// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Malha de Células Recorrentes Otimizada (LSTM) para inferência NAM.

use core::arch::x86_64::*;

/// Uma camada individual do modelo LSTM otimizada para SIMD.
pub struct LstmLayer<const I: usize, const H: usize, const IH: usize, const H4: usize> {
    /// Pesos da camada no layout Gate-Major.
    pub input_hidden_weights: [[[u16; H]; IH]; 4],
    /// Bias lineares da camada.
    pub bias: [f32; H4],
    /// Buffer de estado consolidado [Input | Hidden].
    pub state: [f32; IH],
    /// Espelho do estado em BF16 para aceleração VNNI.
    pub state_bf16: [u16; IH],
    /// Estado da célula (C) do LSTM.
    pub cell_state: [f32; H],
    /// Ativações intermediárias das portas.
    pub gates: [f32; H4],
}

macro_rules! define_lstm_process {
    (
        $fn_name:ident,
        $target_meta:meta,
        $gemv_4gate:path,
        $gemv_4gate_bf16:path,
        $step:expr,
        $load:ident,
        $store:ident,
        $add:ident,
        $mul:ident,
        $tanh:path,
        $sigmoid:path,
        $fused_gates:path,
        $is_bf16:expr
    ) => {
        /// Processa uma amostra através da camada LSTM.
        ///
        /// # Safety
        /// O chamador deve garantir suporte às instruções SIMD especificadas.
        #[$target_meta]
        pub unsafe fn $fn_name(&mut self, input: &[f32]) {
            unsafe {
                self.state[..I].copy_from_slice(&input[..I]);
                if $is_bf16 {
                    for i in 0..I {
                        self.state_bf16[i] = (self.state[i].to_bits() >> 16) as u16;
                    }
                }
                _mm_prefetch::<{ _MM_HINT_T0 }>(self.state.as_ptr().cast::<i8>());
                if $is_bf16 {
                    $gemv_4gate_bf16(
                        &self.state_bf16,
                        self.input_hidden_weights[0].as_flattened(),
                        self.input_hidden_weights[1].as_flattened(),
                        self.input_hidden_weights[2].as_flattened(),
                        self.input_hidden_weights[3].as_flattened(),
                        &self.bias,
                        &mut self.gates,
                        true,
                    );
                } else {
                    $gemv_4gate(
                        &self.state,
                        self.input_hidden_weights[0].as_flattened(),
                        self.input_hidden_weights[1].as_flattened(),
                        self.input_hidden_weights[2].as_flattened(),
                        self.input_hidden_weights[3].as_flattened(),
                        &self.bias,
                        &mut self.gates,
                        true,
                    );
                }
                let f_offset = H;
                let g_offset = 2 * H;
                let o_offset = 3 * H;
                let h_offset = I;
                let mut i = 0;
                while i + $step <= H {
                    let g_f = $load(self.gates.as_ptr().add(i + f_offset));
                    let g_i = $load(self.gates.as_ptr().add(i));
                    let g_g = $load(self.gates.as_ptr().add(i + g_offset));
                    let g_o = $load(self.gates.as_ptr().add(i + o_offset));
                    let c_s = $load(self.cell_state.as_ptr().add(i));
                    let (new_c_s, h_s) = $fused_gates(g_f, g_i, g_g, g_o, c_s);
                    $store(self.cell_state.as_mut_ptr().add(i), new_c_s);
                    $store(self.state.as_mut_ptr().add(h_offset + i), h_s);
                    if $is_bf16 {
                        let mut h_s_arr = [0.0; $step];
                        $store(h_s_arr.as_mut_ptr(), h_s);
                        for j in 0..$step {
                            self.state_bf16[h_offset + i + j] = (h_s_arr[j].to_bits() >> 16) as u16;
                        }
                    }
                    i += $step;
                }
                if i < H {
                    let tail_len = H - i;
                    let mut temp_gf = [0.0; $step];
                    let mut temp_gi = [0.0; $step];
                    let mut temp_gg = [0.0; $step];
                    let mut temp_go = [0.0; $step];
                    let mut temp_cs = [0.0; $step];
                    for j in 0..tail_len {
                        temp_gf[j] = self.gates[i + j + f_offset];
                        temp_gi[j] = self.gates[i + j];
                        temp_gg[j] = self.gates[i + j + g_offset];
                        temp_go[j] = self.gates[i + j + o_offset];
                        temp_cs[j] = self.cell_state[i + j];
                    }
                    let g_f = $load(temp_gf.as_ptr());
                    let g_i = $load(temp_gi.as_ptr());
                    let g_g = $load(temp_gg.as_ptr());
                    let g_o = $load(temp_go.as_ptr());
                    let c_s = $load(temp_cs.as_ptr());
                    let (new_c_s, h_val) = $fused_gates(g_f, g_i, g_g, g_o, c_s);
                    let mut out_cs = [0.0; $step];
                    let mut out_h = [0.0; $step];
                    $store(out_cs.as_mut_ptr(), new_c_s);
                    $store(out_h.as_mut_ptr(), h_val);
                    for j in 0..tail_len {
                        self.cell_state[i + j] = out_cs[j];
                        self.state[i + j + h_offset] = out_h[j];
                        if $is_bf16 {
                            self.state_bf16[i + j + h_offset] = (out_h[j].to_bits() >> 16) as u16;
                        }
                    }
                }
            }
        }
    };
}

macro_rules! define_lstm2_process_pipelined {
    (
        $fn_name:ident,
        $target_meta:meta,
        $layer_proc:ident,
        $dot_prod:path,
        $get_h2:ident
    ) => {
        #[$target_meta]
        unsafe fn $fn_name(&mut self, input: &[f32], output: &mut [f32]) {
            unsafe {
                let len = input.len();
                if len >= 1 {
                    // Prólogo: processa o primeiro frame apenas na camada 1
                    self.layer1.$layer_proc(&[input[0]]);
                    let mut prev_h1 = [0.0; H];
                    prev_h1.copy_from_slice(self.layer1.get_hidden_state());

                    // Loop principal: overlap entre camada 1 (frame i) e camada 2 (frame i-1)
                    for i in 1..len {
                        let current_input = [input[i]];
                        // Estas duas chamadas são agora independentes (processam frames diferentes)!
                        self.layer1.$layer_proc(&current_input);
                        self.layer2.$layer_proc(&prev_h1);

                        let h2 = self.layer2.$get_h2();
                        let dot = $dot_prod(h2, &self.head_weights);
                        output[i - 1] = dot + self.head_bias;

                        prev_h1.copy_from_slice(self.layer1.get_hidden_state());
                    }

                    // Epílogo: processa o último frame na camada 2
                    self.layer2.$layer_proc(&prev_h1);
                    let h2 = self.layer2.$get_h2();
                    let dot = $dot_prod(h2, &self.head_weights);
                    output[len - 1] = dot + self.head_bias;
                }
            }
        }
    };
}

impl<const I: usize, const H: usize, const IH: usize, const H4: usize> LstmLayer<I, H, IH, H4> {
    /// Cria uma nova camada LSTM zerada.
    pub fn new() -> Self {
        Self {
            input_hidden_weights: [[[0u16; H]; IH]; 4],
            bias: [0.0; H4],
            state: [0.0; IH],
            state_bf16: [0u16; IH],
            cell_state: [0.0; H],
            gates: [0.0; H4],
        }
    }
    /// Retorna uma referência ao estado oculto atual.
    #[inline(always)]
    pub fn get_hidden_state(&self) -> &[f32] {
        &self.state[I..]
    }
    /// Retorna uma referência ao estado oculto atual espelhado em BF16.
    #[inline(always)]
    pub fn get_hidden_state_bf16(&self) -> &[u16] {
        &self.state_bf16[I..]
    }

    define_lstm_process!(
        process_sample_avx2,
        inline(always),
        crate::math::simd::gemv_4gate_avx2,
        crate::math::simd::gemv_4gate_bf16_fallback,
        8,
        _mm256_loadu_ps,
        _mm256_storeu_ps,
        _mm256_add_ps,
        _mm256_mul_ps,
        crate::math::fastmath::simd_tanh,
        crate::math::fastmath::simd_sigmoid,
        crate::math::fastmath::fused_lstm_gates_avx2,
        false
    );
    define_lstm_process!(
        process_sample_avx512,
        target_feature(enable = "avx512f,avx512vl"),
        crate::math::simd::gemv_4gate_avx512,
        crate::math::simd::gemv_4gate_bf16_fallback,
        16,
        _mm512_loadu_ps,
        _mm512_storeu_ps,
        _mm512_add_ps,
        _mm512_mul_ps,
        crate::math::fastmath::simd_tanh_avx512,
        crate::math::fastmath::simd_sigmoid_avx512,
        crate::math::fastmath::fused_lstm_gates_avx512,
        false
    );
    define_lstm_process!(
        process_sample_avx2vnni,
        target_feature(enable = "avxvnni"),
        crate::math::simd::gemv_4gate_avx2,
        crate::math::simd::gemv_4gate_bf16_fallback,
        8,
        _mm256_loadu_ps,
        _mm256_storeu_ps,
        _mm256_add_ps,
        _mm256_mul_ps,
        crate::math::fastmath::simd_tanh,
        crate::math::fastmath::simd_sigmoid,
        crate::math::fastmath::fused_lstm_gates_avx2,
        false
    );
    define_lstm_process!(
        process_sample_avx512vnni,
        target_feature(enable = "avx512f,avx512vl,avx512vnni"),
        crate::math::simd::gemv_4gate_avx512,
        crate::math::simd::gemv_4gate_bf16_fallback,
        16,
        _mm512_loadu_ps,
        _mm512_storeu_ps,
        _mm512_add_ps,
        _mm512_mul_ps,
        crate::math::fastmath::simd_tanh_avx512,
        crate::math::fastmath::simd_sigmoid_avx512,
        crate::math::fastmath::fused_lstm_gates_avx512,
        false
    );
    define_lstm_process!(
        process_sample_avx512_vnni_bf16,
        target_feature(enable = "avx512f,avx512vl,avx512bf16"),
        crate::math::simd::gemv_4gate_avx512,
        crate::math::simd::gemv_4gate_bf16_fallback,
        16,
        _mm512_loadu_ps,
        _mm512_storeu_ps,
        _mm512_add_ps,
        _mm512_mul_ps,
        crate::math::fastmath::simd_tanh_avx512,
        crate::math::fastmath::simd_sigmoid_avx512,
        crate::math::fastmath::fused_lstm_gates_avx512,
        true
    );

    /// Processamento escalar (fallback).
    pub fn process_sample_scalar(&mut self, input: &[f32]) {
        let ih = I + H;
        let h = H;
        self.state[..I].copy_from_slice(&input[..I]);
        for k in 0..4 {
            let target_gate_offset = k * h;
            for i in 0..h {
                let mut sum = 0.0;
                for (j, &s) in self.state.iter().enumerate().take(ih) {
                    let w = self.input_hidden_weights[k][j][i];
                    sum += half::f16::from_bits(w).to_f32() * s;
                }
                self.gates[target_gate_offset + i] = sum + self.bias[target_gate_offset + i];
            }
        }
        for j in 0..h {
            let gf = self.gates[j + h];
            let gi = self.gates[j];
            let gg = self.gates[j + 2 * h];
            let go = self.gates[j + 3 * h];
            let cs = self.cell_state[j];
            let f = 0.5 * (1.0 + (gf * 0.5).tanh());
            let i = 0.5 * (1.0 + (gi * 0.5).tanh());
            let g = gg.tanh();
            let o = 0.5 * (1.0 + (go * 0.5).tanh());
            let new_cs = f * cs + i * g;
            let h_val = o * new_cs.tanh();
            self.cell_state[j] = new_cs;
            self.state[I + j] = h_val;
        }
    }
    /// Reseta os estados internos para zero.
    pub fn reset_states(&mut self) {
        self.state.fill(0.0);
        self.cell_state.fill(0.0);
        self.gates.fill(0.0);
    }
}

impl<const I: usize, const H: usize, const IH: usize, const H4: usize> Default
    for LstmLayer<I, H, IH, H4>
{
    fn default() -> Self {
        Self::new()
    }
}

/// Modelo LSTM de 1 camada.
pub struct LstmModel1<const H: usize, const H1_IH: usize, const H_H4: usize> {
    /// Camada única do modelo.
    pub layer: LstmLayer<1, H, H1_IH, H_H4>,
    /// Pesos do cabeçalho de saída (Linear Projection).
    pub head_weights: [u16; H],
    /// Bias do cabeçalho de saída.
    pub head_bias: f32,
}

impl<const H: usize, const H1_IH: usize, const H_H4: usize> LstmModel1<H, H1_IH, H_H4> {
    /// Cria um novo modelo LSTM de 1 camada.
    pub fn new() -> Self {
        Self {
            layer: LstmLayer::new(),
            head_weights: [0u16; H],
            head_bias: 0.0,
        }
    }
    unsafe fn process_avx2(&mut self, input: &[f32], output: &mut [f32]) {
        unsafe {
            let mut i = 0;
            let len = input.len();
            while i < len {
                self.layer.process_sample_avx2(&[input[i]]);
                let hidden = self.layer.get_hidden_state();
                let dot = crate::math::simd::dot_product_avx2(hidden, &self.head_weights);
                output[i] = dot + self.head_bias;
                i += 1;
            }
        }
    }
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn process_avx512(&mut self, input: &[f32], output: &mut [f32]) {
        unsafe {
            let mut i = 0;
            let len = input.len();
            while i < len {
                self.layer.process_sample_avx512(&[input[i]]);
                let hidden = self.layer.get_hidden_state();
                let dot = crate::math::simd::dot_product_avx512(hidden, &self.head_weights);
                output[i] = dot + self.head_bias;
                i += 1;
            }
        }
    }
    #[target_feature(enable = "avxvnni")]
    unsafe fn process_avx2vnni(&mut self, input: &[f32], output: &mut [f32]) {
        unsafe {
            let mut i = 0;
            let len = input.len();
            while i < len {
                self.layer.process_sample_avx2vnni(&[input[i]]);
                let hidden = self.layer.get_hidden_state();
                let dot = crate::math::simd::dot_product_avx2(hidden, &self.head_weights);
                output[i] = dot + self.head_bias;
                i += 1;
            }
        }
    }
    #[target_feature(enable = "avx512f,avx512vl,avx512vnni")]
    unsafe fn process_avx512vnni(&mut self, input: &[f32], output: &mut [f32]) {
        unsafe {
            let mut i = 0;
            let len = input.len();
            while i < len {
                self.layer.process_sample_avx512vnni(&[input[i]]);
                let hidden = self.layer.get_hidden_state();
                let dot = crate::math::simd::dot_product_avx512(hidden, &self.head_weights);
                output[i] = dot + self.head_bias;
                i += 1;
            }
        }
    }
    #[target_feature(enable = "avx512f,avx512vl,avx512bf16")]
    unsafe fn process_avx512_vnni_bf16(&mut self, input: &[f32], output: &mut [f32]) {
        unsafe {
            let mut i = 0;
            let len = input.len();
            while i < len {
                self.layer.process_sample_avx512_vnni_bf16(&[input[i]]);
                let hidden = self.layer.get_hidden_state_bf16();
                let dot = crate::math::simd::dot_product_bf16_avx512(hidden, &self.head_weights);
                output[i] = dot + self.head_bias;
                i += 1;
            }
        }
    }
    /// Processa um bloco de áudio através do modelo (SIMD dispatch).
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        unsafe {
            crate::math::simd::dispatch_simd!(
                self,
                process_avx512_vnni_bf16,
                process_avx512vnni,
                process_avx512,
                process_avx2vnni,
                process_avx2,
                input,
                output
            );
        }
    }
    /// Processamento escalar (fallback).
    pub fn process_scalar(&mut self, input: &[f32], output: &mut [f32]) {
        for i in 0..input.len() {
            self.layer.process_sample_scalar(&[input[i]]);
            let hidden = self.layer.get_hidden_state();
            let mut dot = 0.0;
            for (j, &h_val) in hidden.iter().enumerate().take(H) {
                dot += h_val * half::f16::from_bits(self.head_weights[j]).to_f32();
            }
            output[i] = dot + self.head_bias;
        }
    }
    /// Reseta os estados internos.
    pub fn reset_states(&mut self) {
        self.layer.reset_states();
    }
}

impl<const H: usize, const H1_IH: usize, const H_H4: usize> Default for LstmModel1<H, H1_IH, H_H4> {
    fn default() -> Self {
        Self::new()
    }
}

/// Modelo LSTM de 2 camadas.
pub struct LstmModel2<const H: usize, const H1_IH: usize, const H2_IH: usize, const H_H4: usize> {
    /// Camada 1 do modelo.
    pub layer1: LstmLayer<1, H, H1_IH, H_H4>,
    /// Camada 2 do modelo.
    pub layer2: LstmLayer<H, H, H2_IH, H_H4>,
    /// Pesos do cabeçalho de saída.
    pub head_weights: [u16; H],
    /// Bias do cabeçalho de saída.
    pub head_bias: f32,
}

impl<const H: usize, const H1_IH: usize, const H2_IH: usize, const H_H4: usize>
    LstmModel2<H, H1_IH, H2_IH, H_H4>
{
    /// Cria um novo modelo LSTM de 2 camadas.
    pub fn new() -> Self {
        Self {
            layer1: LstmLayer::new(),
            layer2: LstmLayer::new(),
            head_weights: [0u16; H],
            head_bias: 0.0,
        }
    }
    define_lstm2_process_pipelined!(
        process_avx2,
        inline(always),
        process_sample_avx2,
        crate::math::simd::dot_product_avx2,
        get_hidden_state
    );

    define_lstm2_process_pipelined!(
        process_avx512,
        target_feature(enable = "avx512f,avx512vl"),
        process_sample_avx512,
        crate::math::simd::dot_product_avx512,
        get_hidden_state
    );

    define_lstm2_process_pipelined!(
        process_avx2vnni,
        target_feature(enable = "avxvnni"),
        process_sample_avx2vnni,
        crate::math::simd::dot_product_avx2,
        get_hidden_state
    );

    define_lstm2_process_pipelined!(
        process_avx512vnni,
        target_feature(enable = "avx512f,avx512vl,avx512vnni"),
        process_sample_avx512vnni,
        crate::math::simd::dot_product_avx512,
        get_hidden_state
    );

    define_lstm2_process_pipelined!(
        process_avx512_vnni_bf16,
        target_feature(enable = "avx512f,avx512vl,avx512bf16"),
        process_sample_avx512_vnni_bf16,
        crate::math::simd::dot_product_bf16_avx512,
        get_hidden_state_bf16
    );
    /// Processa um bloco de áudio através do modelo (SIMD dispatch).
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        unsafe {
            crate::math::simd::dispatch_simd!(
                self,
                process_avx512_vnni_bf16,
                process_avx512vnni,
                process_avx512,
                process_avx2vnni,
                process_avx2,
                input,
                output
            );
        }
    }
    /// Processamento escalar (fallback).
    pub fn process_scalar(&mut self, input: &[f32], output: &mut [f32]) {
        for i in 0..input.len() {
            self.layer1.process_sample_scalar(&[input[i]]);
            self.layer2
                .process_sample_scalar(self.layer1.get_hidden_state());
            let hidden2 = self.layer2.get_hidden_state();
            let mut dot = 0.0;
            for (j, &h_val) in hidden2.iter().enumerate().take(H) {
                dot += h_val * half::f16::from_bits(self.head_weights[j]).to_f32();
            }
            output[i] = dot + self.head_bias;
        }
    }
    /// Reseta os estados internos.
    pub fn reset_states(&mut self) {
        self.layer1.reset_states();
        self.layer2.reset_states();
    }
}

impl<const H: usize, const H1_IH: usize, const H2_IH: usize, const H_H4: usize> Default
    for LstmModel2<H, H1_IH, H2_IH, H_H4>
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "lstm_test.rs"]
mod lstm_test;
