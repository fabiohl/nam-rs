// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Malha CNN Causal Dinâmica para inferência WaveNet (Fallback).
//!
//! Permite carregamento de modelos com topologias (canais, dilatações, etc.) não cobertas
//! pelas restrições dos Const Generics. A alocação ocorre apenas na thread hospedeira
//! (durante construtor) permitindo zero-allocation e RT-safety no caminho DSP, trocando
//! unroll do compilador (estático) por iterações dinâmicas de matriz em SIMD.

#![allow(clippy::needless_range_loop)]

use crate::models::wavenet::WaveNetLayerState;

/// Estrutura para convolução causal 1D com dimensões limitadas em runtime.
#[derive(Clone)]
pub struct Conv1dDyn {
    /// Pesos da convolução arranjados contiguamente.
    pub weights: Vec<u16>,
    /// Vetor de bias somado aos canais de saída.
    pub bias: Vec<f32>,
    /// Flag indicando se o bias deve ser aplicado.
    pub do_bias: bool,
    /// Fator de dilatação da camada para acesso casual na fita temporal.
    pub dilation: usize,
    /// Quantidade de canais de entrada.
    pub in_ch: usize,
    /// Quantidade de canais de saída.
    pub out_ch: usize,
    /// Tamanho físico do kernel (fator de retardo causais no buffer).
    pub kernel: usize,
}

impl Conv1dDyn {
    /// Processa um frame temporal aplicando a convolução sobre o histórico em buffer livre de alocação.
    ///
    /// ## Otimização: Software Prefetch Proativo
    ///
    /// Idêntico ao `Conv1d` estático: para dilatações grandes, emite `_mm_prefetch`
    /// para o próximo tap do kernel enquanto o tap atual é processado via FMA,
    /// eliminando L1 cache misses previsíveis (~5–10% ganho em layers dilatadas).
    ///
    /// # Safety
    /// Depende da instância estrita de `SimdMathConfig` referenciar uma SIMD suportada.
    /// Processa bloco iterativo.
    /// # Safety
    /// Pointer must be valid.
    pub unsafe fn process_block<M: crate::math::simd::SimdMath>(
        &self,
        layer_buffer: &[f32],
        block: &mut [f32],
        buffer_start: usize,
        num_frames: usize,
    ) {
        // out_c_base controla o bloco de canais de saída sendo processado (batch de 4 canais para SIMD)
        let mut out_c_base = 0;
        // Processa os canais de saída em blocos de 4 para otimizar o uso de registradores SIMD
        while out_c_base + 4 <= self.out_ch {
            // Itera sobre o tamanho do kernel causal temporal (geralmente kernel_size = 2 ou 3)
            for k in 0..self.kernel {
                // Calcula o offset do elemento passado no tempo baseado na dilatação
                let offset = (self.dilation as isize) * ((k as isize) + 1 - (self.kernel as isize));
                let base_frame_idx = (buffer_start as isize) + offset;

                // Calcula os offsets iniciais dos pesos para os 4 canais de saída simultâneos.
                // Na NAM, os pesos contíguos de convolução 1D são dispostos no formato [out_ch][kernel][in_ch].
                // Isso maximiza o acesso sequencial à memória do canal de entrada (que tem tamanho `in_ch`).
                let w_start0 = ((out_c_base) * self.kernel + k) * self.in_ch;
                let w_start1 = ((out_c_base + 1) * self.kernel + k) * self.in_ch;
                let w_start2 = ((out_c_base + 2) * self.kernel + k) * self.in_ch;
                let w_start3 = ((out_c_base + 3) * self.kernel + k) * self.in_ch;

                let w0 = unsafe { self.weights.get_unchecked(w_start0..w_start0 + self.in_ch) };
                let w1 = unsafe { self.weights.get_unchecked(w_start1..w_start1 + self.in_ch) };
                let w2 = unsafe { self.weights.get_unchecked(w_start2..w_start2 + self.in_ch) };
                let w3 = unsafe { self.weights.get_unchecked(w_start3..w_start3 + self.in_ch) };

                // Se for o primeiro tap do kernel (k == 0), devemos sobrescrever o lixo da memória (= assignment direto)
                // e já somar o bias. Nos taps subsequentes (k > 0), acumulamos (+=).
                if k == 0 {
                    let bias0 = if self.do_bias {
                        self.bias[out_c_base]
                    } else {
                        0.0
                    };
                    let bias1 = if self.do_bias {
                        self.bias[out_c_base + 1]
                    } else {
                        0.0
                    };
                    let bias2 = if self.do_bias {
                        self.bias[out_c_base + 2]
                    } else {
                        0.0
                    };
                    let bias3 = if self.do_bias {
                        self.bias[out_c_base + 3]
                    } else {
                        0.0
                    };

                    for i in 0..num_frames {
                        let frame_idx = base_frame_idx + (i as isize);
                        let in_slice_start = (frame_idx as usize) * self.in_ch;

                        // Prefetch adaptativo para o próximo "tap" do kernel temporal (dilation-aware).
                        if k + 1 < self.kernel {
                            let prefetch_ptr = unsafe {
                                layer_buffer.as_ptr().add(in_slice_start + self.dilation)
                            };
                            unsafe {
                                crate::math::simd::adaptive_prefetch_f32(
                                    prefetch_ptr,
                                    self.dilation,
                                );
                            }
                        }

                        let in_slice = unsafe {
                            layer_buffer.get_unchecked(in_slice_start..in_slice_start + self.in_ch)
                        };

                        // OPERAÇÃO CORE (SIMD 4x):
                        // Aqui acontece a "mágica" da aceleração: calculamos o produto escalar (dot product)
                        // de uma única fatia de entrada (in_slice) contra 4 conjuntos de pesos diferentes
                        // (w0 a w3) simultaneamente.
                        // Benefício: A fatia de entrada (in_slice) é lida apenas uma vez da memória e
                        // permanece nos registradores enquanto as unidades de FMA (Fused Multiply-Add)
                        // processam os 4 canais de saída. Isso maximiza o throughput de dados.
                        let [r0, r1, r2, r3] =
                            unsafe { M::dot_product_4x(w0, w1, w2, w3, in_slice) };

                        // ESCRITA DIRETA + BIAS (k == 0):
                        // Como este é o primeiro "tap" (passo) do kernel temporal, estamos inicializando
                        // a memória do bloco de saída. Substituímos qualquer valor residual pelo bias
                        // somado ao resultado do produto escalar.
                        unsafe {
                            *block.get_unchecked_mut(i * self.out_ch + out_c_base) = bias0 + r0;
                            *block.get_unchecked_mut(i * self.out_ch + out_c_base + 1) = bias1 + r1;
                            *block.get_unchecked_mut(i * self.out_ch + out_c_base + 2) = bias2 + r2;
                            *block.get_unchecked_mut(i * self.out_ch + out_c_base + 3) = bias3 + r3;
                        }
                    }
                } else {
                    for i in 0..num_frames {
                        let frame_idx = base_frame_idx + (i as isize);
                        let in_slice_start = (frame_idx as usize) * self.in_ch;

                        // Prefetch adaptativo para o próximo "tap" do kernel temporal (dilation-aware).
                        if k + 1 < self.kernel {
                            let prefetch_ptr = unsafe {
                                layer_buffer.as_ptr().add(in_slice_start + self.dilation)
                            };
                            unsafe {
                                crate::math::simd::adaptive_prefetch_f32(
                                    prefetch_ptr,
                                    self.dilation,
                                );
                            }
                        }

                        let in_slice = unsafe {
                            layer_buffer.get_unchecked(in_slice_start..in_slice_start + self.in_ch)
                        };

                        // PRODUTO ESCALAR VETORIZADO (Simultâneo para 4 canais):
                        // Mesma lógica de alta performance aplicada no bloco acima.
                        let [r0, r1, r2, r3] =
                            unsafe { M::dot_product_4x(w0, w1, w2, w3, in_slice) };

                        // ACUMULAÇÃO (k > 0):
                        // Como não é o primeiro passo do kernel, não podemos sobrescrever a memória.
                        // Somamos (+=) o novo resultado ao que já foi calculado nos taps anteriores.
                        unsafe {
                            *block.get_unchecked_mut(i * self.out_ch + out_c_base) += r0;
                            *block.get_unchecked_mut(i * self.out_ch + out_c_base + 1) += r1;
                            *block.get_unchecked_mut(i * self.out_ch + out_c_base + 2) += r2;
                            *block.get_unchecked_mut(i * self.out_ch + out_c_base + 3) += r3;
                        }
                    }
                }
            }
            out_c_base += 4;
        }

        // PROCESSO DE CAUDA (REMAINDER):
        // Se a quantidade de canais de saída (out_ch) não for múltiplo de 4, os canais restantes
        // (1, 2 ou 3 canais) são processados individualmente aqui para garantir a corretude.
        while out_c_base < self.out_ch {
            let out_c = out_c_base;
            for k in 0..self.kernel {
                // Cálculo do deslocamento (offset) temporal baseado na dilatação da camada.
                let offset = (self.dilation as isize) * ((k as isize) + 1 - (self.kernel as isize));
                let base_frame_idx = (buffer_start as isize) + offset;

                // Localiza a fatia de pesos específica para este canal de saída e tap do kernel.
                let weight_slice_start = (out_c * self.kernel + k) * self.in_ch;
                let weight_slice = unsafe {
                    self.weights
                        .get_unchecked(weight_slice_start..weight_slice_start + self.in_ch)
                };

                if k == 0 {
                    // Inicialização com Bias: Primeiro tap limpa a memória com o bias + dot product.
                    let bias = if self.do_bias { self.bias[out_c] } else { 0.0 };
                    for i in 0..num_frames {
                        let frame_idx = base_frame_idx + (i as isize);
                        let in_slice_start = (frame_idx as usize) * self.in_ch;

                        // Prefetch adaptativo com lookahead.
                        let lookahead_offset = 16;
                        let prefetch_ptr =
                            unsafe { layer_buffer.as_ptr().add(in_slice_start + lookahead_offset) };
                        unsafe {
                            crate::math::simd::adaptive_prefetch_f32(prefetch_ptr, self.dilation);
                        }

                        let in_slice = unsafe {
                            layer_buffer.get_unchecked(in_slice_start..in_slice_start + self.in_ch)
                        };
                        unsafe {
                            *block.get_unchecked_mut(i * self.out_ch + out_c) =
                                bias + M::dot_product(in_slice, weight_slice);
                        }
                    }
                } else {
                    // Acumulação: Taps subsequentes somam o resultado ao acumulado.
                    for i in 0..num_frames {
                        let frame_idx = base_frame_idx + (i as isize);
                        let in_slice_start = (frame_idx as usize) * self.in_ch;

                        // Prefetch adaptativo com lookahead.
                        let lookahead_offset = 16;
                        let prefetch_ptr =
                            unsafe { layer_buffer.as_ptr().add(in_slice_start + lookahead_offset) };
                        unsafe {
                            crate::math::simd::adaptive_prefetch_f32(prefetch_ptr, self.dilation);
                        }

                        let in_slice = unsafe {
                            layer_buffer.get_unchecked(in_slice_start..in_slice_start + self.in_ch)
                        };
                        unsafe {
                            *block.get_unchecked_mut(i * self.out_ch + out_c) +=
                                M::dot_product(in_slice, weight_slice);
                        }
                    }
                }
            }
            out_c_base += 1;
        }
    }
}

/// Camada Dense (fully-connected / projeção linear 1×1) avaliada dinamicamente.
#[derive(Clone)]
pub struct DenseLayerDyn {
    /// Matriz densa de pesos `[Output][Input]`.
    pub weights: Vec<u16>,
    /// Bias unificado somado a saída.
    pub bias: Vec<f32>,
    /// Flag ativando bias.
    pub do_bias: bool,
    /// Dimensão da entrada (Input Size).
    pub in_size: usize,
    /// Dimensão projetada final (Output Size).
    pub out_size: usize,
}

impl DenseLayerDyn {
    /// ACUMULAÇÃO DENSA (Projeção Linear 1x1):
    /// Processa a multiplicação de matriz (entrada x pesos) e ACUMULA o resultado no vetor `output`.
    /// É fundamentalmente uma camada Fully-Connected executada dinamicamente.
    ///
    /// # Safety
    /// Depende do `SimdMathConfig` estar validamente instanciado para a arquitetura alvo.
    pub unsafe fn process_acc_block<M: crate::math::simd::SimdMath>(
        &self,
        input: &[f32],
        output: &mut [f32],
        num_frames: usize,
    ) {
        for i in 0..num_frames {
            let in_slice = unsafe { input.get_unchecked(i * self.in_size..(i + 1) * self.in_size) };
            let out_slice =
                unsafe { output.get_unchecked_mut(i * self.out_size..(i + 1) * self.out_size) };
            unsafe {
                M::fused_add_gemv(in_slice, &self.weights, &self.bias, out_slice, self.do_bias);
            }
        }
    }

    /// Executa a projeção fundida (W*in + bias) somando ao buffer de saída (Residual Fusion).
    ///
    /// # Safety
    /// O chamador deve garantir que `in_frame` e `out_frame` tenham tamanhos compatíveis com `in_size` e `out_size`.
    #[inline(always)]
    pub unsafe fn process_fused<M: crate::math::simd::SimdMath>(
        &self,
        in_frame: &[f32],
        out_frame: &mut [f32],
    ) {
        unsafe {
            M::fused_add_gemv(in_frame, &self.weights, &self.bias, out_frame, self.do_bias);
        }
    }

    /// ACUMULAÇÃO STRIDED (Salto de Memória):
    /// Similar ao `process_acc_block`, mas suporta `strides` (saltos).
    /// Útil quando os dados de entrada ou saída não estão perfeitamente contíguos
    /// (ex: processando sub-blocos ou canais intercalados).
    ///
    /// # Safety
    /// Requer `SimdMathConfig` válido e ponteiros de buffer com espaço suficiente para os strides.
    pub unsafe fn process_acc_block_strided<M: crate::math::simd::SimdMath>(
        &self,
        input: &[f32],
        output: &mut [f32],
        num_frames: usize,
        in_stride: usize,
        out_stride: usize,
    ) {
        // [PASSO: Buffer Temporário Stack]
        // Para operações strided, projetamos o frame em um buffer contíguo local
        // e depois espalhamos (scatter) para o buffer de saída com os strides corretos.
        debug_assert!(
            self.out_size <= 1024,
            "DenseLayerDyn::process_acc_block_strided: out_size ({}) excede o buffer stack (1024)",
            self.out_size,
        );
        let mut tmp = [0.0f32; 1024];
        let tmp_slice = &mut tmp[..self.out_size];

        for i in 0..num_frames {
            let in_slice =
                unsafe { input.get_unchecked(i * in_stride..i * in_stride + self.in_size) };
            let out_base = i * out_stride;

            // Limpa o buffer temporário
            tmp_slice.fill(0.0);

            unsafe {
                M::fused_add_gemv(in_slice, &self.weights, &self.bias, tmp_slice, self.do_bias);
            }

            // Scatter (Acúmulo)
            for (j, &val) in tmp_slice.iter().enumerate() {
                unsafe {
                    *output.get_unchecked_mut(out_base + j) += val;
                }
            }
        }
    }

    /// PROCESSO DENSO STRIDED (Atribuição Direta):
    /// Realiza a transformação linear com saltos (strides), mas SOBRESCREVE o buffer de saída.
    /// Diferente de `process_acc_block_strided` (que usa `+=`), aqui usamos `=`.
    ///
    /// # Safety
    /// Requer `SimdMathConfig` válido e ponteiros com acesso seguro conforme os strides.
    pub unsafe fn process_block_strided<M: crate::math::simd::SimdMath>(
        &self,
        input: &[f32],
        output: &mut [f32],
        num_frames: usize,
        in_stride: usize,
        out_stride: usize,
    ) {
        debug_assert!(
            self.out_size <= 1024,
            "DenseLayerDyn::process_block_strided: out_size ({}) excede o buffer stack (1024)",
            self.out_size,
        );
        let mut tmp = [0.0f32; 1024];
        let tmp_slice = &mut tmp[..self.out_size];

        for i in 0..num_frames {
            let in_slice =
                unsafe { input.get_unchecked(i * in_stride..i * in_stride + self.in_size) };
            let out_base = i * out_stride;

            tmp_slice.fill(0.0);

            unsafe {
                M::fused_add_gemv(in_slice, &self.weights, &self.bias, tmp_slice, self.do_bias);
            }

            // Scatter (Atribuição Direta)
            for (j, &val) in tmp_slice.iter().enumerate() {
                unsafe {
                    *output.get_unchecked_mut(out_base + j) = val;
                }
            }
        }
    }

    /// PROCESSO DENSO (Contíguo):
    /// Esta é a versão mais simples e rápida da projeção linear, usada quando os buffers
    /// de entrada e saída estão perfeitamente contíguos em memória.
    ///
    /// # Safety
    /// Requer `SimdMathConfig` válido e buffers com tamanho compatível com `in_size` e `out_size`.
    pub unsafe fn process_block<M: crate::math::simd::SimdMath>(
        &self,
        input: &[f32],
        output: &mut [f32],
        num_frames: usize,
    ) {
        for i in 0..num_frames {
            let in_slice = unsafe { input.get_unchecked(i * self.in_size..(i + 1) * self.in_size) };
            let out_slice =
                unsafe { output.get_unchecked_mut(i * self.out_size..(i + 1) * self.out_size) };

            out_slice.fill(0.0);
            unsafe {
                M::fused_add_gemv(in_slice, &self.weights, &self.bias, out_slice, self.do_bias);
            }
        }
    }
}

/// O elemento atomizado de conexão na malha, que interliga as equações diferenciais convolutivas.
#[derive(Clone)]
pub struct WaveNetLayerDyn {
    /// Núcleo convolutivo casual com histórico.
    pub conv1d: Conv1dDyn,
    /// Misturador acoplante de input local (residuum).
    pub input_mixin: DenseLayerDyn,
    /// Transformador 1x1 associado ao output/head da inferência final.
    pub one_by_one: DenseLayerDyn,
    /// Tamanho fixo do número de canais desta arquitetura.
    pub ch: usize,
    /// Ativa o mecanismo de Gated Activation: `tanh(z[0..ch]) ⊙ sigmoid(z[ch..2*ch])`.
    ///
    /// Quando `true`, o `conv1d` deve ter `out_ch = 2 * ch`.
    pub gated: bool,
}

impl WaveNetLayerDyn {
    /// MOTOR DE PASSAGEM DA CAMADA (Fluxo WaveNet):
    /// Aqui as equações de uma camada WaveNet são executadas sequencialmente.
    /// O fluxo segue o padrão: Convolução -> Condicionamento -> Ativação (Gated) -> Skip -> Residual.
    ///
    /// # Safety
    /// Requer instâncias estritas do buffer interno e `block` com tamanho
    /// `ch` (não-gated) ou `2*ch` (gated).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn process_block_internal<M: crate::math::simd::SimdMath>(
        &self,
        condition: &[f32],
        head_input: &mut [f32],
        output: &mut [f32],
        layer_buffer: &[f32],
        buffer_start: usize,
        block: &mut [f32],
        num_frames: usize,
    ) {
        let ch = self.ch;

        // LIMPEZA DE ESTADO:
        // Zeramos o bloco temporário para garantir que não haja vazamento de áudio
        // entre camadas ou frames passados.
        for v in block.iter_mut() {
            *v = 0.0;
        }

        unsafe {
            // 1) CONVOLUÇÃO CAUSAL:
            // Aplica a convolução sobre o histórico de áudio (layer_buffer).
            // O resultado é escrito no buffer temporário `block`.
            self.conv1d
                .process_block::<M>(layer_buffer, block, buffer_start, num_frames);

            // 2) MECANISMO DE GATING (Ativação):
            if self.gated {
                // Mistura a condição externa (condition) nos canais do bloco.
                // Usamos stride de 2*ch porque o bloco contém tanh e sigmoid intercalados.
                self.input_mixin.process_acc_block_strided::<M>(
                    condition,
                    block,
                    num_frames,
                    1,
                    2 * self.ch,
                );

                for i in 0..num_frames {
                    let block_start = i * self.conv1d.out_ch;
                    let block_frame = &mut block[block_start..block_start + self.conv1d.out_ch];

                    // SPLIT GATED: O bloco é dividido em dois: Z1 (tanh) e Z2 (sigmoid).
                    let (z1, z2) = block_frame.split_at_mut(self.ch);
                    M::tanh_slice(z1);
                    M::sigmoid_slice(z2);

                    // OPERAÇÃO GATED: z1 = tanh(z1) * sigmoid(z2).
                    // Isso permite que o modelo aprenda quais informações deixar passar (gate).
                    for j in 0..self.ch {
                        *z1.get_unchecked_mut(j) *= *z2.get_unchecked(j);
                    }
                }
            } else {
                // MODO NÃO-GATED: Mais simples, soma a condição e aplica apenas tanh.
                self.input_mixin
                    .process_acc_block_strided::<M>(condition, block, num_frames, 1, self.ch);
                M::tanh_slice(&mut block[0..num_frames * self.ch]);
            }

            // 3) SKIP CONNECTION (Saída para a Cabeça):
            // O resultado processado (z1) é somado ao `head_input`.
            // Todas as camadas contribuem para este somatório global que gera o áudio final.
            for i in 0..num_frames {
                let head_frame = &mut head_input[i * ch..i * ch + ch];
                let block_frame = &block[i * (if self.gated { 2 * ch } else { ch })
                    ..i * (if self.gated { 2 * ch } else { ch }) + ch];
                for j in 0..ch {
                    *head_frame.get_unchecked_mut(j) += *block_frame.get_unchecked(j);
                }
            }

            // 4) PROJEÇÃO 1x1 + FUSÃO RESIDUAL:
            // Prepara o sinal para a próxima camada, fundindo a projeção com a soma residual.
            let in_stride = if self.gated { 2 * self.ch } else { self.ch };

            for i in 0..num_frames {
                let out_frame = output.get_unchecked_mut(i * ch..i * ch + ch);
                let lb_start = (buffer_start + i) * ch;
                let block_frame = &block[i * in_stride..i * in_stride + ch];

                // [PASSO: Residual Fusion]
                // Inicializamos a saída com o dado residual (Skip original).
                out_frame.copy_from_slice(layer_buffer.get_unchecked(lb_start..lb_start + ch));

                // [PASSO: Fused GEMV]
                // Somamos a projeção 1x1 diretamente sobre o residual.
                self.one_by_one.process_fused::<M>(block_frame, out_frame);
            }
        }
    }
}

/// Representa a topologia vertical inteira de um galho WaveNet dinâmico, suportando múltiplas dilatações seq.
pub struct WaveNetLayerArrayDyn {
    /// Lista empilhada de camadas com suas respectivas dilatações fixadas na RAM em loading.
    pub layers: Vec<WaveNetLayerDyn>,
    /// Registro espelho local de fita de retardo. Mantido para as passagens lock-free circulares.
    pub states: Vec<WaveNetLayerState>,
    /// Redimensionador denso do canal de entrada.
    pub rechannel: DenseLayerDyn,
    /// Redimensionador denso para a malha da cabeça de soma paramétrica.
    pub head_rechannel: DenseLayerDyn,
    /// Acumulador temporário das cadeias sequenciais.
    pub array_outputs: std::vec::Vec<f32>,
    /// Acumulador das ativações projetadas pela malha Head.
    pub head_accum: std::vec::Vec<f32>,
    /// Projeções finais da cabeça em andamento (somatório de projeções de múltiplas camadas).
    pub head_outputs: std::vec::Vec<f32>,
    /// Buffer de estado auxiliar reutilizado para inibir heap allocation nas threads de RT.
    ///
    /// Tamanho: `ch` quando `gated == false`, ou `2 * ch` quando `gated == true`,
    /// para acomodar os dois slots (tanh + sigmoid) do mecanismo gated.
    pub block_buffer: std::vec::Vec<f32>,
    /// Tamanho efetivo do `block_buffer`. Igual a `ch` ou `2*ch` conforme gated.
    pub block_size: usize,
    /// Tamanho analítico global de latência causal desta cascata. (Usado no prewarm).
    pub receptive_field_size: usize,
    /// Eixo transversal de Canais base (`C`).
    pub ch: usize,
    /// Redução projetada somatória.
    pub head: usize,
}

impl WaveNetLayerArrayDyn {
    /// ORQUESTRADOR DE INFERÊNCIA (Cascade Process):
    /// Realiza a inferência síncrona de todas as camadas do Array em cascata.
    /// Este é o ponto de entrada principal para processar um bloco de áudio.
    ///
    /// # Safety
    /// Depende da integridade das matrizes carregadas e dos estados de buffer circular.
    pub unsafe fn process<M: crate::math::simd::SimdMath>(
        &mut self,
        layer_inputs: &[f32],
        condition: &[f32],
        num_frames: usize,
    ) {
        debug_assert_eq!(
            self.layers.len(),
            self.states.len(),
            "WaveNetLayerArrayDyn: layers ({}) ≠ states ({})",
            self.layers.len(),
            self.states.len()
        );
        let ch = self.ch;
        let head = self.head;

        let states_ptr = self.states.as_mut_ptr();

        // 1) RESET DO ACUMULADOR DE CABEÇA (Skip Connections):
        // Antes de começar, zeramos o buffer onde todas as camadas vão somar seus outputs parciais.
        for v in self.head_accum.iter_mut() {
            *v = 0.0;
        }

        unsafe {
            let state_0 = &mut *states_ptr.add(0);
            let start = state_0.buffer_start * ch;

            // 2) RECHANNEL (Entrada -> Residual):
            // Projeta o áudio de entrada (1 canal) para a dimensão residual (`ch`).
            // O resultado é escrito DIRETAMENTE na fita de retardo (buffer circular) da 1ª camada.
            self.rechannel.process_block::<M>(
                layer_inputs,
                &mut state_0.layer_buffer[start..start + num_frames * ch],
                num_frames,
            );

            let num_layers = self.layers.len();
            let last_layer = num_layers - 1;

            let block_size = self.block_size;

            // 3) CASCATEAMENTO DE CAMADAS:
            // Itera sobre todas as camadas da rede.
            for (i, layer) in self.layers.iter().enumerate() {
                let current_state = &mut *states_ptr.add(i);

                if i == last_layer {
                    // ÚLTIMA CAMADA: O output residual final vai para `array_outputs`.
                    layer.process_block_internal::<M>(
                        condition,
                        &mut self.head_accum[0..num_frames * ch],
                        &mut self.array_outputs[0..num_frames * ch],
                        &current_state.layer_buffer,
                        current_state.buffer_start,
                        &mut self.block_buffer[0..num_frames * block_size],
                        num_frames,
                    );
                } else {
                    let next_state = &mut *states_ptr.add(i + 1);
                    let next_start = next_state.buffer_start * ch;

                    // CONEXÃO ENTRE CAMADAS:
                    // A saída residual da camada 'i' é injetada DIRETAMENTE no buffer da camada 'i+1'.
                    // Isso economiza cópias de memória e mantém os dados quentes no cache.
                    layer.process_block_internal::<M>(
                        condition,
                        &mut self.head_accum[0..num_frames * ch],
                        &mut next_state.layer_buffer[next_start..next_start + num_frames * ch],
                        &current_state.layer_buffer,
                        current_state.buffer_start,
                        &mut self.block_buffer[0..num_frames * block_size],
                        num_frames,
                    );
                }

                // 4) ATUALIZAÇÃO DOS PONTEIROS CIRCULARES:
                // Move o índice 'buffer_start' para a frente, "envelhecendo" as amostras no buffer causal.
                current_state.advance_frames(num_frames, ch);
            }

            // 5) HEAD RECHANNEL (Skip Sum -> Output):
            // Pega o somatório de todas as skip connections e realiza a projeção final 1x1.
            // O resultado (`head_outputs`) é o que será usado para prever o próximo valor do áudio.
            self.head_rechannel.process_block::<M>(
                &self.head_accum[0..num_frames * ch],
                &mut self.head_outputs[0..num_frames * head],
                num_frames,
            );
        }
    }

    /// AQUECIMENTO DE ESTADO (Pre-warm):
    /// Preenche os buffers circulares com valores iniciais para evitar artefatos de áudio (cliques/pops)
    /// no início da reprodução. Essencial para redes com histórico (convoluções causais).
    pub fn prewarm<M: crate::math::simd::SimdMath>(
        &mut self,
        layer_inputs: &[f32],
        condition: &[f32],
    ) {
        debug_assert_eq!(
            self.layers.len(),
            self.states.len(),
            "WaveNetLayerArrayDyn: layers ({}) ≠ states ({})",
            self.layers.len(),
            self.states.len()
        );
        let ch = self.ch;
        let head = self.head;
        let states_ptr = self.states.as_mut_ptr();

        // Inicializa acumuladores.
        for v in self.head_accum.iter_mut() {
            *v = 0.0;
        }

        unsafe {
            let state_0 = &mut *states_ptr.add(0);
            let start = state_0.buffer_start * ch;

            // Projeção inicial do frame de aquecimento.
            self.rechannel.process_block::<M>(
                layer_inputs,
                &mut state_0.layer_buffer[start..start + ch],
                1, // O aquecimento processa frame a frame (1 frame).
            );

            let num_layers = self.layers.len();
            let last_layer = num_layers - 1;

            let block_size = self.block_size;

            for (i, layer) in self.layers.iter().enumerate() {
                let current_state = &mut *states_ptr.add(i);

                // POPULANDO O HISTÓRICO:
                // Copia o valor atual para o histórico do buffer circular da camada.
                current_state.copy_buffer(ch);

                if i == last_layer {
                    layer.process_block_internal::<M>(
                        condition,
                        &mut self.head_accum[0..ch],
                        &mut self.array_outputs[0..ch],
                        &current_state.layer_buffer,
                        current_state.buffer_start,
                        &mut self.block_buffer[0..block_size],
                        1,
                    );
                } else {
                    let next_state = &mut *states_ptr.add(i + 1);
                    let next_start = next_state.buffer_start * ch;

                    layer.process_block_internal::<M>(
                        condition,
                        &mut self.head_accum[0..ch],
                        &mut next_state.layer_buffer[next_start..next_start + ch],
                        &current_state.layer_buffer,
                        current_state.buffer_start,
                        &mut self.block_buffer[0..block_size],
                        1,
                    );
                }
            }

            // Projeção final do frame de aquecimento.
            self.head_rechannel.process_block::<M>(
                &self.head_accum[0..ch],
                &mut self.head_outputs[0..head],
                1,
            );
        }
    }
}

/// Invólucro Dinâmico final. Comporta Arrays interconectados conforme a abstração de Inferência `NamModel`.
///
/// **Referência Científica:** van den Oord, A., et al. (2016). *"WaveNet: A Generative Model for Raw Audio."* DeepMind.
pub struct WaveNetDynModel {
    /// O galho Primário com maior parte do campo causal da WaveNet.
    pub array1: WaveNetLayerArrayDyn,
    /// Galho Secundário, redutor mono causal.
    pub array2: WaveNetLayerArrayDyn,
    /// Ajuste de volume master computado pré-linearização.
    pub head_scale: f32,
    /// Carga total de frames que este modelo assimila antes da saída confiável.
    pub receptive_field_size: usize,
    /// Dimensões da convergência interna final do head.
    pub head: usize,
}

impl WaveNetDynModel {
    /// Processa o bloco de áudio na matriz causal.
    ///
    /// **Para Cientistas e Devs:** Todo o processamento passa obrigatoriamente pela macro `dispatch_simd!`.
    /// Essa tática de multiversioning anula custos de ramificação em tempo de execução (branches condicionais)
    /// para selecionar qual conjunto de intruções de CPU usar (AVX2, AVX-512). O compilador gera clones desta
    /// classe de processamento especializados para cada chipset.
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        unsafe {
            crate::math::simd::dispatch_simd!(self, process_internal, input, output);
        }
    }

    /// Variante clonada e puramente otimizada para SIMD `M` do processamento da rede inteira.
    unsafe fn process_internal<M: crate::math::simd::SimdMath>(
        &mut self,
        input: &[f32],
        output: &mut [f32],
    ) {
        let total_frames = input.len();

        let mut pos = 0;
        // Processa em janelas para limitar cache misses (chunking).
        // Os buffers internos circulares dependem destas iterações menores e densas.
        while pos < total_frames {
            let num_frames =
                (total_frames - pos).min(crate::models::wavenet::WAVENET_MAX_NUM_FRAMES);
            let in_slice = &input[pos..pos + num_frames];

            unsafe {
                // array1 engloba a porção majoritária do campo receptivo da WaveNet.
                self.array1.process::<M>(in_slice, in_slice, num_frames);

                let array1_outputs = &self.array1.array_outputs[0..num_frames * self.array1.ch];
                // array2 recebe o processado do array1 e o sinal original no canal de condição.
                // É utilizado para condensar o resultado ou aplicar finalizações paramétricas.
                self.array2
                    .process::<M>(array1_outputs, in_slice, num_frames);
            }

            // Mixagem Final (Master Blend):
            // Combina a predição condensada de head do Array1 com o Head do Array2,
            // e os redimensiona/escala para o intervalo de áudio -1.0 a +1.0.
            for i in 0..num_frames {
                let mut final_sum = 0.0f32;
                for j in 0..self.head {
                    final_sum += self.array1.head_outputs[i * self.head + j];
                }
                final_sum += self.array2.head_outputs[i];
                output[pos + i] = final_sum * self.head_scale;
            }
            pos += num_frames;
        }
    }

    /// Realiza o `Prewarm` inicial para estabilizar os buffers.
    pub fn prewarm(&mut self) {
        unsafe {
            crate::math::simd::dispatch_simd!(self, prewarm_internal);
        }
    }

    /// # Safety
    /// Call this via `dispatch_simd!` macro only.
    unsafe fn prewarm_internal<M: crate::math::simd::SimdMath>(&mut self) {
        let condition = [0.0f32];
        let layer_inputs_1 = [0.0f32];

        // Roda o estado com uma amostra preenchida de 0.0 para preencher os buffers internos
        // e impedir ruídos espúrios ao trocar modelos (os pipelines de convolução precisam
        // de amostras causais passadas).
        self.array1.prewarm::<M>(&layer_inputs_1, &condition);
        let array1_outputs = &self.array1.array_outputs[..];
        self.array2.prewarm::<M>(array1_outputs, &condition);
    }
}

// =============================================================================
// Testes Unitários
// =============================================================================

#[cfg(test)]
#[path = "wavenet_dyn_test.rs"]
mod wavenet_dyn_test;
