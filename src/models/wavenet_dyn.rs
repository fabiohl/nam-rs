// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.
// Portado em grande parte da implementação original em C++ (NeuralAudio) por Mike Oliphant.

//! Malha CNN Causal Dinâmica para inferência WaveNet (Fallback).
//!
//! Permite carregamento de modelos com topologias (canais, dilatações, etc.) não cobertas
//! pelas restrições dos Const Generics. A alocação ocorre apenas na thread hospedeira
//! (durante construtor) permitindo zero-allocation e RT-safety no caminho DSP, trocando
//! unroll do compilador (estático) por iterações dinâmicas de matriz em SIMD.

#![allow(clippy::needless_range_loop)]

use crate::math::simd::SimdMathConfig;
use crate::models::wavenet::WaveNetLayerState;

/// Estrutura para convolução causal 1D com dimensões limitadas em runtime.
#[derive(Clone)]
pub struct Conv1dDyn {
    /// Pesos da convolução arranjados contiguamente.
    pub weights: Vec<f32>,
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
    pub unsafe fn process_block(
        &self,
        layer_buffer: &[f32],
        block: &mut [f32],
        buffer_start: usize,
        num_frames: usize,
        math: &SimdMathConfig,
    ) {
        let mut out_c_base = 0;
        while out_c_base + 4 <= self.out_ch {
            for k in 0..self.kernel {
                if k + 1 < self.kernel {
                    let next_offset =
                        (self.dilation as isize) * ((k as isize) + 2 - (self.kernel as isize));
                    let next_frame_idx = (buffer_start as isize) + next_offset;
                    let next_addr = unsafe {
                        layer_buffer
                            .as_ptr()
                            .add((next_frame_idx as usize) * self.in_ch)
                    };
                    unsafe {
                        core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(
                            next_addr.cast::<i8>(),
                        );
                    }
                }

                let offset = (self.dilation as isize) * ((k as isize) + 1 - (self.kernel as isize));
                let base_frame_idx = (buffer_start as isize) + offset;

                let w_start0 = ((out_c_base) * self.kernel + k) * self.in_ch;
                let w_start1 = ((out_c_base + 1) * self.kernel + k) * self.in_ch;
                let w_start2 = ((out_c_base + 2) * self.kernel + k) * self.in_ch;
                let w_start3 = ((out_c_base + 3) * self.kernel + k) * self.in_ch;

                let w0 = unsafe { self.weights.get_unchecked(w_start0..w_start0 + self.in_ch) };
                let w1 = unsafe { self.weights.get_unchecked(w_start1..w_start1 + self.in_ch) };
                let w2 = unsafe { self.weights.get_unchecked(w_start2..w_start2 + self.in_ch) };
                let w3 = unsafe { self.weights.get_unchecked(w_start3..w_start3 + self.in_ch) };

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
                        let in_slice = unsafe {
                            layer_buffer.get_unchecked(in_slice_start..in_slice_start + self.in_ch)
                        };

                        let [r0, r1, r2, r3] =
                            unsafe { (math.dot_product_4x)(w0, w1, w2, w3, in_slice) };

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
                        let in_slice = unsafe {
                            layer_buffer.get_unchecked(in_slice_start..in_slice_start + self.in_ch)
                        };

                        let [r0, r1, r2, r3] =
                            unsafe { (math.dot_product_4x)(w0, w1, w2, w3, in_slice) };

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

        // Tail process (remainder)
        while out_c_base < self.out_ch {
            let out_c = out_c_base;
            for k in 0..self.kernel {
                if k + 1 < self.kernel {
                    let next_offset =
                        (self.dilation as isize) * ((k as isize) + 2 - (self.kernel as isize));
                    let next_frame_idx = (buffer_start as isize) + next_offset;
                    let next_addr = unsafe {
                        layer_buffer
                            .as_ptr()
                            .add((next_frame_idx as usize) * self.in_ch)
                    };
                    unsafe {
                        core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(
                            next_addr.cast::<i8>(),
                        );
                    }
                }

                let offset = (self.dilation as isize) * ((k as isize) + 1 - (self.kernel as isize));
                let base_frame_idx = (buffer_start as isize) + offset;
                let weight_slice_start = (out_c * self.kernel + k) * self.in_ch;
                let weight_slice = unsafe {
                    self.weights
                        .get_unchecked(weight_slice_start..weight_slice_start + self.in_ch)
                };

                if k == 0 {
                    let bias = if self.do_bias { self.bias[out_c] } else { 0.0 };
                    for i in 0..num_frames {
                        let frame_idx = base_frame_idx + (i as isize);
                        let in_slice_start = (frame_idx as usize) * self.in_ch;
                        let in_slice = unsafe {
                            layer_buffer.get_unchecked(in_slice_start..in_slice_start + self.in_ch)
                        };
                        unsafe {
                            *block.get_unchecked_mut(i * self.out_ch + out_c) =
                                bias + (math.dot_product)(in_slice, weight_slice);
                        }
                    }
                } else {
                    for i in 0..num_frames {
                        let frame_idx = base_frame_idx + (i as isize);
                        let in_slice_start = (frame_idx as usize) * self.in_ch;
                        let in_slice = unsafe {
                            layer_buffer.get_unchecked(in_slice_start..in_slice_start + self.in_ch)
                        };
                        unsafe {
                            *block.get_unchecked_mut(i * self.out_ch + out_c) +=
                                (math.dot_product)(in_slice, weight_slice);
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
    pub weights: Vec<f32>,
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
    /// Acumula as predições no vetor `output`. Utilizado pelo pipeline interno de blend `input_mixin`.
    ///
    /// # Safety
    /// Depende do `SimdMathConfig` ser válido nativamente.
    pub unsafe fn process_acc_block(
        &self,
        input: &[f32],
        output: &mut [f32],
        num_frames: usize,
        math: &SimdMathConfig,
    ) {
        let mut out_c_base = 0;
        while out_c_base + 4 <= self.out_size {
            let w_start0 = (out_c_base) * self.in_size;
            let w_start1 = (out_c_base + 1) * self.in_size;
            let w_start2 = (out_c_base + 2) * self.in_size;
            let w_start3 = (out_c_base + 3) * self.in_size;

            let w0 = unsafe {
                self.weights
                    .get_unchecked(w_start0..w_start0 + self.in_size)
            };
            let w1 = unsafe {
                self.weights
                    .get_unchecked(w_start1..w_start1 + self.in_size)
            };
            let w2 = unsafe {
                self.weights
                    .get_unchecked(w_start2..w_start2 + self.in_size)
            };
            let w3 = unsafe {
                self.weights
                    .get_unchecked(w_start3..w_start3 + self.in_size)
            };

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
                let in_frame = unsafe {
                    input.get_unchecked(i * self.in_size..i * self.in_size + self.in_size)
                };
                let [r0, r1, r2, r3] = unsafe { (math.dot_product_4x)(w0, w1, w2, w3, in_frame) };
                unsafe {
                    *output.get_unchecked_mut(i * self.out_size + out_c_base) += r0 + bias0;
                    *output.get_unchecked_mut(i * self.out_size + out_c_base + 1) += r1 + bias1;
                    *output.get_unchecked_mut(i * self.out_size + out_c_base + 2) += r2 + bias2;
                    *output.get_unchecked_mut(i * self.out_size + out_c_base + 3) += r3 + bias3;
                }
            }
            out_c_base += 4;
        }

        while out_c_base < self.out_size {
            let out_c = out_c_base;
            let weight_slice =
                &self.weights[out_c * self.in_size..out_c * self.in_size + self.in_size];
            let bias = if self.do_bias { self.bias[out_c] } else { 0.0 };
            for i in 0..num_frames {
                let in_frame = unsafe {
                    input.get_unchecked(i * self.in_size..i * self.in_size + self.in_size)
                };
                let sum = unsafe { (math.dot_product)(in_frame, weight_slice) };
                unsafe {
                    *output.get_unchecked_mut(i * self.out_size + out_c) += sum + bias;
                }
            }
            out_c_base += 1;
        }
    }

    /// Processa o dot product substituindo a memória em `output`.
    ///
    /// # Safety
    /// Requer `SimdMathConfig` validamente instanciado.
    pub unsafe fn process_acc_block_strided(
        &self,
        input: &[f32],
        output: &mut [f32],
        num_frames: usize,
        in_stride: usize,
        out_stride: usize,
        math: &SimdMathConfig,
    ) {
        let mut out_c_base = 0;
        while out_c_base + 4 <= self.out_size {
            let w_start0 = (out_c_base) * self.in_size;
            let w_start1 = (out_c_base + 1) * self.in_size;
            let w_start2 = (out_c_base + 2) * self.in_size;
            let w_start3 = (out_c_base + 3) * self.in_size;

            let w0 = unsafe {
                self.weights
                    .get_unchecked(w_start0..w_start0 + self.in_size)
            };
            let w1 = unsafe {
                self.weights
                    .get_unchecked(w_start1..w_start1 + self.in_size)
            };
            let w2 = unsafe {
                self.weights
                    .get_unchecked(w_start2..w_start2 + self.in_size)
            };
            let w3 = unsafe {
                self.weights
                    .get_unchecked(w_start3..w_start3 + self.in_size)
            };

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
                let in_frame =
                    unsafe { input.get_unchecked(i * in_stride..i * in_stride + self.in_size) };
                let [r0, r1, r2, r3] = unsafe { (math.dot_product_4x)(w0, w1, w2, w3, in_frame) };
                unsafe {
                    *output.get_unchecked_mut(i * out_stride + out_c_base) += r0 + bias0;
                    *output.get_unchecked_mut(i * out_stride + out_c_base + 1) += r1 + bias1;
                    *output.get_unchecked_mut(i * out_stride + out_c_base + 2) += r2 + bias2;
                    *output.get_unchecked_mut(i * out_stride + out_c_base + 3) += r3 + bias3;
                }
            }
            out_c_base += 4;
        }

        while out_c_base < self.out_size {
            let out_c = out_c_base;
            let weight_slice =
                &self.weights[out_c * self.in_size..out_c * self.in_size + self.in_size];
            let bias = if self.do_bias { self.bias[out_c] } else { 0.0 };
            for i in 0..num_frames {
                let in_frame =
                    unsafe { input.get_unchecked(i * in_stride..i * in_stride + self.in_size) };
                let sum = unsafe { (math.dot_product)(in_frame, weight_slice) };
                unsafe {
                    *output.get_unchecked_mut(i * out_stride + out_c) += sum + bias;
                }
            }
            out_c_base += 1;
        }
    }

    /// Processa bloco strided.
    /// # Safety
    /// Pointer must be valid.
    pub unsafe fn process_block_strided(
        &self,
        input: &[f32],
        output: &mut [f32],
        num_frames: usize,
        in_stride: usize,
        out_stride: usize,
        math: &SimdMathConfig,
    ) {
        let mut out_c_base = 0;
        while out_c_base + 4 <= self.out_size {
            let w_start0 = (out_c_base) * self.in_size;
            let w_start1 = (out_c_base + 1) * self.in_size;
            let w_start2 = (out_c_base + 2) * self.in_size;
            let w_start3 = (out_c_base + 3) * self.in_size;

            let w0 = unsafe {
                self.weights
                    .get_unchecked(w_start0..w_start0 + self.in_size)
            };
            let w1 = unsafe {
                self.weights
                    .get_unchecked(w_start1..w_start1 + self.in_size)
            };
            let w2 = unsafe {
                self.weights
                    .get_unchecked(w_start2..w_start2 + self.in_size)
            };
            let w3 = unsafe {
                self.weights
                    .get_unchecked(w_start3..w_start3 + self.in_size)
            };

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
                let in_frame =
                    unsafe { input.get_unchecked(i * in_stride..i * in_stride + self.in_size) };
                let [r0, r1, r2, r3] = unsafe { (math.dot_product_4x)(w0, w1, w2, w3, in_frame) };
                unsafe {
                    *output.get_unchecked_mut(i * out_stride + out_c_base) = r0 + bias0;
                    *output.get_unchecked_mut(i * out_stride + out_c_base + 1) = r1 + bias1;
                    *output.get_unchecked_mut(i * out_stride + out_c_base + 2) = r2 + bias2;
                    *output.get_unchecked_mut(i * out_stride + out_c_base + 3) = r3 + bias3;
                }
            }
            out_c_base += 4;
        }

        while out_c_base < self.out_size {
            let out_c = out_c_base;
            let weight_slice =
                &self.weights[out_c * self.in_size..out_c * self.in_size + self.in_size];
            let bias = if self.do_bias { self.bias[out_c] } else { 0.0 };
            for i in 0..num_frames {
                let in_frame =
                    unsafe { input.get_unchecked(i * in_stride..i * in_stride + self.in_size) };
                let sum = unsafe { (math.dot_product)(in_frame, weight_slice) };
                unsafe {
                    *output.get_unchecked_mut(i * out_stride + out_c) = sum + bias;
                }
            }
            out_c_base += 1;
        }
    }

    /// Processa bloco iterativo.
    /// # Safety
    /// Pointer must be valid.
    pub unsafe fn process_block(
        &self,
        input: &[f32],
        output: &mut [f32],
        num_frames: usize,
        math: &SimdMathConfig,
    ) {
        let mut out_c_base = 0;
        while out_c_base + 4 <= self.out_size {
            let w_start0 = (out_c_base) * self.in_size;
            let w_start1 = (out_c_base + 1) * self.in_size;
            let w_start2 = (out_c_base + 2) * self.in_size;
            let w_start3 = (out_c_base + 3) * self.in_size;

            let w0 = unsafe {
                self.weights
                    .get_unchecked(w_start0..w_start0 + self.in_size)
            };
            let w1 = unsafe {
                self.weights
                    .get_unchecked(w_start1..w_start1 + self.in_size)
            };
            let w2 = unsafe {
                self.weights
                    .get_unchecked(w_start2..w_start2 + self.in_size)
            };
            let w3 = unsafe {
                self.weights
                    .get_unchecked(w_start3..w_start3 + self.in_size)
            };

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
                let in_frame = unsafe {
                    input.get_unchecked(i * self.in_size..i * self.in_size + self.in_size)
                };
                let [r0, r1, r2, r3] = unsafe { (math.dot_product_4x)(w0, w1, w2, w3, in_frame) };
                unsafe {
                    *output.get_unchecked_mut(i * self.out_size + out_c_base) = r0 + bias0;
                    *output.get_unchecked_mut(i * self.out_size + out_c_base + 1) = r1 + bias1;
                    *output.get_unchecked_mut(i * self.out_size + out_c_base + 2) = r2 + bias2;
                    *output.get_unchecked_mut(i * self.out_size + out_c_base + 3) = r3 + bias3;
                }
            }
            out_c_base += 4;
        }

        while out_c_base < self.out_size {
            let out_c = out_c_base;
            let weight_slice =
                &self.weights[out_c * self.in_size..out_c * self.in_size + self.in_size];
            let bias = if self.do_bias { self.bias[out_c] } else { 0.0 };
            for i in 0..num_frames {
                let in_frame = unsafe {
                    input.get_unchecked(i * self.in_size..i * self.in_size + self.in_size)
                };
                let sum = unsafe { (math.dot_product)(in_frame, weight_slice) };
                unsafe {
                    *output.get_unchecked_mut(i * self.out_size + out_c) = sum + bias;
                }
            }
            out_c_base += 1;
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
    /// Engata as equações de passagem, transferindo estados de Buffer Circular de camada em camada.
    ///
    /// Quando `gated == false` (padrão): aplica `tanh(x)` sobre `block[0..ch]`.
    /// Quando `gated == true`: aplica ativação gated `tanh(x[0..ch]) ⊙ sigmoid(x[ch..2*ch])`.
    ///   - O `conv1d` deve ter `out_ch = 2 * ch`; `input_mixin` acumula apenas nos primeiros `ch`.
    ///   - Resultado gated em `block[0..ch]` é somado ao head e passado ao one_by_one.
    ///
    /// # Safety
    /// Requer instâncias estritas do buffer interno e `block` com tamanho
    /// `ch` (não-gated) ou `2*ch` (gated).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn process_block_internal(
        &self,
        condition: &[f32],
        head_input: &mut [f32],
        output: &mut [f32],
        layer_buffer: &[f32],
        buffer_start: usize,
        block: &mut [f32],
        num_frames: usize,
        math: &SimdMathConfig,
    ) {
        let ch = self.ch;

        for v in block.iter_mut() {
            *v = 0.0;
        }

        unsafe {
            self.conv1d
                .process_block(layer_buffer, block, buffer_start, num_frames, math);

            if self.gated {
                self.input_mixin.process_acc_block_strided(
                    condition,
                    block,
                    num_frames,
                    1,
                    2 * self.ch,
                    math,
                );

                for i in 0..num_frames {
                    let block_start = i * self.conv1d.out_ch;
                    let block_frame = &mut block[block_start..block_start + self.conv1d.out_ch];

                    let (z1, z2) = block_frame.split_at_mut(self.ch);
                    (math.tanh_slice)(z1);
                    (math.sigmoid_slice)(z2);

                    for j in 0..self.ch {
                        *z1.get_unchecked_mut(j) *= *z2.get_unchecked(j);
                    }
                }
            } else {
                self.input_mixin
                    .process_acc_block_strided(condition, block, num_frames, 1, self.ch, math);
                (math.tanh_slice)(&mut block[0..num_frames * self.ch]);
            }

            for i in 0..num_frames {
                let head_frame = &mut head_input[i * ch..i * ch + ch];
                let block_frame = &block[i * (if self.gated { 2 * ch } else { ch })
                    ..i * (if self.gated { 2 * ch } else { ch }) + ch];
                for j in 0..ch {
                    *head_frame.get_unchecked_mut(j) += *block_frame.get_unchecked(j);
                }
            }

            // in_stride depende do layout do block: 2*ch quando gated (slots tanh+sigmoid),
            // ch quando não-gated (apenas slot tanh). Usar stride errado faz o one_by_one
            // ler de offsets incorretos, produzindo saída silenciosamente corrompida.
            let in_stride = if self.gated { 2 * self.ch } else { self.ch };
            self.one_by_one
                .process_block_strided(block, output, num_frames, in_stride, self.ch, math);
        }

        unsafe {
            for i in 0..num_frames {
                let out_frame = output.get_unchecked_mut(i * ch..i * ch + ch);
                let lb_start = (buffer_start + i) * ch;
                for j in 0..ch {
                    *out_frame.get_unchecked_mut(j) += *layer_buffer.get_unchecked(lb_start + j);
                }
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
    /// Realiza inferência síncrona não recursiva entre todas as camadas do Array. Mutação sem alocação em heap.
    ///
    /// # Safety
    /// Depende nativamente das matrizes preenchidas via alocador estrito no C++ Fallback parser (Loader CLI).
    pub unsafe fn process(
        &mut self,
        layer_inputs: &[f32],
        condition: &[f32],
        num_frames: usize,
        math: &SimdMathConfig,
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

        for v in self.head_accum.iter_mut() {
            *v = 0.0;
        }

        unsafe {
            let state_0 = &mut *states_ptr.add(0);
            let start = state_0.buffer_start * ch;
            self.rechannel.process_block(
                layer_inputs,
                &mut state_0.layer_buffer[start..start + num_frames * ch],
                num_frames,
                math,
            );

            let num_layers = self.layers.len();
            let last_layer = num_layers - 1;

            let block_size = self.block_size;

            for (i, layer) in self.layers.iter().enumerate() {
                let current_state = &mut *states_ptr.add(i);

                if i == last_layer {
                    layer.process_block_internal(
                        condition,
                        &mut self.head_accum[0..num_frames * ch],
                        &mut self.array_outputs[0..num_frames * ch],
                        &current_state.layer_buffer,
                        current_state.buffer_start,
                        &mut self.block_buffer[0..num_frames * block_size],
                        num_frames,
                        math,
                    );
                } else {
                    let next_state = &mut *states_ptr.add(i + 1);
                    let next_start = next_state.buffer_start * ch;

                    layer.process_block_internal(
                        condition,
                        &mut self.head_accum[0..num_frames * ch],
                        &mut next_state.layer_buffer[next_start..next_start + num_frames * ch],
                        &current_state.layer_buffer,
                        current_state.buffer_start,
                        &mut self.block_buffer[0..num_frames * block_size],
                        num_frames,
                        math,
                    );
                }

                current_state.advance_frames(num_frames, ch);
            }

            self.head_rechannel.process_block(
                &self.head_accum[0..num_frames * ch],
                &mut self.head_outputs[0..num_frames * head],
                num_frames,
                math,
            );
        }
    }

    /// Preenche os buffers para aquecimento inicial (Pre-warm).
    pub fn prewarm(&mut self, layer_inputs: &[f32], condition: &[f32], math: &SimdMathConfig) {
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

        for v in self.head_accum.iter_mut() {
            *v = 0.0;
        }

        unsafe {
            let state_0 = &mut *states_ptr.add(0);
            let start = state_0.buffer_start * ch;
            self.rechannel.process_block(
                layer_inputs,
                &mut state_0.layer_buffer[start..start + ch],
                1, // prewarm is 1 frame
                math,
            );

            let num_layers = self.layers.len();
            let last_layer = num_layers - 1;

            let block_size = self.block_size;

            for (i, layer) in self.layers.iter().enumerate() {
                let current_state = &mut *states_ptr.add(i);
                current_state.copy_buffer(ch);

                if i == last_layer {
                    layer.process_block_internal(
                        condition,
                        &mut self.head_accum[0..ch],
                        &mut self.array_outputs[0..ch],
                        &current_state.layer_buffer,
                        current_state.buffer_start,
                        &mut self.block_buffer[0..block_size],
                        1,
                        math,
                    );
                } else {
                    let next_state = &mut *states_ptr.add(i + 1);
                    let next_start = next_state.buffer_start * ch;

                    layer.process_block_internal(
                        condition,
                        &mut self.head_accum[0..ch],
                        &mut next_state.layer_buffer[next_start..next_start + ch],
                        &current_state.layer_buffer,
                        current_state.buffer_start,
                        &mut self.block_buffer[0..block_size],
                        1,
                        math,
                    );
                }
            }

            self.head_rechannel.process_block(
                &self.head_accum[0..ch],
                &mut self.head_outputs[0..head],
                1,
                math,
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
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        let math = crate::math::simd::SimdMathConfig::get();
        let total_frames = input.len();

        let mut pos = 0;
        while pos < total_frames {
            let num_frames =
                (total_frames - pos).min(crate::models::wavenet::WAVENET_MAX_NUM_FRAMES);
            let in_slice = &input[pos..pos + num_frames];

            unsafe {
                self.array1.process(in_slice, in_slice, num_frames, math);

                let array1_outputs = &self.array1.array_outputs[0..num_frames * self.array1.ch];
                self.array2
                    .process(array1_outputs, in_slice, num_frames, math);
            }

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
        let math = crate::math::simd::SimdMathConfig::get();
        let condition = [0.0f32];
        let layer_inputs_1 = [0.0f32];

        self.array1.prewarm(&layer_inputs_1, &condition, math);
        let array1_outputs = &self.array1.array_outputs[..];
        self.array2.prewarm(array1_outputs, &condition, math);
    }
}

// =============================================================================
// Testes Unitários
// =============================================================================

#[cfg(test)]
#[path = "wavenet_dyn_test.rs"]
mod wavenet_dyn_test;
