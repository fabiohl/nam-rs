<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Épico: Erradicação do `rustfft` e RFFT Nativa

**Foco do Épico**: Remover por completo a dependência externa pesada `rustfft`, substituindo-a por um motor in-house focado estritamente em tamanhos de "potência de 2" (Radix-2/Radix-4). Essa alteração viabiliza a integração transparente com o novo layout SoA (Struct-of-Arrays) construído no Épico O3a, mitigando conversões de array, garantindo aderência absoluta de RT-Safety (Zero-Alloc) no hot-path, e melhorando a densidade de cache do DSP.

**Importante**: Use e abuse da feature level x86-64-v3 com otimizações SIMD AVX-2, FMA e outras.

## Sprint 1: Núcleo FFT Nativo (`src/math/dsp/fft.rs`)

**Diretriz Central**: Desenvolver algoritmos *Radix-2 Cooley-Tukey* decimation-in-time in-place e algoritmos reais (RFFT/IRFFT) customizados, abstraindo `f32` e `f64`.

- [x] **Tarefa 1.1 (Base Complexa In-Place)**: Criar módulo `src/math/dsp/fft.rs` contendo a FFT Complexa genérica (`f32` e `f64`).
  - Desenvolver a inicialização (`new`) que pré-calcula:
    1. Arrays estáticos de **Bit-Reversal** baseados no tamanho $N$.
    2. Fatores de rotação (**Twiddle Factors**): matrizes de senos e cossenos pré-computados (para evitar recomputar `sin/cos` na thread de áudio).
  - Implementar o algoritmo DIT Radix-2 iterativo ascendente iterando em arrays SoA (`re: &mut [T], im: &mut [T]`). Nenhuma alocação `Vec` pode ocorrer na rotina `process`.
- [x] **Tarefa 1.2 (RFFT - Real to Complex)**: Implementar wrapper RFFT que converte um buffer puramente real de tamanho $N$ num espectro complexo compacto de tamanho $N/2 + 1$.
  - Método: Empacotar array real de tamanho $N$ num array complexo imaginário de tamanho $N/2$, executar a FFT complexa de $N/2$ e, no final, aplicar um passo $O(N)$ post-processing para separar os espectros originais via simetria Hermitiana.
  - Assinatura alvo: `fn process_forward(&mut self, input: &[T], out_re: &mut [T], out_im: &mut [T])` (scratch buffers pré-alocados em `new`, reutilizados no hot-path para RT-safety).
- [x] **Tarefa 1.3 (IRFFT - Complex to Real)**: Implementar a operação reversa (IRFFT).
  - Método: Aplicar simetria reversa $O(N)$ (pre-processing) combinando `re` e `im` em um sinal empacotado de $N/2$, chamar a FFT Inversa complexa $N/2$ e extrair os valores reais intercalados.
  - Assinatura alvo: `fn process_inverse(&self, in_re: &mut [T], in_im: &mut [T], out: &mut [T])`. O ganho de scaling ($1/N$ ou $2/N$) deve estar embutido no loop final.

## Sprint 2: Integração RT — Cabsim ConvEngine (`src/dsp/cabsim/conv.rs`)

**Diretriz Central**: Conectar o novo motor RFFT ao Cabsim e eliminar a contaminação residual de `Complex<T>` intercalado.

- [x] **Tarefa 2.1 (Substituição de Estruturas)**: Em `conv.rs`:
  - Apagar importações de `rustfft`.
  - Substituir `rustfft::FftPlanner` por `crate::math::dsp::fft::RfftPlanner<f32>`.
  - Instanciar a nova RFFT na função `ConvEngine::new()` com o tamanho `partition_size * 2`.
- [ ] **Tarefa 2.2 (Refatoração SoA Total)**: O buffer `fft_buf` (que era um array de blocos `Complex<f32>` alocados para acomodar os requisitos do `rustfft`) deve ser transformado em dois arrays escalares independentes: `fft_buf_re: AlignedVec<f32>` e `fft_buf_im: AlignedVec<f32>`, preservando o alinhamento de 64 bytes.
- [ ] **Tarefa 2.3 (Alimentação RT do Forward)**: No `process()` (Passo 2), alimentar a convolução não mais empacotando structs nativas, mas injetando o frame em tempo real diretamente em `rfft.process_forward(input_slice, &mut self.fft_buf_re, &mut self.fft_buf_im)`.
- [ ] **Tarefa 2.4 (Reconstrução IFFT)**: No `process()` (Passo 5), passar o acumulador SoA (que processou o SIMD na Sprint 2) diretamente para `irfft.process_inverse(&mut acc_re, &mut acc_im, &mut output_slice)`. Remover o bypass temporário de mixagem. Executar os testes `conv_test`.

## Sprint 3: Migração Offline e Métricas Perceptuais (`sinc_kernel` & `perceptual`)

**Diretriz Central**: Substituir as ocorrências offline `f64` de rustfft por soluções internas SoA.

- [ ] **Tarefa 3.1 (Migração do Cepstrum Real)**: Em `src/dsp/sinc_kernel.rs`, reescrever `to_minimum_phase()` (algoritmo iterativo de convolução logarítmica).
  - Instanciar o novo `FftPlanner<f64>`.
  - Como a implementação nova recebe e emite vetores escalares desacoplados, alterar o payload que manipula `Vec<Complex<f64>>` por dois arrays `buf_re` e `buf_im`. Computar a raiz quadrada (log-magnitude) acessando iterativamente `buf_re[i]` e `buf_im[i]`.
- [ ] **Tarefa 3.2 (Causal Truncation Fix)**: Ajustar a indexação das cópias no truncate de cepstrum em `sinc_kernel.rs` e garantir que `im` é mantido zerado antes da segunda `process_forward`.
- [ ] **Tarefa 3.3 (Migração STFT e LUFS)**: Em `src/testing/perceptual.rs`, na rotina `compute_mr_stft()`:
  - Usar os métodos do `FftPlanner<f64>` interno.
  - Tratar a divisão de arrays SoA na extração do módulo L1 e L2 dentro das janelas (256, 1024, 4096).
  - Validar contra golden metrics via `cargo test` assegurando a integridade perante os hardcodes em `A2ESR_A2_FULL_MEDIAN`.

## Sprint 4: Faxina, Auditoria e Validação

**Diretriz Central**: Desacoplar oficialmente a crate legada, inspecionar a RT-Safety e avaliar ganhos.

- [ ] **Tarefa 4.1 (Clean Cargo.toml)**: Remover `rustfft = ...` do arquivo `Cargo.toml`. Executar `cargo update` caso existam traços da dependência na lockfile.
- [ ] **Tarefa 4.2 (Auditoria O3b)**: Rodar `utils/lints.sh`. Nenhum alerta de performance ou linter derivado da nova matemática em SoA deve subsistir.
- [ ] **Tarefa 4.3 (Testes Anti-Alloc)**: Rodar a suíte `utils/tests-cargo.sh`. O rastreador de heap detectará imediatamente se o novo FFT tentar alocar qualquer `Vec` sob a thread do SPSC loop de processamento do áudio (`RT`).
- [ ] **Tarefa 4.4 (Benchmarking Golden)**: Rodar o motor em modo `cargo bench` garantindo o cômputo `benchmark_convengine` com ganho perceptivo em nano-segundos, finalizando o selo arquitetural da Fase O3b.
