<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Épico: RT-Safety & SIMD (P15 + O3a)

**Foco do Épico**: Erradicar `expect()` da inicialização A2 para garantir 100% de RT-Safety no ciclo de carregamento e implementar a Fase 1 da vetorização do Cabsim (MAC complexo SIMD).

## Sprint 1: P15 — Erradicação de `expect()` no A2 [DONE]

**Risco**: Alto (mudança de assinatura da trait `NamModel` afeta todos os modelos e a inicialização do CLAP host).
**Diretriz Central**: Substituir `expect` por `Result` no construtor `WaveNetA2` e propagar erros da alocação de memória virtual (`MirroredBuffer`) até o loop de eventos RT do host.

- [x] **Tarefa 1.1 (Trait NamModel)**: Em `src/models/mod.rs`, alterar as assinaturas na trait `NamModel` para que os métodos de redimensionamento possam falhar:
  - De `fn set_max_buffer_size(&mut self, _max_buf: usize)` para `fn set_max_buffer_size(&mut self, _max_buf: usize) -> anyhow::Result<()> { Ok(()) }`
  - De `fn reset(&mut self, ...)` para `fn reset(&mut self, ...) -> anyhow::Result<()> { self.prewarm(max_buffer_size); Ok(()) }`
- [x] **Tarefa 1.2 (StaticModel e Container)**: Atualizar a propagação em `src/models/mod.rs` (enum `StaticModel`) e em `src/models/container.rs` (`ContainerModel`), fazendo com que o `set_max_buffer_size` e `reset` repassem os erros dos submodelos via `?` ou loop com verificação. Atualizar `Lstm`, `Linear` e `WaveNetModel` genérico para retornarem `Ok(())`.
- [x] **Tarefa 1.3 (A2 Constructor)**: Em `src/models/a2/model/mod.rs`:
  - Remover `impl Default for WaveNetA2`.
  - Alterar `pub fn new() -> anyhow::Result<Self>`.
  - Substituir `.expect("MirroredBuffer...")` por `?`.
- [x] **Tarefa 1.4 (A2 Resize)**: No mesmo arquivo, alterar `set_max_buffer_size` para retornar `anyhow::Result<()>`. Repassar falhas de realocação usando `?`.
- [x] **Tarefa 1.5 (Dispatcher)**: Em `src/loader/dispatcher/wavenet/mod.rs`, nas linhas ~64 e ~75, tratar a instanciação `WaveNetA2::<3>::new()?` e `WaveNetA2::<8>::new()?` (o método `build_wavenet` já retorna `anyhow::Result`).
- [x] **Tarefa 1.6 (Host RT Loop)**: Em `src/clap/processor/events.rs`, no método `cold_load_model` (linha ~173), tratar a chamada `model.set_max_buffer_size(self.max_frames_count)`.
  - **Crítico para RT-Safety**: O RT-thread *não pode* usar `log::error!` nem dar `panic`. Se `set_max_buffer_size` retornar `Err`, você deve descartar o modelo (`self.push_to_gc(GcItem::Model(self.model_l.take().unwrap()))`) e ativar a flag de erro na atômica compartilhada (`self.rt_status.set_flag(crate::common::spsc::RT_STATUS_MODEL_LOAD_FAILED)`).

## Sprint 2: O3a — MAC Complexo SIMD no Cabsim (Fase 1)

**Risco**: Médio-Alto (manipulação de ponteiros intrínsecos AVX2 e layout SoA - Struct of Arrays).
**Diretriz Central**: Converter buffers FDL e H_FDL para Struct-of-Arrays (separando `re` e `im`) e usar FMA (`_mm256_fmadd_ps`) no domínio da frequência.

- [ ] **Tarefa 2.1 (Estruturas SoA)**: Em `src/dsp/cabsim/conv.rs`, refatorar a estrutura `ConvEngine`. Substituir `h_fdl: AlignedVec<f32>` e `fdl: AlignedVec<f32>` por 4 vetores paralelos com a metade do tamanho (sem o multiplicador `* 2`):
  - `h_fdl_re: AlignedVec<f32>` e `h_fdl_im: AlignedVec<f32>`
  - `fdl_re: AlignedVec<f32>` e `fdl_im: AlignedVec<f32>`
  - Modificar o acumulador para ser SoA nativamente para evitar unpack antes da IFFT: `acc_re: AlignedVec<f32>` e `acc_im: AlignedVec<f32>`.
- [ ] **Tarefa 2.2 (Inicialização)**: Atualizar `ConvEngine::new()`. Ao processar a FFT das partições de impulso (IR), gravar os componentes iterando em `fft_buf`:
  - `h_fdl_re[base + k] = c.re;`
  - `h_fdl_im[base + k] = c.im;`
- [ ] **Tarefa 2.3 (Alimentação RT)**: Em `process()` (Passo 3), quando extrair resultados de `self.fft_buf`, salvar os componentes separadamente em `self.fdl_re` e `self.fdl_im`.
- [ ] **Tarefa 2.4 (Laço MAC AVX2)**: Implementar o MAC SIMD vetorizado em `process()` (Passo 4).
  - Criar bloco `#[cfg(target_arch = "x86_64")]` para usar intrínsecos AVX2 (`core::arch::x86_64::*`).
  - Iterar `k` em steps de `8`. Usar ponteiros `.as_ptr().add(k)`.
  - Usar `_mm256_load_ps` para carregar 8 floats de `x_re`, `x_im`, `h_re`, `h_im`.
  - Calcular parte real: `re_prod = _mm256_mul_ps(h_re, x_re)`. Depois `re_res = _mm256_fnmadd_ps(h_im, x_im, re_prod)` (que faz `re_prod - h_im * x_im`).
  - Calcular parte imaginária: `im_prod = _mm256_mul_ps(h_re, x_im)`. Depois `im_res = _mm256_fmadd_ps(h_im, x_re, im_prod)` (que faz `h_im * x_re + im_prod`).
  - Acumular iterativamente (somar ao acumulador global do particionamento) usando `_mm256_add_ps` com os valores carregados de `acc_re` e `acc_im`, depois `_mm256_store_ps`.
- [ ] **Tarefa 2.5 (Reconstrução IFFT)**: No Passo 5, antes de passar o acumulador para a `ifft.process_with_scratch`, mesclar temporariamente os resultados de `acc_re` e `acc_im` em `self.acc` (tipo `Complex<f32>`), uma vez que a biblioteca `rustfft` requer estrutura intercalada.
- [ ] **Tarefa 2.6 (Validação)**: Executar `cargo test --release --test conv_test`. Validar a ausência de distorção de fase e amplitude no modo `x86_64-v3` vs fallback escalar.
