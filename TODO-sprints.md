<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-sprints.md — Planejamento Ágil dos Épicos

Este documento organiza o trabalho técnico derivado de auditorias e propostas registradas em [TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md) em épicos, sprints e tarefas técnicas detalhadas.

---

## ÉPICO D — Higiene do thread de tempo-real

### Visão Geral e Riscos

Este épico foca exclusivamente no achado [F6](file:///home/fabio/nam-rs/TODO-findings.md#L309) de [TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md). O objetivo é garantir o determinismo temporal absoluto do thread de tempo-real (RT), eliminando varreduras redundantes de memória e cópias de buffers desnecessárias no caminho crítico do PipeWire e do backbone WaveNet.

> [!IMPORTANT]
> **Risco de Regressão Crítico (RT-Safety):** Nenhuma alteração no thread de áudio pode introduzir alocações dinâmicas de heap (`malloc`/`free`), syscalls ou bloqueios (mutexes/locks). Qualquer quebra dessas premissas resultará em *xruns* (glitches de áudio). A paridade numérica dos testes unitários e de integração precisa ser mantida a 100%.

---

### Sprint D1: Box de Buffers e Fatoração de Cópias de Taps (F6)

- **[x] Tarefa D1.1 — Eliminar Varredura/Cópia de 196 KB na Callback PipeWire**
  - **Conclusão:** Seis buffers `[f32; 8192]` em `CaptureState` convertidos para `Box<[f32; 8192]>`, alocados no heap durante `CaptureState::init` (main thread). O `DspBuffers` em `setup.rs` agora usa `&mut *state.resamp_mid_l` para dereferenciar o `Box`. O tamanho do ambiente da closure caiu de ~196 KB para ~1 KB (apenas 6 ponteiros de 8 bytes = 48 bytes + campos escalares). `cargo check`, `cargo test` (pipeline, gate, zero_alloc) — todos passam. RT-safe: zero alocação no hot-path.
  - **Foco:** Corrigir a cópia e varredura gerada pelo tamanho excessivo de `CaptureState` na pilha (devido a seis arrays estáticos `[f32; MAX_RESAMP_BUF]` de 32 KB cada).
  - **Ação:**
    - Modificar [state.rs](file:///home/fabio/nam-rs/src/standalone/pw_host/capture/state.rs) para alterar os buffers `resamp_mid_l`, `resamp_mid_r`, `resamp_out_l`, `resamp_out_r`, `model_out_l` e `model_out_r` do tipo `[f32; MAX_RESAMP_BUF]` para `Box<[f32]>` (ou `Box<[f32; MAX_RESAMP_BUF]>`).
    - Alocar esses buffers no heap durante a inicialização em `CaptureState::init` (que roda fora do thread RT, sob o controle do main thread).
    - Ajustar a passagem em [setup.rs](file:///home/fabio/nam-rs/src/standalone/pw_host/capture/setup.rs) para obter referências mutáveis aos fatiamentos via `&mut *state.resamp_mid_l`, etc.
  - **Validação:** Compilar com `cargo check`. O tamanho do ambiente da closure deve cair de ~196 KB para menos de 1 KB.

- **[x] Tarefa D1.2 — Otimização de Cópia de Taps no Conv1D Estático**
  - **Conclusão:** As 4 chamadas de `copy_from_slice` (2 em `conv1d.rs`, 2 em `conv1d_dual.rs`) foram substituídas por `std::ptr::copy_nonoverlapping` com contagem constante igual ao generic parameter `IN`. A otimização de propagação de constantes do LLVM agora expande a cópia diretamente para instruções SIMD (`vmovups`/`vmovdqu`), eliminando a chamada externa a `memcpy` da libc do caminho crítico RT. `cargo check` limpo (sem warnings), 91 testes unitários do WaveNet passaram.
  - **Foco:** Evitar chamadas externas a `memcpy`/`memmove` em cópias de tamanho constante.
  - **Ação:**
    - Em [conv1d.rs](file:///home/fabio/nam-rs/src/models/wavenet/conv1d.rs) e [conv1d_dual.rs](file:///home/fabio/nam-rs/src/models/wavenet/conv1d_dual.rs), substituir `in_tap.copy_from_slice(...)` e `in_taps_f0[k].copy_from_slice(...)` por `std::ptr::copy_nonoverlapping` com contagem constante igual ao generic parameter `IN`.
    - Garantir que a otimização de propagação de constantes do LLVM abaixe a cópia diretamente para instruções `vmovups`/`vmovdqu` do AVX2, sem chamadas externas para `memcpy` da libc.
  - **Validação:** Confirmar com `cargo build` e executar testes unitários do WaveNet.

---

### Sprint D2: Fused Accumulator Seeding e Validação de Estresse (F6)

- **[x] Tarefa D2.1 — Fusão do Seed Copy do head_accum no WaveNet**
  - **Conclusão:** Adicionado campo `seed: Option<&'a [f32]>` ao `WavenetProcessContext`. Implementada operação `tanh_and_accumulate_with_seed` (scalar, AVX2, AVX-512) que computa `head[i] = seed[i] + tanh(block[i])` em uma única passada SIMD, eliminando o `copy_from_slice` de `num_frames * CH` bytes no hot-path. O `layer_array.rs` e `layer_array_dyn.rs` agora passam `prev_head_outputs` como seed no contexto do primeiro layer, e o `layer.rs`/`layer_dyn.rs` despacham para a operação fundida quando seed está presente. 842 testes passam (0 falhas), incluindo 32 unitários do WaveNet, 18 de integração, 7 de zero-alloc, e o novo `test_tanh_and_accumulate_with_seed` cobrindo 10 comprimentos em AVX2 e AVX-512.
  - **Foco:** Eliminar a chamada `copy_from_slice` dedicada de `num_frames * CH` bytes no início do processamento de blocos de `WaveNetLayerArray`.
  - **Ação:**
    - Adicionar um campo opcional de semente `seed: Option<&'a [f32]>` na estrutura `WavenetProcessContext` em [common.rs](file:///home/fabio/nam-rs/src/models/wavenet/common.rs).
    - Modificar a chamada do cascade de inferência em [layer_array.rs](file:///home/fabio/nam-rs/src/models/wavenet/layer_array.rs) (e [layer_array_dyn.rs](file:///home/fabio/nam-rs/src/models/wavenet/layer_array_dyn.rs)) para não copiar mais o seed diretamente em `self.head_accum`. Em vez disso, passar `prev_head_outputs` no contexto do primeiro layer (`i == 0`).
    - Modificar a trait `SimdMath` em [traits.rs](file:///home/fabio/nam-rs/src/math/common/traits.rs) e suas implementações ([avx2_impl.rs](file:///home/fabio/nam-rs/src/math/common/avx2_impl.rs), [avx512/activations.rs](file:///home/fabio/nam-rs/src/math/common/avx512/activations.rs) e [scalar_ref.rs](file:///home/fabio/nam-rs/src/math/common/scalar_ref.rs)) para incluir a operação fundida `tanh_and_accumulate_with_seed`.
    - Atualizar a lógica do primeiro layer em [layer.rs](file:///home/fabio/nam-rs/src/models/wavenet/layer.rs) (e [layer_dyn.rs](file:///home/fabio/nam-rs/src/models/wavenet/layer_dyn.rs)) para usar essa nova operação se a semente estiver presente, ou `tanh_and_overwrite_block` caso contrário.
  - **Validação:** Executar testes unitários e de paridade do WaveNet para assegurar que a saída numérica permaneça idêntica bit-a-bit.

- **[x] Tarefa D2.2 — Validação de Estresse, Heap Audit e Estabilidade RT**
  - **Conclusão:** Validação completa aprovada em todos os níveis. **(1)** `utils/tests-quick.sh`: 5/5 fases aprovadas — testes unitários, C++ parity (31/31), proptest parsers (13/13), proptest math (3/3), build CLAP heap-audit, testes CLAP (76/77, 1 ignorado proposital), heap-audit (12/12: A2 ×2, cabsim ×4, resampler, diagnostic bundle), clap-validator (19/19 pass, 2 skip por ausência de note-ports). **(2)** Soak test `pipeline_soak`: 3/3 passaram (A1-Nano 10M frames, A2-Lite 5M, A2-Full 2M) — zero NaN/Inf, zero regressão de geração, variação RSS ≤3.7 MB (limite 10 MB), latência máx 65.5 µs (P99 32.8 µs). **(3)** Standalone PipeWire em buffer=64: inicialização correta, RT thread SCHED_FIFO prio=83 em core dedicado, mlockall + huge pages ativos, latência DSP máx 106 µs (gate ativo), 0 µs (gate fechado), **zero xruns**. **(4)** Testes zero-alloc (7/7): pipeline de captura, WaveNet estático/dinâmico, LSTM, container, modelos nondist — todos zero-alloc no hot-path. **RT-Safety confirmada:** zero heap drop, zero vazamento, 100% integridade numérica.

---

## ÉPICO E — Refinamento Pós-Implementação (Findings F8–F12)

### E - Visão Geral e Riscos

Este épico foca nos achados identificados na análise pós-Épicos A–D, documentados em
[TODO-findings.md §4](file:///home/fabio/nam-rs/TODO-findings.md#L480):
[F8](file:///home/fabio/nam-rs/TODO-findings.md#F8), [F9](file:///home/fabio/nam-rs/TODO-findings.md#F9),
[F10](file:///home/fabio/nam-rs/TODO-findings.md#F10), [F11](file:///home/fabio/nam-rs/TODO-findings.md#F11),
[F12](file:///home/fabio/nam-rs/TODO-findings.md#F12).

O objetivo é corrigir a regressão de performance no WaveNet Dinâmico (o item mais urgente),
eliminar overhead de zeragem no hot-path, investigar a margem de fidelidade auditiva e resolver
residuais de cópia de buffer.

> [!IMPORTANT]
> **Risco de Regressão Crítico (F9):** A correção de taps não-contíguos no caminho dinâmico
> deve ser validada bit-a-bit contra os golden vectors do caminho dinâmico (`test_golden_vectors_wavenet_dyn_free`,
> `test_a2_dynamic_blended_ch3`, `test_a2_dynamic_gated_ch8`). Qualquer refatoração que altere a
> ordem de acumulação pode afetar o ESR.
>
> **Risco de Paridade (F10):** A Fidelity Margin ≤ 0.5 dB deve ser investigada com `git bisect`
> antes de qualquer ajuste de limiar. Não afrouxar thresholds sem evidência de que é pré-existente.

---

### Sprint E1: Regressão Dinâmica e Zeragem de Bloco (F8 + F9) — Alta Prioridade

- **[ ] Tarefa E1.1 — Corrigir regressão WaveNet Dinâmico — taps não-contíguos (F9 / B.4.1)**
  - **Foco:** Eliminar a cópia de taps para buffer intermediário no caminho dinâmico que causa
    regressão de +53-57% para shapes pequenos (CH3, CH5).
  - **Ação:**
    1. Auditar `src/models/wavenet/conv1d_dyn_dual.rs` e `conv1d_dyn.rs`: identificar onde a
       cópia para buffer temporário de pilha foi adicionada pelo Épico B para taps não-contíguos.
    2. **Opção A (atalho por threshold):** para `num_frames ≤ FRAMES_THRESHOLD` (avaliar 2 ou 4),
       usar o caminho K-taps-sequencial original (sem buffer intermediário). O overhead de cópia
       domina sobre a fusão para batches pequenos.
    3. **Opção B (kernel sem cópia):** implementar variante do kernel de acumulação fundida que
       aceite ponteiros por tap (`&[*const f32; K]`) em vez de slice contíguo — eliminando a cópia.
    4. **Opção C (especialização const):** para shapes usados por modelos reais (CH3, CH5, CH8),
       adicionar const-generic specializations no dispatcher dinâmico.
  - **Validação:** Criterion `WaveNet_Dynamic_CH5_64samp` deve igualar ou superar `A2Full_CH8_64samp`
    (26.98 µs). Todos os golden vectors dinâmicos devem passar bit-a-bit.

- **[ ] Tarefa E1.2 — Eliminar PLT `memset` no prólogo de `WaveNetLayer` (F8)**
  - **Foco:** Substituir a zeragem de 4 KB via PLT `callq *GOT/memset` a cada chamada de
    `WaveNetLayer::process_block_internal` por uma abordagem inline.
  - **Ação:**
    1. Identificar exatamente onde o compilador emite o zero: `src/models/wavenet/layer.rs`
       (o `block_buffer` recebido como `&mut [f32]` precisa estar zerado antes de `tanh_and_accumulate`).
    2. Verificar se o caminho `is_first_layer = true` (que usa `tanh_and_overwrite`) **sobrescreve**
       completamente o buffer sem lê-lo antes — se sim, a zeragem para esse caminho é redundante.
    3. Para o caminho de acumulação (layers seguintes): substituir `copy_from_slice`/zero por
       `core::ptr::write_bytes(block.as_mut_ptr(), 0, block.len())` com o tamanho como `const`
       (propaga via `num_frames * block_size` onde `block_size = CH` e `num_frames ≤ 64` são
       conhecidos em tempo de compilação nos caminhos estáticos).
    4. Nos caminhos onde o buffer é completamente sobrescrito antes de ser lido: usar
       `assume_init` / `MaybeUninit` e eliminar a zeragem.
  - **Validação:** Confirmar via `perf annotate` que o símbolo `__memset_avx2_unaligned_erms`
    desapareceu do hot path. Soak test + golden vectors obrigatórios.

---

### Sprint E2: Paridade e Residuais (F10 + F11 + F12) — Média Prioridade

- **[ ] Tarefa E2.1 — Investigar Fidelity Margin ≤ 0.5 dB (F10)**
  - **Foco:** Determinar se a `Fidelity Margin` próxima de zero em `live_cross_validation_lstm_dyn_1x7 (v2)`
    e `live_cross_validation_linear (v2)` é pré-existente ou introduzida pelos Épicos B/C.
  - **Ação:**
    1. Executar `cargo test --release --test cpp_parity -- --ignored --nocapture` no commit imediatamente
       anterior à implementação do Épico B (usar `git stash` ou `git bisect`).
    2. Comparar os números de Fidelity Margin/MR-STFT antes e depois.
    3. Se novo: identificar o kernel específico via `cargo test` granular em cada subconjunto de mudanças.
    4. Documentar o resultado com comentário no `threshold_calibration.rs`.
  - **Validação:** Decisão binária: pré-existente (documentar e fechar) ou novo (corrigir kernel).

- **[ ] Tarefa E2.2 — Tail GEMV via `_mm256_maskstore_ps` (F11)**
  - **Foco:** Substituir o loop de `vinsertps` no tratamento de cauda do GEMV por masked load/store
    AVX2 (`_mm256_maskload_ps` / `_mm256_maskstore_ps`).
  - **Ação:**
    1. Em `src/math/gemm/gemv/f32_avx2.rs`: identificar o loop de `vinsertps` no tail handler do
       caminho `out_len >= 8` (o trecho que trata `out_c` para `out_len % 8 ≠ 0`).
    2. Substituir por máscara AVX2 pré-calculada: `let mask = _mm256_set_epi32(...)` baseado em
       `tail_len = out_len % 8`, usando `_mm256_maskload_ps` para leitura segura e
       `_mm256_maskstore_ps` para escrita mascarada.
    3. Verificar se `_mm256_maskstore_ps` é seguro quando o buffer de saída tem exatamente `out_len`
       floats (sem padding): pode exigir um buffer temporário de 8 floats ou ajuste de layout.
    4. Aplicar o mesmo padrão ao `gemv_no_bias_f32_avx2`.
  - **Validação:** Criterion `gemv_with_bias_*` e `gemv_no_bias_*`, `golden_vectors`, `threshold_calibration`.

- **[ ] Tarefa E2.3 — Auditar e eliminar `memmove` residual em caminhos de cópia constante (F12)**
  - **Foco:** Identificar chamadas de `copy_from_slice`/`copy_within` não cobertas por D1.2 que ainda
    geram `__memmove_avx_unaligned_erms` no hot path.
  - **Ação:**
    1. Executar: `rg 'copy_from_slice|copy_within' src/models/wavenet/ src/dsp/ --line-number`
    2. Para cada ocorrência, determinar se o tamanho é constante (baseado em `const generics`) ou variável.
    3. Para tamanhos constantes: substituir por `std::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr(), CONST_N)`.
    4. Para `WaveNetLayerArrayDyn` e `layer_array.rs:81` (seed copy): verificar se D2.1 cobriu todos
       os caminhos ou se há fallback não instrumentado.
    5. Verificar o `DelayLine::push` do resampler (`src/dsp/resampler.rs`): se o push usa `copy_within`
       para mover a janela deslizante, substituir por ponteiro circular (sem cópia).
  - **Validação:** `rg 'memmove\|memcpy' target/dsp_hotpath.asm` (após rebuild) deve retornar zero
    para funções do namespace `nam_rs`. Soak test obrigatório.
