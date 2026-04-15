# TODO — Auditoria WaveNet: Tarefas Técnicas

Plano de correções derivado da auditoria independente da implementação WaveNet.
Cada item contém: escopo, arquivos afetados, estratégia de correção e critério de aceitação.

---

## Sprint 1 — Imediato (Pré-Beta)

### T1 · C2 — Validar campo `activation` no Dispatcher

- [x] **1.1** Em `src/loader/dispatcher.rs` → `build_wavenet_typed()` ou `build_wavenet()` (≈L105):
  - Antes de despachar para o construtor, iterar `data.config.layers` e ler `layer.activation.as_deref().unwrap_or("Tanh")`
  - Aceitar `"Tanh"` (pass-through, sem alteração funcional)
  - Rejeitar qualquer outro valor com `bail!("Ativação '{}' na layer {} não é suportada. Apenas 'Tanh' é implementado.", act, idx)`
- [x] **1.2** Em `src/loader/dispatcher.rs` → `build_wavenet_dynamic()` (≈L274):
  - Mesmo check de validação antes de chamar `build_wavenet_array_dyn`
- [x] **1.3** Em `src/loader/dispatcher.rs` → `mod tests`:
  - Adicionado `test_reject_unsupported_activation`: `NamModelData` com `activation: Some("ReLU".into())` retorna `Err` com mensagem contendo `"ReLU"` (testa layer 0 e layer 1)
  - Adicionado `test_accept_tanh_activation`: `activation: Some("Tanh".into())` retorna `Ok`
  - Adicionado `test_accept_missing_activation`: `activation: None` retorna `Ok` (default = Tanh)
- [x] **1.4** `cargo test` — 71 testes, 0 falhas (sem regressões)

- **Critério de aceitação:** Modelos com `activation != "Tanh"` falham com mensagem descritiva. Modelos existentes (Tanh ou null) sem regressão.

> **✅ Concluído:** 2026-04-15. Helper `validate_layer_activations()` adicionado como função privada reutilizável, invocada antes da leitura de pesos em `build_wavenet_typed()` e `build_wavenet_dynamic()`. `lints.sh` passou limpo.

---

### T2 · C1 — Implementar Gated Activations no Path Dinâmico

- [x] **2.1** Em `src/models/wavenet_dyn.rs` → `WaveNetLayerDyn` (≈L130):
  - Adicionar campo `pub gated: bool`
- [x] **2.2** Em `src/models/wavenet_dyn.rs` → `WaveNetLayerDyn::process()` (≈L144):
  - Se `self.gated == false`: comportamento atual inalterado (`tanh_slice(block)`)
  - Se `self.gated == true`:
    - `block` tem tamanho `2*ch` (preenchido pelo conv1d com `out_ch = 2*ch`)
    - Aplicar `tanh_slice(&mut block[0..ch])`
    - Aplicar `sigmoid_slice(&mut block[ch..2*ch])`
    - Multiplicação element-wise: `block[j] = block[j] * block[ch + j]` para `j in 0..ch`
    - Prosseguir com `head_input[j] += block[j]` e `one_by_one.process(&block[0..ch], ...)` normalmente
- [x] **2.3** Em `src/models/wavenet_dyn.rs` → `WaveNetLayerArrayDyn` (≈L180):
  - `block_buffer` deve ter tamanho `max(ch, 2*ch se gated)`, determinado no construtor
- [x] **2.4** Em `src/loader/dispatcher.rs` → `build_wavenet_array_dyn()`:
  - Aceitar parâmetro adicional `gated: bool`
  - Quando `gated == true`: passar `out_size = 2 * ch` para `read_conv1d_weights_dyn` (em vez de `ch`)
  - Propagar `gated` para cada `WaveNetLayerDyn { ..., gated }`
  - `block_buffer` alocado com `if gated { 2 * ch } else { ch }`
- [x] **2.5** Em `src/loader/dispatcher.rs` → `build_wavenet_dynamic()` (≈L274):
  - Ler `l0.gated.unwrap_or(false)` e `l1.gated.unwrap_or(false)`
  - Passar o valor para `build_wavenet_array_dyn`
- [x] **2.6** Verificar que `get_wavenet_topology()` em `nam_json.rs` já retorna `None` quando `gated == true` (já confirmado, L148)
- [x] **2.7** Testes:
  - Adicionar teste unitário `test_gated_layer_dyn_process`: construir `WaveNetLayerDyn` com `gated=true`, `conv1d.out_ch=2*ch`, pesos sintéticos, e verificar que a saída é `tanh(x) ⊙ sigmoid(x)` numericamente
  - Adicionar teste no dispatcher: `test_build_wavenet_dynamic_gated` com `NamModelData` sintético com `gated: true` e contagem de pesos ajustada para `2*ch` no conv1d

- **Critério de aceitação:** Modelos `gated: true` produzem `tanh(conv) ⊙ sigmoid(conv)` no path dinâmico. Path estático rejeita silenciosamente via `get_wavenet_topology() → None → fallback dinâmico`.

> **✅ Concluído:** 2026-04-15. Campo `gated: bool` em `WaveNetLayerDyn`; campo `block_size: usize` em `WaveNetLayerArrayDyn`; `build_wavenet_array_dyn` recebe `gated: bool`, dobra `conv_out_ch` e `block_buffer`; `build_wavenet_dynamic` lê `l0/l1.gated` do JSON. 5 novos testes (3 em `wavenet_dyn.rs`, 2 em `dispatcher.rs`). 97 testes, 0 falhas. `lints.sh` limpo. A fidelidade numérica foi validada analiticamente: `head_input = tanh(x) ⊙ sigmoid(x)` com eps < 1e-5.

---

### T3 · A2 — Substituir `println!` por `log::info!`

- [x] **3.1** `cargo add log` (sem features extras; facade mínima)
  - Verificar que não arrasta dependências transitivas desnecessárias
- [x] **3.2** Em `src/loader/dispatcher.rs`:
  - Adicionado `use log::info;` no topo
  - Substituídas as 5 ocorrências de `println!` (WaveNet estático, WaveNet dinâmico, LSTM 1×, LSTM 2×, LSTM dinâmico) por `log::info!`
- [x] **3.3** Em `src/main.rs`:
  - `cargo add env_logger` instalado
  - `env_logger::init()` adicionado no início de `main()`
  - Mensagens de prewarm e payload também migradas para `log::info!`
- [x] **3.4** `cargo test 2>&1 | grep "\[Dispatcher\]"` retorna vazio — confirmado

- **Critério de aceitação:** Biblioteca não emite nada em stdout. Logs acessíveis via `RUST_LOG=info` quando subscriber instalado.

> **✅ Concluído:** 2026-04-15. `log = 0.4` e `env_logger = 0.11` adicionados via `cargo add`. 5 `println!` do dispatcher substituídos por `log::info!`; 2 `println!` de lifecycle no `main.rs` também migrados. `env_logger::init()` instalado no topo de `main()`. `cargo test` — todos os testes passam, `[Dispatcher]` ausente em stdout. `lints.sh` limpo.

---

> **📋 Sprint 1 — Revisão Concluída:** 2026-04-15.
> Auditoria independente confirmou todas as 3 tarefas (T1, T2, T3) implementadas e funcionais.
> **97 testes** passando (76 unitários + 18 integração + 2 proptest + 1 PipeWire), 0 falhas.
> Documentação sincronizada: `README.md` (97 verificações) e `docs/architecture.md` (76 unitários, tabela atualizada com `dispatcher.rs:16` e novo `wavenet_dyn.rs:3`).
> Nenhum apontamento de melhoria identificado que necessite adição de tarefas extras.

## Sprint 2 — Curto Prazo

### T6 · M1 — Debug Asserts de Invariante em `process()`/`prewarm()`

- [ ] **6.1** Em `src/models/wavenet.rs` → `WaveNetLayerArray::process()` (≈L271):
  - Inserir no início: `debug_assert_eq!(self.layers.len(), self.states.len(), "WaveNetLayerArray: layers ({}) ≠ states ({})", self.layers.len(), self.states.len());`
- [ ] **6.2** Em `src/models/wavenet.rs` → `WaveNetLayerArray::prewarm()` (≈L333):
  - Mesmo `debug_assert_eq!`
- [ ] **6.3** Em `src/models/wavenet_dyn.rs` → `WaveNetLayerArrayDyn::process()` (≈L210):
  - Mesmo `debug_assert_eq!`
- [ ] **6.4** Em `src/models/wavenet_dyn.rs` → `WaveNetLayerArrayDyn::prewarm()` (≈L277):
  - Mesmo `debug_assert_eq!`
- [ ] **6.5** Verificar que `cargo test` em modo debug (default) exercita os asserts sem falha

- **Critério de aceitação:** Panic em debug se `layers.len() ≠ states.len()`. Zero custo em release.

---

### T7 · M2 — Guard em `rewind_buffer`

- [ ] **7.1** Em `src/models/wavenet.rs` → `rewind_buffer()` (L217):
  - Inserir antes da subtração: `debug_assert!(self.buffer_start >= self.receptive_field_size, "rewind_buffer: buffer_start ({}) < receptive_field_size ({})", self.buffer_start, self.receptive_field_size);`
- [ ] **7.2** Documentar o invariante no docstring de `rewind_buffer`:
  - `/// # Invariante:`buffer_start >= receptive_field_size` (garantido por `advance_frames`).`

- **Critério de aceitação:** Debug panic para estados mal construídos. Comportamento de release inalterado.

---

### T4 · A1 — Corrigir RF Sintético nos Testes de Integração

- [ ] **4.1** Em `tests/nam_infer_test.rs` → `build_synthetic_wavenet_standard()` (≈L240):
  - Substituir `let rf1 = 512 * 2;` (L244) por `let rf1: usize = dilations_1.iter().map(|&d| (3 - 1) * d).sum();` (resultado: 2046)
  - Idem para `rf2` (L245)
  - Substituir `WaveNetLayerState::new(16, final_rf, i)` (L253) por `WaveNetLayerState::new(16, (3 - 1) * d, i)` onde `d` é a dilatação da camada correspondente
  - Idem para `states_2` (L281, CH=8, per-layer RF)
  - Manter `model.receptive_field_size = rf1.max(rf2)` (agora = 2046)
- [ ] **4.2** Verificar que `test_wavenet_computational_stability` (L342) continua passando com os novos RFs
- [ ] **4.3** Ajustar `prewarm()` se necessário (o prewarm do `build_synthetic` chama o interno do modelo que já itera `receptive_field_size` passos)

- **Critério de aceitação:** Cada `WaveNetLayerState` no teste sintético usa `RF = (K-1) * dilation`, espelhando fielmente o construtor de produção (`build_wavenet_array`/`build_wavenet_array_dyn`).

---

### T5 · C3 — Recalibrar Threshold do Golden Vector WaveNet

- [ ] **5.1** Executar `cargo test test_golden_vectors_wavenet -- --nocapture` e anotar o MSE real impresso em `[Golden WaveNet] MSE=...`
- [ ] **5.2** Definir novo threshold:
  - Se MSE real ≤ 1e-3 → threshold = `5e-3` (5× headroom)
  - Se MSE real ≤ 5e-3 → threshold = `2e-2` (4× headroom)
  - Se MSE real ≤ 2e-2 → manter `5e-2` e documentar justificativa
- [ ] **5.3** Em `tests/nam_infer_test.rs` → `test_golden_vectors_wavenet()` (L586):
  - Atualizar o `assert!(mse < NOVO_THRESHOLD, ...)` conforme medição (L628)
  - Atualizar o docstring (L576–L581) para refletir o threshold real e sua justificativa:

    ```rust
    /// **Threshold calibrado:** MSE < Xe-Y
    /// - MSE medido em [data]: Ze-W
    /// - Headroom: N× para variação FastMath entre plataformas
    /// - O `simd_tanh` (Padé grau 5 + rsqrt_ps) difere do C++ que usa
    ///   rational polynomial (`Activation.h`), acumulando ~5e-3 por camada
    ///   em 20 camadas empilhadas.
    ```

  - **Nota:** O docstring atual afirma `MSE < 1e-4` mas o assert real usa `5e-2` — essa inconsistência será corrigida nesta tarefa.

- **Critério de aceitação:** Threshold documentado e calibrado contra medição real. Docstring consistente com o assert.

---

### T13 · PERF — Eliminar Bounds Checks em `Conv1d::process_frame()` (Hot-Path)

**Origem:** Auditoria independente de otimização de delay lines WaveNet (abril/2026).
**Avaliação:** O relatório original propunha substituir a gestão de histórico (delay lines) por um `PotDelayLine` com mascaramento bitwise. A auditoria independente rejeitou a premissa central — **o NAM-rs não usa ring buffer circular com módulo; usa buffer linear flat (SoA) com `rewind` amortizado** — mas confirmou que há **bounds checks elimináveis** no hot-path de `process_frame()`.

**Diagnóstico confirmado:** Dentro do loop de convolução `for k in 0..K` (e `for out_c in 0..OUT`), os slicings de `layer_buffer` e `weights` geram bounds checks (`cmp`+`jcc` pairs) que o LLVM não consegue elimir porque os índices envolvem aritmética com `dilation` e `buffer_start` (não trivialmente provados in-bounds pelo auto-bounds-elimination pass).

**Impacto:** Modelo Standard (CH=16, K=3): 16 × 3 × 2 = **96 bounds checks por frame**, × 64 frames/bloco = **~6.144 branches/bloco** elimináveis. Em camadas empilhadas (10–20 layers × 2 arrays), o total atinge **~120k–250k branches/bloco** desnecessários.

**Prova de segurança (SAFETY):**

- `layer_buffer` é alocado com `buffer_frames * channels` floats (L193–195), e `buffer_start` é mantido dentro de `[0, buffer_frames)` pelas invariantes de `advance_frames()` (L208) e `rewind_buffer()` (L217).
- `self.weights` é alocado com `OUT * K * IN` floats exatos pelo construtor (`read_conv1d_weights`), e os índices `out_c * K * IN + k * IN` com `out_c < OUT`, `k < K` nunca excedem o tamanho.
- O `frame_idx` calculado por `buffer_start + dilation * (k+1-K)` é sempre ≥ 0 porque `buffer_start ≥ receptive_field_size ≥ (K-1) * max_dilation`.

**Custo / Risco:** Zero custo adicional. A cobertura de golden vectors + auto-consistência + determinismo valida a equivalência bit-exact.

- [ ] **13.1** Em `src/models/wavenet.rs` → `Conv1d::process_frame()` (L53–56):
  
  - Substituir os 2 slicings bounds-checked por `get_unchecked`:

    ```rust
    // Antes:
    let in_slice = &layer_buffer[in_slice_start..in_slice_start + IN];
    let weight_slice = &self.weights[weight_slice_start..weight_slice_start + IN];
    
    // Depois:
    // SAFETY: `in_slice_start + IN ≤ layer_buffer.len()` — invariante
    // de `WaveNetLayerState::new()` (buffer_frames * CH) e `advance_frames`
    // que mantém `buffer_start` dentro de limites. `frame_idx` é ≥ 0
    // porque `buffer_start ≥ receptive_field_size ≥ (K-1)*max_dilation`.
    let in_slice = unsafe {
        layer_buffer.get_unchecked(in_slice_start..in_slice_start + IN)
    };
    // SAFETY: `weight_slice_start + IN ≤ OUT*K*IN` por construção
    // imutável de `self.weights` com tamanho exato `OUT*K*IN`.
    let weight_slice = unsafe {
        self.weights.get_unchecked(weight_slice_start..weight_slice_start + IN)
    };
    ```

- [ ] **13.2** Em `src/models/wavenet.rs` → `WaveNetLayer::process()` residual (L158–162):
  
  - Substituir o loop indexado por `get_unchecked`:

    ```rust
    // SAFETY: `buffer_start` validado por `advance_frames()`;
    // `lb_start + CH ≤ layer_buffer.len()` é invariante do construtor.
    unsafe {
        for j in 0..CH {
            *output.get_unchecked_mut(j) += *layer_buffer.get_unchecked(lb_start + j);
        }
    }
    ```

- [ ] **13.3** Em `src/models/wavenet_dyn.rs` → `Conv1dDyn::process_frame()` (L55–60):
  
  - Mesma transformação para variante dinâmica:

    ```rust
    let in_slice = unsafe {
        layer_buffer.get_unchecked(in_slice_start..in_slice_start + self.in_ch)
    };
    let weight_slice = unsafe {
        self.weights.get_unchecked(weight_slice_start..weight_slice_start + self.in_ch)
    };
    ```

- [ ] **13.4** Em `src/models/wavenet_dyn.rs` → `WaveNetLayerDyn::process()` residual (L172–175):
  
  - Mesma transformação para loop residual dinâmico.

- [ ] **13.5** Validação:
  
  - `cargo test` — todos os testes devem passar sem regressão
  - Golden vectors inalterados (MSE tresholds atuais: WaveNet < 5e-2, LSTM < 1e-3)
  - Auto-consistência (MSE = 0.0) mantida entre estático ↔ dinâmico
  - `cargo bench --bench inference_bench` — anotar comparativo antes/depois

- **Critério de aceitação:** Bounds checks eliminados em `process_frame()` e residual de ambas as variantes (estática e dinâmica). Golden vectors e auto-consistência inalterados. Benchmark mostra redução mensurável (ou neutro) em latência/bloco.

---

### T11 · DSP — Métrica SNR nos Golden Tests (Validação Cross-Implementação)

**Origem:** Auditoria independente de golden reference DSP CI/CD (abril/2026).
**Avaliação:** A proposta de utilizar SNR (Signal-to-Noise Ratio) como métrica de validação cross-implementação é tecnicamente sólida e padrão na indústria DSP. Porém, o relatório original continha imprecisões factuais corrigidas durante a avaliação:

- ❌ O codebase **nunca usou** `assert_eq!` binário nem `approx::assert_relative_eq!` nos golden tests (já usava MSE em `f64`).
- ❌ O modelo referenciado como `BossWN-feather.nam` nos golden tests é na verdade `BossWN-standard.nam`.
- ❌ O `calculate_snr()` proposto fazia subtração em `f32` antes do cast a `f64` — corrigido para aritmética integral em `f64`.
- ❌ O threshold SNR ≥ 120 dB é irrealista para FastMath Padé (~5e-3 erro/camada × 20 camadas).

**Decisão:** SNR como métrica **aditiva** (não substitutiva) ao MSE existente.

**Custo computacional:** Zero overhead líquido. A implementação funde MSE, MAE e SNR numa **única passada** sobre o buffer (512 amostras em `f64`), substituindo as 2 passadas atuais (`compute_mse` + `compute_max_abs_error` separadas). Estas funções vivem exclusivamente em `#[test]` — zero impacto em produção.

- [ ] **11.1** Em `tests/nam_infer_test.rs` → seção Helpers (após `compute_max_abs_error`, ≈L94):
  - Adicionar `compute_snr(reference: &[f32], test: &[f32]) -> f64`:
    - Aritmética integral em `f64`: `let r64 = r as f64; let t64 = t as f64;`
    - `signal_power += r64 * r64;` e `noise_power += (r64 - t64) * (r64 - t64);`
    - Guarda contra divisão por zero: `if noise_power <= f64::EPSILON { return f64::INFINITY; }`
    - Retorna `10.0 * (signal_power / noise_power).log10()`
- [ ] **11.2** Adicionar `assert_dsp_fidelity(reference, test, mse_limit, min_snr_db, label)`:
  - Função `#[track_caller]` que calcula MSE, MAE e SNR numa **única passada fundida**:
    - 1 loop `for (&r, &t) in reference.iter().zip(test.iter())` acumulando `signal_power`, `noise_power`, `sum_sq_diff` e `max_abs_diff` simultaneamente
    - Deriva MSE = `sum_sq_diff / n`, MAE = `max_abs_diff`, SNR = `10 * log10(signal_power / noise_power)`
  - Imprime todas as 3 métricas via `println!` para diagnóstico: `[{label}] MSE=..., MaxAbsErr=..., SNR=... dB`
  - Falha com `assert!` se `mse >= mse_limit` **ou** `snr < min_snr_db`
  - Mensagem de falha inclui as 3 métricas para debug rápido
- [ ] **11.3** Em `test_golden_vectors_wavenet()` (L586, assert em L627–630):
  - Substituir bloco de validação atual (`compute_mse` + `compute_max_abs_error` + `println!` + `assert!`) por chamada única:

    ```rust
    assert_dsp_fidelity(&expected, &output, 5e-2, 30.0, "Golden WaveNet");
    ```

  - Threshold MSE mantido em `5e-2` (recalibrado por T5 quando executado)
  - Threshold SNR = `30.0` dB (conservador; ajustado junto com T5)
- [ ] **11.4** Em `test_golden_vectors_lstm()` (L644, assert em L685–688):
  - Substituir bloco de validação por:

    ```rust
    assert_dsp_fidelity(&expected, &output, 1e-3, 50.0, "Golden LSTM 1×16");
    ```

  - Threshold MSE mantido em `1e-3`
  - Threshold SNR = `50.0` dB (LSTM converge melhor que WaveNet — sem acumulação de FastMath)
- [ ] **11.5** Atualizar docstrings de ambos os golden tests:
  - Documentar validação dual MSE + SNR
  - Justificar thresholds SNR: WaveNet 30 dB (conservador para ativações Padé acumuladas em 20 camadas), LSTM 50 dB (menor divergência cross-implementação)
  - Referenciar que a subtração é feita integralmente em `f64` para preservar precisão do residual
- [ ] **11.6** Verificar que `cargo test test_golden_vectors -- --nocapture` imprime SNR em dB para ambos os testes e ambos passam

- **Critério de aceitação:** Golden tests reportam MSE + SNR + MAE. Ambas as métricas (MSE e SNR) são assertadas independentemente. Zero overhead computacional adicional (fusão single-pass). Produção inalterada.

---

## Sprint 3 — Médio Prazo

### T8 · M3 — Adicionar Fixture BossWN-lite.nam e Teste de Integração

- [ ] **8.1** Criar modelo Lite sintético com pesos uniformes (0.01) via helper no teste:
  - CH=12, K=3, HEAD=6, dilations_0 = `[1,2,4,8,16,32,64]`, dilations_1 = `[128,256,512,1,2,4,8,16,32,64,128,256,512]`
  - Calcular peso total via fórmula do dispatcher
  - Gerar `NamModelData` sintético no dispatcher test
- [ ] **8.2** Em `tests/nam_infer_test.rs`:
  - Adicionar `test_wavenet_stability_lite`:
    - Se `BossWN-lite.nam` existir → carrega e testa com modelo real
    - Senão → construir `DynamicModel` via dispatcher com `NamModelData` sintético (CH=12, pesos calculados) e verificar estabilidade (finitude + magnitude ≤ 100.0)
- [ ] **8.3** (Opcional) Se um modelo Lite real for obtido futuramente, comitá-lo em `tests/fixtures/models/BossWN-lite.nam`

- **Critério de aceitação:** Topologia Lite exercitada em integração real (dispatcher → prewarm → process → validação numérica).

---

### T9 · M4 — Documentar FastMath vs Threshold nos Testes

- [ ] **9.1** Em `tests/nam_infer_test.rs`:
  - Expandir docstring de `test_golden_vectors_wavenet` (L570–L584) com seção explícita sobre calibração do threshold vs erro de `simd_tanh`
  - Incluir referência ao erro máximo documentado em `docs/architecture.md` §2 ("Erro máximo ~5e-3")
  - Explicitar a fórmula de acumulação esperada: `erro_máx ≈ √N_camadas × erro_por_camada`
- [ ] **9.2** Em `src/math/fastmath.rs` ou docstring de `simd_tanh`:
  - Documentar: "Erro máximo vs `f32::tanh()`: ~5e-3 (Padé grau 5 + Newton-Raphson rsqrt). Em modelos WaveNet com 20 camadas empilhadas, o erro acumula sublinearmente."

- **Critério de aceitação:** Justificativa do threshold golden documentada junto ao assert. Erro do FastMath documentado na função fonte.

---

### T10 · Campos Privados + Construtores Validados (Futuro)

- [ ] **10.1** Tornar campos de `WaveNetLayerArray` privados (`pub(crate)` ou privados com getters)
- [ ] **10.2** Expor `WaveNetLayerArray::new(layers, states, rechannel, head_rechannel, ...)` com validação de invariantes:
  - `assert_eq!(layers.len(), states.len())`
  - `assert_eq!(head_outputs.len(), HEAD)`
  - `assert_eq!(array_outputs.len(), CH)`
- [ ] **10.3** Refatorar todos os testes que constroem `WaveNetLayerArray` manualmente para usar o novo construtor
- [ ] **10.4** Idem para `WaveNetLayerState`, `Conv1d`, `DenseLayer`

- **Critério de aceitação:** Construção inválida impossível via API pública. Testes usam construtores validados.

---

### T12 · DOC — Documentar Metodologia SNR no README de Fixtures

**Origem:** Mesma auditoria de T11.

- [ ] **12.1** Em `tests/fixtures/README.md`:
  - Adicionar seção "## Metodologia de Validação" após "## Para regenerar"
  - Documentar a estratégia dual MSE + SNR:
    - MSE: erro quadrático médio como métrica de regressão primária (sensível à escala absoluta)
    - SNR: relação sinal-ruído em dB como métrica de equivalência perceptual (normalizada pela potência do sinal)
    - Ambas assertadas independentemente nos golden tests
  - Justificar por que SNR é aditiva e não substitutiva:
    - MSE detecta erros absolutos (útil para regressões estruturais)
    - SNR fornece interpretação DSP padrão (útil para engenheiros de áudio)
  - Documentar thresholds atuais e a fonte de divergência (FastMath Padé vs. C++ rational polynomial)

- **Critério de aceitação:** README de fixtures documenta a política de validação numérica de forma autocontida.

---

## Dependências entre Tarefas

```text
T1 (activation) ──→ T2 (gated) ──→ T5 (threshold recalibrar)
T3 (log) ──→ independente
T4 (RF sintético) ──→ independente
T6, T7 (asserts) ──→ independente
T8 (fixture lite) ──→ depende de T1/T2 para cobertura completa
T9 (doc threshold) ──→ depende de T5
T10 (campos privados) ──→ depende de T6
T11 (SNR golden) ──→ independente (co-beneficia T5 quando executado junto)
T12 (doc SNR fixtures) ──→ depende de T11
T13 (bounds checks) ──→ independente (validada por golden vectors existentes)
```

## Ordem de Execução Recomendada

Sprint 1 (sequencial):

1. **T1** → T2 → T3

Sprint 2 (segurança primeiro, depois performance, depois observabilidade):

1. **T6** + **T7** (debug asserts — paralelos entre si)
2. **T4** (RF sintético — corretude de testes)
3. **T5** (threshold — requer medição real)
4. **T13** (bounds checks — performance, independente)
5. **T11** (SNR — observabilidade, independente)

Sprint 3 (médio prazo):

1. **T8** (fixture Lite)
2. **T9** (doc FastMath)
3. **T10** (campos privados)
4. **T12** (doc SNR README)
