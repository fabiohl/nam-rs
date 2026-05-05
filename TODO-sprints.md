<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva. -->
# TODO-sprints.md — Backlog Técnico NAM-rs

## Épico Z — Teste de Estabilidade de Longa Duração (Soak Test)

**Impacto:** Confiança na estabilidade numérica e operacional para operação contínua
(8-10 horas) — detecção de drift, leaks de estado, overflow de contadores.

> **Nota:** É executado manualmente sob demanda, fora do CI/CD. O objetivo é estressar os **algoritmos**, não a máquina.

### [x] `TZ1` — Soak Test: Suíte de Estabilidade Numérica

**Arquivos:** `tests/soak_test.rs` (novo)
**Prioridade:** Alta (executado manualmente, não no CI)

**Ação:**

1. Criar `tests/soak_test.rs` com os seguintes cenários:
   - **WaveNet Silence Soak:** Processa 10M+ frames de silêncio (0.0) por um
     `WaveNetModel<16, 3, 8>`. Verifica que a saída permanece boundada `[-2.0, 2.0]`.
   - **WaveNet Noise Soak:** Processa 10M+ frames de ruído branco (seeded PRNG
     determinístico) por `WaveNetModel`. Verifica ausência de NaN, ±Inf, divergência.
   - **LSTM Silence Soak:** Processa 10M+ frames de silêncio por `LstmModel2<16>`.
     Verifica que `cell_state` e `hidden_state` não divergem.
   - **LSTM Noise Soak:** Processa 10M+ frames de ruído por `LstmModel2<16>`.
     Verifica estabilidade.
   - **Resampler Drift Soak:** Processa 50M+ amostras (equivalente a ~17 minutos
     a 48kHz) com taxas não-padrão (22050→48000, 96000→48000). Verifica que
     `phase_accum` permanece ≥ 0 e que não há acumulação de erro aritmético.
   - **VirtualRingBuffer Long Run:** Executa 100M+ advance/write cycles no
     `VirtualRingBuffer` e verifica integridade da fronteira mapeada.
   - **Gate FSM Endurance:** Processa 10M+ alternâncias rápidas Open↔Closed↔FadingIn
     para verificar que o `ramp_counter` e `hold_counter` não apresentam overflow.
2. Marcar todos os testes com `#[ignore]` para que não rodem no CI rápido.
3. Adicionar ao final de cada teste um `println!` com as métricas coletadas
   (tempo total, frames processados, min/max saída) para análise manual.

**Critério de Aceite:** 10M+ frames sem panic, NaN, ±Inf ou divergência.
Testes ignorados por padrão. Rodam sob `cargo test -- --ignored --nocapture`.

**✅ Concluído:** `tests/soak_test.rs` com 7 cenários (WaveNet×2, LSTM×2, Resampler,
VRB, Gate FSM). Todos marcados `#[ignore]`. PCG PRNG inline sem dependências externas.
Métricas de tempo/frames/min-max impressas com `println!`.

---

### [x] `TZ2` — Linha de Comando para Disparo Manual do Soak Test

**Arquivos:** `utils/soak-test.sh` (novo)
**Prioridade:** Alta

**Ação:**

1. Criar script shell `utils/soak-test.sh` que:
   - Imprime um aviso de que a bateria pode durar horas.
   - Executa os soak tests com `--release`, `--ignored`, `--nocapture` e `--test-threads=1`.
   - `--release` é mandatório para que os kernels SIMD sejam compilados com otimizações.
   - `--test-threads=1` garante que os testes não competem por recursos de CPU.
   - O log é salvo em `soak-test.log` na raiz do projeto para análise posterior.
   - Ao final, imprime um resumo com exit code.
2. Garantir que `soak-test.log` seja ignorado pelo Git.
3. Adicionar o cabeçalho de copyright no script.

**Critério de Aceite:** `bash utils/soak-test.sh` dispara a suíte corretamente.
O log é gravado. O script é idempotente.

**✅ Concluído:** Script `utils/soak-test.sh` criado, executável (`chmod +x`), com
cabeçalho SPDX. Usa `cargo test --release --test soak_test -- --ignored --nocapture
--test-threads=1 2>&1 | tee soak-test.log` com captura de `${PIPESTATUS[0]}` para
exit code correto via pipe. Logs cobertos pela glob `*.log` no `.gitignore`.

---

### [x] `TZ3` — Benchmarks Longos para Avaliação de Desempenho Fora de Aleatoriedades

**Arquivos:** `benches/inference_bench.rs` (modificação), `utils/long-bench.sh` (novo)
**Prioridade:** Média

Os benchmarks atuais do Criterion usam blocos curtos (64 amostras) com medição
estatística rápida (~50 iterações). Para modelos com variância alta (jitter de cache,
TLB misses em VRB, rewind amortizado), uma medição prolongada com blocos maiores
pode revelar o desempenho *real* em operação contínua.

**Ação:**

1. Adicionar grupo de benchmarks `Long_Run_*` no `inference_bench.rs`:
   - `Long_WaveNet_Standard_CH16_4096samp`: Processa 4096 amostras por iteração
     (equivalente a ~85ms de áudio). `measurement_time(30)`, `sample_size(100)`.
   - `Long_LSTM_2x16_4096samp`: Idem para LSTM 2-camadas.
   - `Long_Resampler_44100_to_48000_4096samp`: Idem para resampler com bloco grande.
2. Anotar estes benchmarks com `#[cfg(feature = "long_bench")]` para que não rodem
   na bateria rápida padrão (ativados via `cargo bench --features long_bench`).
3. Adicionar `long_bench = []` no `[features]` do `Cargo.toml`.
4. Criar `utils/long-bench.sh` com padrão idêntico ao `soak-test.sh` para disparo
   manual controlado (aviso + sleep + `--features long_bench`).

**Critério de Aceite:** Benchmarks longos rodam sob `bash utils/long-bench.sh`.
Resultados gravados pelo Criterion em `target/criterion/`. Sem regressão nos
benchmarks curtos existentes. Script de disparo manual com aviso e log.

**✅ Concluído:** Grupos `Long_Run_WaveNet`, `Long_Run_LSTM`, `Long_Run_Resampler`
implementados com `#[cfg(feature = "long_bench")]`. Feature `long_bench = []` no
`Cargo.toml`. Script `utils/long-bench.sh` criado e executável. Logs cobertos pela
glob `*.log` no `.gitignore`. Benchmarks curtos sem regressão confirmado.

---

> **Épico Z — ✅ CONCLUÍDO** (05/05/2026)
>
> Arquivos entregues: `tests/soak_test.rs`, `utils/soak-test.sh`, `utils/long-bench.sh`,
> `benches/inference_bench.rs` (extensão com `Long_Run_*`), `Cargo.toml` (feature `long_bench`).
> Documentação: `README.md` e `docs/benchmarks.md` atualizados.
> Higiene: Logs gerados (`soak-test.log`, `long-bench.log`) cobertos pelo `.gitignore`.
