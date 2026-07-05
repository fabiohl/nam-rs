<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Findings — Remoção da Quantização de Pesos (F16C/BF16)

> **Origem**: Auditoria revisor-auditor 2026-07-05, solicitada por Fábio.
> **Prioridade declarada**: Fidelidade ao ideal matemático (f64) tem prioridade sobre paridade com NAMCore.
> **Abordagem**: Remoção completa e definitiva — sem feature flags. Git rollback se necessário.

---

## Finding F-Q1: Quantização de pesos `f32 → u16` causa drift recorrente em LSTMs e erro estático em A2

### Contexto F-Q1

O nam-rs comprime pesos de modelos neurais de `f32` (32-bit IEEE 754, ~7.2 dígitos de precisão) para `u16` (16-bit half-precision via F16C ou BF16, ~3.3 dígitos) durante o carregamento do modelo. Na inferência, os kernels SIMD fazem descompressão on-the-fly (`_mm256_cvtph_ps` / `_mm512_cvtph_ps`).

### Impacto por Arquitetura F-Q1

| Arquitetura        | Usa quantização u16?                                                                 | Impacto sonoro                                                                                                                                                        | Impacto em performance                                             |
|:------------------ |:------------------------------------------------------------------------------------ |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------- |:------------------------------------------------------------------ |
| **LSTM**           | ✅ Sim — `input_hidden_weights: [[[u16; H]; IH]; 4]`, `head_weights: [u16; H]`       | **Severo**: drift acumulativo do cell-state. ESR cresce com duração e sample rate (2.61e-2 @ 5s/48kHz → 1.42e-1 @ 192kHz). É o pior problema de fidelidade do nam-rs. | Quantização economiza ~50% de memória de pesos (L1 cache pressure) |
| **WaveNet A2**     | ✅ Parcial — `rechannel_w: AlignedVec<u16>`, conv weights internos (u16 interleaved) | **Baixo**: feedforward, erro não acumula. ESR ≈ 1.21e-8 (79.2 dB SNR)                                                                                                 | Mesmo efeito de cache                                              |
| **WaveNet A1**     | ❌ Não — `DenseLayer` já usa `AlignedVec<f32>` nativo                                | Nenhum — já opera em precisão nativa                                                                                                                                  | Já usa f32, não aplica                                             |
| **ConvNet/Linear** | ❌ Não — f32 nativo                                                                  | Nenhum                                                                                                                                                                | Não aplica                                                         |

### Raiz do Problema no LSTM F-Q1

```text
cₜ = fₜ · cₜ₋₁ + iₜ · gₜ    ← cada time-step injeta εq via gates quantizados
hₜ = oₜ · tanh(cₜ)

ESR_steady ∝ σ²ε / (1 − ⟨f⟩²)    ← forget gate ≈ 0.9–0.99 → decaimento lento do erro
```

- **Cada amostra de áudio** roda o GEMV com pesos u16, injetando erro ~3.9e-3 per-weight
- O forget gate (~0.95 tipicamente) preserva 95% do erro anterior
- Em 5 segundos @ 48kHz = 240.000 time-steps de acumulação

### Locais Exatos no Código F-Q1

**Structs com pesos u16:**

- `src/models/lstm/layer.rs:11` — `input_hidden_weights: Aligned64<[[[u16; H]; IH]; 4]>`
- `src/models/lstm/layer_dyn.rs:21` — `input_hidden_weights: AlignedVec<u16>`
- `src/models/lstm/model1.rs:46` — `head_weights: [u16; H]`
- `src/models/lstm/model2.rs:76` — `head_weights: [u16; H]`
- `src/models/lstm/model_dyn.rs:33` — `head_weights: AlignedVec<u16>`
- `src/models/a2/model/static/mod.rs:54` — `rechannel_w: AlignedVec<u16>`
- `src/models/a2/model/dynamic/mod.rs:56` — `rechannel_w: AlignedVec<u16>`

**Função de quantização:**

- `src/math/common/ops.rs:17` — `quantize_weight(f: f32, is_bf16: bool) -> u16`

**Loaders que quantizam:**

- `src/loader/dispatcher/lstm/weights.rs:30,38` — LSTM backbone weights
- `src/loader/dispatcher/lstm/static_builder.rs:40,92` — LSTM head weights
- `src/loader/dispatcher/lstm/dynamic_builder.rs:49` — LSTM dyn head weights
- `src/models/a2/model/set_weights.rs:57` — A2 rechannel + conv weights

**Kernels SIMD que descomprimem u16→f32 (foco da migração):**

- `src/math/gemm/gemv_4gate/avx2.rs` — LSTM 4-gate GEMV (AVX2)
- `src/math/gemm/gemv_4gate/avx512.rs` — LSTM 4-gate GEMV (AVX-512)
- `src/math/gemm/gemv/f16_avx2.rs` — GEMV genérico f16 (AVX2)
- `src/math/gemm/gemv/f16_avx2_specialized.rs` — GEMV f16 especializado
- `src/math/gemm/gemv/f16_avx512.rs` — GEMV genérico f16 (AVX-512)
- `src/math/gemm/dot.rs` — Dot product f16 (head projection)
- `src/math/gemm/gemm_batch/fused_add_gemm_batch.rs` — Batch GEMM f16
- `src/math/gemm/gemm_batch/fused_residual_batch.rs` — Residual batch GEMM f16
- `src/math/gemm/gemm_batch/avx512.rs` — Batch GEMM AVX-512
- `src/models/lstm/layer_kernels.rs` — LSTM layer kernels

**Pesos BF16/VNNI (remoção):**

- `src/models/lstm/layer.rs:17` — `state_bf16: Aligned64<[u16; IH]>` (mirror de estado)
- `src/models/lstm/layer_dyn.rs:29` — `state_bf16: AlignedVec<u16>` (mirror de estado)

### Solução Proposta F-Q1

**Remoção completa e definitiva** da quantização de pesos. Todos os campos `u16` de pesos passam a `f32`. Todos os kernels SIMD que faziam `_mm256_cvtph_ps` para descomprimir pesos passam a fazer `_mm256_loadu_ps` diretamente. O `state_bf16` mirror do LSTM é removido. O dispatch BF16/VNNI para LSTM perde o sentido e pode ser simplificado.

**Impacto esperado:**

- **Fidelidade LSTM**: ESR deve cair de ~2.61e-2 para a vizinhança do piso Padé (~7.6e-4). Melhoria de ~34× no ESR para LSTM.
- **Fidelidade A2**: ESR deve melhorar ligeiramente (de ~1.21e-8 para ~1e-10+).
- **Fidelidade WaveNet A1**: Sem mudança (já f32).
- **Performance**: O uso de memória de pesos dobra (~40KB → ~80KB para WaveNet Standard). Risco teórico de L1 cache misses em modelos grandes. Contrapartida: elimina o custo da conversão `cvtph_ps` (≈1-2 ciclos/instrução). **Precisa de medição.**
- **Paridade NAMCore**: Vai **divergir** do NAMCore (que também quantiza para f16c). Esta é uma decisão consciente — prioridade é fidelidade ao ideal, não paridade.

---

## Finding F-Q2: Script de Dashboard `utils/quality-dashboard.sh` — ferramenta de uso geral

### Contexto F-Q2

O nam-rs precisa de um instrumento científico permanente que colha dados de **todos** os testes de fidelidade e performance existentes e os apresente de forma amigável a humanos. Este instrumento é uma **ferramenta de uso geral do projeto** — independente da PoC de quantização — e continuará sendo útil para qualquer mudança futura (ativações, oversampling, novos modelos, etc.).

### Cobertura Obrigatória — TODOS os Modelos, Modos e Arquiteturas

#### Arquiteturas (6 famílias)

| Família             | Modelos Fixture                                                                                                                          | Variantes                                                                                   |
|:------------------- |:---------------------------------------------------------------------------------------------------------------------------------------- |:------------------------------------------------------------------------------------------- |
| **WaveNet A1**      | `BossWN-standard.nam`, `BossWN-feather.nam`, `BossWN-lite.nam`, `BossWN-nano.nam`, `wavenet_a1_standard.nam`, `wavenet_official.nam`     | Static CH4/CH8/CH12/CH16                                                                    |
| **WaveNet A2**      | `wavenet_a2_full.nam`, `wavenet_a2_lite.nam`, `wavenet_a2_max.nam`, `wavenet_a2_container.nam`                                           | Static CH3/CH8                                                                              |
| **WaveNet A2 FiLM** | `wavenet_a2_film_lite.nam`, `wavenet_a2_film_full.nam`                                                                                   | Conditioning                                                                                |
| **WaveNet Dynamic** | `wavenet_dyn_free.nam`, `wavenet_condition_dsp.nam`, `a2_dynamic_gated_ch8.nam`, `a2_dynamic_blended_ch3.nam`                            | Dinâmico + Slimmable (`slimmable_container.nam`, `slimmable_wavenet.nam`, `a2_example.nam`) |
| **LSTM**            | `BossLSTM-1x16.nam`, `BossLSTM-2x8.nam`, `lstm.nam`, `lstm_dyn_test.nam`                                                                 | Static 1×16, 2×8; Dyn                                                                       |
| **ConvNet/Linear**  | `convnet_test.nam`, `linear_test.nam`, `linear_fft_rf320.nam`, `linear_fft_rf2048.nam`, `linear_fft_rf4096.nam`, `linear_fft_rf8192.nam` | Variáveis RF                                                                                |

#### Modos de Qualidade (2)

| Modo               | Oversampling | Ativações               | Adaptive Compute |
|:------------------ |:------------ |:----------------------- |:---------------- |
| **Live** (default) | Off          | `Standard` (Padé)       | Ativo            |
| **HQ/Offline**     | 4×           | `HighFidelity` (stdlib) | Desativado       |

#### ISAs Suportadas (3)

- **AVX2** (x86-64-v3) — baseline obrigatório
- **AVX-512** — dispatch automático
- **AVX-512 VNNI BF16** — dispatch automático (LSTM/GEMV otimizado)

#### Suítes de Teste Colhidas

| Suite                  | Comando                                                                  | O que mede                                        | Modelos cobertos                                                                        |
|:---------------------- |:------------------------------------------------------------------------ |:------------------------------------------------- |:--------------------------------------------------------------------------------------- |
| `golden_vectors`       | `cargo test --release --test golden_vectors -- --nocapture`              | ESR, SNR, MSE, MR-STFT vs C++ NAMCore             | Todos os modelos com `.golden.bin` (v1+v2, multi-SR)                                    |
| `reference_oracle_f64` | `cargo test --release --test reference_oracle_f64 -- --nocapture`        | ESR vs f64 ideal + decomposição de fontes de erro | WaveNet, LSTM, A2, A2-FiLM, ConvNet, A2-Generic                                         |
| `spectral_fidelity`    | `cargo test --release --test spectral_fidelity -- --nocapture`           | ASR (aliasing spectral ratio)                     | Todos que tenham spectral tests                                                         |
| `isa_parity`           | `cargo test --release --test isa_parity -- --test-threads=1 --nocapture` | Paridade bitwise AVX2 vs AVX-512                  | Todos                                                                                   |
| `activation_precision` | `cargo test --release --test activation_precision -- --nocapture`        | Impacto Standard vs HighFidelity                  | LSTM (mais sensível)                                                                    |
| `regression_gate`      | `cargo bench --bench regression_gate 2>&1`                               | Latência por bloco (64 samp, 48kHz)               | 10 modelos: WaveNet Std/Feather/Lite/Nano, A2 Full/Lite, LSTM 1×16/2×8, Linear, ConvNet |

### Solução Proposta F-Q2

Script `utils/quality-dashboard.sh` que:

1. Roda **todas** as suítes acima em `--release`
2. Faz **graceful-skip** para componentes ausentes (modelo não presente, goldens não gerados, C++ render não disponível)
3. Parseia o stdout e gera relatório humano com:
   - **Resumo rápido por modelo** — ESR → % de erro + veredicto humano (IDÊNTICO / IMPERCEPTÍVEL / AUDÍVEL COM A/B / AUDÍVEL)
   - **Tabela de performance** — latência mediana vs deadline RT (1333 µs @64samp/48kHz) + % de folga + sugestões (oversampling 2×/4×, HQ mode)
   - **Tabela técnica completa** — ESR, SNR, MSE, MR-STFT, LUFS por modelo e por sample rate
   - **Informações de sistema** — ISA detectada, CPU, baseline Criterion, data/hora
4. Suporta flag `--save <filename>` para persistir o output (usado para baseline A/B)
5. Retorna exit code 0 mesmo com testes skipped (informacional), ≠0 somente em erros de infraestrutura

---

## Finding F-Q3: Documentação e cleanup pós-PoC

### Contexto F-Q3

Se a remoção da quantização provar ser benéfica, vários documentos existentes se tornam parcialmente obsoletos e precisam de atualização:

- `docs/audio_fidelity_map.md` §1 e §3 — descrevem mecanismo de quantização e drift
- `docs/lstm_recurrent_drift.md` — inteiramente sobre o drift causado pela quantização
- `docs/cpp_parity_map.md` §2.5, §2.7, §7.2 — descrevem divergência de quantização
- `tests/common/constants.rs` — thresholds ESR calibrados para o regime quantizado

**Ação**: Esses docs serão reduzidos a um breve registro histórico de aprendizado ("we used to quantize, measured the impact, and decided to remove it").

---

## Épico EQ: Remoção da Quantização de Pesos (F16C/BF16)

| Sprint  | Descrição                                          | Findings   | Risco    |
|:------- |:-------------------------------------------------- |:---------- |:-------- |
| **SQ1** | Dashboard de medição (`utils/quality-dashboard.sh`)| F-Q2       | 🟢 Baixo |
| **SQ2** | Captura do baseline (antes da remoção)             | F-Q2       | 🟢 Baixo |
| **SQ3** | Remoção da quantização: structs + loaders          | F-Q1       | 🔴 Alto  |
| **SQ4** | Remoção da quantização: kernels SIMD               | F-Q1       | 🔴 Alto  |
| **SQ5** | Medição pós-remoção + decisão Go/NoGo              | F-Q1, F-Q2 | 🟡 Médio |
| **SQ6** | Cleanup de documentação e thresholds               | F-Q3       | 🟢 Baixo |
