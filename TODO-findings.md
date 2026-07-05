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

## Finding F-Q2: Script de Dashboard `docs/quality-dashboard.sh` necessário para medir impacto

### Contexto F-Q2

Antes (baseline) e depois (pós-remoção) da mudança, precisamos de um instrumento científico que colha dados dos testes existentes e os apresente de forma amigável. Este instrumento será permanente e reaproveitável para qualquer mudança futura.

### Solução Proposta F-Q2

Script `docs/quality-dashboard.sh` que:

1. Roda `golden_vectors`, `reference_oracle_f64`, `spectral_fidelity`, `isa_parity` em release
2. Roda `regression_gate` benchmark
3. Parseia o stdout e gera relatório humano com:
   - Resumo não-científico (ESR → % de erro + veredicto humano)
   - Latência vs budget RT (% de folga + sugestões de upgrades como oversampling)
   - Tabela técnica detalhada (ESR, SNR, MSE, MR-STFT)

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
| **SQ1** | Dashboard de medição (`docs/quality-dashboard.sh`) | F-Q2       | 🟢 Baixo |
| **SQ2** | Captura do baseline (antes da remoção)             | F-Q2       | 🟢 Baixo |
| **SQ3** | Remoção da quantização: structs + loaders          | F-Q1       | 🔴 Alto  |
| **SQ4** | Remoção da quantização: kernels SIMD               | F-Q1       | 🔴 Alto  |
| **SQ5** | Medição pós-remoção + decisão Go/NoGo              | F-Q1, F-Q2 | 🟡 Médio |
| **SQ6** | Cleanup de documentação e thresholds               | F-Q3       | 🟢 Baixo |
