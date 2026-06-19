<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Mapa de Features do nam-rs — Aderência ao NeuralAmpModelerCore 🧩

> **Propósito**: mapeia **lacunas de funcionalidade** do `nam-rs` frente ao NAMCore v0.5.3.
> Para **problemas de produto** consulte **`TODO-problemas.md`**; para **otimização
> estrutural** consulte **`TODO-optimize.md`**.
>
> **Última auditoria**: jun/2026 (revisor-auditor + planejador-arquiteto).
>
> **Filosofia**: o `nam-rs` quer **superar** o original, mas a superação deve ser **correta**:
> nenhum recurso oficial silenciosamente abandonado; RT-safety preservada; paridade
> auditável (ESR/SNR vs C++ v0.5.3).

---

## Sumário Ativo

| ID      | Feature ausente / parcial                                           | Impacto p/ Produto | Público real         | Esforço |
| ------- | ------------------------------------------------------------------- |:------------------:| -------------------- |:-------:|
| **F2**  | **Multi-condição / FiLM** (`condition_size > 1`, `condition_dsp`)   | 🔴 Crítico (A2)    | Usuários A2 oficiais | Alto    |
| **F3**  | **Motor A2 geral** (gating, ativações, `head1x1`, `bn≠ch`)          | 🟠 Alto            | A2 avançado/futuro   | Alto    |
| **F5**  | **SlimmableWavenet** (slicing dinâmico de canais)                   | 🟠 Médio-Alto      | Pedais/embarcados    | Alto    |
| **F6**  | **Post-stack Head** (sub-objeto `head` multi-camada do WaveNet)     | 🟡 Médio           | Modelos custom       | Médio   |
| **F7**  | **LSTM arbitrário** (`hidden_size`/`num_layers` fora dos 10 perfis) | 🟠 Médio-Alto      | Capturas LSTM custom | Médio   |
| **F8**  | **Biblioteca completa de ativações** (PReLU, SiLU, etc.)            | 🟡 Médio           | A2 geral + custom    | Médio   |
| **F9**  | **Convoluções agrupadas/depthwise** (`groups > 1`)                  | 🟡 Médio           | A2 geral + custom    | Médio   |
| **F4**  | **ConvNet** (arquitetura legada)                                    | 🟢 Baixo           | Nicho/legado         | Médio   |
| **F10** | **Modelos multi-canal** (`in/out_channels > 1`)                     | 🟢 Baixo           | Experimental         | Médio   |
| **F11** | **Container aninhado** + cobertura `SlimmableContainer` real        | 🟢 Baixo           | Quality-scaling      | Baixo   |

> **Dependências**: F2 ⊃ F3 ⊃ {F8, F9}. F7 é ortogonal. F5 depende do motor dinâmico (F1 DONE).

---

## Diagnóstico Verificado (probe de carga, jun/2026 — atualizado)

| Modelo oficial                | Resultado no nam-rs                     | Feature que destrava |
| ----------------------------- | --------------------------------------- | -------------------- |
| `wavenet_a1_standard.nam`     | ✅ Carrega (golden oficial)             | —                    |
| `lstm.nam`                    | ✅ Carrega (golden oficial)             | —                    |
| `wavenet.nam` (CH=3, livre)   | ✅ Carrega (motor dinâmico — F1 DONE)   | —                    |
| `slimmable_wavenet.nam`       | ❌ "A2 shape not recognized"            | **F5**               |
| `wavenet_a2_max.nam` (cond=8) | ❌ "only condition_size=1 is supported" | **F2 / F3**          |
| `wavenet_condition_dsp.nam`   | ❌ "only condition_size=1 is supported" | **F2 / F3**          |
| `slimmable_container.nam`     | ❌ "submodel build failed"              | **F5 / F11**         |

---

## F2 — 🔴 Multi-condição / FiLM (`condition_size > 1`)

**O que é.** Conditioning arbitrário + FiLM (Feature-wise Linear Modulation) e
`condition_dsp`. O `nam-rs` fixa COND=1 como const-generic e rejeita multi-condição
(verificado: `topology.rs:338,587-595`).

**Evidência de rejeição (verificada jun/2026):**

- `topology.rs:338`: `if l0.condition_size != Some(1)` (A2 fast-path)
- `topology.rs:593`: `"condition_size={}, but only condition_size=1 is supported"`
- Estruturas FiLM existem em `src/models/a2/film.rs` (reservadas para futuro).
- Testes de rejeição: `nam_json_test.rs:206,903`.

**Público real — incerteza a resolver.** A fast-path A2 do NAMCore (`a2_fast.cpp`) é
**sem FiLM**. Os modelos A2 de produção do TONE3000 provavelmente casam com a fast-path.
Modelos FiLM/cond podem ser variantes de pesquisa. Conclusão depende de corpus real.

### **Diretrizes**

1. Implementar `FiLM`, `GatingActivation`/`BlendingActivation` e `condition_dsp`.
2. Converter testes de rejeição → golden positivo (sem buraco de cobertura).
3. Paridade ESR/SNR vs C++ v0.5.3.

---

## F3 — 🟠 Motor A2 geral (além da fast-path)

**O que é.** Hoje o `nam-rs` só roda A2 na fast-path rígida (23 camadas, K=6/15,
LeakyReLU, CH∈{3,8}, sem gating/FiLM/`head1x1`). O NAMCore suporta gating
(`GATED/BLENDED`), `head1x1`/`layer1x1` configuráveis, `bottleneck ≠ channels`,
ativações heterogêneas.

**Oportunidade de otimização (O5/S1).** A2 head conv (`head.rs:96-118`) é **100%
escalar** no hot-path (verificado jun/2026: laço aninhado `frames × 16 taps × CH`
sem nenhum intríseco SIMD, `get_unchecked` puro). Para CH=8 são 128 FMAs escalares/frame.
O motor A2 geral deve **nascer SIMD** neste ponto.

### **Diretrizes:**

1. Motor A2 dinâmico (alocação no load) com downcast para fast-path const-generic.
2. Reaproveitar F2 (FiLM/gating) e F8/F9 (ativações/grouped conv).
3. Vetorizar head conv com AVX2/FMA ao generalizar.

---

## F5 — 🟠 SlimmableWavenet

**O que é.** WaveNet com slicing dinâmico de canais por qualidade (`val ∈ [0,1]`).
O trait `SlimmableModel` e `ContainerModel` existem; o slicing real está planejado
(`src/models/slimmable.rs:17`).

### **Diretrizes::**

1. Implementar slicing por extração de pesos no load/stage assíncrono.
2. Swap atômico via SPSC GC (padrão existente).
3. Integrar ao `adaptive.rs`.

---

## F6 — 🟡 Post-stack Head

**O que é.** Head multi-camada pós-stack (`detail::Head`). Rejeitado explicitamente
(verificado: `topology.rs:601-607` — _"WaveNet 'head' (post-stack sub-object) is
not supported (F6)"_).

### **Diretrizes:::**

1. Implementar `Head` (Conv1D em cadeia + ativação).
2. Somar ao `receptive_field` de prewarm.

---

## F7 — 🟠 LSTM arbitrário

**O que é.** O NAMCore aceita LSTM com qualquer `hidden_size`/`num_layers`. O `nam-rs`
tem **10 perfis estáticos** (verificado: `dispatcher/lstm/dispatch.rs:9-51` — match
explícito de (1,3), (1,8), (1,12), (1,16), (1,24), (2,8), (2,12), (2,16), (1,40), (2,24)).

### **Diretrizes-**

1. Caminho LSTM dinâmico (dimensões no load) preservando fast-path const-generic.
2. Mesma filosofia de F1 (dispatch híbrido).

---

## F8 — 🟡 Biblioteca completa de ativações

**O que é.** NAMCore oferece 12 ativações. O `nam-rs` exercita só Tanh (A1) e
LeakyReLU (A2 fast-path). Resta marcado como futuro em `src/models/a2/activations.rs`.

### **Diretrizes--**

1. Portar `ActivationConfig` (string ou objeto com params).
2. Reaproveitar `src/math/activations/` (já há silu/relu/prelu/softsign/sigmoid/tanh).
3. Toggle global de fast-tanh (interação com política de fidelidade).

---

## F9 — 🟡 Convoluções agrupadas/depthwise (`groups > 1`)

**O que é.** Conv1D/Conv1x1 com `groups>1`. O `nam-rs` assume `groups==1`.

### **Diretrizes---**

1. Generalizar kernels Conv1D/Conv1x1 para grupos.
2. Manter fast-path `groups==1` intacto e SIMD-otimizado.

---

## F4 — 🟢 ConvNet (legado)

**O que é.** Arquitetura oficial `ConvNet` (`NAM/convnet.{cpp,h}`). Zero matches no
`src/` do nam-rs (verificado jun/2026). Volume marginal no TONE3000.

**Decisão**: provavelmente **fora de escopo**, salvo demanda.

---

## F10 — 🟢 Modelos multi-canal (`in/out_channels > 1`)

**Decisão**: fora de escopo (o ecossistema NAM é dominantemente mono).

---

## F11 — 🟢 Container aninhado + cobertura real

**O que é.** `SlimmableContainer` existe (`src/models/container.rs`) mas rejeita
aninhamento e depende de F5 para golden com modelo real.

---

## Histórico — Features Concluídas

Expandir itens concluídos (F1, F12)

### F1 — ✅ WaveNet genérico (dispatcher dinâmico, escopo A1 COND=1) — DONE

Despacho híbrido ativo. Geometrias conhecidas usam fast-path const-generic; qualquer
outra geometria A1 válida (COND=1, sem head pós-stack) cai no motor dinâmico
(verificado: `src/loader/dispatcher/wavenet/dynamic.rs:107`, `mod.rs:132`).
Paridade C++ validada por golden `wavenet_official.nam` (CH=3).

### F12 — ✅ TONE3000 + expansão de `tests/fixtures/models/` — DONE

Corpus real de modelos A2 de qualidade em `tests/fixtures/models-nondist`. Modelos A1
Lite e custom encontrados. Encerrado pelo PO: _"Vamos ter de ir nos virando com o que temos"_.

</details>

---

## Recomendações para Golden de Novas Features

- **Modelo oficial real, não sintético.** Usar os `.nam` oficiais que hoje falham como fonte.
- **Cross-reference C++ pinado, gate scale-invariant.** Validar por ESR/SNR (não MSE absoluto).
- **Conversão de `test_loader_gap_*` → golden positivo.** Migrar de "rejeita" para
  "casa com o C++" ao ganhar suporte.
- **Calibração por medição documentada.** Registrar em `get_calibrated_threshold`.
- **Cobertura multi-SR e determinismo.** Acrescentar aos gates v2 e determinismo bitwise.

---

## Nota de Método (para o planejador-arquiteto)

- Respeitar dependências: F2⊃F3⊃{F8,F9}; F7 é ortogonal.
- RT-safety: alocação só no load, hot-path zero-alloc/lock/panic.
- Regra de golden: "todo golden deve poder falhar".
- Ao implementar F3: vetorizar head conv (O5/S1) como critério de aceite.
- Ao implementar F7: mesma filosofia de dispatch híbrido de F1.
