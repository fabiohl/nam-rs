<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Feature à parte (fora do escopo de qualidade de goldens) — WaveNet genérico + FiLM/multi-condição 🧩

## Diagnóstico verificado (probe de carga em 2026-06-14)

O dispatcher do nam-rs aceita hoje **apenas um catálogo fixo** de topologias WaveNet e **rejeita** o resto.
Testando os modelos **oficiais** do `NeuralAmpModelerCore_v0.5.3/example_models/`:

| Modelo oficial              | Geometria                               | Resultado no nam-rs                                              |
| --------------------------- | --------------------------------------- | ---------------------------------------------------------------- |
| `wavenet_a1_standard.nam`   | WaveNet ch=16, cond=1 (real, 407 KB)    | ✅ **Carrega** (já é golden oficial)                             |
| `my_model.nam`              | == `wavenet_a1_standard` (md5 idêntico) | ✅ Carrega (redundante)                                          |
| `lstm.nam`                  | LSTM H=3, L=1                           | ✅ **Carrega** (já é golden oficial)                             |
| `wavenet.nam`               | WaveNet ch=3, cond=1, `[(3,2),(2,1)]`   | ❌ "topology not in catalog / dynamic fallback no longer avail." |
| `slimmable_wavenet.nam`     | WaveNet ch=3, cond=1, geometria livre   | ❌ "A2 shape not recognized"                                     |
| `wavenet_a2_max.nam`        | WaveNet ch=4, **cond=8** (FiLM)         | ❌ "only condition_size=1 is supported"                          |
| `wavenet_condition_dsp.nam` | WaveNet ch=3, **cond=3** (FiLM)         | ❌ "only condition_size=1 is supported"                          |
| `slimmable_container.nam`   | SlimmableContainer (3 submodelos)       | ❌ "submodel build failed" (depende dos acima)                   |

São **duas features distintas**, ambas recursos oficiais do NAMCore:

1. **WaveNet genérico (dispatcher dinâmico).** Suportar **qualquer** nº de camadas/canais/dilatações, não só o
   catálogo rígido (Standard/Lite/Feather/Nano/A2-Full/A2-Lite). Hoje o fallback dinâmico foi **removido** — então
   nam-rs não roda modelos NAM treinados arbitrariamente, que é o **caso de uso central** do ecossistema NAM.
   Risco/atenção: preservar RT-safety — toda alocação deve ocorrer **no load** (fora da audio-thread), mantendo o
   hot-path zero-alloc; o fast-path estático atual deve continuar sendo escolhido quando a geometria casar.
2. **Multi-condição / FiLM (`condition_size > 1`).** Implementar conditioning arbitrário + FiLM
   (refs C++: `NAM/wavenet/film.h`, `NAM/wavenet/gating_activations.h`). É o que destrava os A2 oficiais
   (`wavenet_a2_max`, `wavenet_condition_dsp`) e permitiria **elevar os goldens A2 de sintético→oficial**.

> Nota lateral (baixa prioridade): o NAMCore também tem a arquitetura **ConvNet** (`NAM/convnet.{cpp,h}`), não
> suportada pelo nam-rs e **sem** modelo oficial entre os `example_models`. Decidir em produto se entra no escopo.

## Recomendações para gerar bons goldens **destas** features (quando forem implementadas)

Para manter o padrão exemplar já estabelecido, os goldens das novas features devem seguir as mesmas regras:

- **Modelo oficial real, não sintético.** Usar diretamente os `.nam` oficiais que hoje falham —
  `wavenet.nam` (WaveNet genérico) e `wavenet_a2_max.nam` (FiLM) — como fonte. Isso **promove o A2 de
  sintético→oficial** e elimina a única ressalva de proveniência que resta nos goldens.
- **Cross-reference C++ pinado, gate scale-invariant.** Gerar a referência com o `render` do **v0.5.3** (mesmo
  pipeline do `golden_gen_build.sh`); validar Rust↔C++ por **ESR/SNR** (não MSE absoluto), como no A2 atual.
- **Conversão de `test_loader_gap_*` → golden positivo.** Cada modelo que hoje tem um teste de "rejeição"
  (`test_loader_gap_wavenet_a2_max`, `_condition_dsp`, `_slimmable_wavenet`, `_slimmable_container`) deve, ao
  ganhar suporte, **migrar** de "afirmo que rejeita" para "afirmo que casa com o C++" — sem deixar buraco de cobertura.
- **Calibração por medição documentada.** Registrar a entrada em `get_calibrated_threshold` com
  `// Measured: SNR=…, ESR=…` e margem (6–10 dB), como exige o meta-teste anti-placebo (T4.4).
- **Regime de amplitude realista.** Como os modelos oficiais são treinados, a saída já fica em faixa de áudio sã
  (pico ~0.1–0.5, LUFS ~−20) — o gate de plausibilidade LUFS (T4.3) passa naturalmente, **sem** o reescalonamento
  artificial que o A2 sintético exigiu (T2.5).
- **Cobertura multi-SR e determinismo.** Acrescentar os novos modelos aos gates v2 multi-SR (T4.2) e ao gate de
  determinismo bitwise por arquitetura (T4.5), fechando a malha de cobertura universal.
