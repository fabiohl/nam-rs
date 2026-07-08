<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->
# Pesquisador Inovador & Revisor Auditor: Análise de Qualidade de Modelos (FiLM e LSTM)

Este documento sumariza os achados críticos da investigação de paridade e divergência reportados no `utils/quality-dashboard.sh` para os modelos WaveNet A2-FiLM Lite e BossLSTM.
Ensejará atualização de: `docs/audio_fidelity_map.md`, `docs/fastmath-approximations.md`, `docs/cpp_parity_map.md` e `docs/architecture.md`.

## Achado 1: FiLM Floating-Point Associativity Gap

**Diagnóstico:** O teste de fidelidade acusa um ESR de 1.54e-2 (SNR 18.1 dB) audível para o modelo `WaveNet A2-FiLM-Lite (CH=3)`. Contudo, o oráculo (f64) do `nam-rs` comprova matematicamente que o motor FiLM nativo em Rust produz o limite exato teórico do ponto flutuante de dupla precisão (ESR ≈ 1e-14).
A divergência se concentra em uma diferença estrutural entre o `nam-rs` (que utiliza uma redução `_mm256_fmadd_ps` otimizada em árvore de AVX2) e o `NAMCore` nativo em C++ (que repassa a multiplicação `MatrixXf * VectorXf` para o framework genérico e dinâmico da biblioteca `Eigen`). A biblioteca C++ Eigen adota empacotamentos e passos associativos diferentes (possivelmente column-major vectors ou reduções sequenciais) que propagam arredondamentos numéricos f32 em ordem distinta.

**Solução Exigida:**

1. Adoção do modelo de "Imitar a Origem": Para maximizar a sustentabilidade e paridade exata, não tentaremos contornar a perda com heurísticas locais artificiais (como o *Kahan Summation*, que sozinho não resolveu o interop gap).
2. Investigação estrutural da árvore de redução vetorial do `Eigen` C++ no NAMCore e adequação na função `dot_product_avx2` e no loop `cond_to_scale_shift` dentro de `src/models/a2/film.rs` para mimetizar exatamente os mesmos passos aritméticos do C++.

## Achado 2: LSTM Recurrent State Drift

**Diagnóstico:** Ao contrário do mapeado preliminarmente em algumas seções da documentação oficial, os desvios contínuos dos modelos `BossLSTM 1x16` (1.04e-2 ESR) e `BossLSTM 2x8` (2.68e-3 ESR) *não* foram totalmente solucionados pela remoção do `f16c`. O "bit-exactness" outrora documentado é válido apenas para excitações de curtíssima duração (2000 samples). Em sinais de produção e stream contínuo, a acumulação sucessiva do *Cell State* através das portas (`f32`) invariavelmente produz o *Recurrent Drift*.

**Solução Exigida:**

1. Alteração do loop interno de processamento nas células da LSTM para aplicar correção ativa de somatório.
2. Inclusão de um novo estado dinâmico (`cell_error`) correspondente ao shadow state de erro da célula.
3. Modificação das sub-rotinas SIMD em `src/math/lstm/gates.rs` (AVX2, AVX-512 e escalar) para executar um passo de compensação (soma de Kahan/Neumaier) em cada iteração recorrente (`new_cs = f * cs + i * g`).
4. Execução mandatória de *benchmark* de stress em `cargo bench` e no dashboard de latência. A folga atual de budget de CPU para LSTM (usa menos de 1%) precisará absorver o overhead numérico extra (operações adicionais nos registradores SIMD). Decisão final (GO / No GO) de integração condicionada aos números do benchmark.

## Achado 3: `wavenet_a2_max.nam` Ativamente Quebrado (Divergência Crítica no `condition_dsp`)

**Diagnóstico:** O modelo `wavenet_a2_max` (CH=4, condition_size=8) é o único flagship do A2 ativamente quebrado, gerando uma saída com erro severo (MSE ≈ 2.46e3, SNR ≈ -15.6 dB, ESR ≈ 36.1). Apesar do modelo carregar corretamente e executar via `WaveNetA2Dyn`, seu áudio de saída não faz sentido e, por cautela, ele foi contido de forma hardcoded no dispatcher (`is_disabled_broken_a2_flagship`).
A investigação atual já descartou problemas no layout de pesos na `head1x1`. O desvio fundamental reside firmemente no processamento do sub-modelo interno `condition_dsp` (uma sub-WaveNet com SiLU, bottleneck=6, head_size=4, e FiLM). Historicamente, uma regressão drástica foi injetada no código de produção quando ele tentou mimetizar um oráculo f64 defeituoso (o oráculo possuía um bug dimensional que cuspia 1 valor por frame em vez de 8, e contava incorretamente os pesos da `head1x1`).

**Solução Exigida (Desbloqueio de Sprints):**

1. **Consertar a régua (Oráculo f64)**: O oráculo precisa ser corrigido para alinhar com o comportamento C++ correto (produzir `condition_size=8` valores por frame e calcular corretamente a geometria agrupada da `head1x1`). Sem isso, não há ponto de referência validável.
2. Homologar os testes `test_oracle_*_a2_generic` isolados para garantir precisão flutuante ideal (golden-match do f64 oráculo vs C++ interno).
3. **Consertar a produção (Rust f32)**: Após a validação do oráculo corrigido, implementar as mesmas correções estruturais no `WaveNetA2Dyn` e `condition_dsp` de produção.
4. Remover o flag limitador `is_disabled_broken_a2_flagship` e assegurar validação final via `live_cross_validation` reportando SNR sadio e paridade integral.

---

## Estrutura de Épicos para Execução (Fechamento)

- **Épico 6: Reabilitação do Flagship `wavenet_a2_max`** (Correção mandatória dimensional no oráculo f64 do `condition_dsp` seguida da transposição estrutural em produção, recuperando a paridade de ISA para reativar o modelo com segurança).
