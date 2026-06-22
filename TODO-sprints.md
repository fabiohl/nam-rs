<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved. -->

# TODO-sprints.md — Planejamento Ágil de Sprints e Tarefas

> **Origem:** Skill `planejador-arquiteto`
>
> Este documento traduz os achados apontados em `TODO-findings.md` em um plano executável,
> subdividido em Sprints focadas e Tarefas Técnicas altamente detalhadas, prontas para a
> ação da skill `implementador`.

## Visão Geral das Entregas (Baseado em TODO-findings.md)

Neste planejamento abordamos as seguintes demandas de alta prioridade solicitadas:

* **Épico E1**: Correção do Pipeline de Validação C++ (CI Verde)
* **Épico E8**: Rigor e Cobertura da Suíte de Testes

---

## 🏃 Sprint 1: Recuperação de Paridade do CI (Épico E1)

**Objetivo:** Restaurar a auditoria longa para 6/6 fases verdes (CI Verde), consertando as falhas falso-positivas do gate LUFS.
**Risco:** Baixo
**Base:** [TODO-findings.md - F8](TODO-findings.md)

### 📝 Tarefa 1.1: Tornar o Gate LUFS flexível em `cpp_parity.rs` [DONE]

**Responsável Sugerido:** `implementador` / `debugger`
**Contexto:** O script de teste de paridade C++ compara referências numéricas, mas modelos silenciosos como `wavenet_dyn_free` caem na trava restrita do gate LUFS `[-50, +10]` e quebram falsamente a suite longa de testes.
**Critérios de Aceitação:**

1. Em `tests/cpp_parity.rs`, parametrizar as funções de base (`run_render_comparison`, `run_v1`, e `run_v2_multi_sr`) para aceitar um argumento `check_lufs_gate: bool`.
2. Onde a função aciona `report_dsp_fidelity(...)`, inserir lógica para despachar para `report_dsp_fidelity` (com LUFS gate ativo) quando `check_lufs_gate == true`, ou para `report_dsp_fidelity_no_lufs` (bypass seguro, informando log) quando for `false`.
3. Ajustar todas as chamadas nas funções de testes específicos para passarem `check_lufs_gate = false` nos cenários explicitamente reconhecidos por terem baixa energia: testes afetando `wavenet_dyn`, `v2_wavenet_dyn`, e `v2_lstm_dyn` (e instâncias silenciosas por design como `wavenet_dyn_free` e `lstm_dyn_test`). Os demais continuarão passando `true` por padrão.
4. Executar os testes em `utils/tests-long.sh` (fase de Parity) localmente e comprovar que as 3 falhas de falso-positivo de CI foram consertadas. O log terminal deve evidenciar o skip do gate corretamente via impressão `ⓘ LUFS gate skipped`.

---

## 🏃 Sprint 2: Balanceamento e Melhoria dos Gates de Lane Rápida e Scripts (Épico E8)

**Objetivo:** Aumentar a utilidade dos scripts diários da suite unitária/integração (`tests-quick.sh`) não deixando áreas críticas inteiramente dependentes dos testes longos (ignorados por padrão), fortalecendo também as verificações base dos stress testes.
**Risco:** Baixo
**Base:** [TODO-findings.md - F30, F31, F34]

### 📝 Tarefa 2.1: Promover Testes Críticos "Smoke" para a Lane Rápida [DONE]

**Responsável Sugerido:** `implementador`
**Contexto:** Validações profundas e demoradas são mantidas sobre `#[ignore]` no CI rápido. Porém não ter uma mínima cobertura do Parser ou Parity nos runs locais compromete a qualidade da verificação. (Ref: F30)
**Critérios de Aceitação:**

1. Visitar os módulos isolados de suíte pesada: `tests/proptest_parsers.rs` e `tests/proptest_math.rs`.
2. Remover a anotação `#[ignore]` de apenas **1 caso representativo (smoke-test rápido)** do parser (ex: injeção básica sem estender iterações pra milhares) e 1 de precisão SIMD tanh.
3. Em `tests/cpp_parity.rs`, escolher uma amostra estática/rápida unitária de formato comum (ex. modelo Wavenet menor), e retirar o `#[ignore]`.
4. Assegurar que ao rodar `utils/tests-quick.sh`, todos os novos testes ativados passem em Verde e o tempo total de rodada não aumente abusivamente (< 20-30s extra em média no debug run local).

### 📝 Tarefa 2.2: Suíte de Validação Intermediária [DONE]

**Responsável Sugerido:** `implementador`
**Contexto:** O time requer uma etapa viável antes de aprovar PR sem aguardar longos ciclos e com garantias robustas sobre estresse de parsers proptest. (Ref: F30)
**Critérios de Aceitação:**

1. Desenvolver um script `utils/tests-medium.sh` nos moldes e com a mesma maturidade de headers (`SPDX-License-Identifier: Apache-2.0` no topo). **→ Feito; fusionado com `tests-quick.sh` como Phase 2 (medium validation) por coesão funcional. `tests-medium.sh` removido.**
2. Neste script habilitar `set -euo pipefail` (com trap similar aos scripts utils atuais). **→ Adotado no script fundido.**
3. Lançar o comando do Cargo contendo uma flag combinada ou seleta para despachar todos proptests e o arquivo inteiro `cpp_parity` do repositório (`cargo test --test cpp_parity --test proptest_parsers --test proptest_math -- --ignored` por exemplo). **→ Executado com sucesso.**
4. Garantir que este run seja concluído e passe sem erros. **→ 43 passed, 0 failed.**
5. **Adicional (fusão):** Suíte intermediária fusionada com `utils/tests-quick.sh` como Phase 2, com skip condicional se goldens/NeuralAmpModelerCore ausentes. O script agora é o PR gate unificado: fast tests + medium validation + CLAP audit.

### 📝 Tarefa 2.3: Elevar Rigor contra Drifts Silenciosos nos Testes de Estresse [DONE]

**Responsável Sugerido:** `implementador`
**Contexto:** O loop DSP que verifica processamento baseando-se meramente em finitude `.is_finite()` em arrays não reage a sinais mortos ou zeros inesperados num caso de falha catastrófica matemática ou memory corrompida. E oráculos de threshold não estão calibrados adequadamente. (Ref: F31, F34)
**Critérios de Aceitação:**

1. Navegar nos testes `soak_test.rs`, `nam_infer_test.rs` ou `processor_test.rs`, mapeando os trechos com `assert!(val.is_finite())` processados após alimentação real.
2. Reforçar esses checks calculando adicionalmente o **RMS (Root-Mean-Square)** do buffer de saída se a alimentação não foi zero puro. Estabelecer e inserir uma checagem de banda tipo `assert!(rms > 0.0001 && rms < 10.0, "RMS do loop out of range: {}", rms)`.
3. Em `test_model_gain_calibration` de `processor_test.rs:1875-1882`, o teste tem uma restrição inexpressiva. Ajustar as bandas limitrofes do RMS de `rms < 0.4` para `0.05 <= rms && rms <= 0.20` baseando-se nos golden references reais se providos ou após teste empírico (ou readequar os bounds exatos conforme run-teste, mas documentá-lo propriamente).

---

## 🏃 Sprint 4: Cobertura Unitária e Contratos em Código Unsafe (Épico E8)

**Objetivo:** Eliminar fragilidades não-checadas e testar falhas intencionalmente para assegurar que limites de segurança estão protegendo a integridade do processo de Host RT.
**Risco:** Médio (Requer assertiva e conhecimento das premissas numéricas/matrizes das arquiteturas internas).
**Base:** [TODO-findings.md - F22, F32, F33]

### 📝 Tarefa 3.1: Oráculos Desacoplados e Mapeamento de Testes Críticos Co-localizados

**Responsável Sugerido:** `implementador` / `debugger`
**Contexto:** Componentes chaves de inferência matemática e loaders de fronteiras não contam com verificação cruzada totalmente independente ou não têm baterias de testes focados nativos que capturam comportamentos fora da curva. (Ref: F22, F32)
**Critérios de Aceitação:**

1. Na referência escalar pura `src/math/common/scalar_ref/utility.rs` (no `tanh_slice_fallback`), substituir o backend compartilhado de Padé tanh por chamadas a referências padrão Rust `f32::tanh()` em um slice iterativo, estabelecendo verificação independente cruzada pura. Checar no `test_long.sh` de paridade se necessita de algum epsilon float tweak documentado (devido à reordenação).
2. Adicionar testes ao `src/loader/nam_json/validation.rs` cobrindo o comportamento do validador. Caso o arquivo possua linhas >= 300 (seguir as regras da casa), ele deve ficar em `src/loader/nam_json/validation_test.rs`. Criar mocks injetando `NaN` e dimensões inválidas extremas para assegurar robustez.
3. Repetir o exercício criando baterias básicas de `set_weights.rs` para atestar bounds/finitude.

### 📝 Tarefa 3.2: Garantir os Contratos Críticos em Zonas SIMD e Memórias Diretas

**Responsável Sugerido:** `implementador` / `pesquisador-inovador`
**Contexto:** Kernels transpositores de modelo não blindam ativamente (através de testes focados `should_panic`) operações em que callers pudessem burlar as restrições ou enviar matrizes desbalanceadas. (Ref: F33)
**Critérios de Aceitação:**

1. Selecionar o módulo de `grouped_conv1d.rs` e subjacentes de resampling e avaliar todos os callers `unsafe` das funções publicamente expostas ao kernel RT.
2. Escrever e adicionar cenários com anotações de atributo `#[should_panic]` que entregam shapes quebrados de dims/chans que seriam proibidos nas aproximações SIMD (passar um `vec` desalinhado e um layout faltando posições).
3. Assegurar que o programa abortará a execução por um `debug_assert!` ou bound safe panics internos antes de explodir por um `#GP` (General Protection Fault) / Memory Violation C-style segfault.
