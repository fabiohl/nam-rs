<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-sprints.md — Planejamento Ágil (Sprints e Tarefas)

Este documento organiza a execução física dos planos de melhoria e correções de auditoria em sprints e tarefas de engenharia detalhadas, com foco em segurança, rastreabilidade e mitigação de riscos.

---

## Relação com Achados e Epicos

* **Referência:** `TODO-findings.md` § Epic F1-A — Correção da causa raiz do fixture FiLM (baixo risco, alto valor de credibilidade)

---

## Sprint 1: Correção do Fixture FiLM e Recalibração de Thresholds

### 1. Resumo da Sprint

Esta sprint foca em corrigir a inicialização do canal de `scale` do FiLM no script de fixtures sintéticos de teste. O objetivo é remover o falso positivo "audível" (ESR inflado) gerado pelo esmagamento de energia do sinal, estabelecendo thresholds reais e confiáveis de fidelidade de áudio de acordo com a causa raiz documentada no Achado F1 do `TODO-findings.md`.

---

### 2. Tarefas Técnicas

#### **Tarefa T1.1: Correção do Script Gerador de Fixtures (Python)**

* **Objetivo:** Aplicar o bias de identidade `+1.0` ao canal de `scale` do FiLM no gerador.
* **Detalhes técnicos:**
  * Modificar a função `generate_weights_film` em [generate_a2_fixtures.py](file:///home/fabio/nam-rs/tests/fixtures/generate_a2_fixtures.py) para que a metade correspondente ao bias do canal de `scale` seja deslocada por `+1.0` (tornando-a *identity-biased*), enquanto a metade de `shift` permanece centrada em `0.0`.
  * Preservar a consistência do RNG (Random Number Generator) para não corromper outras partes dos fixtures.
* **Arquivos:** [generate_a2_fixtures.py](file:///home/fabio/nam-rs/tests/fixtures/generate_a2_fixtures.py)
* **Status:**  ✅ Concluído

#### **Tarefa T1.2: Regeneração dos Fixtures e Golden Vectors C++**

* **Objetivo:** Recriar os arquivos `.nam` e os vetores de referência C++ `.bin`.
* **Detalhes técnicos:**
  * Executar o script [golden_gen_build.sh](file:///home/fabio/nam-rs/tests/fixtures/golden_gen_build.sh) para processar o script Python corrigido e renderizar os novos arquivos binários de referência com o NAMCore C++.
* **Arquivos:**
  * [wavenet_a2_film_lite.nam](file:///home/fabio/nam-rs/tests/fixtures/models/wavenet_a2_film_lite.nam)
  * [wavenet_a2_film_full.nam](file:///home/fabio/nam-rs/tests/fixtures/models/wavenet_a2_film_full.nam)
  * [golden_wavenet_a2_film_lite.bin](file:///home/fabio/nam-rs/tests/fixtures/golden_wavenet_a2_film_lite.bin)
  * [golden_wavenet_a2_film_full.bin](file:///home/fabio/nam-rs/tests/fixtures/golden_wavenet_a2_film_full.bin)
* **Status:**  ✅ Concluído

#### **Tarefa T1.3: Medição dos Novos Valores de ESR/SNR**

* **Objetivo:** Coletar as métricas reais de fidelidade obtidas contra a nova referência de alta energia do sinal.
* **Detalhes técnicos:**
  * Executar a suíte de testes de golden vectors para os modelos FiLM, capturando as falhas intencionais de threshold antigos e extraindo as medições exatas de SNR, ESR e MR-STFT.
  * A expectativa é de que o SNR suba para a faixa de `>90 dB` (ESR `<1e-9`).
* **Status:**  ✅ Concluído

#### **Tarefa T1.4: Recalibração de Thresholds de Validação (Rust)**

* **Objetivo:** Atualizar os limites de aceitação na suíte de testes automatizados com base nas medições reais da Tarefa T1.3.
* **Detalhes técnicos:**
  * Ajustar as chaves `"wavenet_a2_film_lite"` (linhas 589-592) e `"wavenet_a2_film_full"` (linhas 599-602) em [validation.rs](file:///home/fabio/nam-rs/tests/common/validation.rs).
  * Atualizar as descrições nos comentários que detalham os valores medidos e a margem de tolerância adicionada.
* **Arquivos:** [validation.rs](file:///home/fabio/nam-rs/tests/common/validation.rs)
* **Status:**  ✅ Concluído

#### **Tarefa T1.5: Homologação e Verificação Final (Lints & Quick Tests)**

* **Objetivo:** Confirmar que toda a base compila perfeitamente e passa sem erros de conformidade ou regressões.
* **Detalhes técnicos:**
  * Rodar `utils/lints.sh` para verificar formatação e cabeçalhos SPDX.
  * Rodar `utils/tests-quick.sh` para atestar a validação completa de fidelidade sob os novos limites.
* **Status:**  ✅ Concluído

---

### 3. Matriz de Risco e Mitigação

| Risco | Impacto | Mitigação |
| :--- | :--- | :--- |
| **Quebra do determinismo dos pesos sintéticos** | Médio | Garantir que o RNG use seed fixa e a lógica de geração preserve a dimensionalidade correta dos tensores. |
| **Tolerâncias de Threshold estritas demais** | Baixo | Adicionar margens adequadas nos novos thresholds (~3 dB em SNR e fator extra de tolerância em ESR) para tolerar pequenas variações ambientais de float. |
| **Divergências estruturais remanescentes** | Baixo | Validar os resultados contra o oráculo f64 (`reference_oracle`) para ter certeza de que o motor de inferência Rust do FiLM está matematicamente correto. |
