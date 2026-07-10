<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-sprints.md — Planejamento Ágil (Sprints e Tarefas)

Este documento organiza a execução física dos planos de melhoria e correções de auditoria em sprints e tarefas de engenharia detalhadas, com foco em segurança, rastreabilidade e mitigação de riscos.

---

## Relação com Achados e Epicos

* **Referência:** `TODO-findings.md` § Epic F1-A (concluído) e Epic F1-B (em planejamento)

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

## Sprint 2: Preservação do Teste de Estresse Numérico e Documentação

Esta sprint foca em preservar a fixture degenerada original de FiLM como um teste dedicado de robustez numérica sob adversidades (denominada `wavenet_a2_film_chaos_stress.nam`), garantindo que o motor de inferência processe regimes sem bias de identidade sem produzir inconsistências numéricas (NaN/Inf). Paralelamente, todas as documentações e mapas de paridade são atualizados para refletir a nova realidade das medições ideais do FiLM.

---

### **Tarefa T2.1: Registro e Geração do Golden para o Chaos Stress Fixture**

* **Objetivo:** Registrar o novo fixture `wavenet_a2_film_chaos_stress.nam` no catálogo e gerar seu vetor binário de referência C++.
* **Detalhes técnicos:**
  * Atualizar a lista `CATALOG` no script [golden_gen_build.sh](file:///home/fabio/nam-rs/tests/fixtures/golden_gen_build.sh) para adicionar a entrada `"wavenet_a2_film_chaos_stress.nam:golden_wavenet_a2_film_chaos_stress:A2-FiLM Chaos Stress (CH=3):none"`.
  * Executar `./tests/fixtures/golden_gen_build.sh` para renderizar o golden binary de referência `golden_wavenet_a2_film_chaos_stress.bin`.
* **Arquivos:** [golden_gen_build.sh](file:///home/fabio/nam-rs/tests/fixtures/golden_gen_build.sh)
* **Status:**  ✅ Concluído

### **Tarefa T2.2: Adição do Teste de Golden Vector e Threshold Próprio**

* **Objetivo:** Adicionar o teste unitário de golden vector do novo modelo e configurar os thresholds originais dele no Rust.
* **Detalhes técnicos:**
  * Configurar a chave `"wavenet_a2_film_chaos_stress"` em [validation.rs](file:///home/fabio/nam-rs/tests/common/validation.rs) usando os antigos thresholds degenerados (ex: SNR = `12.0` dB, ESR = `3.5e-2`).
  * Adicionar o teste unitário `test_golden_vectors_wavenet_a2_film_chaos_stress` em [golden_vectors.rs](file:///home/fabio/nam-rs/tests/models/golden_vectors.rs).
  * Registrar a chave `"wavenet_a2_film_chaos_stress"` no teste de calibração em [threshold_calibration.rs](file:///home/fabio/nam-rs/tests/models/threshold_calibration.rs).
* **Arquivos:**
  * [validation.rs](file:///home/fabio/nam-rs/tests/common/validation.rs)
  * [golden_vectors.rs](file:///home/fabio/nam-rs/tests/models/golden_vectors.rs)
  * [threshold_calibration.rs](file:///home/fabio/nam-rs/tests/models/threshold_calibration.rs)
* **Status:**  ✅ Concluído

### **Tarefa T2.3: Atualização da Documentação Técnica de Fidelidade e Paridade**

* **Objetivo:** Atualizar os arquivos markdown do repositório para refletir a correção da causa raiz e as medições reais de SNR/ESR obtidas.
* **Detalhes técnicos:**
  * Substituir o diagnóstico incorreto de associatividade pelo correto (bias de inicialização do gerador) no Achado 1 de [TODO-parity.md](file:///home/fabio/nam-rs/TODO-parity.md).
  * Atualizar a seção §4.3 de [cpp_parity_map.md](file:///home/fabio/nam-rs/docs/cpp_parity_map.md) explicando a causa real e a mitigação pelo teste de estresse de caos numérico.
  * Atualizar as tabelas de thresholds e medições do FiLM nos arquivos [audio_fidelity_map.md](file:///home/fabio/nam-rs/docs/audio_fidelity_map.md) e [perceptual_validation.md](file:///home/fabio/nam-rs/docs/perceptual_validation.md).
* **Arquivos:**
  * [TODO-parity.md](file:///home/fabio/nam-rs/TODO-parity.md)
  * [cpp_parity_map.md](file:///home/fabio/nam-rs/docs/cpp_parity_map.md)
  * [audio_fidelity_map.md](file:///home/fabio/nam-rs/docs/audio_fidelity_map.md)
  * [perceptual_validation.md](file:///home/fabio/nam-rs/docs/perceptual_validation.md)
* **Status:**  ✅ Concluído

### **Tarefa T2.4: Homologação de Sprints e Execução de Testes Rápidos**

* **Objetivo:** Homologar a entrega e certificar conformidade completa.
* **Detalhes técnicos:**
  * Executar `utils/lints.sh` e `utils/tests-quick.sh`.
* **Status:**  ✅ Concluído

---

### 3. Matriz de Risco e Mitigação de Ambas as Sprints

| Risco | Impacto | Mitigação |
| :--- | :--- | :--- |
| **Quebra do determinismo dos pesos sintéticos (Sprint 1)** | Médio | Garantir que o RNG use seed fixa e a lógica de geração preserve a dimensionalidade correta dos tensores. |
| **Tolerâncias de Threshold estritas demais (Sprint 1)** | Baixo | Adicionar margens adequadas nos novos thresholds (~3 dB em SNR e fator extra de tolerância em ESR) para tolerar pequenas variações ambientais de float. |
| **Divergência imprevista no modelo de caos (Sprint 2)** | Baixo | Como o modelo e os pesos são idênticos aos anteriores, a paridade com o golden original e o C++ deve se manter estável nos níveis reportados antes da correção. |
| **Documentações defasadas (Sprint 2)** | Baixo | Realizar revisão estrita para certificar que nenhum local ainda refira-se ao diagnóstico incorreto de associatividade SIMD de forma ativa. |
