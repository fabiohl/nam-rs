<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO Findings — Auditoria de Paridade e Robusteza (nam-rs)

Este documento centraliza os achados detalhados de auditoria estrutural e matemática baseados na revisão do código-fonte em relação à referência C++ (NAMcore) e aos testes de integridade. A resolução desses itens visa robustecer o nam-rs contra comportamentos indefinidos, mitigar vulnerabilidades e eliminar drifts de paridade.
Obs: Regularmente atualizar o "docs/cpp_parity_map.md" com o progresso obtido.

---

## 1. Achados de Categoria 7.2 🟠 Confirmed Code Bugs (Dormant)

Abaixo estão detalhados os bugs confirmados no código que poderiam produzir saídas silenciosamente incorretas ou pânico/comportamento indefinido (UB) sob condições específicas de modelos.

### Finding 7.2.1: Preaquecimento Incompleto do `condition_dsp` do Tipo LSTM no WaveNet

* **Componente:** `src/models/wavenet/model_dyn.rs` (linhas 348-350)
* **Descrição do Problema:** O método `prewarm_internal()` no motor dinâmico do WaveNet chama `cond_dsp.prewarm(0)` com um argumento fixo em zero.
* **Causa Raiz:** Quando o sub-modelo `condition_dsp` é um LSTM (configuração válida suportada oficialmente pelo C++), o prewarm dele é iterativo. Passar `0` faz com que o LSTM não execute nenhuma iteração de silêncio, mantendo os estados oculto e de célula (`_xh` e `_c`) contendo lixo ou valores não-convergidos. O primeiro `process()` subsequente consome um sinal de condicionamento não estabilizado.
* **Impacto:** Transiente audível indesejado no início do sinal de áudio se um modelo com `condition_dsp` LSTM for utilizado.
* **Solução Proposta:** Alterar a chamada para `cond_dsp.prewarm(cond_dsp.prewarm_samples())`. Desta forma, se o sub-modelo de condicionamento for um LSTM, ele receberá a quantidade correta de amostras de silêncio para convergir (geralmente `0.5 * sample_rate`).

### Finding 7.2.2: Omissão de Gating/FiLM/Head1x1/Layer1x1 no Path Dinâmico WaveNet A1

* **Componente:** `src/loader/nam_json/model.rs` (linhas 62-108), `src/loader/nam_json/topology/wavenet.rs` (linhas 360-380)
* **Descrição do Problema:** O parser `NamLayerConfig` e o resolvedor de topologia A1 dinâmico (`WaveNetModelDyn`) não lêem nem suportam os campos estruturais de gating/FiLM/head1x1/layer1x1 presentes no JSON. O campo booleano legado `gated` é usado apenas para desqualificar SKUs do catálogo, mas é silenciosamente ignorado na construção de modelos livres/dinâmicos.
* **Causa Raiz:** Falta de checagem defensiva (fail-closed) na separação de fluxos de arquitetura A1 e A2, fazendo com que modelos A1 não-catalogados que declarem esses recursos passem sem erro, porém processados incorretamente.
* **Impacto:** Saída matematicamente incorreta sem aviso ou erro em tempo de carregamento.
* **Solução Proposta:** Implementar guards fail-closed rigorosas em `get_wavenet_topology`. Se o modelo for direcionado ao path dinâmico A1 (`WavenetTopologyResult::Free`), rejeitar explicitamente o carregamento caso campos como `gating_mode`, `head1x1`, `layer1x1`, ou objetos de parâmetros `FiLM` estejam presentes/ativos nas camadas do JSON, garantindo que o comportamento falhe de forma segura.

### Finding 7.2.3: Dereferenciamento de Ponteiro Vazio no `LstmModelDyn`

* **Componente:** `src/models/lstm/model_dyn.rs` (métodos SIMD `process_avx2`, `process_avx512`, `process_avx512_vnni_bf16`)
* **Descrição do Problema:** Os loops internos de processamento SIMD extraem `self.layers.as_mut_ptr()` e realizam dereferenciamento incondicional (`*layers_ptr`) e aritmética antes de checar se a quantidade de camadas é maior do que zero (apenas um `debug_assert!` está presente, que é removido em compilações de release).
* **Causa Raiz:** Se o modelo `.nam` carregar com `num_layers: 0`, a chamada para `.as_mut_ptr()` em um `Vec` vazio retorna um ponteiro dangling/desalinhado. A primeira iteração acessará memória inválida.
* **Impacto:** Comportamento indefinido (UB) e possível segfault imediato em ambiente de execução de produção (release).
* **Solução Proposta:** Adicionar uma regra fail-closed restritiva na função `get_lstm_topology` de forma que `num_layers == 0` seja sumariamente rejeitado. Adicionalmente, proteger os métodos SIMD com um check explícito em tempo de execução `if self.layers.is_empty() { return; }` (ou passthrough direto).

### Finding 7.2.4: Catalog-SKU de WaveNet A1 Ignora Presença de `condition_dsp`

* **Componente:** `src/loader/nam_json/topology/wavenet.rs` (linhas 337-339)
* **Descrição do Problema:** A classificação booleana `catalog_compatible` para desviar modelos WaveNet para as especializações const-generic rápidas (Standard, Lite, Feather, Nano) analisa apenas contagem de canais, dilatações e se os blocos possuem bias na cabeça. Ela não verifica se o modelo possui o objeto `condition_dsp` declarado.
* **Causa Raiz:** Omitir a verificação de existência de sub-modelos de condicionamento na determinação do catálogo.
* **Impacto:** Um modelo com o shape exato de um SKU padrão, mas que adicione um sub-modelo de condicionamento, será erroneamente desviado para o fast-path que tem suporte zero a `condition_dsp`, descartando silenciosamente a modulação.
* **Solução Proposta:** Adicionar `&& data.config.condition_dsp.is_none()` na validação de `catalog_compatible`.

### Finding 7.2.5: Falta de Validação de Canais (`in_channels`/`out_channels`) no LSTM Loader

* **Componente:** `src/loader/nam_json/topology/lstm.rs`
* **Descrição do Problema:** O detector de topologia `get_lstm_topology` extrai apenas `num_layers` e `hidden_size` do JSON, omitindo validação de canais de áudio de entrada e saída.
* **Causa Raiz:** O C++ suporta LSTM multi-canal arbitrário, mas o `nam-rs` foi desenhado especificamente para áudio mono. A ausência de validação permite carregar modelos estéreo/multi-canal como se fossem mono.
* **Impacto:** Processamento corrompido de áudio estéreo em LSTMs sem acusar falhas de carregamento.
* **Solução Proposta:** Adicionar `out_channels` no parsing de `NamConfig` e validar na função `get_lstm_topology` que se `in_channels` ou `out_channels` forem declarados no JSON, eles devem obrigatoriamente ser iguais a `1`. Caso contrário, rejeitar o modelo.

---

## 2. Achados de Categoria 7.4 ⚪ Cosmetic Findings (Hygiene)

Achados que não alteram a saída audível hoje devido a particularidades do encadeamento atual, mas que representam inconformidades que podem gerar regressões no futuro.

### Finding 7.4.1: Retorno Sub-reportado de `prewarm_samples()` no WaveNet

* **Componente:** `src/models/wavenet/model.rs` e `model_dyn.rs`
* **Descrição do Problema:** `WaveNetModel::prewarm_samples()` retorna apenas o tamanho do campo receptivo do primeiro array (`array1`), ignorando o segundo. Em `model_dyn.rs`, o cálculo faz um `.max()` com o prewarm do `condition_dsp` em vez de efetuar a soma matemática conforme implementado no C++ (`sum` de todas as camadas + sub-modelo).
* **Solução Proposta:** Implementar a soma cumulativa correta de todos os receptores activos e incluir a demanda de prewarm do `condition_dsp` caso esteja ativo.

### Finding 7.4.2: Propagação Incorreta de Saída Acumulada no Cascade Multi-Array A2

* **Componente:** `src/models/a2/cascade.rs` (ou equivalente de processamento de cascade)
* **Descrição do Problema:** O motor dinâmico propaga o acumulador bruto (`head_accum`) entre os arrays acoplados, em vez de realizar a recanalização do sinal antes de repassar.
* **Solução Proposta:** Corrigir a lógica de propagação do sinal intermediário do cascade para refletir a saída pós-canalização da cabeça do C++.

### Finding 7.4.3: Projeção de Cabeça Simplificada para `head_size > 1` no A2

* **Componente:** `src/models/a2/model/static/process.rs`
* **Descrição do Problema:** Para modelos onde a cabeça do A2 possui tamanho superior a 1, a implementação do Rust realiza uma projeção densa simples. O C++ aplica uma Conv1D completa com bias e escala (`head_scale`).
* **Solução Proposta:** Reestruturar o braço condicional de finalização para aplicar a convolução de cabeça nos moldes canônicos do C++ quando o tamanho for maior que 1.

---

## 3. Achados de Categoria 7.5 📄 Stale Documentation

Inconsistências identificadas entre a documentação de testes/fixturas e o estado real atual do código-fonte.

### Finding 7.5.1: Documentação Obsoleta sobre WaveNet Lite em `docs/testing.md`

* **Local:** `docs/testing.md` §5
* **Correção:** Remover a afirmação de que o teste de WaveNet Lite está marcado com `#[ignore]` e possui baixa fidelidade (SNR=0.9 dB). O teste foi atualizado no branch principal para `EVH-5150-Lite.nam` e está rodando com SNR = 122.3 dB.

### Finding 7.5.2: Tabela de Modelos em `tests/fixtures/README.md`

* **Local:** `tests/fixtures/README.md` (tabela de fixturas)
* **Correção:** Atualizar o status do modelo `wavenet_a2_max.nam`. Ele é classificado atualmente como "Rejected — structure-incompatible" (o que era verdade antes de adicionarmos cascade), mas o comportamento real hoje é que ele carrega e roda, sendo explicitamente rejeitado apenas na camada de dispatch por guardrail de paridade (ver §7.1).

### Finding 7.5.3: Placeholder de Threshold para `wavenet_a2_max`

* **Local:** `tests/common/validation.rs` (linhas 600-612)
* **Correção:** Atualizar os comentários explicativos. O teste faz menção de que o modelo ainda não carrega e usa um ESR provisório de `5.0e-2`, sendo que a medição empírica real indica que o modelo carrega e gera um ESR discrepante de `3.61e1`.

### Finding 7.5.4: Comentários de Baseline SNR/ESR em `tests/golden_vectors.rs`

* **Local:** `tests/golden_vectors.rs` (linhas 1980-2005)
* **Correção:** O comentário documenta um SNR de `4.7 dB` e ESR de `3.4e-1` originados de baselines experimentais obsoletos. Corrigir com os números reais de baseline empiricamente medidos (SNR = -15.6 dB, ESR = 3.61e1).

---

## 4. Achados de Categoria 7.6 Coverage Gaps

Identificação de áreas do codebase que necessitam de enriquecimento de testes funcionais ou dados reais de validação.

### Finding 7.6.1: Ausência de Modelo Comunitário Real para A2 Fast-Path (Lite/Full)

* **Status:** O fast-path estático do A2 é exercitado apenas com fixturas sintéticas criadas em scripts. Não há nenhum captura oficial de amplificador real treinado sob o formato A2 Lite/Full validando a fidelidade final.
* **Mitigação:** Documentar ou adicionar um teste integrando uma captura real de comunidade (se disponível).

### Finding 7.6.2: Ausência de Validação de Regressão Cruzada (`live_cross_validation`) para A2 Gated/Blended/Max

* **Status:** Os modelos dinâmicos do A2 (`wavenet_a2_max.nam`, `a2_dynamic_gated_ch8.nam`, etc.) são testados apenas contra capturas estáticas de arquivos `.bin` (vetores dourados). Não há asserção direta em `tests/cpp_parity.rs` rodando o utilitário binário do C++ contra esses modelos dinâmicos em tempo de build.
* **Mitigação:** Estender as macros de teste cruzado para dar cobertura a esses modelos.

### Finding 7.6.3: Falta de Modelo Real para Cobertura de LstmModelDyn

* **Status:** O parser e execução dinâmica do LSTM são testados apenas através de cenários sintéticos; nenhum arquivo NAM de comunidade faz uso de dimensões que desviem do catálogo estático de LSTMs do nam-rs.

---

## 5. Epics Propostos para Sprints

Para organizar de maneira ágil, segura e lógica as correções destes achados, agrupamos as tarefas nos seguintes Épicos:

* **Épico E-1: Correções de Código e Validação Defensiva (Achados 7.2)**
  * *Objetivo:* Sanar os 5 bugs ativos mapeados na auditoria e estancar riscos latentes de UB e comportamentos silenciados.
  * *Prioridade:* Crítica / Alta.

* **Épico E-2: Refatoração Cosmética e Alinhamento de Latência (Achados 7.4)**
  * *Objetivo:* Ajustar o cálculo de prewarm e propagação de canais em cascata para evitar landmines futuras de host-compensation e modelos multi-array de tamanho grande.
  * *Prioridade:* Média.

* **Épico E-3: Correção de Documentação e Baselines de Testes Obsoletos (Achados 7.5)**
  * *Objetivo:* Atualizar os arquivos de texto e baselines para que reflitam a realidade funcional do motor de processamento.
  * *Prioridade:* Baixa.

* **Épico E-4: Expansão da Matriz de Cobertura de Testes (Achados 7.6)**
  * *Objetivo:* Integrar validação cruzada cruzando saídas em tempo real e cobrir branches dinâmicos de modelagem.
  * *Prioridade:* Média.
