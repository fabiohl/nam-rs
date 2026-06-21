<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-sprints.md — Planejamento de Sprints (NRF)

> **Skill:** `planejador-arquiteto`
> **Origem:** `TODO-audit.md` (§6.3 Novos achados residuais)
> **Data:** 2026-06-21

## Épico G — Resolução de Achados Residuais Menores (NRF)

**Objetivo:** Eliminar as dívidas técnicas residuais (NRF1 a NRF5) apontadas na última re-auditoria. Estes itens não são bloqueantes, mas envolvem clareza em benchmarks, limpeza de código morto, redundâncias em asserts e garantia de testabilidade em CI.

### Sprint 1: Correções de Benchmarks e Código Morto (NRF1, NRF2)

Foco na correção de inconsistências no código de benchmark e na remoção de atributos não utilizados no código de produção para garantir clareza e facilidade de manutenção.

* **Tarefa 1.1: Ativar path Gated no Benchmark do WaveNetA2Dyn (Referência: NRF1 / RF2)**
  * **Arquivo:** `benches/inference_bench.rs`
  * **Ação:** Na função que inicializa os dados, possivelmente `make_wavenet_a2_dyn_data`, alterar a configuração de `gated: Some(false)` para `gated: Some(true)`. Em contrapartida, se for mais condizente com a estrutura atual, renomeie o ID/doc de `A2Dyn_Gated_64samp_48kHz` para não indicar `gated` caso não deva de fato usar gating. Optaremos pela primeira solução (`gated: Some(true)`), conforme sugestão primária da auditoria.
  * **Justificativa:** O nome do bench (`A2Dyn_Gated_64samp_48kHz`) e a documentação indicam o uso do ramo de gating ("gating active on the first layer"). Fornecer `gated: false` perfila outro ramo da inferência, fugindo do perfil pretendido.
  * **Critérios de Aceite (DoD):** Compilar os benchmarks (`cargo check --benches`) com sucesso. Se executado (`cargo bench`), o teste não deve apresentar pânico.

* **Tarefa 1.2: Remoção do campo morto `rt_status` em `WaveNetA2Dyn` (Referência: NRF2 / RF8)**
  * **Arquivo(s):** Principalmente `src/models/a2/model/dynamic.rs`
  * **Ação:**
        1. Remover o campo `rt_status` da definição da `struct WaveNetA2Dyn` (aprox. linha 140).
        2. Remover a sua inicialização na instanciação da struct (aprox. linha 273).
        3. Remover eventuais utilizações que injetam esse valor (ex.: chamada `inject_rt_status` em `dynamic.rs:284` ou semelhante).
  * **Justificativa:** Ao contrário da versão de const-generics (`WaveNetA2<CH>`), a versão genérica dinâmica (`WaveNetA2Dyn`) não possui caminhos de fallback escalar para setar flags de telemetria RT, tornando o campo morto (write-only).
  * **Critérios de Aceite (DoD):** A compilação da library deve rodar limpa sem avisos de código não utilizado, `cargo check --lib` passando com sucesso e sem comprometer as rotas dinâmicas.

### Sprint 2: Melhorias em Testes e Validação (NRF3, NRF5)

Foco em simplificar lógicas de validação (assertivas idiomáticas) e garantir visibilidade sobre coberturas não compulsórias.

* **Tarefa 2.1: Simplificação do `assert_eq!` no `nondist_validation` (Referência: NRF3 / RF4)**
  * **Arquivo:** `tests/nondist_validation.rs`
  * **Ação:** Localizar o trecho (linhas 112-121) do condicional `if class_label == expected` que encerra com um `else` englobando o `assert_eq!`. Substituir o bloco if/else por um direto: `assert_eq!(class_label, expected, "Classification mismatch...");`.
  * **Justificativa:** O bloco if/else que condiciona a falha já faz parte da utilidade nativa do macro `assert_eq!`. O uso idiomático poupa linhas de código e fica visualmente mais limpo.
  * **Critérios de Aceite (DoD):** `cargo test --test nondist_validation` deve passar. O skip de "Unknown" não deve ser afetado (pois acontece antes nas linhas 104-109).

* **Tarefa 2.2: Documentar pré-requisito do WaveNet Lite non-dist (Referência: NRF5 / RF7)**
  * **Arquivo:** `tests/fixtures/README.md` (e/ou `docs/testing.md` conforme alinhamento do projeto)
  * **Ação:** Adicionar um parágrafo/item explícito mencionando que "O gate de paridade cruzada do modelo WaveNet Lite (`golden_vectors.rs`) é condicionado à presença do modelo local `EVH-5150-Lite.nam` dentro de `models-nondist`. Em caso de ausência num ambiente limpo (ex: CI de terceiros), o teste efetuará _skip_ em prol da não-falsidade do teste, mas perderá-se a cobertura."
  * **Justificativa:** Sendo agora condicional para não utilizar gateways placebos, se faz necessário instruir novos desenvolvedores sobre como reativar esse teste.
  * **Critérios de Aceite (DoD):** Documentação atualizada confirmando a condição explícita do ambiente de teste limpo.

---

**Nota Informativa de Fechamento:**

* **NRF4 🔵 (Descoberta recursiva agora em todo log de models-nondist):** Identificado como melhoria aceitável e de comportamento positivo consistente para unificação. Não requer _action item_ ou refatoração. Apenas registrado no histórico do projeto.
