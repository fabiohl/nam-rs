<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Refatoração Rust: Análise do Diretório `/tests` (Achados Confirmados)

Este documento centraliza os achados técnicos consolidados e confirmados sobre ineficiências na topologia de integração da suíte de testes.

## Achado 1: Anti-pattern de Compilação vs. Controle de CI

**Diagnóstico:** A pasta `tests/` possui **50 arquivos `.rs` soltos** na raiz. O comportamento padrão do Cargo trata cada um como um binário de teste de integração independente, forçando a compilação e a linkagem do crate 50 vezes consecutivas, o que compromete severamente a performance de CI/CD.

**O Desafio Estrutural (A Dependência dos Scripts):** A topologia plana não foi acidental. Os scripts automáticos `utils/tests-quick.sh` e `utils/tests-long.sh` dependem *fortemente* do fato de que cada teste é um binário independente. Eles invocam os testes de forma nominal e direta (ex: `cargo test --test a2_loader`), orquestrando e dividindo cargas de trabalho complexas (testes estruturais vs. oráculos de floats).
Se nós simplesmente agruparmos os testes em sub-módulos, quebraremos os scripts da CI imediatamente.

**Solução Exigida:**

1. Os 50 testes precisam ser consolidados em 4-5 entry-points centrais (ex: `tests/models.rs`, `tests/clap.rs`, `tests/rt_constraints.rs`, `tests/perf_soak.rs`, `tests/parity.rs`).
2. Os testes atuais devem virar submódulos (`mod a2_loader;`, etc) dentro desses entry-points.
3. Os scripts `tests-quick.sh` e `tests-long.sh` precisarão ter a sintaxe de invocação refatorada de `--test <nome_do_arquivo>` para filtros de pacote/módulo: `--test <entry_point> <entry_point>::<nome_do_modulo>`, permitindo manter a orquestração granular intacta.

## Achado 2: Lixo Oculto e Dead-files (Fixtures)

**Diagnóstico:** A pasta de dados estáticos `tests/fixtures/` está altamente poluída com dezenas de arquivos que jamais são referenciados no código Rust, acumulando *storage* sem proveito.

**Solução Exigida:**
Exclusão direta e implacável dos seguintes 51 arquivos órfãos (garantidamente não utilizados):

- Todos os `stress_signal_v2_*.wav` e o `stress_signal_v1.wav`.
- Todos os arquivos do resampler (`resampler_input_*.f32` e `resampler_ref_*.f32`).
- Dezenas de vetores *Golden* versão 2 não consumidos (ex: `golden_lstm_*_v2_*.bin`, `golden_wavenet_*_v2_*.bin`).
- O script `CMakeLists_render_ir.txt` residual.

---

## Estrutura de Épicos para Execução (Resumo)

Os achados acima derivam os seguintes épicos de solução (detalhados no `TODO-sprints.md`):

- **Épico 1: Faxina de Repositório** (Remoção imediata dos dead-files. Baixíssimo risco.)
- **Épico 2: Desacoplamento da CI** (Adaptação prévia dos scripts bash para suportar a nova topologia. Alto risco de quebra da automação.)
- **Épico 3: Refatoração Topológica** (Criação dos submódulos e realocação estrutural do código Rust, seguido de validação final única.)
