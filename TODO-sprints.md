<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved. -->

# TODO-sprints.md — Planejamento de Sprints e Tarefas Técnicas

> **Origem:** Skill `planejador-arquiteto`
> **Escopo:** Planejamento ágil (Sprints e Tarefas Técnicas) para os Épicos **E6** e **E7** definidos no documento de achados (`TODO-findings.md`). As entregas foram particionadas em blocos atômicos direcionados à especialidade da equipe de implementação.

---

## 🟢 Épico E6 — Unificação e Deduplicação de Kernels (DRY SIMD)

**Objetivo Estratégico:** Reduzir aproximadamente 850 linhas de código duplicadas entre as implementações AVX2 e AVX-512, mitigando drasticamente o risco de *drift* comportamental.
**Achados Mapeados:** F23, F27.
**Grau de Risco:** Médio (Requer extrema cautela na preservação da paridade bit-a-bit dos cálculos matemáticos de inferência).

### 🏃 Sprint E6.1: Deduplicação Estrutural de Gain e Ativações (F23)

Foco: Extração de *boilerplates* repetitivos nas rotinas matemáticas de ganho e funções de ativação.

- [ ] **Tarefa T6.1.1: Parametrização via Macros em `Gain` (AVX2 / AVX-512)**
  - **Origem:** Achado F23 (`src/math/dsp/gain/avx2.rs` e `gain/avx512.rs`).
  - **Ação:** Inspirando-se no padrão existente `gemv_kernel!`, criar uma macro (ex: `gain_kernel!`) que parametrize as rotinas de ganho (largura, *step*, *tail*). Reimplementar os laços das fatias iterativas (*slice*) invocando a macro unificada.
  - **Riscos/Atenção:** Assegurar o correto tratamento dos elementos escalares (tail/remainder).

- [ ] **Tarefa T6.1.2: Unificação das Funções de Ativação**
  - **Origem:** Achado F23 (laços `slice` das ativações `*_slice_avx2`/`avx512`).
  - **Ação:** Projetar e aplicar macros para unificar a aplicação de funções de ativação sobre *slices* grandes de dados. Compartilhar a lógica de varredura SIMD entre as arquiteturas.
  - **Riscos/Atenção:** Validar se a paridade é mantida em fronteiras de alinhamento com `cargo bench` e `cargo test` dedicados em `tests-quick.sh`.

### 🏃 Sprint E6.2: Deduplicação de Convolução e Rotinas Comuns (F23, F27)

Foco: Consolidação das operações de *Stereo Convolution* e lógicas redundantes de arquitetura do App/Plugin.

- [ ] **Tarefa T6.2.1: Refatoração da Convolução Stereo (AVX2 / AVX-512)**
  - **Origem:** Achado F23 (`src/math/dsp/stereo/convolution_avx2.rs` e `convolution_avx512.rs`).
  - **Ação:** Substituir as 300+ SLOC estruturais idênticas por macros de kernel ou *generics* com especializações SIMD. A macro deverá englobar carregamento alinhado, processamento e *tail escalares*.
  - **Riscos/Atenção:** A convolução é um hot-path crítico para performance e *memory safety*. Validar paridade rigorosamente através dos testes C++.

- [ ] **Tarefa T6.2.2: Centralização de `try_slimmable_rebuild`**
  - **Origem:** Achado F27 (Standalone `commands.rs` x CLAP `events.rs`).
  - **Ação:** Extrair a lógica comum para um *helper* central genérico que receba `&mut Option<Box<StaticModel>>`, adicionando *hooks* de estado ou callbacks para a lógica específica do CLAP (ex: *flags RT*). Substituir os laços nos arquivos chamadores, excluindo a duplicação.

---

## 🟢 Épico E7 — Reorganização Estrutural e Higiene ("Embelezamento")

**Objetivo Estratégico:** Adequação à arquitetura oficial (`testing.md`), otimização da facilidade de manutenção por meio do desmembramento de monolitos e remoção de débitos técnicos silenciosos.
**Achados Mapeados:** F24, F25, F26, F28, F29.
**Grau de Risco:** Baixo (Esforço mecânico, altamente paralelizável; baixo impacto no DSP).

### 🏃 Sprint E7.1: Desmembramento e Normalização de Regras de Teste (F24, F25)

Foco: Divisão de grandes módulos em namespaces menores e re-assentamento de testes de acordo com a política (>300 linhas = arquivo separado, <300 linhas = inline).

- [ ] **Tarefa T7.1.1: Quebra de Monolitos de Domínio (F24)**
  - **Ação 1:** Extrair e reestruturar `src/loader/nam_json/topology.rs` (765 linhas) em submódulos ou diretório próprio abrangendo `{wavenet, lstm, convnet, a2, linear}.rs`.
  - **Ação 2:** Separar as responsabilidades mistas dos testes em `src/clap/processor_test.rs` (2048 linhas) dividindo-os tematicamente em múltiplos arquivos dentro do diretório de testes apropriado.

- [ ] **Tarefa T7.1.2: Realocação Restrita das Regras de Teste (F24, F25)**
  - **Ação 1:** Extrair testes contidos em blocos `inline` para arquivos separados com o sufixo `_test.rs` em:
    - `src/math/dsp/fft.rs` (1284 linhas).
    - `src/models/a2/grouped_conv1d.rs` (1205 linhas).
    - `src/models/a2/model/dynamic.rs` (1140 linhas).
    - `src/models/slimmable.rs` (725 linhas).
    - `src/models/a2/model/mod.rs` e `src/models/a2/film.rs` (arquivos >300 linhas).
  - **Ação 2:** Movimentar testes que estão declarados em um arquivo `_test.rs` separado mas que referenciam um módulo fonte menor que 300 linhas (ex: `a2/activations.rs` que tem 157 linhas) devolvendo-os ao modelo estrutural *inline* com `#[cfg(test)]`.

### 🏃 Sprint E7.2: Higiene, Layout e Código Morto (F26, F28, F29)

Foco: Ajustes visuais, normalização da árvore de arquivos e limpeza de *dead code* indesejado.

- [ ] **Tarefa T7.2.1: Adequação do Layout de Módulos (F29)**
  - **Contexto:** Módulos de arquiteturas similares estão armazenados com layouts de diretório inconsistentes.
  - **Ação:** Promover o arquivo *flat* `src/models/a2/conv1d_ch8.rs` para um diretório `conv1d_ch8/` com a estrutura padrão: `{mod.rs, simd.rs, scalar.rs}`. Atualizar as exportações (`pub use`).

- [ ] **Tarefa T7.2.2: Padronização da Nomenclatura dos Arquivos de Teste (F26)**
  - **Contexto:** Multiplicidade de convenções como `tests.rs`, `test_files/` no lugar da oficial.
  - **Ação:** Renomear sistematicamente as ocorrências de `tests.rs` (ex: em `math/activations/`, `math/common/`, `models/{a2,lstm,wavenet}/`, e `clap/gui/ui/`) para `_test.rs` de modo a honrar o padrão do projeto. Revisar os caminhos e importações (`#[path = ...]`).

- [ ] **Tarefa T7.2.3: Auditoria e Extinção de `#[allow(dead_code)]` (F28)**
  - **Contexto:** Várias rotinas ignoram verificação de código não utilizado, maquiando dívidas de implementação.
  - **Ação:** Localizar as 9 ocorrências (ex: `loader/dispatcher/lstm/weights.rs:97`, 3 ocorrências em `models/a2/conv1d_ch3/mod.rs`, `dsp/pipeline/output_pw.rs:18`, `models/wavenet/conv_input.rs:105`, `loader/dispatcher/wavenet/traits.rs:11`). Remover o código subjacente se inútil, ou remover o macro documentando justificativa plausível. Compilar garantindo ausência de *warnings*.
