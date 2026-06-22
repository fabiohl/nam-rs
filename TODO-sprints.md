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

### 📝 Tarefa 3.1: Oráculos Desacoplados e Mapeamento de Testes Críticos Co-localizados [DONE]

**Nota pós-implementação (2026-06-21):**

* `tanh_slice_fallback` agora usa `f32::tanh()` (oráculo independente), desacoplando do Padé SIMD. Epsilon de paridade mantido em 5e-3 — `f32::tanh()` tem erro ~2e-6, folga adequada.
* `validation_test.rs` (23 testes) cobre NaN/Inf no visitor, limites de profundidade/tamanho, submodels. Documenta que `WeightsVisitor` **ainda aceita NaN/Inf silenciosamente** (F1 — gap conhecido, requer validação futura no loader).
* `model_test.rs` +13 testes de finitude/bounds em `set_weights` (NaN, Inf, subnormals, zeros, extremos finitos, slice vazio).
* **Impacto em Tarefa 3.2:** Kernels de transposição (`transpose_conv1d_interleaved_4wide` etc.) em `set_weights.rs` não validam finitude das entradas — NaN/Inf propagam silenciosamente para os buffers do modelo. Testes `should_panic` de T3.2 devem considerar que pesos não-finitos chegam aos kernels.

**Responsável Sugerido:** `implementador` / `debugger`
**Contexto:** Componentes chaves de inferência matemática e loaders de fronteiras não contam com verificação cruzada totalmente independente ou não têm baterias de testes focados nativos que capturam comportamentos fora da curva. (Ref: F22, F32)
**Critérios de Aceitação:**

1. Na referência escalar pura `src/math/common/scalar_ref/utility.rs` (no `tanh_slice_fallback`), substituir o backend compartilhado de Padé tanh por chamadas a referências padrão Rust `f32::tanh()` em um slice iterativo, estabelecendo verificação independente cruzada pura. Checar no `test_long.sh` de paridade se necessita de algum epsilon float tweak documentado (devido à reordenação).
2. Adicionar testes ao `src/loader/nam_json/validation.rs` cobrindo o comportamento do validador. Caso o arquivo possua linhas >= 300 (seguir as regras da casa), ele deve ficar em `src/loader/nam_json/validation_test.rs`. Criar mocks injetando `NaN` e dimensões inválidas extremas para assegurar robustez.
3. Repetir o exercício criando baterias básicas de `set_weights.rs` para atestar bounds/finitude.

### 📝 Tarefa 3.2: Garantir os Contratos Críticos em Zonas SIMD e Memórias Diretas [DONE]

**Responsável Sugerido:** `implementador` / `pesquisador-inovador`
**Contexto:** Kernels transpositores de modelo não blindam ativamente (através de testes focados `should_panic`) operações em que callers pudessem burlar as restrições ou enviar matrizes desbalanceadas. (Ref: F33)
**Critérios de Aceitação:**

1. Selecionar o módulo de `grouped_conv1d.rs` e subjacentes de resampling e avaliar todos os callers `unsafe` das funções publicamente expostas ao kernel RT.
2. Escrever e adicionar cenários com anotações de atributo `#[should_panic]` que entregam shapes quebrados de dims/chans que seriam proibidos nas aproximações SIMD (passar um `vec` desalinhado e um layout faltando posições).
3. Assegurar que o programa abortará a execução por um `debug_assert!` ou bound safe panics internos antes de explodir por um `#GP` (General Protection Fault) / Memory Violation C-style segfault.

**✅ Tarefa 3.2 concluída (2026-06-21):**

* Adicionados `debug_assert!` nos 4 entry points SIMD (`process_single_frame`, `process_single_frame_avx2`, `grouped_conv1d_single_frame_simd`, `process_single_frame_depthwise_avx2`) + `load_mixin_4` + `process_block` verificando: `out_frame.len() >= out_ch`, `frame_idx >= lookback`, `layer_buffer` suficiente, mixin length.
* 15 testes `#[should_panic]` cobrindo: out_frame curto, frame_idx baixo, mixin/block desbalanceados, pesos truncados, groups=0, in_ch/out_ch não divisíveis por groups.
* Resampler (`src/dsp/resampler.rs`) já possui `debug_assert!` nos acessos `DelayLine::push`/`window_ptr` e bounds guards via `while` loops nos métodos `process_internal`. Layout de transposição (`src/loader/dispatcher/wavenet/layout.rs`) é código seguro com slice indexing. Nenhum gap remanescente de segurança SIMD nestes módulos.
* **Nota para tarefas futuras:** `debug_assert!` só atua em modo debug; em release as verificações são removidas. Kernels SIMD dependem da disciplina do caller — o contrato de segurança está documentado nas assinaturas `unsafe fn`.

---

## 🏃 Sprint 5: Blindagem de Segurança do Loader (Épico E2)

**Objetivo:** Tornar o parsing de modelos (`.nam` e `.namb`) resiliente a entradas hostis, corrompidas ou malformadas, prevenindo ataques de DoS por exaustão de memória, danos a equipamentos por ruídos (`NaN`/`Inf`) e garantindo a integridade dos dados carregados.
**Risco:** Baixo–Médio (Caminho frio de loading, atentar para manter a compatibilidade retroativa, especialmente na validação CRC de v1).
**Base:** [TODO-findings.md - F1, F2, F3, F4, F5, F6, F7](TODO-findings.md)

### 📝 Tarefa 5.1: Defesa contra Valores Numéricos Maliciosos (Finitude) [DONE]

**Nota pós-implementação (2026-06-21):**

* `WeightsVisitor::visit_seq` agora valida `is_finite()` em cada peso extraído, retornando `JsonError::WeightNotFinite { index, value }`.
* `NambError::NonFiniteWeight` e `NambError::InvalidHeaderField` adicionados. Parser `.namb` valida finitude dos pesos (byte a byte) e das variáveis de header (`sample_rate`, `input_level_dbu`, `output_level_dbu`), com rejeição adicional de `sample_rate <= 0.0`.
* `WeightCursor::read_f32_finite()` — choke point único usado por todos os construtores de modelo (WaveNet standard/dynamic, LSTM 1/2-layer/dynamic, ConvNet, Linear) para validar `head_scale`/`head_bias`.
* A2 (static e dynamic): validação de `is_finite()` em `head_b` e `head_scale` nos métodos `set_weights`.
* `NamErrorCode`: 3 novos códigos — `NamJsonWeightNotFinite` (E1211), `NambNonFiniteWeight` (E1212), `NambInvalidHeaderField` (E1213).
* 8 novos testes NAMB: `test_non_finite_weight_*_rejected` (NaN/+Inf/-Inf), `test_invalid_header_*` (sample_rate NaN/0/neg, input_level Inf, output_level -Inf).
* 3 testes de `validation_test.rs` atualizados para esperar rejeição (antes aceitavam NaN/Inf). 4 testes existentes de NAMB ajustados para inicializar `sample_rate`/`input_level_dbu`/`output_level_dbu`.

**Responsável Sugerido:** `implementador` / `revisor-auditor`
**Contexto:** O projeto aceita pesos, ganhos e configs de taxa de amostragem que podem ser `NaN` ou infinitos, o que em processamento DSP causaria ruídos extremos e danos a equipamentos. (Ref: F1, F4)
**Critérios de Aceitação:**

1. **JSON (`src/loader/nam_json/validation.rs`):** Modificar `WeightsVisitor::visit_seq` para validar `is_finite()` em cada valor extraído. Se não for, retornar um novo erro `JsonError::NonFiniteWeight`.
2. **Binário (`src/loader/namb/parse.rs`):** Validar `is_finite()` dos pesos importados do buffer (idealmente usando `iter().all()` antes de prosseguir) e retornar um erro `NambError::NonFiniteWeight` se corrompido.
3. **Escalas/Bias:** Inserir validação `.is_finite()` logo após a leitura de `head_scale` e `head_bias` nos construtores em `wavenet` (`standard.rs`, `dynamic.rs`), `lstm` (`static_builder.rs`, `dynamic_builder.rs`), `convnet` e `linear`.
4. **Header Namb (`parse.rs`):** Validar as variáveis de header lidas (`sample_rate`, `input_level_dbu`, `output_level_dbu`) com `is_finite()`. Além disso, rejeitar `sample_rate <= 0.0`.
5. Prover testes injetando floats maliciosos nestes campos para atestar o reject do modelo (fail-fast, sem panic).

### 📝 Tarefa 5.2: Proteção contra Exaustão de Memória (OOM/DoS) [DONE]

**Nota pós-implementação (2026-06-21):**

* Constantes de limite definidas em `src/loader/nam_json/validation.rs`:
  `MAX_LSTM_LAYERS=16`, `MAX_LSTM_HIDDEN_SIZE=1024`, `MAX_WAVENET_FREE_CHANNELS=512`,
  `MAX_A2_DYN_CHANNELS=256`, `MAX_A2_DYN_BOTTLENECK=256`.
* **LSTM:** `get_lstm_topology()` rejeita `num_layers > 16` ou `hidden_size > 1024`
  retornando `None`, impedindo alocação em `build_lstm_dynamic` (onde
  `Vec::with_capacity(num_layers)` e buffers quadráticos em `hidden_size` causariam OOM).
* **WaveNet Free-shape:** `get_wavenet_topology()` rejeita per-layer `channels > 512`
  com `WavenetTopologyResult::Rejected`, antes que `build_wavenet_array_dyn`
  aloque `MirroredBuffer` e `AlignedVec` multiplicativos.
* **A2-Dyn:** Dispatcher (`wavenet/mod.rs`) rejeita `channels > 256` ou
  `bottleneck > 256` via `bail!()` antes de `WaveNetA2Dyn::new()`, prevenindo
  alocações catastróficas em `head_ring` e `head1x1_w`.
* 8 testes negativos: `test_lstm_rejects_*` (2), `test_wavenet_free_rejects_channels_too_high`,
  `test_a2_dyn_rejects_*` (2), `test_lstm_accepts_max_bounds`,
  `test_wavenet_free_accepts_max_channels`, `test_a2_dyn_accepts_max_channels_and_bottleneck`.
* `validation` module promovido a `pub(crate)` para acesso cross-crate.

**Responsável Sugerido:** `implementador`
**Contexto:** Alguns tipos de modelo especificam sua topologia (canais, camadas ocultas) no arquivo e alocam `Vec` baseados nesses tamanhos antes de verificar a quantidade real de pesos do bloco, abrindo janela para alocações em casa de GBs caso os canais sejam gigantes (DoS). (Ref: F2)
**Critérios de Aceitação:**

1. Definir e documentar constantes de limite máximo baseadas nas arquiteturas (ex. `MAX_DYN_CHANNELS`, `MAX_LSTM_LAYERS`, `MAX_LSTM_HIDDEN`).
2. **LSTM (`nam_json/topology.rs` e dispatch):** Rejeitar tentativas com `num_layers > 16` ou `hidden_size > 1024` logo no parse de `get_lstm_topology`.
3. **WaveNet e A2-Dyn (`wavenet/dynamic.rs`, `wavenet/mod.rs`):** Rejeitar `channels > 512` em WaveNet Free-shape e validar `channels <= 256` e `bottleneck <= 256` para A2-Dyn.
4. Incluir proptest ou um `#[test]` negativo forçando a injeção de parâmetros de topology massivos e atestar que a API retorna um `Err` limpo invés de crash/OOM.

### 📝 Tarefa 5.3: Integridade de Formato e Resiliência Estrutural [TO-DO]

**Responsável Sugerido:** `implementador`
**Contexto:** Arquivos de modelos v1 evadem validação de CRC se o campo for `0`, existem indexações seguras mas assumidas implicitamente, e truncamentos de ponteiro inseguros entre 64/32-bit. (Ref: F3, F5, F6, F7)
**Critérios de Aceitação:**

1. **CRC V1 (`namb/parse.rs`):** Mudar o tratamento do `.namb` v1 para computar e verificar o CRC32 em todo caso, exceto estritamente quando a seção de pesos estiver vazia E o `crc32 == 0`. (Ajustar com compatibilidade para os modelos antigos).
2. **FastPath (`dispatcher/wavenet/mod.rs`):** Substituir a macro `unreachable!()` no pattern-matching catch-all `KnownFastPath(_)` por um fallback grace failure seguro (ex: `bail!("A2 channels inesperado")`).
3. **Limites de Transposição e Indexação (`layout.rs` e `weights.rs`):** Inserir `debug_assert!(raw.len() >= ...)` no início das funções de helpers de transposição (`transpose_conv1d_interleaved_4wide` e indexadores lógicos) garantindo a suficiência do slice providenciado.
4. **Portabilidade 32-bit (`model.rs`):** Refatorar `parse_head` para substituir `v as usize` por um pattern check usando `usize::try_from(v).ok()?` nas configurações do model json.
