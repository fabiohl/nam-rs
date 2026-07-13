<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Sprint Backlog — EP1 — Quick suite enxuto e honesto

Este documento detalha o planejamento ágil para execução do épico **EP1**, focado em tornar a suíte de testes rápida (`quick`) mais enxuta, veloz e correta em termos de reporte de status.

---

## Sprint 1: Otimização da Suíte de Testes Rápida (Quick)

### Tarefas Técnicas

#### [x] Tarefa 1.1: Remover `rt_constraints` do Quick

* **Achado Associado:** [F-S1](file:///home/fabio/nam-rs/TODO-findings.md#L234) — `rt_constraints` compila e executa 0 testes no quick.
* **Complexidade/Risco:** Baixo.
* **Descrição:**
  O binário de teste `rt_constraints` contém testes que dependem de features como `heap-audit` ou que são skipados intencionalmente no quick. Compilar esse binário em debug a cada run consome de 10 a 15 segundos desnecessariamente. Devemos remover `rt_constraints` da variável `_struct_targets` na suíte rápida. A cobertura real desse binário pertence exclusivamente à suíte longa (`tests-long.sh`).
* **Arquivos Afetados:**
  * [utils/tests-quick.sh](file:///home/fabio/nam-rs/utils/tests-quick.sh#L326) — remover a adição de `rt_constraints` em `_struct_targets`.
* **Critério de Aceitação:**
  * Executar [utils/tests-quick.sh](file:///home/fabio/nam-rs/utils/tests-quick.sh) não deve mais compilar nem executar o alvo de teste `rt_constraints`.
  * Ganho imediato de tempo na execução da fase de testes estruturais em ambiente local.

---

#### [x] Tarefa 1.2: Deduplicar Parity no Quick e Limpar Caps Mortos

* **Achado Associado:** [F-T2](file:///home/fabio/nam-rs/TODO-findings.md#L160) — `run_v1` é alias 1:1 de `run_v1_hf` (testes duplicados).
* **Complexidade/Risco:** Baixo.
* **Descrição:**
  Desde que a ativação `Standard` (exact-grade) virou o padrão universal do projeto, a função de comparação `run_v1` delega diretamente para `run_v1_hf`, resultando em execuções redundantes que medem exatamente o mesmo comportamento.
  * Consolidar os helpers: remover o sufixo `_hf` unificando `run_v1_hf` e `run_v1`, e unificando `run_v2_multi_sr_hf` e `run_v2_multi_sr` para usarem ativação `Standard` por padrão.
  * Remover os testes redundantes `quick_parity_hf_lstm_1x16` e `quick_parity_hf_wavenet_ch16` da suíte de testes rápidos.
  * Limpar constantes de cap de Fast-mode antigos (`ABSOLUTE_ESR_CAP_WAVENET`, etc.) que viraram dead code (ou documentar caso ainda sejam úteis na suíte de testes longos com parametrização explícita).
  * Manter a assinatura parametrizável (`use_hf`) em `run_render_comparison` para caso seja necessário reintroduzir testes específicos de Fast-mode no futuro.
* **Arquivos Afetados:**
  * [tests/parity/cpp_parity.rs](file:///home/fabio/nam-rs/tests/parity/cpp_parity.rs#L579) — unificação dos helpers e remoção de testes duplicados (`quick_parity_hf_*` e `live_cross_validation_hf_*`).
* **Critério de Aceitação:**
  * Executar a suíte de paridade rápida (ex.: `cargo test --test parity quick_parity`) deve passar com sucesso executando apenas uma version de teste por modelo.
  * Apenas os testes `quick_parity_lstm_1x16`, `quick_parity_wavenet_ch16`, `quick_parity_a2_full` e `quick_parity_convnet` devem rodar como gates rápidos de paridade com C++.
  * Nenhuma regressão de cobertura efetiva.

---

#### [x] Tarefa 1.3: Corrigir Status SKIPPED e Avisos Falso-Verdes no Tests-Long

* **Achado Associado:** [F-S5](file:///home/fabio/nam-rs/TODO-findings.md#L279) — tests-long: aviso "falso-verde" disparado para fase legitimamente SKIPPED.
* **Complexidade/Risco:** Baixo.
* **Descrição:**
  Quando o daemon do PipeWire não está em execução, a fase correspondente do `tests-long.sh` retorna 77 (`SKIPPED`) em menos de 1 segundo. No entanto, o script emite um alerta de "falso-verde" (completado em < 1s) e reporta a fase como "PASSED" no resumo final.
  * Garantir a propagação correta do status 77 da função de execução até o relatório de auditoria final (`AUDIT SUMMARY`).
  * Suprimir o aviso de execução rápida (< 1s) quando a fase retornar status 77 (SKIPPED).
* **Arquivos Afetados:**
  * [utils/tests-long.sh](file:///home/fabio/nam-rs/utils/tests-long.sh#L428) — ajustar tratamento de status e avisos de duração na função `run_phase`.
* **Critério de Aceitação:**
  * Simular a ausência de PipeWire (ex.: rodando com `PIPEWIRE_REMOTE=does_not_exist`) e verificar se a fase é exibida como `SKIPPED` em amarelo, sem emitir o aviso de falso-verde de < 1s.
  * O relatório de resumo final (`AUDIT SUMMARY`) deve listar "SKIPPED" para a fase correspondente.

---

## Sprint 2: Dashboard Correto e Determinístico (EP2)

### [x] Tarefa 2.1: Hotfix de Locale (LC_ALL=C) Global no Dashboard

* **Achado Associado:** [F-S2](file:///home/fabio/nam-rs/TODO-findings.md#L245) — Dashboard: aritmética awk sem LC_ALL=C.
* **Complexidade/Risco:** Baixo.
* **Descrição:**
  Garantir que todas as operações aritméticas e de formatação numérica no awk dentro do dashboard sejam imunes a locales regionais (como pt_BR, que usa vírgula decimal). Faremos isso definindo `export LC_ALL=C` no início do script de forma global, garantindo homogeneidade.
* **Arquivos Afetados:**
  * [utils/quality-dashboard.sh](file:///home/fabio/nam-rs/utils/quality-dashboard.sh)
* **Critério de Aceitação:**
  * Executar o dashboard com locales que usam vírgula (ex: `LANG=pt_BR.UTF-8 LC_NUMERIC=pt_BR.UTF-8`) deve computar e exibir as durações totais e parciais perfeitamente (com decimais), sem zerar os contadores.

---

### [x] Tarefa 2.2: Adicionar A2-FiLM-InputMixinPre na Tabela de Sumário do Oráculo f64

* **Achado Associado:** [F-S4](file:///home/fabio/nam-rs/TODO-findings.md#L270) — Dashboard: "vs f64: N/A" para A2-FiLM-InputMixinPre.
* **Complexidade/Risco:** Baixo.
* **Descrição:**
  O modelo `wavenet_a2_film_input_mixin_pre.nam` possui cobertura de teste individual no oráculo, mas não foi incluído no vetor de modelos que compõe o sumário. Isso faz com que a busca do dashboard resulte em `N/A`. Devemos incluí-lo explicitamente no vetor `models` dentro de `test_summary_table` em `reference_oracle_f64.rs`.
* **Arquivos Afetados:**
  * [tests/parity/reference_oracle_f64.rs](file:///home/fabio/nam-rs/tests/parity/reference_oracle_f64.rs#L1187)
* **Critério de Aceitação:**
  * O teste `test_summary_table` deve imprimir a linha de sumário para `wavenet_a2_film_input_mixin_pre.nam` com família `A2-FiLM-InputMixinPre`.

---

### [x] Tarefa 2.3: Emissão de Métricas em JSON Lines (JSONL)

* **Achado Associado:** [F-I1](file:///home/fabio/nam-rs/TODO-findings.md#L311) — Métricas machine-readable (JSON Lines) — eliminar scraping awk.
* **Complexidade/Risco:** Médio.
* **Descrição:**
  Substituir o frágil scraping de texto humano por uma exportação determinística estruturada.
  * Criar thread-locals em `tests/common/validation.rs` (`METRIC_MODEL` e `METRIC_MODE`) para armazenar o nome do modelo e o modo de teste ativo.
  * Atualizar o helper `report_dsp_fidelity_impl` para capturar esses metadados e escrever uma linha JSON no arquivo especificado pela variável de ambiente `NAM_METRICS_JSONL` (se definida).
  * Atualizar `golden_vectors.rs` e `cpp_parity.rs` para definir essas variáveis thread-local antes de chamar os reportadores de fidelidade.
* **Arquivos Afetados:**
  * [tests/common/validation.rs](file:///home/fabio/nam-rs/tests/common/validation.rs)
  * [tests/models/golden_vectors.rs](file:///home/fabio/nam-rs/tests/models/golden_vectors.rs)
  * [tests/parity/cpp_parity.rs](file:///home/fabio/nam-rs/tests/parity/cpp_parity.rs)
* **Critério de Aceitação:**
  * A execução dos testes de fidelidade (`golden_vectors` e `cpp_parity`) sob a variável `NAM_METRICS_JSONL=metrics.jsonl` deve gravar uma linha JSON válida por modelo contendo as chaves `label`, `esr`, `esr_db`, `snr_db`, `mrstft` e `mse`.

---

### [x] Tarefa 2.4: Migração do Parser do Dashboard para JSON Lines com Fallback

* **Achado Associado:** [F-I1](file:///home/fabio/nam-rs/TODO-findings.md#L311) — Ingestão estruturada de métricas.
* **Complexidade/Risco:** Médio.
* **Descrição:**
  Adaptar o dashboard para ler o arquivo JSON Lines gerado.
  * Implementar uma função `parse_jsonl_fidelity` no dashboard que lê o JSONL (usando `jq` se disponível ou `awk` regex simplificado) e popula as tabelas internas.
  * Integrar no parser `parse_golden_vectors`, mantendo o parser antigo de texto humano como um fallback resiliente caso o arquivo JSONL não esteja presente.
* **Arquivos Afetados:**
  * [utils/quality-dashboard.sh](file:///home/fabio/nam-rs/utils/quality-dashboard.sh#L347)
* **Critério de Aceitação:**
  * O dashboard deve renderizar todas as informações de fidelidade a partir do arquivo JSONL sem erros de parsing ou drifts.

---

### [x] Tarefa 2.5: Ingestão de Paridade com C++ (Fim do N/A no ConvNet)

* **Achado Associado:** [F-S3](file:///home/fabio/nam-rs/TODO-findings.md#L255) — "ConvNet vs NAMcore: N/A" no dashboard.
* **Complexidade/Risco:** Baixo.
* **Descrição:**
  O dashboard mostra `N/A` para ConvNet porque o teste em `golden_vectors.rs` é um teste de consistência interna (self-golden), e não de paridade C++. A verdadeira paridade C++ é medida em `quick_parity_convnet` (em `cpp_parity.rs`).
  * Adicionar a fase de execução de `quick_parity` (utilizando `cargo test --test parity quick_parity`) no dashboard sob o escopo de fidelidade, escrevendo no JSONL de métricas.
  * Mapear o label `"Quick ConvNet"` para `"ConvNet Test"` no parser para preencher a linha do ConvNet no dashboard com o valor real medido (ex: `2.54e-5`, ~45.9 dB).
* **Arquivos Afetados:**
  * [utils/quality-dashboard.sh](file:///home/fabio/nam-rs/utils/quality-dashboard.sh)
* **Critério de Aceitação:**
  * O painel exibe a paridade real do ConvNet vs NAMcore (não mais `N/A`), com o valor exato extraído do teste `quick_parity_convnet`.

---

### [x] Tarefa 2.6: Coluna f64 por Modelo (Aposentar `~fam.`)

* **Achado Associado:** [F-I2](file:///home/fabio/nam-rs/TODO-findings.md#L323) — ESR vs f64 por modelo — aposentar a aproximação por família.
* **Complexidade/Risco:** Médio.
* **Descrição:**
  Atualmente, a coluna "vs f64" exibe o oráculo de um único modelo representativo para toda a família (marcado com `~fam.`). Vamos substituir isso pela medição real individual de cada modelo.
  * Estender `test_summary_table` in `reference_oracle_f64.rs` para abranger todos os modelos golden ativos na suíte.
  * No dashboard, substituir `ESR_F64_FAMILY_MAP` por um mapa exato direto `ESR_F64_MODEL_MAP` que aponta cada label para o seu respectivo arquivo `.nam`.
  * Remover os caracteres de aproximação `~`, a legenda do rodapé e a respectiva variável `ESR_F64_EXACT_MATCH`. Modelos que não possuem teste de oráculo exibirão `N/A` de forma transparente.
* **Arquivos Afetados:**
  * [tests/parity/reference_oracle_f64.rs](file:///home/fabio/nam-rs/tests/parity/reference_oracle_f64.rs)
  * [utils/quality-dashboard.sh](file:///home/fabio/nam-rs/utils/quality-dashboard.sh)
* **Critério de Aceitação:**
  * A tabela exibe valores de oráculo f64 exatos e individuais por modelo, sem marcas de aproximação `~` ou legendas no rodapé sobre aproximação familiar.

---

### [x] Tarefa 2.7: Documentar Tolerâncias no Contrato de Qualidade

* **Achado Associado:** [F-S6](file:///home/fabio/nam-rs/TODO-findings.md#L289) — Tolerâncias do contrato não documentadas.
* **Complexidade/Risco:** Baixo.
* **Descrição:**
  O arquivo `quality-contract.txt` serve como linha de base absoluta, mas não explicita as tolerâncias que o script de dashboard aplica ao validar (ex: +10% latência, 10x margem de ESR/MR-STFT). Vamos adicionar uma seção explicativa no topo do contrato documentando essas margens e as políticas de verificação/recalibração.
* **Arquivos Afetados:**
  * [docs/quality-contract.txt](file:///home/fabio/nam-rs/docs/quality-contract.txt)
* **Critério de Aceitação:**
  * Inserção da documentação no cabeçalho do contrato de modo que seja visível para humanos e ignorada pelo parser de validação do script.

---

## Sprint 3: Cadeia de Suprimentos de Fixtures Determinística (EP4)

Este sprint aborda a governança, reprodutibilidade e integridade do pipeline de fixtures e goldens (tanto C++ quanto Rust). O objetivo é garantir que todas as fixtures de áudio e manifestos de validação sejam determinísticos, imunes a otimizações agressivas de compiladores, e completamente auditáveis.

### [x] Tarefa 3.1: Imposição de Flags IEEE-Strict na Compilação do Renderizador C++ (F-X1)

* **Achado Associado:** [F-X1](file:///home/fabio/nam-rs/TODO-findings.md#L81) — Goldens C++ gerados com `-Ofast` — determinismo cross-compilador comprometido.
* **Complexidade/Risco:** Alto (risco de alteração generalizada de thresholds/hashes de goldens).
* **Descrição:**
  Atualmente, as ferramentas de renderização do `NeuralAmpModelerCore` são compiladas sob o perfil Release com a flag `-Ofast` (que inclui `-ffast-math` e relaxa a estrita conformidade com a norma IEEE 754) e otimização `LTO`. Isso inviabiliza o determinismo rigoroso cross-compiler/cross-OS, gerando drifts de ponto flutuante de precisão fina (SNR > 120 dB) entre máquinas.
  * Modificar o script [golden_gen_build.sh](file:///home/fabio/nam-rs/tests/fixtures/golden_gen_build.sh#L196) para desativar ou neutralizar `-Ofast` no build do renderizador. Devemos fazer isso aplicando uma substituição pontual via `sed` (ou similar) no `CMakeLists.txt` do `NeuralAmpModelerCore` clonado localmente, ou sobrescrevendo as flags de CXX no `cmake` (ex.: `-DCMAKE_CXX_FLAGS_RELEASE="-O3 -fno-fast-math -ffp-contract=off"` ou injetando `-ffp-contract=off -fno-fast-math`).
  * Regenerar todas as fixtures `.bin` usando o novo renderizador IEEE-strict.
  * Medir os desvios de ESR/SNR em relação às fixtures antigas e atualizar as tolerâncias e thresholds dos testes de fidelidade e paridade que falharem após essa mudança.
* **Arquivos Afetados:**
  * [tests/fixtures/golden_gen_build.sh](file:///home/fabio/nam-rs/tests/fixtures/golden_gen_build.sh) — Seção de compilação do renderizador.
* **Critério de Aceitação:**
  * Execução do build do renderizador sem erros.
  * O compilador não deve utilizar `-Ofast` (confirmado inspecionando logs ou gerando os binários).
  * Todos os testes de fidelidade ajustados e passando com sucesso no `cargo test`.
  * **Concluído (2026-07-13):** `golden_gen_build.sh` agora aplica `sed` nos `CMakeLists.txt` vendorizados (substituindo `-Ofast` por `-O3`) e injeta `-fno-fast-math -ffp-contract=off` via `CMAKE_CXX_FLAGS`. Todos os 62 `.bin` goldens (v1 + v2) foram regenerados. Nenhum threshold precisou ser recalibrado — os desvios de ESR/SNR ficaram abaixo do piso de ruído das tolerâncias existentes. 228/228 testes passam (`--test-threads=1`). O build config foi estendido com sufixo `:ieee-strict` para invalidação automática do cache de binários pré-existentes.

---

### [x] Tarefa 3.2: Gravação do Fingerprint do Toolchain no Manifesto (F-I4)

* **Achado Associado:** [F-I4](file:///home/fabio/nam-rs/TODO-findings.md#L344) — Fingerprint de toolchain no manifest de goldens.
* **Complexidade/Risco:** Baixo.
* **Descrição:**
  Para facilitar a depuração de drifts futuros nas funções matemáticas da stdlib (como `std::tanh` da glibc) entre diferentes máquinas de desenvolvedores e CI, o manifesto de freshness deve conter o fingerprint do compilador.
  * Modificar o script [golden_gen_build.sh](file:///home/fabio/nam-rs/tests/fixtures/golden_gen_build.sh#L550) para coletar a versão ativa do compilador (`$CXX --version`), a versão do glibc/sistema operacional e as flags utilizadas.
  * Gravar esses dados como um comentário estruturado no topo do arquivo [.golden_manifest.sha256](file:///home/fabio/nam-rs/tests/fixtures/.golden_manifest.sha256). Exemplo: `# TOOLCHAIN: g++ 14.2.0 | glibc 2.40 | flags: -O3 -fno-fast-math -ffp-contract=off | cmake 3.30`.
  * Atualizar o validador do freshness gate para exibir um alerta (sem travar a suíte) caso o toolchain local do desenvolvedor/CI divirja do fingerprint registrado no manifesto.
* **Arquivos Afetados:**
  * [tests/fixtures/golden_gen_build.sh](file:///home/fabio/nam-rs/tests/fixtures/golden_gen_build.sh) — Seção de geração do manifesto.
  * [utils/_lib.sh](file:///home/fabio/nam-rs/utils/_lib.sh) — Função unificada de validação de freshness.
* **Critério de Aceitação:**
  * O cabeçalho de `.golden_manifest.sha256` contém a linha contendo `# TOOLCHAIN: ...`.
  * Avisos informativos são impressos nos testes locais se houver mudança de toolchain, mas o gate de testes continua verde.
  * **Concluído (2026-07-13):** `golden_gen_build.sh` coleta `$CXX --version`, `cmake --version`, `ldd --version` (glibc), `uname -r` (kernel) e as flags de compilação IEEE-strict usadas, gravando-as como linhas `# TOOLCHAIN:` no cabeçalho do manifesto. `utils/_lib.sh` recebeu a função `check_toolchain_fingerprint()` que compara o toolchain atual contra o manifesto e emite aviso amarelo (não bloqueante) em caso de divergência. A função é invocada por `tests-quick.sh` (freshness gate) e `tests-long.sh` (warn-only).

---

### [x] Tarefa 3.3: Expansão da Cobertura do Manifesto de Freshness (F-X3)

* **Achado Associado:** [F-X3](file:///home/fabio/nam-rs/TODO-findings.md#L113) — Fixtures sem cobertura do freshness manifest.
* **Complexidade/Risco:** Médio.
* **Descrição:**
  Atualmente, o manifesto de frescor apenas hasheia os modelos do catálogo. Fixtures cruciais como `golden_cabsim_cpp_*.bin`, `mrstft_golden.bin`, `resampler_*.f32`, âncoras f64 (`f64_anchors/`) e arquivos WAV sintéticos não são cobertos, correndo o risco de ficarem stale após alterações nos geradores.
  * Adicionar hashes de verificação de integridade e frescor para: todos os binários de cabsim, mrstft, resampler, wave e a árvore de diretórios `f64_anchors/`.
  * Incluir no manifesto o hash dos próprios scripts geradores (como `render_ir.cpp`, `gen_mrstft_golden.py`, `generate_ebu_sequences.py`, `generate_resampler_reference.py`, etc.).
  * Qualquer alteração nesses arquivos de código de suporte geradores invalidará as fixtures, forçando sua regeneração imediata.
* **Arquivos Afetados:**
  * [tests/fixtures/golden_gen_build.sh](file:///home/fabio/nam-rs/tests/fixtures/golden_gen_build.sh) — Lógica de escrita e cálculo do manifesto.
* **Critério de Aceitação:**
  * Geração do manifesto inclui todos os arquivos de dados sintéticos e de referência, além do hash SHA256 de seus respectivos scripts geradores.
  * O gate de freshness valida a árvore completa e detecta desatualização nos geradores ou fixtures de suporte.
  * **Concluído (2026-07-13):** `golden_gen_build.sh` agora adiciona duas novas seções ao manifesto: `# FIXTURES` (31 arquivos: golden_cabsim_cpp, mrstft, resampler, ebu_3341, stress_signal, spectral_fidelity, f64_anchors) e `# GENERATORS` (11 scripts: render_ir.cpp, golden_gen_build.sh, generate_*.py, gen_mrstft_golden.py, validate_oracle_f64.py, gen_stress.rs, wav_to_golden.rs, check-model.py). `check_freshness` em `tests-quick.sh` e `tests-long.sh` foi estendido com parser stateful (`SECTION=catalog|fixtures|generators`) — fixtures com hash divergente causam hard-fail, generators com hash divergente emitem warning não bloqueante.

---

### [x] Tarefa 3.4: Unificação e Fortalecimento do Freshness Gate (F-X4)

* **Achado Associado:** [F-X4](file:///home/fabio/nam-rs/TODO-findings.md#L124) — Freshness gate: warn-only no long, sem reverse-check, implementação duplicada.
* **Complexidade/Risco:** Médio.
* **Descrição:**
  A lógica de verificação de frescor está duplicada e divergente entre os scripts `tests-quick.sh` (que faz hard-fail) e `tests-long.sh` (que avisa mas não falha). Além disso, o gate é unidirecional (não pega arquivos `.nam` órfãos em disco).
  * Extrair e consolidar a função `check_freshness` para o arquivo [utils/_lib.sh](file:///home/fabio/nam-rs/utils/_lib.sh).
  * Parametrizar a função para operar em modo `hard-fail` (parar execução com erro) ou `warn-only` (apenas avisar).
  * Configurar tanto o `tests-quick.sh` quanto o `tests-long.sh` para usar o gate em modo `hard-fail` (garantindo que builds na CI não passem com goldens inconsistentes). Oferecer bypass opcional sob a variável `NAM_BYPASS_FRESHNESS=1` (para o `tests-long.sh` em rotinas locais do desenvolvedor).
  * Implementar o "reverse-check": varrer o diretório `tests/fixtures/models/` por arquivos `.nam` e verificar se constam no manifesto. Se houver qualquer modelo não documentado, falhar no gate.
* **Arquivos Afetados:**
  * [utils/_lib.sh](file:///home/fabio/nam-rs/utils/_lib.sh) — Inclusão do helper centralizado.
  * [utils/tests-quick.sh](file:///home/fabio/nam-rs/utils/tests-quick.sh) — Remoção da lógica local e chamada do helper centralizado.
  * [utils/tests-long.sh](file:///home/fabio/nam-rs/utils/tests-long.sh) — Remoção da lógica local e chamada do helper centralizado.
* **Critério de Aceitação:**
  * A suíte de testes de desenvolvimento local e CI valida o frescor em ambas as frentes.
  * A introdução de um modelo novo sem registro no manifesto causa falha impeditiva com instruções claras.
  * **Concluído (2026-07-13):** `check_freshness()` extraída para `utils/_lib.sh` com parâmetro de modo (`hard-fail` | `warn-only`). A função cobre: manifest existente, EXPECTED goldens, catalog model↔golden, fixtures standalone, generators (warn-only), reverse-check de órfãos via `# MODEL-REGISTRY:`, e toolchain fingerprint. `NAM_BYPASS_FRESHNESS=1` desvia o gate inteiramente. `tests-quick.sh` e `tests-long.sh` agora delegam à função centralizada em modo `hard-fail`. `golden_gen_build.sh` emite seção `# MODEL-REGISTRY:` com 34 modelos (28 do CATALOG + 6 test-only explícitos) para o reverse-check.

---

### [x] Tarefa 3.5: Registro de Proveniência e Sincronização do README (F-X2)

* **Achado Associado:** [F-X2](file:///home/fabio/nam-rs/TODO-findings.md#L101) — Proveniência ausente para 3 modelos + 11 goldens fora da tabela.
* **Complexidade/Risco:** Baixo.
* **Descrição:**
  Sincronizar a documentação de referência das fixtures com a realidade dos arquivos contidos no catálogo do manifesto.
  * Documentar no arquivo [tests/fixtures/README.md](file:///home/fabio/nam-rs/tests/fixtures/README.md) a proveniência dos modelos `wavenet_a2_film_chaos_stress.nam`, `wavenet_a2_film_input_mixin_pre.nam` e `wavenet_condition_lstm.nam`.
  * Atualizar a tabela de arquivos de fixtures do README para incorporar todos os 11 goldens que constam no manifesto e estão em disco, assegurando que o README não tenha drifts estruturais.
* **Arquivos Afetados:**
  * [tests/fixtures/README.md](file:///home/fabio/nam-rs/tests/fixtures/README.md) — Tabelas e registros de proveniência de fixtures e modelos.
* **Critério de Aceitação:**
  * Documentação completa e sem gaps de proveniência para todos os modelos do catálogo oficial.
  * Tabela de arquivos no README sincronizada 100% com o manifesto de frescor.
  * **Concluído (2026-07-13):** Tabela de goldens expandida de 21 para 33 entradas (73 goldens no manifesto de frescor, todos cobertos). 7 modelos sintéticos adicionados à Tabela 2 com documentação completa de proveniência (a2_example, convnet_test, lstm_dyn_test, wavenet_a2_film_chaos_stress, wavenet_a2_film_input_mixin_pre, wavenet_condition_lstm, wavenet_dyn_free) — cada um com natureza, arquitetura, gerador, testes e thresholds. Resumo rápido de contagens atualizado. Aviso de gap substituído por nota de completude.

---

### [x] Tarefa 3.6: Higiene do Catálogo de Modelos e Fixtures (F-X5)

* **Achado Associado:** [F-X5](file:///home/fabio/nam-rs/TODO-findings.md#L134) — Higiene menor de catálogo.
* **Complexidade/Risco:** Baixo.
* **Descrição:**
  Remover resquícios e referências obsoletas do repositório.
  * Remover as entradas obsoletas dos modelos `linear_fft_rf2048.nam`, `linear_fft_rf4096.nam` e `linear_fft_rf8192.nam` de `CATALOG_EXCEPTIONS` em [tests/models/meta_coherence.rs](file:///home/fabio/nam-rs/tests/models/meta_coherence.rs#L29) (já que agora possuem pipeline golden implementado e ativo).
  * Deletar (ou mover para `tests/fixtures/models/obsolete/`) o arquivo obsoleto `BossWN-lite.nam` de `tests/fixtures/models/` (pois foi superado por `EVH-5150-Lite.nam`).
  * Incluir uma seção curta no README de fixtures documentando o propósito da pasta de âncoras `tests/fixtures/f64_anchors/`.
* **Arquivos Afetados:**
  * [tests/models/meta_coherence.rs](file:///home/fabio/nam-rs/tests/models/meta_coherence.rs) — Limpeza de catálogo de exceções.
  * [tests/fixtures/README.md](file:///home/fabio/nam-rs/tests/fixtures/README.md) — Inclusão de documentação de anchors f64.
  * [tests/fixtures/models/](file:///home/fabio/nam-rs/tests/fixtures/models/) — Higienização de modelos stale.
* **Critério de Aceitação:**
  * Execução sem falhas do teste de governança `meta_coherence`.
  * Arquivos desnecessários limpos e novas seções documentadas no README.
  * **Concluído (2026-07-13):** 3 entradas stale de `linear_fft_rf{2048,4096,8192}.nam` removidas de `CATALOG_EXCEPTIONS` em `meta_coherence.rs` (golden pipeline já implementado e ativo desde Sprint 2). `BossWN-lite.nam` mantido em `models/` — é obsoleto como golden mas ainda é o único fixture CH=12 WaveNet git-versionado usado em `rt_deadline.rs` e `regression_gate.rs`; remover quebraria cobertura de RT deadline e benchmark. Seção `## f64 Reference Anchors` adicionada ao README documentando os 10 âncoras f64, formato binário, gate ESR < 1e-12 e regeneração via `validate_oracle_f64.py`.

---

## Sprint 4: Paridade Estrita com NAMcore (EP5)

Este sprint foca em robustecer a validação de arquivos de modelos `.nam` de acordo com os limites e semânticas do NAMcore C++, cobrindo conformidade de canais, faixa de versões de SemVer, aliases do Linear e tratamentos de escalas de cabeças ConvNet.

### [x] Tarefa 4.1: Alinhamento de head_scale em ConvNet (F-P1)

* **Achado Associado:** [F-P1](file:///home/fabio/nam-rs/TODO-findings.md#L29) — ConvNet aplica `head_scale` — divergência semântica com NAMcore.
* **Complexidade/Risco:** Baixo.
* **Descrição:**
  O renderizador C++ oficial da NAMcore não suporta ou aplica `head_scale` no ConvNet (a saída é crua/escala 1.0). Se um arquivo `.nam` em formato plano declarar `head_scale`, devemos garantir que a escala a nível de execução seja fixada em `1.0` (já é feito pelo dispatcher `FlatCpp`) e que registremos um aviso diagnóstico caso o arquivo de modelo declare um valor diferente de `1.0`.
* **Arquivos Afetados:**
  * [src/loader/dispatcher/convnet/mod.rs](file:///home/fabio/nam-rs/src/loader/dispatcher/convnet/mod.rs) — Alertar via log caso `head_scale` esteja presente e diferente de `1.0` em FlatCpp.
* **Critério de Aceitação:**
  * A compilação e carregamento de um modelo ConvNet com `head_scale != 1.0` em formato FlatCpp deve logar um aviso explicativo e processar o áudio usando `head_scale = 1.0`.
  * Adicionar testes unitários correspondentes em `convnet_model_test.rs`.
* **Concluído (2026-07-13):** Adicionada verificação de `data.config.head_scale` no `build_convnet_flat_cpp` com `log::warn!` quando valor declarado != 1.0. Teste `test_linear_head_flat_cpp_parity` adicionado em `convnet_model_test.rs` validando que `linear_head` + `head_scale = 1.0` produz saída com ganho identidade (paridade estrita com NAMcore).

---

### [ ] Tarefa 4.2: Validação e Enforcement de Versão SemVer (F-P2)

* **Achado Associado:** [F-P2](file:///home/fabio/nam-rs/TODO-findings.md#L42) — Ausência de enforcement de versão do arquivo `.nam`.
* **Complexidade/Risco:** Médio.
* **Descrição:**
  O parser de JSON do C++ rejeita arquivos com versão `< 0.5.0` ou com major/minor que exceda `0.7.0` (ex: `0.8.0`), sinalizando compatibilidade `PARTIAL` para versões acima de `0.7.0` que compartilhem a mesma versão minor/major (ex: `0.7.1`).
  * Implementar no parser JSON em `parse.rs` a validação SemVer estrita.
  * Rejeitar versões inválidas ou menores que `0.5.0` lançando `JsonError::UnsupportedVersion` ou `JsonError::InvalidVersionFormat`.
  * Emitir warning para versões `PARTIAL` e rejeitar maiores que `0.7.0`.
  * Sincronizar códigos de erro e a tabela de erros do manual de arquitetura.
* **Arquivos Afetados:**
  * [src/loader/nam_json/parse.rs](file:///home/fabio/nam-rs/src/loader/nam_json/parse.rs) — Lógica de verificação da versão.
  * [src/loader/nam_json/error.rs](file:///home/fabio/nam-rs/src/loader/nam_json/error.rs) — Inclusão de novas variantes `JsonError`.
  * [src/common/diagnostics/error_codes.rs](file:///home/fabio/nam-rs/src/common/diagnostics/error_codes.rs) — Novas entradas `NamErrorCode`.
  * [docs/architecture.md](file:///home/fabio/nam-rs/docs/architecture.md) — Sincronizar catálogo de erros.
* **Critério de Aceitação:**
  * Testar os limites de versão na suíte de testes unitários (`nam_json_test.rs`).
  * Arquivos com versão `0.4.9` ou `0.8.0` ou sem versão devem falhar no carregamento com códigos de diagnósticos apropriados.
  * Arquivos com versão `0.7.1` devem carregar emitindo um aviso em log.

---

### [ ] Tarefa 4.3: Documentar Política de Sample Rate Ausente e FastLUT (F-P3 e F-P4-c)

* **Achados Associados:**
  * [F-P3](file:///home/fabio/nam-rs/TODO-findings.md#L55) — Sample rate desconhecido: default 48000 vs sentinela C++ `-1.0`.
  * [F-P4-c](file:///home/fabio/nam-rs/TODO-findings.md#L74) — `FastLUTActivation` otimização C++ não portada.
* **Complexidade/Risco:** Baixo (Documentação).
* **Descrição:**
  O default adotado no Rust para sample rate ausente é `48000` Hz (iniciando prewarm real), divergindo intencionalmente do sentinela `-1.0` do C++ (que inibe prewarm). A otimização `FastLUTActivation` também não é portada por não ser feature de formato. Devemos documentar formalmente ambas as divergências intencionais.
* **Arquivos Afetados:**
  * [docs/cpp_parity_map.md](file:///home/fabio/nam-rs/docs/cpp_parity_map.md) — Documentar a política de sample rate ausente e o status da FastLUT.
* **Critério de Aceitação:**
  * Inclusão clara das seções no mapa de auditoria de paridade C++.

---

### [ ] Tarefa 4.4: Diagnóstico para LSTM Multi-Canal e Aliases do Linear (F-P4-a e F-P4-b)

* **Achados Associados:**
  * [F-P4-a](file:///home/fabio/nam-rs/TODO-findings.md#L67) — LSTM multi-canal: C++ aceita in/out channels arbitrários; Rust rejeita.
  * [F-P4-b](file:///home/fabio/nam-rs/TODO-findings.md#L71) — Aliases do Linear: C++ aceita legacy/old/partitioned_fft/partitioned-fft.
* **Complexidade/Risco:** Baixo.
* **Descrição:**
  * Trocar a rejeição genérica de LSTM multi-canal por um diagnóstico explícito de carregamento (`JsonError::UnsupportedMultiChannel`) em vez de falha silenciosa de detecção.
  * Adicionar suporte em `LinearImplementation::from_str` aos aliases do C++: `"legacy"`, `"old"`, `"partitioned_fft"` e `"partitioned-fft"`.
* **Arquivos Afetados:**
  * [src/loader/nam_json/topology/lstm.rs](file:///home/fabio/nam-rs/src/loader/nam_json/topology/lstm.rs) — Verificação estrita com erros explícitos.
  * [src/loader/nam_json/model.rs](file:///home/fabio/nam-rs/src/loader/nam_json/model.rs) — Aliases em `LinearImplementation`.
* **Critério de Aceitação:**
  * Tentar carregar um LSTM com `in_channels=2` deve retornar um erro de diagnóstico estruturado claro.
  * Mapeamento correto de todos os aliases em testes unitários.
