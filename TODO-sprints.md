<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-sprints.md — Planejamento Ágil de Implementação (`nam-rs`)

Este documento detalha o plano de execução ágil (Épicos, Sprints e Tarefas Técnicas Atômicas) derivado dos achados de auditoria e arquitetura registrados em [`TODO-findings.md`](file:///home/fabio/nam-rs/TODO-findings.md).

---

## Épico 01: Núcleo do Engine `NamLogger` e Retenção de Histórico

- **Objetivo**: Criar o motor centralizador de logging thread-safe (`NamLogger`), resolver o blocker de multi-instância CLAP no registro da facade `log::*`, incorporar o histórico recente de logs (`LogBuffer`) nos dumps de diagnóstico/suporte e crash reports, e adicionar rotação automática de arquivos de crash em disk cache.
- **Achados Cobertos em `TODO-findings.md`**:
  - [Finding 01 (parcial)](file:///home/fabio/nam-rs/TODO-findings.md#finding-01-facade-log-desconectada-no-modo-clap-plugin-p0--crítico) — Backend central unificado para a facade `log::*`.
  - [Finding 02](file:///home/fabio/nam-rs/TODO-findings.md#finding-02-dumps-de-diagnóstico-e-crash-reports-sem-rastro-de-execução-p0--crítico) — Rastro cronológico recente em `DiagnosticBundle` e crash reports.
  - [Finding 07](file:///home/fabio/nam-rs/TODO-findings.md#finding-07-risco-de-multi-instância-clap-com-logset_logger-global-p1--alto) — Suporte multi-instância CLAP sem falha no `log::set_logger()`.
  - [Finding 08](file:///home/fabio/nam-rs/TODO-findings.md#finding-08-buffer-de-4-kib-insuficiente-para-log-trace-no-panic-hook-p2--médio) — Expansão stack-safe do buffer no `panic_hook.rs`.
  - [Finding 09](file:///home/fabio/nam-rs/TODO-findings.md#finding-09-ausência-de-rotaçãolimpeza-de-crash-files-p3--baixo) — Rotação automática dos relatórios `crash-*.txt`.

---

### Sprint 1.1: Engine `NamLogger`, `LogBuffer` Ring Buffer e Concorrência Multi-Instância CLAP

#### Task 1.1.1: Design e Implementação do Ring Buffer `LogBuffer` Thread-Safe [DONE]

- **Descrição**: Criar a estrutura `LogBuffer` em [`src/common/diagnostics/logger.rs`](file:///home/fabio/nam-rs/src/common/diagnostics/logger.rs), responsável por reter em memória um histórico circular com capacidade para ~128 a 256 entradas de log formatadas (timestamp UNIX/relativo, log level, target/módulo, mensagem).
- **Direcionamento Técnico**:
  - Utilizar `Mutex<VecDeque<LogRecord>>` ou ring buffer de tamanho fixo com mecanismo de desacoplamento não-bloqueante off-RT (`try_lock()` ou lock otimista sem bloquear callers off-RT em caso de contenção).
  - Expor métodos thread-safe: `pub fn push(&self, record: LogRecord)`, `pub fn snapshot(&self) -> Vec<LogRecord>` e `pub fn render_trace(&self, limit: usize) -> String`.
  - **Restrição RT-Safety**: callers RT continuam proibidos de invocar `log::*`. A gravação no `LogBuffer` é estritamente off-RT.
- **Especialista**: Engenheiro Sênior de Concorrência & Rust (Off-RT Systems Architect).
- **Criticidade / Risco**: **Médio-Alto**. Requer atenção à contabilidade de memória em ring buffer e prevenção de deadlocks em panics.
- **Arquivos Afetados**:
  - `[NEW]` [`src/common/diagnostics/logger.rs`](file:///home/fabio/nam-rs/src/common/diagnostics/logger.rs)
  - `[MODIFY]` [`src/common/diagnostics/mod.rs`](file:///home/fabio/nam-rs/src/common/diagnostics/mod.rs)
- **Critérios de Aceite**:
  - Ring buffer com capacidade máxima garantida (descarte FIFO das entradas mais antigas ao atingir o limite).
  - Snapshot de logs formatados em runtime sem mutação do estado original.

---

#### Task 1.1.2: Implementação da Facade Bridge `NamLogger` e Suporte Multi-Instância CLAP [DONE]

- **Descrição**: Implementar a struct `NamLogger` que satisfaz a trait `log::Log` da crate `log`. Resolver a limitação de chamada única de `log::set_logger()` em ambientes com múltiplas instâncias CLAP rodando no mesmo processo da DAW.
- **Direcionamento Técnico**:
  - Proteger a invocação de `log::set_logger()` usando `std::sync::OnceLock` / `std::sync::Once`. Se o logger global já tiver sido instalado pela primeira instância, chamadas subsequentes em novas instâncias reutilizam o `NamLogger` global em vez de falhar ou causar panic.
  - O `NamLogger` central mantém:
    1. Instância global do `LogBuffer`.
    2. Lista thread-safe de sinks registrados para o plugin CLAP: `Mutex<Vec<Weak<HostLogSink>>>` ou abstração equivalente.
  - Ao receber um registro de log (`log::Log::log`), o `NamLogger`:
    1. Grava no `LogBuffer` global.
    2. Emite para `stdout`/`stderr` se executando em modo Standalone/CLI (respeitando `RUST_LOG` ou `NAM_LOG_LEVEL`).
    3. Percorre os sinks CLAP ativos (`Weak::upgrade()`), despachando a mensagem formatada para a extensão `HostLog` de cada plugin ativo, expurgando automaticamente handles expirados.
- **Especialista**: Arququiteto de Plugins de Áudio & Concorrência Rust.
- **Criticidade / Risco**: **Crítico (P0/P1)**. Blocker de arquitetura para a coexistência de múltiplos plugins nam-rs em um projeto da DAW.
- **Arquivos Afetados**:
  - `[MODIFY]` [`src/common/diagnostics/logger.rs`](file:///home/fabio/nam-rs/src/common/diagnostics/logger.rs)
  - `[MODIFY]` [`src/common/diagnostics/mod.rs`](file:///home/fabio/nam-rs/src/common/diagnostics/mod.rs)
- **Critérios de Aceite**:
  - Múltiplas instâncias do plugin inicializam sem panic e sem falha de `log::set_logger()`.
  - Logs gravados via `log::info!`, `log::warn!`, `log::error!` são direcionados tanto para o `LogBuffer` quanto para os sinks ativos.

---

#### Task 1.1.3: Testes Unitários e de Concorrência do Engine `logger.rs` [DONE]

- **Descrição**: Desenvolver a suíte de testes unitários para a infraestrutura de logging em [`src/common/diagnostics/logger_test.rs`](file:///home/fabio/nam-rs/src/common/diagnostics/logger_test.rs) (incluso via `#[cfg(test)] #[path = "logger_test.rs"] mod logger_test;` em `mod.rs`).
- **Direcionamento Técnico**:
  - Testar estouro/rollover de capacidade do `LogBuffer` (FIFO descarte).
  - Testar emissão paralela de logs a partir de N threads simultâneas.
  - Testar ciclo de vida de sinks CLAP (`Weak::upgrade` / descarte automático de instâncias finalizadas).
  - Testar inicialização idempotente concorrente de `NamLogger::init()`.
- **Especialista**: Engenheiro de QA & Testes Rust.
- **Criticidade / Risco**: **Médio**.
- **Arquivos Afetados**:
  - `[NEW]` [`src/common/diagnostics/logger_test.rs`](file:///home/fabio/nam-rs/src/common/diagnostics/logger_test.rs)
  - `[MODIFY]` [`src/common/diagnostics/mod.rs`](file:///home/fabio/nam-rs/src/common/diagnostics/mod.rs)
- **Critérios de Aceite**:
  - Todos os testes unitários passando em `cargo test --lib common::diagnostics::logger_test`.

---

### Sprint 1.2: Integração nos Relatórios de Diagnóstico, Panic Hook e Rotação de Crash Files [DONE]

#### Task 1.2.1: Anexar Rastro de Execução (`Recent Log Trace`) no `DiagnosticBundle::render()`

- **Descrição**: Modificar o gerador de diagnósticos em [`src/common/diagnostics/bundle.rs`](file:///home/fabio/nam-rs/src/common/diagnostics/bundle.rs) para incluir a seção `──── Recent Log Trace ────` extraída do `LogBuffer`.
- **Direcionamento Técnico**:
  - No método `DiagnosticBundle::render()`, consultar a snapshot do `LogBuffer` em `NamLogger`.
  - Exibir as últimas N mensagens (e.g. 50 a 100 linhas), aplicando a política de redação de caminhos caso `self.full` seja `false`.
  - Atualizar os testes existentes em [`src/common/diagnostics/diagnostic_test.rs`](file:///home/fabio/nam-rs/src/common/diagnostics/diagnostic_test.rs) para cobrir a inclusão da nova seção no output renderizado.
- **Especialista**: Engenheiro de Diagnóstico & UX CLI/Plugin.
- **Criticidade / Risco**: **Médio**.
- **Arquivos Afetados**:
  - `[MODIFY]` [`src/common/diagnostics/bundle.rs`](file:///home/fabio/nam-rs/src/common/diagnostics/bundle.rs)
  - `[MODIFY]` [`src/common/diagnostics/diagnostic_test.rs`](file:///home/fabio/nam-rs/src/common/diagnostics/diagnostic_test.rs)
- **Critérios de Aceite**:
  - Os comandos `nam-rs --diagnose` e o acionamento do botão `ℹ` na interface do plugin CLAP contêm a seção `Recent Log Trace`.

---

#### Task 1.2.2: Expansão do Buffer e Inclusão de Log Trace em `src/common/panic_hook.rs` [DONE]

- **Descrição**: Atualizar o manipulador de pânico em [`src/common/panic_hook.rs`](file:///home/fabio/nam-rs/src/common/panic_hook.rs) para incorporar as entradas mais recentes de log no crash report salvo em `~/.cache/nam-rs/crash-*.txt`.
- **Direcionamento Técnico**:
  - Expandir o buffer stack-allocated `report_buf` de 4 KiB (`[u8; 4096]`) para 16 KiB (`[u8; 16384]`) em `install_panic_hook`.
  - Em `format_panic_report_to_buf`, formatar a seção `──── Recent Log Trace ─────────────────────────────`, capturando até 20 a 30 linhas recentes do `LogBuffer` com verificação estrita de limites via `LimitWriter`.
  - Garantir que a inclusão de logs no crash path continue **zero-alloc** e imune a deadlocks (usando `try_lock()` ou snapshot seguro).
- **Especialista**: Engenheiro de Sistemas Low-Level & Crash Recovery.
- **Criticidade / Risco**: **Alto (Finding 08)**. Risco de estouro de pilha durante pânico se o buffer for mal gerido.
- **Arquivos Afetados**:
  - `[MODIFY]` [`src/common/panic_hook.rs`](file:///home/fabio/nam-rs/src/common/panic_hook.rs)
- **Critérios de Aceite**:
  - Os relatórios de pânico gerados em `~/.cache/nam-rs/crash-*.txt` trazem o rastro final de logs antes do crash.
  - Testes do atributo `#[cfg(feature = "heap-audit")]` em `panic_hook.rs` continuam passando sem regressões.

---

#### Task 1.2.3: Rotação e Limpeza Automática de Crash Files em `~/.cache/nam-rs/` [DONE]

- **Descrição**: Implementar mecanismo de retenção de histórico de crash reports (Finding 09), limitando o total de arquivos mantidos no diretório do usuário.
- **Direcionamento Técnico**:
  - Após salvar com sucesso um novo crash file em `~/.cache/nam-rs/crash-*.txt`, listar os arquivos correspondentes no diretório.
  - Se a contagem total exceder `MAX_CRASH_FILES` (ex: 10 arquivos), ordenar por data de modificação (`mtime`) ou pelo timestamp UNIX no nome do arquivo e remover os mais antigos.
  - Tratar falhas de E/S de forma graciosa (sem disparar novos pânicos dentro do panic hook).
- **Especialista**: Engenheiro de Infraestrutura & File System Linux.
- **Criticidade / Risco**: **Baixo**.
- **Arquivos Afetados**:
  - `[MODIFY]` [`src/common/panic_hook.rs`](file:///home/fabio/nam-rs/src/common/panic_hook.rs)
- **Critérios de Aceite**:
  - O diretório `~/.cache/nam-rs/` retém no máximo 10 relatórios de crash, expurgando automaticamente os registros excedentes mais antigos.

---

#### Task 1.2.4: Auditoria de Conformidade, RT-Safety e Validação do Épico 01 [DONE]

- **Descrição**: Realizar auditoria e suite de validação final no subsistema de logging construído no Épico 01.
- **Direcionamento Técnico**:
  - Verificar presenças dos cabeçalhos SPDX Apache-2.0 e Copyright 2026 em todos os arquivos novos/modificados.
  - Confirmar zero uso de `log::*` no hot-path RT (`src/dsp/`, `process()`).
  - Executar os scripts de verificação permitidos (`utils/lints.sh` e `utils/tests-quick.sh`).
- **Especialista**: Revisor Auditor & Lead QA.
- **Criticidade / Risco**: **Médio**.
- **Arquivos Afetados**:
  - Todos os arquivos do Épico 01.
- **Critérios de Aceite**:
  - `utils/lints.sh` sem nenhum warning ou erro.
  - `utils/tests-quick.sh` com 100% de aprovação nos testes da suíte rápida.

---

## Épico 02: Conexão nos Entrypoints e Encaminhamento CLAP `HostLog`

- **Objetivo**: Garantir que tanto o binário CLI (`main.rs`) quanto o plugin CLAP inicializem a facade central `NamLogger`, repassem as mensagens de log aos seus respectivos destinos (stderr ou extensão `HostLog` da DAW hospedeira), e migrem o código off-RT CLAP para a facade unificada `log::*`.
- **Achados Cobertos em `TODO-findings.md`**:
  - [Finding 01 (completo)](file:///home/fabio/nam-rs/TODO-findings.md#finding-01-facade-log-desconectada-no-modo-clap-plugin-p0--crítico) — Inicialização nos entrypoints e ponte automática entre `NamLogger` e `HostLog` CLAP.

---

### Sprint 2.1: Inicialização do `NamLogger` nos Entrypoints e Ponte Automática CLAP `HostLog`

#### Task 2.1.1: Inicialização do `NamLogger` no Plugin CLAP (`NamClapPlugin::new_shared`) [DONE]

- **Descrição**: Registrar a inicialização do `NamLogger` no ponto de fabricação/instanciação do plugin CLAP e conectar a extensão `HostLog` do host via callback de sink registrado.
- **Direcionamento Técnico**:
  - Em `NamClapPlugin::new_shared()` (ou fábrica associada em [`src/clap/plugin/shared.rs`](file:///home/fabio/nam-rs/src/clap/plugin/shared.rs)), invocar `NamLogger::init_clap()` (idempotente via `OnceLock`).
  - Extrair a extensão `clack_extensions::log::HostLog` quando fornecida pelo host CLAP.
  - Registrar um sink callback em `NamLogger::register_sink()` associado ao ciclo de vida da instância do plugin.
- **Especialista**: Arquiteto de Plugins CLAP & Concorrência Rust.
- **Criticidade / Risco**: **Crítico (P0/P1)**. Blocker para que chamadas `log::*` em módulos compartilhados funcionem dentro de DAWs.
- **Arquivos Afetados**:
  - `[MODIFY]` [`src/clap/plugin/shared.rs`](file:///home/fabio/nam-rs/src/clap/plugin/shared.rs)
  - `[MODIFY]` [`src/clap/plugin/mod.rs`](file:///home/fabio/nam-rs/src/clap/plugin/mod.rs)
  - `[MODIFY]` [`src/clap/plugin/main_thread/logging.rs`](file:///home/fabio/nam-rs/src/clap/plugin/main_thread/logging.rs)
- **Critérios de Aceite**:
  - Chamadas `log::info!`, `log::warn!`, `log::error!` emitidas em qualquer lugar da biblioteca populam o `LogBuffer` e são enviadas à DAW hospedeira.

---

#### Task 2.1.2: Refino e Validação da Inicialização no CLI Standalone (`main.rs`) [DONE]

- **Descrição**: Validar e refinar a inicialização do `NamLogger::init_standalone()` no ponto de entrada do binário CLI (`main.rs`).
- **Direcionamento Técnico**:
  - Garantir o correto parsing das variáveis de ambiente (`RUST_LOG` / `NAM_LOG_LEVEL`).
  - Emitir um log informativo inicial contendo a versão do `nam-rs`, target ISA (`x86-64-v3`) e o modo de execução.
- **Especialista**: Engenheiro de Sistemas CLI/Standalone.
- **Criticidade / Risco**: **Baixo-Médio**.
- **Arquivos Afetados**:
  - `[MODIFY]` [`src/main.rs`](file:///home/fabio/nam-rs/src/main.rs)
- **Critérios de Aceite**:
  - Execução via terminal gera output formatado em stderr respeitando o nível selecionado e popula o `LogBuffer`.

---

### Sprint 2.2: Migração das Chamadas Manuais `HostLog` para a Facade `log::*` nos Módulos CLAP

#### Task 2.2.1: Migração do Ciclo de Vida e Carregamento (`load.rs`, `housekeeping.rs`, `state.rs`, `preset_load.rs`) [DONE]

- **Descrição**: Substituir as chamadas brutas e manuais de `HostLog` nestes módulos off-RT por chamadas equivalentes à facade unificada `log::info!`, `log::warn!`, `log::error!`.
- **Direcionamento Técnico**:
  - Substituir trechos que chamavam `host.get_extension::<HostLog>()` com `CString` manuais por macros `log::*`.
  - Essa migração faz com que as mensagens passem pelo `NamLogger`, alimentando tanto o `LogBuffer` (diagnósticos/crash reports) quanto a DAW via o bridge construído na Sprint 2.1.
- **Especialista**: Engenheiro de Áudio CLAP.
- **Criticidade / Risco**: **Médio**. Requer atenção para preservar as mensagens informativas sem alterações de fluxo.
- **Arquivos Afetados**:
  - `[MODIFY]` [`src/clap/plugin/main_thread/load.rs`](file:///home/fabio/nam-rs/src/clap/plugin/main_thread/load.rs)
  - `[MODIFY]` [`src/clap/plugin/main_thread/housekeeping.rs`](file:///home/fabio/nam-rs/src/clap/plugin/main_thread/housekeeping.rs)
  - `[MODIFY]` [`src/clap/extensions/state.rs`](file:///home/fabio/nam-rs/src/clap/extensions/state.rs)
  - `[MODIFY]` [`src/clap/extensions/state_context.rs`](file:///home/fabio/nam-rs/src/clap/extensions/state_context.rs)
  - `[MODIFY]` [`src/clap/extensions/preset_load.rs`](file:///home/fabio/nam-rs/src/clap/extensions/preset_load.rs)
- **Critérios de Aceite**:
  - Chamadas diretas/manuais a `HostLog` removidas desses módulos.
  - Mensagens de log de estado e carregamento devidamente gravadas no `LogBuffer` e enviadas ao host.

---

#### Task 2.2.2: Migração dos Módulos de Interface Gráfica e Manipuladores de Arquivos (`gui/`, `file_dialogs.rs`, `gui.rs`) [DONE]

- **Descrição**: Migrar chamadas manuais `HostLog` nos módulos da GUI e seletores de arquivo para a facade `log::*`.
- **Direcionamento Técnico**:
  - Substituir requisições de log brutas em `gui.rs`, `gui/window/state.rs` e `gui/ui/zones/file_dialogs.rs` por `log::*`.
  - Manter `emit_pending_logs()` em [`src/clap/plugin/main_thread/logging.rs`](file:///home/fabio/nam-rs/src/clap/plugin/main_thread/logging.rs) operando **exclusivamente** para o consumo off-RT das 9 flags atômicas RT (`RtStatusFlags`).
- **Especialista**: Engenheiro de GUI & Frontend Plugin.
- **Criticidade / Risco**: **Médio**.
- **Arquivos Afetados**:
  - `[MODIFY]` [`src/clap/extensions/gui.rs`](file:///home/fabio/nam-rs/src/clap/extensions/gui.rs)
  - `[MODIFY]` [`src/clap/gui/window/state.rs`](file:///home/fabio/nam-rs/src/clap/gui/window/state.rs)
  - `[MODIFY]` [`src/clap/gui/ui/zones/file_dialogs.rs`](file:///home/fabio/nam-rs/src/clap/gui/ui/zones/file_dialogs.rs)
  - `[MODIFY]` [`src/clap/plugin/main_thread/logging.rs`](file:///home/fabio/nam-rs/src/clap/plugin/main_thread/logging.rs)
- **Critérios de Aceite**:
  - Módulos de GUI utilizam 100% `log::*`.
  - `emit_pending_logs()` fica restrito à drenagem de `RtStatusFlags`.

---

#### Task 2.2.3: Suíte de Testes de Integração CLAP & Logger Verification [DONE]

- **Descrição**: Desenvolver testes unitários e de integração para validar a ponte do `NamLogger` no ambiente CLAP.
- **Direcionamento Técnico**:
  - Criar/atualizar testes em `src/clap/` para instanciar o plugin via `NamClapPlugin` e verificar o registro do logger.
  - Validar que chamadas `log::info!` emitidas durante o ciclo de vida do plugin chegam ao `LogBuffer` e ao mock de `HostLog`.
- **Especialista**: Engenheiro de QA & Testes Rust.
- **Criticidade / Risco**: **Médio**.
- **Arquivos Afetados**:
  - `[MODIFY]` [`src/clap/test_util.rs`](file:///home/fabio/nam-rs/src/clap/test_util.rs)
  - `[MODIFY]` [`src/clap/processor_state_test.rs`](file:///home/fabio/nam-rs/src/clap/processor_state_test.rs)
- **Critérios de Aceite**:
  - Testes de integração CLAP executando sem erros e validando a captura e encaminhamento de logs.

---

#### Task 2.2.4: Auditoria de Conformidade, RT-Safety e Validação do Épico 02 [DONE]

- **Descrição**: Realizar a auditoria final de código e verificação do Épico 02.
- **Direcionamento Técnico**:
  - Confirmar a presença do cabeçalho SPDX Apache-2.0 e Copyright 2026 nos arquivos modificados.
  - Garantir zero chamadas `log::*` na thread de áudio real-time.
  - Executar `utils/lints.sh` e `utils/tests-quick.sh`.
- **Especialista**: Revisor Auditor & Lead QA.
- **Criticidade / Risco**: **Médio**.
- **Arquivos Afetados**:
  - Todos os arquivos do Épico 02.
- **Critérios de Aceite**:
  - `utils/lints.sh` sem avisos ou erros.
  - `utils/tests-quick.sh` com 100% de aprovação.

---

## Épico 03: Auditoria e Cobertura Completa de Logs na Base de Código (`/src/`)

- **Objetivo**: Inserir registros de log informativos, defensivos e precisos em todos os módulos off-RT da biblioteca, expandindo a visibilidade de diagnósticos em `src/loader/`, `src/dsp/`, `src/standalone/` e `src/clap/`, mantendo o isolamento absoluto de zero `log::*` no hot-path de áudio real-time.
- **Achados Cobertos em `TODO-findings.md`**:
  - [Finding 03](file:///home/fabio/nam-rs/TODO-findings.md#finding-03-ausência-de-logs-em-etapas-críticas-do-loader-p2--médio) — Cobertura de logs pre-dispatch e parsing nos módulos de carregamento (`loader/`).
  - [Finding 04](file:///home/fabio/nam-rs/TODO-findings.md#finding-04-ausência-de-logs-no-subsistema-dsp-p1--alto) — Logging off-RT nos construtores, inicializadores e configuradores de DSP (`dsp/`).
  - [Finding 05](file:///home/fabio/nam-rs/TODO-findings.md#finding-05-gaps-pontuais-em-eventos-da-camada-pipewire-p3--baixo) — Logs de renegociação de quantum/buffer PipeWire e detalhes de fallback HugeTLB (`standalone/`).
  - [Finding 06](file:///home/fabio/nam-rs/TODO-findings.md#finding-06-cobertura-parcial-de-eventos-do-ciclo-de-vida-clap-p2--médio) — Logs de instanciação de plugin (DAW/API CLAP), modo de renderização (Realtime vs Offline) e preset loading (`clap/`).

---

### Sprint 3.1: Cobertura de Logs nos Subsistemas de Carregamento (`src/loader/`) e DSP Off-RT (`src/dsp/`)

#### Task 3.1.1: Pre-Dispatch & Detailed Parsing Logs em `src/loader/` [DONE]

- **Descrição**: Adicionar chamadas `log::info!`, `log::warn!` e `log::debug!` nas etapas de pré-despacho, parsing de metadados JSON/NAMB e compilação de pesos nos carregadores de modelo em `src/loader/`.
- **Direcionamento Técnico**:
  - Registrar tipo de arquitetura (WaveNet, LSTM, ConvNet), número de parâmetros, dilatações/camadas e sample rate target durante `loader::build` e `namb::parse`.
  - Registrar falhas de validação de schema ou discrepâncias de versão com contexto rico (`log::warn!` / `log::error!`).
  - Em [`src/dsp/cabsim/loader.rs`](file:///home/fabio/nam-rs/src/dsp/cabsim/loader.rs), registrar o carregamento de arquivos IR WAV/FLAC, tamanho de amostras, quantidade de canais e de amostragem.
- **Especialista**: Engenheiro de Carregamento de Modelos & DSP.
- **Criticidade / Risco**: **Médio (P2)**.
- **Arquivos Afetados**:
  - `[MODIFY]` [`src/loader/build.rs`](file:///home/fabio/nam-rs/src/loader/build.rs)
  - `[MODIFY]` [`src/loader/namb/parse.rs`](file:///home/fabio/nam-rs/src/loader/namb/parse.rs)
  - `[MODIFY]` [`src/dsp/cabsim/loader.rs`](file:///home/fabio/nam-rs/src/dsp/cabsim/loader.rs)
- **Critérios de Aceite**:
  - O carregamento de modelos `.nam` / `.namb` e arquivos IR emite eventos descritivos de nível `info` contendo metadados do modelo/IR.
  - Erros de parsing capturam detalhes claros sobre a discrepância no arquivo sem causar pânico silencioso.

---

#### Task 3.1.2: Logs Off-RT em Construtores e Configuradores do Subsistema DSP (`src/dsp/`) [DONE]

- **Descrição**: Inserir registros de log em todas as funções construtoras, reconfigurações off-RT e alteradores de estado nos módulos de DSP de `src/dsp/` (resampler, oversample, noise gate, cabsim, adaptive compute).
- **Direcionamento Técnico**:
  - [`src/dsp/resampler/mod.rs`](file:///home/fabio/nam-rs/src/dsp/resampler/mod.rs): Logar ao instanciar o resampler (razão de amostragem `in_rate` -> `out_rate`, número de taps, modo de interpolação).
  - [`src/dsp/oversample.rs`](file:///home/fabio/nam-rs/src/dsp/oversample.rs): Logar alteração do fator de oversampling (ex: 2x/4x/8x), latência em amostras introduzida e atraso equivalente em milissegundos.
  - [`src/dsp/gate.rs`](file:///home/fabio/nam-rs/src/dsp/gate.rs): Logar a inicialização do Noise Gate e transições de configuração (threshold, attack, release, ativado/desativado).
  - [`src/dsp/adaptive.rs`](file:///home/fabio/nam-rs/src/dsp/adaptive.rs): Logar alteração nos modos de economia de computação adaptativa.
  - **Restrição Crucial**: Todas as chamadas `log::*` devem residir **exclusivamente em funções off-RT** (`new()`, `reset()`, `set_sample_rate()`, `set_ratio()`, `set_threshold()`). O método `process()` / hot-path de áudio é 100% isento de `log::*`.
- **Especialista**: Cientista de Processamento Digital de Sinais (DSP) & Rust.
- **Criticidade / Risco**: **Alto (P1 / Finding 04)**. Extrema atenção para não colocar nenhuma chamada `log::*` dentro das funções de processamento no áudio callback.
- **Arquivos Afetados**:
  - `[MODIFY]` [`src/dsp/resampler/mod.rs`](file:///home/fabio/nam-rs/src/dsp/resampler/mod.rs)
  - `[MODIFY]` [`src/dsp/oversample.rs`](file:///home/fabio/nam-rs/src/dsp/oversample.rs)
  - `[MODIFY]` [`src/dsp/gate.rs`](file:///home/fabio/nam-rs/src/dsp/gate.rs)
  - `[MODIFY]` [`src/dsp/cabsim/conv.rs`](file:///home/fabio/nam-rs/src/dsp/cabsim/conv.rs)
  - `[MODIFY]` [`src/dsp/adaptive.rs`](file:///home/fabio/nam-rs/src/dsp/adaptive.rs)
- **Critérios de Aceite**:
  - Qualquer reconfiguração de DSP off-RT gera registros claros de `log::info!`.
  - Nenhuma chamada `log::*` está presente nas rotinas executadas no loop real-time de áudio.

---

### Sprint 3.2: Cobertura de Logs nas Camadas Standalone (`src/standalone/`) e CLAP (`src/clap/`)

#### Task 3.2.1: Enriquecimento de Eventos Standalone / PipeWire e HugeTLB [DONE]

- **Descrição**: Preencher as lacunas pontuais de logging na camada PipeWire Host standalone, especificamente durante a renegociação de quantum/buffer e no fallback de HugeTLB.
- **Direcionamento Técnico**:
  - Em [`src/standalone/pw_host/run.rs`](file:///home/fabio/nam-rs/src/standalone/pw_host/run.rs): Adicionar `log::info!` quando o PipeWire renegociar o tamanho do buffer/quantum (ex: de 256 para 128 amostras ou alteração de taxa de amostragem no gráfico).
  - Em [`src/standalone/rt_setup/thread.rs`](file:///home/fabio/nam-rs/src/standalone/rt_setup/thread.rs) ou [`pm_qos.rs`](file:///home/fabio/nam-rs/src/standalone/rt_setup/pm_qos.rs): Enriquecer os logs de fallback de HugeTLB quando a alocação de memória de páginas grandes explícitas falhar e recorrer a THP (Transparent Huge Pages), registrando o código de erro do OS (`errno`) ou motivo detalhado.
- **Especialista**: Engenheiro de Linux Low-Latency & PipeWire.
- **Criticidade / Risco**: **Baixo-Médio (P3 / Finding 05)**.
- **Arquivos Afetados**:
  - `[MODIFY]` [`src/standalone/pw_host/run.rs`](file:///home/fabio/nam-rs/src/standalone/pw_host/run.rs)
  - `[MODIFY]` [`src/standalone/rt_setup/thread.rs`](file:///home/fabio/nam-rs/src/standalone/rt_setup/thread.rs)
- **Critérios de Aceite**:
  - Mudanças de quantum/buffer size no PipeWire são explicitamente logadas com os novos valores.
  - Falha na reserva HugeTLB descreve o motivo específico e confirma a ativação do fallback THP.

---

#### Task 3.2.2: Logs Estruturados de Ciclo de Vida CLAP, Render Mode e Presets [DONE]

- **Descrição**: Adicionar eventos de log via facade `log::*` durante a instanciação do plugin CLAP, transições de modo de renderização e carregamento de presets.
- **Direcionamento Técnico**:
  - [`src/clap/plugin/shared.rs`](file:///home/fabio/nam-rs/src/clap/plugin/shared.rs) / [`load.rs`](file:///home/fabio/nam-rs/src/clap/plugin/main_thread/load.rs): Emitir `log::info!` ao instanciar o plugin registrando o nome/id da DAW hospedeira (`host.name()`) e a versão da API CLAP negociada.
  - [`src/clap/extensions/render.rs`](file:///home/fabio/nam-rs/src/clap/extensions/render.rs): Emitir `log::info!` quando o host CLAP alternar entre modo `Realtime` e modo `Offline` (bounce/export HQ), informando a reconfiguração automática de oversampling.
  - [`src/clap/extensions/preset_load.rs`](file:///home/fabio/nam-rs/src/clap/extensions/preset_load.rs): Emitir `log::info!` ao carregar um preset, registrando o caminho do arquivo e o nome do preset.
- **Especialista**: Engenheiro de Plugins de Áudio CLAP.
- **Criticidade / Risco**: **Médio (P2 / Finding 06)**.
- **Arquivos Afetados**:
  - `[MODIFY]` [`src/clap/plugin/shared.rs`](file:///home/fabio/nam-rs/src/clap/plugin/shared.rs)
  - `[MODIFY]` [`src/clap/extensions/render.rs`](file:///home/fabio/nam-rs/src/clap/extensions/render.rs)
  - `[MODIFY]` [`src/clap/extensions/preset_load.rs`](file:///home/fabio/nam-rs/src/clap/extensions/preset_load.rs)
- **Critérios de Aceite**:
  - A inicialização do plugin na DAW produz registro com nome do host e versão da extensão.
  - Mudanças em modo de renderização (`Offline` vs `Realtime`) e trocas de presets aparecem no histórico de logs.

---

## Épico 04: Completude do `LogBuffer` e Qualidade do Output de Log

- **Objetivo**: Fechar as 5 lacunas residuais identificadas na auditoria pós-implementação dos Épicos 01–03, garantindo que o `LogBuffer` capture a totalidade dos eventos de diagnóstico relevantes (incluindo eventos RT drenados off-RT pela main thread) e melhorar a legibilidade do output de log no modo standalone.
- **Achados Cobertos em `TODO-findings.md`**:
  - [Finding 10](file:///home/fabio/nam-rs/TODO-findings.md#finding-10-eventos-rt-críticos-do-emit_pending_logs-não-alimentam-o-logbuffer-p2--médio) — Eventos RT críticos do `emit_pending_logs()` encaminhados ao `LogBuffer` via facade `log::*`.
  - [Finding 11](file:///home/fabio/nam-rs/TODO-findings.md#finding-11-ausência-de-log-na-destruição-da-instância-do-plugin-clap-p3--baixo) — Log de ciclo de vida na destruição (`Drop`) de `NamClapMainThread` e `NamClapShared`.
  - [Finding 12](file:///home/fabio/nam-rs/TODO-findings.md#finding-12-ausência-de-log-de-confirmação-no-state-save-clap-p3--baixo) — Log de confirmação e tamanho do blob no state save CLAP.
  - [Finding 13](file:///home/fabio/nam-rs/TODO-findings.md#finding-13-mudanças-de-latência-e-cauda-cabsim-reportadas-ao-host-sem-log-p3--baixo) — Logs de telemetria de mudanças de latência e cauda do CabSim reportadas ao host via `housekeeping()`.
  - [Finding 14](file:///home/fabio/nam-rs/TODO-findings.md#finding-14-formato-de-timestamp-no-stderr-do-modo-standalone-ilegível-para-humanos-p4--cosmético) — Formato de timestamp no `stderr` do modo standalone legível para humanos (`HH:MM:SS` / `HH:MM:SS.mmm`).

---

### Sprint 4.1: Captura de Eventos RT e Telemetria de Host no `LogBuffer`

#### Task 4.1.1: Encaminhamento de Eventos RT Drenados para a Facade `log::*` e `LogBuffer` [DONE]

- **Descrição**: Modificar a rotina `emit_pending_logs()` em [`src/clap/plugin/main_thread/logging.rs`](file:///home/fabio/nam-rs/src/clap/plugin/main_thread/logging.rs) para que, além de emitir mensagens para a extensão `HostLog` do host CLAP, cada uma das 9 flags atômicas de status real-time dadas por `RtStatusFlags` seja espelhada via chamadas `log::warn!` ou `log::error!`.
- **Direcionamento Técnico**:
  - Em [`logging.rs`](file:///home/fabio/nam-rs/src/clap/plugin/main_thread/logging.rs), após cada invocação `log.log(&shared, severity, &msg)`, emitir a chamada `log::*` correspondente. Como `emit_pending_logs()` é executado exclusivamente na thread principal (off-RT), a chamada a macros `log::*` é 100% segura.
  - Garantir a cobertura completa das 9 flags drenadas:
    1. `RT_STATUS_HAS_CLIPPED`: `log::warn!("NAM-rs: Output clipping detected!");`
    2. `RT_STATUS_GC_OVERFLOW`: `log::error!("NAM-rs: GC channel overflow! Possible memory leak.");`
    3. `RT_STATUS_GC_TIER3`: `log::warn!("NAM-rs: GC cascade reached Tier 3 (overflow buffer). Sustained GC pressure.");`
    4. `RT_STATUS_GC_CORRUPTED`: `log::error!("NAM-rs: GC overflow buffer corrupted! Forced leak to avoid UB.");`
    5. `RT_STATUS_HEAP_ALLOC`: `log::error!("NAM-rs: Heap allocation detected in audio thread during process()!");`
    6. `RT_STATUS_MODEL_LOAD_FAILED`: `log::error!("NAM-rs: Critical failure! No active model for processing.");`
    7. `RT_STATUS_SLIMMABLE_SLICE_FAILED`: `log::error!("NAM-rs: WaveNet slimmable slice_channels rebuild failed.");`
    8. Flags adicionais de falha de amostragem / estado RT registradas no bitmask.
  - Com essa alteração, se o plugin reportar clipping ou GC overflow e logo em seguida ocorrer um crash, o rastro do evento crítico estará preservado no `LogBuffer` e visível no `DiagnosticBundle::render()` e nos crash reports.
- **Especialista**: Engenheiro de Plugins CLAP & Diagnóstico Off-RT.
- **Criticidade / Risco**: **Médio-Alto (P2 / Finding 10)**. Alta relevância técnica para diagnóstico de falhas graves em suporte remoto.
- **Arquivos Afetados**:
  - `[MODIFY]` [`src/clap/plugin/main_thread/logging.rs`](file:///home/fabio/nam-rs/src/clap/plugin/main_thread/logging.rs)
- **Critérios de Aceite**:
  - Todos os 9 eventos RT drenados por `emit_pending_logs()` são gravados no `LogBuffer` global e aparecem no snapshot renderizado do `DiagnosticBundle`.
  - Nenhuma chamada `log::*` é feita dentro do audio thread (mantida a garantia de RT-safety, pois `emit_pending_logs()` é invocado apenas na main thread).

---

#### Task 4.1.2: Logging de Telemetria para Mudanças de Latência e Cauda CabSim em Housekeeping [DONE]

- **Descrição**: Adicionar registros de log via `log::info!` no método `housekeeping()` em [`src/clap/plugin/main_thread/housekeeping.rs`](file:///home/fabio/nam-rs/src/clap/plugin/main_thread/housekeeping.rs) quando forem detectadas e comunicadas ao host CLAP alterações na latência do plugin ou na cauda (tail length) do CabSim.
- **Direcionamento Técnico**:
  - No bloco de monitoramento de latência (`if current_latency != self.last_reported_latency`):

    ```rust
    self.last_reported_latency = current_latency;
    log::info!("[Housekeeping] Latency changed: {} samples reported to host.", current_latency);
    ```

  - No bloco de monitoramento do CabSim tail (`if cabsim_tail != self.last_reported_cabsim_tail`):

    ```rust
    self.last_reported_cabsim_tail = cabsim_tail;
    log::info!("[Housekeeping] CabSim tail changed: {} samples reported to host.", cabsim_tail);
    ```

  - Isso garante que a auditoria possa correlacionar a latência informada à DAW com a troca de fatores de oversampling ou alteração da resposta ao impulso (IR) do CabSim.
- **Especialista**: Engenheiro de Plugins de Áudio CLAP.
- **Criticidade / Risco**: **Baixo (P3 / Finding 13)**.
- **Arquivos Afetados**:
  - `[MODIFY]` [`src/clap/plugin/main_thread/housekeeping.rs`](file:///home/fabio/nam-rs/src/clap/plugin/main_thread/housekeeping.rs)
- **Critérios de Aceite**:
  - Alterações no valor de latência em amostras e no tail length do CabSim geram entradas claras de `log::info!` no `LogBuffer`.

---

### Sprint 4.2: Ciclo de Vida, Persistência de Estado e Formatação Standalone

#### Task 4.2.1: Log de Ciclo de Vida na Destruição do Plugin CLAP [DONE]

- **Descrição**: Incluir logs de ciclo de vida nos destrutores (`Drop`) das estruturas principais do plugin CLAP em [`src/clap/plugin/main_thread/mod.rs`](file:///home/fabio/nam-rs/src/clap/plugin/main_thread/mod.rs) e [`src/clap/plugin/shared.rs`](file:///home/fabio/nam-rs/src/clap/plugin/shared.rs).
- **Direcionamento Técnico**:
  - Em `impl<'a> Drop for NamClapMainThread<'a>` em [`src/clap/plugin/main_thread/mod.rs`](file:///home/fabio/nam-rs/src/clap/plugin/main_thread/mod.rs):

    ```rust
    fn drop(&mut self) {
        log::info!("NAM-rs: Plugin instance destroying — draining GC.");
        self.drain_gc_final();
    }
    ```

  - Em `impl Drop for NamClapShared` em [`src/clap/plugin/shared.rs`](file:///home/fabio/nam-rs/src/clap/plugin/shared.rs), adicionar `log::debug!("NAM-rs: NamClapShared dropped.");` garantindo que o log ocorra fora do caminho crítico de áudio.
  - Isso permite distinguir se um crash ocorreu com a instância em pleno processamento ou durante o desmantelamento final do plugin pelo host.
- **Especialista**: Engenheiro de Arquitetura de Sistemas Rust.
- **Criticidade / Risco**: **Baixo (P3 / Finding 11)**.
- **Arquivos Afetados**:
  - `[MODIFY]` [`src/clap/plugin/main_thread/mod.rs`](file:///home/fabio/nam-rs/src/clap/plugin/main_thread/mod.rs)
  - `[MODIFY]` [`src/clap/plugin/shared.rs`](file:///home/fabio/nam-rs/src/clap/plugin/shared.rs)
- **Critérios de Aceite**:
  - A destruição da instância do plugin gera mensagens informativas de log capturadas no `LogBuffer`.

---

#### Task 4.2.2: Log de Confirmação e Tamanho do Blob no State Save CLAP [DONE]

- **Descrição**: Adicionar log de confirmação e telemetria de tamanho serializado na função `save()` da extensão de estado CLAP em [`src/clap/extensions/state.rs`](file:///home/fabio/nam-rs/src/clap/extensions/state.rs).
- **Direcionamento Técnico**:
  - Na função `fn save(&mut self, output: &mut OutputStream) -> Result<(), PluginError>`:

    ```rust
    let blob_len = serialized.len();
    output.write_all(&serialized)...?;
    log::debug!("[State] Save completed: {} bytes serialized.", blob_len);
    ```

  - Elimina a assimetria entre `save()` (que não emitia log) e `load()` (que contava com logs detalhados), facilitando o diagnóstico de persistência de projetos na DAW.
- **Especialista**: Engenheiro de Plugins CLAP & Serialização.
- **Criticidade / Risco**: **Baixo (P3 / Finding 12)**.
- **Arquivos Afetados**:
  - `[MODIFY]` [`src/clap/extensions/state.rs`](file:///home/fabio/nam-rs/src/clap/extensions/state.rs)
- **Critérios de Aceite**:
  - Salvamentos de estado com sucesso registram o tamanho do blob serializado em bytes no `LogBuffer`.

---

#### Task 4.2.3: Formatação Legível de Timestamp Wall-Clock no Modo Standalone [DONE]

- **Descrição**: Atualizar a saída para `stderr` do `NamLogger` no modo standalone em [`src/common/diagnostics/logger.rs`](file:///home/fabio/nam-rs/src/common/diagnostics/logger.rs) para incluir horário legível em formato wall-clock (`HH:MM:SS` ou `HH:MM:SS.mmm`).
- **Direcionamento Técnico**:
  - Criar função auxiliar off-RT `format_wall_clock_time()` que converte `SystemTime::now()` em `HH:MM:SS` (ou `HH:MM:SS.mmm`) via aritmética pura de tempo/epoch ou utilizando dependências leves já presentes no projeto.
  - Atualizar o `eprintln!` no branch `if self.standalone_mode`:

    ```rust
    eprintln!(
        "{time} {level:5} {target}: {message}",
        time = format_wall_clock_time(),
        level = level_str(level),
        target = record.target(),
        message = record.args()
    );
    ```

  - **Restrição Importante**: O `LogRecord::timestamp_secs` em `LogBuffer` deve permanecer **estritamente como `u64` UNIX epoch**, assegurando parsing limpo por ferramentas automatizadas e diagnósticos máquina-a-máquina.
- **Especialista**: Engenheiro CLI/Standalone & Rust.
- **Criticidade / Risco**: **Cosmético (P4 / Finding 14)**. Melhora substancial na DX (developer experience) em execuções de terminal.
- **Arquivos Afetados**:
  - `[MODIFY]` [`src/common/diagnostics/logger.rs`](file:///home/fabio/nam-rs/src/common/diagnostics/logger.rs)
- **Critérios de Aceite**:
  - Saída no terminal em modo standalone exibe timestamps legíveis (`HH:MM:SS` ou `HH:MM:SS.mmm`).
  - O `LogBuffer` e `DiagnosticBundle` continuam retendo timestamps `u64` sem quebra de contrato.

---

### Sprint 4.3: Suíte de Testes e Validação Automatizada de Diagnóstico

#### Task 4.3.1: Cobertura de Testes Unitários de Diagnóstico e Validação do Épico 04

- **Descrição**: Desenvolver testes unitários para validar o comportamento das novas emissões de log e garantir conformidade com os critérios de RT-safety e lints.
- **Direcionamento Técnico**:
  - Em [`src/clap/processor_state_test.rs`](file:///home/fabio/nam-rs/src/clap/processor_state_test.rs): Adicionar teste unitário que simula a ativação de `RtStatusFlags` e verifica se a invocação de `emit_pending_logs()` grava com sucesso as entradas correspondentes no `LogBuffer`.
  - Em [`src/common/diagnostics/logger_test.rs`](file:///home/fabio/nam-rs/src/common/diagnostics/logger_test.rs): Testar a formatação de wall-clock e garantir que a estrutura de `LogRecord` retém a integridade do timestamp UNIX epoch.
  - Em [`src/clap/extensions/state.rs`](file:///home/fabio/nam-rs/src/clap/extensions/state.rs) (ou arquivo de teste associado): Validar que a rotina `save()` emite o log de confirmação.
  - Executar `utils/lints.sh` e `utils/tests-quick.sh` para assegurar conformidade total.
- **Especialista**: Lead QA & Revisor Auditor.
- **Criticidade / Risco**: **Médio**.
- **Arquivos Afetados**:
  - `[MODIFY]` [`src/clap/processor_state_test.rs`](file:///home/fabio/nam-rs/src/clap/processor_state_test.rs)
  - `[MODIFY]` [`src/common/diagnostics/logger_test.rs`](file:///home/fabio/nam-rs/src/common/diagnostics/logger_test.rs)
- **Critérios de Aceite**:
  - Testes unitários do `LogBuffer` e `emit_pending_logs()` aprovados.
  - `utils/lints.sh` e `utils/tests-quick.sh` sem nenhuma regressão.
