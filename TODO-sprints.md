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

#### Task 1.1.3: Testes Unitários e de Concorrência do Engine `logger.rs`

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

### Sprint 1.2: Integração nos Relatórios de Diagnóstico, Panic Hook e Rotação de Crash Files

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

#### Task 1.2.2: Expansão do Buffer e Inclusão de Log Trace em `src/common/panic_hook.rs`

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

#### Task 1.2.3: Rotação e Limpeza Automática de Crash Files em `~/.cache/nam-rs/`

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

#### Task 1.2.4: Auditoria de Conformidade, RT-Safety e Validação do Épico 01

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
