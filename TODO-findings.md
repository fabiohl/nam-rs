# Findings & Optimization Opportunities (Resilience and Robustness Specialist)

## 1. Vulnerabilidade de OOM (Out-Of-Memory) e Panics em Alocadores Customizados (Oversized Payloads)

**Descrição:** Os arquivos `nam_json` e `cabsim` são dados fornecidos pelo usuário. Modelos ou IRs maliciosos, corrompidos ou não suportados (ex: milhares de camadas, ou `hidden_size` massivo) podem solicitar uma alocação monstruosa de memória. Em `src/math/common/huge_alloc.rs` e `src/math/common/aligned.rs`, existem chamadas como `Layout::from_size_align(...).unwrap();`. Se o tamanho exceder `isize::MAX`, isso causará um pânico (`unwrap`), derrubando a DAW do usuário ou o host do PipeWire de forma abrupta.
**Proposta de Solução:**

- Implementar limites máximos rígidos (Hard Limits) no parser JSON e no carregador de IRs (ex: `MAX_LAYERS`, `MAX_HIDDEN_SIZE`, `MAX_IR_LENGTH`).
- Substituir todos os `.unwrap()` associados a `Layout::from_size_align` por tratamento de erro robusto (`Result`), propagando a falha (`Error::OutOfMemory` ou `Error::UnsupportedTopology`) para a UI/Host informando que o modelo é inválido, mantendo o plugin vivo.

## 2. Refatoração de `unwrap()` em Threads de Produção e GC (Garbage Collection)

**Descrição:** Foi identificado o uso de `.unwrap()` e `.expect()` durante a inicialização do resampler e trocas de modelo nas threads de produção (ex: `src/standalone/pw_host/rt_callback/resampler_swap.rs` e construtores como `NamResampler::new`). Embora não estejam na thread crítica de áudio (RT), um pânico nessas threads comprometerá o estado do plugin e fatalmente encerrará o processo host.
**Proposta de Solução:**

- Alterar as assinaturas das fábricas de resampler e construtores de modelo de estado de GUI para retornar `Result` sempre que a alocação falhar ou parâmetros incompatíveis forem fornecidos.
- Tratar as falhas graciosamente: fazendo log da falha e cancelando a transição de estado (continuando a operar em "Bypass" ou mantendo o modelo anterior seguro).

## 3. Segurança Estrutural e Prova Matemática de `unsafe` (Fuzzing e Miri)

**Descrição:** O uso de `unsafe` está rigorosamente documentado (`// SAFETY:` flags em buffers de espelhamento, SIMD, e `oversample.rs`). Entretanto, em sistemas de áudio robustos de baixíssima latência, as invariantes de segurança não devem depender apenas de auditoria humana.
**Proposta de Solução:**

- Incorporar alvos de fuzzing focados em buffer (usando `cargo-fuzz` / `libfuzzer`) especificamente nas estruturas: `MirroredBuffer`, `AlignedVec` e iteradores SIMD de `sinc_kernel.rs`.
- Adicionar uma etapa no pipeline de testes longos (`utils/tests-long.sh`) que compile os testes unitários destas estruturas utilizando o `cargo miri test` para detectar cientificamente o uso de ponteiros pendentes, aliasing inválido, falhas de sincronização multithreaded e vazamentos silenciosos de memória.

## 4. Mitigação de SPSC GC Overflow em Condições Extremas

**Descrição:** O sistema SPSC possui um `GcOverflowBuffer` engenhoso (`src/common/spsc/mod.rs`), mas sob cenários de estresse máximo e mal-intencionado (ex: automação massiva da DAW trocando presets a cada ciclo de renderização), as filas "lock-free" podem saturar antes que a thread de limpeza do GC em background consiga atuar, correndo o risco de bloquear ou vazar estado.
**Proposta de Solução:**

- Assegurar e comprovar através de testes de integração (`tests-long.sh`) que a taxa limite de submissão (Rate Limiting) é sempre estrita. Se o produtor falhar ao inserir no SPSC (`push` falha), os recursos devem ir impreterivelmente ao overflow sem causar loops infinitos (spin-lock).

---

## Epics & Sprints (Planejamento Ágil)

### Sprint 1: Blindagem de Alocação de Memória e Sanitização de Entradas (Crítico) [DONE]

**Objetivo:** Garantir que nenhuma entrada do usuário (IRs ou arquivos `.nam`) seja capaz de causar falhas catastróficas de *Out-Of-Memory* (OOM) no host.

- **Epic 1.1: Limites Rígidos (Hard Limits) na Entrada de Dados (Refere-se ao Finding 1)** [DONE]
  - **Task 1.1.1 (Parser `nam_json.rs`):** Declarar constantes de segurança para limites razoáveis de processamento de redes neurais em áudio real-time, tais como: `MAX_LAYERS = 8`, `MAX_HIDDEN_SIZE = 512`. Injetar lógicas de validação rigorosas logo após o parsing bruto do JSON. Se a arquitetura solicitada exceder tais constantes, retornar um enumerador `Error::UnsupportedTopology` imediatamente.
  - **Task 1.1.2 (Loader Wav/IR `cabsim/loader.rs`):** Declarar um limite de segurança rígido para o tamanho máximo permitido de um *Impulse Response* (ex: `MAX_IR_LENGTH = 192000` samples, que corresponde a ~4s em 48kHz). Rejeitar o carregamento do arquivo precocemente (`Result::Err`) antes de submeter os samples a conversões de ponto flutuante ou alocações.
  - **Task 1.1.3 (Suíte de Regressão Negativa):** Criar uma suíte especializada de testes (`loader_malformed_test.rs`) responsável por alimentar propositalmente os módulos com metadados e topologias forjadas ou exageradas, asseverando assertivamente (via `assert!(result.is_err())`) que a biblioteca repulsa as falhas (Soft Reject) graciosamente e com alta performance (< 5ms).

- **Epic 1.2: Refatoração Resiliente de Alocadores (Refere-se ao Finding 1)** [DONE]
  - **Task 1.2.1 (`huge_alloc.rs`):** Remover terminantemente qualquer uso de `.unwrap()` e `.expect()` associado a `Layout::from_size_align()`. A função deve obrigatoriamente retornar um `Result<T, NamErrorCode>`, mapeando falhas de *layout* para `NamErrorCode::OutOfMemory`. **ATENÇÃO:** É expressamente proibido utilizar *fallbacks silenciosos* (como `unwrap_or_else` retornando ponteiros *dangling* ou vetores vazios) para mascarar o erro internamente.
  - **Task 1.2.2 (`aligned.rs`):** Modificar a assinatura dos construtores base de `AlignedVec` (`new`, `with_capacity`, `from_vec`, `resize`, etc.) para que deixem de retornar `Self` e passem a retornar obrigatoriamente `Result<Self, NamErrorCode>`.
  - **Task 1.2.3 (Cascateamento Rigoroso e Propagação de `Result`):** Inspecionar *toda a árvore de dependências* que chama `AlignedVec::new` ou `huge_alloc`. Isso inclui, mas não se limita a, construtores críticos de áudio no *hot path*, como `NamResampler::new` (`resampler.rs`), `CabSimConv::new` (`cabsim/conv.rs`), e `WaveNetLayerState::new`. O implementador **DEVE** alterar a assinatura dessas funções dependentes para que elas também retornem `Result<Self, NamErrorCode>`. Sob nenhuma circunstância a falha de alocação de memória (OOM) deve resultar na injeção de uma estrutura com capacidade 0 (zero-capacity vector) no motor DSP, o que causaria Segfaults fatais por Undefined Behavior. O erro real de OOM deve cascatear até a *API Host* (e.g. SPSC Payload de carregamento ou extensão CLAP), onde será devidamente tratado/negado em segurança.

---

### Sprint 2: Erradicação de Panics em Transições de Estado e Threads Não-RT

**Objetivo:** Assegurar que, após instanciado, mudanças abruptas nos parâmetros do plugin CLAP não levem à queda de instâncias da DAW sob hipótese alguma.

- **Epic 2.1: Robustez no Resampling e Troca Dinâmica (Refere-se ao Finding 2)**
  - **Task 2.1.1 (Inicialização Crítica do `NamResampler`):** Em `src/dsp/resampler.rs`, revisar milimetricamente a função `NamResampler::new`. Identificar qualquer equação matemática ou conversão vetorial onde `unwrap()` ou `.expect()` estejam presentes para checagem de buffer size. Substituir pela construção atômica baseada em `Result`.
  - **Task 2.1.2 (SPSC Producer em `resampler_swap.rs`):** Modificar a pipeline assíncrona que cria o resampler instanciando-o no *Main Thread* antes de submetê-lo à fila `producer.push()`. Validar rigorosamente se o buffer SPSC não alcançou lotação e se a alocação do Resampler teve êxito prévio, mitigando as falhas do lado do produtor (interface gráfica/DAW) de modo a nunca interromper o laço de mensagens telemétricas.

- **Epic 2.2: Blindagem Efetiva no Frontend CLAP (Refere-se ao Finding 2)**
  - **Task 2.2.1 (APIs Nativas em `clap/extensions/`):** Revisar as implementações dos *traits* C-FFI das extensões CLAP (particularmente a extensão de *State* e *Preset Load*). Onde ocorrerem falhas internas previsíveis (ex: disco cheio, permissões negadas ou buffer host inválido), certificar-se estritamente de retornar códigos booleanos previstos no header do CLAP (`false` ou NULL pointer callbacks) ao invés do processo hospedeiro colapsar em um `unwrap()`.

---

### Sprint 3: Provas Formais e Testabilidade Extrema (Quality Assurance Avançado)

**Objetivo:** Instituir barreiras matemáticas rígidas de Integração Contínua para refutar e comprovar definitivamente a ausência de Undefined Behaviors (UBs).

- **Epic 3.1: Fuzzing Direcionado (Refere-se ao Finding 3)**
  - **Task 3.1.1 (Setup do `cargo-fuzz`):** Criar e estruturar o diretório padrão `fuzz/` na raiz do projeto com o workspace necessário e dependências do fuzzer (`libfuzzer-sys`).
  - **Task 3.1.2 (Targets Estressores de Fuzzing):** Escrever arquivos targets focados primariamente em mutações estruturais nas áreas críticas: `mirror_buf.rs` e `aligned.rs`. O utilitário Fuzzer deverá injetar combinações completamente absurdas de offsets de alinhamento, limites nominais e escritas circulares massivas com o intuito de colapsar e estressar os ponteiros lógicos ao limite absoluto suportado.
  - **Task 3.1.3 (Script Operacional de Fuzzing):** Construir o utilitário shell script `/utils/tests-fuzz.sh` (com `chmod +x`) configurado com todos os parâmetros recomendados (`-Z sanitizer=address`) visando propiciar execução reprodutível do `cargo fuzz run`.

- **Epic 3.2: Sanitização Rigorosa via Miri Engine (Refere-se ao Finding 3)**
  - **Task 3.2.1 (Automação de Testes Longos):** Configurar no script base de testes noturnos/longos (`utils/tests-long.sh`) um acionamento isolado, mas obrigatório, para o comando `cargo miri test --target x86_64-unknown-linux-gnu`.
  - **Task 3.2.2 (Isolamento FFI com Decorators):** Demarcar cuidadosamente suítes de testes correntes que utilizam interfaces FFI externas puras de sistema (kernel level ou host DAW) ignorando-as explicitamente da análise via diretiva condicional (`#[cfg(not(miri))]`). Dessa forma, restringe-se eficientemente o tempo computacional da interpretação do Miri apenas à prova de ponteiros lógicos dos módulos iteradores SIMD (`sinc_kernel.rs`) e dos citados alocadores customizados.

---

### 🔥 CORREÇÃO RESIDUAL (Planejamento Ágil - Falha na Execução da Epic 1.2) [DONE]

**Objetivo:** Erradicar o uso de `.expect()` e panics explícitos que foram injetados indevidamente nas funções de inicialização de DSP (`oversample.rs` e `resampler.rs`) em substituição ao fallback silencioso. O pânico derruba o host e viola a regra de segurança.

- **Epic 1.2 (Retrabalho Obrigatório): Cascateamento Genuíno de `Result`** [DONE]
  - **Task 1.2.4 (`src/dsp/oversample.rs`):** A estrutura `X2Stage` e `OversampleEngine` instanciam `AlignedVec::new`. O implementador injetou `.expect("OOM: X2Stage up_ring")`. **Ação Exigida:**
    1. Alterar a assinatura de `X2Stage::new(...)` para retornar `Result<Self, NamErrorCode>`.
    2. Substituir o `.expect()` pelo operador `?` (`AlignedVec::new(...)?`).
    3. Alterar `OversampleEngine::new(...)` para `Result<Self, NamErrorCode>`, cascateando com `?`.
  - **Task 1.2.5 (`src/dsp/resampler.rs`):** O implementador injetou `.expect("failed to allocate resampler delay line")` na inicialização de `ResamplerCore`. **Ação Exigida:**
    1. Alterar `ResamplerCore::new(...)` para `Result<Self, NamErrorCode>`.
    2. Utilizar `DelayLine::new()?` nas atribuições de `state_l` e `state_r`.
    3. Atualizar as chamadas de `ResamplerCore::new` dentro de `NamResampler::new` (que já retorna `Result` via `anyhow`) para utilizar o operador `?`, mapeando o erro apropriadamente se necessário.
  - **Task 1.2.6 (`src/dsp/cabsim/conv.rs` e similares):** Varrer a base por qualquer `.expect("OOM...")` residual inserido fora de diretórios de testes (`#[cfg(test)]`). Nenhuma alocação em código de produção deve entrar em pânico. A falha deve seguir o caminho do `Result` até o caller primário.
