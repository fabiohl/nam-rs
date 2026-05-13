<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva. -->
# TODO — Sprints de Integração CLAP

---

## Sprint 1 — Finalização do Esqueleto e Estabilidade Básica

**Objetivo:** Garantir que o esqueleto do plugin CLAP seja perfeitamente detectado, carregado e gerido pelo Host de Plugins do Bitwig Studio, com bypass puro operando sem alocações e sem interrupções.

### Épico 1.1 — Validação e Estabilidade no Bitwig Studio

- [ ] **Tarefa 1.1.1** — Validação do Lifecycle e Escaneamento no Bitwig 6.0.6 (Ubuntu Linux 25.10+)

  - **Contexto:** Bitwig realiza escaneamento de plugins em background e gerencia falhas via sua aba *Plug-in Errors*.
  - **Ações Técnicas:**
    1. Compilar o `.so` com `./utils/build-clap.sh` garantindo a cópia correta para `~/.clap/`.
    2. Adicionar o caminho nas preferências do Bitwig (*Settings > Plug-ins*) e monitorar o scan.
    3. Garantir a instanciação em uma Device Chain do Bitwig.
  - **Aceite:** O plugin `NAM-rs Neural Amp Modeler` aparece corretamente categorizado e é inserido sem erros ou warnings nos logs do host.

- [ ] **Tarefa 1.1.2** — Validação de Bypass Puro e Modos de Sandboxing

  - **Contexto:** O Bitwig Studio possui modos de isolamento flexíveis (*Sandboxing*). O plugin não deve gerar segfaults IPC.
  - **Ações Técnicas:**
    1. Tocar áudio pela trilha com o plugin e testar bypass do Bitwig (Device Enable).
    2. Alterar o modo de plugin host (*Settings > Plug-ins > Plug-in hosting*) para "Individually" e depois para "Together".
  - **Aceite:** Áudio em bypass passa ileso (0 latência extra). Transições instantâneas entre ativo/inativo. Zero crashes do Bitwig Audio Engine.

---

## Sprint 2 — Roteamento DSP e Modulação (CLAP Extensions)

**Objetivo:** Interligar os buffers de entrada do CLAP com as rotinas RT-safe do `DspPipeline`, e garantir suporte ao poderoso sistema de modulação frame-by-frame do Bitwig.

### Épico 2.1 — Integração do Pipeline DSP

- [ ] **Tarefa 2.1.1** — Interfacing de Buffers RT-Safe
  - **Ações Técnicas:**
    1. No callback `process()`, converter a enum `ChannelPair` (InputOutput/InPlace) do CLAP para `&[f32]` in e `&mut [f32]` out.
    2. Evocar `DspPipeline::process()` diretamente na áudio-thread.
  - **Aceite:** Passagem de ponteiros realizada estritamente com zero-allocation e zero locks mutáveis bloqueantes.
- [ ] **Tarefa 2.1.2** — Carregamento Inicial no `activate()`
  - **Ações Técnicas:**
    1. Aproveitar a chamada de `activate()` (cold-path) para inicializar o `NamModel` vazio ou com preset padrão e pre-alocar os buffers necessários usando o struct `PluginAudioConfiguration`.
- [ ] **Tarefa 2.1.3** — Telemetria de CPU e Integração Acústica
  - **Ações Técnicas:**
    1. Realizar loop de playback de DI com carga no Bitwig.
    2. Observar painel DSP do Bitwig para garantir que picos de DSP (Spikes) causados pelo inferenciador fiquem abaixo de 50%.
- [ ] **Tarefa 2.1.4** — Backend de Logging Unificado
  - **Ações Técnicas:**
    1. Configurar rota de log apontando falhas críticas de alloc/model-load simultaneamente para a extensão `clap_host_log` (para ler no log panel do Bitwig) e para `/tmp/nam-rs.log`.

### Épico 2.2 — Parâmetros e Sincronização Automática

- [ ] **Tarefa 2.2.1** — Mapeamento via `clap_plugin_params`
  - **Ações Técnicas:**
    1. Usando a trait apropriada em `clack-extensions`, mapear os IDs e bounds de `input_gain_db`, `output_gain_db`, `gate_threshold_db` e `bypass`.
  - **Aceite:** Parâmetros gerados aparecem imediatamente nos "Device Parameters" e no painel Remote Controls no rodapé do Bitwig.
- [ ] **Tarefa 2.2.2** — Modulação Contínua Sample-Accurate
  - **Contexto:** Bitwig envia LFOs rápidos que bombardeiam a thread de áudio com `CLAP_EVENT_PARAM_VALUE`.
  - **Ações Técnicas:**
    1. Dentro do `process()`, iterar a fila de eventos *antes e durante* a passagem dos frames.
    2. Sincronizar parâmetros suavizados para prevenir artefatos em modulação via áudio-rate LFO.
- [ ] **Tarefa 2.2.3** — Salvar e Restaurar Sessão (`clap_plugin_state`)
  - **Ações Técnicas:**
    1. Serializar estado do plugin (Path absoluto do modelo `.nam` instanciado e os valores do struct `NamPluginParams`).
    2. Responder adequadamente aos requests de salvamento do host.
  - **Aceite:** Reiniciar o Bitwig com o projeto salvo deve recarregar instantaneamente as redes neurais correspondentes.

---

## Sprint 3 — Portas de Áudio, Compensação de Latência e Estresse

**Objetivo:** Conformar-se com regras estritas de host compliance para blocos de áudio irregulares e Delay Compensation.

### Épico 3.1 — Extensões Vitais para Mixagem

- [ ] **Tarefa 3.1.1** — Declaração Exata de Portas (Audio Ports)
  - **Ações Técnicas:**
    1. Retornar corretamente informações ao solicitar canais via `clap_plugin_audio_ports`. Nomear porta: `Main Stereo`.
    2. Indicar suporte `in_place=true`.
- [ ] **Tarefa 3.1.2** — ADC (Automatic Delay Compensation)
  - **Ações Técnicas:**
    1. Computar `latência = resampler_delay + pipeline_lookahead` (se aplicável).
    2. Notificar o host invocando callback via extension `clap_host_latency` se o modelo for trocado gerando alteração de buffers.
  - **Aceite:** Trilha com NAM-rs em paralelo a trilhas diretas soam alinhadas por fase. O Bitwig relata o número exato de delay samples.

### Épico 3.2 — Estabilidade Extrema (Host-Stress)

- [ ] **Tarefa 3.2.1** — Tratamento de Block Sizes Dinâmicos
  - **Contexto:** O Bitwig frequentemente subdivide os blocos para automação ou rendering offline (ex: envia blocos de 17 samples, depois 53, depois 256).
  - **Ações Técnicas:** Testar e ajustar iteradores/buffers SIMD residuais do modelo para acomodar `frame_count` não-potência de 2.
- [ ] **Tarefa 3.2.2** — Asserções `CountingAllocator`
  - **Ações Técnicas:** Acionar verificador customizado para banir qualquer *malloc* indesejado nos métodos do trait de processamento CLAP.
- [ ] **Tarefa 3.2.3** — Integração CI com CLI Host
  - **Ações Técnicas:** Uso das crates `clack-host` para escrever rotinas de testes que simulem envios de blocos fragmentados parecidos com o Bitwig sem levantar UI.

---

## Sprint 4 — Interface Gráfica Responsiva (egui + baseview)

**Objetivo:** Painel customizado nativo, isolado da UI-thread do Bitwig, com visualização fluida e zero-interferência na thread de áudio.

### Épico 4.1 — Integração de Janela (Embedded Windowing)

- [ ] **Tarefa 4.1.1** — Inserir Dependências UI
  - **Ações Técnicas:** Adicionar deps de GUI sob a feature `clap-plugin` (`egui`, `baseview`, `rfd` para seleção de arquivo de modelo).
- [ ] **Tarefa 4.1.2** — Implementação do Extension `clap_plugin_gui`
  - **Contexto:** No Linux, o Bitwig fornecerá handles X11 ou Wayland dependendo da sessão do usuário no Ubuntu 25.10.
  - **Ações Técnicas:**
    1. Subir uma nova janela `baseview` e acoplar no `raw_window_handle` passado pelo host.
    2. Inicializar o renderizador `egui` e a fila de mensagens cross-thread SPSC com a áudio thread.
  - **Aceite:** O container abre e fecha imediatamente ao comandar a visualização da UI no device chain do Bitwig, sem congelar (hangs).

### Épico 4.2 — Interações Bidirecionais e UX

- [ ] **Tarefa 4.2.1** — UI Controls e Browsing
  - **Ações Técnicas:**
    1. Input/Output Sliders bidirecionais.
    2. Integração nativa de File Dialog (`rfd`) para carregar o modelo neural.
- [ ] **Tarefa 4.2.2** — Sincronia de Gestos de Automação
  - **Ações Técnicas:**
    1. Quando o usuário clica e arrasta um botão na UI customizada, disparar `CLAP_EVENT_PARAM_GESTURE_BEGIN` e `END` para que o Bitwig registre a trilha de automação sem glitches de descontinuidade.
- [ ] **Tarefa 4.2.3** — Visualização a 60fps (Medidores e Telemetria)
  - **Ações Técnicas:**
    1. Fazer pooling atômico de níveis True Peak via SPSC e desenhar VUs de alta atualização, medidores de delay interno e diagnóstico de SIMD ativado na CPU (AVX2, AVX-512).

---

## Sprint 5 — Otimizações Avançadas de Engine (Opcional p/ Alpha)

### Épico 5.1 — Pool de Threads e Desempenho Multi-Instância

- [ ] **Tarefa 5.1.1** — Investigar/Implementar `clap_plugin_thread_pool`
  - **Contexto:** Bitwig possui uma infraestrutura soberba para multithreading de áudio. Se ele provê a extension de thread pool CLAP, podemos injetar workers GEMV na engine do host ao invés de gerenciar threads concorrentes próprias.
  - **Ações Técnicas:** Checar se `HostHandle` implementa `clap_host_thread_pool`. Fallback elegante.
- [ ] **Tarefa 5.1.2** — Compartilhamento de Pesos na RAM (Flyweight Pattern)
  - **Ações Técnicas:** Se Múltiplas instâncias no Bitwig (ex: 8 clones em backing tracks) utilizarem o mesmo modelo, apontar as tabelas const para um `Arc` imutável, cortando o uso de memória por instância drasticamente. Validar via Htop simulando no Bitwig.

### Épico 5.2 — Integrações Nativas Premium (Bitwig First-Class Citizen)

- [ ] **Tarefa 5.2.1** — Integração Visual Dinâmica (`clap_plugin_track_info`)
  - **Contexto:** Bitwig informa ao plugin a cor e o nome da trilha em que está inserido.
  - **Ações Técnicas:** Assinar a extensão de track info. Repassar a cor atual da trilha para o `egui` via SPSC e usar como *accent color* da interface.
- [ ] **Tarefa 5.2.2** — Mapeamento Automático para Hardware (`clap_plugin_remote_controls`)
  - **Contexto:** Bitwig usa "Remote Controls" para mapear plugins instantaneamente para teclados e superfícies de controle (Push, Maschine).
  - **Ações Técnicas:** Definir "Páginas" lógicas (ex: "Main", "Gate") com os IDs dos parâmetros, permitindo controle físico out-of-the-box sem MIDI Learn.
- [ ] **Tarefa 5.2.3** — Economia Extrema de CPU via Auto-Sleep (`clap_plugin_tail`)
  - **Contexto:** Bitwig adormece plugins sem áudio, mas precisa saber do "Tail" (cauda) para não cortar decays abruptamente.
  - **Ações Técnicas:** Implementar a extensão reportando o decay/tail do modelo carregado.
- [ ] **Tarefa 5.2.4** — Suporte a Modulação Não-Destrutiva (The Grid / LFOs)
  - **Contexto:** Bitwig modula usando `CLAP_EVENT_PARAM_MOD` (delta temporal) ao invés de alterar o valor real do knob.
  - **Ações Técnicas:** Processar eventos de modulação e desenhar os "anéis de modulação" na interface `egui` para feedback visual premium.

---

## Sprint 6 — Validação Final e Alpha Release

### Épico 6.1 — Assinatura de Estabilidade Completa

- [ ] **Tarefa 6.1.1** — Sessão de Estresse no Bitwig Studio
  - **Ações Técnicas:** Renderizações offline ultrarrápidas (*Bounce* e *Bounce in Place*), ativação de racks e macros modulando os CLAPs massivamente, gravação em loop contínuo de 1 hora sem leak memory.
- [ ] **Tarefa 6.1.2** — Auditoria Estrita via `clap-validator`
  - **Ações Técnicas:** Executar suite `clap-validator` do spec oficial provando 0 avisos estruturais.

### Épico 6.2 — Publicação v2.0.0 Alpha

- [ ] **Tarefa 6.2.1** — Correção e Publicação de Documentos
  - **Ações Técnicas:** Modificar `docs/architecture.md` detalhando decisões sobre sandboxing no Linux e a estratégia multi-thread. Inserir imagens rodando no Bitwig.
- [ ] **Tarefa 6.2.2** — Suporte a Idiomas na GUI
  - **Ações Técnicas:** Ajustar labels da GUI (PT-BR, EN-US). O Bitwig no Ubuntu já pode oferecer hints de locale pelo ENV.
- [ ] **Tarefa 6.2.3** — Release
  - **Ações Técnicas:** Tag no git e pacote pronto: libnam_rs.so compactado.

> **📋 FINALIZAÇÃO — Sprint 6 concluída**
> O plugin NAM-rs v2.0.0 Alpha estará consolidado com desempenho irretocável em sistemas Linux rodando Bitwig Studio 6.0.6.
