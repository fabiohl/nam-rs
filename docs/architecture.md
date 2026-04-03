# AudioRip: Um extrator bit perfect do stream de audio pipewire salvando no hd

## 1. Camada de Infraestrutura e Negociação Passiva (Áudio)

* **Recepção "Bit Perfect":** O software não deve forçar nenhuma taxa de amostragem (sample rate) ou profundidade de bits (bit depth). Ele deve se adaptar dinamicamente ao formato do stream que o PipeWire e aplicação Host entregar, na sua forma perfeita e evitando processamentos desnecessários.
* **Backend Standalone:** Para o modo standalone do `nih_plug`, priorize a integração mais próxima possível do PipeWire.
* **Captura de Metadados:** Assim que o stream for inicializado (ou sofrer mutação), o aplicativo deve capturar o sample rate e o bit depth reais para formatar o cabeçalho do arquivo WAV corretamente, garantindo zero resampling.
* **qpwgraph:** Um primeiro caso de uso (e método de teste) será usar o qpwgraph (aberto à parte) para conectar a saída de áudio de uma fonte qualquer (um navegador, por exemplo) na entrada do AudioRip.

## 2. Camada de Gravação e I/O Assíncrono (`io_uring`)

* **Objetivo:** Preparar o código fonte para o uso intrínseco de instruções vetorizadas mantendo-se restrito à versão *stable* do Rust, sem depender unicamente da vetorização automática do LLVM.
* **Otimização de Hardware:** Para operações SIMD em Rust estável, a arquitetura deve utilizar o módulo nativo `core::arch::x86_64` (intrinsics como AVX2/SSE) com compilação condicional, ou em alternativa uma crate leve focada em stable (como `safe_arch`). Isso prepara o terreno para operações matemáticas vetorizadas no DSP sem a necessidade da toolchain Nightly.
* **I/O no Kernel Linux:** A gravação em disco deve ser feita utilizando o subsistema `io_uring` estritamente através da crate `tokio-uring`. Isso é crucial para lidar com fluxos de altíssima resolução (ex: 192kHz/32-bit float multicanal) sem onerar o *user space*.

## 3. Concorrência Lock-Free (Zero Latência)

A arquitetura deve dividir estritamente o trabalho em duas threads comunicando-se via um Ring Buffer lock-free de Produtor-Único/Consumidor-Único (SPSC) (ex: crate ringbuf ou rtrb). O dimensionamento do buffer deve ser feito estritamente em tempo de compilação utilizando const generics, e as estruturas de dados compartilhadas devem ser forçadas a um alinhamento de 128 bytes utilizando a macro #[repr(align(128))] para neutralizar o fenômeno de falso compartilhamento (False Sharing) entre os núcleos da CPU.

* **Objetivo:** Garantir a previsibilidade estrita de memória e a mitigação de False Sharing no cache L1/L2 da CPU, além de travar a thread de áudio no tempo do kernel.
* **Thread de Áudio (DSP/Callback do nih_plug):** Baixíssima latência. Deve ser instanciada sob a política de escalonamento SCHED_FIFO do kernel Linux e atrelada a núcleos físicos de performance da CPU (Core Affinity). Zero alocação de memória na heap (proibido uso de Vec, Box, Arc em tempo de execução), zero I/O de disco, zero mutexes. Apenas recebe os blocos de bytes do host e os insere no Ring Buffer.
* **Thread de Gravação (I/O):** Roda fora do contexto de SCHED_FIFO. Consome o Ring Buffer e despacha as gravações para o disco usando o `io_uring` e a crate `hound` para estruturar o arquivo WAV dinamicamente.

## Entregáveis

* **Aplicação Headless (CLI)**: O aplicativo opera exclusivamente em modo purista via linha de comando, sem persistência de estado, salvando a captura no diretório de trabalho corrente como um arquivo wav.
* Funciona plenamente silenciado com graceful shutdown via `Ctrl+C`.
* Executável standalone que, ao ser plugado a uma fonte de áudio, começa a salvar com precisão o wav.
* Se o stream parar o arquivo é fechado e salvo em estado pronto para consumo.
* Volta ao estado de espera. Quando um novo stream começar, reinicia nova captura. E assim por diante.
* Lidar graciosamente com sinais do sistema operacional. Por exemplo, se ele receber um ctrl+c do usuário, toma as devidas providências de encerramento completo de todas as ações.

## Adento: Porque " target/debug/audiorip --backend jack"?

### 1. A Abstração do `nih_plug` e o "PipeWire-First"

Ser "PipeWire-first" significa que a aplicação foi desenhada para viver harmonicamente no grafo dinâmico do PipeWire, aceitando taxas de amostragem dinâmicas e comunicando-se via roteamento inteligente (como o `qpwgraph`).

No entanto, o crate `nih_plug` — que é o motor que permite focar apenas na matemática pura dentro da função `process()` — foi primariamente desenhado para construir plugins VST3/CLAP. Quando você usa a feature `standalone` para compilar um executável direto, o `nih_plug` precisa de um "backend" para conversar com o sistema operacional.

Atualmente, o `nih_plug` *não possui* um backend nativo escrito puramente nas bibliotecas C do PipeWire (`libpipewire`). Ele oferece suporte aos protocolos **ALSA**, **JACK** e **Dummy** (silêncio).

### 2. Por que o ALSA puro falharia aqui?

Se removermos o `--backend jack` do `utils/testar.sh`, o `nih_plug` tentaria fazer fallback para o ALSA. Aqui começam os problemas:

* Quando o PipeWire está rodando no Linux moderno, ele "sequestra" o controle exclusivo da placa de som ALSA em nível de hardware.
* Se o AudioRip tentar acessar o ALSA diretamente, ele baterá de frente com o PipeWire, resultando em erros de "dispositivo ocupado" (Device Busy) ou forçando o uso do *ALSA dmix*, que destrói a garantia de Bit-Perfect e insere latência não determinística.

### 3. A Magia do `pw-jack` (ABI Drop-in)

É aqui que o `pw-jack` brilha e ajuda no comportamento puramente de tempo real.

O `pw-jack` não é um emulador pesado. Ele é uma biblioteca de compatibilidade de interface binária (ABI). O que acontece nos bastidores é elegante:

1. Ao rodar `pw-jack target/debug/audiorip --backend jack`, o PipeWire injeta a sua própria versão da biblioteca `libjack.so` na inicialização do binário.
2. O `nih_plug` "pensa" que está se conectando a um servidor JACK clássico, que possui o contrato de tempo real (DSP) mais rigoroso, de mais baixa latência e mais determinístico da história do Linux.
3. **A Conversão Zero-Copy:** O PipeWire intercepta essas chamadas do JACK e as mapeia *nativamente e sem cópia de memória (zero-copy)* diretamente para o seu moderno *Graph* de nós (nodes).

### 4. Como isso garante o Tempo Real (SCHED_FIFO)?

O protocolo JACK exige um modelo de execução estritamente síncrono. O PipeWire herda isso. Ele entende que o binário é um nó de "Pro Audio" (Áudio Profissional) e aplica regras drásticas ao seu processo:

* **Escalonamento:** Ele acorda a thread `process()` do plugin no exato microssegundo em que o hardware precisa dos bytes, acionando o Ring Buffer lock-free (`SPSC`).
* **Prioridade Crítica:** Ele automaticamente coloca essa thread de processamento em `SCHED_FIFO` (se o usuário estiver no grupo `@audio`), isolando-a do *user space* comum e blindando o Cache L1/L2 de concorrência.

### Conclusão

Embora o projeto seja conceitualmente "PipeWire-first", escrever código nativo para a API crua do PipeWire exigiria abandonar o `nih_plug` e lidar com milhares de linhas de código inseguro (*unsafe C bindings*), gerenciamento manual de grafos, alocações assíncronas e buffers intrincados.

Usar o `pw-jack` é o estado da arte do pragmatismo no Linux hoje: você ganha toda a modernidade, o roteamento limpo do PipeWire e a compatibilidade do `qpwgraph`, mantendo a pureza matemática, zero-alocação e estabilidade de hardware que apenas o modelo do JACK API consegue forçar no backend do `nih_plug`.

## PS: A respeito do erro "[ERROR] nih_plug::wrapper::standalone::backend::jack: Buffer size changed from 512 to 2048. Buffer size changes are currently not supported, aborting..."

Como Especialista em Engenharia de Áudio DSP, afirmo que você **não precisa se preocupar de imediato**, especialmente se ocorreu apenas uma vez. No entanto, é fundamental entender a mecânica por trás desse "abort" para não ser pego de surpresa no futuro.

### O que causou o erro?

Esse erro é um choque de filosofias entre a dinamicidade do **PipeWire** e a implementação estrita do **nih_plug** em modo standalone:

* **A Natureza do PipeWire:** O PipeWire possui um grafo dinâmico. Ele é inteligente e ajusta o "quantum" (o tamanho do buffer de áudio em amostras) em tempo real, dependendo da demanda do sistema. Se você abre ou fecha um navegador, um player de vídeo ou o próprio `qpwgraph`, o PipeWire pode renegociar o grafo global e decidir subir o buffer (neste caso, de 512 para 2048) para otimizar o uso da CPU.
* **A Rigidez do nih_plug (JACK):** O backend standalone do `nih_plug` via JACK assume que o tamanho do buffer negociado na inicialização será fixo e imutável. Ele aloca toda a memória necessária para os blocos de áudio estritamente *antes* de iniciar a thread de tempo real para garantir o determinismo. Se o host (PipeWire) mudar o tamanho do buffer abruptamente, o `nih_plug` aborta intencionalmente porque não suporta redimensionar seus buffers internos on-the-fly sem fazer alocação na heap (o que violaria frontalmente a regra de zero alocação na thread DSP que você definiu).

### Por que aconteceu apenas uma vez?

Provavelmente, no exato momento em que o aplicativo estava capturando o áudio, ocorreu algum evento no seu sistema operacional que forçou o PipeWire a renegociar o grafo global de conexões, alterando temporariamente a topologia e o buffer de toda a cadeia de áudio.

### Como mitigar (se voltar a acontecer)?

Você não precisa alterar o código-fonte do **AudioRip** para consertar isso. O próprio log diz *"Buffer size changes are currently not supported"*, indicando que é uma limitação (ou escolha conservadora) do wrapper do framework, e não um bug da sua lógica lock-free.
Se o erro se tornar recorrente e começar a atrapalhar seus testes, a melhor abordagem é forçar o PipeWire a travar o tamanho do buffer e a taxa de amostragem na hora de invocar o executável. Você pode fazer isso injetando uma variável de ambiente no seu `utils/testar.sh`:

```bash
PIPEWIRE_QUANTUM=1024/48000 pw-jack target/debug/audiorip --backend jack
```

Isso impõe um contrato rigoroso ao PipeWire para aquela execução específica, prevenindo que ele tente alterar o buffer sob os pés do `nih_plug`.
Em resumo: o erro apenas prova que o `nih_plug` prefere capotar com segurança a tentar fazer alocação dinâmica proibida na thread de áudio. Pode seguir em frente com o projeto!
