# TODO-sprints

## Sprint 3: Arquitetura Core e Estabilidade

### Tarefa 3.1: DspBridge com Double-Buffer

Estimativa: ~4h | Complexidade: Média | Ganho: Robustez contra futuras mudanças na topologia de áudio

> Pode ser combinada com outras tarefas de baixa prioridade durante
> refatorações periódicas.

#### 3.1 Motivação

O `DspBridge` atual em `pw_host.rs` (linhas 72–82) usa um **único buffer** com
generation counter atômico. Na implementação atual, ambos os callbacks rodam no
mesmo `ThreadLoop` PW (`node.group = "nam-rs-dsp"`) e o `fence(Release)` +
`generation.fetch_add` só é emitido **após** ambos os canais L e R estarem
escritos (linhas 536–544). Portanto, **não há bug atual**.

Porém, um double-buffer com swap atômico seria mais robusto contra:

- Futuras mudanças na topologia de streams PW
- Cenários onde os callbacks possam executar em threads diferentes
- Simplificação da lógica de sincronização (elimina `n_samples == 0` guard)

#### 3.1 Proposta Técnica

1. Alocar 2 instâncias de buffer (front/back) via `Box::leak`
2. Usar `AtomicBool` ou `AtomicUsize` para indexar qual buffer é o "ativo para leitura"
3. Capture escreve sempre no back-buffer; ao final, swapa o índice com `fence(Release)`
4. Playback lê sempre do front-buffer com `fence(Acquire)`

#### 3.1 Arquivos Afetados

- `src/pw_host.rs` — struct `DspBridge` e lógica de sincronização dos callbacks

#### 3.1 Verificação

- Testes de integração com modelo WaveNet real (sem artefatos audíveis)
- Validar que a latência não aumentou (generation gap ≤ 1 buffer)

---

## Sprint 4: Pesquisa e Inovação (Backlog Avançado)

> Itens de investigação a longo prazo. Requerem prototipagem e benchmarking
> antes de serem promovidos a tarefas formais.

---

### Pesquisa 4.1: VNNI para Dot Products Int8

#### 4.1 Contexto

Quantização dos pesos LSTM/WaveNet para Int8 com acumulação em Int32 via `_mm512_dpbusd_epi32`
ou `_mm256_dpbusd_epi32`.

Ganho teórico: **4× throughput vs FP32 FMA**. O VNNI combina múltiplas multiplicações e adições em um único ciclo de clock de forma muito mais densa que o FMA3 padrão. Isso significa que a CPU faz muito menos força para entregar o mesmo bloco de áudio.

#### 4.1 Desafio Principal

Validar que a perda de precisão (FP32 → Int8) é aceitável para modelos NAM,
onde a fidelidade perceptual de áudio é crítica. Provavelmente requer:

- Calibração de escala por-camada (quantization-aware)
- Comparação A/B audível para cada modelo
- Threshold de SNR mínimo para aceitar quantização

#### 4.1 Próximos Passos

- Prototipar quantização dos pesos de um modelo LSTM 1×16
- Medir MSE e SNR vs referência FP32
- Avaliar se a calibração per-layer é suficiente ou se precisa per-tensor
- Ativado via linha de comando.
- Quantização Just-in-Time (JIT)? Ao carregar o arquivo .nam comum, arredondar os pesos de Float32 para inteiros durante o carregamento.

#### 4.1 VNNI também possui uma versão para x86-64-v3

- Intel: Qualquer processador a partir da 12ª Geração (Alder Lake, final de 2021). Inclui a série Core (i3, i5, i7, i9 com numeração 12000 ou superior) e as novas linhas Core Ultra.
- AMD: Qualquer processador baseado na arquitetura Zen 4 ou superior (lançados a partir do final de 2022). O marco principal é a linha Ryzen série 7000 para desktops.
- grep -E 'avx_vnni' /proc/cpuinfo
- std::is_x86_feature_detected!("avx_vnni")

---

### Pesquisa 4.2: Uso de Cgroup

Enquanto o `pthread_setaffinity_np` (afinidade) é como colocar uma placa de "Reservado" em uma mesa para a sua _thread_ de áudio, o **cgroups (Control Groups)**, especificamente na sua versão v2 unificada, é como construir uma sala à prova de som ao redor dessa mesa e trancar a porta por fora.

A afinidade diz para onde a sua _thread_ deve ir, mas não impede que o escalonador do Linux jogue um processo do navegador ou um atualizador do sistema naquele mesmo núcleo por alguns microssegundos. O `cgroups` resolve isso criando fronteiras físicas e lógicas no nível do kernel.

Para um projeto de DSP de baixíssima latência, aqui está o que pode ser feito estritamente via `cgroups`:

#### 1. Blindagem Absoluta de Núcleos (O Controlador `cpuset`)

Esta é a aplicação mais brutal e eficaz do `cgroups` para áudio. Você particiona os núcleos físicos do processador e proíbe o resto do sistema de enxergá-los.

- **Isolamento de Domínio (CPU Shielding):** Você pode criar um grupo (por exemplo, `system.slice`) e atribuir os núcleos `0,1,2,3` a ele. Todos os processos normais do SO rodarão apenas aí. Em seguida, cria um `audio.slice` com os núcleos `4,5` (os de maior capacidade/menor interrupção que você identificou).
- **Migração de Interrupções:** Ao isolar os núcleos via `cpuset`, o kernel automaticamente tenta afastar o processamento de interrupções de hardware (IRQs de rede, USB, vídeo) desses núcleos isolados, deixando-os em silêncio absoluto aguardando as amostras do PipeWire.
- **Desligamento do Balanceador de Carga:** Dentro do `cpuset` do seu grupo de áudio, você pode sinalizar ao kernel para desligar o _load balancer_ (`cpuset.sched_load_balance=0`). Como você já fixou a afinidade da sua _thread_ manualmente no código, o kernel para de gastar ciclos de clock tentando reavaliar se deve mover sua _thread_ para outro núcleo.

#### 2. Prevenção de Swap e Page Faults (O Controlador `memory`)

O maior inimigo da latência determinística não é a falta de CPU, mas a CPU ter que esperar a memória RAM. Se o sistema operacional decidir fazer paginação (_swap_) de um pedaço da memória do seu motor neural para o disco, o áudio vai estalar severamente.

- **Bloqueio de Swap (`memory.swap.max = 0`):** Você pode instruir o `cgroups` a definir um limite de zero _bytes_ de swap para o grupo onde o servidor de áudio e o cliente NAM estão rodando. Se faltar RAM, o sistema não jogará as matrizes neurais para o SSD; ele as manterá travadas na memória física (L1/L2/L3 e RAM).
- _Nota de Desenvolvimento:_ No código Rust/C++, isso trabalha em conjunto com chamadas de sistema como o `mlockall()` (Memory Lock), que tranca o espaço de endereçamento do processo na RAM, impedindo falhas de página (_page faults_) durante o processamento do _buffer_.

#### 3. Garantia de Tempo de Execução (O Controlador `cpu`)

Como você já utiliza o agendador de tempo real (`SCHED_FIFO` via `rtprio`), este controlador é menos crítico, mas atua como uma rede de segurança contra travamentos completos do sistema (o temido _RT throttling_).

- **Alocação de Banda Larga de CPU (`cpu.max` / `cpu.weight`):** Mesmo rodando em tempo real, se houver uma falha no seu código (um _loop_ infinito, por exemplo), uma _thread_ RT pode congelar a máquina inteira. O `cgroups` permite definir um teto (ex: permitir uso de 95% do tempo do núcleo, deixando 5% para que o kernel possa intervir e você consiga matar o processo no terminal, se necessário).

#### 4. Gestão de Banda de I/O (O Controlador `io`)

Embora a inferência do NAM seja estritamente matemática e dependa de CPU/RAM, a cadeia de áudio pode envolver a leitura de respostas de impulso (IRs) ou gravação de _takes_ em disco.

- **Priorização de Disco (`io.latency` / `io.weight`):** Caso o sistema esteja sob forte estresse de disco (ex: atualizando pacotes ou copiando arquivos grandes), você pode proteger o grupo de áudio garantindo que qualquer leitura de arquivo (como carregar dinamicamente um novo modelo `.nam` ou arquivo `.wav` de _cab sim_) tenha passagem livre pelo agendador de I/O do kernel, reduzindo o tempo de carregamento da interface.

#### Como isso se traduz na Prática (Via Systemd)

Em distribuições modernas (como Ubuntu), manipular os arquivos virtuais do `cgroups` manualmente em `/sys/fs/cgroup/` não é recomendado, pois o `systemd` gerencia a árvore unificada do cgroups v2.

A forma mais limpa de aplicar esse poder é encapsular a execução do seu binário em um serviço ou escopo do systemd. Por exemplo, executando o motor de áudio via comando com as diretivas do cgroups mapeadas:

```bash
systemd-run --user --scope \
    -p AllowedCPUs=4,5 \
    -p MemorySwapMax=0 \
    -p CPUSchedulingPolicy=fifo \
    -p CPUSchedulingPriority=85 \
    ./seu_binario_nam_rs
```

Esse comando cria um escopo (grupo de controle) efêmero que isola os núcleos, zera a paginação, aplica as políticas de tempo real e empacota o processo numa bolha blindada gerenciada nativamente pelo kernel, extraindo o limite do que o hardware de consumidor pode oferecer.
