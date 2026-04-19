# TODO-sprints

Obs: Algumas coisas aqui devem ficar pra versões 1.x - identificar claramente.

---

## Sprint 1

### Tarefa 1.1: Fused Gate GEMV para LSTM + Prefetch Preditivo [Testando]

Estimativa: ~6h | Complexidade: Alta | Ganho: −20–30% latência LSTM/sample (×2 com True Stereo)

**Testando:** "Optimizing LSTM Performance via GEMV"
target/release/nam-rs -m github.com/mikeoliphant/NeuralAudio/Utils/Models/BossLSTM-1x16.nam
target/release/nam-rs -m github.com/mikeoliphant/NeuralAudio/Utils/Models/BossWN-standard.nam

#### Motivação

O hot-path do LSTM executa H4 dot products independentes em série (um por porta),
onde cada invocação de `dot_product` tem overhead fixo de setup (~3 inst.) e reduce
(~8 inst.). Para H=16 (H4=64, IH=17): 1088 FMAs sequenciais limitadas por
latência, não throughput.

**Contexto atualizado (True Stereo):** Desde a implementação de True Stereo em
`pw_host.rs` (linhas 480–493), o callback RT executa `model_l.process()` +
`model_r.process()` sequencialmente. Para modelos LSTM 2×16, isso resulta em
**2× o número de dot products por buffer**, tornando o GEMV duplamente impactante.

O Silence Bypass (`is_buffer_silent_stereo_simd`) mitiga a carga apenas quando não
há sinal — durante uso ativo (músico tocando), a carga LSTM é integral.

#### Proposta Técnica

Processar 4 linhas da matriz de pesos simultaneamente aproveitando que o vetor
`state` é compartilhado entre as 4 portas (Input, Forget, Cell, Output):

- Carregar `state` uma vez, reutilizar 4× → reduz bandwidth de leitura de 4 IH para 1 IH
- 4 FMAs independentes → ILP total no pipeline (4 ciclos de latência × 4 acumuladores)
- H4 é sempre múltiplo de 4 → nenhum remainder

#### **Sub-item: Prefetch Preditivo do State Vector**

Para modelos LSTM 2×16 (IH=32, ou seja 128 bytes = 2 cache lines de 64B), emitir
`_mm_prefetch(_MM_HINT_T0)` sobre `self.state[0..IH]` **antes** do loop GEMV.
Isto garante que ambas cache lines estejam quentes em L1 antes da primeira FMA,
eliminando stalls de cache miss no primeiro tap.

Implementação sugerida:

```rust
// Antes do loop GEMV, prefetch do state (IH floats = IH×4 bytes)
// Para IH=17 (68 bytes): 2 cache lines (0..63, 64..67)
unsafe {
    _mm_prefetch::<{ _MM_HINT_T0 }>(self.state.as_ptr().cast::<i8>());
    if IH > 16 {
        _mm_prefetch::<{ _MM_HINT_T0 }>(self.state.as_ptr().add(16).cast::<i8>());
    }
}
```

#### Arquivos Afetados

- `src/models/lstm.rs` — macro `define_lstm_process!` (linhas 35–140)
  - Substituir o loop `for i in 0..H4` (linhas 57–60) pelo GEMV fused de 4 portas
  - Adicionar prefetch do `self.state` antes do loop
- `src/models/lstm_dyn.rs` — `LstmDynLayer::process_sample()` (linhas 41–92)
  - O fallback dinâmico faz o mesmo padrão H4-serial (linhas 50–55) — aplicar
    a mesma otimização, adaptada para dimensões dinâmicas
- `src/loader/dispatcher.rs` — `read_lstm_layer()` (linhas 623–647)
  - A transposição dos pesos para layout intercalado de 4 portas deve ocorrer aqui,
    na thread Main, preservando zero-alloc no RT
  - Layout atual: `input_hidden_weights[H4][IH]` — row-major por porta
  - Layout novo: intercalar linhas i, i+H, i+2H, i+3H para leitura contígua de 4 portas

#### Pré-Requisitos

- Transposição dos pesos na construção (thread Main) para layout intercalado de 4
  portas (dot contíguo, custo pré-pago no loader, sem impacto RT)
- Manter backward compatibility com formatos `.nam`/`.namb` existentes
  (a transposição é interna ao dispatcher, transparente ao formato de arquivo)

#### Riscos

- Mudança no layout SoA dos pesos afeta o dispatcher e o struct `LstmLayer`
- Requer ajuste nos testes golden e de paridade estático↔dinâmico em
  `tests/nam_infer_test.rs` (função `assert_dsp_fidelity` com MSE + SNR)
- Deve ser executado em **BRANCH ISOLADO** com revisão cuidadosa

#### Verificação

- `cargo test` (todos os golden tests devem manter MSE ≤ threshold em
  `tests/nam_infer_test.rs`)
- `cargo bench --bench inference_bench` (confirmar ganho na latência LSTM)
- Comparação A/B audível com modelo LSTM real (ex: `tests/Neve31102-Standard.nam`)
- Auto-consistência: 2 runs do mesmo modelo devem produzir MSE = 0.0 (bitwise)

---

### Tarefa 1.2: Tanh Padé Grau 7 com Clamp Saturante

Estimativa: ~3h | Complexidade: Média | Ganho: −40% erro MSE golden

#### 1.2 Motivação

O `simd_tanh` atual (em `src/math/fastmath.rs`, linhas 53–98) usa Padé grau 5 com
erro máximo ~5e-3, que acumula ~√20 × 5e-3 ≈ 2.2e-2 por 20 camadas WaveNet.
Um grau 7 reduziria o MSE golden para ~1e-2 com apenas +2 ciclos/tanh.

O MSE golden medido (3.21e-2 em 2026-04-15) é consistente com o threshold `5e-2`
documentado na docstring de `simd_tanh` e nos testes de fidelidade DSP.

> **Nota de prioridade:** O Silence Bypass reduz a frequência de invocações de tanh
> (pula inferência quando não há sinal). O ganho de precisão é real mas incremental
> (~40% MSE). Recomenda-se executar a Tarefa 1.1 primeiro e reavaliar com benchmarks
> A/B se a fidelidade numérica é issue perceptível.

#### 1.2 Proposta Técnica

1. Adicionar coeficiente x^7 ao polinômio: `c2` (derivado via Minimax Remez)
2. Clamp saturante: |x| > 4.97 → ±1.0 (tanh(4.97) = 0.999988)
3. Blend via `_mm256_blendv_ps` entre resultado polinomial e ±1.0

#### 1.2 Pré-Requisitos

- Derivar coeficiente `c2` via Minimax Remez Exchange sobre [-5, 5]
- Ajustar thresholds de aceitação nos testes correspondentes

#### 1.2 Arquivos Afetados

- `src/math/fastmath.rs` — função `simd_tanh` (AVX2, linhas 53–98)
- `src/math/fastmath.rs` — função `simd_tanh_avx512` (AVX-512, linhas 130–172)
- `src/math/fastmath.rs` — teste `test_simd_fastmath_tanh_mse` (linhas 289–312)
  - Threshold atual: `5e-3` — estreitar para ~5e-4 após grau 7
- `tests/nam_infer_test.rs` — testes golden com `assert_dsp_fidelity` (MSE + SNR)
  - Os golden vectors estão neste arquivo, **não** em `fastmath.rs`
  - Threshold de MSE e SNR mínimo devem ser apertados proporcionalmente

#### 1.2 Verificação

- `test_simd_fastmath_tanh_mse` com threshold estreitado para ~5e-4
- `cargo bench -- tanh` (confirmar custo adicional ≤ 2 ciclos/invocação)
- Golden tests em `tests/nam_infer_test.rs` devem melhorar MSE sem violar SNR

---

### Tarefa 1.3: Detecção de Mono via AVX2 (Mono Fast-Path)

Estimativa: ~2h | Complexidade: Baixa | Ganho: ~5ns detecção + código mais limpo

#### 1.3 Motivação

O callback RT de captura em `pw_host.rs` (linhas 446–452) já detecta se o canal
direito é puramente zero ou idêntico ao esquerdo para economizar 50% de CPU
(pula `model_r.process()`). Porém, a detecção atual é **escalar** — um loop
sample-by-sample com branch por amostra:

```rust
let mut process_mono = true;
for i in 0..n_samples {
    if samples_r[i] != 0.0 && samples_r[i] != samples_l[i] {
        process_mono = false;
        break;
    }
}
```

Para 128 samples, isso resulta em até 128 comparações escalares com branch.

#### 1.3 Proposta Técnica

Implementar `is_buffer_mono_simd(left: &[f32], right: &[f32]) -> bool` em
`src/dsp/gain.rs`, usando o mesmo padrão de `is_buffer_silent_stereo_simd`:

1. `_mm256_loadu_ps` — carregar 8 samples de L e R
2. `_mm256_cmp_ps(abs_r, zero, _CMP_NEQ_OQ)` — R ≠ 0?
3. `_mm256_cmp_ps(r, l, _CMP_NEQ_OQ)` — R ≠ L?
4. `_mm256_and_ps(cmp_nz, cmp_ne)` — R ≠ 0 **e** R ≠ L?
5. `_mm256_or_ps(accum, result)` — acumular
6. `_mm256_movemask_ps` — early-exit se algum lane é true (não é mono)

Custo: ~5ns para 128 samples (16 iterações AVX2) vs loop escalar atual.

#### 1.3 Arquivos Afetados

- `src/dsp/gain.rs` — adicionar função `is_buffer_mono_simd()`
  - Seguir o padrão de `is_buffer_silent_stereo_simd` (linhas 126–164)
  - Incluir tail escalar para buffers não-múltiplos de 8
- `src/pw_host.rs` — substituir loop escalar (linhas 446–452) pela chamada
  `is_buffer_mono_simd(&samples_l[..n_samples], &samples_r[..n_samples])`
- Testes unitários em `src/dsp/gain.rs`:
  - Buffer R=zeros → mono=true
  - Buffer R=L (bitwise) → mono=true
  - Buffer R≠L em sample 64 → mono=false
  - Buffer R=zeros exceto último sample (tail escalar) → mono=false

#### 1.3 Pré-Requisitos

Nenhum — ortogonal às outras tarefas.

#### 1.3 Riscos

Baixos — é uma otimização isolada de detecção, sem impacto na fidelidade do áudio.

#### 1.3 Verificação

- `cargo test` — novos testes unitários + testes existentes inalterados
- Validar em runtime com modelo WaveNet mono (output sem mudança audível)

---

## Sprint 2 (Baixa Prioridade)

### Tarefa 2.1: DspBridge com Double-Buffer

Estimativa: ~4h | Complexidade: Média | Ganho: Robustez contra futuras mudanças

> Pode ser combinada com outras tarefas de baixa prioridade durante
> refatorações periódicas.

#### 2.1 Motivação

O `DspBridge` atual em `pw_host.rs` (linhas 72–82) usa um **único buffer** com
generation counter atômico. Na implementação atual, ambos os callbacks rodam no
mesmo `ThreadLoop` PW (`node.group = "nam-rs-dsp"`) e o `fence(Release)` +
`generation.fetch_add` só é emitido **após** ambos os canais L e R estarem
escritos (linhas 536–544). Portanto, **não há bug atual**.

Porém, um double-buffer com swap atômico seria mais robusto contra:

- Futuras mudanças na topologia de streams PW
- Cenários onde os callbacks possam executar em threads diferentes
- Simplificação da lógica de sincronização (elimina `n_samples == 0` guard)

#### 2.1 Proposta Técnica

1. Alocar 2 instâncias de buffer (front/back) via `Box::leak`
2. Usar `AtomicBool` ou `AtomicUsize` para indexar qual buffer é o "ativo para leitura"
3. Capture escreve sempre no back-buffer; ao final, swapa o índice com `fence(Release)`
4. Playback lê sempre do front-buffer com `fence(Acquire)`

#### 2.1 Arquivos Afetados

- `src/pw_host.rs` — struct `DspBridge` e lógica de sincronização dos callbacks

#### 2.1 Verificação

- Testes de integração com modelo WaveNet real (sem artefatos audíveis)
- Validar que a latência não aumentou (generation gap ≤ 1 buffer)

---

## Sprint 3 (Pesquisa e Inovação)

> Itens de investigação a longo prazo. Requerem prototipagem e benchmarking
> antes de serem promovidos a tarefas formais.

### Pesquisa 3.1: Batch Processing WaveNet

#### Contexto

O WaveNet atual processa **1 sample por vez** (loop `for i in 0..num_frames` em
`src/models/wavenet.rs`, linhas 489–510). Para modelos WaveNet Standard (CH=16,
K=3), cada `conv1d.process_frame()` executa `OUT × K = 48` dot products de 16
floats.

#### Ideia

Se conseguirmos processar N samples em batch (acumulando no ring buffer antes de
despachar), os dot products poderiam ser convertidos em small-GEMM com melhor
reuso de cache line.

#### Desafio Principal

A natureza **causal/autoregressive** do WaveNet limita o paralelismo temporal:
cada sample depende do anterior via residual connections e estado do ring buffer.
Pesquisar se há "look-ahead" possível nas residual connections ou se as camadas
independentes entre arrays podem ser batched.

#### Próximos Passos

- Estudar o paper original (van den Oord et al., 2016) para identificar
  dependências exatas entre samples
- Prototipar batch de 2-4 samples e medir impacto em latência e cache miss rate
- Avaliar se o overhead de acumulação no ring buffer compensa o ganho do GEMM

---

### Pesquisa 3.2: VNNI AVX-2 para Dot Products Int8

#### 3.1 Contexto

Quantização dos pesos LSTM para Int8 com acumulação em Int32 via `_mm512_dpbusd_epi32`
ou `_mm256_dpbusd_epi32`.

Ganho teórico: **4× throughput vs FP32 FMA**. O VNNI combina múltiplas multiplicações e adições em um único ciclo de clock de forma muito mais densa que o FMA3 padrão. Isso significa que a CPU faz muito menos força para entregar o mesmo bloco de áudio.

#### 3.2 Desafio Principal

Validar que a perda de precisão (FP32 → Int8) é aceitável para modelos NAM,
onde a fidelidade perceptual de áudio é crítica. Provavelmente requer:

- Calibração de escala por-camada (quantization-aware)
- Comparação A/B audível para cada modelo
- Threshold de SNR mínimo para aceitar quantização

#### 3.2 Próximos Passos

- Prototipar quantização dos pesos de um modelo LSTM 1×16
- Medir MSE e SNR vs referência FP32
- Avaliar se a calibração per-layer é suficiente ou se precisa per-tensor
- Ativado via linha de comando.
- Quantização Just-in-Time (JIT)? Ao carregar o arquivo .nam comum, arredondar os pesos de Float32 para inteiros durante o carregamento.

#### 3.4 VNNI também possui uma versão para x86-64-v3

- Intel: Qualquer processador a partir da 12ª Geração (Alder Lake, final de 2021). Inclui a série Core (i3, i5, i7, i9 com numeração 12000 ou superior) e as novas linhas Core Ultra.
- AMD: Qualquer processador baseado na arquitetura Zen 4 ou superior (lançados a partir do final de 2022). O marco principal é a linha Ryzen série 7000 para desktops.
- grep -E 'avx_vnni' /proc/cpuinfo
- std::is_x86_feature_detected!("avx_vnni")

---

### Pesquisa 3.3: Multiversionamento Explícito do WaveNet `process()`

#### 3.3 Contexto

Atualmente só o LSTM tem versões explícitas `process_sample_avx2` /
`process_sample_avx512` (via macro `define_lstm_process!`). O WaveNet usa a
v-table `SimdMathConfig` (em `src/math/simd.rs`) para dispatch individual de
`dot_product` e `tanh_slice`, mas o `process()` em si não tem
`#[target_feature]`.

#### 3.3 Ideia

Adicionar uma versão `process_avx512()` com `_mm512_*` hardcoded no loop interno
do WaveNet, eliminando o overhead de indireção via function pointers nos dot
products e tanh/sigmoid slices. O compilador poderia então otimizar o loop
completo como uma unidade, potencialmente beneficiando-se de register allocation
ZMM mais agressiva.

#### 3.3 Próximos Passos

- Medir o overhead real da indireção via `SimdMathConfig` function pointers
  (provavelmente < 1% — validar antes de implementar)
- Se o overhead for significativo, criar macro similar a `define_lstm_process!`
  para o loop `WaveNetLayerArray::process()`
- Avaliar se `#[target_feature(enable = "avx512f")]` no `process()` é suficiente
  ou se precisa de intrinsics explícitas

---

### Pesquisa 3.4: Uso de Cgroup

Enquanto o `pthread_setaffinity_np` (afinidade) é como colocar uma placa de "Reservado" em uma mesa para a sua *thread* de áudio, o **cgroups (Control Groups)**, especificamente na sua versão v2 unificada, é como construir uma sala à prova de som ao redor dessa mesa e trancar a porta por fora.

A afinidade diz para onde a sua *thread* deve ir, mas não impede que o escalonador do Linux jogue um processo do navegador ou um atualizador do sistema naquele mesmo núcleo por alguns microssegundos. O `cgroups` resolve isso criando fronteiras físicas e lógicas no nível do kernel.

Para um projeto de DSP de baixíssima latência, aqui está o que pode ser feito estritamente via `cgroups`:

#### 1. Blindagem Absoluta de Núcleos (O Controlador `cpuset`)

Esta é a aplicação mais brutal e eficaz do `cgroups` para áudio. Você particiona os núcleos físicos do processador e proíbe o resto do sistema de enxergá-los.

- **Isolamento de Domínio (CPU Shielding):** Você pode criar um grupo (por exemplo, `system.slice`) e atribuir os núcleos `0,1,2,3` a ele. Todos os processos normais do SO rodarão apenas aí. Em seguida, cria um `audio.slice` com os núcleos `4,5` (os de maior capacidade/menor interrupção que você identificou).
- **Migração de Interrupções:** Ao isolar os núcleos via `cpuset`, o kernel automaticamente tenta afastar o processamento de interrupções de hardware (IRQs de rede, USB, vídeo) desses núcleos isolados, deixando-os em silêncio absoluto aguardando as amostras do PipeWire.
- **Desligamento do Balanceador de Carga:** Dentro do `cpuset` do seu grupo de áudio, você pode sinalizar ao kernel para desligar o *load balancer* (`cpuset.sched_load_balance=0`). Como você já fixou a afinidade da sua *thread* manualmente no código, o kernel para de gastar ciclos de clock tentando reavaliar se deve mover sua *thread* para outro núcleo.

#### 2. Prevenção de Swap e Page Faults (O Controlador `memory`)

O maior inimigo da latência determinística não é a falta de CPU, mas a CPU ter que esperar a memória RAM. Se o sistema operacional decidir fazer paginação (*swap*) de um pedaço da memória do seu motor neural para o disco, o áudio vai estalar severamente.

- **Bloqueio de Swap (`memory.swap.max = 0`):** Você pode instruir o `cgroups` a definir um limite de zero *bytes* de swap para o grupo onde o servidor de áudio e o cliente NAM estão rodando. Se faltar RAM, o sistema não jogará as matrizes neurais para o SSD; ele as manterá travadas na memória física (L1/L2/L3 e RAM).
- *Nota de Desenvolvimento:* No código Rust/C++, isso trabalha em conjunto com chamadas de sistema como o `mlockall()` (Memory Lock), que tranca o espaço de endereçamento do processo na RAM, impedindo falhas de página (*page faults*) durante o processamento do *buffer*.

#### 3. Garantia de Tempo de Execução (O Controlador `cpu`)

Como você já utiliza o agendador de tempo real (`SCHED_FIFO` via `rtprio`), este controlador é menos crítico, mas atua como uma rede de segurança contra travamentos completos do sistema (o temido *RT throttling*).

- **Alocação de Banda Larga de CPU (`cpu.max` / `cpu.weight`):** Mesmo rodando em tempo real, se houver uma falha no seu código (um *loop* infinito, por exemplo), uma *thread* RT pode congelar a máquina inteira. O `cgroups` permite definir um teto (ex: permitir uso de 95% do tempo do núcleo, deixando 5% para que o kernel possa intervir e você consiga matar o processo no terminal, se necessário).

#### 4. Gestão de Banda de I/O (O Controlador `io`)

Embora a inferência do NAM seja estritamente matemática e dependa de CPU/RAM, a cadeia de áudio pode envolver a leitura de respostas de impulso (IRs) ou gravação de *takes* em disco.

- **Priorização de Disco (`io.latency` / `io.weight`):** Caso o sistema esteja sob forte estresse de disco (ex: atualizando pacotes ou copiando arquivos grandes), você pode proteger o grupo de áudio garantindo que qualquer leitura de arquivo (como carregar dinamicamente um novo modelo `.nam` ou arquivo `.wav` de *cab sim*) tenha passagem livre pelo agendador de I/O do kernel, reduzindo o tempo de carregamento da interface.

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
