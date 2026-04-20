# TODO-sprints

## Sprint 4: Pesquisa e Inovação (Backlog Avançado)

---

### Pesquisa 4.1: VNNI para Dot Products Int8 [To Do]

#### 4.1 Contexto

Quantização dos pesos LSTM/WaveNet para Int8 com acumulação em Int32 via `_mm512_dpbusd_epi32` ou `_mm256_dpbusd_epi32`.

Ganho teórico: **4× throughput vs FP32 FMA**. O VNNI combina múltiplas multiplicações e adições em um único ciclo de clock de forma muito mais densa que o FMA3 padrão. Isso significa que a CPU faz muito menos força para entregar o mesmo bloco de áudio.

#### 4.1 Desafio Principal

Validar que a perda de precisão (FP32 → Int8) é aceitável para modelos NAM, onde a fidelidade perceptual de áudio é crítica. Provavelmente requer:

- Calibração de escala por-camada (quantization-aware)
- Comparação A/B audível para cada modelo
- Threshold de SNR mínimo para aceitar quantização

#### 4.1 Passos

- Ativação via linha de comando
- Prototipar quantização dos pesos
- Medir MSE e SNR vs referência FP32
- Avaliar se a calibração per-layer é suficiente ou se precisa per-tensor
- Quantização Just-in-Time (JIT)? Ao carregar o arquivo .nam comum, arredondar os pesos de Float32 para inteiros durante o carregamento.

#### 4.1 VNNI também possui uma versão para x86-64-v3

- Intel: Qualquer processador a partir da 12ª Geração (Alder Lake, final de 2021). Inclui a série Core (i3, i5, i7, i9 com numeração 12000 ou superior) e as novas linhas Core Ultra.
- AMD: Qualquer processador baseado na arquitetura Zen 4 ou superior (lançados a partir do final de 2022). O marco principal é a linha Ryzen série 7000 para desktops.
- grep -E 'avx_vnni' /proc/cpuinfo
- Multiversioning via std::is_x86_feature_detected!("avx_vnni")

---

### Tarefa 4.2: A "Bala de Prata" em Código: `/dev/cpu_dma_latency` [Concluída]

Os processadores modernos tentam "dormir" (entrar em C-States profundos, como C6/C7) para economizar energia, e o tempo que levam para "acordar" causa estalos no áudio.

O Linux permite que um processo em *user space* diga ao kernel: *"Eu exijo que a latência de despertar da CPU seja de **Zero** milissegundos"*. Enquanto o seu programa mantiver essa solicitação aberta, o kernel **proibirá fisicamente a CPU de entrar em estados de dormência**, mantendo os transistores alimentados e prontos para o PipeWire.

### Como implementar no Rust

Você só precisa abrir o arquivo `/dev/cpu_dma_latency`, escrever um inteiro `0` de 32 bits nele, e **manter o arquivo aberto** (segurar o *File Descriptor*) durante a execução do programa. Quando o NAM-rs for fechado, o arquivo é descartado e o kernel devolve a CPU ao modo de economia de bateria normal. Zero scripts externos.

Adicione esta função e chame-a depois do carregamento bem-sucessido, mas antesde iniciar o processamento em si:

```rust
use std::fs::OpenOptions;
use std::io::Write;

/// Impede que o processador entre em C-States de economia de energia,
/// garantindo latência de despertar de 0ms para processamento de áudio RT.
///
/// RETORNO: O arquivo `File`. Ele DEVE ser mantido vivo no escopo principal.
/// Se for "dropado", o kernel anula a proteção.
pub fn lock_cpu_c_states() -> Option<std::fs::File> {
    match OpenOptions::new().write(true).open("/dev/cpu_dma_latency") {
        Ok(mut file) => {
            let zero: i32 = 0;
            if file.write_all(&zero.to_ne_bytes()).is_ok() {
                log::info!("⚡ PM QoS Lock: C-States profundos da CPU desativados (Zero DMA Latency).");
                return Some(file);
            }
            log::warn!("PM QoS: Falha ao escrever em /dev/cpu_dma_latency.");
            None
        }
        Err(e) => {
            log::warn!(
                "PM QoS: Acesso negado a /dev/cpu_dma_latency ({}). \
                 Considere criar uma regra de udev para o grupo 'audio'.", 
                e
            );
            None
        }
    }
}
```

Obs: Cuidado para este estado não continuar após o encerramento do NAM-rs.

### O complemento no Sistema Operacional (A Regra do Udev)

Assim como o `SCHED_FIFO` precisou do arquivo em `limits.d` para funcionar sem root, a abertura deste arquivo exige privilégios. Para que o seu usuário comum consiga aplicar essa trava no processador pelo próprio código, basta delegar a permissão via uma regra do `udev` para o grupo `audio`.

Crie um arquivo em `/etc/udev/rules.d/99-audio-dma-latency.rules` contendo:

```text
KERNEL=="cpu_dma_latency", GROUP="audio", MODE="0664"
```

*(Reinicie a máquina ou rode `sudo udevadm control --reload-rules && sudo udevadm trigger` para aplicar).*

Com isso, o NAM-rs assume o controle total do envelope de energia da CPU de forma programática.

> **Nota (Conclusão da Tarefa 4.2):** A implementação da trava de DMA (`lock_cpu_c_states`), da proteção de páginas de memória (`madvise`) e da nomeação de thread DSP (`pthread_setname_np`) foi efetuada no arquivo `src/pw_host.rs`. As permissões para grupo `audio` no `/etc/udev/rules.d/99-audio-dma-latency.rules` devem ser conferidas pelo usuário. A arquitetura de áudio RT PipeWire agora restringe completamente o retorno de latências derivadas de states ociosos (C-states), impedindo drop-outs na operação em low-latency.

### Ajustes Menores de *Hardening* em C/Rust

Além da gestão de energia, existem mais duas diretivas que você pode inserir junto à sua função `configure_realtime_thread` para "apertar" ainda mais a segurança:

**1. Blindagem de Memória Avançada (`madvise`):**
Você já vai colocar o `mlockall` para evitar swap. Para complementar, você pode avisar o kernel para **não clonar** as páginas de memória do seu áudio se o processo sofrer um *fork*, e para não fazer *core dumps* dessa região.

```rust
#[cfg(target_os = "linux")]
unsafe {
    // Substitua `buffer_ptr` e `tamanho` pelos ponteiros do seu DspBridge.
    // libc::madvise(buffer_ptr, tamanho, libc::MADV_DONTFORK | libc::MADV_DONTDUMP);
}
```

Isso é opcional, mas evita que engasgos ocultos do SO (como serviços do sistema tentando ler a memória) travem a *thread* DSP.

**2. Nomeação Explicita da Thread (`pthread_setname_np`):**
O PipeWire cria a *Data Thread* nos bastidores. Dar um nome a ela via código não melhora a latência, mas é fundamental para você monitorar se o *Core Affinity* realmente funcionou quando rodar ferramentas como `htop` ou `perf` no terminal do Linux.

```rust
#[cfg(target_os = "linux")]
unsafe {
    let name = b"nam_rs_dsp\0";
    libc::pthread_setname_np(libc::pthread_self(), name.as_ptr() as *const libc::c_char);
}
```
