# Instalação NAM-rs: Requisitos Sistêmicos e Dependências (APT)

Este documento centraliza as dependências requeridas para a infraestrutura avançada do motor de inferência algorítmica `NAM-rs`. Diferente de plug-ins universais convencionais, as APIs Standalone demandam bibliotecas hospedeiras presentes.

---

## 1. Ferramentas de Compilação (Build Runtime)

A compilação local foca na extração microarquitetural e ponteiros de kernel para injetar instâncias C estritas da matriz de áudio. É imprescindível injetar o ecossistema C/C++ padrão e dependências PipeWire-RS no SO.

```bash
sudo apt install build-essential pkg-config pipewire libpipewire-0.3-dev clang libclang-dev
```

*Por que usar o `clang`?* Exigido para compatibilidade irrestrita pelo `rust-bindgen` a fim de extrair as referências dinâmicas limpas da matriz PipeWire em ambiente de compilação local (build-time).

---

## 2. Dependências Rust (Cargo.toml) — Estado Atual (pós-Sprint 2)

| Crate      | Versão   | Propósito                                                                 |
| ---------- | -------- | ------------------------------------------------------------------------- |
| `anyhow`   | 1.0.102  | Tratamento de erros ergonômico em toda a aplicação                        |
| `ctrlc`    | 3.5.2    | Handler de sinal CTRL+C para shutdown gracioso                            |
| `libc`     | 0.2.184  | Chamadas de baixo nível: SCHED_FIFO, Core Affinity, pthread               |
| `pipewire` | 0.9.2    | Host PipeWire nativo: ThreadLoopBox, StreamBox, Context e callbacks de RT |
| `rtrb`     | 0.3.3    | SPSC Ring Buffer lock-free para comunicação CLI → DSP                     |

**Total: 5 dependências diretas** — mínimo necessário para a fundação PipeWire + tempo real.

### Dependências Futuras Previstas

| Crate         | Sprint | Propósito                                                    |
| ------------- | ------ | ------------------------------------------------------------ |
| `serde`       | 4      | Framework de serialização para parsing do formato `.nam`     |
| `serde_json`  | 4      | Deserialização JSON do formato `.nam`                        |
| `crc32fast`   | 4      | Verificação CRC32 IEEE 802.3 do formato binário `.namb`      |
| `rubato`      | 5      | Conversão de sample rate FIR Sinc polifásico (fase linear)   |

---

## 3. Máquina do Usuário Final (Runtime)

O binário final extraído em modo `--release` encapsula a matemática lógica, mas invoca conexões assíncronas do Host PipeWire no runtime final.
As dependências restritas do lado do usuário configuram em:

```bash
sudo apt install pipewire
```

**Requisito de CPU:** Processador x86-64 com suporte nativo a AVX2 e FMA (Intel ≥ Haswell 2013, AMD ≥ Excavator 2015). Verificado no startup do aplicativo.

## 4. Políticas de Baixa Latência

Para que o motor execute inabalável sob modelos NAM severos ("Standard"), é fundamental conceder a autorização de políticas SCHED mais avançadas ao binário.
Adicione seu usuário ao grupo de áudio do sistema e edite limits:

1. `sudo usermod -aG audio $USER`
2. Crie ou edite o arquivo de limites (ex: `sudo nano /etc/security/limits.d/audio.conf`):

```text
@audio   -  rtprio     95
@audio   -  memlock    unlimited
```

## 5. Flags de Compilação

O projeto é compilado com `target-cpu` otimizado para a microarquitetura `x86-64-v3`. Isso garante que o compilador LLVM emita instruções AVX2/FMA para a CPU do build.
