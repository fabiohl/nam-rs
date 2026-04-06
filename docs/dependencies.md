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

## 2. Dependências Rust (Cargo.toml)

| Crate      | Propósito                                                                 |
| ---------- | ------------------------------------------------------------------------- |
| `anyhow`   | Tratamento de erros ergonômico em toda a aplicação                        |
| `ctrlc`    | Handler de sinal CTRL+C para shutdown gracioso                            |
| `libc`     | Chamadas de baixo nível: SCHED_FIFO, Core Affinity, pthread               |
| `pipewire` | Host PipeWire nativo: ThreadLoopBox, StreamBox, Context e callbacks de RT |
| `rtrb`     | SPSC Ring Buffer lock-free para comunicação CLI → DSP (Tarefa 1.2)        |

---

## 3. Máquina do Usuário Final (Runtime)

O binário final extraído em modo `--release` encapsula a matemática lógica, mas invoca conexões assíncronas do Host PipeWire no runtime final.
As dependências restritas do lado do usuário configuram em:

```bash
sudo apt install pipewire
```

## 4. Políticas de Baixa Latência

Para que o motor execute inabalável sob modelos NAM severos ("Standard"), é fundamental conceder a autorização de políticas SCHED mais avançadas ao binário.
Adicione seu usuário ao grupo de áudio do sistema e edite limits:

1. `sudo usermod -aG audio $USER`
2. Crie ou edite o arquivo de limites (ex: `sudo nano /etc/security/limits.d/audio.conf`):

```text
@audio   -  rtprio     95
@audio   -  memlock    unlimited
```
