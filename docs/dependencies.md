# AudioRip: Requisitos de Sistema e Dependências (APT)

Este documento centraliza as dependências de sistema exigidas para o desenvolvimento e também para a execução final do **AudioRip** na máquina do usuário.

Devido à integração com a crate `nih_plug` em modo `standalone` e às interfaces nativas de Kernel (`tokio-uring`, `alsa`), alguns pacotes do sistema operacional Ubuntu/Debian são obrigatórios.

---

## 1. Ambiente de Desenvolvimento (Build)

Para que um desenvolvedor consiga baixar o código-fonte e compilar o projeto com `cargo build`, é necessário instalar as bibliotecas de desenvolvimento de C/C++ vinculadas (headers).

Execute o seguinte comando em distribuições Debian-based:

```bash
sudo apt install build-essential pkg-config libasound2-dev libx11-xcb-dev pipewire-jack qpwgraph
```

### Por que pacotes gráficos (X11) estão inclusos num app CLI?

O `libx11-xcb-dev` é puxado porque usamos o ecossistema `nih_plug` com a feature `standalone` habilitada. Essa feature embute capacidades gráficas padronizadas (através do submódulo *baseview*) para o caso de interfaces de áudio VST.
Como o *AudioRip* é agnóstico a painéis GUI (e atua via linha de comando *headless/cli*), essa dependência ocorre unicamente em **nível de compilação/linkagem**.

**Wayland vs X11**: Devido a isso, distribuições modernas que rodam exclusivamente debaixo do display server **Wayland** não terão problemas. A linkagem não exige uma sessão ativa do X11; o binário rodará perfeitamente via emulação nativa no terminal (ou no máximo dependerá do `Xwayland` abstraído no sistema host se alguma janela tentasse ser invocada).

---

## 2. Máquina do Usuário Final (Runtime)

O binário final compilado em modo `--release` já costuma absorver quase todo o comportamento do projeto numa tipologia *statically-linked* na ótica do Rust. Contudo, dependências dinâmicas relacionadas ao driver de áudio e às bibliotecas base da libc precisam estar instaladas.

Por via de regra padrão, na grande maioria das distribuições Linux de desktop (*Ubuntu, Mint, Pop!_OS, Fedora*), estas dependências sistêmicas **já vêm pré-instaladas de fábrica**. Caso ocorra alguma falha de `cannot open shared object file: No such file or directory` ao rodar `./audiorip`, o usuário precisará garantir as seguintes *shared libraries*:

```bash
sudo apt install libasound2 libx11-xcb1 pipewire-jack qpwgraph
```

O *AudioRip* comunicará ativamente com a matriz do PipeWire, que atua integrando o servidor de som e interpretando os metadados dinamicamente via ALSA plugin chain sem corromper o audio (`Bit Perfect`).

## 3. Para usar thread de baixa latência com SCHED_FIFO

Por padrão o kernel bloqueia a elevação de prioridade de baixa latência para usuários comuns.

Você libera a permissão de baixa latência no sistema operacional para o seu usuário. É a prática recomendada e o padrão da indústria para quem desenvolve ou trabalha com áudio profissional no Linux.

* **Comandos:**
  1. Adicione seu usuário ao grupo de áudio do sistema:
     `sudo usermod -aG audio $USER`
  2. Crie ou edite o arquivo de limites (ex: `sudo nano /etc/security/limits.d/audio.conf`) e adicione as seguintes linhas:

```text
@audio   -  rtprio     95
@audio   -  memlock    unlimited
```
