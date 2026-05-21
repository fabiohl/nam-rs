<!-- SPDX-License-Identifier: Apache-2.0 -->

<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved. -->

# Estratégia de Integração CLAP (Clever Audio Plug-in)

Este documento descreve a arquitetura e a estratégia para transformar o motor DSP do NAM-rs em um plugin de áudio compatível com o padrão CLAP.

## 1. Thread Model

A integração CLAP deve respeitar estritamente a segregação de threads já existente no NAM-rs, mapeando-as para o modelo do host (DAW):

- **Main Thread (Host)**:
  - Responsável pela inicialização do plugin, escaneamento de parâmetros e gerenciamento de estado.
  - No NAM-rs, esta thread substituirá o loop principal em [src/main.rs](file:///home/fabio/nam-rs/src/main.rs).
  - Gerencia o carregamento de arquivos `.nam`/`.namb` via [src/loader/](file:///home/fabio/nam-rs/src/loader/).
- **Audio Thread (Real-time)**:
  - Chamada pelo host via callback `process()`.
  - **Requisito Crítico**: Deve manter a política de **ZERO alocações** e **ZERO locks**.
  - Utilizará [src/dsp/pipeline.rs](file:///home/fabio/nam-rs/src/dsp/pipeline.rs) para o processamento, adaptando os buffers do CLAP para o formato interno.
  - Diferente do PipeWire (que é dual-stream), o CLAP fornece buffers de entrada e saída em um único contexto, eliminando a necessidade do `DspBridge`.

## 2. Mapeamento de Parâmetros

Os parâmetros expostos ao host serão mapeados a partir da estrutura `NamPluginParams` (veja [src/common/params.rs](file:///home/fabio/nam-rs/src/common/params.rs)):

| Parâmetro CLAP     | ID                  | Unidade | Descrição                                        |
|:------------------ |:------------------- |:------- |:------------------------------------------------ |
| **Input Gain**     | `input_gain_db`     | dB      | Ganho aplicado antes da inferência neural.       |
| **Output Gain**    | `output_gain_db`    | dB      | Ganho aplicado após a inferência neural.         |
| **Gate Threshold** | `gate_threshold_db` | dB      | Limiar de abertura do Noise Gate.                |
| **Bypass**         | `bypass`            | binário | Desativa o processamento neural (Dry/Wet 0/100). |

O caminho do modelo (`model_path`) será tratado como um **State Property**, permitindo que a DAW salve e carregue o modelo correto no projeto.

## 3. Estratégia de Compilação

O projeto utiliza *feature flags* para permitir múltiplos alvos de build:

- `cargo build --features standalone`: Binário executável com backend PipeWire (padrão).
- `cargo build --no-default-features --features clap-plugin`: Biblioteca dinâmica (`.clap`) sem dependência de PipeWire.

A feature `clap-plugin` omitirá os módulos `pw_host.rs` e `rt_setup.rs`, mantendo o binário final enxuto.

## 4. Framework: `clack-plugin`

- **Motivo**: Oferece controle granular sobre a implementação sem adicionar overhead desnecessário, permitindo uma integração direta com as estruturas RT-safe do NAM-rs. Ao contrário de frameworks mais opinativos, o `clack` mapeia-se quase 1:1 ao spec CLAP enquanto provê segurança de tipos em Rust.
- **Frameworks de alto nível**: Descartados por adicionarem suporte forçado a VST3, uma camada de GUI embutida que conflita com nossa escolha de `egui` puro, e abstrações que poderiam mascarar o determinismo temporal exigido pelo motor DSP do NAM-rs.
- **Link**: [https://github.com/prokopyl/clack](https://github.com/prokopyl/clack)

## 5. Extensões CLAP Implementadas

A integração utiliza o crate `clack-extensions` para implementar as seguintes extensões do protocolo CLAP:

| Extensão                       | Arquivo                                                                                  | Finalidade                                                                                                             |
|:------------------------------ |:---------------------------------------------------------------------------------------- |:---------------------------------------------------------------------------------------------------------------------- |
| `clap_plugin_audio_ports`      | [audio_ports.rs](file:///home/fabio/nam-rs/src/clap/extensions/audio_ports.rs)           | Declaração explícita de portas de entrada/saída estéreo e suporte a processamento in-place                             |
| `clap_plugin_params`           | [params.rs](file:///home/fabio/nam-rs/src/clap/extensions/params.rs)                     | Mapeamento e automação de parâmetros (`input_gain`, `output_gain`, `gate`, `bypass`) com suporte a gesture e `flush()` |
| `clap_plugin_state`            | [state.rs](file:///home/fabio/nam-rs/src/clap/extensions/state.rs)                       | Persistência do estado do plugin (parâmetros e caminho do modelo) no projeto da DAW                                    |
| `clap_plugin_latency`          | [latency.rs](file:///home/fabio/nam-rs/src/clap/extensions/latency.rs)                   | Reporte dinâmico de latência induzida pelo processamento e resampling ao host                                          |
| `clap_plugin_track_info`       | [track_info.rs](file:///home/fabio/nam-rs/src/clap/extensions/track_info.rs)             | Suporte à cor da track do host para adaptar dinamicamente o accent color da GUI                                        |
| `clap_plugin_remote_controls`  | [remote_controls.rs](file:///home/fabio/nam-rs/src/clap/extensions/remote_controls.rs)   | Páginas de controle pré-configuradas ("Main" e "Gate") para integração com controladores de hardware e Device Panel    |
| `clap_plugin_param_indication` | [param_indication.rs](file:///home/fabio/nam-rs/src/clap/extensions/param_indication.rs) | Feedback visual na GUI para indicar parâmetros mapeados, automatizados ou sob override temporário                      |
| `clap_plugin_gui`              | [gui.rs](file:///home/fabio/nam-rs/src/clap/extensions/gui.rs)                           | Interface gráfica nativa construída com `egui` e embutida via `baseview`                                               |

## 6. Plugin Descriptor

O descritor de metadados do plugin seguirá o seguinte padrão:

- **Plugin ID**: `br.eti.fabiolima.nam-rs`
- **Nome**: `NAM-rs Neural Amp Modeler`
- **Vendor**: `Fabio Lima`
- **URL**: [https://github.com/fabiohl/nam-rs](https://github.com/fabiohl/nam-rs)
- **Features**: `["audio-effect", "distortion", "gate", "simulator", "stereo"]`

## 7. DAWs Alvo de Validação

- **Bitwig Studio**: Plataforma de referência absoluta para conformidade CLAP (co-autora do padrão). Essencial para validar o comportamento de sandboxing e automação sample-accurate.
- **REAPER**: Validação de compatibilidade com hosts de baixo custo e testes de buffers irregulares.
  - NOTA: *Descartado* por estar buggy na minha máquina ubuntu linux.
- **Fender Studio Pro**: Garantia de funcionamento em ambientes de produção de larga escala.
- **CLAP-info / CLAP-host**: Ferramentas de linha de comando para validação técnica rigorosa do spec.

## 8. Compilação e Validação do Plugin

### 8.1. Script de Build (`utils/build-clap.sh`)

O projeto fornece um script automatizado para compilar, instalar e realizar uma auditoria preliminar do plugin no formato CLAP:

- **Build Padrão (com GUI)**:
  Por padrão, o script compila o plugin com suporte a interface gráfica nativa (utilizando a feature `clap-plugin-gui`).
  
  ```bash
  ./utils/build-clap.sh
  ```

- **Build Headless (sem GUI)**:
  Caso queira compilar a versão enxuta para testes sem suporte à interface gráfica, utilize a flag `--headless` ou `--no-gui`:
  
  ```bash
  ./utils/build-clap.sh --headless
  ```

- **Modo Debug**:
  Para compilar em modo de depuração (debug), adicione `--debug`:
  
  ```bash
  ./utils/build-clap.sh --debug
  ```

O script gera a biblioteca dinâmica e a instala no diretório padrão de busca de plugins CLAP do usuário (`~/.clap/nam-rs.clap`).

### 8.2. Instalação e Execução do `clap-validator`

O `clap-validator` é a ferramenta de linha de comando oficial da organização `free-audio` para validar conformidade com a especificação CLAP e identificar potenciais problemas ou vazamentos de recursos.

#### Instalação

Como o `clap-validator` é desenvolvido em Rust, ele pode ser compilado e instalado diretamente a partir de seu repositório Git usando `cargo`:

```bash
cargo install --git https://github.com/free-audio/clap-validator.git
```

Isso instalará o executável `clap-validator` no diretório de binários globais do Cargo (`~/.cargo/bin/`).

#### Execução de Testes Automáticos

Após instalar o validador, você pode executar o script de testes integrados do projeto para rodar a suíte completa de testes unitários, testes de benchmark e testes estruturais do `clap-validator`:

```bash
./utils/tests-cargo.sh
```

O script detecta a presença do `clap-validator` (seja no `PATH` global ou em `~/.cargo/bin/`) e executa os testes estruturais sobre o binário instalado em `~/.clap/nam-rs.clap`, confirmando que o resultado é **0 FAILs** e **0 WARNINGs**.
