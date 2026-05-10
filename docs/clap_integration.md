<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva. -->

# Estratégia de Integração CLAP (Clever Audio Plug-in)

Este documento descreve a arquitetura e a estratégia para transformar o motor DSP do NAM-rs em um plugin de áudio compatível com o padrão CLAP.

## 1. Thread Model

A integração CLAP deve respeitar estritamente a segregação de threads já existente no NAM-rs, mapeando-as para o modelo do host (DAW):

- **Main Thread (Host)**:
  - Responsável pela inicialização do plugin, escaneamento de parâmetros e gerenciamento de estado.
  - No NAM-rs, esta thread substituirá o loop principal do `src/main.rs`.
  - Gerencia o carregamento de arquivos `.nam`/`.namb` via `src/loader/`.
- **Audio Thread (Real-time)**:
  - Chamada pelo host via callback `process()`.
  - **Requisito Crítico**: Deve manter a política de **ZERO alocações** e **ZERO locks**.
  - Utilizará `src/dsp/pipeline.rs` para o processamento, adaptando os buffers do CLAP para o formato interno.
  - Diferente do PipeWire (que é dual-stream), o CLAP fornece buffers de entrada e saída em um único contexto, permitindo simplificar o `DspBridge`.

## 2. Mapeamento de Parâmetros

Os parâmetros expostos ao host serão mapeados a partir da estrutura `NamPluginParams` (veja `src/common/params.rs`):

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

## 5. Extensões CLAP Planejadas

A integração utilizará o crate `clack-extensions` para implementar as seguintes funcionalidades do protocolo CLAP:

| Extensão               | Finalidade                                                               |
|:---------------------- |:------------------------------------------------------------------------ |
| `clap-ext-params`      | Automação de parâmetros (`input_gain`, `output_gain`, `gate`, `bypass`). |
| `clap-ext-state`       | Persistência de estado (save/load do projeto na DAW).                    |
| `clap-ext-thread-pool` | Paralelização da inferência via pool de threads do host.                 |
| `clap-ext-latency`     | Reporte de latência induzida pelo processamento/resampling.              |
| `clap-ext-audio-ports` | Declaração explícita de entradas/saídas estéreo e suporte in-place.      |
| `clap-ext-gui`         | Interface gráfica nativa via `egui` + `baseview`.                        |

## 6. Plugin Descriptor

O descritor de metadados do plugin seguirá o seguinte padrão:

- **Plugin ID**: `br.eti.fabiolima.nam-rs`
- **Nome**: `NAM-rs Neural Amp Modeler`
- **Vendor**: `Fabio Lima`
- **URL**: [https://github.com/fabiohl/nam-rs](https://github.com/fabiohl/nam-rs)
- **Features**: `["audio-effect", "distortion", "gate", "simulator", "stereo"]`

## 7. DAWs Alvo de Validação

- **REAPER**: Plataforma primária de desenvolvimento devido ao seu preço acessídel, flexibilidade com buffers variáveis e ferramentas de debug de plugin.
- **Bitwig Studio**: Referência de implementação do padrão CLAP para validação de conformidade.
- **Studio One**: Validação de compatibilidade em hosts comerciais de grande escala.
- **CLAP-info / CLAP-host**: Ferramentas de linha de comando para validação técnica do contrato do plugin.
