<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva. -->

# Estratégia de Integração CLAP (Clever Audio Plug-in)

Este documento descreve a arquitetura e a estratégia para transformar o motor DSP do NAM-rs em um plugin de áudio compatível com o padrão CLAP.

## 1. Thread Model

A integração CLAP deve respeitar estritamente a segregação de threads já existente no NAM-rs, mapeando-as para o modelo do host (DAW):

- **Main Thread (Host)**:
  - Responsável pela inicialização do plugin, escaneamento de parâmetros e gerenciamento de estado.
  - No NAM-rs, esta thread substituirá o loop principal do `src/main.rs`.
  - Gerencia o carregamento de arquivos `.nam`/`.namb` via `src/loader.rs`.
- **Audio Thread (Real-time)**:
  - Chamada pelo host via callback `process()`.
  - **Requisito Crítico**: Deve manter a política de **ZERO alocações** e **ZERO locks**.
  - Utilizará `src/dsp/pipeline.rs` para o processamento, adaptando os buffers do CLAP para o formato interno.
  - Diferente do PipeWire (que é dual-stream), o CLAP fornece buffers de entrada e saída em um único contexto, permitindo simplificar o `DspBridge`.

## 2. Mapeamento de Parâmetros

Os parâmetros expostos ao host serão mapeados a partir da estrutura `NamPluginParams` (veja `src/params.rs`):

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
- `cargo build --features clap-plugin`: Biblioteca dinâmica (`.clap`) sem dependência de PipeWire.

A feature `clap-plugin` omitirá os módulos `pw_host.rs` e `rt_setup.rs`, mantendo o binário final enxuto.

## 4. Decisão de Framework (Pendente)

Estamos avaliando duas abordagens para a implementação final:

1. **`nih-plug`**:
   - *Prós*: Framework de alto nível, lida com GUI, automação de parâmetros e gerenciamento de estado de forma automática. Suporte nativo a CLAP e VST3.
   - *Contras*: Adiciona muitas dependências e tem uma estrutura opinativa.
2. **`clack` / `clap-sys`**:
   - *Prós*: Controle total sobre a implementação, baixo overhead, permite manter o NAM-rs minimalista.
   - *Contras*: Maior esforço de implementação (boilerplates manuais para parâmetros e estado).

**Recomendação**: `nih-plug` é a opção mais pragática para um plugin CLAP de alta qualidade — battle-tested e multi-formato. `clack` é preferível quando se deseja controle granular sem overhead de VST3. A decisão final pode ser diferida: o staging atual foca em tornar o NAM-rs *plugin-ready* via trait `AudioHost` e a pipeline agnóstica em `src/dsp/pipeline.rs`.

> **Nota sobre `cdylib`**: Quando a feature `clap-plugin` for efetivamente implementada, será necessário adicionar `crate-type = ["cdylib"]` ao `[lib]` do `Cargo.toml`. O plugin CLAP é uma shared library (`.clap` = `.so` no Linux). Este item é **diferido** até a Sprint de implementação CLAP real.

## 5. DAWs Alvo de Validação

- **Bitwig Studio**: Referência primária para suporte CLAP.
- **REAPER**: Excelente para depuração de performance e buffers variáveis.
- **CLAP-info / CLAP-host**: Ferramentas de linha de comando para validação de contrato.
