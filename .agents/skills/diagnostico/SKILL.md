---
name: diagnostico
description: Diagnóstico a partir de mensagem de erro do NAM-rs. Cole o bloco de suporte e a IA faz a triagem completa.
---

# Skill: Diagnóstico de Erro NAM-rs

## When to use this skill

Workflow para quando o usuário cola uma mensagem de erro/bloco de suporte gerado pelo NAM-rs.

## Fase 1: Extração e Triagem (Skill: `debugger`)

Acione a skill `debugger` e execute os seguintes passos:

### 1.1. Parse do Bloco de Suporte

Extraia do texto colado pelo usuário:

- **Código de erro** (`Exxxx`) e **mnemônico** (ex: `NAMB_CRC32_MISMATCH`)
- **Parâmetros contextuais** (arquivo, tamanho, rate, etc.)
- **Informações de sistema** (versão, arch, avx2/fma, os, kernel)
- **Timestamp** da ocorrência

Se o usuário colou apenas a mensagem amigável (sem o bloco técnico), peça que cole o bloco completo do shell/terminal — incluindo a seção "Informação para suporte" que o NAM-rs gera.

### 1.2. Localização no Código-Fonte

1. Localize o `NamErrorCode` correspondente em `src/diagnostics.rs`.

2. Use a tabela de faixas para direcionar a investigação:

   | Faixa   | Onde investigar                                        |
   | ------- | ------------------------------------------------------ |
   | `E1xxx` | `src/loader/`, `src/main.rs::load_and_send_model()`    |
   | `E2xxx` | `src/pw_host.rs`, `src/dsp/resampler.rs`               |
   | `E3xxx` | `src/spsc.rs`, `src/main.rs::cli_loop()`               |
   | `E4xxx` | `src/main.rs::cli_loop()`, `src/main.rs::parse_args()` |
   | `E5xxx` | `src/main.rs::main()`                                  |

3. Leia no código-fonte o módulo e a função onde a mensagem é emitida para entender o contexto da situação.

### 1.3. Diagnóstico de Causa-Raiz

Exemplo de questões que podem vir a serem levantadas com os dados extraídos:

- **E1xxx (Modelo)**: O arquivo existe? O formato (JSON/binário) está íntegro? O CRC confere? A topologia (WaveNet/LSTM) está tabelada? Os pesos são suficientes?
- **E2xxx (Áudio)**: O PipeWire está rodando? O sample rate é suportado? O resampler conseguiu criar o resampler? Existe permissão SCHED_FIFO?
- **E3xxx (SPSC)**: O canal está cheio porque o DSP está travado? O modelo anterior foi consumido?
- **E4xxx (CLI)**: O usuário digitou um comando válido? O valor de ganho é um número válido?
- **E5xxx (Sistema)**: A CPU tem AVX2+FMA? A memória é suficiente?

## Fase 2: Proposta de Solução ao Usuário (Skill: `debugger`)

Com o diagnóstico concluído, apresente ao usuário:

1. **O que aconteceu** — explicação em linguagem acessível da causa-raiz.
2. **O que fazer** — passos concretos que o usuário pode tomar (ex: baixar modelo novamente, verificar permissões, atualizar kernel, etc.).
3. **Se aplicável**: informar se é um bug do NAM-rs (prossegue para Fase 3) ou um problema de ambiente/configuração do usuário (encerra aqui com orientações).

## Fase 3: Avaliação de Tarefa de Fix (Skill: `planejador-arquiteto`)

Se a Fase 2 concluir que é um **bug ou deficiência do NAM-rs**:

1. Acione a skill `planejador-arquiteto`.
2. Avalie a severidade e a urgência do problema.
3. Proponha ao desenvolvedor se é necessário criar uma tarefa técnica de fix, detalhando:
   - **Módulo(s) afetado(s)** e linhas de código relevantes
   - **Impacto** (crash? degradação? inconveniência?)
   - **Complexidade estimada** (trivial / moderado / complexo)
   - **Fix proposto** — esboço da correção técnica
4. Se o usuário aprovar, crie a tarefa usando a workflow `/tarefa` ou proceda diretamente ao fix usando a skill `implementador`.
