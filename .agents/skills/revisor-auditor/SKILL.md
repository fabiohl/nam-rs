---
name: revisor-auditor
description: Painel de auditores, caçadores de bugs, cientistas, engenheiros sêniores e especialistas em diversas disciplinas associadas ao projeto. Conjuntamente realizam varredura sistemática e cirúrgica do repositório, caçando bugs latentes, falhas de concorrência, riscos de runtime e desvios arquiteturais - antes que eles se tornem incidentes de produção.
---

# Skill: Revisor Auditor

## When to use this skill

Use esta skill para realizar uma varredura proativa de bugs e oportunidades de melhorias no projeto, focando especialmente em gargalos de performance, condições de corrida lock-free, violações de processamento real-time e falhas de alinhamento em memória cache da CPU.

## Instructions

Entenda os objetivos do projeto e se eles estão sendo alcançados.
Melhore ao máximo a organização, o código, os comentários e a documentação.
Se necessário, promova refatorações. Porém cuidado e bom-senso para não quebrar funcionalidades importantes.
Garanta as melhores práticas de engenharia de software.

### 1. Contextualização Antes de Tudo (Read First)

Antes de qualquer análise de código, lembre de carregar seu contexto mental lendo `docs/` e os fundamentos em `.agent/rules/`.

### 2. Taxonomia de Bugs: O Que Caçar

Analise o código, módulo por módulo, rigorosamente contra estas restrições:

#### 🔴 CRÍTICO — Riscos e Quebras Arquiteturais

Exemplos:

| Categoria                               | O que verificar                                                                                                                                           |
| --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Violação do Tempo Real**              | Alocação indireta (ex: Box, Arc, Vec, String, macros format!) ocorrendo na thread DSP ou no callback do nih_plug.                                         |
| **I/O ou Syscalls no DSP**              | Qualquer operação bloqueante, syscall, I/O síncrono ou semáforo adquirida pelo loop do áudio.                                                             |
| **Thread arriscada**                    | Thread com risco de preempção forçada, migração de cores ou de congelar o sistema inteiro.                                                                |
| **Gargalo de Hardware (False Sharing)** | Variáveis ou estruturas sendo processadas simultaneamente inter-threads, e que não estejam rigorosamente alinhadas a 128 bytes por `#[repr(align(128))]`. |
| **Tratamento Descuidado de Gravação**   | Retornos do subsystema `io_uring` ignorados ou erros silenciados no background, possibilitando geração de um arquivo WAV corrompido ou vazio.             |

#### 🟠 ALTO — Falhas de robustez

Exemplos:

| Categoria                             | O que verificar                                                                                                                                               |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Dados do Cabeçalho WAV incorretos** | O software está deduzindo ou congelando (hardcoding) sample/bit rate no `hound` em vez de inspecionar fisicamente o formato que o Host/PipeWire entrega.      |
| **Sincronia Inconsistente (SPSC)**    | Uso ambíguo de canais MPSC ou Mutexes no Ring Buffer. Lock corrompido. Buffer Overruns e Underruns omitidos ao usuário na CLI.                                |
| **Subversão de Sinal SO**             | A aplicação capotando sem fechar de perto os buffers finais, ou o handler de interrupções ignorando que precisa fazer Graceful Shutdown ao receber um CTRL+C. |

E assim por diante, até os "médios" e "baixos".

### 3. Procedimento de Correção

Estruture em um relatório de artefato as falhas listadas. Na correção destas, aja com precisão cirúrgica sem quebrar as restrições inegociáveis de um audio renderer Bit Perfect. Revalide tudo com script de lint local após finalizar!
