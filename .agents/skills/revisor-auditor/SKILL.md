---
name: revisor-auditor
description: Painel de auditores, caçadores de bugs, cientistas, engenheiros sêniores e especialistas em diversas disciplinas associadas ao projeto. Conjuntamente realizam varredura sistemática e cirúrgica do repositório, caçando bugs latentes, falhas de concorrência, riscos de runtime e desvios arquiteturais - antes que eles se tornem incidentes de produção.
---

# Skill: Revisor Auditor

## When to use this skill

Use para inspecionar, revisar e diagnosticar estrita aderência arquitetural, focando nos princípios cruciais de Inferencia via FMA (Fused Multiply Action), PipeWire Assíncrono e detecção cirúrgica de violações computacionais e lock-free no buffer macro da thread principal (DSP) do projeto Standalone NAM-rs.

## Instructions

### 1. Ingestão de Referenciais

Revise seu contexto mental sob o manifesto presente nos repositórios como as restrições em `.agents/rules/rust.md` e a arquitetura estática estipulada em `docs/NAM-rs-referência.md`.

### 2. Subversão Arquitetural Híbrida: O Que Erradicar

Inspecione linha-de-código detectando categoricamente:

#### 🔴 CRÍTICO — Violações Severas e Predições Daninhas

| Categoria                            | O que verificar                                                                                                                                                                      |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Pulo Condicional Desperdiçado**    | Modelos WaveNet ou LSTMs usando loops iterativos que não foram suprimidos via desdobramento (_loop unrolling_ de Const Generics), custando fatias pesadas do Branch Predictor local. |
| **Latência por Alocação (Box, Vec)** | Instanciações sob a heap dentro da esteira low latency provocando desastre de alocação preempitiva no kernel hospedeiro.                                                             |
| **Cálculo Matemático Letárgico**     | Submissão aos padrões `std::math` substituindo a velocidade necessária requerida da biblioteca FastMath sobre vetores de instrução _std::simd_.                                      |
| **Gargalo L1/L2 (False Sharing)**    | Estruturas assíncronas ativas (Parametros Tone3000 de ganho) comunicando sobre structs não separadas geograficamente a 128-bytes no Ring Buffer (_#[repr(align(128))]_).             |
| **Resquícios Herdados do AudioRip**  | Algoritmos IO persistentes para disco e requisições passivas I/O Uring sendo mantidos ou ressuscitados, onerando o kernel em ações desnecessárias ao NAM-rs Standalone.              |

#### 🟠 ALTO — Condicionamentos Inadequados e Falhas de Domínio

| Categoria                            | O que verificar                                                                                                                                                                                                                                                                                                                 |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Defasagem Operacional PCM**        | Emuladores não aplicando processamento com isolamentos nativos Superamostrados FIR limitados (Linear Phase Filter), causando _digital aliasing_ fatal pela interface de processamento a 48 kHz base da convolucional externa sob intermédios dinâmicos host (ex: conversões errôneas no Pipewire vindo por 96kHz ou similares). |
| **Tratativa Abstrata Parametrizada** | Omissão do leitor nativo falhar ao intervir e balancear silenciosa os níveis operacionais brutos ditados pelos limites dBu de entrada do arquivo _.namb_ provido e estático causará sobrecarga destrutiva nas atuações tangente-hiperbólica do som.                                                                             |

E assim por diante, até os "médios" e "baixos".

### 3. Retificações Perfeitas e Bit-Perfect

Submeta patches letolizando ineficiências em vetores numéricos puristas. A operação final obriga validação incondicional ao ciclo local `utils/lints.sh`. Tudo deve operar nas janelas perfeccionistas de baixa latência.
