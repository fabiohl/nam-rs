---
name: debugger
description: Acionado quando algo não funciona como esperado. Atua como um painel multi-disciplinar de analistas e engenheiros seniores capazes de resolver os problemas mais difíceis.
---

# Skill: Debugger

## When to use this skill

Use esta skill sob a manifestação de um erro ativo, crash, comportamento não-linear de DSP, áudio truncado (clicks/pops) ou impasses de compilação em componentes críticos.

## Instructions

### 1. Foco em Diagnóstico de Áudio e Alta Performance (Analyze First)

- Não proponha remendos ou mutações na esperança mágica de funcionar em um sistema focado em tempo-real. Cada bit de mudança custa latência.
- **Falhas no buffer de áudio (Clicks/Pops)**: Resultam em buffer overrun ou underrun. Analise estaticamente a cadência Produtor/Consumidor. O Host PipeWire gerou os buffers tarde demais? O I/O bloqueou o Consumidor causando overrun no Ring Buffer?
- **Travamentos e CPU 100%**: Reavalie a estratégia de consumo lock-free. Há algum loop ocupado destrutivo (spin loop) sem ceder (yielding) de maneira correta à CPU? Spinlocks estão forçando aquecimento?
- **Corrupção de WAV**: O arquivo gerado está vazio ou truncado? Verifique o `io_uring` e o Graceful Shutdown. Os arquivos não estão recebendo sync ou não terminam a gravação da formatação header após drain.

### 2. Intervenção Baseada em Evidências

- Antes de compilar, observe se as dependências C-level de Kernel/PipeWire estão em conflito com std. Em especial manipulação externa de ponteiros e unsafe pointers se integrando ao `nih_plug`.
- Aplique o patch com estrita observância das regras. Corrija o que estancou sem alocar dentro do callback do áudio de Tempo Real e valide se o seu _fix_ alterou o status macro do lock-free.
- Após sanado, caso tenha usado blocos simuladores e artefatos logs `println!` na thread RTA só para investigar, VOCÊ DEVE APAGÁ-LOS RIGOROSAMENTE para garantir que tudo retorne a zero atraso/io na via de áudio.
