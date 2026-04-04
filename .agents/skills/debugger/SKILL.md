---
name: debugger
description: Acionado quando algo não funciona como esperado. Atua como um painel multi-disciplinar de analistas e engenheiros seniores capazes de resolver os problemas mais difíceis.
---

# Skill: Debugger

## When to use this skill

Use esta skill sob a manifestação de um erro ativo, crash, comportamento não-linear de DSP, áudio truncado com modulação/ringing ou redes neurais resultando em degradação contínua nos cálculos de áudio real-time.

## Instructions

### 1. Foco em Diagnóstico Inferencia e Otimização Híbrida (Analyze First)

- Não proponha remendos ou mutações na esperança mágica de contornar gargalos matemáticos. Cada alocação onera sub-milissegundos perigosos no PipeWire.
- **Falhas de Processamento de Janelas e Clicks**: O buffer de pipeline sofre xruns pois os multiplicadores matriciais demoraram demais? O Host mandou parâmetros incompatíveis com as restrições síncronas do NAM-rs? O preditor de branch do processador descarregou o pipeline?
- **Travamentos e Avaliação de Consumo CPU**: Empregue avaliações focadas se estruturas SPSC estão preenchendo adequadamente as variáveis dinâmicas de DSP. Onde a carga FastMath se atrelou pesadamente na CPU?
- *Nota vital*: Rotinas contendo resquícios de I/O de arquivos (`io_uring`) e disco de sistema herdados não são aceitas. Investigue o PipeWire puramente operando sobre a infra de Thread de DSP Isolcpus.

### 2. Intervenção Baseada em Evidências

- Antes de compilar a solução proposta SIMD, observe atentamente os desdobramentos de registros AVX2 ou AVX-512 via `std::simd` usando macros corretos das arquiteturas base const generics (SoA).
- Aplique correções puristas sem introduzir instancianção de ponteiros `Vec`, `Box`, etc, durante a ativação da submissão DSP.
- Após sanado, arquivos e linhas de log ou simuladores utilizados unicamente como debulhadores visuais na thread devem ser rigorosamente limpos da codebase.
