<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved. -->

# Benchmarks de Performance (Criterion)

O projeto NAM-rs utiliza o **Criterion.rs** como sua suíte oficial de benchmarks de performance. Dada a natureza sensível de latência de um motor de áudio em tempo real (DSP), realizar medições com rigor estatístico é fundamental para não ser enganado por variações do sistema operacional (ruído, context switches, flutuações de clock).

## Como rodar os Benchmarks

Para executar a suíte de performance:

```bash
cargo bench --bench inference_bench
```

### Benchmarks de Longa Duração (Soak Bench)

Para avaliar a performance sob pressão constante e identificar jitter causado por cache misses ou TLB misses em blocos grandes, o projeto oferece uma suíte de benchmarks de longa duração (30s+ por função):

```bash
cargo bench --features long_bench
```

Ou via script de disparo manual (recomendado):

```bash
bash utils/tests-long.sh
```

Estes benchmarks utilizam blocos de **4096 amostras** (~85ms), reduzindo o peso relativo do overhead de invocação e focando puramente no throughput do motor DSP.

## Como interpretar a Saída do Criterion

Quando você roda um benchmark, o Criterion reporta uma saída similar a esta:

```text
WaveNet_Standard_CH16_64samp_48kHz
                        time:   [107.03 µs 107.32 µs 107.61 µs]
                        change: [−9.3273% −6.2506% −3.5233%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 5 outliers among 50 measurements (10.00%)
  ...
```

### Entendendo as Métricas

1. **`time: [A B C]` (Intervalo de Confiança)**
   Mostra o tempo de execução por iteração, expresso através de um **intervalo de confiança de 95%**.

   * O número central (`B`, ex: `107.32 µs`) é a melhor estimativa pontual de tempo médio.
   * Os números externos (`A` e `C`) definem o limite inferior e superior, garantindo estatisticamente (com 95% de certeza) que a performance verdadeira está dentro dessa margem.

2. **`change: [...]` e `(p = X < 0.05)` (Significância Estatística)**

   * `change` exibe o percentual de diferença em relação ao último teste executado na mesma máquina (valores negativos indicam código mais rápido).
   * O **p-value** (`p`) indica a probabilidade dessa variação ter ocorrido ao acaso. Se `p < 0.05` (nível de significância de 5%), o Criterion atesta que a variação observada é real e não apenas ruído de sistema operacional.

3. **Conclusões Textuais**
   Baseado nos cálculos matemáticos, o software sumariará as conclusões:

   * **Performance has improved / regressed**: O p-value confirmou que a alteração no código fonte provocou uma mudança estatística mensurável (positiva ou negativa).
   * **Change within noise threshold**: O p-value é alto, as margens de erro se interceptam, ou a variação é insignificante. A mudança detectada é ruído.

4. **Outliers (Jitter)**
   As amostras são rodadas centenas de vezes e anomalias são reportadas. Em um sistema de tempo-real crítico como o NAM-rs, ocorrências de `high severe` costumam estar atreladas a *jitter* (falhas de processamento, preempção de thread de áudio pelo kernel, cache misses etc). Executar benchmarks em ambientes blindados (SCHED_FIFO e afinidade de CPU ativados) mitiga outliers.

## Histórico Temporal (Baselines)

Você não precisa comparar os tempos mentalmente. **O Criterion salva a linha de base (baseline) da sua última execução automaticamente**.

Todas as métricas históricas de acompanhamento temporal são gravadas em arquivos locais dentro do seu projeto em: `target/criterion/`

*(Nota: O NAM-rs mantém a geração de relatórios HTML com gráficos temporais propositalmente desativada no arquivo `Cargo.toml` (`default-features = false`) para omitir o download de extensas dependências visuais, limitando a avaliação ao console).*

## Resultados Comparativos: LSTM Escalar vs SIMD (Fused Gates T3)

As otimizações introduziram a fusão de portas (*fused gates*) e ativações SIMD (AVX2/AVX-512) no hot-path das redes recorrentes. Abaixo, os ganhos medidos em uma arquitetura x86-64-v3 (AVX2/FMA) para blocos de 64 amostras:

| Topologia     | Implementação       | Latência (Média) | Speedup   |
|:------------- |:------------------- |:---------------- |:--------- |
| **LSTM 1x8**  | Escalar (Baseline)  | ~22.45 µs        | -         |
| **LSTM 1x8**  | **SIMD Fused (T3)** | **~6.36 µs**     | **3.53x** |
| **LSTM 2x16** | Escalar (Baseline)  | ~83.66 µs        | -         |
| **LSTM 2x16** | **SIMD Fused (T3)** | **~20.29 µs**    | **4.12x** |

### Conclusão Técnica

O ganho de performance superior a **4x** em modelos complexos (2x16) valida a estratégia de fusão de kernels. Ao processar as 4 portas LSTM simultaneamente via vetores SIMD e manter os dados em registradores entre as ativações Sigmoid e Tanh, reduzimos drasticamente os ciclos de CPU desperdiçados com *loads/stores* redundantes e latência de memória.

## Orçamento de Ciclos (WaveNet Hot-Path)

Para guiar futuras otimizações, realizamos a instrumentação granular do *hot-path* da WaveNet (`WaveNetLayer::process_block_internal`) utilizando contadores de ciclos de hardware (**RDTSC**). Esta medição identifica onde a CPU gasta a maior parte do tempo durante o processamento de um bloco de áudio.

### Distribuição de Ciclos por Estágio (Per Layer)

Abaixo, a distribuição percentual média de ciclos em uma arquitetura x86-64-v3 (AVX2) para um modelo Standard (CH=16):

| Estágio Operacional        | Operações Envolvidas               | Budget (%) | Justificativa Técnica                                             |
|:-------------------------- |:---------------------------------- |:---------- |:----------------------------------------------------------------- |
| **Conv1D (SIMD GEMV)**     | Convolução causal, MACs, dilatação | **~45%**   | Fase mais intensa em computação (multiplicação de matriz-vetor).  |
| **1x1 & Residual (Fused)** | Projeção densa, soma de resíduo    | **~25%**   | Alta pressão de memória (read-modify-write) e projeção de canais. |
| **Mixin (Conditioning)**   | Injeção de metadados de timbre     | **~15%**   | Operação densa aplicada à entrada de cada camada.                 |
| **Act & Head (Fused)**     | Tanh/Sigmoid, Skip-Connections     | **~15%**   | Custo das funções transcendentais (aproximadas via SIMD).         |

### Análise de Fluxo de Dados (Array Level)

No nível da `WaveNetLayerArray`, a cascata de camadas domina o processamento (**>90% do tempo total**). Os estágios de interface (**Rechannel** de entrada e **Head Rechannel** de saída) representam uma sobrecarga fixa negligenciável à medida que o número de camadas aumenta, validando a escalabilidade da arquitetura NAM-rs para modelos complexos.

> [!TIP]
> A fusão da **Tanh** com o **Head Accumulation** foi a otimização mais impactante do Épico E, reduzindo o budget do estágio de ativação de ~30% para ~15% ao eliminar passagens redundantes pela memória Cache L1.

## Relatório de Experimento: Temporal Tiling (Dual-Frame) na Conv1D

No Épico de otimização de hot-paths, foi projetada e testada uma variante **Temporal Tiling** (processamento "Dual-Frame") para os kernels de `Conv1D`, visando maximizar o reuso dos pesos na Cache L1 ao processar dois frames simultaneamente na inferência da WaveNet.

### Resultados da Medição (64 samples, 48kHz, CH=16, AVX2)

* **Single-Frame (Baseline):** ~84 µs
* **Dual-Frame Tiling:** ~100 µs (Regressão de ~19%)

### Análise e Decisão Arquitetural

Apesar da teoria sugerir que carregar os pesos da memória apenas metade das vezes pouparia largura de banda (L1 cache), na prática a arquitetura x86-64 (AVX2/FMA) revelou-se limitada pelo **Register Pressure** (Pressão de Registradores).
Para processar dois frames em paralelo:

1. O número de acumuladores SIMD necessários dobrou (de 4 YMM para 8 YMM por canal).
2. O overhead de instruções no frontend (ex: *broadcasts* e *blends*) superou a economia de *loads*.
3. O compilador foi forçado a usar *register spilling* ou atingiu gargalos de *execution ports* para instruções de mistura (Port 5).

**Conclusão:** O gargalo primário da `Conv1D` no NAM-rs não está atrelado à largura de banda da Cache L1, e sim ao *throughput* computacional e contenção de registradores do backend (FMA). Por causa disto, embora a implementação do kernel tenha sido mantida no trait `SimdMath` para portabilidade e testes em arquiteturas com mais registradores (ex: AVX-512 ou ARM NEON), o loop principal na `WaveNetLayer` permanece utilizando **processamento Single-Frame** para garantir a menor latência e maior estabilidade em tempo real.

## Relatório de Experimento: Fusão Stereo no Output Stage (T3.2)

A Tarefa T3.2 visou eliminar passagens redundantes de memória no estágio final de saída, fundindo as operações de ganho (Hysteresis/Gate) dos canais L e R em uma única chamada SIMD stereo.

### Resultados da Medição (64 samples, 48kHz, AVX2)

| Topologia       | Antes da Fusão | Após a Fusão (T3.2) | Ganho (%) |
|:--------------- |:-------------- |:------------------- |:--------- |
| **WaveNet Std** | ~107.3 µs      | ~101.4 µs           | **~5.5%** |
| **LSTM 2x16**   | ~15.4 µs       | ~14.7 µs            | **~4.5%** |

### Conclusão

A fusão stereo reduz o tráfego de memória na Cache L1 ao ler os canais L e R simultaneamente e aplicar os pesos de ganho/rampa em um único loop. O ganho é mais pronunciado em blocos menores (ex: 32 samples, onde mediu-se **~8.5%** de melhora), onde o overhead de despacho e os cache misses parciais têm maior peso relativo.
