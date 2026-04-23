<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva. -->

# Benchmarks de Performance (Criterion)

O projeto NAM-rs utiliza o **Criterion.rs** como sua suíte oficial de benchmarks de performance. Dada a natureza sensível de latência de um motor de áudio em tempo real (DSP), realizar medições com rigor estatístico é fundamental para não ser enganado por variações do sistema operacional (ruído, context switches, flutuações de clock).

## Como rodar os Benchmarks

Para executar a suíte de performance:

```bash
cargo bench --bench inference_bench
```

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

Todas as métricas históricas de acompanhamento temporal são gravadas em arquivos locais dentro do seu projeto em:
`target/criterion/`

*(Nota: O NAM-rs mantém a geração de relatórios HTML com gráficos temporais propositalmente desativada no arquivo `Cargo.toml` (`default-features = false`) para omitir o download de extensas dependências visuais, limitando a avaliação ao console).*
