# Planejamento de Sprints e Tarefas (NAM-rs)

## Sprint I - Microarchitectural Hyper-Optimization & Cache Geometry

Esta Sprint tem foco exclusivo na mitigação dos gargalos limítrofes do processador, especificamente tráfego na L1 Cache e Branch Prediction, pavimentando o terreno para inferências de modelos massivos com latência imperceptível.

### [ ] Tarefa I.1. SIMD BFloat16/F16C Weight Compression (VNNI-like)

**Justificativa:** O principal gargalo na avaliação de Redes Neurais densas é o Memory Bound (banda de memória), não o Compute Bound (cálculos matemáticos). A L1 Cache dos núcleos tem tamanho estrito (geralmente 32KB a 48KB para dados). Em redes grandes, os vetores de pesos excedem essa capacidade facilmente, forçando despejos constantes (evicções) para a L2 Cache, atrasando a CPU.
**O que será feito:**

- Introduzir compressão de memória para a estrutura de matrizes de pesos (Weights e Bias) de `f32` (32 bits) para `f16` (Half-precision, 16 bits) no momento do preenchimento da `WaveNet` e `LSTM` (na thread de parser/loader).
- Na Thread RT/DSP (processamento), instanciar as funções intrínsecas F16C (`_mm256_cvtph_ps` e `_mm512_cvtph_ps`) para descompactar os pesos `f16` em registradores YMM/ZMM de volta para `f32` on-the-fly, *apenas* no instante do ciclo da CPU, em blocos de 8 em 8 (ou 16 em 16 no ZMM).
  **Resultados Esperados:**
- Duplicação imediata e gratuita da capacidade virtual da L1 Cache.
- Redução substancial de Cache Misses nas operações temporais.
- Latência marginalmente aprimorada para modelos Feather e Nano; impacto massivo nos tempos limite do WaveNet Standard/Lite.

### [x] Tarefa I.2. LSTM Fused Gate Processing

**Justificativa:** Diferente do modelo estático WaveNet que agora opera em blocos Batch GEMM hiperdensos, a recursividade da rede LSTM exige a computação *sample-by-sample* em função do tempo, obrigando a chamar a engine SIMD quatro vezes por frame para processar os Gates (Input, Forget, Cell, Output). Isso quebra o paralelismo nativo dos pipelines de Unidades de Execução da CPU (Out-Of-Order) adicionando um overhead severo.
**O que será feito:**

- Refatorar a classe gerenciadora (o carregador) do LSTM para *interfolhar* horizontalmente os vetores de pesos ($W$ e $U$).
- Ao invés de executar um *dot product* escalar isolado para cada porta, as 4 matrizes de pesos correspondentes ao Input, Forget, Cell, e Output serão organizadas consecutivamente por linha de modo a formarem um agrupamento massivo que possibilite um único loop `dot_product_4x`.
- No loop principal DSP, o estado e as ativações da LSTM (`tanh` e `sigmoid`) colapsarão essas portas usando um somatório paralelo de múltiplas vias (Multi-Accumulator ILP).
**Nota de Conclusão (Auditoria de Benchmark):** A refatoração (fused gate via `dot_product_4x`) foi auditada. Medições comparativas apontam ausência de regressões e os seguintes ganhos de performance (latência):
- `LSTM_Dynamic_1x16` caiu de ~29.11 µs para ~20.89 µs (**~28.2% de aceleração**).
- `LSTM_2x16` (Estático) caiu de ~29.11 µs para ~25.86 µs (**~11.2% de aceleração**).
  **Resultados Esperados:**
- Subtração significativa nos tempos de inferência de arquiteturas LSTM 1x16 ou 2x16.
- Potencial aceleração de 20% a 30% no loop DSP da classe `LstmModel`.

### [x] Tarefa I.3. Resampler Sinc-SIMD Nativo

**Justificativa:** O NAM-rs atualmente importa o excelente `rubato` para conversão de *sample rates*. No entanto, o `rubato` é uma solução genérica. Ele atua sobre buffers flutuantes `Vec<f32>` escalares na camada interna de filtragem Sinc-Kaiser, deixando oportunidades de desempenho sobre a mesa num software altamente calibrado em DSP e SIMD.
**O que será feito:**

- Eliminar gradativamente o pacote `rubato` da pipeline.
- Escrever um filtro polifásico FIR customizado no diretório `dsp/resampler.rs` que consuma `std::simd` nativo e unrolling estático adaptado a blocos curtos e preditivos (como `1x64`, `1x128` frames de PipeWire).
- Codificar internamente o kernel de janela Kaiser e o multiplicador de fase Sinc com registradores `YMM` e matrizes de const generics garantindo o zero alocação estrito (RT-safe).
  **Resultados Esperados:**
- Resampling bidirecional `96k <-> 48k` efetuado num micro-passo, reduzindo o tempo global do DSP Callback consideravelmente.
- Fim da sobrecarga abstrativa do gerenciamento de alocadores externos, gerando um build sem dependências indiretas para a lógica central de DSP.

### [ ] Tarefa I.4. Zero-Copy DspBridge

**Justificativa:** O modelo `DspBridge` foi concebido usando dupla memória plana de `[f32; 8192]` travada com `Acquire`/`Release` barrier sync. Embora lock-free e altamente previsível (conforme elogiado na Sprint I), exige trânsito contínuo de matrizes na memória da CPU, mesmo em trechos estritamente silenciosos, consumindo banda que poderia ser canalizada para a Inferência Neural.
**O que será feito:**

- Modificar a macro-arquitetura do sistema Dual-Stream do PipeWire em `pw_host.rs` elaborando um conceito de *In-Place Process* nativo.
- Verificar a viabilidade e, se compatível, forçar a thread `Audio/Sink` (Capture) a preencher os dados processados *diretamente* sobre as janelas Ring Buffer disponibilizadas pela thread *Playback*. Isso dispensa um array intermediário ponte alocador e minimiza a instrução compulsória `copy_from_slice`.
- Envolver memory constraints a nível de kernel ou memory map `mmap` (via fd) se estritamente necessário para unificar os fluxos bidirecionais entre a API PipeWire e o loop nativo.
  **Resultados Esperados:**
- Evitará desperdícios em `memmove` e largura de banda intra-processo da RAM à L1 (aproximadamente `~96MB/s` aliviados num clock padrão), focando os ciclos totalmente na convolução vetorial SIMD.
