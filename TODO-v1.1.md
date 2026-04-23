# Ideias para versão 1.1

---

## Idéia 1. Resampler Sinc-SIMD Nativo & Fase Mínima (Tarefa I.3)

Esforço: 🔴 Alto · Impacto: 🟢 Alto (Performance & Latência) · Risco: 🔴 Alto

### 1.1 Diagnóstico e Justificativa

O `NamResampler` em `resampler.rs` utiliza atualmente a biblioteca `rubato` (`SincFixedIn`). Embora excelente, o `rubato` é uma solução genérica que opera sobre buffers flutuantes `Vec<f32>` e loops escalares na camada interna de filtragem Sinc-Kaiser. Isso deixa oportunidades significativas de desempenho em um software altamente calibrado para SIMD.

Além disso, o filtro FIR padrão é de **Fase Linear** (Kaiser-BlackmanHarris2, `sinc_len=256`).

- **Latência:** O group delay simétrico adiciona **~1.5 ms** de latência pura no roundtrip 96k → 48k → 96k.
- **Artefatos:** Filtros lineares geram *pre-ringing* audível, algo inexistente em amplificadores valvulados reais (sistemas de fase mínima).
- **Sobrecarga:** Dependência de alocadores externos e gerenciamento de estado opaco para o loop RT-DSP.

### 1.2 Implementação: Resampler Nativo SIMD

A meta é substituir o `rubato` por um motor polifásico FIR customizado no diretório `dsp/resampler.rs`:

1. **SIMD Nativo (`std::simd`):** Utilizar intrínsecos de 256-bit (YMM) e 512-bit (ZMM) para processar o kernel do resampler.
2. **Unrolling Estático:** Otimizar para blocos curtos e preditivos (ex: `1x64`, `1x128` frames de PipeWire), minimizando o custo de setup do filtro por callback.
3. **Zero-Alocação (RT-Safe):** Utilizar matrizes de `const generics` e buffers circulares estáticos para garantir que nenhum `malloc` ocorra no hot-path.
4. **Kernel Kaiser-Sinc:** Codificar internamente o multiplicador de fase Sinc e a janela de Kaiser para controle total da banda de transição e rejeição de aliasing.

Considerar trabalhar junto ao upstream do Rubato 2.0?

### 1.3 Abordagens de Fase (Linear vs Mínima)

Para reduzir a latência de ~1.5ms para valores próximos de zero no resampler, propõe-se duas abordagens:

#### **Abordagem A — Tabela estática de coeficientes MPS:**

1. Gerar os coeficientes Kaiser-Blackman.
2. Aplicar a **Transformada de Hilbert** (ou algoritmos como o de Cepstrum) para converter para fase mínima.
3. Injetar como tabela `const` no código, usada pelo resampler FIR customizado.

#### **Abordagem B — Controle de Group Delay Dinâmico:**

1. Implementar o resampler polifásico FIR com suporte a múltiplos kernels (Linear vs Minimum Phase).
2. Permitir que o usuário escolha entre "Latência Zero" (Fase Mínima) ou "Fase Perfeita" (Linear) via configuração.

### 1.4 Riscos Críticos e Validação

- **Resposta Audível:** A fase mínima redistribui a energia temporal. Requer validação perceptual (Double-blind ABX) com músicos para garantir que o "feel" do amplificador não foi degradado.
- **Invalidação de Golden Vectors:** Todos os testes de roundtrip e vetores de referência que passam pelo resampler precisarão ser re-gerados.
- **SNR e Aliasing:** Filtros de fase mínima podem ter rejeição de aliasing ligeiramente inferior na stopband em comparação com filtros lineares de mesmo tamanho. Requer análise rigorosa via FFT e impulso de Dirac.
- **Complexidade SIMD:** A implementação de filtros polifásicos com `std::simd` requer cuidado extra com o alinhamento de memória e os *tail blocks*.

---

## Idéia 2. Zero-Copy DspBridge (Processamento In-Place)

Esforço: 🟡 Médio · Impacto: 🟡 Médio (Redução de tráfego L1) · Risco: 🔴 Alto (Alteração de macro-arquitetura PW)

### 2.1 Diagnóstico e Justificativa

A arquitetura dual-stream atual (`Audio/Sink` para Capture e `Stream/Output/Audio` para Playback) requer um buffer intermediário lock-free (`DspBridge`) para passar os dados do processamento do capture para a reprodução. Isso adiciona overhead de cópia obrigatória de memória intra-processo e uso de cache. A versão 1.0.x (RC) mantém essa arquitetura dupla por segurança, mas ela pode ser simplificada.

### 2.2 Implementação: Processamento In-Place Nativo

Modificar a macro-arquitetura do sistema Dual-Stream do PipeWire em `pw_host.rs`:

- Reordenar a inicialização para criar a stream de playback antes do listener da stream de capture.
- Dentro da closure de `process()` da stream de captura, acessar diretamente a stream de playback via ponteiro raw (`*mut pw::stream::Stream`).
- Efetuar o `dequeue_buffer()` da stream de playback *diretamente dentro da thread de captura*, preenchendo os dados in-place e eliminando o array intermediário.
- A thread de Playback do PipeWire ficaria com um callback `process` vazio (no-op), visto que o buffer já foi preenchido pela Capture Stream no mesmo `thread_loop`. Isso elimina os 65KB do `DspBridge`.

### 2.3 Riscos Críticos e Validação

- **Invasividade:** A mudança altera profundamente o ciclo de vida do PipeWire e a dinâmica de enqueue/dequeue de buffers das streams. Considerado demasiadamente invasivo para o estágio Release Candidate da versão 1.0.
- **Underrun/Ordem de Fogo:** É essencial garantir que o PipeWire reaja bem com um `process` vazio na stream de reprodução e que não falte buffer durante requisições de captura (Capture e Playback sendo linkadas e do mesmo grupo de latência).

---

## Idéia 3. SIMD F16C Weight Compression (VNNI-like)

Esforço: 🟡 Médio · Impacto: 🟢 Alto (Redução de L1 Cache Misses) · Risco: 🟡 Médio (Alterações na trait matemática e parsing)

### 3.1 Diagnóstico e Justificativa

O principal gargalo na avaliação de Redes Neurais densas é o Memory Bound (banda de memória), não o Compute Bound (cálculos matemáticos). A L1 Cache dos núcleos tem tamanho estrito (geralmente 32KB a 48KB para dados). Em redes grandes, os vetores de pesos excedem essa capacidade facilmente, forçando despejos constantes (evicções) para a L2 Cache, atrasando a CPU.

### 3.2 Implementação: Compressão f16 in-memory com expansão on-the-fly

- **Crate `half`**: Adicionar a dependência para lidar de forma nativa e agnóstica à arquitetura com a conversão escalar `f32 -> f16 -> f32` durante o carregamento de modelo (`nam_json.rs`, `namb.rs`).
- **SIMD Math**: Modificar a trait `SimdMath` e as assinaturas de operações de Produto Escalar (`dot_product`, `dot_product_4x`) para receber `&[u16]` (pesos em f16) em vez de `&[f32]`.
- **Descompressão intrínseca**: Nas funções AVX2 (`#[target_feature(enable = "avx2,fma,f16c")]`), usar `_mm_loadu_si128` seguido de `_mm256_cvtph_ps` em blocos de 8. No AVX-512, `_mm256_loadu_si256` seguido de `_mm512_cvtph_ps` em blocos de 16.
- **Modelos**: Em `Conv1d`, `DenseLayer`, e `LstmLayer`, mudar a tipagem dos arrays/vectors de `weights` e `bias` para usar `u16`. O `bias` será descomprimido via conversão de bits (`half::f16::from_bits(...).to_f32()`) nas partes escalares.

### 3.3 Riscos Críticos e Validação

- **Invasividade**: A alteração afeta a assinatura estrutural de todos os modelos `WaveNet` e `LSTM`, bem como da engine SIMD inteira. É por esta razão que foi adiado para 1.x ou 2.x, ao invés da versão inicial Release Candidate.
- **Validação Numérica**: Necessário testar extensivamente contra os "golden vectors" (testes do cargo) para certificar-se de que a perda de precisão associada ao f16 não degrada a taxa SNR do plugin ou provoca instabilidades no loop DSP recorrente (LSTM).

---

## Idéia 4. LSTM Batch Head Dot-Product (Mini-GEMM)

Esforço: 🟡 Médio · Impacto: 🟡 Médio (5-10% em LSTM benchmarks) · Risco: 🟡 Médio

### 4.1 Diagnóstico e Justificativa

O LSTM processa estritamente sample-by-sample (IIR: `t` depende de `t-1`), conforme documentado na decisão arquitetural em `architecture.md §2`. Porém, dentro de cada sample, a projeção **head dot_product** (`dot_product_avx2(&self.head_weights, hidden)` em `lstm.rs`) é invocada N vezes independentes.

Atualmente, cada invocação paga o custo de setup de registradores YMM para `head_weights` (load constants) e o overhead de loop do `dot_product`. Para N=64 frames, são 64 setups redundantes dos mesmos pesos.

### 4.2 Implementação Proposta

1. Acumular os hidden states de K samples (4-8) num buffer temporário stack-allocated (`[f32; K * H]`, ~128 bytes para K=8, H=16 — cabe inteiramente em L1).
2. Processar o head dot_product como uma mini-GEMM de K×H → K×1, permitindo reutilização dos pesos em registradores YMM durante K iterações.
3. O loop externo em `LstmModel1::process_avx2/512` ficaria:
   - Para cada batch de K samples: executar LSTM sample-by-sample (acumulando hidden states)
   - Executar head dot_product sobre o batch inteiro (1 load de pesos, K FMAs)

### 4.3 Riscos e Validação

- Requer reestruturação dos loops em `LstmModel1::process_avx2` e `LstmModel2::process_avx2` com cuidado para preservar a semântica sample-by-sample do estado recorrente.
- Golden vectors existentes devem continuar passando (mesma saída numérica).
- Validar com `cargo bench --bench inference_bench -- Lstm` para medir o ganho real.

---

## Idéia 5. Prefetch Adaptativo por Dilatação (WaveNet Conv1D)

Esforço: 🟢 Baixo · Impacto: Variável (dependente do hardware) · Risco: 🟡 Médio

### 5.1 Diagnóstico e Justificativa

O `_mm_prefetch` em `wavenet.rs Conv1d::process_single_frame` usa stride fixo de `+1 tap` para o prefetch do próximo tap. Para dilatações baixas (1, 2, 4), o hardware prefetcher do processador já resolve os acessos sequenciais e o software prefetch é redundante. Para dilatações altas (256, 512), o stride é tão grande que o `_MM_HINT_T0` pode trazer a cache line cedo demais (evicção antes do uso) ou gerar I-Cache pollution desnecessária.

### 5.2 Implementação Proposta

Prefetch adaptativo baseado no campo `dilation` da `Conv1d`:

- **Dilatação ≤ 16:** Omitir o prefetch (o hardware prefetcher detecta o padrão strided para strides pequenos).
- **Dilatação ≥ 32 e < 256:** Manter `_MM_HINT_T0` (L1), comportamento atual.
- **Dilatação ≥ 256:** Usar `_MM_HINT_T1` (L2 em vez de L1) para evitar poluição da L1. A L2 tem mais capacidade e tolera melhor o prefetch agressivo.

### 5.3 Riscos e Validação

- Pode degradar CPUs com hardware prefetchers menos agressivos (laptops com menos unidades de memória).
- Requer `cargo bench` em hardware real com modelos Standard (dilation 512) vs Lite (dilation 16).
- Considerar tornar o comportamento configurável via `const generic` ou parâmetro do modelo.

---

## Idéia 6. Eliminação do `powf` no Callback RT (Gain Staging LUT)

Esforço: 🟢 Baixo · Impacto: 🟡 Médio (latência de I-Cache) · Risco: 🟡 Médio

### 6.1 Diagnóstico e Justificativa

A função `update_gain_multipliers` em `pw_host.rs` calcula `10.0f32.powf(total_db / 20.0)` quando parâmetros de ganho mudam. O `powf` é uma chamada para `libm` que pode levar 100-200 ciclos. Embora protegida por `if param_changed` (evento raro), a mera presença de `powf` importa o corpo da função `libm` na I-Cache do callback, ocupando espaço que poderia ser usado pelo código hot-path.

### 6.2 Implementação Proposta

**Opção A — LUT Pré-computada:**

- Tabela estática `const` de ~120 entradas: `DB_TO_LINEAR_LUT[i] = 10.0^((i - 60) / 20.0)` para i ∈ [0, 120] cobrindo −60dB a +60dB.
- Interpolação linear entre pontos adjacentes para valores fracionários.
- Custo: 1 load + 1 FMA vs 200 ciclos de `powf`.

**Opção B — Polinômio Rápido `exp2`:**

- Usar a identidade `10^(x/20) = 2^(x * log2(10) / 20)` e implementar um polinômio rápido `exp2_fast(x)` baseado na mesma técnica de Minimax já usada em `fastmath.rs`.
- Custo: ~5-8 instruções FMA em registrador vs chamada `libm`.

### 6.3 Riscos e Validação

- A LUT requer validação de precisão nos limites (−60dB, +60dB). Tolerância aceitável: < 0.001 dB.
- A Opção B requer polinômio de grau 3-4 para precisão suficiente no range [−3.0, +3.0] (cobrindo ±60dB/20).
- Testar com `cargo test -- test_combined_gain_staging` e `test_extreme_gain_values`.

---

## Idéia 7. AVX-512 VNNI/BF16 para Dot Product Interleaved (LSTM)

Esforço: 🔴 Alto · Impacto: 🟢 Alto (30-50% teórico em LSTM) · Risco: 🔴 Alto

> **Cruza com Idéia 3 (F16C Weight Compression).** A implementação VNNI requer pesos em formato BF16, portanto é um super-set natural da Idéia 3.

### 7.1 Diagnóstico e Justificativa

O `dot_product_4x_interleaved_avx2` em `simd.rs` usa `_mm256_broadcast_ss` + `_mm256_blend_ps` para construir pares de state values antes do FMA. Em AVX-512 com VNNI (Intel Tiger Lake+, AMD Zen4+), o `_mm512_dpbf16_ps` pode processar dot-products de 2×BF16 acumulando em FP32 nativamente — potencialmente dobrando o throughput.

### 7.2 Implementação Proposta

1. Adicionar target feature `avx512bf16` + `avx512vnni` no `SimdMathConfig` via multiversioning.
2. Converter os pesos LSTM interleaved de `[f32; 4]` para `[bf16; 4]` no loader.
3. Implementar `dot_product_4x_interleaved_vnni` que usa `_mm512_dpbf16_ps` diretamente.
4. O dispatch via `LazyLock` v-table selecionaria automaticamente o path VNNI quando disponível.

### 7.3 Riscos e Validação

- Requer features `avx512bf16` + `avx512vnni` (hardware 2020+: Intel Tiger Lake, AMD Zen4).
- BF16 tem apenas 7 bits de mantissa (vs 10 do FP16). Testar extensivamente a estabilidade do loop LSTM recorrente com pesos BF16 — o erro pode acumular de forma catastrófica em redes profundas.
- Considerar BF16 apenas para pesos (não para ativações/states), preservando FP32 nos acumuladores.
- Golden vectors precisariam de versões BF16-specific ou tolerâncias relaxadas.
