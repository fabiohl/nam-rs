# Ideias de pesquisa para as próximas versões 1.x e/ou 2.x

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
