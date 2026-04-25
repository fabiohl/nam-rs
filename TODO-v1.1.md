# 🚀 Backlog do Produto — Versão 1.1 (Próxima Temporada)

## 📊 Dashboard de Prioridades (Quick View)

| ID     | Iniciativa                           | Impacto  | Esforço  | Risco    | Prioridade | Status   | Tags              |
|:------ |:------------------------------------ |:--------:|:--------:|:--------:|:----------:|:--------:|:----------------- |
| **T3** | SIMD F16C Weight Compression         | 🟢 Alto  | 🟡 Médio | 🟡 Médio | **P1**     | No Radar | #simd #model      |
| **T6** | Fast Gain LUT (Eliminação do `powf`) | 🟡 Médio | 🟢 Baixo | 🟡 Médio | **P1**     | No Radar | #dsp #performance |
| **T1** | Resampler Sinc-SIMD & Fase Mínima    | 🟢 Alto  | 🔴 Alto  | 🔴 Alto  | **P2**     | No Radar | #dsp #latency     |
| **T4** | LSTM Batch Head Dot-Product          | 🟡 Médio | 🟡 Médio | 🟡 Médio | **P2**     | No Radar | #simd #lstm       |
| **T7** | AVX-512 VNNI/BF16 Interleaved        | 🟢 Alto  | 🔴 Alto  | 🔴 Alto  | **P2**     | No Radar | #simd #avx512     |
| **T8** | Fusão Residual + 1x1 (DenseLayer)    | 🟡 Médio | 🟡 Médio | 🟡 Médio | **P2**     | No Radar | #simd #wavenet    |
| **T9** | Non-Temporal Bridge Copy             | 🟡 Médio | 🟢 Baixo | 🟡 Médio | **P2**     | No Radar | #performance #l1  |
| **T2** | Zero-Copy DspBridge (In-Place)       | 🟡 Médio | 🟡 Médio | 🔴 Alto  | **P3**     | No Radar | #pipewire #arch   |
| **T5** | Prefetch Adaptativo por Dilatação    | 🟡 Médio | 🟢 Baixo | 🟡 Médio | **P3**     | No Radar | #simd #l1         |

---

## 🏛️ Épico: Otimização de Modelos & Inferência

*Foco: Reduzir a pegada de memória e aumentar o throughput dos núcleos neurais.*

### [T3] SIMD F16C Weight Compression (VNNI-like)

- **Problema:** Gargalo de Memory Bound (L1 Cache Misses) em redes grandes.
- **Solução:** Implementar compressão f16 in-memory com expansão on-the-fly usando `_mm256_cvtph_ps`.
- **Critérios de Aceite:** Redução de cache misses na L1 e passagem nos Golden Vectors de v1.0.
- **Tags:** #simd #model #memory

### [T4] LSTM Batch Head Dot-Product (Mini-GEMM)

- **Problema:** Overhead redundante de setup de registradores para o head dot-product em cada sample.
- **Solução:** Acumular hidden states de K samples e processar como mini-GEMM (1 load de pesos para K FMAs).
- **Critérios de Aceite:** Ganho de 5-10% em benchmarks de LSTM.
- **Tags:** #simd #lstm

### [T7] AVX-512 VNNI/BF16 para Dot Product Interleaved

- **Problema:** Throughput limitado pelo FMA FP32 tradicional em hardwares modernos.
- **Solução:** Usar `_mm512_dpbf16_ps` para processar dot-products de 2×BF16 nativamente.
- **Critérios de Aceite:** Suporte via dispatch `LazyLock` e estabilidade numérica validada.
- **Tags:** #simd #avx512

### [T8] Fusão Residual + One-by-One no WaveNet Layer

- **Problema:** Passes separados de 1x1 e soma residual gerando excesso de loads/stores.
- **Solução:** Novo método `process_single_frame_with_residual` no `DenseLayer` fundindo as operações.
- **Critérios de Aceite:** Redução mensurável de ciclos no hot-path da WaveNet.
- **Tags:** #simd #wavenet

### [T5] Prefetch Adaptativo por Dilatação (WaveNet Conv1D)

- **Problema:** Prefetch fixo pode ser redundante para dilatações baixas ou poluir L1 para dilatações altas.
- **Solução:** Ajustar o hint de prefetch (`T0` vs `T1`) baseado no valor da dilatação.
- **Critérios de Aceite:** Sem regressões de performance em CPUs mobile.
- **Tags:** #simd #l1

---

## 🎸 Épico: Motor DSP & Performance de Baixo Nível

*Foco: Precisão audível, latência mínima e eficiência do ciclo RT.*

### [T1] Resampler Sinc-SIMD Nativo & Fase Mínima

- **Problema:** Latência de ~1.5ms e pre-ringing causados pelo resampler de fase linear da `rubato`.
- **Solução:** Resampler polifásico FIR customizado com SIMD nativo e suporte a Fase Mínima.
- **Critérios de Aceite:** Redução da latência roundtrip e zero alocações no loop.
- **Tags:** #dsp #latency #simd

### [T6] Eliminação do `powf` no Callback RT (Gain Staging LUT)

- **Problema:** Chamadas para `libm` (powf) poluem a I-Cache do callback e custam ~200 ciclos.
- **Solução:** Implementar LUT estática com interpolação linear ou polinômio rápido `exp2`.
- **Critérios de Aceite:** Precisão de ganho < 0.001 dB e remoção do `powf` do hot-path.
- **Tags:** #dsp #performance

### [T9] Playback Bridge SIMD Copy (Non-Temporal Stores)

- **Problema:** Cópia de saída polui L1 com dados que não serão reusados pela CPU.
- **Solução:** Usar `_mm256_stream_ps` para buffers ≥ 64 samples para bypass de cache.
- **Critérios de Aceite:** Ganho marginal em quantums altos (512+).
- **Tags:** #performance #l1

---

## 🌐 Épico: Arquitetura de Áudio & PipeWire

*Foco: Simplificação da infraestrutura de roteamento.*

### [T2] Zero-Copy DspBridge (Processamento In-Place)

- **Problema:** Overhead de cópia entre as streams de captura e reprodução via `DspBridge`.
- **Solução:** Modificar a macro-arquitetura do PipeWire para processamento in-place na thread de captura.
- **Critérios de Aceite:** Eliminação do buffer intermediário e redução do overhead de inter-stream sync.
- **Tags:** #pipewire #arch #latency
