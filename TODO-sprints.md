# 🚀 Backlog do Produto — Versão 1.1

## 📊 Dashboard de Prioridades (Quick View)

| ID     | Iniciativa                           | Impacto  | Esforço  | Risco    | Prioridade | Status   | Tags              |
|:------ |:------------------------------------ |:--------:|:--------:|:--------:|:----------:|:--------:|:----------------- |
| **T1** | Suporte completo a processadores     | 🟢 Alto  | 🔴 Alto  | 🟡 Médio | **P1**     | No Radar | #simd #cpu        |
| **T2** | Monomorfização de Modelos (Enums)    | 🟢 Alto  | 🟡 Médio | 🟢 Baixo | **P1**     | No Radar | #simd #rust       |
| **T3** | SIMD F16C Weight Compression         | 🟢 Alto  | 🟡 Médio | 🟡 Médio | **P1**     | No Radar | #simd #model      |
| **T4** | Fast Gain LUT (Eliminação do `powf`) | 🟡 Médio | 🟢 Baixo | 🟡 Médio | **P1**     | No Radar | #dsp #performance |
| **T5** | Resampler Sinc-SIMD & Fase Mínima    | 🟢 Alto  | 🔴 Alto  | 🔴 Alto  | **P2**     | No Radar | #dsp #latency     |
| **T6** | AVX-512 VNNI/BF16 Interleaved        | 🟢 Alto  | 🔴 Alto  | 🔴 Alto  | **P2**     | No Radar | #simd #avx512     |
| **T7** | LSTM Batch Head Dot-Product          | 🟡 Médio | 🟡 Médio | 🟡 Médio | **P2**     | No Radar | #simd #lstm       |
| **T8** | Fusão Residual + 1x1 (DenseLayer)    | 🟡 Médio | 🟡 Médio | 🟡 Médio | **P2**     | No Radar | #simd #wavenet    |
| **T9** | Non-Temporal Bridge Copy             | 🟡 Médio | 🟢 Baixo | 🟡 Médio | **P2**     | No Radar | #performance #l1  |
| **T10**| Timing de Baixa Latência (RDTSC)     | 🟡 Médio | 🟢 Baixo | 🟢 Baixo | **P2**     | No Radar | #performance      |
| **T11**| Resampler Short-Circuit (Bypass)     | 🟡 Médio | 🟢 Baixo | 🟢 Baixo | **P2**     | No Radar | #dsp #resampler   |
| **T12**| Zero-Copy DspBridge (In-Place)       | 🟡 Médio | 🟡 Médio | 🔴 Alto  | **P3**     | No Radar | #pipewire #arch   |
| **T13**| Prefetch Adaptativo por Dilatação    | 🟡 Médio | 🟢 Baixo | 🟡 Médio | **P3**     | No Radar | #simd #l1         |
| **T14**| Histerese em Otimizações de Sinal    | 🟢 Baixo | 🟢 Baixo | 🟢 Baixo | **P3**     | No Radar | #dsp              |

---

## 🏛️ Épico: Otimização de Modelos & Inferência

*Foco: Reduzir a pegada de memória e aumentar o throughput dos núcleos neurais.*

### [T1] Suporte completo a processadores (AVX2 / AVX-512)

- **Problema:** Fragmentação de performance em hardware heterogêneo; falta de suporte explícito a instruções modernas.
- **Solução:** Maximizar a quantidade de código otimizado para AVX2-VNNI e AVX-512 via multiversioning.
- **Critérios de Aceite:** Binário otimizado detectando e usando o set de instruções mais rápido disponível no host.
- **Tags:** #simd #cpu #x86

### [T2] Monomorfização de Modelos (Static Dispatch via Enums)

- **Problema:** Overhead de vtable e falha de inlining devido ao `Box<dyn DynamicModel>` no hot-path.
- **Solução:** Substituir o trait object por um `enum` que encapsule os tipos de modelos, permitindo despacho estático.
- **Critérios de Aceite:** Redução do overhead de chamada e permissão para o compilador inlinar o kernel DSP.
- **Tags:** #simd #rust #performance

### [T3] SIMD F16C Weight Compression (VNNI-like)

- **Problema:** Gargalo de Memory Bound (L1 Cache Misses) em redes grandes.
- **Solução:** Implementar compressão f16 in-memory com expansão on-the-fly usando `_mm256_cvtph_ps`.
- **Critérios de Aceite:** Redução de cache misses na L1 e passagem nos Golden Vectors de v1.0.
- **Tags:** #simd #model #memory

### [T7] LSTM Batch Head Dot-Product (Mini-GEMM)

- **Problema:** Overhead redundante de setup de registradores para o head dot-product em cada sample.
- **Solução:** Acumular hidden states de K samples e processar como mini-GEMM (1 load de pesos para K FMAs).
- **Critérios de Aceite:** Ganho de 5-10% em benchmarks de LSTM.
- **Tags:** #simd #lstm

### [T6] AVX-512 VNNI/BF16 para Dot Product Interleaved

- **Problema:** Throughput limitado pelo FMA FP32 tradicional em hardwares modernos.
- **Solução:** Usar `_mm512_dpbf16_ps` para processar dot-products de 2×BF16 nativamente.
- **Critérios de Aceite:** Suporte via dispatch `LazyLock` e estabilidade numérica validada.
- **Tags:** #simd #avx512

### [T8] Fusão Residual + One-by-One no WaveNet Layer

- **Problema:** Passes separados de 1x1 e soma residual gerando excesso de loads/stores.
- **Solução:** Novo método `process_single_frame_with_residual` no `DenseLayer` fundindo as operações.
- **Critérios de Aceite:** Redução mensurável de ciclos no hot-path da WaveNet.
- **Tags:** #simd #wavenet

### [T13] Prefetch Adaptativo por Dilatação (WaveNet Conv1D)

- **Problema:** Prefetch fixo pode ser redundante para dilatações baixas ou poluir L1 para dilatações altas.
- **Solução:** Ajustar o hint de prefetch (`T0` vs `T1`) baseado no valor da dilatação.
- **Critérios de Aceite:** Sem regressões de performance em CPUs mobile.
- **Tags:** #simd #l1

---

## 🎸 Épico: Motor DSP & Performance de Baixo Nível

*Foco: Precisão audível, latência mínima e eficiência do ciclo RT.*

### [T5] Resampler Sinc-SIMD Nativo & Fase Mínima

- **Problema:** Latência de ~1.5ms e pre-ringing causados pelo resampler de fase linear da `rubato`.
- **Solução:** Resampler polifásico FIR customizado com SIMD nativo e suporte a Fase Mínima.
- **Critérios de Aceite:** Redução da latência roundtrip e zero alocações no loop.
- **Tags:** #dsp #latency #simd

### [T4] Eliminação do `powf` no Callback RT (Gain Staging LUT)

- **Problema:** Chamadas para `libm` (powf) poluem a I-Cache do callback e custam ~200 ciclos.
- **Solução:** Implementar LUT estática com interpolação linear ou polinômio rápido `exp2`.
- **Critérios de Aceite:** Precisão de ganho < 0.001 dB e remoção do `powf` do hot-path.
- **Tags:** #dsp #performance

### [T9] Playback Bridge SIMD Copy (Non-Temporal Stores)

- **Problema:** Cópia de saída polui L1 com dados que não serão reusados pela CPU.
- **Solução:** Usar `_mm256_stream_ps` para buffers ≥ 64 samples para bypass de cache.
- **Critérios de Aceite:** Ganho marginal em quantums altos (512+).
- **Tags:** #performance #l1

### [T10] Timing de Baixa Latência (RDTSC / `minstant`)

- **Problema:** `Instant::now()` pode causar syscalls/VDSO polpudos em loops de alta frequência.
- **Solução:** Usar contadores de ciclo de hardware (RDTSC) para monitoramento de carga DSP.
- **Critérios de Aceite:** Redução do jitter de medição e overhead desprezível no callback.
- **Tags:** #performance #lowlevel

### [T11] Resampler Short-Circuit Bypass (Zero-Copy)

- **Problema:** Cópia desnecessária de buffers mesmo quando `pw_rate == nam_rate`.
- **Solução:** Implementar bypass direto que use os slices originais do PipeWire sem passar pelo buffer do resampler.
- **Critérios de Aceite:** Ganho de performance em sessões nativas de 48kHz.
- **Tags:** #dsp #resampler

### [T14] Histerese Dinâmica em Otimizações de Sinal (Mono/Silêncio)

- **Problema:** Re-teste constante de condições de sinal (mono/silêncio) em cada buffer consome ciclos redundantes.
- **Solução:** Implementar máquina de estado com histerese (mínimo de N buffers) para manter otimizações ativas.
- **Critérios de Aceite:** Redução de oscilação de modo e economia de ciclos de comparação.
- **Tags:** #dsp #optimization

---

## 🌐 Épico: Arquitetura de Áudio & PipeWire

*Foco: Simplificação da infraestrutura de roteamento.*

### [T12] Zero-Copy DspBridge (Processamento In-Place)

- **Problema:** Overhead de cópia entre as streams de captura e reprodução via `DspBridge`.
- **Solução:** Modificar a macro-arquitetura do PipeWire para processamento in-place na thread de captura.
- **Critérios de Aceite:** Eliminação do buffer intermediário e redução do overhead de inter-stream sync.
- **Tags:** #pipewire #arch #latency
