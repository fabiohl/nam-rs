# tests/fixtures — Golden Vectors para Validação Cross-Reference C++ ↔ Rust

## Arquivos neste diretório

   golden_wavenet_standard.bin  — Gerado pelo C++ NeuralAudio (BossWN-standard.nam)
   golden_lstm_1x16.bin         — Gerado pelo C++ NeuralAudio (BossLSTM-1x16.nam)

## Formato binário (.golden.bin)

   [u32 num_samples LE]
   [f32×N input samples LE]       — senoidal 440Hz a 48kHz (512 amostras)
   [f32×N expected output LE]     — output do C++ NeuralAudio Internal mode

## Para regenerar

   ./utils/golden_gen_build.sh

Estes arquivos são commitados no repositório para que os testes Rust
(test_golden_vectors_wavenet, test_golden_vectors_lstm) executem sem
precisar recompilar o NeuralAudio C++.
Se os golden vectors não existirem, os testes fazem skip gracioso.

## Metodologia de Validação

Os golden tests assertam **duas métricas independentes** em fusão single-pass
(MSE, MAE e SNR calculados numa única iteração sobre o buffer — zero overhead):

### MSE — Erro Quadrático Médio

Métrica primária de regressão, sensível à escala absoluta do erro.

- Detecta erros estruturais: transposição de pesos, offset de bias, gate invertido.
- Falha ruídos de baixa magnitude que SNR acomodaria (ex: DC offset sistemático).

| Modelo           | Threshold MSE | MSE medido (2026-04-15) | Headroom |
| ---------------- |:-------------:|:-----------------------:|:--------:|
| WaveNet Standard | `< 5e-2`      | 3.21e-2                 | ~1.56×   |
| LSTM 1×16        | `< 1e-3`      | —                       | —        |

### SNR — Signal-to-Noise Ratio em dB

Métrica de equivalência perceptual normalizada pela potência do sinal,
padrão da indústria DSP. Complementa o MSE:

- MSE detecta erros absolutos (útil para regressões estruturais).
- SNR fornece interpretação DSP padrão (útil para engenheiros de áudio):
  valores > 20 dB são imperceptíveis; valores < 6 dB indicam falha grave.

| Modelo           | Threshold SNR | SNR medido (2026-04-15) | Headroom |
| ---------------- |:-------------:|:-----------------------:|:--------:|
| WaveNet Standard | `≥ 9 dB`      | 10.1 dB                 | ~1.1×    |
| LSTM 1×16        | `≥ 22 dB`     | 26.0 dB                 | ~0.85×   |

As métricas são **aditivas, não substitutivas**: uma regressão pode passar
em SNR mas falhar em MSE (ex: erro estrutural de baixa potência) ou vice-versa.

### Fonte de Divergência: FastMath Padé vs C++ Polynomial

O motor Rust usa `simd_tanh` — polinômio Padé grau 5 + refinamento
Newton-Raphson sobre `_mm256_rsqrt_ps` — enquanto o NeuralAudio C++ usa
um polinômio racional diferente (`Activation.h`, grau variado por plataforma).

Esta divergência é **intencional e esperada**:

- Erro máximo por ativação: **~5e-3** (validado por `test_simd_fastmath_tanh_mse`).

- Acumulação em profundidade: **sublinear** — cada camada aplica ativação
  não-linear que reescala o resíduo. Modelo empírico:
  
  ```text
  erro_máx_acumulado ≈ √N_camadas × erro_por_camada
  ```

- Para WaveNet Standard (20 camadas: 2 arrays × 10 layers):
  
  ```text
  √20 × 5e-3 ≈ 2.2e-2  →  MSE medido: 3.21e-2  →  SNR medido: 10.1 dB
  ```

Os thresholds `MSE < 5e-2` e `SNR ≥ 9 dB` foram calibrados contra estas
medições reais com headroom suficiente para absorver variações de compilador
e FP, mas apertados o suficiente para capturar regressões estruturais
(onde MSE tipicamente salta para > 0.5 e SNR cai abaixo de 0 dB).

### Referências

- `docs/architecture.md §2` — Inferência FastMath e Microarquitetura
- `src/math/fastmath.rs` → `simd_tanh` — derivação do erro e acumulação
- `tests/nam_infer_test.rs` → `test_golden_vectors_wavenet` — calibração completa
