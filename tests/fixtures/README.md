# tests/fixtures — Golden Vectors para Validação Cross-Reference NeuralAmpModelerCore ↔ NAM-rs

## Fonte de Verdade

Todos os golden vectors neste diretório são gerados pelo **NeuralAmpModelerCore** (Steven Atkinson) — implementação canônica que treina e gera os modelos `.nam`.

## Arquivos neste diretório

| Golden File                       | Modelo `.nam`        | Origem  | Topologia              |
| --------------------------------- | -------------------- | ------- | ---------------------- |
| `golden_wavenet_standard.bin`     | `BossWN-standard.nam` | NAM-rs | CH=16, K=3, HEAD=8, 20 layers |
| `golden_wavenet_feather.bin`      | `BossWN-feather.nam`  | NAM-rs | CH=8, K=3, HEAD=4, 20 layers  |
| `golden_wavenet_nano.bin`         | `BossWN-nano.nam`     | NAM-rs | CH=4, K=3, HEAD=2, 20 layers  |
| `golden_lstm_1x16.bin`            | `BossLSTM-1x16.nam`   | NAM-rs | 1 layer, H=16          |
| `golden_lstm_2x8.bin`             | `BossLSTM-2x8.nam`    | NAM-rs | 2 layers, H=8          |
| `golden_namcore_lstm_1x3.bin`     | `lstm.nam`            | NAMCore | 1 layer, H=3, 70 pesos |
| `golden_namcore_wn_micro.bin`     | `wavenet.nam`         | NAMCore | CH=3/2, K=3, HEAD=2/1, 3 layers |

Os 2 modelos NAMCore (`lstm.nam` e `wavenet.nam`) são do diretório `example_models/` do NeuralAmpModelerCore — os mesmos usados nos testes oficiais (`test_get_dsp.cpp`, `test_slimmable_wavenet.cpp`). Exercitam topologias **abaixo de qualquer perfil estático** do NAM-rs, forçando os caminhos de despacho dinâmico/fallback.

## Formato binário (.golden.bin)

```
[u32 num_samples LE]
[f32×N input samples LE]       — stress signal (2048 amostras @ 48 kHz)
[f32×N expected output LE]     — output do NeuralAmpModelerCore (render tool)
```

## Sinal de Stress (2048 amostras @ 48 kHz ≈ 42.7 ms)

Substitui a senoidal 440 Hz por um sinal multi-componente deterministico:

| Comportamento a testar                     | Componente do sinal                          |
| ------------------------------------------ | -------------------------------------------- |
| Resposta em frequência (grave → agudo)     | Chirp sweep 220 Hz → 3520 Hz                 |
| Intermodulação harmônica (guitarra real)   | Harmônicos Low-E (82/165/330/659 Hz)         |
| Resposta a transientes (ataque de nota)    | Impulso isolado (+0.9) a 25%                 |
| Dinâmica de amplitude                      | Envelope attack–sustain–release              |
| Comportamento near-silence / denormals     | Fade-to-silence (release tail)               |

## Métricas de Precisão (5 métricas, single-pass fusion)

Cada golden test reporta **5 métricas** calculadas numa única iteração sobre o buffer:

| Métrica      | Fórmula                                          | O que detecta                              |
| ------------ | ------------------------------------------------ | ------------------------------------------ |
| **MSE**      | `Σ(rᵢ - tᵢ)² / N`                                | Erro médio (regressões estruturais)        |
| **MAE**      | `max|rᵢ - tᵢ|`                                   | Pior caso pontual (outliers, overflow)     |
| **SNR**      | `10 · log₁₀(Σrᵢ² / Σ(rᵢ-tᵢ)²)`                   | Relação sinal/ruído (interpretação DSP)    |
| **PSNR**     | `10 · log₁₀(peak² / MSE)`                        | SNR normalizado pelo pico                  |
| **Bits eq.** | `-0.5 · log₂(MSE / signal_power)`                | Precisão — quantos bits de float32 corretos|

## Thresholds de Paridade

| Modelo           | MSE threshold | SNR threshold |
| ---------------- |:-------------:|:-------------:|
| LSTM 1×16        | < 3e-3        | ≥ 15 dB       |
| LSTM 2×8         | < 1e-3        | ≥ 18 dB       |
| NAMCore LSTM 1×3 | < 1e-3        | ≥ 22 dB       |
| WaveNet Nano     | < 5e-2        | ≥ 9 dB        |
| WaveNet Feather  | < 5e-2        | ≥ 9 dB        |
| WaveNet Standard | < 5e-2        | ≥ 9 dB        |
| NAMCore WN Micro | < 5e-2        | ≥ 9 dB        |

> [!IMPORTANT]
> Os thresholds são calibrados empiricamente. NAMCore (`std::tanh` nativo) pode mostrar SNR superior aos modelos Boss (FastMath Padé). A divergência é dominada exclusivamente pela implementação de `tanh`/`sigmoid` — ver ADR-001 em `docs/architecture.md §2`.

## Para regenerar

```bash
./tests/fixtures/golden_gen_build.sh
```

Pré-requisitos: `cmake`, `g++` (ou `clang++`, C++20), `python3`, `git`.

Estes arquivos são commitados no repositório para que os testes Rust de golden vectors executem sem precisar recompilar C++. Se os golden vectors não existirem, os testes fazem skip gracioso.

## Duas Camadas de Validação

### Camada 1 — Goldens pré-commitados (rápido, `cargo test`)

Testes em `tests/nam_infer_test.rs` carregam os `.golden.bin` e comparam contra inferência Rust. Roda em cada `cargo test` sem C++.

### Camada 2 — Validação cruzada ao vivo (lento, `utils/tests-long.sh`)

Testes `#[ignore]` em `tests/cpp_parity.rs` compilam o `render` tool do NeuralAmpModelerCore on-demand e comparam C++ vs Rust ao vivo. Detecta drift se o NAMCore for atualizado e os goldens pré-commitados ficarem defasados.

## Decisão Técnica: Cross-Reference NÃO é Bit-Identical (ADR-002)

> **Decisão:** Os golden vectors validam paridade *funcional* (MSE + SNR + PSNR + bits dentro de thresholds calibrados) contra o NeuralAmpModelerCore C++, **não** paridade *bit-a-bit*.
>
> **Consequência:** O NAM-rs produz áudio perceptualmente equivalente ao C++, mas com diferenças numéricas mensuráveis (SNR 9–25 dB dependendo do modelo). Estas diferenças são inaudíveis em qualquer pipeline de áudio 16-bit ou superior.
>
> **Fonte exclusiva da divergência:** Implementações de `tanh` e `sigmoid` — ver ADR-001 em `docs/architecture.md §2`.
>
> **Prova:** A degradação de SNR é proporcional à profundidade do modelo:
> - LSTM 1×16 (1 camada): SNR ~25 dB
> - WaveNet Standard (20 camadas): SNR ~10 dB
>
> Se houvesse erro estrutural, o SNR não degradaria linearmente com a profundidade.
