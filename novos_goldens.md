# S13.T01 — Suite de Cross-Validation NAM-rs ↔ NeuralAmpModelerCore

## Plano de Implementação — Revisão Final

---

## 1. Escopo da Mudança

### O que ENTRA

- **Fonte única de verdade**: NeuralAmpModelerCore (Steven Atkinson) — código canônico que treina e gera os modelos `.nam`.
- **Sinal de stress multi-componente**: chirp + harmônicos de guitarra + impulsos + fade-to-silence (substitui a senoidal simples 440 Hz).
- **Métricas de precisão expandidas**: MSE, MAE, SNR, PSNR e **bits equivalentes de precisão** — reportadas em cada golden test.
- **Duas camadas de validação**: goldens pré-commitados (rápido, `cargo test`) + validação cruzada ao vivo (lento, `tests-long.sh`).
- **Atualização completa de documentação**: architecture.md, dependencies.md, README.md, fixtures/README.md.

### O que SAI (git rm)

| Artefato                                     | Tipo                               | Ação       |
| -------------------------------------------- | ---------------------------------- | ---------- |
| `tests/regression_goldens.rs`                | Teste autorreferencial (Rust-only) | **DELETE** |
| `tests/golden/*.bin` (7 arquivos)            | Goldens autorreferenciais          | **DELETE** |
| `tests/golden/` (diretório)                  | Container                          | **DELETE** |
| `tests/fixtures/golden_gen.cpp`              | Gerador C++ NeuralAudio            | **DELETE** |
| `tests/fixtures/golden_wavenet_standard.bin` | Golden NeuralAudio                 | **DELETE** |
| `tests/fixtures/golden_wavenet_feather.bin`  | Golden NeuralAudio                 | **DELETE** |
| `tests/fixtures/golden_wavenet_nano.bin`     | Golden NeuralAudio                 | **DELETE** |
| `tests/fixtures/golden_lstm_1x16.bin`        | Golden NeuralAudio                 | **DELETE** |

---

## 2. Design do Sinal de Stress

### Motivação

A senoidal pura 440 Hz é um sinal trivial: frequência única, amplitude constante, sem transientes. Exercita apenas um ponto de operação da rede neural. Para maximizar a cobertura da validação, o sinal deve estressar:

| Comportamento a testar                            | Componente do sinal                          |
| ------------------------------------------------- | -------------------------------------------- |
| Resposta em frequência (grave → agudo)            | Chirp sweep 82 Hz → 3520 Hz                  |
| Intermodulação harmônica (como uma guitarra real) | Harmônicos da nota Low-E (82/165/330/659 Hz) |
| Resposta a transientes (ataque de nota)           | Impulso isolado (+0.9)                       |
| Dinâmica de amplitude                             | Envelope attack–sustain–release              |
| Comportamento near-silence / denormals            | Fade-to-silence (release tail)               |

### Definição Matemática

**2048 amostras @ 48 kHz ≈ 42.7 ms** — 4× o sinal anterior (512), mas golden files continuam pequenos (~16 KB cada).

```python
SR = 48000.0
N  = 2048
T  = N / SR  # duração total

for i in range(N):
    t   = i / SR
    pos = i / N       # 0.0 → 1.0 (posição normalizada)

    # ── Envelope (attack 2ms, sustain, release 5ms) ────────────────
    attack_end  = int(0.002 * SR)   # 96 amostras
    release_beg = N - int(0.005 * SR)  # 2048 - 240 = 1808
    if i < attack_end:
        env = i / attack_end
    elif i >= release_beg:
        env = (N - 1 - i) / (N - release_beg)
    else:
        env = 1.0

    # ── Componente 1: Harmônicos de guitarra (Low-E) ───────────────
    guitar = (0.40 * sin(2π * 82.41 * t)     # fundamental
            + 0.25 * sin(2π * 164.81 * t)    # 2ª harmônica
            + 0.15 * sin(2π * 329.63 * t)    # 4ª harmônica
            + 0.08 * sin(2π * 659.25 * t))   # 8ª harmônica

    # ── Componente 2: Chirp linear 220 Hz → 3520 Hz ───────────────
    #     f(t) = f0 + (f1 - f0) * t / T
    #     φ(t) = 2π * [f0 * t + (f1 - f0) * t² / (2*T)]
    f0, f1 = 220.0, 3520.0
    chirp_phase = 2π * (f0 * t + (f1 - f0) * t * t / (2.0 * T))
    chirp = 0.30 * sin(chirp_phase)

    # ── Componente 3: Impulso transiente no 25% ───────────────────
    impulse = 0.9 if (i == N // 4) else 0.0

    # ── Soma final ────────────────────────────────────────────────
    sample = clamp(env * (guitar + chirp) + impulse, -1.0, 1.0)
    output_f32(sample)
```

### Propriedades do sinal

- **Determinístico**: reprodutível bit-a-bit em Python e Rust (mesma fórmula, `f64` intermediário, cast final para `f32`).
- **Pico máximo**: ~1.0 (clamp garante [-1,1]) — evita clipping acidental no render C++.
- **Espectro**: cobre 82 Hz (Low-E) a 3520 Hz (nota A7, 2 oitavas acima do A4) — faixa completa de uma guitarra.
- **Duração**: 42.7 ms = ~2.7× o receptive field do WaveNet Standard (2046 samples ÷ 48000 = ~42.6 ms).

---

## 3. Métricas de Precisão Expandidas

### Métricas reportadas por teste

| Métrica         | Fórmula                                          | O que detecta                                                 |
| --------------- | ------------------------------------------------ | ------------------------------------------------------------- |
| **MSE**         | `Σ(rᵢ - tᵢ)² / N`                                | Erro médio (regressões estruturais)                           |
| **MAE**         | `max\|rᵢ - tᵢ\|`                                 | Pior caso pontual (outliers, overflow)                        |
| **SNR**         | `10 * log₁₀(Σrᵢ² / Σ(rᵢ-tᵢ)²)`                   | Relação sinal/ruído (interpretação DSP)                       |
| **PSNR**        | `10 * log₁₀(peak² / MSE)` com `peak = max\|rᵢ\|` | SNR normalizado pelo pico (comparável entre modelos)          |
| **Bits equiv.** | `-0.5 * log₂(MSE / signal_power)`                | Precisão intuitiva — "quantos bits de float32 estão corretos" |

### Formato de output do teste

```text
[NeuralAmpModelerCore × NAM-rs — BossWN-standard]
  MSE     = 3.21e-02      (threshold < 5.0e-02)  ✓
  MAE     = 2.84e-01
  SNR     = 10.1 dB       (threshold ≥ 9.0 dB)   ✓
  PSNR    = 14.9 dB
  Bits    = 2.5 bits equiv.
  Samples = 2048 @ 48 kHz (stress signal)
```

### Implementação: single-pass fusion

Todas as 5 métricas são calculadas em **uma única iteração** sobre o buffer (zero overhead adicional vs. a validação atual):

```rust
fn report_dsp_fidelity(reference: &[f32], test: &[f32], ...) {
    let mut sig_pow = 0.0f64;
    let mut noise_pow = 0.0f64;
    let mut max_abs_diff = 0.0f64;
    let mut peak_ref = 0.0f64;
    for (&r, &t) in reference.iter().zip(test.iter()) {
        let (r64, t64) = (r as f64, t as f64);
        let diff = r64 - t64;
        sig_pow += r64 * r64;
        noise_pow += diff * diff;
        max_abs_diff = max_abs_diff.max(diff.abs());
        peak_ref = peak_ref.max(r64.abs());
    }
    let n = reference.len() as f64;
    let mse  = noise_pow / n;
    let snr  = 10.0 * (sig_pow / noise_pow).log10();
    let psnr = 10.0 * (peak_ref * peak_ref / mse).log10();
    let bits = -0.5 * (mse / (sig_pow / n)).log2();
    // ... print + assert
}
```

> [!TIP]
> **Custo x benefício**: as 5 métricas custam exatamente 0 multiplicações extras — todas derivam dos mesmos 3 acumuladores (`sig_pow`, `noise_pow`, `max_abs_diff`) e 1 máximo (`peak_ref`). 2048 amostras × 5 operações = ~10μs no pior caso.

---

## 4. Arquitetura: Duas Camadas

### Camada 1 — Goldens NeuralAmpModelerCore pré-commitados (rápido)

- **Geração**: Script `tests/fixtures/golden_gen_build.sh` (reescrito) executado **uma única vez** pelo developer.
- **Pipeline**: Compila `render` → gera WAV stress → processa cada modelo → extrai f32 do WAV → salva como `.golden.bin`.
- **Saída**: 7 arquivos em `tests/fixtures/` (~16 KB cada, ~112 KB total).
- **Testes Rust**: `test_golden_vectors_*` em `nam_infer_test.rs` — roda em cada `cargo test` sem compilar C++.
- **Naming**: `golden_wavenet_standard.bin`, `golden_wavenet_feather.bin`, `golden_wavenet_nano.bin`, `golden_lstm_1x16.bin`, `golden_lstm_2x8.bin`, `golden_namcore_lstm_1x3.bin`, `golden_namcore_wn_micro.bin`.

### Camada 2 — Validação cruzada ao vivo (lento, `tests-long.sh`)

- **Testes**: `tests/cpp_parity.rs` com `#[test] #[ignore]`.
- **Pipeline**: Compila `render` on-demand → gera WAV stress ao vivo → executa render C++ → compara com Rust.
- **Quando roda**: `utils/tests-long.sh` ou `cargo test --test cpp_parity -- --ignored`.
- **Valor**: Detecta drift se o mirror `github.com/NeuralAmpModelerCore/` for atualizado e os goldens pré-commitados ficarem defasados.

---

## 5. Artefatos — Inventário Completo

### 5.1. DELETES

#### [DELETE] [regression_goldens.rs](file:///home/fabio/nam-rs/tests/regression_goldens.rs)

Testes autorreferenciais Rust-only (436 LoC). Substituídos pelos golden tests baseados em NeuralAmpModelerCore.

#### [DELETE] [tests/golden/](file:///home/fabio/nam-rs/tests/golden/) (diretório inteiro)

7 arquivos `.bin` (28 KB total). Goldens autorreferenciais.

#### [DELETE] [golden_gen.cpp](file:///home/fabio/nam-rs/tests/fixtures/golden_gen.cpp)

Gerador C++ NeuralAudio (141 LoC). Substituído pelo `render` tool do NeuralAmpModelerCore.

#### [DELETE] [golden_wavenet_standard.bin](file:///home/fabio/nam-rs/tests/fixtures/golden_wavenet_standard.bin), [golden_wavenet_feather.bin](file:///home/fabio/nam-rs/tests/fixtures/golden_wavenet_feather.bin), [golden_wavenet_nano.bin](file:///home/fabio/nam-rs/tests/fixtures/golden_wavenet_nano.bin), [golden_lstm_1x16.bin](file:///home/fabio/nam-rs/tests/fixtures/golden_lstm_1x16.bin)

Goldens NeuralAudio. Serão regenerados pelo NeuralAmpModelerCore (mesmos nomes, nova fonte de verdade).

---

### 5.2. CREATES

#### [NEW] [cpp_parity.rs](file:///home/fabio/nam-rs/tests/cpp_parity.rs)

Testes `#[test] #[ignore]` para validação cruzada ao vivo (Camada 2):

- Compila `render` tool on-demand (idempotente, cached).
- Gera WAV stress signal via `generate_stress_signal()`.
- Escreve WAV com `write_wav_f32()`, executa render via `std::process::Command`.
- Lê WAV output com `read_wav_f32()`.
- Compara C++ output vs Rust output com `report_dsp_fidelity()`.
- 5 testes: um por modelo de referência.

#### [NEW] [wav.rs](file:///home/fabio/nam-rs/tests/common/wav.rs)

Helpers minimalistas (~80 LoC) para WAV mono float32 IEEE:

- `write_wav_f32(path, samples, sample_rate)` — escreve WAV com header de 44 bytes + raw f32 LE.
- `read_wav_f32(path) -> Vec<f32>` — lê WAV mono float32, valida header, retorna samples.
- Sem crate externo. Formato fixo: 1 canal, 32-bit IEEE float, little-endian.

---

### 5.3. MODIFIES

#### [MODIFY] [golden_gen_build.sh](file:///home/fabio/nam-rs/tests/fixtures/golden_gen_build.sh)

**Reescrita completa** (~90 LoC) para usar NeuralAmpModelerCore:

1. Verifica pré-requisitos (`cmake`, `g++`/`clang++`, `python3`).
2. Inicializa submodules do NeuralAmpModelerCore (`Dependencies/eigen`, `Dependencies/AudioDSPTools`).
3. Compila `render` target via CMake (idempotente — skip se binário existe).
4. Gera WAV stress signal via Python one-liner embutido (mesma fórmula §2).
5. Executa `render` para cada modelo → produz WAV output.
6. Converte WAV output → `.golden.bin` (extrai header 44B, monta formato `[u32 N][f32×N in][f32×N out]`).
7. Adiciona `golden_lstm_2x8.bin` (novo — não existia).

#### [MODIFY] [mod.rs (tests/common)](file:///home/fabio/nam-rs/tests/common/mod.rs)

- `GOLDEN_NUM_SAMPLES`: 512 → **2048**.
- Renomear `generate_sine_440hz()` → **`generate_stress_signal()`** com a fórmula da §2.
- Manter `generate_sine_440hz()` como wrapper deprecated (usado em auto-consistência — estes testes não mudam).
- Expandir `assert_dsp_fidelity()` → **`report_dsp_fidelity()`** com as 5 métricas (MSE, MAE, SNR, PSNR, bits equivalentes).

#### [MODIFY] [nam_infer_test.rs](file:///home/fabio/nam-rs/tests/nam_infer_test.rs)

- Atualizar docstrings dos `test_golden_vectors_*` (referência muda de NeuralAudio → NeuralAmpModelerCore).
- Atualizar chamada de `assert_dsp_fidelity()` → `report_dsp_fidelity()`.
- Adicionar `test_golden_vectors_lstm_2x8` (novo modelo).
- Recalibrar thresholds MSE/SNR após regeneração dos goldens.
- Remover todas as referências a "NeuralAudio" e "golden_gen.cpp" nos comentários.

#### [MODIFY] [tests-long.sh](file:///home/fabio/nam-rs/utils/tests-long.sh)

Adicionar etapa de cross-validation live:

```bash
echo "=== Cross-Validation NAM-rs ↔ NeuralAmpModelerCore ==="
cargo test --test cpp_parity -- --ignored --nocapture
```

---

### 5.4. DOCUMENTAÇÃO

#### [MODIFY] [architecture.md](file:///home/fabio/nam-rs/docs/architecture.md)

**Seção 2 — ADR FastMath** (L32-46):

- Remover referência a "regression goldens self-reference (7 modelos, MSE < 1e-6)".
- Atualizar validação: "golden vectors cross-NeuralAmpModelerCore (5 modelos), PropTests (10k inputs)".

**Seção 6 — Estratégia de Testes** (L197-225):

- Tabela "Camadas Ativas" (L211-221):
  - **Remover** linha "Golden Vectors" com `regression_goldens.rs`.
  - **Reescrever** linha Golden Vectors: `tests/nam_infer_test.rs` + `tests/cpp_parity.rs` — "Ancoragem ao NeuralAmpModelerCore canônico".
- ADR "Remoção dos Parity Tests" (L223-225): Expandir para documentar a remoção dos goldens autorreferenciais e NeuralAudio.

**Seção 9 — Referências** (L319-327):

- Remover referência ao NeuralAudio como "Inspiração no suporte à arquitetura A1" (mover para nota histórica).

#### [MODIFY] [dependencies.md](file:///home/fabio/nam-rs/docs/dependencies.md)

Nova seção **"6. Dependências para Cross-Validation C++ (Opcional)"**:

```markdown
## 6. Dependências para Cross-Validation C++ (Opcional)

Para regenerar os golden vectors ou executar a validação cruzada ao vivo contra o
NeuralAmpModelerCore, os seguintes pacotes são necessários:

```bash
sudo apt install cmake g++ python3
```

- **cmake** (≥ 3.10): Build system do NeuralAmpModelerCore.
- **g++** (ou `clang++`, C++20): Compilador C++ para o tool `render`.
- **python3**: Geração do WAV de teste (sinal de stress).

> [!NOTE]
> Estas dependências são **opcionais**. Os golden vectors são pré-commitados no
> repositório e os testes de validação rodam sem C++ no `cargo test` normal.
> O C++ é necessário apenas para:
>
> - Regenerar goldens: `./tests/fixtures/golden_gen_build.sh`
> - Validação cruzada ao vivo: `./utils/tests-long.sh`

#### [MODIFY] [README.md](file:///home/fabio/nam-rs/README.md)

- Atualizar seção "Tests & Validation" com instrução de cross-validation.
- Na seção "Acknowledgments": manter crédito a Mike Oliphant mas remover referência a "golden vectors" / "cross-reference".

#### [MODIFY] [README.md (fixtures)](file:///home/fabio/nam-rs/tests/fixtures/README.md)

```text
**Reescrita completa** refletindo:

- Fonte de verdade: NeuralAmpModelerCore (não mais NeuralAudio).
- Sinal de stress (não mais senoidal 440 Hz).
- 7 modelos (5 Boss + 2 NAMCore upstream).
- Novas métricas (MSE, MAE, SNR, PSNR, bits equivalentes).
- ADR-002 atualizada.

---

## 6. Modelos de Referência

| Modelo           | Arquivo `.nam`        | Origem  | Topologia              | Profundidade | Golden                                   |
| ---------------- | --------------------- | ------- | ---------------------- | ------------ | ---------------------------------------- |
| WaveNet Standard | `BossWN-standard.nam` | NAM-rs  | CH=16, K=3, HEAD=8     | 20 layers    | `golden_wavenet_standard.bin`            |
| WaveNet Feather  | `BossWN-feather.nam`  | NAM-rs  | CH=8, K=3, HEAD=4      | 20 layers    | `golden_wavenet_feather.bin`             |
| WaveNet Nano     | `BossWN-nano.nam`     | NAM-rs  | CH=4, K=3, HEAD=2      | 20 layers    | `golden_wavenet_nano.bin`                |
| LSTM 1×16        | `BossLSTM-1x16.nam`   | NAM-rs  | 1 layer, H=16          | 1 layer      | `golden_lstm_1x16.bin`                   |
| LSTM 2×8         | `BossLSTM-2x8.nam`    | NAM-rs  | 2 layers, H=8          | 2 layers     | `golden_lstm_2x8.bin` (**novo**)         |
| NAMCore LSTM 1×3 | `lstm.nam`            | NAMCore | 1 layer, H=3, 70 pesos | 1 layer      | `golden_namcore_lstm_1x3.bin` (**novo**) |
| NAMCore WN Micro | `wavenet.nam`         | NAMCore | CH=3/2, K=3, HEAD=2/1  | 3 layers     | `golden_namcore_wn_micro.bin` (**novo**) |

> [!NOTE]
> Os 2 modelos NAMCore (`lstm.nam` e `wavenet.nam`) são do diretório `example_models/` do NeuralAmpModelerCore — os mesmos que Steven Atkinson usa nos testes oficiais (`test_get_dsp.cpp`, `test_slimmable_wavenet.cpp`). Exercitam topologias **abaixo de qualquer perfil estático** do NAM-rs (LSTM H=3 e WaveNet CH=3/2), forçando os caminhos de despacho dinâmico/fallback.

---

## 7. Thresholds de Paridade

Serão calibrados empiricamente após regeneração dos goldens. Estimativas conservadoras:

| Modelo           | MSE threshold | SNR threshold | Bits estimados |
| ---------------- |:-------------:|:-------------:|:--------------:|
| LSTM 1×16        | < 1e-3        | ≥ 22 dB       | ~5 bits        |
| LSTM 2×8         | < 1e-3        | ≥ 18 dB       | ~4 bits        |
| NAMCore LSTM 1×3 | < 1e-3        | ≥ 22 dB       | ~5 bits        |
| WaveNet Nano     | < 5e-2        | ≥ 9 dB        | ~2.5 bits      |
| WaveNet Feather  | < 5e-2        | ≥ 9 dB        | ~2.5 bits      |
| WaveNet Standard | < 5e-2        | ≥ 9 dB        | ~2.5 bits      |
| NAMCore WN Micro | < 5e-2        | ≥ 9 dB        | ~2.5 bits      |

> [!IMPORTANT]
> É possível que os thresholds NAMCore fiquem **melhores** que os NeuralAudio atuais, pois NAMCore usa `std::tanh` nativo (alta precisão). A divergência será dominada exclusivamente pela FastMath Padé do NAM-rs. Thresholds finais serão ajustados após primeira calibração.

---

## 8. Ordem de Execução

### Fase 1 — Remoção e limpeza

1. `git rm tests/regression_goldens.rs`
2. `git rm -r tests/golden/`
3. `git rm tests/fixtures/golden_gen.cpp`
4. `git rm tests/fixtures/golden_wavenet_standard.bin tests/fixtures/golden_wavenet_feather.bin tests/fixtures/golden_wavenet_nano.bin tests/fixtures/golden_lstm_1x16.bin`
5. Verificar `cargo test` compila (testes golden fazem skip gracioso se `.bin` ausente).

### Fase 2 — Novo gerador e sinal de stress

1. Reescrever `tests/fixtures/golden_gen_build.sh` (NeuralAmpModelerCore render).
2. Implementar `tests/common/wav.rs` (WAV I/O).
3. Implementar `generate_stress_signal()` em `tests/common/mod.rs`.
4. Implementar `report_dsp_fidelity()` expandido.

### Fase 3 — Geração e calibração

1. Executar `./tests/fixtures/golden_gen_build.sh` — gera os novos goldens.
2. Calibrar thresholds MSE/SNR contra os valores medidos.
3. Commitar os novos `.golden.bin`.

### Fase 4 — Atualização dos testes Rust

1. Atualizar `tests/nam_infer_test.rs` (docstrings, thresholds, novo LSTM 2×8).
2. Criar `tests/cpp_parity.rs` (validação ao vivo, Camada 2).
3. Atualizar `utils/tests-long.sh`.

### Fase 5 — Documentação

1. Atualizar `docs/architecture.md` (ADRs, tabela de testes).
2. Atualizar `docs/dependencies.md` (seção 6 — deps C++ opcionais).
3. Reescrever `tests/fixtures/README.md`.
4. Atualizar `README.md` (seção Tests & Validation + Acknowledgments).

### Fase 6 — Verificação final

1. `cargo test` — todos os testes passam.
2. `cargo test --test cpp_parity -- --ignored --nocapture` — validação live.
3. `./utils/tests-long.sh` — suite completa.
4. Verificar que nenhum artefato C++ vaza para o git.

---

## 9. Plano de Verificação

### Testes Automatizados

```bash
# Verificar que remoções não quebram nada
cargo test

# Regenerar goldens
./tests/fixtures/golden_gen_build.sh

# Verificar golden tests com novos goldens
cargo test test_golden_vectors -- --nocapture

# Validação cruzada ao vivo
cargo test --test cpp_parity -- --ignored --nocapture

# Suite completa
./utils/tests-long.sh
```

### Verificação Manual

- [ ] Golden files são ≤ 20 KB cada.
- [ ] `cargo test` sem C++ instalado funciona (goldens pré-commitados, `#[ignore]` skippados).
- [ ] Nenhum arquivo em `tests/golden/` ou `golden_gen.cpp` existe no repo.
- [ ] Documentação em `docs/dependencies.md` inclui `sudo apt install cmake g++ python3`.
- [ ] `docs/architecture.md` não referencia mais "regression_goldens" ou "NeuralAudio".
- [ ] Todas as métricas (MSE, MAE, SNR, PSNR, bits) são exibidas na saída dos testes.
