# S13.T01 — Suite de Cross-Validation NAM-rs ↔ NeuralAmpModelerCore

## Revisão Crítica e Plano de Implementação Refinado

---

## 1. Diagnóstico do Estado Atual

### 1.1. O que já existe

O projeto possui **duas camadas** de validação numérica, mas **ambas têm lacunas**:

| Camada                             | Arquivos                                            | Referência C++              | Formato                              | Execução                               |
| ---------------------------------- | --------------------------------------------------- | --------------------------- | ------------------------------------ | -------------------------------------- |
| **Regressão autorreferencial**     | `tests/golden/*.bin` + `regression_goldens.rs`      | Nenhuma (Rust-only)         | f32 LE direto                        | `cargo test` (rápido)                  |
| **Golden vectors C++ NeuralAudio** | `tests/fixtures/golden_*.bin` + `nam_infer_test.rs` | NeuralAudio (Mike Oliphant) | `[u32 N][f32×N input][f32×N output]` | `cargo test` (rápido, skip se ausente) |

### 1.2. Problemas identificados

1. **Goldens autorreferenciais (`regression_goldens.rs`)** detectam drift do NAM-rs contra **si mesmo** — se há um bug na implementação, o golden perpetua o bug. Valor: proteção contra regressões acidentais, mas zero valor para validação de corretude.

2. **Goldens NeuralAudio (`nam_infer_test.rs`)** comparam contra o motor C++ de **Mike Oliphant** (NeuralAudio), que é uma reimplementação independente do NAM, **não** a referência canônica. O NeuralAudio usa:

   - RTNeural como backend de inferência
   - Otimizações Eigen para GEMM
   - Seus próprios polinômios de ativação (`Activation.h`)

3. **A referência canônica é o `NeuralAmpModelerCore`** de Steven Atkinson — o código que treina e gera os modelos `.nam`. Comparar contra ele é a validação definitiva: "o NAM-rs está processando os modelos **exatamente** como o autor original pretendeu?"

4. **O `NeuralAmpModelerCore` já possui um CLI `render`** em `tools/render.cpp` — compilado e testado com sucesso na pesquisa desta tarefa. Complexidade de produzir o binário: **zero** (CMake + Eigen + nlohmann — tudo já incluso nos submodules).

### 1.3. Diferenças de implementação entre as três engines

```text
NeuralAmpModelerCore (Steven Atkinson)  ←  REFERÊNCIA CANÔNICA
│   Eigen GEMM, std::tanh/exp nativos
│   Buffer size = 64 (fixo no render tool)
│   Reset(sampleRate, bufferSize) + process()
│
├── NeuralAudio (Mike Oliphant)  ←  GOLDEN ATUAL
│       RTNeural backend, math_approx activations
│       Otimizações separadas, divergência parcial
│
└── NAM-rs (este projeto)  ←  SOB TESTE
        SIMD AVX2/512 FMA, Padé tanh/sigmoid FastMath
        Batch GEMM, const-generics, lock-free RT
```

> [!IMPORTANT]
> **O `NeuralAmpModelerCore` usa `std::tanh`/`std::exp` nativos** (não aproximações FastMath). A divergência NAM-rs↔NAMCore será **menor** que NAM-rs↔NeuralAudio para LSTM (menos camadas não-lineares), mas pode ser maior ou menor para WaveNet dependendo da profundidade.

---

## 2. Arquitetura Proposta: Duas Camadas

### Camada 1 — Goldens NeuralAmpModelerCore pré-commitados (rápido)

- **Geração**: Script `utils/cpp_parity/generate_goldens.sh` executado **uma única vez** pelo developer (ou quando `NeuralAmpModelerCore` atualizar).
- **O que faz**: Compila o `render` → gera WAV de teste → processa cada modelo → extrai samples f32 → salva como `.golden.bin`.
- **Saída**: Arquivos `tests/fixtures/cpp_parity/namcore_*.golden.bin` (~4 KB cada, ~25 KB total para 5 modelos).
- **Commitados no git**: Sim — são pequenos, estáticos, e o `.gitignore` já permite `!tests/fixtures/*.bin`.
- **Testes Rust**: `tests/cpp_parity.rs` com `#[test]` normal (NÃO `#[ignore]`). Carregam o `.golden.bin`, processam o mesmo input pelo Rust, comparam MSE/SNR. Executam em **cada `cargo test`** — zero overhead de compilação C++.

> [!TIP]
> Este é **o padrão idêntico** ao já existente para `golden_gen.cpp`/NeuralAudio. A diferença é que a fonte de verdade muda de NeuralAudio para NeuralAmpModelerCore.

### Camada 2 — Validação cruzada ao vivo (lento, `tests-long.sh` only)

- **O que faz**: Compila o `render` (se necessário, cached) → gera WAV → executa C++ render → executa Rust → compara os dois outputs **ao vivo**.
- **Quando roda**: Apenas em `utils/tests-long.sh` e `cargo test --test cpp_parity -- --ignored`.
- **Valor extra**: Detecta drift se alguém atualizar o mirror `github.com/NeuralAmpModelerCore/` e os goldens commitados ficarem defasados.

### Fluxo visual

```text
Developer normal (cargo test):
  tests/fixtures/cpp_parity/namcore_*.golden.bin  →  Rust processa  →  MSE < threshold  ✓
  (Nenhuma compilação C++. Rápido.)

Developer QA (tests-long.sh):
  Build C++ render (se necessário)  →  Gera output C++ ao vivo  →  Compara com Rust  ✓
  (Compilação C++ uma única vez, cached. Lento na primeira vez.)
```

---

## 3. Detalhes Técnicos

### 3.1. Compilação e caching do binário C++

| Aspecto          | Detalhe                                                                           |
| ---------------- | --------------------------------------------------------------------------------- |
| **Binário**      | `github.com/NeuralAmpModelerCore/build/tools/render`                              |
| **Build system** | CMake ≥ 3.10, C++20, Eigen (header-only, submodule)                               |
| **Submodules**   | `Dependencies/eigen` e `Dependencies/AudioDSPTools` — inicializados pelo script   |
| **Caching**      | Script verifica se binário existe antes de compilar. Recompila apenas se ausente. |
| **Git**          | Diretório `/github.com/` inteiro está no `.gitignore` — zero risco de poluição    |
| **Tempo**        | ~30s na primeira compilação, 0s nas subsequentes                                  |

### 3.2. Formato dos goldens

Reutilizaremos o formato já existente em `tests/common/mod.rs::read_golden_bin()`:

```text
[u32 num_samples LE]
[f32×N input samples LE]
[f32×N output samples LE]
```

### 3.3. Sinal de teste

Senoidal **440 Hz @ 48 kHz, 512 amostras** — idêntico ao usado nos golden tests existentes (`generate_sine_440hz(512)`). Para o `render` CLI, será salvo como WAV mono float32 e a saída WAV será convertida para raw f32.

### 3.4. Modelos de referência

| Modelo           | Arquivo               | Topologia          | Profundidade |
| ---------------- | --------------------- | ------------------ | ------------ |
| WaveNet Standard | `BossWN-standard.nam` | CH=16, K=3, HEAD=8 | 20 layers    |
| WaveNet Feather  | `BossWN-feather.nam`  | CH=8, K=3, HEAD=4  | 20 layers    |
| WaveNet Nano     | `BossWN-nano.nam`     | CH=4, K=3, HEAD=2  | 20 layers    |
| LSTM 1×16        | `BossLSTM-1x16.nam`   | 1 layer, H=16      | 1 layer      |
| LSTM 2×8         | `BossLSTM-2x8.nam`    | 2 layers, H=8      | 2 layers     |

### 3.5. Thresholds de paridade

Serão calibrados na implementação com base em medições reais. Estimativas iniciais baseadas na divergência NeuralAudio existente:

| Modelo           | MSE esperado | SNR esperado |
| ---------------- | ------------ | ------------ |
| LSTM 1×16        | < 1e-3       | ≥ 22 dB      |
| LSTM 2×8         | < 1e-3       | ≥ 18 dB      |
| WaveNet Nano     | < 5e-2       | ≥ 9 dB       |
| WaveNet Feather  | < 5e-2       | ≥ 9 dB       |
| WaveNet Standard | < 5e-2       | ≥ 9 dB       |

> [!NOTE]
> **É possível que os thresholds NAMCore fiquem MELHORES que os NeuralAudio**, pois NAMCore usa `std::tanh` nativo (alta precisão). A divergência será dominada pela FastMath Padé do NAM-rs, sem a camada extra de divergência do NeuralAudio. Saberemos com certeza ao calibrar.

---

## 4. Artefatos a Produzir

### Build e Geração

#### [NEW] [generate_goldens.sh](file:///home/fabio/nam-rs/utils/cpp_parity/generate_goldens.sh)

Script idempotente que:

1. Verifica pré-requisitos (`cmake`, `g++`/`clang++`).
2. Inicializa submodules do NeuralAmpModelerCore se necessário.
3. Compila o `render` CLI (se binário não existir).
4. Gera WAV de teste (sine 440 Hz, 48 kHz, 512 samples).
5. Executa `render` para cada modelo de referência.
6. Converte saídas WAV para formato `.golden.bin`.
7. Salva em `tests/fixtures/cpp_parity/`.

### Testes

#### [NEW] [cpp_parity.rs](file:///home/fabio/nam-rs/tests/cpp_parity.rs)

- **Testes normais** (`#[test]`): Carregam goldens pré-commitados de `tests/fixtures/cpp_parity/`, processam pelo Rust, comparam MSE/SNR. Rodam em cada `cargo test`.
- **Testes live** (`#[test] #[ignore]`): Compilam C++, geram output ao vivo, comparam. Rodam em `tests-long.sh`.

#### [NEW] [wav.rs](file:///home/fabio/nam-rs/tests/common/wav.rs)

Helpers minimalistas (~60 LoC) para ler/escrever WAV mono float32 IEEE. Sem crate externo.

### Documentação

#### [MODIFY] [dependencies.md](file:///home/fabio/nam-rs/docs/dependencies.md)

Nova seção documentando pré-requisitos para a suite de cross-validation C++.

#### [MODIFY] [README.md](file:///home/fabio/nam-rs/README.md)

Atualizar seção "Tests & Validation" com instruções da cross-validation.

### Integração

#### [MODIFY] [tests-long.sh](file:///home/fabio/nam-rs/utils/tests-long.sh)

Adicionar etapa de cross-validation live.

#### [MODIFY] [TODO-sprints.md](file:///home/fabio/nam-rs/TODO-sprints.md)

Atualizar tarefa S13.T01 com o resumo deste plano refinado.

---

## 5. Plano de Verificação

### Testes Automatizados

```bash
# 1. Verificar que cargo test normal continua rápido e inclui parity checks
time cargo test

# 2. Executar cross-validation live (requer CMake/C++)
cargo test --test cpp_parity -- --ignored --nocapture

# 3. Suite completa
./utils/tests-long.sh
```

### Verificação Manual

- Confirmar que goldens `.golden.bin` são pequenos (~4 KB cada).
- Confirmar que nenhum artefato C++ (binários, .o, CMakeCache) vaza para o git.
- Confirmar que `cargo test` sem C++ instalado funciona (goldens são pré-commitados, testes `#[ignore]` são skippados).
