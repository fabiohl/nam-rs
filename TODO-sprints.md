# TODO-sprints.md — Hardening da Suite de Testes e Benchmarks

O objetivos das sprints abaixo é reforçar a suite de QA do NAM-rs. São fortemente inspiradas no trabalho de Steven Atkinson, Mike Oliphant e outros (espelhados em github.com/mikeoliphant/NeuralAudio/), consulte-os sempre.
O objetivo é assegurar conformidade aos padrões do NAM, qualidade de código e performance exemplar. Em cada tarefa, construa os testes visando os mais altos padrões e só conclua a tarefa quando a base de código passar aprovada no teste construido.

---

## Sprint 5 — Fundação de Segurança: Fuzz Testing e Zero-Alloc (Prioridade Máxima)

> **Objetivo:** Garantir que os parsers de entrada (`parse_nam_json`, `parse_namb`) sobrevivam a inputs malformados/adversários, e que o hot path `process()` seja comprovadamente zero-allocation.
>
> **Skill:** `implementador`

### Tarefa 5.1 — Fuzz Testing via Proptest para `parse_nam_json()` (Stable Rust) [Concluida]

**Entregável:** Novos testes proptest em `tests/proptest_parsers.rs`.

**Abordagem:** Usar `proptest` (já é dev-dependency, roda em stable Rust) para gerar inputs estruturados e semi-aleatórios. Não requer nightly nem `cargo-fuzz`.

**Implementação detalhada:**

1. Criar `tests/proptest_parsers.rs` com:
   - **`prop_fuzz_nam_json_arbitrary_bytes`** — Gera `Vec<u8>` aleatórios (0..4096 bytes), converte para `String` (lossy), alimenta `parse_nam_json()`. Verifica que **nunca** ocorre panic — apenas `Ok` ou `Err`.
   - **`prop_fuzz_nam_json_near_valid`** — Gera JSON semi-válido com campos `architecture`, `config`, `weights` presentes mas com valores aleatórios (strings, números, arrays de tamanhos variáveis). Valida que erros são retornados graciosamente.
   - **`prop_fuzz_nam_json_truncated`** — Pega um JSON válido de modelo `.nam` (fixture), trunca em posição aleatória (0..len), e alimenta `parse_nam_json()`. Deve retornar `Err` sem panic.
   - **`prop_fuzz_nam_json_weight_overflow`** — JSON válido mas com `weights` contendo `f32::MAX`, `f32::MIN`, `f32::INFINITY`, `f32::NAN`, `f32::MIN_POSITIVE` (subnormals). Verifica que o dispatcher rejeita ou constrói sem crash.

2. Configuração proptest: `ProptestConfig::with_cases(5_000)` (40K+ inputs gerados).

**Critério de aceite:**

- `cargo test --test proptest_parsers` passa com 5000 cases sem panic.
- `parse_nam_json()` retorna `Err` para todos os inputs inválidos (nunca `unwrap` explode).

### Tarefa 5.2 — Fuzz Testing via Proptest para `parse_namb()` (Binário) [Concluida]

**Entregável:** Testes proptest adicionais em `tests/proptest_parsers.rs`.

**Implementação detalhada:**

1. Adicionar ao mesmo arquivo:
   - **`prop_fuzz_namb_arbitrary_bytes`** — `Vec<u8>` aleatórios (0..8192 bytes), alimenta `parse_namb()`. Verifica que nunca ocorre panic.
   - **`prop_fuzz_namb_bad_magic`** — Header com magic number corrompido mas restante válido. Deve retornar `Err`.
   - **`prop_fuzz_namb_bad_crc`** — NAMB válido mas CRC32 alterada em 1 bit. Deve falhar na verificação de integridade.
   - **`prop_fuzz_namb_truncated`** — Buffer NAMB válido truncado em posição aleatória. Deve retornar `Err` sem panic.
   - **`prop_fuzz_namb_oversized_offset`** — `weights_offset` apontando além do buffer. Deve retornar `Err`.

2. Configuração: `ProptestConfig::with_cases(5_000)`.

**Critério de aceite:**

- Todos os testes passam, nenhum panic, nenhum OOB read.
- Verificar que `parse_namb()` não usa `unwrap()` ou `expect()` em dados de input — todo `[]` indexing deve usar `.get()` ou bounds check.

### Tarefa 5.3 — Verificação de Zero-Allocation no `process()` (Counting Allocator) [Concluida]

**Entregável:** Teste de integração em `tests/nam_infer_test.rs` que prova zero-alloc no hot path.

**Abordagem:** `#[global_allocator]` com allocator de contagem, ativo **apenas em `#[cfg(test)]`** — zero impacto no runtime de produção.

**Implementação detalhada:**

1. Criar um módulo `tests/alloc_tracker.rs` (ou inline no teste) com:

   ```rust
   // Allocator de contagem: conta malloc/free durante um intervalo
   // Ativo apenas quando #[cfg(test)]
   use std::alloc::{GlobalAlloc, Layout, System};
   use std::sync::atomic::{AtomicUsize, Ordering};

   static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
   static TRACKING_ENABLED: AtomicBool = AtomicBool::new(false);

   struct CountingAllocator;
   unsafe impl GlobalAlloc for CountingAllocator {
       unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
           if TRACKING_ENABLED.load(Ordering::Relaxed) {
               ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
           }
           unsafe { System.alloc(layout) }
       }
       unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
           unsafe { System.dealloc(ptr, layout) }
       }
   }

   #[cfg(test)]
   #[global_allocator]
   static GLOBAL: CountingAllocator = CountingAllocator;
   ```

2. Teste `test_zero_alloc_process`:
   - Carrega modelo real (`BossWN-standard.nam`), faz prewarm.
   - Reseta `ALLOC_COUNT` para 0, ativa tracking.
   - Roda `model.process()` com 64 amostras de seno 440Hz.
   - Desativa tracking.
   - `assert_eq!(ALLOC_COUNT.load(), 0, "Alocações detectadas no hot path!")`.

3. Repetir para LSTM (`BossLSTM-1x16.nam`) e WaveNet Dynamic.

**Nota arquitetural:** O `#[global_allocator]` em test binary é independente do binary de produção. O `nam-rs` (binary) compila sem o counting allocator. Apenas o test binary (`cargo test`) usa o allocator especial.

**Critério de aceite:**

- `model.process()` completa com 0 alocações heap para WaveNet estático, WaveNet dinâmico e LSTM estático.
- Se WaveNet dinâmico alocar (possível por usar `Vec` internamente), documentar como exceção conhecida.

### Tarefa 5.4 — Auditoria de `unwrap()` nos parsers [Concluida]

**Entregável:** Revisão e correção de qualquer `unwrap()`/`expect()` em `parse_nam_json()` e `parse_namb()` que toque input externo.

**Implementação:**

- `grep -n "unwrap\|expect" src/loader/nam_json.rs src/loader/namb.rs`
- Substituir por `?` operator ou `.ok_or_else(|| anyhow!(...))`.
- Manter `unwrap()` apenas em invariantes internas (nunca em dados do usuário).

**Critério de aceite:**

- Os fuzz tests da 5.1/5.2 passam sem panic em 5000 cases.

---

## Sprint 6 — Cobertura de Block Sizes e Modelos Comunitários

> **Objetivo:** Validar robustez com block sizes variáveis (como NeuralAudio faz) e exercitar os 5 modelos comunitários que existem em `tests/nam_files/` mas nunca foram testados.
>
> **Skill:** `implementador`

### Tarefa 6.1 — Testes com Block Sizes Variáveis (WaveNet + LSTM)

**Entregável:** Novo teste parametrizado em `tests/nam_infer_test.rs`.

**Implementação detalhada:**

1. Teste `test_wavenet_variable_block_sizes`:
   - Carrega `BossWN-standard.nam`, faz prewarm(2048).
   - Para cada `block_size ∈ {1, 16, 32, 64, 128, 256, 512}`:
     - Gera 512 amostras de senoidal 440Hz.
     - Processa em chunks de `block_size`.
     - Verifica finitude de todas as saídas.
     - Verifica RMS ≤ 10.0 (estabilidade numérica).
   - Opcionalmente: verifica que outputs com block_size=64 vs block_size=1 são **idênticos** (propriedade obrigatória — o modelo é state-stationary para block boundaries arbitrárias).

2. Teste `test_lstm_variable_block_sizes`:
   - Mesmo procedimento com `BossLSTM-1x16.nam`.
   - Block sizes: `{1, 16, 32, 64, 128, 256, 512}`.

3. Teste `test_wavenet_dynamic_variable_block_sizes`:
   - Mesmo com path dinâmico (`build_wavenet_dynamic()`).

**Critério de aceite:**

- Todos os block sizes processam sem panic e com saídas finitas.
- Block_size=1 (sample-by-sample) **deve** funcionar — este é o caso extremo que expõe bugs de indexação.

### Tarefa 6.2 — Exercitar Modelos Comunitários de `tests/nam_files/`

**Entregável:** Novo teste parametrizado em `tests/nam_infer_test.rs`.

**Implementação detalhada:**

1. Teste `test_community_models_inference`:
   - Lista estática dos 5 modelos em `tests/nam_files/`:
     - `ChandlerRedd47-Gain34-Standard.nam`
     - `EVH-5150-Lite.nam`
     - `NEVE1073-Standard.nam`
     - `UA610B-Gain+10-Standard.nam`
     - `little-bear-t7_phono-aux-tube-preamp_line-in_Standard.nam`
   - Para cada modelo:
     - `parse_nam_json()` → sucesso
     - `build_model()` → sucesso
     - `prewarm(2048)`
     - `process()` com 64 amostras de seno 440Hz
     - Verificar finitude e magnitude `< 100.0` em todas as saídas.
   - Incluir no `cargo test` principal (sem `#[ignore]`).

2. Verificar que topologias são detectadas corretamente:
   - `ChandlerRedd47` e `NEVE1073` são Standard (CH=16).
   - `EVH-5150` é Lite (CH=12).
   - Assertar `get_wavenet_topology()` para cada um.

**Critério de aceite:**

- Todos os 5 modelos carregam, constroem e processam sem crash.
- Nenhum teste usa `#[ignore]` — roda no CI principal.

### Tarefa 6.3 — Teste de Rejeição de Formatos Não-Suportados

**Entregável:** Novo teste em `tests/nam_infer_test.rs`.

**Implementação detalhada:**

1. Teste `test_reject_keras_legacy_format`:
   - Carrega `tests/fixtures/unsupported/tw40_blues_deluxe_deerinkstudios.json`.
   - Alimenta `parse_nam_json()`.
   - Verifica que retorna `Err` (formato Keras Legacy não suportado).
   - Se `parse_nam_json()` retornar `Ok`, verificar que `build_model()` retorna `Err` (rejeição no dispatcher por topologia desconhecida ou `architecture` não reconhecida).

2. Teste `test_reject_activation_non_tanh`:
   - Construir JSON sintético com `"activation": "ReLU"` (não Tanh).
   - Verificar que `build_model()` retorna `Err` com mensagem sobre ativação não suportada.

**Critério de aceite:**

- Formatos Keras e ativações não-Tanh são rejeitados graciosamente com `Err`.

---

## Sprint 7 — Testes Unitários Granulares e Proptest Expandido

> **Objetivo:** Preencher lacunas de cobertura unitária nas camadas internas (Conv1d, DenseLayer, LSTM state), e expandir proptest para `dot_product`.
>
> **Skill:** `implementador`

### Tarefa 7.1 — Testes Unitários Isolados para Conv1d

**Entregável:** Novos testes em `src/models/wavenet.rs` → `#[cfg(test)] mod tests`.

**Implementação detalhada:**

1. `test_conv1d_identity_kernel`:
   - Cria `Conv1d` com kernel_size=1, CH=4, pesos = identidade (1.0 na diagonal).
   - Input = `[1.0, 2.0, 3.0, 4.0]`, verifica output = input.

2. `test_conv1d_with_bias`:
   - `Conv1d` com `do_bias=true`, bias = `[0.5; CH]`.
   - Verifica que output = conv(input) + 0.5 para cada canal.

3. `test_conv1d_dilation`:
   - Dilation=2, kernel_size=3, CH=2.
   - Verifica que os taps acessam posições `[t, t-2, t-4]` no buffer (não `[t, t-1, t-2]`).
   - Alimenta sequência crescente, verifica resultado contra cálculo manual.

4. `test_conv1d_zero_input`:
   - Input de zeros, pesos arbitrários → output deve ser zero (sem bias) ou bias puro.

5. `test_conv1d_known_output`:
   - Pesos manuais e input manuais com output calculado à mão.

### Tarefa 7.2 — Testes Unitários Isolados para DenseLayer

**Entregável:** Novos testes em `src/models/wavenet.rs` → `#[cfg(test)] mod tests`.

1. `test_dense_layer_identity`:
   - DenseLayer(IN=4, OUT=4) com pesos = identidade. Output = input.

2. `test_dense_layer_with_bias`:
   - `do_bias=true`, bias = [1.0; OUT]. Verifica acréscimo constante.

3. `test_dense_layer_rectangular`:
   - IN=8, OUT=4. Pesos conhecidos, output verificado manualmente.

### Tarefa 7.3 — Testes LSTM Granulares

**Entregável:** Novos testes em `src/models/lstm.rs` → `#[cfg(test)] mod tests`.

1. `test_lstm_state_evolution`:
   - Alimenta step function (0→1) e verifica que hidden/cell states evoluem progressivamente.
   - Compara hidden state após 1 step vs 10 steps — devem ser diferentes.

2. `test_lstm_variable_block_sizes`:
   - Processa 64 amostras em {1, 8, 16, 32, 64} block sizes.
   - Verifica que output final é **idêntico** (state carryover correto entre blocos).

3. `test_lstm_reset_on_prewarm`:
   - Verifica que `prewarm()` zera os hidden/cell states antes de reprocessar.

### Tarefa 7.4 — Proptest para `dot_product_avx2` e `dot_product_avx512`

**Entregável:** Novos testes em `tests/proptest_math.rs`.

**Implementação detalhada:**

1. `prop_dot_product_avx2_vs_scalar`:
   - Gera dois vetores `Vec<f32>` de comprimento aleatório (1..512).
   - Computa dot product via SIMD (`dot_product_avx2`) e via escalar (`a.iter().zip(b).map(|(x,y)| x*y).sum()`).
   - Verifica que erro relativo ≤ 1e-5 (acumulação f32 tolerada).

2. `prop_dot_product_avx512_vs_scalar` (com guarda runtime `is_x86_feature_detected!("avx512f")`):
   - Mesmo procedimento para AVX-512.

3. Configuração: 5000 cases.

### Tarefa 7.5 — Teste de Concorrência Simulada do DspBridge

**Entregável:** Novo teste em `tests/nam_infer_test.rs` ou `src/pw_host.rs` inline.

**Implementação detalhada:**

1. `test_dsp_bridge_concurrent_access`:
   - Cria `DspBridge` via `Box::leak` (como produção).
   - Spawna 2 threads:
     - **Writer** (capture): escreve 1000 buffers de 64 amostras (padrão de contagem crescente) com `fence(Release)`.
     - **Reader** (playback): lê buffers com `fence(Acquire)`, verifica coerência (nenhum buffer parcialmente escrito).
   - Verifica que o reader nunca lê dados corrompidos (mixed entre duas escritas).
   - Usa `generation` counter para detectar atualizações.

2. Tempo de execução alvo: < 100ms.

---

## Sprint 8 — Benchmarks e Golden Vectors Expandidos

> **Objetivo:** Medir performance com block sizes variáveis, benchmarks de componentes isolados, e expandir golden vectors para mais topologias.
>
> **Skill:** `implementador`

### Tarefa 8.1 — Benchmarks com Buffer Sizes Variáveis

**Entregável:** Novos benchmarks em `benches/inference_bench.rs`.

**Implementação detalhada:**

1. `bench_wavenet_standard_block_sizes`:
   - WaveNet Standard com buffer sizes: 32, 128, 256.
   - Nome: `WaveNet_Standard_CH16_{N}samp_48kHz`.

2. `bench_lstm_2x16_block_sizes`:
   - LSTM 2×16 com buffer sizes: 32, 128, 256.

3. Manter os benchmarks existentes de 64 amostras intactos.

### Tarefa 8.2 — Benchmark de `dot_product_avx2` Isolado

**Entregável:** Novo benchmark em `benches/inference_bench.rs`.

1. `bench_dot_product_avx2_256`:
   - Dois vetores de 256 f32, dot product via SIMD.
   - Nome: `DotProduct_AVX2_256elem`.

2. `bench_dot_product_avx2_64`:
   - Dois vetores de 64 f32 (tamanho típico de hidden layer LSTM).
   - Nome: `DotProduct_AVX2_64elem`.

### Tarefa 8.3 — Benchmark de NamResampler (44.1k, 48k, 96k)

**Entregável:** Novo benchmark em `benches/inference_bench.rs`.

1. `bench_resampler_44100_to_48000`:
   - Cria `NamResampler(44100)`, processa 1024 amostras (input → 48k).
   - Nome: `Resampler_44100_to_48k_1024samp`.

2. `bench_resampler_96000_to_48000`:
   - Cria `NamResampler(96000)`, processa 1024 amostras (input → 48k).
   - Nome: `Resampler_96000_to_48k_1024samp`.

3. `bench_resampler_48000_bypass`:
   - `NamResampler(48000)` em bypass — mede overhead mínimo.
   - Nome: `Resampler_48000_bypass_1024samp`.

### Tarefa 8.4 — Benchmark AVX-512 para tanh/sigmoid

**Entregável:** Novos benchmarks em `benches/inference_bench.rs`.

1. `bench_tanh_avx512_256elem`:
   - Envolver com `if std::is_x86_feature_detected!("avx512f")`.
   - Se não suportado, SKIP (eprintln + return).
   - Nome: `FastMath_tanh_AVX512_256elem`.

2. `bench_sigmoid_avx512_256elem`:
   - Mesmo padrão.
   - Nome: `FastMath_sigmoid_AVX512_256elem`.

### Tarefa 8.5 — Golden Vectors para Feather e Nano

**Entregável:** Expandir `tests/fixtures/golden_gen.cpp` e `.sh`.

**Implementação detalhada:**

1. Modificar `golden_gen.cpp`:
   - Adicionar geração para `BossWN-feather.nam` → `golden_wavenet_feather.bin`.
   - Adicionar geração para `BossWN-nano.nam` → `golden_wavenet_nano.bin`.

2. Modificar `golden_gen_build.sh`:
   - Incluir os dois novos modelos na pipeline de build.

3. Novos testes em `tests/nam_infer_test.rs`:
   - `test_golden_vectors_wavenet_feather`:
     - MSE < 5e-2, SNR ≥ 9 dB (mesmos critérios do Standard — FastMath error similar).
   - `test_golden_vectors_wavenet_nano`:
     - MSE < 5e-2, SNR ≥ 9 dB.

4. Gerar e commitar os `.golden.bin`.

---

## Sprint 9 — Mutation Testing e Polimento Final

> **Objetivo:** Validar que a suite de testes realmente detecta bugs via mutation testing. Polir testes existentes e documentar.
>
> **Skill:** `implementador`, `revisor-auditor`

### Tarefa 9.1 — Mutation Testing via `cargo-mutants`

**Entregável:** Report de mutation testing.

**Implementação detalhada:**

1. Instalar: `cargo install cargo-mutants`.
2. Rodar: `cargo mutants --package nam-rs -- --lib` (apenas testes unitários, para velocidade).
3. Analisar: identificar mutantes que sobrevivem (código modificado sem falha de teste).
4. Para cada mutante sobrevivente: adicionar asserção que o captura, ou documentar como falso positivo.

**Critério de aceite:**

- ≥ 80% mutation score nos módulos `loader/`, `models/`, `math/`, `dsp/`.
- Mutantes sobreviventes documentados com justificativa.

### Tarefa 9.2 — Benchmark de `prewarm()`

**Entregável:** Novo benchmark em `benches/inference_bench.rs`.

1. `bench_prewarm_wavenet_standard`:
   - Constrói WaveNet Standard, mede tempo de `prewarm(2048)`.
   - Nome: `Prewarm_WaveNet_Standard_2048samp`.

2. `bench_prewarm_lstm_2x16`:
   - Nome: `Prewarm_LSTM_2x16_2048samp`.

### Tarefa 9.3 — Atualizar `docs/architecture.md` Seção 6

**Entregável:** Documentação atualizada refletindo a suite expandida.

**Skill:** `documentador`

**Implementação:**

- Atualizar contagens de testes na tabela da §6.1.
- Adicionar `tests/proptest_parsers.rs` à §6.2.
- Atualizar tabela de benchmarks na §6.3.
- Adicionar golden vectors Feather/Nano à §6.4.
- Documentar o counting allocator na §6.6.

### Tarefa 9.4 — Revisão Final e Lint

**Entregável:** Suite completa passando, lints limpos.

**Implementação:**

1. `cargo build` — sem erros.
2. `cargo test` — todos os testes passam (target: ≥ 130 testes).
3. `cargo bench --bench inference_bench` — todos os benchmarks rodam sem crash.
4. `utils/lints.sh` — zero warnings.
5. Revisar que todos os novos arquivos `.rs` têm header SPDX.
