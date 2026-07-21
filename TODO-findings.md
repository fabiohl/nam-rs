<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-findings.md — Refatoração estrutural da mega-pasta `benches/`

Auditoria conduzida pela skill `refatora-rust` (foco estrutural, **sem alterar
lógica/algoritmos**; regressões proibidas) com planejamento delegado à skill
`planejador-arquiteto`. Escopo: 14 arquivos `.rs` em `benches/` (3.445 linhas
totais). Não houve `TODO-findings.md` prévio para este escopo; este documento é
a baseline de planejamento.

## Contexto e restrições de regressão (leia antes de qualquer épico)

A pasta `benches/` é acoplada externamente de forma rígida, o que restringe o
espaço de refatoração segura:

1. **Nomes de alvos `[[bench]]` são API pública.** Cada `cargo bench --bench
   <nome>` está documentado em `docs/benchmarks.md` e invocado literalmente em
   `utils/tests-long.sh` (linhas 652–660), `utils/tests-performance-regression.sh`
   (linhas 76, 112) e nos cabeçalhos `//!` de cada arquivo. **Renomear arquivos
   ou nomes de alvo quebra CI e documentação** → fora de escopo.
2. **IDs de benchmark (strings do `bench_function`/`benchmark_group`) são
   contratos de baseline.** `regression_gate` compara contra baselines
   persistidos (`--save-baseline`/`--baseline`). Alterar qualquer ID invalida
   comparações de regressão de performance → os IDs **devem permanecer
   idênticos**.
3. **`mod common;` é o padrão estabelecido** de inclusão de módulo compartilhado
   (`benches/common.rs`), já usado por `inference_bench.rs`, `linear.rs`,
   `long_inference_bench.rs`, `regression_gate.rs`, `clap_bench.rs`. Toda
   decomposição em subarquivos deve seguir este padrão (`mod <x>;` ou
   `#[path = "<x>.rs"] mod <x>;`), mantendo o binário de bench inalterado.
4. **Lint gate vigente.** `utils/lints.sh` roda `cargo clippy --all-targets
   --all-features` (fase 5/5). Isso significa que **imports não utilizados já
   seriam rejeitados hoje** — não há "import morto" especulativo a remover; a
   caça a código morto deve focar em duplicação real, não em `use` não-usados.
5. **`common.rs:11 #![allow(dead_code)]` é intencional** (comentário: "individual
   functions may appear unused in some binaries during phased migration"). O
   módulo é compilado em múltiplos binários de bench; o `allow` é justificado e
   **não deve ser removido**.

Princípio norteador: **preservar 100% dos nomes de alvo, IDs de benchmark e
comportamento observável**, decompondo apenas a estrutura interna via `mod` e
deduplicando helpers em `common.rs`.

---

## Inventário atual (linhas)

| Arquivo                   | Linhas | Veredito estrutural                    |
| ------------------------- | ------:| -------------------------------------- |
| `inference_bench.rs`      | 835    | Monolítico — dividir (F1)              |
| `gemv_bench.rs`           | 482    | Kernels + harness misturados (F4)      |
| `math_bench.rs`           | 330    | Aceitável; opportunidade de macro (F7) |
| `linear.rs`               | 230    | OK; usar `common::synth_ir` (F3)       |
| `kahan_conv1d_bench.rs`   | 228    | OK                                     |
| `clap_bench.rs`           | 206    | OK                                     |
| `long_inference_bench.rs` | 205    | OK; usar `common::synth_ir` (F3)       |
| `dsp_bench.rs`            | 198    | OK                                     |
| `common.rs`               | 182    | Alvo de consolidação (F2, F3)          |
| `dot_4x_bench.rs`         | 146    | OK                                     |
| `fft_radix4_bench.rs`     | 127    | OK                                     |
| `cabsim_bench.rs`         | 116    | OK; usar `common::synth_ir` (F3)       |
| `regression_gate.rs`      | 105    | OK (já usa `common`)                   |
| `head_gemv_bench.rs`      | 55     | OK                                     |

---

## F1 — `inference_bench.rs` (835 linhas) é monolítico

**Localização:** `benches/inference_bench.rs` (inteiro).

**Problema:** Único arquivo mistura 5 famílias de benchmark distintas
(WaveNet, LSTM, A2, Linear/Container/ConvNet, Dynamic/comparativos), com 28
funções `bench_*` e um `criterion_group!` com 28 alvos. Viola "small, atomic,
modular" da skill. Arquivo de bench mais quebrou o limite de 300 linhas por
fator 2,8×.

**Evidência:** `criterion_group!` em `inference_bench.rs:802-833` lista 28
alvos; blocos separados por comentários `// ── A2-Full ──` (`:221`, `:292`)
denotam fronteiras naturais já implícitas.

**Solução proposta — decomposição em submódulos do mesmo binário:**

```text
benches/
  inference_bench.rs          # raiz: mod wavenet; mod lstm; mod a2; mod misc;
                              #       criterion_group!(... wavenet::bench_... ...);
  inference/
    wavenet_bench.rs          # bench_wavenet_standard_process, p10, block_sizes,
                              #   dynamic, comparison, prewarm_wavenet_standard
    lstm_bench.rs             # bench_lstm_* (2x16, 1x8, 2x24, 1x40, comparisons,
                              #   dynamic, block_sizes, prewarm_lstm_2x16)
    a2_bench.rs               # bench_a2_full_*, bench_a2_lite_*, bench_prewarm_a2_*,
                              #   bench_a2_comparison
    misc_bench.rs             # bench_linear_model_dot_product,
                              #   bench_container_crossfade_64samp,
                              #   bench_wavenet_a2_dyn_gated_process,
                              #   bench_lstm_dynamic_process, bench_nondist_models,
                              #   bench_convnet_model_process
```

- `inference_bench.rs` declara `mod common;` (já existe), `mod wavenet;` etc.,
  torna as funções `pub` (ou `pub(super)` se o módulo for filha via `#[path]`).
- O `criterion_group!` raiz referencia `wavenet::bench_wavenet_standard_process`
  etc. — **mesmo conjunto de 28 alvos, mesmos IDs de string** → binário e CLI
  idênticos, `cargo bench --bench inference_bench` inalterado.
- Submódulos acessam `common::` via `crate::common` (ou `super::common` se
  `#[path]`). O helper `load_model_data` (ver F2) é a peça que viabiliza a
  extração sem rescrever o boilerplate em cada submódulo.

**Risco de regressão:** Médio. A decomposição é mecânica (move função + ajusta
visibilidade), mas envolve 28 alvos. **Mitigação:** executar
`cargo bench --bench inference_bench -- --list` antes e depois da refatoração e
confirmar que o conjunto de IDs é byte-a-byte idêntico; rodar `utils/lints.sh`
(fase fmt/check/clippy). Não é necessária execução completa de bench (apenas
`--list` prova que nenhum alvo sumiu).

---

## F2 — Boilerplate de carregamento de modelo duplicado (~8× em `inference_bench.rs`) [DONE]

**Localização:** Padrão recorrente em `inference_bench.rs` em
`:52-64`, `:159-167`, `:182-190`, `:227-235`, `:250-258`, `:275-281`,
`:298-306`, `:321-329`, `:344-350`, `:365-371`, `:514-534`, `:617-627`,
`:660-712`, `:776-786` (≈15 ocorrências do bloco path+exists+read+parse, ~8
delas com `build_model`+`prewarm` completos).

**Padrão duplicado:**

```rust
let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
path.push("tests/fixtures/models/<file>.nam");
if !path.exists() { return; }
let json_data = std::fs::read_to_string(&path).expect("Failed to read <X> model");
let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
let mut model = build_model(&model_data).expect("Dispatcher failed ...");
model.prewarm(2048);
```

**Problema:** ~150 linhas de duplicação; mensagens `.expect()` inconsistentes
("Failed in JSON parser" vs "Dispatcher failed for X"); alto custo de
manutenção e fonte de typos.

**Solução proposta — novo helper em `common.rs`:**

```rust
/// Lê e faz o parse de um modelo do diretório de fixtures (com fallback
/// nondist). Retorna `None` se o arquivo não existir — caller deve `return`
/// para pular o benchmark silenciosamente (comportamento atual).
pub fn load_model_data(filename: &str) -> Option<NamModelData> {
    let path = model_path(filename);          // já trata nondist fallback
    let json_data = fs::read_to_string(&path).ok()?;
    parse_nam_json(&json_data).ok()
}
```

- Reaproveita `common::model_path` (`:158`), que já implementa o fallback
  `tests/fixtures/models-nondist` → `tests/fixtures/models` — **comportamento
  equivalente** ao path hardcoded atual para os modelos catalogados
  (BossWN-standard, wavenet_a2_full/lite, convnet_test).

- O call site colapsa para:

  ```rust
  let model_data = match common::load_model_data("BossWN-standard.nam") {
      Some(d) => d, None => return,
  };
  let mut model = build_model(&model_data).expect("Dispatcher failed for WaveNet");
  model.prewarm(2048);
  ```

- `common::load_and_prewarm` (`:172`) já existe para o caso simples
  (load+build+prewarm retornando `Option<StaticModel>`); **não serve** para
  benches que precisam de `model_data` downstream (`bench_container_crossfade`,
  `bench_prewarm_*` usam `iter_with_setup` reconstruindo de `model_data`). O
  novo `load_model_data` complementa, não substitui.

**Risco de regressão:** Baixo. Equivalência comportamental verificada
(`model_path` retorna o mesmo caminho para modelos catalogados; `None`→`return`
reproduz o `if !path.exists() { return }`). **Mitigação:** confirmar via
`cargo bench --bench inference_bench -- --list` que nenhum alvo sumiu nem mudou
de nome.

---

## F3 — `synth_ir` (senoide exponencial decrescente) duplicado em 3 arquivos

**Localização:**

- `benches/linear.rs:40-47` — `synth_ir(len, freq, decay)` com `2.0 * PI`.
- `benches/cabsim_bench.rs:18-26` — `synth_ir(len, freq, decay)` com `TAU`.
- `benches/long_inference_bench.rs:149-156` — closure inline `synth_ir` com `2.0 * PI`.

**Problema:** Três cópias do mesmo gerador de IR sintética. Diferença
`2.0 * PI` vs `TAU` é **matematicamente nula** (`std::f32::consts::TAU == 2.0 *
PI` exatamente em f32), mas a duplicação esconde isso e fragiliza a manutenção.

**Solução proposta — consolidar em `common.rs`:**

```rust
/// Gera uma IR sintética (senoide exponencialmente decrescente) para benches
/// de CabSim/Linear, sem depender de fixtures externas.
pub fn synth_ir(len: usize, freq: f32, decay: f32) -> Vec<f32> {
    const SR: f32 = 48000.0;
    (0..len)
        .map(|n| {
            let t = n as f32 / SR;
            (std::f32::consts::TAU * freq * t).sin() * (-decay * t).exp()
        })
        .collect()
}
```

- `linear.rs` já tem `mod common;` (`:35`) → troca direta por `common::synth_ir`.
- `long_inference_bench.rs` já tem `#[cfg(feature="long_bench")] #[path =
  "common.rs"] mod common;` → troca direta.
- `cabsim_bench.rs` **não** importa `common`; adicionar `mod common;` é seguro
  (o módulo compila isoladamente) e habilita `common::synth_ir`.

**Risco de regressão:** Baixo. `TAU == 2.0*PI` garante bit-identidade para
`linear.rs` e `long_inference_bench.rs`; `cabsim_bench.rs` já usava `TAU`.
**Mitigação:** `cargo bench --bench cabsim_bench -- --list` + `--bench linear
-- --list` confirmam alvos preservados.

---

## F4 — `gemv_bench.rs` (482 linhas) mistura kernels SIMD especializados com harness

**Localização:** `benches/gemv_bench.rs`. Kernels especializados em `:74-394`
(6 funções `gemv_specialized_*`, ~320 linhas de `#[target_feature]` unsafe);
harness/macro/`criterion_group` em `:396-482`.

**Problema:** Os 6 kernels são **protótipos de bench** (variantes
fully-unrolled não canonizadas na lib), mas vivem no mesmo arquivo que o
harness, inflando-o além do limiar modular.

**Solução proposta — extrair kernels para submódulo irmão:**

```text
benches/
  gemv_bench.rs              # mod kernels; mod common; (não usa common hoje)
                             # + make_test_data + bench_dim! + bench_* + group
  gemv_kernels.rs            # gemv_specialized_1x4/4x4/4x6/8x4/8x6/8x8
```

- `gemv_bench.rs` declara `#[path = "gemv_kernels.rs"] mod kernels;` (ou
  `mod kernels;` se o arquivo for `gemv_kernels.rs` no mesmo dir — Cargo busca
  `benches/gemv_kernels.rs` automaticamente para `mod kernels` dentro de um
  bench binário? **Não**: bench binaries não têm `mod` discovery automático além
  do raiz; usar `#[path]` é o seguro, espelhando `long_inference_bench.rs:20`).
- Kernels marcados `pub(crate)` ou `pub(super)` para o harness os referenciar
  via `kernels::gemv_specialized_1x4` no macro `bench_dim!`.
- **Nenhum ID de bench muda** (`gemv_1x4`...`gemv_8x8`, `generic_avx2`,
  `specialized_avx2`, `scalar_fallback`).

**Risco de regressão:** Médio. Kernels são `unsafe` com `#[target_feature]`;
movê-los é mecânico mas exige atenção a visibilidade e ao caminho `#[path]`.
**Mitigação:** `cargo bench --bench gemv_bench -- --list` antes/depois; manter
`use core::arch::x86_64::*;` no arquivo de kernels (dependência dos intrínsecos).

---

## F5 — Quantizador f32→f16 inline em `gemv_bench.rs` — **NÃO substituir** (guardião de regressão)

**Localização:** `benches/gemv_bench.rs:43-58` (dentro de `make_test_data`).

**Problema aparente:** Reimplementa conversão f32→f16, parecendo duplicação do
helper da lib `nam_rs::math::common::half::f32_to_f16_bits`
(`src/math/common/half.rs:51`), já usado em `kahan_conv1d_bench.rs:5`.

**Análise crítica (por que NÃO é duplicação removível):**

- A versão da lib (`f32_to_f16_bits`) implementa **round-to-nearest-even** com
  tratamento correto de NaN/Inf/subnormais (doc em `half.rs:46`).

- A versão inline de `gemv_bench.rs` implementa **truncagem** (shift puro,
  sem rounding) com clamp para `±0x7BFF` (não `±Inf 0x7C00`):

  ```rust
  let frac = (u & 0x7F_FFFF) >> 13;          // truncation, no +0.5 rounding
  if exp < 112 { 0 }
  else if exp > 142 { (sign | 0x7BFF) as u16 }   // clamp, not Inf
  else { (sign | ((exp - 112) << 10) | (frac & 0x3FF)) as u16 }
  ```

- **Bit patterns divergem** para a maioria das entradas → pesos diferentes →
  workload de FMA diferente → o bench mediria outra coisa. Trocar
  invalidaria baselines e mudaria a semântica do que é medido.

**Solução proposta:** **Manter como está.** Adicionar comentário no local
documentando que é um quantizador por truncagem intencional (não usar
`f32_to_f16_bits`). Alternativa, se unificação for desejada no futuro: adicionar
à lib uma variante `f32_to_f16_bits_trunc` e usá-la — **fora do escopo** desta
refatoração estrutural (tocaria na lógica da lib, proibido aqui).

**Risco de regressão se ignorado:** Alto (regressão semântica silenciosa). Este
finding existe precisamente para **evitar** que um refactor "dedup" bem-intencionado
introduza regressão.

---

## F6 — Inconsistência de nome `linear.rs` (sem sufixo `_bench`) — **manter**

**Localização:** `benches/linear.rs` (único arquivo sem sufixo `_bench`).

**Problema aparente:** Quebra a convenção de nomenclatura dos demais
(`cabsim_bench.rs`, `dsp_bench.rs`, etc.).

**Análise:** Renomear `linear.rs` → `linear_bench.rs` exige mudar
`Cargo.toml:104 [[bench]] name = "linear"` → `name = "linear_bench"`, o que
**muda o nome do alvo de CLI** (`cargo bench --bench linear` → `--bench
linear_bench`). Isso quebra:

- `docs/benchmarks.md` (comando documentado),
- cabeçalho `//!` do próprio arquivo (`:20 cargo bench --bench linear`),
- comentário em `utils/tests-long.sh:639` ("`linear` (bench)").

**Solução proposta:** **Não renomear.** A consistência estética não compensa o
risco de quebra de CI/docs. Registrar a inconsistência como dívida cosmética
aceita. (Se futuramente desejado: renomear arquivo + `Cargo.toml` + todas as
referências em docs/scripts num commit atômico — tarefa separada, não
estrutural.)

**Risco de regressão:** N/A (decisão é não agir).

---

## F7 — `math_bench.rs` (330 linhas): repetição do gate AVX-512 — oportunidade de macro (cauteloso)

**Localização:** `benches/math_bench.rs`. Seis funções
`bench_*_avx512_*_256elem` (`:223-307`) repetem o molde:

```rust
if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512vl") {
    use std::arch::x86_64::*;
    let base: Vec<f32> = (0..256).map(...).collect();
    c.bench_function("...", |b| { /* chunks_exact_mut + _mm512_* */ });
}
```

**Problema:** ~85 linhas de molde repetido; alta similaridade estrutural.

**Solução proposta (opcional, conservadora):** Macro `bench_avx512!` que expande
para o bloco idêntico, parametrizada por `(nome, chunk_size, chamada_simd)`.
Expansão é textualmente equivalente → sem mudança de lógica.

**Risco de regressão:** Médio-baixo. Macro mal escrita pode alterar o corpo
(ex: errar o `chunks_exact_mut(16)` vs `(8)`). **Mitigação:** validar com
`cargo bench --bench math_bench -- --list` (IDs preservados) e `cargo expand`
para conferir expansão; alternativamente, **deixar como está** — 330 linhas é
aceitável e a macro é ganho cosmético. Recomendado **apenas se** F1/F4 já
estiverem resolvidos e houver folga.

---

## F8 — Geradores de dados sintéticos f32 espalhados (consolidação parcial)

**Localização:**

- `gemv_bench.rs:38` `make_test_data` (f32 in/bias + f16 weights via F5).
- `head_gemv_bench.rs:18` `make_test_data` (f32 puro, in_len×out_len).
- `dot_4x_bench.rs:19` `generate_test_data`, `:32` `generate_f32_test_data<const N>`.
- `kahan_conv1d_bench.rs:10-49` `generate_weights`, `generate_weights_f32`,
  `generate_input`.

**Problema:** Quatro variações de "gerar pesos/entradas f32 com padrão
senoidal determinístico". Algumas quase idênticas (`head_gemv` e a parte f32 de
`gemv`).

**Solução proposta (opcional, baixo ganho):** Mover apenas o gerador f32
genérico `(in_len, out_len) -> (Vec<f32>, Vec<f32>, Vec<f32>)` para `common.rs`
e reusar em `head_gemv_bench.rs` + parte f32 de `gemv_bench.rs`. **Não mover**
os que têm layout especializado (interleaving 4-uplo de `kahan_conv1d`, padrão
`[u16;4]` de `dot_4x`) — são dimensão-específicos.

**Risco de regressão:** Baixo, mas **ganho marginal**. Priorizar F1–F4 antes.
O `make_test_data` de `gemv_bench.rs` **não deve** ser tocado (depende do
quantizador de F5).

---

## Épicos (agrupamento para execução segura)

### Épico A — Consolidação de helpers em `common.rs` (F2, F3) [DONE]

**Risco: Baixo.** Quick win de deduplicação lógica-preservante.

- A1: Adicionar `common::load_model_data` (F2); refatorar call sites em
  `inference_bench.rs` (~8 blocos). Validar `--list` do `inference_bench`.
- A2: Adicionar `common::synth_ir` (F3); refatorar `linear.rs`,
  `cabsim_bench.rs` (adicionar `mod common;`), `long_inference_bench.rs`.
  Validar `--list` de `cabsim_bench` e `linear`.
- **Critério de aceite:** `utils/lints.sh` verde; `cargo bench --bench <X>
  -- --list` idêntico pré/pós para cada bench tocado.

### Épico B — Decomposição modular de `inference_bench.rs` (F1) [DONE]

**Risco: Médio.** Depende de A1 (helper `load_model_data`) para viabilizar
submódulos limpos. **Crítico — maior atenção.**

- B1: Criar `benches/inference/{wavenet,lstm,a2,misc}_bench.rs` via
  `#[path = "inference/<x>_bench.rs"] mod <x>;`.
- B2: Mover funções `bench_*` para os submódulos, marcar `pub`, atualizar
  `criterion_group!` raiz com paths qualificados.
- B3: Validar `cargo bench --bench inference_bench -- --list` idêntico (28
  alvos, mesmos IDs); rodar `utils/lints.sh`.

### Épico C — Decomposição modular de `gemv_bench.rs` (F4) [DONE]

**Risco: Médio.** Envolve código `unsafe`/`#[target_feature]`. **Crítico —
atenção redobrada.**

- C1: Criar `benches/gemv_kernels.rs` via `#[path]`; mover os 6
  `gemv_specialized_*` (`pub(super)`).
- C2: Ajustar macro `bench_dim!` para referenciar `kernels::...`. **Não tocar**
  em `make_test_data`/quantizador (F5, F8).
- C3: Validar `cargo bench --bench gemv_bench -- --list` idêntico
  (`gemv_1x4`...`gemv_8x8` + `generic_avx2`/`specialized_avx2`/`scalar_fallback`).

### Épico D — Guardiões de não-regressão documentais (F5, F6)

**Risco: Nenhum (decisão de não-agir + documentação).**

- D1: Adicionar comentário em `gemv_bench.rs:43` explicando o quantizador por
  truncagem intencional (F5).
- D2: Registrar dívida cosmética do nome `linear` (F6) — sem ação de código.

### Épico E — Opcional: dedup de geradores e macro AVX-512 (F7, F8)

**Risco: Baixo-médio.** Executar apenas após A–C estabilizados e se houver
folga; ganho predominantemente estético.

- E1: Macro `bench_avx512!` em `math_bench.rs` (F7) — validar expansão.
- E2: Mover gerador f32 genérico para `common.rs` e reusar em `head_gemv`/
  `gemv` (F8) — sem tocar `kahan_conv1d`/`dot_4x`.

---

## Ordem de execução recomendada

A → B → C → D, com E opcional por último. Cada épico é uma entrega atômica
validável por `utils/lints.sh` + `cargo bench --bench <X> -- --list` (comparação
de IDs pré/pós). **Nunca** executar `utils/tests-long.sh` em tarefa de IA
(regra `testing.md` §2); a validação de benchs longos é responsabilidade do
operador humano.
