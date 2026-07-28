<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Findings — NAM-rs como Crate Pública (crates.io / docs.rs)

Auditoria da jornada do desenvolvedor externo consumindo `NeuralAmpModeler-rs` via
[crates.io](https://crates.io/crates/NeuralAmpModeler-rs) e
[docs.rs](https://docs.rs/NeuralAmpModeler-rs/latest/nam_rs/).

**Nota de segurança**: Todas as correções devem preservar o funcionamento do projeto
original (`standalone`, `clap-plugin`, testes e benchmarks). Nenhuma alteração deve
quebrar o binário `nam-rs`, o plugin CLAP, ou o suite de testes existente.

---

## F1. `crate-type` inclui `cdylib` — Potencial quebra downstream 🔴

**Arquivo:** `Cargo.toml` L178
**Severidade:** Alta

O `[lib]` declara `crate-type = ["rlib", "cdylib"]`. Quando um projeto externo
adiciona `NeuralAmpModeler-rs` como dependência, o Cargo tenta construir **ambos**
os artefatos. O `cdylib` requer:

- O símbolo `clap_entry` (forçado via `-Wl,-u,clap_entry` em `.cargo/config.toml`)
- O linker version script `.cargo/hide-libm-shadow.map`
- Ambos **não presentes** no pacote publicado no crates.io

**Impacto:** Build downstream pode falhar com erros de linker.

**Solução proposta:** Remover `"cdylib"` do `crate-type` no `[lib]`. O build do
plugin CLAP deve continuar funcionando porque a seção `[[bin]]` e o target `cdylib`
podem ser gerenciados via script de build ou workspace separado. Alternativa: usar
`cargo:rustc-cdylib-link-arg` apenas sob feature `clap-plugin`.

**Risco:** Médio — precisa validar que o build CLAP (`cargo build --features clap-plugin`)
ainda produz o `.so` necessário. Se `cdylib` for necessário para o CLAP, a solução é
condicionar o `crate-type` via `build.rs` ou mover o CLAP para um crate wrapper.

---

## F2. `global_asm!` redefine símbolos globais — Conflito em projetos externos 🔴

**Arquivo:** `src/lib.rs` L209-226
**Severidade:** Alta

O bloco `global_asm!` redefine símbolos globais `log10f`, `atan2f`, `acosf` com
binding GLIBC_2.2.5. Esses símbolos são compilados em **todo binário** que linka
`nam_rs`.

**Investigação necessária:** O `build.rs` (raiz) já aplica um version script
(`.cargo/hide-libm-shadow.map`) que força esses símbolos para `local`. Pode haver
**redundância** entre o `global_asm!` e o version script. O `global_asm!` pode
ser desnecessário se o version script já resolve o problema.

**Cenário de risco para downstream:**

1. Projeto externo linka `nam_rs` como `rlib`
2. `global_asm!` emite símbolos `log10f` etc. com binding GLOBAL
3. Se o version script não for aplicado (porque o consumidor não herda o `build.rs`),
   esses símbolos podem causar interposição
4. Se o consumidor usa `lld` ou outra toolchain, comportamento pode variar

**Solução proposta:** Guardar o `global_asm!` sob
`#[cfg(any(feature = "standalone", feature = "clap-plugin"))]` para que consumidores
de `rlib` (com `default-features = false`) não herdem esses símbolos. Validar que
o `build.rs` sozinho (com o version script) é suficiente para resolver o hang original.

**Risco:** Alto — requer testes cuidadosos. O hang original documentado em
`docs/postmortem-libm-symbol-interposition.md` deve ser reproduzível antes/depois
da mudança.

---

## F3. `pub use common::*` — Poluição de namespace 🔴

**Arquivo:** `src/lib.rs` L180-181
**Severidade:** Alta (DX)

O `pub use common::*` re-exporta transitivamente:

- `diagnostics::*` → `NamDiagnostic`, `NamErrorCode`, `NamLogger`, `LogBuffer`,
  `HostLogFn`, `DiagnosticBundle`, `ErrorContext`, `SystemSnapshot`, `ModelInfo`,
  `AudioInfo`, `RtInfo`, `RuntimeSnapshot`, `TelemetrySnapshot`, `HasRuntimeSnapshot`,
  `ACTIVE_MODEL_INFO`, `ACTIVE_MODEL_NAME`, `ACTIVE_SAMPLE_RATE`
- `panic_hook::*` → Tipos internos do crash reporter
- `params::*` → Tipos de parâmetros CLAP internos
- `spsc::*` → `RtStatusFlags`, `GcItem`, constantes `RT_STATUS_*`

O consumidor externo vê **~40+ tipos** no namespace raiz `nam_rs::` que são
infraestrutura interna. Isso polui autocompletion e torna a API confusa.

**Solução proposta:** Remover `pub use common::*` e `pub use standalone::*`.
Criar re-exports seletivos apenas dos tipos genuinamente públicos:

```rust
pub use common::diagnostics::SystemSnapshot;
```

**Risco:** Baixo para downstream (nenhum consumidor externo ainda depende desses
re-exports em produção). Porém **médio internamente** — os módulos `standalone`,
`clap` e testes podem importar via `crate::common::...` em vez de `crate::`.
Requer busca e atualização de imports internos.

---

## F4. Versão hardcoded nos doc-examples 🟡

**Arquivo:** `src/lib.rs` L43, L49, L55
**Severidade:** Média

Os exemplos de `Cargo.toml` no doc-comment mostram `version = "3.0.2"` enquanto
a versão atual é `3.1.0`. Na próxima publicação, os exemplos na docs.rs estarão
desatualizados.

**Solução proposta:** Usar `version = "3"` para resiliência semver.

**Risco:** Nulo.

---

## F5. `StaticModel` não implementa `Debug` 🟡

**Arquivo:** `src/models/mod.rs` L106-153
**Severidade:** Média

O tipo central retornado por `load_and_build_model` não implementa `Debug`.
O `LoadedModelPair` contorna isso com hardcode `"StaticModel"` no seu `Debug` impl.
Isso impede o consumidor de fazer `dbg!(model_pair)` e ver qual variante foi
construída.

**Solução proposta:** Implementar `Debug` para `StaticModel` delegando ao
`class_label()` existente.

**Risco:** Nulo.

---

## F6. `SystemSnapshot::capture()` no Quick Start sem contexto 🟡

**Arquivo:** `src/lib.rs` L79-80
**Severidade:** Média (DX)

O exemplo Quick Start usa `SystemSnapshot::capture()` sem explicar o que é,
por que é necessário, ou de onde importar.

**Solução proposta:** Adicionar comentário explicativo no doctest.

**Risco:** Nulo.

---

## F7. Dupla/tripla indireção em `LoadedModelPair.model_l` 🟡

**Arquivo:** `src/loader/loaded_model_pair.rs` L20
**Severidade:** Média (DX — confusa, mas não um bug)

`Option<Box<StaticModel>>` onde `StaticModel` contém `Box<WaveNetModel<...>>`.
Indireção deliberada para swap SPSC, mas não documentada para o consumidor.

**Solução proposta:** Adicionar doc-comment explicando o porquê da indireção.

**Risco:** Nulo.

---

## F8. Módulos internos do `loader` expostos como `pub` 🟡

**Arquivo:** `src/loader/mod.rs` L10-17
**Severidade:** Média

`dispatcher`, `transpose`, `namb_encoder` são módulos de implementação interna
expostos publicamente.

**Solução proposta:** Tornar `pub(crate)`.

**Risco:** Baixo — validar que nenhum teste de integração em `tests/` importa
esses módulos diretamente.

---

## F9. `slicing` exposto via `pub mod` em `slimmable.rs` 🟡

**Arquivo:** `src/models/slimmable.rs` L88
**Severidade:** Média

Funções como `slice_conv1d`, `slice_dense`, `slice_wavenet_model` são implementação
interna do adaptive compute. Não devem ser API pública.

**Solução proposta:** Tornar `pub(crate) mod slicing;`.

**Risco:** Baixo.

---

## F10. Tom informal e typo no doc-comment do `lib.rs` 🟡

**Arquivo:** `src/lib.rs` L11
**Severidade:** Média (profissionalismo)

```text
since some dude took my name during the phase na-rs wa in development
```

Typo ("wa" → "was") e tom informal na primeira impressão da docs.rs.

**Solução proposta:** Reformular profissionalmente.

**Risco:** Nulo.

---

## F11. `cabsim` gated por features específicas 🟢

**Arquivo:** `src/dsp/mod.rs` L8-9
**Severidade:** Baixa

O módulo `cabsim` só está disponível com `standalone`, `clap-plugin` ou `test`.
Consumidores externos que querem convolver IRs sem essas features ficam sem acesso.

**Solução proposta:** Avaliar se o `cabsim` faz sentido como API pública
independente. Se sim, remover o gate. Se não (porque depende de WAV I/O e
assets internos), manter e documentar.

**Risco:** Baixo.

---

## F12. `serde` derives sempre ativos em tipos públicos 🟢

**Arquivo:** `src/math/activations/mod.rs` L70
**Severidade:** Baixa

`Serialize`/`Deserialize` em `ActivationPrecision` e outros tipos é sempre ativo
porque `serde` é dependência normal (não optional).

**Solução proposta:** Para iteração futura. Considerar feature-gate `serde` se
a audiência downstream não usa serde.

**Risco:** Baixo, mas requer refatoração significativa se implementado.

---

## F13. Falta `examples/` com exemplo público 🟢

**Severidade:** Baixa (DX)

Não existe diretório `examples/` no pacote publicado. Um `examples/basic_inference.rs`
seria o melhor recurso para o desenvolvedor — mais concreto que documentação.

**Solução proposta:** Criar `examples/basic_inference.rs` exercitando toda a API
pública core. Adicionar ao `include` no `Cargo.toml`.

**Risco:** Nulo.

---

## F14. `build.rs` emite linker args incondicionalmente para downstream 🟡

**Arquivo:** `build.rs` (raiz)
**Severidade:** Média

O `build.rs` emite `cargo:rustc-link-arg` com o version script e `--undefined-version`
para **todos** os targets. Quando o crate é consumido como `rlib` por um projeto
externo, esses args são propagados ao binário final do consumidor.

O version script `.cargo/hide-libm-shadow.map` **está incluído** no pacote via
`build.rs` (que referencia `CARGO_MANIFEST_DIR`), mas o arquivo `.map` em si
pode não estar no `include` do Cargo.toml.

**Solução proposta:** Condicionar a emissão de link args no `build.rs`:

- Verificar se está construindo `cdylib` ou `bin` antes de emitir
- Incluir `.cargo/hide-libm-shadow.map` no `include` do Cargo.toml

**Risco:** Médio — requer validação cuidadosa.

---

## F15. `build.rs` e `x86-64-v3`: consumidor externo não herda target flags 🟡

**Arquivo:** `.cargo/config.toml` L12
**Severidade:** Média

O `.cargo/config.toml` define `-Ctarget-cpu=x86-64-v3`, mas este arquivo **não é
distribuído** no pacote crates.io. Consumidores externos compilam com o target
padrão de suas máquinas, potencialmente sem AVX2/FMA.

Já existe `compile_error!` para `!target_arch = "x86_64"` em `lib.rs` L176-177,
mas não há verificação de feature level (AVX2, FMA).

**Solução proposta:** Adicionar no `build.rs` verificação via
`cfg!(target_feature = "avx2")` e `cfg!(target_feature = "fma")` emitindo
`compile_error!` ou ao menos `cargo:warning` quando ausentes. Documentar no
`lib.rs` doctest que o consumidor deve configurar `-Ctarget-cpu=x86-64-v3`.

**Risco:** Baixo.
