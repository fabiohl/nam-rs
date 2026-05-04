<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva. -->
# 🚀 Backlog do Produto e Planejamento de Sprints Técnicas

## Épicos A–G — Sprints Anteriores (v1.1–v1.3) [DONE]

> Todos os épicos A–G foram concluídos. Consulte o histórico git para detalhes.
> Resumo: Otimizações SIMD (LSTM fused GEMV, WaveNet Conv1D tiling, prefetch,
> tanh+head fusion, gated activation fusion, fused GEMV residual), fuzz testing,
> paridade WaveNet Dyn, documentação atualizada.

---

## Sprint v1.4 — Preparações para A2 e CLAP

> **Meta**: "Organizar a casa e preparar o terreno" para a Arquitetura A2 e suporte CLAP.
> Nenhuma implementação de inferência A2 ou plugin CLAP. Apenas scaffolding,
> refatoração de boundaries, docs e testes de regressão.
>
> **Referência**: Relatório em `v1.4_research_and_planning.md`
> **Snapshot C++**: `/github.com/sdatkinson/NeuralAmpModelerCore/` (v0.5.2)

---

## Épico H — Staging Arquitetura A2 (Scaffolding) [DONE]

> Criar tipos, enums e módulos-esqueleto para A2. Zero lógica de inferência.
> Ref: `github.com/sdatkinson/NeuralAmpModelerCore/NAM/`
> O repositório `https://github.com/sdatkinson/NeuralAmpModelerCore` está espelhado na subpasta `/github.com/sdatkinson/NeuralAmpModelerCore/`.

### TH1 · Enum `ActivationType` e Trait `ActivationFn` [DONE]

- **Arquivo(s):** `src/models/activations.rs` (novo)
- **O quê:** Enum com 11 variantes do C++ (`NAM/activations.h` L27-40: Tanh, HardTanh, FastTanh, ReLU, LeakyReLU, PReLU, Sigmoid, SiLU, HardSwish, LeakyHardTanh, Softsign). Trait `ActivationFn` com `apply(&self, data: &mut [f32])`. Impls escalares. Registrar em `src/models/mod.rs`.
- **Ref C++:** `NAM/activations.h` L59-428
- **Validação:** Testes unitários vs valores C++.
- [x] Criar módulo + enum + trait + impls
- [x] Testes unitários (golden values)

### TH2 · Enum `GatingMode` e Structs de Gating [DONE]

- **Arquivo(s):** `src/models/gating.rs` (novo)
- **O quê:** Enum `GatingMode { None, Gated, Blended }` (`NAM/wavenet/params.h` L18-22). Config structs para GatingActivation/BlendingActivation. Sem `process()`.
- **Ref C++:** `NAM/gating_activations.h` L25-246
- [x] Criar módulo com enums + config structs
- [x] Doc-tests

### TH3 · Struct `FiLMConfig` (Stub) [DONE]

- **Arquivo(s):** `src/models/film.rs` (novo)
- **O quê:** `FiLMConfig { active: bool, shift: bool, groups: u32 }` de `_FiLMParams` (`NAM/wavenet/params.h` L76-91). Trait `FiLMLayer` com `process()` → `todo!()`.
- **Ref C++:** `NAM/film.h` L19-209
- [x] Criar módulo com config + trait stub

### TH4 · Structs de Parâmetros A2 WaveNet [DONE]

- **Arquivo(s):** `src/models/wavenet_params.rs` (novo)
- **O quê:** Portar structs data-only de `NAM/wavenet/params.h`: `Head1x1Params`, `Layer1x1Params`, `LayerParamsA2` (19+ campos), `LayerArrayParamsA2`, `HeadParams`.
- **Ref C++:** `NAM/wavenet/params.h` L36-316
- [x] Criar módulo com todas as structs A2
- [x] Doc-tests de construção

### TH5 · Variante `DynamicModel::WavenetA2` (Placeholder) [DONE]

- **Arquivo(s):** `src/models/mod.rs`
- **O quê:** Adicionar `WavenetA2(Box<WavenetA2Placeholder>)` ao enum. Impl `NamModel` com `process()` que retorna zeros + log warning. Permite loader aceitar A2 sem panic.
- **Ref:** `DynamicModel` em `src/models/mod.rs` L62-89
- [x] Criar placeholder struct + NamModel impl
- [x] Adicionar variante ao enum + match arms
- [x] Teste unitário

### TH6 · Forward-Compatible Parsing no Loader [DONE]

- **Arquivo(s):** `src/loader/nam_json.rs`, `src/loader/dispatcher/wavenet.rs`
- **O quê:** Garantir que JSON com campos A2 extras não cause panic → fallback gracioso para placeholder. Modelos A1 inalterados.
- **Ref C++:** `NAM/get_dsp.cpp` L237-260
- [x] Auditar forward-compatibility do loader (`NamModelData::is_wavenet_a2()`)
- [x] Fixture JSON mock A2 em `tests/fixtures/models/mock_a2.nam`
- [x] Teste integração: A2 mock → placeholder (`test_forward_compatibility_wavenet_a2`)
- [x] Teste regressão: A1 fixtures ok (`test_accept_a2_activation_with_fallback` + suite completa)

---

## Épico I — Staging CLAP Plugin (Trait `AudioHost`)

> Desacoplar motor DSP do PipeWire. Zero dependência CLAP adicionada.
> Ref: [free-audio/clap](https://github.com/free-audio/clap)
> Opções futuras: `nih-plug` (framework) ou `clack` (safe wrapper)

### TI1 · Trait `AudioHost`

- **Arquivo(s):** `src/audio_host.rs` (novo)
- **O quê:** Trait com `sample_rate()`, `max_buffer_size()`, `run()`. Abstrai lifecycle (não hot-path). Será implementado para PipeWireHost e futuramente ClapPlugin.
- [ ] Criar trait + registrar em `lib.rs`

### TI2 · Feature Flags: `standalone` vs `clap-plugin`

- **Arquivo(s):** `Cargo.toml`, `src/lib.rs`, `src/main.rs`
- **O quê:** Feature `standalone` (default) → `dep:pipewire`. Feature `clap-plugin` vazia. Condicionar `pw_host.rs`/`rt_setup.rs` a `#[cfg(feature = "standalone")]`.
- **Cuidado:** `cargo build` default inalterado. `cargo check --no-default-features` compila engine puro.
- [ ] Feature flags no `Cargo.toml`
- [ ] `#[cfg]` em `lib.rs`, `pw_host.rs`, `rt_setup.rs`, `main.rs`
- [ ] Verificar ambas compilações

### TI3 · Struct `NamPluginParams`

- **Arquivo(s):** `src/params.rs` (novo)
- **O quê:** Parâmetros agnósticos ao host: `input_gain_db`, `output_gain_db`, `gate_threshold_db`, `model_path`, `bypass`. Coexiste com `ParamPayload`.
- [ ] Criar struct + `Default` impl + registrar em `lib.rs`

### TI4 · Documentação CLAP

- **Arquivo(s):** `docs/clap_integration.md` (novo)
- **O quê:** Thread model CLAP vs NAM-rs, mapeamento de params, estratégia de compilação, DAWs alvo, decisão de crate pendente.
- [ ] Criar documento

---

## Épico J — Consolidação de Documentação

> Atualizar docs para v1.3 (completo) + preparações v1.4.

### TJ1 · Atualização de `docs/architecture.md`

- **Arquivo(s):** `docs/architecture.md`
- **O quê:** Kernels SIMD dos Épicos E/F, seção "Preparação A2", seção "Roadmap CLAP", tabela de módulos atualizada.
- [ ] Atualizar e acionar skill `documentador`

### TJ2 · Enxugamento de `.agents/rules/rust.md`

- **Arquivo(s):** `.agents/rules/rust.md`
- **O quê:** Adicionar menção a feature flags, compilação sem PW, módulos A2 stub. Manter < 40 linhas.
- [ ] Revisar e condensar

### TJ3 · Atualização do `README.md`

- **Arquivo(s):** `README.md`
- **O quê:** Seção "Roadmap" (A2 + CLAP como futuro). Versão 1.4.0-staging.
- [ ] Atualizar README

---

## Épico K — Cobertura de Testes v1.4

> Garantir zero regressão com as preparações.

### TK1 · Testes `ActivationType`

- **Arquivo(s):** `src/models/activations.rs` (inline)
- **O quê:** Testes para cada variante em 0.0, ±1.0, ±5.0 vs golden values C++.
- [ ] Testes inline para 11 ativações

### TK2 · Testes Forward-Compatibility Loader

- **Arquivo(s):** `tests/loader_a2_compat.rs` (novo)
- **O quê:** Fixture A2 mock → placeholder. Regressão A1 fixtures.
- [ ] Fixture + testes integração

### TK3 · Teste Compilação Condicional

- **Arquivo(s):** `utils/check_features.sh` (novo)
- **O quê:** Verificar `--features standalone`, `--no-default-features`, `--features clap-plugin`.
- [ ] Criar script de verificação

### TK4 · Smoke Tests `NamPluginParams` + `AudioHost`

- **Arquivo(s):** `src/params.rs`, `src/audio_host.rs` (inline)
- **O quê:** `Default` retorna valores sensatos. Mock impl de `AudioHost`.
- [ ] Testes inline

---

## Observações — Itens Diferidos (v1.4)

### Diferidos A2

- Implementação FiLM SIMD, ConvNet model, Slimmable inference, ConfigParserRegistry, NAMB v3

### Diferidos CLAP

- Dependência CLAP (clap-sys/nih-plug/clack), cdylib target, GUI, MIDI, State save/load

---

## Notas Pós-Auditoria — Épico H

> **Status**: Épico H concluído com 100% de cobertura. Todos os 6 TH tasks implementados
> e validados. Suite de testes: 91 unitários + 30 integração, todos passando.

### Legado do Épico H para Épicos Futuros

- **Para Épico I (CLAP)**: O trait `NamModel` existente + enum `DynamicModel` já suportam
  A2 via `WavenetA2Placeholder`. A interface de plugin pode usar `build_model()` sem
  modificações — o dispatcher já é forward-compatible.

- **Para Kernel A2 Real**: Quando o kernel A2 for implementado, o caminho de substituição
  é trivial: substituir `WavenetA2Placeholder` por `WavenetA2Kernel` em `DynamicModel`.
  Toda a estrutura de parâmetros (`LayerParamsA2`, `FiLMConfig`, `GatingMode`) já está
  pronta em `src/models/`.

- **Detecção A2**: `NamModelData::is_wavenet_a2()` detecta via versão (`0.6.x`) OU
  ativação não-Tanh. A heurística é conservadora — não pode gerar falsos positivos em
  modelos A1 existentes.

- **TK2 em Épico K**: O teste `tests/loader_a2_compat.rs` separado foi incorporado
  diretamente em `tests/nam_infer_test.rs` (via `test_forward_compatibility_wavenet_a2`).
  O Épico K/TK2 pode ser marcado como pré-satisfeito ou adaptado para cenários adicionais.
