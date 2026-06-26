<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-findings.md — Auditoria NeuralAmpModelerCore v0.5.4 → NAM-rs

> **Origem:** Revisão técnica da release [v0.5.4](https://github.com/sdatkinson/NeuralAmpModelerCore/releases/tag/v0.5.4) do NeuralAmpModelerCore.
> **Data da auditoria:** 2026-06-25
> **Referência upstream:** `v0.5.3 → v0.5.4` ([diff](https://github.com/sdatkinson/NeuralAmpModelerCore/compare/v0.5.3...v0.5.4))
> **Fixture pin atualizado:** `utils/mod-update.sh` → `v0.5.4` @ `1f42f88535884450104b8711d7595019afa0495b`

---

## Resumo Executivo

A release v0.5.4 do NeuralAmpModelerCore traz **sete mudanças acionáveis** com impacto direto para o NAM-rs:

| #   | Área                                                     | PRs upstream                                                                                                                             | Impacto NAM-rs                                    |
| --- | -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| F1  | **Convolução FFT particionada** para modelo Linear       | [#278](https://github.com/sdatkinson/NeuralAmpModelerCore/pull/278), [#280](https://github.com/sdatkinson/NeuralAmpModelerCore/pull/280) | 🔴 Alto — Performance de IRs longas (~4000+ taps) |
| F2  | **Controle granular de Prewarm-on-Reset**                | Embedded no diff principal                                                                                                               | 🟡 Médio — API pública, UX de carregamento        |
| F3  | **DspLoadOptions + desacoplamento prewarm do get_dsp**   | Embedded no diff principal                                                                                                               | 🟡 Médio — API de loading                         |
| F4  | **Novos kernels GEMV/Conv1x1 unrolled** + `NAM_RESTRICT` | Embedded no diff principal                                                                                                               | 🟢 Baixo — NAM-rs já usa kernels SIMD próprios    |
| F5  | **GetSlimmableSizeBreakpoints** (Slimmable API)          | Embedded no diff principal                                                                                                               | 🟡 Médio — CLAP parameter mapping                 |
| F6  | **Novo modelo de exemplo A2.nam** (SlimmableContainer)   | Embedded no diff principal                                                                                                               | 🟢 Baixo — Fixture de teste de paridade           |
| F7  | **Propagação `SetPrewarmOnReset`** em Container/WaveNet  | Embedded no diff principal                                                                                                               | 🟡 Médio — Gap de paridade em Container/Slimmable |

---

## F1 — Convolução FFT Particionada para Modelo Linear

### F1 — Contexto

O modelo `Linear` no NeuralAmpModelerCore era originalmente uma convolução direta pura (FIR dot-product), o que funciona bem para IRs curtas (< 256 taps). No entanto, IRs longas (cab sims com 2048–8192+ taps a 48 kHz) tornam a convolução direta proibitivamente cara: **O(N) por sample**, onde N é o tamanho do receptive field.

A v0.5.4 introduz:

- **Separação em arquivos:** `NAM/linear.h` e `NAM/linear.cpp` (antes o Linear estava inline em `dsp.cpp`/`dsp.h`).
- **Enum `LinearImplementation`**: `Auto`, `Direct`, `FFT`.
- **Convolução FFT zero-latência particionada** (overlap-save/overlap-add com ring buffer):
  - Partição automática baseada no tamanho do receptive field.
  - Block sizes adaptativos: 256 (< 2048 taps), 512 (< 8192 taps), 1024 (≥ 8192 taps).
  - Taps iniciais processados em convolução direta (zero latência), cauda via FFT particionada.
  - Estado FFT em struct opaca `LinearFFTState` com `Eigen::FFT<float>`.
- **Campo `implementation` no JSON** da config do modelo (suporta `"auto"`, `"direct"`, `"fft"`, `"legacy"`, etc).

### F1 — Estado Atual no NAM-rs

O [`LinearModel`](file:///home/fabio/nam-rs/src/models/linear.rs) do NAM-rs implementa **apenas convolução direta** via dot-product com `MirroredBuffer` + SIMD (`convolve_mono`). Não existe caminho FFT.

### F1 — Proposta de Solução

1. **Reutilizar a infraestrutura interna de FFT e convolução do NAM-rs.** O projeto já possui:
   - [`RfftPlanner`](file:///home/fabio/nam-rs/src/math/dsp/fft.rs) — FFT real nativa Radix-2 DIT com SIMD AVX2/AVX-512, zero-alloc após setup.
   - [`ConvEngine`](file:///home/fabio/nam-rs/src/dsp/cabsim/conv.rs) — Motor UPOLS (Uniform-Partitioned Overlap-Save) completo, já usado no CabSim, com FDL (Frequency Delay Line) pré-alocada em SoA, hot-path zero-alloc.
   - **Não usar `rustfft`**: decisão de projeto já tomada — foco em implementação interna opinativa e otimizada para as necessidades do NAM-rs (SoA layout, `AlignedVec`, dispatch SIMD nativo).
2. **Estratégia de auto-seleção**: Se `receptive_field <= 256` → Direct (atual), senão → FFT particionada.
3. **Layout do módulo**:
   - `src/models/linear.rs` → enum `LinearImpl { Direct, Fft }` + dispatch no `process()`.
   - `src/models/linear_fft.rs` → adaptador que instancia `RfftPlanner` + lógica de convolução particionada.
4. **Diferença chave vs. `ConvEngine`**: O `ConvEngine` do CabSim impõe latência de `partition_size` samples. O modelo Linear do upstream usa **convolução híbrida zero-latência**: os primeiros `direct_taps` são calculados por dot-product direto (sem latência), e apenas a cauda (partições FFT) é processada via overlap-save. Isso é um requisito de paridade — o modelo `.nam` Linear NÃO pode introduzir latência.
5. **RT-Safety obrigatória**:
   - O `RfftPlanner` pré-computa twiddle factors na construção — `process` é zero-alloc.
   - Todos os buffers (input ring, spectrum FDL, accumulator, IFFT output) devem ser `AlignedVec<f32>` pré-alocados.
   - Ring buffer de output com wrap-around módulo (sem branches no hot-path).
   - Nenhum `unwrap()`, nenhuma alocação heap, nenhum lock.
6. **Paridade de testes**: Gerar golden outputs com o C++ v0.5.4 para um IR longo e comparar bit-a-bit.

### F1 — Risco e Prioridade

- **Prioridade:** 🔴 Alta — Sem FFT, carregar um modelo `.nam` com IR de 4000+ taps será ordens de magnitude mais lento que o C++. Isso é especialmente crítico para simulação de cabinets (cab sims) que tipicamente usam receptive fields longos.
- **Risco:** Médio — A lógica híbrida (direct + FFT particionada zero-latência) é mais complexa que o UPOLS puro do CabSim, mas o `RfftPlanner` e a lógica de particionamento são componentes já testados.

---

## F2 — Controle Granular de Prewarm-on-Reset

### F2 — Contexto

A v0.5.4 introduz uma refatoração significativa na API de prewarm:

1. **`PrewarmSamples()` → `GetPrewarmSamples()`**: Renomeado de `protected` para `public virtual`. Agora todas as subclasses (`ConvNet`, `WaveNet`, `LSTM`, `A2FastModel`, `ContainerModel`, `SlimmableWavenet`) expõem o método publicamente.

2. **`ScopedPrewarmOnResetDefault`** (classe RAII, thread_local):

   - Permite controlar se novas instâncias DSP criadas na thread atual terão prewarm-on-reset ativado ou não.
   - Usa `thread_local bool gPrewarmOnResetDefault = true`.
   - Instanciada como `ScopedPrewarmOnResetDefault scoped(false)` → todas as DSP criadas dentro do escopo nascem com prewarm desligado.

3. **`DSP::mPrewarmOnReset`** (`std::atomic<bool>`):

   - Cada instância DSP guarda seu próprio flag.
   - `Reset()` agora consulta `GetPrewarmOnReset()` antes de chamar `prewarm()`.
   - `SetPrewarmOnReset(bool)` é virtual para propagar para sub-modelos (Container, WaveNet com condition_dsp, Slimmable).

4. **`ResetAndPrewarm()` marcado como deprecated** — Reset() agora faz prewarm por padrão.

### F2 — Estado Atual no NAM-rs

- `NamModel::prewarm_samples()` já existe como método público (default retorna 0).
- `NamModel::reset()` chama `self.prewarm()` internamente — similar ao novo comportamento default do C++.
- Não existe flag `prewarm_on_reset` ou equivalente ao `ScopedPrewarmOnResetDefault`.
- Ao carregar um modelo no [`loader/build.rs`](file:///home/fabio/nam-rs/src/loader/build.rs), prewarm é chamado explicitamente após construção.

### F2 — Proposta de Solução

1. **Adicionar campo `prewarm_on_reset: bool`** em cada modelo (via trait default ou campo nos structs concretos).
2. **Método `set_prewarm_on_reset(&mut self, val: bool)`** no trait `NamModel` com implementação default que seta o campo.
3. **Propagação em Container/Slimmable**: Override que propaga para sub-modelos.
4. **Não portar `ScopedPrewarmOnResetDefault`**: O padrão thread_local + RAII com escopo é um idioma C++ para contornar limitações de API. Em Rust, podemos resolver via `LoadOptions` (ver F3) passado explicitamente, sem recorrer a estado global thread_local.

### F2 — Risco e Prioridade

- **Prioridade:** 🟡 Média — Atualmente NAM-rs sempre prewarms ao Reset. Se um host DAW faz Reset frequente (mudança de sample rate, etc.), o prewarm desnecessário causa spikes de CPU.
- **Risco:** Baixo — Mudança de API simples.

---

## F3 — DspLoadOptions e Desacoplamento do Prewarm no Carregamento

### F3 — Contexto

A v0.5.4 introduz `DspLoadOptions` como parâmetro em todas as sobrecargas de `get_dsp()`:

```cpp
struct DspLoadOptions {
  std::optional<bool> prewarm = std::nullopt;
};
```

- Se `prewarm` é `std::nullopt` → usa o default thread_local atual.
- Se `prewarm` é `false` → cria o modelo sem prewarm (carregamento rápido).
- Se `prewarm` é `true` → força prewarm.

Além disso, **`create_dsp()` não chama mais `prewarm()` diretamente** — o prewarm é feito por `Reset()` (que agora consulta o flag `mPrewarmOnReset`).

### F3 — Estado Atual no NAM-rs

O [`loader/build.rs`](file:///home/fabio/nam-rs/src/loader/build.rs) chama `m.prewarm()` explicitamente após construir o modelo. Não existe estrutura `LoadOptions`.

### F3 — Proposta de Solução

1. **Criar `LoadOptions`** em `src/loader/mod.rs`:

   ```rust
   pub struct LoadOptions {
       pub prewarm: Option<bool>, // None = default (true), Some(false) = skip
   }
   impl Default for LoadOptions {
       fn default() -> Self { Self { prewarm: None } }
   }
   ```

2. **Propagar para `build_model_pipeline()`** e funções de loading.

3. **Remover prewarm explícito** de `create_dsp()` equivalente — delegar ao `reset()`.

### F3 — Risco e Prioridade

- **Prioridade:** 🟡 Média — Permite carregamento instantâneo (skip prewarm) para preview/browsing de modelos.
- **Risco:** Baixo.

---

## F4 — Novos Kernels GEMV/Conv1x1 Unrolled + NAM_RESTRICT

### F4 — Contexto

A v0.5.4 adiciona vários novos kernels inline GEMV especializados:

#### F4 — Conv1D (conv1d.cpp)

- **8×4 fully unrolled** — Conv1D com out_ch=8, in_ch=4.
- **1×4 fully unrolled** — Conv1D com out_ch=1, in_ch=4.

#### F4 — Conv1x1 (dsp.cpp)

- **4×4 com bias fusionado** — Conv1x1 com bias integrado no loop (evita segundo passe).
- **4×6** — Conv1x1 com out_ch=4, in_ch=6 (novo).
- **8×6** — Conv1x1 com out_ch=8, in_ch=6 (novo).

#### F4 — Portabilidade

- **`NAM_RESTRICT`** macro (compiler.h): Unifica `__restrict__` (GCC/Clang) vs `__restrict` (MSVC) num macro único. Aplicado em **todos** os kernels GEMV.

### F4 — Estado Atual no NAM-rs

NAM-rs **não porta diretamente os kernels GEMV do C++**. Em vez disso, usa kernels SIMD genéricos próprios em `src/math/` que operam com tamanho dinâmico via `chunks_exact()` + intrínsecas AVX2/FMA. As dimensões de canal são tratadas genericamente pelo dispatch SIMD, sem unrolling por dimensão específica.

A macro `NAM_RESTRICT` é irrelevante em Rust — o compilador Rust já trata `&mut` como `noalias` (equivalente a `restrict`).

### F4 — Proposta de Solução

1. **Avaliar se os kernels especializados por dimensão trazem ganho mensurável em Rust/AVX2.**
   - Hipótese: O LLVM já unrolla loops pequenos com `chunks_exact(4)` quando o trip count é constante.
   - Benchmarkar com `criterion` os tamanhos 1×4, 4×4, 4×6, 8×4, 8×6, 8×8 vs. o kernel genérico atual.
2. **Se houver ganho > 5%**: Criar kernels monomorphizados via const generics `<const IN: usize, const OUT: usize>` em `src/math/gemv.rs`.
3. **Se não houver ganho**: Documentar que o dispatch genérico já cobre esses casos e fechar o finding.

### F4 — Risco e Prioridade

- **Prioridade:** 🟢 Baixa — NAM-rs já tem kernels SIMD otimizados. O ganho potencial é marginal.
- **Risco:** Baixo — Benchmark-driven, sem risco arquitetural.

> **Nota:** Este finding complementa o Épico C já existente sobre "Kernels GEMV/LSTM afinados" (se houver referência em conversas anteriores).

---

## F5 — GetSlimmableSizeBreakpoints (API Slimmable)

### F5 — Contexto

A v0.5.4 adiciona ao trait `SlimmableModel`:

```cpp
virtual std::vector<double> GetSlimmableSizeBreakpoints() const { return {}; }
```

Retorna os **valores de breakpoint normalizados** (0.0, 1.0) que dividem os sub-modelos selecionáveis no `ContainerModel` ou `SlimmableWavenet`.

- `ContainerModel::GetSlimmableSizeBreakpoints()` retorna os `max_value` dos submodelos (exceto o último).
- `SlimmableWavenet::GetSlimmableSizeBreakpoints()` computa breakpoints a partir dos ratios de canais permitidos por array.

### F5 — Utilidade

Isso é essencial para **plugins CLAP/VST** que precisam discretizar o parâmetro "Model Size" no host DAW. Os breakpoints indicam os valores exatos onde o modelo muda de sub-modelo/configuração, permitindo:

- Snapping do knob do host nos pontos de transição.
- Exibição de labels ("Small", "Medium", "Large").
- Automação precisa.

### F5 — Estado Atual no NAM-rs

O [`SlimmableModel`](file:///home/fabio/nam-rs/src/models/slimmable.rs) do NAM-rs **não expõe breakpoints**. O trait não tem método equivalente. O [`ContainerModel`](file:///home/fabio/nam-rs/src/models/container.rs) também não.

### F5 — Proposta de Solução

1. **Adicionar `fn slimmable_breakpoints(&self) -> Vec<f64>`** ao trait `SlimmableModel` com default `vec![]`.
2. **Implementar em `ContainerModel`**: retornar os `max_value` dos sub-modelos.
3. **Implementar em `SlimmableWavenet`**: computar breakpoints por ratio de canais.
4. **Expor via CLAP parameter info** no plugin, usando os breakpoints para definir steps discretos ou labels.

### F5 — Risco e Prioridade

- **Prioridade:** 🟡 Média — Necessário para UX adequada do parâmetro Model Size no plugin CLAP.
- **Risco:** Baixo.

---

## F6 — Novo Modelo de Exemplo A2.nam (SlimmableContainer)

### F6 — Contexto

A v0.5.4 inclui um novo modelo de exemplo `example_models/A2.nam` que usa a arquitetura `SlimmableContainer` com dois sub-modelos WaveNet (channels=3 e channels=6) e breakpoints definidos. Este é o primeiro exemplo público de modelo A2 slimmable.

### F6 — Estado Atual no NAM-rs

NAM-rs já suporta `SlimmableContainer` e A2. Porém, este modelo pode ser usado como **fixture de teste adicional** para validação de paridade C++.

### F6 — Proposta de Solução

1. Incluir o `A2.nam` nos testes de integração de paridade C++, se não estiver já presente.
2. Usar como golden test para a combinação `SlimmableContainer` + `WaveNet A2`.

### F6 — Risco e Prioridade

- **Prioridade:** 🟢 Baixa — Melhoria de cobertura de testes.
- **Risco:** Nenhum.

---

## F7 — `ContainerModel::SetPrewarmOnReset` Propagação

### F7 — Contexto

A v0.5.4 adiciona `ContainerModel::SetPrewarmOnReset()` que propaga o flag para todos os sub-modelos:

```cpp
void ContainerModel::SetPrewarmOnReset(const bool prewarmOnReset) {
  DSP::SetPrewarmOnReset(prewarmOnReset);
  for (auto& submodel : _submodels)
    submodel.model->SetPrewarmOnReset(prewarmOnReset);
}
```

Mesmo padrão em `WaveNet::SetPrewarmOnReset()` (propaga para `_condition_dsp`) e `SlimmableWavenet::SetPrewarmOnReset()` (propaga para `_active_model` e pending model).

### F7 — Estado Atual no NAM-rs

Container e Slimmable **não propagam** configurações de prewarm para sub-modelos. Este é um gap de paridade.

### F7 — Proposta de Solução

Coberto por F2 (propagação do flag `prewarm_on_reset`).

---

## Épicos de Implementação

### Épico D — "Convolução FFT Particionada para Linear" (F1)

**Escopo:** Implementar convolução FFT zero-latência particionada para o modelo Linear, com auto-seleção baseada no tamanho do receptive field.

**Findings:** F1

**Dependências:** Nenhuma (módulo isolado).

**Estimativa:** ~3 sprints (pesquisa/PoC, implementação, testes de paridade).

---

### Épico E — "Controle de Prewarm e LoadOptions" (F2, F3, F7)

**Escopo:** Implementar flag `prewarm_on_reset` nos modelos, `LoadOptions` no loader, e propagação para Container/Slimmable/WaveNet com condition_dsp.

**Findings:** F2, F3, F7

**Dependências:** Nenhuma.

**Estimativa:** ~1 sprint.

---

### Épico F — "API Slimmable Breakpoints" (F5)

**Escopo:** Expor breakpoints de transição do SlimmableModel para uso em plugins CLAP.

**Findings:** F5

**Dependências:** Nenhuma (pode ser feito a qualquer momento).

**Estimativa:** < 1 sprint.

---

### Épico G — "Benchmark Kernels GEMV Especializados" (F4)

**Escopo:** Benchmark-driven investigation: medir se kernels especializados por dimensão (1×4, 4×4, 4×6, 8×4, 8×6, 8×8) superam os kernels genéricos do NAM-rs. Implementar apenas se ganho > 5%.

**Findings:** F4

**Dependências:** Épico C (GEMV/LSTM kernels) se existente.

**Estimativa:** ~1 sprint (benchmark + decisão).

---

### Épico H — "Fixture de Teste A2 Slimmable" (F6)

**Escopo:** Adicionar o modelo de exemplo `A2.nam` como fixture de teste de paridade C++.

**Findings:** F6

**Dependências:** Nenhuma.

**Estimativa:** < 1 sprint.

---

## Prioridade Recomendada de Execução

| Ordem | Épico                       | Justificativa                                                 |
| ----- | --------------------------- | ------------------------------------------------------------- |
| 1     | **E** (Prewarm/LoadOptions) | Requisito de paridade de API, baixo risco, alto impacto em UX |
| 2     | **D** (FFT Linear)          | Feature nova de alta performance, necessária para IRs longas  |
| 3     | **F** (Breakpoints)         | Necessário para plugin CLAP, implementação simples            |
| 4     | **H** (Fixture A2)          | Melhoria de cobertura de testes                               |
| 5     | **G** (Kernels GEMV)        | Investigação benchmark-driven, ganho incerto                  |
