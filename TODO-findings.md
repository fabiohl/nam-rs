<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-findings.md — Auditoria NeuralAmpModelerCore v0.5.4 → NAM-rs

> **Origem:** Revisão técnica da release [v0.5.4](https://github.com/sdatkinson/NeuralAmpModelerCore/releases/tag/v0.5.4) do NeuralAmpModelerCore + **Auditoria complementar de paridade plena** de todo o NeuralAmpModelerCore.
> **Data da auditoria:** 2026-06-25 (v0.5.4), 2026-06-26 (paridade plena)
> **Referência upstream:** `v0.5.3 → v0.5.4` ([diff](https://github.com/sdatkinson/NeuralAmpModelerCore/compare/v0.5.3...v0.5.4)) + revisão integral da árvore C++ v0.5.4
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

**Auditoria complementar de paridade plena (2026-06-26):**

| #   | Área                                                         | Fonte                   | Impacto NAM-rs                                                |
| --- | ------------------------------------------------------------ | ----------------------- | ------------------------------------------------------------- |
| F8  | **LSTM `prewarm_samples` retorna 0** (deveria ser 0.5s × SR) | `lstm.cpp:125-132`      | 🔴 Alto — LSTM não prewarms corretamente após Reset           |
| F9  | **`ContainerModel::Reset` — Reset seletivo só do sub-ativo** | `container.cpp:71-83`   | 🟡 Médio — Desperdício de CPU ao resetar sub-modelos inativos |
| F10 | **SlimmableWavenet — rebuild atômico lock-free**             | `slimmable.cpp:489-498` | 🟡 Médio — NAM-rs `SlimmableModel` diverge na API de staging  |
| F11 | **Input/Output Level + Loudness metadata**                   | `dsp.h:110-149`         | 🟢 Baixo — Campos parcialmente expostos                       |
| F12 | **`LinearImplementation` enum no JSON**                      | `linear.h:11-16`        | 🟢 Baixo — Parser não lê campo `implementation` do JSON       |

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
   - **Resultado (2026-06-26): Ganho massivo em todas as 6 dimensões testadas.**
     - 1×4: +40.0% (15.0ns → 9.0ns)
     - 4×4: +50.4% (23.2ns → 11.5ns)
     - 4×6: +49.5% (28.1ns → 14.2ns)
     - 8×4: +72.9% (46.8ns → 12.7ns)
     - 8×6: +65.5% (62.6ns → 21.6ns)
     - 8×8: +21.2% (15.6ns → 12.3ns)
   - **Decisão: Prosseguir com implementação definitiva (Tarefa 3 do Sprint 6).**
   - O LLVM **não** unrolla suficientemente os loops do kernel genérico — o overhead de loop+branch é significativo mesmo no caso ótimo (8×8, bloco interno alinhado).
2. **Criar kernels monomorphizados via const generics** em `src/math/gemm/gemv/f16_avx2.rs`.
3. **Corrigir UB dos protótipos de benchmark** ao promover para produção:
   - **Store YMM→buffer parcial**: `_mm256_storeu_ps` escreve 32 bytes — usar temp `[f32; 8]` + cópia parcial para `out_len < 8`.
   - **Load YMM de slice parcial**: `_mm256_loadu_ps` lê 32 bytes de alocações de 16–24 bytes (UB). Usar temp `[f32; 8]` zero-padded + `_mm_loadu_ps` (128-bit) ou `_mm256_insertf128_ps`.

### F4 — Risco e Prioridade

- **Prioridade:** 🟢 Baixa → 🟡 Média — Reavaliada após benchmarks: ganho é massivo (21–73%), justificando implementação.
- **Risco:** Baixo — Benchmark-driven, sem risco arquitetural.

> **Nota:** Este finding complementa o Épico G em andamento. Benchmarks executados em `benches/gemv_bench.rs` (Sprint 6, Tarefas 1–2).

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

## F8 — LSTM `prewarm_samples` Retorna 0 (Deveria Ser 0.5s × Sample Rate)

### F8 — Contexto

No C++ (`lstm.cpp:125-132`), o método `LSTM::GetPrewarmSamples()` retorna:

```cpp
int nam::lstm::LSTM::GetPrewarmSamples() {
  int result = (int)(0.5 * mExpectedSampleRate);
  return result <= 0 ? 1 : result;
}
```

Isto significa que ao chamar `Reset()` (com `prewarmOnReset=true`), o LSTM processa **meio segundo** de silêncio a cada reset. Isso é fundamental para estabilizar o estado recorrente do LSTM — modelos LSTM não têm receptive field fixo, e a memória recorrente precisa de tempo para "convergir" a um estado estacionário.

### F8 — Estado Atual no NAM-rs

As implementações de LSTM no NAM-rs (`LstmModel1`, `LstmModel2`, `LstmModelDyn`) **não implementam `prewarm_samples()`**, herdando o default do trait `NamModel` que retorna `0`. Isto significa:

1. **O LSTM nunca prewarma via `Reset()`.** Quando um host DAW chama Reset (mudança de sample rate), o modelo LSTM é resetado com `reset_states()` mas **não reprocessa silêncio**, violando paridade.
2. **O LSTM não armazena `expected_sample_rate`.** Não existe campo equivalente a `mExpectedSampleRate` nos structs LSTM do NAM-rs. Este valor vem do JSON do modelo e é necessário para calcular o número de samples de prewarm.
3. **A implementação de `reset()` no LSTM é "light" (apenas `reset_states()`)**, enquanto o C++ faz `Reset()` → `prewarm()` com `GetPrewarmSamples()` frames de silêncio.

### F8 — Proposta de Solução

1. **Armazenar `expected_sample_rate: f64`** em cada struct LSTM (`LstmModel1`, `LstmModel2`, `LstmModelDyn`), preenchido durante a construção pelo loader.

2. **Implementar `prewarm_samples(&self) -> usize`** nos LSTMs:

   ```rust
   fn prewarm_samples(&self) -> usize {
       let result = (0.5 * self.expected_sample_rate) as isize;
       if result <= 0 { 1 } else { result as usize }
   }
   ```

3. **Atualizar `reset()`** para chamar `prewarm(self.prewarm_samples())` em vez de apenas `reset_states()`, alinhando com o comportamento do C++ `Reset()` → `prewarm()`.

4. **Propagar `expected_sample_rate`** durante a construção em todos os dispatchers LSTM (`src/loader/dispatcher/lstm/`).

### F8 — Risco e Prioridade

- **Prioridade:** 🔴 Alta — Sem isso, o LSTM pode produzir artefatos audíveis (clicks, DC offset) ao mudar de sample rate em uma DAW, pois o estado recorrente não é estabilizado.
- **Risco:** Baixo — Mudança localizada (campos + override de trait method).

---

## F9 — `ContainerModel::Reset` — Reset Seletivo Apenas do Sub-Modelo Ativo

### F9 — Contexto

No C++ (`container.cpp:71-83`), `ContainerModel::Reset()` reseta **apenas o sub-modelo ativo**:

```cpp
void ContainerModel::Reset(const double sampleRate, const int maxBufferSize) {
  std::lock_guard<std::mutex> lock(_slim_set_mutex);
  mExternalSampleRate = sampleRate;
  mHaveExternalSampleRate = true;
  SetMaxBufferSize(maxBufferSize);
  const size_t active_index = _active_index.load(std::memory_order_acquire);
  _submodels[active_index].model->Reset(sampleRate, maxBufferSize);
}
```

Além disso, `SetSlimmableSize()` faz reset do sub-modelo **antes** de ativá-lo, garantindo que qualquer sub-modelo que entre no path de processamento já esteja resetado:

```cpp
_submodels[active_index].model->Reset(sr, GetMaxBufferSize());
_active_index.store(active_index, std::memory_order_release);
```

### F9 — Estado Atual no NAM-rs

O [`ContainerModel::reset()`](file:///home/fabio/nam-rs/src/models/container.rs) do NAM-rs **não implementa reset seletivo** — analisa-se que `reset()` herda o default do trait que chama `prewarm(max_buffer_size)`, que processa o sub-modelo ativo. No entanto:

1. **Não armazena `external_sample_rate` / `max_buffer_size`** persistentemente para uso no `set_slimmable_size`.
2. **`set_slimmable_size()` não faz reset/prewarm do sub-modelo alvo antes de ativá-lo**, podendo causar artefatos se o sub-modelo destino nunca foi resetado no sample rate corrente.

### F9 — Proposta de Solução

1. **Armazenar `external_sample_rate: u32` e `external_max_buffer_size: usize`** no `ContainerModel`, atualizados em `reset()`.
2. **No `set_slimmable_size()`**: antes de ativar um sub-modelo diferente, chamar `sub_model.reset(external_sample_rate, external_max_buffer_size)` e `sub_model.prewarm(sub_model.prewarm_samples())` no novo sub-modelo.
3. Manter o lock do crossfade existente como mecanismo de serialização.

### F9 — Risco e Prioridade

- **Prioridade:** 🟡 Média — Sem isso, trocar de sub-modelo via `set_slimmable_size` pode produzir artefatos quando o sub-modelo destino não foi resetado para o sample rate corrente.
- **Risco:** Baixo.

---

## F10 — SlimmableWavenet: Rebuild Atômico Lock-Free com Staging

### F10 — Contexto

O C++ `SlimmableWavenet` (`slimmable.cpp:489-498`) implementa um modelo de staging com swap atômico lock-free no hot-path:

```cpp
void SlimmableWavenet::process(NAM_SAMPLE** input, ...) {
  if (auto pack = _pending_exchange_take_acq_rel()) {
    _active_model = std::move(pack->model);
    _current_channels = std::move(pack->channels);
  }
  if (_active_model)
    _active_model->process(input, output, num_frames);
}
```

- `SetSlimmableSize()` chama `_stage_rebuild_model()` que constrói o novo WaveNet em background e publica via `_pending_store_release()` (atômico).
- `process()` faz `exchange` atômico para tomar o modelo staged — **zero locks no hot-path**.
- `SetPrewarmOnReset()` propaga para `_active_model` e `_pending_staged`.

### F10 — Estado Atual no NAM-rs

O NAM-rs [`SlimmableModel`](file:///home/fabio/nam-rs/src/models/slimmable.rs) usa um design diferente:

- `slice_wavenet_model()` + `try_slimmable_rebuild_single()` fazem o rebuild e swap via SPSC GC pipeline (`GcItem::Model`).
- O design é funcional, mas **a API de staging é diferente**: o C++ usa `shared_ptr` atômico (acquire/release) para swap no `process()`, enquanto NAM-rs usa `Option<Box<StaticModel>>` com `replace()` e GC offload.
- **Gap**: o NAM-rs não implementa o equivalente a `_stage_rebuild_model()` — o rebuild ocorre no path do caller de `try_slimmable_rebuild_single()`, não em uma thread de staging separada.

### F10 — Proposta de Solução

1. **Documentar que o design diverge intencionalmente** — o NAM-rs usa SPSC GC em vez de `atomic<shared_ptr>` por design (zero contention vs. ref-counting atômico).
2. **Verificar que `SetPrewarmOnReset` propaga para modelos staged/pending** (gap coberto por F7/F2).
3. **Considerar staging thread** para o rebuild do SlimmableWavenet, se profiles indicarem latência no path do caller.

### F10 — Risco e Prioridade

- **Prioridade:** 🟡 Média — O design é funcionalmente equivalente, mas diverge na ergonomia de staging.
- **Risco:** Baixo — Já funcional.

---

## F11 — Input/Output Level e Loudness Metadata: Exposição Parcial

### F11 — Contexto

A classe `DSP` do C++ (`dsp.h:110-199`) expõe uma API completa de metadados de nível:

- `GetInputLevel()` / `SetInputLevel(double)` — nível de entrada em dBu RMS.
- `GetOutputLevel()` / `SetOutputLevel(double)` — nível de saída em dBu RMS.
- `GetLoudness()` / `SetLoudness(double)` — loudness do modelo em dB.
- `HasInputLevel()` / `HasOutputLevel()` / `HasLoudness()` — flags de disponibilidade.

Estes valores são lidos do JSON via `apply_metadata()` em `get_dsp.cpp:214-222`:

```cpp
void apply_metadata(DSP& dsp, const ModelMetadata& metadata) {
  if (metadata.loudness.has_value()) dsp.SetLoudness(metadata.loudness.value());
  if (metadata.input_level.has_value()) dsp.SetInputLevel(metadata.input_level.value());
  if (metadata.output_level.has_value()) dsp.SetOutputLevel(metadata.output_level.value());
}
```

### F11 — Estado Atual no NAM-rs

O loader do NAM-rs ([`build.rs`](file:///home/fabio/nam-rs/src/loader/build.rs)) já lê `loudness`, `input_level_dbu`, e `output_level_dbu` do JSON e os usa para computar `output_db_adj` (ajuste de ganho de saída). No entanto:

1. **Os valores não são armazenados nos modelos** — são consumidos apenas durante a construção do `LoadedModelPair` para calcular ajustes de ganho.
2. **Não existe API pública** equivalente a `GetLoudness()`, `HasLoudness()`, etc. no trait `NamModel`.
3. **O `input_level_dbu` é lido** mas o tratamento de "has/hasn't" é implícito (default fallback), não exposto como flag.

### F11 — Proposta de Solução

1. **Adicionar campos opcionais ao `LoadedModelPair`** (ou struct equivalente de metadata):

   ```rust
   pub struct ModelMetadata {
       pub loudness: Option<f32>,
       pub input_level_dbu: Option<f32>,
       pub output_level_dbu: Option<f32>,
   }
   ```

2. **Expor via API de consulta** no `LoadedModelPair` ou estrutura análoga — útil para plugins CLAP/VST que precisam informar o host sobre as características do modelo.

3. **Prioridade baixa** — o ajuste de ganho já funciona corretamente; esta finding é sobre completude de API.

### F11 — Risco e Prioridade

- **Prioridade:** 🟢 Baixa — A funcionalidade crítica (ajuste de ganho) já funciona. Falta apenas exposição formal da API.
- **Risco:** Nenhum.

---

## F12 — Campo `implementation` no JSON do Modelo Linear

### F12 — Contexto

A v0.5.4 do C++ suporta um campo `implementation` na config JSON do modelo Linear (`linear.h:84`):

```json
{ "architecture": "Linear", "config": { "receptive_field": 4096, "implementation": "auto" } }
```

Valores suportados: `"auto"`, `"direct"`, `"fft"`, `"legacy"`. Quando `"auto"`, o C++ seleciona FFT para receptive fields grandes.

### F12 — Estado Atual no NAM-rs

O parser de Linear no NAM-rs ([`topology/linear.rs`](file:///home/fabio/nam-rs/src/loader/nam_json/topology/linear.rs)) **não lê o campo `implementation`** do JSON. Como não existe implementação FFT (F1), isso não tem impacto funcional hoje, mas será necessário quando F1 for implementado.

### F12 — Proposta de Solução

1. **Parser**: Adicionar leitura opcional do campo `"implementation"` no `parse_linear_config()`, mapeando para um enum `LinearImpl { Auto, Direct, Fft }`.
2. **Propagar** para o construtor do `LinearModel`.
3. **Implementar junto com F1** — esta finding é dependência direta do Épico D.

### F12 — Risco e Prioridade

- **Prioridade:** 🟢 Baixa — Dependência de F1 (FFT Linear). Sem impacto até F1 ser implementado.
- **Risco:** Nenhum.

---

## Épicos de Implementação

### Épico D — "Convolução FFT Particionada para Linear" (F1) [DONE]

**Escopo:** Implementar convolução FFT zero-latência particionada para o modelo Linear, com auto-seleção baseada no tamanho do receptive field.

**Findings:** F1

**Dependências:** Nenhuma (módulo isolado).

**Estimativa:** ~3 sprints (pesquisa/PoC, implementação, testes de paridade).

---

### Épico E — "Controle de Prewarm e LoadOptions" (F2, F3, F7) [DONE]

**Escopo:** Implementar flag `prewarm_on_reset` nos modelos, `LoadOptions` no loader, e propagação para Container/Slimmable/WaveNet com condition_dsp.

**Findings:** F2, F3, F7

**Dependências:** Nenhuma.

**Estimativa:** ~1 sprint.

---

### Épico F — "API Slimmable Breakpoints" (F5) [DONE]

**Escopo:** Expor breakpoints de transição do SlimmableModel para uso em plugins CLAP.

**Findings:** F5

**Dependências:** Nenhuma (pode ser feito a qualquer momento).

**Estimativa:** < 1 sprint.

---

### Épico G — "Benchmark Kernels GEMV Especializados" (F4) [DONE]

**Escopo:** Benchmark-driven investigation: medir se kernels especializados por dimensão (1×4, 4×4, 4×6, 8×4, 8×6, 8×8) superam os kernels genéricos do NAM-rs. Implementar apenas se ganho > 5%.

**Findings:** F4

**Dependências:** Épico C (GEMV/LSTM kernels) se existente.

**Estimativa:** ~1 sprint (benchmark + decisão).

**Status:** Sprint 6 em andamento. Tarefa 1 (bench suite) e Tarefa 2 (análise/decisão) concluídas. Ganho 21–73% aprovado — prosseguir para Tarefa 3 (implementação).

---

### Épico H — "Fixture de Teste A2 Slimmable" (F6) [DONE]

**Escopo:** Adicionar o modelo de exemplo `A2.nam` como fixture de teste de paridade C++.

**Findings:** F6

**Dependências:** Nenhuma.

**Estimativa:** < 1 sprint.

---

### Épico I — "LSTM Prewarm por Sample Rate" (F8) [DONE]

**Escopo:** Armazenar `expected_sample_rate` nos structs LSTM, implementar `prewarm_samples()` retornando `0.5s × SR`, e alinhar `reset()` com o comportamento do C++ (`Reset()` → `prewarm(GetPrewarmSamples())`).

**Findings:** F8

**Dependências:** Épico E (prewarm_on_reset flag) para interação correta com `prewarm_on_reset`.

**Estimativa:** ~1 sprint.

---

### Épico J — "Container Reset Seletivo + Staging" (F9, F10) [DONE]

**Escopo:** Implementar reset seletivo apenas do sub-modelo ativo no ContainerModel, armazenar `external_sample_rate`/`max_buffer_size` para uso em `set_slimmable_size()`, e verificar propagação de prewarm no SlimmableWavenet.

**Findings:** F9, F10

**Dependências:** Épico E (prewarm_on_reset).

**Estimativa:** ~1 sprint.

---

### Épico K — "Metadados e Parser Linear" (F11, F12) [DONE]

**Escopo:** Expor API completa de metadados (loudness, input/output levels) e adicionar parsing do campo `implementation` no JSON do modelo Linear.

**Findings:** F11, F12

**Dependências:** F12 depende de Épico D (FFT Linear).

**Estimativa:** < 1 sprint.

---

## Prioridade Recomendada de Execução

| Ordem | Épico                           | Justificativa                                                 |
| ----- | ------------------------------- | ------------------------------------------------------------- |
| 1     | **E** (Prewarm/LoadOptions)     | Requisito de paridade de API, baixo risco, alto impacto em UX |
| 2     | **I** (LSTM Prewarm SR)         | 🔴 Bug de paridade sonora — LSTM não estabiliza após Reset    |
| 3     | **D** (FFT Linear)              | Feature nova de alta performance, necessária para IRs longas  |
| 4     | **J** (Container Reset/Staging) | Paridade de comportamento em Container/Slimmable              |
| 5     | **F** (Breakpoints)             | Necessário para plugin CLAP, implementação simples            |
| 6     | **H** (Fixture A2)              | Melhoria de cobertura de testes                               |
| 7     | **K** (Metadados/Parser Linear) | Completude de API, baixa prioridade                           |
| 8     | **G** (Kernels GEMV)            | Investigação benchmark-driven, ganho incerto                  |
