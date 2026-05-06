<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva. -->
# TODO-sprints.md — Backlog Técnico NAM-rs

## Épico Alpha — Inovação Microarquitetural e Ultra-Baixa Latência (Pesquisa & P&D)

**Impacto:** Extrair a máxima performance da CPU através de otimizações não-óbvias e agressivas de pipeline de hardware, visando latências sub-milissegundo para os modelos WaveNet e LSTM.

### `TA1` — Otimização do Hot-path: Temporal Tiling e Fusão de Kernel (WaveNet) [DONE]

**Status:** Parcialmente implementado. A lógica de Temporal Tiling foi validada matematicamente (sem quebra de regressão), porém o `cargo bench` apontou **regressão de 25% na performance**.
**Motivo Identificado:** A barreira de abstração do trait `SimdMath::dot_product_4x_interleaved` exige o passe do slice de pesos, forçando o intrínseco (ex: em `avx2.rs`) a ler do cache L1 a cada frame independentemente. O reuso de registradores YMM0 (para evitar L1 hit) não ocorreu, pois a trait não suporta múltiplos frames simultâneos.

### `TA1.5` — Extensão do Trait SimdMath para Multi-Frame Tiling [DONE]

**Pesquisa e Implementação:**

- **Problema:** Para que a `TA1` efetivamente reduza as leituras L1, a carga dos pesos para os registradores YMM deve ocorrer *antes* do loop sobre os múltiplos frames do Tiling.
- **Solução:** Criar e implementar uma nova interface no trait `SimdMath` (ex: `dot_product_4x_interleaved_dual_frame`) que receba os estados de 2 frames (f0 e f1) e execute as multiplicações de ambos utilizando o mesmo carregamento `_mm256_loadu_si128` dos pesos.
- **Como implementar:**
  1. Modificar `src/math/simd/traits.rs` adicionando `dot_product_4x_interleaved_dual`.
  2. Implementar em `avx2.rs`, mantendo os pesos em `vw` e acumulando `vs_0` e `vs_1` em registradores separados.
  3. Aplicar o fallback escalar equivalente em `fallback.rs`.
  4. Retomar e re-aplicar o código de Temporal Tiling na `Conv1D` e `WaveNetLayer` usando a nova instrução.

### `TA2` — Unrolling Agressivo e Hiding de Latência em FastMath [DONE]

**Status:** Implementado (loop unrolling 16-floats em `simd_tanh_dual_avx2` e `simd_sigmoid_dual_avx2` para melhor Instruction Level Parallelism).
**Pesquisa e Implementação:**

- **Problema:** Funções como `simd_tanh` em `fastmath.rs` dependem de intrínsecas como `_mm256_rsqrt_ps` (que possuem latência de 5~7 ciclos) e de aproximações de Newton-Raphson com longas cadeias de dependência de dados (dependency chains). O pipeline da CPU fica ocioso aguardando os resultados dessas operações.
- **Solução (Instruction Level Parallelism - ILP):** Intercalar instruções independentes para manter o pipeline totalmente alimentado (*latency hiding*).
- **Como implementar:**
  1. Modificar funções como `tanh_slice_avx2` para iterar em blocos de 16 floats (2 registradores YMM por vez) ao invés de 8.
  2. Carregar `va1` e `va2` simultaneamente.
  3. Aplicar as macros do polinômio de forma intercalada: `let x_sq1 = _mm256_mul_ps(va1, va1); let x_sq2 = _mm256_mul_ps(va2, va2);`
  4. Quando `rr1 = _mm256_rsqrt_ps(radicand1)` for despachado para a Execution Unit, imediatamente despachar `rr2 = _mm256_rsqrt_ps(radicand2)`. A CPU processará ambas em paralelo (se tiver múltiplas portas FMA) ou esconderá a latência da primeira com o despacho da segunda.

### `TA3` — Compactação de Status SPSC e Redução de Cache Bouncing [DONE]

**Status:** Implementado (condensação de `AtomicBool` em bitmask `AtomicU64`).
**Pesquisa e Implementação:**

- **Problema:** A struct `RtStatusFlags` (em `spsc.rs`) contém múltiplas variáveis atômicas separadas (`has_clipped: AtomicBool`, `gc_overflow: AtomicBool`, etc). Quando a thread RT atualiza essas flags e a Main Thread as lê em `poll_rt_status()`, ocorre contenção em múltiplas linhas de cache diferentes, exigindo sincronização de barramento de memória (Cache-line Bouncing) que gera jitter.
- **Solução:** Condensar estados não-contadores em um único bitmask atômico.
- **Como implementar:**
  1. Substituir os múltiplos `AtomicBool` por um único `status_bits: AtomicU64`.
  2. Definir constantes para os bits: `FLAG_CLIPPED = 1 << 0`, `FLAG_GC_OVERFLOW = 1 << 1`, etc.
  3. Na thread DSP (escrita rápida): Usar `status_bits.fetch_or(FLAG_CLIPPED, Ordering::Relaxed)`. Sendo um único endereço de memória, a invalidação do cache ocorre em apenas uma cache line (64 bytes).
  4. Na Main Thread (leitura/limpeza): Usar `fetch_and(!mask, Ordering::Relaxed)` para ler e zerar simultaneamente de forma segura.

### `TA4` — Isolamento Térmico e Avanços no `SCHED_FIFO`

**Pesquisa e Implementação:**

- **Problema:** A heurística de `select_optimal_cpu` (em `rt_setup.rs`) escolhe a CPU de maior capacidade e menor quantidade de IRQs. No entanto, ela pode inadvertidamente selecionar um core que o usuário ou o kernel já isolou propositalmente (ex: `taskset`, `cgroups` ou `isolcpus` do GRUB), o que resulta em falha catastrófica ao aplicar a afinidade via `pthread_setaffinity_np`.
- **Solução:** O NAM-rs precisa respeitar os limites de contexto do processo (sua "máscara de afinidade permitida" real) antes de tentar fixar a thread.
- **Como implementar:**
  1. No início de `select_optimal_cpu`, usar `libc::sched_getaffinity(0, ...)` ou ler `/proc/self/status` (campo `Cpus_allowed_list`) para obter o bitmask de CPUs que o sistema operacional autorizou para o processo atual.
  2. Filtrar a lista total de cores disponíveis mantendo apenas aqueles presentes no bitmask autorizado.
  3. Aplicar a heurística de capacidade e contagem de interrupções APENAS nesse subconjunto permitido. Isso garante que nunca selecionaremos um núcleo inacessível, melhorando a resiliência em sistemas complexos de áudio Linux.

### `TA5` — Reestruturação Lógica e Extração de Acessórios (`src/models/wavenet.rs`)

**Pesquisa e Implementação:**

- **Problema:** `wavenet.rs` possui ~1200 linhas englobando matemática central, gerenciamento de estado das camadas e lógicas construtivas de contexto, dificultando o isolamento mental da arquitetura e das futuras otimizações do *TA1*.
- **Solução:** Manter a "inteligência de DSP e loops críticos" no `wavenet.rs` e extrair as "estruturas burocráticas" para novos arquivos ou para o `wavenet_common.rs`.
- **Como implementar:**
  1. Extrair definições acessórias puras, como construtores ou configurações do `WaveNetLayerState`, inicialização de arrays e construtores de buffer temporal para módulos separados ou para o próprio `wavenet_common.rs`.
  2. Mover a struct `WavenetProcessContext` ou parsers secundários para fora do núcleo se não exigirem inlining estrito no laço.
  3. **Restrição Crítica (Atenção):** As implementações do hot-path crítico (os laços internos de `Conv1D::process_single_frame`, `DenseLayer::process_fused_block` e `WaveNetLayer::process_block_internal`) DEVEM permanecer em `wavenet.rs`. O compilador depende fortemente de estarem no mesmo escopo de tradução para resolver as anotações agressivas de `#[inline(always)]` perfeitamente, sem depender estritamente de LTO (Link-Time Optimization).
- **Ação:** Chamar a skill `planejador-arquiteto` e `revisor-auditor` após essa reestruturação para assegurar paridade de perf.
