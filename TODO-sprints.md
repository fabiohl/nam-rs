# Pesquisa e Inovação — Backlog de Otimizações DSP

Propostas de otimizações pesquisadas para o hot-path do NAM-rs, organizadas por domínio e ordenadas por prioridade dentro de cada seção. Cada proposta contém análise detalhada de viabilidade, plano de execução e critérios de validação.

---

## Sprint I — Otimizações de Compute (Hot-Path Neural)

---

### Tarefa I.1. WaveNet Dinâmico: Processamento em Bloco (Batch GEMM)

**Prioridade: P0** · Esforço: 🟢 Baixo · Impacto: 🔴 Alto (2–4× speedup) · Risco: 🟢 Baixo
**Minhas Notas:** Assegurar o backport para o estático dos mesmos ganhos que por ventura possam ser obtidos aqui.
**Nota de Conclusão:** O processamento em bloco foi implementado no WaveNet Dinâmico (`wavenet_dyn.rs`). O caminho estático (`wavenet.rs`) já possuía essa otimização implementada (processamento em blocos iterativos limitados a `WAVENET_MAX_NUM_FRAMES`), portanto, nenhum backport foi necessário. Testes e lints rodaram com sucesso sem regressão matemática.

#### 1.1 Diagnóstico

O `WaveNetDynModel::process()` em `wavenet_dyn.rs:553–577` itera sample-by-sample:

```rust
for i in 0..num_frames {
    let condition = [input[i]];
    self.array1.process(&layer_inputs, &condition, math); // num_frames=1
    self.array2.process(array1_outputs, &condition, math); // num_frames=1
}
```

Isso é herança direta do porte C++ e contrasta com o WaveNet estático, que já processa blocos de até `WAVENET_MAX_NUM_FRAMES=64` frames via `process_block_internal<M>`. O overhead do sample-by-sample é severo:

- **20 layers × 20 convoluções** por sample, cada uma com overhead de chamada, indireção de v-table (`math.dot_product`) e evicção de cache L1 ao alternar entre layers.
- Os pesos de cada layer são relidos da cache para cada sample individual, destruindo a reutilização temporal.

Os benchmarks do path estático demonstram que o processamento em bloco (64 frames) é **2–4× mais rápido** que sample-by-sample, pois amortiza o overhead de controle e maximiza a reutilização dos pesos na cache.

#### 1.2 Solução

Replicar a estratégia de **Batch GEMM temporal** do WaveNet estático para o path dinâmico:

1. `WaveNetDynModel::process()` chunka o input em blocos de até `WAVENET_MAX_NUM_FRAMES`:

   ```rust
   let mut offset = 0;
   while offset < num_frames {
       let block_len = (num_frames - offset).min(WAVENET_MAX_NUM_FRAMES);
       self.process_block(&input[offset..offset+block_len],
                          &mut output[offset..offset+block_len]);
       offset += block_len;
   }
   ```

2. `WaveNetLayerArrayDyn::process()` passa `num_frames` real (não hardcoded 1) para `process_block_internal()`.

3. `Conv1dDyn::process_block()` **já aceita** `num_frames` — o gargalo está exclusivamente em `WaveNetLayerArrayDyn::process()` que hardcoda `num_frames=1`.

#### 1.3 Mudanças Necessárias

| Arquivo          | Mudança                                                                                                           |
| ---------------- | ----------------------------------------------------------------------------------------------------------------- |
| `wavenet_dyn.rs` | Redimensionar `array_outputs`, `head_accum`, `head_outputs` de `ch` → `ch × WAVENET_MAX_NUM_FRAMES` no construtor |
| `wavenet_dyn.rs` | `WaveNetDynModel::process()` → loop em blocos, chamando `process_block()`                                         |
| `wavenet_dyn.rs` | `WaveNetLayerArrayDyn::process()` → receber e propagar `num_frames`                                               |
| `wavenet_dyn.rs` | Gated activation loop → processar `2*ch*num_frames` no split `tanh⊙sigmoid`                                       |
| `wavenet_dyn.rs` | `advance_frames(num_frames, ch)` no loop das layers                                                               |

#### 1.4 Validação

- Testes de golden vectors existentes (`test_golden_vectors_wavenet`) devem passar sem alteração de MSE/SNR.
- Benchmark `criterion` comparando path dinâmico antes/depois com modelo BossWN-standard.
- Teste de paridade: saída sample-by-sample vs bloco deve ser **bitwise idêntica** (ambos usam os mesmos `dot_product` e `tanh`).

---

### Tarefa I.2. Tanh Polinômio de Grau 7: Precisão sem Custo

**Prioridade: P1** · Esforço: 🟢 Baixo · Impacto: 🟡 Médio (100× menos erro) · Risco: 🟢 Baixo
**Minhas Notas:** Medir efetivamente o "antes e o depois" para assegurar impacto baixíssimo em performance.
**Nota de Conclusão:** O polinômio de grau 7 foi derivado e aplicado com sucesso nas implementações (AVX2 e AVX-512) usando Diferencial Evolution Minimax. O erro máximo caiu de ~5e-3 para ~1.2e-5. Os testes unitários e de golden vector foram ajustados para o novo threshold de 1e-4 e todos rodaram com sucesso, sem degradação do throughput. Lints conferidos.

#### 2.1 Diagnóstico

A `simd_tanh` em `fastmath.rs:53–97` utiliza um polinômio Padé de grau 5 com dois coeficientes (`c0=0.165326984`, `c1=0.009702402`). O erro máximo por ativação é ~5e-3 vs `f32::tanh()`. Para o modelo BossWN-standard (20 camadas), o erro acumulado atinge ~2.2e-2 (MSE golden medido: 3.21e-2). O threshold de golden vectors precisa ser `5e-2`, o que é amplo demais para garantir paridade rigorosa com o C++.

O pipeline atual por ativação:

```text
x_sq = x * x                           // 1 mul
y_3_5 = fmadd(c1, x_sq, c0)            // 1 fma
y_1_3_5 = fmadd(y_3_5, x_sq, 1.0)      // 1 fma
p_x = x * y_1_3_5                      // 1 mul
radicand = fmadd(p_x, p_x, 1.0)        // 1 fma
rr = rsqrt(radicand)                    // 1 rsqrt (~3 ciclos)
// Newton-Raphson: 4 instruções
result = p_x * rr                      // 1 mul
// Total: ~12 instruções, ~6–8 ciclos (throughput-bound)
```

#### 2.2 Solução: Polinômio de Grau 7

Adicionar um terceiro coeficiente `c2` ao polinômio, elevando para grau 7:

```rust
let x_sq = _mm256_mul_ps(x, x);
let x_sq_sq = _mm256_mul_ps(x_sq, x_sq);        // +1 mul (NOVO)
let y_3_5 = _mm256_fmadd_ps(c1, x_sq, c0);
let y_3_5_7 = _mm256_fmadd_ps(c2, x_sq_sq, y_3_5); // +1 fma (NOVO)
let y_full = _mm256_fmadd_ps(y_3_5_7, x_sq, one);
let p_x = _mm256_mul_ps(x, y_full);
// ... rsqrt + Newton-Raphson inalterados
```

**Custo incremental:** +2 instruções (`mul` + `fma`) num pipeline de ~12. Em CPUs com 2+ portas FMA (todas x86-64-v3), essas instruções são emitidas em paralelo com as existentes — **0% impacto mensurável no throughput**.

**Ganho de precisão:** Erro por ativação cai de ~5e-3 → ~5e-5 (duas ordens de magnitude). Erro acumulado em 20 camadas: ~2.2e-4 vs ~2.2e-2. Permite apertar golden vectors para `MSE < 1e-3`.

#### 2.3 Mudanças Necessárias

| Arquivo       | Mudança                                                                                |
| ------------- | -------------------------------------------------------------------------------------- |
| `fastmath.rs` | `simd_tanh()` — adicionar `c2`, `x_sq_sq`, `y_3_5_7`                                   |
| `fastmath.rs` | `simd_tanh_avx512()` — mesma alteração com intrinsics ZMM                              |
| `fastmath.rs` | `simd_sigmoid()` e `simd_sigmoid_avx512()` — derivam de `tanh`, herdam automaticamente |
| `fastmath.rs` | Docstring — atualizar erro máximo documentado                                          |

#### 2.4 Derivação dos Coeficientes

Os coeficientes `c0`, `c1`, `c2` devem ser re-derivados via **otimização Minimax (algoritmo de Remez)** no intervalo `[-8, 8]`:

1. Usar ferramenta como `sollya` ou script Python com `scipy.optimize.minimize`:

   ```python
   # Minimizar max|tanh(x) - approx(x)| para x ∈ [-8, 8]
   # approx(x) = x * (1 + (c0 + c1*x² + c2*x⁴) * x²) * rsqrt(p²+1)
   ```

2. Validar MSE contra `f64::tanh()` em 10M pontos aleatórios no intervalo.

3. Confirmar que o clamp `[-1, 1]` da saturação permanece correto.

#### 2.5 Validação

- `test_simd_fastmath_tanh_mse` — MSE deve cair de ~5e-3 para ~5e-5.
- `test_golden_vectors_wavenet` — MSE deve cair; threshold pode ser apertado para `1e-3`.
- Benchmark `criterion` — latência de `tanh_slice` deve ser **idêntica** (±1%) ao grau 5.

---

### Tarefa I.3. Fusão de Operadores no WaveNet Layer

**Prioridade: P2** · Esforço: 🟡 Médio · Impacto: 🟡 Médio (~40% menos loads) · Risco: 🟡 Médio
**Minhas Notas:** Medir efetivamente o "antes e o depois" para assegurar impacto baixíssimo em performance.

#### 3.1 Diagnóstico

`WaveNetLayer::process_block_internal<M>` em `wavenet.rs:241–280` executa 6 passes de memória sequenciais por layer:

```text
Passo  Operação                        Acesso a Memória
─────  ──────────────────────────────  ──────────────────────
  1    conv1d.process_block()          Escreve block[0..CH×N]
  2    input_mixin.process_acc_block() Lê+escreve block[0..CH×N]
  3    M::tanh_slice()                 Lê+escreve block[0..CH×N]
  4    Sum block → head_input          Lê block, escreve head_input
  5    one_by_one.process_block()      Lê block, escreve output
  6    Residual addition               Lê layer_buffer, escreve output
```

Para `num_frames=64` e `CH=16`: cada passo percorre `64 × 16 × 4 = 4096 bytes`. São **6 × 4096 = 24 KB** de tráfego L1 por layer, e 20 layers totalizam **480 KB** de tráfego — estourando a L1 (32 KB típica).

#### 3.2 Solução: Kernel Fusionado por Frame

Fusionar os passos 1–6 numa **única passada por frame**:

```rust
for i in 0..num_frames {
    // temp[0..CH] vive em registradores YMM (64 bytes para CH=16)

    // 1. Conv1d para frame i → temp
    conv1d_single_frame(layer_buffer, &mut temp, buffer_start + i);
    // 2. Input mixin acumula em temp
    input_mixin_single_frame(condition_i, &mut temp);
    // 3. tanh(temp) in-place — operação sobre 2 registradores YMM
    tanh_register(&mut temp);
    // 4. Acumula temp em head_input[i]
    head_input[i*CH..(i+1)*CH] += temp;
    // 5. one_by_one(temp) → output[i]
    one_by_one_single_frame(&temp, &mut output[i*CH..]);
    // 6. output[i] += residual
    output[i*CH..(i+1)*CH] += layer_buffer[...];
}
```

**Impacto:** Os vetores `temp[0..CH]` (16 floats = 64 bytes = 2 registradores YMM) vivem **inteiramente em registradores** para os passos 3–6. Tráfego L1 reduzido de ~6 passes a ~2 passes (conv1d + one_by_one são os únicos que acessam matrizes de pesos).

#### 3.3 Desafio Estrutural

O `Conv1d::process_block` atual itera `for out_c in 0..OUT` externamente e `for i in 0..num_frames` internamente. Para fusionar, é necessário **inverter a ordem dos loops**: frame-first, channel-second. Isso requer uma nova função `conv1d_fused_single_frame<M>()` que processa todos os canais de saída de um frame numa única chamada.

Esta proposta complementa a Proposta 4 (FMA Multi-Channel): se combinadas, os ganhos se multiplicam.

#### 3.4 Validação

- Golden vectors devem permanecer **bitwise idênticos** (mesmas operações, mesma ordem aritmética).
- Benchmark `criterion` comparando `process_block_internal` antes/depois.
- Perfil `perf stat` medindo L1-dcache-load-misses antes/depois.

---

### Tarefa I.4. FMA Multi-Channel: Dot Product 4× para Conv1d

**Prioridade: P2** · Esforço: 🟡 Médio · Impacto: 🟡 Médio (~75% menos loads) · Risco: 🟡 Médio
**Minhas Notas:** Medir efetivamente o "antes e o depois" para assegurar impacto baixíssimo em performance.

#### 4.1 Diagnóstico

Em `Conv1d::process_block<M>` (`wavenet.rs:51–109`), o laço externo itera pelos canais de saída individualmente:

```rust
for out_c in 0..OUT {       // OUT=16 para Standard
    for k in 0..K {
        for i in 0..num_frames {
            let in_slice = &layer_buffer[...];   // relido OUT vezes!
            block[i*OUT + out_c] = M::dot_product(in_slice, weight_slice);
        }
    }
}
```

Para CH=16 (Standard), cada frame carrega o **mesmo** `in_slice` (16 floats = 64 bytes) 16 vezes da L1 para registradores. São **16 loads redundantes** onde 1 seria suficiente.

O `dot_product_4x_avx2` já existe em `simd.rs` para o LSTM (processa 4 dot products simultaneamente com o mesmo vetor de entrada), mas **não é exposto** via a trait `SimdMath` nem utilizado pelo WaveNet.

#### 4.2 Solução

1. **Expor `dot_product_4x`** na trait `SimdMath` e no `SimdMathConfig`:

   ```rust
   pub trait SimdMath {
       // ... existente
       unsafe fn dot_product_4x(
           input: &[f32],
           w0: &[f32], w1: &[f32], w2: &[f32], w3: &[f32],
       ) -> (f32, f32, f32, f32);
   }
   ```

2. **Criar `Conv1d::process_block_4x<M>()`** que itera `out_c` em passos de 4:

   ```rust
   for out_c_base in (0..OUT).step_by(4) {
       for k in 0..K {
           for i in 0..num_frames {
               let in_slice = &layer_buffer[...]; // carregado 1 vez para 4 canais
               let (r0,r1,r2,r3) = M::dot_product_4x(in_slice, w0, w1, w2, w3);
               block[i*OUT + out_c_base + 0] = r0;
               // ... r1, r2, r3
           }
       }
   }
   // Tail escalar para OUT % 4 != 0
   ```

3. **Tail handling:** Para topologias onde `OUT` não é múltiplo de 4 (ex: CH=8 no Lite), o tail usa `dot_product` individual para os canais restantes.

#### 4.3 Análise Quantitativa

| Topologia | CH (OUT) | Loads `in_slice` (atual) | Loads `in_slice` (4×) | Redução |
| --------- | -------- | ------------------------ | --------------------- | ------- |
| Standard  | 16       | 16 × N                   | 4 × N                 | 75%     |
| Lite      | 12       | 12 × N                   | 3 × N                 | 75%     |
| Feather   | 8        | 8 × N                    | 2 × N                 | 75%     |
| Nano      | 4        | 4 × N                    | 1 × N                 | 75%     |

#### 4.4 Validação

- Saída deve ser **bitwise idêntica** ao `process_block` original (mesma aritmética FMA).
- Benchmark isolado: `dot_product` × 16 vs `dot_product_4x` × 4 com vetores de 16 floats.
- Golden vectors inalterados.

---
---

## Sprint II — Hardening de Tempo Real (Sistema)

---

### Tarefa II.5. Desabilitação de THP e Advisory de IRQ Isolation

**Prioridade: P2** · Esforço: 🟢 Baixo · Impacto: 🟡 Médio (jitter) · Risco: 🟢 Baixo

#### 5.1 Diagnóstico

O NAM-rs já configura `SCHED_FIFO`, Core Affinity, PM QoS e `mlockall`. Dois vetores de jitter do kernel permanecem descobertos:

1. **Transparent Huge Pages (THP):** O thread `khugepaged` do kernel executa compactação e promoção de páginas em background. Quando toca memória na região do processo NAM-rs, pode causar latency spikes de **50–500 µs**. O processo não desabilita THP.

2. **IRQ Affinity:** Interrupções de hardware (NIC, USB, timer) podem ser despachadas para o core afixado à thread DSP, causando preempção involuntária de ~10–50 µs por IRQ.

#### 5.2 Solução

**5.2a — Desabilitar THP via `prctl`:**

Adicionar em `main.rs`, antes de `mlockall`:

```rust
unsafe { libc::prctl(libc::PR_SET_THP_DISABLE, 1, 0, 0, 0); }
```

Isso desabilita THP **apenas para este processo** — sem afetar o restante do sistema.

**5.2b — Advisory de IRQ no diagnóstico de startup:**

Adicionar ao `SystemSnapshot` uma detecção do core afixado e emitir sugestão:

```text
💡 Para latência mínima, isole o core 3 das IRQs do sistema:
   echo 0xFFFFFFF7 > /proc/irq/default_smp_affinity
```

Não é possível automatizar isso sem `root`; a sugestão é informativa.

#### 5.3 Mudanças Necessárias

| Arquivo          | Mudança                                                           |
| ---------------- | ----------------------------------------------------------------- |
| `main.rs`        | Adicionar `prctl(PR_SET_THP_DISABLE, 1, ...)` antes de `mlockall` |
| `diagnostics.rs` | `SystemSnapshot` — detectar core afixado, emitir advisory de IRQ  |
| `Cargo.toml`     | Nenhuma — `libc` já é dependência                                 |

#### 5.4 Validação

- Verificar via `/proc/self/status` que `THP_enabled: 0` após o `prctl`.
- Medir jitter com `cyclictest` antes/depois em host com THP habilitado.
- Testes existentes não são afetados (THP e IRQ não alteram a lógica DSP).

---
---

## Sprint III — Pesquisa de Longo Prazo

---

### Tarefa III.6. Filtros de Fase Mínima (Minimum Phase) no Resampler

**Prioridade: P4** · Esforço: 🔴 Alto · Impacto: 🟡 Médio (~1 ms latência) · Risco: 🔴 Alto

#### 6.1 Diagnóstico

O `NamResampler` em `resampler.rs` usa `rubato::SincFixedIn` com filtro FIR de **Fase Linear** (Kaiser-BlackmanHarris2, `sinc_len=256`, interpolação Hermite cúbica). Filtros de fase linear possuem group delay simétrico de `sinc_len/2 = 128` taps. Para o roundtrip 96k → 48k → 96k, isso adiciona **~1.5 ms** de latência pura na matemática do resampler, além de *pre-ringing* audível.

Amplificadores valvulados reais são sistemas de fase mínima por natureza — o pre-ringing dos FIR lineares é um artefato estranho à experiência de tocar guitarra.

#### 6.2 Solução Proposta

Migrar para coeficientes FIR de **Fase Mínima** (Minimum Phase Sinc). Duas abordagens:

**Abordagem A — Tabela estática de coeficientes:**

1. Extrair os coeficientes Kaiser-Blackman do `rubato` (ou gerá-los independentemente).
2. Aplicar a **Transformada de Hilbert** para converter para fase mínima.
3. Injetar como tabela `const` no código, usada por um resampler FIR customizado.

**Abordagem B — Resampler FIR customizado:**

1. Implementar resampler polifásico FIR sem dependência de `rubato`.
2. Usar coeficientes MPS (Minimum Phase Sinc) pré-computados.
3. Controle total da API e dos buffers internos.

#### 6.3 Riscos Críticos

- **Resposta audível muda:** A fase mínima redistribui a energia temporal do filtro. Isso altera a assinatura sonora do resampler e requer validação perceptual com músicos.
- **Invalidação de golden vectors:** Todos os testes de roundtrip do resampler e golden vectors que passam pelo resampling path precisariam ser re-gerados.
- **API do `rubato`:** Não expõe coeficientes de fase mínima. Qualquer solução requer implementação custom ou fork.
- **SNR:** Filtros de fase mínima podem ter rejeição de aliasing ligeiramente inferior em stopband. Requer validação com impulso de Dirac e FFT.

#### 6.4 Decisão Atual

**Não implementar agora.** O ganho de ~1 ms é relevante para `pw_rate ≠ 48000` (cenário não-padrão), mas o risco de regressão audível e o esforço de implementação/validação são desproporcionais. Reavaliar quando houver demanda de usuários com setups a 96 kHz.

---
---

## Matriz de Prioridades Consolidada

| Prioridade | #   | Proposta                      | Esforço | Impacto          | Domínio  |
|:----------:|:---:| ----------------------------- |:-------:|:----------------:| -------- |
| **P0**     | 1   | WaveNet Dinâmico Batch GEMM   | 🟢      | 🔴 2–4×          | Compute  |
| **P1**     | 2   | Tanh Grau 7 (FastMath)        | 🟢      | 🟡 100× precisão | Compute  |
| **P2**     | 3   | Operator Fusion (Layer)       | 🟡      | 🟡 40% loads     | Compute  |
| **P2**     | 4   | FMA Multi-Channel (Conv1d 4×) | 🟡      | 🟡 75% loads     | Compute  |
| **P2**     | 5   | THP Disable + IRQ Advisory    | 🟢      | 🟡 jitter        | Sistema  |
| **P4**     | 6   | Filtros de Fase Mínima        | 🔴      | 🟡 ~1 ms         | Pesquisa |

### Definição de Prioridades

- **P0 — Imediato:** Alto impacto, baixo risco. Implementar na próxima sprint.
- **P1 — Curto prazo:** Ganho claro com esforço mínimo. Sprint seguinte.
- **P2 — Médio prazo:** Requer benchmarks antes/depois para validar ganho real.
- **P4 — Pesquisa futura:** Alto risco ou dependência externa. Reavaliar sob demanda.

### Dependências entre Propostas

```text
1 (Batch GEMM) ──→ independente
2 (Tanh Grau 7) ──→ independente
3 (Operator Fusion) ←──→ 4 (FMA 4×)  [complementares, ganhos multiplicativos]
5 (THP/IRQ) ──→ independente
6 (Fase Mínima) ──→ independente
```

As propostas 3 e 4 são **complementares**: a fusão de operadores (3) requer inverter o loop para frame-first, e o FMA 4× (4) otimiza o kernel interno da convolução. Implementar ambas em conjunto maximiza o ganho e evita refatoração dupla do `Conv1d::process_block`.
