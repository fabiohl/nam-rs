<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Relatório de Achados do nam-rs — Problemas e Riscos de Produto 🐛

> **Propósito**: mapeia **problemas, estranhezas e riscos de produto** que os testes e a
> auditoria de código expõem. Para **lacunas de funcionalidade** consulte **`TODO-features.md`**;
> para **otimização estrutural** consulte **`TODO-optimize.md`**.
>
> **Última auditoria**: jun/2026 (revisor-auditor + planejador-arquiteto).

---

## Sumário Ativo

| ID      | Achado                                                                                   | Severidade     | Eixo                 |
| ------- | ---------------------------------------------------------------------------------------- |:--------------:| -------------------- |
| **P5**  | Pico de latência WaveNet em silêncio (~677 µs) — possível penalidade de denormais        | 🟡 Média-Baixa | RT/Performance       |
| **P6**  | Telemetria de latência quantizada em potências de 2 (leitura imprecisa)                  | 🟢 Baixa       | Observabilidade      |
| **P8**  | 137 testes `ignored` na suíte padrão; cross-validação live vs C++ só na longa            | 🟢 Informativo | Cobertura            |
| **P15** | `expect()` no hot-path construtivo da A2 (`MirroredBuffer` no `new/set_max_buffer_size`) | 🟡 Média-Baixa | RT-safety            |
| **P16** | A2 fallback escalar (`model/mod.rs:393-444`) inalcançável mas compilado — código morto   | 🟢 Baixa       | Manutenibilidade     |
| **P17** | `SeqCst` no shutdown flag do standalone (`main.rs:76,79`)                                | 🟢 Baixa       | Performance/Ordering |
| **P18** | Cabsim `ConvEngine` usa `Vec` em vez de `AlignedVec` (64-byte) para buffers DSP          | 🟡 Média-Baixa | Performance/SIMD     |

---

## P5 — 🟡 Pico de latência por bloco no WaveNet em silêncio (~677 µs)

### **Evidências**

- `test_denormal_stability_silence`: `max_block_time=677μs ... exceeds 500μs`.
- Benchmark: `Long_Run_WaveNet/Standard_CH16_4096samp` ≈ 5,94 ms (amortizado confortável).

**O que significa**
O tempo amortizado é confortável, mas **um bloco individual** atingiu 677 µs em silêncio —
sintoma de **penalidade de denormais**. Com buffers RT pequenos e carga de CPU alta, picos
assim podem causar **xruns**.

### **Conexões**

- P4 (resolvido): resíduo de silêncio ≈ −89 dBFS é fiel ao C++ (bias propagado), não zerar.
- O2 (pendente): unificação do clock TSC melhoraria a medição precisa destes picos.

### **Diretrizes de resolução**

1. Verificar DAZ/FTZ ativo em **todo** o hot-path WaveNet (há reasserção em `ops.rs:163` e
   `processor/mod.rs:268` — confirmar que `standalone/rt_setup` também asserteia).
2. Implementar um gate de latência **percentil** (P99 sob carga controlada) em vez do
   _wall-clock_ ingênuo removido — mais robusto e menos flaky.
3. Investigar se o pico ocorre no primeiro bloco após silêncio (warm-up de pipeline) ou
   persiste (denormais genuínos).

---

## P6 — 🟢 Telemetria de latência quantizada em potências de 2

**Evidência**: `Telemetry max latency: 65536 ns` e `131072 ns` (= 2¹⁶ e 2¹⁷) — fronteiras
de bucket, não medidas reais.

### **Diretrizes**

- Avaliar buckets mais finos ou registrar `max` exato em paralelo ao histograma.
- Conecta-se a **O2** (clock TSC unificado): ao internalizar `minstant`, o histograma pode
  usar resolução TSC nativa (ns reais) em vez de buckets log2.

---

## P8 — 🟢 Cobertura: 137 testes `ignored` + C++ live só na suíte longa

**Contexto**: A suíte padrão valida contra **goldens congelados** (commit pinado `9c7b185`
do NAMCore v0.5.3). A cross-validação live vs C++ só ocorre na `tests-long.sh`.

### **Diretrizes:**

- Manter cadência regular da suíte longa.
- Verificação (jun/2026): goldens estão atrelados ao commit pinado `9c7b185` ✅.

---

## P15 — 🟡 [NOVO] `expect()` no construtivo da A2 (RT-safety latente)

**Origem**: auditoria complementar jun/2026.

**Evidência**: `src/models/a2/model/mod.rs:150,213` — `MirroredBuffer::new(...)
.expect("MirroredBuffer allocation for A2 layer ring failed")` é chamado tanto no
`new()` (cold, aceitável) quanto no `set_max_buffer_size()` (chamado por
`events.rs:173` via host CLAP, ainda cold mas potencialmente em contexto sensível).

**Risco**: se a alocação falhar (memória esgotada), o `expect` causa `panic` → stack
unwinding → alocação → violação RT. Embora `set_max_buffer_size` rode na ativação
(não no process), o `panic` é contra a filosofia "zero-panic" do projeto.

### **Diretrizes::**

- Substituir `expect` por `match` com log de erro e retorno `Result<(), anyhow::Error>`.
- Propagar o erro para que o host CLAP possa negar a ativação.

---

## P16 — 🟢 [NOVO] Fallback escalar da A2 inalcançável — código morto

**Origem**: auditoria complementar jun/2026.

**Evidência**: `src/models/a2/model/mod.rs:393-444` — o bloco de fallback escalar
(`// ── Scalar fallback: per-frame, per-layer ──`) no `process()` só é alcançado
quando `layer.ch3_conv` **e** `layer.ch8_conv` são ambos `None`. Mas `set_weights`
sempre popula um dos dois (CH=3 → `ch3_conv`, CH=8 → `ch8_conv`), e o match no
dispatcher garante CH∈{3,8}. Logo este fallback é **inalcançável** em produção.

### **Diretrizes-**

- Substituir por `unreachable!("A2 layers always have ch3 or ch8 conv")` (ou
  `debug_unreachable` para release) para documentar a invariante.
- Alternativa: reter como oracle escalar para testes de paridade unitária (mover para `#[cfg(test)]`).

---

## P17 — 🟢 [NOVO] `SeqCst` no shutdown flag (`main.rs:76,79`)

**Origem**: auditoria complementar jun/2026.

**Evidência**: `spsc::SHUTDOWN.load/store(Ordering::SeqCst)` no loop principal
do standalone. O flag é uma boolean simples lida/escrita por duas threads.

### **Diretrizes+**

- Substituir por `Acquire`/`Release` (load com `Acquire`, store com `Release`).
- `SeqCst` não é errado aqui (é cold-path), mas viola a convenção do projeto
  ("No `SeqCst` on the hot-path") e é mais caro que o necessário.

---

## P18 — 🟡 [NOVO] Cabsim `ConvEngine` usa `Vec` padrão em vez de `AlignedVec`

**Origem**: auditoria complementar jun/2026.

**Evidência**: `src/dsp/cabsim/conv.rs:48-66` — todos os buffers internos (`h_fdl`,
`fdl`, `input_buf`, `fft_buf`, `acc`, `fft_scratch`, `ifft_scratch`) são `Vec<f32>`
ou `Vec<Complex<f32>>` padrão (alinhamento 4/8 bytes), **não** `AlignedVec<T>`
(64-byte). Quando O3a vetorizar o MAC com AVX2/AVX-512, loads/stores desalinhados
causarão penalidade ou crash com instruções alinhadas.

**Conexões**: pré-requisito para **O3a** (MAC SIMD do cabsim).

### **Diretrizes=**

- Migrar `h_fdl`, `fdl` e `acc` para `AlignedVec<f32>` (64-byte) quando O3a for
  implementado. Alternativamente, fazer a migração como prep task independente (baixo risco).
- Os buffers `fft_buf`/`fft_scratch` podem permanecer `Vec<Complex>` enquanto usarem `rustfft`
  (que tem seu próprio planner e scratch) — migrar junto com O3b.

---

## Histórico — Achados Resolvidos (verificados no código, jun/2026)

Expandir itens resolvidos (P1–P4, P9–P14)

### P1 — ✅ WaveNet Lite (CH=12/6) + Official (CH=3) — RESOLVIDO (T1.2+T1.3+T1.4)

Root cause: wrap do ring-buffer `MirroredBuffer` dessincronizava quando `channels ∤ page_size`.
Fix: `MirroredBuffer::new_aligned` (verificado em `src/dsp/mirror_buf/alloc.rs:60`).
Validação: Lite + Official bit-exatos contra C++.

### P2 — ✅ Fidelidade WaveNet vs C++ — RESOLVIDO (T-HF6.6)

Eliminação completa do modo lo-fi. ESR colapsou de ~6e-3 para ~1e-13.

### P3 — ✅ Gates de golden frouxos — RESOLVIDO (T-HF6.6)

Thresholds recalibrados SNR 85-105 dB com base em medição real pós-nuke f32.

### P4 — ✅ WaveNet silêncio não-nulo — RESOLVIDO (S1.T1.3)

Diagnóstico concluído: resíduo 3,58e-5 origina-se de `tanh(conv1d_bias)` — **fiel ao NAMCore C++**.

### P9 — ✅ A2 UB com blocos >64 — RESOLVIDO (T2.1+T2.2+T2.3)

Re-chunk interno implementado em `A2::process` (verificado em `model/mod.rs:276-277`:
`let nf = (nf_total - pos).min(WAVENET_MAX_NUM_FRAMES)`).

### P10 — ✅ Modo lo-fi sob júdice — RESOLVIDO (T-HF6.6)

Lo-fi eliminado do WaveNet A1. Feature flag `high-fidelity` removida do `Cargo.toml` (verificado).

### P11 — ✅ Bug carry encoder f16 — RESOLVIDO

Correção em `half.rs:117`: `sign | (e + 0x0400)`. Teste exaustivo 65.536 + regressão.

### P12 — ✅ Corrida CMake em cpp_parity — RESOLVIDO

Serialização por `Mutex` em `ensure_render_compiled`.

### P13 — ✅ Golden v2 wavenet_official — RESOLVIDO (T1.2+T1.4)

Mesma raiz P1. v2 golden reabilitado SNR 121 dB.

### P14 — ✅ Inconsistência tripla gerador goldens v2 — RESOLVIDO

Fontes da verdade alinhadas. ~33 MB removidos.

</details>

---

## Nota de Método

- Itens DONE foram **verificados contra o código-fonte** (grep + leitura direta dos
  arquivos `.rs` e `Cargo.toml`) antes de serem arquivados.
- Novos achados (P15–P18) são de **baixa a média severidade** — nenhum é bloqueante para
  produção, mas melhoram robustez, manutenibilidade e preparam o terreno para otimizações.
- P5 e P6 conectam-se a **O2** (`TODO-optimize.md`).
- P18 é pré-requisito para **O3a** (`TODO-optimize.md`).
