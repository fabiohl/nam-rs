<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Mapa de Otimização do nam-rs — Internalização e Performance ⚙️

> **Propósito**: mapeia **oportunidades de performance e enxugamento** via internalização de
> dependências e vetorização de lacunas SIMD no hot-path.
>
> Para **problemas de produto** consulte **`TODO-problemas.md`**; para **lacunas de
> funcionalidade** consulte **`TODO-features.md`**.
>
> **Última auditoria**: jun/2026 (revisor-auditor + planejador-arquiteto).
>
> **Filosofia**: internalizar é justificado quando a dependência é genérica demais, arrasta
> cadeia de build desproporcional, ou deixa performance na mesa que só o controle total
> destrava. Quando nenhum se aplica, **manter a dependência** (ex.: `rtrb`).

---

## Sumário Ativo

| ID     | Achado                                            | Ganho esperado                          | Esforço   | Hot-path? | Status         |
| ------ | ------------------------------------------------- | --------------------------------------- |:---------:|:---------:| -------------- |
| **O2** | `minstant` → `RtClock` TSC unificado              | −1 proc-macro, −`syn` v1; 1 clock único | Baixo-Méd | Morno     | 🟢 Fazer       |
| **O3** | `rustfft` → RFFT real interna + MAC complexo SIMD | ~2× FFT + metade memória do cabsim      | Alto      | **Sim**   | ★ PoC faseada  |
| **O4** | `rtrb` → fila SPSC interna                        | Marginal                                | Médio     | Controle  | ⚪ Deferir     |
| **O5** | Cobertura SIMD x86-64-v3 (índice + guard-rail)    | Latência ↓ no hot-path                  | Baixo     | **Sim**   | 🟢 (via F1/F3) |

### Matriz de Priorização (esforço × impacto)

```text
IMPACTO
  ▲
A │                               O3  rustfft → RFFT interna  ★ flagship
L │
T │      O2  minstant
O │
  │        O5  cobertura SIMD (itens em F1/F3)         O4  rtrb  ⚪ (deferir)
B │
A │
I │
X └──────────────────────────────────────────────────────────────────►
       BAIXO              MÉDIO                 ALTO         ESFORÇO

🟢 = fazer já    ★ = PoC profunda faseada    ⚪ = opcional/deferido
```

**Ordem recomendada**: O2 → O5 (lacunas escalares, baixo risco) → O3a (MAC SIMD) →
O3b (RFFT interna) → (O4, opcional).

---

## O2 — 🟢 `minstant`: unificar com o clock TSC interno

**O que é.** `minstant` fornece `Instant`/`Anchor` baseado em TSC. Usado no path CLAP
(`src/clap/processor/mod.rs:32`, `orchestrator.rs:12`, `telemetry.rs:5`) e no standalone
(`src/standalone/rt_setup/telemetry.rs:12`, `src/main.rs:179`, `dsp/pipeline/output_pw.rs:11`).

**Duplicidade verificada (jun/2026).** O standalone já tem calibração RDTSC própria em
`src/standalone/rt_setup/tsc.rs`. Convivem **duas implementações de clock**.

**Custo da dependência (verificado no Cargo.toml).** `minstant` (linha 29) arrasta `ctor`
(proc-macro) + `syn` v1 → **duas major versions** do `syn` no grafo.

### **Diretrizes**

1. Promover `tsc.rs` a `RtClock` / `TscInstant` único (CLAP + standalone).
2. Adicionar probe CPUID de invariant-TSC (`0x80000007` bit 8) + fallback `CLOCK_MONOTONIC`.
3. Teste calibração ±0.1% vs `clock_gettime`; **remover `minstant`** do `Cargo.toml`.

**Conexões**: P5 (pico de latência), P6 (telemetria quantizada).

---

## O3 — ★ `rustfft` (cabsim UPOLS): RFFT real interna + MAC SIMD

**O que é.** `rustfft` fornece FFT complexa mixed-radix. Usado no motor UPOLS do cabsim
(`src/dsp/cabsim/conv.rs:26`), `sinc_kernel.rs:18` e `perceptual.rs:11`.

**Oportunidades verificadas (jun/2026):**

1. **RFFT em vez de FFT complexa cheia** — entrada é real (`Complex::new(sample, 0.0)`,
   `conv.rs:240`); da IFFT só se usa parte real (`conv.rs:303`). ~2× desperdício.
2. **Metade da pegada de memória** — `h_fdl`/`fdl` passam de N para N/2+1 bins.
3. **MAC complexo escalar** — `conv.rs:259-292` é loop escalar puro. Vetorizar com AVX2/FMA.
4. **FFT sempre power-of-two** (`conv.rs:94`) → radix-2/4 especializado viável.

**Pré-requisito**: P18 (alinhar buffers do cabsim com `AlignedVec`).

**Estratégia faseada:**

| Fase | Entrega                                      | Esforço | Risco |
| ---- | -------------------------------------------- | ------- | ----- |
| O3a  | MAC complexo SIMD + layout split `re[]/im[]` | Médio   | Baixo |
| O3b  | RFFT/IRFFT interna power-of-two              | Alto    | Médio |
| O3c  | Kernels AVX-512 dedicados + fusão FFT↔MAC    | Alto    | Médio |

**Obs:** P18 (`TODO-problemas.md`) é pré-requisito para **O3a**.

### **Diretrizes:**

1. Bench de baseline do cabsim (FFT, MAC, IFFT separados) antes de qualquer mudança.
2. O3a primeiro (independente da FFT): converter `h_fdl`/`fdl` para layout split `re[]/im[]`,
   vetorizar MAC com FMA AVX2. ESR do cabsim inalterado.
3. O3b com `rustfft` como golden; só remover `rustfft` quando `sinc_kernel.rs`/`perceptual.rs`
   também migrarem.
4. Validar ESR < 1e-5 vs `rustfft` em IRs reais.

---

## O4 — ⚪ `rtrb`: deferir (baixo retorno)

**O que é.** `rtrb` é fila SPSC lock-free bounded, **zero-dependências**. Encapsulada em
`src/common/spsc/{mod,gc}.rs`. Usada **só para controle** (parâmetros, GC cascade, swap).

**Decisão**: deferir. `rtrb` é minúsculo, zero-dep, bem testado. Risco de errar ordering
lock-free supera o ganho marginal. Reconsiderar somente se perfil apontar false sharing.

---

## O5 — 🟢 Cobertura SIMD x86-64-v3 (índice + guard-rail)

**Contexto.** `.cargo/config.toml` fixa x86-64-v3 → AVX2, FMA e F16C garantidos em compilação.

**Achados verificados (jun/2026):**

| Achado  | Local                     | Status                                 | Roteamento                    |
| ------- | ------------------------- | -------------------------------------- | ----------------------------- |
| S1      | `a2/head.rs:96-118`       | Head conv 100% escalar (128 FMA/frame) | → F3 (motor A2)               |
| S2      | `a2/model/mod.rs:264-270` | ✅ DONE — rechannel AVX2 SIMD (CH=8)   | Verificado :281-294           |
| S3      | `wavenet/model.rs:96-98`  | ✅ DONE — `M::apply_gain` SIMD         | Verificado :98                |
| S4      | `cabsim/conv.rs:259-292`  | MAC complexo escalar                   | → O3a (acima)                 |
| Limpeza | `avx2_impl.rs:42-99`      | ✅ DONE — `unreachable!()` BF16        | Verificado :42,76,192,242,303 |

**Guard-rail ativo**: com x86-64-v3 garantido, **nenhum laço de aritmética f32/f16
por-amostra/por-bloco** deve permanecer escalar. Regra de revisão para qualquer sprint
que toque inferência.

---

## Histórico — Achados Resolvidos

Expandir item resolvido (O1)

### O1 — ✅ `half`: internalizado (verificado jun/2026)

- `src/math/common/half.rs` (313 linhas) contém: `f16_bits_to_f32` (software),
  `f32_to_f16_bits` (RNE com fix de carry P11), variantes F16C de hardware.
- **`half` removido do `Cargo.toml`** (verificado: não presente nas linhas 18-39).
- Todas as 16 caudas escalares dos kernels SIMD usam F16C hardware.
- Testes exaustivos: decode 65.536 + encode 65.536 + regressão carry.

---

## Recomendações de Validação

- **Golden contra dependência original.** Correção provada contra a dependência como
  referência antes de removê-la.
- **RT-safety preservada.** Toda alocação no `new()`/load; hot-path zero-alloc/lock/panic.
- **Bench antes e depois.** Baseline `criterion` por sub-operação.
- **Remoção só após verde total.** Dependência só sai do `Cargo.toml` quando todos os usos
  migraram e os goldens/benches passaram.

---

## Nota de Método (para o planejador-arquiteto)

- Ao planejar sprints: O2 é independente (baixo risco, boa primeira sprint); O3 é faseado
  (O3a → O3b → O3c), cada fase com critério de aceite numérico próprio; O4 fica deferido.
- O5 é transversal: S1 e S4 executados dentro de F3 e O3a respectivamente.
- Cruzar com P5/P6 ao planejar O2, e com P18 ao planejar O3a.
