<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-findings.md — Auditoria de Performance & Low-Latency (x86-64-v3)

Auditoria realizada em 2026-07-17 pelas skills `revisor-auditor` (role **Performance and
Low-Latency Master**) + `pesquisador-inovador`, consolidada pela `planejador-arquiteto`.

**Fontes de evidência:**

- `testes.log` (tests-quick + quality-dashboard + build-release + tests-long, run de 2026-07-17,
  CPU AMD Ryzen 7 5700U / Zen 2, ISA AVX2 x86-64-v3, rustc 1.97.1);
- `target/dsp_hotpath.asm` (perf annotate do binário PGO, 4 kHz cycles:u, gerado pelo
  `utils/build-release.sh`, workload WaveNet Standard CH16 via PipeWire);
- Leitura dirigida do código-fonte (kernels AVX2, pipeline DSP, rt_setup, CLAP, PipeWire host).

**Escopo e restrições (obrigatórios para quem implementar):**

1. **PROIBIDA** qualquer regressão de paridade com o NAMcore ou penalização de fidelidade
   sonora. Ferramentas de defesa: `utils/quality-dashboard.sh` (contrato de ESR/latência) e
   `utils/tests-performance-regression.sh`. Todo finding indica seu "gate de validação".
2. Foco exclusivo no baseline **x86-64-v3 (AVX2+FMA+BMI2)**. Multiversioning de outras ISAs
   (AVX-512 etc.) está fora de escopo nesta rodada.
3. **Não atacar** os temas de `TODO-wavenet_a2_max.md` e `TODO-convnet_parity.md`.
   Nenhum finding abaixo adentra esses temas.
4. Apenas ferramentas/APIs **stable** (Rust stable, flags estáveis do cargo/rustc).

**Estado de referência medido (quality-dashboard, bloco de 64 amostras @48 kHz, deadline
1333 µs):** WaveNet Standard CH16 42,7 µs (3,2%), Lite CH12 65,0 µs (4,9%), Feather CH8
26,7 µs, Nano CH4 21,7 µs, A2 Full 27,6 µs, A2 Lite 18,2 µs, LSTM 1x16 7,6 µs, LSTM 2x8
7,4 µs, ConvNet 10,4 µs, Linear 0,3 µs. Todos os contratos de fidelidade OK (ESR 1e-11…1e-14,
exceto ConvNet 2.54e-5 — tema diferido).

---

## SEÇÃO A — Microarquitetura dos kernels AVX2 (evidência: `target/dsp_hotpath.asm`)

Distribuição de amostras do perfil (cycles:u, total ≈ 2.444 amostras):

| Função                                                        | Amostras | % do total |
| ------------------------------------------------------------- | -------- | ---------- |
| `WaveNetLayer<1,16,3>::process_block_internal::<Avx2Math>`    | 1408     | ≈ 57,6%    |
| `fused_gemm_residual_batch_f32_avx2`                          | 523      | ≈ 21,4%    |
| `WaveNetLayer<1,8,3>::process_block_internal::<Avx2Math>`     | 347      | ≈ 14,2%    |
| `gemv_no_bias_f32_avx2`                                       | 155      | ≈ 6,3%     |
| Restante do pipeline (closure de captura, gate, bridge, etc.) | ~15      | < 1%       |

> Nota metodológica: o pipeline não-inferência (gain, gate, resampler, cabsim, bridge) já está
> exemplarmente vetorizado — auditoria dedicada não encontrou loops escalares relevantes.
> O ganho está concentrado nos 4 kernels acima.

### F-P1 — Spills de registradores no kernel dual 16-lane (`dot_product_16x_f32_dual_accumulate_avx2`) 🔴 CRÍTICO

**Onde:** `src/math/gemm/dot_16x/dot_f32_avx2.rs:327-429`, chamado por
`Conv1D::process_dual_frame_with_mixin` (`src/models/wavenet/conv1d_dual.rs:35-197`) a partir de
`WaveNetLayer::process_block_internal` (`src/models/wavenet/layer.rs:34-71`). É o kernel da
convolução dilatada do WaveNet Standard (CH=16, K=3) — **~58% de todos os ciclos do DSP**.

**Evidência (asm, loop interno em `0x1a42b0..0x1a43fd`):** o kernel Rust declara **16
acumuladores `__m256`** (unroll 4× taps × 2 frames × 2 metades lo/hi — linhas 338-353 do
fonte). Isso consome os 16 registradores ymm arquiteturais e não sobra nada para os 8 loads de
pesos + 8 broadcasts por iteração. O compilador então:

1. Derrama (spill) 4 acumuladores para a pilha, com round-trip load+FMA+store a cada iteração:

   ```text
   5.09%  vfmadd231ps %ymm0, %ymm1, %ymm11
   5.98%  vmovups %ymm11, 0xa0(%rsp)      ; spill do acumulador
   5.97%  vmovups 0xe0(%rsp), %ymm7       ; reload de outro acumulador
   2.53%  vmovups %ymm7, 0x100(%rsp)
   ```

2. Gera uma cadeia de rotação registrador-a-registrador sem função aritmética:

   ```text
   1.68%  vmovaps %ymm9, %ymm10
   4.13%  vmovaps %ymm8, %ymm7
   4.74%  vmovaps %ymm10, %ymm9
   ```

Somando as instruções de spill/reload/rotação: **≈ 25-30% dos ciclos da função são tráfego de
registradores, não matemática**. Sobre os 57,6% globais, isso representa ~15% do tempo total
de DSP do WaveNet Standard desperdiçado.

**Proposta de solução (bit-exact, ordem de somas preservada):** reestruturar em **duas
passadas** sobre os 48 taps (`K*IN = 3*16`):

- *Passada 1 (metade lo):* somente os 8 acumuladores lo (`acc_f0_lo0..3`, `acc_f1_lo0..3`),
  carregando apenas `w_lo` por tap. Registradores vivos: 8 acum + 1 peso + 2 broadcasts ≈ 11.
- *Passada 2 (metade hi):* idem com `acc_*_hi0..3` e `w_hi`.

Cada lane de saída mantém **exatamente** a mesma ordem de FMAs e a mesma árvore de redução
`(acc0+acc1)+(acc2+acc3)` → resultado **bit-idêntico** ao atual (zero risco de paridade).
Custo: os 48 `vbroadcastss` são re-executados na passada 2 e as linhas de pesos (3 KB,
L1-resident) são lidas 2×. Na Zen 2/Zen 3+ (2 loads de 32B/ciclo), esse custo é ordens de
magnitude menor que os spills eliminados.

*Alternativa B (maior ganho, exige re-baseline):* reduzir o split de acumuladores de 4-way
para 2-way (8 acumuladores em passada única). Muda a árvore de redução → resultado deixa de
ser bit-idêntico (ESR vs NAMcore muda dentro do mesmo tier ~1e-14). Só considerar se a
Alternativa A não bastar, com revalidação completa do contrato.

**Aplicar também em:** `dot_product_16x_f32_dual_avx2` (variante sem accumulate, mesmo padrão
de 16 acumuladores) e conferir `dot_8x` dual (CH=8 usa 2 frames × 1 ymm × 4-way = 8 acum —
provavelmente já saudável; o asm de `WaveNetLayer<1,8,3>` mostra bem menos spill).

**Ganho estimado:** −20…30% na latência do WaveNet Standard CH16 (42,7 µs → ~32 µs) e ganho
proporcional no Feather/head paths. **Validação:** `cargo bench dot_4x_bench inference_bench
regression_gate`, goldens bit-exact (`tests-quick`), contrato via `utils/quality-dashboard.sh`,
`utils/tests-performance-regression.sh`, e conferência do novo `dsp_hotpath.asm` (spills = 0
no loop interno).

---

### F-P2 — WaveNet Lite (CH=12) cai no kernel 4-wide: SKU mais leve é mais LENTO que o Standard 🔴 CRÍTICO

**Onde:** `src/loader/dispatcher/wavenet/layout.rs:346-354` (`select_interleave_width`) e
`src/models/wavenet/conv1d_dual.rs:162-194` (branch genérico 4-wide).

**Evidência (testes.log, dashboard):** Lite CH12 = **65,0 µs** vs Standard CH16 = **42,7 µs**,
apesar de o Lite ter ~56% dos MACs do Standard (12²/16² por camada, mesma malha de dilations).
Eficiência por MAC ~2,7× pior. Causa estrutural: `select_interleave_width(12)` retorna **4**
(12 não é múltiplo de 8 nem de 16), então cada frame processa **3 blocos seriais de 4 canais**
(`dot_product_4x_f32_dual_accumulate`) em vez de 1 bloco 16-wide — desperdiçando metade das
lanes de cada ymm e triplicando overhead de laço/prólogo.

**Proposta de solução (preferida — "pad-to-16"):** para `out_ch == 12`, armazenar os pesos no
layout 16-wide interleaved com **4 lanes zeradas** (mudança apenas no loader:
`transpose_conv1d_interleaved` + bias/init padding) e rotear para o mesmo kernel 16-wide dual
do F-P1. A saída usa as 12 primeiras lanes (store 8+4: `vmovups ymm` + `vmovups xmm`).

- Custo de memória: +33% nos pesos conv do Lite (~0,6 KB/camada) — irrelevante para L1/L2.
- Paridade: cada lane de saída computa seu próprio dot product; lanes de padding não afetam as
  lanes reais. **Atenção:** verificar se a ordem de acumulação por lane do kernel 16x é igual à
  do kernel 4x atual (ambos usam split 4-way por índice de tap com a mesma árvore de redução —
  se confirmado, o resultado é bit-idêntico; caso contrário, o ESR muda dentro do tier e o
  contrato precisa ser re-baselinado com aprovação explícita).

*Alternativa B:* blocos híbridos 8+4 (1 bloco 8-wide + 1 bloco 4-wide por frame). Menos
elegante, ganho menor, mas sem padding de memória.

**Bônus relacionado:** o head do Lite (HEAD=6) usa `gemv` com `out_len=6 < 8` (caminho
mascarado). Impacto menor; medir depois do fix principal.

**Ganho estimado:** Lite 65 µs → ~30-38 µs (−40…50%). **Validação:** goldens do Lite
(EVH-5150-Lite no dashboard: ESR 1.20e-12 deve se manter no tier), `inference_bench`,
`regression_gate`, contrato completo.

---

### F-P3 — `fused_gemm_residual_batch_f32_avx2`: strides runtime geram cadeias de `leaq` e reloads de ponteiros 🟠 ALTO

**Onde:** `src/math/gemm/gemm_batch/fused_residual_batch.rs:229-399`, chamado por
`DenseLayer::process_residual_batch` (`src/models/wavenet/dense.rs:28-46`) para a projeção 1×1
de cada camada WaveNet (16×16 no Standard, 8×8 no Feather...). **~21% dos ciclos do DSP.**

**Evidência (asm, `0x18b040..0x18b0cc`):** o endereçamento
`weights[in_c * out_len + out_c]` e `in_frames[f * in_len + in_c]` com `in_len`/`out_len`
**derivados em runtime** (`out_len = out_frames.len() / num_frames`, linha 245) impede o LLVM
de fazer strength-reduction completa. Resultado: por iteração do loop interno há 4-5 `leaq`
encadeados + reloads de ponteiros da pilha, somando **~25-30% dos ciclos da função**:

```text
3.07%  movq 0x70(%rsp), %rax     ; reload de ponteiro derramado
4.63%  movq %r10, 0x28(%rsp)
4.49%  leaq (%rdx,%rbp), %rdi    ; aritmética de índice runtime
7.04%  leaq (%rdx,%rsi), %rdi
3.68%  leaq (%rdx,%r11), %rdi
13.01% vfmadd231ps %ymm11, %ymm9, %ymm5   ; a matemática de verdade
```

**Proposta de solução:**

1. **Especialização const-generic:** o call-site (`WaveNetLayer<COND, CH, K>`) conhece `CH` em
   compile-time. Adicionar variante `fused_gemm_residual_batch_f32_const::<const IN: usize,
   const OUT: usize>` (função livre por ISA, despachada pelo `DenseLayer::process_residual_batch`
   quando as dimensões batem com o const do chamador). Com `IN=OUT=16` constantes, o LLVM
   dobra os offsets em modos de endereçamento imediatos e elimina os `leaq` encadeados.
2. **Strength-reduction manual (fallback dinâmico):** substituir `in_c * out_len + out_c` por
   ponteiros incrementais (`wp = wp.add(out_len)` a cada `in_c`), hoisted fora do loop de
   frames — remove as multiplicações mesmo no caminho runtime.

A ordem das FMAs por lane não muda → **bit-exact**.

**Ganho estimado:** −20…30% na função ⇒ −4…7% no total do WaveNet. **Validação:**
`gemv_bench`, goldens bit-exact, contrato, novo `dsp_hotpath.asm`.

---

### F-P4 — `gemv_no_bias_f32_avx2` com `in_len==1`: rechannel/input_mixin pagam prólogo de GEMV genérico 🟡 MÉDIO

**Onde:** `src/math/gemm/gemv/f32_avx2.rs:332-590` (branch `in_len == 1` na linha 348).
Call-sites: rechannel (`layer_array.rs:81-83`, IN=1→CH), input_mixin (`layer.rs:56-57`) e
head_rechannel (`layer_array.rs:171-175`, CH→HEAD). **~6% dos ciclos.**

**Evidência (asm):** o trabalho útil por chamada é minúsculo (para CH=16: 2 ymm mults por
frame), então prólogo/verificações dominam: `xorl %r11d,%r11d` com **9,16%** e o branch
`jae +0x140` com **5,57%** dos ciclos da função.

**Proposta de solução:** kernel dedicado `broadcast_scale_add_f32_avx2(weights[OUT],
in[frames], out[frames*OUT])` (produto externo colunar) com `OUT` const-generic (16/12/8/4),
`#[inline(always)]`, sem os branches do GEMV genérico; opcionalmente fundir o
rechannel+input_mixin no laço da primeira camada (os dois são broadcasts sobre o mesmo frame).
Bit-exact (mesma ordem de operações por lane).

**Ganho estimado:** −3…5% no total do WaveNet. **Validação:** goldens bit-exact, `gemv_bench`,
contrato.

---

### F-P5 — Tanh high-fidelity: cadeia serial de dependência estrangula o throughput (stall no `vminps`) 🟠 ALTO

**Onde:** `src/math/wavenet/accumulate/avx2.rs:46-60` (`tanh_and_accumulate_block_avx2` e
variantes overwrite/seed) chamando `simd_tanh_poly_avx2`
(`src/math/activations/tanh/high_fidelity.rs:82-96`).

**Evidência (asm, `0x1a4ae0..0x1a4b61` e mesmo padrão no layer CH8):** o laço processa **um
único ymm por iteração**. A cadeia por vetor é serial: clamp → round → 7 FMAs (polinômio) →
`vcvtps2dq`/`vpslld`/`vpaddd` (2^k) → 2 mults → `vdivps` (latência ~13c na Zen 2) → clamp
final. O sampling acumula no primeiro consumidor após a divisão:

```text
10.48% vminps %ymm12, %ymm9, %ymm12     ; layer CH16 — espera da vdivps
13.60% vminps %ymm12, %ymm9, %ymm12     ; layer CH8 — idem
```

Com blocos de 64 frames × 16 canais = 1024 floats por camada, há paralelismo de dados de sobra
para esconder a latência — hoje desperdiçado.

**Proposta de solução:** desenrolar o laço da ativação para **2 (ou 4) vetores ymm
independentes por iteração**, intercalando as cadeias (software pipelining). As operações por
lane são idênticas → **bit-exact**. Não tocar na matemática (a `vdivps` exata é requisito de
paridade — proibido substituir por `rcpps`+NR).

**Ganho estimado:** o pass de ativação encolhe ~40-50%; −5…8% no total do WaveNet.
**Validação:** goldens bit-exact, `math_bench` (tanh HF), contrato, asm.

---

### F-P6 — Bounds-checks residuais e caminhos de pânico nos kernels quentes 🟢 BAIXO (higiene defensiva)

**Onde:** finais de função com `callq core::slice::index::slice_index_fail` /
`panic_bounds_check` em `WaveNetLayer::process_block_internal`,
`fused_gemm_residual_batch_f32_avx2` (tail single-frame usa indexação checada —
`fused_residual_batch.rs:181,200-205,351-364`) e `gemv_no_bias_f32_avx2`.

**Análise:** os checks estão fora dos laços internos (custo direto ~0), mas (a) inflam o
código e criam branches extras no prólogo; (b) o caminho de tail com `[]` checado é
inconsistente com o resto do kernel (`get_unchecked` com invariantes documentadas).

**Proposta:** homogeneizar: hoist de asserts de tamanho no início da função
(`assert!`/`debug_assert!` + `let` de slices com tamanho fixo via `split_at`), permitindo ao
LLVM eliminar os panics; manter a documentação de invariantes de segurança. Ganho de
performance marginal; ganho real é de robustez/consistência.

**Validação:** `utils/lints.sh`, testes estruturais, asm sem `panic_bounds_check` nos kernels.

---

## SEÇÃO B — Insights ocultos do `testes.log`

### F-L1 — BUG: `madvise(MADV_DONTFORK | MADV_DONTDUMP)` — advice não é bitmask ⇒ EINVAL silencioso em produção 🟠 ALTO (correção trivial)

**Onde:** `src/standalone/pw_host/bridge.rs:44-58`.

**Evidência (testes.log:3246):**

```text
[WARN nam_rs::standalone::pw_host::bridge] madvise(MADV_DONTFORK|MADV_DONTDUMP)
returned -1 (errno: Invalid argument (os error 22)).
```

O parâmetro `advice` do `madvise(2)` é um **enum, não um bitmask**. `MADV_DONTFORK (10) |
MADV_DONTDUMP (16) = 26`, valor inválido → a proteção pretendida (excluir os buffers do bridge
de forks/core-dumps) **nunca é aplicada**, em todo boot, desde sempre.

**Proposta de solução:** duas chamadas separadas:

```rust
libc::madvise(ptr, len, libc::MADV_DONTFORK);
libc::madvise(ptr, len, libc::MADV_DONTDUMP);
```

com verificação de retorno individual e log de warn específico por advice. Adicionar teste de
integração que verifica ausência do warn no boot (ou teste unitário do wrapper com ambos os
advices).

**Ganho:** correção funcional (robustez RT: evita CoW pós-fork tocando páginas do bridge) e
eliminação de um WARN enganoso. **Validação:** `tests-quick` + execução standalone sem WARN.

---

### F-L2 — Incoerência de THP: `PR_SET_THP_DISABLE` process-wide anula o alocador de huge pages dos pesos em hot-swap 🟠 ALTO

**Onde:** `src/standalone/rt_setup/thread.rs:26-29` (prctl) × `src/math/common/huge_alloc.rs:109-120`
e `src/dsp/mirror_buf/alloc.rs:233-234` (MADV_HUGEPAGE + MADV_COLLAPSE, códigos de retorno
**ignorados**: `let _madvise_rc`, `let _collapse_rc`).

**Evidência e análise:**

- `main.rs:163` carrega o modelo inicial **antes** de `rt_setup::configure_process_wide()`
  (`main.rs:239`) → o modelo inicial ganha THP normalmente.
- Depois do prctl, a documentação do kernel é explícita: `PR_SET_THP_DISABLE, 1, 0` desabilita
  THP para o processo *"irrespective of global THP controls or madvise(..., MADV_COLLAPSE)
  being used"*. Logo, **todo modelo trocado em runtime (hot-swap via CLI/GUI) perde as huge
  pages silenciosamente** (o `MADV_COLLAPSE` falha e o rc é descartado).
- A telemetria (`testes.log:3251` — "THP advice active — kernel may promote to 2 MB") e o flag
  `THP_ACTIVE` (`src/common/spsc/status.rs:97`) reportam sucesso sem confirmar de fato —
  informação potencialmente mentirosa para diagnóstico.
- Estado resultante: dTLB/paridade de layout de memória **diferente entre o modelo do boot e os
  modelos trocados ao vivo** — exatamente o tipo de variabilidade que a auditoria de jitter quer
  eliminar.

**Proposta de solução (kernel 7.0 disponível na stack alvo):**

1. Trocar o prctl para o modo moderno:
   `prctl(PR_SET_THP_DISABLE, 1, PR_THP_DISABLE_EXCEPT_ADVISED, 0, 0)` — mantém o processo
   livre do khugepaged/THP background (motivação original), mas **permite** THP onde o código
   pede explicitamente (`MADV_HUGEPAGE`/`MADV_COLLAPSE` dos arenas de pesos e do MirroredBuffer).
   Fallback: se o prctl com flag retornar EINVAL (kernel antigo), manter comportamento atual e
   registrar o downgrade em log.
2. Parar de descartar os retornos: propagar `madvise`/`MADV_COLLAPSE` rc até o flag
   `THP_ACTIVE` (setar apenas se o collapse retornou 0; opcionalmente confirmar via
   `/proc/self/smaps_rollup` `AnonHugePages` em teste).
3. Teste de integração: hot-swap de modelo + verificação de que o arena novo está em 2 MB
   (smaps) com o prctl ativo.

**Ganho:** consistência determinística de dTLB para *todos* os modelos (não só o do boot);
telemetria honesta. **Validação:** teste novo + `tests-long` (heap-audit) + inspeção smaps.

---

### F-L3 — Pipeline PGO/BOLT deixa ganho na mesa: perfil raso, sem LBR, quantum irreal e CLAP sem BOLT 🟠 ALTO

**Onde:** `utils/build-release.sh:295-380` (fase 4).

**Evidências (testes.log:3236-3267 + build script):**

1. **Perfil raso:** Zen 2 (5700U) não tem LBR/BRS → fallback para sampling 4 kHz por apenas
   **3 segundos** ⇒ ~2,4 mil amostras. A recomendação do BOLT é cobrir ~1 bilhão de instruções;
   com basic-events (`--basic-events`) o próprio BOLT avisa que o ganho é limitado.
2. **Quantum irreal:** a telemetria do run de profiling mostra "3 blocks" em 10 s com sampling
   1/16 ⇒ quantum efetivo ≈ 8192 amostras. O binário é otimizado com um perfil onde prólogos/
   epílogos/branches de bloco pequeno (o caso de produção: 64-256 amostras) quase não aparecem.
   O `dsp_hotpath.asm` herda o mesmo viés.
3. **Workload mono-modelo:** só WaveNet Standard é perfilado no BOLT (o PGO usa 3 modelos);
   usuários de LSTM/A2 recebem layout sub-ótimo.
4. **CLAP sem BOLT:** só o executável standalone é BOLTado; o `nam-rs.clap` (o artefato que
   roda dentro de DAW, caso de uso mais sensível a jitter) recebe apenas PGO. BOLT suporta
   shared libraries (`--relocs`; exige symtab não-stripped — atenção: `Cargo.toml` usa
   `strip = true` no profile release; será preciso manter símbolos até o passo BOLT e stripar
   depois).
5. `--no-huge-pages` está fixo — coerente com o THP atual, mas deve ser reavaliado junto com
   F-L2 (`-hugify` mapeia o texto quente em páginas de 2 MB → menos iTLB miss; hoje seria
   anulado pelo prctl).

**Proposta de solução (em ordem de valor):**

1. **Migrar para instrumentação BOLT** (`llvm-bolt -instrument` → rodar
   `pgo_profiling_workload` (ou o standalone com pw-play) → `merge-fdata` → otimizar). Segundo
   os autores do BOLT, instrumentação é *mais precisa* que hardware counters e independe de
   LBR — ideal para máquinas AMD Zen 2. Elimina a dependência de `perf_event_paranoid`.
2. Enquanto sampling for usado: aumentar duração (3 s → 30-60 s), forçar quantum de produção
   (`PIPEWIRE_QUANTUM=64/48000` ou `node.force-quantum`) e ciclar pelos modelos do catálogo
   (WaveNet Std/Lite, A2, LSTM) como o PGO já faz.
3. Aplicar BOLT também ao `nam-rs.clap` (build com símbolos → BOLT → strip).
4. Reavaliar `--no-huge-pages` → `-hugify` após F-L2.
5. Regenerar `target/dsp_hotpath.asm` com quantum 64 para que a próxima auditoria de asm veja
   o perfil de produção.

**Ganho estimado:** BOLT bem alimentado costuma render 2-8% adicionais em código front-end
bound; determinismo maior entre builds. **Validação:** `utils/tests-performance-regression.sh`
antes/depois; contrato de latência do dashboard.

---

### F-L4 — Dashboard: adicionar métrica de eficiência (µs/MMAC) para flagrar SKUs degenerados 🟢 BAIXO

**Onde:** `utils/quality-dashboard.sh` (tabela de performance).

**Racional:** a anomalia do F-P2 (Lite 52% mais lento que Standard com 56% dos MACs) ficou
**invisível** por meses porque o contrato compara cada SKU apenas consigo mesmo. Uma coluna
derivada "µs por MMAC" (MACs do SKU calculados do topology) tornaria qualquer futura
degeneração de kernel imediatamente gritante no dashboard, sem custo de runtime.

**Proposta:** computar MACs por bloco por SKU (função no gerador do dashboard) e imprimir a
coluna + gate suave (warn se um SKU destoar >2× da mediana de eficiência da família).

---

## SEÇÃO C — Stack moderna: Kernel Linux 7.0 · PipeWire 1.6 · CLAP 1.2

*(Filtro crítico aplicado: somente itens com benefício palpável para o caso de uso NAM-rs.
Descartados conscientemente: io_uring (não há I/O no RT), sched_ext/uclamp (não-stable ou sem
ganho sobre SCHED_FIFO já configurado), AVX-512/AMX (fora do baseline v3), CRC via PCLMUL no
loader NAMB (cold path, arquivos de KB.)*

### F-S1 — Avaliar nó único `pw_filter` vs dual-stream + DspBridge (potencial de −1 quantum de latência) ✅ ARQUIVADO (WONT-FIX)

**Onde:** `src/standalone/pw_host/` (arquitetura documentada em `mod.rs:11-26`),
`src/dsp/pipeline/bridge.rs`.

**Análise:** a arquitetura atual usa 2 streams (capture `Audio/Sink` + playback
`Stream/Output/Audio`) conectados pelo `DspBridge` lock-free. Os props `node.group` +
`node.link-group` (`capture/setup.rs:61-62`, `playback.rs:37-38`) já garantem o mesmo driver
cycle, mas **não garantem a ordem capture→playback dentro do ciclo** — quando a ordem não
favorece, o playback lê o bloco do ciclo anterior ⇒ +1 quantum de latência (1,33 ms @64) além
de 2 cópias e sincronização atômica.

**Resolução (2026-07-19):** A ordem capture→playback **é garantida** pelo `node.group`/`node.link-group`

- ordem de registro no driver (capture criado primeiro em `run.rs`) + `PRIORITY_DRIVER=2000` no capture.
  O resultado é **0 quantum de latência extra intra-ciclo**. O protótipo `pw_filter` e Fases 2+ estão
  arquivados como WONT-FIX. Ver relatório completo: [`docs/latency_sprint1_analysis.md`](./docs/latency_sprint1_analysis.md).

**Proposta original (arquivada):**

O PipeWire oferece exatamente a primitiva para esse caso: **`pw_filter`** — um único nó com
porta de entrada e de saída processadas **no mesmo ciclo do grafo** (`PW_FILTER_FLAG_RT_PROCESS`),
com buffers DSP f32 e latência declarável via `spa_process_latency_build`. Eliminaria o
DspBridge, as 2 cópias e o risco de +1 quantum.

**Gap identificado:** `pipewire-rs 0.10` **não expõe bindings do `pw_filter`** (módulos
disponíveis: stream, core, context, ...; sem `filter`). Implementar exigiria FFI direto via
`pipewire-sys` (já é dependência transitiva) ou contribuição upstream.

**Proposta em fases (medir antes de construir):**

1. **Medição:** instrumentar com `pw_stream::time()` (`Time.delay`) / `pw-top` a latência real
   capture→playback da configuração atual. Se a ordem intra-ciclo já estiver correta na prática
   (0 quantum extra), **arquivar o finding** com os números.
2. Se houver +1 quantum: protótipo `pw_filter` via `pipewire-sys` (unsafe isolado no módulo
   `pw_host`, mesmas regras RT), mantendo o modo dual-stream como fallback runtime.
3. Reavaliar o campo `node.latency` e reportar a latência do resampler/oversample via
   `spa_process_latency` no caminho novo.

**Ganho potencial:** −1,33 ms de latência ponta-a-ponta @64/48k (enorme para o músico) + menos
2 cópias/bloco. **Risco:** médio-alto (FFI novo em código RT) — por isso a fase de medição é
obrigatória antes de qualquer código.

</details>

---

### F-S2 — CLAP: implementar `clap.tail` (e considerar `audio-ports-activation`) 🟡 MÉDIO

**Onde:** `src/clap/extensions/` (extensões atuais: audio_ports, latency, params, state,
state_context, render, preset_load, remote_controls, param_indication, track_info, gui).

**Análise:** o NAM-rs com CabSim ativo tem cauda real (IR de até segundos) e o host não tem
como saber — sem `clap.tail`, DAWs conservadores processam o plugin para sempre (CPU
desperdiçada em faixas silenciosas) ou cortam a cauda do IR em bounce. A extensão é **stable**
no CLAP 1.2 e o `clack-extensions 0.1` já embarca o módulo `tail` (verificado no crate
vendorizado) — implementação é pequena: reportar `ir_len + oversample_latency +
resampler_latency` em samples, atualizando quando o IR muda (via `tail.changed()`).

**Complemento de baixo custo:** avaliar `clap.audio-ports-activation` — permite ao host
desativar o canal R quando o plugin está num chain mono; hoje o NAM-rs detecta mono por
heurística (`compute_max_diff` por bloco). Um sinal explícito do host elimina a heurística e
garante o caminho `process_mono` (metade do custo de gain/copy stages) de forma determinística.

**Ganho:** economia real de CPU no DAW (suspensão em silêncio) + bounce correto da cauda do
CabSim + mono determinístico. **Validação:** `clap-validator` (já integrado ao build),
`clap_bench`, testes de lifecycle existentes.

---

## Anti-escopo registrado (decisões conscientes desta auditoria)

| Item                                                          | Motivo do descarte                                                                                             |
| ------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| Aproximações `rcpps`/`rsqrtps` no tanh/sigmoid                | Viola paridade/fidelidade (proibido)                                                                           |
| Multiversioning AVX-512/VNNI/BF16                             | Fora do foco x86-64-v3 desta rodada (será tratado oportunamente)                                               |
| ConvNet parity & WaveNet A2 max                               | Diferidos por decisão do PO (`TODO-convnet_parity.md`, `TODO-wavenet_a2_max.md`)                               |
| io_uring, sched_ext, ferramentas nightly (`-Ztune-cpu`, etc.) | Política stable-only / sem ganho no RT path                                                                    |
| CRC32 do NAMB via PCLMUL                                      | Cold path de loader; ganho imensurável                                                                         |
| Prioridade FIFO 83 (PipeWire) vs 90 (config própria)          | Comportamento correto por design (`thread.rs:119` respeita FIFO pré-existente do data-loop); apenas documentar |

---

## ÉPICOS (agrupamento para execução segura e incremental)

### EPIC-1 — "Zero Spill": microarquitetura dos kernels WaveNet AVX2 🔴 [DONE]

> Findings: **F-P1, F-P3, F-P5, F-P4, F-P6** (nesta ordem). Ganho combinado estimado no
> WaveNet Standard: **42,7 µs → ~30 µs**; beneficia Feather/Nano/head paths por arrasto.

Sequência recomendada (cada passo com gate completo antes do próximo):

1. F-P1 (two-pass lo/hi — bit-exact, maior ganho isolado);
2. F-P3 (const-generic no fused GEMM residual — bit-exact);
3. F-P5 (ILP ×2/×4 na ativação — bit-exact);
4. F-P4 (kernel broadcast-scale para in_len==1 — bit-exact);
5. F-P6 (higiene de bounds-checks).

Gate por passo: goldens bit-exact (`tests-quick`), `cargo bench` dirigido (dot_4x/gemv/math),
`utils/quality-dashboard.sh` (contrato integral), `utils/tests-performance-regression.sh`,
diff do `dsp_hotpath.asm` (spills eliminados). Risco: baixo (transformações bit-exact);
qualquer desvio de ESR = abortar e investigar.

### EPIC-2 — "Lite à altura do nome": caminho 16-wide para CH=12 🔴 [DONE]

> Finding: **F-P2** (+ coluna µs/MMAC do **F-L4** como guarda-corpo permanente).

Depende do kernel 16x saudável (fazer após F-P1). Entregável: `select_interleave_width` com
caminho pad-to-16 para out_ch=12, layout de pesos com padding no loader, store 8+4, goldens do
Lite revalidados. Meta: Lite ≤ 38 µs.

**Status da meta (T2.S4.2, 2026-07-19):** Após implementação do loader 16-wide (T2.S3.1) combinado com a
correção do `store_16_accums` (T2.S3.3: store SIMD 8+4) e a especialização `fused_gemm_residual_batch_f32_12x12`
(YMM+XMM no OneByOne), a latência do Lite CH12 no dashboard caiu significativamente de **63,3 µs → 51.88 µs**
(**−18%** de redução). O contrato global de performance e fidelidade do dashboard passou com sucesso (CONTRATO OK).

A investigação de profiling de sub-etapas revelou a barreira física:

1. Conv1D (SIMD 16x): ~37% do tempo (~19 us)
2. OneByOne Residual (Dense SIMD 12x12): ~42% do tempo (~22 us)

O gargalo principal reside no desalinhamento de memória nativo dos 12 canais (passo de 12 floats / striding não-alinhado), que provoca cache-line splits frequentes e impede a CPU de atingir a latência máxima que obtém com o Standard CH16 (37 us para 18 camadas com passo de 16 floats). Para atingir o teto de ≤ 38 µs, a rota definitiva é o **pad-to-16 homogêneo (CH=16 estático)** em todos os tensores internos da WaveNet Lite.

### EPIC-3 — Coerência de memória & kernel moderno 🟠 [DONE]

> Findings: **F-L1** (madvise split — quick win, pode ser feito imediatamente) e **F-L2**
> (PR_THP_DISABLE_EXCEPT_ADVISED + telemetria THP honesta + teste de hot-swap com smaps).

Independente dos épicos 1-2; não toca em kernels DSP. Risco baixo; todo o código é cold-path
de setup, mas exige teste em kernel alvo (7.0) + fallback documentado para kernels antigos.

### EPIC-4 — Build pipeline de próxima geração (PGO + BOLT instrumentado) 🟠 [DONE]

> Finding: **F-L3** (instrumentação BOLT, quantum de produção no profiling, BOLT no CLAP,
> workload multi-modelo, reavaliar hugify pós-EPIC-3).

Somente `utils/build-release.sh` + perfis de build (símbolos até o passo BOLT). Validar com
A/B do `tests-performance-regression.sh` e contrato de latência. Executar por último (para que
os épicos 1-2 sejam medidos sem viés de layout antigo) ou antes com re-run após.

### EPIC-5 — Stack PipeWire/CLAP (latência ponta-a-ponta e cidadania de host) 🟡 [DONE COM RESSALVA]

> Findings: **F-S1** (fase 1 de medição é pré-requisito duro; só avançar com números) e
> **F-S2** (clap.tail + avaliação de audio-ports-activation).

F-S2 é independente e pequeno (pode entrar em qualquer sprint). F-S1 fase 2+ só com aprovação
após a medição comprovar o +1 quantum.

**Status (2026-07-19):** F-S1 concluído com decisão NO-GO bem fundamentada (ver
`docs/latency_sprint1_analysis.md`) — arquivado corretamente sem código FFI arriscado. F-S2
entregou `clap.tail` + bônus `audio-ports-activation`, mas a auditoria de reverificação
(seção abaixo) encontrou um **defeito de semântica** no valor reportado por `clap.tail` — ver
**F-A1**. EPIC-5 permanece "done" para fins de escopo original, mas com dívida técnica nova
registrada.

---

## AUDITORIA DE REVERIFICAÇÃO (2026-07-19) — "Concluído" revisitado

Segunda passada da `revisor-auditor`, agora em modo *verificação de implementação*: toda a
árvore de commits do EPIC-1 ao EPIC-5 foi lida linha a linha, cruzada com testes/benchmarks
existentes e, para os kernels AVX2, **revalidada de forma independente regenerando e
reinspecionando `target/dsp_hotpath.asm`** — não apenas confiando nas mensagens de commit.

### Veredito por épico

| Épico                     | Veredito                                                                             | Evidência                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ------------------------- | ------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| EPIC-1 (F-P1,P3,P4,P5,P6) | ✅ **Implementado corretamente, bit-exact confirmado**                               | asm regenerado: 0 spills de acumulador em `WaveNetLayer<1,16,3>` (era 25-30% dos ciclos); `vminps` do tanh agora aparece em pares intercalados (era 1 instrução a 10-13%, hoje maior valor isolado é 3,6-5,2%); kernels const-generic `fused_gemm_residual_batch_f32_const::<16,16>`/`<8,8>` aparecem no perfil real (não é código morto); sem cadeias `leaq` nos hot-spots do GEMM residual                                        |
| EPIC-2 (F-P2)             | 🟡 **Implementado parcialmente — meta original não atingida, gap bem diagnosticado** | Lite CH12: 65,0 µs → **52,2 µs** (dashboard atual, −19,7%), mas meta era ≤ 38 µs. µs/MMAC = 6043,98 vs 2407,55 do Standard (2,5× menos eficiente por MAC). Causa-raiz corretamente identificada pela própria equipe (cache-line splits por stride de 12 floats) com solução seguinte já apontada (pad-to-16 em todos os tensores) mas **não implementada** — ver **F-A2**                                                           |
| EPIC-3 (F-L1,L2)          | ✅ **Implementado corretamente**                                                     | `madvise` dividido em 2 chamadas independentes; `PR_THP_DISABLE_EXCEPT_ADVISED` com fallback automático por `errno==EINVAL`; `HugePageStatus::Transparent` agora condicionado ao `collapse_rc==0` real. Ressalva menor de higiene de teste — ver **F-A3**                                                                                                                                                                           |
| EPIC-4 (F-L3)             | ✅ **Implementado corretamente, além do escopo mínimo pedido**                       | Migrou para instrumentação BOLT (mais precisa, sem depender de LBR/perf_event_paranoid); aplica BOLT a standalone **e** ao `.so` do CLAP; `CARGO_PROFILE_DIST_STRIP=false` + `strip` manual pós-BOLT (resolve a contradição com `strip=true` do profile `dist` que eu mesmo não havia detalhado explicitamente); `-hugify` habilitado coerentemente após o fix de THP do EPIC-3; quantum de profiling casado com produção (`-b 64`) |
| EPIC-5 (F-S1,S2)          | 🟡 **F-S1 correto; F-S2 com bug de semântica**                                       | F-S1: decisão NO-GO bem fundamentada e documentada, com instrumentação real (`pw_stream::time()`) para verificação empírica contínua — abordagem exemplar de "medir antes de construir". F-S2: `clap.tail` **implementado com o valor errado** — ver **F-A1** (crítico para a intenção original do finding)                                                                                                                         |

### Ganhos medidos vs. estimados (honestidade de calibração)

| SKU                   | Baseline (findings original) | Medido agora (`docs/quality-contract.txt`) | Estimativa original                                                       | Realidade                                                                                                                                                                                                                                 |
| --------------------- | ---------------------------- | ------------------------------------------ | ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| WaveNet Standard CH16 | 42,7 µs                      | **37,0 µs (−13,3%)**                       | −20…30% (→ ~30-32 µs)                                                     | Ganho real, abaixo da estimativa otimista, mas F-P1 sozinho já eliminou 100% dos spills medidos — o restante do gap para 30 µs provavelmente está nos custos de tap-gathering escalar (`vmovss`) não cobertos por nenhum finding original |
| WaveNet Feather CH8   | 26,7 µs                      | 19,4 µs (−27,3%)                           | Ganho por arrasto (F-P5 aplicado; F-P1 não se aplica, kernel já saudável) | Superou a expectativa qualitativa — bom                                                                                                                                                                                                   |
| WaveNet Lite CH12     | 65,0 µs                      | **52,2 µs (−19,7%)**                       | −40…50% (→ ≤ 38 µs)                                                       | Abaixo da meta — ver F-A2                                                                                                                                                                                                                 |
| WaveNet Nano CH4      | 21,7 µs                      | 17,2 µs (−20,7%)                           | Ganho por arrasto                                                         | Bom                                                                                                                                                                                                                                       |

---

## NOVOS FINDINGS (da reverificação de 2026-07-19)

### F-A1 — BUG: `clap.tail` reporta o valor errado — usa `current_latency` (delay fixo) em vez da duração real da cauda do CabSim 🔴 CRÍTICO

**Onde:** `src/clap/extensions/tail.rs:15-22`.

```rust
fn get(&self) -> TailLength {
    TailLength::Finite(self.shared.rt_to_ui.current_latency.load(Ordering::Relaxed))
}
```

**O problema:** `current_latency` (calculado em `src/clap/processor/events.rs:83-88`) é:

```rust
let mut effective_latency = self.resampler.latency_samples(host_rate);
effective_latency += self.os_l.latency_samples() as u32;
// + conv.latency_samples()  (== partition_size da UPOLS, tipicamente 64-512 amostras)
```

Isso é o **delay fixo de processamento** (o mesmo valor já reportado por `clap.latency`,
`src/clap/extensions/latency.rs:18`) — **não** é a duração de "ring-out" após o input silenciar.
Para CabSim com IR de cabine real (centenas de ms a poucos segundos), a cauda verdadeira é
aproximadamente `ConvEngine::num_partitions() × partition_size` (o comprimento total do IR
particionado, já exposto publicamente via `num_partitions()` em `src/dsp/cabsim/conv.rs:199-201`),
que pode ser **50-100× maior** que o `partition_size` isolado.

**Consequência prática:** exatamente o problema que o finding original (F-S2) queria resolver
permanece sem solução — um host conservador que confia em `clap.tail` para decidir quando
parar de processar/cortar o áudio no bounce vai **cortar a cauda da reverberação do cabinet**
porque o valor reportado é ordens de magnitude menor que o real. Pior: como o valor é idêntico
ao de `clap.latency`, um host que já lida bem com latência ganha zero informação adicional —
a extensão está, na prática, um no-op mais custoso.

**Proposta de solução:**

```rust
fn get(&self) -> TailLength {
    let base = self.shared.rt_to_ui.current_latency.load(Ordering::Relaxed);
    let cabsim_tail = self
        .cabsim_engine_ref() // acessor existente ou novo, leitura Relaxed de um AtomicU32
        .map(|conv| (conv.num_partitions() * conv.partition_size()) as u32)
        .unwrap_or(0);
    TailLength::Finite(base + cabsim_tail)
}
```

Publicar `cabsim_tail_samples` como um `AtomicU32` no mesmo local de `current_latency`
(`src/clap/plugin/shared.rs:137`), atualizado no mesmo ponto onde o CabSim é hot-swapped
(`src/standalone/pw_host/rt_callback/cabsim_swap.rs` e equivalente CLAP) — **não** ler o
`ConvEngine` diretamente na main-thread para evitar data race com a troca RT do IR.

**Teste que falta (nenhum existe hoje):** teste de integração que carrega um IR longo (ex.:
`tests/fixtures/*.wav` de vários segundos), consulta `PluginTailImpl::get()` via
`clack-host`/harness de teste do CLAP, e afirma que o valor retornado é
`>= ir_samples * 0.9` (tolerância de arredondamento de partição) — não apenas `> 0`.

**Validação:** novo teste de integração (acima) + `clap-validator` (não cobre semântica de
valor, apenas presença da extensão) + inspeção manual em uma DAW real (Reaper/Bitwig) com
bounce de um sinal com silêncio final e CabSim ativo, confirmando que a cauda não é cortada.

---

### F-A2 — WaveNet Lite: meta de eficiência (≤ 38 µs / paridade de µs-por-MMAC com o Standard) não atingida — causa-raiz mapeada, solução pad-to-16 completo pendente 🟠 ALTO

**Onde:** `src/models/wavenet/conv_input.rs`, `src/models/wavenet/layer.rs`,
`src/models/wavenet/conv1d_dual.rs` (buffers internos ainda dimensionados para CH=12 nativo,
não CH=16 com padding).

**Estado atual (medido, `docs/quality-contract.txt:87-90`):** Lite CH12 = **52,2 µs**
(6043,98 µs/MMAC) vs Standard CH16 = **37,0 µs** (2407,55 µs/MMAC) — **2,51× menos eficiente
por MAC**, apesar do EPIC-2 já ter entregue uma melhoria real de −19,7% (65,0→52,2 µs) via
padding dos **pesos** da convolução para o layout 16-wide (T2.S3.1) e um kernel dedicado
`fused_gemm_residual_batch_f32_12x12` para a projeção 1×1.

**Causa-raiz (já diagnosticada e documentada pela própria equipe em EPIC-2/T2.S4.2, linha 520
deste documento):** os **demais tensores internos** (buffers de estado da camada, scratch de
mixin/conv, ring buffers de dilatação) continuam dimensionados nativamente para 12 canais —
um stride não-potência-de-2 que gera cache-line splits frequentes toda vez que a CPU acessa
essas estruturas, mesmo com os pesos já no layout 16-wide. Ou seja: **o kernel de compute foi
corrigido (F-P1/F-P2 nos pesos), mas o layout de memória ao redor dele não** — o gargalo
migrou de "register spill" (resolvido) para "cache-line split por desalinhamento estrutural"
(não resolvido).

**Proposta de solução — pad-to-16 estrutural completo:**

1. Redimensionar `WaveNetLayer<COND, 12, K>`'s buffers internos (`scratch_mixin`,
   `scratch_conv`, estado de dilatação em `conv1d_dual.rs`) para `16` de facto, com as 4
   lanes extras permanentemente zeradas e **nunca lidas para produzir a saída** (a saída
   continua sendo as primeiras 12 lanes — já é o padrão usado pelo store 8+4 do T2.S3.3).
2. Isso alinha *todos* os acessos de memória do Lite ao mesmo stride de 16 floats (64 bytes
   = 1 cache-line) do Standard, eliminando os cache-line splits identificados.
3. Custo de memória: +33% nos buffers de estado do Lite (kilobytes, não relevante para
   L1/L2/L3 em nenhum SKU do catálogo).
4. **Risco de paridade:** as 4 lanes de padding devem ser garantidamente zero em todo o
   ciclo de vida (inicialização, reset, hot-swap) — um teste de fuzzing/proptest dedicado
   (similar ao já existente para `T2.S3.2`, `layout.rs`) deve cobrir "lane de padding nunca
   vira NaN/Inf/subnormal ao longo de N blocos consecutivos" para blindar contra UB sutil.

**Meta realista revisada:** com pad-to-16 completo, a expectativa é aproximar (não
necessariamente igualar) a eficiência do Standard, já que o Lite ainda paga overhead de
loop/setup proporcionalmente maior (menos trabalho por chamada). Meta sugerida: **≤ 42 µs**
(µs/MMAC ≤ 3000, redução do gap de 2,51× para ≤ 1,25×) — mais realista que o ≤ 38 µs original,
que implicitamente assumia paridade perfeita de eficiência com o Standard.

**Validação:** golden EVH-5150-Lite (ESR deve permanecer em 1e-12, já confirmado bit-exato
até aqui em T2.S4.1), `inference_bench`, `regression_gate`, novo teste de padding-zero acima,
`utils/quality-dashboard.sh` com a coluna µs/MMAC (F-L4) como critério de aceite explícito.

---

### F-A3 — Teste `test_prctl_thp_except_advised_no_crash` modifica estado global do processo sem isolamento 🟢 BAIXO (higiene de teste)

**Onde:** `tests/models/thp_coherence.rs:86-117`.

**Problema:** o teste chama `prctl(PR_SET_THP_DISABLE, 1, PR_THP_DISABLE_EXCEPT_ADVISED, 0, 0)`
e, ao final, **incondicionalmente** chama `prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0)` (linha 116) —
desabilitando THP **completamente para todo o processo de teste**, não apenas revertendo ao
estado anterior. `libtest` executa testes como threads dentro do mesmo processo por padrão;
qualquer teste que rode concorrentemente ou depois deste no mesmo binário (incluindo
`test_thp_coherence_smaps_consistency`, que **depende** de THP estar potencialmente ativo)
pode observar um estado de THP diferente do que teria sem este teste ter rodado antes/junto.

**Proposta de solução:** ler o estado original via `PR_GET_THP_DISABLE` antes de qualquer
`PR_SET_THP_DISABLE`, e restaurar exatamente esse valor no `Drop`/final do teste (ou usar
`#[serial]` do crate `serial_test`, já que ambos os testes deste arquivo tocam estado
process-wide, combinado com a leitura/restauração do valor original).

**Validação:** `cargo test --test thp_coherence -- --test-threads=1` deve produzir o mesmo
resultado que a suíte completa em paralelo (hoje pode não produzir, por ordem de execução).

---

### F-A4 — WaveNet Feather CH8: eficiência por MAC pior que o Standard, dashboard já sinaliza (guarda-corpo funcionando) — investigar oportunamente 🟢 BAIXO

**Onde:** `src/models/wavenet/conv1d_dual.rs` (branch 8-wide), evidenciado pela própria
métrica nova do **F-L4**: `docs/quality-contract.txt:88` mostra Feather CH8 em
**5046,88 µs/MMAC** vs 2407,55 do Standard (2,1× pior).

**Análise:** a reinspeção do `dsp_hotpath.asm` para `WaveNetLayer<1,8,3>` confirma que **não
há spills de acumulador** (diferente do que motivou F-P1/F-P2) — o custo aqui é dominado por
`vmovss` escalares (tap-gathering elemento-a-elemento para os buffers `flat_taps`) e overhead
fixo de loop/setup (`xorl`, `jbe`), proporcionalmente maior porque o Feather processa menos
trabalho por chamada. Isto é estruturalmente similar ao caso do Nano (CH4, 17947,92 µs/MMAC),
onde o overhead fixo é aceitável porque a latência absoluta já é pequena (19,4 µs / 17,2 µs).

**Recomendação:** **não atacar agora** — nenhum SKU do catálogo com Feather/Nano está perto do
budget RT (Folga > 96% em todos), e o ganho absoluto de otimizar o tap-gathering escalar seria
de poucos µs. Registrado aqui apenas para constar que o guarda-corpo `µs/MMAC` do F-L4 está
funcionando exatamente como projetado — cumpriu seu propósito de auditoria contínua nesta
própria reverificação. Revisitar somente se, no futuro, o `partition_size`/bloco RT diminuir
(ex.: buffers de 32 amostras) e o overhead fixo passar a pesar proporcionalmente mais.

---

## EPIC-6 — Fechamento da reverificação: correção de semântica e continuidade dos gaps abertos 🔴

> Findings desta segunda auditoria: **F-A1** (crítico, bug real de comportamento no host),
> **F-A2** (continuação do EPIC-2, meta revisada), **F-A3** (higiene de teste), **F-A4**
> (nenhuma ação — apenas registro).

Ordem recomendada:

1. **F-A1 primeiro e isolado** — é uma correção de bug com risco de regressão comportamental
   em produção (hosts que já confiam no valor atual de `clap.tail`, mesmo que errado, podem
   reagir à mudança de valor — testar em pelo menos 2 hosts CLAP reais antes de mesclar).
   Gate: novo teste de integração com IR longo (proposto em F-A1) + `clap-validator` +
   verificação manual em DAW.
2. **F-A3** (trivial, pode entrar na mesma PR de qualquer outra tarefa de teste).
3. **F-A2** — maior escopo, mexe em layout de memória interno do Lite. Seguir o mesmo rigor de
   gate do EPIC-2 original (goldens bit-exact, dashboard completo, teste de padding-zero
   dedicado). Não é urgente (Lite já está dentro do budget RT com folga de 96,1%) — é uma
   questão de qualidade de engenharia/consistência, não de correção funcional.
4. **F-A4** — sem ação; apenas manter no radar via o próprio dashboard (F-L4).

---

*Gerado pela auditoria `revisor-auditor`/`pesquisador-inovador` de 2026-07-17, reverificado em
2026-07-19. Nenhuma linha de código de produção foi alterada nesta reverificação (apenas
leitura, regeneração de `dsp_hotpath.asm` para inspeção, e este documento). `TODO-sprints.md`
será atualizado com o EPIC-6 quando solicitado.
solicitado.*
