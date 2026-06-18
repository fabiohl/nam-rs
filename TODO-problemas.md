<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Relatório de Achados do nam-rs — Visão do PO (a partir dos testes)

> **Origem**: leitura crítica dos resultados da bateria completa (`tests-cargo.log` —
> suíte padrão + 🚦 GATE L2 da `tests-long.sh`, todas verdes) e dos logs por fase em
> `target/logs/`, agora que a infraestrutura de testes está estabilizada e confiável
> (Épicos 1–4). Os "guardiões" do nam-rs estão em pé; **este documento separa o que os
> guardiões apontam sobre o _produto_** (não sobre a infra de testes, já resolvida).
>
> **Natureza**: nenhum dos achados abaixo é uma falha de teste — a suíte está 100% verde.
> São **estranhezas e riscos de produto** que os testes _expõem como sinal_ e que merecem
> análise calma e condução oportuna pelo PO.
>
> **Macro-objetivo de referência**: "suíte perfeita, cobrindo todos os casos críticos,
> ignorando os irrelevantes". À luz disso, alguns achados são sobre **o nam-rs**; outros
> são sobre **onde os guardiões ainda estão fracos** (gates frouxos).
>
> **Documentos irmãos**: para **lacunas de funcionalidade** vs NeuralAmpModelerCore consulte
> **`TODO-features.md`** (achados "F"); para **otimização estrutural via internalização de
> dependências** consulte **`TODO-optimize.md`** (achados "O"). Em particular, **§P5/§P6**
> (picos de latência e telemetria quantizada) conectam-se a **`TODO-optimize.md §O2`**
> (clock TSC unificado).

---

## Sumário de severidade

| ID      | Achado                                                                                                                                                                                           | Severidade         | Eixo                  |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |:------------------:| --------------------- |
| **P1**  | WaveNet **"Lite" (CH=12)** diverge do C++ de referência (SNR ≈ **0,9 dB**) — arquitetura inteira fora de paridade                                                                                | 🔴 Alta            | Fidelidade            |
| **P2**  | Família **WaveNet** tem fidelidade vs C++ muito inferior à LSTM/Linear (custo do FastMath: ESR ~0,3–1%) — ✅ [RESOLVIDO] (T-HF6.6, nuke completa) | 🟠 Média-Alta      | Fidelidade            |
| **P3**  | **Gates de golden muito frouxos** em alguns cenários (SNR ≥ **7,0 / 8,5 dB**) — guardião fraco onde o produto é menos fiel — ✅ [RESOLVIDO] (T-HF6.6, thresholds SNR 85-105 dB) | 🟠 Média           | Cobertura/Fidelidade  |
| **P4**  | WaveNet emite **saída não-nula no silêncio** (~3,6e-5; ≈ −89 dBFS); A2 emite **0 exato**                                                                                                         | 🟡 Média-Baixa     | Correção/DSP          |
| **P5**  | **Pico de latência** de bloco no WaveNet em silêncio (~**677 µs**) — possível penalidade de denormais                                                                                            | 🟡 Média-Baixa     | RT/Performance        |
| **P6**  | Telemetria de latência **quantizada em potências de 2** (65536/131072 ns) — leitura imprecisa                                                                                                    | 🟢 Baixa           | Observabilidade       |
| **P7**  | `MirroredBuffer` só tem implementação real no **Linux** (stub nas demais plataformas)                                                                                                            | 🟢 Baixa           | Portabilidade         |
| **P8**  | **137 testes `ignored`** na suíte padrão; cross-validação **live** vs C++ só roda na suíte longa                                                                                                 | 🟢 Informativo     | Cobertura             |
| **P9**  | **A2** `set_max_buffer_size` cresce > 64 mas conv kernels têm scratch de stack fixo de 64 — `panic` (debug) / **UB** (release) com bloco > 64                                                    | 🟠 Média (latente) | RT-safety/Soundness   |
| **P10** | **Modo "baixa fidelidade" (padrão) sob júdice** — ganho real de performance/latência vs modo exato ainda não medido; A2 já entregou qualidade E eficiência, questionando a premissa do trade-off | 🟠 Estratégico     | Arquitetura/Qualidade |

---

## P1 — 🔴 WaveNet "Lite" (CH=12) não bate com o C++ de referência

**Evidência** (Wavenet)

- `tests/golden_vectors.rs:484`:
  `#[ignore = "known-divergent: WaveNet Lite (CH=12) exhibits numerical drift (SNR = 0.9 dB vs C++)"]`
- `tests-cargo.log` (Phase 3, `cpp_parity`): `SKIP: WaveNet Lite (CH=12) is known-divergent (T1.2) - skipping to avoid false gate` (v1 **e** v2).
- `tests/self_consistency.rs:26`: documentado como "Lite (CH=12, known-divergent output but engine [estável])".

**Verificação T-HF6.6 (jun/2026, pós-nuke lo-fi):** O caminho f32 exato (pesos f32 + tanh
poly, sem quantização) **não** resolveu P1. BossWN-lite manteve SNR=0.9 dB; v2 wavenet_lite
@ 44.1 kHz registrou SNR=-0.4 dB. A divergência do CH=12 **não é causada por quantização de
pesos nem por FastMath** — é um problema arquitetural/implementacional separado (suspeitos:
bias-tuning, interleaving 4-wide, ou caminho CH=12 no motor estático).

**O que significa**
SNR ≈ **0,9 dB** vs a referência NeuralAmpModelerCore significa que a saída do nam-rs para
essa arquitetura é **quase totalmente descorrelacionada** do esperado — não é "um pouco
diferente", é **outro som**. O engine é determinístico e auto-consistente (não trava, não
gera NaN), mas **um usuário que carregue um modelo "Lite" ouvirá algo diferente do NAM
canônico**. Hoje isso está **mascarado por `#[ignore]`** para não derrubar o gate.

**Nova manifestação (auditoria jun/2026 — S2).** O teste novo `tests/nondist_validation.rs`
expôs que um modelo **Lite real** (`EVH-5150-Lite.nam`) também **viola invariância de tamanho de
bloco**: processar o mesmo sinal em blocos de 1 vs 16 frames diverge (MSE ≈ 2,1e-2). Como o caminho
Lite é const-generic (intocado pela S2), isto é a **mesma raiz P1** se manifestando como divergência
**single-frame vs dual-frame tiling** no caminho **CH=12** — exatamente o suspeito apontado na
condução abaixo ("ordem de interleaving 4-wide"). No teste, a verificação foi rebaixada a **warning
visível** (`WARN [P1]`) para não derrubar a suíte por um problema já rastreado; restaurar o
`assert` rígido quando P1 for resolvido.

**Por que importa ao PO**
É o achado de maior impacto de produto: uma classe inteira de modelos (WaveNet Lite,
CH=12) está silenciosamente "errada" em relação à referência que o projeto promete
espelhar. Afeta credibilidade ("é um NAM fiel?").

**Sugestão de condução**
A hipótese de que o caminho f32 exato resolveria P1 foi **refutada** (T-HF6.6). A divergência
é arquitetural/implementacional no caminho CH=12. Suspeitos remanescentes: _bias-tuning_,
ordem de interleaving 4-wide, ou o próprio caminho `StaticModel` para CH=12. Decidir entre
**corrigir** (reabilitar o golden) ou **documentar formalmente** a limitação e, idealmente,
**avisar o usuário** ao carregar um modelo Lite.

---

## P2 — 🟠 Fidelidade da família WaveNet vs C++ é muito inferior à de LSTM/Linear — ✅ [DONE] (T-HF6.6, nuke completa)

> **Status final (jun/2026, T-HF6.6 — recalibração pós-nuke):** Com a eliminação completa do
> modo lo-fi (S-HF6), todos os modelos WaveNet **não-Lite** agora operam exclusivamente com pesos
> f32 + tanh poly, atingindo fidelidade comparável a LSTM/Linear:
>
> - **WaveNet A1 Standard**: SNR=123.4 dB, ESR=4.58e-13 (live v1); SNR=101.8 dB @ 192 kHz (v2)
> - **WaveNet Standard CH=16**: SNR=134.6 dB, ESR=3.45e-14 (live v1); SNR=123.0 dB (v2 worst)
> - **WaveNet Feather**: SNR=133.1 dB, ESR=4.92e-14; SNR=117.6 dB @ 192 kHz (v2 worst)
> - **WaveNet Nano**: SNR=132.0 dB, ESR=6.30e-14; SNR=114.6 dB @ 192 kHz (v2 worst)
>
> A assimetria de fidelidade entre WaveNet e LSTM/Linear está **eliminada** para todos os SKUs
> exceto Lite (CH=12, P1 permanece como achado arquitetural separado).
> Thresholds recalibrados para SNR 85-105 dB (ver `tests/common/validation.rs`).
> P10 (modo baixa fidelidade) também está resolvido: lo-fi já não existe.

**Evidência** (`tests-cargo.log`, Phase 3, baselines de ESR impressos):

- WaveNet **A1-Std**: ESR `6,23e-3` (≈0,6% erro de energia); **A2-Full** `3,34e-3`; **A2-Lite** `5,00e-3`.
- Cenários com **SNR 19,8 dB / ESR 1,04e-2** (≈1%) passando com folga (threshold ≥ 12 dB).
- Em contraste, **LSTM/Linear** atingem **67–91 dB** (ESR ~1e-7 a 1e-9 — praticamente bit-exato).

**O que significa**
A família WaveNet troca exatidão por velocidade via aproximações FastMath (documentado em
`docs/fastmath-approximations.md`). O resultado é um **erro de ~0,3–1%** vs o C++, enquanto
LSTM/Linear são essencialmente exatos. Não é bug — é **decisão de design** — mas é uma
**assimetria de fidelidade** entre arquiteturas.

**Por que importa ao PO**
Define a "barra de fidelidade" do produto. ~1% de erro de energia pode ser perceptível em
material crítico (transientes, high-gain). Vale uma decisão consciente: aceitar como está,
ou oferecer um **modo "exato"** (sem FastMath) opcional para quem prioriza fidelidade sobre
latência.

**Sugestão de condução**
Quantificar perceptualmente (MUSHRA/ESR já disponíveis) o impacto do FastMath na WaveNet e
registrar a política de fidelidade. Conectar com P1 (Lite é o caso extremo dessa família).

**📋 Parecer revisor-auditor (jun/2026) — planejado em `TODO-sprints.md` (Épico E-WN).** Decomposição
das fontes de drift (via `docs/fastmath-approximations.md` + código): a **quantização de pesos
BF16/F16** (~3,9e-3 por elemento, `src/math/common/ops.rs:16`) é a fonte **dominante**, **maior** que o
tanh Padé [5,4] (~2,3e-3, `src/math/activations/tanh/production.rs`). **Não existe modo "exato"** no
projeto. Recomendação: (1) **medir e isolar** cada fonte primeiro (quick-win, baixo risco — `S1.T1.4`);
(2) oferecer um **modo alta-fidelidade opcional** (pesos f32 + tanh exato, **off por padrão**, sem
afetar produção) — `S4.T4.1`; (3) **documentar a política** de fidelidade (`S4.T4.2`). **Constraint
para F1**: o motor genérico **não pode piorar** o ESR atual. Sprints: **S1** (medição), **S4**
(modo exato + doc, opcional/deferida).

**Nota do PO:** Incluir uma sprint ou apenas tarefa-tecnica só para fazer uma comparação rigorosa entre os
modos de alta fidelidade e baixa fidelidade(atual). Afinal qual o ganho real em performance, latência e
estabilidade que a baixa fidelidade traz? Porque se não houver uma justificativa muito forte, abandona-la!

**→ Aberto como P10** (`TODO-problemas.md` abaixo) — ver também `TODO-sprints.md §Frente P10`.

---

## P3 — 🟠 Gates de golden frouxos em alguns cenários (guardião fraco onde mais importa) — ✅ [DONE] (recalibrado T-HF6.6)

> **Status final (jun/2026, T-HF6.6 — recalibração pós-nuke f32):** Thresholds de golden
> **endurecidos** com base em medição real pós-nuke. Com o caminho f32 exato, o ESR dos modelos
> WaveNet não-Lite colapsou de ~6e-3 (lo-fi) para ~1e-13 (≈ melhoria de 10 ordens de grandeza).
> Os novos floors SNR 85-105 dB para WaveNet A1 Standard/Feather/Nano (com margem de 16-37 dB
> sobre o worst-case multi-SR) garantem que **qualquer regressão genuína será detectada**.
> Cenários de threshold baixo (nondist, estímulos difíceis) seguem seu ciclo de triagem
> documentado.
>
> A regra de ouro mantida: _"todo golden deve poder falhar"_.

**Evidência** (`tests-cargo.log`, Phase 3, goldens v2 multi-SR):

- Thresholds calibrados em **≥ 7,0 dB**, **≥ 8,5 dB**, **≥ 8,8 dB**, **≥ 11,6 dB** para
  vários cenários; SNRs medidos correspondentes de **12,7 / 15,8 / 16,2 / 19,3 dB**.

**O que significa**
SNR de 7–16 dB equivale a **~5–20% de erro** tolerado. Esses gates (calibrados, com
meta-teste anti-placebo) são **honestos** sobre a divergência real — mas tão baixos que
**uma regressão genuína poderia passar despercebida** exatamente nas configurações onde o
produto já é menos fiel (provavelmente estímulos difíceis e/ou taxas ≠ 48 kHz após
reamostragem).

**Por que importa ao PO**
"Guardião" fraco = risco de regressão silenciosa. Onde o nam-rs é menos fiel é justamente
onde a vigilância é menor.

**Sugestão de condução**
Triagem dos cenários de threshold baixo: (a) se a divergência é inerente ao estímulo/SR,
documentar; (b) se há espaço de melhoria no engine (resampler, FastMath em alta
frequência), abrir investigação; (c) avaliar gates perceptuais (MR-STFT/LUFS) como
complemento ao SNR cru nesses casos.

**📋 Parecer revisor-auditor (jun/2026) — planejado em `TODO-sprints.md` (Épico E-WN).** Os gates são
**honestos** (calibrados com medição + meta-teste anti-placebo, `tests/threshold_calibration.rs:199`),
mas a chegada de **novas geometrias** via F1 amplia a superfície de risco. Endurecimento **conduzido
por triagem** (`S3.T3.3`): manter onde a divergência é inerente (estímulo/SR/reamostragem) com
justificativa documentada; recalibrar com margem honesta onde houver regressão evitável do engine
(inclusive o dinâmico); avaliar gates perceptuais (MR-STFT/LUFS) como complemento. **Cautela do
auditor**: **não** apertar gate sem entender a origem da divergência — geraria flakiness de CI. Regra
de ouro mantida: _"todo golden deve poder falhar"_. Sprints: **S3** (T3.2 goldens de paridade C++ +
T3.3 endurecimento dos gates).

---

## P4 — 🟡 WaveNet não é "silencioso no silêncio" — ✅ [DONE]

> **Status (jun/2026):** **diagnóstico concluído** (S1.T1.3, commit `a7df957`: o resíduo de 3,58e-5
> origina-se 100% da propagação de `tanh(conv1d_bias)` — **fiel ao NAMCore C++** `dsp.h:67`).
> Mantido por design (não zerar à força); documentado.

**Evidência** (`target/logs/phase1-soak.log`):

- `--- WaveNet Silence Soak ---` → `Max output: 0.000035848654` (≈ **−89 dBFS**).
- `--- A2-Lite/A2-Full Silence Soak ---` → `Max output: 0` (**zero exato**).

**O que significa**
Com entrada de silêncio puro, a WaveNet produz um resíduo DC/bias minúsculo (~3,6e-5),
enquanto as A2 produzem zero absoluto. Inaudível em termos absolutos (−89 dBFS), mas
**revela um termo de bias/offset ou artefato de denormal** que não zera. Notar que esse
resíduo é ~8% do nível de saída de **ruído** desse mesmo modelo (Max ruído ≈ 4,3e-4).

**Por que importa ao PO**
Interage com **noise gate**, **true-bypass** e detecção de silêncio. Um piso de saída
não-nulo no silêncio pode atrapalhar a lógica de gate/economia de CPU e indica que o modelo
não converge exatamente a zero.

**Sugestão de condução**
Verificar se é o termo de _bias_ do head/dense ou flush de denormais (DAZ/FTZ). Decidir se
deve ser zerado no caminho de silêncio (consistência com A2) ou aceito/documentado.

**📋 Parecer revisor-auditor (jun/2026) — planejado em `TODO-sprints.md` (Épico E-WN).**
**Não é bug — é comportamento fiel ao NAMCore.** Os biases das camadas (conv/mixin/1x1/rechannel)
tornam `tanh(bias) ≠ 0`, e o próprio C++ documenta _"don't expect the model to be outputting zeroes
after this"_ (`tests/fixtures/NeuralAmpModelerCore/NAM/dsp.h:67`). A A2 só zera por usar **LeakyReLU(0)=0**

- pesos sintéticos — não é o caso da WaveNet A1 real. **Cautela crítica do auditor: NÃO zerar o
  silêncio à força** — isso **divergiria da bíblia** (C++) e quebraria a paridade. Encaminhamento:
  diagnosticar a origem do resíduo (bias propagado vs quantização vs denormal), **confirmar paridade**
  gerando o mesmo silêncio no `render` v0.5.3, e **documentar** que −89 dBFS é fiel. DAZ/FTZ já está
  ativo (`src/math/common/ops.rs:163`, reasserção em `src/clap/processor/mod.rs:268`); a interação com
  noise-gate/true-bypass é responsabilidade da camada de gate, não do modelo. Liga-se a **P5**
  (penalidade de denormal). Sprint: **S1.T1.3** (diagnóstico + política documentada, 🟢 baixo risco).

---

## P5 — 🟡 Pico de latência por bloco no WaveNet em silêncio (~677 µs)

**Evidências** (Wavenet)

- `tests/nam_infer_test.rs` (`test_denormal_stability_silence`, gate rebaixado a warning na
  T4.2.1): `WaveNet denormal OK — 4096 silence blocks, max_block_time=677μs ... exceeds 500μs — possible denormal penalty`.
- Benchmark (`tests-cargo.log`, Phase 6): `Long_Run_WaveNet/Long_WaveNet_Standard_CH16_4096samp` ≈ **5,94 ms**.

**O que significa?**
O tempo amortizado por bloco é confortável (5,94 ms para 4096 amostras vs ~85 ms de
orçamento a 48 kHz). Porém **um bloco individual chegou a 677 µs** num cenário de silêncio —
sintoma clássico de **penalidade de denormais** (valores subnormais desaceleram a FPU). Com
**buffers RT pequenos** e sem DAZ/FTZ garantido em todo o caminho, picos assim podem se
aproximar do orçamento e causar **xruns**.

**Por que importa ao PO**
RT-safety/robustez ao vivo. O gate virou _warning_ (correto, para não ser flaky), então a
condição **não falha mais o teste** — mas o fenômeno de produto permanece e deixou de ser
vigiado por um gate rígido.

**Sugestão de condução**
Confirmar DAZ/FTZ ativos em todo o hot-path da WaveNet (há `test_set_daz_ftz`); investigar
o pico em silêncio (ligado a P4). Considerar um gate de latência **dedicado e robusto**
(percentil sob carga controlada) em vez do _wall-clock_ ingênuo removido.

---

## P6 — 🟢 Telemetria de latência quantizada em potências de 2

**Evidência** (`target/logs/phase1-soak.log`): `Telemetry max latency: 65536 ns` e
`131072 ns` (= 2¹⁶ e 2¹⁷) — valores de **fronteira de bucket**, não medidas reais.

**O que significa**
O histograma de latência reporta apenas buckets log2 grossos. A "latência máxima" exibida
pode **super/subestimar** o valor real, prejudicando o diagnóstico de picos (justamente o
que P5 precisa medir bem).

**Sugestão de condução**
Avaliar buckets mais finos ou registrar o `max` exato em paralelo ao histograma.

---

## P7 — 🟢 `MirroredBuffer` é Linux-only (stub nas demais plataformas)

**Evidência**: `src/dsp/mirror_buf/alloc.rs:83` — "memfd on Linux, **stub fallback on
other platforms**".

**O que significa**
O ring-buffer _zero-copy_ (espelhamento de páginas) só tem implementação real no Linux.
Provavelmente intencional (foco PipeWire/Linux), mas significa que **portar para
macOS/Windows exige trabalho** e que builds não-Linux não exercitam esse caminho.

**Sugestão de condução**
Confirmar se cross-plataforma é objetivo. Se sim, planejar a implementação real (mach
`vm_remap` no macOS, `MapViewOfFileEx`/placeholder no Windows) e cobertura de teste.

**Nota do PO**
Croos-platform não é escopo. O NAM-rs é opinado a explorar ao máximo as potencialidades do Linux e Pipewire.

---

## P8 — 🟢 Cobertura: 137 testes `ignored` na suíte padrão; C++ "live" só na longa

**Evidência**: 137 linhas `ignored` na execução padrão (`tests-cargo.log`); as 22
`live_cross_validation_*` de `cpp_parity` rodam **apenas** na suíte longa.

**O que significa**
A suíte padrão (rápida, rodada a cada commit) valida fidelidade contra **goldens
congelados** — nunca recompara com o C++ ao vivo. A cross-validação real vs C++ só ocorre
na `tests-long.sh` (38 min). A maioria dos 137 _ignored_ é legítima (soak, proptest pesado,
multi-SR), mas vale o PO ter ciência de que **o guardião do dia-a-dia confia nos goldens**,
não na referência viva.

**Sugestão de condução**
Garantir cadência regular da suíte longa (ela é a única que recomputa a paridade C++) e
confirmar que os goldens congelados estão atrelados a um commit fixo e auditável do
NeuralAmpModelerCore (há indícios disso — commit pinado `9c7b185`).

**Nota do PO**
Sim, a suite longa é rodada frequentemente. Quanto aos goldens congelados, favor confirmar isto.

---

## P9 — 🟠 A2: `set_max_buffer_size` cresce além de 64, mas os conv kernels têm scratch fixo de 64 (potencial UB em release)

**Origem**: auditoria da S2 (jun/2026), exposto pelo teste novo `tests/nondist_validation.rs`.
**Evidências**

- `src/models/a2/conv1d_ch8.rs:291-293` e `:387-389`: `let max_frames = 64;` +
  `debug_assert!(num_frames <= max_frames);` + **scratch de stack `let mut z_buf = [0.0f32; 64 * 8];`**.
- `src/models/a2/model/mod.rs:185-189`: `set_max_buffer_size` **cresce** `max_buffer_size` (e realoca
  buffers de modelo) para valores > 64; `model_test.rs:120-124` confirma crescimento até 256.
- `src/clap/processor/events.rs:173`: produção chama `model.set_max_buffer_size(max_frames_count)` na
  ativação; `src/dsp/pipeline/stages/inference.rs:118` chama `model.process(model_in_l, …)` com o
  **bloco inteiro** (sem rechunk a ≤64).

**O que significa**
A A2 **anuncia** suporte a blocos > 64 (via `set_max_buffer_size`), mas os kernels `conv1d_ch8`
(e provavelmente `conv1d_ch3`) usam um buffer de stack **fixo em 64 frames**. Com `num_frames > 64`:
em **debug** dá `panic` (o `debug_assert`); em **release** o `debug_assert` some e o kernel escreve
`num_frames*8 > 512` elementos num array de 512 → **estouro de buffer de stack (UB)**. É alcançável em
produção **se** um host negociar bloco > 64 com um modelo A2. Hoje o pico está em parte contido porque
o WaveNet A1 re-chunka internamente a 64 e a maioria dos hosts usa blocos pequenos — mas o contrato A2
é **inconsistente** (cresce o que não pode crescer).

**Por que importa ao PO**
RT-safety/soundness. Não é regressão da S2 (a A2 não foi tocada); é latente e foi **descoberto** pela
nova cobertura. Risco real de UB sob host com bloco grande + modelo A2.

**Sugestão de condução**
Decidir entre: (a) **chunkar internamente** a A2 em sub-blocos de ≤64 no `process` (espelha o WaveNet
A1; corrige a raiz); ou (b) tornar `set_max_buffer_size` **honesto** — clampar o teto efetivo da A2 a
64 e documentar; ou (c) generalizar os kernels `conv1d_ch{3,8}` para scratch dimensionado no load
(heap, alinhado) — sinergia com o motor A2 geral (`TODO-features.md §F3`). Enquanto não resolvido, o
teste `nondist_validation` limita a varredura de tamanho de bloco a 64 (contrato universal seguro).

---

## P10 — 🟠 Modo "baixa fidelidade" (padrão) sob júdice — ganho real não mensurado

**Origem**: observação estratégica do PO (jun/2026) durante o fechamento do Épico E-WN.

**Contexto**
O Épico E-WN implementou dois modos para WaveNet:

- **Padrão / "baixa fidelidade"**: pesos `u16` (BF16/F16) + tanh Padé [5,4]; ESR ~3e-3..1e-2 vs C++.
- **Alta fidelidade** (`--features high-fidelity`): pesos `f32` + `f32::tanh` exato; ESR < 1e-5,
  comparable ao LSTM/Linear. Implementado em `64e2c7f`.

**A premissa do trade-off nunca foi medida com dados reais.**
O modo padrão foi herdado como decisão histórica ("menor latência, menor memória"), mas:

1. A **magnitude real** do ganho de performance em hardware x86-64-v3 moderno (com AVX2/FMA
   disponíveis) nunca foi quantificada por benchmark reprodutível.
2. A arquitetura **A2** já entregou **qualidade superior E melhor eficiência** simultaneamente
   (via LeakyReLU + caminhos SIMD otimizados) — o que questiona a premissa de que qualidade
   e performance são necessariamente contrapostos no WaveNet A1.
3. O Padé [5,4] usa `_mm256_div_ps` (latência 10-14 ciclos em Ice Lake/Zen 3); `f32::tanh`
   usa lib matemática (costuma ser mais lento, mas o impacto real no throughput do _sistema_
   — que é dominado por memória/conv — é desconhecido). O modo f32 elimina a decode u16→f32
   por amostra, o que pode **compensar** parcialmente.
4. O modo padrão introduz ~1% de erro de energia **sem justificativa quantificada**. Se o
   ganho de performance for marginal (ex: <5%), o custo de fidelidade pode não se pagar.

### **O que precisamos medir: (critério de decisão)**

| Métrica                         | Pergunta                                                                       |
| ------------------------------- | ------------------------------------------------------------------------------ |
| Latência de inferência          | Quanto mais rápido é o modo padrão vs alta-fidelidade? (WaveNet A1 Std, bs=64) |
| Uso de memória                  | Quanto maior é o footprint de memória no modo f32?                             |
| Throughput (frames/s)           | O ganho é significativo sob carga contínua (soak 5M frames)?                   |
| ESR com pesos reais             | Com modelos reais (não sintéticos 0.01), qual o ESR real de cada modo?         |
| Perceptual (MUSHRA/AB informal) | A diferença de ~1% de energia é perceptível em material real?                  |

**Hipótese do PO**
O modo padrão pode não ter justificativa forte o suficiente. Se confirmado, o **caminho de
menor resistência é abandonar o modo padrão quantizado e adotar f32 + tanh exato como único
caminho**, eliminando toda uma classe de problemas de fidelidade (P2, fragmentos de P1) e
simplificando o código (remove dualidade u16/f32, feature flag, bias-tune, etc.).

**Por que importa ao PO**
Simplificação radical do produto + resolução de P2 pela raiz + aproximação ao padrão de
fidelidade do LSTM/Linear. Conexão direta com P1 (Lite): o modo exato pode (ou não) resolver
a divergência CH=12, a ser verificado.

**Sugestão de condução**
Abrir frente técnica dedicada (`TODO-sprints.md §Frente P10`):

1. Benchmark criterion: `bench_wavenet_standard_default` vs `bench_wavenet_standard_hifi`
   em blocos de 1, 16, 64 frames — registrar throughput e latência de pico.
2. Medição de ESR com pesos reais (usar `wavenet_official.nam` e modelos nondist).
3. Perfil de memória: `AlignedVec<f32>` vs `AlignedVec<u16>` por modelo.
4. Avaliação perceptual informal (AB em material high-gain).
5. Decisão: manter dualidade / promover hi-fi a padrão / abandonar lo-fi.

**Nota sobre A2**: a A2 já usa f32 nativo internamente (sem quantização de pesos, LeakyReLU
exata) e é simultaneamente mais eficiente e mais fiel que WaveNet A1 padrão. Isso reforça
empiricamente que qualidade e eficiência **não** são mutuamente exclusivos no contexto de
inferência neural com SIMD moderno.

---

## Não-achados (verificados e descartados)

- **SIGSEGV em release (Mono CLAP)**: era **UB de código de teste** (`transmute` de `i32`
  de stack para `HostSharedHandle` em `src/clap/gui/ui/test.rs`), **não** do produto.
  Corrigido (T4.1.1). _Lição:_ rodar testes em **release** tem valor — pegou UB que o debug
  mascarava.
- **`mock_a2.nam` "fails to build"**: é _fixture_ de **teste negativo** (exercita rejeição
  de build), não bug do loader.
- **Vazamento de memória / drift nos soaks**: RSS estável (ex.: A2-Lite 6.742.016 →
  6.742.016 bytes em 5M frames); sem vazamento. ✅

---

## Recomendação de priorização ao PO

> **Atualização (jun/2026, pós-T-HF6.6 — recalibração pós-nuke):**
> P2 e P3 resolvidos definitivamente (nuke completa, thresholds endurecidos SNR 85-105 dB).
> P10 resolvido pela raiz (lo-fi eliminado). P1 confirmado como arquitetural (não quantização).

1. **P10/P2/P3** — ✅ **Resolvidos (T-HF6.6)**. Lo-fi eliminado; thresholds endurecidos SNR 85-105 dB.
2. **P1** (WaveNet Lite divergente) — Confirmado como arquitetural (não quantização). f32 path
   não resolveu. Investigar causa no caminho CH=12 (bias-tuning, interleaving, etc.).
3. **P9** (A2 UB blocos >64) — corrigir antes de publicar suporte a blocos grandes;
   opções documentadas em P9.
4. **P4 + P5** (silêncio não-nulo + pico de latência) — P4 documentado/fiel C++; P5 pendente
   de gate de latência percentil.
5. **P6, P7, P8** — observabilidade/portabilidade/cadência: acompanhar, baixa urgência.
