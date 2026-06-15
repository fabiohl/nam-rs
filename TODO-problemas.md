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

---

## Sumário de severidade

| ID     | Achado                                                                                                                     | Severidade     | Eixo                 |
| ------ | -------------------------------------------------------------------------------------------------------------------------- |:--------------:| -------------------- |
| **P1** | WaveNet **"Lite" (CH=12)** diverge do C++ de referência (SNR ≈ **0,9 dB**) — arquitetura inteira fora de paridade          | 🔴 Alta        | Fidelidade           |
| **P2** | Família **WaveNet** tem fidelidade vs C++ muito inferior à LSTM/Linear (custo do FastMath: ESR ~0,3–1%)                    | 🟠 Média-Alta  | Fidelidade           |
| **P3** | **Gates de golden muito frouxos** em alguns cenários (SNR ≥ **7,0 / 8,5 dB**) — guardião fraco onde o produto é menos fiel | 🟠 Média       | Cobertura/Fidelidade |
| **P4** | WaveNet emite **saída não-nula no silêncio** (~3,6e-5; ≈ −89 dBFS); A2 emite **0 exato**                                   | 🟡 Média-Baixa | Correção/DSP         |
| **P5** | **Pico de latência** de bloco no WaveNet em silêncio (~**677 µs**) — possível penalidade de denormais                      | 🟡 Média-Baixa | RT/Performance       |
| **P6** | Telemetria de latência **quantizada em potências de 2** (65536/131072 ns) — leitura imprecisa                              | 🟢 Baixa       | Observabilidade      |
| **P7** | `MirroredBuffer` só tem implementação real no **Linux** (stub nas demais plataformas)                                      | 🟢 Baixa       | Portabilidade        |
| **P8** | **137 testes `ignored`** na suíte padrão; cross-validação **live** vs C++ só roda na suíte longa                           | 🟢 Informativo | Cobertura            |

---

## P1 — 🔴 WaveNet "Lite" (CH=12) não bate com o C++ de referência

**Evidência** (Wavenet)

- `tests/golden_vectors.rs:484`:
  `#[ignore = "known-divergent: WaveNet Lite (CH=12) exhibits numerical drift (SNR = 0.9 dB vs C++)"]`
- `tests-cargo.log` (Phase 3, `cpp_parity`): `SKIP: WaveNet Lite (CH=12) is known-divergent (T1.2) - skipping to avoid false gate` (v1 **e** v2).
- `tests/self_consistency.rs:26`: documentado como "Lite (CH=12, known-divergent output but engine [estável])".

**O que significa**
SNR ≈ **0,9 dB** vs a referência NeuralAmpModelerCore significa que a saída do nam-rs para
essa arquitetura é **quase totalmente descorrelacionada** do esperado — não é "um pouco
diferente", é **outro som**. O engine é determinístico e auto-consistente (não trava, não
gera NaN), mas **um usuário que carregue um modelo "Lite" ouvirá algo diferente do NAM
canônico**. Hoje isso está **mascarado por `#[ignore]`** para não derrubar o gate.

**Por que importa ao PO**
É o achado de maior impacto de produto: uma classe inteira de modelos (WaveNet Lite,
CH=12) está silenciosamente "errada" em relação à referência que o projeto promete
espelhar. Afeta credibilidade ("é um NAM fiel?").

**Sugestão de condução**
Investigar a causa (skill `pesquisador-inovador`/`debugger`): suspeitos prováveis no
caminho CH=12 — _bias-tuning_, ordem de interleaving 4-wide, ou a cadeia FastMath
(tanh/sigmoid Padé) amplificada nessa topologia. Decidir entre **corrigir** (reabilitar o
golden) ou **documentar formalmente** a limitação e, idealmente, **avisar o usuário** ao
carregar um modelo Lite.

---

## P2 — 🟠 Fidelidade da família WaveNet vs C++ é muito inferior à de LSTM/Linear

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

---

## P3 — 🟠 Gates de golden frouxos em alguns cenários (guardião fraco onde mais importa)

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

---

## P4 — 🟡 WaveNet não é "silencioso no silêncio"

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

1. **P1** (WaveNet Lite divergente) — investigar/decidir: é o maior risco de credibilidade.
2. **P2 + P3** (fidelidade WaveNet e gates frouxos) — definir a **política de fidelidade**
   do produto (aceitar FastMath? modo exato? apertar gates?).
3. **P4 + P5** (silêncio não-nulo + pico de latência) — provavelmente **mesma raiz**
   (bias/denormal); tratar em conjunto, com olho em RT-safety.
4. **P6, P7, P8** — observabilidade/portabilidade/cadência: acompanhar, baixa urgência.
