<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-findings — Auditoria de Caça a Bugs (análise forense de `testes.log`)

> **Origem:** skill `revisor-auditor`, role _"Ace of bug hunting"_, com foco exclusivo na
> análise meticulosa de `testes.log` (3686 linhas) "atrás de insights ocultos nas entrelinhas".
> **Data:** 2026-06-26.
> **Estado de superfície dos testes:** ✅ tudo verde — `0 failed` em todas as suítes
> (quick: 948+ ok; long: 6/6 fases PASSED; clap-validator: 19/19, 2 skipped).
> **Estado nas entrelinhas:** a rede de segurança tem furos estruturais e o instrumento de
> diagnóstico de paridade mais crítico fica **ilegível justamente sob falha**.

## Metodologia e fontes

- Leitura integral de `testes.log` (suíte `utils/tests-quick.sh` + `utils/tests-long.sh`).
- Correlação cruzada com o código-fonte:
  - `tests/cpp_parity.rs` (orquestração de cross-validation C++).
  - `tests/common/validation.rs` (`report_dsp_fidelity_impl`, gates e thresholds).
  - `src/clap/extensions/state.rs` (higiene de log do plugin).
  - `docs/perceptual_validation.md` (intenção documentada: ESR como gate primário; LUFS diagnóstico).
- Toda evidência abaixo cita `arquivo:linha` para rastreabilidade.

---

## F-1 · 🔴 CRÍTICO — Diagnóstico de paridade C++ ilegível sob falha (saída intercalada byte-a-byte)

### F-1 · Sintoma (evidência no log)

- `testes.log:1561` — `Loading model [Loading model [Loading model [/…/BossLSTM-2x8.nam/…/BossLSTM-1x16.nam/…/linear_test.nam]`
  → **três subprocessos C++ escrevendo simultaneamente** no mesmo descritor de arquivo, sem newline entre eles.
- `testes.log:1970` — `Wrote   MR-STFT = 6.2966e-1      (relative)` seguido de `testes.log:1977` — `240001 samples to …_88200_live.wav`
  → uma linha `println!` do relatório Rust **rasgou ao meio** a linha `Wrote N samples` do processo filho (tearing em nível de **byte**, não de linha).
- Em **todo** o trecho `testes.log:1569-3321`, cada rodapé (`MR-STFT … LUFS … Fidelity Margin … Samples`)
  aparece **destacado/órfão** do seu cabeçalho `[NeuralAmpModelerCore × NAM-rs — … MSE/SNR …]`.

### F-1 · Causa-raiz (confirmada no código)

1. **stdout do filho é herdado.** `tests/cpp_parity.rs:267-271`:

   ```rust
   let status = Command::new(&bin)
       .arg(model_path…).arg(stress_wav…).arg(output_wav…)
       .status();   // .status() herda stdin/stdout/stderr do processo de teste
   ```

   O render tool C++ (`Loading model…`, `Model loaded successfully`, `Wrote N samples…`)
   escreve direto no fd compartilhado, **sem** sincronizar com o lock do `println!` do Rust
   nem com outros filhos concorrentes.

2. **Harness multi-thread.** O test runner do Rust roda em paralelo por padrão. A long-suite
   (`testes.log:3621`) passa `--test-threads=1` para `pipeline_soak` mas **não** para `cpp_parity`.

3. **Relatório fragmentado.** `tests/common/validation.rs:180-250` emite **~20 chamadas `println!`
   separadas** por modelo (cabeçalho MSE/SNR e rodapé MR-STFT/LUFS/Fidelity em chamadas distintas),
   permitindo que linhas de outras threads/filhos se intercalem no meio do bloco.

### F-1 · Impacto

Quando um `assert!` de paridade finalmente disparar (`tests/common/validation.rs:252-265`,
marcadores `✗` em `validation.rs:184/190`), a mensagem de pânico **e** o bloco de métricas do
modelo culpado estarão intercalados com a saída de N outros modelos e N subprocessos C++.
**Triagem fica quase impossível**: não se identifica qual modelo falhou nem se leem suas métricas.
É a pior falha possível num instrumento de medição — ele mente exatamente quando é mais necessário.

### F-1 · Proposta de solução

- **(A) Capturar a saída do filho** em vez de herdá-la — trocar `.status()` por `.output()`,
  guardar `stdout`/`stderr` e **só imprimir em caso de `!success`** (ou anexar ao painço do `assert`):

  ```rust
  let out = Command::new(&bin).args([…]).output();
  match out {
      Ok(o) if o.status.success() => { /* silêncio em sucesso */ }
      Ok(o) => { eprintln!("SKIP/FAIL {label}: rc={:?}\n{}", o.status.code(),
                           String::from_utf8_lossy(&o.stderr)); return; }
      Err(e) => { eprintln!("SKIP {label}: {e}"); return; }
  }
  ```

- **(B) Emitir o relatório atomicamente** — montar o bloco inteiro em uma única `String`
  e emitir com **um** `print!`, protegido por um `static REPORT_LOCK: Mutex<()>`
  em `validation.rs` (serializa a escrita entre threads).

- **(C) Defesa em profundidade** — fixar `cpp_parity` em `--test-threads=1` no
  `utils/tests-long.sh:3621` (como já é feito para `pipeline_soak`).

Recomenda-se aplicar **(A) + (B)** (causa-raiz) e usar **(C)** como cinto-e-suspensório.

### F-1 · Critérios de aceite

- Rodar `cargo test --release --test cpp_parity -- --ignored --nocapture` produz blocos de
  relatório **contíguos** (cabeçalho+rodapé juntos), 1 por modelo, sem linhas de filho intercaladas.
- Forçar uma falha sintética (ex.: corromper 1 amostra) ⇒ o pânico identifica **inequivocamente**
  o modelo e suas métricas.

### F-1 · Esforço estimado

Baixo (≈ 0,5–1 dia). Alto ROI: desbloqueia todo debug futuro de paridade.

---

## F-2 · 🟠 ALTO — Ponto cego de CI: gates "hard" se auto-afrouxam por sample rate e o gate perceptual não bloqueia

### F-2 · Sintoma (evidência no log)

- **27** estouros do soft-gate MR-STFT, chegando a **1.9587e0** (`testes.log:2812`) — 195% de erro
  espectral relativo — todos rotulados _"informational, not a hard assertion"_.
- **9** `Fidelity Margin` abaixo do alvo (`> 8.0 dB`), marcados com `?`, **incluindo dois NEGATIVOS**:
  - `testes.log:2189` → `-0.8 dB` @ 88200 Hz;
  - `testes.log:2381` → `-0.9 dB` @ 96000 Hz.
  - Margem negativa = a saída NAM-rs diverge da referência C++ **mais** do que uma âncora
    deliberadamente degradada (low-pass 3,5 kHz; `validation.rs:174-178`).
- **Não é só taxa exótica**: há sub-alvo nas taxas **nativas** — `0.5 dB @ 44100` (`testes.log:1809`)
  e `0.4 dB @ 48000` (`testes.log:1953`), com MR-STFT de 0.82 e 0.87.
- Correção bruta no pior caso: LSTM 1×16 @ 192 kHz com **ESR = 1.42e-1** (`testes.log:2519`)
  — ~23× o baseline A1-Std (6.23e-3) — ainda assim **passa**.

### F-2 · Causa-raiz (confirmada no código)

1. **Os três gates hard (MSE/SNR/ESR) são afrouxados em função do sample rate**
   (`tests/cpp_parity.rs:320-352`):
   - LSTM v2: `snr_relaxation = (3.5·sr_ratio).min(10.0)` ⇒ a 192 kHz, **−10 dB de SNR**,
     **×10 no `mse_limit`** e **×10 no `max_esr`** (`cpp_parity.rs:331-332`), com piso `min_snr_db.max(7.0)`.
   - WaveNet v2: até **−4 dB / ×2.5**; resampling adiciona **−1.5 dB / ×1.5** (`cpp_parity.rs:345-352`).
   - Os `assert!` existem e são reais (`validation.rs:252-265`), mas o teto foi elevado **acima** do
     valor observado: a 192 kHz o ESR de LSTM passou com `esr_linear = 1.42e-1`, logo `max_esr > 1.4e-1`.
2. **O gate perceptual MR-STFT é não-bloqueante por design** (`validation.rs:216-225`):
   imprime aviso, mas **não** há `assert!`. O `Fidelity Margin` idem (marca `?`, `validation.rs:240-248`).

> **Nota de justiça:** esse afrouxamento é **deliberado e documentado** —
> `docs/perceptual_validation.md:78-80` afirma que o gate é "deliberately loose to tolerate variation
> in less-covered sample rates (44.1k, 192k)". O achado **não** é "bug acidental"; é um **risco de
> cobertura conhecido**: a barra de aprovação se auto-afrouxa exatamente onde a implementação mais
> diverge, e a métrica que capturaria isso (MR-STFT) não reprova.

### F-2 · Impacto

Uma regressão **real** de fidelidade espectral (ex.: bug num kernel SIMD de upsample, num activation
de alta frequência, ou no resampler) em taxas ≠ 48 kHz seria **absorvida** pelo afrouxamento dos gates
hard e **silenciada** pelo gate perceptual não-bloqueante. O CI permaneceria verde. O caso mais
preocupante são os **sub-alvo nas taxas nativas 44.1/48 kHz**, onde não há desculpa de "fora da
distribuição de treino".

### F-2 · Proposta de solução

- **Promover MR-STFT a gate hard com limiar calibrado por modelo.** A infraestrutura **já existe**:
  - `tests/threshold_calibration.rs` (testes `test_all_golden_models_have_calibrated_thresholds`,
    `test_all_thresholds_anti_placebo`, `test_all_calibrated_entries_have_measurement_comments`
    — `testes.log:1493-1495`);
  - `validation.rs:get_calibrated_threshold()` já segue o padrão "SNR_medido − margem / ESR_medido × fator".
  - Estender a tupla calibrada para incluir `mrstft_max` por modelo×taxa.
- **No mínimo, tornar MR-STFT bloqueante em 44.1/48 kHz** (taxas nativas), mantendo soft-gate em
  88.2/96/192 kHz enquanto a divergência de LSTM por sample rate for caracterizada.
- **Investigar especificamente** o WaveNet/o caso `0.4 dB @ 48000` (`testes.log:1953`, MR-STFT 0.87):
  divergência espectral alta com MSE baixo em taxa nativa merece RCA dedicada (suspeitos:
  conteúdo de alta frequência/borda de banda, dither, ou caminho de resample no harness).
- **Limitar o piso de relaxamento**: hoje LSTM pode cair a `min_snr_db = 7.0` e `max_esr` ~1e-1.
  Definir um teto absoluto de relaxamento (ex.: ESR nunca acima do baseline A1-Std 6.23e-3) para
  que "passar" continue significando "paridade", e não apenas "não totalmente quebrado".

### F-2 · Resolução (2026-06-27, Tarefa 3.3)

**RCA concluída.** O caso `MR-STFT 0.87 / margin 0.4 dB @ 48 kHz` foi integralmente investigado:

1. **Causa-raiz identificada:** _Recurrent state quantization drift_ — acúmulo inerente de erro de
   quantização f16c no estado de célula do LSTM (`cₜ = fₜ·cₜ₋₁ + iₜ·gₜ`). A cada iteração
   (240k samples no v2 @ 48k), o erro de quantização dos pesos f16c (~3.3 dígitos decimais
   significativos) se acumula no estado recorrente. O forget gate fₜ < 1 limita o acúmulo a um
   steady-state, não deixando o ESR crescer livremente.

2. **Evidência experimental:**
   - v1 (2048 samples, 42.7ms) @ 48 kHz: ESR=1.04e-2, MR-STFT=0.098 ✅
   - v2 (240k samples, 5s) @ 48 kHz: ESR=2.61e-2, MR-STFT=0.87 ❌
   - Multi-SR v2: ESR cresce de 2.39e-2 (44.1k) a 1.42e-1 (192k), proporcional ao número de steps
   - ASR = −68.8 dB (aliasing insignificante — não explica o ESR de 2.61e-2)

3. **Hipóteses refutadas:** borda de banda (sem DC-block no pipeline), dither de denormal
   (±1e-11 simétrico, −220 dBFS), aliasing (ASR=−68.8 dB), resample do harness (48 kHz bypassa
   resampler). **Nenhuma dessas hipóteses explica o ESR=2.61e-2.**

4. **Divergência vs NAMCore × vs oráculo f64:** O ESR=2.61e-2 é a diferença _interop_ (nam-rs vs
   NAMCore). Ambos compartilham o piso numérico f16c+f32 — o oráculo f64 mostra ESR≈1.0 (0 dB)
   desde os primeiros 512 samples. Ou seja: **"ambos divergem do ideal"**, não é uma regression
   exclusiva do nam-rs. A limitação é inerente ao formato .nam com pesos f16c.

5. **Ações tomadas:** O `ABSOLUTE_ESR_CAP` de 6.23e-3 (A1-Std baseline) introduzido em T3.2
   funciona como sentinela: força triagem consciente quando algum modelo excede o baseline
   absoluto. Os thresholds calibrados por modelo (6.5e-2 para LSTM 1×16) refletem a realidade
   física do formato. **Não há correção possível sem alterar o formato do modelo (quebrando
   interoperabilidade com NAMCore).**

6. **Mitigação de longo prazo encaminhada ao Épico E4 (S5):** acúmulo compensado (Kahan) no
   head do LSTM; oversampling do estado recorrente como modo HQ opcional.

### F-2 · Critérios de aceite

- Existe `mrstft_max` calibrado e **asserido** para todos os modelos golden em 44.1/48 kHz.
- O teste anti-placebo cobre o novo gate (impede limiar frouxo demais).
- Uma regressão sintética que eleve MR-STFT acima do baseline calibrado **reprova** o CI.

### F-2 · Esforço estimado

Médio (≈ 2–3 dias), majoritariamente calibração + RCA do caso 48 kHz.

---

## F-3 · 🟡 MÉDIO — Processo: o loop rápido de dev dá falsa confiança (paridade quase toda ignorada)

### F-3 · Sintoma (evidência no log)

No `utils/tests-quick.sh` (ciclo diário, ~3 min) a validação de paridade está majoritariamente `#[ignore]`:

- `cpp_parity`: **32 de 33** ignorados — só `wavenet_nano` roda (`testes.log:1109-1144`),
  ironicamente o caso mais benigno.
- `cabsim_cpp_parity`: **3/3** ignorados (`testes.log:1031-1033`).
- `golden_vectors` v2: **15** ignorados (`testes.log:1184-1198`).
- Fuzz de parsers (`proptest_parsers`): **13/14** ignorados (`testes.log:1405-1417`).
- `proptest_math` (precisão SIMD): **3/4** ignorados (`testes.log:1395-1397`).
- Soak: **19/20** ignorados (`testes.log:1458-1476`); parity bf16 LSTM ignorados (`testes.log:1300-1302`).

A cobertura **existe** na long-suite (`testes.log:3620-3623`, Fase 3, 214 s), porém uma regressão de
paridade C++ em taxas ≠ 48 kHz fica **invisível** no ciclo curto.

### F-3 · Impacto

Interage com F-1 e F-2: o único teste de paridade que roda no loop rápido (`wavenet_nano`) é o que
**menos** estressa o sistema. O desenvolvedor recebe "tudo verde" em 3 min sem nenhuma garantia de
paridade hard nas condições onde os bugs realmente aparecem.

### F-3 · Proposta de solução

- Adicionar ao `tests-quick.sh` um **subconjunto representativo de paridade hard** rodando em
  release: 1 LSTM + 1 WaveNet (CH16) + 1 A2 @ 48 kHz, com o gate MR-STFT de F-2 ativo.
- Manter o orçamento de tempo: usar sinal curto (2048 amostras, como o `wavenet_nano` atual) e
  pré-condicionar o binário do render tool C++ via cache (já há `BUILD_LOCK`, `cpp_parity.rs:92-93`).

### F-3 · Critérios de aceite

- `tests-quick.sh` executa ≥ 3 cross-validations hard @ 48 kHz em < +30 s de orçamento total.

### F-3 · Esforço estimado

Baixo (≈ 0,5 dia).

---

## F-4 · 🟢 BAIXO — Clareza do relatório: o gate **primário** (ESR) não exibe veredito ✓/✗

### F-4 · Sintoma / Causa-raiz

`tests/common/validation.rs` imprime `✓/✗` para **MSE** (`:184`) e **SNR** (`:190`), mas a linha de
**ESR** (`:205-214`) é impressa **sem** marcador de aprovação/reprovação — embora
`docs/perceptual_validation.md:82-89` (T16.4) declare **ESR como o threshold primário, scale-robust**,
e MSE/SNR como secundários/legados (MSE explicitamente "not robust to scale mismatch").

### F-4 · Impacto

O olho de quem lê o log é atraído para os gates secundários (que têm ✓), enquanto a métrica que
realmente decide (ESR, asserida em `validation.rs:260-265`) não mostra veredito visual. Em conjunto
com F-1, a "linguagem de verdes/vermelhos" do relatório fica incompleta e potencialmente enganosa.

### F-4 · Proposta de solução

Adicionar o marcador na linha de ESR, espelhando MSE/SNR:

```rust
println!("  ESR     = {esr_linear:.2e}  ({esr_db:.1} dB)  (threshold < {limit:.1e})  {}",
         if max_esr.map_or(true, |l| esr_linear < l) { "✓" } else { "✗" });
```

(e destacar visualmente que ESR é o gate primário).

### F-4 · Critérios de aceite

- Todo bloco de relatório mostra ✓/✗ explícito para ESR alinhado ao `assert!` real.

### F-4 · Esforço estimado

Trivial (< 0,5 dia).

---

## Diligência: hipóteses investigadas e **refutadas** (não são bugs)

Registradas para evitar retrabalho e documentar a robustez observada.

### D-1 · `[CLAP_PLUGIN_ERROR] Empty state buffer` é comportamento **correto e esperado**

`testes.log:3513`. Verificado em `src/clap/extensions/state.rs:90-95`: o plugin loga via
**`HostLog` em severidade `Debug`** (rota correta) e retorna `Err(PluginError::Message(...))`.
A linha `[CLAP_PLUGIN_ERROR]` é o **clap-validator** reportando o erro **esperado** do teste
`state-invalid`, que **PASSOU** (`testes.log:3580-3581`). Bônus: `src/clap/` tem **zero**
`println!/eprintln!` — higiene de log exemplar (nenhuma I/O de console na thread de áudio,
aderente à regra RT-safety do projeto).

### D-2 · "GOLDEN DEFECT (T2.5 lesson)" nos modelos dinâmicos é **benigno** (não indica sub-teste)

11 ocorrências (`testes.log:1877, 1958, 2027, 2144, 2250, 2369, 2440, 2531, 2673, 2798, 2885`),
todas `WaveNetDyn Free-Shape` / `LSTM-Dyn`, com LUFS de referência implausível (−55 a −71 LUFS).
**Porém:**

- O LUFS é **diagnóstico, não-CI-gate** por design (`docs/perceptual_validation.md:56`), e o gate é
  explicitamente **opt-out** para goldens de convolução-IR (`validation.rs:266-282`,
  `check_lufs_gate=false`).
- Nesses mesmos modelos, os gates **primários** passam com folga enorme — ex.: WaveNetDyn com
  `MSE 1.55e-19`, `SNR 123.9 dB`, `ESR 4.09e-13` (`testes.log:2666-2671`). Como o ESR é
  **scale-invariant**, referências de baixa loudness **não** enfraquecem a cobertura (na verdade
  aumentam a sensibilidade).
- A label "✗ — GOLDEN DEFECT" (`validation.rs:234`) descreve a lição **T2.5** (um caso real de
  LUFS −67 que escapou no passado e motivou criar o gate) — é design **maduro**, não defeito ativo.

**Único nit residual (informativo, sem ação obrigatória):** a string alarmante "GOLDEN DEFECT" aparece
em **log de sucesso**, o que confunde. Opcional: reduzir o ruído para `ⓘ` quando `check_lufs_gate=false`
(o caminho já existe em `validation.rs:276-281`), evitando "vermelho cosmético" em runs verdes.

---

## Observações menores (radar — sem severidade atribuída)

- **Latência de hot-swap de modelo:** `gc_stress_1000_swaps` = 106,9 s (`testes.log:3673`)
  ⇒ ~107 ms/swap (inclui load). Off-RT, mas se a troca é acionável pelo usuário, vale confirmar o
  orçamento de UX (perceptível acima de ~100 ms).
- **`resampler_heap_audit` lento:** 83,6 s (`testes.log:3667`) — alto para um heap-audit;
  possível excesso de cenários ou caminho lento; vale um perfil rápido.
- **Positivo a preservar:** guarda de determinismo SHA256 fase3 == fase5 (`testes.log:3510-3512`)
  impede TOCTOU no pipeline de validação CLAP. Manter.
- **Positivo:** `param-fuzz-basic` (50 conjuntos, 0 NaN/Inf, `testes.log:3559-3561`) e
  `process-audio-out-of-place-basic` (0 subnormais, `testes.log:3566-3568`) passam — boa robustez RT.

---

## Parte II — Pesquisa Inovadora: Precisão & Qualidade Sonora

> **Origem:** skill `pesquisador-inovador`. Complementa a auditoria de bug-hunting (Parte I) com
> pesquisa científica de meios para **elevar os índices de precisão e qualidade sonora** do nam-rs e,
> sobretudo, **instrumentar a medição correta** desses índices (qualidade, performance, precisão).
> **Data:** 2026-06-26.
> **Método:** mapeamento profundo do código (math/dsp/models/testing) cruzado com literatura DAFx/JAES/IEEE
> e normas ITU-R/AES/EBU (ver §Referências ao final desta parte).
> **Diagnóstico-síntese (entre as linhas dos dados):** a fidelidade _no domínio do tempo_ vs C++ é
> excelente (WaveNet f32-nativo ESR ~1e-13; LSTM/Linear bit-exatos). Mas **três eixos de qualidade
> sonora ficam fora do tempo-domínio e hoje são invisíveis ao QA**: (1) **aliasing** gerado pelas
> não-linearidades; (2) **fidelidade do resampler** (banda-passante / aliasing); (3) **distorção
> harmônica/IMD/true-peak**. Nenhum é medido. Pior: o golden de referência é o **próprio C++ NAMCore
> em f32** — logo erros de modo-comum f32 são, por construção, **inobserváveis**.

---

## A. Melhorias de Qualidade Sonora & Precisão

### P-1 · 🔴 ALTO — Aliasing das ativações não-lineares: não mitigado e não medido

**Base científica.** Sato & Smith III (DAFx 2025, arXiv:2505.04082) demonstram que modelos neurais de
amp (WaveNet/TCN) sofrem **aliasing significativo** originado nas **funções de ativação não-lineares**,
especialmente em fundamentais agudas e alto ganho — a não-linearidade expande a banda além de Nyquist e
as componentes dobram para dentro da banda audível. Eles introduzem a métrica **ASR (Aliasing-to-Signal
Ratio)** e mostram que **ativações mais suaves** (tanh, Snake) reduzem o ASR **sem** aumentar
significativamente o ESR. Carson, Wright & Bilbao (DAFx 2025, arXiv:2505.11375) mostram que reduzir
aliasing (via fine-tuning ou 2× oversampling) é frequentemente necessário; Kahles, Esqueda & Välimäki
(JAES 67(6), 2019) detalham o projeto correto de filtros para oversampling de não-linearidades; a linha
ADAA (Parker/Zavalishin/Le Bivic, DAFx-16; Bilbao/Esqueda/Parker/Välimäki, IEEE SPL 24(7) 2017; Holters,
DAFx-19 para sistemas com estado) oferece alternativa de baixo custo ao oversampling.

**Estado atual no nam-rs (evidência).**

- **Zero oversampling em torno do modelo não-linear.** O modelo roda na taxa nativa/host
  (`src/dsp/pipeline/stages/inference.rs`, PATH A taxas iguais / PATH B resample host↔nam — nenhum
  estágio sobe a taxa _para a não-linearidade_). Confirmado por varredura: não há `oversample/2x/4x`
  anti-alias em `src/dsp/`.
- As não-linearidades são `tanh` Padé (`src/math/activations/tanh/production.rs`), `sigmoid` minimax
  (`src/math/activations/sigmoid.rs`) e, no A2, **ReLU/LeakyReLU** (`src/models/a2/activations.rs`).
  ReLU/LeakyReLU têm **canto duro** (derivada descontínua) → por Sato & Smith, os **piores** geradores de
  aliasing. A `tanh` Padé é _clampeada_ em |x|<4 (`production.rs`), introduzindo descontinuidade de
  derivada na fronteira do clamp — fonte extra de aliasing **nunca medida**.
- **Nenhuma métrica de aliasing** (ASR ou equivalente) existe no QA (confirmado em `src/testing/` e
  `tests/`).

**Insight não-óbvio.** As aproximações fast-math foram validadas **apenas** por ESR/paridade no
**tempo**. Como ESR é quase insensível a aliasing de baixa energia espalhado no espectro, **é possível
que a escolha de aproximação esteja trocando precisão temporal (ótima) por aliasing espectral (não
medido)**. O `docs/fastmath-approximations.md:159` já alerta: com pesos reais (|x|>1) o erro da Padé
"torna-se significativo" — mas o efeito **espectral** disso jamais foi quantificado.

**Proposta de solução.**

1. **(Qualidade) Oversampling opcional 2×/4× ao redor do estágio neural**, com meia-banda anti-imagem /
   anti-alias projetada conforme Kahles et al. 2019 — reaproveitando a infra polifásica já existente
   (`src/dsp/resampler.rs`, `src/dsp/sinc_kernel.rs`). Expor como modo **"HQ/offline"** (latência+CPU)
   vs **"live"** (bypass), espelhando a prática real da indústria (hardware desliga OS ao vivo).
2. **(Alternativa de baixo custo) ADAA de 1ª/2ª ordem** nas ativações memoryless (tanh/ReLU) — reduz
   aliasing a ~½–¼ do custo do oversampling para casos memoryless; avaliar viabilidade dentro do laço
   convolucional.
3. **(QA — essencial) Implementar a métrica ASR** (Sato & Smith 2025): injeta senoides puras em pitches
   musicais, FFT da saída, classifica bins como harmônicos vs aliased (critério aritmético) e reporta
   energia-aliased/energia-harmônica por modelo. Esta é **a** medida que falta para um amp-sim.

**Critérios de aceite.** ASR medido e versionado por SKU; modo HQ reduz ASR ≥ X dB vs baseline em
entrada agressiva (E2/A5 high-gain, fundamental ≥ 2 kHz); ESR não regride além de tolerância calibrada.

**Esforço.** Médio-alto (ASR: ~2–3 dias; oversampling opcional: ~1–2 semanas com QA).

> **Nota do PO:** Criar um mecanismo claro e simples para o usuário acionar este oversampling OPCIONAL.
> Linha de comando no modo standalone e controle visual no GUI CLAP.

**Resolução (resposta à Nota do PO).** Sim — controle explícito do usuário, **OFF por padrão** (o caminho
_live_ de baixa latência fica intocado). Reaproveitamos padrões **já existentes** no projeto:

- **Standalone (CLI):** novo argumento `--oversample off|2x|4x` (alias `--os`), no mesmo molde do já
  existente `--slim auto|full|lite` (`src/standalone/cli.rs` → `CliArgs` + `print_help`). Default `off`.
- **CLAP (GUI + automação):** parâmetro _stepped_ "Oversampling" {Off, 2×, 4×}, no molde de
  `AdaptiveComputeMode::from_f32` (`src/clap/processor/params.rs`); exposto na GUI como combo/segmented
  control (`src/clap/gui/ui/`) e **persistido no estado** do plugin (`src/clap/extensions/state.rs`).
- **RT-safety (crítico):** trocar o fator de OS **realoca** os up/downsamplers e os buffers de trabalho do
  modelo na taxa oversampled → a troca ocorre **fora da thread de áudio** (mesma via do hot-swap de
  modelo / GC SPSC), com swap atômico; **nunca** realocar mid-`process`. O parâmetro é tratado como
  "exige re-preparação" (rebuild off-RT), não automatável amostra-a-amostra.

→ Estas ações entram como **tarefa dedicada de UX (CLI+GUI)** no Sprint **S5** e são documentadas em **S6**.

---

### P-2 · 🟠 ALTO — Fidelidade do resampler muito abaixo da aspiração; rolloff de topo atinge a base 44.1 kHz

**Estado atual (evidência).** O próprio teste documenta o problema: `src/dsp/resampler_test.rs:384-390`
— _"currently the polyphase filter achieves ~24 dB SNR across the passband"_, com _"~-1.7 dB at 9.9 kHz
for 44.1k→48k"_ e **threshold de apenas 20 dB** ("guards against gross filter quality regressions").
A documentação, porém, **anuncia >120 dB** (`src/dsp/sinc_kernel.rs:30-32`) e "quality > 120 dB SNR"
(`resampler.rs:23`). Config atual: **256 fases × 32 taps**, Kaiser β=12 (`sinc_kernel.rs:26-33`).

**Impacto sonoro.** Modelos NAM são nativos **48 kHz**; hosts a **44.1 kHz** (fração enorme da base)
**sempre** engajam o resampler → **perda de brilho no topo** (~-1.7 dB perto de 10 kHz) e fidelidade
real ~24 dB vs uma referência de alta qualidade (libsoxr). Conecta-se ao **F-2 da Parte I**: a relaxação
de thresholds de paridade por resampling tem **aqui** sua causa física.

**Proposta de solução.**

1. Reprojetar o filtro: **mais taps** (ex.: 48–64) e/ou desenho **half-band** otimizado; reavaliar se a
   transformada **minimum-phase via cepstrum** (`sinc_kernel.rs:156-223`) introduz ripple de magnitude
   (comparar magnitude pré/pós-cepstrum). Oferecer opção **linear-phase** para uso offline.
2. **QA de resampler de verdade:** medir **resposta em frequência, ripple de banda-passante, atenuação de
   stopband e THD/aliasing** via varredura senoidal (Farina) contra o **ideal analítico** (não apenas vs
   soxr), e **elevar progressivamente** o gate de 20 dB conforme a qualidade sobe (o próprio teste pede
   isso em `resampler_test.rs:389-390`).

**Critérios de aceite.** Ripple de banda-passante < 0,1 dB até 0,45×Nyquist; stopband ≥ 100 dB medido;
gate do teste elevado de 20 dB → faixa coerente com a medição; documentação reconciliada com a medição.

**Esforço.** Médio (reprojeto + QA: ~1 semana).

> **Nota do PO:** Observar o custo computacional extra. Se for considerável avaliar viabilidade de torna-lo
> opicional: linha de comando no modo standalone e controle visual no GUI CLAP.

**Resolução (resposta à Nota do PO).** O custo extra é **localizado e condicional**: o resampler **só roda
quando o host ≠ 48 kHz** (a 48 kHz há bypass total — `resampler.rs:428-431`); dobrar os taps (32→64)
~dobra apenas o custo do _resampler_, que é pequeno frente ao modelo neural. Plano dirigido por dado:

1. **Medir** o custo (Δμs por bloco) via bench do Sprint **S2** (P-7), nos cenários 44.1→48 e 96→48 kHz.
2. **Se desprezível** (esperado): HQ vira o **padrão** — é correção de qualidade e não onera a maioria.
3. **Se considerável:** expor um seletor "Resampler Quality", igual ao P-1 — `--resampler standard|hq`
   (standalone) e parâmetro _stepped_ + combo na GUI CLAP (persistido no estado); troca **off-RT** (realoca
   o banco polifásico, mesma via do P-1).

→ A decisão e o _default_ ficam registrados na **Conclusão** da tarefa de resampler (**S5**) com os números
do bench; a superfície de controle (se necessária) compartilha a **tarefa de UX** do P-1.

---

### P-5 · 🟡 MÉDIO — Erro de ativação sob pesos reais: medir e oferecer modo exato/alta-fidelidade

**Base científica + lacuna interna.** `docs/fastmath-approximations.md:162` **recomenda explicitamente**
"a follow-up measurement with real model weights … to quantify the tanh contribution under realistic
conditions" — **nunca executada**. A medição existente (S1.T1.4) usou pesos sintéticos pequenos (0,01)
onde tanh≈linear, **subestimando** o erro da Padé. Wright & Välimäki (ICASSP 2020, arXiv:1911.08922)
mostram ainda que ponderação perceptual (pré-ênfase A-weighting) altera a relevância do erro por banda.

**Estado atual.** Existe `src/math/activations/tanh/high_fidelity.rs` (~2,4e-7, exp-based) **pronto e
testado, porém não usado em produção**. Produção usa Padé (~2,32e-3).

**Proposta.** (1) Cumprir a medição pendente: quantificar contribuição da Padé ao ESR **e ao ASR (P-1)**
com **pesos reais** (modelos golden). (2) Oferecer **modo "exato/HF" opt-in** (high_fidelity ou
`f32::tanh`) para offline/mixdown — duplo ganho: menor erro **e** menor aliasing (ativação mais suave,
P-1). (3) Reportar ESR **com pré-ênfase** (perceptual) em complemento ao ESR plano.

**Critérios de aceite.** Tabela "contribuição Padé vs exato" (ESR/ASR) por modelo real; modo HF
selecionável com custo medido em bench.

**Esforço.** Baixo-médio (medição: ~1–2 dias; modo HF: já existe o kernel).

---

## B. Mecanismos de Medição & QA (aferição correta dos índices)

### P-3 · 🟠 ALTO — Suíte de fidelidade no domínio da frequência: THD/THD+N, IMD, resposta em frequência e true-peak

**Base científica/normas.** Farina (AES Conv. 108, 2000) — varredura senoidal exponencial separa, por
deconvolução, **resposta linear e cada ordem harmônica** em _time-lags_ distintos (1 medida → IR + THD).
AES17 — **THD+N** com tom de 997 Hz e notch Q∈[1,5]. **IMD** SMPTE/DIN (dois tons). ITU-R BS.1770-4
**Anexo 2** — **true-peak (dBTP)** por **4× oversampling** + LP + |·|.

**Lacuna no nam-rs (confirmada).** **Zero** testes de THD/THD+N/IMD/resposta-em-frequência/sweep
(varredura em `src/testing/`+`tests/` não encontrou nada); a infra FFT existe (`src/math/dsp/fft/`) mas
nunca é usada para isso. **A detecção de clipping é só de sample-peak** (`src/dsp/pipeline/stages/output.rs`,
`apply_gain_and_detect_clipping_*` → `RT_STATUS_HAS_CLIPPED`), **ignorando inter-sample peaks** — um modelo
pode gerar _overs_ inter-amostrais inaudíveis ao flag atual e que **clipam no DAC**.

**Proposta.** Nova suíte `tests/spectral_fidelity.rs` (e helpers em `src/testing/`):

1. **Resposta em frequência + THD por ordem** via sweep exponencial + deconvolução de Farina, por modelo.
2. **THD+N @ 997 Hz** (AES17) e **IMD** SMPTE (60 Hz + 7 kHz, 4:1).
3. **True-peak dBTP** (BS.1770-4 Anexo 2, 4× — reusar a polifásica) **e** corrigir a detecção de clipping
   para inter-sample (flag `RT_STATUS_HAS_CLIPPED` baseada em true-peak).

**Critérios de aceite.** Cada SKU tem fingerprint espectral versionado (FR/THD/IMD); regressão espectral
reprova; true-peak detecta _overs_ inter-amostrais sintéticos.

**Esforço.** Médio (~1 semana; reusa FFT/resampler existentes).

---

### P-4 · 🟠 MÉDIO-ALTO — Oráculo de referência em f64 (piso absoluto de precisão, independente do C++)

**Problema de fundo.** Todo golden vem do **C++ NAMCore, que é f32** (`tests/cpp_parity.rs`,
`tests/fixtures/golden_gen_build.sh`). Logo, **erros de modo-comum f32** (acúmulo, ordem de FMA,
aproximação de ativação compartilhada conceitualmente) são **estruturalmente inobserváveis** — a paridade
mede "igual ao C++", não "correto". Não há **verdade-terreno absoluta**.

**Proposta (inovadora).** Implementar um **motor de referência em Rust puro, f64**, com `f64::tanh`/`exp`
exatos e **acúmulo f64 (ou compensado Kahan/Neumaier)** — usado **somente no QA** (não-RT). Ele fornece:

- O **piso absoluto de erro** do caminho de produção f32 (mede o erro **próprio** do nam-rs, não vs C++).
- Decomposição rigorosa das fontes (quantização de peso vs ativação vs acúmulo) **com pesos reais**,
  cumprindo a recomendação de `fastmath-approximations.md:162` e estendendo a S1.T1.4.
- Base para decidir, com dados, se vale **acúmulo compensado** no head/skip
  (`src/models/.../layer_array.rs`, soma f32 sequencial por camada) e no estado recorrente do LSTM (bf16).

**Critérios de aceite.** `tests/reference_oracle_f64.rs` reporta, por modelo, ESR(produção f32 vs oráculo
f64) e a decomposição por fonte; documentado em `docs/perceptual_validation.md`.

**Esforço.** Médio (~3–5 dias; reaproveita topologias existentes com tipos genéricos/f64).

> **Nota do PO:** Dúvida: mas a "fonte de verdade" será quem? O NAMcore obrigado a rodar como f64? Ou o
> próprio NAM-rs. Se for o próprio NAM-rs, qual o sentido de usar a si mesmo como referência?

**Resolução (resposta à Nota do PO).** A "fonte de verdade" **não é o NAM-rs nem o NAMCore** — é a
**matemática ideal do modelo**: a saída que a rede _deveria_ produzir, dados seus pesos, com aritmética de
precisão infinita. Nem o NAMCore nem o NAM-rs a calculam — **ambos** usam f32 + aproximações (Padé/minimax)
e, portanto, **compartilham as mesmas limitações**. Três esclarecimentos:

1. **Por que o NAMCore (f32) não serve de verdade-terreno de _precisão_.** Comparar nam-rs(f32) × NAMCore(f32)
   mede **paridade de implementação** (interoperabilidade), não **correção numérica**. Erros de _modo-comum_
   (acúmulo f32, ordem de FMA, aproximação de ativação) são **invisíveis**, pois os dois lados os cometem
   igualmente. **Não** se obriga o NAMCore a rodar em f64 (não o modificamos).

2. **O oráculo NÃO é "o NAM-rs medindo a si mesmo".** É uma implementação **independente** da _mesma
   matemática_, porém: em **f64** (~15–16 dígitos vs ~7 do f32 → ~8 ordens de grandeza mais preciso); com
   **ativações exatas** (`f64::tanh`/`exp`, não Padé); e **acúmulo f64/Kahan** (não a soma f32+FMA plana).
   Ele **não compartilha nenhuma** fonte de erro do caminho de produção. Seu erro próprio (~1e-15) é
   desprezível frente ao que se mede (~1e-3…1e-7) → _para medir o erro do f32_, o oráculo **é** a
   verdade-terreno. Validar uma rotina rápida/baixa-precisão contra a **mesma fórmula em dupla precisão**
   é prática-padrão em computação numérica (cf. MPFR/double-double) — **não** é circularidade.

3. **Duas referências complementares, para duas perguntas distintas** (mantemos as duas):
   - **NAMCore f32** → _"Paridade/Interop"_: o nam-rs soa idêntico ao player de referência da comunidade?
   - **Oráculo f64** → _"Correção absoluta"_: qual o erro numérico **próprio** do caminho f32 vs a
     matemática ideal — e a **decomposição** por fonte (peso × ativação × acúmulo), que o C++ **não**
     permite (modo-comum).

**Para eliminar qualquer suspeita de auto-referência, o oráculo é _ancorado uma vez_ numa referência externa
de altíssima precisão:** a **rede original em PyTorch/NumPy executada em f64** (a verdadeira origem dos
pesos). Casando o oráculo Rust-f64 com o PyTorch-f64 dentro de ~1e-12 para **≥ 1 modelo por família**
(WaveNet / LSTM / A2), ele fica **validado como verdade-terreno** e passa a servir de referência barata,
determinística e _em-repo_ — sem depender do PyTorch no CI. Assim respondemos "qual o sentido": não usamos o
NAM-rs como referência de si; usamos um **cálculo independente, em precisão muito maior, ancorado na origem
dos pesos**.

→ A **Tarefa 2.1 (S2)** passa a incluir o **passo de validação externa do oráculo** como critério de aceite.

---

### P-6 · 🟡 MÉDIO — LUFS completo (ITU-R BS.1770-4, 2 passes) + LRA + true-peak

**Estado atual.** `src/testing/perceptual.rs:217-273` implementa LUFS **simplificado** (gate absoluto de
1 passe, sem gate relativo de −10 LU), **diagnóstico-only** (`docs/perceptual_validation.md:56`), sem
**LRA** nem **true-peak**. Isso enfraquece o gate de plausibilidade (origem do "GOLDEN DEFECT", ver D-2 da
Parte I).

**Proposta.** Implementar BS.1770-4 pleno (2 passes: −70 LUFS absoluto → −10 LU relativo) + **LRA**
(EBU Tech 3342) + **true-peak** (compartilhado com P-3). Com loudness confiável, o gate de plausibilidade
pode ser promovido de informativo a (soft/hard), e a label "GOLDEN DEFECT" deixa de ser necessária.

**Critérios de aceite.** LUFS integrado validado contra vetores de referência EBU (±0,1 LU); LRA e dBTP
reportados; D-2 (Parte I) resolvido sem mislabel.

**Esforço.** Baixo-médio (~2–3 dias).

---

### P-7 · 🟡 MÉDIO — Gates de regressão de performance e de deadline RT

**Estado atual.** O bench rápido usa parâmetros **estatisticamente fracos** (`testes.log:3636`:
`--sample-size 10 --measurement-time 0.5 --warm-up-time 0.2`) — incapaz de detectar regressões < ~10–20%.
O bench longo (Fase 6) só reporta **top-5 mais lentos**, sem **pass/fail vs deadline**. A infra de
telemetria existe e é boa (`src/dsp/telemetry.rs` LatencyHistogram p50/p99/max; `tests/nam_infer_test.rs:686-704`
já compara p99 vs 1,33 ms) **mas não há gate** que reprove se o p99 estourar o deadline.

**Proposta.** (1) **Gate de deadline RT**: p99 do bloco de 64 amostras < **1,33 ms** @ 48 kHz para
**todos os SKUs** e **todos os estados adaptativos** (Full/Reduced/Minimal). (2) **Gate de regressão**
com baselines salvos (Criterion + amostragem adequada em core fixado/`taskset`). (3) Medir **xrun/jitter
sob pressão de escalonamento** (carga concorrente), não só RSS.

**Critérios de aceite.** CI reprova em regressão > limiar estatístico (p<0,05) ou em p99 > deadline;
relatório de jitter sob carga.

**Esforço.** Médio (~3–5 dias).

---

### P-8 · 🟡 MÉDIO — Matriz de determinismo cruzado entre ISAs (scalar / AVX2 / AVX-512 / VNNI-bf16)

**Estado atual.** Há paridade **unitária** de kernels (scalar vs AVX2 vs AVX-512 — `testes.log:251-497`),
mas **não há paridade ponta-a-ponta de saída de modelo entre ISAs**, nem entre CPUs (contração de FMA,
ordem de redução). O guard de determinismo é só **SHA do binário** (`testes.log:3510-3512`), não da
**saída de áudio** entre ISAs.

**Proposta.** Suíte que roda os golden vectors forçando cada caminho (scalar/AVX2/AVX-512/VNNI quando
presente) e **assere paridade de saída entre ISAs** dentro de um orçamento ULP/ESR calibrado. Entrega
duplo valor: pega regressões específicas de ISA **e** quantifica o piso de erro SIMD-vs-scalar
(dado útil de "aferição de precisão").

**Critérios de aceite.** `tests/isa_parity.rs` verde com orçamento documentado por modelo; CI cobre ao
menos scalar+AVX2 (AVX-512/VNNI quando o runner suportar).

**Esforço.** Médio (~3–4 dias).

---

## Referências (Parte II)

- R. Sato, J. O. Smith III, **"Aliasing Reduction in Neural Amp Modeling by Smoothing Activations"**,
  DAFx 2025 (arXiv:2505.04082) — métrica **ASR**; ativações suaves reduzem aliasing sem subir ESR.
- A. Carson, A. Wright, S. Bilbao, **"Anti-aliasing of neural distortion effects via model fine tuning"**,
  DAFx 2025 (arXiv:2505.11375) — fine-tuning anti-aliasing frequentemente supera 2× oversampling.
- J. Kahles, F. Esqueda, V. Välimäki, **"Oversampling for Nonlinear Waveshaping: Choosing the Right
  Filters"**, JAES 67(6):440-449, 2019.
- J. D. Parker, V. Zavalishin, E. Le Bivic, **"Reducing the Aliasing of Nonlinear Waveshaping Using
  Continuous-Time Convolution"**, DAFx-16 — ADAA (1ª ordem).
- S. Bilbao, F. Esqueda, J. D. Parker, V. Välimäki, **"Antiderivative Antialiasing for Memoryless
  Nonlinearities"**, IEEE Signal Process. Lett. 24(7):1049-1053, 2017.
- M. Holters, **"Antiderivative Antialiasing for Stateful Systems"**, DAFx-19 (relevante p/ LSTM).
- A. Wright, V. Välimäki, **"Perceptual Loss Function for Neural Modelling of Audio Systems"**, ICASSP
  2020 (arXiv:1911.08922) — pré-ênfase A-weighting; relevância perceptual do ESR.
- A. Wright et al., **"Real-Time Guitar Amplifier Emulation with Deep Learning"**, Applied Sciences
  10(3):766, 2020 — origem da métrica ESR usada pelo NAM.
- A. Farina, **"Simultaneous measurement of impulse response and distortion with a swept-sine
  technique"**, AES Convention 108, 2000 — FR + THD por deconvolução.
- **AES17** — medição de THD+N (997 Hz, notch Q 1–5).
- **ITU-R BS.1770-4** (Anexo 2, true-peak 4× oversample) / **EBU R128** / **EBU Tech 3342** (LRA).

---

## Épicos (agrupamento para resolução rápida, segura e ordenada)

> Sequência otimizada (Parte I): **E1 primeiro** (desbloqueia o debug de tudo o mais), depois **E2**
> (fecha o ponto cego de correção), depois **E3** (processo/clareza). **Parte II:** **E5 (medição)
> antes de E4 (correção sonora)** — não se corrige o que não se mede; o ASR/THD/oráculo f64 precedem o
> oversampling e o reprojeto do resampler.

### Épico E1 — "Diagnóstico de paridade confiável sob falha"  _(desbloqueador)_

- **Findings:** F-1 (crítico), F-4 (clareza ESR).
- **Objetivo:** garantir que qualquer falha de paridade produza um relatório contíguo, atribuível e
  com veredito visual completo (incluindo ESR primário).
- **Por que primeiro:** sem isto, qualquer trabalho em E2 que **provoque** uma falha de gate gera
  logs ilegíveis. É pré-requisito de produtividade.
- **Risco:** baixo (mudança restrita a `tests/`); cuidado apenas para não engolir saída útil de
  diagnóstico ao capturar o stdout do filho.

### Épico E2 — "Fechar o ponto cego de fidelidade perceptual"  _(crítico de correção)_

- **Findings:** F-2 (alto).
- **Objetivo:** promover MR-STFT a gate hard calibrado (≥ 44.1/48 kHz), limitar o piso de
  relaxamento dos gates hard, e fazer RCA do caso `0.4 dB @ 48000` (taxa nativa).
- **Depende de:** E1 (para depurar as falhas que surgirem ao apertar os gates).
- **Risco:** **médio-alto** — apertar gates pode expor divergências reais (objetivo!) e exigir
  correção de DSP. Tratar com cautela e calibração orientada por medição (anti-placebo).

### Épico E3 — "Confiança do loop rápido + higiene de relatório"  _(processo)_

- **Findings:** F-3 (médio), D-2 nit (opcional).
- **Objetivo:** levar um subconjunto de paridade hard @ 48 kHz para o `tests-quick.sh` e reduzir o
  ruído cosmético "GOLDEN DEFECT" em runs verdes.
- **Depende de:** E2 (para que o subconjunto rápido herde os gates calibrados).
- **Risco:** baixo.

### Épico E5 — "Instrumentação de Medição & QA Científica"  _(habilitador da Parte II)_

- **Findings:** P-3 (suíte espectral THD/IMD/FR/true-peak), P-4 (oráculo f64), P-6 (LUFS BS.1770-4),
  P-7 (gates de perf/deadline), P-8 (matriz cross-ISA); inclui a **métrica ASR** de P-1.
- **Objetivo:** dar ao projeto os instrumentos que **medem corretamente** aliasing, distorção, resposta
  em frequência, loudness, true-peak, performance e determinismo — produzindo dados confiáveis.
- **Por que antes de E4:** _não se corrige o que não se mede_. ASR/THD/oráculo f64 são pré-requisito
  para validar qualquer ganho de qualidade sonora.
- **Risco:** baixo-médio (código de teste/medição, fora do hot-path RT). Cuidado com a corretude das
  próprias métricas (validar contra vetores de referência EBU/AES e goldens Python/Farina).

### Épico E4 — "Qualidade Sonora: Anti-aliasing & Fidelidade"  _(correção sonora)_

- **Findings:** P-1 (oversampling opcional + ADAA), P-2 (reprojeto/QA do resampler), P-5 (ativação
  exata/HF opt-in).
- **Objetivo:** reduzir aliasing e melhorar a fidelidade espectral (resampler + topo de banda) com
  modos HQ/offline, sem comprometer o caminho live de baixa latência.
- **Depende de:** **E5** (precisa de ASR/THD/FR/true-peak para provar ganho e evitar regressão).
- **Risco:** **médio-alto** — toca o hot-path DSP e o trade-off latência×qualidade; tratar com
  feature-flags (live vs offline) e benchmarks (P-7).

---

_Findings produzidos pelas skills `revisor-auditor` (Parte I) e `pesquisador-inovador` (Parte II).
Para transformá-los em sprints/tasks ágeis, acione a skill `planejador-arquiteto` para gerar
`TODO-sprints.md` referenciando os Épicos E1–E5 acima._
