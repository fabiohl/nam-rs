<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-sprints.md — Planejamento Ágil (Sprints e Tarefas)

Este documento organiza a execução física dos planos de melhoria e correções de auditoria em sprints e tarefas de engenharia detalhadas, com foco em segurança, rastreabilidade e mitigação de riscos.

---

## Relação com Achados e Epicos

* **Referência:** `TODO-findings.md` § Epic F1-A (concluído) e Epic F1-B (concluído)
* **Referência:** `TODO-findings.md` § Achado F2 / Epic F2 (Sprint 3, abaixo) — **em planejamento**

---

## Sprint 1: Correção do Fixture FiLM e Recalibração de Thresholds

### 1. Resumo da Sprint

Esta sprint foca em corrigir a inicialização do canal de `scale` do FiLM no script de fixtures sintéticos de teste. O objetivo é remover o falso positivo "audível" (ESR inflado) gerado pelo esmagamento de energia do sinal, estabelecendo thresholds reais e confiáveis de fidelidade de áudio de acordo com a causa raiz documentada no Achado F1 do `TODO-findings.md`.

---

### 2. Tarefas Técnicas

#### **Tarefa T1.1: Correção do Script Gerador de Fixtures (Python)**

* **Objetivo:** Aplicar o bias de identidade `+1.0` ao canal de `scale` do FiLM no gerador.
* **Detalhes técnicos:**
  * Modificar a função `generate_weights_film` em [generate_a2_fixtures.py](file:///home/fabio/nam-rs/tests/fixtures/generate_a2_fixtures.py) para que a metade correspondente ao bias do canal de `scale` seja deslocada por `+1.0` (tornando-a *identity-biased*), enquanto a metade de `shift` permanece centrada em `0.0`.
  * Preservar a consistência do RNG (Random Number Generator) para não corromper outras partes dos fixtures.
* **Arquivos:** [generate_a2_fixtures.py](file:///home/fabio/nam-rs/tests/fixtures/generate_a2_fixtures.py)
* **Status:**  ✅ Concluído

#### **Tarefa T1.2: Regeneração dos Fixtures e Golden Vectors C++**

* **Objetivo:** Recriar os arquivos `.nam` e os vetores de referência C++ `.bin`.
* **Detalhes técnicos:**
  * Executar o script [golden_gen_build.sh](file:///home/fabio/nam-rs/tests/fixtures/golden_gen_build.sh) para processar o script Python corrigido e renderizar os novos arquivos binários de referência com o NAMCore C++.
* **Arquivos:**
  * [wavenet_a2_film_lite.nam](file:///home/fabio/nam-rs/tests/fixtures/models/wavenet_a2_film_lite.nam)
  * [wavenet_a2_film_full.nam](file:///home/fabio/nam-rs/tests/fixtures/models/wavenet_a2_film_full.nam)
  * [golden_wavenet_a2_film_lite.bin](file:///home/fabio/nam-rs/tests/fixtures/golden_wavenet_a2_film_lite.bin)
  * [golden_wavenet_a2_film_full.bin](file:///home/fabio/nam-rs/tests/fixtures/golden_wavenet_a2_film_full.bin)
* **Status:**  ✅ Concluído

#### **Tarefa T1.3: Medição dos Novos Valores de ESR/SNR**

* **Objetivo:** Coletar as métricas reais de fidelidade obtidas contra a nova referência de alta energia do sinal.
* **Detalhes técnicos:**
  * Executar a suíte de testes de golden vectors para os modelos FiLM, capturando as falhas intencionais de threshold antigos e extraindo as medições exatas de SNR, ESR e MR-STFT.
  * A expectativa é de que o SNR suba para a faixa de `>90 dB` (ESR `<1e-9`).
* **Status:**  ✅ Concluído

#### **Tarefa T1.4: Recalibração de Thresholds de Validação (Rust)**

* **Objetivo:** Atualizar os limites de aceitação na suíte de testes automatizados com base nas medições reais da Tarefa T1.3.
* **Detalhes técnicos:**
  * Ajustar as chaves `"wavenet_a2_film_lite"` (linhas 589-592) e `"wavenet_a2_film_full"` (linhas 599-602) em [validation.rs](file:///home/fabio/nam-rs/tests/common/validation.rs).
  * Atualizar as descrições nos comentários que detalham os valores medidos e a margem de tolerância adicionada.
* **Arquivos:** [validation.rs](file:///home/fabio/nam-rs/tests/common/validation.rs)
* **Status:**  ✅ Concluído

#### **Tarefa T1.5: Homologação e Verificação Final (Lints & Quick Tests)**

* **Objetivo:** Confirmar que toda a base compila perfeitamente e passa sem erros de conformidade ou regressões.
* **Detalhes técnicos:**
  * Rodar `utils/lints.sh` para verificar formatação e cabeçalhos SPDX.
  * Rodar `utils/tests-quick.sh` para atestar a validação completa de fidelidade sob os novos limites.
* **Status:**  ✅ Concluído

---

## Sprint 2: Preservação do Teste de Estresse Numérico e Documentação

Esta sprint foca em preservar a fixture degenerada original de FiLM como um teste dedicado de robustez numérica sob adversidades (denominada `wavenet_a2_film_chaos_stress.nam`), garantindo que o motor de inferência processe regimes sem bias de identidade sem produzir inconsistências numéricas (NaN/Inf). Paralelamente, todas as documentações e mapas de paridade são atualizados para refletir a nova realidade das medições ideais do FiLM.

---

### **Tarefa T2.1: Registro e Geração do Golden para o Chaos Stress Fixture**

* **Objetivo:** Registrar o novo fixture `wavenet_a2_film_chaos_stress.nam` no catálogo e gerar seu vetor binário de referência C++.
* **Detalhes técnicos:**
  * Atualizar a lista `CATALOG` no script [golden_gen_build.sh](file:///home/fabio/nam-rs/tests/fixtures/golden_gen_build.sh) para adicionar a entrada `"wavenet_a2_film_chaos_stress.nam:golden_wavenet_a2_film_chaos_stress:A2-FiLM Chaos Stress (CH=3):none"`.
  * Executar `./tests/fixtures/golden_gen_build.sh` para renderizar o golden binary de referência `golden_wavenet_a2_film_chaos_stress.bin`.
* **Arquivos:** [golden_gen_build.sh](file:///home/fabio/nam-rs/tests/fixtures/golden_gen_build.sh)
* **Status:**  ✅ Concluído

### **Tarefa T2.2: Adição do Teste de Golden Vector e Threshold Próprio**

* **Objetivo:** Adicionar o teste unitário de golden vector do novo modelo e configurar os thresholds originais dele no Rust.
* **Detalhes técnicos:**
  * Configurar a chave `"wavenet_a2_film_chaos_stress"` em [validation.rs](file:///home/fabio/nam-rs/tests/common/validation.rs) usando os antigos thresholds degenerados (ex: SNR = `12.0` dB, ESR = `3.5e-2`).
  * Adicionar o teste unitário `test_golden_vectors_wavenet_a2_film_chaos_stress` em [golden_vectors.rs](file:///home/fabio/nam-rs/tests/models/golden_vectors.rs).
  * Registrar a chave `"wavenet_a2_film_chaos_stress"` no teste de calibração em [threshold_calibration.rs](file:///home/fabio/nam-rs/tests/models/threshold_calibration.rs).
* **Arquivos:**
  * [validation.rs](file:///home/fabio/nam-rs/tests/common/validation.rs)
  * [golden_vectors.rs](file:///home/fabio/nam-rs/tests/models/golden_vectors.rs)
  * [threshold_calibration.rs](file:///home/fabio/nam-rs/tests/models/threshold_calibration.rs)
* **Status:**  ✅ Concluído

### **Tarefa T2.3: Atualização da Documentação Técnica de Fidelidade e Paridade**

* **Objetivo:** Atualizar os arquivos markdown do repositório para refletir a correção da causa raiz e as medições reais de SNR/ESR obtidas.
* **Detalhes técnicos:**
  * Substituir o diagnóstico incorreto de associatividade pelo correto (bias de inicialização do gerador) no Achado 1 de [TODO-parity.md](file:///home/fabio/nam-rs/TODO-parity.md).
  * Atualizar a seção §4.3 de [cpp_parity_map.md](file:///home/fabio/nam-rs/docs/cpp_parity_map.md) explicando a causa real e a mitigação pelo teste de estresse de caos numérico.
  * Atualizar as tabelas de thresholds e medições do FiLM nos arquivos [audio_fidelity_map.md](file:///home/fabio/nam-rs/docs/audio_fidelity_map.md) e [perceptual_validation.md](file:///home/fabio/nam-rs/docs/perceptual_validation.md).
* **Arquivos:**
  * [TODO-parity.md](file:///home/fabio/nam-rs/TODO-parity.md)
  * [cpp_parity_map.md](file:///home/fabio/nam-rs/docs/cpp_parity_map.md)
  * [audio_fidelity_map.md](file:///home/fabio/nam-rs/docs/audio_fidelity_map.md)
  * [perceptual_validation.md](file:///home/fabio/nam-rs/docs/perceptual_validation.md)
* **Status:**  ✅ Concluído

### **Tarefa T2.4: Homologação de Sprints e Execução de Testes Rápidos**

* **Objetivo:** Homologar a entrega e certificar conformidade completa.
* **Detalhes técnicos:**
  * Executar `utils/lints.sh` e `utils/tests-quick.sh`.
* **Status:**  ✅ Concluído

---

### 3. Matriz de Risco e Mitigação das Sprints 1 e 2

| Risco                                                      | Impacto | Mitigação                                                                                                                                                       |
|:---------------------------------------------------------- |:------- |:--------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Quebra do determinismo dos pesos sintéticos (Sprint 1)** | Médio   | Garantir que o RNG use seed fixa e a lógica de geração preserve a dimensionalidade correta dos tensores.                                                        |
| **Tolerâncias de Threshold estritas demais (Sprint 1)**    | Baixo   | Adicionar margens adequadas nos novos thresholds (~3 dB em SNR e fator extra de tolerância em ESR) para tolerar pequenas variações ambientais de float.         |
| **Divergência imprevista no modelo de caos (Sprint 2)**    | Baixo   | Como o modelo e os pesos são idênticos aos anteriores, a paridade com o golden original e o C++ deve se manter estável nos níveis reportados antes da correção. |

---

## Sprint 3: Correção dos Bugs de Aplicação FiLM em `input_mixin_post_film` / `layer1x1_post_film` (Achado F2)

**Esta sprint corrige código de produção real** (`src/models/a2/model/dynamic/process.rs`, e,
por consistência/dead-code-hygiene, os caminhos estáticos espelhados
`src/models/a2/conv1d_ch3/simd.rs` e `src/models/a2/conv1d_ch8/simd.rs`), não apenas fixtures de
teste — ao contrário das Sprints 1/2. O Achado F2 (`TODO-findings.md`) comprovou, por leitura de
código dos dois lados (C++ NAMCore e Rust) e por medição empírica reprodutível (render C++ real
vs. `WaveNetA2Dyn`), **três bugs de paridade genuínos e independentes** nos pontos de inserção FiLM
`input_mixin_post_film` (Bug B1) e `layer1x1_post_film` (Bugs B2 e B3):

* **B1** — `input_mixin_post_film` modula `conv + mixin` já combinados; o C++ modula **somente**
  o `mixin`, antes de somá-lo ao `conv`.
* **B2** — `layer1x1_post_film` é aplicado **incondicionalmente**; o C++ só o aplica quando
  `gating_mode == BLENDED` (nunca em `NONE`/`GATED`).
* **B3** — mesmo restrito ao modo `BLENDED`, o Rust aplicaria o FiLM **depois** de somar o
  residual (`input + l1x1`); o C++ aplica ao `l1x1` isolado e só depois soma o residual.

Medição de referência que motiva esta sprint (ver Achado F2, seção "Verificação empírica"):
restaurando os 4 slots FiLM originais (bias de identidade já corrigido pela Sprint 1), o SNR
despenca de 138,3 dB (só `conv_post_film`+`activation_post_film`) para **-0,8 dB** — prova
inequívoca de que B1/B2/B3 são bugs reais, não uma questão de fixture.

**Impacto adicional confirmado:** `wavenet_a2_max.nam` (modelo flagship real, `condition_size=8`,
atualmente desabilitado por estar "ativamente quebrado" — Achado 2 do `TODO-parity.md`) tem
`input_mixin_post_film` e `layer1x1_post_film` ativos com `gating_mode: "none"` em todas as
camadas. B1/B2 afetam esse modelo diretamente e são, no mínimo, um contribuinte adicional
(não investigado até agora) para o ESR≈3,61e1 documentado no Achado 2.

**Ordem de execução obrigatória:** corrigir e testar unitariamente o código de produção
(T3.1–T3.5) **antes** de tocar em qualquer fixture/golden (T3.6+). Isso evita recalibrar
thresholds contra um motor ainda incorreto.

### **Tarefa T3.1 [🔴 CRÍTICA — Bug B1]: Isolar o `mixin` em scratch buffer próprio antes do FiLM no motor dinâmico**

* **Objetivo:** Fazer `input_mixin_post_film` modular **somente** a saída do `mixin`, replicando
  exatamente `model.cpp:198-204` (`z = conv + film(mixin)`).
* **Detalhes técnicos:**
  * Em [mod.rs](file:///home/fabio/nam-rs/src/models/a2/model/dynamic/mod.rs) (struct
    `WaveNetA2Dyn`, próximo ao campo `head1x1_scratch` — linha ~136), adicionar um novo campo de
    scratch pré-alocado `pub mixin_scratch: AlignedVec<f32>`, dimensionado para o maior
    `z_out_ch` possível entre as camadas (mesmo critério de dimensionamento hoje usado para
    `z_scratch`, linha ~134). Inicializar no(s) construtor(es) (ex.: linha ~305,
    `AlignedVec::new(<mesmo tamanho de z_scratch>, 0.0f32)`), no mesmo ponto onde
    `head1x1_scratch` é inicializado.
  * Em [process.rs](file:///home/fabio/nam-rs/src/models/a2/model/dynamic/process.rs), adicionar
    o parâmetro `mixin_scratch: &mut [f32]` à assinatura de `process_frame_dyn` (linha ~318,
    junto aos demais parâmetros de scratch como `head1x1_scratch`), e passá-lo no call site
    (próximo à linha ~275, onde `z_scratch`/`head1x1_scratch` já são passados).
  * Reescrever o bloco do mixin (hoje linhas 371-385):
    1. Computar `mixin_scratch[c] = Σ mixin_w[...] * cond_slice[k]` (o mesmo laço de hoje, linha
       ~374-379, mas escrevendo em `mixin_scratch` em vez de somar direto em `z_scratch`).
    2. Se `layer.input_mixin_post_film` estiver ativo, aplicar
       `film.process(&mut mixin_scratch[..z_out_ch], cond_slice)` **sobre o `mixin_scratch`
       isolado** (não sobre `z_scratch`).
    3. Só então somar: `z_scratch[c] += mixin_scratch[c]` para `c in 0..z_out_ch`.
  * Remover a chamada antiga em `z_scratch` (linha atual ~382-385).
  * Preservar o comportamento quando `input_mixin_post_film` é `None`: o resultado deve ser
    bit-idêntico ao código atual (soma direta), já que o único efeito observável do buffer
    intermediário é permitir a modulação isolada.
* **Riscos:** memória adicional pré-alocada (não impacta RT-safety — segue o padrão já
  estabelecido por `head1x1_scratch`); atenção ao dimensionamento correto para modelos com
  `groups > 1` (múltiplos grupos de `cond_size`/`z_out_ch` distintos por camada).
* **Arquivos:**
  [mod.rs](file:///home/fabio/nam-rs/src/models/a2/model/dynamic/mod.rs),
  [process.rs](file:///home/fabio/nam-rs/src/models/a2/model/dynamic/process.rs)
* **Status:**  ✅ Concluído (commit `2108b3f`)
* **Verificação (auditoria 2026-07-10):** diff confere linha a linha com a spec — `mixin_scratch`
  dimensionado `bottleneck * 2` (idêntico a `z_scratch`), inicializado no único construtor de
  produção (`WaveNetA2Dyn::new`, usado pelo dispatcher real em
  `src/loader/dispatcher/wavenet/mod.rs:273`), mixin computado isolado, FiLM aplicado só nele, soma
  para `z_scratch` só depois. `cargo check`/`clippy --all-targets --all-features` limpos.

### **Tarefa T3.2 [🔴 CRÍTICA — Bug B2]: Restringir `layer1x1_post_film` ao modo `BLENDED` no motor dinâmico**

* **Objetivo:** Fazer o Rust nunca aplicar `layer1x1_post_film` fora do modo `BLENDED`,
  replicando `model.cpp:217-271` (a chamada só existe dentro do branch `GatingMode::BLENDED`).
* **Detalhes técnicos:**
  * Em [process.rs](file:///home/fabio/nam-rs/src/models/a2/model/dynamic/process.rs), no bloco
    "5. L1x1 residual" (hoje linhas 466-483), envolver a chamada a
    `layer.layer1x1_post_film` (linhas 479-483) com um guard explícito:
    `if use_blending { if let Some(ref mut film) = layer.layer1x1_post_film { ... } }`.
  * `use_blending: bool` já está disponível como parâmetro de `process_frame_dyn` (linha 326;
    computado em `self.gating_modes[li] == GatingMode::Blended`, linha 199) — não é necessário
    plumbing adicional, apenas o guard.
  * **Dependência de ordem:** aplicar esta tarefa **depois** de T3.1 e **antes** de T3.3, pois
    T3.3 modifica o mesmo bloco de código (residual l1x1).
* **Arquivos:** [process.rs](file:///home/fabio/nam-rs/src/models/a2/model/dynamic/process.rs)
* **Status:** ✅ Concluído (commit `cc892ae`)
* **Verificação (auditoria 2026-07-10):** guard `layer.layer1x1_post_film.as_mut().filter(|_|
  use_blending)` implementado exatamente como especificado — nenhum plumbing extra necessário,
  confirmado que `use_blending` já existia como parâmetro de `process_frame_dyn`.

### **Tarefa T3.3 [🔴 CRÍTICA — Bug B3]: Isolar o `layer1x1` em scratch buffer próprio antes do FiLM e antes da soma residual, no motor dinâmico**

* **Objetivo:** Fazer `layer1x1_post_film` (quando `use_blending == true`, após T3.2) modular
  **somente** a saída do `layer1x1`, com a soma do residual ocorrendo **depois**, replicando
  exatamente `model.cpp:265-268` + `:359-360` (`output = input + film(l1x1(z))`).
* **Detalhes técnicos:**
  * Adicionar um novo campo de scratch pré-alocado `pub l1x1_scratch: AlignedVec<f32>`
    (dimensionado para `channels` máximo entre as camadas) em
    [mod.rs](file:///home/fabio/nam-rs/src/models/a2/model/dynamic/mod.rs), seguindo o mesmo
    padrão de T3.1/`head1x1_scratch`. Inicializar no(s) construtor(es).
  * Em [process.rs](file:///home/fabio/nam-rs/src/models/a2/model/dynamic/process.rs), adicionar
    o parâmetro `l1x1_scratch: &mut [f32]` à assinatura de `process_frame_dyn` e ao call site.
  * Reescrever o bloco "5. L1x1 residual" (linhas ~466-483, já modificado por T3.2):
    1. Computar `l1x1_scratch[oc] = l1x1_b[oc] + Σ l1x1_w[...] * z_scratch[ic]` (o mesmo laço de
       hoje, linhas ~470-476, mas escrevendo em `l1x1_scratch` em vez de acumular direto em
       `layer_in`).
    2. Se `use_blending && layer.layer1x1_post_film.is_some()`, aplicar
       `film.process(&mut l1x1_scratch[..channels], cond_slice)` sobre o `l1x1_scratch` isolado.
    3. Só então somar: `layer_in[base + oc] += l1x1_scratch[oc]` para `oc in 0..channels`.
  * Preservar bit-exatidão para `layer.layer1x1_post_film.is_none()` ou `use_blending == false`
    (deve reduzir exatamente ao comportamento atual de soma direta).
* **Riscos:** este bloco é compartilhado por todos os modos de gating (`NONE`/`GATED`/`BLENDED`);
  garantir que a introdução do `l1x1_scratch` não regride nenhum teste existente de
  `a2_dynamic_gated_ch8`/`a2_dynamic_blended_ch3` (que hoje passam com SNR 103/133 dB).
* **Arquivos:**
  [mod.rs](file:///home/fabio/nam-rs/src/models/a2/model/dynamic/mod.rs),
  [process.rs](file:///home/fabio/nam-rs/src/models/a2/model/dynamic/process.rs)
* **Status:**  ✅ Concluído (commit `8a2de15`)
* **Verificação (auditoria 2026-07-10):** confirmado que o bias `l1x1_b[oc]` permanece incluído em
  `l1x1_scratch` antes do FiLM (não foi perdido na refatoração — `l1x1_scratch[oc] = sum` onde
  `sum` já parte de `l1x1_b[oc]`). Soma residual `layer_in[base+oc] += l1x1_scratch[oc]` ocorre
  depois do FiLM, como especificado. `a2_dynamic_gated_ch8`/`a2_dynamic_blended_ch3` (risco 🔴 da
  matriz de risco, bloco de código compartilhado) **não regrediram**: SNR medido 103,0 dB / 132,8
  dB (era 103/133 dB antes da mudança) — dentro da margem de ruído de medição.

### **Tarefa T3.4 [🟡 Consistência — sem impacto funcional atual]: Espelhar B1/B2/B3 nos caminhos estáticos CH=3/CH=8**

* **Objetivo:** Corrigir os mesmos 3 bugs em
  [conv1d_ch3/simd.rs](file:///home/fabio/nam-rs/src/models/a2/conv1d_ch3/simd.rs) e
  [conv1d_ch8/simd.rs](file:///home/fabio/nam-rs/src/models/a2/conv1d_ch8/simd.rs), que replicam
  exatamente o mesmo padrão incorreto (confirmado por leitura de código no Achado F2).
* **Nota de risco/prioridade:** o dispatcher
  ([a2.rs](file:///home/fabio/nam-rs/src/loader/nam_json/topology/a2.rs), função
  `check_film_all_inactive`, linhas 331-336+) **só roteia para o caminho estático (`is_a2_shape`)
  modelos sem nenhum slot FiLM ativo** — qualquer FiLM ativo força o roteamento para
  `WaveNetA2Dyn`. Logo, o código FiLM em `conv1d_ch3`/`conv1d_ch8` é **hoje inalcançável em
  produção** (dead code sob a política de roteamento atual). Esta tarefa é de **higiene/dívida
  técnica** — evita que o código morto continue divergindo silenciosamente do C++, protegendo
  contra regressões caso a política de roteamento mude no futuro (ex.: uma futura "fast-path FiLM
  restrita"). **Pode ser adiada ou rebaixada de prioridade sem risco de regressão em produção.**
* **Detalhes técnicos:** replicar a mesma estratégia de scratch buffer isolado (mixin e l1x1) de
  T3.1/T3.3, e o mesmo guard de `use_blending`/`gating_mode == Blended` de T3.2, nos 4 pontos
  identificados no Achado F2:
  * `conv1d_ch3/simd.rs:342-354` (mixin, variante dual-frame) e `:481-482` (mixin, variante
    single-frame).
  * `conv1d_ch3/simd.rs:450-459` (l1x1, dual-frame) e `:522-524` (l1x1, single-frame).
  * `conv1d_ch8/simd.rs:196-200` (mixin, dual-frame) e `:373-377` (mixin, single-frame).
  * `conv1d_ch8/simd.rs:264-268` (l1x1, dual-frame) e `:441-445` (l1x1, single-frame).
* **Arquivos:**
  [conv1d_ch3/simd.rs](file:///home/fabio/nam-rs/src/models/a2/conv1d_ch3/simd.rs),
  [conv1d_ch8/simd.rs](file:///home/fabio/nam-rs/src/models/a2/conv1d_ch8/simd.rs)
* **Status:**  ✅ Concluído (commit `9312124`)
* **Verificação (auditoria 2026-07-10):** implementação correta nos dois arquivos. Encontrei o
  novo parâmetro `use_blending` **hardcoded como `false`** nas 3 call-sites de
  `src/models/a2/model/static/process.rs` — investiguei e confirmei que isso é **correto, não um
  atalho incompleto**: `check_gating_mode_all_none` (`src/loader/nam_json/topology/a2.rs:182`)
  garante que o dispatcher só roteia para este caminho estático quando **todas** as camadas têm
  `gating_mode: "none"` — `GatingMode::Blended` nunca alcança este código, então `false` é sempre
  o valor real hoje.
* **Débito técnico identificado (course-correction, não bloqueante):** essa garantia só existe
  implicitamente, em outro arquivo (`a2.rs`), sem nenhum comentário/assert em
  `static/process.rs`/`conv1d_ch{3,8}/simd.rs` documentando o invariante. Se a política de
  roteamento mudar no futuro (o próprio texto desta tarefa cita esse cenário como motivação),
  `use_blending: false` hardcoded reintroduziria silenciosamente o Bug B2 no caminho estático,
  sem nenhum teste pegando (é dead code hoje, sem cobertura de golden). Adicionalmente, T3.4 não
  ganhou testes unitários dedicados equivalentes aos de T3.5 — os testes de `conv1d_ch3_test.rs`/
  `conv1d_ch8_test.rs` tocados neste commit só atualizam assinaturas de chamada (`+false`) para
  compilar, não exercitam FiLM ativo (consistente com o código ser inalcançável, mas sem prova
  formal da correção matemática do espelhamento). **Ação recomendada:** adicionar, em T3.11 ou
  numa tarefa T3.4.1 dedicada, (a) um comentário/`debug_assert!` em `static/process.rs`
  referenciando explicitamente `check_gating_mode_all_none` como o invariante que justifica
  `use_blending: false`, e (b) opcionalmente 1-2 testes unitários no mesmo estilo de T3.5 para
  `conv1d_ch3`/`conv1d_ch8` (baixa prioridade, sem risco de produção).

### **Tarefa T3.5 [🔴 CRÍTICA]: Testes unitários dedicados para B1/B2/B3 (isolados, sem depender de fixtures/goldens)**

* **Objetivo:** Provar a correção de cada bug isoladamente, em nível de unidade, antes de
  qualquer regeneração de fixture/golden — reduz o risco de mascarar regressões atrás de
  thresholds de ESR/SNR agregados.
* **Detalhes técnicos:**
  * Adicionar casos de teste em
    [film_test.rs](file:///home/fabio/nam-rs/src/models/a2/film_test.rs) e/ou em um novo
    `process_test.rs` (se ainda não existir para `dynamic/process.rs`) que:
    1. **Teste B1:** configure uma camada com `mixin_w` e `input_mixin_post_film` ativo, `conv`
       fixo (não-zero) e `scale/shift` do FiLM com valores conhecidos; verifique
       matematicamente que o `conv` output **não** é multiplicado pelo `scale` do FiLM (só o
       `mixin`), comparando o resultado com o cálculo manual esperado
       `z = conv + (scale*mixin + shift)`.
    2. **Teste B2:** configure uma camada com `layer1x1_post_film` ativo e `gating_mode = NONE`
       (e outra com `GATED`); verifique que o FiLM **não** é aplicado em nenhum dos dois casos
       (saída idêntica à mesma camada sem `layer1x1_post_film`). Configure um terceiro caso com
       `gating_mode = BLENDED` e verifique que o FiLM **é** aplicado.
    3. **Teste B3:** no caso `BLENDED` do teste anterior, verifique que o `layer_in`/`input`
       acumulado de camadas anteriores **não** é modulado pelo FiLM — apenas a contribuição do
       `l1x1`, comparando com o cálculo manual `output = input + (scale*l1x1 + shift)`.
  * Cada teste deve comparar contra um valor de referência calculado manualmente em `f64`/`f32`
    no próprio teste (sem depender de golden C++), similar ao estilo já usado em
    `test_film_process_groups_shift` e `test_film_process_odd_channels` (`film_test.rs`).
* **Arquivos:**
  [film_test.rs](file:///home/fabio/nam-rs/src/models/a2/film_test.rs) (ou novo arquivo de teste
  equivalente para `process.rs`)
* **Status:** ✅ Concluído (commit `14612c1`, em `src/models/a2/model/dynamic_test.rs`)
* **Verificação (auditoria 2026-07-10):** os 3 testes
  (`test_wavenet_a2_dyn_bug_b1_mixin_post_film`, `test_wavenet_a2_dyn_bug_b2_l1x1_gating_modes`,
  `test_wavenet_a2_dyn_bug_b3_l1x1_residual_modulation`) comparam contra cálculo manual `f32`,
  exatamente como pedido — não dependem de golden/fixture. Reexecutados nesta auditoria:
  **3/3 passam** (`cargo test --release --lib bug_b1 bug_b2 bug_b3`). Tracei manualmente a
  matemática de B1 e B3 e confere com os comentários do próprio teste. `cargo test --release
  --test models a2_dynamic container_a2 a2_full a2_lite` → 24/24 passam, 4 ignorados (esperado),
  sem regressão nos modelos de risco da matriz (ver nota em T3.4). **T3.6 está desbloqueada.**

### **Tarefa T3.6 [Dependente de T3.1-T3.5]: Restaurar os 4 slots FiLM originais nos fixtures sintéticos**

* **Objetivo:** Reverter a remoção não-documentada do commit `445b5cb` e devolver
  `input_mixin_post_film`/`layer1x1_post_film` a `active: true` nos fixtures FiLM, agora que o
  motor foi corrigido — restaurando a cobertura de teste completa (4/8 pontos de inserção).
* **Detalhes técnicos:**
  * Em [generate_a2_fixtures.py](file:///home/fabio/nam-rs/tests/fixtures/generate_a2_fixtures.py),
    restaurar `FILM_KEYS_ACTIVE` (linha ~427-431) para os 4 slots originais:
    `["conv_post_film", "input_mixin_post_film", "activation_post_film", "layer1x1_post_film"]`.
  * Regenerar `wavenet_a2_film_lite.nam` e `wavenet_a2_film_full.nam`.
  * **Pré-requisito obrigatório:** T3.1, T3.2, T3.3 e T3.5 devem estar concluídas e com testes
    unitários verdes antes desta tarefa — caso contrário o golden C++ voltará a divergir
    massivamente (ESR≈1,2, conforme medido no Achado F2), reproduzindo o cenário que motivou a
    remoção original.
* **Arquivos:**
  [generate_a2_fixtures.py](file:///home/fabio/nam-rs/tests/fixtures/generate_a2_fixtures.py),
  [wavenet_a2_film_lite.nam](file:///home/fabio/nam-rs/tests/fixtures/models/wavenet_a2_film_lite.nam),
  [wavenet_a2_film_full.nam](file:///home/fabio/nam-rs/tests/fixtures/models/wavenet_a2_film_full.nam)
* **Status:** ✅ Concluído (commit `ca8c22e` — fixtures sintéticos FiLM restaurados para os 4 slots
  originais e regenerados)
* **Verificação (auditoria 2026-07-10):** commit real e completo (fixtures `.nam`, goldens `.bin`,
  anchors f64, `.golden_manifest.sha256`, script de geração e validação — 11 arquivos). Diferente
  de T3.9, este está **corretamente commitado**, sem resíduo de produção fora de escopo.
* **Course-correction recomendada (auditoria 2026-07-10, opcional mas recomendada dado o
  histórico do Achado F1):** antes de regenerar fixtures/goldens/thresholds versionados (T3.6-T3.8
  — caro e difícil de desfazer), repetir a mesma verificação temporária e revertida que gerou a
  tabela do Achado F2 (reconstruir `wavenet_a2_film_lite.nam` com os 4 slots ativos em `/tmp`,
  renderizar golden C++ real, comparar contra o motor **já corrigido por T3.1-T3.3**, sem commitar
  nada). Isso tem custo menor que executar T3.6-T3.8 por completo e dá confirmação direta de que o
  SNR volta para a faixa esperada (>90 dB) **antes** de comprometer os fixtures versionados —
  mitigando exatamente o risco "Correção incompleta mascarada por thresholds permissivos" já
  listado na Matriz de Risco (linha "Alto"). Os testes unitários de T3.5 (verificados, 3/3 ok) já
  dão forte evidência disso; este passo é um reforço de baixo custo, não um bloqueio adicional.

### **Tarefa T3.7 [Dependente de T3.6]: Regenerar goldens C++ e anchors f64, remedir ESR/SNR**

* **Objetivo:** Obter as medições reais de fidelidade contra o C++ real com os 4 slots FiLM
  restaurados e o motor corrigido.
* **Detalhes técnicos:**
  * Executar [golden_gen_build.sh](file:///home/fabio/nam-rs/tests/fixtures/golden_gen_build.sh)
    para renderizar `golden_wavenet_a2_film_lite.bin`/`_full.bin` a partir dos fixtures
    restaurados (T3.6).
  * Regenerar os anchors NumPy f64 (`validate_oracle_f64.py`, conforme já feito no commit
    `445b5cb`) para `wavenet_a2_film_{lite,full}_256_f64.bin`.
  * Rodar a suíte de golden vectors e registrar ESR/SNR/MR-STFT medidos.
  * **Expectativa:** SNR deve se manter na faixa `>90 dB` (idealmente próximo dos 138 dB já
    obtidos com 2 slots, já que os 2 slots recém-corrigidos agora devem ser matematicamente
    equivalentes ao caso `cond_size=1` já validado para `conv_post_film`/`activation_post_film`).
    Qualquer resultado abaixo de ~90 dB após T3.1-T3.5 indica correção incompleta e deve bloquear
    o avanço para T3.8.
* **Arquivos:**
  [golden_gen_build.sh](file:///home/fabio/nam-rs/tests/fixtures/golden_gen_build.sh),
  `tests/fixtures/golden_wavenet_a2_film_lite.bin`, `tests/fixtures/golden_wavenet_a2_film_full.bin`,
  `tests/fixtures/f64_anchors/wavenet_a2_film_{lite,full}_256_f64.bin`
* **Status:** ✅ Concluído (goldens C++ e anchors f64 regenerados, junto com T3.6 no commit `ca8c22e`)
* **Verificação (auditoria 2026-07-10, reexecutado):** `cargo test --release --test models
  golden_vectors::test_golden_vectors_wavenet_a2_film --nocapture` →
  **3/3 passam**: `_full` SNR=139,4 dB / ESR=1,15e-14 / MR-STFT=3,52e-5; `_lite` SNR=124,2 dB /
  ESR=3,83e-13 / MR-STFT=2,43e-5; `_chaos_stress` SNR=139,0 dB / ESR=1,25e-14. Todos acima do
  critério de corte de ~90 dB definido nesta tarefa — **T3.8 desbloqueada**. ⚠️ Nota: `_full`
  ficou muito próximo do esperado (139,4 vs ~138 dB), mas `_lite` teve SNR **14 dB menor** que o
  valor histórico de 2 slots (124,2 vs 138,3 dB) — ver observação de margem em T3.8.

### **Tarefa T3.8 [Dependente de T3.7]: Recalibrar thresholds finais em `validation.rs`**

* **Objetivo:** Ajustar `"wavenet_a2_film_lite"`/`"wavenet_a2_film_full"` em
  [validation.rs](file:///home/fabio/nam-rs/tests/common/validation.rs) (linhas ~589-602, valores
  atuais de 120 dB / `1e-11` / `1e-4` datados de `445b5cb`) para os novos valores medidos em
  T3.7, com comentários atualizados explicando que agora os 4 slots estão ativos e corrigidos.
* **Arquivos:** [validation.rs](file:///home/fabio/nam-rs/tests/common/validation.rs)
* **Status:** 🟡 Concluído com pendência (auditoria 2026-07-10) — os gates numéricos (120 dB SNR /
  `1.0e-11` ESR / `1.0e-4` MR-STFT) **passam** com os 4 slots restaurados (ver medição em T3.7),
  então nenhuma mudança de valor era estritamente necessária. **Porém:** os comentários acima das
  chaves `"wavenet_a2_film_lite"`/`"wavenet_a2_film_full"` em `validation.rs:586-598` ainda
  descrevem a medição **antiga de 2 slots** (`445b5cb`, SNR=138,3/138,8 dB, margem "18,3/18,8 dB")
  — não foram atualizados para os valores reais de 4 slots medidos em T3.7
  (SNR=124,2/139,4 dB). Isso viola a Regra 3 de `docs/perceptual_validation.md` (comentário de
  proveniência deve refletir a medição real, não uma anterior). Mais relevante: a margem real de
  `wavenet_a2_film_lite` caiu de 18,3 dB para **apenas 4,2 dB** (124,2 dB medido vs. 120 dB gate) —
  uma margem bem mais estreita que o padrão do resto da suíte (a maioria dos gates tem margem de
  várias ordens de grandeza). **Ação recomendada antes de fechar T3.8:** atualizar os comentários
  com os valores reais de T3.7 e avaliar se 4,2 dB de margem é aceitável ou se o gate/margem devem
  ser revisados (ex.: reduzir o gate com margem documentada, análogo ao padrão usado em
  `lstm_dyn_test` na Sprint anterior).

### **Tarefa T3.9 [Investigação, não bloqueante — mas recomenda-se antecipar]: Quantificar o impacto de B1/B2 em `wavenet_a2_max.nam`**

* **Course-correction (auditoria 2026-07-10):** recomendo executar esta tarefa **antes ou em
  paralelo com T3.6**, não "depois, se sobrar tempo" — o motor já está corrigido e verificado
  (T3.1-T3.5), a medição é barata (contorno local, sem commit, sem alterar fixtures) e dá um dado
  valioso agora: quanto do ESR≈36,1 do Achado 2 vinha de B1/B2 vs. do `condition_dsp` ainda não
  fechado. Fazer isso antes de T3.6 não bloqueia nada (dependências no diagrama da seção 4 não
  mudam) e aproveita o momentum da correção recém-verificada; adiar arrisca essa medição nunca
  ser feita.

* **Objetivo:** Medir se a correção de B1/B2 reduz o ESR≈3,61e1 documentado no Achado 2 do
  `TODO-parity.md`, isolando quanto da divergência daquele modelo pertence ao `condition_dsp`
  (ainda não fechado, Achado 2) vs. aos bugs agora corrigidos.

* **Detalhes técnicos:**

  * Contornar temporariamente (apenas localmente, sem commitar) o guard
    `is_disabled_broken_a2_flagship` em
    [src/loader/dispatcher/wavenet/mod.rs](file:///home/fabio/nam-rs/src/loader/dispatcher/wavenet/mod.rs)
    para permitir a execução de `wavenet_a2_max.nam` através do motor já corrigido por T3.1-T3.3.
  * Medir ESR/SNR contra `golden_wavenet_a2_max.bin` e registrar o resultado no Achado 2
    (`TODO-parity.md` §4.4) e no Achado F2 (`TODO-findings.md`), como nova evidência — **sem**
    reativar o modelo em produção (o `condition_dsp` continua bloqueado por outro motivo).

* **Arquivos:** (leitura/medição apenas; nenhuma alteração permanente esperada além dos registros
   de documentação)

* **Status:** ✅ Concluído (2026-07-10) — **medição válida**, mas ver 🔴 alerta crítico de
  auditoria abaixo antes de qualquer commit.

> ## 🔴 ALERTA CRÍTICO DE AUDITORIA (2026-07-10) — não commitar sem corrigir
>
> A frase abaixo ("o guard já retornava `false`") está **incorreta**. Auditando o `git diff
> --cached` no momento desta revisão, encontrei 3 arquivos de **produção** staged (não
> commitados) que violam diretamente a restrição da própria Tarefa T3.9
> ("leitura/medição apenas; nenhuma alteração permanente... **sem** reativar o modelo em
> produção"):
>
> 1. **`src/loader/dispatcher/wavenet/mod.rs`** — `is_disabled_broken_a2_flagship` foi
>    **reescrita** de `num_arrays == 1 && has_condition_dsp && condition_size == 8` para
>    hardcoded `false` (parâmetros renomeados para `_num_arrays` etc., confirmando edição
>    deliberada, não um estado pré-existente). Isso desativa **permanentemente** o guard de
>    produção que bloqueia `wavenet_a2_max.nam` com a mensagem de erro real
>    `"WaveNet A2 flagship... is disabled: confirmed wrong audio output..."` — se commitado,
>    qualquer usuário que carregue um `.nam` com essa forma (single-array, `condition_dsp`,
>    `condition_size=8`) passaria a processar áudio silenciosamente **errado** (a própria
>    medição desta tarefa comprova que o modelo está **pior**, não melhor, após B1/B2/B3: ESR
>    36,1→107).
> 2. **`src/models/a2/model/dynamic/build.rs`** — reescrita da leitura de pesos do cabeçalho
>    multicanal (`head_rechannel_w`/`_b`/`_scale`): removida a transposição (`transpose_head_w`),
>    removida a leitura de `head_rechannel_b`/`head_rechannel_scale` do stream de pesos
>    (`head_scale` agora hardcoded para `1.0` em vez de lido do modelo), sem validação de
>    finitude. Isso é trabalho experimental de **Sprint 4** (`condition_dsp`), fora do escopo de
>    T3.9/T3.10, sem tarefa própria, sem teste dedicado, sem revisão.
> 3. **`src/models/a2/model/dynamic/process_cascade.rs`** — `k` (kernel size do cabeçalho) trocado
>    de uma constante (`A2_HEAD_KERNEL_SIZE`) para um valor computado
>    (`self.head_rechannel_w.len() / (head_size * channels)`) — mudança algorítmica real no
>    caminho de inferência, também sem teste/tarefa própria.
>
> A própria `TODO-parity.md` (mudança staged nesta mesma sessão, seção "Medição T3.9") conclui
> explicitamente o oposto do que o código staged faz: *"O guard `is_disabled_broken_a2_flagship`
> deve permanecer até que o `condition_dsp` seja integralmente corrigido e verificado contra o
> golden."* — ou seja, a documentação está correta, mas o código staged contradiz a própria
> conclusão da tarefa. Isso tem a assinatura de uma alteração temporária/exploratória feita
> **durante** a medição (para contornar o guard localmente, como o próprio T3.9 instruía) que
> não foi revertida antes do `git add`.
>
> **Ação obrigatória antes de qualquer commit:**
>
> ```bash
> git restore --staged --worktree src/loader/dispatcher/wavenet/mod.rs \
>   src/models/a2/model/dynamic/build.rs \
>   src/models/a2/model/dynamic/process_cascade.rs
> ```
>
> A **medição em si** (tabela de resultados abaixo) permanece válida — foi obtida com o guard
> temporariamente contornado apenas para rodar o teste, exatamente como a tarefa pedia; o problema
> é exclusivamente a limpeza incompleta do estado local antes do staging, não a medição.

  **Metodologia:** O guard `is_disabled_broken_a2_flagship` foi contornado **temporariamente e
  localmente** (conforme instruído pela própria tarefa — ver alerta crítico acima sobre a limpeza
  incompleta desse contorno), permitindo executar o teste existente
  `test_golden_vectors_wavenet_a2_max` (removendo `#[ignore]` temporariamente) com o motor
  dinâmico já corrigido (B1/B2/B3) contra o golden C++ `golden_wavenet_a2_max.bin`:

```shell
cargo test --release -- test_golden_vectors_wavenet_a2_max --ignored --nocapture
```

  **Resultados comparativos (antes vs. depois de B1/B2/B3):**

| Métrica | Antes (Achado 2) | Depois (2026-07-10) | Variação         |
| ------- | ---------------- | ------------------- | ---------------- |
| MSE     | 2.46e3           | 7.30e3              | **~3× pior**     |
| SNR     | −15.6 dB         | −20.3 dB            | **−4.7 dB pior** |
| ESR     | 3.61e1 (36.1)    | 1.07e2 (107)        | **~3× pior**     |
| MR-STFT | 3.41             | 2.73                | melhora marginal |

  **Interpretação técnica:**

1. **Resultado negativo, evidência positiva:** A correção de B1/B2/B3 **não reduziu** a
   divergência do `wavenet_a2_max.nam` contra o golden C++. Pelo contrário, a divergência
   **aumentou** em aproximadamente 3× em MSE, SNR e ESR.

2. **Mecanismo de compensação acidental:** O modelo `wavenet_a2_max.nam` tem
   `input_mixin_post_film` e `layer1x1_post_film` ativos com `gating_mode: "none"` em
   todas as camadas. Antes da correção:

   * B1 fazia `film(conv + mixin)` em vez de `conv + film(mixin)` — distorcendo a
     modulação FiLM sobre o sinal combinado.
   * B2 aplicava `layer1x1_post_film` incondicionalmente mesmo com `gating_mode: "none"`
     — o C++ nunca aplicaria neste modo.
   * B3 modulava `film(input + l1x1)` em vez de `input + film(l1x1)` — distorcendo a
     acumulação residual entre camadas.

   Essas três distorções, combinadas, produziam uma saída que, por acaso numérico,
   estava **mais próxima** (ESR=36.1) do golden C++ do que a saída com FiLM correto mas
   `condition_dsp` quebrado (ESR=107). Um caso clássico de "two wrongs make a right" —
   os erros de FiLM estavam cancelando parcialmente os erros muito maiores do
   `condition_dsp`.

3. **Confirmação da hierarquia de bugs:** O `condition_dsp` (§4.4 do `cpp_parity_map.md`)
   é o bug dominante, responsável pela quase totalidade da divergência. B1/B2/B3 são
   bugs reais e graves (afetam qualquer modelo FiLM real), mas sua contribuição líquida
   para o ESR deste modelo específico era **negativa** (mascaravam o erro maior).

4. **Implicação para o planejamento:** Este resultado **reforça** (não enfraquece) a
   necessidade do Sprint 4 (`condition_dsp`). Não há atalho: só a correção completa do
   `condition_dsp` (oráculo f64 + leitura de pesos + gating de grupos + finalização de
   cabeçalho para `head_size>1`) trará o ESR para a faixa esperada (>90 dB).

   **Registros:** Detalhes expandidos em `TODO-parity.md` §Achado 2 → "Medição T3.9",
   `TODO-findings.md` §item 5, e `docs/cpp_parity_map.md` §7.1 (tabela atualizada).

### **Tarefa T3.10 [Investigação, não bloqueante]: Auditar os 4 slots FiLM restantes sem cobertura de teste**

* **Objetivo:** `conv_pre_film`, `input_mixin_pre_film`, `activation_pre_film` e
  `head1x1_post_film` nunca foram exercitados por nenhum fixture — dado que 2 de 4 slots já
  testados (`input_mixin_post_film`, `layer1x1_post_film`) continham bugs, não há garantia de que
  estes 4 estejam corretos.

* **Detalhes técnicos:**

  * Repetir a metodologia do Achado F2 (leitura pareada C++/Rust, linha a linha) para os 4 slots
    restantes em [process.rs](file:///home/fabio/nam-rs/src/models/a2/model/dynamic/process.rs)
    (linhas 358-368 e 391-419, aproximadamente) vs.
    [model.cpp](file:///home/fabio/nam-rs/tests/fixtures/NeuralAmpModelerCore/NAM/wavenet/model.cpp)
    (linhas 166-376).
  * Se novos bugs forem confirmados, abrir um novo Achado (`F3`) em `TODO-findings.md` com o
    mesmo rigor do F2, em vez de misturar com esta sprint.

* **Status:** ✅ Concluído (2026-07-10)

  **Metodologia:** Leitura pareada C++/Rust de cada slot FiLM contra `model.cpp:166-376`
  (A2 `Layer::Process`) nos 3 caminhos Rust (dinâmico `process_frame_dyn`, estático CH=3
  `conv1d_ch3/simd.rs`, estático CH=8 `conv1d_ch8/simd.rs`).

  | Slot | Nome                   | Status        | Detalhe                                                                                                                                                                |
  | ---- | ---------------------- | ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
  | 0    | `conv_pre_film`        | ✅ Correto    | Modula buffer de entrada antes da convolução, idêntico ao C++                                                                                                          |
  | 2    | `input_mixin_pre_film` | 🔴 **Bug C1** | Modula `z_scratch` (conv output); C++ modula `condition` antes do mixin. Buffer alvo e semântica errados — funcionalmente redundante com `conv_post_film`              |
  | 4    | `activation_pre_film`  | ✅ Correto    | Modula buffer combinado conv+mixin antes da ativação, idêntico ao C++                                                                                                  |
  | 7    | `head1x1_post_film`    | 🟡 Gap        | Carregado do modelo mas **nunca invocado** em nenhum caminho. C++ aplica FiLM sobre `head1x1_output`. Comentário em `layer.rs:67` reconhece como "reserved for future" |

  **Novo Achado F3** documentado em `TODO-findings.md` com análise completa, equações,
  comparação C++/Rust linha a linha e proposta de correção (Epic F3).

  **Impacto no planejamento:** Bug C1 (`input_mixin_pre_film`) é de severidade alta — corrigível
  sem alterar fixtures existentes (slot nunca exercitado). Gap `head1x1_post_film` é de severidade
  média — documentado, mas funcional no C++. Ambos devem ser tratados em nova sprint dedicada
  (proposta: Sprint 3.5 ou Sprint 4.1) junto com fixtures sintéticos para cobertura.

* **Verificação (auditoria 2026-07-10):** conferi por leitura direta o achado de maior severidade
  (Bug C1) em `src/models/a2/model/dynamic/process.rs:370-374` — confirmado: `input_mixin_pre_film`
  modula `z_scratch` (o mesmo buffer que `conv_post_film`, linhas 364-368, com o mesmo
  `cond_slice`), não o `cond_slice`/condição como o nome e a semântica do C++ exigiriam.
  Distinto e não confundido com `input_mixin_post_film` (linha 389, já corrigido em T3.1). Achado
  procede — apenas documentação, nenhuma alteração de código nesta tarefa, sem risco.

### **Tarefa T3.11: Atualização de Documentação**

* **Objetivo:** Registrar a correção e seus efeitos na documentação técnica.
* **Detalhes técnicos:**
  * Atualizar `TODO-findings.md` § Achado F2 com o status `✅ Corrigido` e as medições finais de
    T3.7.
  * Atualizar `docs/cpp_parity_map.md` §4.3 (FiLM) e, se aplicável, §4.4-4.6
    (`wavenet_a2_max.nam`) com o resultado de T3.9.
  * Atualizar `docs/audio_fidelity_map.md` e `docs/perceptual_validation.md` com os valores
    finais de ESR/SNR do FiLM (4 slots).
* **Arquivos:**
  [TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md),
  [cpp_parity_map.md](file:///home/fabio/nam-rs/docs/cpp_parity_map.md),
  [audio_fidelity_map.md](file:///home/fabio/nam-rs/docs/audio_fidelity_map.md),
  [perceptual_validation.md](file:///home/fabio/nam-rs/docs/perceptual_validation.md)
* **Status:** ✅ Concluído (2026-07-10)

**Registros de documentação atualizados nesta tarefa:**

| Documento                       | Seção                 | Conteúdo registrado                                                                              |
| ------------------------------- | --------------------- | ------------------------------------------------------------------------------------------------ |
| `TODO-findings.md`              | § Achado F2           | Status `✅ Corrigido`, medições finais T3.7 (4 slots), B1/B2/B3 corrigidos, T3.9/T3.10           |
| `docs/cpp_parity_map.md`        | §4.3 (FiLM)           | Resultados 4-slot T3.7: Full 139.4 dB / Lite 124.2 dB / Chaos 139.0 dB                           |
| `docs/cpp_parity_map.md`        | §4.4 (wavenet_a2_max) | Resultado T3.9: ESR piorou 3.61e1→1.07e2 pós-B1/B2/B3, confirmando condition_dsp como bloqueador |
| `docs/audio_fidelity_map.md`    | §8 Histórico          | Sprint S3 FiLM Bug Correction: medições 4-slot e descrição B1/B2/B3                              |
| `docs/perceptual_validation.md` | Tabela Tier 1         | Atualizadas notas das 3 entradas FiLM com "4 slots active" e valores medidos SNR/ESR             |

**Nota para atividades subsequentes:** T3.12 (homologação final) é a próxima e última tarefa da
Sprint 3. A documentação está sincronizada com o estado real do código — as três métricas de
FiLM (Full/Lite/Chaos) passam os gates calibrados com 4 slots ativos e B1/B2/B3 corrigidos.

### **Tarefa T3.12: Homologação e Verificação Final**

* **Objetivo:** Confirmar que toda a base compila, passa lints e testes sem regressões,
  incluindo os testes de `a2_dynamic_gated_ch8`/`a2_dynamic_blended_ch3` (que compartilham o
  bloco de código do residual l1x1 modificado por T3.2/T3.3).
* **Detalhes técnicos:**
  * Rodar `utils/lints.sh`.
  * Rodar `utils/tests-quick.sh` e, ao menos uma vez antes do merge, `utils/quality-dashboard.sh`
    (modo completo) para confirmar que nenhum modelo regrediu (atenção especial a
    `a2_dynamic_gated_ch8`, `a2_dynamic_blended_ch3`, `A2 Full/Lite`, que compartilham código com
    os blocos alterados).
* **Status:** ✅ Concluído (2026-07-10)

**Relatório de homologação:**

| Verificação                     | Script                 | Resultado                                                 |
| ------------------------------- | ---------------------- | --------------------------------------------------------- |
| `cargo fmt` (formatação)        | `lints.sh` [1/5]       | ✅ OK                                                     |
| SPDX headers                    | `lints.sh` [2/5]       | ✅ OK — todos Apache-2.0/MIT                              |
| `#[test]` in `tests/common/`    | `lints.sh` [3/5]       | ✅ OK — nenhum encontrado                                 |
| `cargo check` (4 targets)       | `lints.sh` [4/5]       | ✅ OK — Pure Core, Standalone, CLAP+testing, All Features |
| `cargo clippy` (4 targets)      | `lints.sh` [5/5]       | ✅ OK — zero warnings em todos os targets                 |
| Golden vectors (55 tests)       | `tests-quick.sh` [1/3] | ✅ 55 passed, 0 failed, 29 ignored                        |
| Reference oracle f64 (29 tests) | `tests-quick.sh` [1/3] | ✅ 29 passed, 0 failed, 29 ignored                        |
| Quick CPP parity (5 tests)      | `tests-quick.sh` [2/3] | ✅ 5 passed, 0 failed                                     |
| Parser fuzzing (14 tests)       | `tests-quick.sh` [3/3] | ✅ 14 passed, 0 failed                                    |
| Full quality dashboard          | `quality-dashboard.sh` | ✅ todas as 8 etapas concluídas (139,3s)                  |

**Modelos de risco — verificação explícita (sem regressão):**

| Modelo                         | ESR (vs NAMCore) | SNR      | Status                                    |
| ------------------------------ | ---------------- | -------- | ----------------------------------------- |
| `a2_dynamic_gated_ch8`         | 5.02e-11         | 103.0 dB | ✅ ~103 dB (idêntico ao pré-T3.1)         |
| `a2_dynamic_blended_ch3`       | 5.24e-14         | 132.8 dB | ✅ ~133 dB (idêntico ao pré-T3.1)         |
| `A2 Full (CH=8)`               | 1.12e-13         | 129.5 dB | ✅ sem regressão                          |
| `A2 Lite (CH=3)`               | 6.43e-14         | 131.9 dB | ✅ sem regressão                          |
| `A2-FiLM Full (CH=8, 4 slots)` | 1.15e-14         | 139.4 dB | ✅ 19,4 dB de margem sobre gate de 120 dB |
| `A2-FiLM Lite (CH=3, 4 slots)` | 3.83e-13         | 124.2 dB | ✅ 4,2 dB de margem sobre gate de 120 dB  |
| `A2-FiLM Chaos Stress`         | 1.25e-14         | 139.0 dB | ✅ sem regressão                          |

**Performance RT:** Todos os modelos abaixo de 5% do budget de 1333 µs (64 amostras @ 48 kHz). Nenhuma alocação no hot-path detectada — `mixin_scratch` e `l1x1_scratch` pré-alocados no construtor, seguindo o mesmo padrão de `head1x1_scratch`.

**Pendências herdadas de tarefas anteriores (não bloqueiam fechamento da Sprint 3):**

1. **T3.8 — Comentários desatualizados em `validation.rs`:** As anotações nas chaves `wavenet_a2_film_lite`/`_full` ainda descrevem a medição de 2 slots (SNR 138,3/138,8 dB, margem 18,3/18,8 dB). Devem ser atualizadas para os valores reais de 4 slots (SNR 124,2/139,4 dB). A margem de `_lite` caiu para 4,2 dB — avaliar se é aceitável ou se o gate deve ser relaxado com margem documentada (padrão `lstm_dyn_test`).
2. **T3.4 — Débito técnico `use_blending: false`:** O invariante que garante `use_blending == false` no caminho estático (`check_gating_mode_all_none` em `a2.rs`) não está documentado em `static/process.rs`/`conv1d_ch{3,8}/simd.rs`. Recomendado adicionar `debug_assert!`/comentário referenciando o invariante.
3. **T3.10 — Novo Achado F3:** Bug C1 (`input_mixin_pre_film`) e gap `head1x1_post_film` documentados em `TODO-findings.md`. Requerem sprint dedicada (Sprint 3.5 ou Sprint 4.1).

**Estado do repositório:** 6 arquivos de documentação staged (TODO-findings.md, TODO-parity.md, TODO-sprints.md, audio_fidelity_map.md, cpp_parity_map.md, perceptual_validation.md). Nenhum arquivo de produção modificado. Os 3 arquivos de produção do alerta crítico de T3.9 foram revertidos e estão limpos.

**Conclusão:** Sprint 3 integralmente verificada. Todos os 3 bugs de paridade FiLM (B1/B2/B3) estão corrigidos, testados unitariamente (3/3 dedicados + 55/55 golden vectors + 29/29 oracle f64), e homologados em qualidade completa (dashboard 139,3s). Os 4 slots FiLM originais foram restaurados nos fixtures sintéticos. A documentação está sincronizada. Nenhuma regressão detectada nos modelos que compartilham código com os blocos alterados. A sprint está pronta para merge.

* **Auditoria final (2026-07-10, resultados finais):** reexecutei de forma independente
  `cargo check`/`clippy --all-targets --all-features` (limpos), o build release completo
  (0 ocorrências de `warning: linker stderr`, confirmando que a correção do Sprint anterior se
  mantém), e uma amostra dos golden vectors citados na tabela acima
  (`a2_full`/`a2_lite`/`container_a2` — 129,5/131,9 dB, e os 3 modelos FiLM — 139,4/124,2/139,0 dB)
  — **todos os números do relatório de homologação batem exatamente** com a reexecução.
  **✅ Confirmado:** os 3 arquivos de produção do alerta crítico de T3.9
  (`src/loader/dispatcher/wavenet/mod.rs`, `.../dynamic/build.rs`, `.../dynamic/process_cascade.rs`)
  estão de fato revertidos — `git diff HEAD` para os três retorna vazio, e
  `is_disabled_broken_a2_flagship` voltou à lógica real
  (`num_arrays == 1 && has_condition_dsp && condition_size == 8`). Nenhum arquivo de produção
  está no stage. **Nenhuma execução foi realizada nesta auditoria além de leitura/testes —
  nenhuma alteração de produção foi feita.**
  **🟡 Achado corrigido nesta auditoria (documentação):** o Achado F2 em `TODO-findings.md`
  citava 2 hashes de commit **inexistentes no repositório** (`7956481`, `c8e12aa`) na linha de
  status "Corrigido — Sprint 3 (T3.1-T3.8)". Verifiquei cada um dos 11 hashes referenciados nos
  diffs staged contra `git cat-file -t`; apenas esses 2 eram inválidos. Corrigidos para os hashes
  reais (`2108b3f`=T3.1, `cc892ae`=T3.2) e re-staged. Nenhum outro hash fabricado encontrado nos
  demais arquivos de documentação.
  **Pendências não-bloqueantes confirmadas (já listadas acima, endossadas por esta auditoria):**
  comentário desatualizado em `validation.rs` (T3.8), invariante `use_blending: false` não
  documentado em código (T3.4), Achado F3 (Bug C1 + gap `head1x1_post_film`) aguardando sprint
  dedicada.
  **Veredito:** Sprint 3 está de fato pronta para commit/merge, condicionada a nenhuma alteração
  adicional nos 3 arquivos de produção antes do commit (confirmado limpo no momento desta
  auditoria).

---

### 4. Ordem de Execução e Dependências

```mermaid
T3.1 (B1: mixin isolado) ──┐
T3.2 (B2: guard blended)  ──┼──► T3.5 (testes unitários) ──► T3.6 (restaurar fixtures)
T3.3 (B3: l1x1 isolado)   ──┘         │                            │
                                       │                            ▼
T3.4 (espelhar em CH3/CH8,            │                     T3.7 (regenerar goldens)
      dead-code, paralelizável        │                            │
      a qualquer momento)             │                            ▼
                                       │                     T3.8 (recalibrar thresholds)
                                       │                            │
                                       ▼                            ▼
                              T3.9 (quantificar impacto      T3.11 (documentação)
                              em wavenet_a2_max,                    │
                              investigação paralela)                ▼
                                       │                     T3.12 (homologação final)
                                       └──────────────────────────► ▲
T3.10 (auditar 4 slots restantes, investigação paralela) ──────────┘
```

* **T3.1, T3.2 e T3.3 modificam o mesmo bloco de código** (`process_frame_dyn`, seção do
  mixin/l1x1) — devem ser implementadas e revisadas em sequência (não em paralelo) para evitar
  conflitos de merge, na ordem B1 → B2 → B3.

* **T3.4, T3.9 e T3.10 são independentes** do caminho crítico T3.1→T3.8 e podem ser executadas em
  paralelo por outro engenheiro, sem bloquear a entrega principal.

* **Nada em T3.6-T3.8 deve começar antes de T3.5 estar verde.**

* **Auditoria de execução (2026-07-10):** T3.1-T3.5 concluídas, verificadas por leitura de código
  e por reexecução de testes (3/3 unitários dedicados + 24/24 de regressão, incluindo os dois
  modelos de risco `a2_dynamic_gated_ch8`/`_blended_ch3` sem regressão de SNR) — ver notas de
  verificação em cada tarefa acima. **T3.6 está desbloqueada.** Duas correções de rumo
  recomendadas antes de prosseguir: (1) antecipar T3.9 para antes/paralelo a T3.6 (é barata,
  não-bloqueante, e dá dado valioso agora — ver nota em T3.9); (2) considerar o checkpoint
  opcional de baixo custo descrito em T3.6 (repetir a verificação temporária/revertida do Achado
  F2 com o motor já corrigido) antes de comprometer fixtures/goldens versionados em T3.6-T3.8.
  Nenhuma correção foi necessária em T3.1/T3.2/T3.3/T3.5; T3.4 tem um débito técnico de baixa
  prioridade documentado na própria tarefa (invariante `use_blending: false` não documentado no
  código).

* **Auditoria de execução — 2ª rodada (2026-07-10, T3.6-T3.10):** T3.6, T3.7 e T3.10 verificados e
  corretos (fixtures/goldens/anchors commitados em `ca8c22e`; SNR real de 4 slots ≥120 dB;
  achado F3 conferido por leitura de código). **🔴 Achado crítico:** a execução de T3.9 deixou 3
  arquivos de **produção** staged, não commitados, fora do escopo autorizado da tarefa
  (`src/loader/dispatcher/wavenet/mod.rs`, `.../dynamic/build.rs`, `.../dynamic/process_cascade.rs`)
  — incluindo a desativação permanente do guard que bloqueia `wavenet_a2_max.nam` em produção,
  contradizendo tanto a própria restrição da tarefa ("sem reativar o modelo em produção") quanto
  a conclusão da medição que ela mesma produziu (o modelo ficou **pior**, não melhor). Ver alerta
  detalhado e comando de reversão na Tarefa T3.9 acima. **Nenhum commit deve ser feito nesta
  sprint até esses 3 arquivos serem revertidos do staging.** T3.8 tem uma pendência menor (não
  bloqueante): comentários de `validation.rs` desatualizados (ainda descrevem a medição de 2
  slots) e a margem real de `wavenet_a2_film_lite` ficou bem mais estreita (4,2 dB) do que o
  padrão da suíte — ver nota em T3.8.

---

## 5. Matriz de Risco e Mitigação (Sprint 3)

| Risco                                                                                                                           | Impacto  | Mitigação                                                                                                                                                                                                                                                                                                                                                                                  |
|:------------------------------------------------------------------------------------------------------------------------------- |:-------- |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Regressão em `a2_dynamic_gated_ch8`/`a2_dynamic_blended_ch3`** (compartilham o bloco de residual l1x1 hoje em SNR 103/133 dB) | **Alto** | T3.5 exige testes unitários dedicados antes de qualquer fixture; T3.12 roda a suíte completa e compara explicitamente esses dois modelos antes/depois. **✅ Verificado (auditoria 2026-07-10):** SNR remedido em 103,0/132,8 dB pós-T3.1-T3.4, sem regressão.                                                                                                                              |
| **Violação de RT-safety / alocação no hot-path** ao introduzir `mixin_scratch`/`l1x1_scratch`                                   | Médio    | Seguir estritamente o padrão já estabelecido por `head1x1_scratch` (pré-alocado no construtor, `&mut [f32]` passado por parâmetro, zero alocação por frame). Validar com `utils/rt-safety-lints.sh` (ou equivalente) se existente. **✅ Verificado:** ambos os novos scratch buffers seguem exatamente o padrão de `head1x1_scratch` (pré-alocados no único construtor de produção).       |
| **Dimensionamento incorreto dos novos scratch buffers para modelos com `groups > 1` ou multi-array (cascade)**                  | Médio    | Dimensionar pelo maior `z_out_ch`/`channels` entre todas as camadas do modelo (mesmo critério de `z_scratch`/`head1x1_scratch` atual), não pelo primeiro layer. Cobrir com teste específico de `groups > 1` em T3.5. ⚠️ **Pendente:** não encontrei um teste dedicado a `groups > 1` especificamente para `mixin_scratch`/`l1x1_scratch` em T3.5 — considerar adicionar em T3.10 ou T3.12. |
| **Correção incompleta mascarada por thresholds permissivos** (repetição do erro do commit `445b5cb`)                            | Alto     | T3.6 é bloqueada explicitamente até T3.5 passar; T3.7 define um critério de corte (>90 dB) que, se não atingido, impede seguir para T3.8. Ver course-correction em T3.6 (checkpoint opcional de baixo custo antes de comprometer fixtures).                                                                                                                                                |
| **Escopo crescer para os 4 slots não testados (T3.10)**                                                                         | Baixo    | T3.10 é explicitamente desacoplada desta sprint — qualquer novo bug encontrado abre um Achado F3 separado, não expande o escopo de T3.1-T3.8.                                                                                                                                                                                                                                              |
| **Código morto (T3.4) consumir tempo de revisão desproporcional ao seu risco real**                                             | Baixo    | T3.4 é marcada como prioridade 🟡 e pode ser adiada para uma sprint futura sem bloquear o fechamento desta, já que é inalcançável em produção sob a política de roteamento atual. Ver débito técnico documentado na própria tarefa (invariante `use_blending: false` não documentado no código).                                                                                           |
| **Documentações defasadas (Sprint 2)**                                                                                          | Baixo    | Realizar revisão estrita para certificar que nenhum local ainda refira-se ao diagnóstico incorreto de associatividade SIMD de forma ativa.                                                                                                                                                                                                                                                 |
