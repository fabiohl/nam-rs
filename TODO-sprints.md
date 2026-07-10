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
* **Status:** ⬜ Pendente
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
* **Status:** ⬜ Pendente

### **Tarefa T3.8 [Dependente de T3.7]: Recalibrar thresholds finais em `validation.rs`**

* **Objetivo:** Ajustar `"wavenet_a2_film_lite"`/`"wavenet_a2_film_full"` em
  [validation.rs](file:///home/fabio/nam-rs/tests/common/validation.rs) (linhas ~589-602, valores
  atuais de 120 dB / `1e-11` / `1e-4` datados de `445b5cb`) para os novos valores medidos em
  T3.7, com comentários atualizados explicando que agora os 4 slots estão ativos e corrigidos.
* **Arquivos:** [validation.rs](file:///home/fabio/nam-rs/tests/common/validation.rs)
* **Status:** ⬜ Pendente

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
* **Status:** ⬜ Pendente

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
* **Status:** ⬜ Pendente

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
* **Status:** ⬜ Pendente

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
* **Status:** ⬜ Pendente

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

---

## 5. Matriz de Risco e Mitigação (Sprint 3)

| Risco                                                                                                                           | Impacto  | Mitigação                                                                                                                                                                                                                          |
|:------------------------------------------------------------------------------------------------------------------------------- |:-------- |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Regressão em `a2_dynamic_gated_ch8`/`a2_dynamic_blended_ch3`** (compartilham o bloco de residual l1x1 hoje em SNR 103/133 dB) | **Alto** | T3.5 exige testes unitários dedicados antes de qualquer fixture; T3.12 roda a suíte completa e compara explicitamente esses dois modelos antes/depois. **✅ Verificado (auditoria 2026-07-10):** SNR remedido em 103,0/132,8 dB pós-T3.1-T3.4, sem regressão. |
| **Violação de RT-safety / alocação no hot-path** ao introduzir `mixin_scratch`/`l1x1_scratch`                                   | Médio    | Seguir estritamente o padrão já estabelecido por `head1x1_scratch` (pré-alocado no construtor, `&mut [f32]` passado por parâmetro, zero alocação por frame). Validar com `utils/rt-safety-lints.sh` (ou equivalente) se existente. **✅ Verificado:** ambos os novos scratch buffers seguem exatamente o padrão de `head1x1_scratch` (pré-alocados no único construtor de produção). |
| **Dimensionamento incorreto dos novos scratch buffers para modelos com `groups > 1` ou multi-array (cascade)**                  | Médio    | Dimensionar pelo maior `z_out_ch`/`channels` entre todas as camadas do modelo (mesmo critério de `z_scratch`/`head1x1_scratch` atual), não pelo primeiro layer. Cobrir com teste específico de `groups > 1` em T3.5. ⚠️ **Pendente:** não encontrei um teste dedicado a `groups > 1` especificamente para `mixin_scratch`/`l1x1_scratch` em T3.5 — considerar adicionar em T3.10 ou T3.12. |
| **Correção incompleta mascarada por thresholds permissivos** (repetição do erro do commit `445b5cb`)                            | Alto     | T3.6 é bloqueada explicitamente até T3.5 passar; T3.7 define um critério de corte (>90 dB) que, se não atingido, impede seguir para T3.8. Ver course-correction em T3.6 (checkpoint opcional de baixo custo antes de comprometer fixtures). |
| **Escopo crescer para os 4 slots não testados (T3.10)**                                                                         | Baixo    | T3.10 é explicitamente desacoplada desta sprint — qualquer novo bug encontrado abre um Achado F3 separado, não expande o escopo de T3.1-T3.8.                                                                                      |
| **Código morto (T3.4) consumir tempo de revisão desproporcional ao seu risco real**                                             | Baixo    | T3.4 é marcada como prioridade 🟡 e pode ser adiada para uma sprint futura sem bloquear o fechamento desta, já que é inalcançável em produção sob a política de roteamento atual. Ver débito técnico documentado na própria tarefa (invariante `use_blending: false` não documentado no código). |
| **Documentações defasadas (Sprint 2)**                                                                                          | Baixo    | Realizar revisão estrita para certificar que nenhum local ainda refira-se ao diagnóstico incorreto de associatividade SIMD de forma ativa.                                                                                         |
