<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-convnet_parity.md — Lacuna de precisão ConvNet ↔ NAMcore (diagnosticada; correção proposta)

> **Status (2026-07-27): CAUSA RAIZ CONFIRMADA COM EVIDÊNCIA NUMÉRICA FECHADA.**
> A lacuna de 2.54e-5 **não é** divergência aritmética f32, fusão de BatchNorm, nem resíduo
> de build do golden. É exclusivamente uma **divergência de semântica de inicialização de
> estado (prewarm)** entre NAMcore e nam-rs, confinada às **primeiras 62 amostras** do render.
> A correção é cirúrgica (~1 função), de baixíssimo risco, e está especificada na seção
> "Correção proposta". Aguarda apenas decisão do Product Owner para execução.

Documento criado pela auditoria de compliance de 2026-07-14 e **integralmente reescrito**
pelo peer review de 2026-07-27 (skill `revisor-auditor`), que refutou o enquadramento
original e confirmou a causa raiz por triangulação NumPy/f64 contra o golden C++.

Agente de IA: o diagnóstico está concluído — **não reinvestigar do zero**. Resta executar a
correção proposta (seção "Plano de execução"), que só será atacada por decisão do Product
Owner do NAM-rs.

## O fato

| Medição                                              | Valor                                          |
| ---------------------------------------------------- | ---------------------------------------------- |
| ConvNet prod × golden C++ (`quick_parity_convnet`)   | **ESR 2.54e-5 · SNR 45.9 dB · MR-STFT 2.7e-3** |
| ConvNet prod × oráculo f64 (prewarm-paired)          | **ESR 3.57e-15 · −144.5 dB**                   |
| Âncora NumPy do oráculo ConvNet                      | ESR 5.23e-33 (−322.8 dB)                       |
| Demais famílias prod × golden C++                    | 1e-11 … 1e-14                                  |

O aparente paradoxo ("oráculo verde, referência vermelha") está **resolvido**: as duas
medições observam janelas diferentes do sinal. O oráculo mede apenas o **regime permanente**
(o teste `run_oracle_esr_paired` processa 24 000 amostras de warmup e compara **somente as
últimas 256** — `tests/parity/reference_oracle_f64.rs:431`), enquanto a comparação com o
golden C++ cobre o arquivo **inteiro de 2048 amostras, incluindo o transiente de partida**.
Toda a energia do erro está nesse transiente.

## Causa raiz (confirmada)

**O NAMcore renderiza goldens com prewarm-on-reset; o nam-rs inicializa o ConvNet com
histórico de zeros literais. As duas inicializações divergem exatamente durante o campo
receptivo da rede (63 amostras).**

1. **NAMcore (`tests/fixtures/NeuralAmpModelerCore/NAM/dsp.cpp`)**: `DSP::Reset()` chama
   `prewarm()` por padrão (`mPrewarmOnReset`, default `true`, linhas 130–139/20). O
   `prewarm()` (linhas 67–96) **processa `GetPrewarmSamples()` = 1 + Σdilations = 64
   amostras de silêncio através de toda a rede**. Como cada bloco ConvNet tem BatchNorm com
   offset `loc = β − scale·μ ≠ 0`, a resposta a entrada zero de cada bloco é
   `tanh(loc) ≠ 0` — ou seja, após o prewarm, os ring buffers internos contêm o
   **estado estacionário de entrada-zero** (não zeros). O render (`tools/render.cpp:148`)
   chama `Reset(sr, 64)` e **não descarta** as saídas do prewarm; o golden portanto começa
   já nesse estado estacionário.

2. **nam-rs (`src/models/convnet/model.rs:216-234` → `block.rs:192-230`)**:
   `ConvNetModel::prewarm()` delega a cada bloco um `prewarm_internal` que **escreve um
   frame de silêncio e o replica por todo o histórico da conv** — ou seja, enche cada ring
   buffer com **zeros literais**, sem propagar a resposta de silêncio através da cadeia
   (bloco *i* deveria receber o `tanh(loc)` do bloco *i−1*, não zero). É um cold start que o
   C++ **nunca** exibe no render.

3. **Consequência**: as duas saídas divergem apenas enquanto o histórico inicial coexiste na
   janela da convolução — as primeiras ~62 amostras (RF = Σ(k−1)·d = 63). A partir daí, os
   dois motores computam **a mesma matemática** (ver evidências abaixo: concordância de
   4.2e-9, o piso de quantização f32 do WAV golden).

Observação arquitetural: a família WaveNet não sofre disso porque seu prewarm
(`src/models/wavenet/model.rs:182-196` + `layer_array.rs:192-203`) **propaga** a resposta
de silêncio pela cadeia (array1 → array2) e faz backfill com os **valores computados**,
aproximando o mesmo estado estacionário do C++ — verificado empiricamente a 1e-14. O
ConvNet é a única família cujo prewarm não propaga nada.

## Evidências (reproduzível: script de triangulação NumPy/f64)

Replicação independente do forward ConvNet em NumPy f64 (pesos de `convnet_test.nam`,
semântica NAMcore lida de `convnet.cpp`/`conv1d.cpp`), comparada ao golden commitado
`tests/fixtures/golden_convnet_test.bin` (formato `[u32 N][f32×N input][f32×N output]`):

| Experimento                                                        | ESR vs golden C++ | max abs Δ  |
| ------------------------------------------------------------------ | ----------------- | ---------- |
| NumPy f64 **cold start** (histórico zero)                          | **2.5418412e-05** | 2.32e-3    |
| NumPy f64 **com prewarm de 64 zeros processados**                  | **1.7203856e-15** | 4.2e-9     |

Piso interno do modelo (não envolve o golden): ESR(NumPy f32 × NumPy f64, cold) =
1.20e-15 — o modelo é numericamente benigno; o f32 intrínseco está 10 ordens de magnitude
abaixo da lacuna observada.

* O ESR do cold start (2.5418412e-05) **reproduz exatamente** o valor medido no dashboard
  (2.54e-5; SNR −10·log10(2.54e-5) = 45.95 dB = 45.9 dB reportado). A medição live é,
  portanto, inteiramente explicada pelo transiente de inicialização.
* **Topografia do erro** (|golden − numpy cold|, janelas de 128): amostras 0–61 carregam
  ~100% da energia do erro (max 2.3e-3); amostra 62 em diante: ≤ 4.2e-9 (piso f32). 62 ≈
  RF−1 da rede — assinatura textual de divergência de estado inicial, não de aritmética.
* Valor da 1ª amostra: golden = −5.1883e-2 (DC estacionário de entrada-zero); cold start =
  −5.0148e-2. O "desvio de ganho" que motivou a expressão "ganho incerto" no estudo original
  era esse DC de inicialização, não ganho.
* **Invariância de ponto fixo**: prewarm com 62, 63, 64, 65, 128 ou 256 zeros produz estado
  idêntico (ESR 1.65e-15 nos 256 primeiros pontos, todos iguais). A correção é, portanto,
  **robusta a qualquer buffer size** (relevante para o plugin CLAP, onde o C++ processaria
  `ceil(64/N)·N` zeros — o estado final é o mesmo para qualquer contagem ≥ RF).
* Input do golden é bit-idêntico ao `stress_signal.wav` commitado (max diff 0.0) — sem
  hipótese de divergência de sinal de entrada.

## Veredito do peer review sobre o estudo anterior (2026-07-14)

O estudo original enquadrou a lacuna como *"a divergência está entre o NAMcore e a
matemática ideal — o C++ computa o ConvNet numa ordem de operações f32 diferente"* e a
classificou como escolha de fidelidade ("idêntico ao NAMcore" vs "idêntico à matemática").
**Esse enquadramento está refutado**:

1. **Magnitude impossível para reordenação f32.** sqrt(ESR 2.54e-5) ≈ 5e-3 de erro relativo
   RMS ≈ 42 000× o eps de f32. Reordenação de somas de 16 parcelas e diferenças de FMA
   produzem ~1e-7 relativo (ESR ~1e-14) — **7 ordens de magnitude abaixo** do observado.
   O modelo é numericamente benigno (var BN ≈ 1, scale ≈ 1, sem amplificação; piso f32
   medido 1.2e-15). Nenhuma ordem de acumulação explica 0.5% RMS.
2. **Hipótese 1 (fusão de BN nos pesos) — REFUTADA e desatualizada.** O Rust **não** funde
   BN nos pesos da conv: `ConvNetBlock` aplica conv → BN afim (`x·scale + loc`) → ativação,
   estruturalmente idêntico ao C++ (`block.rs:153-170` vs `convnet.cpp:71-91`). O comentário
   sobre `from_fused` descrevia um estado antigo do código. (Diferenças residuais — BN scale
   em f64→f32 no C++ vs f32 no Rust, FMA único vs mul+add — valem ≤ 1 ulp → ESR ~1e-14.)
3. **Hipótese 2 (ordem Eigen) — REFUTADA** pelo item 1 e pela concordância de 4.2e-9 em
   regime permanente.
4. **Hipótese 3 (resíduo de build do golden) — REFUTADA.** O golden commitado (IEEE-strict,
   F-X1) e o render live (que usa apenas `-w`, sem `-fno-fast-math` — divergência de harness
   registrada abaixo em "Lições") produzem a mesma assinatura 2.54e-5; as flags são
   imateriais para esta lacuna.
5. **Corolário**: não existe escolha de fidelidade a fazer. Em regime permanente, NAMcore e
   nam-rs já são **o mesmo motor** (4.2e-9). A paridade total exige apenas alinhar a
   inicialização — que é, ademais, o comportamento de produção correto (ver abaixo).

## Correção proposta (segura, paridade estrita NAMcore)

**Substituir o prewarm de preenchimento-por-zeros do `ConvNetModel` pelo processamento real
de silêncio através da cadeia — réplica exata de `nam::DSP::prewarm()`:**

```rust
// src/models/convnet/model.rs — substituir prewarm()/prewarm_internal()
#[cold]
pub fn prewarm(&mut self) {
    // Paridade NAMcore (`dsp.cpp:67-96`): o prewarm processa GetPrewarmSamples()
    // (= 1 + Σdilations = 64 para o formato flat) amostras de silêncio através
    // de toda a rede, deixando os estados internos no ponto fixo de entrada-zero
    // (com BN, tanh(loc) ≠ 0). Qualquer contagem ≥ RF produz o mesmo estado
    // (invariância verificada 2026-07-27), portanto o resultado é independente
    // de buffer size/chunking.
    let n = self.receptive_field_size + 1;
    let zeros = vec![0.0f32; n];
    let mut sink = vec![0.0f32; n * self.out_channels()];
    self.process(&zeros, &mut sink); // saída descartada — só o estado importa
}
```

Pontos de decisão e segurança:

* **Escopo cirúrgico**: toca apenas o caminho frio de inicialização. Hot path, pesos,
  layouts, kernels SIMD e APIs públicas permanecem intocados.
* **`ConvNetBlock::prewarm`/`prewarm_internal` tornam-se código morto** — remover
  (papel do especialista em resiliência: eliminar dead code). Testes de unidade que chamam
  `model.prewarm()` (`convnet_model_test.rs`) continuam válidos.
* **`prewarm_samples()`**: ajustar para `receptive_field_size + 1` (= 64), espelhando
  `ConvNet::GetPrewarmSamples()` = 1 + Σdil (hoje retorna 63; incoerência cosmética, pois o
  argumento é ignorado — alinhar e documentar).
* **Alocações**: dois `Vec` pequenos (≤ 64 + 64·out_ch floats) em caminho frio — espelha o
  próprio `prewarm()` do C++ (que aloca buffers) e o prewarm atual do bloco (que já aloca
  `vec![0.0f32; in_ch]`). Refinamento opcional (não bloqueante): pré-alocar scratch de
  prewarm no struct para reset sem alocação.
* **Benefício colateral em produção**: o comportamento atual (zeros literais) produz um
  transiente frio de ~63 amostras (~1.3 ms @ 48 kHz) a cada reset no plugin; o prewarm
  correto o elimina — esta é justamente a razão de existir do prewarm no NAMcore.
* **Alternativa rejeitada**: descartar/mascarar as primeiras 64 amostras na comparação do
  teste. O golden C++ (a referência) **mantém** a região de prewarm no arquivo; igualá-lo
  bit a bit é estritamente mais forte, e mascarar enfraqueceria a suíte permanentemente.
* **Impacto em outros testes**: oráculo f64 (janela de regime permanente) — inalterado;
  self-golden determinism (duas instâncias) — inalterado; soak/RT — inalterados (só as
  primeiras 63 amostras pós-reset mudam).

## Plano de execução (quando o PO priorizar)

1. **Código** (commit isolado): aplicar a correção acima + remover o prewarm de bloco morto
   * alinhar `prewarm_samples()`.
2. **Validação**: rodar `quick_parity_convnet`, `live_cross_validation_convnet`,
   `test_golden_vectors_convnet_test`, suíte oráculo (`reference_oracle_f64::convnet`) e
   `utils/tests-quick.sh`. **Expectativa fundamentada**: ESR ≈ 1e-14…1e-15, SNR ≈ 130+ dB,
   MR-STFT ≈ 1e-5 (o degrau NumPy já validou 1.72e-15; o f32 do motor adiciona o piso usual
   das demais famílias). Registrar os valores reais medidos.
3. **Recalibração de gates** (commit separado, com os valores medidos em 2):
   * `tests/common/validation.rs:913` (`convnet_test`): de SNR 35 dB / ESR 1e-4 / MR-STFT
     0.03 para o piso de família — proposta inicial **SNR ≥ 120 dB, ESR ≤ 1e-12,
     MR-STFT ≤ 1e-4** (margem ~10–20 dB sobre o medido, conforme política de calibração).
   * `tests/parity/cpp_parity.rs:415` (`ABSOLUTE_ESR_CAP_CONVNET_HF`): de 1e-3 para
     **1e-10**, alinhando ao cap do WaveNet.
4. **Guarda de regressão**: o gate de janela completa do item 3 já captura qualquer
   regressão de prewarm para sempre. Adicionalmente, teste unitário novo:
   pós-`prewarm()`, processar 64 zeros deve produzir saída constante (DC estacionário)
   igual à constante obtida por convergência explícita (ponto fixo).
5. **Docs**: `docs/cpp_parity_map.md` — veredito ConvNet: **corrigido (inicialização de
   estado, não aritmética)**; regenerar `docs/quality-contract.txt`; limpar comentários
   desatualizados ("no C++ golden available / v0.5.3 incompatible" em `validation.rs:980` e
   `golden_vectors.rs:2008`); mover este arquivo para apêndice "resolvido" ou deletá-lo com
   ponteiro para o commit.
6. **Rollback**: reverter ambos os commits (o gate novo falharia com o comportamento
   antigo, por construção — rollback é o par commits 1+3).

**Risco: BAIXO.** A correção foi pré-validada de forma independente (NumPy/f64) contra o
golden real; o motor em regime permanente já concorda com o C++ a 4.2e-9.

## Lições de processo (auditoria)

1. **Calibração de gate não pode canonizar anomalia.** O gate de 35 dB foi calibrado sobre a
   primeira medição (45.9 dB) sem o sanity check contra o piso de família (100–140 dB das
   demais arquiteturas). Um desvio de ~90 dB abaixo da família deveria ter disparado
   investigação de causa raiz, não calibração. Proposta de emenda à Gate Calibration Policy:
   gate mais frouxo que o piso de família por > 20 dB exige causa raiz documentada.
2. **Comentários desatualizados custaram meses de diagnóstico errado**: "v0.5.3
   incompatível" (o golden flat já existia desde T4.7) e "BN fundida nos pesos" (já não era
   verdade no código) direcionaram o estudo original para hipóteses de aritmética.
3. **Divergência de harness registrada**: `golden_gen_build.sh` compila o render com
   `-fno-fast-math -ffp-contract=off` (F-X1), mas `cpp_parity.rs::ensure_render_compiled`
   usa apenas `-DCMAKE_CXX_FLAGS=-w`. Imaterial para esta lacuna (verificado), mas a
   divergência deve ser eliminada para que live e golden sejam estritamente o mesmo binário.
4. **Invariante de projeto a registrar**: todo motor nam-rs deve replicar a semântica
   `DSP::Reset` + `prewarm()` do NAMcore (processar ≥ RF amostras de silêncio pela cadeia).
   WaveNet/LSTM/A2 já a satisfazem empiricamente; ConvNet era a exceção.

## Por que estava diferido — e por que agora é barato

O diferimento original assumia esforço de "bissecção C++↔Rust por camada" desproporcional.
O peer review de 2026-07-27 **concluiu a bissecção analiticamente** (sem instrumentar o
C++): a topografia do erro + a triangulação NumPy fecharam a causa raiz. Restam ~15 linhas
de código, uma recalibração de gates e docs — custo de uma tarefa pequena, com ganho de
paridade total (ConvNet entra na banda "IDÊNTICO" do dashboard) e eliminação do transiente
de reset no plugin. Recomenda-se repriorizar.
