<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-convnet_parity.md — Lacuna de precisão ConvNet ↔ NAMcore (diferido)

Documento especializado criado pela auditoria de compliance de 2026-07-14 para registrar, de
forma individualizada, uma questão **deliberadamente diferida**: ganho incerto, arquitetura
secundária ao objetivo principal do NAM-rs (WaveNet/LSTM para amp modeling), e nenhuma
evidência de audibilidade. Atacar em momento oportuno.

Agente de IA: Ainda não tente resolve-lo. É lícito tomar conhecimento dele e contribuir com novos aprendizados.
Mas ele só será atacado em momento oportuno por decisão do Product Owner do NAM-rs.

## O fato

Com o suporte ao formato flat C++ do ConvNet (T4.7) e goldens IEEE-strict regenerados, a
paridade real ConvNet↔NAMcore ficou mensurável e é **ordens de magnitude mais frouxa** que a
das demais famílias:

| Medição (run 2026-07-14)                           | Valor                                          |
| -------------------------------------------------- | ---------------------------------------------- |
| ConvNet prod × golden C++ (`quick_parity_convnet`) | **ESR 2.54e-5 · SNR 45.9 dB · MR-STFT 2.7e-3** |
| ConvNet prod × oráculo f64 (prewarm-paired)        | **ESR 3.57e-15 · −144.5 dB**                   |
| Âncora NumPy do oráculo ConvNet                    | ESR 5.23e-33 (−322.8 dB)                       |
| Demais famílias prod × golden C++                  | 1e-11 … 1e-14                                  |

## A leitura correta

A produção Rust está **matematicamente ideal** (piso −144.5 dB vs f64, âncora NumPy
essencialmente exata). A divergência de 45.9 dB está, portanto, **entre o NAMcore e a
matemática ideal** — o C++ computa o ConvNet numa ordem de operações f32 diferente. Não é bug
do NAM-rs; é uma escolha de fidelidade: "idêntico ao NAMcore" vs "idêntico à matemática".
Como o NAMcore é o alvo declarado do projeto, a paridade máxima exigiria reproduzir a ordem de
operações do C++, mesmo sendo menos precisa.

## Hipóteses de causa (a investigar quando atacado)

1. **Fusão de BatchNorm:** o Rust pré-funde BN nos pesos da conv
   (`src/models/convnet/batch_norm.rs`, `from_fused`: `scale = γ/√(σ²+ε)` aplicado ao peso);
   o C++ (`NAM/convnet.cpp`) avalia a BN como passo separado pós-conv em f32. Duas ordens de
   arredondamento distintas — candidata principal para os ~46 dB.
2. **Ordem de acumulação Eigen:** o C++ usa GEMM/colwise do Eigen; o Rust usa kernels próprios
   com somas em ordem diferente (e Kahan em alguns caminhos).
3. **Resíduo de build do golden:** os goldens foram regenerados com IEEE-strict
   (F-X1, `-fno-fast-math`), mas conferir se o alvo do render do ConvNet herdou os flags em
   todas as translation units (verificar fingerprint de toolchain no manifest).

## Plano de ataque (quando priorizado)

1. Reproduzir a BN **não fundida** num caminho de referência Rust (fora do hot path) e medir
   ESR vs golden C++: se cair para ≤1e-10, a causa é a fusão → decidir entre (a) computar a
   fusão em f64 e arredondar no final (barato, pode recuperar dB sem tocar o hot path),
   (b) modo de compatibilidade não fundido, ou (c) aceitar e documentar.
2. Se a fusão não explicar, diferenciar por bissecção de camada (dump intermediário por bloco
   vs instrumentação equivalente no C++ `render`).
3. Recalibrar `convnet_test` (`tests/common/validation.rs`) ao novo piso medido e atualizar
   `docs/cpp_parity_map.md` com o veredito (intrínseco vs corrigido).

## Por que diferido

* ESR 2.54e-5 está na faixa "audível apenas com A/B científico" (legenda do dashboard), longe
  da zona vermelha; nenhum relato ou evidência perceptual.
* ConvNet é raro no ecossistema NAM real (WaveNet domina os captures publicados).
* O esforço de bissecção C++↔Rust por camada é desproporcional ao ganho enquanto existirem
  itens de correção de áudio em zona audível.

## Gatilhos para repriorizar

* Surgimento de modelos ConvNet reais relevantes na comunidade.
* Fechamento do EP-A (libera a metodologia de triangulação e o oráculo plenamente confiável para servir de instrumento na bissecção).
* Qualquer medição futura do ConvNet pior que SNR 40 dB (hoje o gate é 35 dB — margem 10.9 dB).
