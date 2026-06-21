<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-sprints.md — Plano de Execução (Épicos C e D)

Este documento detalha o planejamento das Sprints e Tarefas Técnicas para os Épicos C e D derivados da auditoria Pós-Paridade (`TODO-audit.md`). As tarefas foram concebidas com o máximo de detalhamento, voltadas a especialistas de implementação (conforme a skill `planejador-arquiteto`).

---

## ÉPICO C — Robustez RT & Caça a Bugs

**Objetivo:** Garantir zero superfícies de panic na thread de áudio, mitigar riscos em código `unsafe`, adotar parsing defensivo e corrigir vulnerabilidades de versionamento (`.gitignore`).
**Criticidade Global:** 🟠 Média (Atenção redobrada ao F11, que afeta a thread real-time).

### SPRINT C.1 — Remoção de Panics e Unwraps no Hot-path (RT-Safety)

**Foco:** Tratar achados críticos que podem derrubar a thread de áudio (F11) ou falhar de forma não-tratada no parser (F14).

- [x] **Tarefa C.1.1 (F11): Converter `unreachable!()` no A2 fixo para fallback RT-safe**
  - **Especialista:** `implementador` / Especialista em DSP e Rust RT-safe.
  - **Arquivo alvo:** `src/models/a2/model/mod.rs` (linhas ~532-538).
  - **Ação:** No método `process()` do A2 Fixo (`WaveNetA2<CH>`), o ramo de fallback escalar do bloco `if cfg!(...)` possui um `unreachable!("A2 layers always have ch3 or ch8 conv; ...")`.
  - **Implementação:** Substituir a diretriz de pânico incondicional por um `debug_assert!(false, "...")` (para falhar ativamente em debug builds durante os testes) e, em runtime (`release`), implementar um fallback 100% RT-safe. Este fallback deve silenciar o buffer de saída do bloco atual ou retornar sem processar, nunca disparando pânico. Se apropriado, sinalizar uma flag interna de telemetria de erro.
  - **Risco:** Alta criticidade, viola a regra de ouro de RT-safety se o invariante falhar no futuro.

- [x] **Tarefa C.1.2 (Auditoria e Blindagem): Varrer hot-paths de inferência (process)**
  - **Especialista:** `implementador`.
  - **Alvos:** Todo o código nas pastas `src/models/**` e `src/dsp/pipeline/**`.
  - **Ação:** Realizar grep extensivo por `unwrap()`, `expect()`, `panic!()`, `unreachable!()` e indexações em array diretas sem assert (`[i]`) em funções executadas pela thread de áudio (`process`, `process_block`).
  - **Implementação:** Transmutar qualquer ocorrência perigosa para tratamentos seguros e resilientes. Para indexações que são provadamente seguras, deve-se adicionar um comentário de bloco documentando formalmente a garantia do invariante.
  - **Resultado:** Auditoria completa em 33 funções `process/process_block` em 17 arquivos. Nenhum `unwrap()/expect()/panic!` em produção. Encontrados e corrigidos 2 `unreachable!()` residuais em `src/models/a2/conv1d_ch3/simd.rs:203` e `src/models/a2/conv1d_ch3/mod.rs:156` (substituídos por `debug_assert!` + fallback de silenciamento). Hot-path livre de panics.

- [x] **Tarefa C.1.3 (F14): Eliminar `unwrap()` inseguro no parser de topologia**
  - **Especialista:** `implementador`.
  - **Arquivo alvo:** `src/loader/nam_json/topology.rs` (linha ~226).
  - **Ação:** Onde ocorre `first_channels.unwrap()`, a asserção de que os canais estão presentes é vulnerável a refatorações.
  - **Implementação:** Propagar o erro corretamente: converter a chamada para `ok_or_else(|| NamError::...)` (ou equivalente no ecossistema de erros atual) e repassá-lo via operador `?`.
  - **Justificativa:** É um cold-path (fase de parsing e load), sendo ideal fechar essa vulnerabilidade de segurança para prevenir corrupção via json adulterado.

### SPRINT C.2 — Proteção de Memória e Versionamento Seguro

**Foco:** Fechar as vulnerabilidades em cast unsafe (F13) e evitar perda de arquivos no Git (F12).

- [x] **Tarefa C.2.1 (F13): Proteger e checar cast `unsafe` do LSTM**
  - **Especialista:** `implementador`.
  - **Arquivo alvo:** `src/loader/dispatcher/lstm/weights.rs` (linhas ~56-61).
  - **Ação:** O uso da função `std::slice::from_raw_parts_mut` ocorre sob um buffer bruto sem validação de pré-condições sobre o tamanho esperado (`H4 * IH`).
  - **Implementação:** Antes do bloco `unsafe`, inserir validações fortes de estado, como um `assert_eq!(buf.len(), expected_len, "...")`, ou encapsular dentro de um construtor que valide os bounds de memória, evitando *Undefined Behavior* por falha de casting.

- [x] **Tarefa C.2.2 (F12): Corrigir regra global de `*.json` no `.gitignore`**
  - **Especialista:** `implementador`.
  - **Arquivo alvo:** `.gitignore` (raiz).
  - **Ação:** O arquivo de configuração proíbe track de json, arriscando a integridade dos fixtures.
  - **Implementação:** Adicionar a regra de exceção explícita `!tests/fixtures/**/*.json` para un-ignorar essa pasta ou escopar rigidamente o ignore json apenas à raiz da workspace/IDE. Garantir que `tests/fixtures/models/keras_unsupported.json` figure perfeitamente versionado.

---

## ÉPICO D — Saneamento da Suíte de Testes (Anti-placebo e Consolidação)

**Objetivo:** Erradicar testes classificados como "placebos" que inflacionam a cobertura sem prover detecção real, abolir o "skip incondicional" e consolidar asserts matemáticos contra tolerâncias justificadas.
**Criticidade Global:** 🟠 Média. **Cuidado Vital:** Não apagar ou invalidar propriedades fundamentais como determinismo (`MSE == 0`) e invariance de block-size contidas nestes testes.

### SPRINT D.1 — Erradicação de Testes Placebo (F6)

**Foco:** Refatorar, apertar bounds ou consolidar testes cujo único gate efetivo é genérico: `is_finite()` ou `abs() < 100.0`.

- [x] **Tarefa D.1.1 (F6): Restringir pass-gates e fortalecer prewarm do WaveNet**
  - **Especialista:** `implementador` / `revisor-auditor`.
  - **Arquivos alvo:** `tests/nam_infer_test.rs` (`test_wavenet_stability_*`), `tests/wavenet_prewarm_edge.rs`.
  - **Ação:** Identificar testes validando a saída via métricas nulas (finitude/limites arbitrariamente gigantes).
  - **Implementação:** Onde disponível referência C++, engatar comparações diretas de fidelidade paramétrica (MSE/ESR). Sem referência, extrair um teto (bound) correlacionado fisicamente ao sinal de excitação de entrada do teste, garantindo margens contidas (ex: saída não desvia acima de 4x do pico máximo injetado).

- [x] **Tarefa D.1.2 (F6): Rastreabilidade rígida em testes SPSC e A2 (a2_loader, spsc_pipeline)**
  - **Arquivos alvo:** `tests/a2_loader.rs` (`test_a2_*_finite_output`), `tests/spsc_pipeline.rs`.
  - **Ação:** Refatorar as verificações que aceitam qualquer output contanto que não seja NaN (ou abs < 100.0 com ruído 0.01).
  - **Implementação:** Substituir asserts de `abs < 100.0` por envelopes rígidos provando que o bloco de saída segue matematicamente a dinâmica alimentada ao pipeline de SPSC e RingBuffer sem overflow injustificado.

- [x] **Tarefa D.1.3 (F6): Consolidação para Dinâmicos / Edge Cases (lstm_model_dyn_validation)**
  - **Arquivos alvo:** `tests/lstm_model_dyn_validation.rs`.
  - **Ação:** Testes isolados para finitude como `test_model_dyn_no_panic_edge` geram volume sem garantia total.
  - **Implementação:** Agregar as verificações de não-pânico (No-Panic/Zero-Input) nos testes core determinísticos e aplicar gates matemáticos (comparação block/frame). O resultado será menor volume bruto de `#[test]`, porém com asserção estrita de confiabilidade.

### SPRINT D.2 — Reativação de Goldens e Fim dos Dead Tests (F7)

**Foco:** Substituir modelos "lite" de baixíssima eficácia e purgar o padrão `skip incondicional` de validações cross-C++.

- [ ] **Tarefa D.2.1 (F7): Elevar o Golden e a Validação de modelos Lite**
  - **Especialista:** `implementador`.
  - **Arquivos alvo:** `tests/threshold_calibration.rs`, Infraestrutura de goldens.
  - **Ação:** O modelo atual `BossWN-lite.nam` apresenta baixíssima acurácia (SNR de 0.9dB) e desabilita implicitamente a eficácia do gate anti-regressões.
  - **Implementação:** Substituir definitivamente este modelo por uma contra-parte do mundo real listada em `models-nondist` (candidato forte indicado pelo auditor: `EVH-5150-Lite.nam` ou similar - validado pelo `CATALOG.txt` como Real WaveNet Lite). Recalcular o SNR esperado no arquivo e registrá-lo utilizando a diretiva `// Measured:` obrigatória no projeto.

- [ ] **Tarefa D.2.2 (F7): Purgar Skip Incondicional em validações V2 Lite (`cpp_parity.rs`)**
  - **Arquivo alvo:** `tests/cpp_parity.rs` (linhas ~551-555 - teste `live_cross_validation_v2_wavenet_lite`).
  - **Ação:** O teste possui um hard `return` logando `"SKIP: ... known-divergent (T1.2)"`, efetivamente morto.
  - **Implementação:** Substituir por tratamento de paridade com o novo modelo Lite real da Tarefa D.2.1. Se a divergência V2 C++ para Lite ainda ocorrer fundamentalmente por conta do bug T1.2 de upstream, encapsular adequadamente através de uma flag explícita `#[cfg(feature = "known-divergent")]`, `#[ignore]` justificado ou assert específico em faixa perdoável, devolvendo seu status à "Validação Confiável".
