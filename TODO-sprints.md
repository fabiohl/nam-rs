<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-sprints.md — Planejamento Ágil (Paridade V2 Multi-SR)

Este planejamento divide a resolução dos problemas descritos em [TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md) em sprints e tarefas técnicas acionáveis.

---

## Épico 1 — Correção de Sanidade Espectral e LUFS em Paridades V2

Este épico visa desativar gates incompatíveis e ajustar tetos de tolerância para modelos dinâmicos e baseados em FiLM que possuem divergências de processamento legítimas e documentadas frente ao fallback Eigen C++.

### Sprint 1 — Ajuste de Gates e Tetos Absolutos (cpp_parity)

* **Risco**: Baixo. As alterações modificam apenas as regras de validação nos testes e não afetam o código do motor DSP de produção.
* **Complexidade**: Média. Requer mapeamento correto e verificação cuidadosa dos tipos de modelos no harness de teste.

#### Tarefa T1.1 — Desativação do LUFS Gate em Modelos A2 Dinâmicos

* **Descrição**: Atualizar as chamadas de `run_v2_multi_sr` em [cpp_parity.rs](file:///home/fabio/nam-rs/tests/cpp_parity.rs) para desativar a validação do LUFS gate para modelos dinâmicos/gated.
* **Ações**:
  * Alterar o parâmetro `check_lufs_gate` de `true` para `false` nas funções `live_cross_validation_v2_a2_dynamic_gated` e `live_cross_validation_v2_a2_dynamic_blended`.
* **Referência**: Finding 1 em [TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md).

#### Tarefa T1.2 — Implementação de ESR Cap Específico para Modelos FiLM

* **Descrição**: Detectar se o modelo sendo avaliado é um modelo FiLM no harness de teste do [cpp_parity.rs](file:///home/fabio/nam-rs/tests/cpp_parity.rs) e relaxar o teto absoluto de ESR.
* **Ações**:
  * Em `run_render_comparison`, verificar se `golden_name` ou `model_filename` contém `"film"`.
  * Se for um modelo FiLM, definir `esr_cap` para `0.08` (modo Live) ou `0.15` (modo HighFidelity) em vez de aplicar o cap padrão `ABSOLUTE_ESR_CAP_WAVENET` (de `6.23e-3`).
* **Referência**: Finding 2 em [TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md).

#### Tarefa T1.3 — Ajuste de Teto Abspectral (MR-STFT Cap) para Modelos FiLM

* **Descrição**: Adaptar o teto absoluto do gate de MR-STFT (`ABSOLUTE_MRSTFT_CAP`) para acomodar a divergência espectral legítima do modelo FiLM de 8 canais a 48000 Hz.
* **Ações**:
  * Se for um modelo FiLM, redefinir a variável/teto de MR-STFT (ou ajustar a lógica de cap para `mrstft_max`) para aplicar um teto de `1.20` ao invés do cap fixo padrão de `0.95`.
* **Referência**: Finding 3 em [TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md).

---

## Validação e Critérios de Aceitação

1. **Compilação sem Avisos**: O projeto deve compilar limpo com `cargo check` e `cargo clippy`.
2. **Execução de Paridade Local**: A execução individual dos três testes afetados com `--ignored` deve passar com sucesso:
   * `cargo test --release --test cpp_parity live_cross_validation_v2_a2_dynamic_gated -- --ignored`
   * `cargo test --release --test cpp_parity live_cross_validation_v2_wavenet_a2_film_lite -- --ignored`
   * `cargo test --release --test cpp_parity live_cross_validation_v2_wavenet_a2_film_full -- --ignored`
3. **Validação de Testes Rápidos**: A suíte `utils/tests-quick.sh` deve continuar passando integralmente.
