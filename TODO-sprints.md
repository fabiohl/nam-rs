<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-sprints.md — Planejamento de Sprints e Tarefas Técnicas

Este documento detalha o planejamento ágil para resolução dos achados identificados em [TODO-findings.md](TODO-findings.md), com foco inicial no **Épico A — Confiabilidade do harness de paridade C++**.

---

## Sprint 1: Confiabilidade do Harness C++ (Épico A)

**Objetivo:** Garantir o isolamento e determinismo de testes concorrentes que alteram a precisão global de ativação (`ActivationPrecision`), calibrar e testar de forma efetiva os modos `Fast` vs `Standard` na suíte `quick_parity`, e otimizar o tempo e a poluição de logs do build C++ vendorizado.

**Risco:** Médio (as alterações são limitadas à infraestrutura de testes e build auxiliar, com risco zero de regressão sobre o áudio de produção).

---

### Tarefas Técnicas

#### Épico A.1 — Guard RAII Simétrico e Thread-Safe para `ActivationPrecision` (F9)

* **S1.T01 — Criação do modulo de precisão comum** `[x]`
  * **Ação:** Criar [precision.rs](file:///home/fabio/nam-rs/tests/common/precision.rs) com uma estrutura `PrecisionGuard` contendo:
    * Um `Mutex` estático (`PRECISION_MUTEX`) para serializar qualquer teste que altere a precisão de ativação global do processo.
    * Atributo `original_mode` para armazenar o valor antes da modificação.
    * Implementação de `Drop` que restaura `original_mode` de forma segura.
  * **Arquivos:** [precision.rs](file:///home/fabio/nam-rs/tests/common/precision.rs) `[NEW]`

* **S1.T02 — Registro do módulo comum** `[x]`
  * **Ação:** Registrar e reexportar o novo módulo em [mod.rs](file:///home/fabio/nam-rs/tests/common/mod.rs).
  * **Arquivos:** [mod.rs](file:///home/fabio/nam-rs/tests/common/mod.rs) `[MODIFY]`

* **S1.T03 — Auditoria e refatoração dos call-sites de `set_activation_precision`** `[x]`
  * **Ação:** Substituir chamadas diretas que alteram o atômico global pelo uso do `PrecisionGuard` para garantir exclusão mútua e restauração automática.
  * **Arquivos:**
    * [cpp_parity.rs](file:///home/fabio/nam-rs/tests/parity/cpp_parity.rs) `[MODIFY]`
    * [isa_parity.rs](file:///home/fabio/nam-rs/tests/parity/isa_parity.rs) `[MODIFY]`
    * [activation_precision.rs](file:///home/fabio/nam-rs/tests/models/activation_precision.rs) `[MODIFY]`
    * [lstm_activation_precision.rs](file:///home/fabio/nam-rs/tests/models/lstm_activation_precision.rs) `[MODIFY]`
    * [reference_oracle_f64.rs](file:///home/fabio/nam-rs/tests/parity/reference_oracle_f64.rs) `[MODIFY]`

---

#### Épico A.2 — Redefinir Semântica HF/não-HF dos `quick_parity_*` (F1)

* **S1.T04 — Atualização do pipeline de testes não-HF (modo `Fast`)** `[x]`
  * **Ação:** Configurar os testes não-HF para usar explicitamente `ActivationPrecision::Fast` via `PrecisionGuard` (anteriormente eles rodavam silenciosamente no modo default `Standard`).
  * **Arquivos:** [cpp_parity.rs](file:///home/fabio/nam-rs/tests/parity/cpp_parity.rs) `[MODIFY]`

* **S1.T05 — Atualização do pipeline de testes HF (modo `Standard`)** `[x]`
  * **Ação:** Configurar os testes HF (`quick_parity_hf_*`) para fixar explicitamente `ActivationPrecision::Standard`.
  * **Arquivos:** [cpp_parity.rs](file:///home/fabio/nam-rs/tests/parity/cpp_parity.rs) `[MODIFY]`

* **S1.T06 — Ajuste de limiares de erro (ESR) para o modo `Fast`** `[x]`
  * **Ação:** Como o C++ upstream não implementa aproximações Padé (roda sempre em modo exato), a comparação de Rust `Fast` vs C++ `Standard` resultará em um erro de aproximação conhecido de tanh (~2.3e-3). Ajustar os limites de tolerância de ESR (`ABSOLUTE_ESR_CAP`) nos testes rápidos que rodam em modo `Fast` para evitar falsos-positivos.
  * **Arquivos:** [cpp_parity.rs](file:///home/fabio/nam-rs/tests/parity/cpp_parity.rs) `[MODIFY]`

---

#### Épico A.3 — Correção de Comentários e Limiar do WaveNet HF (F1.1, F1.2)

* **S1.T07 — Correção do comentário descritivo** `[x]`
  * **Ação:** Corrigir a descrição em [cpp_parity.rs](file:///home/fabio/nam-rs/tests/parity/cpp_parity.rs) (linhas 1228-1241) para refletir que o C++ na verdade usa matemática exata (`std::tanh`), enquanto o Rust em modo `Fast` usa aproximações Padé.
  * **Arquivos:** [cpp_parity.rs](file:///home/fabio/nam-rs/tests/parity/cpp_parity.rs) `[MODIFY]`

* **S1.T08 — Ajuste de cap de erro no WaveNet HF** `[x]`
  * **Ação:** Corrigir a fórmula/limiar de tolerância de ESR do WaveNet HF (evitando o cap relaxado incoerente `* 5.0` e adotando um limite adequado de ~1e-10, compatível com a medição real de ~2.4e-14).
  * **Arquivos:** [cpp_parity.rs](file:///home/fabio/nam-rs/tests/parity/cpp_parity.rs) `[MODIFY]`

---

#### Épico A.4 — Otimização de Build C++ e Silenciamento de Warnings (F5, F10.3)

* **S1.T09 — Suprimir warnings do compilador C++ vendorizado** `[ ]`
  * **Ação:** Inserir a flag `-DCMAKE_CXX_FLAGS="-w"` nos processos de build do `render` C++ (em `ensure_render_compiled` e `golden_gen_build.sh`) para silenciar avisos `-Weffc++` que poluem desnecessariamente os consoles e logs de auditoria.
  * **Arquivos:**
    * [cpp_parity.rs](file:///home/fabio/nam-rs/tests/parity/cpp_parity.rs) `[MODIFY]`
    * [golden_gen_build.sh](file:///home/fabio/nam-rs/tests/fixtures/golden_gen_build.sh) `[MODIFY]`

* **S1.T10 — Compilação preventiva do binário `render` no script `tests-quick.sh`** `[ ]`
  * **Ação:** Executar a compilação de forma preventiva no script de validação de testes rápidos, garantindo que o tempo gasto de compilação ocorra de forma isolada antes do disparo de `cargo test`.
  * **Arquivos:** [tests-quick.sh](file:///home/fabio/nam-rs/utils/tests-quick.sh) `[MODIFY]`

---

## Verificação e Compliance

1. **Compilação Incremental:**
   * Executar `cargo check` e `cargo clippy --tests` para garantir a ausência de warnings.
2. **Execução de Testes Rápidos:**
   * Executar `utils/tests-quick.sh` e assegurar que todas as etapas terminem com sucesso (verde), verificando se o log `testes.log` está limpo de warnings C++.
3. **Não-Regressão de Performance:**
   * Assegurar que os testes não entrem em deadlock devido ao lock do mutex do `PrecisionGuard`.
