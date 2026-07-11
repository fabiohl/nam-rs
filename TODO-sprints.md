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

* **S1.T09 — Suprimir warnings do compilador C++ vendorizado** `[x]`
  * **Ação:** Inserir a flag `-DCMAKE_CXX_FLAGS="-w"` nos processos de build do `render` C++ (em `ensure_render_compiled` e `golden_gen_build.sh`) para silenciar avisos `-Weffc++` que poluem desnecessariamente os consoles e logs de auditoria.
  * **Arquivos:**
    * [cpp_parity.rs](file:///home/fabio/nam-rs/tests/parity/cpp_parity.rs) `[MODIFY]`
    * [golden_gen_build.sh](file:///home/fabio/nam-rs/tests/fixtures/golden_gen_build.sh) `[MODIFY]`

* **S1.T10 — Compilação preventiva do binário `render` no script `tests-quick.sh`** `[x]`
  * **Ação:** Executar a compilação de forma preventiva no script de validação de testes rápidos, garantindo que o tempo gasto de compilação ocorra de forma isolada antes do disparo de `cargo test`.
  * **Arquivos:** [tests-quick.sh](file:///home/fabio/nam-rs/utils/tests-quick.sh) `[MODIFY]`

---

## Sprint 2: Metodologia do Oráculo f64 (Épico C)

**Objetivo:** Eliminar assimetrias de inicialização (cold-start) nas decomposições e tabelas do oráculo f64, mapear o oráculo WaveNet A2 corrigido e assegurar a rastreabilidade via meta-coerência do build.

**Risco:** Baixo (alterações focadas na infraestrutura de testes/oráculos off-RT e ferramentas de validação estática).

---

### Épico C.1 — Decomposições Pareadas com Prewarm para WaveNet/LSTM/A2/ConvNet (F3)

* **S2.T01 — Migrar Decomposições para `run_decomposition_paired`** `[x]`
  * **Ação:** Refatorar os testes de decomposição no arquivo `tests/parity/reference_oracle_f64.rs` (`test_decomposition_wavenet`, `test_decomposition_lstm`, `test_decomposition_a2`, `test_decomposition_convnet`) para que utilizem `run_decomposition_paired` com sinal de estresse e `WARMUP_LEN = 24_000` e `MEASURE_LEN = 4_096`. Isso garante que o cálculo de ESR por componente ocorra em regime permanente.
  * **Arquivos:** [reference_oracle_f64.rs](file:///home/fabio/nam-rs/tests/parity/reference_oracle_f64.rs) `[MODIFY]`

---

### Épico C.2 — Tabela de Resumo de ESR Pareada e Simétrica (F3)

* **S2.T02 — Atualizar `test_summary_table`** `[x]`
  * **Ação:** Refatorar a tabela de resumo no teste `test_summary_table` para consumir a metodologia de medição pareada (através do helper `run_oracle_esr_paired` com 24k de warmup + 256 de sweep), alinhando a fidelidade real relatada com os limites de tolerância física dos modelos.
  * **Arquivos:** [reference_oracle_f64.rs](file:///home/fabio/nam-rs/tests/parity/reference_oracle_f64.rs) `[MODIFY]`

---

### Épico C.3 — Correção Estrutural do Oráculo WaveNet A2 (F4)

* **S2.T03 — Correção do Oráculo WaveNet A2** `[x]`
  * **Ação:** Corrigir os caminhos de cálculo e dimensionalidades associadas a `condition_dsp` e leitura de pesos de `head1x1` no oráculo f64 em `src/testing/reference_oracle/a2.rs` conforme detalhado no Épico 6 do `TODO-wavenet_a2_max.md` (não-bloqueante para produção).
  * **Arquivos:** [a2.rs](file:///home/fabio/nam-rs/src/testing/reference_oracle/a2.rs) `[MODIFY]`

---

### Épico C.4 — Mensagens de Ignore e Rastreabilidade de Metamodelo (F4.1, F4.2)

* **S2.T04 — Atualizar Mensagens descritivas do `#[ignore]`** `[x]`
  * **Ação:** Modificar os comentários dos atributos `#[ignore]` nos testes de oráculo A2 Generic em `reference_oracle_f64.rs` para refletir as causas-raiz reais identificadas e referenciar explicitamente o plano `TODO-wavenet_a2_max.md`.
  * **Arquivos:** [reference_oracle_f64.rs](file:///home/fabio/nam-rs/tests/parity/reference_oracle_f64.rs) `[MODIFY]`

* **S2.T05 — Estender Cobertura de Rastreamento em `meta_coherence`** `[x]`
  * **Ação:** Incluir o arquivo de testes do oráculo `tests/parity/reference_oracle_f64.rs` na lista de arquivos escaneados pelo meta-teste `test_ignored_models_are_in_catalog` para garantir que qualquer modelo `.nam` mencionado em testes ignorados do oráculo f64 esteja formalmente registrado no catálogo de goldens do build.
  * **Arquivos:** [meta_coherence.rs](file:///home/fabio/nam-rs/tests/models/meta_coherence.rs) `[MODIFY]`

---

## Sprint 3: Metrologia MR-STFT (Épico B)

**Objetivo:** Implementar o condicionamento da métrica MR-STFT com máscara de piso de ruído baseada no pico do frame espectral, regerar as âncoras golden bit-a-bit e revalidar os testes de paridade; calibrar o soft gate e integrar a família Linear na metrologia; unificar a nomenclatura e documentar expoentes de relaxação de `mrstft_max`; e introduzir um modo silencioso no harness para evitar o vazamento de relatórios e mensagens de pânico nos logs de meta-testes.

**Risco:** Alto (alterações na métrica de fidelidade espectral exigem regeração sincronizada dos goldens Python e calibração meticulosa para evitar quebra silenciosa de cobertura ou falsos alarmes persistentes).

---

### Épico B.1 — Condicionamento Relativo da Métrica MR-STFT e Regeração de Golden (F2)

* **S3.T01 — Condicionamento da Métrica por Frame em `compute_mr_stft`** `[ ]`
  * **Ação:** Refatorar a computação em `src/testing/perceptual.rs` para substituir o piso absoluto `eps = 1e-8` por um piso relativo ao pico espectral do frame (ex.: −80 dB em relação ao pico do frame).
  * **Arquivos:** [perceptual.rs](file:///home/fabio/nam-rs/src/testing/perceptual.rs) `[MODIFY]`

* **S3.T02 — Espelhamento no Script Python e Regeração de `mrstft_golden.bin`** `[ ]`
  * **Ação:** Criar ou atualizar o script `tests/fixtures/scripts/gen_mrstft_golden.py` com o mesmo piso relativo do Rust, e rodar o script para gerar um novo `tests/fixtures/mrstft_golden.bin` determinístico e bit-a-bit idêntico.
  * **Arquivos:**
    * [gen_mrstft_golden.py](file:///home/fabio/nam-rs/tests/fixtures/scripts/gen_mrstft_golden.py) `[NEW]`
    * [mrstft_golden.bin](file:///home/fabio/nam-rs/tests/fixtures/mrstft_golden.bin) `[MODIFY]`

---

### Épico B.2 — Calibração do Soft Gate e Cobertura da Família Linear (F2.1, F2.2)

* **S3.T03 — Calibração de `MRSTFT_SOFT_THRESHOLD` no Metateste de Calibração** `[ ]`
  * **Ação:** Calibrar empiricamente e documentar o limiar do soft gate de MR-STFT, integrando-o ao escrutínio automático em `tests/models/threshold_calibration.rs`.
  * **Arquivos:**
    * [validation.rs](file:///home/fabio/nam-rs/tests/common/validation.rs) `[MODIFY]`
    * [threshold_calibration.rs](file:///home/fabio/nam-rs/tests/models/threshold_calibration.rs) `[MODIFY]`

* **S3.T04 — Mapeamento e Habilitação do Gate para Modelos Linear** `[ ]`
  * **Ação:** Adicionar mapeamento de goldens Linear FFT em `golden_bin_to_model_name()` e configurar limiares de `mrstft_max` realísticos (não-`None`) para a família Linear no harness.
  * **Arquivos:**
    * [validation.rs](file:///home/fabio/nam-rs/tests/common/validation.rs) `[MODIFY]`
    * [threshold_calibration.rs](file:///home/fabio/nam-rs/tests/models/threshold_calibration.rs) `[MODIFY]`
    * [linear_fft_test.rs](file:///home/fabio/nam-rs/tests/models/linear_fft_test.rs) `[MODIFY]`

---

### Épico B.3 — Higiene de Rótulos e Rationale de Relaxação (F2.3, F2.4)

* **S3.T05 — Rótulo da Métrica de `(relative)` para `(log-mag abs)`** `[ ]`
  * **Ação:** Corrigir a identificação visual impressa pelo harness de teste para que declare a métrica de forma precisa e coerente com a implementação final.
  * **Arquivos:** [validation.rs](file:///home/fabio/nam-rs/tests/common/validation.rs) `[MODIFY]`

* **S3.T06 — Unificação e Documentação da Relaxação de `mrstft_max`** `[ ]`
  * **Ação:** Unificar o expoente de relaxamento de `mrstft_max` (`/5.0` vs `/10.0`) e documentar o rationale físico/espectral associado no harness e na especificação.
  * **Arquivos:**
    * [golden_vectors.rs](file:///home/fabio/nam-rs/tests/models/golden_vectors.rs) `[MODIFY]`
    * [perceptual_validation.md](file:///home/fabio/nam-rs/docs/perceptual_validation.md) `[MODIFY]`

---

### Épico B.4 — Modo Silencioso e Supressão de Pânico em Metatestes (F7)

* **S3.T07 — Modo Silencioso no Harness para Testes de Regressão** `[ ]`
  * **Ação:** Implementar controle de supressão de relatório (ex.: flag thread-local `SUPPRESS_REPORT` ou estrutura similar) e registrar panic hook temporário em `test_mrstft_hard_gate_catches_regression` para impedir poluição com símbolos "✗" e "panicked" na suíte de testes verde.
  * **Arquivos:**
    * [validation.rs](file:///home/fabio/nam-rs/tests/common/validation.rs) `[MODIFY]`
    * [golden_vectors.rs](file:///home/fabio/nam-rs/tests/models/golden_vectors.rs) `[MODIFY]`
