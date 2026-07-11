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

* **S1.T03 — Auditoria e refatoração dos call-sites de `set_activation_precision`** `[x]` ⚠ **REABERTO — ver S6.T01 (Achado F11)**
  * **Ação:** Substituir chamadas diretas que alteram o atômico global pelo uso do `PrecisionGuard` para garantir exclusão mútua e restauração automática.
  * **Arquivos:**
    * [cpp_parity.rs](file:///home/fabio/nam-rs/tests/parity/cpp_parity.rs) `[MODIFY]`
    * [isa_parity.rs](file:///home/fabio/nam-rs/tests/parity/isa_parity.rs) `[MODIFY]`
    * [activation_precision.rs](file:///home/fabio/nam-rs/tests/models/activation_precision.rs) `[MODIFY]`
    * [lstm_activation_precision.rs](file:///home/fabio/nam-rs/tests/models/lstm_activation_precision.rs) `[MODIFY]`
    * [reference_oracle_f64.rs](file:///home/fabio/nam-rs/tests/parity/reference_oracle_f64.rs) `[MODIFY]`
  * **Verificação de acompanhamento (2026-07-11):** auditoria de execução (não
    apenas revisão estática) encontrou 3 call-sites ainda desprotegidos em
    `activation_precision.rs` e reproduziu falhas intermitentes reais em
    `namb_v2_roundtrip.rs`/`namb_v2_validation.rs` causadas por essa lacuna
    (2 de 4 execuções de `cargo test --release --test models` falharam, cada
    vez em um teste diferente). Ver `TODO-findings.md` Achado F11 e Sprint 6
    abaixo para o fechamento.

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

* **S2.T03 — Correção do Oráculo WaveNet A2** `[x]` ⚠ **PARCIAL — REABERTO, ver S6.T02 (Achado F12)**
  * **Ação:** Corrigir os caminhos de cálculo e dimensionalidades associadas a `condition_dsp` e leitura de pesos de `head1x1` no oráculo f64 em `src/testing/reference_oracle/a2.rs` conforme detalhado no Épico 6 do `TODO-wavenet_a2_max.md` (não-bloqueante para produção).
  * **Arquivos:** [a2.rs](file:///home/fabio/nam-rs/src/testing/reference_oracle/a2.rs) `[MODIFY]`
  * **Verificação de acompanhamento (2026-07-11):** executar (não apenas
    revisar) `test_oracle_vs_python_anchor_a2_generic -- --ignored` revela um
    **panic** ("range end index 826 out of range for slice of length 818") —
    o commit `b7a8fb4` corrigiu fórmulas de dimensão mas manteve a estrutura
    de leitura por-array (mesmo Bug A da produção, replicado no oráculo).
    Ver `TODO-findings.md` Achado F12 e Sprint 6 abaixo para o fechamento.

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

* **S3.T01 — Condicionamento da Métrica por Frame em `compute_mr_stft`** `[x]`
  * **Ação:** Refatorar a computação em `src/testing/perceptual.rs` para substituir o piso absoluto `eps = 1e-8` por um piso relativo ao pico espectral do frame (ex.: −80 dB em relação ao pico do frame).
  * **Concluído:** Substituído `eps = 1e-8` por `eps_frame = max(frame_peak * 1e-4, eps_abs)`. O pico espectral (`frame_peak`) é calculado por frame como o máximo de magnitude entre os espectros de referência e teste. O floor absoluto `1e-8` é mantido como fallback para frames silenciosos. Todos os 46 testes passam (45 perceptual + 1 parity).
  * **Arquivos:** [perceptual.rs](file:///home/fabio/nam-rs/src/testing/perceptual.rs) `[MODIFY]`

* **S3.T02 — Espelhamento no Script Python e Regeração de `mrstft_golden.bin`** `[x]`
  * **Ação:** Criar ou atualizar o script `tests/fixtures/scripts/gen_mrstft_golden.py` com o mesmo piso relativo do Rust, e rodar o script para gerar um novo `tests/fixtures/mrstft_golden.bin` determinístico e bit-a-bit idêntico.
  * **Concluído:** Script criado em `tests/fixtures/scripts/gen_mrstft_golden.py` com o mesmo algoritmo de piso relativo (-80 dB por frame). Golden `mrstft_golden.bin` gerado com 4800 samples (seed=42). `test_mr_stft_parity_with_python` passa com resultado idêntico (1.150806e-2).
  * **Arquivos:**
    * [gen_mrstft_golden.py](file:///home/fabio/nam-rs/tests/fixtures/scripts/gen_mrstft_golden.py) `[NEW]`
    * [mrstft_golden.bin](file:///home/fabio/nam-rs/tests/fixtures/mrstft_golden.bin) `[MODIFY]`

---

### Épico B.2 — Calibração do Soft Gate e Cobertura da Família Linear (F2.1, F2.2)

* **S3.T03 — Calibração de `MRSTFT_SOFT_THRESHOLD` no Metateste de Calibração** `[x]`
  * **Ação:** Calibrar empiricamente e documentar o limiar do soft gate de MR-STFT, integrando-o ao escrutínio automático em `tests/models/threshold_calibration.rs`.
  * **Concluído:** `MRSTFT_SOFT_THRESHOLD` elevado a `pub const` com documentação de calibração (S3.T03). Valor calibrado empiricamente para 0.50 — teto anti-placebo (Rule 4), com margem de 0.05 sobre o maior hard gate não-degenerado (wavenet_official: 0.45). Meta-teste `test_mrstft_soft_threshold_is_calibrated` adicionado ao escrutínio automático, validando: (1) valor ≤ 0.5, (2) valor > 0.0, (3) comentário `// Measured:` de proveniência. Todos os 6 meta-testes de threshold_calibration passam; 34 golden_vectors sem regressão.
  * **Arquivos:**
    * [validation.rs](file:///home/fabio/nam-rs/tests/common/validation.rs) `[MODIFY]`
    * [threshold_calibration.rs](file:///home/fabio/nam-rs/tests/models/threshold_calibration.rs) `[MODIFY]`

* **S3.T04 — Mapeamento e Habilitação do Gate para Modelos Linear** `[x]`
  * **Ação:** Adicionar mapeamento de goldens Linear FFT em `golden_bin_to_model_name()` e configurar limiares de `mrstft_max` realísticos (não-`None`) para a família Linear no harness.
  * **Concluído:** Mapeamento de `golden_linear_fft_rf{2048,4096,8192}` adicionado em `golden_bin_to_model_name()`. Entradas calibradas em `get_calibrated_threshold()` com MR-STFT gate=0.12 — calibrado empiricamente do pior caso (impulse response RF=4096: MR-STFT=0.109, margem 10%). `topology_thresholds()` e `live_parity_thresholds()` atualizados de `None` para `Some(0.12)` para Linear. Math oracle tests (`verify_against_oracle`, `verify_nam_file_against_oracle`) com MR-STFT gate habilitado. 6 meta-testes passam, 51 golden_vectors OK, 17 linear_fft OK.
  * **Arquivos:**
    * [validation.rs](file:///home/fabio/nam-rs/tests/common/validation.rs) `[MODIFY]`
    * [threshold_calibration.rs](file:///home/fabio/nam-rs/tests/models/threshold_calibration.rs) `[MODIFY]`
    * [linear_fft_test.rs](file:///home/fabio/nam-rs/tests/models/linear_fft_test.rs) `[MODIFY]`
  * **Nota para S3.T05:** O rótulo `(relative)` ainda está presente na saída de `report_dsp_fidelity` para o soft gate — será corrigido no próximo task.

---

### Épico B.3 — Higiene de Rótulos e Rationale de Relaxação (F2.3, F2.4)

* **S3.T05 — Rótulo da Métrica de `(relative)` para `(log-mag abs)`** `[x]`
  * **Ação:** Corrigir a identificação visual impressa pelo harness de teste para que declare a métrica de forma precisa e coerente com a implementação final.
  * **Arquivos:** [validation.rs](file:///home/fabio/nam-rs/tests/common/validation.rs) `[MODIFY]`

* **S3.T06 — Unificação e Documentação da Relaxação de `mrstft_max`** `[x]`
  * **Ação:** Unificar o expoente de relaxamento de `mrstft_max` (`/5.0` vs `/10.0`) e documentar o rationale físico/espectral associado no harness e na especificação.
  * **Arquivos:**
    * [golden_vectors.rs](file:///home/fabio/nam-rs/tests/models/golden_vectors.rs) `[MODIFY]`
    * [perceptual_validation.md](file:///home/fabio/nam-rs/docs/perceptual_validation.md) `[MODIFY]`

---

### Épico B.4 — Modo Silencioso e Supressão de Pânico em Metatestes (F7)

* **S3.T07 — Modo Silencioso no Harness para Testes de Regressão** `[x]`
  * **Ação:** Implementar controle de supressão de relatório (ex.: flag thread-local `SUPPRESS_REPORT` ou estrutura similar) e registrar panic hook temporário em `test_mrstft_hard_gate_catches_regression` para impedir poluição com símbolos "✗" e "panicked" na suíte de testes verde.
  * **Arquivos:**
    * [validation.rs](file:///home/fabio/nam-rs/tests/common/validation.rs) `[MODIFY]`
    * [golden_vectors.rs](file:///home/fabio/nam-rs/tests/models/golden_vectors.rs) `[MODIFY]`

---

## Sprint 4: Quality Dashboard (Épico D)

**Objetivo:** Ajustar a apresentação de métricas e corrigir inconsistências de exibição no script `utils/quality-dashboard.sh` (normalização do Container A2-Lite, separação de metodologias f64 cold vs paired, declaração explícita de não-cobertura das seções ISA/Spectral no modo rápido), adicionar testes de interoperação (cross-validation) para o modelo ConvNet contra a implementação C++ de referência, e alinhar cosmeticamente o cabeçalho.

**Risco:** Baixo (alterações focadas na exibição de dados e na adição de um teste de paridade específico, sem alteração nos caminhos de produção).

---

### Épico D.1 — Investigar e Normalizar a Linha Container A2-Lite (F6.1)

* **S4.T01 — Correção de vazamento de contexto no parser de golden vectors** [x]
  * **Ação:** No script [quality-dashboard.sh](file:///home/fabio/nam-rs/utils/quality-dashboard.sh), na função `parse_golden_vectors`, redefinir `label = ""` quando o parser casar a linha `/^\[ConvNet Self-Golden/`. Isso impede que as métricas do teste de determinismo do ConvNet (ESR=0, SNR=inf) sobrescrevam de forma errônea as métricas reais do Container A2-Lite registradas no bloco anterior.
  * **Arquivos:** [quality-dashboard.sh](file:///home/fabio/nam-rs/utils/quality-dashboard.sh) `[MODIFY]`

---

### Épico D.2 — Separar Proveniência Cold/Paired no `ESR_F64` (F3, F6.3)

* **S4.T02 — Separação de metodologias de medição F64 no Dashboard** [x]
  * **Ação:** Refatorar o armazenamento do f64 oracle em [quality-dashboard.sh](file:///home/fabio/nam-rs/utils/quality-dashboard.sh). Em vez de usar um único array `ESR_F64` compartilhado, criar dois arrays associativos distintos:
    * `ESR_F64_COLD`: Para os dados de decomposição (256 samples sem warmup, transiente ativo).
    * `ESR_F64_PAIRED`: Para as tabelas de fidelidade e resumos (warmup de 24k samples).
    * Adaptar as funções `_lookup_esr_f64`, `parse_oracle_f64` e `render_f64_decomposition` para utilizarem os respectivos arrays.
  * **Arquivos:** [quality-dashboard.sh](file:///home/fabio/nam-rs/utils/quality-dashboard.sh) `[MODIFY]`

---

### Épico D.3 — Seções ISA/Spectral no Modo Quick (F6.2)

* **S4.T03 — Declaração explícita de cobertura rápida nas seções ISA/Spectral** [x]
  * **Ação:** Modificar as funções de renderização `render_isa_parity` e `render_spectral_summary` em [quality-dashboard.sh](file:///home/fabio/nam-rs/utils/quality-dashboard.sh). Caso os arrays de dados correspondentes estejam vazios (indicando uma rodada rápida sem testes de ISA ou fidelidade espectral estendida), exibir explicitamente a mensagem `"Não coberto no modo quick — rode tests-long para verificação completa"` em vez de indicar `"Nenhum resultado disponivel"`.
  * **Arquivos:** [quality-dashboard.sh](file:///home/fabio/nam-rs/utils/quality-dashboard.sh) `[MODIFY]`

---

### Épico D.4 — Cobertura de Interoperação ConvNet (F6.4)

* **S4.T04 — Adicionar teste de paridade C++ para ConvNet** [x]
  * **Ação:**
    * Em [cpp_parity.rs](file:///home/fabio/nam-rs/tests/parity/cpp_parity.rs), adicionar os testes de paridade `quick_parity_convnet` (para o loop rápido) e `live_cross_validation_convnet` (para o loop completo) usando o modelo `convnet_test.nam`.
    * No helper `topology_thresholds` em [validation.rs](file:///home/fabio/nam-rs/tests/common/validation.rs), garantir que existam limites de tolerância calibrados para a família ConvNet.
  * **Arquivos:**
    * [cpp_parity.rs](file:///home/fabio/nam-rs/tests/parity/cpp_parity.rs) `[MODIFY]`
    * [validation.rs](file:///home/fabio/nam-rs/tests/common/validation.rs) `[MODIFY]`

---

### Épico D.5 — Correções Cosméticas e Referências no Dashboard (F10.5)

* **S4.T05 — Alinhamento cosmético do cabeçalho e referências textuais**[x]
  * **Ação:**
    * Corrigir a largura e o preenchimento de campos (`printf`) na função `render_header` do script [quality-dashboard.sh](file:///home/fabio/nam-rs/utils/quality-dashboard.sh) para evitar estouro de bordas quando os rótulos de CPU ou compilador forem longos.
    * Na função `render_f64_decomposition`, atualizar a referência textual `"TODO-findings.md Achado A1"` para `"TODO-findings.md Achado F3"`.
  * **Arquivos:** [quality-dashboard.sh](file:///home/fabio/nam-rs/utils/quality-dashboard.sh) `[MODIFY]`

---

## Sprint 5: Suítes de execução (Épico E)

**Objetivo:** Otimizar e documentar a suíte de testes rápidos (`utils/tests-quick.sh`), estabelecendo políticas claras sobre a remoção de compilações redundantes (como `--test clap` sem feature flags) e consolidando a robustez de skips sob uso contínuo.

**Risco:** Baixo (apenas otimização da suíte de testes locais, sem risco para caminhos de áudio real-time).

---

### Épico E.1 — Validação e robustecimento de Skips no script de testes rápidos (F5 follow-up)

* **S5.T01 — Validação e robustecimento de Skips no script de testes rápidos** [x]
  * **Ação:** Auditar e documentar no script [tests-quick.sh](file:///home/fabio/nam-rs/utils/tests-quick.sh) a saída graciosa (exit code 0 com skips relatados) e as condições sob as quais skips ocorrem (ausência de compilation toolchain C++, falta de fixtures committed), garantindo integridade e ausência de falsos alarmes em ambientes de CI.
  * **Concluído:** Auditoria completa dos 4 cenários de skip identificados: (1) golden vectors v1/v2 ausentes, (2) toolchain C++ não encontrada, (3) falha de CMake configure/build, (4) NAMCore não checkout. Adicionado header documentando cada cenário com condição/consequência/tracking. Introduzida variável `CPP_PARITY_SKIPPED` para rastrear skip do cpp_parity, independentemente de `MEASUREMENT_STATUS`. Resumo final agora cobre 4 combinações de skips (GOLDEN_RAN × CPP_PARITY_SKIPPED). Todos os caminhos de saída verificados: skips produzem exit 0 com mensagens informativas; apenas falhas reais em testes mandatórios produzem exit 1. Nenhum falso alarme identificado para ambientes CI sem golden fixtures e sem toolchain C++.
  * **Arquivos:** [tests-quick.sh](file:///home/fabio/nam-rs/utils/tests-quick.sh) `[MODIFY]`

---

### Épico E.2 — Decisão sobre `--test clap` na Fase 1 (F5 follow-up)

* **S5.T02 — Remoção do target `--test clap` na Fase 1 (sem features)** [x]
  * **Ação:** Remover o parâmetro `--test clap` da chamada de `cargo test` na Fase 1 de [tests-quick.sh](file:///home/fabio/nam-rs/utils/tests-quick.sh) quando a feature `clap-plugin` não estiver ativa. Isso evita o custo de compilação e linkagem de um binário de teste vazio (0 testes executados) e acelera o tempo de feedback do desenvolvedor.
  * **Concluído:** Adicionada função `_has_clap_plugin()` que detecta se a feature está ativa via `NAM_FEATURES` (override explícito) ou parsing do `Cargo.toml` (default features). Como `clap-plugin` NÃO está nas default features (`standalone` + `testing`), `--test clap` é excluído na execução padrão, eliminando ~15-30s de compilação de binário vazio. Quando o desenvolvedor define `NAM_FEATURES="standalone,testing,clap-plugin"`, o target é reincluído automaticamente. O caminho legacy (pre-Sprint 3) não é afetado — `STRUCT_TESTS` nunca incluiu testes clap.
  * **Arquivos:** [tests-quick.sh](file:///home/fabio/nam-rs/utils/tests-quick.sh) `[MODIFY]`

---

## Sprint 6: Fechamento Residual Verificado por Execução (Épico G)

**Objetivo:** Fechar as duas lacunas encontradas pela auditoria de
acompanhamento de 2026-07-11, que **executou** (em vez de apenas revisar
estaticamente) os artefatos das Sprints 1–5 e reproduziu, empiricamente, uma
falha de flakiness real (F11) e um panic no oráculo A2 (F12). Ver
`TODO-findings.md` Achado F11/F12 e Épico G para o detalhamento completo.

**Risco:** F11 é baixo esforço/alto valor (causa raiz já isolada a 3 funções);
F12 é médio esforço e deve ser coordenado com o início do Épico F
(`TODO-wavenet_a2_max.md` Epic 2), já que ambos tocam a mesma estrutura de
leitura de pesos do A2.

---

### Épico G.1 — Fechar o Rollout do `PrecisionGuard` (F11)

* **S6.T01 — Proteger os call-sites remanescentes em `activation_precision.rs`** `[x]`
  * **Ação:** Envolver `test_zero_alloc_activation_switch_primitive`,
    `test_zero_alloc_activation_hot_path_switch` e
    `test_zero_alloc_cli_activation_flow` com `PrecisionGuard::new(...)`,
    adquirido **antes** do `TrackingGuard`, para não contaminar a contagem de
    alocações medida. Adicionar meta-teste estático (grep-based, estilo
    `threshold_calibration.rs`) que falha o build se `set_activation_precision(`
    aparecer em `tests/**/*.rs` fora de `tests/common/precision.rs` sem
    `PrecisionGuard::new` na mesma função.
  * **Critério de aceite:** `cargo test --release --test models` roda **≥ 10×
    consecutivas sem falha** (hoje falha em ~2 de 4 execuções).
  * **Arquivos:**
    * [activation_precision.rs](file:///home/fabio/nam-rs/tests/models/activation_precision.rs) `[MODIFY]`
    * [threshold_calibration.rs](file:///home/fabio/nam-rs/tests/models/threshold_calibration.rs) `[MODIFY]` (novo meta-teste)

* **S6.T02 — Decidir sobre `--test-threads=1` como defesa em profundidade na Fase 1** `[x]`
  * **Ação:** Avaliar o custo/benefício de forçar `--test-threads=1` no
    binário `models` em [tests-quick.sh](file:///home/fabio/nam-rs/utils/tests-quick.sh)
    Fase 1, e documentar a decisão (feita ou rejeitada, com rationale) em
    [testing.md](file:///home/fabio/nam-rs/docs/testing.md) §2.
  * **Decisão:** REJEITADO. O custo (+15–30s por execução, várias vezes/dia)
    supera o benefício marginal dado que: (a) `ACTIVATION_MODE` é o único estado
    global mutável vulnerável, (b) o meta-teste estático de S6.T01 já captura
    regressões do padrão conhecido em tempo de build, (c) o projeto investiu
    explicitamente em suporte paralelo (`REPORT_LOCK`). Documentado em
    `docs/testing.md` §2.

---

### Épico G.2 — Corrigir Estruturalmente o Oráculo A2 (F12)

* **S6.T03 — Reestruturar leitura de `head1x1` para dentro do laço por-camada** `[ ]`
  * **Ação:** Em [a2.rs](file:///home/fabio/nam-rs/src/testing/reference_oracle/a2.rs),
    mover a leitura de `head1x1_w`/`head1x1_b` (hoje após o laço `for li in
    0..num_layers`) para **dentro** do laço, na ordem correta do stream de
    pesos do C++ (a confirmar por reconciliação campo-a-campo, coordenada com
    a correção do Bug A na produção — Epic 2 do `TODO-wavenet_a2_max.md`).
    Corrigir também a condição de formato do head final para depender da
    presença real de `cfg.head` (não de `head_size == 1`).
  * **Arquivos:** [a2.rs](file:///home/fabio/nam-rs/src/testing/reference_oracle/a2.rs) `[MODIFY]`

* **S6.T04 — Gate de reconciliação automático de orçamento de pesos** `[ ]`
  * **Ação:** Ao final de `oracle_a2_forward`, adicionar
    `assert_eq!(cursor.pos, model_data.weights.len(), "resíduo de pesos não
    consumidos/lidos em excesso para {model_filename}")` (não-ignorado) —
    transforma a reconciliação manual de `TODO-wavenet_a2_max.md` §3.5 em
    salvaguarda permanente.
  * **Arquivos:** [a2.rs](file:///home/fabio/nam-rs/src/testing/reference_oracle/a2.rs) `[MODIFY]`

* **S6.T05 — Reabilitar e verificar `test_oracle_vs_python_anchor_a2_generic`** `[ ]`
  * **Ação:** Após S6.T03/S6.T04, executar
    `cargo test --release --test parity reference_oracle_f64::test_oracle_vs_python_anchor_a2_generic -- --ignored --nocapture`
    e confirmar que produz um valor de ESR (não um panic). O teste permanece
    `#[ignore]`d até os Bugs A/B/C de produção (Épico F) também serem
    corrigidos, mas a mensagem de `#[ignore]` deve ser atualizada para
    refletir que a medição agora é possível.
  * **Arquivos:** [reference_oracle_f64.rs](file:///home/fabio/nam-rs/tests/parity/reference_oracle_f64.rs) `[MODIFY]`
