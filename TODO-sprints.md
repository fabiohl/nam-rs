<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-sprints.md — Planejamento Ágil de Sprints (NAM-rs)

Este documento contém o planejamento de sprints e tarefas técnicas estruturadas para o desenvolvimento do NAM-rs, garantindo paridade total com o NeuralAmpModelerCore v0.5.4.

---

## Sprints 3, 4 e 5: Épico D — "Convolução FFT Particionada para Linear" (F1, F12)

**Escopo:** Implementar a convolução FFT particionada com latência zero para o modelo Linear, com auto-seleção inteligente baseada no tamanho do receptive field e suporte ao controle de implementação via JSON.
**Objetivo de Paridade:** Processar eficientemente Impulse Responses (IRs) longas (como cab sims de 2048 a 8192+ amostras) sem comprometer o orçamento de CPU de tempo real (RT-Safety), enquanto mantém a latência estritamente zerada usando convolução híbrida (Direct Head + FFT Tail).
**Estimativa Total:** 3 Sprints (1 de setup/modelagem, 1 de core do motor de convolução FFT, 1 de paridade/testes/benchmarks).
**Risco Geral:** 🟡 Médio — Requer sincronização temporal perfeita e livre de aliasing entre o bloco de convolução direta (Head) e os blocos calculados via FFT no domínio da frequência (Tail), além de estrita garantia de segurança em tempo real (RT-Safety).

---

### Sprint 3: Fase 1 — Arquitetura, Parser JSON e Modelagem (F1, F12)

#### 1. [LOADER] Adicionar Enum `LinearImplementation` e Desserialização no JSON (F12) [DONE]

- **Status:** `[x]`
- **Arquivos Alvo:**
  - [`src/loader/nam_json/model.rs`](file:///home/fabio/nam-rs/src/loader/nam_json/model.rs)
  - [`src/loader/nam_json/topology/linear.rs`](file:///home/fabio/nam-rs/src/loader/nam_json/topology/linear.rs)
- **Descrição:**
  - Definir o enum `LinearImplementation` com suporte a `Auto`, `Direct` e `Fft`.
  - Adicionar o campo opcional `implementation: Option<String>` na struct `NamConfig`.
  - Atualizar [`get_linear_topology`](file:///home/fabio/nam-rs/src/loader/nam_json/topology/linear.rs) para ler e expor a flag de implementação de forma limpa.
- **Risco:** 🟢 Baixo. Alteração puramente estrutural no parsing JSON do modelo.

#### 2. [LOADER] Propagar Configuração de Implementação para o Dispatcher (F12) [DONE]

- **Status:** `[x]`
- **Arquivo Alvo:** [`src/loader/dispatcher/linear/mod.rs`](file:///home/fabio/nam-rs/src/loader/dispatcher/linear/mod.rs)
- **Descrição:**
  - Atualizar o retorno de [`get_linear_topology`](file:///home/fabio/nam-rs/src/loader/nam_json/topology/linear.rs) para passar o enum de implementação.
  - Ajustar [`build_linear`](file:///home/fabio/nam-rs/src/loader/dispatcher/linear/mod.rs) para aceitar e propagar a opção de implementação durante a chamada de criação do `LinearModel`.
- **Risco:** 🟢 Baixo. Conexão mecânica entre componentes.

#### 3. [MODEL] Definir Variantes de Modo no `LinearModel` (F1) [DONE]

- **Status:** `[x]`
- **Arquivo Alvo:** [`src/models/linear.rs`](file:///home/fabio/nam-rs/src/models/linear.rs)
- **Descrição:**
  - Criar o enum `LinearMode` contendo as variantes `Direct` e `Fft(Box<LinearFftState>)` para o modelo Linear.
  - Adicionar o campo `mode: LinearMode` na struct `LinearModel`.
  - Integrar a assinatura do construtor `LinearModel::new` para aceitar a opção de implementação e configurar o modo direto inicial/fallback.
- **Risco:** 🟢 Baixo. Modificação de setup sem alteração no processamento ativo.

---

### Sprint 4: Fase 2 — Core do Motor de Convolução FFT Zero-Latência (F1)

#### 4. [MODEL] Desenhar e Estruturar o `LinearFftState` (F1) [DONE]

- **Status:** `[x]`
- **Arquivo Alvo:** [`src/models/linear_fft.rs`](file:///home/fabio/nam-rs/src/models/linear_fft.rs)
- **Descrição:**
  - Criar o arquivo `linear_fft.rs` incluindo cabeçalho SPDX e copyright obrigatório.
  - Implementar a estrutura `LinearFftState` com todos os buffers necessários para overlap-save zero-latência pré-alocados via `AlignedVec<f32>` (para compatibilidade SIMD x86-64-v3):
    - `rfft`: `RfftPlanner<f32>` do NAM-rs.
    - `h_fdl_re`, `h_fdl_im`: espectros das partições de cauda (IR de `P` a `N-1`).
    - `fdl_re`, `fdl_im`: Frequency Delay Line circular.
    - `input_buf`: buffer de entrada de tamanho `2P`.
    - `fft_re`, `fft_im`: saída do forward RFFT (`P+1` bins).
    - `acc_re`, `acc_im`: buffers de acumulação complexa.
    - `output_buf`: buffer de saída IFFT (`2P`).
    - `tail_output_buf`: buffer circular com os resultados prontos para leitura direta por sample.
    - `sample_counter`: índice de leitura interno do bloco de cauda (0 a `P-1`).
  - Construtor `LinearFftState::new(p, weights)` que pré-computa os espectros `h_fdl_*` das partições de cauda e aloca todos os buffers.
  - Método `reset()` que zera buffers de runtime sem realocar (allocation-free).
  - 9 testes unitários cobrindo: P==N, N==2P, IR 8192, partição irregular, pânico em P não-potência-de-2, pânico em P>N, reset, Debug, verificação de espectros.
- **Risco:** 🟡 Médio. Requer precisão nos buffers de alinhamento e determinação do tamanho da partição `P` com base no receptive field (256, 512, 1024).

#### 5. [MODEL] Implementar o Loop de Processamento e FFT de Cauda (F1)

- **Status:** `[ ]`
- **Arquivo Alvo:** [`src/models/linear_fft.rs`](file:///home/fabio/nam-rs/src/models/linear_fft.rs)
- **Descrição:**
  - Implementar o método `process_tail_block` em `LinearFftState` para rodar na fronteira de blocos:
    - Copiar a janela de tamanho `2P` correspondente das últimas amostras contíguas diretamente do `MirroredBuffer` para `input_buf`.
    - Computar o forward RFFT de `input_buf`.
    - Atualizar a FDL e realizar o acúmulo no domínio da frequência usando dispatch SIMD (`complex_mac_overwrite` / `complex_mac_accumulate`) conforme suporte ISA (`Avx2` ou `Avx512`).
    - Executar o IFFT e armazenar a cauda válida (amostras `P..2P-1`) no `tail_output_buf`.
- **Risco:** 🔴 Alto. Hot-path precisa ser rigorosamente RT-Safe: zero heap drops, zero locks, zero loops sem eliminação estática de bounds checks.

#### 6. [MODEL] Integrar o Despacho no Hot-path do `LinearModel` (F1)

- **Status:** `[ ]`
- **Arquivo Alvo:** [`src/models/linear.rs`](file:///home/fabio/nam-rs/src/models/linear.rs)
- **Descrição:**
  - Adaptar o hot-path do `process_sample` do `LinearModel` para realizar o despacho:
    - **Modo Direct:** Executa o dot product SIMD de todo o receptive field.
    - **Modo FFT:**
      - Computa o dot product da parte Head (tamanho `P`) de forma direta (tempo-real/latência zero).
      - Soma o bias do modelo e o valor correspondente da cauda: `y_tail = tail_output_buf[sample_counter]`.
      - Incrementa `sample_counter`. Se atingir `P`, invoca a computação do próximo bloco via `process_tail_block` e reseta o contador.
  - Atualizar os métodos `reset` e `prewarm` do `LinearModel` para reinicializar os estados do `LinearFftState` de forma determinística e livre de alocações.
- **Risco:** 🔴 Alto. Risco de falha de paridade de processamento se o timing de leitura/escrita do buffer circular estiver deslocado por 1 sample.

---

### Sprint 5: Fase 3 — Paridade, Testes de Extremo e Benchmarks (F1)

#### 7. [TEST] Escrever Testes Unitários de Equivalência Numérica (F1)

- **Status:** `[ ]`
- **Arquivo Alvo:** [`src/models/linear.rs`](file:///home/fabio/nam-rs/src/models/linear.rs) (ou arquivo separado de teste)
- **Descrição:**
  - Escrever testes unitários em `tests` validando que as saídas geradas em Direct e FFT sob o mesmo modelo e sinal de teste são numericamente idênticas dentro de uma tolerância estrita de `1e-6`.
  - Garantir tratamento correto das regras de tamanho de arquivos de testes (inline vs `_test.rs`) de acordo com a quantidade de linhas do arquivo final.
- **Risco:** 🟢 Baixo. Validação da corretude matemática básica.

#### 8. [TEST] Criar Fixtures de Integração e Golden Tests vs C++ v0.5.4 (F1)

- **Status:** `[ ]`
- **Arquivo Alvo:** `tests/linear_fft_test.rs` (Novo arquivo)
- **Descrição:**
  - Adicionar fixture de teste contendo IRs de comprimento longo (ex: 2048, 4096, 8192 taps).
  - Alimentar os modelos no NAM-rs e comparar o output final amostra por amostra com a saída gerada pela biblioteca de referência C++.
  - Certificar execução e passagem total no QA pipeline via `utils/tests-quick.sh`.
- **Risco:** 🟡 Médio. Eventuais desvios de arredondamento de float em FFTs longas precisam ser inspecionados.

#### 9. [BENCH] Benchmark de Performance e Ajuste de Limiar (F1)

- **Status:** `[ ]`
- **Arquivo Alvo:** `benches/linear.rs` (Novo arquivo)
- **Descrição:**
  - Criar benchmarks baseados no framework `criterion` comparando o tempo de processamento por bloco do modo Direct vs FFT em diferentes tamanhos de receptive field (128, 256, 512, 1024, 2048, 4096, 8192).
  - Validar empiricamente se o limiar de corte para auto-seleção (256 taps) é ótimo para a arquitetura alvo `x86-64-v3`.
- **Risco:** 🟢 Baixo. Coleta estatística de dados de latência de CPU.
