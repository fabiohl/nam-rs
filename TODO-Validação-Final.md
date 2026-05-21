<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved. -->

# TODO — Validações Finais Pré-Release

---

## Épico 1 — Auditoria Final de Código (Dev Humano)

* [x] **Tarefa 1.1** — Auditoria Completa com `clap-validator`

  * **Contexto:**
    O `clap-validator` da CLAP-SDK é a ferramenta oficial de conformidade para atestar que o plugin obedece de forma irrestrita às especificações e ciclos de vida do padrão CLAP (inclusão de portas, transição de estados atômicos, ativação/desativação dinâmica e carregamento de parâmetros). Esta é a auditoria final que executará toda a suíte de testes de estresse estático contra o binário unificado com todas as extensões ativas.
  * **Ações Técnicas:**
    1. Instalar o `clap-validator` mais recente compatível com o spec CLAP v1.2+.
    2. Rodar a validação cruzada do plugin em formato JSON para ambas as compilações (**Debug** com assertions de panic ativados e **Release** otimizado).
    3. Integrar uma verificação ativa de alocação de heap (`CountingAllocator`) acionada durante os testes dinâmicos de processamento de áudio do validador.
    4. Tornar o processo de validação tolerância-zero, forçando a quebra do script caso o arquivo de saída contenha qualquer `warning` ou `failure`:

       ```bash
       # Executar validação rigorosa com fallback de erro explícito para warnings/failures
       clap-validator validate ~/.clap/nam-rs.clap --json > /tmp/final-validation.json
       jq -e '[.. | objects | select(.code? == "failure" or .code? == "warning")] | length == 0' /tmp/final-validation.json
       ```

    5. Caso algum teste dinâmico falhe ou gere warning, depurar usando `gdb` ou a skill `debugger` e corrigir os estados dinâmicos no ciclo de vida do `clack-plugin`.
  * **Aceite:**
    * `clap-validator` reporta conformidade total (zero FAILs e zero WARNINGs na suíte oficial de validação).
    * Execução do validador em modo de estresse não gera erros de concorrência, crashes ou vazamento de memória no arquivo de dump.

* [ ] **Tarefa 1.2** — Regressão do Modo Standalone
  * **Contexto:**
    A introdução de feature flags e a refatoração do pipeline DSP principal podem ter afetado o modo standalone (PipeWire/JACK nativo). É vital atestar que o NAM-rs rodando em modo CLI standalone preserva suas propriedades de baixa latência e compatibilidade.
  * **Ações Técnicas:**
    1. Executar a suíte de testes completa: `utils/lints.sh`, `utils/tests-cargo.sh` e `utils/tests-long.sh`.
    2. Testar a inicialização do executável standalone (`utils/run-standalone.sh`) conectando-o dinamicamente ao PipeWire sob taxas de amostragem comuns (44.1kHz e 48kHz), verificando se o escalonamento em tempo real (RT) e as prioridades do kernel são atribuídos corretamente.
    3. Solicitar ao git apontamento de todos os trechos de arquivos fora do `src/clap/`, que foram editados, para leitura e revisão.
  * **Aceite:**
    * Executável standalone compila e todos os testes unitários/integração integrados passam com status de sucesso.
    * Desempenho do benchmark DSP de inferência e resampling standalone mantido com variação máxima tolerada de ±2% em comparação ao baseline pré-sprint.

* [ ] **Tarefa 1.3** — Revisão Crítica de Todo Código CLAP Adicionado
  * **Contexto:**
    Ao longo das sprints rápidas, múltiplos módulos foram criados em `src/clap/`. É imperativo realizar uma auditoria rigorosa baseada nas Diretrizes Técnicas e de Performance para garantir coesão arquitetural, nomenclatura aderente à convenção do Rust, ausência de vazamento de memória e que todas as operações no hot-path obedeçam às garantias de RT-Safety (ex: zero blocos de lock, zero heap drops indiretos e tratamento seguro de concorrência).
  * **Ações Humanas:**
    1. Skill `revisor-auditor` focado exclusivamente na pasta `src/clap/`. Demais pastas apenas nas' suas realações com `src/clap/`.
    2. Revisão completa do `README.md` e do `docs/*.md` para refletir o estado real do código.
    3. Leitura humana o código em `src/clap/` e assegurar completa cobertura de docsys e comentários inline no código fonte.

* [ ] **Tarefa 1.4** — Revisão Crítica da necessidade de continuidade de certos artefatos.
  * **Contexto:**
    Ao longo das sprints rápidas, múltiplos artefatos foram sendo criados por um propósito útil na época - mas que hoje pode simplesmente ser apenas lixo.
  * **Ações Humanas:**
    1. `proptest-regressions/`.
    2. `third_party/`.
    3. Suítes de testes e benches: O que não traz valor real ainda? O que está faltando?

---

## Épico 2 — Sessão de Estresse Final em DAWs

* [ ] **Tarefa 2.1** — Testes A/B de Fidelidade e Integridade de Inferência
  * **Contexto:**
    Para garantir a acurácia matemática e fidelidade do processamento DSP e das aproximações matemáticas de funções de ativação do NAM-rs (WaveNet/LSTM) em relação à implementação original de referência em Python/C++ (sdatkinson), devemos realizar testes de verificação de sinal. A divergência entre o NAM-rs e o C++ é **intencional** (Padé SIMD vs Polynomial escalar — ADR-002 em `tests/fixtures/README.md`) e os thresholds são calibrados contra medições reais.
  * **Pré-requisito Automatizado:**
    Confirmar que os testes de golden vectors passam sem regressão:

    ```bash
    cargo test -- test_golden_vectors --include-ignored
    ```

    Os golden vectors em `tests/fixtures/` validam 4 modelos (WaveNet Standard/Feather/Nano + LSTM 1×16) com thresholds MSE+SNR calibrados.
  * **Ações Técnicas (Validação Manual Complementar):**
    1. Coletar um conjunto representativo de arquivos de modelo (`.nam` e `.namb`) do repositório público (como Tone3000) cobrindo diferentes arquiteturas e tamanhos (WaveNet Standard, LSTM 1 camada, LSTM 2 camadas).
    2. No NAM-rs standalone (`utils/run-standalone.sh`), processar sinais de entrada de referência (sine sweep 20Hz-20kHz e ruído branco) e capturar a saída em arquivo WAV.
    3. Comparar auditivamente a saída do NAM-rs com a do plugin oficial NAM (VST3/CLAP) na mesma DAW, com parâmetros idênticos (ganhos em 0 dB, gate desativado).
    4. Se discrepâncias audíveis forem detectadas, computar diferença RMS ponto a ponto e investigar a causa (resampler, gate, denormals).
  * **Aceite:**
    * Testes automatizados de golden vectors passam com os thresholds calibrados (ADR-002):
      * **WaveNet:** MSE < 5e-2, SNR ≥ 9 dB.
      * **LSTM:** MSE < 1e-3, SNR ≥ 22 dB.
    * Zero pânicos ou silenciamento nos blocos processados.
    * Avaliação auditiva subjetiva: sem diferenças tonais perceptíveis em comparação A/B cega.

---

* [ ] **Tarefa 2.2** — Bateria de Stress Test (Smoke Test Cross-Host)
  * **Contexto:** Garantir que o plugin sobreviva a operações estressantes, uso frenético de interface e modulações agressivas sem derrubar a DAW (panic/crash) ou vazar recursos de sistema (memória RAM ou File Descriptors).
  * **Configuração de áudio recomendada:** 48 kHz, buffer de 128 samples (worst case realista para latência).
  * **Ações do Testador:**
    1. **Spamming de Bypass e Interface:**
       * Alternar o bypass do plugin alucinadamente via GUI (cliques muito rápidos) e via botão nativo de bypass do host (20+ vezes consecutivamente).
       * Abrir e fechar a GUI do plugin consecutivamente 20+ vezes em menos de 30 segundos com áudio em execução.
       * No Bitwig Studio, alternar os modos de hosting (*Together*, *Individually* e *Individually strict*) e refazer o teste. O sandbox de plugins da DAW não pode crashar.
    2. **Carga Rápida Concorrente de Modelos:**
       * Abrir o File Picker e carregar modelos diferentes de forma rápida e consecutiva (10x em menos de 1 minuto) enquanto a DAW processa sinal de áudio real.
       * Monitorar o consumo de RAM (RSS) do processo sandbox ou da DAW no `htop`. A memória deve permanecer estável (crescimento RSS < 2 MB após 10 reloads consecutivos), provando que o coletor de lixo lock-free (`GcItem` e `GcOverflowBuffer`) na main thread está liberando corretamente os buffers e pesos de modelos anteriores do heap.
    3. **Modulação por LFO e Sinais de Alta Frequência (The Grid no Bitwig):**
       * **No Bitwig Studio:** Adicionar um LFO rápido (ex: 20Hz a 100Hz) modulando o `input_gain_db` ou conectar uma cadeia complexa de modulação no *The Grid* direcionada a parâmetros do NAM-rs.
       * **Configuração crítica:** Usar buffer de 128 samples a 48 kHz (375 callbacks/segundo) para estressar o `ParamSmoother` ao máximo.
       * Deixar rodando por no mínimo 5 minutos sob playback contínuo.
       * Ouvir com atenção a saída para certificar-se de que o `ParamSmoother` eliminou completamente qualquer zipper noise ou artefato de clique, mesmo sob modulação sample-accurate rápida.
       * **No Fender Studio Pro:** Utilizar envelopes de modulação ou LFOs de canal para estressar os parâmetros em playback por 5 minutos.
    4. **Comportamento com Silêncio (Tail & CPU):**
       * ⚠️ **Nota:** O NAM-rs atualmente retorna `ProcessStatus::Continue` sempre — o host não recebe indicação de auto-sleep. Este item valida apenas que o **Gate FSM** opera corretamente durante transições silêncio → sinal.
       * Parar o playback e gerar silêncio no canal por 10 segundos. Observar que o Gate FSM fecha (multiplicador → 0.0) e a saída é silêncio limpo (sem ruído residual, sem denormals audíveis).
       * Iniciar o playback novamente e validar que a saída retoma sem estalos ou perda de transientes (fade-in rampa do Gate FSM funcionando perfeitamente).
    5. **Auditoria de Recursos:**
       * Antes de iniciar: anotar baseline de File Descriptors e thread count:

         ```bash
         PID=$(pgrep -f "bitwig-studio" | head -n 1)
         echo "FDs: $(ls /proc/$PID/fd | wc -l)"
         echo "Threads: $(ls /proc/$PID/task | wc -l)"
         ```

       * Deixar o plugin em execução em sessões de áudio por 15 minutos em ambos os hosts.
       * Repetir a contagem de FDs e threads. Crescimento > 0 FDs persistentes = ⚠️ leak provável.
       * Executar `lsof -p <PID> | grep -c '.nam\|rfd\|zenity'` para verificar que não há File Descriptors retidos do File Picker ou de portais D-Bus.
  * **Aceite:** Zero crashes, zero panics na DAW ou no processo sandbox. Uso de memória RSS estabilizado (crescimento < 2 MB após trocas de modelos). Zero XRUNs adicionais acusados pelo host de áudio. Zero leak de FDs ou threads.

---

* [ ] **Tarefa 2.3** — Endurance de 1 Hora com Carga Máxima
  * **Contexto:**
    É crucial assegurar que o plugin não apresenta vazamentos de recursos (memória, file descriptors) ou degradação de performance ao longo de sessões prolongadas sob alta carga nas DAWs.
  * **Ações Técnicas:**
    1. Montar um projeto de teste de endurance na DAW contendo 4 instâncias concorrentes do NAM-rs utilizando modelos de redes neurais distintos (2× WaveNet Standard + 2× LSTM 1×16) e com 2 moduladores LFO associados a cada parâmetro.
    2. **Null Test de Bypass (verificação de integridade):** Adicionar uma trilha extra com o NAM-rs em **bypass**, sinal idêntico em paralelo com fase invertida e compensação automática de atraso (ADC) ativa. O resultado deve ser silêncio absoluto (null perfeito), confirmando que o bypass não altera o sinal. *(Nota: o null test NÃO se aplica com processamento neural ativo, pois o NAM transforma a forma de onda.)*
    3. **Teste de Determinismo:** Com processamento ativo, renderizar (bounce) o projeto 2 vezes consecutivas e comparar os arquivos WAV bit-a-bit (`cmp` ou `diff`). Resultados devem ser idênticos (determinismo de processamento).
    4. Iniciar um script de monitoramento em segundo plano que captura a cada 30 segundos:
       * Memória residente (RSS) e virtual (VSZ) via `/proc/<PID>/status`.
       * Contagem de File Descriptors (`ls /proc/<PID>/fd | wc -l`).
       * Contagem de threads (`ls /proc/<PID>/task | wc -l`).
       * XRUNs do PipeWire/JACK (`pw-top -b` ou log do host).
       * Gerar CSV (`timestamp,rss_kb,vsz_kb,fds,threads,xruns`) para análise pós-teste.
    5. Inspecionar o log de kernel (`dmesg -T`) e o arquivo de logs do host à procura de violações de thread access ou pânicos silenciosos.
  * **Aceite:**
    * Zero crashes ou travamentos nos hosts ao longo dos 60 minutos de reprodução contínua.
    * O consumo de memória RSS do processo do plugin/host deve estabilizar (variação menor que 5 MB após os primeiros 5 minutos de aquecimento, confirmando ausência de vazamentos no loop da GUI e nos buffers circulares).
    * CPU usage estável (variação < ±5% no percentil p95 após warmup de 5 minutos).
    * Zero crescimento de FDs ou threads após warmup.
    * Latência de áudio estável com zero XRUNs acumulados decorrentes do NAM-rs.
    * Null test de bypass: silêncio absoluto (< -120 dBFS).
    * Teste de determinismo: bounce 1 == bounce 2 (bit-identical).

---

* [ ] **Tarefa 2.4** — Stress Test Multi-Instância e Interação
  * **Contexto:**
    Cenários de uso real envolvem múltiplas instâncias do NAM-rs na mesma sessão. A interação entre instâncias (compartilhamento de recursos globais, contention de GC, race conditions de File Picker) deve ser testada explicitamente.
  * **Ações do Testador:**
    1. **Carregar/Destruir Instâncias durante Playback:**
       * Com playback ativo e 2 instâncias do NAM-rs processando, adicionar uma 3ª instância em tempo real. Verificar que as 2 primeiras não sofrem interrupção.
       * Remover (deletar) a 3ª instância durante playback. As 2 restantes devem continuar processando sem falhas.
    2. **File Picker Concorrente:**
       * Abrir o File Picker em 2 instâncias simultaneamente. Ambos devem funcionar independentemente (sem deadlock GTK/portal).
       * Selecionar arquivos em ambos — cada instância carrega seu modelo corretamente.
    3. **GUI Independente:**
       * Fechar a GUI de uma instância enquanto outra está aberta. A instância com GUI fechada deve continuar processando áudio normalmente.
       * Reabrir a GUI da instância fechada — o estado deve estar preservado (parâmetros, modelo carregado).
    4. **Bypass Independente:**
       * Com 3 instâncias: colocar a instância 1 em bypass, manter 2 ativa, carregar modelo na 3. Verificar que cada instância opera independentemente.
  * **Aceite:**
    * Zero interferência entre instâncias.
    * Cada instância mantém estado e processamento independentes.
    * Zero crashes ao adicionar/remover instâncias durante playback.
