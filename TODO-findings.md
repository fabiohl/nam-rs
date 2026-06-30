<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-findings.md — Levantamento de Pontos e Propostas de Solução

Este documento agrupa as análises e constatações obtidas pelas auditorias da CLI, do CLAP e do processo de build de release, propondo soluções técnicas.

---

## Findings (Constatações)

### 1. Simplificação de Parâmetros de Oversampling (HQ Mode) na CLI

* **ID:** F01
* **Problema:** A CLI do nam-rs expõe tanto a opção `--oversample` quanto o atalho `--os`. Para simplificar a interface de usuário da CLI, o atalho `--os` deve ser removido, mantendo-se exclusivamente `--oversample`.
* **Proposta de Solução:**
  * No arquivo [cli.rs](file:///home/fabio/nam-rs/src/standalone/cli.rs), remover a exibição de `--os` na função `print_help()`.
  * Na função `parse_args_from()`, remover a correspondência de `Long("os")` na leitura do parser do `lexopt`.
  * Ajustar eventuais testes unitários em [cli.rs](file:///home/fabio/nam-rs/src/standalone/cli.rs) para garantir que `--oversample` funcione e `--os` seja rejeitado ou não testado.

### 2. Auditoria do Switch de Oversampling no CLAP

* **ID:** F02
* **Problema:** O usuário deseja verificar se a alteração do switch de oversampling no GUI do CLAP realmente surte efeito antes de carregar o modelo NAM e durante o processamento.
* **Análise Técnica:**
  * **Antes de carregar o NAM:** A função `model_process_stereo_with_os` em [inference.rs](file:///home/fabio/nam-rs/src/dsp/pipeline/stages/inference.rs) processa o áudio mesmo que `active_model_l` seja `None`. Se o oversampling estiver ativado (2x ou 4x), o sinal é upsampled, passa pelo bloco `passthru`, e é downsampled para a taxa de amostragem nativa. Portanto, o switch funciona e altera a cadeia de processamento (filtros de banda intermediária continuam operando) mesmo sem modelo carregado.
  * **Durante o processamento:** Quando um modelo NAM está ativo, o processamento ocorre integralmente na taxa sobreamostrada (2x ou 4x). A mudança do fator de sobreamostragem dispara `RT_STATUS_NEEDS_OS_REBUILD`. A thread principal reconstrói as estruturas off-RT em [housekeeping.rs](file:///home/fabio/nam-rs/src/clap/plugin/main_thread/housekeeping.rs) e as envia via canal SPSC de forma segura para a thread de áudio, que realiza o hot-swap em `process_events()`.
  * **Conclusão:** O mecanismo é robusto, seguro em tempo real (RT-safe) e tem efeito imediato. Não há necessidade de alterações de código no CLAP para este ponto, apenas confirmação da integridade.

### 3. Simplificação de Parâmetros de Activation Precision na CLI

* **ID:** F03
* **Problema:** A CLI expõe `--activation` e o atalho `--act`. Para simplificar, o atalho `--act` deve ser removido, restando apenas `--activation`.
* **Proposta de Solução:**
  * Em [cli.rs](file:///home/fabio/nam-rs/src/standalone/cli.rs), remover `--act` do help text.
  * Em `parse_args_from()`, remover a correspondência de `Long("act")` do loop de parsing de argumentos.
  * Atualizar o teste unitário `test_parse_args_activation_hf` para utilizar `--activation` em vez de `--act`.

### 4. Exibição de Activation Precision no Terminal (CLI)

* **ID:** F04
* **Problema:** A CLI não imprime a informação de qual modo de precisão de ativação (`ActivationPrecision`) está sendo utilizado no início, diferentemente do que ocorre para outros parâmetros.
* **Proposta de Solução:**
  * No arquivo [main.rs](file:///home/fabio/nam-rs/src/main.rs), logo após chamar `set_activation_precision(args.activation);`, adicionar uma linha de log informativa usando `log::info!`:

    ```rust
    log::info!(
        "{} Activation precision set to {:?}",
        "⚡".yellow(),
        args.activation
    );
    ```

### 5. Auditoria do Switch de Activation Precision no CLAP

* **ID:** F05
* **Problema:** O usuário deseja saber se o switch de precisão de ativação no GUI do CLAP surte efeito antes do carregamento e durante o processamento.
* **Análise Técnica:**
  * **Mecanismo:** A alteração do parâmetro no GUI atualiza o atômico global `ACTIVATION_MODE` (através de `set_activation_precision()`).
  * **Antes de carregar o NAM:** Sem modelo carregado, não há cálculos de rede neural ativos (portanto, não há chamadas a funções de ativação como tanh ou sigmoid). O switch altera o estado global atômico, mas não impacta o processamento bypass de áudio.
  * **Durante o processamento:**
    * **Modelos LSTM e A2:** Estes modelos realizam chamadas a `activation_precision()` em seu loop interno. Dependendo do valor (`Standard` ou `HighFidelity`), eles chaveiam dinamicamente entre aproximações rápidas (como Padé ou Minimax) e polinômios de alta precisão baseados em exp. A mudança do switch no GUI tem efeito imediato no próximo bloco de áudio.
    * **Modelos WaveNet:** As camadas internas de convolução do WaveNet no nam-rs utilizam uma aproximação polinomial de alta fidelidade constante (`simd_tanh_poly_avx2`/`avx512`) para estabilidade numérica e controle de aliasing inerentes a sua arquitetura. O switch de precisão de ativação não afeta os kernels WaveNet, mas afeta as saídas e outros modelos.
  * **Conclusão:** O switch funciona conforme o projetado. Nenhuma modificação de código necessária, apenas documentação/explicação.

### 6. Descrição Confusa do Parâmetro `--slim` na CLI

* **ID:** F06
* **Problema:** A descrição de `--slim` como `"Force quality level (overrides adaptive FSM) [default: auto]"` é confusa.
* **Proposta de Solução:**
  * Em [cli.rs](file:///home/fabio/nam-rs/src/standalone/cli.rs), alterar a string do help de `--slim` para:
    `Adaptive Compute (downgrades the active model) Auto = Under CPU Pressure [default: auto]`

### 7. Lentidão do `perf annotate` no Script de Build de Release

* **ID:** F07
* **Problema:** O comando `perf annotate` no script [build-release.sh](file:///home/fabio/nam-rs/utils/build-release.sh) demora cerca de 14 minutos para rodar.
* **Causa:** O comando é executado sem indicar o binário alvo específico (`perf annotate --stdio -i "$BOLT_DIR/perf.data"`). Isso faz com que o `perf` tente analisar e gerar anotações de código de máquina para todos os executáveis, bibliotecas compartilhadas do sistema (PipeWire, libc) e símbolos do kernel registrados no arquivo `perf.data`. Além disso, passar `"$NAM_RS_BIN"` diretamente faz o `perf annotate` tratá-lo como um nome de símbolo (não um binário/DSO), resultando em um arquivo `target/dsp_hotpath.asm` vazio (0 bytes).
* **Proposta de Solução:**
  * Limitar a anotação ao binário do nam-rs passando a flag `-d nam-rs` (ou `-d $(basename "$NAM_RS_BIN")`) ao comando `perf annotate`:

    ```bash
    perf annotate --stdio -i "$BOLT_DIR/perf.data" -d nam-rs > "target/dsp_hotpath.asm" 2>/dev/null || true
    ```

### 8. Erro Falso de Símbolo `clap_entry` Ausente no Build de Release

* **ID:** F08
* **Problema:** O script de build exibe o erro `"Erro: Símbolo 'clap_entry' ausente no artefato CLAP distribuído!"`, mas os arquivos são gerados e funcionam.
* **Causa:** O script executa a validação usando:

  ```bash
  if ! nm -D "$CLAP_TARGET" | grep -q "clap_entry"; then
  ```

  Sob a opção `set -o pipefail` do Bash, se o comando da direita do pipe (`grep -q`) encontra o padrão buscado, ele sai imediatamente com código `0` e fecha seu descritor de entrada. Se o comando da esquerda (`nm -D`) ainda está despejando texto, ele recebe um sinal `SIGPIPE` ao tentar escrever na pipe fechada, encerrando com erro `141`. Como o `pipefail` propaga o erro de qualquer estágio da pipe, o pipeline inteiro retorna `141` (erro), fazendo o script acreditar que o símbolo não foi encontrado e imprimindo o erro falso (embora os arquivos já tivessem sido copiados anteriormente).
* **Proposta de Solução:**
  * Substituir `grep -q` por `grep ... >/dev/null`. O `grep` comum lê todo o fluxo de dados até o fim antes de fechar, evitando o `SIGPIPE` no `nm`.

    ```bash
    if ! nm -D "$CLAP_TARGET" | grep "clap_entry" >/dev/null; then
    ```

  * Corrigir o teste do `SONAME` adjacente pelo mesmo motivo:

    ```bash
    if ! readelf -d "$CLAP_TARGET" | grep SONAME >/dev/null; then
    ```

---

## Epics (Agrupamentos)

### Épico 1: Ajustes e Polimento de Interface CLI e Logs [DONE]

* Inclui **F01**, **F03**, **F04**, **F06**.
* Foco em limpar atalhos redundantes (`--os`, `--act`), esclarecer o `--slim` e introduzir a exibição no terminal do modo de precisão de ativação selecionado.

### Épico 2: Correção do Pipeline de Build de Release [DONE]

* Inclui **F07**, **F08**.
* Foco em acelerar o passo do `perf annotate` de 14 minutos para alguns segundos e corrigir a falha silenciosa de pipe do validador de símbolo `clap_entry` (eliminando a saída de erro falsa).

### Épico 3: Correção de Sintaxe do Relatório perf annotate

* Inclui **F07** (correção de sintaxe).
* Foco em alterar o argumento de `$NAM_RS_BIN` para `-d nam-rs` no comando `perf annotate` de `utils/build-release.sh` para restaurar a geração correta do relatório de assembly hotspot (`target/dsp_hotpath.asm`).
