<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Regressões do Proptest

Este diretório contém os arquivos de persistência de falha (failure persistence seeds) do framework `proptest`.

## Finalidade e Funcionamento

Quando um teste baseado em propriedades (Property-Based Testing) falha, o `proptest` gera e salva uma semente (seed) contendo a entrada exata que causou o pânico ou a quebra de asserção.

Nosso projeto está configurado para salvar essas regressões em `tests/proptest-regressions/` de forma organizada (via `FileFailurePersistence::SourceParallel`), evitando poluir a raiz do repositório.

## Importância do Versionamento

Manter estas sementes de falha rastreadas no controle de versão (Git) é uma **boa prática recomendada pelo `proptest`** por dois motivos principais:

1. **Repetibilidade no CI:** Garante que a integração contínua (CI) e outros desenvolvedores reexecutem imediatamente os casos de teste específicos que falharam no passado, prevenindo que bugs corrigidos reapareçam (regressão).
2. **Determinismo:** Testes com entradas aleatórias podem ser difíceis de reproduzir sem a semente exata da falha. O arquivo de persistência remove essa aleatoriedade para os erros conhecidos.

Se um teste que anteriormente falhava agora passa de forma consistente e a correção foi consolidada, o arquivo correspondente continuará servindo de base permanente para atestar a estabilidade daquela lógica.
