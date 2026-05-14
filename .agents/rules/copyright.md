---
trigger: glob
description: Diretriz obrigatória para inclusão de aviso de Copyright nos arquivos de código.
globs: **/*
---

# Aviso de Copyright e Licenciamento

* **Obrigatoriedade**: Em todo arquivo de código (novo ou modificado), você deve **sempre assegurar a presença**, em seu cabeçalho, do comentário com o identificador SPDX e o aviso de copyright.
* **Texto Padrão**: Utilize obrigatoriamente o seguinte modelo de texto (adaptando como bloco ou linhas de comentário para a linguagem do arquivo correspondente):

  ``` text
  SPDX-License-Identifier: Apache-2.0
  Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
  ```

* **Boas Práticas e Posicionamento**: Sempre coloque o aviso de forma fluida e profissional no topo do arquivo. Em arquivos de script que contenham *shebang/hashbang* (ex.: `#!/bin/bash`), aloque o bloco de copyright imediatamente abaixo do mesmo. Tenha zelo para que o formato do comentário não quebre a sintaxe do arquivo de código em vigência. Atentar para que o ano 2026 seja substituído pelo ano correspondente (se for o caso), porém apenas nos arquivos que estiverem sendo editados.
* **Arquivos Não Aplicáveis**: `Cargo.lock`, arquivos temporários, arquivos binários e recursos de imagem gerados automaticamente não precisam de cabeçalho.
