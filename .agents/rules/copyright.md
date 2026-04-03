---
trigger: glob
description: Diretriz obrigatória para inclusão de aviso de Copyright nos arquivos de código.
globs: **/*
---

# Aviso de Copyright e Propriedade

* **Obrigatoriedade**: Em todo arquivo de código (novo ou modificado), você deve **sempre assegurar a presença**, em seu cabeçalho, do comentário de aviso de copyright.
* **Texto Padrão**: Utilize obrigatoriamente o seguinte modelo de texto (adaptando como bloco ou linhas de comentário para a linguagem do arquivo correspondente):

  ``` text
  Copyright 2026 Fábio Henrique de Lima Silva. Todos os direitos reservados.
  Este arquivo é confidencial e propriedade de Fábio Henrique de Lima Silva. O uso não autorizado é estritamente proibido.
  ```

* **Boas Práticas e Posicionamento**: Sempre coloque o aviso de forma fluida e profissional no topo do arquivo. Em arquivos de script que contenham *shebang/hashbang* (ex.: `#!/bin/bash`), aloque o bloco de copyright imediatamente abaixo do mesmo. Tenha zelo para que o formato do comentário não quebre a sintaxe do arquivo de código em vigência. Atentar para que o ano 2026 seja substituído pelo ano correspondente (se for o caso), porém apenas nos arquivos que estiverem sendo editados.
* **Formatos por Linguagem**:
  * **Rust / SQL / JavaScript / TypeScript / C**: `// Copyright …`
  * **Python / Shell / TOML / ini**: `# Copyright …`
  * **HTML / Markdown / XML**: `<!-- Copyright … -->` — em Markdown (`*.md`), prefira omitir o comentário para não poluir a renderização, exceto quando explicitamente solicitado. Arquivos `.md` em `docs/` não carregam copyright obrigatório.
* **Arquivos Não Aplicáveis**: `Cargo.lock`, arquivos binários e recursos de imagem gerados automaticamente não precisam de cabeçalho.
