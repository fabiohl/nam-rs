---
trigger: always_on
---

<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Convenções de Testes

Para manter a organização e a performance do projeto, seguimos as seguintes regras de organização de testes:

## 1. Testes Unitários (Unit Tests)

Os testes unitários devem testar a lógica interna de cada módulo.

- **Arquivos Pequenos (< 300 linhas):** Devem manter os testes **inline** no final do arquivo, dentro de um bloco:

  ```rust
  #[cfg(test)]
  mod tests {
      use super::*;
      // ... testes ...
  }
  ```

- **Arquivos Grandes (>= 300 linhas):** Devem mover os testes para um arquivo separado com o sufixo `_test.rs` no mesmo diretório. O arquivo principal deve incluir o teste no final:

  ```rust
  #[cfg(test)]
  #[path = "nome_do_modulo_test.rs"]
  mod nome_do_modulo_test;
  ```

## 2. Testes de Integração (Integration Tests)

Testes que exercitam a API pública do crate ou múltiplos módulos integrados devem ser colocados no diretório `tests/` na raiz do projeto.

## 3. Benchmarks

Benchmarks de performance usando o framework `criterion` devem ser colocados no diretório `benches/` na raiz do projeto.

## 4. Requisitos de Código

- Todos os novos arquivos de teste devem incluir o cabeçalho de Copyright e Licença.
- Os testes não devem realizar alocações no heap se estiverem testando código do hot-path DSP (usar `CountingAllocator` quando necessário).
