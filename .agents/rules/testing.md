---
trigger: always_on
---

<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Testing Conventions

To maintain project organization and performance, we follow these test organization rules:

## 1. Unit Tests

Unit tests should test the internal logic of each module.

- **Small Files (< 300 lines):** Should keep tests **inline** at the end of the file, inside a block:

  ```rust
  #[cfg(test)]
  mod tests {
      use super::*;
      // ... tests ...
  }
  ```

- **Large Files (>= 300 lines):** Should move tests to a separate file with the suffix `_test.rs` in the same directory. The main file should include the test at the end:

  ```rust
  #[cfg(test)]
  #[path = "module_name_test.rs"]
  mod module_name_test;
  ```

## 2. Integration Tests

Tests that exercise the crate's public API or multiple integrated modules should be placed in the root `tests/` directory.
Important: When creating new tests, always judge whether a test is worth running with every `cargo test` or if it can be moved to `utils/tests-long.sh`. Very long tests are a temptation to skip. `cargo test` should be reserved only for things with a risk of breaking on every commit, which truly need to be verified always.

## 3. Benchmarks

Performance benchmarks using the `criterion` framework should be placed in the root `benches/` directory.
Important: When creating new benchmarks, always judge whether a benchmark is worth running with every `cargo bench` or if it can be moved to `utils/tests-long.sh`. Very long benchmarks are a temptation to skip. `cargo bench` should be reserved only for things with a risk of breaking on every commit, which truly need to be verified always.

## 4. Code Requirements

- All new test files must include the Copyright and License header.
- Tests must not perform heap allocations if testing hot-path DSP code (use `CountingAllocator` when necessary).
