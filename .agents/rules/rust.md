---
trigger: glob
description: Diretrizes e restrições para desenvolvimento Rust focado em Áudio, Alta Performance e Kernel Linux.
globs: **/*.rs, **/*.toml
---

# Diretrizes de Aplicação (Rust focado em Áudio e Kernel Linux)

* **Stack Moderna e Perf**: Como padrão, assuma **Rust 1.94+** (edição 2024). A arquitetura deve utilizar o módulo nativo `core::arch::x86_64` (intrinsics como AVX2/SSE) com compilação condicional, ou em alternativa uma crate leve focada em stable (como `safe_arch`), preparando o projeto para operações matemáticas vetorizadas no DSP sem depender do Nightly nem unicamente da vetorização automática do LLVM.
* **Gestão de Dependências e Crates**: Busque ser estritamente minimalista nos `dependencies` e `features` do `Cargo.toml` - ativando apenas o necessário e reduzindo a superfície de bugs. Utilize sempre `cargo add` para testar adequação. Prefira a stdlib onde possível.
* **Divisão Estrita de Concorrência**:
  * O core de processamento de áudio deve ser dividido em estritamente duas threads:
    1. **Thread de Áudio (DSP/Callback)**: Restrita ao tempo-real absoluto. O papel desta thread é apenas receber bytes, inferir metadados perfeitamente (Bit Perfect) e empurrá-los.
    2. **Thread de I/O (Gravação)**: Opera fora do tempo-real. Grava os bytes vindos no buffer.
  * A comunicação entre elas obrigatoriamente se dará por um Ring Buffer Lock-Free de Produtor-Único/Consumidor-Único (SPSC) (ex: `ringbuf` ou `rtrb`).
* **Regras de Tempo-Real (Thread de Áudio)**:
  * Deve ser configurada com `SCHED_FIFO` ou política de baixa latência equivalente do OS hospedeiro, e atrelada a núcleos de CPU físicos de alta performance, livre de interrupções (Core Affinity).
  * Thread enxuta, sem risco de preempção forçada, migração de cores ou de congelar o sistema inteiro.
  * **ZERO** alocação na heap em tempo de execução (`Vec`, `Box`, `Arc` ou strings dinâmicas são probidos).
  * **ZERO** operações de entrada e saída (I/O) de disco.
  * **ZERO** mutexes ou primitivas de bloqueio de thread. Toda comunicação é via lock-free.
* **Tratamento de Falso Compartilhamento (False Sharing)**:
  * Dimensione buffers via const generics no tempo de compilação.
  * Todas as estruturas transientes compartilhadas pelo Ring Buffer devem ser estritamente alinhadas a 128 bytes usando a macro `#[repr(align(128))]`.
  * Mitigar falhas de branch prediction e cache L1/L2 sempre "hot" nas CPUs.
* **I/O Assíncrono com o Kernel Linux (Thread de Gravação)**:
  * A gravação ao disco deve obrigatoriamente usar o subsistema `io_uring` via crate `tokio-uring`.
  * Evite a todo custo syscalls síncronas de gravação direta.
* **Tratamento de Erros**: Utilize `anyhow::Result` e contextualize as falhas (`.context()`) nas threads de controle/IO. Em threads DSP não é possível fazer unwrap que gere panics. Erros devem ser sinalizados silenciosamente ao lock-free log. Ignorar erros silenciosamente é proibido na thread não-RTA.
* **Comentários de código-fonte:** Pratique a documentação limpa.
  * **Módulo (//!):** Cada arquivo com uma explicação de alto nível: o quê o módulo faz, por quê existe e como se encaixa na arquitetura.
  * **structs/traits/funções (///):** Contratos públicos documentados.
  * **Comentários inline (//)** onde as otimizações e magias lock-free não forem óbvias.
