---
trigger: glob
description: Diretrizes e restrições para desenvolvimento Rust focado em Áudio Inferenciais, Processamento Neural e Kernel Linux.
globs: **/*.rs, **/*.toml
---

# Diretrizes de Aplicação (Rust focado em Inferência e PipeWire)

* **Stack Matemática e Perf:** Como padrão, assuma **Rust 1.94+** (edição 2024). A arquitetura apoia-se nativamente no `std::simd` para operações de Fused Multiply-Add (FMA), calculando topologias complexas (LSTM e WaveNet). Assegure conformidade irrestrita aos desdobramentos de microarquiteturas x86-64-v3 e AVX-512 via Multiversioning, expurgando cálculos `std::math` abstratos por polinômios otimizados (FastMath Minimax).
* **Gestão de Dependências e Crates:** Prefira abstenções; ative limitadas dependências como `pipewire-rs` em substituição aos pesados plugins de encapsulamento como `nih_plug`. Tudo reside de forma Standalone (PipeWire Client nativo). 
* **Divisão Estrita de Concorrência Lock-Free:**
  * O core executa fundamentalmente comunicação sob **SPSC Ring Buffers** transpondo parâmetros entre estado CLI Assíncrono (ex: input/output gain, troca de .namb, metadados de tone3000) para o injetor de DSP passivo.
  * O processo NÃO performa transações de E/S de captura no Kernel Linux. Todas as rotinas que invocam persistência de arquivos devem ser rejeitadas.
* **Regras de Tempo-Real absolutas (Thread de DSP):**
  * Configurada intrusivamente como `SCHED_FIFO` no sistema, se disponível.
  * O DSP de Áudio afixa estritamente um processador Lógico usando **Core Affinity** - inibindo falhas no Cache de L1/L2 (Core Migration).
  * **ZERO** alocação na Heap; Proibido instanciar `Vec`, `Box`, e fechamentos em tempo de execução.
  * Tensores modelados orientam-se exclusivamente como _SoA_ (Structure of Arrays) pré-alocados e submetidos via _const generics_ para abster-se da compilação de salto direcional dinâmico na microarquitetura (Branch Prediction Failures).
* **Tratamento de Falso Compartilhamento (False Sharing):**
  * Todas as estruturas transientes compartilhadas pelo Ring Buffer lock-free devem ser rigorosamente alinhadas usando a macro `#[repr(align(128))]`.
* **Tratamento de Erros:** Unwraps são restritos. Em threads DSP não se deve incorrer a interrupções dinâmicas sem tratamentos tolerantes lógicos ou avisos assíncronos silentes.
* **Comentários de código-fonte:** Pratique a documentação limpa.
  * Módulo (`//!`), structs (`///`), e logs de linha visando expor magias das iterações dos _Const Generics_ das CNN WaveNet ou portas LSTM.
