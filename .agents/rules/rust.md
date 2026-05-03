---
trigger: glob
description: Diretrizes e restrições para desenvolvimento Rust focado em Áudio Inferenciais, Processamento Neural e Kernel Linux.
globs: **/*.rs, **/*.toml
---

# Diretrizes de Aplicação (Rust focado em Inferência e PipeWire)

* **Porte de código C++:** Muito do trabalho a ser realizado envolverá portar código escrito em C++ a partir de outros projetos que subsidiam este projeto. Esteja preparado para navegar nestes projetos espelhados na pasta github.com/ ou em outros locais.
* **Stack Matemática e Perf:** Como padrão, assuma **Rust 1.95+** (edição 2024). A arquitetura apoia-se nativamente no `std::simd` para operações de Fused Multiply-Add (FMA), calculando topologias complexas (LSTM e WaveNet).
  * Assegure conformidade irrestrita aos desdobramentos de microarquiteturas x86-64-v3, expurgando cálculos `std::math` abstratos por polinômios otimizados (FastMath Minimax).
  * Busque sempre algoritimos que tirem proveito das modernas instruções x86-64-v3 (obrigatório) ou x86-64-v4/avx-512 (opcional, se agregar valor considerável, usando multiversioning).
  * Busque sempre construções de código que dêem ao compilador oportunidade a mais otimizações de código binário.
* **Gestão de Dependências e Crates:** Mantenha o mais enxuto possível. Nunca insira diretamente no toml, sempre use "cargo add" como helper. Procure ser o mais específico possível em crates e features, de modo a não inserir dependências não utilizadas e abrir possíveis brechas para bugs.
* **Divisão Estrita de Concorrência Lock-Free:**
  * O core executa fundamentalmente comunicação sob **SPSC Ring Buffers** transpondo parâmetros entre estado CLI Assíncrono (ex: input/output gain, troca de .namb, metadados) para o injetor de DSP passivo.
  * **DspBridge (Dual-Stream):** A comunicação inter-stream (capture→playback) usa um buffer `#[repr(align(128))]` compartilhado via ponteiro raw (`Box::leak`), sincronizado por `fence(Release/Acquire)` + `AtomicU64` (generation counter). Nunca use `Mutex`, `RwLock`, ou qualquer primitiva bloqueante entre callbacks RT.
  * O processo NÃO performa transações de E/S de captura no Kernel Linux. Todas as rotinas não essenciais deve ficar fora da thread RT Áudio/DSP.
* **Regras de Tempo-Real absolutas (Thread de DSP):**
  * Configurada como `SCHED_FIFO` (prioridade ~90).
  * O DSP de Áudio afixa estritamente um processador Lógico usando **Core Affinity** - inibindo falhas no Cache de L1/L2 (Core Migration).
  * **ZERO alocação na Heap:** Proibido instanciar `Vec`, `Box`, `String` ou fechamentos em tempo de execução no closure `.process()`.
  * **ZERO I/O:** Proibido `println!`, `eprintln!`, `dbg!`, ou qualquer syscall bloqueante (leitura/escrita de arquivos, sockets).
  * **ZERO Locks:** Proibido `Mutex`, `RwLock` ou `Spinlock` no caminho RT.
  * Tensores modelados orientam-se exclusivamente como _SoA_ (Structure of Arrays) pré-alocados e submetidos via _const generics_ para abster-se da compilação de salto direcional dinâmico na microarquitetura (Branch Prediction Failures).
* **Tratamento de Falso Compartilhamento (False Sharing):**
  * Todas as estruturas transientes compartilhadas via Ring Buffer lock-free devem ser rigorosamente alinhadas usando a macro `#[repr(align(128))]` (ex: `ParamPayload`).
* **Tratamento de Erros:** Unwraps são restritos. No caminho DSP use `.unwrap_or_else()` ou fallbacks silenciosos. Fora do RT, utilize o sistema estruturado em `src/diagnostics.rs` via `NamDiagnostic`.
* **Comentários de código-fonte:** Pratique a documentação limpa.
  * Módulo (`//!`), structs (`///`), e logs de linha visando expor magias das iterações dos _Const Generics_ das CNN WaveNet ou portas LSTM.
* **Debug Friendly:** Crie código amigável à skill .agents/skills/debugger/SKILL.md.
