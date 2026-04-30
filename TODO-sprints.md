# 🚀 Backlog do Produto e Planejamento de Sprints Técnicas

**Modelo de prompt web:** Você é uma equipe de arquitetos sêniores e desenvolvedores especialistas na linguagem Rust e no subsistema de áudio Linux. Em anexo temos o aglutinado repomix do github atualizado do NAM-rs (vide subanexo README.md). Assumindo a persona/workflow .agents/workflows/diagnostico.md e .agents/skills/planejador-arquiteto/SKILL.md você vai detalhar ao máximo a implementação da tarefa técnica demandada logo abaixo. Sua resposta será concisa e direta, usando toda a sua janela de processamento e contexto para orientar o trabalho do(s) implementador(es).

---

## 🏛️ Épico: Otimização de Modelos & Inferência ✅ Concluído (2026-04-30)

*Foco: Reduzir a pegada de memória e aumentar o throughput dos núcleos neurais.*

> **Notas de Auditoria (2026-04-30):**
>
> - Todas as 7 tarefas (T1, T2, T3, T6, T7, T8, T13) aprovadas na revisão.
> - Benchmarks: WaveNet Standard 64samp = ~198 µs (85% margem RT @48kHz).
> - LSTM 2×16 64samp = ~19 µs (99% margem RT @48kHz).
> - Path dinâmico (fallback): regressão de ~6% investigada e **descartada** (Criterion p=0.74, dentro do ruído estatístico).
> - `clippy::missing_safety_doc` supressão removida; `# Safety` docs adicionadas a 16 funções em `simd.rs`.
> - `debug_assert!` adicionados a buffers stack `[0.0f32; 1024]` em wavenet.rs e wavenet_dyn.rs.
> - Artefatos de debug (`src/scratch.rs`, `src/scratch`) removidos.

### [T1] Suporte completo a processadores (AVX2 / AVX-512) [Concluido]

- **Problema:** Fragmentação de performance em hardware heterogêneo; falta de suporte explícito a instruções modernas.
- **Solução:** Maximizar a quantidade de código otimizado para AVX2-VNNI e AVX-512 via multiversioning. *(Nota: Esta sprint trata estritamente da infraestrutura de despacho e detecção; a lógica matemática com intrinsics VNNI ocorrerá nas sprints T3 e T6).*
- **Critérios de Aceite:** Binário otimizado detectando e usando o set de instruções mais rápido disponível no host.
- **Tags:** #simd #cpu #x86

### [T2] Monomorfização de Modelos (Static Dispatch via Enums) [Concluido]

- **Problema:** Overhead de vtable e falha de inlining devido ao `Box<dyn DynamicModel>` no hot-path.
- **Solução:** Substituir o trait object por um `enum` que encapsule os tipos de modelos, permitindo despacho estático.
- **Critérios de Aceite:** Redução do overhead de chamada e permissão para o compilador inlinar o kernel DSP.
- **Tags:** #simd #rust #performance

### [T3] SIMD F16C Weight Compression (VNNI-like) [Concluido]

- **Problema:** Gargalo de Memory Bound (L1 Cache Misses) em redes grandes.
- **Solução:** Implementar compressão f16 in-memory com expansão on-the-fly usando `_mm256_cvtph_ps`.
- **Critérios de Aceite:** Redução de cache misses na L1 e passagem nos Golden Vectors de v1.0.
- **Tags:** #simd #model #memory

### [T7] LSTM Batch Head Dot-Product (Mini-GEMM) [Concluido]

- **Problema:** Overhead redundante de setup de registradores para o head dot-product em cada sample.
- **Solução:** Acumular hidden states de K samples e processar como mini-GEMM (1 load de pesos para K FMAs).
- **Critérios de Aceite:** Ganho de 5-10% em benchmarks de LSTM.
- **Tags:** #simd #lstm

### [T6] AVX-512 VNNI/BF16 para Dot Product Interleaved [Concluido]

- **Contexto Arquitetural:** O projeto já possui despacho inteligente (`SimdMathConfig`) que prevê sub-classes AVX2 e AVX512. No entanto, em hardware de altíssima gama, o path FP32 puro desperdiça a imensa capacidade FMA da `_mm512_dpbf16_ps` que calcula produtos escalares simultâneos BFloat16 nativamente.
- **Problema:** O throughput do "dot product" nas convoluções é restrito pelo teto das portas FMA em FP32, limitando a execução de modelos densos com baixíssima latência.
- **Solução Proposta:** Incorporar ao trait genérico `SimdMath` (e nas funções de `SimdMathConfig` / `fastmath`) o novo kernel BF16. Envolver a conversão em tempo de load de pesos (f32 -> BF16 in-memory) ou converter iterativamente registradores `_mm512_cvtne2ps_pbh`. O método deve persistir assinaturas genéricas sem instabilizar matrizes estáticas.
- **Arquivos-Alvo:** `src/math/simd.rs`, `src/math/fastmath.rs`
- **Critérios de Aceite:** Benchmarks de infração validando ganhos absolutos sobre AVX2 em topologias pesadas (LSTM 2x16 / WaveNet Standard). Preservação da exatidão via `proptest_math.rs`.
- **Perfil do Implementador:** Engenheiro de Alta Performance (C/Rust Intrinsics).
- **Tags:** #simd #avx512 #bf16

### [T8] Fusão Residual + One-by-One no WaveNet Layer [Concluido]

- **Contexto Arquitetural:** A rotina do path do `DenseLayer` em WaveNet e `wavenet_dyn.rs` hoje divide computação de convolução, mix in, Tanh, e passes de 1x1 em estágios, resultando num array intermediário de soma e passes extras à cache L1.
- **Problema:** Múltiplos descarregamentos (stores e loads) redundantes da L1 Cache durante saltos do one-by-one de volta pra onda residual.
- **Solução Proposta:** Escrever e transicionar o modelo para uma abstração `process_single_frame_with_residual` no interior da struct `DenseLayer`. Essa função fará com que o Head Accumulate, o mix de 1x1 e o Residual Add operem no contexto exclusivo dos registradores do SIMD (YMM/ZMM), efetuando store no vetor final apenas no final da esteira.
- **Arquivos-Alvo:** `src/models/wavenet.rs`, `src/models/wavenet_dyn.rs`
- **Critérios de Aceite:** Zero overhead adicional em block_size pequenos e redução estatisticamente provada no benchmark WaveNet do Criterion (`cargo bench`). O `cargo test --test nam_infer_test` não pode ter degradação estocástica em Golden Vectors.
- **Perfil do Implementador:** Engenheiro DSP / Performance.
- **Tags:** #simd #wavenet #l1

### [T13] Prefetch Adaptativo por Dilatação (WaveNet Conv1D) [Concluido]

- **Contexto Arquitetural:** O loop WaveNet causal em suas camadas altas salta em taxas enormes de dilatação (e.g. D=512, D=1024), pulando trechos da memória.
- **Problema:** Software prefetch forçado via `T0` (todas as caches) não distingue padrões curtos de longos. Dilatações curtas (ex. D=1) já são inferidas com perfeição pelo *hardware prefetcher*, o hint de software é custoso e redundante. Dilatações vastas acabam engarrafando L1 prematuramente.
- **Solução Proposta:** Modificar o pipeline das camadas convolucionais (`Conv1d`) ou no macro-pass de block, determinando o flag do intrinsic `_mm_prefetch` (ex. `_MM_HINT_T0`, `_MM_HINT_T1`, ou `_MM_HINT_NTA`) a partir do valor estático ou da variante do kernel em Runtime.
- **Arquivos-Alvo:** `src/models/wavenet.rs`
- **Critérios de Aceite:** Nenhuma regressão nas camadas curtas e melhora marginais na latência Roundtrip final.
- **Perfil do Implementador:** Especialista em Microarquitetura.
- **Tags:** #simd #l1 #prefetch

---

## 🎸 Épico: Motor DSP & Performance de Baixo Nível

*Foco: Precisão audível, latência mínima e eficiência do ciclo RT.*

### [T5] Resampler Sinc-SIMD Nativo & Fase Mínima

- **Contexto Arquitetural:** Hoje usamos o crate `rubato 0.16` operando em FIR Sinc de fase linear, bidirecional planar.
- **Problema:** O filtro de fase linear causa ringing assimétrico "pré-eco" (pré-ringing), que em transients drásticos (e.g., palhetada forte de guitarras) suprime o *feel* de resposta da corda e adiciona ~1.5ms de latência pura matemática desnecessária (delay algorítmico).
- **Solução Proposta:** Abandonar `rubato` no núcleo quente. Implementar localmente em `src/dsp/resampler.rs` um filtro FIR Sinc Polifásico customizado (com suporte à Fase Mínima), otimizado por via vetorial com loops const-generics parecidos com a arquitetura do modelo.
- **Arquivos-Alvo:** `src/dsp/resampler.rs`
- **Critérios de Aceite:** Remoção drástica na latência final e fase audível mais cristalina e alinhada ao tempo zero; aderência total de `cargo bench` e `cargo test` para bypass planar 48kHz.
- **Perfil do Implementador:** Cientista DSP.
- **Tags:** #dsp #latency #simd

### [T4] Eliminação do `powf` no Callback RT (Gain Staging LUT)

- **Contexto Arquitetural:** O motor normaliza níveis baseado na calibração DBu com a equação logarítmica exp2. O ganho de entrada e saída é calculado com base nos perfis do modelo ou configurações da linha de comando de maneira iterativa.
- **Problema:** Empregos de `powf` vindo da biblioteca `libm` ou `std` destroem o pipeline de branch da CPU dentro de uma região crítica de inferência milisegundo e causam L1 I-Cache miss constante com saltos pesados de 200+ ciclos.
- **Solução Proposta:** Introduzir tabela estática pré-calculada (LUT) no construtor DSP ou formular uma aproximação de Padé / Minimax em `fastmath.rs` nomeada `fast_exp2` / `fast_pow2_f32` voltada a `gain_staging`.
- **Arquivos-Alvo:** `src/dsp/gain.rs`, `src/math/fastmath.rs`
- **Critérios de Aceite:** Margem de desvio absoluto de atenuação em decibéis ficar rigidamente contida num intervalo < 0.001 dB se aferida usando golden tests ou testes unitários locais com extrema varredura de `-120dB` a `+24dB`. Loop RT liberado da stdlib.
- **Perfil do Implementador:** Matemático Computacional / Engenheiro Rust.
- **Tags:** #dsp #performance

### [T9] Playback Bridge SIMD Copy (Non-Temporal Stores)

- **Contexto Arquitetural:** O Buffer compartilhado `DspBridge` despacha áudio já computado (saída final) usando cópia simples para os consumos da thread do Playback no WirePlumber.
- **Problema:** Como os buffers da stream de saída jamais serão re-lidos pela CPU (apenas transportados ao DMA via PipeWire), povoar o Cache de L1 com estas amostras expulsa as tabelas vitais de pesos da rede Neural da RAM rápida.
- **Solução Proposta:** Utilizar stores nativos não temporais (NTA). A instrução `_mm256_stream_ps` joga o slice do Array alinhado a 128-bytes diretamente na main-memory, despoluindo as linhas do L1.
- **Arquivos-Alvo:** `src/pw_host.rs` (nas operações do DspBridge).
- **Critérios de Aceite:** Benefício notório em block_sizes de >128. Coerência dos dados intocável.
- **Perfil do Implementador:** Especialista em Microarquitetura.
- **Tags:** #performance #l1

### [T10] Timing de Baixa Latência (RDTSC / `minstant`)

- **Contexto Arquitetural:** O `Instant::now()` é usado para medir o custo de DSP e exportá-lo na struct `RtStatusFlags` via diagnósticos.
- **Problema:** No kernel do linux, em determinados hosts, chamar Instant/Time triggers VDSO context switches onerosos, acarretando spikes inesperados de micro-stutter de CPU no SCHED_FIFO.
- **Solução Proposta:** Usar a instrução CPU-native `core::arch::x86_64::_rdtsc()` no inicio e fim de loops, abstraindo numa estrutura inline. O fator de conversão de clock/us pode ser amostrado fora da thread do RT com um sleep curto.
- **Arquivos-Alvo:** `src/pw_host.rs`, `src/diagnostics.rs`
- **Critérios de Aceite:** Zero variação sistemática, o loop RT precisa exibir latência de relógio constante de ~0 ciclos OS level (100% via CPU registers).
- **Perfil do Implementador:** Engenheiro Kernel Linux.
- **Tags:** #performance #lowlevel

### [T11] Resampler Short-Circuit Bypass (Zero-Copy)

- **Contexto Arquitetural:** Quando o hardware de áudio opera em 48.000 Hz, o resampler faz um bypass de software.
- **Problema:** Apesar do bypass aliviar a matemática complexa FIR, ele ainda copia fatias (slices) por rotinas do tipo `.copy_from_slice()`. Isso tem custo de L1 em sessões de produção rigorosas.
- **Solução Proposta:** Se `pw_rate == 48000`, o resampler não atua. Invés disso, os fatiadores recebem uma referência planar `&[f32]` provinda diretamente do Sink PipeWire do RT, canalizando os slices diretamente ao `DynamicModel` (que opera lock-free e in-place) sem intermediários.
- **Arquivos-Alvo:** `src/pw_host.rs`, `src/dsp/resampler.rs`
- **Critérios de Aceite:** Redução da pegada de memória do pipeline RT em 48kHz com ganhos no roundtrip-latency de ponta. Código unitário atestando estabilidade estática e comportamental do buffer original.
- **Perfil do Implementador:** Arquiteto Rust sênior.
- **Tags:** #dsp #resampler #zerocopy

### [T14] Histerese Dinâmica em Otimizações de Sinal (Mono/Silêncio)

- **Contexto Arquitetural:** Atualmente o `apply_gain_simd` e a inferência detectam instantaneamente blocos R/L clonados (uso mono) ou silêncio espectral puro (`0.0`), pulando ciclos matemáticos em R.
- **Problema:** Tocar acordes ou inserir transientes dinâmicos causa um cenário em que a condição silêncio/mono oscila furiosamente a cada mini-bloco de áudio. Avaliar a lógica a cada chamada RT adiciona carga e intermitência de aquecimento.
- **Solução Proposta:** Engajar *Histerese / Threshold Temporal*. A entrada num estado "Mono" ocorre quando `L==R` por no mínimo `N` buffers (ex. 10ms-20ms) - ou a saída do estado Mono pro Estéreo deve ser imediata na variação - garantindo estabilidade acústica contra o ruído analógico e otimizações fixas em longos períodos de backing-tracks com L/R hard.
- **Arquivos-Alvo:** `src/pw_host.rs`, `src/dsp/gain.rs`
- **Critérios de Aceite:** Smooth transitions em benchmarks de áudio real com fade-ins contínuos.
- **Perfil do Implementador:** Engenheiro DSP / Áudio.
- **Tags:** #dsp #optimization

---

## 🌐 Épico: Arquitetura de Áudio & PipeWire

*Foco: Simplificação da infraestrutura de roteamento.*

### [T12] Zero-Copy DspBridge (Processamento In-Place)

- **Contexto Arquitetural:** A abordagem PipeWire se declara como um `Audio/Sink` para receber fluxos do sistema, porém, a documentação atesta que o `pw_stream` copia os dados antes da callback. Devido a isso, NAM-rs injeta um segundo Node de Output e faz as vias de um `DspBridge` entre eles com atomics e buffers lock-free.
- **Problema:** A macro-arquitetura atual requer um ping-pong de stream e aloca 8192 floats x2 canais ininterruptamente. Isso agrava L1 evictions gerais e dificulta sync-clocks absolutos com interface USB.
- **Solução Proposta:** Evoluir e refatorar a infraestrutura de PipeWire adotando o formato `Audio/Filter` (SPA Node Type). Este formato recebe *In* e *Out* no mesmo pulso de DSP RT do kernel, processando o áudio 100% via In-place Replacement e extirpando a barreira DspBridge por completo!
- **Arquivos-Alvo:** `src/pw_host.rs`, `src/main.rs`
- **Tarefas de Implementação:**
  1. Alterar a lógica do PipeWire de Sink/Src para `pw_filter` e `media.class = Audio/Filter`.
  2. Apagar toda a lógica de atomics, fence generation e copy slices pertinente ao DspBridge antigo.
  3. Roteamento transparente via WirePlumber será garantido.
- **Critérios de Aceite:** Carga de CPU em idle e inferência atestadamente menor. Nenhuma instabilidade na detecção do hardware de placa de som.
- **Perfil do Implementador:** Especialista em Kernel Linux / PipeWire / Rust nativo.
- **Tags:** #pipewire #arch #latency
