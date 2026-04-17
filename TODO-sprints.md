# TODO-sprints

## Sprint 1

### Tarefa 1.1: Fused Gate GEMV para LSTM (Item 2 — Sprint Avançado)

Estimativa: ~6h | Complexidade: Alta | Ganho: −20–30% latência LSTM/sample

#### Motivação

O hot-path do LSTM executa H4 dot products independentes em série (um por porta), onde cada invocação de dot_product tem overhead fixo de setup (~3 inst.) e reduce (~8 inst.). Para H=16 (H4=64, IH=17): 1088 FMAs sequenciais limitadas por
latência, não throughput.

#### Proposta Técnica

Processar 4 linhas da matriz de pesos simultaneamente aproveitando que o vetor `state` é compartilhado entre as 4 portas (Input, Forget, Cell, Output):

- Carregar `state` uma vez, reutilizar 4× → reduz bandwidth de leitura de 4 IH para 1 IH
- 4 FMAs independentes → ILP total no pipeline (4 ciclos de latência × 4 acumuladores)
- H4 é sempre múltiplo de 4 → nenhum remainder

#### Arquivos Afetados

- src/models/lstm.rs — macro `define_lstm_process!` (linhas ~35-140)
- src/models/lstm_dyn.rs — LstmDynLayer::process_sample()
- src/models/dispatcher.rs — Ajustar loader de pesos para layout

#### Pré-Requisitos

- Transposição dos pesos na construção (thread Main) para layout (dot contíguo, custo pré-pago no loader, sem impacto RT)
- Manter backward compatibility com formatos .nam/.namb existentes

#### Riscos

- Mudança no layout SoA dos pesos afeta o dispatcher
- Requer ajuste nos testes golden e de paridade estático↔dinâmico
- Deve ser executado em BRANCH ISOLADO com revisão cuidadosa

#### Verificação

- cargo test (todos os golden tests devem manter MSE ≤ threshold)
- cargo bench --bench inference_bench (confirmar ganho na latência LSTM)
- Comparação A/B audível com modelo LSTM real

---

### Tarefa 1.2: Tanh Padé Grau 7 com Clamp Saturante (Item 4 — Sprint de Precisão)

Estimativa: ~3h | Complexidade: Média | Ganho: −40% erro MSE golden

#### Motivação

O simd_tanh atual usa Padé grau 5 com erro máximo ~5e-3, que acumula ~sqrt(20) × 5e-3 ≈ 2.2e-2 por 20 camadas WaveNet. Um grau 7 reduziria o MSE golden para ~1e-2 com apenas +2 ciclos/tanh.

#### Proposta Técnica

1. Adicionar coeficiente x^7 ao polinômio: c2 (derivado via Minimax Remez)
2. Clamp saturante: |x| > 4.97 → ±1.0 (tanh(4.97) = 0.999988)
3. Blend via _mm256_blendv_ps entre resultado polinomial e ±1.0

#### Pré-Requisitos

- Derivar coeficiente c2 via Minimax Remez Exchange sobre [-5, 5]
- Re-gerar golden vectors ou ajustar thresholds de aceitação

#### Arquivos Afetados

- src/math/fastmath.rs — simd_tanh (AVX2 + AVX-512)
- tests/ — golden vector thresholds

#### Verificação

- test_simd_fastmath_tanh_mse com threshold estreitado para ~5e-4
- cargo bench -- tanh (confirmar custo adicional ≤ 2 ciclos/invocação)
