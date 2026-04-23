# Ideias de pesquisa para as próximas versões 1.x e/ou 2.x

---

## Idéia 1. Resampler Sinc-SIMD Nativo & Fase Mínima (Tarefa I.3)

Esforço: 🔴 Alto · Impacto: 🟢 Alto (Performance & Latência) · Risco: 🔴 Alto

### 1.1 Diagnóstico e Justificativa

O `NamResampler` em `resampler.rs` utiliza atualmente a biblioteca `rubato` (`SincFixedIn`). Embora excelente, o `rubato` é uma solução genérica que opera sobre buffers flutuantes `Vec<f32>` e loops escalares na camada interna de filtragem Sinc-Kaiser. Isso deixa oportunidades significativas de desempenho em um software altamente calibrado para SIMD.

Além disso, o filtro FIR padrão é de **Fase Linear** (Kaiser-BlackmanHarris2, `sinc_len=256`).

- **Latência:** O group delay simétrico adiciona **~1.5 ms** de latência pura no roundtrip 96k → 48k → 96k.
- **Artefatos:** Filtros lineares geram *pre-ringing* audível, algo inexistente em amplificadores valvulados reais (sistemas de fase mínima).
- **Sobrecarga:** Dependência de alocadores externos e gerenciamento de estado opaco para o loop RT-DSP.

### 1.2 Implementação: Resampler Nativo SIMD

A meta é substituir o `rubato` por um motor polifásico FIR customizado no diretório `dsp/resampler.rs`:

1. **SIMD Nativo (`std::simd`):** Utilizar intrínsecos de 256-bit (YMM) e 512-bit (ZMM) para processar o kernel do resampler.
2. **Unrolling Estático:** Otimizar para blocos curtos e preditivos (ex: `1x64`, `1x128` frames de PipeWire), minimizando o custo de setup do filtro por callback.
3. **Zero-Alocação (RT-Safe):** Utilizar matrizes de `const generics` e buffers circulares estáticos para garantir que nenhum `malloc` ocorra no hot-path.
4. **Kernel Kaiser-Sinc:** Codificar internamente o multiplicador de fase Sinc e a janela de Kaiser para controle total da banda de transição e rejeição de aliasing.

Considerar trabalhar junto ao upstream do Rubato 2.0?

### 1.3 Abordagens de Fase (Linear vs Mínima)

Para reduzir a latência de ~1.5ms para valores próximos de zero no resampler, propõe-se duas abordagens:

#### **Abordagem A — Tabela estática de coeficientes MPS:**

1. Gerar os coeficientes Kaiser-Blackman.
2. Aplicar a **Transformada de Hilbert** (ou algoritmos como o de Cepstrum) para converter para fase mínima.
3. Injetar como tabela `const` no código, usada pelo resampler FIR customizado.

#### **Abordagem B — Controle de Group Delay Dinâmico:**

1. Implementar o resampler polifásico FIR com suporte a múltiplos kernels (Linear vs Minimum Phase).
2. Permitir que o usuário escolha entre "Latência Zero" (Fase Mínima) ou "Fase Perfeita" (Linear) via configuração.

### 1.4 Riscos Críticos e Validação

- **Resposta Audível:** A fase mínima redistribui a energia temporal. Requer validação perceptual (Double-blind ABX) com músicos para garantir que o "feel" do amplificador não foi degradado.
- **Invalidação de Golden Vectors:** Todos os testes de roundtrip e vetores de referência que passam pelo resampler precisarão ser re-gerados.
- **SNR e Aliasing:** Filtros de fase mínima podem ter rejeição de aliasing ligeiramente inferior na stopband em comparação com filtros lineares de mesmo tamanho. Requer análise rigorosa via FFT e impulso de Dirac.
- **Complexidade SIMD:** A implementação de filtros polifásicos com `std::simd` requer cuidado extra com o alinhamento de memória e os *tail blocks*.
