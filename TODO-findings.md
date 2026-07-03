<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-findings.md — Análise de Diagnóstico do BUG-3

Este documento apresenta as descobertas detalhadas da análise estática e da preparação para o diagnóstico do BUG-3 associado ao travamento indefinido e potencial reset de sessão de desktop durante o teste `test_x2_aliasing_rejection`.

## 1. Descrição do Bug (BUG-3 ⚠️)

- **Componente afetado:** `src/dsp/oversample.rs` (exercitado por `src/dsp/oversample_test.rs`).
- **Sintomas:**
  - O teste `test_x2_aliasing_rejection` trava indefinidamente em builds `--release` (> 30s sem resposta).
  - O operador humano relatou reset completo da sessão GNOME/desktop ao tentar rodar esse teste em sua estação de trabalho principal.
  - O teste é atualmente ignorado (`#[ignore]`) e excluído da suíte de execução automática (`tests-long.sh`).

---

## 2. Análise Técnica e Causa Raiz Provável

### A. Ausência de Loops Infinitos no DSP Core

Uma análise estática de `OversampleEngine` e `X2Stage` revela que todos os loops de processamento interno de sinal são rigorosamente limitados:

- Loops externos iteram sobre o tamanho do slice de entrada.
- Loops internos do filtro half-band são limitados por constantes pequenas: `HB_DELAY = 12`, `HB_TAPS = 25`, `HB_ODD_COUNT = 12`.
- A inicialização do filtro utiliza a função `bessel_i0` que possui um limite rígido de 20 iterações.

Desta forma, um loop infinito matemático ou algorítmico puro **não** é o causador óbvio no processamento do sinal DSP.

### B. Vulnerabilidade de Comportamento Indefinido (Undefined Behavior - UB) em `AlignedVec`

A estrutura `AlignedVec<T>` em [aligned.rs](file:///home/fabio/nam-rs/src/math/common/aligned.rs) possui uma inconsistência crítica em seu gerenciamento de ciclo de vida de memória:

1. **Alocação vs Liberação:**
   Na inicialização via `with_capacity(capacity)`, a memória é alocada para `capacity` elementos:

   ```rust
   let layout = Layout::from_size_align(capacity * std::mem::size_of::<T>(), Self::ALIGN).unwrap();
   let ptr = unsafe { alloc(layout) };
   ```

   No entanto, o campo `len` é inicializado como `0`.

2. **Método `drop` Incorreto:**
   Durante o descarte do vetor (`drop`), o layout de liberação é recalculado com base no comprimento lógico `self.len`, e não na capacidade real alocada (`capacity`):

   ```rust
   let layout = Layout::from_size_align(self.len * std::mem::size_of::<T>(), Self::ALIGN).unwrap();
   unsafe { dealloc(self.ptr.as_ptr() as *mut u8, layout); }
   ```

   - **Se `len < capacity`:** O layout fornecido ao `dealloc` difere do layout original de alocação. Em Rust e C/C++, passar uma assinatura de tamanho incorreta ao desalocador corrompe os metadados do heap do alocador do sistema.
   - **Consequências:** Corrupção de memória do sistema, travamento (hang) do processo de testes, ou falha catastrófica da pilha de drivers gráficos/servidor de display (causando o reset do GNOME).

### C. Indexação não verificada com `get_unchecked`

A implementação de `upsample` e `downsample` faz uso intensivo de acessos diretos não verificados com `unsafe get_unchecked`. Se ocorrer qualquer desalinhamento aritmético residual (por exemplo, na lógica de atraso do filtro interpolador com tamanhos de bloco ímpares ou no estágio `X4`), isso pode levar a acessos fora dos limites que silenciosamente corrompem a pilha ou o heap em compilações de release.

---

## 3. Plano de Resolução Proposto (Epis)

### Épico E-1: Diagnóstico e Isolamento de Recursos Seguro

- **Objetivo:** Reproduzir o travamento sob isolamento e capturar informações de execução de baixo nível sem risco à máquina host.
- **Abordagem:**
  - Configurar um container temporário ou isolamento de cgroups com limite estrito de memória (e.g., 1 GB) para evitar reset de desktop por exaustão de recursos.
  - Executar o teste alvo envelopado com `timeout -s KILL 10` e suporte de instrumentação (`RUSTFLAGS="-Zsanitizer=address"` ou depuradores nativos).
  - Executar trace do processo (`perf top` ou `strace`) para discernir entre spin de CPU ou bloqueio em syscall.

### Épico E-2: Correção Estrutural de `AlignedVec` e Remoção de UB

- **Objetivo:** Corrigir o gerenciador de memória para que siga o contrato de desalocação do Rust.
- **Abordagem:**
  - Armazenar explicitamente a capacidade original (`capacity`) no struct `AlignedVec` ou forçar o alinhamento de `len` a `capacity` se a estrutura for de tamanho fixo.
  - Substituir temporariamente `get_unchecked` por indexações seguras na engine de oversampling durante a fase de depuração para verificar se algum pânico de fora-dos-limites é disparado.

### Épico E-3: Validação, Regressão e Reativação do Teste

- **Objetivo:** Assegurar integridade e performance do processamento sem aliasing.
- **Abordagem:**
  - Re-executar toda a suíte de testes rápida (`tests-quick.sh`).
  - Habilitar novamente `test_x2_aliasing_rejection` na suíte oficial assim que provado estável.
  - Aferir performance em microssegundos para atestar que nenhuma regressão de tempo real foi introduzida no callback de áudio.
