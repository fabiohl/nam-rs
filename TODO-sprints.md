<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva. -->

# TODO Sprints — NAM-rs Performance & Usability Roadmap

> **Escopo:** Ganhos de performance tangíveis no hot-path DSP, bugs relevantes e limpeza de código com impacto significativo.
> **Fora de escopo:** Arquitetura A2, suporte CLAP, refatorações cosméticas ou ganhos meramente marginais.

---

## Sprint 1 — Performance: Hot-Path SIMD & WaveNet

> **Objetivo:** Eliminar gargalos mensuráveis no inner loop do WaveNet (estático e dinâmico).
> **Validação global:** `cargo test` + `cargo bench` (grupos `WaveNet_*`, `LSTM_*`) + Golden Vectors.

### T1.1 — Eliminar `unwrap_or` do inner loop do Conv1D dinâmico [CONCLUIDO]

- [ ] **Arquivos:** `src/models/wavenet_common.rs`
- [ ] **Problema:** As funções `process_single_frame` (linhas 292-307), `process_single_frame_bf16` (linhas 622-636), e `process_dual_frame_bf16` (linhas 414-417, 443-446) usam `m.get(out_c).copied().unwrap_or(0.0)` em ~20 call-sites dentro do inner loop mais quente. Cada chamada gera: (1) bounds check `cmp+jae`, (2) branch `Option::unwrap_or`, (3) cópia `copied()`. O path F32 de `process_dual_frame` (linhas 90-114) **já usa `*m.get_unchecked(out_c)`** sem `unwrap_or` — há assimetria.
- [ ] **Ação exata:**
  1. Em `process_single_frame` (L292-307): substituir cada `m.get(out_c).copied().unwrap_or(0.0)` por `*m.get_unchecked(out_c)`. O `debug_assert_eq!(num_frames * self.out_ch, block.len())` do caller já garante tamanho. Fazer o mesmo no remainder loop (L347).
  2. Em `process_single_frame_bf16` (L622-636): idem ao passo 1. Idem remainder (L676).
  3. Em `process_dual_frame_bf16` (L414-417 para f0, L443-446 para f1): substituir os 8 call-sites `m.get(out_c).copied().unwrap_or(0.0)` por `*m.get_unchecked(out_c)`, `*m.get_unchecked(out_c+1)`, etc. — espelhar exatamente o padrão de `process_dual_frame` F32 (L90-103) que já usa `*m.get_unchecked(out_c)`.
- [ ] **Não alterar:** `process_dual_frame` F32 (já está correto).
- [ ] **Teste:** `cargo test` sem regressão. `cargo bench` grupo `WaveNet_Dynamic` deve melhorar.

### T1.2 — `horizontal_sum` SIMD nativo (substituir fallback escalar) [CONCLUIDO]

- [ ] **Arquivos:** `src/math/simd/avx2.rs`, `src/math/simd/avx512.rs`, `src/math/simd/dispatch.rs`
- [ ] **Problema:** Na vtable de dispatch (`dispatch.rs` linhas 87-88, 106-107, 124-125, 143-144, 161-162), **todos** os backends roteiam `horizontal_sum` para `horizontal_sum_fallback` (loop escalar em `fallback.rs`). No hot-path do WaveNet (`wavenet.rs` L1305), `M::horizontal_sum::<HEAD>` é chamado uma vez **por frame** — para HEAD=8 e num_frames=64, são 64 invocações escalares por bloco.
- [ ] **Ação exata:**
  1. **AVX2** — Criar `horizontal_sum_avx2(ptr: *const f32, len: usize) -> f32` em `avx2.rs`:

     ```rust
     // Para len <= 8 (HEAD=8 típico):
     // Load 8 floats → _mm256_hadd_ps x2 → extract upper 128 → _mm_add_ps → extract scalar
     // Para len > 8: acumular em loop de 8, depois hadd o acumulador.
     ```

     Marcar com `#[target_feature(enable = "avx2")]`.
  2. **AVX-512** — Criar `horizontal_sum_avx512(ptr: *const f32, len: usize) -> f32` em `avx512.rs`:

     ```rust
     // Para len <= 16: _mm512_loadu_ps (com mask para len < 16) → _mm512_reduce_add_ps
     // Para len > 16: acumular em loop de 16, depois reduce.
     ```

     Marcar com `#[target_feature(enable = "avx512f")]`.
  3. **dispatch.rs** — Substituir os closures `|ptr, len| unsafe { super::fallback::horizontal_sum_fallback(ptr, len) }` pelas funções nativas:
     - Backends AVX2/Avx2Vnni: `horizontal_sum: |ptr, len| unsafe { super::avx2::horizontal_sum_avx2(ptr, len) }`
     - Backends AVX-512/*: `horizontal_sum: |ptr, len| unsafe { super::avx512::horizontal_sum_avx512(ptr, len) }`
     - Fallback: manter `horizontal_sum_fallback`.
- [ ] **Teste:** Reutilizar ou criar teste unitário comparando saída SIMD vs fallback para len=1,4,8,16,32.

### T1.3 — Vetorizar o loop de soma Head do WaveNet

- [ ] **Arquivo:** `src/models/wavenet.rs` (linhas 1303-1312)
- [ ] **Problema:** O loop final em `process_internal` é:

  ```rust
  for i in 0..num_frames {
      let head1_sum = unsafe { M::horizontal_sum::<HEAD>(head_ptr.add(i * HEAD)) };
      let final_sum = head1_sum + self.array2.head_outputs[i];
      output[pos + i] = final_sum * self.head_scale;
  }
  ```

  Após T1.2 o `horizontal_sum` será SIMD, mas o loop externo ainda itera frame-a-frame com adição e multiplicação escalares. Para 64 frames, são 64 iterações.
- [ ] **Ação exata:** Após o loop de `horizontal_sum`, processar a adição com `array2.head_outputs` e a escala por `head_scale` em batch SIMD:

  ```rust
  // 1. Reduzir cada HEAD-grupo para um f32 (já feito pelo horizontal_sum)
  //    Armazenar os resultados em um buffer temporário na stack:
  let mut head_sums = [0.0f32; WAVENET_MAX_NUM_FRAMES]; // 64 × 4B = 256B stack
  for i in 0..num_frames {
      head_sums[i] = unsafe { M::horizontal_sum::<HEAD>(head_ptr.add(i * HEAD)) };
  }
  // 2. Somar com array2 + escalar, em blocos de 8 (AVX2):
  unsafe {
      let scale = _mm256_set1_ps(self.head_scale);
      let mut j = 0;
      while j + 8 <= num_frames {
          let h1 = _mm256_loadu_ps(head_sums.as_ptr().add(j));
          let h2 = _mm256_loadu_ps(self.array2.head_outputs.as_ptr().add(j));
          let sum = _mm256_add_ps(h1, h2);
          let out = _mm256_mul_ps(sum, scale);
          _mm256_storeu_ps(output.as_mut_ptr().add(pos + j), out);
          j += 8;
      }
      // Tail escalar
      while j < num_frames {
          output[pos + j] = (head_sums[j] + self.array2.head_outputs[j]) * self.head_scale;
          j += 1;
      }
  }
  ```

  **Nota:** Se preferível, usar `M::` trait methods em vez de intrínsecas diretas para manter o dispatch.
- [ ] **Teste:** Golden vectors sem regressão numérica. `cargo bench` grupo WaveNet.

### T1.4 — Batch GEMV para `DenseLayerDyn::process_block`

- [ ] **Arquivo:** `src/models/wavenet_common.rs` (linhas 767-781, 791-811)
- [ ] **Problema:** `process_block` itera frame-a-frame chamando `M::gemv_overwrite` individualmente. O kernel batch `fused_add_gemm_batch` (usado em `process_acc_block`) já existe e processa todos os frames com melhor localidade de cache. A camada Dense de `rechannel` (IN=1→CH=16) e `head_rechannel` (CH=16→HEAD=8) são chamadas **uma vez por bloco** de até 64 frames — alto impacto.
- [ ] **Ação exata para `process_block` (F32):**
  Substituir o loop:

  ```rust
  // ANTES (loop por frame):
  for i in 0..num_frames {
      let in_slice = input.get_unchecked(i * self.in_size..(i + 1) * self.in_size);
      let out_slice = output.get_unchecked_mut(i * self.out_size..(i + 1) * self.out_size);
      M::gemv_overwrite(in_slice, &self.weights, &self.bias, out_slice, self.do_bias);
  }
  ```

  Por uma chamada batch:

  ```rust
  // DEPOIS (batch):
  M::gemv_overwrite_batch(input, &self.weights, &self.bias, output, num_frames, self.do_bias);
  ```

  **Se `gemv_overwrite_batch` não existir no trait `SimdMath`:** criar o método no trait com implementação default que faz o loop (fallback), e implementações otimizadas nos backends AVX2/AVX-512 que processem múltiplos frames por iteração reutilizando os pesos carregados nos registradores.
  **Alternativa mais simples (se a criação do método batch for complexa demais):** Não criar novo método — apenas manter a nota como oportunidade futura e focar nos T1.1-T1.3.
- [ ] **Idem para `process_block_bf16`** (L791-811): mesma transformação para o path BF16.
- [ ] **Teste:** `cargo bench` + golden vectors.
