// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Kernel GEMV BF16 — operação de projeção linear com pesos em formato BF16.
//!
//! Placeholder: `gemv_overwrite_bf16` ainda não possui implementação SIMD nativa.
//! O dispatch atualmente usa `gemv_overwrite_bf16_fallback` (escalar) em `common/scalar_ref.rs`.
//!
//! # Futuro
//!
//! Quando um kernel AVX-512 BF16 para GEMV for implementado, ele será adicionado aqui.
