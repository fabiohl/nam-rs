// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Módulo raiz de operações matemáticas e inferência neural.
//!
//! O NAM-rs organiza sua infraestrutura matemática de forma modular para garantir
//! performance extrema (SIMD) e manutenibilidade. Este módulo coordena os kernels
//! de álgebra linear, funções de ativação e utilitários DSP.
//!
//! # Estrutura
//! - `common`: Traits, despacho dinâmico e implementações base (AVX2/AVX-512).
//! - `activations`: Funções de ativação otimizadas (tanh, sigmoid, etc.).
//! - `gemm`: Operações de matriz-vetor e dot product de alto throughput.
//! - `dsp`: Processamento de áudio (ganho, stereo, conversão).
//! - `lstm` & `wavenet`: Kernels especializados para cada arquitetura de modelo.
//! Contém funcionalidades estruturais para simulações como algoritmos vetoriais
//! massivamente paralelos.

pub mod activations;
pub mod common;
pub mod constants;
pub mod dsp;
pub mod gemm;
pub mod lstm;
pub mod wavenet;
