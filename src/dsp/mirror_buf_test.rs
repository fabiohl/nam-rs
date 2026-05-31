// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use libc::sysconf;

#[test]
fn test_mirror_buf_page_alignment() -> Result<(), Box<dyn std::error::Error>> {
    let page_size = unsafe { sysconf(_SC_PAGESIZE) } as usize;
    let element_size = std::mem::size_of::<f32>();

    // Pede 1 elemento, deve arredondar para 1 página
    let buf = MirroredBuffer::<f32>::new(1)?;
    let expected_elements = page_size / element_size;

    assert_eq!(buf.size(), expected_elements);
    assert_eq!(buf.len(), expected_elements * 2);
    Ok(())
}

#[test]
fn test_mirror_buf_mirroring() -> Result<(), Box<dyn std::error::Error>> {
    // Teste do Espelhamento: Verifica se o "truque" da memória virtual está funcionando.
    // Qualquer valor escrito na primeira metade deve aparecer instantaneamente na
    // segunda metade (o espelho), pois ambas as janelas apontam para o mesmo lugar físico.
    // Cria um buffer pequeno (será arredondado para 1 página)
    let mut buf = MirroredBuffer::<u32>::new(1)?;
    let size = buf.size();

    // 1. Escrita no início da primeira metade
    buf[0] = 0x12345678;
    // Deve ser visível no início da segunda metade (espelho)
    assert_eq!(buf[size], 0x12345678);

    // 2. Escrita no final da primeira metade
    buf[size - 1] = 0xDEADBEEF;
    // Deve ser visível no final da segunda metade
    assert_eq!(buf[2 * size - 1], 0xDEADBEEF);

    // 3. Acesso contíguo cruzando a fronteira (o "Pulo do Gato")
    // Vamos escrever uma sequência de valores que atravessa exatamente o meio do buffer.
    // Em um buffer comum isso exigiria dois loops ou um 'if', mas aqui é linear.
    let middle = size;
    let start = middle - 8;
    for i in 0..16 {
        buf[start + i] = i as u32;
    }

    // Verifica se a primeira metade (original) reflete as mudanças
    // Os primeiros 8 valores foram escritos no final da primeira metade
    for i in 0..8 {
        assert_eq!(buf[size - 8 + i], i as u32);
    }
    // Os próximos 8 valores foram escritos no início da segunda metade,
    // o que deve ter modificado o início da PRIMEIRA metade física.
    for i in 8..16 {
        assert_eq!(buf[i - 8], i as u32);
    }
    Ok(())
}

#[test]
fn test_mirror_buf_clone() -> Result<(), Box<dyn std::error::Error>> {
    let mut buf = MirroredBuffer::<i32>::new(100)?;
    buf[0] = 42;

    let buf2 = buf.clone();
    assert_eq!(buf2[0], 42);
    assert_eq!(buf2.size(), buf.size());

    // Modifica o original, o clone deve permanecer inalterado
    buf[0] = 99;
    assert_eq!(
        buf2[0], 42,
        "Clones de MirroredBuffer devem ser independentes"
    );
    Ok(())
}

#[test]
fn test_mirror_buf_zst_error() {
    assert!(MirroredBuffer::<()>::new(1024).is_err());
}

#[test]
fn test_mirror_buf_large_allocation() -> Result<(), Box<dyn std::error::Error>> {
    // Testa alocação de ~1MB
    let size = 1024 * 1024 / 4;
    let buf = MirroredBuffer::<f32>::new(size)?;
    assert!(buf.size() >= size);
    // Apenas garante que não deu panic e o mmap foi bem sucedido
    Ok(())
}

#[test]
fn test_mirror_buf_debug() -> Result<(), Box<dyn std::error::Error>> {
    let buf = MirroredBuffer::<f32>::new(1024)?;
    let debug_str = format!("{:?}", buf);
    assert!(debug_str.contains("MirroredBuffer"));
    assert!(debug_str.contains("ptr"));
    assert!(debug_str.contains("size_elements"));
    Ok(())
}
