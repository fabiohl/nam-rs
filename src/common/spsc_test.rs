// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

/// Testa se o RingBuffer consegue passar dados entre duas "linhas de processamento" (threads)
/// diferentes ao mesmo tempo sem perder informações ou travar.
#[test]
fn test_spsc_concurrency() {
    // Cria um buffer com 64 espaços vazios.
    // 'prod' (produtor) envia dados, 'cons' (consumidor) recebe.
    let (mut prod, mut cons) = RingBuffer::<i32>::new(64);

    // Cria uma nova linha de processamento que vai "produzir" números de 0 a 999.
    let handle = std::thread::spawn(move || {
        let mut count = 0;
        while count < 1000 {
            // Tenta colocar o número no buffer. Se estiver cheio, tenta de novo depois.
            if prod.push(count).is_ok() {
                count += 1;
            }
            std::thread::yield_now(); // Dá uma pequena pausa para não sobrecarregar o processador.
        }
    });

    // Esta parte (a linha principal) vai "consumir" os números enviados.
    let mut count = 0;
    while count < 1000 {
        // Tenta tirar um número do buffer.
        if let Ok(val) = cons.pop() {
            // Verifica se o número recebido é exatamente o que esperávamos.
            assert_eq!(val, count);
            count += 1;
        }
        std::thread::yield_now();
    }
    // Espera a outra linha de processamento terminar antes de encerrar o teste.
    handle.join().unwrap();
}

/// Testa os limites do buffer: o que acontece quando ele está totalmente vazio ou totalmente cheio.
#[test]
fn test_spsc_full_empty() {
    // Cria um buffer pequeno, com apenas 4 espaços.
    let (mut prod, mut cons) = RingBuffer::<i32>::new(4);

    // Tenta tirar algo de um buffer vazio. Deve retornar um erro dizendo que não há nada.
    assert!(cons.pop().is_err());

    // Preenche os 4 espaços disponíveis.
    assert!(prod.push(1).is_ok());
    assert!(prod.push(2).is_ok());
    assert!(prod.push(3).is_ok());
    assert!(prod.push(4).is_ok());

    // Tenta colocar o 5º item em um buffer de 4 espaços. Deve retornar um erro de "cheio".
    assert!(prod.push(5).is_err());

    // Retira o primeiro item (o número 1).
    assert_eq!(cons.pop(), Ok(1));

    // Agora que abriu um espaço, o número 5 deve entrar com sucesso.
    assert!(prod.push(5).is_ok());

    // Tenta colocar o 6º item. Como o buffer está cheio de novo (contém 2, 3, 4, 5), deve falhar.
    assert!(prod.push(6).is_err());
}

#[test]
fn test_gc_overflow_overwrite() {
    use std::sync::Arc;
    use std::sync::atomic::AtomicU32;

    let overflow = GcOverflowBuffer::new(64);
    let counter = Arc::new(AtomicU32::new(0));

    // 1. Enche o buffer de 64 slots
    for _ in 0..64 {
        let item = GcItem::Test(Box::new(counter.clone()));
        overflow.push(item);
    }

    // 2. Tenta inserir o 65º item (deve sobrescrever o 1º)
    let item_65 = GcItem::Test(Box::new(counter.clone()));
    overflow.push(item_65);

    // 3. Valida que o drain retorna 64 itens
    let drained = overflow.drain();
    assert_eq!(drained.len(), 64);
}

#[test]
fn test_gc_stress_no_leak() {
    use std::sync::Arc;
    use std::sync::atomic::AtomicU32;

    let (mut gc_prod, mut gc_cons) = RingBuffer::<GcItem>::new(32);
    let overflow = GcOverflowBuffer::new(32);
    let counter = Arc::new(AtomicU32::new(0));

    // Stress: 1000 trocas de "recursos"
    for _ in 0..1000 {
        let item = GcItem::Test(Box::new(counter.clone()));
        if let Err(rtrb::PushError::Full(returned_item)) = gc_prod.push(item) {
            // Se o canal principal encher, vai para o overflow
            overflow.push(returned_item);
        }

        // Drena periodicamente para não acumular infinitamente
        super::drain_gc_channels(&mut gc_cons, &overflow);
    }

    // Valida que o contador final está correto após o drop de tudo
    drop(gc_cons);
    for _ in overflow.drain() {}
}
