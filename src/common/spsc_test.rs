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
    for i in 0..64 {
        let item = GcItem::Test(Box::new(counter.clone()));
        let ptr = Box::into_raw(Box::new(item));
        let old = overflow.push_raw(ptr);
        assert!(old.is_none(), "Slot {} deveria estar vazio", i);
    }

    // 2. Tenta inserir o 65º item (deve sobrescrever o 1º)
    let item_65 = GcItem::Test(Box::new(counter.clone()));
    let ptr_65 = Box::into_raw(Box::new(item_65));
    let old_ptr = overflow.push_raw(ptr_65);

    assert!(
        old_ptr.is_some(),
        "Deveria ter retornado o ponteiro do item sobrescrito"
    );

    // Limpeza manual do item sobrescrito (simulando o que o RT faz ou o leak intencional)
    unsafe {
        let _ = Box::from_raw(old_ptr.unwrap());
    }

    // 3. Valida que o drain retorna 64 itens
    let items = overflow.drain();
    assert_eq!(items.len(), 64);
}

#[test]
fn test_gc_stress_no_leak() {
    use std::sync::Arc;
    use std::sync::atomic::AtomicU32;

    let (mut gc_prod, mut gc_cons) = RingBuffer::new(32);
    let overflow = GcOverflowBuffer::new(32);
    let counter = Arc::new(AtomicU32::new(0));

    // Stress: 1000 trocas de "recursos"
    for _ in 0..1000 {
        let item = GcItem::Test(Box::new(counter.clone()));
        if let Err(rtrb::PushError::Full(returned_item)) = gc_prod.push(item) {
            // Se o canal principal encher, vai para o overflow
            let ptr = Box::into_raw(Box::new(returned_item));
            if let Some(leaked_ptr) = overflow.push_raw(ptr) {
                // Em caso de sobrescrita no overflow, dropamos manualmente aqui para o teste
                // (No RT real isso seria um forget/leak para não travar a thread)
                unsafe { drop(Box::from_raw(leaked_ptr)) };
            }
        }

        // Drena periodicamente para não acumular infinitamente
        super::drain_gc_channels(&mut gc_cons, &overflow);
    }

    // Drenagem final
    super::drain_gc_channels(&mut gc_cons, &overflow);

    // O contador de drops deve ser exatamente 1000 se não houve vazamentos
    // Nota: Arc::strong_count == 1 significa que todas as cópias enviadas ao GC foram dropadas
    assert_eq!(
        Arc::strong_count(&counter),
        1,
        "Ainda existem referências ativas! Vazamento detectado."
    );
}

#[test]
#[should_panic(expected = "GcOverflowBuffer: capacity deve ser maior que 0")]
fn test_gc_overflow_invalid_capacity() {
    let _ = GcOverflowBuffer::new(0);
}
