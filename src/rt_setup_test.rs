// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

use super::*;

#[test]
fn test_get_allowed_cpus_not_empty() {
    let allowed = get_allowed_cpus();
    // Em um sistema Linux normal, pelo menos a CPU atual deve estar permitida.
    assert!(
        !allowed.is_empty(),
        "A lista de CPUs permitidas não deve estar vazia."
    );
}

#[test]
fn test_select_optimal_cpu_returns_something() {
    // select_optimal_cpu deve retornar um núcleo válido no ambiente de teste.
    let cpu = select_optimal_cpu();
    assert!(
        cpu.is_some(),
        "Deveria ser possível selecionar um núcleo ideal."
    );

    if let Some(cpu_idx) = cpu {
        let allowed = get_allowed_cpus();
        assert!(
            allowed.contains(&cpu_idx),
            "O núcleo selecionado deve estar na lista de permitidos."
        );
    }
}

#[test]
fn test_parse_interrupts_basic() {
    let allowed = get_allowed_cpus();
    let irqs = parse_interrupts_per_cpu(allowed.len() + 1);
    assert_eq!(irqs.len(), allowed.len() + 1);
}

#[test]
fn test_rdtsc_nanos_monotonic() {
    // Garante que o tempo avança mesmo em fallback
    let t1 = rdtsc_nanos();
    std::thread::sleep(std::time::Duration::from_millis(1));
    let t2 = rdtsc_nanos();
    assert!(t2 > t1, "Tempo não avançou: {} -> {}", t1, t2);
}

#[test]
fn test_rdtsc_nanos_significant() {
    // Garante que o tempo retornado não é quase zero (problema do Instant::now().elapsed())
    // Note: se o sistema for muito rápido, pode ser pequeno, mas rdtsc_nanos
    // usa um BOOT_TIME estático, então deve ser pelo menos alguns microssegundos
    // desde que o binário de teste começou.
    std::thread::sleep(std::time::Duration::from_millis(5));
    let t = rdtsc_nanos();
    assert!(t > 1_000_000, "Tempo reportado muito baixo: {} ns", t);
}
