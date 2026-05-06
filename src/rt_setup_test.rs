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
