// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Seleção de núcleo de CPU ideal e análise de interrupções.
//!
//! Implementa heurísticas para fixar a thread DSP no core com maior
//! capacidade de processamento e menor carga de IRQ, respeitando
//! máscaras de afinidade do kernel (isolcpus, cgroups).

/// Seleciona o núcleo de CPU ideal para fixar a thread RT (core affinity).
///
/// A heurística prioriza:
/// 1. **Filtro de Afinidade** — Respeita os núcleos autorizados pelo SO (isolcpus, cgroups).
/// 2. **Capacidade Máxima** — Em arquiteturas híbridas (ex: big.LITTLE), prefere núcleos de alta performance.
/// 3. **Menor Carga de IRQ** — Minimiza jitter causado por interrupções de hardware (rede, disco, etc).
/// 4. **Índice Numérico** — Desempate determinístico.
pub fn select_optimal_cpu() -> Option<usize> {
    use std::fs;

    // 1. Obtém os núcleos que o sistema operacional permitiu para este processo.
    let allowed_cpus = get_allowed_cpus();
    if allowed_cpus.is_empty() {
        log::warn!("Could not detect allowed CPUs via sched_getaffinity. Using total fallback.");
    }

    // Varre /sys para encontrar todos os núcleos lógicos disponíveis.
    let cpu_dir = fs::read_dir("/sys/devices/system/cpu").ok()?;
    let mut cpus: Vec<usize> = cpu_dir
        .filter_map(|entry| {
            let name = entry.ok()?.file_name();
            let name = name.to_str()?;
            // Filtra apenas diretórios no formato 'cpuX'.
            let cpu_idx = name.strip_prefix("cpu")?.parse::<usize>().ok()?;

            // Se tivermos a lista de permitidos, filtramos por ela.
            if !allowed_cpus.is_empty() && !allowed_cpus.contains(&cpu_idx) {
                return None;
            }

            Some(cpu_idx)
        })
        .collect();

    if cpus.is_empty() {
        return None;
    }
    cpus.sort_unstable();

    // Obtém a capacidade de processamento de cada núcleo.
    let capacities: Vec<(usize, u64)> = cpus
        .iter()
        .map(|&cpu| {
            let path = format!("/sys/devices/system/cpu/cpu{cpu}/cpu_capacity");
            let cap = fs::read_to_string(path)
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok())
                .unwrap_or(1024); // Fallback padrão do kernel.
            (cpu, cap)
        })
        .collect();

    // Totaliza as interrupções processadas por cada core para detectar "vizinhos barulhentos".
    let irq_totals = parse_interrupts_per_cpu(cpus.len());

    capacities
        .iter()
        .map(|&(cpu, cap)| {
            let irqs = irq_totals.get(cpu).copied().unwrap_or(u64::MAX);
            (cpu, cap, irqs)
        })
        .max_by(|a, b| {
            // Heurística de Ordenação:
            // 1. Maior capacidade (cap)
            // 2. Menor número de interrupções (irqs) — invertido no cmp
            // 3. Maior índice de CPU (cpu)
            a.1.cmp(&b.1)
                .then_with(|| b.2.cmp(&a.2))
                .then_with(|| a.0.cmp(&b.0))
        })
        .map(|(cpu, _, _)| cpu)
}

/// Obtém a lista de CPUs permitidas para o processo atual via `sched_getaffinity`.
///
/// Respeita isolamento de CPUs (isolcpus), cgroups e máscaras de afinidade
/// impostas pelo sistema operacional ou pelo usuário (ex: taskset).
pub fn get_allowed_cpus() -> Vec<usize> {
    let mut allowed = Vec::new();
    unsafe {
        let mut cpuset: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut cpuset);

        // Chamada ao kernel para obter a máscara de afinidade do processo atual (pid 0)
        if libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut cpuset) == 0 {
            for i in 0..libc::CPU_SETSIZE as usize {
                if libc::CPU_ISSET(i, &cpuset) {
                    allowed.push(i);
                }
            }
        }
    }
    allowed
}

/// Analisa /proc/interrupts para extrair a carga de interrupções por núcleo.
///
/// Esta função realiza o parsing do formato tabular do kernel, onde cada coluna
/// (após a primeira) representa um contador para um núcleo específico.
pub fn parse_interrupts_per_cpu(num_cpus: usize) -> Vec<u64> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    let mut totals = vec![0u64; num_cpus];

    // Refatoração para Streaming: Evita alocação monolítica de string para /proc/interrupts.
    // Essencial para sistemas com alto número de CPUs onde o arquivo pode ser grande.
    let file = match File::open("/proc/interrupts") {
        Ok(f) => f,
        Err(_) => return totals,
    };
    let reader = BufReader::new(file);

    // Pula o cabeçalho (CPU0 CPU1 ...)
    for line_result in reader.lines().skip(1) {
        let line = match line_result {
            Ok(l) => l,
            Err(_) => break,
        };
        let trimmed = line.trim_start();
        // Localiza o delimitador do nome da interrupção (ex: "7: ...")
        let irq_end = trimmed.find(':').unwrap_or(0);
        if irq_end == 0 {
            continue;
        }

        // Filtra apenas interrupções numéricas (ignora NMI, LOC, etc por simplicidade).
        if !trimmed[..irq_end]
            .trim()
            .bytes()
            .all(|b| b.is_ascii_digit())
        {
            continue;
        }

        let after_colon = match trimmed.get(irq_end + 1..) {
            Some(s) => s,
            None => continue,
        };

        // Acumula os contadores para cada core presente na linha.
        for (cpu_idx, token) in after_colon.split_whitespace().enumerate() {
            if cpu_idx >= num_cpus {
                break;
            }
            if let Ok(count) = token.parse::<u64>() {
                totals[cpu_idx] += count;
            } else {
                // Para no primeiro token não numérico (geralmente o nome do driver/dispositivo).
                break;
            }
        }
    }

    totals
}
