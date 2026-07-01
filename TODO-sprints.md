<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Roadmap de Sprints — Paridade A2 Max vs NAMCore C++

> Roadmap gerado pela skill `planejador-arquiteto` sob a **premissa estratégica**: o C++ NAMCore
> é a única fonte de verdade; o motor Rust deve reproduzir exatamente a saída do C++ (golden
> vectors). O oráculo f64 é ferramenta de decomposição, **não** gate. Findings detalhados em
> [`TODO-findings.md`](TODO-findings.md) (PM-A a PM-F, Épicos N1 a N6).

---

## Princípio de execução (inviolável)

1. **Medir antes de cortar:** registrar ESR/SNR/MSE vs `golden_wavenet_a2_max.bin` **antes** e **depois** de qualquer mudança na produção.
2. **C++ é o veredito:** nunca "corrigir" a produção rumo ao oráculo. Se oráculo↔produção divergem, o C++ golden arbitra; se o oráculo diverge do C++, o oráculo está errado (PM-A).
3. **Mudanças atômicas e reversíveis:** uma divergência por commit; golden como feedback loop.
4. **Barreira real = golden vs C++:** `test_golden_vectors_wavenet_a2_max` é o gate; os testes `test_oracle_*_a2_generic` são decomposição (reativados só em S6, após PM-A reconciliada).

---

## Sprint S1 — Reverter o erro da Rodada 4 e restaurar o baseline correto vs C++ (Épico N1)

**Risco:** ZERO (revert a estado já commitado em `84c3ad1`). **Bloqueante.**
**Especialista:** implementador (Rust).

* **Tarefa S1.1:** Reverter as mudanças de código não-commitadas da Rodada 4 ao estado do commit `84c3ad1`: [DONE]

> ```shell
> git checkout 84c3ad1 -- \
>   src/models/a2/model/dynamic/build.rs \
>   src/models/a2/model/dynamic/mod.rs \
>   src/models/a2/model/dynamic/process.rs \
>   src/models/a2/model/dynamic_test.rs
> ```
>
> (Estes arquivos foram alterados erroneamente para casar com o oráculo defeituoso — PM-C.
> Reverter restaura a contagem de pesos correta vs C++ — PM-B.)

* **Tarefa S1.2:** Confirmar que `git diff --stat` para os 4 arquivos acima está **vazio** [DONE]

> (working tree = `84c3ad1`). Os únicos diffs pendentes devem ser os TODO files (editados
> nesta sessão) e `.agents/skills/revisor-auditor/SKILL.md` (não-relacionado).

* **Tarefa S1.3:** Rodar `cargo test --lib` — deve passar (0 failed; o teste [DONE]

> `test_wavenet_a2_dyn_bottleneck_neq_channels` volta a `assert head1x1_w.len() == 4*4`,
> consistente com a contagem `head_accum_size × h1_in` do `build.rs` committed).

* **Tarefa S1.4 (registro):** Documentar no relatório de sprint: "baseline `84c3ad1` [DONE]

> restaurado; produção lê `head_accum_size × h1_in` (= `out_channels × bottleneck/groups`),
> alinhado ao C++ `detail.h:76`."

* **Critério de aceite:** `cargo test --lib` verde; `git diff` dos 4 arquivos vazio.

---

## Sprint S2 — Estabelecer a verdade empírica vs C++ golden (Épico N2)

 **Risco:** Baixo (medição, sem mudança de produção). **Decisório.**
 **Especialista:** QA / revisor-auditor.

* [ ] **Tarefa S2.1:** Verificar a integridade do golden: `tests/fixtures/golden_wavenet_a2_max.bin` (16388 bytes = 4 [u32] + 2048×4 [input] + 2048×4 [output]). Confirmar formato (`golden_vectors.rs:9-14`: `[u32 N][f32×N input][f32×N expected]`). Se ausente/inválido, regenerar via `tests/fixtures/golden_gen_build.sh` com o C++ vendored (v0.5.4) — registrar a versão do C++ usada.

* [ ] **Tarefa S2.2:** Des-ignorar **temporariamente** `test_golden_vectors_wavenet_a2_max` (`golden_vectors.rs:1952`): remover o `#[ignore]` apenas para medição.

* [ ] **Tarefa S2.3:** Rodar em release (o caminho do golden é otimizado):

  ```shell
  cargo test --release --test golden_vectors test_golden_vectors_wavenet_a2_max -- --nocapture
  ```

Registrar **ESR, SNR (dB), MSE, MRSTFT** (via `report_dsp_fidelity`).

* [ ] **Tarefa S2.4 (decisório):**
  * **(a) ESR < `max_esr` calibrado (topology_thresholds "wavenet_a2_max"):** produção está
   **correta vs C++**. PM-D rejeitada. → Pular para S5 (travar gate) e depois S6.
* **(b) ESR ≥ threshold:** produção diverge do C++. → Prosseguir S3 (spec C++) e S4 (fix).
   **Re-ignorar** o teste (preservar o marcador `"S14.2-followup: ..."`) até S5.

* [ ] **Tarefa S2.5 (registro):** Anotar o veredito (a/b) com os números no relatório. **Não mudar produção neste sprint.**
      * **Critério de aceite:** Números empíricos registrados; decisão (a) ou (b) tomada.

---

## Sprint S3 — Especificação exata do C++ (Conv1x1, head1x1, cascade, condition_dsp) (Épico N3)

**Risco:** Nulo (read-only). **Pré-requisito:** S2-caso-(b).
**Especialista:** pesquisador-inovador + documentador.

* [ ] **Tarefa S3.1 — Conv1x1:** Estudar `NAM/wavenet/` `Conv1x1::set_weights_` (ordem de leitura da stream) e `Conv1x1::process_` (ordem de acesso). Responder: pesos armazenados em `[out][in]` ou `[in][out]`? Há transposição? Definir o **layout canônico**.

* [ ] **Tarefa S3.2 — head1x1 apply:** Estudar o loop de aplicação do head1x1 (grupos, `ch_per_group = out_channels/groups`, ordem de `grp`, `oc`, `ic` e acumulação). Confirmar contra o Rust `process.rs` (que hoje itera `oc` em `grp*ch_per_group..`).

* [ ] **Tarefa S3.3 — cascade:** Estudar `_layer_arrays` loop, `Process` (com/sem head input), acumulação de head em `final_head_outputs` (`model.cpp:~775+`). Comparar com `WaveNetA2Cascade` (`cascade.rs`) e `cascade_head_finalize`.

* [ ] **Tarefa S3.4 — condition_dsp:** Estudar `_process_condition` (`model.cpp:700-727`), `NumOutputChannels` do sub-modelo, `_condition_output` (N canais/frame). Confirmar que o Rust `condition_dsp_output` (N×max_buf) está alinhado.

* [ ] **Tarefa S3.5 — head finalize:** Estudar head_size==1 (conv k) vs head_size>1 (rechannel). Confirmar contagem `head_accum_size × head_size` (Rust `build.rs:308`) vs C++.

* [ ] **Tarefa S3.6 (entrega):** Especificação escrita (em `docs/cpp_parity_map.md` §6 ou doc dedicado) com `file:line` do C++ para cada ponto. Esta é a **fonte de verdade para S4**.
      * **Critério de aceite:** Spec cobre os 5 pontos com referências C++ verificáveis.

---

## Sprint S4 — Fix da produção ao C++ (cirúrgico, golden como feedback) (Épico N4)

**Risco:** Médio (hot-path DSP). **Pré-requisito:** S3. **Múltiplas iterações.**
**Especialista:** implementador (Rust/DSP).

* **Tarefa S4.1 (PM-D — prioridade):** Resolver o layout do `head1x1`:
  * Se C++ **não transpõe** e acessa `[out][in]`: **remover** `transpose_dense_f32` do head1x1 no `build.rs` e manter o acesso `[oc*h1_in+ic]`.
  * Se C++ **transpõe**: ajustar o acesso do `process.rs` para `[ic*out_channels+oc]`. Após cada tentativa, rodar `test_golden_vectors_wavenet_a2_max` e registrar o ESR. Manter a variante com menor ESR.

* [ ] **Tarefa S4.2:** Para cada outra divergência Rust↔spec-C++ (S3), aplicar a mudança mínima. Uma divergência por sub-tarefa. Após **cada** mudança: `cargo test --release --test golden_vectors test_golden_vectors_wavenet_a2_max -- --nocapture` + `cargo test --lib`.

* [ ] **Tarefa S4.3:** Iterar até `ESR < max_esr` (topology_thresholds "wavenet_a2_max"). Se estagnado, usar o oráculo **apenas como diagnóstico** (decompor onde diverge) — mas o **veredito** permanece o golden.

* [ ] **Tarefa S4.4:** Garantir nenhum regresso nos goldens ativos (A1/Lite/Full/LSTM/ConvNet) rodando `cargo test --test golden_vectors` completo.
      * **Critério de aceite:** `test_golden_vectors_wavenet_a2_max` passa (ESR < threshold) em release; nenhum golden ativo regredido; `cargo test --lib` verde.

---

## Sprint S5 — Travar a barreira defensiva (golden vs C++ como gate) (Épicos N5, PM-E, PM-F)

**Risco:** Baixo (test-infra). **Pré-requisito:** S4 (ou S2-caso-(a)).
**Especialista:** revisor-auditor + documentador.

* [ ] **Tarefa S5.1:** Des-ignorar **definitivamente** `test_golden_vectors_wavenet_a2_max` (remover o `#[ignore = "S14.2-followup..."]`). Atualizar o docstring para refletir que é **gate de paridade vs C++**.

* [ ] **Tarefa S5.2:** Adicionar `test_a2_max_vs_cpp_golden` (separado do oráculo) com threshold calibrado, roteando para `WavenetA2Dyn` e comparando via `report_dsp_fidelity`. Gate obrigatório.

* [ ] **Tarefa S5.3:** Verificar que `utils/tests-quick.sh` e `utils/tests-long.sh` executam este golden como gate (não apenas manual). Documentar a cobertura.

* [ ] **Tarefa S5.4 (PM-E):** Atualizar o cabeçalho de `src/testing/reference_oracle.rs` para declarar explicitamente: o oráculo é **decomposição de erro**; divergências oráculo↔produção são arbitral pelo **C++ golden** antes de qualquer mudança na produção. *Não* é âncora de correção.

* [ ] **Tarefa S5.5 (docs):** Atualizar `docs/cpp_parity_map.md` §6/§13/§13.1 com o estado empírico **real** (ESR/SNR medidos vs C++), removendo a narrativa da Rodada 4.
      * **Critério de aceite:** golden ativo e verde; doc reflete os números reais; oráculo rotulado como decomposição.

---

## Sprint S6 — Reconciliar o oráculo f64 e a âncora NumPy ao C++ (Épico N6, PM-A) [CONSEQUÊNCIA]

**Risco:** Médio (test-only, amplo alcance). **Pré-requisito:** S4/S5. **Baixa prioridade** (ferramenta de debug).
**Especialista:** pesquisador-inovador + implementador.

* [ ] **Tarefa S6.1:** Corrigir `oracle_forward` do `condition_dsp` (`reference_oracle.rs:932-936`) para respeitar o `head_size` do último array do sub-modelo, produzindo `condition_size` valores/frame (como o C++ `NumOutputChannels`), em vez do escalar por frame.

* [ ] **Tarefa S6.2:** Corrigir a leitura de `head1x1` no oráculo (`reference_oracle.rs:1124-1130`) para `out_channels × (bottleneck/groups)` e bias `out_channels` (em vez de `channels`).

* [ ] **Tarefa S6.3:** Aplicar as mesmas correções em `tests/fixtures/scripts/validate_oracle_f64.py` (linhas 785-798, 975+) — a âncora NumPy.

* [ ] **Tarefa S6.4:** Re-rodar `validate_oracle_f64.py` → confirmar ESR ≤ 1e-12 entre oráculo e âncora (agora ambas corretas vs C++). Re-rodar `test_oracle_vs_python_anchor_a2_generic`.

* [ ] **Tarefa S6.5:** Des-ignorar os 3 testes `test_oracle_*_a2_generic` (decomposição), agora consistentes com o C++. Eles medem a **diferença f32↔f64** (erro de precisão), **não** a paridade vs C++ (que é o golden).
      * **Critério de aceite:** oráculo e NumPy ≤ 1e-12; 3 testes reativados e verdes; oráculo produz `condition_size`/frame.

---

## Sequência e dependências

```text
S1 (revert, zero-risco) ──► S2 (medir vs C++ golden)
                                ├─ caso (a): ESR OK ──► S5 (gate) ──► S6 (oráculo)
                                └─ caso (b): ESR diverge ──► S3 (spec C++) ──► S4 (fix vs C++) ──► S5 ──► S6
```

* **S1** é bloqueante e zero-risco: desfaz o dano da Rodada 4.
* **S2** é decisório: evita trabalho desnecessário se a produção já casa com o C++.
* **S3** é a fundação: sem spec C++ exata, qualquer "fix" é chute.
* **S4** é o trabalho de paridade real, com o golden como feedback loop contínuo.
* **S5** trava o que foi validado para não regredir.
* **S6** é consequência de baixa prioridade: conserta a ferramenta de debug, não o motor.

---

**Lembrete final:** O sucesso desta estratégia mede-se por **um número** — o ESR de `test_golden_vectors_wavenet_a2_max` contra `golden_wavenet_a2_max.bin`. Tudo o mais (oráculo, NumPy, decomposição) é consequência. Se em qualquer ponto o oráculo e o C++ discordarem, **o C++ vence**.
