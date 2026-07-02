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

* [x] **Tarefa S2.1:** Verificar a integridade do golden: `tests/fixtures/golden_wavenet_a2_max.bin` (16388 bytes = 4 [u32] + 2048×4 [input] + 2048×4 [output]). Confirmar formato (`golden_vectors.rs:9-14`: `[u32 N][f32×N input][f32×N expected]`). Se ausente/inválido, regenerar via `tests/fixtures/golden_gen_build.sh` com o C++ vendored (v0.5.4) — registrar a versão do C++ usada. [DONE]
    > **Golden íntegro.** Arquivo presente (16388 bytes), N=2048, input/output válidos.
    > C++ vendored: **v0.5.4** (tag, commit `1f42f88`, "Fix inline GEMM restrict annotations on MSVC").
    > **Nota:** `golden_gen_build.sh:9` ainda declara v0.5.3 como canônico; o vendored local está em v0.5.4.

* [x] **Tarefa S2.2:** Des-ignorar **temporariamente** `test_golden_vectors_wavenet_a2_max` (`golden_vectors.rs:1952`): remover o `#[ignore]` apenas para medição. [DONE]

* [x] **Tarefa S2.3:** Rodar em release (o caminho do golden é otimizado):

  ```shell
  cargo test --release --test golden_vectors test_golden_vectors_wavenet_a2_max -- --nocapture
  ```

Registrar **ESR, SNR (dB), MSE, MRSTFT** (via `report_dsp_fidelity`).
  > **Resultados:** MSE=2.46e3, SNR=−15.6 dB, ESR=3.61e1, MR-STFT=3.41, MAE=3.10e2.
  > Todos os thresholds violados por 3+ ordens de magnitude. **Caso (b).** [DONE]

* [x] **Tarefa S2.4 (decisório):** [DONE] — **Caso (b): ESR=3.61e1 ≫ 5.0e-2.** Produção diverge do C++. → S3 (spec) → S4 (fix). Teste re-ignorado.

* [x] **Tarefa S2.5 (registro):** Anotar o veredito (a/b) com os números no relatório. **Não mudar produção neste sprint.** [DONE]
      * **Critério de aceite:** Números empíricos registrados; decisão (a) ou (b) tomada.
      > **Veredito consolidado:** **Caso (b)** — produção diverge massivamente do C++ golden.
      > MSE=2.46e3, SNR=−15.6 dB, ESR=3.61e1, MR-STFT=3.41. Golden gerado com C++ v0.5.4 (vendored, commit `1f42f88`).
      > **Impacto:** caminho completo S3 (spec C++) → S4 (fix vs C++) → S5 (gate) → S6 (oráculo).
      > Nenhuma mudança de produção neste sprint; decisão 100% empírica.

---

## Sprint S3 — Especificação exata do C++ (Conv1x1, head1x1, cascade, condition_dsp) (Épico N3)

**Risco:** Nulo (read-only). **Pré-requisito:** S2-caso-(b).
**Especialista:** pesquisador-inovador + documentador.

* [x] **Tarefa S3.1 — Conv1x1:** ~Ler TODO-findings.md:~ **Respondido.** Layout canônico determinado:
  * **Stream:** pesos chegam em row-major `[out_ch][in_ch]` por grupo (grupos outer, out middle, in inner). Evidência: `dsp.cpp:384-393` (`for g.. for i(out).. for j(in).. _weight(g*opg+i, g*ipg+j)=*weights++`), confirmado por `test_wavenet/test_layer.cpp:366` (`// Weight layout for Conv1x1: for each out_channel, for each in_channel`).
  * **Matriz:** `_weight.resize(out_channels, in_channels)` em Eigen column-major (`dsp.cpp:345`, `conv1d.cpp:97-98`). Row=output channel, col=input channel.
  * **Multiplicação:** `output = weight * input`, padrão `(out_ch×in_ch) * (in_ch×frames) = (out_ch×frames)`, **sem transposição** (`dsp.cpp:427` `result = _weight * input`, `conv1d.cpp:664` `_weight[k] * input_block`).
  * **Consequência para Rust:** `transpose_dense_f32` no head1x1 do `build.rs:236` **está errada** — lê stream row-major `[oc][ic]`, transpõe para `[ic][oc]`, mas `process.rs:582` acessa como `[oc*h1_in+ic]` (row-major). Solução: remover `transpose_dense_f32` do head1x1 (S4.1).

* [x] **Tarefa S3.2 — head1x1 apply:** ~Estudar loop de aplicação do head1x1 no C++ vs Rust.~ **Respondido. Rust estruturalmente correto.**
  * **C++ — Layer::Process (model.cpp:273-282):** `head1x1->process_(z.leftCols(), nf)` — input são todas as `bottleneck` linhas de `_z` (non-gated) ou `z.topRows(bottleneck)` (gated/blended). A Conviv1x1::process_ faz `output=weight*input` com weight block-diagonal `(head1x1_params.out_channels × bottleneck)` para grupos — o agrupamento é implícito na estrutura bloco-diagonal.
  * **C++ — LayerArray::ProcessInner (model.cpp:474-493):** acumulação element-wise: `_head_inputs += cada_layer.GetOutputHead()` sobre todos os frames. Sem lógica de grupo explícita.
  * **C++ — Construção head1x1 (detail.h:75-76):** `Conv1x1(bottleneck, head1x1_params.out_channels, true, groups)` → `_weight(out_channels, in_channels)` = `(head1x1_params.out_channels, bottleneck)`. `_head_output_size` = `head1x1_params.out_channels` (model.cpp:384).
  * **Rust — process.rs:566-594:** `h1_in = h1_in_size = bottleneck/groups`, `h1_groups = bottleneck/h1_in = groups`, `ch_per_group = head_accum_size/h1_groups = out_per_group`. Loop: `grp→oc→ic` com `z_scratch[grp*h1_in+ic]` (subconjunto de entrada do grupo) e `head1x1_w[oc*h1_in+ic]`. Acumulação: primeira layer copia, subsequentes somam a `head_accum`. **Estruturalmente idêntico ao C++** (que faz o mesmo via matrix block-diagonal implícita). Único bug: layout dos pesos `head1x1_w` (S3.1 — `transpose_dense_f32` espúria).

* [x] **Tarefa S3.3 — cascade:** ~Estudar loop `_layer_arrays` C++ vs Rust `WaveNetA2Cascade`.~ **Respondido.**
  * **C++ — WaveNet::process (model.cpp:744-832):** (1) `_set_condition_array` → copia input áudio para `_condition_input`. (2) `_process_condition` → roda condition_dsp (se existir) → `_condition_output`. (3) Loop sobre `_layer_arrays[i]`: Array 0 chama `Process(condition_input, condition_output, nf)` sem head input; arrays seguintes chamam `Process(prev_layer_outputs, condition_output, prev_head_outputs, nf)`. (4) `final_head_outputs = _layer_arrays.back().GetHeadOutputs()` — output pós-head_rechannel. (5) Se `_post_stack_head != nullptr`: head_scale × final → post_stack_head → output; senão: `output = head_scale × final_head_outputs`.
  * **C++ — LayerArray::ProcessInner (model.cpp:450-511):** `rechannel(layer_inputs)` → loop de layers acumulando `_head_inputs += layer.GetOutputHead()` → `_head_rechannel.Process(_head_inputs)` (Conv1D causal `_head_output_size → head_size`). `GetHeadOutputs()` retorna saída pós-rechannel.
  * **Rust — WaveNetA2Dyn (single-array, process.rs:63-119):** `rechannel_prescale` → `layer_forward_dispatch` per layer (acumula head_accum) → `head_finalize` (A2HeadConv aplica head_scale internamente). **Estruturalmente equivalente ao C++ single-array.** Para A2 Max (single array, sem post_stack_head): ambos fazem rechannel¹ → layer loop → head accumulation → causal head conv → output com head_scale.
  * **Rust — WaveNetA2Cascade (multi-array, cascade.rs:132-170):** Array 0: `cascade_write_mono_input` → `cascade_set_condition` → `cascade_layer_loop(is_first=true)`. Arrays 1..N-1: `cascade_seed_head(prev)` (copia head_accum RAW do array anterior) → `cascade_write_residual_input` → `cascade_layer_loop(is_first=false)`. Último array: `cascade_head_finalize` (head_conv ou head_rechannel). **Diferença vs C++:** C++ propaga entre arrays o head **pós-head_rechannel** (`GetHeadOutputs()`, model.cpp:769), Rust propaga o head **RAW pré-rechannel** (`head_accum`). Isso é correto para single-array (golden atual) mas é um bug latente para multi-array com `head_kernel_size > 1`. Adicionalmente, C++ tem `post_stack_head` (repeated activation+Conv1D, model.cpp:776-805) que Rust não implementa para A2 dinâmico (irrelevante para `wavenet_a2_max.nam` que não o usa).
  * **Nota:** `wavenet_a2_max.nam` é single-array → usa `WaveNetA2Dyn` (não cascade). A diferença cascade multi-array não afeta o golden atual.

* [x] **Tarefa S3.4 — condition_dsp:** ~Estudar `_process_condition`, `NumOutputChannels`, `_condition_output` vs Rust.~ **Respondido. Alinhado.**
  * **C++ (model.cpp:699-729):** Sem condition_dsp: `_condition_output = _condition_input` (pass-through, `condition_dim = _get_condition_dim() = NumInputChannels()`). Com condition_dsp: copia `_condition_input → dsp input buffers → condition_dsp->process() → _condition_output` com `condition_output_channels = _condition_dsp->NumOutputChannels()`. `_condition_output` tem dimensão `condition_output_channels × maxBufferSize` (Eigen column-major = interleaved per frame).
  * **Rust (process.rs:89-98):** Sem condition_dsp: `use_cond_dsp = false`, condition é `input[pos..pos+nf]` (raw mono). Com condition_dsp: `cond_dsp.process(input[pos..], condition_dsp_output[..nf*cond_size])`. `condition_dsp_output` tem dimensão `condition_size × max_buf` (flat, interleaved per frame), mesma do C++ Eigen column-major.
  * **Validação de consistência (dispatcher/mod.rs:271-276):** `cond_out == condition_size` — garante que `condition_dsp.NumOutputChannels()` == `condition_size` da layer array, igual ao C++ `model.cpp:594`.
  * **Para A2 Max (golden atual):** **tem condition_dsp** — modelo WaveNet com 2 layers (ch=3, bn=6, SiLU gated, head_size=4, FiLM). Issso invalida a premissa "sem condition_dsp" que S3.4 usou para declarar o alinhamento. A divergência do condition_dsp é a provável causa dominante do ESR alto (3.61e1 no baseline). O layout e dimensões estão alinhados com o C++ para a interface (cond_size=8 na saída), mas o processamento interno do sub-modelo condition_dsp (grupos, FiLM, gating) pode divergir.

* [x] **Tarefa S3.5 — head finalize:** ~Estudar head_size==1 vs head_size>1, verificar `head_accum_size × head_size` vs C++.~ **Respondido.**
  * **C++ — head_size==1 (a2_fast.cpp / generic model.cpp):** O `_head_rechannel` é Conv1D(in=_head_output_size, out=head_size=1, kernel=head_kernel_size, bias, dilation=1) aplicado sobre ring buffer da head acumulada. Em `WaveNet::set_weights_` (model.cpp:632), `head_scale` é lida como último peso `*(it++)`. Fluxo: `_head_rechannel.Process() → GetHeadOutputs() → output = head_scale × final_head`. O a2_fast.cpp usa kernel=16 fixo. O generic model.cpp usa `head_kernel_size` do JSON (validado para 16 no A2 estático, valor livre no dinâmico).
  * **Rust — head_size==1 (A2HeadConv, head.rs:38-162):** Conv1D f32 com kernel=16 fixo (`A2_HEAD_KERNEL_SIZE`), bias e head_scale, operando sobre ring buffer (`head_accum`) com acesso `col & ring_mask`. `head_finalize` (process.rs:293-307) e `cascade_head_finalize` (process.rs:348-358) delegam ao A2HeadConv. **Estruturalmente equivalente ao C++ a2_fast.**
  * **C++ — head_size>1:** `_head_rechannel` = Conv1D(in=_head_output_size, out=head_size, kernel=head_kernel_size, bias=head_bias). Weight count: `head_kernel_size × _head_output_size × head_size + head_bias × head_size`. Fluxo: conv → head_scale → output.
  * **Rust — head_size>1 (cascade_head_finalize, process.rs:360-378):** Projeção densa pontual (não Conv1D): `output[oc] = sum_ic(head_accum[ic] × hw[ic * head_size + oc])`, sem kernel, sem bias, sem head_scale. Weight count: `head_accum_size × head_size` (build.rs:300). **Só equivale ao C++ quando head_kernel_size==1 e head_bias==false.** Para head_kernel_size > 1, Rust tem menos pesos e ausência de contexto temporal (bug latente para multi-array com head_size>1).
  * **Para A2 Max (golden atual):** head_size==1, kernel_size=16. Ambas usam Conv1D com kernel=16, bias e head_scale. Weight count: `16 × head_accum_size + 2` em ambas. ✓

* [x] **Tarefa S3.6 (entrega):** ~Especificação escrita (em `docs/cpp_parity_map.md` §6 ou doc dedicado) com `file:line` do C++ para cada ponto. Esta é a **fonte de verdade para S4**.~ **Entregue.** Spec adicionada em `docs/cpp_parity_map.md` §6.1, cobrindo os 5 pontos (Conv1x1, head1x1 apply, cascade, condition_dsp, head finalize) com referências C++ `file:line` verificáveis contra `tests/fixtures/NeuralAmpModelerCore/`.

---

## Sprint S4 — Fix da produção ao C++ (cirúrgico, golden como feedback) (Épico N4)

**Risco:** Médio (hot-path DSP). **Pré-requisito:** S3. **Múltiplas iterações.**
**Especialista:** implementador (Rust/DSP).

* [x] **Tarefa S4.1 (PM-D — prioridade):** Resolver o layout do `head1x1`:
  * Se C++ **não transpõe** e acessa `[out][in]`: **remover** `transpose_dense_f32` do head1x1 no `build.rs` e manter o acesso `[oc*h1_in+ic]`.
  * Se C++ **transpõe**: ajustar o acesso do `process.rs` para `[ic*out_channels+oc]`. Após cada tentativa, rodar `test_golden_vectors_wavenet_a2_max` e registrar o ESR. Manter a variante com menor ESR.
    * **Variante A (remove transpose, row-major):** ESR=2.43e2 (MSE=1.66e4, SNR=−23.9dB) — pior que baseline.
    * **Variante B (keep transpose, column-major `[ic*ch+oc]`):** ESR=2.43e2 — idêntico (para head_accum_size=4, h1_in=2, acesso isomórfico entre row/col-major).
    * **Original (keep transpose, row-major `[oc*h1_in+ic]` — access pattern bug):** ESR=3.61e1 — menor de todos. Mantido como baseline.
    * **Conclusão:** Nenhuma correção do layout head1x1 melhora ESR isoladamente. O condition_dsp do `wavenet_a2_max.nam` (WaveNet c/ 2 layers, ch=3, bn=6, SiLU gated, head_size=4, FiLM — ver S3.4 corrigido) é a divergência dominante. **Bloqueado:** S4.2 deve investigar condition_dsp primeiro.

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
