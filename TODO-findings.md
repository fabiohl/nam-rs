<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Findings da Auditoria CLAP

## Escopo e método

Esta auditoria cobre o plugin CLAP de ponta a ponta, com foco no comportamento
observável em Bitwig Studio, REAPER e hosts Linux/X11/XWayland. Foram revisados:

- ciclo de vida CLAP, FFI, ativação, desativação, reativação e múltiplas instâncias;
- processamento de áudio, automação sample-accurate, bypass, resampling,
  oversampling, CabSim, Adaptive Compute, GC e segurança de tempo real;
- estado, state-context, preset discovery/load, parâmetros, latência e cauda;
- GUI, janela embedded/floating, HiDPI, diálogos, teclado, feedback de erro,
  telemetria e integração `egui`/`baseview`;
- paridade matemática com o caminho de inferência validado contra NAMcore;
- testes, scripts de QA e documentação declarada como fonte de verdade.

As evidências foram obtidas por inspeção do código, comparação com os contratos
oficiais CLAP 1.2.x e execução das seguintes validações:

- 92 testes unitários CLAP em release, serializados: aprovados;
- 1.229 testes da biblioteca em release com `debug-assertions`: aprovados;
- 10 testes de integração CLAP não ignorados: aprovados;
- build release do `cdylib`: aprovado;
- `clap-validator`: 19 testes aplicáveis aprovados, 2 não aplicáveis, zero avisos.

O fato de essas baterias passarem é parte dos findings: os oráculos atuais não
exercitam vários caminhos que quebram na prática.

## Escala de severidade

| Severidade | Critério                                                                                                                                     |
|:---------- |:-------------------------------------------------------------------------------------------------------------------------------------------- |
| Crítica    | Perda/corrupção de áudio, restauração com timbre incorreto, uso inseguro de FFI ou funcionalidade principal anunciada que não atua no sinal. |
| Alta       | Não conformidade CLAP capaz de quebrar hosts, automação/latência incorreta, violação RT ou UX severamente enganosa.                          |
| Média      | Falha defensiva, diagnóstico incorreto, degradação relevante ou risco condicionado a uma sequência menos comum.                              |
| Baixa      | Redundância, documentação divergente ou melhoria estrutural sem falha imediata no áudio.                                                     |

## Resumo executivo

| ID        | Severidade | Área              | Resumo                                                                                             |
|:--------- |:----------:|:----------------- |:-------------------------------------------------------------------------------------------------- |
| CLAP-F001 | Crítica    | CabSim            | O IR é carregado e reportado, mas nunca é aplicado ao áudio CLAP.                                  |
| CLAP-F002 | Crítica    | Lifecycle         | `deactivate()`/`activate()` perde modelo e parte relevante do estado RT.                           |
| CLAP-F003 | Crítica    | State/SR          | Restore antes de `activate()` usa resampler de 48 kHz em projetos não-48 kHz.                      |
| CLAP-F004 | Crítica    | Controle/latência | Rebuilds dependem de `on_main_thread()` não agendado e a latência muda ilegalmente enquanto ativo. |
| CLAP-F005 | Alta       | Tail/FFI          | `host.tail.changed()` é chamado na thread errada por meio de handle forjado.                       |
| CLAP-F006 | Crítica    | Multi-instância   | Precisão de ativação é global e uma instância altera o som das demais.                             |
| CLAP-F007 | Crítica    | Buffers           | Blocos acima de 8.192 amostras têm o excedente silenciosamente descartado.                         |
| CLAP-F008 | Alta       | Bypass/PDC        | Bypass ativo impede eventos de saída do bypass e não preserva a latência declarada.                |
| CLAP-F009 | Alta       | Offline           | O modo offline não cumpre a matriz HQ e pode manter/restaurar precisão errada.                     |
| CLAP-F010 | Alta       | RT-safety         | `log::info!` é executado no audio thread em mudanças de qualidade/modelo.                          |
| CLAP-F011 | Alta       | Automação         | Smoothing de ganho depende do tamanho do bloco e ignora o IIR configurado.                         |
| CLAP-F012 | Alta       | Eventos           | Mais de 1.024 eventos por bloco são truncados sem sinalização.                                     |
| CLAP-F013 | Alta       | GUI/params        | Eventos GUI podem ser perdidos e usam alvo PCKN incorreto.                                         |
| CLAP-F014 | Crítica    | Estado            | Falha ao restaurar assets mantém silenciosamente modelo/IR antigos.                                |
| CLAP-F015 | Alta       | State-context     | Presets não restauram todos os parâmetros nem são realmente portáveis/equivalentes.                |
| CLAP-F016 | Alta       | SPSC              | Falhas de envio de recursos são ignoradas e dessincronizam UI, estado e DSP.                       |
| CLAP-F017 | Alta       | Preset load       | A extensão retorna sucesso antes de carregar e não notifica o host.                                |
| CLAP-F018 | Média      | Preset discovery  | Indexação lê arquivos inteiros e o extrator manual falha com Unicode.                              |
| CLAP-F019 | Alta       | GUI lifecycle     | `show`/`hide` não fazem nada e floating/transient/close não cumprem o protocolo.                   |
| CLAP-F020 | Alta       | File picker       | Cancelar deixa a GUI travada em Loading; teardown pode congelar a DAW por 120 s.                   |
| CLAP-F021 | Alta       | GUI/diagnóstico   | Footer fica fora da janela e o comando de clipboard do egui é descartado.                          |
| CLAP-F022 | Alta       | GUI/performance   | Repaint incondicional torna o idle-skip inalcançável.                                              |
| CLAP-F023 | Alta       | HiDPI             | Tamanho físico CLAP e escala lógica baseview podem produzir janela 2x maior que o parent.          |
| CLAP-F024 | Média      | Diagnóstico       | Destruir uma instância desativa permanentemente o panic report de todas as outras.                 |
| CLAP-F025 | Alta       | QA                | A suíte declarada não cobre os contratos críticos e pode testar binário instalado obsoleto.        |
| CLAP-F026 | Média      | Robustez          | Estado sem limite, ativação f64 aceita indevidamente e dead code aumentam a superfície de falha.   |

---

## CLAP-F001 — CabSim não participa do áudio do plugin

**Severidade:** Crítica

**Evidência:** `src/clap/plugin/main_thread/load.rs:214-265`,
`src/clap/processor/events.rs:178-193`,
`src/clap/processor/dsp/orchestrator.rs:158-205`,
`src/dsp/pipeline/stages/inference.rs:231-643` e
`src/dsp/pipeline/capture.rs:131-162`.

**Cadeia causal:** A GUI e o state loader constroem um `ConvEngine` e o enviam via
`ClapParamPayload::LoadCabIr`. O audio thread recebe o engine e o coloca em
`self.conv_engine`. O orquestrador também injeta esse engine em
`DspPipelineContext::conv`. Porém, o caminho CLAP chama diretamente
`run_inference()` e depois `apply_output_stage()`; `run_inference()` nunca lê
`ctx.conv`. A única aplicação efetiva de `conv.process()` está em
`capture_dsp_pipeline()`, caminho do standalone com `DspBridge`, não usado pelo
CLAP.

**Efeito na DAW:** O usuário seleciona um IR, vê o nome do arquivo, o botão Clear,
latência e tail mudarem, mas o áudio permanece sem CabSim. Como a latência do
engine inexistente no sinal é somada em `current_latency`, a DAW ainda aplica PDC
para um atraso que não existe, deslocando a faixa no tempo.

**Risco adicional:** Não basta inserir uma chamada simples no orquestrador. O
`ConvEngine::process()` exige exatamente `partition_size` amostras, enquanto o
CLAP divide blocos em sub-blocos por automação. O gate também retorna silêncio
antes de drenar uma eventual cauda. Uma correção ingênua introduziria asserts,
caudas truncadas ou áudio intermitente.

**Solução proposta:** Criar um adaptador RT-safe de fluxo variável para CabSim,
com FIFO/accumulator pré-alocado em `activate()`, saída causal e latência fixa.
Aplicar a convolução após inferência e antes do ganho final, preservando a ordem
do standalone. Em modo mono, processar um único estado de convolução e copiar a
saída; não reutilizar o mesmo estado sequencialmente para L e R. Continuar
drenando o engine durante silêncio até a cauda terminar.

**Critérios de aceite:** Um teste impulso end-to-end pelo host CLAP deve coincidir
com a convolução de referência; Clear IR deve voltar ao sinal sem IR; blocos
irregulares e eventos no meio do bloco devem produzir saída invariável; a
latência medida por correlação deve coincidir exatamente com a extensão; a cauda
deve ser renderizada integralmente após cessar a entrada.

---

## CLAP-F002 — Reativação perde modelo e estado de processamento

**Severidade:** Crítica

**Evidência:** `src/clap/processor/mod.rs:151-304` e
`src/clap/processor/mod.rs:307-333`.

**Cadeia causal:** Toda ativação cria `model_l: None`, oversampling Off,
`RtPluginParams::default()` e multiplicadores de calibração em 1.0. Na
desativação, somente `param_rx`, `gc_tx` e `slimmable_rx` voltam ao shared. O
modelo ativo, resampler, engines de oversampling e calibração permanecem no
processor consumido e são destruídos na main thread. Não existe payload para
recolocá-los na próxima ativação nem reload automático a partir de
`main_thread.params.model_path`.

**Efeito na DAW:** Uma DAW pode desativar e reativar plugins ao mudar sample rate,
buffer, dispositivo, suspensão da faixa ou configuração do grafo. Após esse
ciclo, o plugin passa a processar sem o modelo anterior. Parâmetros recebidos por
eventos de `process()` também podem voltar aos defaults porque a ativação só usa
os atômicos para inicializar os dois smoothers; bypass, gate, slim, activation e
oversampling não são materializados integralmente em `self.params`.

**Por que os testes passam:** `test_smoother_warm_reset_on_reactivate` verifica
somente o valor inicial do smoother e não carrega um modelo. O teste de lifecycle
faz apenas um ciclo activate/process/deactivate, sem segunda ativação.

**Solução proposta:** Introduzir um `DeactivatedDspState` exclusivo da main
thread/shared, para receber no `deactivate()` os recursos que devem sobreviver e
entregá-los na próxima ativação. Inicializar `RtPluginParams` a partir de um
snapshot coerente dos atômicos, construir o oversampling selecionado durante
`activate()` e definir `last_seen_generation` somente depois do snapshot. Em caso
de falha parcial na ativação, devolver todos os endpoints e recursos já tomados.

**Critérios de aceite:** Carregar modelo, IR e parâmetros não default; processar;
desativar; reativar com mesmo e com novo sample rate/buffer; comparar saída,
modelo, calibração, parâmetros e latência. O teste deve cobrir WaveNet, LSTM e
modelo sem IR, sem depender de novo state load do host.

---

## CLAP-F003 — Restore pré-ativação constrói resampler para taxa errada

**Severidade:** Crítica

**Evidência:** `src/clap/plugin/main_thread/load.rs:51-66`,
`src/clap/plugin/main_thread/load.rs:137-176` e
`src/clap/processor/mod.rs:151-160,241-252`.

**Cadeia causal:** O fluxo normal de restore pode chamar `state.load()` antes de
`activate()`. Como `ColdShared::sample_rate` ainda é zero, `load_model()` assume
48 kHz e cria um `NamResampler` para `48000 -> model_rate`. Esse resampler é
guardado em `PendingModel`. Durante `activate()` um resampler correto é criado
com a taxa real do host, mas `flush_pending_model()` envia depois o resampler
antigo. No primeiro `process()`, `cold_load_model()` substitui o resampler correto
pelo resampler construído com fallback de 48 kHz.

**Efeito na DAW:** Projetos em 44,1, 88,2, 96 ou 192 kHz restaurados antes da
ativação alimentam o modelo como se a entrada estivesse a 48 kHz. O resultado
pode ter resposta temporal/timbre incorretos e divergência total de paridade. O
mesmo padrão afeta IR carregado antes da ativação: as amostras guardadas como
“raw” já foram resampleadas para o fallback de 48 kHz e são reutilizadas após a
taxa real ser conhecida.

**Solução proposta:** Tornar o pending declarativo. Guardar modelo, taxa nativa,
calibração e asset de IR original, mas construir recursos dependentes do host
somente após `PluginAudioConfiguration` estar disponível. Para IR, guardar a
fonte original e sua taxa, ou recarregar/resamplear fora do RT na ativação; não
tratar amostras já convertidas como raw.

**Critérios de aceite:** State restore antes e depois de activate em 44,1/48/96
kHz deve gerar a mesma saída, latência e duração de IR. Um teste deve inspecionar
explicitamente `resampler.is_bypass()` para impedir bypass indevido em host não-48
kHz.

---

## CLAP-F004 — Rebuilds não são agendados e a latência viola o lifecycle CLAP

**Severidade:** Crítica

**Evidência:** `src/clap/processor/events.rs:72-100`,
`src/clap/plugin/main_thread/housekeeping.rs:124-169,311-325`,
`src/clap/gui/ui/zones/controls.rs:123-182` e contrato oficial
`clap/ext/latency.h`.

**Cadeia causal:** Mudanças de oversampling e slimming no audio thread apenas
marcam flags atômicas. O comentário afirma que housekeeping “polls on its regular
cycle”, mas CLAP não chama `on_main_thread()` periodicamente: ele é agendado por
`host.request_callback()`. O audio thread deliberadamente não chama esse método e
não há timer extension. Portanto o rebuild pode nunca ocorrer. Mesmo quando um
callback alheio acontece, a latência é alterada enquanto o plugin está ativo e
`HostLatency::changed()` é chamado. A especificação permite mudar a latência
durante `activate()`; se já ativo, o plugin deve solicitar restart e postergar a
mudança.

**Efeito na DAW:** A GUI pode mostrar 2x/4x enquanto o engine continua Off. Um
oversampling restaurado do projeto sinaliza rebuild no primeiro bloco e fica
pendente indefinidamente. Adaptive Compute pode solicitar um modelo slim e nunca
recebê-lo. A DAW também pode manter PDC obsoleto ou observar mudança de latência
fora do ponto permitido pelo protocolo.

**Solução proposta:** Definir uma única estratégia de mudanças estruturais. A
opção mais conforme é guardar a configuração desejada, chamar
`host.request_restart()` fora do RT e construir tudo no próximo `activate()`.
Alternativamente, manter latência externa fixa no pior caso e compensar
internamente modos menores. Um timer/main-thread scheduler pode cuidar de GC e
logs, mas não torna legal mudar latência ativa. Remover a suposição de callback
periódico não solicitado.

**Critérios de aceite:** Um host mock deve contabilizar `request_callback` e
`request_restart`; uma mudança para 4x deve ou entrar em vigor com latência fixa,
ou somente após deactivate/activate. O valor anunciado nunca pode mudar durante
estado ativo sem restart.

---

## CLAP-F005 — Notificação de tail usa thread e handle incorretos

**Severidade:** Alta

**Evidência:** `src/clap/plugin/main_thread/housekeeping.rs:327-349` e contrato
oficial `clap/ext/tail.h`.

**Cadeia causal:** `housekeeping()` roda na main thread, converte unsafe o ponteiro
do `HostMainThreadHandle` em `HostAudioProcessorHandle` e chama
`HostTail::changed()`. O contrato de `clap_host_tail.changed()` é exclusivamente
`[audio-thread]`. A justificativa no comentário local afirma o oposto da
especificação.

**Efeito na DAW:** Hosts com thread-check estrito podem rejeitar, abortar ou tocar
estado do handler de áudio pela main thread. Como o handle Rust foi forjado, um
host implementado com garantias de exclusividade por thread pode sofrer data
race lógica ou UB na fronteira FFI.

**Solução proposta:** Guardar a última tail reportada no processor e chamar
`HostTail::changed()` diretamente no audio thread quando o valor efetivo mudar.
Remover a conversão unsafe. Se a mudança depender de recurso estrutural, alinhar
com o restart do finding CLAP-F004.

**Critérios de aceite:** Host mock com `thread-check` deve comprovar que `changed`
ocorre no audio thread. Testar load, replace e clear do IR real, sem simular apenas
escritas em atômicos.

---

## CLAP-F006 — Activation Precision global quebra isolamento entre instâncias

**Severidade:** Crítica

**Evidência:** `src/math/activations/mod.rs:101-163`,
`src/clap/processor/params.rs:88-97,238-248`,
`src/clap/processor/events.rs:97-124` e
`src/clap/processor/dsp/orchestrator.rs:689-697`.

**Cadeia causal:** Cada instância expõe `Activation Precision` como parâmetro
próprio, mas sua aplicação escreve em `ACTIVATION_MODE`, um `static AtomicUsize`
global ao processo. Os kernels consultam esse global durante inferência. O
módulo já oferece override thread-local, mas o CLAP não o instala ao entrar no
callback. Duas instâncias em threads diferentes podem alternar o modo no meio da
inferência; duas instâncias na mesma thread herdam o modo da última processada.

**Efeito na DAW:** Configurar Fast em uma faixa pode degradar silenciosamente uma
LSTM configurada como Standard em outra faixa. O efeito é ordem-dependente e pode
mudar entre playback realtime e bounce paralelo, comprometendo determinismo e
paridade NAMcore.

**Solução proposta:** Tornar a precisão parte do contexto de execução. Como
correção mínima, instalar um guard thread-local no início de cada `process()` e
`flush()` relevante, usando o valor da instância, sem escrever no global pelo RT.
A solução definitiva é propagar a precisão ao modelo/kernel e eliminar estado
global mutável do caminho multi-instância.

**Critérios de aceite:** Duas instâncias com Fast/Standard devem produzir seus
goldens independentemente em ordem alternada e em duas threads simultâneas. O
resultado deve ser invariável à ordem de scheduling e ThreadSanitizer/loom não
deve encontrar compartilhamento indevido.

---

## CLAP-F007 — Blocos maiores que MAX_RESAMP_BUF perdem áudio

**Severidade:** Crítica

**Evidência:** `src/clap/processor/dsp/orchestrator.rs:65-79,140-205` e
`src/dsp/pipeline/stages/inference.rs:249-250`.

**Cadeia causal:** A ativação aceita `max_frames_count` arbitrário e aloca buffers
com esse tamanho. O orquestrador passa o sub-bloco completo para
`run_inference()`, que reduz silenciosamente `n_samples` para
`MAX_RESAMP_BUF` (8.192). O `block_offset` avança pelo tamanho original, enquanto
`output_offset` avança apenas por `n_out`. O restante da saída fica antigo ou
não inicializado.

**Efeito na DAW:** Hosts que usam buffers grandes em bounce offline, render
freewheel ou condições de carga podem perder toda a saída após a amostra 8.191 de
cada bloco. Não há erro nem status para o usuário.

**Solução proposta:** Fragmentar cada intervalo entre eventos em chunks de no
máximo `MAX_RESAMP_BUF`, preservando estado do resampler, modelo, smoothers e
offset de saída. Como alternativa defensiva temporária, rejeitar `activate()` com
limite documentado, nunca truncar.

**Critérios de aceite:** Testar 8.191, 8.192, 8.193, 16.384 e 65.536 amostras, com
sentinelas no output, in-place/out-of-place, 44,1/48/96 kHz e eventos nos limites
dos chunks. Toda amostra deve ser escrita uma única vez.

---

## CLAP-F008 — Bypass ativo engole automação e quebra alinhamento de latência

**Severidade:** Alta

**Evidência:** `src/clap/processor/dsp/orchestrator.rs:84-86`,
`src/clap/processor/dsp/bypass.rs:9-83` e
`src/clap/extensions/latency.rs:12-19`.

**Cadeia causal:** Os eventos do bloco são coletados, mas se `self.params.bypass`
já está ativo, `process_bypass()` copia o áudio e o loop faz `continue` antes de
aplicar qualquer evento. Um evento host `Bypass=0` no próprio `process()` não
consegue retirar o plugin do bypass; outros eventos do mesmo bloco também são
ignorados. Além disso, o dry é copiado sem delay enquanto a extensão continua
reportando latência de resampler/oversampling/CabSim.

**Efeito na DAW:** Automação de bypass pode ficar presa até que a GUI ou `flush()`
altere o atômico por outro caminho. Em configurações com latência, a faixa
bypassada fica adiantada em relação às demais após PDC, criando problemas de fase
e comb filtering em processamento paralelo.

**Solução proposta:** Integrar bypass ao mesmo loop sample-accurate de sub-blocos,
aplicando eventos antes de decidir o caminho do intervalo. Implementar bypass
latency-compensated com delay pré-alocado, preferencialmente com crossfade curto,
mantendo latência externa estável.

**Critérios de aceite:** Automatizar on/off em offsets 0, meio e fim do bloco,
partindo dos dois estados. Fazer null test com PDC para cada combinação de sample
rate, oversampling e IR.

---

## CLAP-F009 — Render offline não implementa a política HQ declarada

**Severidade:** Alta

**Evidência:** `README.md:128-135`, `.agents/rules/rust.md:46-50`,
`src/clap/extensions/render.rs:25-55` e
`src/clap/processor/events.rs:97-135`.

**Cadeia causal:** A matriz pública define offline como 4x, Adaptive Off e
qualidade máxima. A implementação apenas força Adaptive Off e Standard; o log
diz `oversample=max quality`, mas nenhum engine 4x é solicitado. Ao voltar para
realtime, `old_activation` é capturado depois de `self.params` já estar Standard,
portanto Fast não é restaurado. Um evento de activation aplicado mais tarde no
mesmo bloco também pode voltar a Fast, pois só Adaptive recebe um guard contínuo.
`SlimOverride::ForceLite` tampouco é neutralizado.

**Conflito de arquitetura:** `docs/audio_fidelity_map.md:98-103` alerta que
oversampling altera constantes temporais de LSTM e não é recomendado. Logo,
forçar 4x universalmente também não é uma solução correta frente ao objetivo de
paridade NAMcore.

**Solução proposta:** Formalizar uma política offline por topologia. Para
WaveNet/feed-forward, aplicar o modo HQ validado; para LSTM, preservar a taxa
temporal do modelo salvo se os testes de paridade rejeitarem oversampling. Guardar
explicitamente um snapshot realtime de activation, adaptive, slim e oversampling
e restaurá-lo na saída. Aplicar a política antes do primeiro bloco offline, em
conjunto com o lifecycle de latência do finding CLAP-F004.

**Critérios de aceite:** Matriz realtime/offline/realtime para todas as
topologias, incluindo tentativa de automação durante offline. Dois bounces devem
ser binariamente idênticos e cada topologia deve passar o golden NAMcore
apropriado.

---

## CLAP-F010 — Logging bloqueante e alocador é chamado no audio thread

**Severidade:** Alta

**Evidência:** `src/dsp/adaptive.rs:185-213,475-483`, com chamadas a partir de
`src/clap/processor/params.rs`, `src/clap/processor/events.rs` e
`src/clap/processor/dsp/orchestrator.rs`.

**Cadeia causal:** `AdaptiveCompute::set_mode()`, `set_slim_override()` e
`set_wavenet_full_ch()` executam `info!`. No CLAP esses métodos são chamados por
eventos de parâmetros e model swap no audio thread. `NamLogger` formata Strings,
trava `LogBuffer`, trava a lista de sinks e chama o logger do host. Isso viola os
contratos de zero allocation, zero lock e zero host I/O no RT.

**Efeito na DAW:** Alterar Adaptive/Slim, entrar em offline ou trocar um WaveNet
pode causar jitter, xrun ou inversão de prioridade. O risco cresce com múltiplas
instâncias porque cada log é despachado a todos os sinks registrados.

**Solução proposta:** Remover logs de setters alcançáveis pelo RT. Emitir flags e
payloads triviais em `RtStatusFlags`, consumidos fora do RT. Se o mesmo método
também for usado off-RT, separar a mutação pura de uma função de logging do
control plane.

**Critérios de aceite:** Heap-audit deve cobrir eventos Adaptive, Slim,
Activation, model swap e transições de render. Um sink instrumentado deve provar
zero chamadas durante `process()`.

---

## CLAP-F011 — Smoothing de ganho é dependente do tamanho do bloco

**Severidade:** Alta

**Evidência:** `src/dsp/smoother.rs:9-115` e
`src/clap/processor/dsp/orchestrator.rs:423-589`.

**Cadeia causal:** `ParamSmoother` é configurado como IIR de 20 Hz, mas o CLAP só
usa `tick()` quando o sub-bloco tem menos de 8 amostras. Para sub-blocos maiores,
o código interpola linearmente de `current` até `target` ao longo do tamanho
inteiro do sub-bloco e chama `smoother.set(target)`. Assim a duração do smoothing
é 64 amostras num host e 1.024 em outro, em vez da constante temporal declarada.

**Efeito na DAW:** A mesma automação soa diferente conforme buffer e densidade de
eventos. Blocos pequenos produzem transições muito rápidas e suscetíveis a click;
blocos grandes alteram a curva. A paridade de render entre realtime e offline
fica comprometida.

**Solução proposta:** Preservar uma única lei temporal sample-accurate. Vetorizar
a recorrência IIR por blocos ou manter um ramp state com número de amostras
restantes calculado pela taxa, sem snap no fim arbitrário do callback.

**Critérios de aceite:** Renderizar a mesma sequência com blocos 1, 7, 8, 16, 64,
128, 512, 1.024 e blocos aleatórios; comparar amostra a amostra dentro de limite
numérico rigoroso. Medir tempo até 63,2% e 95% do target.

---

## CLAP-F012 — Flood de eventos é descartado silenciosamente

**Severidade:** Alta

**Evidência:** `src/clap/processor/dsp/orchestrator.rs:21-61`.

**Cadeia causal:** Quatro arrays de stack guardam no máximo 1.024 eventos. O loop
faz `break` ao atingir o limite e não seta flag. Em offline, um host pode entregar
buffers grandes com automação ou modulação densa de vários parâmetros.

**Efeito na DAW:** Pontos posteriores do bloco desaparecem; o valor final pode
ficar incorreto e o bounce divergir do realtime. O usuário não recebe diagnóstico.

**Solução proposta:** Consumir o `InputEvents` ordenado de forma streaming,
mantendo apenas o próximo evento, sem cópia e sem limite artificial. Se a API
exigir cache, prealocar em `activate()` e definir fallback explícito que preserve
ao menos o último valor por parâmetro e sinalize overflow.

**Critérios de aceite:** Processar mais de 100 mil eventos em bloco offline sem
heap no RT e sem perda; verificar valor final, offsets e ordem de eventos
coincidentes.

---

## CLAP-F013 — Protocolo de eventos da GUI pode perder automação

**Severidade:** Alta

**Evidência:** `src/clap/plugin/shared.rs:375-450`,
`src/clap/gui/ui/zones/controls.rs:149-181,227-246` e contrato
`clap/events.h`/`clap/ext/params.h`.

**Cadeia causal:** `write_gui_events()` limpa cada bit com `fetch_and` antes de
saber se `output.try_push()` aceitou o evento. Se a fila do host estiver cheia,
Begin, Value ou End desaparece definitivamente e a sequência pode ficar
incompleta. `ParamValueEvent` usa `Pckn::new(0,0,0,0)`, direcionando a mudança a
uma nota específica, embora os parâmetros não tenham flags per-note; mudanças
globais devem usar wildcards. Os controles segmentados de Oversampling e
Activation marcam apenas Changed, sem Begin/End.

**Efeito na DAW:** A DSP local muda por atômicos, mas o host pode não gravar a
automação, manter um gesto aberto ou interpretar o evento como note-scoped. A UI
parece funcionar enquanto o projeto salvo não reproduz a ação.

**Solução proposta:** Implementar uma pequena máquina de estado retryable por
parâmetro. Só confirmar/limpar cada fase depois de `try_push()` bem-sucedido;
preservar ordenação global; usar `Pckn::match_all()`; emitir Begin/Value/End para
ações discretas.

**Critérios de aceite:** Host mock com capacidade 0, 1 e 2 deve forçar retries e
receber posteriormente uma sequência completa e ordenada. Validar PCKN raw como
`-1/-1/-1/-1`.

---

## CLAP-F014 — Falha de asset no state restore mantém DSP antigo

**Severidade:** Crítica

**Evidência:** `src/clap/extensions/state.rs:102-207` e
`src/clap/extensions/state_context.rs:101-163`.

**Cadeia causal:** O state é comprometido em `self.params` e nos atômicos antes de
resolver modelo e IR. Se o caminho não existe ou o build falha, o código apenas
loga warning e retorna sucesso. Nenhum payload descarrega o modelo/IR anterior.
Em uma instância reutilizada, o audio thread continua com o asset antigo, e a GUI
pode continuar exibindo o nome antigo.

**Efeito na DAW:** Ao carregar projeto, preset, undo ou duplicação com arquivo
ausente/corrompido, a faixa pode soar com outro amplificador ou gabinete sem
alerta claro. Esse é um erro de integridade de projeto: o host considera o state
restaurado, mas o áudio não corresponde ao state.

**Solução proposta:** Implementar restore transacional em duas fases:
`prepare_state` valida e constrói todos os recursos off-RT; `commit_state` publica
parâmetros e payloads somente quando o conjunto está pronto. Definir política
explícita para asset ausente: falhar o load e manter todo o estado anterior, ou
commit de “sem modelo/sem IR” com erro persistente visível; nunca misturar state
novo com DSP antigo.

**Critérios de aceite:** Restaurar B válido sobre A, B ausente sobre A e B
corrompido sobre A. Estado, UI e áudio devem ser todos A ou todos B/empty conforme
a política, sem combinações híbridas.

---

## CLAP-F015 — State-context não é equivalente nem portátil

**Severidade:** Alta

**Evidência:** `src/clap/extensions/state_context.rs:42-48,72-100,165-194`,
`src/clap/extensions/state.rs:139-176` e contrato oficial
`clap/ext/state-context.h`.

**Cadeia causal:** O save ForPreset remove apenas `model_path`, mas mantém
`model_search_paths` e `ir_path` absolutos. O load ForPreset não copia
`oversample` nem `activation_precision`, e não publica seus atômicos. Ele procura
o basename apenas nos diretórios absolutos serializados, sem fallback canônico.
Além disso, `state.load(state_context.save(ForPreset))` não procura basename
quando `model_path` é None, enquanto `state_context.load(...ForPreset)` procura.
Isso viola a equivalência obrigatória entre as três combinações de save/load
descritas pela extensão.

**Efeito na DAW:** Presets perdem qualidade/CPU configurada, não encontram o
modelo após mover de máquina e podem vazar caminhos locais. IR é serializado mas
ignorado no branch ForPreset. Diferentes hosts restauram resultados distintos
conforme a extensão escolhida.

**Solução proposta:** Definir um schema de asset portátil com basename, hash ou
ID, tipo e roots canônicos, sem caminhos absolutos em ForPreset. Centralizar um
único `apply_loaded_state(context)` usado por state e state-context. Restaurar
todos os parâmetros de áudio ou documentar/excluir explicitamente os não
aplicáveis, mantendo equivalência.

**Critérios de aceite:** Exercitar as três equivalências oficiais com comparação
de estado e áudio. Mover modelo/IR entre diretórios e usuários; verificar ausência
de paths absolutos no blob ForPreset.

---

## CLAP-F016 — Falhas SPSC de recursos são ignoradas

**Severidade:** Alta

**Evidência:** `src/clap/extensions/state.rs:192-201`,
`src/clap/extensions/state_context.rs:157-194`,
`src/clap/plugin/main_thread/housekeeping.rs:114-168,289-308` e
`src/clap/plugin/main_thread/load.rs:243-265`.

**Cadeia causal:** Diversos `push()` de `LoadCabIr`, `SetOversample`, modelo slim
e Params têm o resultado descartado. Housekeeping pode limpar a flag de rebuild
mesmo sem entregar os engines. Clear IR apaga caminho e amostras da UI/state
mesmo se o audio thread continuar usando o engine antigo. `load_cabsim()` altera
cold state antes do push. O Adaptive Compute atualiza a noção de canal slim
atual antes de o rebuild ser confirmado, impedindo retry.

**Efeito na DAW:** Sob fila cheia, UI e state afirmam uma configuração enquanto o
DSP usa outra. O problema é intermitente e tende a aparecer durante reloads,
automação e stress, exatamente quando é mais difícil diagnosticar.

**Solução proposta:** Tratar comandos de recurso como transações com geração e
ack. Manter o comando pendente até enqueue e aplicação confirmadas; aplicar
coalescing “latest wins” para oversampling/modelo/IR; só atualizar metadados
visíveis após commit. Nunca limpar pedido em erro; emitir status off-RT.

**Critérios de aceite:** Saturar deliberadamente cada canal e provar retry,
coalescing, ausência de drop RT e convergência final entre parâmetro, UI, state e
engine ativo.

---

## CLAP-F017 — Preset load retorna sucesso antes do resultado real

**Severidade:** Alta

**Evidência:** `src/clap/extensions/preset_load.rs:16-64`,
`src/clap/plugin/main_thread/housekeeping.rs:191-227` e contrato
`clap/ext/preset-load.h`.

**Cadeia causal:** `from_location()` apenas coloca o path em
`ui_pending_model`, solicita callback e retorna `Ok(())`. O arquivo ainda não foi
lido nem validado. Uma falha posterior não pode ser retornada ao chamador. A
implementação também não chama `clap_host_preset_load.loaded()` no sucesso nem
`on_error()` na falha. O `load_key` do discovery é ignorado.

**Efeito na DAW:** O browser pode indicar preset carregado antes de o timbre
mudar, manter seleção dessincronizada e não receber erro para arquivo inválido.
Uma captura de state imediatamente após o retorno pode salvar o preset anterior.

**Solução proposta:** Como o callback é main-thread, carregar/validar
sincronamente em `from_location()` ou guardar uma operação pendente completa com
location/load_key e concluir via host extension. Retornar false quando não houver
sucesso conforme o contrato e notificar `loaded/on_error`.

**Critérios de aceite:** Host mock deve observar ordem `from_location -> DSP/state
commit -> loaded`. Arquivo inválido deve retornar erro e gerar `on_error`, sem
alterar seleção ou state.

---

## CLAP-F018 — Preset discovery escala mal e falha com Unicode

**Severidade:** Média

**Evidência:** `src/clap/factory/preset_discovery.rs:171-250`.

**Cadeia causal:** O extrator chamado “lightweight” usa `std::fs::read()` e carrega
todo o `.nam`, incluindo pesos, para localizar metadata. Em bibliotecas grandes,
o host indexa muitos arquivos e multiplica I/O e memória. `extract_balanced_json`
itera por `char`, mas usa contador de caracteres como índice de bytes em
`&s[..end_idx]`; metadata com UTF-8 multibyte pode truncar o JSON ou formar índice
inválido. `.namb` é anunciado, mas não tem metadata extraída. O provider também
fornece path como `load_key` para arquivo direto, embora load_key seja reservado a
presets dentro de container.

**Efeito na DAW:** Scan/preset browser lento, pico de RSS e metadata ausente ou
panic capturado em modelos com nomes/autores internacionais.

**Solução proposta:** Usar deserialização streaming/visitor que ignore weights,
com limite de bytes e parser JSON real. Para extração manual inevitável, usar
`char_indices()` e índices de bytes. Implementar metadata NAMB ou declarar a
limitação. Corrigir load_key.

**Critérios de aceite:** Indexar corpus grande sob budget de tempo/RSS, testar
Unicode, escapes, metadata profunda, arquivo truncado, NAMB e arquivos
adversariais.

---

## CLAP-F019 — Lifecycle da GUI não cumpre show/hide/floating

**Severidade:** Alta

**Evidência:** `src/clap/extensions/gui.rs:96-287`,
`src/clap/gui/window/handler.rs:127-267` e contrato oficial `clap/ext/gui.h`.

**Cadeia causal:** `set_parent()`/`set_transient()` já criam e mapeiam a janela
antes de `show()`. `show()` e `hide()` sempre retornam sucesso sem alterar
visibilidade. `set_transient()` ignora o parent, abre top-level independente e
mesmo assim retorna sucesso. `WindowEvent::WillClose` não notifica
`clap_host_gui.closed()`. Uma janela floating fechada pelo usuário pode deixar o
host acreditando que continua aberta. A thread floating pode ser abandonada após
timeout, embora ela retenha ponteiros raw com lifetime artificial.

**Risco de memória:** `NamClapSharedRef` afirma apontar para Arc leaked, mas recebe
um borrow do shared do plugin. O teste `alive_fence` não torna check+deref
atômico; se uma thread abandonada sobreviver ao plugin, existe janela de UAF.

**Efeito na DAW:** Janela fantasma, editor que não reabre, stacking errado, janela
em outro workspace e risco de acesso após destruição em teardown anômalo.

**Solução proposta:** Implementar uma máquina de estados GUI explícita
Created/Parented/Visible/Hidden/Destroyed. No X11, mapear/desmapear no thread da
janela por fila de comandos; configurar `WM_TRANSIENT_FOR`; chamar
`HostGui::closed()` em WillClose. Substituir ponteiro raw por um bridge Arc com
vida própria e impedir qualquer caminho de thread detached com acesso ao plugin.

**Critérios de aceite:** Sequências create/set_parent/show/hide/show/destroy e
floating/set_transient/close-WM/reopen repetidas, com thread-check e contagem de
threads/FD/RSS. Testar também host que chama destroy sem hide.

---

## CLAP-F020 — File picker pode travar loading e teardown

**Severidade:** Alta

**Evidência:** `src/clap/gui/ui/zones/identity.rs:113-124,265-275`,
`src/clap/gui/ui/zones/file_dialogs.rs:8-74` e
`src/clap/extensions/gui.rs:49-58`.

**Cadeia causal:** O botão marca `ColdShared::ui_loading`, mas cancel/timeout só
limpa `DialogSharedState::loading`, campo distinto que nunca é espelhado de volta.
O mesmo ocorre com IR. Assim cancelar deixa Loading permanente e bloqueia novas
tentativas. Cada operação ainda cria uma thread externa e outra interna. No
teardown, a main thread da DAW faz `join()` sem watchdog da thread externa, que
pode esperar até 120 segundos pelo picker.

**Efeito na DAW:** Depois de Cancel, o plugin parece congelado em Loading. Fechar
janela/projeto com diálogo aberto pode congelar a DAW por até dois minutos; após
timeout, a thread interna do rfd continua detached até o diálogo terminar.

**Solução proposta:** Ter uma única fonte de verdade para estado do diálogo e um
único worker cancelável. Usar API async/cancellation token; nunca fazer join
bloqueante na main thread. Publicar Selected/Cancelled/TimedOut/Failed e sempre
limpar loading. Associar owner window para stacking correto.

**Critérios de aceite:** Selecionar, cancelar, expirar e destruir plugin com cada
diálogo aberto. Cada sequência deve terminar em menos de 100 ms no host, sem
thread, janela ou flag residual.

---

## CLAP-F021 — Footer/clipboard e erro de IR não funcionam como documentado

**Severidade:** Alta

**Evidência:** `src/clap/gui/mod.rs:13-16`,
`src/clap/gui/ui/mod.rs:68-84,94-126`,
`src/clap/gui/ui/zones/identity.rs:28,128-145,323-340`,
`src/clap/gui/window/handler.rs:63-110` e
`src/clap/gui/ui/status_bar/telemetry.rs:178-233`.

**Cadeia causal:** A janela tem 275 px de altura, mas Zone 1 reserva 280 px dentro
da linha principal; Zone 5 é desenhada depois dessa linha e tende a ficar fora do
viewport. O botão de diagnóstico chama `egui_ctx.copy_text()`, que gera
`OutputCommand::CopyText`, porém o handler ignora todo `full_output.platform_output`.
Logo nada chega ao clipboard do sistema, embora o toast diga “copied”. Para erros,
model e IR compartilham um único `error_expiration/error_msg`; qualquer erro de IR
ativa primeiro o banner de model. A detecção de `ir_is_error` depende de o texto
já ser “IR load failed”, condição que não é inicializada, então o erro é atribuído
ao asset errado.

**Efeito na DAW:** Telemetria e botão de suporte podem estar invisíveis; quando
visíveis, Copy não copia; falha de IR acusa modelo. Isso elimina a principal rota
de diagnóstico prevista na documentação.

**Solução proposta:** Recalcular layout dentro de 275 px ou aumentar/negociar o
tamanho; adicionar golden screenshot/used-rect. Consumir explicitamente
`PlatformOutput.commands`, cursor e demais outputs suportados com backend de
clipboard X11 testado. Separar estados de erro de modelo e IR por enum/asset.

**Critérios de aceite:** `used_rect` deve caber em 600x275; teste com clipboard
mock deve receber o dump exato; erros NAM e IR isolados/simultâneos devem apontar
para o card correto.

---

## CLAP-F022 — Idle-skip da GUI é logicamente inalcançável

**Severidade:** Alta

**Evidência:** `src/clap/gui/ui/mod.rs:155` e
`src/clap/gui/window/handler.rs:69-113`.

**Cadeia causal:** `draw_ui()` sempre chama `request_repaint_after(30 ms)`. O
handler considera qualquer delay menor que 50 ms um `has_short_repaint`; portanto
`should_skip` sempre é false. Além disso, a decisão de pular acontece depois de
construir toda a UI, ler locks e consumir peaks, economizando apenas tessellation
e paint mesmo se fosse alcançada.

**Efeito na DAW:** Cada editor aberto repinta continuamente mesmo sem áudio ou
interação. Muitas instâncias aumentam CPU/GPU, aquecimento e risco de jitter da
interface, contrariando o teste manual de idle.

**Solução proposta:** Remover repaint incondicional. Separar cadências: frame
ativo para VU/interação, 1 Hz para telemetria e wake explícito para alterações.
Decidir o idle antes da construção cara ou manter snapshots de gerações. Não
consumir peak atomics em frame que não será pintado.

**Critérios de aceite:** Medir frames/s e CPU com 1, 8 e 32 janelas em silêncio e
com áudio. O idle deve convergir à cadência mínima definida e retomar sem atraso
perceptível.

---

## CLAP-F023 — Contrato físico de escala X11 conflita com baseview

**Severidade:** Alta

**Evidência:** `src/clap/extensions/gui.rs:85-92,136-163,176-196` e
`src/clap/gui/window/state.rs:127-141`.

**Cadeia causal:** CLAP X11 expressa tamanho em pixels físicos. `get_size()` sempre
retorna 600x275 e `set_size()` rejeita qualquer outro tamanho. Entretanto a janela
baseview é criada com `SystemScaleFactor`, interpretando 600x275 como tamanho
lógico e podendo criar 1.200x550 físicos em escala 2x. O scale informado pelo host
é usado apenas no egui, independentemente da escala que baseview consulta no SO.

**Efeito na DAW:** Em HiDPI, o child pode ser maior que o parent fornecido pelo
host, ficando cortado; em combinações XWayland/host scale pode haver double scale,
primeiro frame com salto ou texto borrado.

**Solução proposta:** Escolher uma autoridade única de escala. Para API X11,
negociar tamanho físico coerente após `set_scale` e criar baseview com política que
não reaplique a escala do SO, ou retornar tamanho físico escalado e aceitar esse
tamanho. Cobrir mudança de monitor/scale quando suportada.

**Critérios de aceite:** Escalas 1.0, 1.25, 1.5, 2.0 e parent físico conhecido,
em X11 e XWayland. Child e parent devem ter dimensões idênticas e conteúdo não
pode exceder o viewport.

---

## CLAP-F024 — Drop de uma instância desativa crash report global

**Severidade:** Média

**Evidência:** `src/clap/plugin/shared.rs:318-323` e
`src/common/panic_hook.rs:30-49,236-250`.

**Cadeia causal:** `NamClapShared::drop()` chama `set_shutdown_in_progress()`. O
estado é um `OnceLock<bool>` global e irreversível. Ao remover uma única instância
num projeto com outras instâncias ativas, o panic hook passa a considerar o
processo inteiro em shutdown e nunca mais gera o relatório NAM-rs.

**Efeito na DAW:** Crashes posteriores das instâncias restantes perdem diagnóstico
justamente em sessões multi-instância. O comportamento também torna testes
dependentes da ordem.

**Solução proposta:** Remover shutdown global por instância. Usar contador de
instâncias ou fence local; somente o teardown real do processo/entry deve marcar
shutdown global. O panic hook deve continuar ativo enquanto houver qualquer
instância viva.

**Critérios de aceite:** Criar três instâncias, destruir uma e disparar panic
controlado na segunda; o relatório deve ser produzido. Destruir todas deve evitar
deadlock sem desabilitação prematura.

---

## CLAP-F025 — QA fornece falsa confiança para os fluxos de DAW

**Severidade:** Alta

**Evidência:** `utils/tests-long.sh:569-634`,
`tests/clap/clap_lifecycle_test.rs:22-143`,
`tests/clap/tail_semantics.rs:19-53`,
`tests/clap/clap_multi_instance.rs:21-162` e `docs/testing.md:146-150`.

**Lacunas confirmadas:** A fase longa não invoca `tail_semantics` e não invoca
`test_gc_drain_on_destroy_no_leak`. O único teste multi-instance verifica
prioridade RT; a documentação afirma isolamento/no bleed sem teste correspondente.
Os testes tail apenas escrevem atômicos e simulam a fórmula, sem carregar IR nem
ouvir a extensão; ainda usam `mem::forget()` para vazar instâncias. O lifecycle
não reativa e prioriza `~/.clap/nam-rs.clap`, podendo validar um binário instalado
antigo em vez do `target/clap-audit/release/libnam_rs.so` recém-construído. Grande
parte dos testes fornece duas channels a uma porta declarada mono, exercitando
configuração que um host conforme não deve fornecer. O `clap-validator` não abre
GUI nem testa mudanças dinâmicas de latência/tail.

**Oráculos ausentes:** CabSim end-to-end, state restore pré-ativação em não-48 kHz,
deactivate/activate com modelo, bypass on->off por evento, bloco >8.192, fila de
eventos cheia, callback scheduling, thread de tail, PDC por correlação, clipboard,
show/hide, cancel de diálogo, HiDPI e precisão multi-instância.

**Solução proposta:** Criar um host harness com handlers reais de callback,
thread-check, latency, tail, preset-load e GUI mock. Fazer a fase longa apontar
explicitamente `CLAP_PLUGIN_PATH` para o artefato recém-buildado. Adicionar
meta-teste que enumera testes CLAP existentes e falha se um `#[ignore]` não tiver
invocação documentada.

**Critérios de aceite:** Cada finding crítico deve possuir teste que falha antes
da correção. `docs/testing.md` deve ser gerado ou validado contra os scripts, sem
afirmações de cobertura não executada.

---

## CLAP-F026 — Robustez periférica e redundância estrutural

**Severidade:** Média

**Evidência:** `src/clap/extensions/state.rs:90-102`,
`src/clap/extensions/state_context.rs:59-70`,
`src/clap/extensions/audio_ports_activation.rs:30-73` e
`src/clap/processor/dsp/gain.rs:1-128`.

**Problemas:** State e state-context fazem `read_to_end()` sem limite, permitindo
blob de projeto consumir memória arbitrária. Audio Ports não anuncia suporte
64-bit, mas `audio-ports-activation.set_active()` aceita `SampleSize::Float64` e
retorna sucesso; o processor só usa `into_f32()` e deixaria saída f64 sem
processamento. A extensão ainda descreve “canal R”, embora opere a única porta
mono. `dsp/gain.rs` mantém duas implementações inteiras mortas do ganho, enquanto
o orquestrador possui cópia ativa, aumentando risco de correção no lugar errado.
Erros de ativação são construídos com `Box::leak`, e uma falha depois de tomar
endpoints SPSC não faz rollback.

**Solução proposta:** Limitar state a tamanho pequeno e versionado; rejeitar f64
explicitamente; remover ou implementar corretamente audio-ports-activation;
eliminar dead code de ganho; usar erros estáticos/owned sem leak e guard de
rollback para activation.

**Critérios de aceite:** State acima do limite falha sem OOM; f64 recebe rejeição
coerente; zero dead code allow no módulo; fault injection em cada etapa de
activate permite retry bem-sucedido.

---

## Epics de implementação

Os epics abaixo ordenam a correção para minimizar retrabalho. Não constituem
`TODO-sprints.md`; são agrupamentos de findings para planejamento posterior.

### Épico E0 — Contenção imediata de comportamento enganoso

**Findings:** CLAP-F001, CLAP-F004, CLAP-F009, CLAP-F014 e CLAP-F021.

**Objetivo:** Evitar que uma build de release anuncie processamento, qualidade ou
diagnóstico que não ocorre de fato.

- [ ] E0-T01 Adicionar testes vermelhos mínimos para CabSim, reativação, restore 96 kHz, bypass on->off e bloco 8.193.
- [ ] E0-T02 Adicionar status persistente de asset/rebuild pendente ou falho, sem afirmar sucesso antes do commit.
- [ ] E0-T03 Corrigir textos/logs que hoje afirmam `max quality`, clipboard copiado ou IR ativo sem confirmação do engine.
- [ ] E0-T04 Definir se CabSim e mudança dinâmica de oversampling devem ficar temporariamente ocultos até os epics definitivos.

### Épico E1 — Lifecycle e state RT persistente

**Findings:** CLAP-F002, CLAP-F003, CLAP-F014 e CLAP-F026.

**Objetivo:** Garantir que cada ativação materialize exatamente o state visível e
que mudanças de configuração do host nunca percam o modelo.

- [ ] E1-T01 Projetar `DeactivatedDspState` e ownership de modelo, calibração, resampler, OS e CabSim.
- [ ] E1-T02 Tornar restore pré-ativação declarativo e construir recursos dependentes da taxa somente em activate.
- [ ] E1-T03 Inicializar snapshot RT completo a partir dos atômicos e validar invariantes antes de publicar o processor.
- [ ] E1-T04 Implementar rollback de endpoints/recursos em toda saída de erro de activate.
- [ ] E1-T05 Criar matriz deactivate/reactivate para sample rate, buffer, topologia, parâmetros e assets.

### Épico E2 — Pipeline determinístico e sem truncamento

**Findings:** CLAP-F007, CLAP-F008, CLAP-F011 e CLAP-F012.

**Objetivo:** Tornar a saída invariável a buffer size, densidade de eventos e
estado inicial de bypass.

- [ ] E2-T01 Substituir arrays de 1.024 eventos por consumo streaming ordenado.
- [ ] E2-T02 Fragmentar intervalos em chunks limitados por capacidade sem perder offsets.
- [ ] E2-T03 Unificar bypass e processamento wet no mesmo scheduler sample-accurate.
- [ ] E2-T04 Implementar smoothing temporal único e vetorizado.
- [ ] E2-T05 Criar property tests de block invariance e event flooding.

### Épico E3 — CabSim causal, verificável e lifecycle-safe

**Findings:** CLAP-F001, CLAP-F003, CLAP-F005 e CLAP-F014.

**Objetivo:** Entregar convolução audível, correta em taxa/bloco variável e com
tail/PDC exatos.

- [ ] E3-T01 Projetar adaptador de blocos variáveis e medir sua latência real por impulso.
- [ ] E3-T02 Integrar CabSim à ordem correta do pipeline CLAP e preservar mono estrito.
- [ ] E3-T03 Manter processamento de tail após gate/silêncio e definir comprimento exato, sem dupla contagem de latência.
- [ ] E3-T04 Reconstruir IR a partir da fonte correta em mudanças de sample rate/buffer.
- [ ] E3-T05 Validar contra convolução direta e oracle C++ nos comprimentos de IR suportados.

### Épico E4 — Control plane e conformidade CLAP

**Findings:** CLAP-F004, CLAP-F005, CLAP-F013, CLAP-F016 e CLAP-F017.

**Objetivo:** Fazer toda mudança estrutural e notificação ocorrer na thread e fase
permitidas pelo protocolo.

- [ ] E4-T01 Implementar scheduler/handshake de comandos com ack e coalescing.
- [ ] E4-T02 Definir política de latência fixa versus restart e aplicá-la a OS/CabSim/resampler.
- [ ] E4-T03 Mover tail.changed para o audio thread e remover handle unsafe.
- [ ] E4-T04 Tornar eventos GUI retryable, globais e com gestures completos.
- [ ] E4-T05 Implementar HostPresetLoad loaded/on_error e semântica síncrona/assíncrona coerente.

### Épico E5 — Isolamento multi-instância e RT-safety

**Findings:** CLAP-F006, CLAP-F010 e CLAP-F024.

**Objetivo:** Eliminar estado de processamento global e qualquer alocação/lock em
callbacks de áudio.

- [ ] E5-T01 Substituir ActivationPrecision global por contexto por instância ou guard TLS obrigatório.
- [ ] E5-T02 Remover logs dos setters/cold swaps RT e mapear transições para status flags.
- [ ] E5-T03 Tornar panic shutdown consciente da contagem de instâncias.
- [ ] E5-T04 Adicionar testes multi-instância alternados, paralelos, heap-audit e thread-check.

### Épico E6 — Estado e presets transacionais/portáveis

**Findings:** CLAP-F014, CLAP-F015, CLAP-F017 e CLAP-F018.

**Objetivo:** Restaurar exatamente o mesmo som ou falhar de forma atômica e
explicável.

- [ ] E6-T01 Centralizar prepare/validate/commit para state e state-context.
- [ ] E6-T02 Criar identidade portátil de assets e busca em roots canônicos.
- [ ] E6-T03 Garantir equivalência oficial das combinações state-context.
- [ ] E6-T04 Substituir parser manual de metadata por streaming bounded e suportar Unicode/NAMB.
- [ ] E6-T05 Adicionar testes cross-machine simulados e assets ausentes/corrompidos.

### Épico E7 — GUI previsível, econômica e acessível

**Findings:** CLAP-F019, CLAP-F020, CLAP-F021, CLAP-F022 e CLAP-F023.

**Objetivo:** Fazer a janela obedecer ao host, nunca bloquear a DAW e apresentar
feedback verdadeiro em desktop e HiDPI.

- [ ] E7-T01 Implementar state machine show/hide/transient/closed/destroy.
- [ ] E7-T02 Remover ponteiro raw do bridge GUI e tornar teardown bounded sem thread detached.
- [ ] E7-T03 Reescrever file picker com cancelamento e estado único.
- [ ] E7-T04 Corrigir layout, erros por asset e consumo de PlatformOutput/clipboard.
- [ ] E7-T05 Implementar política de repaint orientada a atividade.
- [ ] E7-T06 Corrigir negociação física HiDPI e criar harness visual/layout.
- [ ] E7-T07 Completar semântica de foco, cursor e saída de acessibilidade do egui.

### Épico E8 — QA que reproduz DAWs reais

**Findings:** CLAP-F025 e todos os findings críticos.

**Objetivo:** Transformar cada garantia documental em oráculo automatizado capaz
de falhar.

- [ ] E8-T01 Criar host harness com callback, restart, latency, tail, preset-load, thread-check e filas limitadas.
- [ ] E8-T02 Testar sempre o `.so` recém-construído por path explícito e hash registrado no log.
- [ ] E8-T03 Corrigir seleção de testes ignorados e inventário docs/scripts por meta-teste.
- [ ] E8-T04 Adicionar paridade CLAP end-to-end contra NAMcore em 44,1/48/96 kHz e buffers irregulares.
- [ ] E8-T05 Adicionar testes headful Xvfb/X11 para lifecycle GUI, clipboard, picker e HiDPI.
- [ ] E8-T06 Atualizar `docs/clap_integration.md`, `docs/testing.md`, `docs/functional-tests.md`, `README.md` e `docs/architecture.md` somente após os contratos serem comprovados.

## Ordem recomendada

1. E0 cria contenção e testes vermelhos.
2. E1 e E2 estabilizam ownership e fluxo de amostras.
3. E4 fixa o protocolo de mudanças estruturais.
4. E3 integra CabSim sobre a base estável.
5. E5 elimina interferência multi-instância e violações RT.
6. E6 corrige persistência e presets.
7. E7 corrige GUI e UX.
8. E8 consolida gates e sincroniza documentação.

## Riscos de implementação

- Alterar latência durante execução sem uma decisão explícita de fixed-latency ou restart continuará incompatível com CLAP.
- Inserir CabSim antes de resolver chunking, tail e block-size variável cria uma correção aparentemente funcional, mas não determinística.
- Forçar 4x em LSTM sem golden por topologia pode piorar a paridade NAMcore; “mais oversampling” não equivale automaticamente a “mais fidelidade”.
- Corrigir a GUI sem remover lifetimes artificiais mantém o risco de UAF em caminhos de timeout.
- Aumentar capacidades SPSC apenas adia a dessincronização; comandos de recurso precisam de retry/ack, não de filas maiores.

## Condição de encerramento da auditoria

Um finding só pode ser encerrado quando houver teste reproduzindo a falha anterior,
correção aprovada no modo release, heap/RT audit quando aplicável, validação no
host harness e atualização das fontes de verdade afetadas. Aprovação isolada do
`clap-validator` não é suficiente para nenhum finding dinâmico desta lista.
