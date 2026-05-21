<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved. -->

# TODO — Validação Humana em DAWs (Bitwig & Fender Studio Pro)

* **Contexto:** Validação completa da experiência do usuário ponta-a-ponta nas DAWs de referência (Bitwig Studio 6+ e Fender Studio Pro), certificando que o comportamento visual, a latência de interface e o áudio funcionam de forma impecável e harmoniosa. Cada seção abaixo é um teste independente com critérios de PASS/FAIL explícitos.

## Pré-Requisitos do Ambiente de Teste

* [ ] Sistema operacional: Ubuntu 26.04+ (ou derivado) com sessão gráfica X11 ou Wayland+XWayland.
* [ ] Plugin instalado: `~/.clap/nam-rs.clap` (build Release: `cargo build --no-default-features --features clap-plugin-gui --lib --release`).
* [ ] DAWs configuradas: Bitwig Studio 6+ e Fender Studio Pro instalados e funcionais.
* [ ] Modelo de teste: Ter pelo menos 2 modelos `.nam` ou `.namb` disponíveis (ex: `jcm800.nam`, `twin_reverb.nam`). Recomenda-se um modelo WaveNet e um LSTM para variar a carga de CPU.
* [ ] Sinal de áudio de entrada: DI de guitarra gravado (ex: `tests/amostra-guitarra.wav`) ou gerador de sinal (tone/sweep) conectado a uma trilha de áudio com o NAM-rs inserido.
* [ ] Monitoramento de XRUNs: Abrir em um terminal separado: `pw-top` (PipeWire) ou `jack_ipc` (JACK) para observar contadores de XRUNs durante toda a sessão de teste.
* [ ] Arquivo de modelo **inválido** para testes de erro: arquivo texto `.txt` renomeado para `.nam` (ex: `invalid_model.nam`).
* [ ] *(Opcional, se Tarefa B.3 implementada)* Sessão Wayland disponível (Sway ou GNOME Wayland) para teste de compatibilidade do File Picker.
* [ ] ⚠️ **Volume seguro:** Antes de iniciar qualquer teste, certificar-se de que o volume de monitoração está em nível seguro (≤ -20 dB nos outputs). Testes de clipping podem gerar picos altos.

## Teste 1 — File Picker & Thread Safety

> **Hosts:** Bitwig Studio e Fender Studio Pro.

* [ ] **1.1** Inserir o NAM-rs em uma trilha de áudio e abrir a GUI. A janela deve ter exatamente **600×260 pixels** (tamanho fixo). Verificar que a GUI abre sem delay perceptível (< 500ms).
  * ❌ FAIL se: A janela aparece preta, com artefatos, ou em tamanho diferente de 600×260.
* [ ] **1.2** Iniciar playback na DAW com sinal de áudio passando pelo plugin (sem modelo carregado). Confirmar que a Zona 1 exibe `"No model loaded"` na caixa de nome do modelo.
* [ ] **1.3** Clicar no botão `[📂 Load Model]` na Zona 1 (canto superior esquerdo).
  * ✅ PASS: O File Picker do sistema (zenity/xdg-portal) abre em janela separada. A GUI do plugin e a DAW permanecem responsivas — arraste a janela da DAW ou redimensione-a enquanto o picker está aberto para confirmar.
  * ❌ FAIL se: A DAW congela, a GUI do plugin trava, ou o playback de áudio para.
* [ ] **1.4** Com o picker aberto, mover outros controles na DAW (ex: faders de outra trilha). O event loop da DAW deve permanecer 100% funcional.
* [ ] **1.5** Selecionar um arquivo `.nam` válido (ex: `jcm800.nam`). Observar a GUI do plugin:
  * A caixa do nome do modelo deve exibir imediatamente a animação de loading: `"Loading"` → `"Loading."` → `"Loading.."` → `"Loading..."` (ciclo ASCII sem emoji, ~250ms por frame).
  * Após o carregamento concluir, a caixa deve exibir o basename do arquivo (ex: `jcm800.nam`).
  * **Truncamento na Zona 1 (caixa de nome):** Se o nome for muito longo, o `egui` trunca visualmente com `"…"` conforme o espaço disponível (~120px de largura fixa). A truncagem é visual, não por contagem de caracteres.
  * **Truncamento na Zona 5 (status bar):** Se o nome exceder 35 caracteres, será truncado com `"..."` (ex: `"mesa_boogie_triple_rectifier_chann..."`).
  * ✅ PASS: Transição suave "Loading..." → nome do modelo. Áudio processado audível imediatamente após o carregamento.
  * ❌ FAIL se: A caixa permanece em "Loading..." por mais de 10s, exibe texto cortado/ilegível, ou o áudio não muda.
* [ ] **1.6** Cancelar o picker (fechar sem selecionar arquivo). A GUI deve retornar ao estado anterior sem erros. O botão `[📂 Load Model]` deve voltar a ser clicável.
* [ ] **1.7** Carregar um segundo modelo diferente (ex: `twin_reverb.nam`) por cima do anterior. Confirmar que o nome do modelo atualiza e que o áudio muda imediatamente (sem necessidade de parar/reiniciar o playback).

## Teste 2 — Controles, Fine-Tune & Resposta em Tempo Real

> **Hosts:** Bitwig Studio e Fender Studio Pro.
> **Pré-requisito:** Modelo carregado com sucesso (Teste 1 concluído).

* [ ] **2.1 — Ranges dos Knobs:** Verificar os limites exatos de cada controle:

  | Knob       | Range                   | Default    | Cor do Arco        | Tamanho |
  | ---------- | ----------------------- | ---------- | ------------------ | ------- |
  | **INPUT**  | `-96.0 dB` a `+30.0 dB` | `0.0 dB`   | Turquesa `#00D4AA` | 70×70px |
  | **OUTPUT** | `-96.0 dB` a `+30.0 dB` | `0.0 dB`   | Turquesa `#00D4AA` | 70×70px |
  | **GATE**   | `-90.0 dB` a `-40.0 dB` | `-70.0 dB` | Âmbar `#F5A623`    | 42×42px |

  Arrastar cada knob até o limite inferior e verificar que o tooltip (hover) mostra o valor mínimo. Repetir para o limite superior.

  * ❌ FAIL se: Algum knob permite valores fora dos limites listados.

* [ ] **2.2 — Drag Normal:** Arrastar o knob INPUT verticalmente. O arco turquesa deve acompanhar o arrasto de forma fluida, sem saltos. O valor em dB (label abaixo do knob, formato `X.X dB`) deve atualizar a cada frame.

  * Verificar também que o tooltip de hover (ao posicionar o mouse sobre o knob sem arrastar) mostra o valor com 2 casas decimais (formato `X.XX dB`).
  * Observar o áudio: o volume deve mudar proporcionalmente ao arraste, sem estalos (zipper noise).
  * ✅ PASS: Resposta fluida e contínua, sem zipper noise audível.

* [ ] **2.3 — Ctrl+Drag (Fine-Tune):** Manter `Ctrl` pressionado e arrastar o knob OUTPUT. A sensibilidade deve ser **10× menor** que o arraste normal (o mesmo arraste vertical produz 1/10 da variação em dB).

  * **Procedimento de verificação:** Posicionar o knob OUTPUT em `0.0 dB`. Arrastar ~50px para cima sem Ctrl e anotar o valor final (ex: `+7.5 dB`). Retornar a `0.0 dB`. Repetir o mesmo arraste com Ctrl — o valor final deve ser ~10× menor (ex: `+0.75 dB ± 0.3 dB`).
  * Verificar que o mesmo comportamento vale para o scroll do mouse: scroll normal é mais rápido, Ctrl+scroll é 10× mais lento.
  * ✅ PASS: Diferença de sensibilidade claramente perceptível (~10×).

* [ ] **2.4 — Double-Click Reset:** Clicar duas vezes (double-click rápido) no knob INPUT previamente arrastado para um valor qualquer.

  * ✅ PASS: O knob retorna imediatamente a `0.0 dB` (default). O arco e o label atualizam instantaneamente.
  * Repetir para OUTPUT (deve resetar a `0.0 dB`) e GATE (deve resetar a `-70.0 dB`).
  * ❌ FAIL se: O knob não retorna ao default ou demora mais de 1 frame (~30ms) para atualizar.

* [ ] **2.5 — Glow durante Arraste:** Enquanto arrasta qualquer knob (ex: INPUT), observar que o arco ativo ganha um **halo externo semitransparente** (efeito glow) durante o drag. Ao soltar o mouse, o glow desaparece.

  * ❌ FAIL se: Sem glow visível, ou glow permanece após soltar o mouse.

* [ ] **2.6 — Bypass Toggle:** Clicar no botão BYPASS (Zona 4, extrema direita). Verificar:

  * **Estado Ativo (padrão):** LED retangular turquesa (`#00D4AA`) com halo de glow. Label inferior: `"ACTIVE"`.
  * **Estado Bypass:** LED cinza escuro (`#4A4F5A`). Label inferior: `"BYPASSED"`. O áudio de saída deve ser o sinal de entrada limpo (dry), sem processamento neural.
  * Clicar novamente para retornar ao modo ativo. O processamento neural deve retomar instantaneamente.
  * ✅ PASS: Transição instantânea (< 1 frame) sem estalos.

## Teste 3 — Medição VU, Peak Hold & LEDs de Clipping

> **Hosts:** Bitwig Studio e Fender Studio Pro.
> **Pré-requisito:** Modelo carregado, playback ativo com sinal dinâmico.

* [ ] **3.1 — Gradiente Tricolor:** Alimentar o canal com sinal dinâmico (DI de guitarra com variações ou gerador de sinal com envelope). Observar os medidores VU na Zona 3 (barras verticais "L" e "R", 16px de largura cada):

  * Faixa de **-60 dB a -12 dB**: cor verde (`#43E97B`).
  * Faixa de **-12 dB a -3 dB**: cor amarela (`#F5CE62`).
  * Faixa de **-3 dB a +6 dB**: cor vermelha (`#F74E4E`).
  * A transição entre faixas deve ser suave (gradiente interpolado), não cortes abruptos.
  * ✅ PASS: As 3 cores são claramente distinguíveis com transições suaves.

* [ ] **3.2 — Responsividade a ~33fps:** Com sinal de transientes rápidos (ex: pick attack de guitarra), os medidores VU devem responder sem atraso perceptível (repaint a cada ~30ms ≈ 33fps). As barras devem subir em < 1 frame e descer com suavidade.

  * ❌ FAIL se: Barras congeladas, atualizando com atraso visível (>100ms), ou com "saltos" em vez de transições fluidas.

* [ ] **3.3 — Peak Hold (Retenção de 2.0 segundos):** Provocar um pico alto e parar o sinal abruptamente. Uma marca horizontal fina deve permanecer na posição do pico por exatamente **2.0 segundos**. Após esse período:

  * A marca deve iniciar um **decaimento exponencial suave** (multiplicador ×0.95 por frame de repaint, equivalente a ~1.5 dB/s de queda).
  * Usar cronômetro (ex: relógio do celular) para medir os 2.0s de retenção (tolerância ±300ms).
  * ✅ PASS: Retenção de ~2 segundos seguida de queda suave.
  * ❌ FAIL se: Marca desaparece instantaneamente, não aparece, ou permanece indefinidamente.

* [ ] **3.4 — LED de Clipping:** Elevar o ganho INPUT até provocar saturação digital (o sinal de saída excede 0 dBFS = 1.0 linear):

  * Um LED retangular vermelho (`#F74E4E`) deve acender no topo de cada medidor VU (acima do "L" e "R"). O LED é **persistente** — ele permanece aceso mesmo após o pico passar.
  * ⚠️ **Cuidado com o volume de monitoração** antes de aumentar o ganho!
  * ✅ PASS: LEDs acendem ao atingir clipping (≥ 0 dBFS). Permanecem acesos até serem resetados manualmente.

* [ ] **3.5 — Reset de Clipping:** Com LEDs de clipping acesos:

  * Clicar sobre o LED vermelho do canal L. Apenas o LED L deve apagar (o LED R permanece se também estiver aceso).
  * Clicar sobre o corpo do medidor VU do canal R. O LED R deve apagar.
  * ✅ PASS: Reset individual por canal, imediato ao clique.
  * ❌ FAIL se: Ambos resetam juntos, ou clique não tem efeito.

## Teste 4 — Automações e Gravação de Gestos

> **Host primário:** Bitwig Studio. Host secundário: Fender Studio Pro.

* [ ] **4.1 — Gravação de Automação (Bitwig Studio):**

  1. Configurar a trilha do NAM-rs em modo de escrita de automação (**Write** ou **Latch**).
  2. Iniciar playback e arrastar o knob INPUT da GUI do plugin de `0.0 dB` até `+10.0 dB` ao longo de ~3 segundos. Soltar o mouse.
  3. Parar o playback e abrir o grid de arranjo da automação de `input_gain_db`.
  * ✅ PASS: Nós de automação desenhados no grid com curva suave que reflete o movimento do mouse. O trecho antes e depois do arraste mantém os valores originais.
  * ❌ FAIL se: Nenhuma automação gravada, automação com saltos/descontinuidades, ou valores errados.

* [ ] **4.2 — Verificação de Gesture Begin/End (Bitwig Studio):**

  * Na gravação acima, o Bitwig deve ter recebido corretamente:
    * `CLAP_EVENT_PARAM_GESTURE_BEGIN` no instante em que o mouse clicou sobre o knob (início do drag).
    * `CLAP_EVENT_PARAM_GESTURE_END` no instante em que o mouse foi solto.
  * **Verificação indireta:** No Bitwig, a automação deve ter pontos de ancoragem precisos no início e no fim do gesto (sem "cauda" fantasma de automação antes ou depois).
  * Repetir para os knobs OUTPUT, GATE e o botão BYPASS.

* [ ] **4.3 — Playback de Automação (Bitwig Studio):**

  1. Desenhar manualmente uma rampa de automação no grid do Bitwig para o parâmetro `output_gain_db` (ex: de `-10 dB` a `+5 dB` em 4 compassos).
  2. Iniciar playback e observar a GUI do plugin.
  * ✅ PASS: O knob OUTPUT na GUI se move suavemente acompanhando a automação, o valor em dB atualiza continuamente, e o áudio reflete a mudança em tempo real (sem zipper noise).

* [ ] **4.4 — Sincronia Bidirecional (Fender Studio Pro):**

  * Repetir as etapas 4.1 e 4.3 no Fender Studio Pro. Validar que:
    * Mover o knob na GUI do plugin → a automação é gravada no host.
    * Mover o controle no mixer/painel do host → o knob na GUI do plugin se move.
  * ✅ PASS: Sincronia perfeita em ambas as direções.

## Teste 5 — Remote Controls e Integração de Hardware

> **Host:** Bitwig Studio.
> **⚠️ Dependência:** Requer implementação da extensão `CLAP_EXT_REMOTE_CONTROLS` (vide `TODO-sprints.md`, Tarefa A.2). **Pular se ainda não implementado.**

* [ ] **5.1** Abrir o Device Panel do Bitwig Studio para o NAM-rs. Verificar que 2 páginas aparecem:

  * **Página 0 — "Main":** 3 controles: `INPUT` (slot 0), `OUTPUT` (slot 1), `BYPASS` (slot 2). Slots 3–7 vazios.
  * **Página 1 — "Gate":** 1 controle: `GATE` (slot 0). Slots 1–7 vazios.
  * Ambas devem aparecer sob a seção `"NAM-rs"`.
  * ❌ FAIL se: Páginas não aparecem, parâmetros estão fora de ordem, ou nomes incorretos.

* [ ] **5.2** Mover o knob INPUT na GUI do plugin → o controle correspondente no Device Panel deve se mover simultaneamente. Mover no Device Panel → o knob na GUI do plugin deve refletir.

  * ✅ PASS: Sincronia bidirecional instantânea (< 100ms de latência visual).

* [ ] **5.3** Se um controlador MIDI/Hardware estiver conectado: Validar que as páginas do Device Panel são mapeáveis aos encoders do hardware e que o feedback visual funciona em todas as direções (hardware ↔ Device Panel ↔ GUI).

## Teste 6 — Accent Color Dinâmico

> **Host:** Bitwig Studio.
> **⚠️ Dependência:** Requer implementação da extensão `CLAP_EXT_TRACK_INFO` (vide `TODO-sprints.md`, Tarefa A.1). **Pular se ainda não implementado.**

* [ ] **6.1** No Bitwig Studio, clicar com botão direito na cabeça da trilha do NAM-rs e alterar a cor para **vermelho**. Observar a GUI do plugin:

  * Os arcos dos knobs INPUT e OUTPUT devem mudar de turquesa (`#00D4AA`) para vermelho (cor da trilha) em < 100ms.
  * O medidor VU **não** muda de cor (suas cores são fixas: verde/amarelo/vermelho técnico).
  * O LED do Bypass deve usar a cor da trilha quando ativo.
  * ✅ PASS: Transição instantânea, cor visualmente próxima à cor definida no Bitwig.

* [ ] **6.2** Alterar a cor da trilha para **azul** e depois para **verde**. Verificar que a cor do accent acompanha cada mudança.

* [ ] **6.3** Remover a cor da trilha (se o host permitir, restaurar a cor padrão). A GUI deve reverter ao turquesa padrão (`#00D4AA`).

* [ ] **6.4 (Fender Studio Pro):** Se o Fender Studio Pro não expor `track_info`, a GUI deve manter a cor padrão turquesa sem erros ou warnings no log.

## Teste 7 — Persistência de Sessão e Estado

> **Hosts:** Bitwig Studio e Fender Studio Pro.

* [ ] **7.1 — Preparação do Estado:** Com um modelo carregado (ex: `jcm800.nam`), ajustar os parâmetros para valores não-default:

  * INPUT: `+3.5 dB`
  * OUTPUT: `-6.0 dB`
  * GATE: `-55.0 dB`
  * BYPASS: `OFF` (ativo)
  * Anotar os valores exatos.

* [ ] **7.2 — Save/Reload (Bitwig Studio):**

  1. Salvar o projeto (`Ctrl+S`).
  2. Fechar completamente o Bitwig Studio (não apenas o projeto, mas sair do aplicativo).
  3. Reabrir o Bitwig Studio e carregar o projeto salvo.
  4. Abrir a GUI do NAM-rs.
  * ✅ PASS: Todos os parâmetros estão nos valores exatos anotados em 7.1. O modelo `jcm800.nam` está carregado (nome visível na caixa de modelo). O áudio processado é idêntico ao de antes do save.
  * ❌ FAIL se: Qualquer parâmetro retorna ao default, o modelo não é recarregado, ou a GUI exibe `"No model loaded"`.

* [ ] **7.3 — Save/Reload (Fender Studio Pro):** Repetir os passos 7.1-7.2 no Fender Studio Pro.

* [ ] **7.4 — Caminho Relativo/Absoluto:** Mover o arquivo do modelo para um diretório diferente **antes** de reabrir o projeto. Verificar que:

  * O plugin tenta carregar do path original.
  * Se o arquivo não existir, o plugin exibe `"No model loaded"` (sem crash ou panic).
  * O host loga a falha via `HostLog` (verificar no log da DAW se disponível).
  * ❌ FAIL se: Crash, panic, ou congelamento da DAW.

## Teste 8 — Integridade Visual e Layout

> **Hosts:** Bitwig Studio e Fender Studio Pro.

* [ ] **8.1 — Zona 1 (Identidade):** Verificar presença de:

  * Logo `"NAM-rs⚡"` em turquesa, 24pt, canto superior esquerdo.
  * Subtítulo `"Neural Amp Modeler"` em cinza muted, 9.5pt.
  * Versão `"vX.Y.Z"` e badge SIMD (ex: `"AVX2"` em turquesa ou `"GENERIC"` em cinza).
  * Botão `[📂 Load Model]` com borda visível e texto legível.
  * Caixa do nome do modelo com fundo escuro (`#1A1D23`).

* [ ] **8.2 — Zona 5 (Status Bar):** Na barra inferior, confirmar:

  * Nome do modelo (ou `"-"` se nenhum).
  * Sample rate (ex: `"48kHz"` ou `"Rate: Off"` se desativado).
  * Latência em samples (ex: `"512 samples"`).
  * Botão `"RT"` à direita (toggle de telemetria). Ao clicar, deve abrir/fechar o painel expandido com: Cycles, Last N, RT Prio, Overloads, Flags.

* [ ] **8.3 — Separadores:** Verificar que existem 3 separadores verticais finos (`#2E3440`) entre as 4 zonas principais (Identidade | Controles | Medidores | Bypass).

* [ ] **8.4 — Sem Artefatos:** Redimensionar a janela da DAW, minimizar e restaurar. Verificar que a GUI do plugin não apresenta:

  * Pixels pretos ou brancos espúrios.
  * Flicker ou cintilação.
  * Offsets de mouse (clique em um lugar e o feedback visual aparece em outro).
  * Sobreposição de texto ou widgets cortados.

## Teste 9 — Features Avançadas de UX

> **Hosts:** Bitwig Studio e Fender Studio Pro.
> **⚠️ Dependência:** Requer implementação das Tarefas C.1, C.2, C.3, C.4, C.5 do `TODO-sprints.md`. **Pular itens cujas features ainda não foram implementadas.**

* [ ] **9.1 — DSP Load Meter (Tarefa C.4):** Na status bar, entre a latência e o botão "RT", verificar presença do indicador `"DSP: XX.X%"`:

  * Com modelo leve: cor verde (< 50%).
  * Com modelo pesado (WaveNet grande): cor âmbar ou vermelha.
  * Hover mostra tooltip com `"DSP Load: XX.X% (YYμs / ZZμs budget)"`.
  * ✅ PASS: Indicador presente, cor muda conforme carga, tooltip informativo.
  * ❌ FAIL se: Indicador ausente, cor não muda, ou tooltip incorreto.

* [ ] **9.2 — Drag & Drop de Modelo (Tarefa C.1):** Arrastar um arquivo `.nam` do file manager (Nautilus/Dolphin) sobre a janela do plugin:

  * Overlay semitransparente com texto "Drop NAM Model Here ⬇️" deve aparecer ao entrar na zona de drop.
  * Ao soltar, o modelo carrega (mesmo fluxo de Loading → nome).
  * Arrastar um `.wav` → overlay aparece mas nada acontece ao soltar.
  * ✅ PASS: Drag funcional, overlay visual presente, extensões inválidas ignoradas.
  * ❌ FAIL se: Crash, modelo não carrega, ou overlay não aparece/desaparece.

* [ ] **9.3 — Undo de Modelo (Tarefa C.3):** Carregar modelo A (`jcm800.nam`), depois modelo B (`twin_reverb.nam`):

  * Botão "⬅" deve aparecer ao lado da caixa de nome do modelo.
  * Clicar no "⬅" ou pressionar `Ctrl+Z` → reverte a modelo A.
  * Carregar 6+ modelos → apenas os últimos 5 ficam no histórico.
  * ✅ PASS: Undo funcional, botão visível, limite de 5 entradas.
  * ❌ FAIL se: `Ctrl+Z` não funciona, botão ausente, ou histórico sem limite.

* [ ] **9.4 — Hot-Reload de Modelo (Tarefa C.5):** Com toggle "🔄 Auto" ativado na Zona 1:

  * Modificar o `.nam` no disco (ex: copiar outro arquivo por cima com `cp`).
  * O plugin detecta a mudança em ≤ 1s e recarrega automaticamente.
  * Toast "Model updated ⚡" deve aparecer na status bar por ~2 segundos.
  * Desativar o toggle → modificar o arquivo → plugin **não** recarrega.
  * ✅ PASS: Reload automático quando ativo, toast visível, toggle funcional.
  * ❌ FAIL se: Reload não ocorre, toast ausente, ou crash.

* [ ] **9.5 — LUFS Metering (Tarefa C.2):** Com modelo carregado e sinal dinâmico passando:

  * Toggle "dBFS / LUFS" abaixo dos medidores VU deve alternar modos.
  * No modo LUFS-S, valor numérico grande (18pt) exibido em vez das barras.
  * Cores do valor: verde (-24 a -14 LUFS), amarelo (-14 a -9 LUFS), vermelho (> -9 LUFS).
  * Com silêncio completo, exibe `-inf` ou `---`.
  * ✅ PASS: Alternância instantânea, valores coerentes com a dinâmica do sinal.
  * ❌ FAIL se: Toggle não funciona, valor fixo, ou crash.

## Teste 10 — Parameter Indication

> **Host:** Bitwig Studio.
> **⚠️ Dependência:** Requer implementação da Tarefa A.3 do `TODO-sprints.md`. **Pular se ainda não implementado.**

* [ ] **10.1** Mapear o knob INPUT a um controlador MIDI no Bitwig (via MIDI Learn ou Device Panel) → halo pontilhado com dots azuis ao redor do knob deve aparecer.

  * ✅ PASS: Halo visível, 6 dots azuis (`#5e81ac`), ao redor do corpo do knob.
  * ❌ FAIL se: Sem indicação visual de mapeamento.

* [ ] **10.2** Durante playback de automação de `output_gain_db` desenhada no grid do Bitwig → arco do knob OUTPUT deve pulsar suavemente (alpha oscilando 0.3 → 1.0 em ciclo de ~1s).

  * ✅ PASS: Pulsação visível e suave.
  * ❌ FAIL se: Arco estático, sem pulsação.

* [ ] **10.3** Ao override manual de parâmetro (mover knob na GUI) enquanto automação está ativa → arco muda para cor âmbar (`#F5A623`) temporariamente.

  * ✅ PASS: Cor âmbar durante override, retorna à cor normal ao soltar.
  * ❌ FAIL se: Sem feedback de override.

## Teste 11 — Erro de Carregamento de Modelo

> **Hosts:** Bitwig Studio e Fender Studio Pro.
> **⚠️ Dependência:** Requer implementação da Tarefa D.3 do `TODO-sprints.md`. **Pular se ainda não implementado.**

* [ ] **11.1** Tentar carregar o arquivo inválido preparado nos pré-requisitos (`invalid_model.nam` — arquivo texto renomeado):

  * A caixa de modelo deve exibir `"⚠ Load failed"` em vermelho (`COL_VU_RED`) por ~3 segundos.
  * Após 3s, a caixa retorna ao estado anterior (modelo anterior se havia um, ou `"No model loaded"`).
  * ✅ PASS: Feedback de erro claro, recuperação automática.
  * ❌ FAIL se: Crash, panic, loading infinito, ou sem feedback visual.

* [ ] **11.2** Tentar carregar um arquivo `.nam` de 0 bytes:

  * Mesmo feedback de erro que 11.1.
  * ✅ PASS: Tratamento idêntico a 11.1.

* [ ] **11.3** Com um modelo válido carregado (ex: `jcm800.nam`), tentar carregar o arquivo inválido:

  * O modelo anterior (`jcm800.nam`) deve permanecer carregado após o erro.
  * O áudio processado não deve ser interrompido durante a tentativa de carga.
  * ✅ PASS: Modelo anterior preservado, áudio contínuo.
  * ❌ FAIL se: Modelo anterior perdido ou áudio cortado.

### Critérios Globais de Aceite (PASS em todos os testes)

* [ ] Renderização visual a ~33fps (repaint a cada 30ms) sem oscilações ou flicker em ambas as DAWs.
* [ ] Zero XRUNs acusados pelo PipeWire/JACK durante toda a sessão de teste (verificar `pw-top`).
* [ ] Zero crashes, panics ou congelamentos da DAW durante qualquer operação.
* [ ] Zero zipper noise audível durante arraste de knobs ou playback de automação.
* [ ] O plugin responde corretamente ao fluxo completo: instanciar → carregar modelo → ajustar parâmetros → salvar → fechar DAW → reabrir → estado preservado.

### Em Caso de Falha

> Ao identificar qualquer FAIL, registrar um bug report com:
>
> 1. **Identificador do teste** (ex: "Teste 3.4").
> 2. **DAW e versão** (ex: "Bitwig Studio 6.0.1").
> 3. **Modelo carregado** e **sample rate** da sessão.
> 4. **Comportamento esperado** vs **comportamento observado**.
> 5. **Screenshot ou vídeo** da GUI no momento da falha.
> 6. **Log da DAW** (se disponível) e saída do `pw-top` (contagem de XRUNs).
> 7. **Painel RT** (clicar no botão "RT" na status bar e copiar os valores: Cycles, Last N, Overloads, Flags).
