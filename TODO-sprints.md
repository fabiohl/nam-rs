<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved. -->

# TODO — Melhorias variadas para DAWs

---

## Épico A — Extensões CLAP Ausentes

* [x] **Tarefa A.1** — Accent Color Dinâmico via `CLAP_EXT_TRACK_INFO`

  * **Contexto:**
    O `TODO-Testes-Funcionais.md` exige que a GUI do plugin adapte sua cor de destaque (arcos de knobs, VU meters, LED de bypass) à cor da trilha definida pelo host. Atualmente, `COL_ACCENT` é uma constante `#00D4AA` hardcoded em `src/clap/gui/ui.rs:L15`. Nenhuma referência a `track_info` existe no codebase. A extensão `CLAP_EXT_TRACK_INFO` (`clap/ext/track-info.h`) é parte estável do spec CLAP 1.2 e é implementada pelo Bitwig Studio, que expõe a cor da trilha via `clap_track_info.color` quando o flag `CLAP_TRACK_INFO_HAS_TRACK_COLOR` está setado.

  * **Ações Técnicas:**

    1. **Dependência:** Adicionar a feature `track-info` ao `clack-extensions` em `Cargo.toml` (se disponível). Caso `clack-extensions` não exponha a extensão, implementar via FFI raw (`clap_host_track_info`, `clap_plugin_track_info`) em um novo arquivo `src/clap/extensions/track_info.rs`.

    2. **Shared State:** Adicionar ao `NamClapShared` (`src/clap/plugin.rs`) um campo atômico para a cor da trilha:

       ```rust
       /// Cor accent do host (ARGB empacotado em u32). 0 = sem cor fornecida (fallback para COL_ACCENT).
       pub track_accent_color: AtomicU32,
       ```

    3. **Plugin-Side Callback:** Implementar `clap_plugin_track_info.changed()` na `NamClapMainThread`. Ao receber a notificação:

       * Chamar `host.get_extension::<HostTrackInfo>().get()` para obter a struct `clap_track_info`.
       * Verificar se `flags & CLAP_TRACK_INFO_HAS_TRACK_COLOR != 0`.
       * Extrair os canais RGBA da `clap_color_t` e empacotar em `u32` via `(r << 24) | (g << 16) | (b << 8) | a`.
       * Armazenar no `track_accent_color` com `Ordering::Relaxed`.

    4. **GUI Integration:** Em `src/clap/gui/ui.rs`, criar uma função helper:

       ```rust
       fn resolve_accent(shared: &NamClapShared) -> egui::Color32 {
           let packed = shared.track_accent_color.load(Ordering::Relaxed);
           if packed == 0 { COL_ACCENT } // Fallback
           else {
               egui::Color32::from_rgb(
                   ((packed >> 24) & 0xFF) as u8,
                   ((packed >> 16) & 0xFF) as u8,
                   ((packed >> 8) & 0xFF) as u8,
               )
           }
       }
       ```

       Substituir todas as 8 referências a `COL_ACCENT` (L15, L432, L495, L514, L641, L659, L841, e o glow do bypass L440) por chamadas a `resolve_accent(shared)`.

    5. **Declaração da Extensão:** Registrar em `declare_extensions()` em `src/clap/plugin.rs`.

    6. **Testes:** Teste unitário validando a conversão ARGB ↔ `egui::Color32` com 5 cores representativas (branco, preto, vermelho puro, verde puro, azul Bitwig `#5e81ac`).

  * **Aceite:**

    * Ao mudar a cor da trilha no Bitwig Studio, a cor dos arcos dos knobs e LEDs da GUI muda em < 100ms.
    * Se o host não fornecer cor (Fender Studio Pro ou hosts sem suporte), o plugin usa o fallback turquesa `#00D4AA` sem qualquer erro.
    * `clap-validator` não reporta warnings relacionados à extensão `track-info`.

---

* [x] **Tarefa A.2** — Remote Controls Extension (Páginas de Hardware)

  * **Contexto:**
    O `TODO-Testes-Funcionais.md` exige que o Bitwig Device Panel exiba 2 páginas pré-configuradas: "Main" (INPUT, OUTPUT, BYPASS) e "Gate" (GATE). Sem a extensão `CLAP_EXT_REMOTE_CONTROLS` (`clap.remote-controls/2`), o Bitwig mostra uma lista flat não-organizada dos 4 parâmetros. Cada página suporta até 8 slots mapeáveis a `param_id`s.

  * **Ações Técnicas:**

    1. **Dependência:** Adicionar a feature `remote-controls` ao `clack-extensions` em `Cargo.toml` (se disponível). Caso contrário, implementar via FFI raw em `src/clap/extensions/remote_controls.rs`.
    2. **Implementação da Trait `PluginRemoteControls`:** Criar o arquivo `src/clap/extensions/remote_controls.rs` com:
       * `fn count() -> u32` → retorna `2` (duas páginas).
       * `fn get(page_index: u32, page: &mut RemoteControlsPageWriter)` → preenche as páginas:
         * **Página 0 — "Main":** `section_name = "NAM-rs"`, `page_name = "Main"`, `page_id = 0`. Slots: `[0] = PARAM_INPUT_GAIN`, `[1] = PARAM_OUTPUT_GAIN`, `[2] = PARAM_BYPASS`, `[3..7] = CLAP_INVALID_ID`.
         * **Página 1 — "Gate":** `section_name = "NAM-rs"`, `page_name = "Gate"`, `page_id = 1`. Slots: `[0] = PARAM_GATE_THRESH`, `[1..7] = CLAP_INVALID_ID`.
    3. **Declaração da Extensão:** Registrar em `declare_extensions()` em `src/clap/plugin.rs`.
    4. **Módulo:** Adicionar `pub mod remote_controls;` em `src/clap/extensions/mod.rs`.

  * **Aceite:**

    * No Bitwig Studio, ao abrir o Device Panel do NAM-rs, as páginas "Main" e "Gate" aparecem com os parâmetros corretamente mapeados.
    * Mover um knob na GUI deve refletir no Device Panel. Mover no Device Panel deve refletir na GUI.
    * `clap-validator` passa sem warnings para a extensão `remote-controls`.

---

* [x] **Tarefa A.3** — Parameter Indication Extension (Feedback de Automação/Mapping)

  * **Contexto:**
    A extensão `CLAP_EXT_PARAM_INDICATION` (`clap.param-indication/4`) permite que o host informe ao plugin quando um parâmetro está mapeado a um controlador físico, reproduzindo automação, ou sendo overridado pelo usuário. É complementar à Tarefa A.2 (Remote Controls) — juntas, permitem a sincronia bidirecional GUI ↔ Hardware prevista no item 5 do Épico 1.

  * **Ações Técnicas:**

    1. **Dependência:** Verificar suporte em `clack-extensions`. Se ausente, implementar via FFI raw em `src/clap/extensions/param_indication.rs`.

    2. **Shared State:** Adicionar ao `NamClapShared` um array de `AtomicU8` para indicação por parâmetro:

       ```rust
       /// Bits de indicação por parâmetro: bit 0 = IS_MAPPED, bit 1 = IS_AUTOMATING, bit 2 = IS_OVERRIDING.
       pub param_indication: [std::sync::atomic::AtomicU8; 4],
       ```

    3. **Implementação:** No handler `set_mapping()`:

       * Se `has_mapping == true`, setar bit 0 no `param_indication[param_index]`.
       * Se `has_mapping == false`, limpar bit 0.
         Em `set_automation()`:
       * Setar/limpar bits 1 e 2 conforme o `automation_state`.

    4. **GUI Visual Feedback:** Em `src/clap/gui/ui.rs`, no `knob_widget()`:

       * Se bit 0 (IS_MAPPED): desenhar um halo pontilhado com 6 dots azuis (`#5e81ac`) ao redor do corpo do knob (raio externo + 5px).
       * Se bit 1 (IS_AUTOMATING): aplicar pulsação de alpha (0.3 → 1.0) no arco ativo, com ciclo de ~1s via `sin(time * PI)`.
       * Se bit 2 (IS_OVERRIDING): alternar a cor do arco para `COL_AMBER` temporariamente.

    5. **Declaração da Extensão:** Registrar em `declare_extensions()`.

    6. **Módulo:** Adicionar em `src/clap/extensions/mod.rs`.

  * **Aceite:**

    * Ao mapear um knob do NAM-rs a um controlador MIDI no Bitwig, um halo visual aparece ao redor do knob na GUI do plugin.
    * Durante playback de automação, o arco do knob pulsa suavemente.
    * O `clap-validator` não reporta warnings sobre `param-indication`.

* [x] **Tarefa A.4** —  Atualização da Tabela de Extensões em `docs/architecture.md`

| Extensão                       | Arquivo                          | Responsabilidade                                                                    |
|:------------------------------ |:-------------------------------- |:----------------------------------------------------------------------------------- |
| `clap_plugin_track_info`       | `extensions/track_info.rs`       | Adaptação da cor de destaque da GUI à cor da trilha do host (Accent Color dinâmico) |
| `clap_plugin_remote_controls`  | `extensions/remote_controls.rs`  | Páginas pré-configuradas para Device Panel e controladores de hardware              |
| `clap_plugin_param_indication` | `extensions/param_indication.rs` | Feedback visual de mapeamento, automação e override de parâmetros                   |

---

## Épico B — Robustez RT e Performance

* [x] **Tarefa B.1** — VU Meters via GPU Shader (`PaintCallback` + GLSL)

  * **Contexto:**
    O repaint a 30ms (`ui.rs:L862`) força a reconstrução completa de toda a cena `egui` (tessellation de ~200+ primitivas Shape) a cada frame, mesmo quando apenas os 2 medidores VU precisam ser atualizados. Em cenários com múltiplas instâncias do plugin (4+ no Épico 3), a carga acumulada da GUI thread torna-se significativa.

  * **Ações Técnicas:**

    1. **Shader GLSL:** Criar `src/clap/gui/vu_shader.glsl` (ou incluir inline como `const &str`) com um fragment shader procedural:
       * **Inputs (Uniforms):** `u_peak` (f32), `u_hold` (f32), `u_clip` (bool), `u_meter_rect` (vec4 xywh), `u_colors` (3x vec3 para verde/amarelo/vermelho).
       * **Lógica:** O shader calcula o gradiente tricolor e a linha de peak hold diretamente na GPU, sem tessellation de primitivas no CPU.
    2. **Compilação do Shader:** No `NamPluginWindow::new()` (`src/clap/gui/window.rs`), compilar o shader via `glow::Context::create_program()` uma única vez e armazenar o `ProgramId` no estado da janela.
    3. **Paint Callback:** Refatorar `draw_vertical_meter()` em `src/clap/gui/ui.rs`:
       * Substituir a construção de `Shape::line` e `Shape::rect_filled` por um `egui::PaintCallback` que invoca o shader compilado.
       * O callback recebe os valores atuais de `peak_val`, `hold_val` e `clipped` como uniforms.
       * Renderiza um único quad (2 triângulos) por medidor.
    4. **Manter fallback:** O LED de clipping e os labels "L"/"R" continuam como primitivas egui (baixa frequência de atualização, ~1Hz).
    5. **Benchmark:** Comparar CPU time da UI thread com e sem o shader via `puffin` profiler ou `minstant` timing no `on_frame()`. Meta: ≥ 30% de redução.

  * **Aceite:**

    * Os medidores VU são visualmente idênticos à implementação atual (gradiente tricolor, peak hold, clipping LED).
    * A carga de CPU da GUI thread medida por `minstant` no `on_frame()` reduz em ≥ 30% em repaints contínuos a 60fps.
    * Zero artefatos visuais, flicker ou desalinhamento.

---

* [x] **Tarefa B.2** — ParamSmoother: Convergência Relativa e Denormal Guard

  * **Contexto:**
    O `ParamSmoother` em `src/clap/param_smoother.rs` usa um threshold de snap fixo `1e-6` (L55). Para ganhos altos (+12dB = ~3.98 linear), a convergência desperdiça centenas de samples extras. Adicionalmente, quando `target ≈ 0` e `alpha` é pequeno, o campo `current` pode entrar em território subnormal (< 1e-38), causando slowdown catastrófico na FPU x86 — violação direta da Diretriz Técnica §2.1 (Prevenção de Subnormais).

  * **Ações Técnicas:**

    1. **Threshold Relativo:** Substituir em `src/clap/param_smoother.rs`, método `tick()` (L54-L61):

       ```rust
       #[inline]
       pub fn tick(&mut self) -> f32 {
           let diff = self.current - self.target;
           // Threshold proporcional ao target: convergência 2-5x mais rápida para valores altos.
           let threshold = 1e-6 * self.target.abs().max(1.0);
           if diff.abs() < threshold {
               self.current = self.target;
           } else {
               self.current = self.alpha * self.target + (1.0 - self.alpha) * self.current;
               // Denormal guard (RT-Safety §2.1): flush a zero para evitar FPU slowdown.
               if self.current.abs() < 1e-15 {
                   self.current = 0.0;
               }
           }
           self.current
       }
       ```

    2. **Testes Unitários:** Adicionar no `mod tests` existente:

       * `test_smoother_convergence_high_gain`: Verificar que para target = 3.98 (≈ +12dB), o smoother converge em ≤ 2400 samples a 48kHz (50ms).
       * `test_smoother_denormal_prevention`: Verificar que para target = 0.0 e initial = 1e-20, o tick() retorna exatamente 0.0 após ≤ 10 iterações.
       * `test_smoother_relative_threshold`: Verificar que target = 0.001 ainda converge corretamente (não snap prematuro).

    3. **Benchmark:** Rodar `cargo bench` para garantir que a mudança não introduz regressão.

  * **Aceite:**

    * Convergência para +12dB ocorre em ≤ 2400 samples (50ms) a 48kHz.
    * Zero subnormais no campo `current` em qualquer cenário.
    * Todos os testes existentes + novos passam. Benchmark sem regressão.

---

* [x] **Tarefa B.3** — File Picker: Robustez Wayland, Alive Fence e Timeout

  * **Contexto:**
    O File Picker atual usa `rfd::FileDialog::new()` síncrono em uma thread spawnada (`src/clap/gui/ui.rs:L558-L574`). Existem 3 riscos documentados: (1) Falha silenciosa em sessões Wayland puras por falta de `xdg-desktop-portal`; (2) Deadlock de re-entrancy GTK se o host também usar GTK; (3) `std::mem::transmute` do `HostSharedHandle` (`ui.rs:L555-L556`) fica dangling se o host destruir o plugin enquanto o picker está aberto.

  > **⚠ Nota de Auditoria (Épico A, 2026-05-21):** O risco (3) foi confirmado durante a revisão do Épico A. O ponteiro cru `shared_addr` em `ui.rs:L709` e o `transmute` do host em `ui.rs:L711` permanecem ativos sem proteção de alive fence. Esta tarefa é de alta prioridade de segurança.

  * **Ações Técnicas:**

    1. **Alive Fence:** Adicionar ao `NamClapShared` um `Arc<AtomicBool>`:

       ```rust
       /// Fence de vida útil: true enquanto o plugin existe. Verificado pela thread do File Picker.
       pub alive_fence: Arc<std::sync::atomic::AtomicBool>,
       ```

       Inicializar como `true` no `new_shared()`. Setar `false` no `destroy()` da GUI (`extensions/gui.rs`). Na thread do picker (`ui.rs`), antes de usar o `shared_addr`, verificar `alive_fence.load(Relaxed)`. Se `false`, abortar silenciosamente.

    2. **Timeout de Segurança:** Envolver a chamada ao `rfd::FileDialog` em uma thread com `JoinHandle` + timeout de 120s. Se exceder, abortar via `alive_fence = false` + resetar `ui_loading`.

    3. **Detecção de Backend:** Na inicialização da GUI (`NamPluginWindow::new()`), verificar `std::env::var("XDG_SESSION_TYPE")`. Se `"wayland"`, logar via `HostLog` uma mensagem informativa: `"NAM-rs: Wayland session detected, using XDG portal for file dialogs"`. Isso facilita o diagnóstico em caso de problemas.

    4. **Refatoração do Transmute:** Documentar com SAFETY comment detalhado explicando a vida útil garantida pelo `alive_fence`, mantendo o `transmute` mas adicionando a verificação runtime.

    5. **Teste Manual:** Testar em sessão Wayland (Sway ou GNOME Wayland) com e sem `xdg-desktop-portal-gtk` instalado.

  * **Aceite:**

    * O File Picker funciona em sessões X11 e Wayland sem congelamento da DAW.
    * Se o plugin for destruído enquanto o picker está aberto, não ocorre crash (alive fence previne o acesso ao ponteiro liberado).
    * Se o portal D-Bus falhar ou travar, o timeout de 120s reseta o estado de loading na GUI.

---

> **📝 Auditoria de Sprint (Épico B - 2026-05-21):** O Épico B passou por auditoria completa. As métricas de RT-Safety e performance foram atendidas.
>
> * A refatoração do shader VU (B.1) e o `ParamSmoother` (B.2) reduziram estresse térmico/CPU e mitigaram overhead de subnormais.
> * A infraestrutura adicionada na B.3 (`alive_fence`, Wayland fallback) proverá um alicerce seguro para as funcionalidades de UI interativas planejadas para o Épico C (ex: Tarefa C.1 e C.3), garantindo a estabilidade da GUI em DAWs complexas.
> * **Status:** Épico concluído com sucesso. Aprovado para avançar ao Épico C.

---

## Épico C — Inovações de UX

* [x] **Tarefa C.1** — Drag & Drop de Arquivos de Modelo

  * **Contexto:**
    O carregamento de modelos atualmente é exclusivo via File Picker (`📂 Load Model`). Drag & Drop de arquivos `.nam`/`.namb` diretamente sobre a janela do plugin é uma expectativa de UX em 2026 — sua ausência é percebida como regressão pelos músicos acostumados a organizar modelos em file managers.

  * **Ações Técnicas:**

    1. **Baseview Drop Events:** Investigar se `baseview` já expõe eventos de Drag & Drop (X11 `XdndDrop` protocol). Se não, verificar se a versão mais recente do crate suporta ou se é necessário um fork/PR.
    2. **Fallback X11 Raw:** Se `baseview` não expor D&D nativamente, implementar o listener XDnD diretamente via `xcb` ou `x11rb` no `NamPluginWindow`, capturando `XdndEnter`, `XdndPosition`, `XdndDrop` e `XdndLeave`.
    3. **Validação do Arquivo:** Ao receber o drop:
       * Extrair o path do URI recebido (`file:///path/to/model.nam`).
       * Verificar extensão (`.nam` ou `.namb`). Se inválida, ignorar silenciosamente.
       * Setar `ui_pending_model` e chamar `host.request_callback()` — mesmo fluxo do File Picker.
    4. **Overlay Visual:** Ao receber `XdndEnter` com tipo MIME válido:
       * Desenhar um overlay semitransparente sobre toda a GUI (`egui::Area::new("drop_overlay")`) com texto "Drop NAM Model Here ⬇️" centralizado, cor de fundo `COL_BG` com alpha 0.85.
       * Remover o overlay em `XdndLeave` ou após o processamento do drop.
    5. **Testes Manuais:** Arrastar um arquivo `.nam` do Nautilus/Dolphin sobre a janela do plugin no Bitwig. O modelo deve ser carregado sem precisar do File Picker.

  * **Aceite:**

    * Arrastar um `.nam` ou `.namb` sobre a janela do plugin carrega o modelo com sucesso.
    * Arrastar um arquivo de extensão inválida (ex: `.wav`) não causa erro nem crash.
    * O overlay visual aparece ao entrar na zona de drop e desaparece ao sair ou ao concluir.

---

* [x] **Tarefa C.2** — DSP Load Meter na Status Bar (Sempre Visível)

  * **Contexto:**
    O painel de telemetria RT mostra cycles/overloads, mas está escondido atrás de um toggle "RT" e exibe valores brutos em nanosegundos que são difíceis de interpretar. O músico precisa ver IMEDIATAMENTE se o modelo está consumindo recursos demais — informação que todo plugin profissional deveria expor na interface principal.

  * **Ações Técnicas:**

    1. **Cálculo:** Na status bar de `src/clap/gui/ui.rs` (Zona 5, L790+), adicionar um indicador de carga:

       ```rust
       let sr = shared.sample_rate.load(Ordering::Relaxed);
       let last_n = shared.rt_status.last_n_samples.load(Ordering::Relaxed);
       let cycle_ns = shared.rt_status.dsp_cycle_time.load(Ordering::Relaxed);
       let load_pct = if sr > 0 && last_n > 0 {
           let budget_ns = (last_n as u64 * 1_000_000_000) / sr as u64;
           ((cycle_ns as f64 / budget_ns as f64) * 100.0).min(999.0)
       } else { 0.0 };
       ```

    2. **Widget Visual:** Inserir entre o indicador de latência e o botão "RT" na status bar:

       * Texto: `"DSP: XX.X%"` em fonte monospace 10pt.
       * Cor dinâmica: verde (< 50%), `COL_AMBER` (50-80%), `COL_VU_RED` (> 80%).
       * Tooltip no hover: `"DSP Load: XX.X% (YYμs / ZZμs budget)"`.

    3. **Amortecimento:** Usar o mesmo ciclo de atualização de 1s já existente para telemetria (`state.last_telem_update`) para evitar flickering do valor. Armazenar `telem_load_pct: f64` no `UiState`.

    4. **Separador:** Adicionar pipe `"|"` antes e depois do indicador para manter o padrão visual da status bar.

  * **Aceite:**

    * O indicador "DSP: XX.X%" é visível permanentemente na status bar, sem precisar abrir o painel de telemetria.
    * A cor muda conforme a carga (verde/amarelo/vermelho).
    * O tooltip exibe o detalhamento em microsegundos e budget.

---

> **📝 Auditoria de Sprint (Épico C - 2026-05-21):** O Épico C passou por auditoria completa. As métricas de UX e estabilidade RT foram validadas.
>
> * A implementação do Drag & Drop (C.1) foi validada, provendo o fallback overlay e operando de forma correta e síncrona aos estados da janela (`ui_pending_model`).
> * O indicador de Carga DSP (C.2) está perfeitamente implementado na Status bar, atualizando amortizadamente (1s) e indicando percentuais RT-Safe com precisão.
> * **Status:** Épico concluído com sucesso e sem regressões na bateria de testes (`cargo check && cargo test` pass). Aprovado para avançar ao Épico D.

---

## Épico D — Higiene, Feedback e Qualidade (Quick-Wins)

---

* [x] **Tarefa D.1** — Mensagens de UI/Log em Inglês Internacional

  * **Contexto:**
    Mensagens residuais em português na UI ou nos logs devem ser migradas para inglês para consistência internacional. Comentários de código permanecem em português conforme convenção do projeto.
  * **Ações Técnicas:**
    1. Grep por strings user-facing (labels, logs, mensagens de erro, etc) em português no diretório `src/`.
    2. Traduzir o `README.md` e esclarecer que muitas coisas devem continuar em português por mais um tempo ainda.
    3. Traduzir para inglês conciso, sem tecnicismos e de simples entendimentos para o público internacional.
    4. Manter comentários de código em português. Fazer o mesmo para a pasta `docs/`.
  * **Aceite:** Todas as strings user-facing em inglês.

---

* [ ] **Tarefa D.2** — Feedback Visual de Erro no Carregamento de Modelo

  * **Contexto:**
    Se o `load_model()` falhar (arquivo corrompido, formato inválido, path inexistente), o plugin atualmente reseta silenciosamente o flag `ui_loading` sem feedback visual. O músico vê apenas a caixa voltar para `"No model loaded"` sem entender o que aconteceu. Isso viola o Teste 7.4 e o Teste 1.6 do `TODO-Testes-Funcionais.md`.
  * **Ações Técnicas:**
    1. Adicionar ao `NamClapShared` um `AtomicBool` `ui_load_error` e uma `Mutex<String>` `ui_load_error_msg`.
    2. No `on_main_thread()`, ao falhar o carregamento, setar ambos com mensagem breve (ex: `"Invalid format"`, `"File not found"`).
    3. Na GUI, se `ui_load_error` estiver true, exibir na caixa de modelo: `"⚠ Load failed"` em `COL_VU_RED` por 3 segundos (timer no `UiState`), depois reverter ao estado anterior.
    4. Logar detalhes completos via `HostLog`.
  * **Aceite:**
    * Ao carregar arquivo corrompido ou inexistente, a caixa exibe `"⚠ Load failed"` por 3s e reverte.
    * Zero crash ou panic.

---

* [ ] **Tarefa D.3** — Tooltip Contextualizado para Gate

  * **Contexto:**
    O tooltip de hover mostra `"{:.2} dB"` genérico para todos os knobs. Para o GATE (threshold), um contexto adicional melhora a compreensibilidade.
  * **Ações Técnicas:**
    1. Parametrizar o sufixo do tooltip em `handle_knob()` com um `tooltip_suffix: &str`.
    2. Para GATE, usar `" dB (Threshold)"`.
    3. Para INPUT/OUTPUT, manter `" dB"`.
  * **Aceite:** Tooltip do GATE mostra contexto adicional. INPUT/OUTPUT permanecem limpos.

---

* [ ] **Tarefa D.4** — Acessibilidade: Contraste e Navegação por Teclado

  * **Contexto:**
    Boas práticas de UX de plugins profissionais (referência: WCAG 2.1 AA, FabFilter Pro-Q, Surge XT). O NAM-rs atualmente não oferece navegação por teclado (Tab entre controles, Arrow keys para ajuste) — funcionalidade esperada para acessibilidade e workflows power-user.
  * **Ações Técnicas:**
    1. **Verificação de contraste:** Calcular relações de contraste de todas as combinações texto/fundo na paleta. Se alguma estiver abaixo de 4.5:1, propor ajustes mínimos (ex: `COL_MUTED` `#8B95A5` sobre `COL_PANEL` `#232830` → verificar).
    2. **Tab Focus Ring:** No `knob_widget()`, se o widget tiver keyboard focus, desenhar um anel de focus de 2px na cor accent ao redor do knob body.
    3. **Tab Order:** Tab navega: INPUT → OUTPUT → GATE → BYPASS → Load Model.
    4. **Arrow Keys para Knobs:** Se um knob tiver focus, ↑/↓ ajustam ±1 step, Ctrl+↑/↓ ajustam ±0.1 step (fine-tune por teclado).
  * **Aceite:**
    * Tab navega entre os 5 controles interativos.
    * Focus ring visível.
    * Arrow keys ajustam valores.

---

* [ ] **Tarefa D.5** — Remoção Completa do Modo Headless

  * **Contexto:**
    O modo headless não é útil, ao menos agora, para o público do NAM-rs. Focaremos no modo CLI e no modo Plugin.
  * **Ações Técnicas:**
    1. Grep por referências a "headless" em todo o codebase (`src/`, `Cargo.toml`, `docs/`, scripts em `utils/`).
    2. Remover imports, cfg flags, e trechos de código morto associados.
    3. Atualizar documentação (`README.md`, `docs/architecture.md`) para refletir a remoção.
  * **Aceite:** Zero referências a "headless" no código ou documentação.
