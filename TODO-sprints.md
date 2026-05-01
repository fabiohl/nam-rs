# 🚀 Backlog do Produto e Planejamento de Sprints Técnicas

**Modelo de prompt web:** Você é uma equipe de arquitetos sêniores e desenvolvedores especialistas na linguagem Rust e no subsistema de áudio Linux. Em anexo temos o aglutinado repomix do github atualizado do NAM-rs (vide subanexo README.md). Assumindo a persona/workflow .agents/workflows/diagnostico.md e .agents/skills/planejador-arquiteto/SKILL.md você vai detalhar ao máximo a implementação da tarefa técnica demandada logo abaixo. Sua resposta será concisa e direta, usando toda a sua janela de processamento e contexto para orientar o trabalho do(s) implementador(es).

---

## 🌐 Épico: Arquitetura de Áudio & PipeWire

### [T21] Resampler Sinc-SIMD Nativo & Fase Mínima

- **Contexto Arquitetural:** Hoje usamos o crate `rubato 0.16` operando em FIR Sinc de fase linear, bidirecional planar.
- **Problema:** O filtro de fase linear causa ringing assimétrico "pré-eco" (pré-ringing), que em transients drásticos (e.g., palhetada forte de guitarras) suprime o *feel* de resposta da corda e adiciona ~1.5ms de latência pura matemática desnecessária (delay algorítmico).
- **Solução Proposta:** Abandonar `rubato` no núcleo quente. Implementar localmente em `src/dsp/resampler.rs` um filtro FIR Sinc Polifásico customizado (com suporte à Fase Mínima), otimizado por via vetorial com loops const-generics parecidos com a arquitetura do modelo. Esta tarefa possui alta complexidade técnica e requer estudo detalhado antes da implementação.
- **Arquivos-Alvo:** `src/dsp/resampler.rs`
- **Critérios de Aceite:** Remoção drástica na latência final e fase audível mais cristalina e alinhada ao tempo zero; aderência total de `cargo bench` e `cargo test` para bypass planar 48kHz.
- **Perfil do Implementador:** Cientista DSP.
- **Tags:** #dsp #latency #simd
