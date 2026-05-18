---
description: Tarefa Técnica conforme /TODO-sprints.md
---

# Automatizador de Tarefa Técnica conforme /TODO-sprints.md (caso não exista, crie-o)

* Se for fornecido uma "tarefa" (Exemplos: "Tarefa 1.2" ou "[T9]"): O objetivo é a implementação desta tarefa conforme explicado na sua descrição. Use a skill `implementador` para executá-la.
* Se for fornecido uma "sprint" ou "épico" (Exemplos: "Sprint 1" ou "Épico 3"): O objetivo é passar em revista/auditoria todo a sprint/épico para assegurar que todos os objetivos micro e macro daquela sprint (bem como suas tarefas) foram cumpridos exemplarmente.
  * Atentar ao que pode ter passado batido (desde o plano inicial das sprint, até o que foi identificado posteriormente), entendendo o "espirito" daquela sprint/épico.
  * Neste caso, acionar a skill `planejador-arquiteto` para planejar a execução do que for identificado.
* Considerando o número crescente de testes e benchmarks, sempre que possível rode apenas os diretamente envolvidos no que está sendo feito.
* Ao final da conclusão das atividades, se houver informações importantes de impacto em outras atividades previstas para adiante, deixe anotado no local mais adequado do /TODO-sprints.md.
* Se, ao final da revisão geral da sprint, forem identificados apontamentos relevantes de melhorias - grandes demais para ser resolvido na atividade atual - é permitido a melhoria das tarefas e/ou sprints posteriores à que está sendo revisada. Ou mesmo a adição de nova(s) sprint(s) e tarefa(s) ao final.
* Conclua propondo um texto para o git message de uma linha resumindo o que foi feito.
