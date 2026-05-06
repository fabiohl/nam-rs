# Planos para próximas versões

* Épico "Cold Review"
Olhar clínico e revisão sistemática da pasta "src/". "Código bom e bem escrito".
Código organizado, limpo e inteligível. Buscar problemas (bugs, segurança e performance) potenciais. Remover código morto, inlining, cold/hot, fazer mais com menos, etc. Capriche mesmo!
Ir metódicamente lendo linha a linha, arquivo a arquivo, para realizar uma análise completa.
Focar nas áreas de infra, sem entrar na parte de DSP propriamente dita. Exs: /, /loader, /dsp.
No primeiro estágio (criar o plano de implementação) foque apenas em identificar os pontos de melhoria para review e aprovação de continuidade.
Aprovado o plano de implementação, o entregável final é alimentar o TODO-sprints.md com as tarefas técnicas.

* Épico "Hot Review"
Focar nas áreas centrais e críticas de processamento em tempo real. Exs: /models, /math.

* Épico "Test Review"
Olhar clínico e revisão sistemática das pastas "tests/", "fuzz/" e "benchs/". "Nenhum bug passa batido".
Assegurar que a cobertura de testes e de benchmarks está completa e não deixa nenhum setor do nam-rs desprotegido.

* Suporte completo a modo plugin CLAP
  * Benchmarking comparativo de overhead entre Standalone e Plugin
  * Interface Gráfica (GUI) minimalista para controle de parâmetros
* Finalização da Arquitetura "A2" (Acompanhar trabalho do Steven Atkinson)
* Tradução completa para Inglês Internacional
