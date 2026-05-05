---
trigger: glob
description: Diretrizes mandatórias de garantia de qualidade (Linting) para o encerramento das submissões da IA.
globs: **/*
---

# Qualidade e Linting ao Fim das Atividades

1. **Regra de Conclusão**: Ao término de *qualquer atividade* executada por você (a IA) que crie ou altere arquivos Rust (pode ignorar se não foi editado nenhum arquivo .rs), você jamais deve considerar a tarefa encerrada ou reportar término definitivo ao usuário sem antes realizar a validação final.
2. **Documentação Atualizada:** Verifique se houve alguma alteração arquitetural relevante. Acione a skill `documentador` se for o caso.
3. **Compilação sem Erros**: É salutar, em momentos oportunos no decorrer da ativade, a execução de `cargo check` e de `cargo build` para descobir problemas logo cedo.
4. **Validação Obrigatória**: Como antepenúltima fase obrigatória, sempre execute os scripts de lint `utils/lints.sh`.
   * SE - E SOMENTE SE - algum arquivo rust .rs tiver sido alterado, rode as verificações `cargo test` para assegurar nenhuma quebra de funcionalidade.
   * SE - E SOMENTE SE - algum arquivo rust .rs tiver sido alterado com OBJETIVO DE GANHO DE PERFORMANCE, rode `cargo bench` para averiguar se houve ganho - ou ao menos não houve perda.
5. **Correção Exaustiva**: Analise o resultado de cada fase e só passe para a seguinte quando aquela passar sem quaisquer quebras, erros, warnings ou "mensagens suspeitas" de qualquer tipo.
6. **Ciclo de Ajustes**: Identifique a fonte dos problemas apresentados (skill `debugger`), realize as correções necessárias no código e reexecute o passo que apontou o erro. Só prossiga se a checagem passar com status de sucesso absoluto.
7. **Higiene do Repositório**: E ao final de toda atividade, verifique se não há arquivos temporários, de log, ou artefatos de debug não listados no `.gitignore` que possam ter sido gerados durante a execução (ex: `console.log`, dumps, `.tmp`). Não polua o histórico git com artefatos de trabalho.
