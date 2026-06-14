---
name: tarefa
description: Technical Task skill as per /TODO-sprints.md
---

# Skill: Tarefa Técnica

## When to use this skill

When the user requests the execution of a task, sprint, or epic as per `/TODO-sprints.md`.

## Instructions

* If a "task" is provided (Examples: "Task 1.2" or "[T9]"): The goal is the implementation of this task as explained in its description. Use the `implementador` skill to execute it.
* If a "sprint" or "epic" is provided (Examples: "Sprint 1" or "Epic 3"): The goal is to review/audit the entire sprint/epic to ensure that all micro and macro objectives of that sprint (as well as its tasks) have been exemplarily fulfilled.
  * Pay attention to what may have slipped through (from the initial sprint plan, to what was identified later), understanding the "spirit" of that sprint/epic.
  * In this case, trigger the `planejador-arquiteto` skill to plan the execution of what was identified.
* Know the context of the sprints, epics and whole TODO-sprints.md file the task is inserted. Make informed decisions.
* Considering the growing number of tests and benchmarks, whenever possible run only those directly involved in what is being done.
* At the end of completing activities, if there is important information impacting other activities planned for later, leave a note in the most appropriate location in /TODO-sprints.md.
* If, at the end of the general sprint review, relevant improvement notes are identified — too large to be resolved in the current activity — it is allowed to improve the tasks and/or sprints subsequent to the one being reviewed. Or even add new sprint(s) and task(s) at the end.
* Conclude by proposing a one-line git message summarizing what was done.
