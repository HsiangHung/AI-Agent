# AI Agents

## Definition 

Agent = LLM + Planning + Memory + Tool


## ReAct (Reasoning + Action)

Iterative process of the follows:
1. Thought: According state to reason next step
2. Action: Ask or call tools
3. Observation: Get feedback and update state

Thought 1 -> Action 1 -> Observation 1 -> Thought 2 -> Action 2 -> Observation 2 -> ...


## Reflect

```Python
while not task_complete:
    result = agent.execute(task)
    is_passed, feedback = audit_agent.verify(result, standard)
    if not is_passed:
        task = task + f"feedback：{feedback}"
```
