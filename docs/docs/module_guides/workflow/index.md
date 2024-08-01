# Workflows

A `Workflow` in LlamaIndex is an event-driven abstraction used to chain together several events. Workflows are made up of `steps`, with each step responsible for handling certain event types and emitting new events.

`Workflow`s in LlamaIndex work by decorating function with a `@step()` decorator. This is used to infer the input and output types of each workflow for validation, and ensures each step only runs when an accepted event is ready.

You can create a `Workflow` to do anything! Build an agent, a RAG flow, an extraction flow, or anything else you want.

Workflows are also automatically instrumented, so you get observability into each step using tools like [Arize Pheonix](../observability/index.md#arize-phoenix-local). (**NOTE:** Observability works for integrations that take advantage of the newer instrumentation system. Usage may vary.)

!!! tip
    Workflows make async a first-class citizen, and this page assumes you are running in an async environment. What this means for you is setting up your code for async properly. If you are already running in a server like FastAPI, or in a notebook, you can freely use await already!

    If you are running your own python scripts, its best practice to have a single async entry point.

    ```python
    async def main():
        w = MyWorkflow(...)
        result = await w.run(...)
        print(result)


    if __name__ == "__main__":
        import asyncio

        asyncio.run(main())
    ```

## Getting Started

As an illustrative example, let's consider a naive workflow where a joke is generated and then critiqued.

```python
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)

# `pip install llama-index-llms-openai` if you don't already have it
from llama_index.llms.openai import OpenAI


class JokeEvent(Event):
    joke: str


class JokeFlow(Workflow):
    llm = OpenAI()

    @step()
    async def generate_joke(self, ev: StartEvent) -> JokeEvent:
        topic = ev.get("topic")

        prompt = f"Write your best joke about {topic}."
        response = await self.llm.acomplete(prompt)
        return JokeEvent(joke=str(response))

    @step()
    async def critique_joke(self, ev: JokeEvent) -> StopEvent:
        joke = ev.joke

        prompt = f"Give a thorough analysis and critique of the following joke: {joke}"
        response = await self.llm.acomplete(prompt)
        return StopEvent(result=str(response))


w = JokeFlow(timeout=60, verbose=False)
result = await w.run(topic="pirates")
print(str(result))
```

There's a few moving pieces here, so let's go through this piece by piece.

### Defining Workflow Events

```python
class JokeEvent(Event):
    joke: str
```

Events are user-defined pydantic objects. You control the attributes and any other auxiliary methods. In this case, our workflow relies on a single user-defined event, the `JokeEvent`.

### Setting up the Workflow Class

```python
class JokeFlow(Workflow):
    llm = OpenAI(model="gpt-4o-mini")
    ...
```

Our workflow is implemented by subclassing the `Workflow` class. For simplicity, we attached a static `OpenAI` llm instance.

### Workflow Entry Points

```python
class JokeFlow(Workflow):
    ...

    @step()
    async def generate_joke(self, ev: StartEvent) -> JokeEvent:
        topic = ev.get("topic")

        prompt = f"Write your best joke about {topic}."
        response = await self.llm.acomplete(prompt)
        return JokeEvent(joke=str(response))

    ...
```

Here, we come to the entry-point of our workflow. While events are use-defined, there are two special-case events, the `StartEvent` and the `StopEvent`. Here, the `StartEvent` signifies where to send the initial workflow input. Since the input can be anything, its accessed safely using the `.get()` method.

At this point, you may have noticed that we haven't explicitly told the workflow what events are handled by which steps. Instead, the `@step()` decorator is used to infer the input and output types of each step. Furthermore, these inferred input and output types are also used to verify for you that the workflow is valid before running!

### Workflow Exit Points

```python
class JokeFlow(Workflow):
    ...

    @step()
    async def critique_joke(self, ev: JokeEvent) -> StopEvent:
        joke = ev.joke

        prompt = f"Give a thorough analysis and critique of the following joke: {joke}"
        response = await self.llm.acomplete(prompt)
        return StopEvent(result=str(response))

    ...
```

Here, we have our second, and last step, in the workflow. We know its the last step because the special `StopEvent` is returned. When the workflow encounters a returned `StopEvent`, it immediately stops the workflow and returns whatever the result was.

In this case, the result is a string, but it could be a dictionary, list, or any other object.

### Running the Workflow

```python
w = JokeFlow(timeout=60, verbose=False)
result = await w.run(topic="pirates")
print(str(result))
```

Lastly, we create and run the workflow. There are some settings like timeouts (in seconds) and verbosity to help with debugging.

The `.run()` method is async, so we use await here to wait for the result.

## Drawing the Workflow

Workflows can be visualized, using the power of type annotations in your step definitions. You can either draw all possible paths through the workflow, or the most recent execution, to help with debugging.

```python
from llama_index.core.workflow import (
    draw_all_possible_paths,
    draw_most_recent_execution,
)

# Draw all
draw_all_possible_paths(JokeFlow, filename="joke_flow_all.html")

# Draw an execution
w = JokeFlow()
await w.run(topic="Pirates")
draw_most_recent_execution(w, filename="joke_flow_recent.html")
```

## Working with Global Context/State

Optionally, you can choose to use global context between steps. For example, maybe multiple steps access the original `query` input from the user. You can store this in global context so that every step has access.

```python
from llama_index.core.workflow import Context


@step(pass_context=True)
async def query(self, ctx: Context, ev: QueryEvent) -> StopEvent:
    # retrieve from context
    query = ctx.data.get("query")

    # store in context
    ctx["key"] = "val"

    result = run_query(query)
    return StopEvent(result=result)
```

## Waiting for Multiple Events

The context does more than just hold data, it also provides utilities to buffer and wait for multiple events.

For example, you might have a step like:

```python
@step(pass_context=True)
async def query(
    self, ctx: Context, ev: QueryEvent | RetrieveEvent
) -> StopEvent | None:
    data = ctx.collect_events(evm[QueryEvent, RetrieveEvent])
    # check if we can run
    if data is None:
        return None

    # use buffered events
    print(data[0])  # QueryEvent
    print(data[1])  # RetrieveEvent
```

Using `ctx.collect_events()` we can buffer and wait for ALL expected events to arrive. This function will only return data (in the requested order) once all events have arrived.

## Stepwise Execution

Workflows have built-in utilities for stepwise execution, allowing you to control execution and debug state as things progress.

```python
w = JokeFlow(...)

# Kick off the workflow
w.run_step(topic="Pirates")

# Iterate until done
while not w.is_done():
    w.run_step()

# Get the final result
result = w.get_result()
```

## Decorating non-class Functions

You can also decorate and attach steps to a workflow without subclassing it.

Below is the `JokeFlow` from earlier, but defined without subclassing.

```python
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)


class JokeEvent(Event):
    joke: str


joke_flow = Workflow(timeout=60, verbose=True)


@step(workflow=joke_flow)
async def generate_joke(ev: StartEvent) -> JokeEvent:
    topic = ev.get("topic")

    prompt = f"Write your best joke about {topic}."
    response = await llm.acomplete(prompt)
    return JokeEvent(joke=str(response))


@step(workflow=joke_flow)
async def critique_joke(ev: JokeEvent) -> StopEvent:
    joke = ev.joke

    prompt = (
        f"Give a thorough analysis and critique of the following joke: {joke}"
    )
    response = await llm.acomplete(prompt)
    return StopEvent(result=str(response))
```

## Examples

You can find many useful examples of using workflows in the notebooks below:

- [RAG + Reranking](../../examples/workflow/rag.ipynb)
- [Reliable Structured Generation](../../examples/workflow/reflection.ipynb)
- [Function Calling Agent](../../examples/workflow/function_calling_agent.ipynb)
- [ReAct Agent](../../examples/workflow/react_agent.ipynb)
