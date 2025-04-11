from langchain_anthropic import ChatAnthropic
from langgraph.graph import MessagesState, END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_experimental.utilities import PythonREPL
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langchain_core.tools import tool
import json
import os
from influxdb import InfluxDBClient
from langgraph.checkpoint.memory import MemorySaver
from uuid import uuid4
from pathlib import Path
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI


INFLUXDB_HOST = os.environ.get("INFLUXDB_HOST")
INFLUXDB_PORT = 8086
INFLUXDB_USER = os.environ.get("INFLUXDB_USER")
INFLUXDB_PASSWORD = os.environ.get("INFLUXDB_PASSWORD")
influx_client = InfluxDBClient(
    host=INFLUXDB_HOST,
    username=INFLUXDB_USER,
    password=INFLUXDB_PASSWORD,
    ssl=True,
    database="SailGP"
)
INFLUX_QUERY_OUTPUTS = Path(__file__).parent / "outputs" / "influx_query_outputs"
INFLUX_QUERY_OUTPUTS.mkdir(exist_ok=True)


@tool
def influxdb_tool(
        query: Annotated[str, "The InfluxQL query to run."]
):
    """
    Run an InfluxQL query against an InfluxDB v1.x database.
    """
    try:
        result = influx_client.query(query)
        points = list(result.get_points())
        out_file = INFLUX_QUERY_OUTPUTS / f"{uuid4()}.txt"
        out_file.write_text(json.dumps(points, indent=2))
        return json.dumps(points, indent=2)
        # return f"Saved influx output to {out_file}"
    except Exception as e:
        return f"Query failed: {str(e)}"


repl = PythonREPL()


@tool
def python_repl_tool(
        code: Annotated[str, "The python code to execute"],
):
    """
    Use this to execute python code and do math. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user.
    """
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result = {
        "result": "success",
        "code": code,
        "stdout": {result}
    }


members = ["influx", "coder"]
options = members + ["FINISH"]

system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    f" following workers: {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a",
    " task and respond with their results and status. When finished"
    " respond with FINISH"
)


class Router(TypedDict):
    next: Literal[*options]


rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.1,
    check_every_n_seconds=0.1,
    max_bucket_size=10
)
llm = ChatOpenAI(model="gpt-4o-mini", rate_limiter=rate_limiter)


class State(MessagesState):
    next: str


def supervisor_node(state: State) -> Command[Literal[*members, "__end__"]]:
    messages = [
        SystemMessage(content=system_prompt)
    ] + state["messages"]

    response = llm.with_structured_output(Router).invoke(messages)
    goto = response["next"]
    if goto == "FINISH":
        goto = END
    return Command(goto=goto, update={"next": goto})


coder_agent = create_react_agent(llm, tools=[python_repl_tool])


def code_node(state: State) -> Command[Literal["supervisor"]]:
    result = coder_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][1].content, name="coder")
            ]
        }
    )


influx_agent = create_react_agent(llm, tools=[influxdb_tool])


def influx_node(state: State) -> Command[Literal["supervisor"]]:
    result = influx_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][1].content, name="influx")
            ]
        }
    )


workflow = StateGraph(MessagesState)
workflow.add_edge(START, "supervisor")
workflow.add_node("supervisor",  supervisor_node)
workflow.add_node("coder", code_node)
workflow.add_node("influx", influx_node)
checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)


config = {"configurable": {"thread_id": 1}}


def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": ["user", user_input]}, config=config, subgraphs=True):
        message = event[1]
        if "agent" in message:
            print(message["agent"]["messages"])


def main():
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)

    for s in graph.stream(
            {"messages": [("user", "What boats are available in the database?")]}, subgraphs=True
    ):
        print(s)
        print("----")


if __name__ == "__main__":
    main()
