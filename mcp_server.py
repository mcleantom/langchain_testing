from typing import List, Literal, Annotated
from mcp.server.fastmcp import FastMCP
from langchain_experimental.utilities import PythonREPL
from pydantic import BaseModel


mcp = FastMCP("Math")


repl = PythonREPL()


class PythonCodeResult(BaseModel):
    result: Literal["success", "failure"]
    code: str
    stdout: str
    errors: str


@mcp.tool()
def python(code: Annotated[str, "The python code to execute"]) -> PythonCodeResult:
    """Run Python Code. Use a print statement to get the result of a variable to stdout"""
    try:
        result = repl.run(code)
    except BaseException as e:
        return PythonCodeResult(
            result="failure",
            code=code,
            stdout="",
            errors=f"{repr(e)}"
        )
    return PythonCodeResult(
        result="success",
        code=code,
        stdout=result,
        errors=""
    )


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@mcp.tool()
def get_weather(location: str) -> str:
    """Multiply two numbers"""
    return f"Its always sunny in {location}"


if __name__ == "__main__":
    mcp.run(transport="sse")
