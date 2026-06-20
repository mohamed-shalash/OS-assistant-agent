import subprocess
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun


search_tool = DuckDuckGoSearchRun()

@tool
def create_file(path: str, content: str) -> str:
    """Create a file and write content to it."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Created file {path}"



@tool
def execute_list_of_commands(commands: list[str]) -> str:
    """Execute shell commands and return outputs."""

    outputs = []

    for cmd in commands:
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )

            outputs.append(
                f"""
Command: {cmd}

Exit Code: {result.returncode}

STDOUT:
{result.stdout}

STDERR:
{result.stderr}
"""
            )

        except subprocess.TimeoutExpired:
            outputs.append(
                f"Command: {cmd}\nError: Command timed out"
            )

        except Exception as e:
            outputs.append(
                f"Command: {cmd}\nError: {e}"
            )

    return "\n".join(outputs)



def get_tools():
    tools = [create_file, search_tool, execute_list_of_commands]
    return tools