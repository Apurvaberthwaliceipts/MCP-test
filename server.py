# server.py
from fastmcp import FastMCP

mcp = FastMCP("HTTP Tools")

@mcp.tool()
def hello(name: str) -> str:
    "Greets by name."
    return f"Hello, {name}!"

@mcp.tool()
def add(a: int, b: int) -> int:
    "Adds two numbers."
    return a + b

if __name__ == "__main__":
    # Start HTTP server at http://127.0.0.1:8000/mcp
    mcp.run(transport="http", host="127.0.0.1", port=8005, path="/mcp")
