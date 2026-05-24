from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from mcp_ml_lab.tools import inspect_data_impl

mcp = FastMCP("mcp-ml-lab")


@mcp.tool()
def inspect_data(csv_path: str) -> dict:
    """Inspect a CSV file: shape, dtypes, null counts, summary stats."""
    return inspect_data_impl(csv_path)


def main() -> None:
    """Console-script entrypoint. Runs the MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()