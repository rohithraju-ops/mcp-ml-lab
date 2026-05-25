from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from pathlib import Path

from mcp_ml_lab.tools import inspect_data_impl, define_task_impl

mcp = FastMCP("mcp-ml-lab")


@mcp.tool()
def inspect_data(csv_path: str) -> dict:
    """Inspect a CSV file: shape, dtypes, null counts, summary stats."""
    return inspect_data_impl(csv_path)


@mcp.tool()
def define_task(
    csv_path: str,
    target_column: str,
    task_type: str = "classification",
    ignore_columns: list[str] | None = None,
    seed: int = 42,) -> dict:
    """Register an ML task in the store and return its task_id."""
    return define_task_impl(
        csv_path=csv_path,
        target_column=target_column,
        task_type=task_type,
        ignore_columns=ignore_columns,
        seed=seed,
    )


def main() -> None:
    """Console-script entrypoint. Runs the MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()

