# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Sorter Environment.

This module creates an HTTP server that exposes the SorterEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from models import SorterAction, SorterObservation
    from server.sorter_environment import SorterEnvironment
except ImportError:
    from ..models import SorterAction, SorterObservation
    from .sorter_environment import SorterEnvironment


# Create the app with web interface and README integration
app = create_app(
    SorterEnvironment,
    SorterAction,
    SorterObservation,
    env_name="sorter",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """
    Start the FastAPI server with the provided host and port.

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn sorter.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


def main():
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m sorter.server.app
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
