#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Callable

import uvicorn
from mcp.server import Server
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Mount, Route
from starlette.types import Receive, Scope, Send


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None
if load_dotenv is not None:
    load_dotenv(REPO_ROOT / ".env")

DEFAULT_SEGMENTATION_CKPT = SRC_DIR / "checkpoints" / "segmentation.ckpt"
DEFAULT_SIMCLR_CKPT = SRC_DIR / "checkpoints" / "BrainIAC.ckpt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "inference" / "mcp_outputs"
DEFAULT_HTTP_HOST = "127.0.0.1"
DEFAULT_HTTP_PORT = 8001


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    else:
        path = path.resolve()
    return path


SEGMENTATION_CKPT = _resolve_path(os.environ.get("BRAINIAC_SEGMENTATION_CKPT", str(DEFAULT_SEGMENTATION_CKPT)))
SIMCLR_CKPT = _resolve_path(os.environ.get("BRAINIAC_SIMCLR_CKPT", str(DEFAULT_SIMCLR_CKPT)))
OUTPUT_DIR = _resolve_path(os.environ.get("BRAINIAC_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
GPU_DEVICE = os.environ.get("BRAINIAC_GPU_DEVICE", "0")
HTTP_HOST = os.environ.get("BRAINIAC_MCP_HOST", DEFAULT_HTTP_HOST)
HTTP_PORT = int(os.environ.get("BRAINIAC_MCP_PORT", str(DEFAULT_HTTP_PORT)))

# Set device visibility before any CUDA model/tensor is created.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", GPU_DEVICE)


mcp = FastMCP(
    name="brainiac-infarct-segmentation",
    instructions=(
        "Provide an absolute NIfTI path for one patient image and return the absolute "
        "path of the generated infarct segmentation mask."
    ),
)

_MODEL_BUNDLE: tuple[Any, dict[str, Any]] | None = None
_TORCH: Any | None = None
_INFER_FNS: dict[str, Callable[..., Any]] | None = None


def _require_existing_file(path: Path, field_name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{field_name} does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"{field_name} is not a file: {path}")


def _mask_filename(image_path: Path) -> str:
    image_name = image_path.name
    suffix = ""
    stem = image_name
    if image_name.endswith(".nii.gz"):
        stem = image_name[: -len(".nii.gz")]
        suffix = ".nii.gz"
    elif image_name.endswith(".nii"):
        stem = image_name[: -len(".nii")]
        suffix = ".nii"
    else:
        raise ValueError(
            f"Unsupported image format for '{image_path}'. Expected .nii or .nii.gz."
        )

    digest = hashlib.sha1(str(image_path.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{stem}_{digest}_seg{suffix}"


def _load_runtime() -> tuple[Any, dict[str, Callable[..., Any]]]:
    global _TORCH, _INFER_FNS
    if _TORCH is not None and _INFER_FNS is not None:
        return _TORCH, _INFER_FNS

    import torch
    from generate_segmentation import (
        generate_segmentation as _generate_segmentation,
        load_model_for_inference as _load_model_for_inference,
        preprocess_image as _preprocess_image,
        save_segmentation as _save_segmentation,
    )

    _TORCH = torch
    _INFER_FNS = {
        "generate_segmentation": _generate_segmentation,
        "load_model_for_inference": _load_model_for_inference,
        "preprocess_image": _preprocess_image,
        "save_segmentation": _save_segmentation,
    }
    return _TORCH, _INFER_FNS


def _load_or_get_model_bundle() -> tuple[Any, dict[str, Any]]:
    global _MODEL_BUNDLE
    if _MODEL_BUNDLE is not None:
        return _MODEL_BUNDLE

    _require_existing_file(SEGMENTATION_CKPT, "Segmentation checkpoint")
    _require_existing_file(SIMCLR_CKPT, "SimCLR checkpoint")

    torch, infer_fns = _load_runtime()
    checkpoint = torch.load(str(SEGMENTATION_CKPT), map_location="cpu")
    if "hyper_parameters" not in checkpoint or "state_dict" not in checkpoint:
        raise ValueError(
            f"Unexpected checkpoint format: {SEGMENTATION_CKPT}. "
            "Expected keys: 'hyper_parameters' and 'state_dict'."
        )

    config = checkpoint["hyper_parameters"]
    state_dict = checkpoint["state_dict"]
    config.setdefault("pretrain", {})
    config["pretrain"]["simclr_checkpoint_path"] = str(SIMCLR_CKPT)

    model = infer_fns["load_model_for_inference"](config, state_dict)
    _MODEL_BUNDLE = (model, config)
    return _MODEL_BUNDLE


@mcp.tool(
    name="segment_infarct_single_patient",
    description=(
        "Run BrainIAC infarct segmentation for one patient. "
        "Input must be an absolute path to a .nii/.nii.gz image file. "
        "Returns the absolute output mask path."
    ),
)
def segment_infarct_single_patient(patient_image_path: str) -> dict[str, str]:
    image_path = Path(patient_image_path).expanduser()
    if not image_path.is_absolute():
        raise ValueError(
            "patient_image_path must be an absolute path because MCP server and agent "
            "run on the same host."
        )
    _require_existing_file(image_path, "patient_image_path")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _, infer_fns = _load_runtime()
    model, config = _load_or_get_model_bundle()
    image_tensor, meta_dict = infer_fns["preprocess_image"](str(image_path), config)
    segmentation_tensor = infer_fns["generate_segmentation"](model, image_tensor, config)

    output_path = OUTPUT_DIR / _mask_filename(image_path)
    infer_fns["save_segmentation"](segmentation_tensor.cpu(), meta_dict, str(output_path))

    return {
        "input_image_path": str(image_path.resolve()),
        "output_mask_path": str(output_path.resolve()),
    }


def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    sse = SseServerTransport("/messages/")
    session_manager = StreamableHTTPSessionManager(
        app=mcp_server,
        event_store=None,
        json_response=True,
        stateless=True,
    )

    async def handle_sse(request: Request) -> None:
        async with sse.connect_sse(
            request.scope,
            request.receive,
            request._send,
        ) as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options(),
            )

    async def handle_streamable_http(scope: Scope, receive: Receive, send: Send) -> None:
        await session_manager.handle_request(scope, receive, send)

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        async with session_manager.run():
            yield

    return Starlette(
        debug=debug,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/mcp", app=handle_streamable_http),
            Mount("/messages/", app=sse.handle_post_message),
        ],
        lifespan=lifespan,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BrainIAC MCP service for single-patient infarct segmentation."
    )
    parser.add_argument(
        "--http",
        action="store_true",
        help="Run with Streamable HTTP + SSE transport instead of STDIO.",
    )
    parser.add_argument(
        "--sse",
        action="store_true",
        help="Deprecated alias of --http.",
    )
    parser.add_argument(
        "--host",
        default=None,
        help=f"Host for HTTP mode (default from BRAINIAC_MCP_HOST or {DEFAULT_HTTP_HOST}).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"Port for HTTP mode (default from BRAINIAC_MCP_PORT or {DEFAULT_HTTP_PORT}).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Starlette debug mode in HTTP mode.",
    )
    args = parser.parse_args()

    use_http = args.http or args.sse
    if not use_http and (args.host is not None or args.port is not None):
        parser.error("Host/port are only valid with --http (or --sse).")
    return args


def parse_args() -> argparse.Namespace:
    # Backward compatible wrapper for previous function name.
    return _parse_args()


def main() -> None:
    args = parse_args()
    use_http = args.http or args.sse
    if use_http:
        host = args.host if args.host else HTTP_HOST
        port = args.port if args.port is not None else HTTP_PORT
        starlette_app = create_starlette_app(mcp._mcp_server, debug=args.debug)
        uvicorn.run(starlette_app, host=host, port=port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
