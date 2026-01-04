#!/usr/bin/env python3
"""
MCP Server: NanoBanana Image Generation via Vertex AI

This server provides a tool to generate images using a custom
"nanobanana" model hosted on Google Cloud Vertex AI.

Tools:
  - nanobanana_generate(prompt: str, ...)
"""

import os
import sys
import time, asyncio
import mimetypes
from pathlib import Path
from typing import Dict, Any
from fastmcp import FastMCP, Context
#from mcp.server.fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from google import genai
from google.genai import types as gtypes
from fastmcp.utilities.logging import configure_logging, get_logger

# ---------- Logging ----------
# Set log level from LOG_LEVEL env var, defaulting to INFO
log_level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
if log_level_name not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
    log_level_name = "INFO"

configure_logging(log_level_name)
logger = get_logger(__name__)

log_file = os.environ.get("LOG_FILE", "gemini_media_gen.log")
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s"))
logger.addHandler(file_handler)

logger.info("--- Log level set to %s ---", log_level_name)
logger.info("--- Logging to file: %s ---", log_file)

def print_environment_variables():
    """Logs relevant GOOGLE_* environment variables for debugging."""
    logger.debug("--- Google Environment Variables ---")
    for var, value in os.environ.items():
        if var.startswith("GOOGLE_"):
            if "API_KEY" in var and value:
                logger.debug(f"{var}: {'*' * 8}")
            else:
                logger.debug(f"{var}: {value}")
    logger.debug("----------------------------------\n")

from dotenv import load_dotenv
# ---------- Load environment variables ----------
# First load local .env which might contain DOTENV_PATH
load_dotenv()

dotenv_path_from_env = os.environ.get("DOTENV_PATH")
if dotenv_path_from_env:
    # Expand ~ if present
    dotenv_path_from_env = os.path.expanduser(dotenv_path_from_env)
    if os.path.exists(dotenv_path_from_env):
        logger.info(f"--- Loading .env file from redirect: {dotenv_path_from_env} ---")
        load_dotenv(dotenv_path=dotenv_path_from_env, override=True)
print_environment_variables()

# ---------- MCP server ----------
mcp = FastMCP("Gemini Media MCP")
logger.info("start gemini media mcp server...")

def _initialize_vertex_client_and_paths(out_dir: str) -> tuple[genai.Client, Path]:
    """
    Initializes and returns the Vertex AI client and the output directory path.

    Raises:
        ToolError: If Vertex AI environment variables are not set.
    """
    project = os.environ.get("VERTEXAI_PROJECT")
    location = os.environ.get("VERTEXAI_LOCATION")
    if not project or not location:
        raise ToolError("Missing Vertex AI configuration. Please set the GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION environment variables.")

    client = genai.Client(vertexai=True, project=project, location=location)
    out_dir_p = Path(os.path.expanduser(out_dir))
    out_dir_p.mkdir(parents=True, exist_ok=True)
    return client, out_dir_p

@mcp.tool()
async def nanobanana_generate(
    prompt: str,
    input_paths: str | None = None,
    out_dir: str = ".",
    model: str = "gemini-2.5-flash-image", # Your custom model
    n: int = 1,
    ctx: Context = None,
) -> Dict[str, Any]:
    """
    Generate image(s) from a text prompt, optionally guided by input image(s),
    using a model on Vertex AI. Saves files to out_dir and returns their paths.

    Args:
      prompt: Text instruction for the model.
      input_paths: Optional comma-delimited string of image file paths for image-to-image generation.
      out_dir: Directory to write generated files.
      model: The image generation model to use.
      n: Desired number of images (best-effort; stream may emit 1+).
    """
    client, out_dir_p = _initialize_vertex_client_and_paths(out_dir)

    # Build parts: text + optional input images
    parts: list[gtypes.Part] = [gtypes.Part.from_text(text=prompt)]
    path_list: list[str] = []
    if input_paths:
        path_list = [p.strip() for p in input_paths.split(',') if p.strip()]

    for p in path_list:
        try:
            with open(p, "rb") as f:
                data = f.read()
            mt, _ = mimetypes.guess_type(p)
            if not mt or not mt.startswith("image/"):
                mt = "image/jpeg"  # Default to jpeg if type is unknown/not an image
            parts.append(gtypes.Part.from_bytes(data=data, mime_type=mt))
        except Exception as e:
            raise ToolError(f"Failed to read input image '{p}': {e}")

    contents = [gtypes.Content(role="user", parts=parts)]
    config = gtypes.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"])

    saved: list[str] = []
    texts: list[str] = []
    try:
        if ctx:
            await ctx.info(f"Starting image generation with model {model}...")
            await ctx.report_progress(0, n)
            
        stream = client.models.generate_content_stream(model=model, contents=contents, config=config)
        for i, chunk in enumerate(stream):
            if not getattr(chunk, "candidates", None): continue
            part = chunk.candidates[0].content.parts[0]
            if getattr(chunk, "text", None): texts.append(chunk.text)
            
            inline = getattr(part, "inline_data", None)
            if inline and getattr(inline, "data", None):
                ext = mimetypes.guess_extension(getattr(inline, "mime_type", "image/png")) or ".png"
                fname = f"nanobanana_{time.strftime('%Y%m%d_%H%M%S')}_{int((time.time() % 1) * 1000):03d}_{i:02d}{ext}"
                fpath = out_dir_p / fname
                with open(fpath, "wb") as f: f.write(inline.data)
                saved.append(str(fpath))
                logger.info("NanoBanana (Vertex AI) saved: %s", fpath)
                if n > 0 and len(saved) >= n:
                    if ctx:
                        await ctx.report_progress(len(saved), n)
                    break
                if ctx:
                    await ctx.report_progress(len(saved), n)
    except Exception as e:
        logger.error("Vertex AI generation failed: %s", e, exc_info=True)
        raise ToolError(f"Vertex AI generation failed: {e}")

    return {
        "ok": True,
        "paths": saved,
        "text": "\n".join(texts).strip(),
        "model": model,
        "backend": "VertexAI",
    }

@mcp.tool()
async def veo_generate_video(
    prompt: str,
    input_paths: str | None = None,
    out_dir: str = ".",
    model: str = "veo-3.1-fast-generate-preview", # Your custom model
    poll_seconds: int = 10,
    max_wait_seconds: int = 900,
    ctx: Context = None,
) -> Dict[str, Any]:
    """
    Generate video(s) from a text prompt, optionally guided by input image(s),
    using a Veo model on Vertex AI. This is a long-running operation.

    Args:
      prompt: Text instruction for the model.
      input_paths: Optional comma-delimited string of image file paths for image-to-video generation.
      out_dir: Directory to write generated files.
      model: The video generation model to use.
      poll_seconds: How often to check the status of the generation job.
      max_wait_seconds: Maximum time to wait for the video to be generated.
    """
    client, out_dir_p = _initialize_vertex_client_and_paths(out_dir)

    image_obj = None
    path_list: list[str] = []
    if input_paths:
        # Veo on Vertex currently supports one input image
        path_list = [p.strip() for p in input_paths.split(",") if p.strip()]
        if len(path_list) > 1:
            logger.warning(f"Veo only supports one input image, using the first one: {path_list[0]}")
        image_path = path_list[0]
        try:
            with open(image_path, "rb") as f:
                data = f.read()
            mt, _ = mimetypes.guess_type(image_path)
            image_obj = gtypes.Image(image_bytes=data, mime_type=mt or "image/png")
        except Exception as e:
            raise ToolError(f"Failed to read input image '{image_path}': {e}")

    try:
        if ctx:
            await ctx.info(f"Submitting Veo generation request (model: {model})...")
            
        op = client.models.generate_videos(model=model, prompt=prompt, image=image_obj)
        logger.info(f"Started Veo generation (operation: {op.name}). Waiting for completion...")
        
        if ctx:
            await ctx.info(f"Veo generation started. Operation ID: {op.name}")
    except Exception as e:
        raise ToolError(f"Failed to start Veo generation: {e}")

    waited = 0
    while not op.done:
        if waited >= max_wait_seconds:
            raise ToolError(f"Timeout waiting for Veo generation after {max_wait_seconds}s")
        
        if ctx:
            # Veo generation doesn't give precise progress, so we report based on time or just an indeterminate "working"
            await ctx.info(f"Waiting for video generation... ({waited}s elapsed)")
            await ctx.report_progress(waited, max_wait_seconds)

        await asyncio.sleep(poll_seconds)
        waited += poll_seconds
        logger.info(f"Polling Veo operation... (waited {waited}s)")
        op = client.operations.get(op)

    if not op.result or not op.result.generated_videos:
        raise ToolError("Veo generation finished but returned no video.")

    # Save the first generated video
    video_file = op.result.generated_videos[0].video
    fname = f"veo_{time.strftime('%Y%m%d_%H%M%S')}_{int((time.time() % 1) * 1000):03d}.mp4"
    fpath = out_dir_p / fname
    video_file.save(str(fpath))
    logger.info(f"Veo video saved to: {fpath}")

    if ctx:
        await ctx.report_progress(max_wait_seconds, max_wait_seconds)
        await ctx.info(f"Video generation complete. Saved to {fpath}")

    return {"ok": True, "path": str(fpath), "model": model, "backend": "VertexAI"}

if __name__ == "__main__":
    mcp.run()
