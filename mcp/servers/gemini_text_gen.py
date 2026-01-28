#!/usr/bin/env python3
"""
MCP Server: Gemini Text Generation via Google AI Studio or Vertex AI

This server provides a tool to generate text using Gemini models hosted on Google AI Studio or Google Cloud Vertex AI.

Tools:
  - gemini_generate_text(prompt: str, ...)
"""

import os
import sys
import logging
import mimetypes
import json
from typing import Dict, Any
from fastmcp import FastMCP, Context
from fastmcp.exceptions import ToolError
from google import genai
from google.genai import types as gtypes
from fastmcp.utilities.logging import configure_logging, get_logger
from dotenv import load_dotenv

# ---------- Logging ----------
# Set log level from LOG_LEVEL env var, defaulting to INFO
log_level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
if log_level_name not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
    log_level_name = "INFO"
configure_logging(log_level_name)
logger = get_logger(__name__)

# Force timestamp on console/stream handlers configured by fastmcp
formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
for handler in logging.getLogger().handlers:
    handler.setFormatter(formatter)

log_file = os.environ.get("LOG_FILE", "gemini_text_gen.log")
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

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

# Check and log configuration at startup
if os.environ.get("GOOGLE_API_KEY"):
    logger.info("Startup: Configured for Google AI Studio (GOOGLE_API_KEY detected).")
elif os.environ.get("VERTEXAI_PROJECT") and os.environ.get("VERTEXAI_LOCATION"):
    logger.info(f"Startup: Configured for Vertex AI (Project: {os.environ.get('VERTEXAI_PROJECT')}, Location: {os.environ.get('VERTEXAI_LOCATION')}).")
else:
    logger.warning("Startup: No valid configuration found for Google AI Studio or Vertex AI.")

# ---------- Pricing & Usage Helpers ----------
import litellm
import io
import pypdf

def _get_pdf_metadata(data: bytes, source_name: str) -> Dict[str, Any]:
    """
    Extracts page count and size from PDF bytes.
    """
    size_mb = len(data) / (1024 * 1024)
    page_count = 0
    
    try:
        f = io.BytesIO(data)
        reader = pypdf.PdfReader(f)
        page_count = len(reader.pages)
    except Exception as e:
        logger.warning(f"Failed to count pages for {source_name}: {e}")
        
    return {
        "file_size_mb": round(size_mb, 2),
        "page_count": page_count
    }

def _extract_usage_dict(usage_meta: Any, model: str) -> Dict[str, Any]:
    """Helper to extract all available token counts from usage metadata."""
    if not usage_meta:
        return {
            "prompt_token_count": 0,
            "candidates_token_count": 0,
            "total_token_count": 0,
            "estimated_cost_usd": 0.0
        }
    
    data = {}
    
    # Dynamically extract all attributes ending in '_token_count'
    # This covers standard fields plus 'cached_content_token_count', 'thought_token_count', etc.
    for attr in dir(usage_meta):
        if attr.endswith("_token_count") and not attr.startswith("_"):
            try:
                val = getattr(usage_meta, attr)
                if isinstance(val, int):
                    data[attr] = val
            except Exception:
                pass
                
    # Ensure vital fields exist (default to 0)
    for key in ["prompt_token_count", "candidates_token_count", "total_token_count"]:
        if key not in data:
            data[key] = 0
            
    # Calculate Cost
    data["estimated_cost_usd"] = calculate_cost(
        model, 
        data["prompt_token_count"], 
        data["candidates_token_count"]
    )
    
    return data


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate estimated cost in USD based on model and token counts using LiteLLM.
    """
    try:
        # litellm.cost_per_token returns cost or raises error / returns 0 if unknown
        cost, _ = litellm.cost_per_token(model=model, prompt_tokens=input_tokens, completion_tokens=output_tokens)
        return cost
    except Exception as e:
        logger.warning(f"LiteLLM cost calculation failed for {model}: {e}")
        return 0.0





def format_tool_error(error_msg: str, error_code: str = "TOOL_ERROR", details: Dict[str, Any] = None) -> str:
    """Returns a structured JSON error string."""
    return json.dumps({
        "error": error_msg,
        "error_code": error_code,
        "details": details or {}
    })

# ---------- MCP server ----------
mcp = FastMCP("Gemini Text MCP")
logger.info("start gemini text mcp server...")

def _get_client() -> genai.Client:
    """
    Initializes and returns the GenAI client (AI Studio or Vertex AI).

    Raises:
        ToolError: If neither API key nor Vertex AI environment variables are set.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key:
        logger.info("Using Google AI Studio environment (GOOGLE_API_KEY detected).")
        return genai.Client(api_key=api_key)

    project = os.environ.get("VERTEXAI_PROJECT")
    location = os.environ.get("VERTEXAI_LOCATION")
    if project and location:
        logger.info(f"Using Vertex AI environment (Project: {project}, Location: {location}).")
        return genai.Client(vertexai=True, project=project, location=location)

    raise ToolError("Missing configuration. Please set GOOGLE_API_KEY for AI Studio, or VERTEXAI_PROJECT and VERTEXAI_LOCATION for Vertex AI.")

def format_grading_report(data: dict) -> str:
    """
    Parses the JSON grading data and returns a formatted Markdown report.
    """
    md = f"# Exam Grading Report\n\n"
    total_awarded = data.get('total_marks', 0)
    total_available = data.get('total_available_marks', '?')
    md += f"**Total Marks:** {total_awarded} / {total_available}\n\n"
    
    summary = data.get("summary_comments")
    if summary:
        md += f"## Summary Comments\n{summary}\n\n"
    
    questions = data.get("question_breakdown", [])
    if questions:
        md += "## Question Breakdown\n\n"
        for q in questions:
            q_id = q.get('question_id', 'Unknown')
            marks = q.get('marks_awarded', 0)
            max_marks = q.get('max_marks_available', '?')
            md += f"### Question {q_id} ({marks}/{max_marks} marks)\n\n"
            
            reasoning = q.get('reasoning')
            if reasoning:
                md += f"**Reasoning:** {reasoning}\n\n"
            
            citations = q.get('citations', [])
            if citations:
                md += "**Citations:**\n"
                for c in citations:
                    source_text = c.get('source_text', '')
                    source_type = c.get('source_type', 'unknown')
                    md += f"- *\"{source_text}\"* ({source_type})\n"
                md += "\n"
            md += "---\n\n"
            
    return md

import requests

def _load_file_parts(input_paths: str | None) -> list[gtypes.Part]:
    parts = []
    if input_paths:
        for p in [p.strip() for p in input_paths.split(',') if p.strip()]:
            try:
                # Determine MIME type
                mt, _ = mimetypes.guess_type(p)
                if not mt:
                    mt = "application/pdf" if p.lower().endswith(".pdf") else "application/octet-stream"

                if p.startswith("gs://"):
                    # For GCS objects, use the URI directly
                    parts.append(gtypes.Part.from_uri(file_uri=p, mime_type=mt))
                elif p.startswith("http://") or p.startswith("https://"):
                    # For HTTP(S) URLs, download the content
                    logger.info(f"Downloading file from URL: {p}")
                    response = requests.get(p)
                    response.raise_for_status()
                    
                    # Use Content-Type from header if available, otherwise fall back to guessed type
                    content_type = response.headers.get("Content-Type")
                    if content_type:
                        # Strip parameters (e.g. "application/pdf; charset=utf-8" -> "application/pdf")
                        mt = content_type.split(";")[0].strip()
                    
                    parts.append(gtypes.Part.from_bytes(data=response.content, mime_type=mt))
                else:
                    # For local files, read the bytes
                    with open(p, "rb") as f:
                        data = f.read()
                    parts.append(gtypes.Part.from_bytes(data=data, mime_type=mt))
            except Exception as e:
                raise ToolError(f"Failed to load input file '{p}': {e}")
    return parts

@mcp.tool()
async def gemini_generate_text(
    prompt: str,
    input_paths: str | None = None,
    model: str = "gemini-3-flash-preview",
    system_instruction: str | None = None,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
    response_mime_type: str | None = None,
    response_schema: str | None = None,
    ctx: Context = None,
) -> str:
    """
    Generate text from a prompt using a Gemini model.

    Args:
      prompt: The input text prompt.
      input_paths: Optional comma-delimited string of file paths (e.g. PDFs, images) to include.
      model: The model to use (default: gemini-3-flash-preview).
      system_instruction: Optional system instruction to guide the model's behavior.
      temperature: Controls randomness in output (0.0 to 2.0).
      max_output_tokens: Maximum number of tokens to generate.
      response_mime_type: Optional MIME type for the response (e.g., "application/json").
      response_schema: Optional JSON string schema for controlled generation.
    """
    try:
        client = _get_client()
        
        config_args = {}
        if temperature is not None:
            config_args["temperature"] = temperature
        if max_output_tokens is not None:
            config_args["max_output_tokens"] = max_output_tokens
        if system_instruction:
            config_args["system_instruction"] = system_instruction
        if response_mime_type:
            config_args["response_mime_type"] = response_mime_type

        if response_schema:
            try:
                config_args["response_schema"] = json.loads(response_schema)
                if not response_mime_type:
                    config_args["response_mime_type"] = "application/json"
            except json.JSONDecodeError as e:
                return format_tool_error(f"Invalid JSON provided for response_schema: {e}", "INVALID_SCHEMA")

        config = gtypes.GenerateContentConfig(**config_args) if config_args else None

        if ctx:
            await ctx.info(f"Starting Gemini text generation with model {model}...")

        parts = [gtypes.Part.from_text(text=prompt)]
        parts.extend(_load_file_parts(input_paths))
        contents = [gtypes.Content(role="user", parts=parts)]

        logger.info(f"Generating text with model {model}...")
        if ctx:
            await ctx.report_progress(1, 2)
            
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config
        )
        
        if ctx:
            await ctx.report_progress(2, 2)
            await ctx.info("Text generation complete.")
            
        # Check if we should inject usage stats for JSON output
        if response_mime_type == "application/json":
            try:
                # Attempt to parse the text as JSON
                data = json.loads(response.text)
                if isinstance(data, dict):
                    # Extract usage
                    usage_meta = response.usage_metadata
                    data["llm_usage_metadata"] = _extract_usage_dict(usage_meta, model)
                    return json.dumps(data)
            except json.JSONDecodeError:
                pass # Return original text if not valid JSON

        return response.text

    except Exception as e:
        logger.error(f"Gemini text generation failed: {e}", exc_info=True)
        return format_tool_error(f"Gemini text generation failed: {e}", "GENERATION_FAILED")

@mcp.tool()
async def gemini_estimate_tokens(
    prompt: str,
    input_paths: str | None = None,
    model: str = "gemini-3-flash-preview",
    system_instruction: str | None = None,
) -> str:
    """
    Estimate token count and cost for a given prompt and input files.
    
    Args:
      prompt: The input text prompt.
      input_paths: Optional comma-delimited string of file paths.
      model: The model to use (default: gemini-3-flash-preview).
      system_instruction: Optional system instruction.
    """
    try:
        client = _get_client()
        
        parts = [gtypes.Part.from_text(text=prompt)]
        parts.extend(_load_file_parts(input_paths))
        contents = [gtypes.Content(role="user", parts=parts)]
        
        # Note: system_instruction might need to be passed differently or might not affect input token count significantly enough 
        # for a rough estimate, but the API supports it in count_tokens config if needed.
        # For simplicity, we just count the user content, which is the bulk. 
        # Use more advanced count_tokens args if the library supports it.
        
        response = client.models.count_tokens(
            model=model,
            contents=contents
        )
        
        total_tokens = response.total_tokens
        cost = calculate_cost(model, total_tokens, 0) # Assumes 0 output for input cost estimate
        
        return json.dumps({
            "total_tokens": total_tokens,
            "estimated_cost_usd": cost,
            "currency": "USD",
            "note": "Cost is for input tokens only."
        })
    except Exception as e:
        logger.error(f"Gemini token estimation failed: {e}", exc_info=True)
        return format_tool_error(f"Gemini token estimation failed: {e}", "ESTIMATION_FAILED")


@mcp.tool()
async def gemini_grade_exam(
    exam_paper_path: str,
    rubric_path: str,
    model: str = "gemini-3-flash-preview",
    ctx: Context = None,
) -> str:
    """ 
    Grade an exam paper (PDF) against a marking rubric (PDF), returning a structured JSON report.

    Args:
      exam_paper_path: Path to the exam paper PDF.
      rubric_path: Path to the marking rubric PDF.
      model: The model to use (default: gemini-3-flash-preview).
    """
    try:
        return await _gemini_grade_exam_impl(exam_paper_path, rubric_path, model, ctx)
    except Exception as e:
        logger.error(f"Grading failed: {e}", exc_info=True)
        return format_tool_error(f"Grading failed: {e}", "GRADING_FAILED")

async def _gemini_grade_exam_impl(
    exam_paper_path: str,
    rubric_path: str,
    model: str = "gemini-3-flash-preview",
    ctx: Context = None,
) -> str:
    """
    Grade an exam paper (PDF) against a marking rubric (PDF), returning a structured JSON report.

    Args:
      exam_paper_path: Path to the exam paper PDF.
      rubric_path: Path to the marking rubric PDF.
      model: The model to use (default: gemini-3-flash-preview).
    """
    schema_path = os.path.join(os.path.dirname(__file__), "exam_grading_schema.json")
    logger.info(f"--- Starting Grade Exam: Exam='{exam_paper_path}', Rubric='{rubric_path}' ---")
    try:
        with open(schema_path, "r") as f:
            schema = json.load(f)
    except Exception as e:
        raise ToolError(f"Failed to load grading schema from {schema_path}: {e}")

    prompt_path = os.path.join(os.path.dirname(__file__), "exam_grading_prompt.txt")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
    except Exception as e:
        # Fallback prompt if file read fails, or raise error. 
        # Raising error is safer to ensure we don't silently use a bad prompt.
        raise ToolError(f"Failed to load grading prompt from {prompt_path}: {e}")
    
    client = _get_client()
    config = gtypes.GenerateContentConfig(response_mime_type="application/json", response_schema=schema, temperature=0.0)
    
    # Load files and extract metadata
    parts = [gtypes.Part.from_text(text=prompt + "\n\n--- MARKING RUBRIC ---")]
    
    rubric_parts = _load_file_parts(rubric_path)
    parts.extend(rubric_parts)
    
    parts.append(gtypes.Part.from_text(text="\n\n--- EXAM SUBMISSION ---"))
    
    exam_parts = _load_file_parts(exam_paper_path)
    parts.extend(exam_parts)
    logger.info(f"Loaded file parts. Rubric segments: {len(rubric_parts)}, Exam segments: {len(exam_parts)}")

    # Helper to extract metadata async
    import asyncio
    import concurrent.futures
    from google.cloud import storage as gcs
    
    loop = asyncio.get_running_loop()
    
    def _download_and_extract(part, name):
        data = None
        if part.inline_data:
            data = part.inline_data.data
        elif part.file_data and part.file_data.file_uri.startswith("gs://"):
            try:
                # Download from GCS
                uri = part.file_data.file_uri
                # Parse gs://bucket/blob
                path_parts = uri.replace("gs://", "").split("/", 1)
                bucket_name = path_parts[0]
                blob_name = path_parts[1]
                
                storage_client = gcs.Client()
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_name)
                data = blob.download_as_bytes()
            except Exception as e:
                logger.warning(f"Failed to download GCS file {uri} for metadata: {e}")

        if data:
            return _get_pdf_metadata(data, name)
        return {"page_count": 0, "file_size_mb": 0.0}
    
    async def extract_meta_async(parts_list, name):
        if parts_list:
             return await loop.run_in_executor(None, _download_and_extract, parts_list[0], name)
        return {"page_count": 0, "file_size_mb": 0.0}

    # Extract concurrently
    rubric_meta, exam_meta = await asyncio.gather(
        extract_meta_async(rubric_parts, "Rubric"),
        extract_meta_async(exam_parts, "Exam")
    )
    logger.info(f"Metadata extracted: Exam Pages={exam_meta.get('page_count')}, Rubric Pages={rubric_meta.get('page_count')}")
    
    # Check Limits (example: 30 pages)
    LIMIT = 30
    if exam_meta["page_count"] > LIMIT:
        msg = f"Exam exceeds basic plan page limit ({LIMIT} pages). Submitted: {exam_meta['page_count']} pages."
        logger.error(msg)
        return format_tool_error(msg, "PAGE_LIMIT_EXCEEDED", {
            "page_count": exam_meta["page_count"],
            "limit": LIMIT
        })
    
    logger.info("Page limit check passed. Proceeding to generation...")

    try:
        logger.info(f"Grading exam with model {model}...")
        if ctx:
            await ctx.info(f"Preparing grading request with model {model}...")
            await ctx.report_progress(1, 3)

        response = client.models.generate_content(model=model, contents=[gtypes.Content(role="user", parts=parts)], config=config)
        
        if ctx:
            await ctx.info("Received grading response.")
            await ctx.report_progress(2, 3)
            
        # Optional: could add some parsing logic here if we wanted to stream back partial results
        # but for now we follow the simple progress pattern.
        
        if ctx:
            await ctx.report_progress(3, 3)
            await ctx.info("Grading complete.")
            
        # Parse the output and inject usage stats
        try:
            logger.info("Parsing structured JSON response from Gemini...")
            # The response text should be JSON because we requested it
            grading_report = json.loads(response.text)
            
            # Extract usage
            usage_meta = response.usage_metadata
            usage_dict = _extract_usage_dict(usage_meta, model)
            
            # Inject into the report
            if isinstance(grading_report, dict):
                grading_report["llm_usage_metadata"] = usage_dict
                grading_report["version"] = "1.0.0"
                # Inject Doc Metadata
                grading_report["doc_metadata"] = {
                    "exam": exam_meta,
                    "scheme": rubric_meta
                }
                return json.dumps(grading_report)
            else:
                # If it's a list or something else, return original but maybe log a warning
                return response.text
                
        except json.JSONDecodeError:
            # If parsing fails, just return the text
            return response.text

    except Exception as e:
        logger.error(f"Gemini exam grading failed: {e}", exc_info=True)
        raise ToolError(f"Gemini exam grading failed: {e}")

if __name__ == "__main__":
    mcp.run()
