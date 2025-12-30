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
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from google import genai
from google.genai import types as gtypes
from dotenv import load_dotenv

# ---------- Logging ----------
# Set log level from LOG_LEVEL env var, defaulting to INFO
log_level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
if log_level_name not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
    log_level_name = "INFO"

handlers = [logging.StreamHandler(sys.stderr)]
log_file = os.environ.get("LOG_FILE", "gemini_text_gen.log")
handlers.append(logging.FileHandler(log_file))

logging.basicConfig(
    level=getattr(logging, log_level_name),
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=handlers
)
log = logging.getLogger("gemini_text_gen")
log.info("--- Log level set to %s ---", log_level_name)

# ---------- Load environment variables ----------
dotenv_path_from_env = os.environ.get("DOTENV_PATH")
if dotenv_path_from_env and os.path.exists(dotenv_path_from_env):
    log.info(f"--- Loading .env file from: {dotenv_path_from_env} ---")
    load_dotenv(dotenv_path=dotenv_path_from_env)
else:    
    load_dotenv()

# ---------- MCP server ----------
mcp = FastMCP("Gemini Text MCP")
log.info("start gemini text mcp server...")

def _get_client() -> genai.Client:
    """
    Initializes and returns the GenAI client (AI Studio or Vertex AI).

    Raises:
        ToolError: If neither API key nor Vertex AI environment variables are set.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key:
        return genai.Client(api_key=api_key)

    project = os.environ.get("VERTEXAI_PROJECT")
    location = os.environ.get("VERTEXAI_LOCATION")
    if project and location:
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

def _load_file_parts(input_paths: str | None) -> list[gtypes.Part]:
    parts = []
    if input_paths:
        for p in [p.strip() for p in input_paths.split(',') if p.strip()]:
            try:
                with open(p, "rb") as f:
                    data = f.read()
                mt, _ = mimetypes.guess_type(p)
                if not mt:
                    mt = "application/pdf" if p.lower().endswith(".pdf") else "application/octet-stream"
                parts.append(gtypes.Part.from_bytes(data=data, mime_type=mt))
            except Exception as e:
                raise ToolError(f"Failed to read input file '{p}': {e}")
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
            raise ToolError(f"Invalid JSON provided for response_schema: {e}")

    config = gtypes.GenerateContentConfig(**config_args) if config_args else None

    parts = [gtypes.Part.from_text(text=prompt)]
    parts.extend(_load_file_parts(input_paths))
    contents = [gtypes.Content(role="user", parts=parts)]

    try:
        log.info(f"Generating text with model {model}...")
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config
        )
        return response.text
    except Exception as e:
        log.error(f"Gemini text generation failed: {e}", exc_info=True)
        raise ToolError(f"Gemini text generation failed: {e}")

@mcp.tool()
async def gemini_grade_exam(
    exam_paper_path: str,
    rubric_path: str,
    model: str = "gemini-3-flash-preview",
) -> str:
    """
    Grade an exam paper (PDF) against a marking rubric (PDF), returning a structured JSON report.

    Args:
      exam_paper_path: Path to the exam paper PDF.
      rubric_path: Path to the marking rubric PDF.
      model: The model to use (default: gemini-3-flash-preview).
    """
    schema_path = os.path.join(os.path.dirname(__file__), "exam_grading_schema.json")
    try:
        with open(schema_path, "r") as f:
            schema = json.load(f)
    except Exception as e:
        raise ToolError(f"Failed to load grading schema from {schema_path}: {e}")

    prompt = "Please grade the following exam submission based on the provided marking rubric."
    
    client = _get_client()
    config = gtypes.GenerateContentConfig(response_mime_type="application/json", response_schema=schema, temperature=0.0)
    parts = [gtypes.Part.from_text(text=prompt + "\n\n--- MARKING RUBRIC ---")]
    parts.extend(_load_file_parts(rubric_path))
    parts.append(gtypes.Part.from_text(text="\n\n--- EXAM SUBMISSION ---"))
    parts.extend(_load_file_parts(exam_paper_path))
    
    try:
        log.info(f"Grading exam with model {model}...")
        response = client.models.generate_content(model=model, contents=[gtypes.Content(role="user", parts=parts)], config=config)
        return response.text
    except Exception as e:
        log.error(f"Gemini exam grading failed: {e}", exc_info=True)
        raise ToolError(f"Gemini exam grading failed: {e}")

if __name__ == "__main__":
    mcp.run()
