# Gemini MCP Servers

A collection of Model Context Protocol (MCP) servers powered by Google Gemini.

## Prerequisites

- Python 3.10+
- [FastMCP](https://github.com/jlowin/fastmcp)
- Google Gemini API Key (configured in `.env`)

## Installation

Install the required dependencies:

```bash
pip install -r mcp/requirements.txt
```

## Running the Servers

### Gemini Text Generation

Provides tools for text generation and exam grading.

**Tools:**
- `gemini_generate_text`: Generate text from prompts and files.
- `gemini_grade_exam`: Grade exam papers against a rubric.

**Run command:**

```bash
fastmcp run mcp/servers/gemini_text_gen.py -t streamable-http
```

### Gemini Media Generation

Provides tools for image and video generation.

**Tools:**
- `gemini_generate_image`: Create images from text descriptions.
- `gemini_generate_video`: Create short videos/animations from text descriptions.

**Run command:**

```bash
fastmcp run mcp/servers/gemini_media_gen.py -t streamable-http
```
