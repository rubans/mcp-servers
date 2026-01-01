import pytest
import os
import json
import sys
from datetime import datetime
import pytest_asyncio
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.utilities.tests import run_server_async

# Add the servers directory to sys.path to resolve import conflict
# Assuming structure: root/tests/Integration_tests/test_file.py
# We want root/mcp/servers
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, os.path.join(repo_root, "mcp", "servers"))

import gemini_text_gen
server = gemini_text_gen.mcp

@pytest_asyncio.fixture
async def http_server():
    """Start server in-process for testing."""
    async with run_server_async(server) as url:
        yield url

@pytest.mark.asyncio
async def test_gemini_grade_exam(http_server: str):
    """Test gemini_grade_exam with actual PDF files."""
    
    # Define paths to the specific files
    base_dir = os.path.dirname(os.path.abspath(__file__))
    exam_parent_path = "History_June_2023_Paper3"
    exam_paper_path = os.path.join(base_dir, exam_parent_path, "Exam.pdf")
    rubric_path = os.path.join(base_dir, exam_parent_path, "Schema.pdf")

    # Verify files exist before running the test
    assert os.path.exists(exam_paper_path), f"Exam paper not found at {exam_paper_path}"
    assert os.path.exists(rubric_path), f"Rubric not found at {rubric_path}"

    async with Client(
        transport=StreamableHttpTransport(http_server)
    ) as client:
        # Check basic connectivity
        result = await client.ping()
        assert result is True

        # Call the grading tool
        print(f"Calling gemini_grade_exam with:\nExam: {exam_paper_path}\nRubric: {rubric_path}")
        result = await client.call_tool(
            "gemini_grade_exam", 
            arguments={
                "exam_paper_path": exam_paper_path,
                "rubric_path": rubric_path
            }
        )
        
        # Verify response
        assert result is not None
        text_response = result.content[0].text
        print(f"Response from server: {text_response[:200]}...") # Print start of response for debug
        
        # It should be a JSON string, try parsing it
        try:
            grading_report = json.loads(text_response)
            assert isinstance(grading_report, dict)
            # Basic check for expected fields (assuming common structure, but at least it's valid JSON)
        except json.JSONDecodeError:
            pytest.fail(f"Response was not valid JSON: {text_response}")

        # Convert to Markdown using the new helper function
        # We need to import the function from the server module we already have access to via `server` object's module,
        # but since `server` is an instance of FastMCP, we should access the module where format_grading_report is defined.
        # However, we imported `gemini_text_gen` as module earlier.
        
        md_report = gemini_text_gen.format_grading_report(grading_report)
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(base_dir, exam_parent_path, f"grading_report_{timestamp}.md")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(md_report)
            
        print(f"Grading report saved to: {output_file}")
        assert os.path.exists(output_file)
        assert len(md_report) > 0
