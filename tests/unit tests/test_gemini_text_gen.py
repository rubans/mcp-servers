import unittest
import asyncio
import os
import sys
import sys
import json
import types
from unittest.mock import patch, MagicMock, mock_open

# Add the server directory to sys.path so we can import the module
# Current file: c:\Code\mcp-servers\unit tests\test_gemini_text_gen.py
# Target file: c:\Code\mcp-servers\mcp\servers\gemini_text_gen.py
SERVER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../mcp/servers'))
sys.path.insert(0, SERVER_DIR)

# Import the module tools
# Note: This requires the dependencies (fastmcp, google-genai, python-dotenv) to be installed.
from gemini_text_gen import gemini_generate_text, gemini_grade_exam

class TestGeminiTextGen(unittest.TestCase):
    def setUp(self):
        self.mock_client_instance = MagicMock()
        self.mock_response = MagicMock()
        self.mock_response.text = "Mocked Gemini Response"
        self.mock_client_instance.models.generate_content.return_value = self.mock_response

    def test_gemini_generate_text_success(self):
        """Test basic text generation with mocked client."""
        # Patch the Client class where it is used in the module
        with patch('gemini_text_gen.genai.Client') as MockClient:
            MockClient.return_value = self.mock_client_instance
            
            # Mock environment variables to ensure _get_client works
            with patch.dict(os.environ, {"GOOGLE_API_KEY": "fake-key"}):
                result = asyncio.run(gemini_generate_text.fn(prompt="Hello world"))
            
            self.assertEqual(result, "Mocked Gemini Response")
            self.mock_client_instance.models.generate_content.assert_called_once()
            
            # Verify arguments
            call_args = self.mock_client_instance.models.generate_content.call_args
            kwargs = call_args.kwargs
            self.assertEqual(kwargs['model'], "gemini-3-flash-preview")
            
            # Verify content structure
            # We can check the text of the first part
            content = kwargs['contents'][0]
            self.assertEqual(content.parts[0].text, "Hello world")

    def test_gemini_generate_text_json_with_usage(self):
        """Test text generation with JSON response and usage metadata injection."""
        # Mock usage metadata on the response
        mock_usage = types.SimpleNamespace(
            prompt_token_count=10,
            candidates_token_count=20,
            total_token_count=30,
            reasoning_token_count=5 # Extra dynamic field
        )
        self.mock_response.usage_metadata = mock_usage
        
        # Valid JSON response from model
        self.mock_response.text = json.dumps({"key": "value"})

        with patch('gemini_text_gen.genai.Client') as MockClient, \
             patch('gemini_text_gen.litellm.cost_per_token', return_value=(0.00005, 0.0)):
            
            MockClient.return_value = self.mock_client_instance
            
            with patch.dict(os.environ, {"GOOGLE_API_KEY": "fake-key"}):
                result = asyncio.run(gemini_generate_text.fn(
                    prompt="Generate JSON", 
                    response_mime_type="application/json"
                ))
            
            # Verify result
            result_data = json.loads(result)
            self.assertEqual(result_data["key"], "value")
            self.assertIn("llm_usage_metadata", result_data)
            self.assertEqual(result_data["llm_usage_metadata"]["total_token_count"], 30)
            self.assertEqual(result_data["llm_usage_metadata"]["reasoning_token_count"], 5)
            self.assertEqual(result_data["llm_usage_metadata"]["estimated_cost_usd"], 0.00005)

    def test_gemini_grade_exam_success(self):
        """Test exam grading with mocked file inputs and schema."""
        
        mock_schema_content = json.dumps({
            "type": "OBJECT",
            "properties": {"total_marks": {"type": "NUMBER"}},
            "required": ["total_marks"]
        })
        
        rubric_content = b"Fake Rubric PDF Content"
        exam_content = b"Fake Exam PDF Content"

        # Mock usage metadata on the response
        mock_usage = types.SimpleNamespace(
            prompt_token_count=100,
            candidates_token_count=50,
            total_token_count=150
        )
        self.mock_response.usage_metadata = mock_usage
        
        # Set response text to valid JSON so injection logic triggers
        self.mock_response.text = json.dumps({"total_marks": 10, "summary_comments": "Good job."})

        # Side effect for open() to handle different files
        def open_side_effect(file, mode='r', *args, **kwargs):
            file_str = str(file)
            if file_str.endswith("exam_grading_schema.json"):
                return mock_open(read_data=mock_schema_content)(file, mode)
            
            if file_str.endswith("rubric.pdf"):
                return mock_open(read_data=rubric_content)(file, mode)
                
            if file_str.endswith("exam.pdf"):
                return mock_open(read_data=exam_content)(file, mode)
                
            raise FileNotFoundError(f"File {file} not found in mock")

        with patch('gemini_text_gen.genai.Client') as MockClient, \
             patch('builtins.open', side_effect=open_side_effect), \
             patch('gemini_text_gen.litellm.cost_per_token', return_value=(0.000123, 0.000001)): # Mock cost calculation
            
            MockClient.return_value = self.mock_client_instance
            
            with patch.dict(os.environ, {"GOOGLE_API_KEY": "fake-key"}):
                result = asyncio.run(gemini_grade_exam.fn(
                    exam_paper_path="path/to/exam.pdf",
                    rubric_path="path/to/rubric.pdf"
                ))

            # Result should be a JSON string with usage_metadata injected
            result_data = json.loads(result)
            self.assertEqual(result_data["total_marks"], 10)
            self.assertIn("llm_usage_metadata", result_data)
            self.assertEqual(result_data["llm_usage_metadata"]["estimated_cost_usd"], 0.000123)
            
            # Verify the client was called
            call_args = self.mock_client_instance.models.generate_content.call_args
            kwargs = call_args.kwargs
            
            # Check model
            self.assertEqual(kwargs['model'], "gemini-3-flash-preview")
            
            # Check config (schema)
            self.assertEqual(kwargs['config'].response_mime_type, "application/json")
            # The schema loaded from mock file
            self.assertEqual(kwargs['config'].response_schema, json.loads(mock_schema_content))
            
            # Check contents (parts)
            parts = kwargs['contents'][0].parts
            # Expecting:
            # 1. Text (Rubric Intro)
            # 2. Rubric PDF
            # 3. Text (Exam Intro)
            # 4. Exam PDF
            self.assertEqual(len(parts), 4)
            
            self.assertIn("--- MARKING RUBRIC ---", parts[0].text)
            self.assertEqual(parts[1].inline_data.data, rubric_content)
            self.assertIn("--- EXAM SUBMISSION ---", parts[2].text)
            self.assertEqual(parts[3].inline_data.data, exam_content)

    def test_load_file_parts_gcs_support(self):
        """Test that _load_file_parts handles gs:// URIs correctly."""
        # We need to import the internal function for testing
        from gemini_text_gen import _load_file_parts
        
        with patch('gemini_text_gen.gtypes.Part.from_uri') as mock_from_uri, \
             patch('gemini_text_gen.gtypes.Part.from_bytes') as mock_from_bytes, \
             patch('builtins.open', mock_open(read_data=b"local data")):
            
            # Case 1: Only GCS URI
            _load_file_parts("gs://bucket/file.pdf")
            mock_from_uri.assert_called_once_with(file_uri="gs://bucket/file.pdf", mime_type="application/pdf")
            mock_from_bytes.assert_not_called()
            
            mock_from_uri.reset_mock()
            mock_from_bytes.reset_mock()
            
            # Case 2: Mixed GCS and local
            _load_file_parts("gs://bucket/file.pdf, local_file.txt")
            self.assertEqual(mock_from_uri.call_count, 1)
            self.assertEqual(mock_from_bytes.call_count, 1)
            
            mock_from_uri.assert_called_with(file_uri="gs://bucket/file.pdf", mime_type="application/pdf")
            # For local_file.txt, mimetypes might return text/plain or None
            # Let's just check it was called.
            mock_from_bytes.assert_called_once()

if __name__ == "__main__":
    unittest.main()