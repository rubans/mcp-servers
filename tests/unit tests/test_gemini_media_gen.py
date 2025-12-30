import unittest
import asyncio
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

# Since the server script is not a package, we need to add its directory to the path
# to import it. A better long-term solution would be to structure the project with packages.
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gemini_media_gen import nanobanana_generate
from fastmcp.exceptions import ToolError


class TestGeminiMediaGen(unittest.TestCase):
    def test_nanobanana_generate_success_text_only(self):
        """
        Tests the successful generation of an image from a text prompt.
        """
        # Mock the genai client to avoid real API calls
        mock_client = MagicMock()

        # Mock the response stream from the client
        mock_chunk = MagicMock()
        mock_part = MagicMock()
        mock_part.inline_data.data = b"fake-image-bytes"
        mock_part.inline_data.mime_type = "image/png"
        mock_chunk.candidates = [MagicMock(content=MagicMock(parts=[mock_part]))]
        mock_chunk.text = "Generated image."
        mock_client.models.generate_content_stream.return_value = [mock_chunk]

        # Mock the Path object and file operations
        mock_path_obj = MagicMock()
        mock_path_obj.__truediv__.return_value = "mock/path/nanobanana_test.png"

        # 2. Run the test in an async event loop
        async def run_test():
            # Use patch as a context manager
            with patch("gemini_media_gen.genai.Client", return_value=mock_client) as mock_genai_client, \
                 patch("gemini_media_gen.Path", return_value=mock_path_obj) as mock_path, \
                 patch("gemini_media_gen.open", mock_open()) as mocked_file, \
                 patch.dict(os.environ, {"VERTEXAI_PROJECT": "test-project", "VERTEXAI_LOCATION": "test-location"}):

                # 3. Call the function being tested
                result = await nanobanana_generate(prompt="a test prompt", out_dir="mock/path")

                # 4. Assertions
                # Check if the client was initialized and the model was called
                mock_genai_client.assert_called_once_with(
                    vertexai=True, project="test-project", location="test-location"
                )
                mock_client.models.generate_content_stream.assert_called_once()

                # Check if the output directory was created
                mock_path.assert_called_with(os.path.expanduser("mock/path"))
                mock_path_obj.mkdir.assert_called_once_with(parents=True, exist_ok=True)

                # Check if the image file was written correctly
                mocked_file.assert_called_once_with("mock/path/nanobanana_test.png", "wb")
                handle = mocked_file()
                handle.write.assert_called_once_with(b"fake-image-bytes")

                # Check the returned result
                self.assertTrue(result["ok"])
                self.assertEqual(len(result["paths"]), 1)
                self.assertEqual(result["paths"][0], "mock/path/nanobanana_test.png")
                self.assertEqual(result["text"], "Generated image.")

        # Run the async test
        asyncio.run(run_test())

    def test_nanobanana_generate_missing_env_vars(self):
        """
        Tests that a ToolError is raised if Vertex AI environment variables are missing.
        """
        async def run_test():
            # Ensure the environment variables are not set for this test
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(ToolError) as cm:
                    await nanobanana_generate(prompt="a test prompt")
                
                self.assertIn("Missing Vertex AI configuration", str(cm.exception))

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()