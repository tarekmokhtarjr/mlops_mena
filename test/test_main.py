import unittest
from unittest.mock import patch
import os
from fastapi.testclient import TestClient
from app.main import app

os.environ["MODEL_DIR"] = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "all-MiniLM-L6-v2",
)

client = TestClient(app)


class TestMain(unittest.TestCase):
    def test_get_model_info(self):
        response = client.get("/info")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("model", data)
        self.assertIn("description", data)
        self.assertIn("repository", data)
        self.assertIn("version", data)
        self.assertIn("license", data)
        self.assertIn("architecture", data)
        self.assertEqual(data["model"], "MiniLM-L6-v2")
        self.assertEqual(
            data["repository"],
            "https://huggingface.co/onnx-models/all-MiniLM-L6-v2-onnx",
        )
        self.assertEqual(data["version"], "v2")
        self.assertEqual(data["license"], "Apache-2.0")
        self.assertEqual(data["architecture"], "Transformer-based")

    def test_get_embedding(self):
        text = "Hello, world!"
        response = client.get(f"/embed?text={text}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("tokens", data)
        self.assertIn("token_ids", data)
        self.assertIn("attention_mask", data)
        self.assertIn("embedding", data)
        self.assertIsInstance(data["token_ids"], list)
        self.assertIsInstance(data["attention_mask"], list)
        self.assertIsInstance(data["embedding"], list)
        self.assertTrue(
            all(isinstance(item, list) for item in data["embedding"])
        )

    def test_get_embedding_no_text(self):
        response = client.get("/embed")
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertIn("detail", data)
        self.assertEqual(data["detail"][0]["loc"], ["query", "text"])
        self.assertEqual(data["detail"][0]["msg"], "Field required")

    def test_get_embedding_empty_text(self):
        text = ""
        response = client.get(f"/embed?text={text}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("error", data)
        self.assertEqual(data["error"], "Text parameter is required")

    def test_get_embedding_special_characters(self):
        text = "Hello, world! @#$%^&*()"
        response = client.get(f"/embed?text={text}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("tokens", data)
        self.assertIn("token_ids", data)
        self.assertIn("attention_mask", data)
        self.assertIn("embedding", data)
        self.assertIsInstance(data["token_ids"], list)
        self.assertIsInstance(data["attention_mask"], list)
        self.assertIsInstance(data["embedding"], list)
        self.assertTrue(
            all(isinstance(item, list) for item in data["embedding"])
        )

    def test_get_embedding_long_text(self):
        text = "a" * 1000  # 1000 'a' characters
        response = client.get(f"/embed?text={text}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("tokens", data)
        self.assertIn("token_ids", data)
        self.assertIn("attention_mask", data)
        self.assertIn("embedding", data)
        self.assertIsInstance(data["token_ids"], list)
        self.assertIsInstance(data["attention_mask"], list)
        self.assertIsInstance(data["embedding"], list)
        self.assertTrue(
            all(isinstance(item, list) for item in data["embedding"])
        )

    def test_get_embedding_tokenizer_error(self):
        with patch("app.main.Tokenizer.from_file") as mock_tokenizer:
            mock_tokenizer.side_effect = Exception("Mocked Tokenizer error")
            response = client.get("/embed?text=Hello, world!")
            self.assertEqual(response.status_code, 500)
            data = response.json()
            self.assertIn("detail", data)
            self.assertEqual(
                data["detail"], "Tokenizer error: Mocked Tokenizer error"
            )

    def test_get_embedding_inference_session_error(self):
        with patch("app.main.InferenceSession") as mock_session:
            mock_session.side_effect = Exception(
                "Mocked InferenceSession error"
            )
            response = client.get("/embed?text=Hello, world!")
            self.assertEqual(response.status_code, 500)
            data = response.json()
            self.assertIn("detail", data)
            self.assertEqual(
                data["detail"],
                "Inference session error: Mocked InferenceSession error",
            )

    def test_get_embedding_encoding_error(self):
        with patch("app.main.Tokenizer.encode") as mock_encode:
            mock_encode.side_effect = Exception("Mocked Encoding error")
            response = client.get("/embed?text=Hello, world!")
            self.assertEqual(response.status_code, 422)
            data = response.json()
            self.assertIn("detail", data)
            self.assertEqual(
                data["detail"], "Encoding error: Mocked Encoding error"
            )

    def test_get_embedding_session_run_error(self):
        with patch("app.main.InferenceSession.run") as mock_run:
            mock_run.side_effect = Exception("Mocked Session run error")
            response = client.get("/embed?text=Hello, world!")
            self.assertEqual(response.status_code, 500)
            data = response.json()
            self.assertIn("detail", data)
            self.assertEqual(
                data["detail"], "Session run error: Mocked Session run error"
            )


# Run the test case
if __name__ == "__main__":
    unittest.main()
