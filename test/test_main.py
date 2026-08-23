import unittest
import os
from fastapi.testclient import TestClient
from app.main import app

# Set the MODEL_DIR environment variable
os.environ["MODEL_DIR"] = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "all-MiniLM-L6-v2",
)

# Create a TestClient instance
client = TestClient(app)


class TestMain(unittest.TestCase):
    def test_get_model_info(self):
        # Send a GET request to the /info endpoint
        response = client.get("/info")

        # Check if the response status code is 200 (OK)
        self.assertEqual(response.status_code, 200)

        # Parse the JSON response
        data = response.json()

        # Check if the response contains the expected keys
        self.assertIn("model", data)
        self.assertIn("description", data)
        self.assertIn("repository", data)
        self.assertIn("version", data)
        self.assertIn("license", data)
        self.assertIn("architecture", data)

        # Check if the values are as expected
        self.assertEqual(data["model"], "MiniLM-L6-v2")
        self.assertEqual(
            data["repository"],
            "https://huggingface.co/onnx-models/all-MiniLM-L6-v2-onnx",
        )
        self.assertEqual(data["version"], "v2")
        self.assertEqual(data["license"], "Apache-2.0")
        self.assertEqual(data["architecture"], "Transformer-based")

    def test_get_embedding(self):
        # Define the text parameter
        text = "Hello, world!"

        # Send a GET request to the /embed endpoint with the text parameter
        response = client.get(f"/embed?text={text}")

        # Check if the response status code is 200 (OK)
        self.assertEqual(response.status_code, 200)

        # Parse the JSON response
        data = response.json()

        # Check if the response contains the expected keys
        self.assertIn("tokens", data)
        self.assertIn("token_ids", data)
        self.assertIn("attention_mask", data)
        self.assertIn("embedding", data)

        # Check if the token_ids, attention_mask, and embedding are lists
        self.assertIsInstance(data["token_ids"], list)
        self.assertIsInstance(data["attention_mask"], list)
        self.assertIsInstance(data["embedding"], list)

        # Check if the embedding is a list of lists (since it's a 2D array)
        self.assertTrue(
            all(isinstance(item, list) for item in data["embedding"])
        )

    def test_get_embedding_no_text(self):
        # Send a GET request to the /embed endpoint without the text parameter
        response = client.get("/embed")

        # Check if the response status code is 422 (Unprocessable Entity)
        self.assertEqual(response.status_code, 422)

        # Parse the JSON response
        data = response.json()

        # Check if the response contains the expected keys
        self.assertIn("detail", data)

        # Check if the detail contains the expected error message
        self.assertEqual(data["detail"][0]["loc"], ["query", "text"])
        self.assertEqual(
            data["detail"][0]["msg"], "Field required"
        )  # Adjusted to match the expected case

    def test_get_embedding_empty_text(self):
        # Define the text parameter as empty string
        text = ""

        # Send a GET request to the /embed endpoint with the text parameter
        response = client.get(f"/embed?text={text}")

        # Check if the response status code is 200 (OK)
        self.assertEqual(response.status_code, 200)

        # Parse the JSON response
        data = response.json()

        # Check if the response contains the expected key
        self.assertIn("error", data)

        # Check if the error message is as expected
        self.assertEqual(data["error"], "Text parameter is required")

    def test_get_embedding_special_characters(self):
        # Define the text parameter with special characters
        text = "Hello, world! @#$%^&*()"

        # Send a GET request to the /embed endpoint with the text parameter
        response = client.get(f"/embed?text={text}")

        # Check if the response status code is 200 (OK)
        self.assertEqual(response.status_code, 200)

        # Parse the JSON response
        data = response.json()

        # Check if the response contains the expected keys
        self.assertIn("tokens", data)
        self.assertIn("token_ids", data)
        self.assertIn("attention_mask", data)
        self.assertIn("embedding", data)

        # Check if the token_ids, attention_mask, and embedding are lists
        self.assertIsInstance(data["token_ids"], list)
        self.assertIsInstance(data["attention_mask"], list)
        self.assertIsInstance(data["embedding"], list)

        # Check if the embedding is a list of lists (since it's a 2D array)
        self.assertTrue(
            all(isinstance(item, list) for item in data["embedding"])
        )

    def test_get_embedding_long_text(self):
        # Define the text parameter with a long string
        text = "a" * 1000  # 1000 'a' characters

        # Send a GET request to the /embed endpoint with the text parameter
        response = client.get(f"/embed?text={text}")

        # Check if the response status code is 200 (OK)
        self.assertEqual(response.status_code, 200)

        # Parse the JSON response
        data = response.json()

        # Check if the response contains the expected keys
        self.assertIn("tokens", data)
        self.assertIn("token_ids", data)
        self.assertIn("attention_mask", data)
        self.assertIn("embedding", data)

        # Check if the token_ids, attention_mask, and embedding are lists
        self.assertIsInstance(data["token_ids"], list)
        self.assertIsInstance(data["attention_mask"], list)
        self.assertIsInstance(data["embedding"], list)

        # Check if the embedding is a list of lists (since it's a 2D array)
        self.assertTrue(
            all(isinstance(item, list) for item in data["embedding"])
        )


# Run the test case
if __name__ == "__main__":
    unittest.main()
