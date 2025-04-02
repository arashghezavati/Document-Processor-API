import os
from dotenv import load_dotenv
from embedding_function import GeminiEmbeddingFunction
from qdrant_wrapper import get_qdrant_client

# Load environment variables
load_dotenv()

# Initialize Qdrant client
qdrant_client = get_qdrant_client()
embedding_model = GeminiEmbeddingFunction(api_key=os.getenv('GOOGLE_GEMINI_API_KEY'))

# Retrieve the collection
collection = qdrant_client.get_collection(name="customer_123", embedding_function=embedding_model)

# Count documents
print(f"Total documents in customer_123: {collection.count()}")
