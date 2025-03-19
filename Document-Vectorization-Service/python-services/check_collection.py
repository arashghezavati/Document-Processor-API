import os
import chromadb
from dotenv import load_dotenv
from embedding_function import GeminiEmbeddingFunction

# Load environment variables
load_dotenv()
db_path = os.path.join(os.path.dirname(__file__), '..', 'vector-database', 'store')

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path=db_path)
embedding_model = GeminiEmbeddingFunction(api_key=os.getenv('GOOGLE_GEMINI_API_KEY'))

# Retrieve the collection
collection = chroma_client.get_collection(name="customer_123", embedding_function=embedding_model)

# Count documents
print(f"Total documents in customer_123: {collection.count()}")
