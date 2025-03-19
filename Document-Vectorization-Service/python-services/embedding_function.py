import google.generativeai as genai
from chromadb.api.types import Documents, EmbeddingFunction
from typing import List
import numpy as np
import os
from dotenv import load_dotenv

# Load configuration from environment
load_dotenv()
EMBEDDING_DIMENSION = int(os.getenv('EMBEDDING_DIMENSION', '768'))
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-004')

class GeminiEmbeddingFunction(EmbeddingFunction):
    """Embedding function that uses Google's Gemini API for embeddings."""
    
    def __init__(self, api_key: str):
        """Initialize the Gemini embedding function.
        
        Args:
            api_key (str): Gemini API key
        """
        genai.configure(api_key=api_key)
        
    def __call__(self, texts: Documents) -> List[List[float]]:
        """Generate embeddings for the given texts using Gemini's embedding API.
        
        Args:
            texts (Documents): List of texts to generate embeddings for
            
        Returns:
            List[List[float]]: List of embeddings
        """
        embeddings = []
        
        for text in texts:
            # Limit text size to avoid payload size errors
            if len(text) > 8000:
                text = text[:8000]
            
            # Use the model name with 'models/' prefix as suggested by the error message
            model_name = f"models/{EMBEDDING_MODEL}"
            print(f"Using model: {model_name}")
            
            # Call the API to get embeddings
            result = genai.embed_content(
                model=model_name,
                content=text,
                task_type="retrieval_document"
            )
            
            # Print the result structure to debug
            print(f"API Response structure: {type(result)}")
            
            # Extract the embedding values based on the correct structure
            # The structure might be different than expected, so we need to handle it properly
            if hasattr(result, 'embedding'):
                # If result is an object with embedding attribute
                embedding = result.embedding
            elif hasattr(result, 'embeddings'):
                # If result has embeddings attribute
                embedding = result.embeddings[0]
            elif isinstance(result, dict):
                # If result is a dictionary
                if 'embedding' in result:
                    embedding = result['embedding']
                    if isinstance(embedding, dict) and 'values' in embedding:
                        embedding = embedding['values']
                elif 'embeddings' in result:
                    embedding = result['embeddings'][0]
            else:
                # Try to convert to a list directly
                embedding = list(result)
            
            # Convert to list if it's not already
            if not isinstance(embedding, list):
                embedding = list(embedding)
            
            # Ensure consistent size
            if len(embedding) < EMBEDDING_DIMENSION:
                embedding.extend([0.0] * (EMBEDDING_DIMENSION - len(embedding)))
            else:
                embedding = embedding[:EMBEDDING_DIMENSION]
            
            # Normalize the embedding to unit length (L2 norm)
            norm = np.linalg.norm(embedding)
            if norm > 0:  # Avoid division by zero
                embedding = [x / norm for x in embedding]
            
            embeddings.append(embedding)
        
        return embeddings
