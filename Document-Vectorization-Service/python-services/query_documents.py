import os
import time
import google.generativeai as genai
from dotenv import load_dotenv
from embedding_function import GeminiEmbeddingFunction
from qdrant_wrapper import get_qdrant_client

def get_all_collections(qdrant_client):
    """Retrieve a list of all available collections in the vector database."""
    try:
        return [col.name for col in qdrant_client.list_collections()]
    except Exception as e:
        print(f"❌ Error fetching collections: {str(e)}")
        return []

def get_total_documents(collection):
    """Get the total number of documents in a collection."""
    try:
        return max(1, collection.count())  # Ensure at least 1 document is used
    except Exception as e:
        print(f"⚠ Error counting documents: {str(e)}")
        return 1

def retrieve_documents(query_text, collection_name=None):
    """Retrieve all documents from a specific collection or all available collections."""
    load_dotenv()
    api_key = os.getenv('GOOGLE_GEMINI_API_KEY')

    # Initialize Qdrant client
    qdrant_client = get_qdrant_client()
    embedding_model = GeminiEmbeddingFunction(api_key=api_key)

    retrieved_docs = []

    try:
        if collection_name and collection_name.lower() != "all":
            # Search in a specific collection
            collections = [qdrant_client.get_collection(name=collection_name, embedding_function=embedding_model)]
        else:
            # Search in ALL available collections
            all_collections = get_all_collections(qdrant_client)
            if not all_collections:
                print("⚠ No collections found in the database.")
                return []
            collections = [qdrant_client.get_collection(name=col, embedding_function=embedding_model) for col in all_collections]

        for collection in collections:
            total_docs = get_total_documents(collection)
            results = collection.query(
                query_texts=[query_text], 
                n_results=total_docs,
                include=["documents", "metadatas", "distances"]
            )

            if 'documents' in results and results['documents']:
                for doc, distance in zip(results['documents'][0], results['distances'][0]):
                    similarity = max(0, 1 - distance)  # Ensure similarity is never negative
                    retrieved_docs.append((doc, similarity))

        retrieved_docs.sort(key=lambda x: x[1], reverse=True)
        return retrieved_docs

    except Exception as e:
        print(f"❌ Error querying collection: {str(e)}")
        return []

def generate_response_gemini(prompt):
    """Generate a response using the Gemini API with retry logic."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    genai.configure(api_key=api_key)

    retries = 3
    for attempt in range(retries):
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"⚠ Error generating AI response (Attempt {attempt+1}): {str(e)}")
            time.sleep(2 ** attempt)  # Exponential backoff

    return "⚠ Unable to generate AI-enhanced response after multiple attempts."

def query_collection(query_text, collection_name=None, mode="strict"):
    retrieved_docs = retrieve_documents(query_text, collection_name)

    if not retrieved_docs:
        print("⚠ No relevant documents found in the database.")
        return

    document_context = "\n\n".join([doc[0] for doc in retrieved_docs])

    response = (
        generate_response_gemini(f"Here are relevant documents:\n\n{document_context}\n\nNow answer: {query_text}")
        if mode == "comprehensive"
        else f"Based on the following documents, answer: '{query_text}'\n\n{document_context}"
    )

    print("\n🤖 AI Response:\n", response)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python query_documents.py <query_text> [collection_name] [mode]")
        print("Use 'all' to search across all collections.")
        sys.exit(1)

    query_text = sys.argv[1]
    collection_name = sys.argv[2] if len(sys.argv) > 2 else None
    mode = sys.argv[3] if len(sys.argv) > 3 else "strict"

    query_collection(query_text, collection_name, mode)
