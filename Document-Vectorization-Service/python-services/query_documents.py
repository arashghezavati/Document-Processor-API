# import os
# import chromadb
# from dotenv import load_dotenv
# from embedding_function import GeminiEmbeddingFunction

# def query_collection(query_text, collection_name="default", n_results=5):
#     """Query a collection in ChromaDB.
    
#     Args:
#         query_text (str): Text to search for
#         collection_name (str): Name of collection to search in
#         n_results (int): Number of results to return
#     """
#     # Load environment variables
#     load_dotenv()
#     api_key = os.getenv('GOOGLE_GEMINI_API_KEY')
    
#     # Initialize ChromaDB
#     db_path = os.path.join(os.path.dirname(__file__), '..', 'vector-database', 'store')
#     chroma_client = chromadb.PersistentClient(path=db_path)
    
#     # Create embedding function
#     embedding_model = GeminiEmbeddingFunction(api_key=api_key)
    
#     try:
#         # Get collection
#         collection = chroma_client.get_collection(
#             name=collection_name,
#             embedding_function=embedding_model
#         )
        
#         # Query the collection
#         results = collection.query(
#             query_texts=[query_text],
#             n_results=n_results
#         )
        
#         # Print results
#         print(f"\n🔍 Results for query: '{query_text}' in collection '{collection_name}'")
#         print("=" * 80)
        
#         for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
#             similarity = 1 - distance  # Convert distance to similarity score
#             print(f"\n📄 Result {i+1} (Similarity: {similarity:.2%}):")
#             print("-" * 40)
#             print(doc)
            
#     except Exception as e:
#         print(f"❌ Error querying collection: {str(e)}")

# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) < 2:
#         print("Usage: python query_documents.py <query_text> [collection_name] [n_results]")
#         sys.exit(1)
    
#     query_text = sys.argv[1]
#     collection_name = sys.argv[2] if len(sys.argv) > 2 else "default"
#     n_results = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    
#     query_collection(query_text, collection_name, n_results)

import os
import chromadb
import google.generativeai as genai
import time
from dotenv import load_dotenv
from embedding_function import GeminiEmbeddingFunction

def get_all_collections(chroma_client):
    """Retrieve a list of all available collections in the vector database."""
    try:
        return [col.name for col in chroma_client.list_collections()]
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

    db_path = os.path.join(os.path.dirname(__file__), '..', 'vector-database', 'store')
    chroma_client = chromadb.PersistentClient(path=db_path)
    embedding_model = GeminiEmbeddingFunction(api_key=api_key)

    retrieved_docs = []

    try:
        if collection_name and collection_name.lower() != "all":
            # Search in a specific collection
            collections = [chroma_client.get_collection(name=collection_name, embedding_function=embedding_model)]
        else:
            # Search in ALL available collections
            all_collections = get_all_collections(chroma_client)
            if not all_collections:
                print("⚠ No collections found in the database.")
                return []
            collections = [chroma_client.get_collection(name=col, embedding_function=embedding_model) for col in all_collections]

        for collection in collections:
            total_docs = get_total_documents(collection)
            results = collection.query(query_texts=[query_text], n_results=total_docs)

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
