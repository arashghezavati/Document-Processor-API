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
            # First try to get total docs count
            try:
                total_docs = get_total_documents(collection)
                print(f"Collection has {total_docs} documents")
            except Exception as e:
                print(f"Error getting document count: {str(e)}")
                total_docs = 10  # Default to retrieving 10 documents

            # Increase n_results to retrieve more potential matches
            n_results = min(total_docs, 20)  # Get up to 20 results
            
            try:
                # Generate query embedding
                print(f"Generating embedding for query: {query_text}")
                # Query with more results and lower threshold
                results = collection.query(
                    query_texts=[query_text], 
                    n_results=n_results,
                    include=["documents", "metadatas", "distances"]
                )
                
                if 'documents' in results and results['documents'] and len(results['documents'][0]) > 0:
                    print(f"Found {len(results['documents'][0])} matching documents")
                    for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
                        # Include documents even with lower similarity scores
                        similarity = max(0, 1 - distance)  # Ensure similarity is never negative
                        retrieved_docs.append((doc, similarity))
                        print(f"  Doc {i+1}: Similarity score: {similarity:.4f}")
                else:
                    print("No matching documents found with semantic search")
                    # Fallback: get all documents from collection
                    print("Falling back to retrieving all documents")
                    all_docs = collection.get(include=["documents"])
                    if all_docs and 'documents' in all_docs and all_docs['documents']:
                        print(f"Retrieved {len(all_docs['documents'])} documents as fallback")
                        for doc in all_docs['documents']:
                            # Assign a moderate similarity score to all documents
                            retrieved_docs.append((doc, 0.5))
            except Exception as query_error:
                print(f"Error during query: {str(query_error)}")
                # Try a fallback approach
                try:
                    print("Attempting fallback retrieval method")
                    all_docs = collection.get(limit=10, include=["documents"])
                    if all_docs and 'documents' in all_docs and all_docs['documents']:
                        for doc in all_docs['documents']:
                            retrieved_docs.append((doc, 0.5))  # Default similarity
                except Exception as fallback_error:
                    print(f"Fallback retrieval failed: {str(fallback_error)}")

        # Sort by similarity score, highest first
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
        return "No relevant documents found to answer your query. Please try a different question or add more documents."

    # Print what we found
    print(f"Retrieved {len(retrieved_docs)} documents for query: '{query_text}'")
    
    # Use all documents but focus on the top ones
    document_context = "\n\n".join([doc[0] for doc in retrieved_docs[:10]])  # Limit to 10 docs
    
    # Create more focused prompt
    if mode == "comprehensive":
        prompt = f"""Here are relevant documents:

{document_context}

Based on ONLY the information provided above, please answer the following question:
"{query_text}"

If the answer cannot be found in the provided documents, please say so instead of making up information."""
    else:
        prompt = f"""Based on the following documents, answer this question: '{query_text}'

Documents:
{document_context}

Answer ONLY using information from the documents provided. If the information is not in the documents, say "I don't have that information in the documents."
"""
    
    print(f"Generating AI response based on {len(retrieved_docs[:10])} relevant documents")
    response = generate_response_gemini(prompt)
    
    print("\n🤖 AI Response:\n", response)
    return response

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
