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
        doc_count = collection.count()
        print(f"Collection count reports {doc_count} documents")
        return max(1, doc_count)  # Ensure at least 1 document is used
    except Exception as e:
        print(f"⚠ Error counting documents: {str(e)}")
        return 10  # Default to 10 if we can't get count

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
            print(f"Searching in specific collection: {collection_name}")
            collections = [qdrant_client.get_collection(name=collection_name, embedding_function=embedding_model)]
        else:
            # Search in ALL available collections
            all_collections = get_all_collections(qdrant_client)
            if not all_collections:
                print("⚠ No collections found in the database.")
                return []
            print(f"Searching across {len(all_collections)} collections: {all_collections}")
            collections = [qdrant_client.get_collection(name=col, embedding_function=embedding_model) for col in all_collections]

        for collection in collections:
            collection_name = getattr(collection, "name", "unknown")
            print(f"Processing collection: {collection_name}")
            
            # First try semantic search
            try:
                # Generate query embedding
                print(f"Generating embedding for query: {query_text}")
                
                # Get up to 20 results (or all if less than 20)
                try:
                    total_docs = get_total_documents(collection)
                except Exception as count_err:
                    print(f"Error getting document count: {str(count_err)}")
                    total_docs = 10  # Fallback
                
                n_results = min(total_docs, 20)
                print(f"Querying for up to {n_results} results")
                
                # Try semantic search first
                results = collection.query(
                    query_texts=[query_text], 
                    n_results=n_results,
                    include=["documents", "metadatas", "distances"]
                )
                
                if 'documents' in results and results['documents'] and len(results['documents'][0]) > 0:
                    print(f"Found {len(results['documents'][0])} matching documents via semantic search")
                    for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
                        # Include documents with lower scores but adjust confidence
                        similarity = max(0, 1 - distance)  # Cosine distance to similarity
                        retrieved_docs.append((doc, similarity))
                        if i < 5:  # Print the first 5 for debugging
                            print(f"  Doc {i+1}: Similarity score: {similarity:.4f}")
                            print(f"  First 100 chars: {doc[:100]}...")
                else:
                    print("No matching documents found with semantic search, trying fallback")
                    # Fallback 1: Try direct retrieval
                    all_docs = collection.get(include=["documents"])
                    if all_docs and 'documents' in all_docs and all_docs['documents']:
                        print(f"Retrieved {len(all_docs['documents'])} documents via direct retrieval")
                        for doc in all_docs['documents']:
                            # Assign a moderate similarity score to all documents
                            retrieved_docs.append((doc, 0.5))
                        print(f"First document sample: {all_docs['documents'][0][:100]}...")
                    else:
                        print("No documents found via direct retrieval either")
            except Exception as query_error:
                print(f"Error during semantic query: {str(query_error)}")
                # Fallback 2: Last resort, try very simple retrieval
                try:
                    print("Attempting final fallback retrieval")
                    # Use a dummy vector with a very large limit
                    vector_size = int(os.getenv("EMBEDDING_DIMENSION", "768"))
                    dummy_vector = [0.0] * vector_size
                    
                    search_result = qdrant_client.client.search(
                        collection_name=collection_name,
                        query_vector=dummy_vector,
                        limit=100,  # Get up to 100 results
                        with_payload=True
                    )
                    
                    if search_result:
                        print(f"Retrieved {len(search_result)} documents via fallback search")
                        for point in search_result:
                            if hasattr(point, 'payload') and point.payload and 'text' in point.payload:
                                retrieved_docs.append((point.payload['text'], 0.4))  # Lower confidence for fallback
                except Exception as fallback_error:
                    print(f"Final fallback retrieval failed: {str(fallback_error)}")

        # Sort by similarity score, highest first
        retrieved_docs.sort(key=lambda x: x[1], reverse=True)
        print(f"Total documents retrieved across all methods: {len(retrieved_docs)}")
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

    # Print prompt length for debugging
    print(f"Prompt length: {len(prompt)} characters")
    
    # Truncate very long prompts to avoid token limits
    max_prompt_length = 30000  # Adjust based on model limits
    if len(prompt) > max_prompt_length:
        # Keep the instruction part and truncate the documents part
        # Find the instruction part (usually at the beginning and end)
        prompt_parts = prompt.split("Documents:")
        if len(prompt_parts) >= 2:
            instructions = prompt_parts[0]
            documents = prompt_parts[1]
            # Truncate the documents part
            truncated_documents = documents[:max_prompt_length - len(instructions) - 100]
            prompt = instructions + "Documents (truncated due to length):" + truncated_documents
            print(f"Prompt truncated to {len(prompt)} characters")
        else:
            # Simple truncation if we can't identify parts
            prompt = prompt[:max_prompt_length]
            print("Prompt truncated with simple method")

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
    
    # Use all documents but focus on the top ones (limit to avoid token limits)
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
