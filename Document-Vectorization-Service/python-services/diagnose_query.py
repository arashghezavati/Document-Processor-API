#!/usr/bin/env python3
"""
Diagnostic script to test the document query process
This helps identify issues in the document retrieval pipeline
"""
import os
import sys
import json
from dotenv import load_dotenv
from qdrant_wrapper import get_qdrant_client
from embedding_function import GeminiEmbeddingFunction
import google.generativeai as genai

def diagnose_query(collection_name, query_text, debug=True):
    """Test document retrieval and diagnosis issues"""
    load_dotenv()
    api_key = os.getenv('GOOGLE_GEMINI_API_KEY')
    model_name = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash')
    
    if debug:
        print(f"🔍 Diagnosing query: '{query_text}' on collection: {collection_name}")
        print(f"Using API key: {api_key[:4]}...{api_key[-4:]}")
        print(f"Using model: {model_name}")
    
    # Initialize clients
    qdrant_client = get_qdrant_client()
    embedding_func = GeminiEmbeddingFunction(api_key=api_key)
    
    # Verify the collection exists
    collections = qdrant_client.list_collections()
    if debug:
        print(f"Available collections: {[col.name for col in collections]}")
    
    if collection_name not in [col.name for col in collections]:
        print(f"❌ Collection {collection_name} not found!")
        return False
    
    # Get the collection
    collection = qdrant_client.get_collection(name=collection_name, embedding_function=embedding_func)
    
    # Count documents in collection
    try:
        count = collection.count()
        print(f"✅ Collection has {count} documents")
    except Exception as e:
        print(f"❌ Error counting documents: {str(e)}")
        return False
    
    if count == 0:
        print("❌ Collection is empty - no documents to query")
        return False
    
    # Generate embedding for query
    if debug:
        print(f"Generating embedding for query: '{query_text}'")
    
    try:
        query_embedding = embedding_func([query_text])[0]
        print("✅ Query embedding generated successfully")
    except Exception as e:
        print(f"❌ Error generating query embedding: {str(e)}")
        return False
    
    # Try basic retrieval (without semantic search)
    print("\n🔍 Testing basic document retrieval...")
    try:
        basic_docs = collection.get(limit=3, include=["documents", "metadatas"])
        if basic_docs and len(basic_docs.get("documents", [])) > 0:
            print(f"✅ Basic retrieval successful - found {len(basic_docs['documents'])} documents")
            print("Sample document content:")
            for i, doc in enumerate(basic_docs["documents"][:2]):
                print(f"  Doc {i+1} ({len(doc)} chars): {doc[:150]}...")
        else:
            print("❌ Basic retrieval failed - no documents returned")
    except Exception as e:
        print(f"❌ Error in basic retrieval: {str(e)}")
    
    # Test vector search
    print("\n🔍 Testing semantic search...")
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            include=["documents", "metadatas", "distances"]
        )
        
        if results and results.get("documents") and len(results["documents"][0]) > 0:
            print(f"✅ Semantic search successful - found {len(results['documents'][0])} matching documents")
            
            # Print similarity scores and samples
            print("\nTop matches:")
            for i, (doc, distance) in enumerate(zip(results["documents"][0], results["distances"][0])):
                similarity = max(0, 1 - distance)
                print(f"\nDocument {i+1} - Similarity: {similarity:.4f}")
                print(f"Content sample: {doc[:200]}...")
                
                if i < 2 and "metadatas" in results and results["metadatas"][0]:
                    print(f"Metadata: {results['metadatas'][0][i]}")
        else:
            print("❌ Semantic search returned no results")
    except Exception as e:
        print(f"❌ Error in semantic search: {str(e)}")
    
    # Test AI generation with the query
    print("\n🔍 Testing AI generation...")
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        
        # Use a simple test prompt
        test_prompt = f"Respond with 'Test successful!' if you can see this query: '{query_text}'"
        
        response = model.generate_content(test_prompt)
        print(f"✅ AI generation successful: {response.text}")
    except Exception as e:
        print(f"❌ Error in AI generation: {str(e)}")
    
    print("\n✅ Diagnosis completed")
    return True

def main():
    """Main function to run the diagnosis"""
    if len(sys.argv) < 3:
        print("Usage: python diagnose_query.py <collection_name> <query_text>")
        sys.exit(1)
    
    collection_name = sys.argv[1]
    query_text = sys.argv[2]
    
    diagnose_query(collection_name, query_text)

if __name__ == "__main__":
    main()
