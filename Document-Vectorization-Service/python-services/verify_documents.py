#!/usr/bin/env python3
"""
Utility script to verify document storage and retrieval
"""
import os
import sys
from dotenv import load_dotenv
from qdrant_wrapper import get_qdrant_client
from embedding_function import GeminiEmbeddingFunction

def verify_collection(collection_name):
    """Verify the contents of a collection"""
    load_dotenv()
    api_key = os.getenv('GOOGLE_GEMINI_API_KEY')
    
    # Get the client and collection
    qdrant_client = get_qdrant_client()
    embedding_model = GeminiEmbeddingFunction(api_key=api_key)
    
    print(f"Verifying collection: {collection_name}")
    
    try:
        # Check if collection exists
        collections = qdrant_client.list_collections()
        if collection_name not in [col.name for col in collections]:
            print(f"❌ Collection {collection_name} not found")
            return False
        
        # Get the collection
        collection = qdrant_client.get_collection(name=collection_name, embedding_function=embedding_model)
        
        # Count documents
        try:
            doc_count = collection.count()
            print(f"✅ Collection contains {doc_count} documents")
        except Exception as e:
            print(f"❌ Error counting documents: {str(e)}")
            doc_count = 0
        
        if doc_count == 0:
            print("❌ Collection is empty")
            return False
        
        # Get all documents to check metadata and content
        try:
            docs = collection.get(include=["documents", "metadatas"])
            
            if not docs or not docs.get("documents") or len(docs["documents"]) == 0:
                print("❌ No documents returned from collection")
                return False
                
            # Print some stats and samples
            print(f"✅ Retrieved {len(docs['documents'])} documents")
            
            # Check metadata
            metadata_counts = {}
            for metadata in docs["metadatas"]:
                for key in metadata:
                    metadata_counts[key] = metadata_counts.get(key, 0) + 1
            
            print(f"📊 Metadata fields found: {metadata_counts}")
            
            # Print some document samples
            print("\n📄 Document Samples:")
            for i in range(min(3, len(docs["documents"]))):
                doc = docs["documents"][i]
                metadata = docs["metadatas"][i]
                print(f"\nDocument {i+1}:")
                print(f"  Metadata: {metadata}")
                print(f"  Content sample: {doc[:200]}...")
                print("  Length:", len(doc))
            
            print("\n✅ Collection verification complete")
            return True
            
        except Exception as e:
            print(f"❌ Error retrieving documents: {str(e)}")
            return False
            
    except Exception as e:
        print(f"❌ Error verifying collection: {str(e)}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_documents.py <collection_name>")
        return
    
    collection_name = sys.argv[1]
    verify_collection(collection_name)

if __name__ == "__main__":
    main()
