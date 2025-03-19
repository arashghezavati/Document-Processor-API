import os
import chromadb
import json

# Initialize ChromaDB Client
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'vector-database', 'store-new')
chroma_client = chromadb.PersistentClient(path=DB_PATH)

# List all collections
print("Available collections:")
collections = chroma_client.list_collections()
for collection in collections:
    print(f"- {collection.name}")

# Try to access a specific collection (replace with your username)
username = "arash"  # Change this to your actual username
collection_name = f"user_{username}"

try:
    print(f"\nTrying to access collection: {collection_name}")
    collection = chroma_client.get_collection(collection_name)
    
    # Get all documents
    print("\nDocuments in collection:")
    documents = collection.get()
    
    if not documents['ids']:
        print("No documents found in collection")
    else:
        print(f"Found {len(documents['ids'])} documents")
        
        # Print document details
        print("\nDocument details:")
        for i, doc_id in enumerate(documents['ids']):
            metadata = documents['metadatas'][i]
            print(f"Document ID: {doc_id}")
            print(f"Document Name: {metadata.get('document_name', 'Unknown')}")
            print(f"Folder: {metadata.get('folder_name', 'No folder')}")
            print("-" * 30)
except Exception as e:
    print(f"Error accessing collection: {str(e)}")
