import os
import chromadb
from auth import get_user

# Initialize ChromaDB Client
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'vector-database', 'store-new')
chroma_client = chromadb.PersistentClient(path=DB_PATH)

def get_user_collection_name(username):
    """Generate a collection name for a specific user."""
    return f"user_{username}_docs"

def clear_user_documents(username):
    """Clear all documents for a specific user."""
    collection_name = get_user_collection_name(username)
    
    # Check if collection exists
    collections = chroma_client.list_collections()
    collection_exists = False
    for collection in collections:
        if collection.name == collection_name:
            collection_exists = True
            break
    
    if not collection_exists:
        print(f"Collection '{collection_name}' does not exist. Nothing to delete.")
        return False
    
    # Get the collection
    collection = chroma_client.get_collection(collection_name)
    
    # Get all documents
    try:
        documents = collection.get()
        if not documents['ids'] or len(documents['ids']) == 0:
            print(f"No documents found in collection '{collection_name}'.")
            return True
        
        # Delete all documents
        num_docs = len(documents['ids'])
        collection.delete(ids=documents['ids'])
        print(f"Successfully deleted {num_docs} documents from collection '{collection_name}'.")
        return True
    except Exception as e:
        print(f"Error deleting documents: {str(e)}")
        return False

if __name__ == "__main__":
    # Get username from input
    username = input("Enter username to clear documents for: ")
    
    # Verify user exists
    user = get_user(username)
    if not user:
        print(f"User '{username}' not found.")
    else:
        # Clear documents
        success = clear_user_documents(username)
        if success:
            print(f"All documents for user '{username}' have been cleared.")
        else:
            print(f"Failed to clear documents for user '{username}'.")
