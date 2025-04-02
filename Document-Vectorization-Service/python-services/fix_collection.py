#!/usr/bin/env python3
"""
Tool to fix or re-index collections that might be damaged
"""
import os
import sys
import time
from dotenv import load_dotenv
from qdrant_wrapper import get_qdrant_client
from embedding_function import GeminiEmbeddingFunction
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("fix_collection")

def reindex_collection(collection_name):
    """Re-index a collection to fix potential issues"""
    load_dotenv()
    api_key = os.getenv('GOOGLE_GEMINI_API_KEY')
    
    if not api_key:
        logger.error("GOOGLE_GEMINI_API_KEY not found in environment variables")
        return False
    
    # Initialize clients
    qdrant_client = get_qdrant_client()
    embedding_func = GeminiEmbeddingFunction(api_key=api_key)
    
    try:
        # Verify the collection exists
        collections = qdrant_client.list_collections()
        if collection_name not in [col.name for col in collections]:
            logger.error(f"Collection {collection_name} not found")
            return False
        
        # Get the collection
        collection = qdrant_client.get_collection(name=collection_name, embedding_function=embedding_func)
        
        # Backup the data
        logger.info(f"Retrieving all data from collection {collection_name} for backup")
        
        try:
            backup_data = collection.get(include=["documents", "metadatas"])
            
            if not backup_data or not backup_data.get("documents") or len(backup_data["documents"]) == 0:
                logger.warning("Collection is empty or data retrieval failed")
                return False
            
            logger.info(f"Retrieved {len(backup_data['documents'])} documents for backup")
            
            # Create a new temporary collection
            temp_collection_name = f"{collection_name}_temp_{int(time.time())}"
            logger.info(f"Creating temporary collection: {temp_collection_name}")
            
            temp_collection = qdrant_client.get_or_create_collection(
                name=temp_collection_name,
                embedding_function=embedding_func
            )
            
            # Reindex the data into the new collection
            logger.info("Re-indexing documents with fresh embeddings...")
            chunks = backup_data["documents"]
            metadatas = backup_data["metadatas"]
            
            # Generate unique document IDs
            import uuid
            doc_ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
            
            # Add to temporary collection in smaller batches
            batch_size = 50
            for i in range(0, len(chunks), batch_size):
                end_idx = min(i + batch_size, len(chunks))
                batch_chunks = chunks[i:end_idx]
                batch_metadatas = metadatas[i:end_idx]
                batch_ids = doc_ids[i:end_idx]
                
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
                
                try:
                    temp_collection.add(
                        documents=batch_chunks,
                        ids=batch_ids,
                        metadatas=batch_metadatas
                    )
                    logger.info(f"Successfully added batch of {len(batch_chunks)} documents")
                except Exception as batch_error:
                    logger.error(f"Error adding batch to temp collection: {str(batch_error)}")
            
            # Rename collections
            try:
                # Rename the original collection
                backup_collection_name = f"{collection_name}_backup_{int(time.time())}"
                logger.info(f"Renaming original collection to {backup_collection_name}")
                qdrant_client.client.collection_ops.rename_collection(
                    collection_name=collection_name,
                    new_collection_name=backup_collection_name
                )
                
                # Rename the temporary collection to the original name
                logger.info(f"Renaming temporary collection to {collection_name}")
                qdrant_client.client.collection_ops.rename_collection(
                    collection_name=temp_collection_name,
                    new_collection_name=collection_name
                )
                
                logger.info(f"✅ Successfully reindexed collection {collection_name}")
                return True
                
            except Exception as rename_error:
                logger.error(f"Error renaming collections: {str(rename_error)}")
                return False
                
        except Exception as get_error:
            logger.error(f"Error retrieving data for backup: {str(get_error)}")
            return False
            
    except Exception as e:
        logger.error(f"Error reindexing collection: {str(e)}")
        return False

def main():
    """Main function to run the fix operation"""
    if len(sys.argv) < 2:
        print("Usage: python fix_collection.py <collection_name>")
        return
    
    collection_name = sys.argv[1]
    reindex_collection(collection_name)

if __name__ == "__main__":
    main()
