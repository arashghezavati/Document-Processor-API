#!/usr/bin/env python3
"""
Migration script to transfer data from ChromaDB to Qdrant
"""

import os
import uuid
import chromadb
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv
import time
from tqdm import tqdm
import argparse

# Load environment variables from main .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

def get_chroma_client():
    """Get ChromaDB client with the current database path"""
    # Get the database path from your existing setup
    db_path = os.path.join(os.path.dirname(__file__), '..', 'vector-database', 'store')
    if not os.path.exists(db_path):
        db_path = os.path.join(os.path.dirname(__file__), '..', 'vector-database', 'store-new')
    
    print(f"Using ChromaDB path: {db_path}")
    return chromadb.PersistentClient(path=db_path)

def get_qdrant_client():
    """Get Qdrant client - local or cloud based on environment variables"""
    # Check if we have Qdrant cloud credentials
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    if qdrant_url and qdrant_api_key:
        print(f"Connecting to Qdrant cloud at {qdrant_url}")
        return QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    else:
        # Use local Qdrant instance
        print("Using local Qdrant instance at localhost:6333")
        return QdrantClient(host="localhost", port=6333)

def migrate_collection(chroma_client, qdrant_client, collection_name, batch_size=100):
    """Migrate a single collection from ChromaDB to Qdrant"""
    print(f"\nMigrating collection: {collection_name}")
    
    try:
        # Get the ChromaDB collection
        chroma_collection = chroma_client.get_collection(name=collection_name)
        
        # Get all items from the collection
        all_items = chroma_collection.get(include=['embeddings', 'documents', 'metadatas'])
        
        if not all_items['ids']:
            print(f"Collection {collection_name} is empty, skipping.")
            return 0
        
        # Get embedding dimension from the first item
        embedding_dim = len(all_items['embeddings'][0])
        print(f"Embedding dimension: {embedding_dim}")
        print(f"Total items to migrate: {len(all_items['ids'])}")
        
        # Create or recreate the collection in Qdrant
        try:
            qdrant_client.get_collection(collection_name)
            print(f"Collection {collection_name} already exists in Qdrant")
        except Exception:
            print(f"Creating collection {collection_name} in Qdrant")
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=embedding_dim,
                    distance=models.Distance.COSINE
                )
            )
        
        # Process in batches to avoid memory issues
        total_items = len(all_items['ids'])
        total_batches = (total_items + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(total_batches), desc="Migrating batches"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_items)
            
            batch_ids = all_items['ids'][start_idx:end_idx]
            batch_embeddings = all_items['embeddings'][start_idx:end_idx]
            batch_documents = all_items['documents'][start_idx:end_idx]
            batch_metadatas = all_items['metadatas'][start_idx:end_idx] if all_items['metadatas'] else [{}] * len(batch_ids)
            
            # Prepare points for Qdrant
            points = []
            for i, (doc_id, embedding, document, metadata) in enumerate(zip(batch_ids, batch_embeddings, batch_documents, batch_metadatas)):
                # Ensure metadata is a dictionary
                if metadata is None:
                    metadata = {}
                
                # Add the document text to the payload
                payload = {
                    "text": document,
                    **metadata
                }
                
                points.append(models.PointStruct(
                    id=doc_id,
                    vector=embedding,
                    payload=payload
                ))
            
            # Upload points to Qdrant
            qdrant_client.upsert_points(
                collection_name=collection_name,
                points=points
            )
        
        print(f"Successfully migrated {total_items} items to Qdrant collection {collection_name}")
        return total_items
    
    except Exception as e:
        print(f"Error migrating collection {collection_name}: {str(e)}")
        return 0

def migrate_all_collections(chroma_client, qdrant_client, specific_collection=None):
    """Migrate all collections from ChromaDB to Qdrant"""
    start_time = time.time()
    
    # Get all collections from ChromaDB
    collections = chroma_client.list_collections()
    
    if specific_collection:
        collections = [c for c in collections if c.name == specific_collection]
        if not collections:
            print(f"Collection {specific_collection} not found in ChromaDB")
            return
    
    print(f"Found {len(collections)} collections in ChromaDB")
    
    total_migrated = 0
    for collection in collections:
        migrated = migrate_collection(chroma_client, qdrant_client, collection.name)
        total_migrated += migrated
    
    end_time = time.time()
    print(f"\nMigration completed in {end_time - start_time:.2f} seconds")
    print(f"Total items migrated: {total_migrated}")

def verify_migration(chroma_client, qdrant_client, collection_name):
    """Verify that the migration was successful by comparing item counts"""
    print(f"\nVerifying migration for collection: {collection_name}")
    
    try:
        # Get the ChromaDB collection
        chroma_collection = chroma_client.get_collection(name=collection_name)
        chroma_count = chroma_collection.count()
        
        # Get the Qdrant collection
        qdrant_count = qdrant_client.count(collection_name=collection_name).count
        
        print(f"ChromaDB count: {chroma_count}")
        print(f"Qdrant count: {qdrant_count}")
        
        if chroma_count == qdrant_count:
            print("✅ Migration verified: Item counts match")
            return True
        else:
            print("❌ Migration verification failed: Item counts do not match")
            return False
    
    except Exception as e:
        print(f"Error verifying migration: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Migrate data from ChromaDB to Qdrant")
    parser.add_argument("--collection", help="Specific collection to migrate (default: all collections)")
    parser.add_argument("--verify", action="store_true", help="Verify migration after completion")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for migration (default: 100)")
    
    args = parser.parse_args()
    
    # Get clients
    chroma_client = get_chroma_client()
    qdrant_client = get_qdrant_client()
    
    # Perform migration
    migrate_all_collections(chroma_client, qdrant_client, args.collection)
    
    # Verify migration if requested
    if args.verify and args.collection:
        verify_migration(chroma_client, qdrant_client, args.collection)
    elif args.verify:
        for collection in chroma_client.list_collections():
            verify_migration(chroma_client, qdrant_client, collection.name)

if __name__ == "__main__":
    main()
