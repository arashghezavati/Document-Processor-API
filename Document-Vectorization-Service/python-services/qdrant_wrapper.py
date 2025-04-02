#!/usr/bin/env python3
"""
Qdrant wrapper to provide similar functionality to ChromaDB
"""

import os
import uuid
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from the main .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

class QdrantWrapper:
    """
    A wrapper for Qdrant that provides similar functionality to ChromaDB
    """
    
    def __init__(self):
        """Initialize the Qdrant client"""
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        if not self.qdrant_url or not self.qdrant_api_key:
            raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in environment variables")
        
        self.client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)
        logger.info(f"Connected to Qdrant at {self.qdrant_url}")
    
    def list_collections(self):
        """List all collections in Qdrant"""
        collections = self.client.get_collections().collections
        # Return objects with a name attribute to match ChromaDB's interface
        return [type('Collection', (), {'name': collection.name}) for collection in collections]
    
    def get_collection(self, name, embedding_function=None):
        """
        Get a collection by name
        
        Args:
            name (str): Name of the collection
            embedding_function: Function to generate embeddings (not used in retrieval)
            
        Returns:
            QdrantCollection: A wrapper around a Qdrant collection
        """
        try:
            # Check if collection exists
            try:
                self.client.get_collection(collection_name=name)
                return QdrantCollection(self.client, name, embedding_function)
            except Exception as e:
                # If there's a validation error but the collection exists, we can still proceed
                if "validation error" in str(e).lower() and "already exists" not in str(e).lower():
                    logger.warning(f"Validation error when getting collection {name}, but proceeding anyway: {str(e)}")
                    # Try to check if the collection exists by listing all collections
                    collections = self.client.get_collections().collections
                    collection_names = [collection.name for collection in collections]
                    if name in collection_names:
                        return QdrantCollection(self.client, name, embedding_function)
                logger.error(f"Error getting collection {name}: {str(e)}")
                raise ValueError(f"Collection {name} does not exist")
        except Exception as e:
            logger.error(f"Error getting collection {name}: {str(e)}")
            raise ValueError(f"Collection {name} does not exist")
    
    def get_or_create_collection(self, name, embedding_function=None):
        """
        Get a collection by name or create it if it doesn't exist
        
        Args:
            name (str): Name of the collection
            embedding_function: Function to generate embeddings
            
        Returns:
            QdrantCollection: A wrapper around a Qdrant collection
        """
        try:
            # Try to get the collection
            try:
                return self.get_collection(name, embedding_function)
            except ValueError:
                # Collection doesn't exist, create it
                pass
            
            # Determine vector size from environment variable
            try:
                vector_size = int(os.getenv("EMBEDDING_DIMENSION", "768"))
                logger.info(f"Using embedding dimension from environment: {vector_size}")
            except (TypeError, ValueError):
                vector_size = 768
                logger.warning(f"Could not parse EMBEDDING_DIMENSION, using default: {vector_size}")
            
            # Create the collection
            try:
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
                
                logger.info(f"Created collection {name} with vector size {vector_size}")
                return QdrantCollection(self.client, name, embedding_function)
            except Exception as e:
                # If the collection already exists, try to get it again
                if "already exists" in str(e).lower():
                    logger.info(f"Collection {name} already exists, trying to get it")
                    return QdrantCollection(self.client, name, embedding_function)
                
                logger.error(f"Error creating collection {name}: {str(e)}")
                # Try one more time to get the collection in case it was created by another process
                return QdrantCollection(self.client, name, embedding_function)
        except Exception as e:
            logger.error(f"Error in get_or_create_collection for {name}: {str(e)}")
            # As a last resort, return a QdrantCollection object anyway
            # This might work for operations that don't require the collection to exist
            return QdrantCollection(self.client, name, embedding_function)
    
    def add(self, documents, metadatas=None, ids=None, embeddings=None):
        """
        Add documents to the collection
        
        Args:
            documents (List[str]): List of documents to add
            metadatas (List[Dict], optional): List of metadata for each document
            ids (List[str], optional): List of IDs for each document
            embeddings (List[List[float]], optional): List of embeddings for each document
        """
        if not documents:
            return
        
        # Generate IDs if not provided
        if not ids:
            ids = [str(uuid.uuid4()) for _ in range(len(documents))]
        
        # Convert string IDs to UUIDs to ensure compatibility with Qdrant
        # Qdrant only accepts integers or UUIDs as IDs
        converted_ids = []
        for id_str in ids:
            try:
                # Try to parse as UUID first
                uuid_obj = uuid.UUID(id_str)
                converted_ids.append(str(uuid_obj))
            except ValueError:
                # If not a valid UUID, create a new UUID based on the string
                # This ensures consistent UUID generation for the same string
                uuid_obj = uuid.uuid5(uuid.NAMESPACE_DNS, id_str)
                converted_ids.append(str(uuid_obj))
        
        # Generate embeddings if not provided
        if not embeddings and self.embedding_function:
            embeddings = self.embedding_function(documents)
        
        if not embeddings:
            raise ValueError("Either embeddings must be provided or embedding_function must be set")
        
        # Ensure metadatas is a list of dictionaries
        if not metadatas:
            metadatas = [{} for _ in range(len(documents))]
        
        # Create points
        points = []
        for i, (doc_id, embedding, document, metadata) in enumerate(zip(converted_ids, embeddings, documents, metadatas)):
            # Add document to payload
            payload = {
                "text": document,
                **metadata
            }
            
            points.append(PointStruct(
                id=doc_id,
                vector=embedding,
                payload=payload
            ))
        
        try:
            # Add points to collection
            self.client.upsert(
                collection_name=self.name,
                points=points
            )
            
            logger.info(f"Added {len(documents)} documents to collection {self.name}")
        except Exception as e:
            logger.error(f"Error adding documents to collection {self.name}: {str(e)}")
            raise
    
    def query(self, query_embeddings=None, query_texts=None, n_results=10, where=None, include=None):
        """
        Query the collection
        
        Args:
            query_embeddings (List[List[float]], optional): List of query embeddings
            query_texts (List[str], optional): List of query texts
            n_results (int): Number of results to return
            where (Dict, optional): Filter condition
            include (List[str], optional): Fields to include in the response
            
        Returns:
            Dict: Query results
        """
        # Generate embeddings if not provided
        if not query_embeddings and query_texts and self.embedding_function:
            query_embeddings = self.embedding_function(query_texts)
        
        if not query_embeddings:
            raise ValueError("Either query_embeddings must be provided or query_texts and embedding_function must be set")
        
        # Convert where filter to Qdrant filter
        filter_query = None
        if where:
            conditions = []
            for key, value in where.items():
                conditions.append(FieldCondition(
                    key=key,
                    match=MatchValue(value=value)
                ))
            
            filter_query = Filter(
                must=conditions
            )
        
        # Execute query for each embedding
        results = {
            "ids": [],
            "distances": [],
            "metadatas": [],
            "documents": [],
            "embeddings": []
        }
        
        try:
            for embedding in query_embeddings:
                search_result = self.client.search(
                    collection_name=self.name,
                    query_vector=embedding,
                    limit=n_results,
                    with_payload=True,
                    with_vectors="include" in include and "embeddings" in include,
                    query_filter=filter_query  # Use query_filter instead of filter
                )
                
                # Extract results
                batch_ids = []
                batch_distances = []
                batch_metadatas = []
                batch_documents = []
                batch_embeddings = []
                
                for point in search_result:
                    batch_ids.append(point.id)
                    batch_distances.append(1.0 - point.score)  # Convert similarity to distance
                    
                    # Extract metadata and document from payload
                    if hasattr(point, 'payload') and point.payload:
                        metadata = {k: v for k, v in point.payload.items() if k != "text"}
                        batch_metadatas.append(metadata)
                        
                        if "documents" in include or not include:
                            batch_documents.append(point.payload.get("text", ""))
                    else:
                        # Handle case where payload is missing
                        batch_metadatas.append({})
                        if "documents" in include or not include:
                            batch_documents.append("")
                    
                    if ("include" in include and "embeddings" in include) and hasattr(point, "vector"):
                        batch_embeddings.append(point.vector)
                
                results["ids"].append(batch_ids)
                results["distances"].append(batch_distances)
                results["metadatas"].append(batch_metadatas)
                results["documents"].append(batch_documents)
                
                if batch_embeddings:
                    results["embeddings"].append(batch_embeddings)
        except Exception as e:
            logger.error(f"Error in query method: {str(e)}")
            # If there's an error, return empty results
            return {
                "ids": [[]],
                "distances": [[]],
                "metadatas": [[]],
                "documents": [[]],
                "embeddings": [[]] if "embeddings" in include else None
            }
        
        return results
    
    def get(self, ids=None, where=None, limit=None, include=None):
        """
        Get documents from the collection
        
        Args:
            ids (List[str], optional): List of IDs to retrieve
            where (Dict, optional): Filter condition
            limit (int, optional): Maximum number of results to return
            include (List[str], optional): Fields to include in the response
            
        Returns:
            Dict: Retrieved documents
        """
        if not include:
            include = ["documents", "metadatas"]
        
        try:
            # Convert where filter to Qdrant filter
            filter_query = None
            if where:
                conditions = []
                for key, value in where.items():
                    conditions.append(FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    ))
                
                filter_query = Filter(
                    must=conditions
                )
            
            # If IDs are provided, add them to the filter
            if ids:
                # Convert string IDs to UUIDs
                converted_ids = []
                for id_str in ids:
                    try:
                        # Try to parse as UUID first
                        uuid_obj = uuid.UUID(id_str)
                        converted_ids.append(str(uuid_obj))
                    except ValueError:
                        # If not a valid UUID, create a new UUID based on the string
                        uuid_obj = uuid.uuid5(uuid.NAMESPACE_DNS, id_str)
                        converted_ids.append(str(uuid_obj))
                
                id_conditions = [FieldCondition(
                    key="id",
                    match=MatchValue(value=id_)
                ) for id_ in converted_ids]
                
                if filter_query:
                    filter_query.must.extend(id_conditions)
                else:
                    filter_query = Filter(must=id_conditions)
            
            points = []
            
            # If we have a filter or IDs, use search instead of scroll
            if filter_query:
                try:
                    # Use search with a dummy vector that will match everything
                    # This is a workaround to get filtered results
                    vector_size = int(os.getenv("EMBEDDING_DIMENSION", "768"))
                    dummy_vector = [0.0] * vector_size
                    
                    # Set a high limit to get all matching documents
                    search_limit = limit if limit else 10000
                    
                    search_result = self.client.search(
                        collection_name=self.name,
                        query_vector=dummy_vector,
                        query_filter=filter_query,
                        limit=search_limit,
                        with_payload=True,
                        with_vectors="embeddings" in include
                    )
                    
                    points.extend(search_result)
                except Exception as search_error:
                    logger.error(f"Error during search operation: {str(search_error)}")
                    # Fall back to empty results
                    points = []
            else:
                # If no filter, use scroll to get all documents
                offset = None
                try:
                    while True:
                        scroll_result = self.client.scroll(
                            collection_name=self.name,
                            limit=100,  # Batch size
                            with_payload=True,
                            with_vectors="embeddings" in include,
                            offset=offset
                        )
                        
                        batch_points, offset = scroll_result
                        points.extend(batch_points)
                        
                        # Stop if we've reached the limit or there are no more points
                        if limit and len(points) >= limit:
                            points = points[:limit]
                            break
                        
                        if not offset:
                            break
                except Exception as scroll_error:
                    logger.error(f"Error during scroll operation: {str(scroll_error)}")
                    # If scroll fails but we have some points, continue with what we have
                    if not points:
                        # Fall back to empty results
                        points = []
            
            # Extract results
            result_ids = []
            result_documents = []
            result_metadatas = []
            result_embeddings = []
            
            for point in points:
                result_ids.append(point.id)
                
                # Extract metadata and document from payload
                if hasattr(point, 'payload') and point.payload:
                    metadata = {k: v for k, v in point.payload.items() if k != "text"}
                    result_metadatas.append(metadata)
                    
                    if "documents" in include:
                        result_documents.append(point.payload.get("text", ""))
                else:
                    # Handle case where payload is missing
                    result_metadatas.append({})
                    if "documents" in include:
                        result_documents.append("")
                
                if "embeddings" in include and hasattr(point, "vector"):
                    result_embeddings.append(point.vector)
            
            # Construct result
            result = {
                "ids": result_ids,
            }
            
            if "documents" in include:
                result["documents"] = result_documents
            
            if result_metadatas:
                result["metadatas"] = result_metadatas
            
            if result_embeddings:
                result["embeddings"] = result_embeddings
            
            # If no documents were found, return empty arrays for consistency
            if not result_ids:
                result = {
                    "ids": [],
                    "metadatas": [] if "metadatas" in include or not include else None,
                    "documents": [] if "documents" in include else None,
                    "embeddings": [] if "embeddings" in include else None
                }
            
            return result
        
        except Exception as e:
            logger.error(f"Error in get method: {str(e)}")
            # Return empty result on error rather than propagating exception
            return {
                "ids": [],
                "metadatas": [] if "metadatas" in include or not include else None,
                "documents": [] if "documents" in include else None,
                "embeddings": [] if "embeddings" in include else None
            }
    
    def count(self):
        """
        Get the number of documents in the collection
        
        Returns:
            int: Number of documents
        """
        try:
            result = self.client.count(collection_name=self.name)
            return result.count
        except Exception as e:
            logger.error(f"Error counting documents in collection {self.name}: {str(e)}")
            return 0
    
    def delete(self, ids=None, where=None):
        """
        Delete documents from the collection
        
        Args:
            ids (List[str], optional): List of IDs to delete
            where (Dict, optional): Filter condition
        """
        # Convert where filter to Qdrant filter
        filter_query = None
        if where:
            conditions = []
            for key, value in where.items():
                conditions.append(FieldCondition(
                    key=key,
                    match=MatchValue(value=value)
                ))
            
            filter_query = Filter(
                must=conditions
            )
        
        # Delete by IDs
        if ids:
            self.client.delete(
                collection_name=self.name,
                points_selector=models.PointIdsList(
                    points=ids
                )
            )
            logger.info(f"Deleted {len(ids)} documents from collection {self.name}")
        
        # Delete by filter
        elif filter_query:
            self.client.delete(
                collection_name=self.name,
                points_selector=models.FilterSelector(
                    filter=filter_query
                )
            )
            logger.info(f"Deleted documents matching filter from collection {self.name}")

class QdrantCollection:
    """
    A wrapper around a Qdrant collection that provides similar functionality to ChromaDB's Collection
    """
    
    def __init__(self, client, name, embedding_function=None):
        """
        Initialize the collection
        
        Args:
            client (QdrantClient): Qdrant client
            name (str): Name of the collection
            embedding_function: Function to generate embeddings
        """
        self.client = client
        self.name = name
        self.embedding_function = embedding_function
    
    def add(self, documents, metadatas=None, ids=None, embeddings=None):
        """
        Add documents to the collection
        
        Args:
            documents (List[str]): List of documents to add
            metadatas (List[Dict], optional): List of metadata for each document
            ids (List[str], optional): List of IDs for each document
            embeddings (List[List[float]], optional): List of embeddings for each document
        """
        if not documents:
            return
        
        # Generate IDs if not provided
        if not ids:
            ids = [str(uuid.uuid4()) for _ in range(len(documents))]
        
        # Convert string IDs to UUIDs to ensure compatibility with Qdrant
        # Qdrant only accepts integers or UUIDs as IDs
        converted_ids = []
        for id_str in ids:
            try:
                # Try to parse as UUID first
                uuid_obj = uuid.UUID(id_str)
                converted_ids.append(str(uuid_obj))
            except ValueError:
                # If not a valid UUID, create a new UUID based on the string
                # This ensures consistent UUID generation for the same string
                uuid_obj = uuid.uuid5(uuid.NAMESPACE_DNS, id_str)
                converted_ids.append(str(uuid_obj))
        
        # Generate embeddings if not provided
        if not embeddings and self.embedding_function:
            embeddings = self.embedding_function(documents)
        
        if not embeddings:
            raise ValueError("Either embeddings must be provided or embedding_function must be set")
        
        # Ensure metadatas is a list of dictionaries
        if not metadatas:
            metadatas = [{} for _ in range(len(documents))]
        
        # Create points
        points = []
        for i, (doc_id, embedding, document, metadata) in enumerate(zip(converted_ids, embeddings, documents, metadatas)):
            # Add document to payload
            payload = {
                "text": document,
                **metadata
            }
            
            points.append(PointStruct(
                id=doc_id,
                vector=embedding,
                payload=payload
            ))
        
        try:
            # Add points to collection
            self.client.upsert(
                collection_name=self.name,
                points=points
            )
            
            logger.info(f"Added {len(documents)} documents to collection {self.name}")
        except Exception as e:
            logger.error(f"Error adding documents to collection {self.name}: {str(e)}")
            raise
    
    def query(self, query_embeddings=None, query_texts=None, n_results=10, where=None, include=None):
        """
        Query the collection
        
        Args:
            query_embeddings (List[List[float]], optional): List of query embeddings
            query_texts (List[str], optional): List of query texts
            n_results (int): Number of results to return
            where (Dict, optional): Filter condition
            include (List[str], optional): Fields to include in the response
            
        Returns:
            Dict: Query results
        """
        # Generate embeddings if not provided
        if not query_embeddings and query_texts and self.embedding_function:
            query_embeddings = self.embedding_function(query_texts)
        
        if not query_embeddings:
            raise ValueError("Either query_embeddings must be provided or query_texts and embedding_function must be set")
        
        # Convert where filter to Qdrant filter
        filter_query = None
        if where:
            conditions = []
            for key, value in where.items():
                conditions.append(FieldCondition(
                    key=key,
                    match=MatchValue(value=value)
                ))
            
            filter_query = Filter(
                must=conditions
            )
        
        # Execute query for each embedding
        results = {
            "ids": [],
            "distances": [],
            "metadatas": [],
            "documents": [],
            "embeddings": []
        }
        
        try:
            for embedding in query_embeddings:
                search_result = self.client.search(
                    collection_name=self.name,
                    query_vector=embedding,
                    limit=n_results,
                    with_payload=True,
                    with_vectors="include" in include and "embeddings" in include,
                    query_filter=filter_query  # Use query_filter instead of filter
                )
                
                # Extract results
                batch_ids = []
                batch_distances = []
                batch_metadatas = []
                batch_documents = []
                batch_embeddings = []
                
                for point in search_result:
                    batch_ids.append(point.id)
                    batch_distances.append(1.0 - point.score)  # Convert similarity to distance
                    
                    # Extract metadata and document from payload
                    if hasattr(point, 'payload') and point.payload:
                        metadata = {k: v for k, v in point.payload.items() if k != "text"}
                        batch_metadatas.append(metadata)
                        
                        if "documents" in include or not include:
                            batch_documents.append(point.payload.get("text", ""))
                    else:
                        # Handle case where payload is missing
                        batch_metadatas.append({})
                        if "documents" in include or not include:
                            batch_documents.append("")
                    
                    if ("include" in include and "embeddings" in include) and hasattr(point, "vector"):
                        batch_embeddings.append(point.vector)
                
                results["ids"].append(batch_ids)
                results["distances"].append(batch_distances)
                results["metadatas"].append(batch_metadatas)
                results["documents"].append(batch_documents)
                
                if batch_embeddings:
                    results["embeddings"].append(batch_embeddings)
        except Exception as e:
            logger.error(f"Error in query method: {str(e)}")
            # If there's an error, return empty results
            return {
                "ids": [[]],
                "distances": [[]],
                "metadatas": [[]],
                "documents": [[]],
                "embeddings": [[]] if "embeddings" in include else None
            }
        
        return results
    
    def get(self, ids=None, where=None, limit=None, include=None):
        """
        Get documents from the collection
        
        Args:
            ids (List[str], optional): List of IDs to retrieve
            where (Dict, optional): Filter condition
            limit (int, optional): Maximum number of results to return
            include (List[str], optional): Fields to include in the response
            
        Returns:
            Dict: Retrieved documents
        """
        if not include:
            include = ["documents", "metadatas"]
        
        try:
            # Convert where filter to Qdrant filter
            filter_query = None
            if where:
                conditions = []
                for key, value in where.items():
                    conditions.append(FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    ))
                
                filter_query = Filter(
                    must=conditions
                )
            
            # If IDs are provided, add them to the filter
            if ids:
                # Convert string IDs to UUIDs
                converted_ids = []
                for id_str in ids:
                    try:
                        # Try to parse as UUID first
                        uuid_obj = uuid.UUID(id_str)
                        converted_ids.append(str(uuid_obj))
                    except ValueError:
                        # If not a valid UUID, create a new UUID based on the string
                        uuid_obj = uuid.uuid5(uuid.NAMESPACE_DNS, id_str)
                        converted_ids.append(str(uuid_obj))
                
                id_conditions = [FieldCondition(
                    key="id",
                    match=MatchValue(value=id_)
                ) for id_ in converted_ids]
                
                if filter_query:
                    filter_query.must.extend(id_conditions)
                else:
                    filter_query = Filter(must=id_conditions)
            
            points = []
            
            # If we have a filter or IDs, use search instead of scroll
            if filter_query:
                try:
                    # Use search with a dummy vector that will match everything
                    # This is a workaround to get filtered results
                    vector_size = int(os.getenv("EMBEDDING_DIMENSION", "768"))
                    dummy_vector = [0.0] * vector_size
                    
                    # Set a high limit to get all matching documents
                    search_limit = limit if limit else 10000
                    
                    search_result = self.client.search(
                        collection_name=self.name,
                        query_vector=dummy_vector,
                        query_filter=filter_query,
                        limit=search_limit,
                        with_payload=True,
                        with_vectors="embeddings" in include
                    )
                    
                    points.extend(search_result)
                except Exception as search_error:
                    logger.error(f"Error during search operation: {str(search_error)}")
                    # Fall back to empty results
                    points = []
            else:
                # If no filter, use scroll to get all documents
                offset = None
                try:
                    while True:
                        scroll_result = self.client.scroll(
                            collection_name=self.name,
                            limit=100,  # Batch size
                            with_payload=True,
                            with_vectors="embeddings" in include,
                            offset=offset
                        )
                        
                        batch_points, offset = scroll_result
                        points.extend(batch_points)
                        
                        # Stop if we've reached the limit or there are no more points
                        if limit and len(points) >= limit:
                            points = points[:limit]
                            break
                        
                        if not offset:
                            break
                except Exception as scroll_error:
                    logger.error(f"Error during scroll operation: {str(scroll_error)}")
                    # If scroll fails but we have some points, continue with what we have
                    if not points:
                        # Fall back to empty results
                        points = []
            
            # Extract results
            result_ids = []
            result_documents = []
            result_metadatas = []
            result_embeddings = []
            
            for point in points:
                result_ids.append(point.id)
                
                # Extract metadata and document from payload
                if hasattr(point, 'payload') and point.payload:
                    metadata = {k: v for k, v in point.payload.items() if k != "text"}
                    result_metadatas.append(metadata)
                    
                    if "documents" in include:
                        result_documents.append(point.payload.get("text", ""))
                else:
                    # Handle case where payload is missing
                    result_metadatas.append({})
                    if "documents" in include:
                        result_documents.append("")
                
                if "embeddings" in include and hasattr(point, "vector"):
                    result_embeddings.append(point.vector)
            
            # Construct result
            result = {
                "ids": result_ids,
            }
            
            if "documents" in include:
                result["documents"] = result_documents
            
            if result_metadatas:
                result["metadatas"] = result_metadatas
            
            if result_embeddings:
                result["embeddings"] = result_embeddings
            
            # If no documents were found, return empty arrays for consistency
            if not result_ids:
                result = {
                    "ids": [],
                    "metadatas": [] if "metadatas" in include or not include else None,
                    "documents": [] if "documents" in include else None,
                    "embeddings": [] if "embeddings" in include else None
                }
            
            return result
        
        except Exception as e:
            logger.error(f"Error in get method: {str(e)}")
            # Return empty result on error rather than propagating exception
            return {
                "ids": [],
                "metadatas": [] if "metadatas" in include or not include else None,
                "documents": [] if "documents" in include else None,
                "embeddings": [] if "embeddings" in include else None
            }
    
    def count(self):
        """
        Get the number of documents in the collection
        
        Returns:
            int: Number of documents
        """
        try:
            result = self.client.count(collection_name=self.name)
            return result.count
        except Exception as e:
            logger.error(f"Error counting documents in collection {self.name}: {str(e)}")
            return 0
    
    def delete(self, ids=None, where=None):
        """
        Delete documents from the collection
        
        Args:
            ids (List[str], optional): List of IDs to delete
            where (Dict, optional): Filter condition
        """
        # Convert where filter to Qdrant filter
        filter_query = None
        if where:
            conditions = []
            for key, value in where.items():
                conditions.append(FieldCondition(
                    key=key,
                    match=MatchValue(value=value)
                ))
            
            filter_query = Filter(
                must=conditions
            )
        
        # Delete by IDs
        if ids:
            self.client.delete(
                collection_name=self.name,
                points_selector=models.PointIdsList(
                    points=ids
                )
            )
            logger.info(f"Deleted {len(ids)} documents from collection {self.name}")
        
        # Delete by filter
        elif filter_query:
            self.client.delete(
                collection_name=self.name,
                points_selector=models.FilterSelector(
                    filter=filter_query
                )
            )
            logger.info(f"Deleted documents matching filter from collection {self.name}")

# Create a singleton instance
def get_qdrant_client():
    """Get a QdrantWrapper instance"""
    return QdrantWrapper()
