import os
import google.generativeai as genai
import shutil
import tempfile
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, status, BackgroundTasks, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, HttpUrl
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
import importlib.util
from qdrant_wrapper import get_qdrant_client
from websocket_manager import manager, notify_document_change, EventTypes
from datetime import datetime, timedelta

# Import authentication module
from auth import (
    User, UserCreate, Token, 
    authenticate_user, create_user, create_access_token,
    get_current_user, create_folder, get_user_folders
)

# Load environment variables
load_dotenv()
GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Debugging: Check API Key and Model
print("GOOGLE_GEMINI_API_KEY:", GOOGLE_GEMINI_API_KEY)
print("GEMINI_MODEL:", GEMINI_MODEL)

# Configure Gemini AI
genai.configure(api_key=GOOGLE_GEMINI_API_KEY)

# Initialize FastAPI app
app = FastAPI(
    title="Document Processor API",
    description="API for processing and querying documents",
    version="1.0.0",
    root_path=""  # Explicitly set empty root path for Render deployment
)

# Add middleware to handle proxy headers from Render
from fastapi.middleware.trustedhost import TrustedHostMiddleware
app.add_middleware(
    TrustedHostMiddleware, allowed_hosts=["*"]
)

# Configure CORS - Updated to explicitly allow localhost and Render domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now to debug
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,
)

# Add CORS headers directly to responses as a fallback
@app.middleware("http")
async def add_cors_headers(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

# Initialize Qdrant Client
qdrant_client = get_qdrant_client()

# Define request models
class ChatRequest(BaseModel):
    query: str
    folder_name: Optional[str] = None
    document_name: Optional[str] = None
    mode: str = "strict"  # Optional: strict or comprehensive
    conversation_id: Optional[str] = None  # Add conversation ID

class FolderCreate(BaseModel):
    folder_name: str

class UrlProcessRequest(BaseModel):
    url: HttpUrl
    folder_name: Optional[str] = None
    document_name: Optional[str] = None
    follow_links: bool = True
    max_links: int = 5

class BatchUrlProcessRequest(BaseModel):
    urls: List[HttpUrl]
    folder_name: Optional[str] = None
    follow_links: bool = True
    max_links: int = 3

def get_user_collection_name(username: str):
    """
    Generate a collection name for a specific user.
    """
    return f"user_{username}_docs"

def retrieve_documents(username: str, folder_name: Optional[str] = None, document_name: Optional[str] = None, query: Optional[str] = None):
    """
    Fetches documents from Qdrant based on user, folder, and document filters.
    If query is provided, performs semantic search to find relevant documents.
    """
    try:
        # Get user's collection
        collection_name = get_user_collection_name(username)
        collection = qdrant_client.get_or_create_collection(name=collection_name)
        
        # Build query filter based on folder and document
        where_filter = {}
        if folder_name:
            where_filter["folder_name"] = folder_name
        if document_name:
            where_filter["document_name"] = document_name
        
        # If query is provided, perform semantic search
        if query:
            # Import embedding function
            from embedding_function import GeminiEmbeddingFunction
            embedding_func = GeminiEmbeddingFunction(api_key=GOOGLE_GEMINI_API_KEY)
            
            # Convert query to embedding
            query_embedding = embedding_func([query])[0]
            
            # Perform semantic search with metadata filtering
            try:
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=5,  # Return top 5 most similar chunks
                    where=where_filter if where_filter else None,
                    include=["documents", "metadatas", "distances"]
                )
                
                if not results or not results["documents"] or len(results["documents"][0]) == 0:
                    print(f"⚠️ No semantic search results found for query: {query}")
                    return None  # No documents found
                
                # Join the most relevant documents into a single string
                documents_text = "\n\n".join(results["documents"][0])
                print(f"🔍 Retrieved Documents via semantic search for {username}: {len(results['documents'][0])} chunks")
                return documents_text
                
            except Exception as e:
                print(f"❌ Error in semantic search: {str(e)}")
                # Fall back to metadata-only retrieval if semantic search fails
                print("⚠️ Falling back to metadata-only retrieval")
                
                # Continue with regular metadata filtering below
        
        # Query documents with filter (used when no query is provided or as fallback)
        if where_filter:
            docs = collection.get(where=where_filter, include=["documents", "metadatas"])
        else:
            docs = collection.get(include=["documents", "metadatas"])
        
        if not docs or not docs["documents"]:
            return None  # No documents found
        
        # Join all documents into a single string
        documents_text = "\n\n".join(docs["documents"])
        print(f"🔍 Retrieved Documents for {username} via metadata filter: {len(docs['documents'])} chunks")
        return documents_text
    
    except Exception as e:
        print(f"❌ Error retrieving documents: {str(e)}")
        return None

def call_ai_model(prompt: str):
    """
    Sends a query to Google Gemini AI and retrieves a response.
    """
    try:
        # Initialize the correct model
        model = genai.GenerativeModel(GEMINI_MODEL)

        # Generate AI response
        response = model.generate_content(prompt)

        # Return AI response text without unnecessary introductions
        return response.text.strip() if response and response.text else "⚠️ AI did not return a response."

    except Exception as e:
        print(f"❌ AI Model Error: {str(e)}")
        return "⚠️ AI service error. Please try again."

# Dynamically import process_document module
spec = importlib.util.spec_from_file_location("process_document", os.path.join(os.path.dirname(__file__), "process_document.py"))
process_document = importlib.util.module_from_spec(spec)
spec.loader.exec_module(process_document)

# Authentication endpoints
@app.options("/signin")
async def signin_options():
    """
    Handle preflight OPTIONS requests for the signin endpoint
    """
    return {}

@app.options("/signup")
async def signup_options():
    """
    Handle preflight OPTIONS requests for the signup endpoint
    """
    return {}

@app.post("/signup", response_model=User)
async def signup_endpoint(user_data: UserCreate):
    """
    Register a new user
    """
    user = create_user(user_data)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username or email already registered"
        )
    return user

@app.post("/signin", response_model=Token)
async def signin_endpoint(username: str = Form(...), password: str = Form(...)):
    """
    Authenticate a user and return a token
    """
    user = authenticate_user(username, password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/me", response_model=User)
async def get_user_info(current_user: User = Depends(get_current_user)):
    """
    Get information about the currently authenticated user
    """
    return current_user

# Folder management endpoints
@app.post("/folders")
async def create_folder_endpoint(
    folder_data: FolderCreate,
    current_user: User = Depends(get_current_user)
):
    """
    Create a new folder for the current user
    """
    success = create_folder(current_user.username, folder_data.folder_name)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Folder already exists or user not found"
        )
    
    return {"status": "success", "message": f"Folder '{folder_data.folder_name}' created successfully"}

@app.get("/folders")
async def get_folders_endpoint(current_user: User = Depends(get_current_user)):
    """
    Get all folders for the current user
    """
    folders = get_user_folders(current_user.username)
    return {"folders": folders}

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    folder_name: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user)
):
    """
    Upload and process a document into a user's collection
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    # Create a temporary file to store the uploaded content
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, file.filename)
    
    try:
        # Save the uploaded file to the temporary location
        with open(temp_file_path, "wb") as f:
            contents = await file.read()
            f.write(contents)
        
        # Get the user's collection name
        collection_name = get_user_collection_name(current_user.username)
        
        # Prepare metadata
        metadata = {
            "username": current_user.username,
        }
        
        if folder_name:
            metadata["folder_name"] = folder_name
        
        # Process the document
        process_document.process_document(
            file_path=temp_file_path,
            collection_name=collection_name,
            metadata=metadata
        )
        
        # Notify connected clients about the new document
        document_info = {
            "name": file.filename,
            "folder": folder_name
        }
        
        # Use asyncio.create_task to run the notification in the background
        import asyncio
        asyncio.create_task(notify_document_change(
            current_user.username, 
            EventTypes.DOCUMENT_ADDED, 
            document_info
        ))
        
        return {"status": "success", "message": f"Document '{file.filename}' processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
    finally:
        # Clean up temporary files
        shutil.rmtree(temp_dir, ignore_errors=True)

@app.post("/upload-multiple")
async def upload_multiple_documents(
    files: List[UploadFile] = File(...),
    folder_name: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user)
):
    """
    Upload and process multiple documents into a user's collection
    """
    results = []
    
    try:
        # Validate folder if provided
        if folder_name and folder_name not in get_user_folders(current_user.username):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Folder '{folder_name}' does not exist"
            )
        
        # Process each file
        for file in files:
            try:
                # Create temp file
                temp_dir = tempfile.mkdtemp()
                temp_file_path = os.path.join(temp_dir, file.filename)
                
                # Save uploaded file
                with open(temp_file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                
                # Process the document with user-specific collection
                collection_name = get_user_collection_name(current_user.username)
                
                # Add metadata for folder and document name
                metadata = {
                    "document_name": file.filename
                }
                if folder_name:
                    metadata["folder_name"] = folder_name
                    
                # Process the document
                process_document.process_document(
                    temp_file_path, 
                    collection_name, 
                    metadata=metadata
                )
                
                # Clean up
                shutil.rmtree(temp_dir)
                
                results.append({
                    "filename": file.filename,
                    "status": "success",
                    "message": f"Document {file.filename} successfully processed" + 
                              (f" into folder {folder_name}" if folder_name else "")
                })
                
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "message": str(e)
                })
        
        return {"results": results}
    
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/documents")
async def get_documents(
    folder_name: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Get all documents for the current user, optionally filtered by folder
    """
    try:
        # Get user's collection
        collection_name = get_user_collection_name(current_user.username)
        collection = qdrant_client.get_or_create_collection(name=collection_name)
        
        # Build query filter based on folder
        where_filter = {}
        if folder_name:
            where_filter["folder_name"] = folder_name
        
        # Query documents with filter
        try:
            if where_filter:
                results = collection.get(where=where_filter, include=["metadatas"])
            else:
                results = collection.get(include=["metadatas"])
        except Exception as query_error:
            print(f"DEBUG ERROR: Error during query: {str(query_error)}")
            raise query_error
        
        # Extract document names from metadata and deduplicate
        documents = []
        seen_documents = set()  # Track unique document names
        
        if results and "metadatas" in results and results["metadatas"]:
            for metadata in results["metadatas"]:
                if "document_name" in metadata:
                    doc_name = metadata["document_name"]
                    folder = metadata.get("folder_name", None)
                    
                    # Create a unique key for each document+folder combination
                    doc_key = f"{doc_name}|{folder}"
                    
                    # Only add if we haven't seen this document before
                    if doc_key not in seen_documents:
                        seen_documents.add(doc_key)
                        documents.append({
                            "name": doc_name,
                            "folder": folder
                        })
        
        return {"documents": documents}
    except Exception as e:
        print(f"DEBUG CRITICAL ERROR: {str(e)}")
        print(f"DEBUG ERROR TYPE: {type(e)}")
        import traceback
        print(f"DEBUG TRACEBACK: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving documents: {str(e)}"
        )

@app.post("/chat")
async def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user)  # Ensure the user is authenticated
):
    """
    Receives chat message and returns AI response based on user's documents.
    """
    query = request.query
    folder_name = request.folder_name
    document_name = request.document_name
    mode = request.mode
    conversation_id = request.conversation_id or f"{current_user.username}_default"

    # Retrieve conversation history
    history = conversation_memory.get(conversation_id, [])

    # Retrieve documents based on filters AND query for semantic search
    customer_data = retrieve_documents(
        current_user.username,
        folder_name,
        document_name,
        query  # Pass the query for semantic search
    )

    if not customer_data:
        scope_description = "your documents"
        if folder_name:
            scope_description = f"folder '{folder_name}'"
        if document_name:
            scope_description = f"document '{document_name}'"
            
        return {
            "response": f"⚠️ No relevant documents found in {scope_description}. Please ensure data is uploaded."
        }

    # Construct AI prompt with conversation history
    history_text = "\n".join([f"User: {h['query']}\nAI: {h['response']}" for h in history])
    prompt = (
        f"{history_text}\n\n"
        f"{customer_data}\n\n"
        f"Act as an immigration consultant and answer the question carefully. Provide different formats of answers based on the question. "
        f"For example, if the question asks for a roadmap, provide a roadmap format answer. "
        f"If the question asks for which program you are eligible or fit for, consider all programs not just one."
        f"if the question is like that it could have multiple answers, provide all possible answers, but explain each answer and explain them why."
        f"do not start your answer with Based on the provided document, just tell your answer"
        f"Answer concisely and directly based on the provided data. "
        f"Just return the response for the following request:\n\n"
        f"Query: {query}"
    )

    print(f"📩 AI Debug - Constructed Prompt: {prompt}")  # Debugging

    # Get AI response
    ai_response = call_ai_model(prompt)

    # Update conversation history
    history.append({"query": query, "response": ai_response})
    conversation_memory[conversation_id] = history[-10:]  # Keep only the last 10 exchanges

    return {"response": ai_response, "conversation_id": conversation_id}

@app.post("/chat/clear")
async def clear_chat(
    conversation_id: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Clears the conversation memory for the given conversation ID.
    """
    global conversation_memory
    conversation_id = conversation_id or f"{current_user.username}_default"

    if conversation_id in conversation_memory:
        del conversation_memory[conversation_id]
        return {"status": "success", "message": f"Conversation memory for '{conversation_id}' cleared."}
    else:
        return {"status": "success", "message": f"No conversation memory found for '{conversation_id}'."}

# Delete a document
@app.delete("/documents/{document_name}")
async def delete_document(document_name: str, current_user: User = Depends(get_current_user)):
    """
    Delete a document from a user's collection
    """
    try:
        # Get user's collection
        collection_name = get_user_collection_name(current_user.username)
        
        # Check if collection exists
        try:
            collection = qdrant_client.get_collection(collection_name)
        except Exception as e:
            print(f"Error getting collection: {str(e)}")
            raise HTTPException(status_code=404, detail="User collection not found")
        
        # Get all documents to find the one to delete
        results = collection.get(include=["metadatas"])
        
        if not results or not results["metadatas"]:
            raise HTTPException(status_code=404, detail=f"No documents found for user {current_user.username}")
        
        # Find documents with matching name
        documents_to_delete = []
        print(f"Available documents: {[meta.get('document_name') for meta in results['metadatas']]}")
        
        for i, metadata in enumerate(results["metadatas"]):
            document_name_in_db = metadata.get("document_name")
            print(f"Comparing: '{document_name_in_db}' with '{document_name}'")
            
            if document_name_in_db == document_name:
                print(f"Match found at index {i}")
                documents_to_delete.append(results["ids"][i])
        
        if not documents_to_delete:
            raise HTTPException(status_code=404, detail=f"Document '{document_name}' not found")
        
        # Delete the document
        for doc_id in documents_to_delete:
            print(f"Deleting document with ID: {doc_id}")
            collection.delete(ids=[doc_id])
        
        # Notify connected clients about the deleted document
        document_info = {
            "name": document_name
        }
        
        # Use asyncio.create_task to run the notification in the background
        import asyncio
        asyncio.create_task(notify_document_change(
            current_user.username, 
            EventTypes.DOCUMENT_DELETED, 
            document_info
        ))
        
        return {"status": "success", "message": f"Document '{document_name}' deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

# Delete a folder and all its documents
@app.delete("/folders/{folder_name}")
async def delete_folder(folder_name: str, current_user: User = Depends(get_current_user)):
    """
    Delete a folder and all its documents
    """
    try:
        # First, delete the folder from MongoDB
        try:
            # Check if the folder exists
            user_folders = get_user_folders(current_user.username)
            folder_exists = False
            
            for folder in user_folders:
                if folder["name"] == folder_name:
                    folder_exists = True
                    break
            
            if not folder_exists:
                raise HTTPException(status_code=404, detail=f"Folder '{folder_name}' not found")
            
            # Delete the folder from MongoDB
            from pymongo import MongoClient
            
            # Get MongoDB connection string from environment
            mongodb_uri = os.getenv("MONGODB_URI")
            if not mongodb_uri:
                raise HTTPException(status_code=500, detail="MongoDB connection string not found in environment")
            
            client = MongoClient(mongodb_uri)
            db = client.get_default_database()
            
            # Delete the folder
            result = db.folders.delete_one({
                "username": current_user.username,
                "name": folder_name
            })
            
            if result.deleted_count == 0:
                raise HTTPException(status_code=404, detail=f"Folder '{folder_name}' not found in database")
            
        except Exception as mongo_error:
            print(f"Error updating MongoDB: {str(mongo_error)}")
            # Continue with document deletion even if folder deletion fails
        
        # Now delete all documents in this folder from Qdrant
        try:
            # Check if collection exists first
            collections = qdrant_client.list_collections()
            collection_exists = False
            collection_name = get_user_collection_name(current_user.username)
            
            for coll in collections:
                if coll.name == collection_name:
                    collection_exists = True
                    break
            
            if not collection_exists:
                # If the collection doesn't exist, we can consider the folder deleted
                return {"message": f"Folder '{folder_name}' deleted successfully (no collection found)"}
            
            collection = qdrant_client.get_collection(collection_name)
            
            # Get all documents
            documents = collection.get()
            
            if not documents or not documents["metadatas"]:
                # No documents found, folder is effectively deleted
                return {"message": f"Folder '{folder_name}' deleted successfully (no documents found)"}
            
            # Find documents in the folder to delete
            documents_to_delete = []
            
            for i, metadata in enumerate(documents["metadatas"]):
                if metadata.get("folder_name") == folder_name:
                    documents_to_delete.append(documents["ids"][i])
            
            if documents_to_delete:
                # Delete the documents
                collection.delete(ids=documents_to_delete)
                print(f"Deleted {len(documents_to_delete)} documents from folder '{folder_name}'")
            
        except Exception as chroma_error:
            print(f"Error deleting documents from Qdrant: {str(chroma_error)}")
            # Continue with folder deletion even if document deletion fails
        
        # Notify connected clients about the deleted folder
        folder_info = {
            "name": folder_name
        }
        
        # Use asyncio.create_task to run the notification in the background
        import asyncio
        asyncio.create_task(notify_document_change(
            current_user.username, 
            EventTypes.FOLDER_DELETED, 
            folder_info
        ))
        
        return {"message": f"Folder '{folder_name}' and all its documents deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting folder: {str(e)}")

@app.post("/process_url")
async def process_url_endpoint(
    request: UrlProcessRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """
    Process a URL and add its content to the user's collection
    """
    try:
        # Validate folder if provided
        if request.folder_name and request.folder_name not in get_user_folders(current_user.username):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Folder '{request.folder_name}' does not exist"
            )
        
        # Prepare metadata
        metadata = {
            "folder_name": request.folder_name if request.folder_name else "Uncategorized",
            "source_type": "web"
        }
        
        # Add document name if provided
        if request.document_name:
            metadata["document_name"] = request.document_name
        
        # Get user's collection name
        collection_name = get_user_collection_name(current_user.username)
        
        # Process URL in background
        background_tasks.add_task(
            process_document.process_url,
            str(request.url),
            collection_name,
            metadata,
            request.follow_links,
            request.max_links
        )
        
        return {
            "status": "processing",
            "message": f"URL {request.url} is being processed in the background",
            "url": str(request.url)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing URL: {str(e)}"
        )

@app.post("/process_urls_batch")
async def process_urls_batch_endpoint(
    request: BatchUrlProcessRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """
    Process multiple URLs in batch and add their content to the user's collection
    """
    try:
        # Validate folder if provided
        if request.folder_name and request.folder_name not in get_user_folders(current_user.username):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Folder '{request.folder_name}' does not exist"
            )
        
        # Prepare metadata
        metadata = {
            "folder_name": request.folder_name if request.folder_name else "Uncategorized",
            "source_type": "web"
        }
        
        # Get user's collection name
        collection_name = get_user_collection_name(current_user.username)
        
        # Convert URLs to strings
        url_strings = [str(url) for url in request.urls]
        
        # Process URLs in background
        background_tasks.add_task(
            process_document.process_urls_batch,
            url_strings,
            collection_name,
            metadata,
            request.follow_links,
            request.max_links
        )
        
        return {
            "status": "processing",
            "message": f"Batch of {len(request.urls)} URLs is being processed in the background",
            "urls_count": len(request.urls)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing URLs batch: {str(e)}"
        )

@app.post("/chat/clear_all")
async def clear_all_chat_memory(current_user: User = Depends(get_current_user)):
    """
    Clears all conversation memory for the current user.
    """
    global conversation_memory
    user_prefix = f"{current_user.username}_"
    keys_to_clear = [key for key in conversation_memory if key.startswith(user_prefix)]

    for key in keys_to_clear:
        del conversation_memory[key]

    return {"status": "success", "message": f"All conversation memory for user '{current_user.username}' cleared."}

# Health check endpoint for Render
@app.get("/health")
def health_check():
    """
    Health check endpoint for monitoring service status
    """
    return {"status": "healthy", "timestamp": datetime.now().isoformat(), "cors_enabled": True}

# Special debug endpoint to test CORS
@app.get("/cors-test")
def cors_test():
    """
    Special endpoint to test if CORS is working correctly
    """
    return {
        "message": "CORS is working correctly if you can see this message",
        "timestamp": datetime.now().isoformat()
    }

# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str):
    """
    WebSocket endpoint for real-time updates
    """
    try:
        # Validate the token
        user = await get_current_user_ws(token)
        if not user:
            await websocket.close(code=1008, reason="Invalid token")
            return
        
        # Accept the connection
        await manager.connect(websocket, user.username)
        
        # Send initial confirmation
        await websocket.send_json({
            "type": "connection_established",
            "data": {
                "username": user.username
            }
        })
        
        # Keep the connection alive and handle messages
        try:
            while True:
                # Wait for messages from the client
                data = await websocket.receive_text()
                # You can handle client messages here if needed
        except WebSocketDisconnect:
            manager.disconnect(websocket, user.username)
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        try:
            await websocket.close(code=1011, reason=f"Internal server error: {str(e)}")
        except:
            pass

# Helper function to get user from token for WebSocket connections
async def get_current_user_ws(token: str):
    """
    Get the current user from a token for WebSocket connections
    """
    from auth import get_current_user_from_token
    try:
        return get_current_user_from_token(token)
    except:
        return None

# In-memory storage for conversation history (can be replaced with a database)
conversation_memory = {}