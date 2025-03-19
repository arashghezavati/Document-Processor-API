import os
import chromadb
import google.generativeai as genai
import shutil
import tempfile
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, status, BackgroundTasks
from pydantic import BaseModel, HttpUrl
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
import importlib.util

# Import authentication module
from auth import (
    User, UserCreate, Token, 
    authenticate_user, create_user, create_access_token,
    get_current_user, create_folder, get_user_folders
)
from datetime import timedelta

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
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (adjust for security)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Initialize ChromaDB Client
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'vector-database', 'store-new')
chroma_client = chromadb.PersistentClient(path=DB_PATH)

# Define request models
class ChatRequest(BaseModel):
    query: str
    folder_name: Optional[str] = None
    document_name: Optional[str] = None
    mode: str = "strict"  # Optional: strict or comprehensive

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

def retrieve_documents(username: str, folder_name: Optional[str] = None, document_name: Optional[str] = None):
    """
    Fetches documents from ChromaDB based on user, folder, and document filters.
    """
    try:
        # Get user's collection
        collection_name = get_user_collection_name(username)
        collection = chroma_client.get_or_create_collection(name=collection_name)
        
        # Build query filter based on folder and document
        where_filter = {}
        if folder_name:
            where_filter["folder_name"] = folder_name
        if document_name:
            where_filter["document_name"] = document_name
        
        # Query documents with filter
        if where_filter:
            docs = collection.get(where=where_filter, include=["documents", "metadatas"])
        else:
            docs = collection.get(include=["documents", "metadatas"])
        
        if not docs or not docs["documents"]:
            return None  # No documents found
        
        # Join all documents into a single string
        documents_text = "\n\n".join(docs["documents"])
        print(f"🔍 Retrieved Documents for {username}: {documents_text}")  # Debugging
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
    try:
        # Validate folder if provided
        if folder_name and folder_name not in get_user_folders(current_user.username):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Folder '{folder_name}' does not exist"
            )
        
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
        
        return {
            "status": "success", 
            "message": f"Document {file.filename} successfully processed" + 
                      (f" into folder {folder_name}" if folder_name else "")
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

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
        collection = chroma_client.get_or_create_collection(name=collection_name)
        
        # Build query filter based on folder
        where_filter = {}
        if folder_name:
            where_filter["folder_name"] = folder_name
        
        # Query documents with filter
        if where_filter:
            results = collection.get(where=where_filter, include=["metadatas"])
        else:
            results = collection.get(include=["metadatas"])
        
        # Extract document names from metadata and deduplicate
        documents = []
        seen_documents = set()  # Track unique document names
        
        if results and results["metadatas"]:
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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving documents: {str(e)}"
        )

@app.post("/chat")
async def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Receives chat message and returns AI response based on user's documents.
    """
    query = request.query
    folder_name = request.folder_name
    document_name = request.document_name
    mode = request.mode
    
    # Retrieve documents based on filters
    customer_data = retrieve_documents(
        current_user.username,
        folder_name,
        document_name
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

    # Construct AI prompt ensuring NO introduction text
    prompt = (
        f"{customer_data}\n\n"
        f"Answer concisely and directly based on the provided data. Do NOT include introductions or explanations. "
        f"Just return the response for the following request:\n\n"
        f"Query: {query}"
    )

    print(f"📩 AI Debug - Constructed Prompt: {prompt}")  # Debugging

    # Get AI response
    ai_response = call_ai_model(prompt)

    return {"response": ai_response}

# Delete a document
@app.delete("/documents/{document_name}")
async def delete_document(document_name: str, current_user: User = Depends(get_current_user)):
    try:
        # Get user's collection with correct naming
        collection_name = f"user_{current_user.username}_docs"
        
        # Debug print
        print(f"Attempting to delete document: '{document_name}' from collection: '{collection_name}'")
        
        # Check if collection exists
        try:
            collection = chroma_client.get_collection(collection_name)
        except Exception as e:
            print(f"Error getting collection: {str(e)}")
            raise HTTPException(status_code=404, detail="User collection not found")
        
        # Check if collection is empty
        try:
            documents = collection.get()
            if not documents['ids']:
                raise HTTPException(status_code=404, detail="No documents found in collection")
        except Exception as e:
            print(f"Error getting documents: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error retrieving documents: {str(e)}")
        
        document_ids = documents['ids']
        document_metadatas = documents['metadatas']
        
        # Debug print all document names
        print(f"Available documents: {[meta.get('document_name', 'Unknown') for meta in document_metadatas]}")
        
        # Find the document by name
        document_index = None
        for i, metadata in enumerate(document_metadatas):
            doc_name = metadata.get('document_name', '')
            print(f"Comparing: '{doc_name}' with '{document_name}'")
            if doc_name == document_name:
                document_index = i
                print(f"Match found at index {i}")
                break
        
        if document_index is None:
            raise HTTPException(status_code=404, detail=f"Document '{document_name}' not found")
        
        # Delete the document
        document_id = document_ids[document_index]
        print(f"Deleting document with ID: {document_id}")
        collection.delete(ids=[document_id])
        
        return {"message": f"Document '{document_name}' deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in delete_document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

# Delete a folder and all its documents
@app.delete("/folders/{folder_name}")
async def delete_folder(folder_name: str, current_user: User = Depends(get_current_user)):
    try:
        # Get user's collection with correct naming
        collection_name = f"user_{current_user.username}_docs"
        
        print(f"Attempting to delete folder: '{folder_name}' for user: '{current_user.username}'")
        
        # First check if the folder exists in user's folders
        user_folders = get_user_folders(current_user.username)
        print(f"User folders: {user_folders}")
        
        if folder_name not in user_folders:
            print(f"Folder '{folder_name}' not found in user's folders list")
            # We'll continue anyway to clean up any documents that might be in this folder
        
        # Delete folder from MongoDB - use try/except to handle potential errors
        try:
            result = users_collection.update_one(
                {"username": current_user.username},
                {"$pull": {"folders": folder_name}}
            )
            print(f"MongoDB update result: {result.modified_count} documents modified")
        except Exception as mongo_error:
            print(f"Error updating MongoDB: {str(mongo_error)}")
            # Continue with document deletion even if folder deletion fails
        
        # Now delete all documents in this folder from ChromaDB
        try:
            # Check if collection exists first
            collections = chroma_client.list_collections()
            collection_exists = False
            for coll in collections:
                if coll.name == collection_name:
                    collection_exists = True
                    break
            
            if not collection_exists:
                print(f"Collection '{collection_name}' does not exist")
                # If the collection doesn't exist, we can consider the folder deleted
                return {"message": f"Folder '{folder_name}' deleted successfully (no collection found)"}
            
            collection = chroma_client.get_collection(collection_name)
            
            # Get all documents
            documents = collection.get()
            if not documents['ids'] or len(documents['ids']) == 0:
                print("No documents found in collection")
                return {"message": f"Folder '{folder_name}' deleted successfully (no documents found)"}
            
            document_ids = documents['ids']
            document_metadatas = documents['metadatas']
            
            print(f"Found {len(document_ids)} documents in collection")
            
            # Find documents in the folder
            documents_to_delete = []
            for i, metadata in enumerate(document_metadatas):
                folder = metadata.get('folder_name', '')
                print(f"Document {i} folder: '{folder}'")
                if folder == folder_name:
                    documents_to_delete.append(document_ids[i])
            
            print(f"Found {len(documents_to_delete)} documents to delete in folder '{folder_name}'")
            
            # Delete documents in the folder
            if documents_to_delete:
                collection.delete(ids=documents_to_delete)
                print(f"Deleted {len(documents_to_delete)} documents from folder '{folder_name}'")
            
        except Exception as chroma_error:
            print(f"Error deleting documents from ChromaDB: {str(chroma_error)}")
            # Continue with folder deletion even if document deletion fails
        
        return {"message": f"Folder '{folder_name}' and all its documents deleted successfully"}
    except Exception as e:
        print(f"Error in delete_folder: {str(e)}")
        # Include the full error details in the response
        import traceback
        error_details = traceback.format_exc()
        print(f"Full error traceback: {error_details}")
        raise HTTPException(status_code=500, detail=f"Failed to delete folder: {str(e)}")

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