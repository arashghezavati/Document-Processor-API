import os
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from passlib.context import CryptContext
from pymongo import MongoClient
from pydantic import BaseModel, Field, EmailStr
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

# MongoDB connection
MONGO_URI = "mongodb://localhost:27017"
client = MongoClient(MONGO_URI)
db = client["Document-Vectorization-Service"]
users_collection = db["users"]

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="signin")

# Models
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    
class UserInDB(BaseModel):
    username: str
    email: EmailStr
    hashed_password: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    folders: list = Field(default_factory=list)
    
class User(BaseModel):
    username: str
    email: EmailStr
    created_at: datetime
    folders: list
    
class Token(BaseModel):
    access_token: str
    token_type: str
    
class TokenData(BaseModel):
    username: Optional[str] = None

# Helper functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(username: str):
    user_dict = users_collection.find_one({"username": username})
    if user_dict:
        return UserInDB(**user_dict)
    return None

def authenticate_user(username: str, password: str):
    user = get_user(username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except jwt.PyJWTError:
        raise credentials_exception
    user = get_user(username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

# User management functions
def create_user(user: UserCreate):
    # Check if username already exists
    if users_collection.find_one({"username": user.username}):
        return None
    
    # Check if email already exists
    if users_collection.find_one({"email": user.email}):
        return None
    
    # Create new user with hashed password
    user_dict = UserInDB(
        username=user.username,
        email=user.email,
        hashed_password=get_password_hash(user.password),
        folders=[]  # Initialize with empty folders list
    ).dict()
    
    # Insert into MongoDB
    users_collection.insert_one(user_dict)
    
    # Return user without password
    return User(
        username=user.username,
        email=user.email,
        created_at=user_dict["created_at"],
        folders=[]
    )

# Folder management functions
def create_folder(username: str, folder_name: str):
    """Create a new folder for a user"""
    # Check if folder already exists
    user = get_user(username)
    if not user:
        return False
    
    if folder_name in user.folders:
        return False
    
    # Add folder to user's folders list
    result = users_collection.update_one(
        {"username": username},
        {"$push": {"folders": folder_name}}
    )
    
    return result.modified_count > 0

def get_user_folders(username: str):
    """Get all folders for a user"""
    user = get_user(username)
    if not user:
        return []
    
    return user.folders
