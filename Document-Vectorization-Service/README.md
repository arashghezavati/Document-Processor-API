# Document Processing & AI Chatbot Service

A powerful Python service that processes documents (PDF, DOCX, TXT, HTML) and makes them searchable and interactive using AI-powered vector embeddings. Built with ChromaDB for efficient vector storage and retrieval, and Google Gemini AI for intelligent response generation.

Features a simplified UI for document management and an AI agent system powered by LangGraph for complex document processing tasks.

## 🚀 Features & Benefits

### 📖 AI-Powered Document Search

- Converts documents into vector embeddings for deep search.
- Stores and retrieves context-aware information using ChromaDB.
- Supports PDF, DOCX, TXT, HTML, and more.

### 🤖 AI Chatbot

- Interactive Chatbot UI – Ask AI about stored business data.
- Customer-Specific Responses – Query business info, customers, and products.
- CORS-enabled API – Works seamlessly with frontend and Postman.

### 📤 Document & Folder Management

- Upload documents directly through the simplified UI.
- Create folders to organize documents.
- Name documents with custom titles during upload.
- Assign documents to specific folders.
- Delete documents and folders as needed.
- View all documents regardless of folder in the chat interface.

- Natural language task requests (e.g., "Summarize my documents about AI agents").
- User-specific document access with proper authentication.
- Document context integration for informed responses.
- Dynamic workflow planning and execution using LangGraph.

### 🔐 Enhanced Authentication System task requests (e.g., "Summarize my documents about AI agents").

- Secure JWT token-based authentication- User-specific document access with proper authentication.
- Strong password validation with multiple criteriaformed responses.
- Automatic token validation and expiration handling
- Protected routes and API endpointsw It Works
- Environment-specific configurations
- User session management and clean logout processg & Storage

## 🔹 How It Works

ph LR

### 1️⃣ Document Processing & Storage

````mermaid
graph LR
    A[Input Document] --> B[Text Extraction]
    B --> C[Text Chunking]```
    C --> D[Vector Embedding]
    D --> E[ChromaDB Storage] text from PDFs, DOCX, TXT, and HTML.
``` text into manageable chunks.
- Extracts text from PDFs, DOCX, TXT, and HTML.
- Splits text into manageable chunks.retrieval.
- Embeds text using Google Gemini AI.
- Stores vectors in ChromaDB for fast retrieval.

### 2️⃣ AI Chatbot Interactionmermaid
```mermaid
graph LR
    A[User Query] --> B[Retrieve Documents]
    B --> C[Vector Search in ChromaDB]    C --> D[Construct AI Prompt]
    C --> D[Construct AI Prompt] AI Response]
    D --> E[Google Gemini AI Response]F[Chatbot UI]
    E --> F[Chatbot UI]
````

- Accepts customer-specific queries or queries across all document categories.c queries or queries across all document categories.
- Searches stored knowledge for relevant data. relevant data.
- Uses Google Gemini AI to generate responses.generate responses.

### 3️⃣ AI Agent Workflow 3️⃣ AI Agent Workflow

````mermaid
graph LR
    A[User Task Request] --> B[Authentication]
    B --> C[Plan Creation]on]
    C --> D[Document Retrieval]
    D --> E[Task Execution]    C --> D[Document Retrieval]
    E --> F[Response Generation]tion]
```    E --> F[Response Generation]
- Processes natural language task requests.
- Creates a plan for completing the task.
- Retrieves relevant documents from the user's collection.nguage task requests.
- Executes the task using document context.
- Generates a comprehensive response.om the user's collection.
- Executes the task using document context.
### 4️⃣ Authentication Flowsponse.
```mermaid
graph LR
    A[User Credentials] --> B[JWT Token Generation]
    B --> C[Token Storage]### 1️⃣ Clone & Setup
    C --> D[Protected Routes/APIs]
    D --> E[Token Validation]
    E --> F[Auto Refresh/Logout]lone the repository
```git clone https://github.com/arashghezavati/Document-Vectorization-Service.git
- Secures user sessions with JWT tokensService
- Validates credentials against database
- Protects sensitive routes and API endpointsreate a virtual environment
- Handles token expiration gracefully
inux/Mac
## ⚡ Quick Start Guides

### 1️⃣ Clone & Setup
```sh install -r python-services/requirements.txt
# Clone the repository```
git clone https://github.com/arashghezavati/Document-Vectorization-Service.git
cd Document-Vectorization-Service

# Create a virtual environment following content:
python -m venv venv
source venv/bin/activate  # Linux/Mac```
.\venv\Scripts\activate   # Windowsyour_api_key_here
2.0-flash
# Install dependencies_SECRET_KEY=your_jwt_secret_key
pip install -r python-services/requirements.txtEMBEDDING_DIMENSION=768
````

### 2️⃣ Configure API Keys

Create a `.env` file in the root directory with the following content:I

```
GOOGLE_GEMINI_API_KEY=your_api_key_heretbot functionality:
GEMINI_MODEL=gemini-2.0-flash
JWT_SECRET_KEY=your_jwt_secret_key
EMBEDDING_DIMENSION=768
EMBEDDING_MODEL=text-embedding-004
```

# Start the API server

### 3️⃣ Start the AI Chatbot API

Run the FastAPI backend to enable chatbot functionality:

```sh
# Make sure you're in the python-services directory
cd python-services

# Start the API serveron UI
python run_server.py
```

- Backend URL: http://localhost:8000- Navigate through the simplified interface:
- API Docs: http://localhost:8000/docserview of your documents and folders.
  - **Documents**: Create folders and upload documents.

### 4️⃣ Open the Application UIabout your documents.

- Open `http://localhost:3000` in your browser.ex tasks to the AI agent.
- Register or log in to your account.
- Navigate through the simplified interface:
  - **Dashboard**: Overview of your documents and folders.
  - **Documents**: Create folders and upload documents.
  - **Chat**: Interact with the AI chatbot about your documents.- Upload documents with custom names and assign them to folders.
  - **Tasks**: Submit complex tasks to the AI agent.ll be available in the chat interface regardless of folder.
- Use the document filter in chat to focus on specific documents.

### 5️⃣ Working with Documents

- Create folders to organize your documents.### 6️⃣ Using the AI Agent
- Upload documents with custom names and assign them to folders.
- All documents will be available in the chat interface regardless of folder.
- Use the document filter in chat to focus on specific documents.nts about AI agents").
  s, and generate a response.

### 6️⃣ Using the AI Agent

- Navigate to the Tasks tab.## 🔧 Technical Stack
- Enter a natural language task request (e.g., "Summarize my documents about AI agents").
- The agent will process your request, retrieve relevant documents, and generate a response.- **Backend**: FastAPI, ChromaDB, LangGraph

For support, please open an issue in the GitHub repository or contact the maintainers directly.## 📩 Support5. Open a Pull Request4. Push to the branch (`git push origin feature/amazing-feature`)3. Commit your changes (`git commit -m 'Add some amazing feature'`)2. Create your feature branch (`git checkout -b feature/amazing-feature`)1. Fork the repositoryContributions are welcome! Please feel free to submit a Pull Request.## 🤝 Contributing- **Document Processing**: PyPDF2, python-docx, BeautifulSoup- **Database**: ChromaDB for vector storage- **Authentication**: JWT-based authentication- **Frontend**: React.js- **AI**: Google Gemini API- **Backend**: FastAPI, ChromaDB, LangGraph## 🔧 Technical Stack- **Protected Resources**: Secure access to documents and chat functionality- **Token Management**: Automatic handling of token expiration- **JWT-based Sessions**: Secure, stateless authentication- **Secure Sign-Up**: With strong password requirements (uppercase, lowercase, numbers, special characters)Our authentication implementation provides:## 🔧 Authentication SystemSet the environment using `REACT_APP_ENV` environment variable.- **Test**: For running automated tests- **Production**: Live deployment with optimized settings- **Development**: Local testing with debugging enabledThe application supports multiple environments through the configuration system:## 🛠️ Environment Configuration- **Frontend**: React.js

- **Authentication**: JWT-based authentication
- **Database**: ChromaDB for vector storage
- **Document Processing**: PyPDF2, python-docx, BeautifulSoup

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📩 Support

For support, please open an issue in the GitHub repository or contact the maintainers directly.
