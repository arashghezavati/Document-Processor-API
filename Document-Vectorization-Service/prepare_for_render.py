import os
import zipfile
import shutil

def create_deployment_package():
    """
    Creates a ZIP file for deployment to Render, excluding unnecessary files
    """
    # Define the output zip file
    output_zip = "render_deployment.zip"
    
    # Define directories and files to exclude
    exclude_dirs = [
        "__pycache__",
        "venv",
        ".git",
        "vector-database",
        "node_modules",
    ]
    
    exclude_files = [
        ".env",
        ".DS_Store",
        "render_deployment.zip",
    ]
    
    # Create a temporary directory for the files to include
    temp_dir = "temp_deployment"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    # Copy files to the temporary directory, excluding unwanted ones
    for root, dirs, files in os.walk("."):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs and not any(ed in os.path.join(root, d) for ed in exclude_dirs)]
        
        # Create the corresponding directory in the temp directory
        rel_dir = os.path.relpath(root, ".")
        if rel_dir == ".":
            rel_dir = ""
        temp_root = os.path.join(temp_dir, rel_dir)
        if not os.path.exists(temp_root):
            os.makedirs(temp_root)
        
        # Copy files, excluding the ones in the exclude list
        for file in files:
            if file not in exclude_files and not any(ef in os.path.join(root, file) for ef in exclude_files):
                src_file = os.path.join(root, file)
                dst_file = os.path.join(temp_root, file)
                shutil.copy2(src_file, dst_file)
    
    # Create a .env.example file with placeholders instead of actual values
    with open(os.path.join(temp_dir, ".env.example"), "w") as f:
        f.write("""GOOGLE_GEMINI_API_KEY=your_api_key_here
EMBEDDING_DIMENSION=768
GEMINI_MODEL=gemini-2.0-flash
MAX_OUTPUT_TOKENS=2048
QDRANT_URL=your_qdrant_url_here
QDRANT_API_KEY=your_qdrant_api_key_here
JWT_SECRET_KEY=your_jwt_secret_key_here
""")
    
    # Create the ZIP file
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, temp_dir)
                zipf.write(file_path, arcname)
    
    # Clean up the temporary directory
    shutil.rmtree(temp_dir)
    
    print(f"Deployment package created: {output_zip}")
    print(f"Size: {os.path.getsize(output_zip) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    create_deployment_package()
