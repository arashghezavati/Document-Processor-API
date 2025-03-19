from extraction import extract_text

def chunk_text(text, chunk_size=1000):
    """Split text into chunks of roughly equal size."""
    # Split by double newlines to preserve paragraph structure
    paragraphs = text.split('\n\n')
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for paragraph in paragraphs:
        # Skip empty paragraphs
        if not paragraph.strip():
            continue
            
        # If adding this paragraph would exceed chunk size, save current chunk
        if current_size + len(paragraph) > chunk_size and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = []
            current_size = 0
            
        current_chunk.append(paragraph)
        current_size += len(paragraph)
    
    # Add the last chunk if there's anything left
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks if chunks else [text]  # Return original text if no chunks were created

def chunk_file(file_path, chunk_size=1000):
    text = extract_text(file_path)
    return chunk_text(text, chunk_size)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        chunks = chunk_file(sys.argv[1])
        print(f"Created {len(chunks)} chunks")
        for chunk in chunks:
            print(f"--- Chunk ---\n{chunk}\n")
