"""
Semantic chunking module for document processing
Implements advanced chunking strategies with overlap and semantic boundaries
"""
import re
from typing import List, Dict, Any, Optional, Tuple

def semantic_chunk_text(
    text: str, 
    chunk_size: int = 4000, 
    chunk_overlap: int = 200,
    min_chunk_size: int = 500
) -> List[str]:
    """
    Split text into chunks based on semantic boundaries with overlap.
    
    Args:
        text: The text to split into chunks
        chunk_size: Target size for each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
        min_chunk_size: Minimum chunk size to avoid tiny chunks
        
    Returns:
        List of text chunks
    """
    # If text is shorter than chunk_size, return it as a single chunk
    if len(text) <= chunk_size:
        return [text]
    
    # Define semantic boundary patterns in order of priority
    section_patterns = [
        # Headers with numbers
        r'\n#{1,6}\s+.*\n',  # Markdown headers
        r'\n\d+\.\s+.*\n',   # Numbered sections
        
        # Natural language section indicators
        r'\n(Section|SECTION|Chapter|CHAPTER)\s+\d+.*\n',
        
        # Strong paragraph boundaries
        r'\n\n+',  # Multiple newlines
        
        # Sentence boundaries (fallback)
        r'(?<=[.!?])\s+(?=[A-Z])',
    ]
    
    chunks = []
    start_idx = 0
    
    while start_idx < len(text):
        # Determine end index for current chunk (target size)
        target_end_idx = min(start_idx + chunk_size, len(text))
        
        # If we're near the end of the text, just include the rest
        if target_end_idx >= len(text) - min_chunk_size:
            chunks.append(text[start_idx:])
            break
        
        # Look for semantic boundaries near the target end
        best_split_idx = None
        
        # Try each pattern in order of priority
        for pattern in section_patterns:
            # Look for matches in a window around the target end
            search_window_start = max(start_idx + min_chunk_size, target_end_idx - chunk_size // 2)
            search_window_end = min(target_end_idx + chunk_size // 2, len(text))
            search_window = text[search_window_start:search_window_end]
            
            matches = list(re.finditer(pattern, search_window))
            
            # Find the closest match to our target end
            if matches:
                # Convert match position to absolute position in text
                match_positions = [search_window_start + m.start() for m in matches]
                
                # Find the closest match to our target
                closest_idx = min(match_positions, key=lambda x: abs(x - target_end_idx))
                
                # If it's a reasonable boundary, use it
                if abs(closest_idx - target_end_idx) < chunk_size // 3:
                    best_split_idx = closest_idx
                    break
        
        # If no good semantic boundary found, fall back to character boundary
        if best_split_idx is None:
            # Try to split at a space to avoid cutting words
            space_before = text.rfind(' ', target_end_idx - 100, target_end_idx)
            if space_before > start_idx + min_chunk_size:
                best_split_idx = space_before
            else:
                # Last resort: just split at the target size
                best_split_idx = target_end_idx
        
        # Add the chunk
        chunks.append(text[start_idx:best_split_idx])
        
        # Move start index for next chunk, accounting for overlap
        start_idx = best_split_idx - chunk_overlap
        
        # Ensure we're making forward progress
        if start_idx <= 0 or best_split_idx <= start_idx:
            start_idx = best_split_idx
    
    # Post-process chunks to add context and clean up
    processed_chunks = []
    for i, chunk in enumerate(chunks):
        # Add overlap marker if this isn't the first chunk
        if i > 0:
            # Clean up the chunk start (remove leading whitespace/newlines)
            chunk = chunk.lstrip()
            
        # Add overlap marker if this isn't the last chunk
        if i < len(chunks) - 1:
            # Make sure the chunk doesn't end with partial sentence
            last_period = max(chunk.rfind('.'), chunk.rfind('!'), chunk.rfind('?'))
            if last_period > len(chunk) - 20 and last_period > 0:
                # If we have a sentence end near the chunk end, trim to it
                chunk = chunk[:last_period+1]
        
        processed_chunks.append(chunk)
    
    return processed_chunks

def extract_metadata_from_chunk(chunk: str) -> Dict[str, Any]:
    """
    Extract useful metadata from a chunk to enhance retrieval.
    
    Args:
        chunk: The text chunk
        
    Returns:
        Dictionary of metadata about the chunk
    """
    metadata = {}
    
    # Try to extract a title/header
    header_match = re.search(r'^\s*#+\s+(.*?)$', chunk, re.MULTILINE)
    if header_match:
        metadata['title'] = header_match.group(1).strip()
    
    # Extract keywords (simplified approach)
    words = re.findall(r'\b[A-Za-z]{4,}\b', chunk.lower())
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Get top keywords
    keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
    metadata['keywords'] = [k for k, _ in keywords]
    
    return metadata

def chunk_document(
    text: str,
    document_name: str,
    metadata: Optional[Dict[str, Any]] = None,
    chunk_size: int = 4000,
    chunk_overlap: int = 200
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Process a document into chunks with metadata.
    
    Args:
        text: The document text
        document_name: Name of the document
        metadata: Base metadata to include with all chunks
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        
    Returns:
        Tuple of (chunks, metadatas)
    """
    # Create semantic chunks
    chunks = semantic_chunk_text(
        text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Prepare metadata for each chunk
    if metadata is None:
        metadata = {}
    
    # Add document name to base metadata
    base_metadata = metadata.copy()
    if "document_name" not in base_metadata:
        base_metadata["document_name"] = document_name
    
    # Create metadata for each chunk
    metadatas = []
    for i, chunk in enumerate(chunks):
        # Start with base metadata
        chunk_metadata = base_metadata.copy()
        
        # Add chunk-specific metadata
        chunk_metadata["chunk_index"] = i
        chunk_metadata["chunk_count"] = len(chunks)
        
        # Extract semantic metadata from chunk content
        semantic_metadata = extract_metadata_from_chunk(chunk)
        chunk_metadata.update(semantic_metadata)
        
        metadatas.append(chunk_metadata)
    
    return chunks, metadatas

if __name__ == "__main__":
    # Simple test
    test_text = """
# Introduction to Document Processing

This is a sample document to test the semantic chunking algorithm.
It contains multiple paragraphs and sections to demonstrate how the chunking works.

## Section 1: Basic Concepts

Document processing involves several steps:
1. Text extraction
2. Chunking
3. Embedding generation
4. Storage and indexing

### Subsection 1.1: Text Extraction

Text extraction is the process of extracting plain text from various file formats.
This can include PDF, DOCX, HTML, and many other formats.

## Section 2: Advanced Topics

Semantic chunking is more effective than simple character-based chunking.
It preserves the meaning and context of the document.

### Subsection 2.1: Overlap

Using overlap between chunks helps maintain context across chunk boundaries.
This is especially important for question answering systems.

### Subsection 2.2: Metadata

Adding metadata to chunks improves retrieval quality.
Metadata can include section titles, keywords, and other contextual information.

## Conclusion

Proper document processing is essential for building effective retrieval systems.
The quality of chunking directly impacts the quality of search results.
    """
    
    chunks = semantic_chunk_text(test_text, chunk_size=1000, chunk_overlap=100)
    
    print(f"Created {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ---")
        print(chunk[:100] + "..." if len(chunk) > 100 else chunk)
        print(f"Length: {len(chunk)} characters")
