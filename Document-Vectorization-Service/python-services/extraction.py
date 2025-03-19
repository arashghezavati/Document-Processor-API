from unstructured.partition.auto import partition
import os

def extract_text(file_path):
    """Extract text from various document formats."""
    extension = os.path.splitext(file_path)[1].lower()
    
    if extension in ['.pdf', '.docx', '.txt', '.html']:
        # Use unstructured library for these formats
        elements = partition(filename=file_path)
        return "\n".join([str(el) for el in elements])
    else:
        raise ValueError(f"Unsupported file format: {extension}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        text = extract_text(sys.argv[1])
        print(text)
