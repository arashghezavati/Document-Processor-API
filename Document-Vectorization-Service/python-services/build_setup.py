#!/usr/bin/env python3
"""
Build setup script for Render deployment
This script installs dependencies that require special handling
"""
import os
import sys
import subprocess
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("build_setup")

def install_dependencies():
    """Install dependencies that need special handling"""
    try:
        logger.info("Starting custom dependency installation")
        
        # Install basic requirements first
        logger.info("Installing base requirements")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install unstructured with DOCX support (using quotes in subprocess call)
        logger.info("Installing unstructured with DOCX support")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "unstructured[docx]"])
        
        # Install other document format support
        logger.info("Installing additional document format support")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "unstructured[pdf]"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "unstructured[pptx]"])
        
        logger.info("Custom dependency installation completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error installing dependencies: {str(e)}")
        return False

if __name__ == "__main__":
    success = install_dependencies()
    sys.exit(0 if success else 1)
