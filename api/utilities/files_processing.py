import os
from pathlib import Path
import tempfile
import uuid


def save_file_to_temp_dir(file):
    """
    Save the uploaded file to a temporary directory
    And return the path of the file
    """

    original_name = file.name
    file_extension = os.path.splitext(original_name)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    
    temp_dir = Path(tempfile.gettempdir()) / "chatbot_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    
    temp_path = temp_dir / unique_filename
    
    with open(temp_path, 'wb') as temp_file:
        for chunk in file.chunks():
            temp_file.write(chunk)
    
    return temp_path

def clean_temp_file(path):
    try:
        os.remove(path)
    except OSError:
        print(f"Failed to delete temporary file: {path}")