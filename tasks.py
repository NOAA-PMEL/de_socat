from datetime import datetime, timedelta, timezone
from pathlib import Path

def delete_old_files(directory_path: str):
    # Convert string path to a Path object
    target_dir = Path(directory_path)
    
    # Define the 24-hour time threshold (using timezone-aware UTC)
    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
    
    print(f"Scanning '{target_dir}' for files older than: {cutoff_time}")
    
    # Iterate through all files in the directory (non-recursive)
    # Use target_dir.rglob('*') instead if you want to search subfolders
    for file_path in target_dir.glob('*'):
        
        # Ensure we are only targeting files, not folders
        if file_path.is_file():
            
            # Get the last modification time of the file
            file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
            
            # Compare and delete if the file is older than the cutoff
            if file_mtime < cutoff_time:
                try:
                    file_path.unlink()
                    print(f"Deleted: {file_path.name} (Modified: {file_mtime})")
                except Exception as e:
                    print(f"Failed to delete {file_path.name}: {e}")