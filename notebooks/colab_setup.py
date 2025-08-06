# notebooks/colab_setup.py
from google.colab import drive
import os

def setup(project_root = "Colab_Projects/your-repo", subdir=""):
    """
    Mounts Google Drive and sets the working directory.
    
    Parameters:
    - project_root: path under 'MyDrive' to your project root
    - subdir: optional subfolder inside your repo to cd into
    """
    # Mount Google Drive if not already mounted
    if not os.path.ismount("/content/drive"):
        drive.mount("/content/drive")

    base_path = os.path.join("/content/drive/MyDrive", project_root)
    
    if subdir:
        full_path = os.path.join(base_path, subdir)
    else:
        full_path = base_path

    os.chdir(full_path)
    print("Drive mounted and working directory set to:\n", os.getcwd())

