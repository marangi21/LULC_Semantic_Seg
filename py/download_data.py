import subprocess
import os

dataset_folder_path = "https://drive.google.com/drive/folders/1J9-BNenq46mPPi8GN1g88BQ2uT5DMehh?usp=drive_link"
destination_path = os.path.join(os.path.pardir(os.path.abspath(__file__)), "./data")

subprocess.run([
    "sudo", "apt", "install", "rclone"
])