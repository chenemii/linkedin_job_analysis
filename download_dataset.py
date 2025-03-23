import kagglehub
import os
# Download latest version
path = kagglehub.dataset_download("arshkon/linkedin-job-postings")

# mv dataset to current directory
current_directory = os.getcwd()
os.rename(os.path.join(path, "postings.csv"),
          os.path.join(current_directory, "postings.csv"))

print("Path to dataset files:", path)
