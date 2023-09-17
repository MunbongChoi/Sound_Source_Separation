import pandas as pd

def preprocess(raw_file_path, processed_file_path):
    df = pd.read_csv(raw_file_path)
    # Add preprocessing logic here
    df.to_csv(processed_file_path, index=False)