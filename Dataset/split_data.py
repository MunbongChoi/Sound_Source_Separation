from sklearn.model_selection import train_test_split
import pandas as pd

def split_data(raw_file_path, train_file_path, test_file_path, test_size=0.2):
    df = pd.read_csv(raw_file_path)
    train_df, test_df = train_test_split(df, test_size=test_size)
    train_df.to_csv(train_file_path, index=False)
    test_df.to_csv(test_file_path, index=False)