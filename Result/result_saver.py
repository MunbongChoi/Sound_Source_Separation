import pandas as pd

def save_results(metrics, file_name):
    df = pd.DataFrame([metrics])
    df.to_csv(f'Result/{file_name}.csv', index=False)