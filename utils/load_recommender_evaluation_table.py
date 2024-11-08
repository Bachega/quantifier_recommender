import os
import pandas as pd

def load_recommender_evaluation_table(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found")
    
    if not path.endswith(".csv"):
        raise ValueError("File must be a CSV file")
    
    evaluation_table = pd.read_csv(path)

    evaluation_table['predicted_ranking'] = evaluation_table['predicted_ranking'].apply(eval)
    evaluation_table['true_ranking'] = evaluation_table['true_ranking'].apply(eval)
    evaluation_table['predicted_ranking_error'] = evaluation_table['predicted_ranking_error'].apply(eval)
    evaluation_table['true_ranking_error'] = evaluation_table['true_ranking_error'].apply(eval)

    return evaluation_table