import pandas as pd
import os

from quantifier_recommender import QuantifierRecommender
from quantifier_evaluator import QuantifierEvaluator
from utils_ import generate_train_test_data
from quantifier_evaluator import QuantifierEvaluator

def load_train_test_data(dataset_name):
    train_df = pd.read_csv(f"./data/train_data/{dataset_name}.csv")
    y_train = train_df.pop(train_df.columns[-1])
    X_train = train_df

    test_df = pd.read_csv(f"./data/test_data/{dataset_name}.csv")
    y_test = test_df.pop(test_df.columns[-1])
    X_test = test_df

    return X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()

if __name__ == "__main__":
    dataset_path = "./datasets"
    # recommender = QuantifierRecommender(supervised=True)

    # recommender.construct_meta_table(dataset_path=dataset_path, supervised=True)
    # recommender.save_meta_table('meta-features.csv')

    # generate_train_test_data("./datasets")

    evaluation_table = pd.read_csv('./evaluation_table.csv')
    quantifier_evaluator = QuantifierEvaluator(evaluation_table=evaluation_table)

    # X_train, y_train, X_test, y_test = load_train_test_data("AedesQuinx")
    # quantifier_evaluator.evaluate_internal_quantifiers("AedesQuinx", X_train, y_train, X_test, y_test)
    # quantifier_evaluator.sort_evaluation_table()
    # quantifier_evaluator.save_evaluation_table()

    processed_list = evaluation_table['dataset'].unique().tolist()
    dataset_list = [csv for csv in os.listdir("./datasets/") if csv.endswith(".csv")]
    for dataset in dataset_list:
        dataset_name = dataset.split(".csv")[0]

        if dataset_name in processed_list:
            continue

        X_train, y_train, X_test, y_test = load_train_test_data(dataset_name)
        
        quantifier_evaluator.evaluate_internal_quantifiers(dataset_name, X_train, y_train, X_test, y_test)
        quantifier_evaluator.save_evaluation_table()

    # quantifier_evaluator.aggregate_evaluation_table()
    # quantifier_evaluator.save_evaluation_table('./aggregated_evaluation_table.csv')