import pandas as pd
import os
import pdb

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

def load_test_data(dataset_name, supervised = True):
    test_df = pd.read_csv(f"./data/test_data/{dataset_name}.csv")
    y_test = test_df.pop(test_df.columns[-1])
    X_test = test_df

    if supervised:
        return X_test.to_numpy(), y_test.to_numpy()
    
    return X_test.to_numpy()

if __name__ == "__main__":
    dataset_path = "./datasets"
    # recommender = QuantifierRecommender(supervised=True)

    # recommender.construct_meta_table(dataset_path=dataset_path, supervised=True)
    # recommender.save_meta_table('meta-features.csv')

    # generate_train_test_data("./datasets")

    # quantifier_evaluator = QuantifierEvaluator()

    supervised_quantifier_recommender = QuantifierRecommender(supervised=True)
    unsupervised_quantifier_recommender = QuantifierRecommender(supervised=False)
    
    supervised_quantifier_recommender.fit(datasets_path="./datasets/",
                                          train_data_path="./data/train_data/",
                                          test_data_path="./data/test_data/")
    
    unsupervised_quantifier_recommender.fit(datasets_path="./datasets/",
                                            train_data_path="./data/train_data/",
                                            test_data_path="./data/test_data/")
    
    X_test, y_test = load_test_data("BNG", supervised=True)
    s_ranking = supervised_quantifier_recommender.predict(X_test, y_test)
    
    X_test = load_test_data("BNG", supervised=False)
    u_ranking = unsupervised_quantifier_recommender.predict(X_test)

    supervised_quantifier_recommender.persist_model("s_qtf_recommender.pkl")
    unsupervised_quantifier_recommender.persist_model("u_qtf_recommender.pkl")

    s_qtf_rec = QuantifierRecommender.load_model("s_qtf_recommender.pkl")
    u_qtf_rec = QuantifierRecommender.load_model("u_qtf_recommender.pkl")

    X_test, y_test = load_test_data("BNG", supervised=True)
    new_s_ranking = s_qtf_rec.predict(X_test, y_test)

    X_test = load_test_data("BNG", supervised=False)
    new_u_ranking = u_qtf_rec.predict(X_test)

    pdb.set_trace()

    
    # quantifier_recommender.save_evaluation_table()
    # quantifier_recommender.save_meta_features_table()

    



    

    # X_train, y_train, X_test, y_test = load_train_test_data("AedesQuinx")
    # quantifier_evaluator.evaluate_internal_quantifiers("AedesQuinx", X_train, y_train, X_test, y_test)
    # quantifier_evaluator.sort_evaluation_table()
    # quantifier_evaluator.save_evaluation_table()

    # processed_list = evaluation_table['dataset'].unique().tolist()
    # dataset_list = [csv for csv in os.listdir("./datasets/") if csv.endswith(".csv")]
    # for dataset in dataset_list:
    #     dataset_name = dataset.split(".csv")[0]

    #     # if dataset_name in processed_list:
    #     #     continue

    #     X_train, y_train, X_test, y_test = load_train_test_data(dataset_name)
        
    #     qtf_eval = quantifier_evaluator.evaluate_internal_quantifiers(dataset_name, X_train, y_train, X_test, y_test)
    #     qtf_eval.to_csv(f"{dataset_name}.csv", index=False)
    #     # quantifier_evaluator.save_evaluation_table()

    # quantifier_evaluator.aggregate_evaluation_table()
    # quantifier_evaluator.save_evaluation_table('./aggregated_evaluation_table.csv')