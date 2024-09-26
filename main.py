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
    # supervised_quantifier_recommender = QuantifierRecommender(supervised=True)
    # unsupervised_quantifier_recommender = QuantifierRecommender(supervised=False)
    
    # supervised_quantifier_recommender.fit(datasets_path="./datasets/",
    #                                       train_data_path="./data/train_data/",
    #                                       test_data_path="./data/test_data/")
    # supervised_quantifier_recommender.persist_model("supervised_quantifier_recommender.pkl")
    



    # unsupervised_quantifier_recommender.fit(datasets_path="./datasets/",
    #                                         train_data_path="./data/train_data/",
    #                                         test_data_path="./data/test_data/")
    
    # X_test, y_test = load_test_data("BNG", supervised=True)
    # s_ranking = supervised_quantifier_recommender.predict(X_test, y_test)
    
    # X_test = load_test_data("BNG", supervised=False)
    # u_ranking = unsupervised_quantifier_recommender.predict(X_test)

    # supervised_quantifier_recommender.persist_model("s_qtf_recommender.pkl")
    # unsupervised_quantifier_recommender.persist_model("u_qtf_recommender.pkl")

    # s_qtf_rec = QuantifierRecommender.load_model("s_qtf_recommender.pkl")
    # u_qtf_rec = QuantifierRecommender.load_model("u_qtf_recommender.pkl")

    # X_test, y_test = load_test_data("BNG", supervised=True)
    # new_s_ranking = s_qtf_rec.predict(X_test, y_test)

    # X_test = load_test_data("BNG", supervised=False)
    # new_u_ranking = u_qtf_rec.predict(X_test)

    # qtf_rec = QuantifierRecommender.load_model("s_qtf_recommender.pkl")


    supervised_quantifier_recommender = QuantifierRecommender.load_model("supervised_quantifier_recommender.pkl")

    recommender_evaluation_table = supervised_quantifier_recommender.leave_one_out_evaluation("recommender_data/recommender_evaluation_table_2.csv")
    old_recommender_evaluation_table = supervised_quantifier_recommender.OLD_leave_one_out_evaluation("recommender_data/OLD_recommender_evaluation_table_2.csv")
    
    pdb.set_trace()