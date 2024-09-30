import pandas as pd
import os
import pdb
import time
import timeit

from quantifier_recommender import QuantifierRecommender
from quantifier_evaluator import QuantifierEvaluator
from utils_ import generate_train_test_data
from quantifier_evaluator import QuantifierEvaluator

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.base import is_regressor

from k_quantifier import KQuantifier

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

def predict_wrapper():
    q.predict(X_test, y_test)

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


    # supervised_quantifier_recommender = QuantifierRecommender.load_model("supervised_quantifier_recommender.pkl")

    
    # qtf_rec_randomforests = QuantifierRecommender(supervised=True, recommender_model=RandomForestRegressor())
    # qtf_rec_linear = QuantifierRecommender(supervised=True, recommender_model=LinearRegression())

    # qtf_rec_randomforests.fit(datasets_path="./datasets/",
    #                           train_data_path="./data/train_data/",
    #                           test_data_path="./data/test_data/")
    
    # qtf_rec_linear.fit(datasets_path="./datasets/",
    #                       train_data_path="./data/train_data/",
    #                       test_data_path="./data/test_data/")
       
    # qtf_rec_randomforests.save_meta_table("recommender_data/random_forests_meta_features_table.csv", "recommender_data/random_forests_evaluation_table.csv")
    # qtf_rec_linear.save_meta_table("recommender_data/linear_meta_features_table.csv", "recommender_data/linear_evaluation_table.csv")

    # qtf_rec_randomforests.load_meta_table_and_fit("recommender_data/random_forests_meta_features_table.csv", "recommender_data/random_forests_evaluation_table.csv")
    # qtf_rec_linear.load_meta_table_and_fit("recommender_data/linear_meta_features_table.csv", "recommender_data/linear_evaluation_table.csv")

    # qtf_rec_randomforests.leave_one_out_evaluation("recommender_data/random_forests_leave_one_out.csv")
    # qtf_rec_linear.leave_one_out_evaluation("recommender_data/linear_leave_one_out.csv")


    # recommender_evaluation_table = supervised_quantifier_recommender.leave_one_out_evaluation("recommender_data/recommender_evaluation_table_2.csv")
    # old_recommender_evaluation_table = supervised_quantifier_recommender.OLD_leave_one_out_evaluation("recommender_data/OLD_recommender_evaluation_table_2.csv")
    
    # pdb.set_trace()

    q = QuantifierRecommender(supervised=True)
    q.load_meta_table_and_fit("recommender_data/meta_features_table.csv", "recommender_data/evaluation_table.csv")
    X_test, y_test = load_test_data("namao", supervised=True)
    
    execution_time = timeit.timeit(predict_wrapper, number=20)
    print(f"Average execution time over 1000 runs: {execution_time / 20} s")
    
    start = time.perf_counter()
    ranking = q.predict(X_test, y_test)
    stop = time.perf_counter()
    
    print(f"Time: {stop - start} s")


    # k_quantifier = KQuantifier(k=11)
    # X_train, y_train, X_test, y_test = load_train_test_data("BNG")
    # ranking = k_quantifier.fit(X_train, y_train)
    # print(ranking)