from qtfrecommender import KNNRecommender, RegressionRecommender
from ensemble_quantifier import EnsembleQuantifier
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR


def knn_and_eval(n_neighbors: int):
    print(f"Running KNN with n_neighbors = {n_neighbors}")
    recommender = KNNRecommender(supervised=True, load_default=False, n_neighbors=n_neighbors)
    recommender.fit(full_set_path="./data/full_set/", train_set_path="./data/train_set/", test_set_path="./data/test_set/")
    recommender.save_meta_table("./qtfrecommender/methods/data/knn_meta_table.h5")
    recommender.persist_model(f"./qtfrecommender/methods/data/KNNRecommender_{n_neighbors}.joblib")
    knn_recommender_eval, knn_quantifiers_eval = recommender.leave_one_out_evaluation(f"./plot_data/knn_recommender_evaluation_table_{n_neighbors}.csv",
                                                                                  f"./plot_data/knn_quantifiers_evaluation_table_{n_neighbors}.csv")
    ensemble_qtf = EnsembleQuantifier()
    ensemble_qtf.evaluation("knn", knn_recommender_eval, knn_quantifiers_eval, f"./plot_data/knn_ensemble_quantifier_evaluation_table_{n_neighbors}.csv")
    print(f"Finished KNN with n_neighbors = {n_neighbors}\n")

def reg_and_eval(model):
    if isinstance(model, RandomForestRegressor):
        model_name = "RandomForests"
    elif isinstance(model, XGBRegressor):
        model_name = "XGBoost"
    elif isinstance(model, SVR):
        model_name = "SVR"
    
    print(f"Running Regression with model = {model_name}")

    recommender = RegressionRecommender(supervised=True, load_default=False, model=model)
    recommender.fit(full_set_path="./data/full_set/", train_set_path="./data/train_set/", test_set_path="./data/test_set/")
    recommender.save_meta_table("./qtfrecommender/methods/data/regression_meta_table.h5")
    recommender.persist_model(f"./qtfrecommender/methods/data/RegressionRecommender_{model_name}.joblib")
    reg_recommender_eval, reg_quantifiers_eval = recommender.leave_one_out_evaluation(f"./plot_data/reg_recommender_evaluation_table_{model_name}.csv",
                                                                                  f"./plot_data/reg_quantifiers_evaluation_table_{model_name}.csv")
    ensemble_qtf = EnsembleQuantifier()
    ensemble_qtf.evaluation("regression", reg_recommender_eval, reg_quantifiers_eval, f"./plot_data/reg_ensemble_quantifier_evaluation_table_{model_name}.csv")
    print(f"Finished Regression with model = {model_name}\n")

def knn():
    knn_and_eval(n_neighbors=1)
    knn_and_eval(n_neighbors=3)
    knn_and_eval(n_neighbors=5)
    knn_and_eval(n_neighbors=7)
    knn_and_eval(n_neighbors=9)
    knn_and_eval(n_neighbors=11)

def reg():
    reg_and_eval(model=RandomForestRegressor(n_jobs=-1))
    reg_and_eval(model=XGBRegressor(n_jobs=-1))
    reg_and_eval(model=SVR())

def run():
    reg()  
    knn()

if __name__ == "__main__":
    run()






    # =============================================================
    # =============================================================
    # =============================================================
    # =============================================================
    # =============================================================
    # =============================================================
    # =================== EXPERIMENT RUNNING... ===================
    # =============================================================
    # =============================================================
    # =============================================================
    # =============================================================
    # =============================================================