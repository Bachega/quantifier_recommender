from qtfrecommender import KNNRecommender, RegressionRecommender
from ensemble_quantifier import EnsembleQuantifier
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR

import pandas as pd
import seaborn as sns
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score

import mlflow


def knn_and_eval(n_neighbors: int):
    print(f"Running KNN with n_neighbors = {n_neighbors}")
    recommender = KNNRecommender(supervised=True, load_default=False, n_neighbors=n_neighbors)
    recommender.fit(full_set_path="./data/full_set/", train_set_path="./data/train_set/", test_set_path="./data/test_set/")
    
    if(n_neighbors == 1):
        recommender.save_meta_table("./qtfrecommender/methods/data/knn_meta_table.h5")

        # For Azure
        Path("outputs/qtfrecommender/methods/data/knn_meta_table.h5").parent.mkdir(parents=True, exist_ok=True)
        recommender.save_meta_table("outputs/qtfrecommender/methods/data/knn_meta_table.h5")
    
    recommender.persist_model(f"./qtfrecommender/methods/data/KNNRecommender_{n_neighbors}.joblib")

    # For Azure
    Path(f"outputs/qtfrecommender/methods/data/KNNRecommender_{n_neighbors}.joblib").parent.mkdir(parents=True, exist_ok=True)
    recommender.persist_model(f"outputs/qtfrecommender/methods/data/KNNRecommender_{n_neighbors}.joblib")
    

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
    
    if isinstance(model, RandomForestRegressor):
        recommender.save_meta_table("./qtfrecommender/methods/data/regression_meta_table.h5")

        # For Azure
        Path("outputs/qtfrecommender/methods/data/regression_meta_table.h5").parent.mkdir(parents=True, exist_ok=True)
        recommender.save_meta_table("outputs/qtfrecommender/methods/data/regression_meta_table.h5")
    
    recommender.persist_model(f"./qtfrecommender/methods/data/RegressionRecommender_{model_name}.joblib")

    # For Azure
    Path(f"outputs/qtfrecommender/methods/data/RegressionRecommender_{model_name}.joblib").parent.mkdir(parents=True, exist_ok=True)
    recommender.persist_model(f"outputs/qtfrecommender/methods/data/RegressionRecommender_{model_name}.joblib")

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

def generate_plots():
    # We concatenate the evaluation results of the quantifiers and the ensemble quantifier
    # to have a single table for the evaluation results

    # EXPERIMENT 1
    # Meta-features extraidas de full set, sem normalização
    # Quantificadores treinados em train set, com zscore
    # Quantificadores avaliados em test set, com zscore

    # EXPERIMENT 2
    # Meta-features extraidas de train set, com zscore
    # Quantificadores treinados em train set, com zscore
    # Quantificadores avaliados em test set, com zscore

    # EXPERIMENT 3 (AINDA NÃO FEITO)
    # Meta-features extraidas de train set, sem normalização
    # Quantificadores treinados em train set, com zscore
    # Quantificadores avaliados em test set, com zscore

    # path = "./plot_data/experiment-1/"

    path = "./plot_data/"

    quantifiers_eval = pd.read_csv(path+"reg_quantifiers_evaluation_table.csv")
    reg_ensemble_quantifier_eval = pd.read_csv(path+"reg_ensemble_quantifier_evaluation_table.csv")
    knn_ensemble_quantifier_eval = pd.read_csv(path+"knn_ensemble_quantifier_evaluation_table.csv")
    eval_table = pd.concat([quantifiers_eval, reg_ensemble_quantifier_eval, knn_ensemble_quantifier_eval], axis=0)


    eval_table = eval_table.groupby(["quantifier", "dataset"]).agg(
            abs_error = pd.NamedAgg(column="abs_error", aggfunc="mean"),
            run_time = pd.NamedAgg(column="run_time", aggfunc="mean")
        )
    eval_table.reset_index(inplace=True)

    def remove_quantifiers(df, quantifiers):
        return df[~df['quantifier'].isin(quantifiers)]

    quantifiers_to_remove = [
        # '(KNN)Top-1',
        '(KNN)Top-2',
        '(KNN)Top-3',
        '(KNN)Top-4',
        '(KNN)Top-5',
        '(KNN)Top-6',
        '(KNN)Top-7',
        '(KNN)Top-8',
        '(KNN)Top-9',
        '(KNN)Top-10',
        '(KNN)Top-11',
        '(KNN)Top-1+W',
        '(KNN)Top-2+W',
        '(KNN)Top-3+W',
        '(KNN)Top-4+W',
        '(KNN)Top-5+W',
        '(KNN)Top-6+W',
        '(KNN)Top-7+W',
        '(KNN)Top-8+W',
        '(KNN)Top-9+W',
        '(KNN)Top-10+W',
        '(KNN)Top-11+W',
        '(REG)Top-1',
        '(REG)Top-2',
        '(REG)Top-3',
        '(REG)Top-4',
        '(REG)Top-5',
        '(REG)Top-6',
        '(REG)Top-7',
        '(REG)Top-8',
        '(REG)Top-9',
        '(REG)Top-10',
        '(REG)Top-11',
        '(REG)Top-1+W',
        '(REG)Top-2+W',
        '(REG)Top-3+W',
        '(REG)Top-4+W',
        '(REG)Top-5+W',
        '(REG)Top-6+W',
        '(REG)Top-7+W',
        '(REG)Top-8+W',
        '(REG)Top-9+W',
        '(REG)Top-10+W',
        '(REG)Top-11+W',
    ]
    eval_table = remove_quantifiers(eval_table, quantifiers_to_remove)

    margin_left = 0.05
    margin_right= 0.99
    margin_top = 0.99
    margin_bottom = .24 # .23
    plt_width = 52 # 38
    plt_heigth = 24 # 18
    plot_rotation = 75
    axis_font_size = 56 # 50
    labels_size = 60 # 60

    def boxplotMae(sample, file=""):
        sample['error_rank'] = sample.groupby(['dataset'], as_index=False )['abs_error'].rank(method='average', ascending = True)

        order = sample.groupby('quantifier')['error_rank'].mean().sort_values().index

        palette = sns.color_palette('Spectral', sample['quantifier'].nunique())

        with sns.axes_style("whitegrid"):
            plt.figure(figsize=(plt_width, plt_heigth))
            plt.subplots_adjust(left=margin_left, bottom=margin_bottom, right=margin_right, top=margin_top)
            ax=sns.boxplot(data=sample, x='quantifier', y='error_rank', order = order, palette = palette, hue='quantifier', legend=False)

            plt.xticks(rotation =plot_rotation ,fontsize = axis_font_size)
            plt.yticks(fontsize = axis_font_size)

            ax.set_xlabel("Quantifiers",fontsize=labels_size)
            ax.set_ylabel("Avg. ranking",fontsize=labels_size)

        fig = ax.figure
        mlflow.log_figure(fig, "boxplot_mae.png")
        # plt.show()
        # if file != "":
        #     ax.figure.savefig('./results/'+file+ '.pdf', format="pdf", facecolor='w')

        plt.close()
        return sample

    boxplotMae(eval_table, file="boxplot_mae")

    # path = "./plot_data/experiment-1/"
    path = "./plot_data/"


    reg_ensemble_quantifier_eval = pd.read_csv(path+"reg_recommender_evaluation_table.csv", index_col=0)
    knn_ensemble_quantifier_eval = pd.read_csv(path+"knn_recommender_evaluation_table.csv", index_col=0)



    qtf_list = reg_ensemble_quantifier_eval['predicted_ranking'].apply(eval).iloc[0]
    qtf_dict = {qtf: {"predicted": [], "true": []} for qtf in qtf_list}



    reg_ensemble_quantifier_eval['predicted_ranking'] = reg_ensemble_quantifier_eval['predicted_ranking'].apply(eval)
    reg_ensemble_quantifier_eval['true_ranking'] = reg_ensemble_quantifier_eval['true_ranking'].apply(eval)
    reg_ensemble_quantifier_eval['predicted_ranking_mae'] = reg_ensemble_quantifier_eval['predicted_ranking_mae'].apply(eval)
    reg_ensemble_quantifier_eval['true_ranking_mae'] = reg_ensemble_quantifier_eval['true_ranking_mae'].apply(eval)

    knn_ensemble_quantifier_eval['predicted_ranking'] = knn_ensemble_quantifier_eval['predicted_ranking'].apply(eval)
    knn_ensemble_quantifier_eval['true_ranking'] = knn_ensemble_quantifier_eval['true_ranking'].apply(eval)
    knn_ensemble_quantifier_eval['predicted_ranking_arr'] = knn_ensemble_quantifier_eval['predicted_ranking_arr'].apply(eval)
    knn_ensemble_quantifier_eval['true_ranking_arr'] = knn_ensemble_quantifier_eval['true_ranking_arr'].apply(eval)

    for i in range(len(reg_ensemble_quantifier_eval)):
        row = reg_ensemble_quantifier_eval.iloc[i]

        predicted_ranking = row['predicted_ranking']
        predicted_ranking_mae = row['predicted_ranking_mae']
        true_ranking = row['true_ranking']
        true_ranking_mae = row['true_ranking_mae']

        for qtf, mae in zip(predicted_ranking, predicted_ranking_mae):
            qtf_dict[qtf]['predicted'].append(mae)

        for qtf, mae in zip(true_ranking, true_ranking_mae):
            qtf_dict[qtf]['true'].append(mae)

    for key, value in qtf_dict.items():
        quantifier = key
        predicted_ranking_mae = value['predicted']
        true_ranking_mae = value['true']

        r2 = r2_score(true_ranking_mae, predicted_ranking_mae)
        print(f"R^2 value for {quantifier}: {r2}")

        plt.figure(figsize=(10, 6))
        plt.scatter(true_ranking_mae, predicted_ranking_mae, color='blue')
        plt.plot([min(true_ranking_mae), max(true_ranking_mae)], [min(true_ranking_mae), max(true_ranking_mae)], color='red', linestyle='--')
        plt.xlabel('True Ranking MAE', fontsize=16)
        plt.ylabel('Predicted Ranking MAE', fontsize=16)
        plt.title(f'R^2 value for {quantifier} Recommender: {r2}', fontsize=16)

        mlflow.log_figure(plt.gcf(), f"scatter_plots/{quantifier}_mae_comparison.png")
        # pdf_filename = f"./results/{quantifier}_recommender_performance.pdf"
        # plt.savefig(pdf_filename, format='pdf', bbox_inches='tight')

        plt.close()
        # plt.show()

def run():
    reg()  
    knn()
    generate_plots()

if __name__ == "__main__":
    run()