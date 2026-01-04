import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

from utils.getTrainingScores import getTrainingScores
from utils.getTPRFPR import getTPRFPR
from utils.applyquantifiers import apply_quantifier

import pdb

class EnsembleQuantifier:
    def __init__(self, ranking: list = None, weights: list = None, method: str ="median") -> None:
        if method not in ["median", "weighted"]:
            raise ValueError("Method must be 'median' or 'weighted'")
        
        # self.k = k
        if ranking is None:
            self.__ranking = None
        else:
            self.ranking = ranking
        self.weights = weights
        self.__clf = None
        self.__calib_clf = None
        self.__scores = None
        self.__pos_scores = None
        self.__neg_scores = None
        self.__tprfpr = None
        self.method = method

    # @property
    # def k(self):
    #     return self.__k
    
    # @k.setter
    # def k(self, k):
    #     if not isinstance(k, int):
    #         raise TypeError("k must be an integer")
        
    #     if k == 0 or k < -1:
    #         raise ValueError("k needs to be a positive number or -1 (to select all quantifiers)")
        
    #     self.__k = k
    
    @property
    def ranking(self):
        return self.__ranking

    @ranking.setter
    def ranking(self, ranking):
        assert isinstance(ranking, list) or isinstance(ranking, tuple), "ranking must be a list/tuple of quantifiers (list of str)."
        self.__ranking = ranking
    
    @property
    def weights(self):
        return self.__weights
    
    @weights.setter
    def weights(self, weights):
        if weights is None or weights == []:
            self.__weights = [] if self.__ranking is None else [1] * len(self.__ranking)
        else:
            assert isinstance(weights, list) or isinstance(weights, tuple), "weights must be a list/tuple of floats."
            self.__weights = weights
    
    @property
    def method(self):
        return self.__method
    
    @method.setter
    def method(self, method):
        if method not in ["median", "weighted"]:
            raise ValueError("Method must be 'median' or 'weighted'")
        self.__method = method
    
    def __str__(self):
        return f"EnsembleQuantifier(ranking={self.ranking}, weights={self.weights}, method={self.method})"
    
    def fit(self, X_train, y_train):
        if not isinstance(X_train, np.ndarray) or not isinstance(y_train, np.ndarray):
            raise TypeError("X_train and y_train must be numpy arrays")
        
        ###### VERIFICAR ######
        self.__scaler = StandardScaler()
        X_train = self.__scaler.fit_transform(X_train)
        ###### VERIFICAR ######

        self.__clf = LogisticRegression(random_state=42, n_jobs=-1)
        self.__calib_clf = CalibratedClassifierCV(self.__clf, cv=3, n_jobs=-1)
        self.__calib_clf.fit(X_train, y_train)
        self.__scores = getTrainingScores(X_train, y_train, 10, self.__clf)[0]
        self.__pos_scores = self.__scores[self.__scores["class"]==1]["scores"]
        self.__neg_scores = self.__scores[self.__scores["class"]==0]["scores"]
        self.__tprfpr = getTPRFPR(self.__scores)
        self.__clf.fit(X_train, y_train)

    def predict(self, X_test):
        assert self.ranking is not None, "The ranking of quantifiers must be provided. Set the 'ranking' attribute."

        ###### VERIFICAR ######
        X_test = self.__scaler.transform(X_test)
        ###### VERIFICAR ######
        
        if self.__method == "median":
            return self.__median_method(X_test)
        elif self.__method == "weighted":
            return self.__weighted_method(X_test)        

    def __median_method(self, X_test):
        test_scores = self.__clf.predict_proba(X_test)[:,1]
        quantifiers = self.__ranking

        predicted_prevalence_list = []
        for quantifier in quantifiers:
            predicted_prevalence_list.append(apply_quantifier(qntMethod=quantifier,
                                                              clf=self.__calib_clf,
                                                              scores=self.__scores,
                                                              p_score=self.__pos_scores,
                                                              n_score=self.__neg_scores,
                                                              train_labels=None,
                                                              test_score=test_scores,
                                                              TprFpr=self.__tprfpr,
                                                              thr=0.5,
                                                              measure='hellinger',
                                                              test_data=X_test,
                                                              test_quapy=None,
                                                              external_qnt=None))
        return np.median(predicted_prevalence_list)
    
    def __weighted_method(self, X_test):
        assert len(self.__weights) == len(self.__ranking), "The number of weights in 'weights' must be equal to the number of quantifiers in 'ranking'."
        test_scores = self.__clf.predict_proba(X_test)[:,1]
        quantifiers = self.__ranking
        weight_list = self.__weights

        final_predicted_prevalence = 0
        i = 0
        for quantifier in quantifiers:
            predicted_prevalence = apply_quantifier(qntMethod=quantifier,
                                                    clf=self.__calib_clf,
                                                    scores=self.__scores,
                                                    p_score=self.__pos_scores,
                                                    n_score=self.__neg_scores,
                                                    train_labels=None,
                                                    test_score=test_scores,
                                                    TprFpr=self.__tprfpr,
                                                    thr=0.5,
                                                    measure='hellinger',
                                                    test_data=X_test,
                                                    test_quapy=None,
                                                    external_qnt=None)
            final_predicted_prevalence += weight_list[i] * predicted_prevalence
            i += 1
        return final_predicted_prevalence

    def evaluation(self, recommender_type, recommender_evaluation, quantifiers_evaluation, k_evaluation_path: str = None):
        assert recommender_type == "regression" or recommender_type == "knn", "recommender_type must be 'regression' or 'knn'."

        ensemble_quantifier_eval = pd.DataFrame(columns=["quantifier", "dataset", "sample_size", "sampling_seed",
                                          "iteration", "alpha", "pred_prev", "abs_error", "run_time"])
        for dataset in quantifiers_evaluation["dataset"].unique().tolist():
            ranking = recommender_evaluation.loc[dataset]["predicted_ranking"]
            rows_by_dataset = quantifiers_evaluation[quantifiers_evaluation["dataset"] == dataset]
            alphas = rows_by_dataset["alpha"].unique().tolist()
            iterations = rows_by_dataset["iteration"].unique().tolist()

            for k in range(1, len(ranking) + 1):
                for alph in alphas:
                    for iter in iterations:
                        predicted_prev_list = []
                        run_time_sum = 0
                        sample_size = 0
                        sampling_seed = 0

                        if recommender_type == "regression":
                            error_list = recommender_evaluation.loc[dataset]['predicted_ranking_mae'][:k]
                            if np.any(error_list == 0):
                                error_list = np.array([1e-6 if x == 0 else x for x in error_list])
                            denominator = sum([1/err for err in error_list])
                            weight_list = [(1/err)/denominator for err in error_list]
                            recommender_type_ = "REG"
                        elif recommender_type == "knn":
                            arr_list = recommender_evaluation.loc[dataset]['predicted_ranking_arr'][:k]
                            weight_list = [arr/sum(arr_list) for arr in arr_list]
                            recommender_type_ = "KNN"
                        for i in range(0, k):
                            row = rows_by_dataset[(rows_by_dataset["alpha"] == alph) & (rows_by_dataset["quantifier"] == ranking[i]) & (rows_by_dataset["iteration"] == iter)]
                            sampling_seed = row["sampling_seed"].values[0]
                            predicted_prev_list.append(row["pred_prev"].values[0])
                            run_time_sum += row["run_time"].values[0]
                            sample_size = row["sample_size"].values[0]
                        # MEDIAN METHOD
                        ensemble_quantifier_row = {"quantifier": "("+recommender_type_+")Top-" + str(k),
                                            "dataset": dataset,
                                            "sample_size": sample_size,
                                            "sampling_seed": sampling_seed,
                                            "iteration": iter,
                                            "alpha": alph,
                                            "pred_prev": np.median(predicted_prev_list),
                                            "abs_error": np.abs(np.median(predicted_prev_list) - alph),
                                            "run_time": run_time_sum}
                        ensemble_quantifier_eval.loc[len(ensemble_quantifier_eval)] = ensemble_quantifier_row

                        # WEIGHTED METHOD
                        ensemble_quantifier_row = {"quantifier": "("+recommender_type_+")Top-" + str(k) + "+W",
                                            "dataset": dataset,
                                            "sample_size": sample_size,
                                            "sampling_seed": sampling_seed,
                                            "iteration": iter,
                                            "alpha": alph,
                                            "pred_prev": np.sum(np.array(predicted_prev_list) * np.array(weight_list)),
                                            "abs_error": np.abs(np.sum(np.array(predicted_prev_list) * np.array(weight_list)) - alph),
                                            "run_time": run_time_sum}
                        ensemble_quantifier_eval.loc[len(ensemble_quantifier_eval)] = ensemble_quantifier_row
            
            print(f"Finished {dataset}")
    
        ensemble_quantifier_eval.sort_values(by=['quantifier', 'dataset'], inplace=True)
        ensemble_quantifier_eval.reset_index(drop=True, inplace=True)
        if k_evaluation_path is not None:
            ensemble_quantifier_eval.to_csv(k_evaluation_path, index=False)
            
        return ensemble_quantifier_eval
    
    def evaluation_opt(self, recommender_type, recommender_evaluation, quantifiers_evaluation, k_evaluation_path: str = None):
        assert recommender_type == "regression" or recommender_type == "knn", "recommender_type must be 'regression' or 'knn'."

        # 1. Preparação da estrutura de dados
        # Pivotamos a tabela para que os QUANTIFICADORES virem COLUNAS.
        # Isso alinha os dados para operações matriciais (vetorização).
        # index: Identificadores únicos da amostra (exceto dataset, que tratamos no loop externo)
        # columns: O nome do quantificador
        # values: As métricas que precisamos agregar
        df_pivot = quantifiers_evaluation.pivot_table(
            index=["dataset", "sample_size", "sampling_seed", "iteration", "alpha"], 
            columns="quantifier", 
            values=["pred_prev", "run_time"],
            aggfunc='first' # Garante pegar o valor único existente
        )

        results = []
        datasets = quantifiers_evaluation["dataset"].unique().tolist()

        for dataset in datasets:
            # Recupera o ranking específico deste dataset
            ranking = recommender_evaluation.loc[dataset]["predicted_ranking"]
            
            # Seleciona apenas as linhas deste dataset na matriz pivotada
            # .loc[dataset] remove o nível 'dataset' do índice, facilitando o uso
            try:
                # O xs garante que pegamos apenas o dataset atual de forma eficiente
                dataset_data = df_pivot.xs(dataset, level="dataset") 
            except KeyError:
                continue # Dataset não encontrado no pivot (caso raro de incompatibilidade)

            # Extrai as matrizes de dados alinhadas com o Ranking
            # Se o ranking tem ['Q1', 'Q2'], pegamos as colunas na ordem exata ['Q1', 'Q2']
            # Shape: (n_amostras, n_quantificadores_total)
            preds_full = dataset_data["pred_prev"][ranking].values
            times_full = dataset_data["run_time"][ranking].values
            
            # Recupera os metadados (alpha, iteration, etc) que estão no índice
            # index_vals é um DataFrame com as colunas: sample_size, sampling_seed, iteration, alpha
            index_df = dataset_data.index.to_frame(index=False)
            alphas = index_df["alpha"].values
            
            # Lógica de Pesos (Pré-cálculo para todos os K)
            # Calculamos todos os pesos antes para não repetir conta dentro do loop
            all_k_weights = []
            
            if recommender_type == "regression":
                full_error_list = np.array(recommender_evaluation.loc[dataset]['predicted_ranking_mae'])
                # Tratamento de zero idêntico ao original, mas vetorizado
                # (Aplica a correção no array todo, o slice depois respeita a lógica original)
                full_error_list = np.where(full_error_list == 0, 1e-6, full_error_list)
                
                for k in range(1, len(ranking) + 1):
                    k_errors = full_error_list[:k]
                    denominator = np.sum(1 / k_errors)
                    weights = (1 / k_errors) / denominator
                    all_k_weights.append(weights)
                recommender_tag = "REG"

            elif recommender_type == "knn":
                full_arr_list = np.array(recommender_evaluation.loc[dataset]['predicted_ranking_arr'])
                for k in range(1, len(ranking) + 1):
                    k_arr = full_arr_list[:k]
                    weights = k_arr / np.sum(k_arr)
                    all_k_weights.append(weights)
                recommender_tag = "KNN"

            # Loop Principal sobre K (O único loop necessário)
            for k_idx, k in enumerate(range(1, len(ranking) + 1)):
                # Fatia as matrizes para pegar apenas os Top-K quantificadores
                # Shape: (n_amostras, k)
                curr_preds = preds_full[:, :k]
                curr_times = times_full[:, :k]
                weights = all_k_weights[k_idx]

                # --- Lógica Matemática (Idêntica à original) ---
                
                # 1. Run Time: Soma dos tempos dos k quantificadores
                sum_runtime = np.sum(curr_times, axis=1)

                # 2. Median Method: Mediana das predições
                median_val = np.median(curr_preds, axis=1)
                median_err = np.abs(median_val - alphas)

                # 3. Weighted Method: Soma ponderada das predições
                # np.dot faz a multiplicação linha a linha pelo vetor de pesos e soma
                weighted_val = np.dot(curr_preds, weights)
                weighted_err = np.abs(weighted_val - alphas)

                # --- Montagem Otimizada dos Resultados ---
                # Criamos dicionários em lote. Isso é muito mais rápido que append em DataFrame.
                
                # Preparamos os dados comuns
                common_data = index_df.copy()
                common_data["dataset"] = dataset
                common_data["run_time"] = sum_runtime

                # Bloco Median
                df_med = common_data.copy()
                df_med["quantifier"] = f"({recommender_tag})Top-{k}"
                df_med["pred_prev"] = median_val
                df_med["abs_error"] = median_err
                
                # Bloco Weighted
                df_wei = common_data.copy()
                df_wei["quantifier"] = f"({recommender_tag})Top-{k}+W"
                df_wei["pred_prev"] = weighted_val
                df_wei["abs_error"] = weighted_err
                
                results.append(df_med)
                results.append(df_wei)

            print(f"Finished {dataset}")

        # Concatena tudo de uma vez (extremamente rápido)
        ensemble_quantifier_eval = pd.concat(results, ignore_index=True)

        # Garante a ordem exata das colunas do código original
        cols_order = ["quantifier", "dataset", "sample_size", "sampling_seed", 
                      "iteration", "alpha", "pred_prev", "abs_error", "run_time"]
        ensemble_quantifier_eval = ensemble_quantifier_eval[cols_order]

        # Ordenação final para garantir match perfeito com CSVs antigos
        ensemble_quantifier_eval.sort_values(by=['quantifier', 'dataset'], inplace=True)
        ensemble_quantifier_eval.reset_index(drop=True, inplace=True)
        
        if k_evaluation_path is not None:
            ensemble_quantifier_eval.to_csv(k_evaluation_path, index=False)
            
        return ensemble_quantifier_eval