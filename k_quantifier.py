from quantifier_recommender import QuantifierRecommender
import pdb

class KQuantifier:
    def __init__(self, k = 1) -> None:
        self.k = k
        self.__quantifier_recommender = QuantifierRecommender(supervised=True)

    @property
    def k(self):
        return self.__k
    
    @k.setter
    def k(self, k):
        if not isinstance(k, int):
            raise TypeError("k must be an integer")
        
        if k == 0 or k < -1:
            raise ValueError("k needs to be a positive number or -1 (to select all quantifiers)")
        
        self.__k = k
    
    def fit(self, X_train, y_train):
        meta_features_path = "recommender_data/meta_features_table.csv"
        evaluation_table_path = "recommender_data/evaluation_table.csv"
        self.__quantifier_recommender.load_meta_table_and_fit(meta_features_path, evaluation_table_path)

        ranking = self.__quantifier_recommender.predict(X_train, y_train, k=self.k)
        pdb.set_trace()
        for (quantifier, _) in ranking:
            print(quantifier)
        # return ranking
    
    def predict(self, X_test, y_test):
        pass