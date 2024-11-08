import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR


def regressor_recommender_grid_search(model, X, y):
    pass
    # reg = 

    # C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    # class_weight = [None, "balanced"]
    
    # grid = {"C": C,
    #         "class_weight": class_weight}
    
    # search = GridSearchCV(estimator=clf,
    #                       param_grid=grid,
    #                       cv=3,
    #                       verbose=2,
    #                       n_jobs=-1)
    
    # search.fit(X_train, y_train)
    # return search.best_estimator_
        