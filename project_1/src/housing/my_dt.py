import pandas as pd
import numpy as np
import sys
import math

from typing import Optional

import xgboost as xgb

from sklearn.metrics import log_loss
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from sklearn import tree
import matplotlib.pyplot as plt


class MyDT:
    def __init__(self, random_state: int, num_features, cat_features) -> None:
        self.random_state = random_state

        self.clf = tree.DecisionTreeRegressor(random_state=random_state)

        self.num_features = num_features
        self.cat_features = cat_features

        self.unknown_level = -1
        self.missing_level = -9999

    def _create_pipeline(self, X: pd.DataFrame, y: Optional[pd.Series],
                         training_or_scoring: str,
                         imputation_strategy: str = 'median',
                         fill_value=None) -> [pd.DataFrame, pd.Series]:

        X[self.cat_features] = X[self.cat_features].astype('category')

        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy=imputation_strategy,
                                      fill_value=fill_value,
                                      add_indicator=True)),
            ('scaler', StandardScaler())  # Required for RBF kernel
        ])

        # Pandas categoricals apparently encode missing as -1 anyway
        # See https://medium.com/bigdatarepublic/integrating-pandas-and-scikit-learn-with-pipelines-f70eb6183696
        cat_pipeline = Pipeline([
            ('missing', SimpleImputer(strategy="constant", fill_value=self.missing_level)),
            ('imputer', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=self.unknown_level))
        ])

        full_pipeline = ColumnTransformer([
            ("numeric", num_pipeline, self.num_features),
            ("cat", cat_pipeline, self.cat_features)
        ])

        if training_or_scoring == 'training':

            self.pipeline = full_pipeline
            X = self.pipeline.fit_transform(X)

        elif training_or_scoring == 'scoring':
            X = self.pipeline.transform(X)

        else:
            raise ValueError("Please specify either 'training' or 'scoring")

        return X, y

    def tune_parameters(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Runs k-fold validation to find the best parameters

        Note: In general you run tune_parameters on the training set
        and leave the validation set as an out of sample check on
        performance

        Note: if the spec were up to me I'd include parameters for num of folds,
        which hyperparams to tune, etc.

        :param X: pandas dataframe to be used for training
        :param y: pandas series that contains the targets
        :return: a dictionary containing the best params and the average scores
        """

        X, _ = self._create_pipeline(X, y, "training",
                                     "constant", -9999)

        parameters = {"criterion": ['mse', 'friedman_mse'],
                      "max_features": ['auto', 'sqrt'],
                      "max_depth": [3, 5, 9, 15, 25],
                      "min_samples_leaf": [1, 3, 5, 10, 25]
                      # "ccp_alpha": [0.0, 0.1, 0.3, 0.5]
                      }

        self.clf = GridSearchCV(self.clf, parameters, scoring=('neg_mean_absolute_error',
                                                               'neg_root_mean_squared_error'),
                                refit="neg_root_mean_squared_error", n_jobs=-1, verbose=3)

        self.clf.fit(X, y)

        cv_results = self.clf.cv_results_
        results_df = pd.DataFrame({"params": cv_results['params'],
                                   "mean_fit_time": cv_results['mean_fit_time'],
                                   "mean_score_time": cv_results['mean_score_time'],
                                   "mse_rank": cv_results['rank_test_neg_mean_absolute_error'],
                                   "mse_results": cv_results['mean_test_neg_mean_absolute_error'],
                                   "rmse_rank": cv_results['rank_test_neg_root_mean_squared_error'],
                                   "rmse_results": cv_results['mean_test_neg_root_mean_squared_error']
                                   })

        return self.clf, results_df


    def prune_tree(self, parameters, X_train, y_train, X_valid, y_valid):
        # Following code adapted from sklearn documentation for pruning
        # https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html

        print("Decision Tree Parameters: ", parameters)
        clf = tree.DecisionTreeRegressor(**parameters)
        path = clf.cost_complexity_pruning_path(X_train, y_train)
        ccp_alphas, impurities = path.ccp_alphas, path.impurities

        clfs = []
        for ccp_alpha in ccp_alphas:
            parameters['ccp_alpha'] = ccp_alpha
            clf = tree.DecisionTreeRegressor(**parameters)
            clf.fit(X_train, y_train)
            clfs.append(clf)

        train_scores = [clf.score(X_train, y_train) for clf in clfs]
        test_scores = [clf.score(X_valid, y_valid) for clf in clfs]

        fig, ax = plt.subplots()
        ax.set_xlabel("alpha")
        ax.set_ylabel("accuracy")
        ax.set_title("Accuracy vs alpha for training and testing sets")
        ax.plot(ccp_alphas, train_scores, marker='o', label="train",
                drawstyle="steps-post")
        ax.plot(ccp_alphas, test_scores, marker='o', label="test",
                drawstyle="steps-post")
        ax.legend()
        plt.show()