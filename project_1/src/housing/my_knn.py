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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import learning_curve

from sklearn.neighbors import KNeighborsRegressor


class MyKNN:
    def __init__(self, random_state: int, num_features, cat_features) -> None:
        self.random_state = random_state

        self.clf = KNeighborsRegressor(n_jobs=-1)

        self.num_features = num_features
        self.cat_features = cat_features

        self.missing_level = -9999
        self.unknown_level = -1

    def _create_pipeline(self, X: pd.DataFrame, y: Optional[pd.Series],
                         training_or_scoring: str,
                         imputation_strategy: str = 'median') -> [pd.DataFrame, pd.Series]:

        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy=imputation_strategy, add_indicator=True)),
            ('scaler', StandardScaler())  # Required for RBF kernel
        ])

        cat_pipeline = Pipeline([
            ('missing', SimpleImputer(strategy="constant", fill_value=self.missing_level)),
            ('imputer', OneHotEncoder(handle_unknown='ignore'))
        ])

        full_pipeline = ColumnTransformer([
            ("numeric", num_pipeline, self.num_features),
            ("cat", cat_pipeline, ['zipcode'])
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

        X, _ = self._create_pipeline(X, y, "training")

        parameters = {"n_neighbors": [1,3,5,7,9],
                      "weights": ['uniform', 'distance']
                      }

        self.clf = GridSearchCV(self.clf, parameters,
                                scoring=('neg_mean_absolute_error', 'neg_root_mean_squared_error'),
                                refit="neg_root_mean_squared_error", n_jobs=-1, verbose=3)

        self.clf.fit(X, y)

        return self.clf

    def run_learning_curve(self, X, y, parameters):

        clf = KNeighborsRegressor(**parameters)
        clf.fit(X,y)

        train_sizes, train_scores, valid_scores,  \
            fit_times, score_times = learning_curve(clf, X, y,
                                                    n_jobs=-1, verbose=3, shuffle=True,
                                                    scoring='neg_mean_absolute_error',
                                                    random_state=self.random_state,
                                                    return_times=True)

        # TODO: return average train & test scores for each learning partition
        return train_sizes, np.mean(train_scores, axis=1), np.mean(valid_scores, axis=1),  \
               np.mean(fit_times, axis=1), np.mean(score_times, axis=1)
