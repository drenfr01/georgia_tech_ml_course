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

class MyDT:
    def __init__(self, random_state: int, num_features, cat_features) -> None:
        self.random_state = random_state

        self.clf = tree.DecisionTreeRegressor(random_state=random_state)

        self.num_features = num_features
        self.cat_features = cat_features

        self.unknown_level = -1
        self.missing_level_name = -9999

    def _create_pipeline(self, X: pd.DataFrame, y: Optional[pd.Series],
                         training_or_scoring: str,
                         imputation_strategy: str = 'median',
                         fill_value = None) -> [pd.DataFrame, pd.Series]:

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
            ('missing', SimpleImputer(strategy="constant", fill_value=self.missing_level_name)),
            ('imputer', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=self.unknown_level))
        ])

        full_pipeline = ColumnTransformer([
            ("numeric", num_pipeline, self.num_features),
            ("cat", cat_pipeline, self.cat_features)
        ])

        if training_or_scoring == 'training':

            self.pipeline = full_pipeline

            self.pipeline = self.pipeline.fit(X)
            X = self.pipeline.transform(X)

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
        num_features = X.shape[1]
        sqrt_num_features = round(math.sqrt(num_features))
        max_percent_features = round(0.4 * num_features)

        min_samples_range = np.linspace(0.1, 1.0, 5, endpoint=True)

        parameters = {"criterion": ['mse', 'friedman_mse'],
                      "max_features": ['auto', 'sqrt'],
                      "max_depth": [3, 5, 7, 11],
                      "ccp_alpha": [0.0, 0.1, 0.3, 0.5]
                      }

        self.clf = GridSearchCV(self.clf, parameters, scoring=('neg_mean_absolute_error',
                                                               'neg_root_mean_squared_error'),
                                refit="neg_root_mean_squared_error", n_jobs=-1, verbose=3)

        self.clf.fit(X, y)

        return self.clf
