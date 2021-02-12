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

from sklearn.kernel_approximation import Nystroem
from sklearn import svm
from sklearn.svm import SVR

class MySVM:
    def __init__(self, random_state: int, num_features, cat_features) -> None:
        self.random_state = random_state

        self.clf = svm.SVR(cache_size=1000)

        self.num_features = num_features
        self.cat_features = cat_features

        self.missing_level_name = 'missing'

    def _create_pipeline(self, X: pd.DataFrame, y: Optional[pd.Series],
                         training_or_scoring: str,
                         imputation_strategy: str = 'median') -> [pd.DataFrame, pd.Series]:

        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy=imputation_strategy, add_indicator=True)),
            ('scaler', StandardScaler())  # Required for RBF kernel
        ])

        # Pandas categoricals apparently encode missing as -1 anyway
        # See https://medium.com/bigdatarepublic/integrating-pandas-and-scikit-learn-with-pipelines-f70eb6183696
        cat_pipeline = Pipeline([
            ('imputer', OneHotEncoder(handle_unknown='ignore'))
        ])

        full_pipeline = ColumnTransformer([
            ("numeric", num_pipeline, self.num_features),
            ("cat", cat_pipeline, ['zipcode'])
        ])

        if training_or_scoring == 'training':
            # Note: I might be able to handle setting this state in
            # a more elegant way
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
        num_features = X.shape[1]
        sqrt_num_features = round(math.sqrt(num_features))
        max_percent_features = round(0.4 * num_features)

        min_samples_range = np.linspace(0.1, 1.0, 5, endpoint=True)

        parameters = {"kernel": ['linear', 'rbf'],
                      "C": [0.1,1.0, 2.0, 10.0]
                      }

        self.clf = GridSearchCV(self.clf, parameters, scoring=('neg_mean_absolute_error',
                                                               'neg_root_mean_squared_error'),
                                refit="neg_root_mean_squared_error", n_jobs=-1, verbose=1)
        # Use Nystroem approximation to reduce training time
        # See https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.Nystroem.html

        # TODO: do this for each kernel
        feature_map_nystroem = Nystroem(random_state=self.random_state)
        data_transformed = feature_map_nystroem.fit_transform(X)
        self.clf.fit(data_transformed, y)

        best_params = self.clf.best_params_
        average_mae = -np.average(self.clf.cv_results_['mean_test_mean_absolute_error'])
        average_rmse = -np.average(self.clf.cv_results_['mean_test_root_mean_absolute_error'])
        best_params['scores'] = {"mae": average_mae, "rmse": average_rmse }

        return best_params
