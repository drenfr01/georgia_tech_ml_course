import pandas as pd
import numpy as np
import sys
import math

from typing import Optional

import xgboost as xgb

from sklearn.metrics import log_loss
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

from sklearn.model_selection import learning_curve, cross_validate

class MyXGB:
    """ A class used to implement an XGBoost.

    Implements an sklearn pipeline with various configuration options

    Attributes:
        random_state: an integer seed for reproducible training
        clf: a GBM classifier
        cat_codes_dict: a dictionary of categorical encodings for use at train time and scoring
        unknown_level_name: a string value for new categorical levels at scoring time
        missing_level_name: a string value to encode missing for categoricals

    Methods:
        fit: fits a GBM classifer using a training dataset and target
        tune_parameters: uses k-fold cv and grid search to return a gbm trained on optimal hyperparams
    """
    def __init__(self, random_state: int, num_features, cat_features) -> None:
        """
        :param random_state: ensures repeatable experiments & results

        """

        # TODO: not getting reproducible results, even after implementing
        self.random_state = random_state
        self.clf = xgb.XGBRegressor()

        self.num_features = num_features
        self.cat_features = cat_features

        self.cat_codes_dict = {}
        self.unknown_level_name = -9999
        self.missing_level_name = -1

    def fit(self, X: pd.DataFrame, y: pd.Series) -> xgb:
        """Fits a model to the provided features and target

        Note: if the spec was up to me I'd include a verbose mode
        here to see fitted params of GBM

        :param X: a pandas dataframe
        :param y: a pandas series
        :return: None
        """
        # TODO: add input checking for X & y
        X, y = self._run_preprocessing_pipeline(X, y, "training")
        self.clf.fit(X, y)


    # TODO: clean up columns
    # TODO: incorporate option for advanced iterative imputer with missing forest
    # TODO: crazy thought, include embedding layer for representation learning
    def _run_preprocessing_pipeline(self, X: pd.DataFrame, y: Optional[pd.Series],
                                    training_or_scoring: str,
                                    imputation_strategy: str = 'median') -> [pd.DataFrame, pd.Series]:
        """
        Clean up both features dataframe and target series

        :param X:
        :param y:
        :param training_or_scoring
        :param imputation_strategy:
        :return:
        """

        # Note: for linear models would use add_indicator flag here
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy=imputation_strategy))
        ])

        # Note: it would be much easier to just use one_hot_encoder form sklearn
        # because it has a param to handle unknown values. But it often reduces performance
        # on tree based algorithms
        cat_pipeline = Pipeline([
            ('missing', SimpleImputer(strategy="constant", fill_value=self.unknown_level_name)),
            ('imputer', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=self.unknown_level_name))
        ])

        full_pipeline = ColumnTransformer([
            ("numeric", num_pipeline, self.num_features),
            ("cat", cat_pipeline, self.cat_features)
        ])

        if training_or_scoring == 'training':
            # Note: I might be able to handle setting this state in
            # a more elegant way
            self.pipeline = full_pipeline
            X = self.pipeline.fit_transform(X)

        elif training_or_scoring == 'scoring':
            X = self.pipeline.transform(X)
        else:
            sys.exit("Please specify either 'training' or 'scoring'")

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

        X, _ = self._run_preprocessing_pipeline(X, y, "training")
        num_features = X.shape[1]
        sqrt_num_features = round(math.sqrt(num_features))
        max_percent_features = round(0.4 * num_features)

        min_samples_range = np.linspace(0.1, 1.0, 5, endpoint=True)

        parameters = {"learning_rate": [0.01, 0.05, 0.1, 0.3],
                      "max_depth": [3, 5, 7],
                      "subsample": [0.8, 1.0],
                      }

        self.clf = GridSearchCV(self.clf, parameters,
                                scoring=('neg_mean_absolute_error', 'neg_root_mean_squared_error'),
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

    def run_learning_curve(self, X, y, parameters):

        clf = xgb.XGBRegressor(**parameters)
        clf.fit(X,y)

        train_sizes, train_scores, valid_scores, \
        fit_times, score_times = learning_curve(clf, X, y,
                                                n_jobs=-1, verbose=3, shuffle=True,
                                                scoring='neg_mean_absolute_error',
                                                random_state=self.random_state,
                                                return_times=True)

        return train_sizes, np.mean(train_scores, axis=1), np.mean(valid_scores, axis=1), \
               np.mean(fit_times, axis=1), np.mean(score_times, axis=1)

    def run_cv(self, X, y, parameters, k):

        clf = xgb.XGBRegressor(**parameters)

        scores = cross_validate(clf, X, y,
                                n_jobs=-1, verbose=3,
                                scoring='neg_mean_absolute_error',
                                cv=k, return_train_score=True)


        return {k: np.mean(v) for k, v in scores.items()}

