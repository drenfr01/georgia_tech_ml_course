import pandas as pd
import numpy as np
import sys
import math

from typing import Optional

from sklearn.metrics import log_loss
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder


class MyGBM:
    """ A class used to implement a gradient boosted machine.

    Implements an sklearn pipeline with various configuration options

    Attributes:
        random_state: an integer seed for reproducible training
        clf: a GBM classifier
        cat_codes_dict: a dictionary of categorical encodings for use at train time and scoring
        unknown_level_name: a string value for new categorical levels at scoring time
        missing_level_name: a string value to encode missing for categoricals

    Methods:
        fit: fits a GBM classifer using a training dataset and target
        predict: returns a binary prediction
        predict_proba: returns the probability of each binary class
        evaluate: returns both an F1 and logloss score
        tune_parameters: uses k-fold cv and grid search to return a gbm trained on optimal hyperparams
    """
    def __init__(self, random_state: int) -> None:
        """
        :param random_state: ensures repeatable experiments & results

        """

        # TODO: not getting reproducible results, even after implementing
        self.random_state = random_state
        self.clf = GradientBoostingClassifier(random_state=self.random_state)

        self.cat_codes_dict = {}
        self.unknown_level_name = 'unknown'
        self.missing_level_name = 'missing'

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
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

    # TODO: Drop any missing target values here?
    def partition(self, X: pd.DataFrame, y: pd.Series,
                  test_percentage: float = 0.2,
                  val_percentage: float = 0.2,
                  shuffle: bool = True) -> [pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                            pd.Series, pd.Series, pd.Series]:
        """ Partition the dataset into train, validation, and test sets

        We will never look at the test set except as a final check to
        mitigate over-fitting

        :param X: the features as a pandas dataframe
        :param y: the target as a pandas series
        :param test_percentage: the percentage of the dataset reserved as test
        :param val_percentage: the percentage of the dataset for validation
        :param shuffle: whether to shuffle the rows of dataframe

        returns: X_train, X_val, X_test, y_train, y_val, y_test
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percentage,
                                                            random_state=self.random_state,
                                                            shuffle=shuffle)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=val_percentage,
                                                              random_state=self.random_state,
                                                              shuffle=shuffle)

        return X_train, X_valid, X_test, y_train, y_valid, y_test

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

        # Transform feature datatypes
        X['emp_length'] = pd.to_numeric(X['emp_length'], errors='coerce')

        # Assemble pipeline
        num_attribs = X.select_dtypes(include='number').columns
        cat_attribs = X.select_dtypes(include='object').columns

        X[cat_attribs] = X[cat_attribs].astype('category')

        # Note: for linear models would use add_indicator flag here
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy=imputation_strategy))
        ])

        # Note: it would be much easier to just use one_hot_encoder form sklearn
        # because it has a param to handle unknown values. But it often reduces performance
        # on tree based algorithms
        cat_pipeline = Pipeline([
            ('missing', SimpleImputer(strategy="constant", fill_value=self.unknown_level_name)),
            ('imputer', OrdinalEncoder())
        ])

        full_pipeline = ColumnTransformer([
            ("numeric", num_pipeline, num_attribs),
            ("cat", cat_pipeline, cat_attribs)
        ])

        if training_or_scoring == 'training':
            # Note: I might be able to handle setting this state in
            # a more elegant way
            self.pipeline = full_pipeline

            # create an unknown level for cat codes
            X_handle_unk = self._create_novel_levels(X, cat_attribs)

            # transform all object dtypes to categoricals because they are faster
            X_handle_unk[cat_attribs] = X_handle_unk[cat_attribs].astype('category')

            # we store the cat categories we want to use later
            self.cat_codes_dict = {col: dict(enumerate(X_handle_unk[col].cat.categories))
                                   for col in cat_attribs}

            self.pipeline = self.pipeline.fit(X_handle_unk)
            X = self.pipeline.transform(X)

        elif training_or_scoring == 'scoring':
            X = self._handle_novel_levels(X, cat_attribs)
            X = self.pipeline.transform(X)
        else:
            sys.exit("Please specify either 'training' or 'scoring'")

        return X, y

    # TODO: put in input checking
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict a binary target for a dataset

        :param X: a pandas Dataframe of features for evaluation
        :return: an ndarray of shape (len(X), 1) with a binary prediction
        """

        # Note: ensure that you only fit on training data otherwise
        # you introduce target leakage for imputation
        X, _ = self._run_preprocessing_pipeline(X, None, 'scoring')
        return self.clf.predict(X)

    # TODO: check input
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the probability of each label

        :param X: a pandas Dataframe of features for evaluation
        :return: an ndarray of shape (len(X), 2) with probabilities for each class
        """
        X, _ = self._run_preprocessing_pipeline(X, None, 'scoring')
        return self.clf.predict_proba(X)

    # TODO: create input checking function that ensures length 1, etc.
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Returns an F1 score and Logloss score

        :param X: pandas dataframe of features
        :param y: pandas series of targets
        :return: dictionary with f1_score and logloss
        """

        y_pred_proba = self.predict_proba(X)
        y_pred = self.predict(X)

        my_log_loss = log_loss(y, y_pred_proba)
        my_f1_score = f1_score(y, y_pred)

        return {'f1_score': my_f1_score, 'logloss': my_log_loss}

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
                      "n_estimators": [10, 25, 50, 100],
                      "max_depth": [3, 5, 7],
                      "subsample": [0.8, 1.0],
                      "min_samples_split": min_samples_range,
                      "max_features": [sqrt_num_features, max_percent_features]
                      }

        self.clf = GridSearchCV(self.clf, parameters, scoring=('neg_log_loss', "f1"),
                                refit="neg_log_loss", n_jobs=-1, verbose=1)
        self.clf.fit(X, y)

        best_params = self.clf.best_params_
        average_logloss = -np.average(self.clf.cv_results_['mean_test_neg_log_loss'])
        average_f1 = np.average(self.clf.cv_results_['mean_test_f1'])
        best_params['scores'] = {"f1_score": average_f1, "logloss": average_logloss}

        return best_params

    def _create_novel_levels(self, X: pd.DataFrame, cat_attribs: pd.Index) -> pd.DataFrame:
        # The idea here is to create a copy of the training dataset,
        # add a new row, and then set all categoricals in new row to unknown
        X_handle_unk = X.copy()
        X_handle_unk = X_handle_unk.append(X.iloc[-1])

        # have to reset index otherwise duplicates
        X_handle_unk.reset_index(inplace=True)
        X_handle_unk.drop(columns=['Id'], axis=1, inplace=True)

        X_handle_unk.loc[X_handle_unk.index[-1], cat_attribs] = self.unknown_level_name

        return X_handle_unk

    def _handle_novel_levels(self, X: pd.DataFrame, cat_attribs: pd.Index) -> pd.DataFrame:
        # if a new level isn't present it will be mapped to N/A
        # exactly like the 'unknown' value at training time
        X[cat_attribs] = X[cat_attribs].astype('category')
        for col in cat_attribs:
            # TODO: I probably want to change this dictionary to only contain the specific categories in order
            # So make it a list instead
            cat_mappings = self.cat_codes_dict[col]

            # fun fact: in Python 3.7 dicts keep insertion order!
            t = pd.CategoricalDtype(categories=cat_mappings.values())
            X[col] = X[col].astype(dtype=t)

        return X

    def get_mygbm_params(self):
        """Getter to help with debugging

        :return: GBM parameters
        """
        return self.clf.get_params()
