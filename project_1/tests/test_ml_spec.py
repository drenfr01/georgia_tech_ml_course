import pytest
import numpy as np

from src.helper_methods import LoadDataset
from src.my_gbm import MyGBM


# TODO: rethink fixture strategy. If I'm copying why am I using a shared data structure?
class TestMyGBM:
    # Note: not sure of the best pattern with pytest
    # to setup fixtures. This seems to work and
    # only runs the expensive operation once per test session
    # could probably also cache this and make it faster?
    @pytest.fixture(scope="session")
    def execute_setup(self):
        # depending on runtime requirements I would either
        # create my own test dataset or use a small one
        # here I'll just use a subset of the existing dataset
        dataset = "data/DR_Demo_Lending_Club_reduced.csv"
        load_dataset = LoadDataset(dataset, "is_bad")

        X, y = load_dataset.split_target()

        # define GBM for testing
        gbm = MyGBM(random_state=42)
        X_train, X_valid, X_test, \
            y_train, y_valid, y_test = gbm.partition(X, y,
                                                     test_percentage=0.2,
                                                     val_percentage=0.2)

        return {"X": X,
                "y": y,
                "X_train": X_train,
                "y_train": y_train,
                "X_valid": X_valid,
                "y_valid": y_valid,
                "gbm":  gbm
                }

    def test_is_reproducible(self, execute_setup):

        X_train = execute_setup["X_train"].copy()
        y_train = execute_setup["y_train"].copy()

        X_valid = execute_setup["X_valid"].copy()

        test_gbm = MyGBM(random_state=42)
        new_gbm = MyGBM(random_state=42)

        test_gbm.fit(X_train, y_train)
        new_gbm.fit(X_train, y_train)

        orig_pred_prob = test_gbm.predict_proba(X_valid)
        new_pred_prob = new_gbm.predict_proba(X_valid)

        assert np.array_equal(orig_pred_prob, new_pred_prob)

    def test_handles_missing(self, execute_setup):
        # we should test here all types of missing
        # in python. Also test both numeric & categoricals
        X_train = execute_setup["X_train"].copy()
        y_train = execute_setup["y_train"].copy()

        X_valid = execute_setup["X_valid"]
        y_valid = execute_setup["y_valid"]
        gbm = execute_setup["gbm"]

        missing_types = [np.nan, None]
        for i, missing_type in enumerate(missing_types):
            # select last row
            X_train.loc[X_train.index == 6091, "zip_code"] = missing_type
            X_train.loc[X_train.index == 6091, "annual_inc"] = missing_type

        gbm.fit(X_train, y_train)
        evaluation_scores = gbm.evaluate(X_valid, y_valid)

        assert 'f1_score', 'logloss' in evaluation_scores

    def test_handles_new_levels(self, execute_setup):
        X_train = execute_setup["X_train"].copy()
        y_train = execute_setup["y_train"].copy()

        X_valid = execute_setup["X_valid"]
        y_valid = execute_setup["y_valid"]

        new_gbm = MyGBM(random_state=42)

        X_train.loc[X_train.index == 6091, "home_ownership"] = "SELLER_CARRY"

        new_gbm.fit(X_train, y_train)
        evaluation_scores = new_gbm.evaluate(X_valid, y_valid)

        assert 'f1_score', 'logloss' in evaluation_scores

    def test_returns_formatted_results(self, execute_setup):
        gbm = execute_setup["gbm"]
        valid_length = len(execute_setup["X_valid"])

        fit_results = gbm.fit(execute_setup["X_train"], execute_setup["y_train"])
        predict = gbm.predict(execute_setup["X_valid"])
        predict_proba = gbm.predict_proba(execute_setup["X_valid"])
        results = gbm.evaluate(execute_setup["X_valid"], execute_setup["y_valid"])

        # only do the first 1000 rows for speed
        tuned_params = gbm.tune_parameters(execute_setup["X_train"][:1000], execute_setup["y_train"][:1000])

        assert fit_results is None

        assert predict.shape == (valid_length, )
        assert isinstance(predict, np.ndarray)
        assert set(predict) == {0, 1}

        assert predict_proba.shape == (valid_length, 2)
        assert isinstance(predict_proba, np.ndarray)
        assert predict_proba.max(initial=0.0) <= 1.0
        assert predict_proba.min(initial=1.0) >= 0.0

        assert 'f1_score', 'logloss' in results

        tuned_param_keys = ["learning_rate", 'max_depth', 'max_features', 'min_samples_split',
                            'n_estimators', 'subsample', 'scores']
        assert all(tuned_param_key in tuned_params.keys() for tuned_param_key in tuned_param_keys)
        assert all(key in tuned_params['scores'].keys() for key in ['f1_score', 'logloss'])
