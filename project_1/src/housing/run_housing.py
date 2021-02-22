from helper_files.helper_methods import LoadDataset
from helper_files.data_munging_methods import DataMunge

from housing.my_svm import MySVM
from housing.my_dt import MyDT
from housing.my_knn import MyKNN
from housing.my_xgboost import MyXGB

import pandas as pd

class Housing:
    def __init__(self):
        self.dataset = "data/housing/kc_house_data.csv"
        self.date_features = ['date']
        self.target = 'price'
        self.index_col = 'id'

    def define_features(self):
        # See https://www.slideshare.net/PawanShivhare1/predicting-king-county-house-prices
        # for data dictionary
        drop_features = ['zipcode']

        # Note: view, condition, and grade could all be categorical as well
        # TODO: play around with different representations?
        cat_features = []

        # TODO: engineer features for price differences between prices?
        num_features = ['bedrooms',
                        'bathrooms',
                        'sqft_living',
                        'sqft_lot',
                        'floors',
                        'waterfront',
                        'sqft_above',
                        'sqft_basement',
                        'view',
                        'condition',
                        'grade',
                        'yr_built',
                        'yr_renovated',
                        'sqft_living15',  # Compared to 15 nearest neighbors
                        'sqft_lot15',
                        'lat',
                        'long'
                        ]

        return cat_features, num_features, drop_features

    def run_dt(self, X_train, y_train, X_valid, y_valid, num_features, cat_features):
        my_dt = MyDT(random_state=42, num_features=num_features,
                     cat_features=cat_features)

        my_dt_clf, results_df = my_dt.tune_parameters(X_train, y_train)
        results_df.to_csv("dt_results_df.csv", index=False)

        # my_dt.prune_tree(my_dt_clf.best_params_, X_train, y_train, X_valid, y_valid)

        parameters = {"min_samples_leaf": 10}
        train_sizes, train_scores, valid_scores, \
        fit_times, score_times = my_dt.run_learning_curve(X_train, y_train, parameters)

        learning_curve_dt = pd.DataFrame({"train_sizes": train_sizes,
                                          "train_scores": train_scores,
                                          "valid_scores": valid_scores,
                                          "fit_times": fit_times,
                                          "score_times": score_times})
        learning_curve_dt.to_csv("dt_learning_curve_results.csv", index=False)


        """
        parameters = my_knn_clf.best_params_
        train_sizes, train_scores, valid_scores, \
        fit_times, score_times = my_knn.run_learning_curve(X_train, y_train, parameters)

        results = my_knn.run_cv( X_train, y_train, parameters, 5)

        return my_knn_clf
        """

    def run_svm(self, X_train, y_train, num_features, cat_features):
        my_svm = MySVM(random_state=42, num_features=num_features,
                       cat_features=cat_features)
        """
        my_svm_clf, results_df = my_svm.tune_parameters(X_train, y_train)
        results_df.to_csv("svm_results_df.csv", index=False)
        """

        parameters = {"kernel": 'linear', "C": 25}
        print("Staring learning curve")
        train_sizes, train_scores, valid_scores, \
        fit_times, score_times = my_svm.run_learning_curve(X_train, y_train, parameters)

        learning_curve_dt = pd.DataFrame({"train_sizes": train_sizes,
                                          "train_scores": train_scores,
                                          "valid_scores": valid_scores,
                                          "fit_times": fit_times,
                                          "score_times": score_times})

        learning_curve_dt.to_csv("svm_learning_curve_results.csv", index=False)


    def run_knn(self, X_train, y_train, X_valid, y_valid,  num_features, cat_features):

        my_knn = MyKNN(random_state=42, num_features=num_features,
                       cat_features=cat_features)
        """
        my_knn_clf, results_df = my_knn.tune_parameters(X_train, y_train)
        results_df.to_csv("knn_results_df.csv", index=False)
        parameters = my_knn_clf.best_params_
        print('Best parameters', parameters)
        """

        parameters = {"n_neighbors": 10, "metric": "euclidean"}
        train_sizes, train_scores, valid_scores, \
        fit_times, score_times = my_knn.run_learning_curve(X_train, y_train, parameters)

        learning_curve_dt = pd.DataFrame({"train_sizes": train_sizes,
                                          "train_scores": train_scores,
                                          "valid_scores": valid_scores,
                                          "fit_times": fit_times,
                                          "score_times": score_times})

        learning_curve_dt.to_csv("knn_learning_curve_results.csv", index=False)

        # results = my_knn.run_cv( X_train, y_train, parameters, 5)

        # return my_knn_clf

    def run_xgb(self, X_train, y_train, X_valid, y_valid, num_features, cat_features):
        my_xgb = MyXGB(random_state=42, num_features=num_features,
                       cat_features=cat_features)

        """
        my_xgb_clf, results_df = my_xgb.tune_parameters(X_train, y_train)
        results_df.to_csv("xgb_results_df.csv", index=False)

        parameters = my_xgb_clf.best_params_
        """
        parameters = {'max_depth': 6}
        train_sizes, train_scores, valid_scores, \
        fit_times, score_times = my_xgb.run_learning_curve(X_train, y_train, parameters)

        learning_curve_dt = pd.DataFrame({"train_sizes": train_sizes,
                                          "train_scores": train_scores,
                                          "valid_scores": valid_scores,
                                          "fit_times": fit_times,
                                          "score_times": score_times})

        learning_curve_dt.to_csv("xgb_learning_curve_results.csv", index=False)

        """
        results = my_xgb.run_cv( X_train, y_train, parameters, 5)

        parameters={"random_state": 42}
        exp_results = my_xgb.run_learning_iteration_curve(X_train, y_train, X_valid, y_valid, parameters)

        iterations_curve_df = pd.DataFrame({"training": exp_results["validation_0"]['mae'],
                                            "validation": exp_results['validation_1']['mae']})
        iterations_curve_df.to_csv("xgb_iterations_curve_results.csv", index=True)
        # return my_xgb_clf
        """

    def run_housing(self):
        print("Running housing experiment")
        cat_features, num_features, drop_features = self.define_features()

        load_dataset = LoadDataset(self.dataset, self.target,
                                   self.index_col, date_cols=self.date_features)
        X, y = load_dataset.split_target()

        datamunge = DataMunge()

        X, new_datepart_features = datamunge.transform_variables(X, cat_features,
                                                                 num_features,
                                                                 self.date_features,
                                                                 drop_features)

        num_features.extend(new_datepart_features)

        X_train, X_valid, X_test, y_train, y_valid, y_test = load_dataset.partition(X, y)

        print("Dataset size: ", X_train.shape)

        self.run_svm(X_train, y_train, num_features, cat_features)
