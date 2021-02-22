import pandas as pd

from helper_files.helper_methods import LoadDataset
from helper_files.data_munging_methods import DataMunge

from lemons.my_svm import MySVM
from lemons.my_dt import MyDT
from lemons.my_knn import MyKNN
from lemons.my_xgboost import MyXGB


class Lemons:
    def __init__(self):
        self.dataset = "data/DontGetKicked/training.csv"
        self.date_features = ['PurchDate']
        self.index_col = "RefId"
        self.target = "IsBadBuy"

    def define_features(self) -> tuple[list, list, list]:
        drop_features = ['WheelTypeID', 'WheelType']

        cat_features = ['Auction',
                        'VehYear',
                        'Make',
                        'Model',
                        'SubModel',
                        'Color',
                        'Transmission',
                        'Nationality',
                        'Size',
                        'PRIMEUNIT',
                        'AUCGUART',
                        'BYRNO',
                        'VNZIP1',
                        'VNST',  # TODO potentially redundant
                        ]

        # TODO: engineer features for price differences between prices?
        num_features = ['VehicleAge',
                        'VehOdo',
                        'MMRAcquisitionAuctionAveragePrice',
                        'MMRAcquisitionAuctionCleanPrice',
                        'MMRAcquisitionRetailAveragePrice',
                        'MMRAcquisitonRetailCleanPrice',
                        'MMRCurrentAuctionAveragePrice',
                        'MMRCurrentAuctionCleanPrice',
                        'MMRCurrentRetailAveragePrice',
                        'MMRCurrentRetailCleanPrice',
                        'VehBCost',
                        'WarrantyCost'
                        ]

        return cat_features, num_features, drop_features

    def run_dt(self, X_train, y_train, X_valid, y_valid, num_features, cat_features):
        my_dt = MyDT(random_state=42, num_features=num_features,
                     cat_features=cat_features)


        # my_dt_clf, results_df = my_dt.tune_parameters(X_train, y_train)
        # results_df.to_csv("dt_results_df.csv", index=False)


        parameters = {"max_depth": 5}
        # my_dt.prune_tree(parameters, X_train, y_train, X_valid, y_valid)
        # print("Best params: ", my_dt_clf.best_params_)
        train_sizes, train_scores, valid_scores, \
        fit_times, score_times = my_dt.run_learning_curve(X_train, y_train, parameters)

        learning_curve_dt = pd.DataFrame({"train_sizes": train_sizes,
                                          "train_scores": train_scores,
                                          "valid_scores": valid_scores,
                                          "fit_times": fit_times,
                                          "score_times": score_times})
        learning_curve_dt.to_csv("dt_learning_curve_results.csv", index=False)



    def run_svm(self, X_train, y_train, num_features, cat_features):
        my_svm = MySVM(random_state=42, num_features=num_features,
                       cat_features=cat_features)
        """
        my_svm_clf, results_df = my_svm.tune_parameters(X_train, y_train)
        results_df.to_csv("svm_results_df.csv", index=False)
        """

        parameters = {"kernel": 'linear', "C": 10}
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

    def run_lemons(self):
        print("Running lemons experiment")
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

        X_train, X_valid, X_test, y_train, y_valid, y_test = load_dataset.partition(X, y, test_percentage=0.6, val_percentage=0.2)

        print(X_train.shape)

        self.run_dt(X_train, y_train, X_valid, y_valid, num_features, cat_features)

        """
        nn = MyNet(input_size=len(num_features), num_epochs=10, batch_size=128,
                   num_features=num_features, cat_features=cat_features,
                   missing_value="Missing", X=X_train, y=y_train, save_path="./my_nn")


        nn.train_nn()
        """
