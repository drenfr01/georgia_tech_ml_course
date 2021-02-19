from helper_files.helper_methods import LoadDataset
from helper_files.data_munging_methods import DataMunge

from housing.my_svm import MySVM
from housing.my_dt import MyDT
from housing.my_knn import MyKNN
from housing.my_xgboost import MyXGB


class Housing:
    def __init__(self):
        self.dataset = "data/housing/kc_house_data.csv"
        self.date_features = ['date']
        self.target = 'price'
        self.index_col = 'id'

    def define_features(self):
        # See https://www.slideshare.net/PawanShivhare1/predicting-king-county-house-prices
        # for data dictionary
        drop_features = ['lat', 'long']

        # Note: view, condition, and grade could all be categorical as well
        # TODO: play around with different representations?
        cat_features = ['zipcode']

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
                        ]

        return cat_features, num_features, drop_features

    def run_dt(self, X_train, y_train, num_features, cat_features):
        my_dt = MyDT(random_state=42, num_features=num_features,
                     cat_features=cat_features)
        my_dt_clf = my_dt.tune_parameters(X_train, y_train)

    def run_svm(self, X_train, y_train, num_features, cat_features):
        my_svm = MySVM(random_state=42, num_features=num_features,
                       cat_features=cat_features)
        best_params = my_svm.tune_parameters(X_train, y_train)

    def run_knn(self, X_train, y_train, num_features, cat_features):

        my_knn = MyKNN(random_state=42, num_features=num_features,
                       cat_features=cat_features)
        my_knn_clf, results_df = my_knn.tune_parameters(X_train, y_train)
        results_df.to_csv("knn_results_df.csv", index=False)

        parameters = my_knn_clf.best_params_
        train_sizes, train_scores, valid_scores, \
        fit_times, score_times = my_knn.run_learning_curve(X_train, y_train, parameters)

        results = my_knn.run_cv( X_train, y_train, parameters, 5)

        return my_knn_clf

    def run_xgb(self, X_train, y_train, num_features, cat_features):
        my_xgb = MyXGB(random_state=42, num_features=num_features,
                       cat_features=cat_features)
        my_xgb_clf, results_df = my_xgb.tune_parameters(X_train, y_train)
        results_df.to_csv("xgb_results_df.csv", index=False)

        parameters = my_xgb_clf.best_params_
        train_sizes, train_scores, valid_scores, \
        fit_times, score_times = my_xgb.run_learning_curve(X_train, y_train, parameters)

        results = my_xgb.run_cv( X_train, y_train, parameters, 5)

        return my_xgb_clf

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

        self.run_xgb(X_train, y_train, num_features, cat_features)