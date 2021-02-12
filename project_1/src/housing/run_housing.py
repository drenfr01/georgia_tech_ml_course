from helper_files.helper_methods import LoadDataset
from helper_files.data_munging_methods import DataMunge


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

        # Note: view, condition, and grade could all be numeric as well
        # TODO: play around with different representations?
        cat_features = ['view',
                        'condition',
                        'grade',
                        'zipcode',
                        ]

        # TODO: engineer features for price differences between prices?
        num_features = ['bedrooms',
                        'bathrooms',
                        'sqrt_living',
                        'sqft_lot',
                        'floors',
                        'waterfront',
                        'sqft_above',
                        'sqft_basement',
                        'yr_built',
                        'yr_renovated',
                        'sqft_living15',  # Compared to 15 nearest neighbors
                        'sqft_lot15'
                        ]

        return cat_features, num_features, drop_features

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

        """
        nn = MyNet(input_size=len(num_features), num_epochs=10, batch_size=128,
                   num_features=num_features, cat_features=cat_features,
                   missing_value="Missing", X=X_train, y=y_train, save_path="./my_nn")


        nn.train_nn()
        """
