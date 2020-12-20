from my_xgboost import MyXGB
from helper_methods import LoadDataset
from data_munging_methods import  DataMunge


def define_features() -> tuple[list, list, list]:
    drop_features = ['WheelTypeID']

    cat_features = ['Auction',
                    'VehYear',
                    'Make',
                    'Model',
                    'Trim',
                    'SubModel',
                    'Color',
                    'Transmission',
                    'WheelType',
                    'Nationality',
                    'Size',
                    'TopThreeAmericanName',
                    'PRIMEUNIT',
                    'AUCGUART',
                    'BYRNO',
                    'VNZIP1',
                    'VNST',  # TODO potentially redundant
                    'WarrantyCost'
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


    return (cat_features, num_features, drop_features)

if __name__ == '__main__':
    dataset = "data/DontGetKicked/training.csv"
    date_features = ['PurchDate']
    cat_features, num_features, drop_features = define_features()

    load_dataset = LoadDataset(dataset, "IsBadBuy", "RefId", date_cols=date_features)
    X, y = load_dataset.split_target()


    datamunge = DataMunge()

    X = datamunge.transform_variables(X, cat_features, num_features, date_features, drop_features)

    X_train, X_valid, X_test, y_train, y_valid, y_test = load_dataset.partition(X, y)


