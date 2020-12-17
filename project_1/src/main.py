from my_xgboost import MyXGB
from helper_methods import LoadDataset

if __name__ == '__main__':
    dataset = "data/fast iron 100k data.csv"
    load_dataset = LoadDataset(dataset, "SalePrice")
    X, y = load_dataset.split_target()

    # Setup
    my_xgb = MyXGB(random_state=42)
    X_train, X_valid, X_test, y_train, y_valid, y_test = my_xgb.partition(X, y,
                                                                          test_percentage=0.2,
                                                                          val_percentage=0.2)
    print(X_train.info())
    print(X_train.dtypes.value_counts())
    """
    # Fit
    gbm.fit(X_train, y_train)

    # Predict
    print(gbm.predict(X_valid))
    print(gbm.predict_proba(X_valid))

    # Evaluate
    print("Results: ", gbm.evaluate(X_valid, y_valid))

    # Tune
    tuned_gbm = gbm.tune_parameters(X_train, y_train)
    gbm.evaluate(X_valid, y_valid)
    print(tuned_gbm)
    """