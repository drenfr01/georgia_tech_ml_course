import pandas as pd
import sys

from sklearn.model_selection import train_test_split


class LoadDataset:
    """ A class used to load and partition dataset

        It contains a helper method to return the feature and target


    """
    def __init__(self, data_file: str, target_feature: str, index_col: str,
                 date_cols: list[str]) -> None:
        """
        Parameters:

        :param data_file: str
            The relative path to the file from the root directory
        :param target_feature: str
            The value you are trying to predict
        :raises FileNotFoundError
        """
        try:
            self.df = self._load_dataset(data_file, index_col, date_cols)
        except FileNotFoundError:
            sys.exit("Could not find specified file, double-check filepath")

        self.y = target_feature

    @staticmethod
    def _load_dataset(data_file: str, index_col: str, date_cols: list[str],
                      print_info: bool = False) -> pd.DataFrame:

        df = pd.read_csv(data_file, index_col=index_col, parse_dates=date_cols, low_memory=False)
        if print_info:
            print(df.info())
        return df

    def split_target(self) -> [pd.DataFrame, pd.Series]:
        """
        :return:
            Returns the target and features
        """
        y = self.df[self.y]
        x = self.df.drop(self.y, axis=1)

        return x, y

    @staticmethod
    def partition(X: pd.DataFrame, y: pd.Series,
                  random_state: int = 42,
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
                                                            random_state=random_state,
                                                            shuffle=shuffle)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=val_percentage,
                                                              random_state=random_state,
                                                              shuffle=shuffle)

        return X_train, X_valid, X_test, y_train, y_valid, y_test
