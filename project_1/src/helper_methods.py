import pandas as pd
import sys


class LoadDataset:
    """ A class used to load and partition dataset

        It contains a helper method to return the feature and target


    """
    def __init__(self, data_file: str, target_feature: str) -> None:
        """
        Parameters:

        :param data_file: str
            The relative path to the file from the root directory
        :param target_feature: str
            The value you are trying to predict
        :raises FileNotFoundError
        """
        try:
            self.df = self._load_dataset(data_file)
        except FileNotFoundError:
            sys.exit("Could not find specified file, double-check filepath")

        self.y = target_feature

    @staticmethod
    def _load_dataset(data_file: str, print_info: bool = False) -> pd.DataFrame:
        df = pd.read_csv(data_file, index_col="SalesID", low_memory=False)
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
