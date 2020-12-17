import pandas as pd


class DataMunge:
    """ Collection of useful methods for data munging

    """
    def __init__(self, df):
        pass

    @staticmethod
    def convert_to_cat(df, cat_feats, include_object_feats = True):
        """Transforms all listed features to categoricals

        :param df
        :param cat_feats:
        :param include_object_feats
        :return:
        """

        if include_object_feats:
            cat_feats.append(df.select_dtypes(include="object").columns)

        df[cat_feats].astype('categorical')
