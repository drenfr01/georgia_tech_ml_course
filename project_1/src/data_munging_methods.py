import pandas as pd


class DataMunge:
    """ Collection of useful methods for data munging

    """
    def __init__(self):
        pass

    @staticmethod
    def _convert_to_cat(df: pd.DataFrame, cat_features: list[str]) -> pd.DataFrame:
        """Transforms all listed features to categoricals

        :param df
        :param cat_feats:
        :param include_object_feats
        :return:
        """
        df[cat_features] = df[cat_features].astype('category')

        return df

    @staticmethod
    def _extract_date_parts(df: pd.DataFrame, date_features: list[str], drop_date_feature: bool) -> pd.DataFrame:
        # Don't use year because we won't see that again at scoring time
        for date_feat in date_features:
            day_col_name = f"{date_feat}_day_of_month"
            month_col_name = f"{date_feat}_month"
            df[day_col_name] = df[date_feat].dt.day
            df[month_col_name] = df[date_feat].dt.month

        if drop_date_feature:
            df.drop(labels=date_features, axis=1, inplace=True)

        return df


    def transform_variables(self, df: pd.DataFrame, cat_features: list[str],
                            num_features: list[str], date_features: list[str],
                            drop_features: list[str]) -> pd.DataFrame:

        df = self._convert_to_cat(df, cat_features)

        # note: need to support OTV
        df = self._extract_date_parts(df, date_features, True)

        return df


