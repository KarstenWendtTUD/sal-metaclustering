"""
Hilfsmittel zur Exploration der Daten
"""

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

import pandas as pd
import numpy as np


def missing_value_correlation(df: pd.DataFrame) -> pd.Series:
    """Pearson-Korrelation zwischen der Anzahl von
    :ref:`fehlenden Werten <Fehlende Werte>` und Features.

    :param df: DataFrame
    :type df: pd.DataFrame
    :rtype: pd.Series
    """
    df['missing_values'] = df.isnull().sum(axis=1)
    imp = SimpleImputer(strategy='most_frequent')
    hot_enc = OneHotEncoder(sparse=False)

    df_obj = df.select_dtypes(exclude=['int', 'float']).copy()
    df.drop(columns=df_obj.columns.to_list(), inplace=True)
    df_obj = pd.DataFrame(imp.fit_transform(df_obj), columns=df_obj.columns.to_list())
    df_obj_ = hot_enc.fit_transform(df_obj)
    df_obj = pd.DataFrame(df_obj_, columns=hot_enc.get_feature_names(df_obj.columns.to_list()))

    df_ = imp.fit_transform(df)
    df = pd.DataFrame(df_, columns=df.columns.to_list())
    df = pd.DataFrame(df.join(df_obj), columns=df.columns.to_list() + df_obj.columns.to_list())

    return df.corrwith(df['missing_values'], axis=0).sort_values(ascending=False)


def pairwise_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """paarweise Pearson-Korrelation zwischen den Attributen. Vorraussetzung ist
    die Vollständigkeit der Werte.

    :param df: DataFrame
    :type df: pd.DataFrame
    :rtype: pd.DataFrame
    :return: DataFrame mit Spalten für die Attribute und Korrelationskoeffizient
    """
    assert df.isnull().sum().sum() == 0, 'Werte sind nicht vollständig. ' \
                                         'Ggf. Imputation anwenden.'
    df_ = df.corr().abs()
    df_corr_triu = df_.where(~np.tril(np.ones(df_.shape)).astype(np.bool))
    df_corr_triu = df_corr_triu.stack()

    df_corr_triu.name = 'corr'
    df_corr_triu.index.names = ['A', 'B']
    return df_corr_triu.to_frame()
