"""
Hilfsmittel zur Kodierung von Daten.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def binary(df: pd.DataFrame) -> pd.DataFrame:
    """Binäre Attribute in numerische Attribute umwandeln. Fehlende Werte bleiben erhalten.
    Einsatz primär bei :ref:`Mutationsindikatoren`.

    :param df: DataFrame
    :type df: pd.DataFrame
    :return: DataFrame mit neuen Attributen entsprechend One Hot Encoding
    :rtype: pd.DataFrame
    """
    df_ = df.applymap(lambda v: {
        'y': 1.0, '1.0': 1.0,
        'n': 0.0, '0.0': 0.0,
        'nan': np.nan
    }[str(v).lower()])

    return df_


def binary_one_hot(df: pd.DataFrame) -> pd.DataFrame:
    """Binäre Attribute in ternäre Attribute (Y/N/U) umwandeln,
    wobei 'U' für unbekannte/fehlende Werte steht. Einsatz z.B.
    bei :ref:`Mutationsindikatoren`.

    .. code-block::

        binary(df[['CEBPA']]) == df[['CEBPA_Y', 'CEBPA_N', 'CEBPA_U']]

    :param df: DataFrame
    :type df: pd.DataFrame
    :return: DataFrame mit neuen Attributen entsprechend One Hot Encoding
    :rtype: pd.DataFrame
    """
    df_ = df.applymap(lambda v: {
        'y': 'Y', '1.0': 'Y', '1': 'Y', 1: 'Y',
        'n': 'N', '0.0': 'N', '0': 'N', 0: 'N',
        'nan': 'U'
    }[str(v).lower()])

    df_ = pd.concat([df_, pd.get_dummies(df_)], axis=1)
    df_.drop(df.columns.to_list(), axis=1, inplace=True)

    return df_


class BinaryNominalEncoder(BaseEstimator, TransformerMixin):
    """
    Kodierer nominaler Binärattribute für die Verwendung in einer Pipeline. Die Abbildung
    erfolgt auf *Y* und *N*, bzw. ``np.nan`` im Falle fehlender Werte.
    """

    #: Mapping für die Zuordnung eingehender Werte
    mapping = {
        'y': 'Y', '1.0': 'Y', '1': 'Y',
        'n': 'N', '0.0': 'N', '0': 'N',
        'nan': np.nan
    }

    def fit(self, X, y=None):
        """Dieser Transformierer betreibt kein Fitting"""
        return self

    def transform(self, X: pd.DataFrame, y=None):
        """
        Tranformiere alle Werte innerhalb eines Pandas DataFrame nach dem definierten ``mapping``

        :param X: DataFrame
        :type X: pd.DataFrame
        :param y: ungenutzte Labels
        :return: DataFrame mit transformierten Werten
        :rtype: pd.DataFrame
        """
        assert type(X) == pd.DataFrame, 'BinaryNominalEncoder takes DataFrame only'
        return X.applymap(self._transform_value)

    def _transform_value(self, value):
        str_value = str(value).lower()

        if str_value in self.mapping:
            return self.mapping[str_value]
        else:
            return value
