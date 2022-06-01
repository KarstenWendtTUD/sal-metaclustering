"""
Hilfsmittel zum Laden der Daten
"""

import pandas as pd


def typed_view(path: str = 'data/interim/typed_view.csv'):
    """Laden der :ref:`Schema-behafteten Projektion <Vorverarbeitung>`
    auf die Rohdaten aus der :download:`CSV-Datei <../../../data/interim/typed_view.csv>`.
    Zellen mit dem Wert `NULL` werden als missing values interpretiert. Der Index entspricht
    der Spalte "SUBJID" und die Spalten sind alphabetisch sortiert.

    :param path: Pfad zum Typed-View
    :type path: str
    :return: Pandas DataFrame mit allen Zeilen und Spalten
    :rtype: pd.DataFrame
    """
    df = pd.read_csv(path, na_values='NULL')
    df.set_index('SUBJID', inplace=True)
    return df.reindex(sorted(df.columns), axis=1)
