"""
Allgemeine Hilfmittel.
"""
from typing import Optional

import pandas as pd
import pytablewriter


def rst_table_for(df: pd.DataFrame, title: Optional[str] = None) -> str:
    """RestructuredText-Tabelle aus Pandas DataFrame erstellen.

    :param df: DataFrame
    :type df: pd.DataFrame
    :param title: Titel der Tabelle
    :type title: str
    :return: RST-Tabelle
    :rtype: str
    """
    rst = pytablewriter.RstSimpleTableWriter()
    rst.from_dataframe(df)

    if title is not None:
        rst.table_name = 'Vergleich der Modelle'

    table = rst.dumps()
    table = table.split('\n')
    table[1] = '    :align: center'
    table.insert(2, ' ')
    return '\n'.join(table)
