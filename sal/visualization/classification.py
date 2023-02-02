"""
Hilfsmittel für die Visualisierung von Klassifikationsproblemen.
"""
from typing import Tuple, Optional, List

import pandas as pd

from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from .rocaur import ROCAUC
from yellowbrick.classifier import ConfusionMatrix, ClassificationReport, PrecisionRecallCurve


def performance(model: BaseEstimator,
                X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series,
                classes: List[str], figsize: Tuple[int, int] = (12, 8), save_fig_to: Optional[str] = None) -> None:
    """Erzeugt :ref:`Konfusionsmatrix`, :ref:`Klassifikationsreport`, :ref:`Precision-Recall-<Precision-Recall-Kurve>`
    und :ref:`ROC-Kurve` für Klassifikationsmodell anhand des Test-Set.

    :param model: Klassifikator
    :type model: BaseEstimator
    :param X_train: Trainingsdaten
    :type X_train: pd.DataFrame
    :param X_test: Testdaten
    :type X_test: pd.DataFrame
    :param y_train: Traingslabels
    :type y_train: pd.Series
    :param y_test: Testlabels
    :type y_test: pd.Series
    :param classes: Bezeichner für Labels
    :type classes: list[str]
    :param figsize: Tupel mit Breite und Höhe der Diagramme
    :type figsize: tuple
    :param save_fig_to: Pfad für das zu erstellende Diagramm
    :type save_fig_to: str
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    visualgrid = [
        ConfusionMatrix(model, ax=axes[0][0], classes=classes),
        ClassificationReport(model, ax=axes[0][1], classes=classes),
        PrecisionRecallCurve(model, ax=axes[1][0], classes=classes),
        ROCAUC(model, ax=axes[1][1], macro=False, micro=True, per_class=False)
    ]

    for viz in visualgrid:
        viz.fit(X_train, y_train)
        viz.score(X_test, y_test)
        viz.finalize()

    if save_fig_to is not None:
        fig.savefig(save_fig_to)
