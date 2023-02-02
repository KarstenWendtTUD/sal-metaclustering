"""
Modelle für supervised Learning Methoden
"""

import logging
import os

from warnings import simplefilter
from typing import List
from operator import itemgetter

import pandas as pd
import numpy as np
import yaml
import pickle

from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_validate, StratifiedKFold, KFold

from sal.data.load import typed_view
from sal.data.sampling import stratified_train_test_split, random_train_test_split
from sal.features.scaling import z_score
from sal.features.selection import drop_labels, drop_features_by_support
from sal.models import model_factory_mapping
from sal.visualization.classification import performance as classification_performance


simplefilter(action='ignore', category=FutureWarning)


def _create_models(model_configurations: dict):
    models = []

    for config_name in model_configurations:
        config: dict = model_configurations[config_name]
        model_type = config.get('type', None)
        params = config.get('params', None)

        if model_type is None:
            continue

        model = model_factory_mapping[model_type]()

        if params is not None:
            model.set_params(**params)

        models.append((config_name, model))

    return models


def conduct_classification_experiment(experiment_name, input_file_path, config: dict):
    """Train and evaluate of classification models.

    :param experiment_name: experiment name
    :type experiment_name: str
    :param input_file_path: path the feature CSV file (e.g. *data/processed/default.csv*)
    :type input_file_path: str
    :param config: configuration
    :type config: dict
    """
    logger = logging.getLogger(__name__)
    logger.info('conduct "{}" ...'.format(experiment_name))

    df = typed_view(input_file_path)

    label_column_name = config['label_column_name']

    experiment_report_path = os.path \
        .join('reports/classification', label_column_name, experiment_name)

    os.makedirs(experiment_report_path, exist_ok=True)

    # extract rows with classification unknown
    if 'label_unknown_column_name' in config and 'label_unknown_value' in config:
        df_unknown_index = df.index[df[config['label_unknown_column_name']]
                                    == config['label_unknown_value']]
        df_unknown = df.loc[df_unknown_index].copy()
        df_unknown.drop(inplace=True, columns=label_column_name)
        df_unknown.to_csv('data/processed/{}_unknown.csv'.format(label_column_name))
        df.drop(inplace=True, index=df_unknown_index)

    if 'test_set' in config:
        df_test = typed_view(config['test_set'])
        y_train = df[label_column_name].copy()
        X_train = df.copy()
        y_test = df_test[label_column_name].copy()
        X_test = df_test.copy()
        logger.info('using test set from "{}"'.format(config['test_set']))
    else:
        # Split Sets
        X_train, X_test, y_train, y_test = stratified_train_test_split(df, label_column_name)
        logger.info('using stratified train/test split')

    # Handle Labels
    drop_labels([X_train, X_test], more=(config['drop']))

    # Scale Data
    X_train = z_score(X_train)
    X_test = z_score(X_test)

    # Select Features
    logger.info('selecting features ...'.format(experiment_name))
    X_train, X_test = drop_features_by_support(
        X_train, X_test, y_train, classification=False,
        top_feature_count=config['top_feature_count'], top_feature_support_min=config['top_feature_support_min'],
        experiment_report_path=experiment_report_path)

    results = pd.DataFrame()
    for name, model in _create_models(config['classifiers']):
        logger.info('Evaluating {}'.format(name))

        try:
            result = cross_validate(
                model, X_train, y_train,
                cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=90210), n_jobs=-1,
                scoring=['precision_weighted', 'recall_weighted', 'f1_weighted'])

            results = results.append(pd.DataFrame([[
                name,
                np.mean(result['test_f1_weighted']),
                np.mean(result['test_precision_weighted']),
                np.mean(result['test_recall_weighted'])
            ]], columns=['Model', 'F1', 'Precision', 'Recall']))

            classification_performance(model, X_train, X_test, y_train, y_test,
                                       classes=(config['label_values']),
                                       save_fig_to=os.path.join(experiment_report_path, name + '.jpg'))

            model.fit(X_train, y_train)
            model_path = os.path.join('data/models', label_column_name, experiment_name)
            os.makedirs(model_path, exist_ok=True)
            model_file = os.path.join(model_path, f'{name}.pkl')
            with open(model_file, 'wb') as file:
                pickle.dump(dict(
                    model=model,
                    features=X_train.columns.tolist()
                ), file)
            logger.info(f'Model written to "{model_file}"')

        except Exception as e:
            logger.error(e)

    results.sort_values(['F1']).to_csv(os.path.join(experiment_report_path, 'vergleich.csv'))

    results.set_index('Model')\
        .sort_values(['F1'])\
        .plot.barh().get_figure().savefig(
            os.path.join(experiment_report_path, 'vergleich.jpg'),
            pad_inches=0.0, bbox_inches='tight')

    logger.info('Results written to "{}"'.format(experiment_report_path))


def train_classfication_models(input_file_path: str, config_file_path: str):
    """Train and evaluate of classifier models w.r.t. a
    config file. The script *scripts/train_supervised_classification_models.py*
    executes this function.

    :param input_file_path: path to the feature CSV file (e.g. *data/processed/default.csv*)
    :type input_file_path: str
    :param config_file_path: path to the config file (z.B. *config/training/supervised/classification/cr1.yml*)
    :type config_file_path: str
    """

    logger = logging.getLogger(__name__)

    with open(config_file_path) as f:
        logger.info('config loaded from "{}"'.format(config_file_path))
        config = yaml.load(f, Loader=yaml.FullLoader)

        for experiment_name in config:
            if config[experiment_name] is not None:
                conduct_classification_experiment(
                    experiment_name, input_file_path, config[experiment_name])


