"""
Alle Funktionen/Module die das Umwandeln der Daten in Features zur Modellierung betreffen.
"""
import logging
import os
import re
from typing import List

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, Binarizer

from sal.data import Attrs
from sal.data.cleaning import MissingValuesIndication
from sal.data.encode import BinaryNominalEncoder
from sal.data.load import typed_view
from sal.features.selection import DefaultAttributeSelector, drop_constant_attributes

ZWEI_JAHRE = 24 # Monate

class DefaultColumnTransformer(ColumnTransformer, Attrs):
    """Transformierer, der die Spalten des SAL-Datensatzes in Features umwandelt.
    """

    def __init__(self):
        self._attrs_nominal = \
            self.quali_nominal + \
            self.quali_bin + \
            self.quali_nominal_mutation_indicators + \
            self.quant_discrete_mutation_indicators

        self._attrs_eln = ['ELNRisk', 'CGELN', 'CGSTUD']
        self._attrs_discrete = self.quant_discrete + self.dates
        self._attrs_continuous = self.quant_continuous

        super().__init__(transformers=[
            ('OSTM', Pipeline(steps=[
                ('dichotomization', Binarizer(threshold=ZWEI_JAHRE))]),
             ['OSTM']),

            ('ECOG', Pipeline(steps=[
                ('missing_indicator', MissingValuesIndication(suffix='unknown')),
                ('imputer', SimpleImputer(strategy='median'))]),
             ['ECOG']),

            ('nominal', Pipeline(steps=[
                ('binary_encoder', BinaryNominalEncoder()),
                ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
                ('one_hot_encoder', OneHotEncoder(sparse=False))
            ]), self._attrs_nominal),

            ('ELN', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
                ('ordinal_encoder', OrdinalEncoder(
                    categories=[['unknown', 'adv', 'int', 'fav']] * len(self._attrs_eln)))
            ]), self._attrs_eln),

            ('discrete', Pipeline(steps=[
                ('missing_indicator', MissingValuesIndication(suffix='unknown')),
                ('imputation', SimpleImputer(strategy='median'))]),
             self._attrs_discrete),

            ('continuous', Pipeline(steps=[
                ('imputation', SimpleImputer(strategy='median'))]),
             self._attrs_continuous)
        ])

    def _feature_names_from(self, path: str, attrs: List[str]) -> List[str]:
        pathList = path.split('/')

        return self \
            .named_transformers_[pathList[0]] \
            .named_steps[pathList[1]] \
            .get_feature_names(attrs).tolist()

    def get_feature_names(self, params: dict):
        """Feature-Namen

        :return: Liste von Feature-Namen
        :rtype: list[str]
        """
        try:
            ostm_threshold = params['preprocessor__OSTM__dichotomization__threshold']
        except KeyError:
            ostm_threshold = ZWEI_JAHRE

        return np.array([
            f'OSTM_above_{ostm_threshold}',
            *self._feature_names_from('ECOG/missing_indicator', attrs=['ECOG']),
            *self._feature_names_from('nominal/one_hot_encoder', attrs=self._attrs_nominal),
            *self._attrs_eln,
            *self._feature_names_from('discrete/missing_indicator', attrs=self._attrs_discrete),
            *self._attrs_continuous], dtype=object)


def build_processed_view(input_file_path: str, output_directory_path: str, yaml_config_path: str):
    """Features unter Berückstichtigung einer Konfigurationsdatei aus den Rohdaten erzeugen.
    Das Script *scripts/build_features.py* führt diese Funktion aus.

    :param input_file_path: Pfad zur SAL-CSV (im Normalfall *data/external/sal.csv*)
    :type input_file_path: str
    :param output_directory_path: Pfad zum Ausgabeverzeichnis (es entsteht eine CSV pro Konfiguration)
    :type output_directory_path: str
    :param yaml_config_path: Pfad zur Konfigurationsdatei (im Normalfall *config/build_features.yml*)
    :type yaml_config_path: str
    """
    logger = logging.getLogger(__name__)

    logger.info('reading dataset from "{}"'.format(input_file_path))
    df = typed_view(input_file_path)

    with open(yaml_config_path) as f:
        logger.info('config loaded from "{}"'.format(yaml_config_path))
        logger.info('writing views to "{}"'.format(output_directory_path))
        config = yaml.load(f, Loader=yaml.FullLoader)

        for name in config:
            if config[name] is None:
                continue

            logger.info('building "{}" ...'.format(name))
            params = pd.json_normalize(config[name], sep='__').to_dict(orient='records')[0]
            
            # FIXME Pandas-Bug in json_normalize? Adds sep as prefix if len(sep) > 1
            params = { re.sub('^__', '', key): val for key, val in params.items() }

            pipeline_params = params.copy()

            if 'filter_query' in pipeline_params: del pipeline_params['filter_query']
            if 'target_cols' in pipeline_params: del pipeline_params['target_cols']
            if 'drop' in pipeline_params: del pipeline_params['drop']
            if 'add' in pipeline_params: del pipeline_params['add']
            if 'subject_id_prefix_filter' in pipeline_params: del pipeline_params['subject_id_prefix_filter']

            pipeline = Pipeline(steps=[
                ('selector', DefaultAttributeSelector(drop_constant_attributes=False, drop_low_information_gain=False, drop_text_attributes=True)),
                ('preprocessor', DefaultColumnTransformer()),
            ])

            pipeline.set_params(**pipeline_params)
            df_transformed = pipeline.fit_transform(df)
            df_transformed_cols = pipeline.named_steps['preprocessor'].get_feature_names(params)

            # print('#' * 100)
            # cols = df_transformed_cols
            # cols = cols.tolist()
            # cols.sort()
            # [print(col) for col in cols]
            # print('#' * 100)

            if 'target_cols' in params:
                cols = params['target_cols']

            # print(df_transformed_cols)
            processed = pd.DataFrame(df_transformed, columns=df_transformed_cols, index=df.index)

            if 'subject_id_prefix_filter' in params:
                processed = processed.loc[processed.index.str.startswith(params['subject_id_prefix_filter'], na=False)]

            if 'preprocessor__OSTM__dichotomization__threshold' in params:
                processed.drop(columns=['OSTM'], inplace=True)
            else:
                processed.drop(columns=[f'OSTM_above_{ZWEI_JAHRE}'], inplace=True)

            if 'filter_query' in params:
                processed = processed.query(params['filter_query']).copy()

            drop_constant_attributes(processed)  # e.g. AGE_missing
            processed.to_csv(os.path.join(output_directory_path, name + '.csv'))
