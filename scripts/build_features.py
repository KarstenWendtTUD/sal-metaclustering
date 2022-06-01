# -*- coding: utf-8 -*-
from typing import List

import os
import click
import logging

from sal.features import build_processed_view


@click.command()
@click.argument('input_file_path', type=click.Path(exists=True))
@click.argument('output_directory_path', type=click.Path())
@click.option('--config-file', default='config/build_features.yml', type=click.Path())
def main(input_file_path, output_directory_path, config_file):
    logger = logging.getLogger(__name__)
    logger.info('building features ...')

    build_processed_view(input_file_path,
                         output_directory_path,
                         config_file)

    logger.info('done!')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
