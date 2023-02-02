# -*- coding: utf-8 -*-
import click
import logging

from sal.models.supervised import predict_classification


@click.command()
@click.argument('input_file_path', type=click.Path(exists=True))
@click.argument('training_config_file_path', type=click.Path(exists=True))
def main(input_file_path, training_config_file_path):
    logger = logging.getLogger(__name__)
    logger.info('predicting classification ...')
    predict_classification(input_file_path, training_config_file_path)
    logger.info('done!')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
