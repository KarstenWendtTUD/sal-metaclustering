import click
import logging

from sal.models.unsupervised import start_UL_experiments

# TODO @karsten "UL" in Dateiname/überall ersetzen

@click.command()
@click.argument('input_file_path', type=click.Path(exists=True))
@click.argument('config_file_path', type=click.Path(exists=True))
def main(input_file_path, config_file_path):
    logger = logging.getLogger(__name__)
    logger.info('performing UL experiments ...')
    start_UL_experiments(input_file_path, config_file_path)
    logger.info('done!')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()