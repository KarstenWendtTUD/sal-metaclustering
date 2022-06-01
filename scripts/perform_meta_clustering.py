import click
import logging

from sal.models.meta_clustering import start


@click.command()
@click.argument('config_file_path', type=click.Path(exists=True))
def main(config_file_path):
    logger = logging.getLogger(__name__)
    logger.info('performing meta clustering ...')
    start(config_file_path)
    logger.info('done!')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()