import click
from config import settings

from src.run import split_in_three

@click.group()
def cli():
    pass

@click.command()
def do_split_in_three():
    split_in_three(settings)

@click.command()
#@click.option('--which', default="internal")
def do_training():
    raise NotImplementedError()

@click.command()
#@click.option('--which', default="internal")
def do_evaluation():
    raise NotImplementedError()

cli.add_command(do_split_in_three)
cli.add_command(do_training)
cli.add_command(do_evaluation)

if __name__ == '__main__':
    cli()