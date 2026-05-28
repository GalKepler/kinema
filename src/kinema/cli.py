"""CLI entry point for kinema."""

import logging
import sys

import click

from kinema import __version__

logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        stream=sys.stderr,
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )


@click.group()
@click.version_option(version=__version__, prog_name="kinema")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Enable debug logging.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """Bouldering performance analytics from phone video."""
    _configure_logging(verbose)
