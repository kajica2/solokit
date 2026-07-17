"""`solokit download-corpora` Click command."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from solokit.data import CORPORA, data_status, download_all

console = Console()
err_console = Console(stderr=True)


@click.command("download-corpora")
@click.option(
    "-c",
    "--corpus",
    "only",
    multiple=True,
    type=click.Choice(list(CORPORA.keys())),
    help="Only download the specified corpus. Repeat for multiple. Default: all.",
)
@click.option(
    "-d",
    "--data-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Target directory. Default: <solokit>/data/.",
)
@click.option(
    "--force/--no-force",
    default=False,
    help="Re-download even if the file is already present.",
)
@click.option(
    "--status/--no-status",
    default=False,
    help="Just print which corpora are present, don't download anything.",
)
def download_corpora(
    only: tuple[str, ...],
    data_dir: Path | None,
    force: bool,
    status: bool,
) -> None:
    """Download the bundled local corpora (WJAZD, Omnibook).

    The data is not included in the git repo or the PyPI sdist because
    the files total ~51MB and have license attribution requirements.

    Examples::

        solokit download-corpora
        solokit download-corpora --corpus wjazzd
        solokit download-corpora --corpus omnibook --force
        solokit download-corpora --status
    """
    from solokit import data as data_module

    if data_dir is None:
        # Default: <solokit>/data/
        # src/solokit/data.py is at <solokit>/src/solokit/data.py
        # so the data dir is ../../data relative to that
        data_dir = Path(__file__).parent.parent.parent / "data"

    if status:
        _print_status(data_module, data_dir)
        return

    targets = list(only) if only else None
    try:
        results = download_all(data_dir, force=force, only=targets)
    except Exception as exc:
        err_console.print(f"[red]Download failed: {exc}[/red]")
        raise click.Abort from exc

    console.print()
    console.print("[green]Done.[/green] Local corpora:")
    for name, path in results.items():
        size = path.stat().st_size if path.is_file() else sum(
            f.stat().st_size for f in path.glob("*.xml")
        )
        size_mb = size / 1_000_000
        console.print(f"  [cyan]{name}[/cyan]: {path}  ({size_mb:.1f} MB)")


def _print_status(data_module, data_dir: Path) -> None:
    status = data_status(data_dir)
    table = Table(show_header=True, header_style="bold", title="solokit data status")
    table.add_column("corpus")
    table.add_column("present", justify="center")
    table.add_column("size")
    table.add_column("details")

    for name, info in status.items():
        present = info.get("present", False)
        present_str = "[green]✓[/green]" if present else "[red]✗[/red]"
        if "file_count" in info:
            size = f"{info['file_count']} files"
        elif "size_bytes" in info:
            size = f"{info['size_bytes'] / 1_000_000:.1f} MB"
        else:
            size = "—"
        details = str(info.get("expected_path", ""))
        table.add_row(name, present_str, size, details)

    console.print(table)
