"""Command-line interface for solokit.

Usage examples::

    # Pattern search
    solokit search -1 -1 4 -5 -2 --transformation interval --min-similarity 0.8

    # Feature extraction
    solokit features path/to/solo.mid --config features/basic.yaml

    # Transcribe an audio file
    solokit transcribe path/to/solo.wav

    # Start the API server
    solokit serve --port 8000

    # Health check against a running server
    solokit health --server http://localhost:8000
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import click
from rich.console import Console
from rich.table import Table

from solokit import __version__
from solokit.corpora import DTLCorpus
from solokit.patterns.transformations import transform
from solokit.cli_download import download_corpora

console = Console()
err_console = Console(stderr=True)


@click.group()
@click.version_option(version=__version__, prog_name="solokit")
def cli() -> None:
    """solokit — pattern search and analysis for jazz solo transcriptions."""


# ----------------------------------------------------------------------------
# search
# ----------------------------------------------------------------------------


@cli.command()
@click.option(
    "-p",
    "--pattern",
    "pattern",
    required=True,
    metavar="PATTERN",
    help='Pattern as a space- or comma-separated string. e.g. "-1 -1 4 -5 -2" or "60,62,64,65,67".',
)
@click.option(
    "-t",
    "--transformation",
    type=click.Choice(["pitch", "interval", "fuzzyinterval", "cdpcx"]),
    default="interval",
    show_default=True,
    help="Pattern transformation. `interval` matches across keys.",
)
@click.option(
    "-c",
    "--corpus",
    "corpus_name",
    type=click.Choice(["dtl", "omnibook", "wjazzd"]),
    default="dtl",
    show_default=True,
    help="Which corpus to search. 'omnibook' (50 solos) and 'wjazzd' (456 solos) "
    "are local and instant. 'dtl' is remote (larger but sometimes down).",
)
@click.option(
    "--min-similarity",
    default=0.8,
    type=click.FloatRange(0.5, 1.0),
    show_default=True,
)
@click.option(
    "--max-length-difference",
    default=0,
    type=click.IntRange(0, 5),
    show_default=True,
)
@click.option(
    "--max-edit-distance",
    default=None,
    type=int,
    help="Hard cap on edit distance. Defaults to derived from min-similarity.",
)
@click.option(
    "--min-frequency",
    default=1,
    type=int,
    show_default=True,
    help="Minimum number of identical instances.",
)
@click.option(
    "--limit",
    default=20,
    type=click.IntRange(1, 500),
    show_default=True,
)
@click.option(
    "--show-audio/--no-audio",
    default=False,
    help="Print the audio snippet URL for each match (DTL only).",
)
def search(
    pattern: str,
    transformation: str,
    corpus_name: str,
    min_similarity: float,
    max_length_difference: int,
    max_edit_distance: int | None,
    min_frequency: int,
    limit: int,
    show_audio: bool,
) -> None:
    """Search a corpus for matches to PATTERN.

    PATTERN is a space- or comma-separated list of integers, e.g.
    "-1 -1 4 -5 -2" or "60,62,64,65,67".

    Example: solokit search "-1 -1 4 -5 -2" --corpus omnibook

    Tip: --corpus omnibook uses the local Charlie Parker corpus
    (no network required, instant results).
    """
    # Parse the pattern string: space- or comma-separated
    raw = [int(x) for x in pattern.replace(",", " ").split()]
    if len(raw) < 2:
        err_console.print(
            f"[red]Pattern must have at least 2 integers (got {len(raw)}).[/red]"
        )
        raise click.Abort

    # The transformation tells us what the user's input represents:
    #   "interval"        → user already passed intervals (-1 -1 4 -5 -2)
    #   "pitch"           → user passed MIDI pitches (60 62 64 65), convert to intervals
    #   "fuzzyinterval"   → user passed fuzzy intervals
    #   "cdpcx"           → user passed chord-diatonic pitch classes
    # So we DON'T re-apply the transformation; we just pass the pattern
    # through with the transformation label.
    if transformation in ("interval", "fuzzyinterval", "cdpcx"):
        # Pattern is already in the target representation
        query_pattern = raw
    else:  # "pitch" — convert MIDI pitches to intervals
        query_pattern = transform(raw, transformation)  # type: ignore[arg-type]
        if not query_pattern:
            err_console.print(
                f"[red]Pattern too short for {transformation!r} (need ≥2 pitches).[/red]"
            )
            raise click.Abort

    err_console.print(
        f"[dim]Querying {corpus_name} for {transformation} pattern {query_pattern}...[/dim]"
    )

    if corpus_name == "dtl":
        # Remote corpus — DTL-specific parameters
        with DTLCorpus() as corpus:
            results = corpus.search(
                query_pattern,
                transformation=transformation,  # type: ignore[arg-type]
                databases=("dtl",),
                min_similarity=min_similarity,
                max_length_difference=max_length_difference,
                max_edit_distance=max_edit_distance,
                min_frequency=min_frequency,
                limit=limit,
            )
    else:  # omnibook or wjazzd
        from solokit.corpora import OmnibookCorpus, WJAZDCorpus

        corpus_cls = WJAZDCorpus if corpus_name == "wjazzd" else OmnibookCorpus
        corpus = corpus_cls()
        results = corpus.search(
            query_pattern,
            transformation=transformation,  # type: ignore[arg-type]
            min_similarity=min_similarity,
            max_length_difference=max_length_difference,
            max_edit_distance=max_edit_distance,
            min_frequency=min_frequency,
            limit=limit,
        )

    if not results:
        console.print("[yellow]No matches found.[/yellow]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("sim", justify="right", style="cyan")
    table.add_column("ed", justify="right")
    table.add_column("performer", style="bold")
    table.add_column("title")
    table.add_column("db", justify="center", style="dim")
    if show_audio:
        table.add_column("audio", style="dim")

    for r in results:
        row = [
            f"{r.match.similarity:.2f}",
            str(r.match.edit_distance),
            r.performer,
            r.title,
            r.database,
        ]
        if show_audio:
            row.append(r.audio_url or "")
        table.add_row(*row)

    console.print(table)
    console.print(f"\n[dim]{len(results)} match(es).[/dim]")


# ----------------------------------------------------------------------------
# features
# ----------------------------------------------------------------------------


@cli.command()
@click.argument(
    "input_path",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "-c",
    "--config",
    "config",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="YAML config file. If omitted, uses the `basic` config from solokit/features/.",
)
@click.option(
    "--tempo",
    default=120.0,
    type=float,
    help="Default tempo in BPM if the file doesn't specify one.",
)
def features(input_path: Path, config: Path | None, tempo: float) -> None:
    """Extract features from a transcription file (MIDI or JSON)."""
    from solokit.core.solo import Solo, SoloMetadata
    from solokit.core.transcription import NoteEvent, Transcription
    from solokit.features import FeatureMachine

    transcription = _load_transcription(input_path, tempo_bpm=tempo)
    solo = Solo(
        metadata=SoloMetadata(
            melid=input_path.stem,
            title=input_path.stem,
            performer="unknown",
        ),
        transcription=transcription,
    )

    if config is None:
        # Use the bundled basic config
        bundled = Path(__file__).parent.parent.parent / "features" / "basic.yaml"
        if bundled.exists():
            config = bundled
        else:
            err_console.print("[red]No config specified and basic.yaml not found.[/red]")
            raise click.Abort

    machine = FeatureMachine.from_yaml(config)
    results = machine.extract(solo)

    import numpy as np
    import yaml as _yaml

    for name, value in results.items():
        if isinstance(value, np.ndarray):
            console.print(f"[bold]{name}[/bold]: ndarray shape={value.shape}")
        elif isinstance(value, dict):
            console.print(f"[bold]{name}[/bold]:")
            console.print(_yaml.dump(value, sort_keys=False).rstrip())
        else:
            console.print(f"[bold]{name}[/bold]: {value}")


# ----------------------------------------------------------------------------
# transcribe
# ----------------------------------------------------------------------------


@cli.command()
@click.argument(
    "audio_path",
    type=click.Path(exists=True, path_type=Path),
)
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Write JSON to file.")
@click.option(
    "--model",
    default="pyin",
    type=click.Choice(["pyin", "basic-pitch"]),
    show_default=True,
    help="Transcription model. 'pyin' is best for monophonic jazz solos. "
    "'basic-pitch' is for polyphonic / multi-instrument audio.",
)
def transcribe(audio_path: Path, output: Path | None, model: str) -> None:
    """Transcribe a .wav file to a transcription (requires [audio] extra)."""
    try:
        from solokit.audio import transcribe_wav
    except ImportError as exc:
        err_console.print(
            f"[red]Audio transcription requires: pip install 'solokit[audio]'[/red]\n{exc}"
        )
        raise click.Abort from exc

    err_console.print(f"[dim]Transcribing {audio_path} (model={model})...[/dim]")
    t = transcribe_wav(audio_path, model=model)

    import json

    payload = {
        "tempo_bpm": t.tempo_bpm,
        "time_signature": t.time_signature,
        "key_signature": t.key_signature,
        "notes": [
            {
                "pitch": n.pitch,
                "onset_beat": n.onset_beat,
                "duration_beats": n.duration_beats,
                "velocity": n.velocity,
            }
            for n in t.notes
        ],
    }
    text = json.dumps(payload, indent=2)
    if output:
        output.write_text(text)
        console.print(f"[green]Wrote {output}[/green]")
    else:
        console.print(text)


# ----------------------------------------------------------------------------
# serve
# ----------------------------------------------------------------------------


@cli.command()
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8000, type=int, show_default=True)
@click.option("--reload/--no-reload", default=False, help="Auto-reload on code changes.")
def serve(host: str, port: int, reload: bool) -> None:
    """Start the FastAPI server."""
    try:
        import uvicorn
    except ImportError as exc:
        err_console.print(
            f"[red]API server requires: pip install 'solokit[api]'[/red]\n{exc}"
        )
        raise click.Abort from exc

    err_console.print(f"[green]Starting solokit API on http://{host}:{port}[/green]")
    uvicorn.run("solokit.api.server:app", host=host, port=port, reload=reload)


# ----------------------------------------------------------------------------
# health
# ----------------------------------------------------------------------------


@cli.command()
@click.option(
    "--server",
    default="http://localhost:8000",
    show_default=True,
    help="Base URL of a running solokit server.",
)
def health(server: str) -> None:
    """Check the health of a running solokit server."""
    from solokit.api.client import SolokitClient

    with SolokitClient(base_url=server, timeout=5.0) as c:
        try:
            h = c.health()
        except Exception as exc:
            err_console.print(f"[red]Server unreachable: {exc}[/red]")
            raise click.Abort from exc
    console.print(f"[green]OK[/green] — version {h['version']}, corpora: {h['corpora']}")


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


def _load_transcription(path: Path, *, tempo_bpm: float = 120.0):
    """Load a transcription from a MIDI or JSON file.

    TODO: add MusicXML support via music21.
    """
    from solokit.core.transcription import NoteEvent, Transcription

    if path.suffix.lower() in (".mid", ".midi"):
        # Lazy import music21 to avoid hard dep
        try:
            import music21 as m21
        except ImportError as exc:
            msg = "MIDI loading requires music21: pip install music21"
            raise click.UsageError(msg) from exc
        score = m21.converter.parse(str(path))
        notes: list[NoteEvent] = []
        for n in score.flat.notes:
            if isinstance(n, m21.note.Rest):
                continue
            notes.append(
                NoteEvent(
                    pitch=n.pitch.midi,
                    onset_beat=float(n.offset),
                    duration_beats=float(n.duration.quarterLength),
                )
            )
        return Transcription.from_note_sequence(notes, tempo_bpm=tempo_bpm)

    if path.suffix.lower() == ".json":
        import json
        data = json.loads(path.read_text())
        notes = [
            NoteEvent(
                pitch=n.get("pitch"),
                onset_beat=float(n["onset_beat"]),
                duration_beats=float(n["duration_beats"]),
                velocity=n.get("velocity"),
            )
            for n in data["notes"]
        ]
        return Transcription.from_note_sequence(
            notes,
            tempo_bpm=data.get("tempo_bpm", tempo_bpm),
            time_signature=tuple(data.get("time_signature", (4, 4))),
            key_signature=data.get("key_signature"),
        )

    msg = f"Unsupported file format: {path.suffix}"
    raise click.UsageError(msg)


if __name__ == "__main__":
    cli()


# Register additional subcommands (defined in separate modules to keep
# this file from growing unboundedly)
cli.add_command(download_corpora)
