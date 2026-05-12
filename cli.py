from __future__ import annotations

import argparse
from pathlib import Path

from models import PreprocessConfig
from preprocess import preprocess_file, preprocess_raw_data_to_files
from readers import available_readers, get_reader
from utils import parse_yes_no


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Actigraphy processing command line interface."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    read_parser = subparsers.add_parser(
        "read",
        help="Run Step 1A reader and export sample-level CSV data.",
    )
    read_parser.add_argument(
        "--reader",
        choices=available_readers(),
        default="geneactive",
        help="Device reader to use.",
    )
    read_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input device file.",
    )
    read_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to the input file directory.",
    )
    read_parser.add_argument(
        "--mode",
        choices=["motion", "full"],
        default="full",
        help="Reader output mode.",
    )
    read_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    read_parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    read_parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum number of pages to process. Defaults to all pages.",
    )
    read_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress messages.",
    )

    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Run Step 1B preprocessing and export epoch-level CSV data.",
    )
    preprocess_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input Step 1A CSV file.",
    )
    preprocess_parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help=(
            "Input Step 1A metadata JSON. Defaults to the input CSV path with "
            ".metadata.json suffix."
        ),
    )
    preprocess_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to the input file directory.",
    )
    preprocess_parser.add_argument(
        "--epoch",
        default="60s",
        help="Epoch length, for example 60s.",
    )
    preprocess_parser.add_argument(
        "--filter",
        choices=["yes", "no"],
        default="yes",
        help="Apply Butterworth bandpass filter to Ax, Ay and Az.",
    )
    preprocess_parser.add_argument(
        "--low",
        type=float,
        default=0.5,
        help="Bandpass low cutoff in Hz.",
    )
    preprocess_parser.add_argument(
        "--high",
        type=float,
        default=20.0,
        help="Bandpass high cutoff in Hz.",
    )
    preprocess_parser.add_argument(
        "--mode",
        choices=["svm", "motion-summary", "full-summary"],
        default="full-summary",
        help="Epoch summary output mode.",
    )
    preprocess_parser.add_argument(
        "--sample-rate",
        type=float,
        default=None,
        help="Fallback sample rate in Hz if metadata is unavailable.",
    )
    preprocess_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    preprocess_parser.add_argument(
        "--metadata-output",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    preprocess_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress messages.",
    )

    process_parser = subparsers.add_parser(
        "process",
        help="Run Step 1A and Step 1B without saving intermediate files.",
    )
    process_parser.add_argument(
        "--reader",
        choices=available_readers(),
        default="geneactive",
        help="Device reader to use.",
    )
    process_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input device file.",
    )
    process_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to the input file directory.",
    )
    process_parser.add_argument(
        "--epoch",
        default="60s",
        help="Epoch length, for example 60s.",
    )
    process_parser.add_argument(
        "--filter",
        choices=["yes", "no"],
        default="yes",
        help="Apply Butterworth bandpass filter to Ax, Ay and Az.",
    )
    process_parser.add_argument(
        "--low",
        type=float,
        default=0.5,
        help="Bandpass low cutoff in Hz.",
    )
    process_parser.add_argument(
        "--high",
        type=float,
        default=20.0,
        help="Bandpass high cutoff in Hz.",
    )
    process_parser.add_argument(
        "--summary-mode",
        choices=["svm", "motion-summary", "full-summary"],
        default="full-summary",
        help="Epoch summary output mode.",
    )
    process_parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    process_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress messages.",
    )

    batch_parser = subparsers.add_parser(
        "batch",
        help="Run the full pipeline for every supported device file in a directory.",
    )
    batch_parser.add_argument(
        "--reader",
        choices=available_readers(),
        default="geneactive",
        help="Device reader to use.",
    )
    batch_parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing input device files.",
    )
    batch_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to each input file directory.",
    )
    batch_parser.add_argument(
        "--epoch",
        default="60s",
        help="Epoch length, for example 60s.",
    )
    batch_parser.add_argument(
        "--filter",
        choices=["yes", "no"],
        default="yes",
        help="Apply Butterworth bandpass filter to Ax, Ay and Az.",
    )
    batch_parser.add_argument(
        "--low",
        type=float,
        default=0.5,
        help="Bandpass low cutoff in Hz.",
    )
    batch_parser.add_argument(
        "--high",
        type=float,
        default=20.0,
        help="Bandpass high cutoff in Hz.",
    )
    batch_parser.add_argument(
        "--summary-mode",
        choices=["svm", "motion-summary", "full-summary"],
        default="full-summary",
        help="Epoch summary output mode.",
    )
    batch_parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    batch_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress messages.",
    )

    return parser


def run_read(args: argparse.Namespace) -> tuple[Path, Path]:
    max_pages = None if args.max_pages in (None, 0) else args.max_pages

    if max_pages is not None and max_pages < 0:
        raise ValueError("--max-pages must be greater than or equal to 0.")

    reader = get_reader(args.reader)
    return reader.read(
        input_path=args.input,
        output_csv_path=args.output,
        output_metadata_path=args.metadata,
        output_dir=args.output_dir,
        mode=args.mode,
        max_pages=max_pages,
        verbose=args.verbose,
    )


def run_preprocess(args: argparse.Namespace) -> tuple[Path, Path]:
    config = PreprocessConfig(
        epoch=args.epoch,
        filter_enabled=parse_yes_no(args.filter),
        low_cutoff_hz=args.low,
        high_cutoff_hz=args.high,
        summary_mode=args.mode,
    )

    return preprocess_file(
        input_csv_path=args.input,
        metadata_path=args.metadata,
        output_csv_path=args.output,
        output_metadata_path=args.metadata_output,
        output_dir=args.output_dir,
        config=config,
        fallback_sample_rate_hz=args.sample_rate,
        verbose=args.verbose,
    )


def infer_reader_mode(summary_mode: str) -> str:
    if summary_mode == "full-summary":
        return "full"

    if summary_mode in {"svm", "motion-summary"}:
        return "motion"

    raise ValueError(
        "summary_mode must be one of 'svm', 'motion-summary', or 'full-summary'."
    )


def build_preprocess_config_from_summary_args(
    args: argparse.Namespace,
) -> PreprocessConfig:
    return PreprocessConfig(
        epoch=args.epoch,
        filter_enabled=parse_yes_no(args.filter),
        low_cutoff_hz=args.low,
        high_cutoff_hz=args.high,
        summary_mode=args.summary_mode,
    )


def normalized_max_pages(max_pages: int | None) -> int | None:
    if max_pages in (None, 0):
        return None

    if max_pages < 0:
        raise ValueError("--max-pages must be greater than or equal to 0.")

    return max_pages


def run_process(args: argparse.Namespace) -> tuple[Path, Path]:
    config = build_preprocess_config_from_summary_args(args)
    reader_mode = infer_reader_mode(config.summary_mode)
    reader = get_reader(args.reader)

    raw_data = reader.load_samples(
        input_path=args.input,
        mode=reader_mode,
        max_pages=normalized_max_pages(args.max_pages),
        verbose=args.verbose,
    )

    return preprocess_raw_data_to_files(
        raw_data=raw_data,
        input_path=args.input,
        output_dir=args.output_dir,
        config=config,
        verbose=args.verbose,
    )


def iter_batch_inputs(input_dir: Path, extensions: tuple[str, ...]) -> list[Path]:
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    if not input_dir.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")

    normalized_extensions = tuple(extension.lower() for extension in extensions)
    paths = [
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in normalized_extensions
    ]

    return sorted(paths)


def run_batch(args: argparse.Namespace) -> list[tuple[Path, Path]]:
    reader = get_reader(args.reader)
    input_paths = iter_batch_inputs(args.input_dir, reader.supported_extensions)

    if not input_paths:
        extensions = ", ".join(reader.supported_extensions)
        raise ValueError(
            f"No input files with supported extensions found in {args.input_dir}: "
            f"{extensions}"
        )

    outputs: list[tuple[Path, Path]] = []
    for input_path in input_paths:
        if args.verbose:
            print(f"Processing: {input_path}")

        command_args = argparse.Namespace(
            reader=args.reader,
            input=input_path,
            output_dir=args.output_dir,
            epoch=args.epoch,
            filter=args.filter,
            low=args.low,
            high=args.high,
            summary_mode=args.summary_mode,
            max_pages=args.max_pages,
            verbose=args.verbose,
        )
        outputs.append(run_process(command_args))

    return outputs


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "read":
        try:
            output_csv_path, output_metadata_path = run_read(args)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))

        if not args.verbose:
            print(f"CSV saved to: {output_csv_path}")
            print(f"Metadata saved to: {output_metadata_path}")
        return

    if args.command == "preprocess":
        try:
            output_csv_path, output_metadata_path = run_preprocess(args)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))

        if not args.verbose:
            print(f"CSV saved to: {output_csv_path}")
            print(f"Metadata saved to: {output_metadata_path}")
        return

    if args.command == "process":
        try:
            output_csv_path, output_metadata_path = run_process(args)
        except (OSError, NotImplementedError, ValueError) as exc:
            parser.error(str(exc))

        if not args.verbose:
            print(f"CSV saved to: {output_csv_path}")
            print(f"Metadata saved to: {output_metadata_path}")
        return

    if args.command == "batch":
        try:
            outputs = run_batch(args)
        except (OSError, NotImplementedError, ValueError) as exc:
            parser.error(str(exc))

        if not args.verbose:
            for output_csv_path, output_metadata_path in outputs:
                print(f"CSV saved to: {output_csv_path}")
                print(f"Metadata saved to: {output_metadata_path}")
        return

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
