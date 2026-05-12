from __future__ import annotations

import argparse
from pathlib import Path

from readers import available_readers, get_reader


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
        required=True,
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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "read":
        try:
            output_csv_path, output_metadata_path = run_read(args)
        except ValueError as exc:
            parser.error(str(exc))

        if not args.verbose:
            print(f"CSV saved to: {output_csv_path}")
            print(f"Metadata saved to: {output_metadata_path}")
        return

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
