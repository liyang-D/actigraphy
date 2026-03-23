import argparse
import sys

from io import load_raw_csv, save_epoch_svm_csv
from models import EpochSVMData, PreprocessConfig, RawTriAxialData
from preprocess.epoch import epoch_svm
from preprocess.filter import apply_bandpass_filter
from preprocess.svm import compute_svm


def run_preprocessing(
    raw_data: RawTriAxialData,
    config: PreprocessConfig,
    verbose: bool = False,
) -> EpochSVMData:
    raw_data.validate()
    config.validate()

    working_data = raw_data.copy()

    if config.apply_filter:
        try:
            if verbose:
                print(" ** Apply Butterworth Filter")
                print(f" --- Low = {config.low_cutoff} Hz")
                print(f" --- High = {config.high_cutoff} Hz")

            working_data = apply_bandpass_filter(
                raw_data=working_data,
                low_cutoff=config.low_cutoff,
                high_cutoff=config.high_cutoff,
                sample_rate=config.sample_rate,
                order=4,
            )
        except Exception:
            print(
                "Error Bandpass Filtering: check CSV file for compatibility, "
                "correct columns and ordering: Time, Ax, Ay, Az"
            )
            raise

    try:
        if verbose:
            print(" ** Computing Signal Vector Magnitude")

        data_with_svm = compute_svm(working_data)
    except Exception:
        print(
            "Error computing SVM, check CSV file for compatibility and "
            "correct columns and ordering: Time, Ax, Ay, Az"
        )
        sys.exit(1)

    try:
        if verbose:
            print(" ** Downsampling")

        return epoch_svm(data_with_svm, config.binsize)
    except Exception:
        print("Error dowmsampling data: check that --binsize value makes sense")
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Takes raw CSV accelerometer data with columns Time, Ax, Ay, Az, "
            "optionally applies a Butterworth bandpass filter, computes signal "
            "vector magnitude (SVM), and downsamples to bins by mean."
        )
    )

    parser.add_argument(
        "source",
        type=str,
        help="Full path to source CSV file",
    )
    parser.add_argument(
        "out",
        type=str,
        help="Full path to output CSV file",
    )
    parser.add_argument(
        "-samplerate",
        type=float,
        default=100.0,
        help="Sample rate in Hz of source data (default = 100Hz)",
    )
    parser.add_argument(
        "-binsize",
        type=str,
        required=True,
        help=(
            "Downsampling window specification accepted by pandas resample. "
            "For example, 3T for 3 minutes, 30S for 30 seconds."
        ),
    )
    parser.add_argument(
        "-filter",
        type=str,
        required=True,
        choices=["yes", "no"],
        help="Apply 4th-order Butterworth bandpass filter",
    )
    parser.add_argument(
        "-low",
        type=float,
        default=0.5,
        help="Bandpass low cutoff in Hz (default = 0.5)",
    )
    parser.add_argument(
        "-high",
        type=float,
        default=20.0,
        help="Bandpass high cutoff in Hz (default = 20)",
    )
    parser.add_argument(
        "-verbose",
        type=str,
        default="no",
        choices=["yes", "no"],
        help="Print progress messages during preprocessing (default = no)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verbose = args.verbose == "yes"

    try:
        if verbose:
            print(" ** Loading source CSV file")

        raw_data = load_raw_csv(
            path=args.source,
            sample_rate=args.samplerate,
        )
    except FileNotFoundError:
        print("Source file: " + args.source + " does not exist")
        sys.exit(1)
    except Exception:
        print(
            "Error loading source CSV file: check CSV file for compatibility, "
            "correct columns and ordering: Time, Ax, Ay, Az"
        )
        sys.exit(1)

    config = PreprocessConfig(
        binsize=args.binsize,
        apply_filter=(args.filter == "yes"),
        sample_rate=args.samplerate,
        low_cutoff=args.low,
        high_cutoff=args.high,
    )

    processed_data = run_preprocessing(
        raw_data=raw_data,
        config=config,
        verbose=verbose,
    )

    try:
        if verbose:
            print(" ** Saving results as CSV")

        save_epoch_svm_csv(processed_data, args.out)
    except FileNotFoundError:
        print("Output file: " + args.out + " cannot be written -- check path exists")
        sys.exit(1)
    except Exception:
        print("Error saving output CSV file")
        sys.exit(1)


if __name__ == "__main__":
    main()