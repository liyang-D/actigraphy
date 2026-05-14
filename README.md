# Actigraphy Processing Pipeline

This repository converts device actigraphy files into CSV outputs for research analysis.

It currently supports GENEActiv `.bin` files and is structured so other device readers can be added later.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m pip install -r requirements.txt
```

## Workflow

```text
Device file
  -> Step 1A: Reader
  -> sample-level CSV + metadata
  -> Step 1B: Preprocessing
  -> epoch-level CSV + metadata
```

The CSV contains only the processed data values. Metadata such as device / experiment / participant information and preprocessing parameters is stored separately in a paired metadata file. This keeps the CSV lightweight and makes downstream data loading and anonymisation easier.

Metadata is always written automatically next to the CSV:

```text
sample.csv
sample.metadata.json
```

## Quick Start

For most use cases, run the full pipeline directly:

```bash
python -m cli process \
  --input data/raw/sample.bin \
  --verbose
```

This uses the default settings:

```text
reader = geneactive
epoch = 60s
filter = no
summary mode = full-summary
workers = 1
```

The processed CSV is saved automatically in the same directory as the input file.

Example output:

```text
sample_60s.csv
sample_60s.metadata.json
```

For a folder of `.bin` files:

```bash
python -m cli batch \
  --input-dir data/raw \
  --verbose
```

Add `--verbose` when processing large files so progress is printed while pages are decoded and epochs are generated.

To generate a PDF report from the processed CSV:

```bash
python -m scripts.generate_sleep_report \
  --input data/processed/sample_60s.csv
```

## Step 1A: Reader

Step 1A converts a device file into a standard sample-level CSV.

Input:

```text
GENEActiv .bin file
```

Output modes:

```text
motion:
Time, Ax, Ay, Az

full:
Time, Ax, Ay, Az, Lux, Button, Temperature
```

Simple command:

```bash
python -m cli read \
  --input data/raw/sample.bin \
  --verbose
```

Customised command:

```bash
python -m cli read \
  --input data/raw/sample.bin \
  --output data/intermediate/sample_raw.csv \
  --mode full \
  --workers 2 \
  --verbose
```

Useful optional parameters: `--output-dir`, `--output`, `--mode`, `--workers`, and `--verbose`.

Increasing `--workers` allows parallel processing and may speed up decoding depending on available hardware performance.

If `--output` is used, it must end with `.csv`. The metadata file name is generated automatically from that CSV path.

## Step 1B: Preprocessing

Step 1B converts sample-level CSV data into epoch-level summaries. It reads the paired metadata file automatically.

Input columns can be motion-only:

```text
Time, Ax, Ay, Az
```

or full:

```text
Time, Ax, Ay, Az, Lux, Button, Temperature
```

Output modes:

```text
svm:
Time, SVM_sum

motion-summary:
Time, Ax_mean, Ay_mean, Az_mean, SVM_sum, Ax_sd, Ay_sd, Az_sd

full-summary:
Time, Ax_mean, Ay_mean, Az_mean, Lux_mean, Button_sum,
Temperature_mean, SVM_sum, Ax_sd, Ay_sd, Az_sd, Lux_peak
```

Simple command:

```bash
python -m cli preprocess \
  --input data/intermediate/sample_raw.csv \
  --verbose
```

Customised command:

```bash
python -m cli preprocess \
  --input data/intermediate/sample_raw.csv \
  --output data/processed/sample_60s.csv \
  --epoch 60s \
  --filter yes \
  --low 0.5 \
  --high 20 \
  --mode full-summary \
  --verbose
```

Useful optional parameters: `--output-dir`, `--output`, `--epoch`, `--filter`, `--low`, `--high`, `--mode`, `--sample-rate`, and `--verbose`.

The default Step 1B settings are:

```text
epoch = 60s
filter = no
filter type = Butterworth bandpass
filter order = 4
low cutoff = 0.5 Hz
high cutoff = 20 Hz
summary mode = full-summary
SVM method = GENEActiv-style abs(vector_magnitude - 1)
time label = epoch start
epoch anchor = first sample time
```

Optional filtering uses a Butterworth bandpass filter on `Ax`, `Ay`, and `Az` only. It can be enabled with `--filter yes`, and `--low` / `--high` can be adjusted to smooth motion signals. `Lux`, `Button`, and `Temperature` are not filtered. To reproduce GENEActiv official exports, `--filter no` is usually closer.

## Full Pipeline

The full pipeline runs Step 1A and Step 1B without saving the intermediate raw CSV.

The reader mode is inferred automatically from `--summary-mode`:

```text
full-summary:
reads all columns

svm / motion-summary:
reads motion columns only
```

Simple command:

```bash
python -m cli process \
  --input data/raw/sample.bin \
  --verbose
```

Customised command:

```bash
python -m cli process \
  --input data/raw/sample.bin \
  --output data/processed/sample_60s.csv \
  --epoch 60s \
  --filter yes \
  --low 0.5 \
  --high 20 \
  --summary-mode full-summary \
  --workers 2 \
  --verbose
```

Useful optional parameters: `--reader`, `--output-dir`, `--output`, `--epoch`, `--filter`, `--low`, `--high`, `--summary-mode`, `--workers`, and `--verbose`.

Default parameter values are the same as the Step 1B defaults above.

## Batch Processing

Batch mode runs the full pipeline for every supported file in a directory.

Simple command:

```bash
python -m cli batch \
  --input-dir data/raw \
  --verbose
```

Customised command:

```bash
python -m cli batch \
  --reader geneactive \
  --input-dir data/raw \
  --output-dir data/processed \
  --epoch 60s \
  --filter yes \
  --low 0.5 \
  --high 20 \
  --summary-mode full-summary \
  --workers 2 \
  --verbose
```

Useful optional parameters: `--reader`, `--output-dir`, `--epoch`, `--filter`, `--low`, `--high`, `--summary-mode`, `--workers`, and `--verbose`.

## Sleep Report PDF

The report script is separate from the processing pipeline. It reads a standard CSV and renders a simple `Actigraphy Sleep Report` PDF.

Quick start:

```bash
python -m scripts.generate_sleep_report \
  --input data/processed/sample_60s.csv
```

The report plots `SVM_sum` as activity. If light columns are available, it overlays light on a `log10(lux + 1)` scale. The default activity y-axis maximum is the 99th percentile of `SVM_sum` across the full file.

Customised command:

```bash
python -m scripts.generate_sleep_report \
  --input data/processed/sample_60s.csv \
  --output data/reports/sample_sleep_report.pdf \
  --report-date "28 Aug 2025" \
  --day-start-hour 15 \
  --days-per-page 4 \
  --activity-scale 100 \
  --lux-log-scale-max 5 \
  --verbose
```

Useful optional parameters: `--output`, `--title`, `--report-date`, `--day-start-hour`, `--days-per-page`, `--activity-scale`, `--lux-log-scale-max`, and `--verbose`.

Use `--activity-scale` to set a fixed activity axis maximum.

Use `--lux-log-scale-max` to adjust the light axis maximum.

## CSV Comparison Utility

Use this utility to compare numeric values in two CSV files for validation against official software exports.

If no row or column ranges are supplied, it skips the candidate header row, aligns the reference CSV to the first candidate timestamp, and compares the overlapping aligned rows.

It also checks column counts and timestamp intervals, which helps catch accidental raw-vs-epoch comparisons.

Automatic comparison:

```bash
python -m scripts.compare_csv \
  --reference data/reference/export.csv \
  --candidate data/processed/sample_60s.csv
```

Manual range comparison:

```bash
python -m scripts.compare_csv \
  --reference data/reference/export.csv \
  --candidate data/processed/sample_60s.csv \
  --reference-rows 1:1001 \
  --candidate-rows 1:1001 \
  --reference-cols 1:4 \
  --candidate-cols 1:4 \
  --output data/reports/sample_compare.json
```

Optional range parameters:

```text
--reference-rows
--candidate-rows
--reference-cols
--candidate-cols
```

Omit `--output` to print only. Provide it to also save the comparison JSON.

To compare two saved comparison JSON files, for example when checking whether filtered or unfiltered output has smaller errors:

```bash
python -m scripts.compare_comparison_json \
  --left data/reports/sample_no_filter_compare.json \
  --right data/reports/sample_filter_compare.json \
  --left-label no_filter \
  --right-label filter
```

## Reference

This project references and builds upon the following resources:

- https://activinsights.com/wp-content/uploads/2025/01/GENEActiv-1.2-IFU-rev-7.pdf
- https://sleeptoolkit.activinsights.net/
- https://github.com/danwjoyce/accel-scripts

## Feedback

If you encounter any issues or have suggestions for improvements, feel free to email [leon.dou@kmms.ac.uk](mailto:leon.dou@kmms.ac.uk).

## License

This project is licensed under the MIT License.

See the `LICENSE` file for details.
