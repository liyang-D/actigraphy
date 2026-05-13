# Actigraphy Processing Pipeline

This repository converts device actigraphy files into CSV outputs for research analysis.
It currently supports GENEActiv `.bin` files and is structured so other device readers can be added later.

## Installation

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

Metadata is always written automatically next to the CSV:

```text
sample.csv
sample.metadata.json
```

If `--output` is used, it must end with `.csv`. The metadata file name is generated automatically from that CSV path.

## Quick Start

For most use cases, run the full pipeline directly:

```bash
python -m cli process \
  --input data/raw/sample.bin \
  --output-dir data/processed \
  --verbose
```

This uses the default settings: GENEActiv reader, 60-second epochs, filtering enabled, full summary output, and one reader worker.

To generate a PDF report from the processed CSV:

```bash
python -m scripts.generate_sleep_report \
  --input data/processed/sample_60s.csv
```

For a folder of `.bin` files:

```bash
python -m cli batch \
  --input-dir data/raw \
  --output-dir data/processed \
  --verbose
```

Add `--verbose` when processing large files so progress is printed while pages are decoded and epochs are generated.

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

## Step 1B: Preprocessing

Step 1B converts sample-level CSV data into epoch-level summaries. It reads the paired metadata file automatically, for example `sample_raw.csv` uses `sample_raw.metadata.json`.

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

## Full Pipeline

The full pipeline runs Step 1A and Step 1B without saving the intermediate raw CSV. The reader mode is inferred from `--summary-mode`: `full-summary` reads all columns, while `svm` and `motion-summary` read motion columns only.

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

## Batch Processing

Batch mode runs the full pipeline for every supported file in a directory.

Simple command:

```bash
python -m cli batch \
  --input-dir data/raw \
  --output-dir data/processed \
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

## Preprocessing Defaults

The default Step 1B settings are:

```text
epoch = 60s
filter = yes
filter type = Butterworth bandpass
filter order = 4
low cutoff = 0.5 Hz
high cutoff = 20 Hz
summary mode = full-summary
SVM method = GENEActiv-style abs(vector_magnitude - 1)
time label = epoch start
```

Filtering is applied only to `Ax`, `Ay`, and `Az`. `Lux`, `Button`, and `Temperature` are not filtered.

## CSV Comparison Utility

Use this utility to compare numeric values in two CSV files. If no row or column ranges are supplied, it skips the candidate header row, aligns the reference CSV to the first candidate timestamp, and compares the overlapping aligned rows. It also checks column counts and timestamp intervals, which helps catch accidental raw-vs-epoch comparisons.

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

Optional range parameters: `--reference-rows`, `--candidate-rows`, `--reference-cols`, and `--candidate-cols`. Omit `--output` to print only; provide it to also save the comparison JSON.

To compare two saved comparison JSON files, for example when checking whether filtered or unfiltered output has smaller errors:

```bash
python -m scripts.compare_comparison_json \
  --left data/reports/sample_no_filter_compare.json \
  --right data/reports/sample_filter_compare.json \
  --left-label no_filter \
  --right-label filter
```

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

Useful optional parameters: `--output`, `--title`, `--report-date`, `--day-start-hour`, `--days-per-page`, `--activity-scale`, `--lux-log-scale-max`, and `--verbose`. Use `--activity-scale` to set a fixed activity axis maximum, and `--lux-log-scale-max` to adjust the light axis maximum.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
