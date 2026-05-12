# Actigraphy Processing Pipeline

This repository provides a scriptable pipeline for converting device-specific actigraphy files into standardised activity summaries for research analysis.

The codebase is designed to support multiple devices through interchangeable readers. The current development focus is the `geneactive` reader.

## Workflow

```text
Device file
  -> Step 1A: Reader
  -> Standard sample-level data
  -> Step 1B: Preprocessing
  -> Epoch-level summary CSV + metadata
```

## Step 1A: Reader

The reader converts a device-specific file (e.g. `GENEActiv .bin`) into a standard sample-level table.

### Input

```text
GENEActiv .bin file
```

### Output modes

#### `motion`

```text
Time, Ax, Ay, Az
```

Use this when only motion/activity analysis is needed.

#### `full`

```text
Time, Ax, Ay, Az, Lux, Button, Temperature
```

Use this when a fuller GENEActiv-style sample-level export is required.

### Metadata

The reader also writes a sidecar metadata file:

```text
sample_raw.csv
sample_raw.metadata.json
```

The metadata contains available source information, such as device, recording, sampling frequency, and reader settings. It is stored as a separate file with the same base name to keep the CSV easy to read and to support anonymisation workflows.

## Step 1B: Preprocessing

The preprocessing stage takes standard sample-level data and produces epoch-level summaries.

### Input

Motion-only input:

```text
Time, Ax, Ay, Az
```

Full input:

```text
Time, Ax, Ay, Az, Lux, Button, Temperature
```

### Parameters

```text
epoch length
filter on/off
low cutoff frequency
high cutoff frequency
summary mode
```

The filter implementation is fixed internally. Users only control whether filtering is applied and the cutoff frequencies.

Sampling frequency should normally be read from the metadata produced by Step 1A.

Default preprocessing settings are `epoch=60s`, `filter=yes`, Butterworth bandpass filtering with order 4, `low=0.5`, `high=20`, `mode=full-summary`, GENEActiv-style gravity-subtracted SVM, and epoch-end time labels.

### Output Modes

#### `svm`

Minimal activity output:

```text
Time, SVM_sum
```

#### `motion-summary`

Motion-only epoch summary:

```text
Time, Ax_mean, Ay_mean, Az_mean, SVM_sum, Ax_sd, Ay_sd, Az_sd
```

#### `full-summary`

GENEActiv-style epoch summary where full input data are available:

```text
Time,
Ax_mean,
Ay_mean,
Az_mean,
Lux_mean,
Button_sum,
Temperature_mean,
SVM_sum,
Ax_sd,
Ay_sd,
Az_sd,
Lux_peak
```

### Metadata

The preprocessing stage also writes a sidecar metadata file:

```text
sample_60s.csv
sample_60s.metadata.json
```

The metadata extends the raw metadata from Step 1A with preprocessing parameters, such as epoch length, filter settings, cutoff frequencies, and summary mode. It is stored as a separate file with the same base name to keep the CSV easy to read and to support anonymisation workflows.


## Command Line Interface

Output file and metadata file names do not need to be specified manually. If `--output-dir` is not specified, outputs are written to the same directory as the input file.

### Step 1A only

For the reader stage, the output CSV is written to the output directory using the input file’s base name with `_raw.csv` suffix. Each metadata file is written next to the output CSV using the same base name and the `_raw.metadata.json` suffix.

```bash
python -m cli read \
  --input data/raw/sample.bin \
  --output-dir data/intermediate \  # optional
  --output data/intermediate/sample_raw.csv \  # optional
  --metadata data/intermediate/sample_raw.metadata.json \  # optional
  --mode full
```

### Step 1B only

For the preprocessing stage, the output CSV is written to the output directory using the input file’s base name plus the epoch length with `.csv` suffix. Each metadata file is written next to the output CSV using the same base name and the `.metadata.json` suffix.

```bash
python -m cli preprocess \
  --input data/intermediate/sample_raw.csv \
  --metadata data/intermediate/sample_raw.metadata.json \
  --output-dir data/processed \  # optional
  --output data/processed/sample_60s.csv \  # optional
  --metadata-output data/processed/sample_60s.metadata.json \  # optional
  --epoch 60s \
  --filter yes \
  --low 0.5 \
  --high 20 \
  --mode full-summary
```

### Full pipeline

The process and batch commands do not save intermediate files. Only `--summary-mode` needs to be specified, and the required reader mode is inferred automatically from the selected summary mode.

```bash
python -m cli process \
  --input data/raw/sample.bin \
  --output-dir data/processed \  # optional
  --output data/processed/sample_60s.csv \  # optional
  --metadata-output data/processed/sample_60s.metadata.json \  # optional
  --epoch 60s \
  --filter yes \
  --low 0.5 \
  --high 20 \
  --summary-mode full-summary
```

### Batch processing

```bash
python -m cli batch \
  --reader geneactive \
  --input-dir data/raw \
  --output-dir data/processed \  # optional
  --epoch 60s \
  --filter yes \
  --low 0.5 \
  --high 20 \
  --summary-mode full-summary
```

## CSV Comparison Utility

The CSV comparison script compares values by row and column position, not by header name. Ranges use Python-style indexing. For example, `1:4` selects columns 1, 2, and 3; `0:1000` selects the first 1000 rows; `:` selects all rows or columns.


Optional arguments: `--reference-rows`, `--candidate-rows`, `--reference-cols`, `--candidate-cols`, and `--output`. 

- If no row or column ranges are supplied, it skips the candidate CSV header row, aligns the reference CSV to the first candidate timestamp in column 0, and compares all candidate data rows.
- In automatic mode, each compared row must have the same column count, and timestamps must keep the same adjacent and cumulative sampling interval in both files. This helps catch accidental raw-vs-epoch comparisons.
- Omit `--output` to print the comparison summary only, or provide it to also save the comparison as JSON.

```bash
python -m scripts.compare_csv \
  --reference data/reference/export.csv \
  --candidate data/intermediate/sample_raw.csv
```

```bash
python -m scripts.compare_csv \
  --reference data/reference/export.csv \
  --candidate data/intermediate/sample_raw.csv \
  --reference-rows 1:1001 \
  --candidate-rows 1:1001 \
  --reference-cols 1:4 \
  --candidate-cols 1:4 \
  --output data/reports/sample_raw_compare.json  # optional
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
