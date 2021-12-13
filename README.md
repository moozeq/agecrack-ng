# agecrack-ng

agecrack-ng is a tool for searching and extracting age-related features from data.

## Run

```bash
# clone
git clone git@github.com:moozeq/agecrack-ng.git
cd agecrack-ng

# setup venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# run
./agecrack-ng.py -h
```

## Analysis

### Vertebrates

```bash
# best vertebrates predictor
./agecrack-ng.py -vv --mode predictor --model rf --filters repair --mmseq-vectors-mode bool --models-rand 1 --mmseq-params '{"min_seq_id": 0.8, "c": 0.2, "cov_mode": 2}' --models-params '{"n_estimators": 300, "max_depth": 18}'

# plots on unprocessed data
./agecrack-ng.py -vv --mode predictor --model rf --filters repair --mmseq-vectors-mode bool --models-rand 1 --mmseq-params '{"min_seq_id": 0.8, "c": 0.2, "cov_mode": 2}' --models-params '{"n_estimators": 300, "max_depth": 18}' --models-plots-unprocess

# gather multiple models and analyse them all
./agecrack-ng.py -vv --mode ontology --model rf --filters repair --mmseq-vectors-mode bool --models-rand 1 --mmseq-params '{"min_seq_id": 0.8, "c": 0.2, "cov_mode": 2}'
```

### Mammalia

```bash
# best mammalia predictor
./agecrack-ng.py -vv --mode predictor --model rf --filter-class Mammalia --exclude 'Homo sapiens' --extract-threshold 1000 --models-plots-annotate --models-rand 17 --mmseq-params '{"min_seq_id": 0.8, "c": 0.8, "cov_mode": 0}' --models-params '{"n_estimators": 30, "max_depth": 7}'

# plots on unprocessed data
./agecrack-ng.py -vv --mode predictor --model rf --filter-class Mammalia --exclude 'Homo sapiens' --extract-threshold 1000 --models-plots-annotate --models-plots-annotate --models-plots-annotate-threshold 15 --models-rand 17 --mmseq-params '{"min_seq_id": 0.8, "c": 0.8, "cov_mode": 0}' --models-params '{"n_estimators": 30, "max_depth": 7}'

# gather multiple models and analyse them all
./agecrack-ng.py -vv --mode ontology --model rf --filter-class Mammalia --exclude 'Homo sapiens' --extract-threshold 1000 --models-plots-annotate --models-rand 17 --mmseq-params '{"min_seq_id": 0.8, "c": 0.8, "cov_mode": 0}'
```
