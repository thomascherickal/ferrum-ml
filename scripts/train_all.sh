#!/usr/bin/env bash
# scripts/train_all.sh
# Train all 10 dataset models and place them in models/*/model.bin.
# Builds the train_cli binary automatically.
# Usage: bash scripts/train_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Building train_cli ==="
cargo build --release -p ferrum_ml

echo ""
echo "=== Training all 10 datasets ==="

run() {
  local csv="$1" dst="$2" name="$3" hidden="$4" epochs="$5"
  mkdir -p "$(dirname "$dst")"
  echo "--- $name ---"
  cargo run -p ferrum_ml --bin train_cli --release -- "$csv" "$dst" "$name" "$hidden" "$epochs" \
    2>&1 | grep -E "accuracy|RMSE|Saved|error" | head -4
}

run datasets/tabular/iris.data    models/iris/model.bin      "Iris"               32 500
run datasets/tabular/wine.csv     models/wine/model.bin      "Wine Quality"       64 600
run datasets/tabular/diabetes.csv models/diabetes/model.bin  "Pima Diabetes"      48 600
run datasets/tabular/titanic.csv  models/titanic/model.bin   "Titanic"            32 500
run datasets/tabular/housing.csv  models/housing/model.bin   "California Housing" 64 400
run datasets/tabular/heart.csv    models/heart/model.bin     "Heart Disease"      32 600
run datasets/tabular/cancer.csv   models/cancer/model.bin    "Breast Cancer"      64 600
run datasets/tabular/penguins.csv models/penguins/model.bin  "Palmer Penguins"    32 500
run datasets/tabular/mpg.csv      models/mpg/model.bin       "Auto MPG"           48 500
run datasets/tabular/seeds.csv    models/seeds/model.bin     "Wheat Seeds"        24 500

echo ""
echo "=== All models trained. Models are in models/<name>/model.bin. ==="
