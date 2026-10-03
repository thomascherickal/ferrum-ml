# Ferrum-ML

**Tabular machine learning on the zero-dependency Ferrum engine.** Train small
MLP classifiers and regressors from any numeric CSV, entirely on the CPU, and
ship each one as a single self-contained FINF model file. The model runs
natively or in a browser.

Ferrum-ML was split out of the [Ferrum](../Ferrum/ReadMe.md) SLM repository on
2026-10-03. These demos work well and are useful on their own, but they have
nothing to do with training language models. They reuse Ferrum's engine
(tensors, layers, optimizers, quantization, and the FINF format) through a path
dependency on `../Ferrum/ferrum_core`. Keep the two folders side by side.

---

## Layout

| Path | What it is |
|---|---|
| `ferrum_ml/` | Library: `CsvDataset` (CSV parsing with header and task auto-detection, plus split helpers) and `Net` (trainable MLP with SGD + momentum and optional int8 QAT) |
| `ferrum_ml/src/bin/train_cli.rs` | The `train_cli` binary: CSV in, trained FINF model out |
| `tabular_wasm/` | `wasm-bindgen` binding `TabularModel` for in-browser prediction |
| `datasets/tabular/` | Ten cleaned public datasets (see below) |
| `models/<name>/model.bin` | One pretrained model per dataset |
| `scripts/` | `download_datasets.sh` (idempotent fetch + clean), `train_all.sh` (retrain all ten), `build_wasm.sh` |

## Build and test

```bash
cargo build --release
cargo test  --release        # 78 tests, including inference checks on all ten shipped models
```

## Train a model

```text
train_cli <csv_path> <model_output.bin> [dataset_name] [hidden_size] [epochs] [--verbose]
```

```bash
./target/release/train_cli datasets/tabular/iris.data models/iris/model.bin "Iris" 32 500
# Full-dataset accuracy : 98.7%
./target/release/train_cli datasets/tabular/housing.csv models/housing/model.bin "California Housing" 64 400
# Full-dataset RMSE : 52834.88 (raw scale)
```

The pipeline works as follows:

1. Detect the header row.
2. Choose the task: **classification** if the last column is non-numeric,
   otherwise **regression**.
3. Fit a per-column normalizer on the training split only.
4. Train with SGD and momentum.
5. Write a FINF model that embeds feature names, ranges, class labels, and the
   normalizer. A UI can build itself from the file alone.

`bash scripts/train_all.sh` retrains all ten models.

## Datasets

| File | Task | Shipped model |
|---|---|---|
| `iris.data` | 3-class classification | `models/iris` |
| `wine.csv` | classification (quality) | `models/wine` |
| `diabetes.csv` | binary classification (Pima) | `models/diabetes` |
| `titanic.csv` | binary classification | `models/titanic` |
| `housing.csv` | regression (California) | `models/housing` |
| `heart.csv` | binary classification | `models/heart` |
| `cancer.csv` | binary classification (breast cancer) | `models/cancer` |
| `penguins.csv` | 3-class classification (Palmer) | `models/penguins` |
| `mpg.csv` | regression (Auto MPG) | `models/mpg` |
| `seeds.csv` | 3-class classification (wheat) | `models/seeds` |

## Use from Rust

```rust
use ferrum_core::{to_bytes, Rng, Sgd};
use ferrum_ml::{train_epoch, train_val_split, CsvDataset, Net};

let ds = CsvDataset::from_str(&std::fs::read_to_string("datasets/tabular/iris.data")?)?;
let mut rng = Rng::new(42);
let (train, val) = train_val_split(&ds, 0.2, &mut rng);
// …fit a Normalizer, build Net::mlp(inputs, hidden, outputs, &mut rng),
// loop train_epoch(&mut net, &x, &y, batch, &Sgd::with_momentum(0.05, 0.9), &mut rng),
// then to_bytes(&net.to_inference()?, &norm, &meta) — see src/bin/train_cli.rs.
```

## In the browser

```bash
bash scripts/build_wasm.sh   # → tabular_wasm/pkg/
```

```js
import init, { TabularModel } from "./pkg/tabular_wasm.js";
await init();
const m = new TabularModel(new Uint8Array(await (await fetch("models/iris/model.bin")).arrayBuffer()));
const meta = JSON.parse(m.metadata());           // feature_names, feature_ranges, class_names, task
console.log(m.predict(new Float32Array([5.1, 3.5, 1.4, 0.2])));   // raw values; normalizer applied inside
```

## License

MIT, the same as Ferrum.
