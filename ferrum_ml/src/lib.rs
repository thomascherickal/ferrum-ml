//! # ferrum_ml — tabular machine learning on the ferrum engine
//!
//! Train small MLP classifiers and regressors from any numeric CSV, entirely on
//! CPU with zero external dependencies, and save them as FINF model files that
//! run natively or in the browser (see the `tabular_wasm` crate).
//!
//! - [`CsvDataset`] parses a CSV, detecting the header and the task
//!   (classification when the last column is non-numeric, regression otherwise).
//! - [`Net`] is the trainable MLP (SGD with momentum, optional int8 QAT); its
//!   [`Net::to_inference`] output is a `ferrum_core::Sequential`.
#![forbid(unsafe_code)]

pub mod csv;
pub mod train;

pub use csv::{fit_normalizer_with_target, train_val_split, CsvDataset, CsvRow};
pub use train::{accuracy, train_epoch, EmbedT, Net};
