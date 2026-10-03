//! WASM bindings for ferrum tabular models (classification / regression).
//!
//! `TabularModel` loads a FINF model trained by `train_cli` and predicts from a
//! row of raw feature values (the stored normalizer is applied internally).

use ferrum_core::{argmax_rows, from_bytes, TaskType};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct TabularModel {
    model: ferrum_core::Sequential,
    norm: ferrum_core::Normalizer,
    meta_json: String,
    task: TaskType,
}

#[wasm_bindgen]
impl TabularModel {
    #[wasm_bindgen(constructor)]
    pub fn new(bytes: &[u8]) -> Result<TabularModel, JsValue> {
        let (model, norm, meta) =
            from_bytes(bytes).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let task = meta.task;
        let meta_json = meta.to_json();
        Ok(Self {
            model,
            norm,
            meta_json,
            task,
        })
    }

    pub fn metadata(&self) -> String {
        self.meta_json.clone()
    }

    pub fn norm_encoded(&self) -> String {
        self.norm.encode()
    }

    pub fn predict(&self, values: &[f32]) -> Result<String, JsValue> {
        let raw = ferrum_core::Tensor::row(values.to_vec())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let input = self
            .norm
            .transform(&raw)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let out = self
            .model
            .forward(&input)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let json = match self.task {
            TaskType::Classification => {
                let class_idx =
                    argmax_rows(&out).map_err(|e| JsValue::from_str(&e.to_string()))?[0];
                let confidence = out.data[class_idx];
                let probs = out
                    .data
                    .iter()
                    .map(|p| format!("{p:.6}"))
                    .collect::<Vec<_>>()
                    .join(",");
                format!(
                    r#"{{"type":"classification","class_index":{class_idx},"confidence":{confidence:.6},"probabilities":[{probs}]}}"#
                )
            }
            TaskType::Regression => {
                let pred_norm = out.data[0];
                let pred_raw = self.norm.denormalise_target(pred_norm);
                format!(
                    r#"{{"type":"regression","value":{pred_raw:.4},"value_norm":{pred_norm:.6}}}"#
                )
            }
            TaskType::TransformerSLM => {
                r#"{"type":"error","message":"Use TransformerSLMModel for SLM inference"}"#
                    .to_string()
            }
        };
        Ok(json)
    }
}

#[cfg(test)]
mod tests {
    use ferrum_core::{from_bytes, to_bytes, ModelMetadata, Normalizer, Rng, TaskType, Tensor};
    use ferrum_ml::Net;

    fn clf_bytes() -> Vec<u8> {
        let mut rng = Rng::new(1);
        let net = Net::mlp(4, 8, 3, &mut rng);
        let model = net.to_inference().unwrap();
        let norm = Normalizer {
            means: vec![5.8, 3.0, 3.7, 1.2],
            stds: vec![0.8, 0.4, 1.7, 0.8],
        };
        let meta = ModelMetadata {
            dataset_name: "test".into(),
            task: TaskType::Classification,
            feature_names: vec!["a".into(), "b".into(), "c".into(), "d".into()],
            feature_ranges: vec![[0.0, 10.0]; 4],
            class_names: vec!["X".into(), "Y".into(), "Z".into()],
            target_name: "".into(),
            target_range: [0.0, 2.0],
            input_dim: 4,
            output_dim: 3,
            tokenizer_state: String::new(),
        };
        to_bytes(&model, &norm, &meta).unwrap()
    }

    #[test]
    fn classification_loads_and_predicts() {
        let bytes = clf_bytes();
        let (model, norm, _meta) = from_bytes(&bytes).unwrap();
        let raw = Tensor::row(vec![5.1f32, 3.5, 1.4, 0.2]).unwrap();
        let input = norm.transform(&raw).unwrap();
        let out = model.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
        let sum: f32 = out.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn corrupt_bytes_error_gracefully() {
        assert!(from_bytes(b"CORRUPT").is_err());
        assert!(from_bytes(&[]).is_err());
    }
}
