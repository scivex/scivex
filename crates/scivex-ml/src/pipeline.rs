use scivex_core::{Float, Tensor};

use crate::error::{MlError, Result};
use crate::traits::{Predictor, Transformer};

// ── Pipeline ──────────────────────────────────────────────────────────────

/// A machine learning pipeline that chains [`Transformer`] steps followed by
/// an optional [`Predictor`].
///
/// During `fit`, each transformer is fit-transformed sequentially; the final
/// transformed data is passed to the predictor's `fit`.
///
/// During `predict`, each transformer applies `transform` (not re-fitted),
/// then the predictor predicts on the result.
pub struct Pipeline<T: Float> {
    steps: Vec<(String, Box<dyn Transformer<T>>)>,
    predictor: Option<(String, Box<dyn Predictor<T>>)>,
}

impl<T: Float> Default for Pipeline<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> Pipeline<T> {
    /// Create an empty pipeline.
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            predictor: None,
        }
    }

    /// Append a transformer step.
    pub fn add_step(mut self, name: &str, transformer: Box<dyn Transformer<T>>) -> Self {
        self.steps.push((name.to_string(), transformer));
        self
    }

    /// Set the final predictor.
    pub fn set_predictor(mut self, name: &str, predictor: Box<dyn Predictor<T>>) -> Self {
        self.predictor = Some((name.to_string(), predictor));
        self
    }

    /// Number of transformer steps (excluding the final predictor).
    pub fn n_steps(&self) -> usize {
        self.steps.len()
    }

    /// Fit all transformers and pass through data.
    fn fit_transform_steps(&mut self, x: &Tensor<T>) -> Result<Tensor<T>> {
        let mut current = x.clone();
        for (_, t) in &mut self.steps {
            current = t.fit_transform(&current)?;
        }
        Ok(current)
    }

    /// Transform data through all fitted steps.
    fn transform_steps(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        let mut current = x.clone();
        for (_, t) in &self.steps {
            current = t.transform(&current)?;
        }
        Ok(current)
    }
}

impl<T: Float> Predictor<T> for Pipeline<T> {
    fn fit(&mut self, x: &Tensor<T>, y: &Tensor<T>) -> Result<()> {
        let transformed = self.fit_transform_steps(x)?;
        match self.predictor {
            Some((_, ref mut pred)) => pred.fit(&transformed, y),
            None => Err(MlError::InvalidParameter {
                name: "predictor",
                reason: "pipeline has no predictor set",
            }),
        }
    }

    fn predict(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        let transformed = self.transform_steps(x)?;
        match &self.predictor {
            Some((_, pred)) => pred.predict(&transformed),
            None => Err(MlError::NotFitted),
        }
    }
}

// ── FeatureUnion ──────────────────────────────────────────────────────────

/// Concatenates results of multiple transformers column-wise.
///
/// Each transformer is fit on the **full** input; their outputs are
/// horizontally stacked (`hstack`).
pub struct FeatureUnion<T: Float> {
    transformers: Vec<(String, Box<dyn Transformer<T>>)>,
}

impl<T: Float> Default for FeatureUnion<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> FeatureUnion<T> {
    /// Create an empty feature union.
    pub fn new() -> Self {
        Self {
            transformers: Vec::new(),
        }
    }

    /// Add a named transformer.
    pub fn add(mut self, name: &str, transformer: Box<dyn Transformer<T>>) -> Self {
        self.transformers.push((name.to_string(), transformer));
        self
    }

    /// Number of transformers.
    pub fn len(&self) -> usize {
        self.transformers.len()
    }

    /// Whether empty.
    pub fn is_empty(&self) -> bool {
        self.transformers.is_empty()
    }
}

impl<T: Float> Transformer<T> for FeatureUnion<T> {
    fn fit(&mut self, x: &Tensor<T>) -> Result<()> {
        for (_, t) in &mut self.transformers {
            t.fit(x)?;
        }
        Ok(())
    }

    fn transform(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        if self.transformers.is_empty() {
            return Err(MlError::InvalidParameter {
                name: "transformers",
                reason: "feature union has no transformers",
            });
        }
        let results: Vec<Tensor<T>> = self
            .transformers
            .iter()
            .map(|(_, t)| t.transform(x))
            .collect::<Result<Vec<_>>>()?;
        hstack(&results)
    }
}

// ── ColumnTransformer ─────────────────────────────────────────────────────

/// A named transformer with its target column indices.
type ColumnStep<T> = (String, Box<dyn Transformer<T>>, Vec<usize>);

/// Applies different transformers to different column subsets, then
/// concatenates the results horizontally.
pub struct ColumnTransformer<T: Float> {
    transformers: Vec<ColumnStep<T>>,
}

impl<T: Float> Default for ColumnTransformer<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> ColumnTransformer<T> {
    /// Create an empty column transformer.
    pub fn new() -> Self {
        Self {
            transformers: Vec::new(),
        }
    }

    /// Add a named transformer that operates on the given column indices.
    pub fn add(
        mut self,
        name: &str,
        transformer: Box<dyn Transformer<T>>,
        columns: Vec<usize>,
    ) -> Self {
        self.transformers
            .push((name.to_string(), transformer, columns));
        self
    }
}

impl<T: Float> Transformer<T> for ColumnTransformer<T> {
    fn fit(&mut self, x: &Tensor<T>) -> Result<()> {
        for (_, t, cols) in &mut self.transformers {
            let sub = select_columns(x, cols)?;
            t.fit(&sub)?;
        }
        Ok(())
    }

    fn transform(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        if self.transformers.is_empty() {
            return Err(MlError::InvalidParameter {
                name: "transformers",
                reason: "column transformer has no transformers",
            });
        }
        let results: Vec<Tensor<T>> = self
            .transformers
            .iter()
            .map(|(_, t, cols)| {
                let sub = select_columns(x, cols)?;
                t.transform(&sub)
            })
            .collect::<Result<Vec<_>>>()?;
        hstack(&results)
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────

/// Select specific columns from a 2-D tensor.
fn select_columns<T: Float>(x: &Tensor<T>, cols: &[usize]) -> Result<Tensor<T>> {
    let s = x.shape();
    if s.len() != 2 {
        return Err(MlError::DimensionMismatch {
            expected: 2,
            got: s.len(),
        });
    }
    let (n, p) = (s[0], s[1]);
    for &c in cols {
        if c >= p {
            return Err(MlError::DimensionMismatch {
                expected: p,
                got: c + 1,
            });
        }
    }
    let data = x.as_slice();
    let out_p = cols.len();
    let mut out = vec![T::zero(); n * out_p];
    for i in 0..n {
        for (j, &c) in cols.iter().enumerate() {
            out[i * out_p + j] = data[i * p + c];
        }
    }
    Tensor::from_vec(out, vec![n, out_p]).map_err(MlError::from)
}

/// Horizontally stack multiple 2-D tensors (concatenate along columns).
fn hstack<T: Float>(tensors: &[Tensor<T>]) -> Result<Tensor<T>> {
    if tensors.is_empty() {
        return Err(MlError::EmptyInput);
    }
    let n = tensors[0].shape()[0];
    let total_cols: usize = tensors.iter().map(|t| t.shape()[1]).sum();

    let mut out = vec![T::zero(); n * total_cols];
    let mut col_offset = 0;
    for t in tensors {
        if t.shape()[0] != n {
            return Err(MlError::DimensionMismatch {
                expected: n,
                got: t.shape()[0],
            });
        }
        let p = t.shape()[1];
        let data = t.as_slice();
        for i in 0..n {
            for j in 0..p {
                out[i * total_cols + col_offset + j] = data[i * p + j];
            }
        }
        col_offset += p;
    }
    Tensor::from_vec(out, vec![n, total_cols]).map_err(MlError::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear::LinearRegression;
    use crate::preprocessing::StandardScaler;

    #[test]
    fn test_pipeline_scaler_and_regression() {
        // y = 2x + 1
        let x = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0], vec![5, 1]).unwrap();
        let y = Tensor::from_vec(vec![3.0, 5.0, 7.0, 9.0, 11.0], vec![5]).unwrap();

        let mut pipe = Pipeline::new()
            .add_step("scaler", Box::new(StandardScaler::<f64>::new()))
            .set_predictor("lr", Box::new(LinearRegression::<f64>::new()));

        pipe.fit(&x, &y).unwrap();
        let preds = pipe.predict(&x).unwrap();
        for (&p, &t) in preds.as_slice().iter().zip(y.as_slice()) {
            assert!((p - t).abs() < 1.0, "prediction {p} far from {t}");
        }
    }

    #[test]
    fn test_pipeline_no_predictor() {
        let x = Tensor::from_vec(vec![1.0_f64, 2.0], vec![2, 1]).unwrap();
        let y = Tensor::from_vec(vec![1.0, 2.0], vec![2]).unwrap();
        let mut pipe =
            Pipeline::<f64>::new().add_step("scaler", Box::new(StandardScaler::<f64>::new()));
        assert!(pipe.fit(&x, &y).is_err());
    }

    #[test]
    fn test_feature_union() {
        let x = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap();

        let mut fu = FeatureUnion::new()
            .add("std", Box::new(StandardScaler::<f64>::new()))
            .add("std2", Box::new(StandardScaler::<f64>::new()));

        let out = fu.fit_transform(&x).unwrap();
        assert_eq!(out.shape(), &[3, 4]); // 2 cols * 2 transformers
    }

    #[test]
    fn test_column_transformer() {
        // 3 samples, 3 features
        let x = Tensor::from_vec(
            vec![1.0_f64, 10.0, 100.0, 2.0, 20.0, 200.0, 3.0, 30.0, 300.0],
            vec![3, 3],
        )
        .unwrap();

        let mut ct = ColumnTransformer::new()
            .add("first", Box::new(StandardScaler::<f64>::new()), vec![0])
            .add(
                "last_two",
                Box::new(StandardScaler::<f64>::new()),
                vec![1, 2],
            );

        let out = ct.fit_transform(&x).unwrap();
        assert_eq!(out.shape(), &[3, 3]); // 1 + 2 columns
    }

    #[test]
    fn test_select_columns() {
        let x = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let sub = select_columns(&x, &[0, 2]).unwrap();
        assert_eq!(sub.shape(), &[2, 2]);
        assert_eq!(sub.as_slice(), &[1.0, 3.0, 4.0, 6.0]);
    }

    #[test]
    fn test_hstack() {
        let a = Tensor::from_vec(vec![1.0_f64, 2.0], vec![2, 1]).unwrap();
        let b = Tensor::from_vec(vec![3.0_f64, 4.0, 5.0, 6.0], vec![2, 2]).unwrap();
        let out = hstack(&[a, b]).unwrap();
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out.as_slice(), &[1.0, 3.0, 4.0, 2.0, 5.0, 6.0]);
    }

    #[test]
    fn test_pipeline_n_steps() {
        let pipe = Pipeline::<f64>::new()
            .add_step("a", Box::new(StandardScaler::<f64>::new()))
            .add_step("b", Box::new(StandardScaler::<f64>::new()));
        assert_eq!(pipe.n_steps(), 2);
    }
}
