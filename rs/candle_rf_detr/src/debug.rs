use candle_core::Result;
use candle_core::Tensor;

/// Tensor statistics for debugging and validation
#[derive(Debug)]
pub struct TensorStats {
    pub shape: Vec<usize>,
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub sum: f32,
}

impl TensorStats {
    /// Compute statistics for a tensor
    pub fn from_tensor(tensor: &Tensor) -> Result<Self> {
        let shape = tensor.dims().to_vec();
        let flat = tensor.flatten_all()?.to_dtype(candle_core::DType::F32)?;
        let data: Vec<f32> = flat.to_vec1()?;

        let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = data.iter().sum();
        let mean = sum / data.len() as f32;

        Ok(Self {
            shape,
            min,
            max,
            mean,
            sum,
        })
    }
}
