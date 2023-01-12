use ndarray::{Array, Array1, Axis, Dimension, Zip};
use serde::{Deserialize, Serialize};

/// Defines a batch normalization layer
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BatchNormalization {
    gamma: Array1<f32>,
    beta: Array1<f32>,
    moving_mean: Array1<f32>,
    moving_variance: Array1<f32>,
    epsilon: f32,
}

impl BatchNormalization {
    /// Construct a new [`BatchNormalization`] from predefined parameters.
    ///
    /// Will panic if `beta`, `moving_mean` and `moving_variance`
    /// don't have identical shapes.
    #[must_use]
    pub fn new(
        gamma: Array1<f32>,
        beta: Array1<f32>,
        moving_mean: Array1<f32>,
        moving_variance: Array1<f32>,
        epsilon: f32,
    ) -> BatchNormalization {
        assert!(
            gamma.shape() == beta.shape()
                && beta.shape() == moving_mean.shape()
                && moving_mean.shape() == moving_variance.shape(),
            "gamma, beta, moving_mean and moving_variance must all have the same shape!"
        );
        BatchNormalization {
            gamma,
            beta,
            moving_mean,
            moving_variance,
            epsilon,
        }
    }

    /// Apply batch normalization to be used at inference time
    /// Returns a normalized array
    #[must_use]
    pub fn apply<D: Dimension>(&self, data: &Array<f32, D>) -> Array<f32, D> {
        let mut output = data.clone();
        self.apply_mut(&mut output);

        output
    }

    /// Apply batch normalization inplace, to be used at inference time
    pub fn apply_mut<D: Dimension>(&self, data: &mut Array<f32, D>) {
        let axis = data.ndim() - 1;

        // the input data's last axis shape must match the shape of one of self.[beta, moving_average, moving_variance]
        assert!(
            data.shape()[axis] == self.beta.shape()[0],
            "Input data's last axis's shape must match beta/moving_mean/moving_variance"
        );

        for mut lane in data.lanes_mut(Axis(axis)) {
            Zip::from(&mut lane)
                .and(&self.gamma)
                .and(&self.beta)
                .and(&self.moving_mean)
                .and(&self.moving_variance)
                .apply(|elem, g, b, m, v| *elem = (*elem - m) / (v + self.epsilon).sqrt() * g + b);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr2, arr3, Array2, Array3};

    #[test]
    fn test_batchnormalization_1d() {
        let data = Array::from_shape_vec(2, vec![1.0, 2.0]).unwrap();

        let gamma = Array::from_shape_vec(2, vec![2.25, 2.25]).unwrap();
        let beta = Array::from_shape_vec(2, vec![3.0, 4.0]).unwrap();
        let moving_mean = Array::from_shape_vec(2, vec![1.0, 1.5]).unwrap();
        let moving_variance = Array::from_shape_vec(2, vec![0.5, 0.7]).unwrap();
        let epsilon = 0.0001;

        let batch_normalization_layer =
            BatchNormalization::new(gamma, beta, moving_mean, moving_variance, epsilon);

        let result = batch_normalization_layer.apply(&data);
        let expected = Array::from_shape_vec(2, vec![3.0, 5.344536]).unwrap();

        assert_eq!(expected, result);
    }

    #[test]
    fn test_batchnormalization_2d() {
        let data: Array2<f32> = arr2(&[
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0],
        ]);

        let gamma = Array::from_shape_vec(2, vec![2.25, 2.25]).unwrap();
        let beta = Array::from_shape_vec(2, vec![3.0, 4.0]).unwrap();
        let moving_mean = Array::from_shape_vec(2, vec![1.0, 1.5]).unwrap();
        let moving_variance = Array::from_shape_vec(2, vec![0.5, 0.7]).unwrap();
        let epsilon = 0.0001;

        let batch_normalization_layer =
            BatchNormalization::new(gamma, beta, moving_mean, moving_variance, epsilon);

        let result = batch_normalization_layer.apply(&data);
        let expected: Array2<f32> = arr2(&[
            [3.0, 5.344536],
            [9.363325, 10.722681],
            [15.726649, 16.100824],
            [22.089973, 21.47897],
            [28.453299, 26.857113],
            [34.816624, 32.23526],
        ]);

        assert_eq!(expected, result);
    }

    #[test]
    fn test_batchnormalization_3d() {
        let data: Array3<f32> = arr3(&[
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
        ]);

        let gamma = Array::from_shape_vec(2, vec![2.25, 2.25]).unwrap();
        let beta = Array::from_shape_vec(2, vec![3.0, 4.0]).unwrap();
        let moving_mean = Array::from_shape_vec(2, vec![1.0, 1.5]).unwrap();
        let moving_variance = Array::from_shape_vec(2, vec![0.5, 0.7]).unwrap();
        let epsilon = 0.0001;

        let batch_normalization_layer =
            BatchNormalization::new(gamma, beta, moving_mean, moving_variance, epsilon);

        let result = batch_normalization_layer.apply(&data);
        let expected: Array3<f32> = arr3(&[
            [
                [3.0, 5.344536],
                [9.363325, 10.722681],
                [15.726649, 16.100824],
            ],
            [
                [22.089973, 21.47897],
                [28.453299, 26.857113],
                [34.816624, 32.23526],
            ],
        ]);

        assert_eq!(expected, result);
    }
}
