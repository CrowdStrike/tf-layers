use crate::activations::Activation;
use ndarray::{Array1, Array2, Array3, Axis};
use serde::{Deserialize, Serialize};

/// Defines a regular densely-connected Neural Network layer.
/// Dense implements the operation:
///     output = activation(dot(input, kernel) + bias)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DenseLayer {
    /// kernel weights matrix
    weights: Array2<f32>,
    /// bias vector
    bias: Array1<f32>,
    /// activation function type
    activation: Activation,
}

impl DenseLayer {
    /// Returns a new [`DenseLayer`] from predefined parameters.
    #[must_use]
    pub fn new(weights: Array2<f32>, bias: Array1<f32>, activation: Activation) -> DenseLayer {
        DenseLayer {
            weights,
            bias,
            activation,
        }
    }

    /// Returns the result of the dense layer operation on the input 2D data
    ///
    /// # Panics
    /// Will panic when `data` axes are not equal lengths.
    #[must_use]
    pub fn apply2d(&self, data: &Array2<f32>) -> Array2<f32> {
        // Since we need to compute data * self.weights + self.bias
        // we must assert that the multiplication step can take place
        // self.weights shape: (features_in, features_out)
        // data shape: (batch_size, features_in)
        assert_eq!(self.weights.len_of(Axis(0)), data.len_of(Axis(1)));

        // result shape: (batch_size, features_out)
        let mut result: Array2<f32> = data.dot(&self.weights);
        result += &self.bias;

        // Apply in-place activation on the result matrix.
        self.activation.activation_mut(&mut result);

        result
    }

    /// Returns the result of the dense layer operation on the input 3D data
    ///
    /// # Panics
    /// `weights` has to be the same shape as data.
    #[must_use]
    pub fn apply3d(&self, data: &Array3<f32>) -> Array3<f32> {
        // Since we need to compute data * self.weights + self.bias
        // we must assert that the multiplication step can take place
        // self.weights shape: (features_in, features_out)
        // data shape: (batch_size, _, features_in)
        assert_eq!(self.weights.len_of(Axis(0)), data.len_of(Axis(2)));

        // result shape: (batch_size, _, features_out)
        let mut result = Array3::zeros((data.shape()[0], data.shape()[1], self.weights.shape()[1]));
        for (mut out2d, arr2d) in result.axis_iter_mut(Axis(0)).zip(data.axis_iter(Axis(0))) {
            let mut tmp2d = arr2d.dot(&self.weights);
            tmp2d += &self.bias;
            out2d.assign(&tmp2d);
        }

        // Apply in-place activation on the result matrix.
        self.activation.activation_mut(&mut result);

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array, Array1, Array2, Array3};

    #[test]
    fn test_dense_2d_simple() {
        let data = Array2::from_elem([11, 7], 1.0);

        let weights = Array2::from_elem([7, 19], 1.0);
        let bias = Array1::from_elem([19], 1.0);
        let activation = Activation::Linear;
        let dense_layer = DenseLayer::new(weights, bias, activation);

        let result = dense_layer.apply2d(&data);
        let expected = Array2::from_elem([11, 19], 8.0);

        assert_eq!(expected, result);
    }

    #[test]
    fn test_dense_3d_simple() {
        let data = Array3::from_elem([10, 11, 7], 1.0);

        let weights = Array2::from_elem([7, 19], 1.0);
        let bias = Array1::from_elem([19], 1.0);
        let activation = Activation::Linear;
        let dense_layer = DenseLayer::new(weights, bias, activation);

        let result = dense_layer.apply3d(&data);
        let expected = Array3::from_elem([10, 11, 19], 8.0);

        assert_eq!(expected, result);
    }

    #[test]
    fn test_dense_2d_complex() {
        let data: Array2<f32> = Array::linspace(1., 33., 33).into_shape([3, 11]).unwrap();

        let weights: Array2<f32> = Array::linspace(1., 88., 88).into_shape([11, 8]).unwrap();
        let bias: Array1<f32> = Array::linspace(1., 8., 8).into_shape([8]).unwrap();
        let activation = Activation::Linear;
        let dense_layer = DenseLayer::new(weights, bias, activation);

        let result = dense_layer.apply2d(&data);
        let expected = array![
            [3587.0, 3654.0, 3721.0, 3788.0, 3855.0, 3922.0, 3989.0, 4056.0],
            [8548.0, 8736.0, 8924.0, 9112.0, 9300.0, 9488.0, 9676.0, 9864.0],
            [13509.0, 13818.0, 14127.0, 14436.0, 14745.0, 15054.0, 15363.0, 15672.0]
        ];

        assert_eq!(expected, result);
    }

    #[test]
    fn test_dense_3d_complex() {
        let data: Array3<f32> = Array::linspace(1., 66., 66).into_shape([2, 3, 11]).unwrap();

        let weights: Array2<f32> = Array::linspace(1., 88., 88).into_shape([11, 8]).unwrap();
        let bias: Array1<f32> = Array::linspace(1., 8., 8).into_shape([8]).unwrap();
        let activation = Activation::Linear;
        let dense_layer = DenseLayer::new(weights, bias, activation);

        let result = dense_layer.apply3d(&data);
        let expected = array![
            [
                [3587.0, 3654.0, 3721.0, 3788.0, 3855.0, 3922.0, 3989.0, 4056.0],
                [8548.0, 8736.0, 8924.0, 9112.0, 9300.0, 9488.0, 9676.0, 9864.0],
                [13509.0, 13818.0, 14127.0, 14436.0, 14745.0, 15054.0, 15363.0, 15672.0]
            ],
            [
                [18470.0, 18900.0, 19330.0, 19760.0, 20190.0, 20620.0, 21050.0, 21480.0],
                [23431.0, 23982.0, 24533.0, 25084.0, 25635.0, 26186.0, 26737.0, 27288.0],
                [28392.0, 29064.0, 29736.0, 30408.0, 31080.0, 31752.0, 32424.0, 33096.0]
            ]
        ];

        assert_eq!(expected, result);
    }

    #[test]
    #[should_panic]
    fn test_dense_2d_panic_wrong_in_features() {
        let data: Array2<f32> = Array::linspace(1., 33., 33).into_shape([3, 11]).unwrap();

        // instead of 10 it should be 11 (features_in)
        let weights: Array2<f32> = Array::linspace(1., 80., 80).into_shape([10, 8]).unwrap();
        let bias: Array1<f32> = Array::linspace(1., 8., 8).into_shape([8]).unwrap();
        let activation = Activation::Linear;
        let dense_layer = DenseLayer::new(weights, bias, activation);

        _ = dense_layer.apply2d(&data);
    }

    #[test]
    #[should_panic]
    fn test_dense_2d_panic_bias() {
        let data: Array2<f32> = Array::linspace(1., 33., 33).into_shape([3, 11]).unwrap();

        let weights: Array2<f32> = Array::linspace(1., 88., 88).into_shape([11, 8]).unwrap();
        // instead of 7 it should be 8 (features_out)
        let bias: Array1<f32> = Array::linspace(1., 7., 7).into_shape([7]).unwrap();
        let activation = Activation::Linear;
        let dense_layer = DenseLayer::new(weights, bias, activation);

        _ = dense_layer.apply2d(&data);
    }
}
