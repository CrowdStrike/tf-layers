use ndarray::{Array, Axis, Dimension};
use serde::{Deserialize, Serialize};
use std::cmp::PartialEq;

/// Activation functions supported
#[derive(Serialize, Deserialize, PartialEq, Debug, Clone, Copy)]
pub enum Activation {
    /// The linear unit activation function, `linear(x) = x`
    Linear,
    /// The rectified linear unit activation function, `relu(x) = max(x, 0)`.
    Relu,
    /// The thresholded rectified linear unit function, defined as:
    ///     f(x) = x, for x > theta
    ///     f(x) = 0 otherwise`
    ThresholdedRelu(f32),
    /// The scaled exponential linear unit activation, defined as:
    ///     if x > 0: return scale * x
    ///     if x < 0: return scale * alpha * (exp(x) - 1)
    Selu,
    /// Exponential activation function
    Exp,
    /// The hard sigmoid activation function, defined as:
    ///     f(x) = 0, for x < -2.5
    ///     f(x) = 1, for x > 2.5
    ///     f(x) = 0.2*x + 0,5, otherwise
    HardSigmoid,
    /// The sigmoid activation function, `sigmoid(x) = 1 / (1 + exp(-x))`.
    Sigmoid,
    /// The softmax activation function. Softmax converts a real vector to a
    /// vector of categorical probabilities. The elements of the output vector
    /// are in range (0, 1) and sum to 1.
    Softmax,
    /// Softplus activation function, `softplus(x) = ln(exp(x) + 1)`
    Softplus,
    /// Softsign activation function, `softsign(x) = x / (abs(x) + 1)`
    Softsign,
    /// Swish activation function, `swish(x) = x * sigmoid(x)`
    Swish,
    /// Hyperbolic tangent activation function.
    Tanh,
    /// No-Op
    None,
}

/// An implementation of multiple activations.
impl Activation {
    /// Applies the specified activation onto an array
    /// Returns a newly allocated array
    #[must_use]
    pub fn activation<D: Dimension>(self, data: &Array<f32, D>) -> Array<f32, D> {
        let mut result = data.clone();
        self.activation_mut(&mut result);

        result
    }

    /// Applies the specified activation onto an array in place
    pub fn activation_mut<D: Dimension>(self, data: &mut Array<f32, D>) {
        match self {
            // Softmax makes use of a per-row computation process
            // Converts from [a,b,c] to [e^a / (e^a + e^b + e^c), e^b / (e^a + e^b + e^c), e^c / (e^a + e^b + e^c)] (in place)
            Self::Softmax => {
                let axis = data.ndim() - 1;

                // Since Softmax([a_1, a_2, ... , a_n]) <=> Softmax([a_1 - x, a_2 - x, ..., a_n - x]), we will choose x = max(row)
                // and rewrite the data in order to avoid overflow (computing exp^1000 will generate this for instance).
                for mut row in data.lanes_mut(Axis(axis)) {
                    // find the maximum element from the array
                    let maximum_elem: f32 = row.fold(f32::NEG_INFINITY, |a, b| f32::max(a, *b));
                    // subtract the maximum from each element of the array and compute the exponential function.
                    row.mapv_inplace(|elem| f32::exp(elem - maximum_elem));
                    // get the sum all of the exponentials
                    let sum_of_row_exponentials = row.sum();
                    // normalize
                    row /= sum_of_row_exponentials;
                }
            }

            // The other activations use a per-element function
            // Convert from [a,b,c] to [activation(a), activation(b), activation(c)] (inplace)
            Self::Relu => data.mapv_inplace(|elem| f32::max(0.0, elem)),

            Self::Exp => data.mapv_inplace(f32::exp),

            Self::ThresholdedRelu(x) => data.mapv_inplace(|elem| f32::max(x, elem)),

            Self::Selu => {
                // constants taken from: https://www.tensorflow.org/api_docs/python/tf/keras/activations/selu
                #[allow(clippy::excessive_precision)]
                const SCALE: f32 = 1.050_700_98;
                #[allow(clippy::excessive_precision)]
                const ALPHA: f32 = 1.673_263_24;

                data.mapv_inplace(|elem| {
                    if elem < 0.0 {
                        SCALE * ALPHA * (elem.exp() - 1.0)
                    } else {
                        SCALE * elem
                    }
                });
            }

            Self::HardSigmoid => data.mapv_inplace(|elem| {
                if elem < -2.5 {
                    0.0
                } else if elem > 2.5 {
                    1.0
                } else {
                    0.2 * elem + 0.5
                }
            }),

            Self::Sigmoid => data.mapv_inplace(|elem| 1.0 / (1.0 + (-elem).exp())),

            Self::Softplus => data.mapv_inplace(|elem| (1.0 + elem.exp()).ln()),

            Self::Softsign => data.mapv_inplace(|elem| elem / (1.0 + elem.abs())),

            Self::Swish => data.mapv_inplace(|elem| elem * (1.0 / (1.0 + (-elem).exp()))),

            Self::Tanh => data.mapv_inplace(f32::tanh),

            Self::Linear | Self::None => {} // no-op
        };
    }
}

#[cfg(test)]
#[allow(clippy::excessive_precision)]
#[allow(clippy::approx_constant)]
#[allow(clippy::unreadable_literal)]
mod tests {
    use super::*;
    use ndarray::{array, Array1, Array2, Array3};

    #[test]
    fn test_relu_1d() {
        let data: Array1<f32> = Array1::from_shape_vec([4], vec![-2.0, 2.0, 4.0, -7.0]).unwrap();

        let result: Array1<f32> = Activation::Relu.activation(&data);
        let expected: Array1<f32> = array![0.0, 2.0, 4.0, 0.0];

        assert_eq!(expected, result);
    }

    #[test]
    fn test_softmax_1d() {
        let data: Array1<f32> = Array1::from_shape_vec([4], vec![-2.0, 2.0, 4.0, -7.0]).unwrap();

        let result: Array1<f32> = Activation::Softmax.activation(&data);
        let expected: Array1<f32> = array![0.0021784895, 0.118941486, 0.87886536, 0.000014678546];

        assert_eq!(expected, result);
    }

    #[test]
    fn test_softmax_1d_large_exponents() {
        let data: Array1<f32> =
            Array1::from_shape_vec([4], vec![-250.0, -250.0, -250.0, 250.0]).unwrap();

        let result: Array1<f32> = Activation::Softmax.activation(&data);
        let expected: Array1<f32> = array![0.0, 0.0, 0.0, 1.0];

        assert_eq!(expected, result);
    }

    #[test]
    fn test_relu_2d() {
        let data: Array2<f32> = Array2::from_shape_vec([2, 2], vec![-2.0, 2.0, 4.0, -7.0]).unwrap();

        let result: Array2<f32> = Activation::Relu.activation(&data);
        let expected: Array2<f32> = array![[0.0, 2.0], [4.0, 0.0]];

        assert_eq!(expected, result);
    }

    #[test]
    fn test_selu_2d() {
        let data: Array2<f32> = Array2::from_shape_vec([2, 2], vec![-2.0, 2.0, 4.0, -7.0]).unwrap();

        let result: Array2<f32> = Activation::Selu.activation(&data);
        let expected: Array2<f32> = array![[-1.5201665, 2.101402], [4.202804, -1.7564961]];

        assert_eq!(expected, result);
    }

    #[test]
    fn test_hardsigmoid_1d() {
        let data: Array1<f32> = Array1::from_shape_vec(5, vec![-3.0, -1.0, 0.0, 1.0, 3.0]).unwrap();

        let result: Array1<f32> = Activation::HardSigmoid.activation(&data);
        let expected = array![0.0, 0.3, 0.5, 0.7, 1.0];

        assert_eq!(expected, result);
    }

    #[test]
    fn test_hardsigmoid_2d() {
        let data: Array2<f32> = Array2::from_shape_vec((2, 2), vec![-3.0, -1.0, 0.0, 1.0]).unwrap();

        let result: Array2<f32> = Activation::HardSigmoid.activation(&data);
        let expected = array![[0.0, 0.3], [0.5, 0.7]];

        assert_eq!(expected, result);
    }

    #[test]
    fn test_hardsigmoid_3d() {
        let data: Array3<f32> = array![
            [[-2.0, 2.0, 0.55], [4.0, -7.0, -2.15]],
            [[-3.0, -0.5, 3.33], [-3.66, 0.0, 2.75]]
        ];

        let result: Array3<f32> = Activation::HardSigmoid.activation(&data);
        let expected = array![
            [[0.099999994, 0.9, 0.61], [1.0, 0.0, 0.06999996]],
            [[0.0, 0.4, 1.0], [0.0, 0.5, 1.0]]
        ];

        assert_eq!(expected, result);
    }

    #[test]
    fn test_sigmoid_2d() {
        let data: Array2<f32> = Array2::from_shape_vec([2, 2], vec![-2.0, 2.0, 4.0, -7.0]).unwrap();

        let result: Array2<f32> = Activation::Sigmoid.activation(&data);
        let expected: Array2<f32> = array![[0.11920292, 0.880797], [0.98201376, 0.0009110512]];

        assert_eq!(expected, result);
    }

    #[test]
    fn test_softmax_2d() {
        let data: Array2<f32> = Array2::from_shape_vec([2, 2], vec![-2.0, 2.0, 4.0, -7.0]).unwrap();

        let result: Array2<f32> = Activation::Softmax.activation(&data);
        let expected: Array2<f32> = array![[0.01798621, 0.98201376], [0.9999833, 0.000016701422]];

        assert_eq!(expected, result);
    }

    #[test]
    fn test_no_activation_2d() {
        let data: Array2<f32> = Array2::from_shape_vec([2, 2], vec![-2.0, 2.0, 4.0, -7.0]).unwrap();

        let result: Array2<f32> = Activation::None.activation(&data);
        let expected = data;

        assert_eq!(expected, result);
    }

    #[test]
    fn test_relu_3d() {
        let data: Array3<f32> = array![
            [[-2.0, 2.0, 0.55], [4.0, -7.0, -2.15]],
            [[-3.0, -0.5, 3.33], [-3.66, 0.0, 2.75]]
        ];

        let result: Array3<f32> = Activation::Relu.activation(&data);
        let expected: Array3<f32> = array![
            [[0., 2., 0.55], [4., 0., 0.]],
            [[0., 0., 3.33], [0., 0., 2.75]]
        ];

        assert_eq!(expected, result);
    }

    #[test]
    fn test_selu_3d() {
        let data: Array3<f32> = array![
            [[-2.0, 2.0, 0.55], [4.0, -7.0, -2.15]],
            [[-3.0, -0.5, 3.33], [-3.66, 0.0, 2.75]]
        ];

        let result: Array3<f32> = Activation::Selu.activation(&data);
        let expected: Array3<f32> = array![
            [
                [-1.5201665, 2.101402, 0.57788557],
                [4.202804, -1.7564961, -1.5533086]
            ],
            [
                [-1.6705687, -0.69175816, 3.4988344],
                [-1.712859, 0., 2.889428]
            ]
        ];

        assert_eq!(expected, result);
    }

    #[test]
    fn test_sigmoid_3d() {
        let data: Array3<f32> = array![
            [[-2.0, 2.0, 0.55], [4.0, -7.0, -2.15]],
            [[-3.0, -0.5, 3.33], [-3.66, 0.0, 2.75]]
        ];

        let result: Array3<f32> = Activation::Sigmoid.activation(&data);
        let expected: Array3<f32> = array![
            [
                [0.11920292, 0.880797, 0.6341356],
                [0.98201376, 0.0009110512, 0.10433122]
            ],
            [
                [0.047425874, 0.37754068, 0.9654438],
                [0.02508696, 0.5, 0.93991333]
            ]
        ];

        assert_eq!(expected, result);
    }

    #[test]
    fn test_softmax_3d() {
        let data: Array3<f32> = array![
            [[-2.0, 2.0, 0.55], [4.0, -7.0, -2.15]],
            [[-3.0, -0.5, 3.33], [-3.66, 0.0, 2.75]]
        ];

        let result: Array3<f32> = Activation::Softmax.activation(&data);
        let expected: Array3<f32> = array![
            [
                [0.01461876, 0.7981573, 0.18722397],
                [0.9978544, 0.000016665866, 0.0021289042]
            ],
            [
                [0.0017411319, 0.021211328, 0.97704756],
                [0.0015437938, 0.05999389, 0.9384623]
            ]
        ];

        assert_eq!(expected, result);
    }

    #[test]
    fn test_exp_1d() {
        let data: Array1<f32> = Array1::from_shape_vec(5, vec![-3.0, -1.0, 0.0, 1.0, 3.0]).unwrap();

        let result = Activation::Exp.activation(&data);
        let expected: Array1<f32> =
            Array1::from_shape_vec(5, vec![0.049787067, 0.36787945, 1., 2.7182817, 20.085537])
                .unwrap();

        assert_eq!(expected, result)
    }

    #[test]
    fn test_exp_2d() {
        let data: Array2<f32> = Array2::from_shape_vec((2, 2), vec![-3.0, -1.0, 0.0, 1.0]).unwrap();

        let result = Activation::Exp.activation(&data);
        let expected = array![[0.049787067, 0.36787945], [1., 2.7182817]];

        assert_eq!(expected, result)
    }

    #[test]
    fn test_softplus_1d() {
        let input: Array1<f32> =
            Array1::from_shape_vec(5, vec![-20.0, -1.0, 0.0, 1.0, 20.0]).unwrap();
        let expected =
            Array1::from_shape_vec(5, vec![0.0, 0.31326166, 0.69314718, 1.3132616, 20.0]).unwrap();

        assert_eq!(Activation::Softplus.activation(&input), expected);
    }

    #[test]
    fn test_softplus_2d_mut() {
        let mut input: Array2<f32> =
            Array2::from_shape_vec((2, 3), vec![-20.0, -1.0, 0.0, 1.0, 20.0, 10.0]).unwrap();
        let expected = Array2::from_shape_vec(
            (2, 3),
            vec![0.0, 0.31326166, 0.69314718, 1.3132616, 20.0, 10.000046],
        )
        .unwrap();

        Activation::Softplus.activation_mut(&mut input);

        assert_eq!(input, expected);
    }

    #[test]
    fn test_softsign_1d() {
        let input: Array1<f32> = Array1::from_shape_vec(3, vec![-1.0, 0.0, 1.0]).unwrap();
        let expected = Array1::from_shape_vec(3, vec![-0.5, 0.0, 0.5]).unwrap();

        assert_eq!(Activation::Softsign.activation(&input), expected);
    }

    #[test]
    fn test_softsign_2d_mut() {
        let mut input: Array2<f32> =
            Array2::from_shape_vec((2, 2), vec![-1.0, 0.0, f32::MIN, 1.0]).unwrap();
        let expected = Array2::from_shape_vec((2, 2), vec![-0.5, 0.0, -1.0, 0.5]).unwrap();

        Activation::Softsign.activation_mut(&mut input);

        assert_eq!(input, expected);
    }

    #[test]
    fn test_swish_1d() {
        let data: Array1<f32> = Array1::from_shape_vec(4, vec![-2.0, 2.0, 4.0, -7.0]).unwrap();

        let result: Array1<f32> = Activation::Swish.activation(&data);
        let expected: Array1<f32> = array![-0.23840584, 1.761594, 3.92805516, -0.006377358];

        assert_eq!(expected, result);
    }

    #[test]
    fn test_swish_2d() {
        let data: Array2<f32> = Array2::from_shape_vec((2, 2), vec![-2.0, 2.0, 4.0, -7.0]).unwrap();

        let result: Array2<f32> = Activation::Swish.activation(&data);
        let expected: Array2<f32> = array![[-0.23840584, 1.761594], [3.92805516, -0.006377358]];

        assert_eq!(expected, result)
    }

    #[test]
    fn test_tanh_1d() {
        let input: Array1<f32> =
            Array1::from_shape_vec(5, vec![-3.0, -1.0, 0.0, 1.0, 3.0]).unwrap();
        let expected =
            Array1::from_shape_vec(5, vec![-0.9950548, -0.7615942, 0.0, 0.7615942, 0.9950548])
                .unwrap();

        assert_eq!(Activation::Tanh.activation(&input), expected);
    }

    #[test]
    fn test_tanh_2d_mut() {
        let mut input: Array2<f32> =
            Array2::from_shape_vec((2, 3), vec![-3.0, -1.0, 0.0, 1.0, 3.0, f32::MIN]).unwrap();
        let expected = Array2::from_shape_vec(
            (2, 3),
            vec![-0.9950548, -0.7615942, 0.0, 0.7615942, 0.9950548, -1.0],
        )
        .unwrap();

        Activation::Tanh.activation_mut(&mut input);

        assert_eq!(input, expected);
    }

    #[test]
    fn test_no_activation_3d() {
        let data: Array3<f32> = array![
            [[-2.0, 2.0, 0.55], [4.0, -7.0, -2.15]],
            [[-3.0, -0.5, 3.33], [-3.66, 0.0, 2.75]]
        ];

        let result: Array3<f32> = Activation::None.activation(&data);
        let expected: Array3<f32> = data;

        assert_eq!(expected, result);
    }

    #[test]
    fn test_relu_1d_mut() {
        let mut data: Array1<f32> =
            Array1::from_shape_vec([4], vec![-2.0, 2.0, 4.0, -7.0]).unwrap();

        Activation::Relu.activation_mut(&mut data);
        let expected: Array1<f32> = array![0.0, 2.0, 4.0, 0.0];

        assert_eq!(expected, data);
    }

    #[test]
    fn test_softmax_1d_mut() {
        let mut data: Array1<f32> =
            Array1::from_shape_vec([4], vec![-2.0, 2.0, 4.0, -7.0]).unwrap();

        Activation::Softmax.activation_mut(&mut data);
        let expected: Array1<f32> = array![0.0021784895, 0.118941486, 0.87886536, 0.000014678546];

        assert_eq!(expected, data);
    }

    #[test]
    fn test_relu_2d_mut() {
        let mut data: Array2<f32> =
            Array2::from_shape_vec([2, 2], vec![-2.0, 2.0, 4.0, -7.0]).unwrap();

        Activation::Relu.activation_mut(&mut data);
        let expected: Array2<f32> = array![[0.0, 2.0], [4.0, 0.0]];

        assert_eq!(expected, data);
    }

    #[test]
    fn test_softmax_2d_mut() {
        let mut data: Array2<f32> =
            Array2::from_shape_vec([2, 2], vec![-2.0, 2.0, 4.0, -7.0]).unwrap();

        Activation::Softmax.activation_mut(&mut data);
        let expected: Array2<f32> = array![[0.01798621, 0.98201376], [0.9999833, 0.000016701422]];

        assert_eq!(expected, data);
    }

    #[test]
    fn test_relu_3d_mut() {
        let mut data: Array3<f32> = array![
            [[-2.0, 2.0, 0.55], [4.0, -7.0, -2.15]],
            [[-3.0, -0.5, 3.33], [-3.66, 0.0, 2.75]]
        ];

        Activation::Relu.activation_mut(&mut data);
        let expected: Array3<f32> = array![
            [[0., 2., 0.55], [4., 0., 0.]],
            [[0., 0., 3.33], [0., 0., 2.75]]
        ];

        assert_eq!(expected, data);
    }

    #[test]
    fn test_softmax_3d_mut() {
        let mut data: Array3<f32> = array![
            [[-2.0, 2.0, 0.55], [4.0, -7.0, -2.15]],
            [[-3.0, -0.5, 3.33], [-3.66, 0.0, 2.75]]
        ];

        Activation::Softmax.activation_mut(&mut data);
        let expected: Array3<f32> = array![
            [
                [0.01461876, 0.7981573, 0.18722397],
                [0.9978544, 0.000016665866, 0.0021289042]
            ],
            [
                [0.0017411319, 0.021211328, 0.97704756],
                [0.0015437938, 0.05999389, 0.9384623]
            ]
        ];

        assert_eq!(expected, data);
    }
}
