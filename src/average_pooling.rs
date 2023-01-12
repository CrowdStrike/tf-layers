use crate::padding::padding;
use ndarray::{Array, Axis, Dimension};
use serde::{Deserialize, Serialize};
use std::cmp::min;

/// Defines a 1D average pooling layer type.
/// Downsamples the input representation by taking the average value
/// over the window defined by `pool_size`. The window is shifted by `strides`.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AveragePooling1DLayer {
    /// Size of the pooling windows.
    pool_size: usize,
    /// Factor by which to downscale. E.g. 2 will halve the input.
    strides: usize,
    /// Vector of (prefix, suffix) tuples, one for every input dimension,
    /// used for resizing of input, by adding padding as prefix and suffix.
    padding: Vec<(usize, usize)>,
}

impl AveragePooling1DLayer {
    /// Returns a new [`AveragePooling1DLayer`] from predefined parameters.
    ///
    /// Will panic if `pool_size` or `strides` are 0 or if the padding is empty.
    #[must_use]
    pub fn new(
        pool_size: usize,
        strides: usize,
        padding: Vec<(usize, usize)>,
    ) -> AveragePooling1DLayer {
        assert!(
            strides > 0 && pool_size > 0,
            "Strides and pool_size should be non-zero!"
        );
        assert!(!padding.is_empty(), "Padding vector should not be empty!");

        AveragePooling1DLayer {
            pool_size,
            strides,
            padding,
        }
    }

    /// Apply average pooling on the input data.
    /// Note: The pooling shape is `(self.pool_size,)` for 1d arrays and `(self.pool_size, 1, 1, ...)` for Nd arrays.
    ///       Simmilarly, the stride is `(self.strides,)` for 1d arrays and `(self.strides, 1, 1, ...)` for Nd arrays
    #[must_use]
    pub fn apply<D: Dimension>(&self, data: &Array<f32, D>) -> Array<f32, D> {
        // Data must be padded before applying the pooling layer.
        // padding will fail if data.ndim() != pdding.len() \
        let data = padding(data, &self.padding);

        // Compute the output shape
        let mut out_shape = data.raw_dim();
        // the second axis will be the different one, unless this is an Array1
        let axis = min(1, data.ndim() - 1);
        // check bounds
        assert!(
            data.shape()[axis] >= self.pool_size,
            "Pooling size({}) cannot be larger than the data size({})!",
            data.shape()[axis],
            self.pool_size
        );
        // the number of sliding windows of `pool_size` size, with a slide size of
        // `strides`, that fit into `(data.shape()[axis]`
        out_shape[axis] = (data.shape()[axis] - self.pool_size) / self.strides + 1;

        let mut result = Array::zeros(out_shape);

        for (mut out_lane, lane) in result
            .lanes_mut(Axis(axis))
            .into_iter()
            .zip(data.lanes(Axis(axis)))
        {
            let pool_lane = lane
                .windows(self.pool_size)
                .into_iter()
                .step_by(self.strides);

            for (elem, window_matrix) in out_lane.iter_mut().zip(pool_lane) {
                *elem = window_matrix.mean().unwrap();
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1, Array2, Array3};

    #[test]
    fn test_averagepooling1d() {
        let data: Array1<f32> = array![4.16634429, 3.6800784, 7.14640084, 5.70240999, 1.75683464];

        let averagepooling_layer = AveragePooling1DLayer::new(3, 2, vec![(0, 0)]);
        let result = averagepooling_layer.apply(&data);
        let expected: Array1<f32> = array![4.9976077, 4.868549];

        assert_eq!(expected, result);
    }

    #[test]
    fn test_averagepooling1d_identity() {
        let data: Array1<f32> = array![4.16634429, 3.6800784, 7.14640084, 5.70240999, 1.75683464];

        let averagepooling_layer = AveragePooling1DLayer::new(1, 1, vec![(0, 0)]);
        let result = averagepooling_layer.apply(&data);
        // no change should happen
        assert_eq!(data, result);
    }

    #[test]
    fn test_averagepooling2d() {
        let data: Array2<f32> = array![
            [4., 3., 1., 5.],
            [1., 3., 4., 8.],
            [4., 5., 4., 3.],
            [6., 5., 9., 4.]
        ];

        let averagepooling_layer = AveragePooling1DLayer::new(2, 2, vec![(0, 0), (0, 0)]);
        let result = averagepooling_layer.apply(&data);
        let expected: Array2<f32> = array![[3.5, 3.0], [2.0, 6.0], [4.5, 3.5], [5.5, 6.5]];

        assert_eq!(expected, result);
    }

    #[test]
    fn test_averagepooling3d() {
        let data: Array3<f32> = array![
            [
                [4.16634429, 3.6800784, 7.14640084, 5.70240999, 1.75683464],
                [7.30367663, 9.564133, 4.76055381, -0.07668671, 1.63573266],
                [-0.96895455, -0.38939883, 4.20417899, 2.02164234, 4.17297862],
                [-0.65123754, 1.9421113, 0.08885265, 7.81152724, 5.85272977]
            ],
            [
                [5.97592671, -1.39181533, 5.77478317, 4.33229714, 3.36414305],
                [1.71159761, 3.1096064, 2.43456038, 2.94875466, 1.45737179],
                [4.9765289, 5.64986778, 2.21295668, 1.3772863, 4.30951371],
                [-0.99992831, 1.10193819, 1.15754957, 0.05423748, -1.58379326]
            ]
        ];

        let averagepooling_layer = AveragePooling1DLayer::new(3, 2, vec![(0, 0), (0, 0), (0, 0)]);
        let result = averagepooling_layer.apply(&data);
        let expected: Array3<f32> = array![
            [[3.5003555, 4.2849374, 5.370378, 2.5491219, 2.5218484]],
            [[4.221351, 2.4558864, 3.4740999, 2.8861125, 3.0436764]]
        ];

        assert_eq!(expected, result);
    }

    #[test]
    fn test_averagepooling3d_with_padding() {
        let data: Array3<f32> = array![
            [
                [4.16634429, 3.6800784, 7.14640084, 5.70240999, 1.75683464],
                [7.30367663, 9.564133, 4.76055381, -0.07668671, 1.63573266],
                [-0.96895455, -0.38939883, 4.20417899, 2.02164234, 4.17297862],
                [-0.65123754, 1.9421113, 0.08885265, 7.81152724, 5.85272977]
            ],
            [
                [5.97592671, -1.39181533, 5.77478317, 4.33229714, 3.36414305],
                [1.71159761, 3.1096064, 2.43456038, 2.94875466, 1.45737179],
                [4.9765289, 5.64986778, 2.21295668, 1.3772863, 4.30951371],
                [-0.99992831, 1.10193819, 1.15754957, 0.05423748, -1.58379326]
            ]
        ];

        let averagepooling_layer = AveragePooling1DLayer::new(2, 2, vec![(1, 0), (0, 1), (1, 1)]);
        let result = averagepooling_layer.apply(&data);
        let expected: Array3<f32> = array![
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ],
            [
                [0.0, 5.73501, 6.6221056, 5.9534774, 2.8128617, 1.6962836, 0.0],
                [0.0, -0.810096, 0.7763562, 2.1465158, 4.916585, 5.012854, 0.0]
            ],
            [
                [0.0, 3.8437622, 0.8588956, 4.1046715, 3.6405258, 2.4107575, 0.0],
                [0.0, 1.9883004, 3.3759031, 1.6852531, 0.7157619, 1.3628602, 0.0]
            ]
        ];

        assert_eq!(expected, result);
    }

    #[test]
    #[should_panic]
    fn test_averagepooling_panic_higher_pool_size() {
        let data: Array3<f32> = Array3::from_shape_fn([2, 3, 3], |(i, j, k)| {
            if i % 2 == 0 {
                return (i + j + k) as f32;
            } else {
                return -((i + j + k) as f32);
            }
        });
        // panics because pool_size=7 is greater than 3 (rows) + 2 + 1 (padding) = 6
        let averagepooling_layer = AveragePooling1DLayer::new(7, 2, vec![(0, 0), (2, 1), (0, 0)]);
        let _ = averagepooling_layer.apply(&data);
    }
}
