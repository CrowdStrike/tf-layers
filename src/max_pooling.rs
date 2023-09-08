use crate::padding::padding;
use ndarray::{Array, Axis, Dimension};
use serde::{Deserialize, Serialize};
use std::cmp::min;

/// Defines a 1D maximum pooling layer type
/// Downsamples the input representation by taking the maximum value
/// over the window defined by `pool_size`. The window is shifted by `strides`.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MaxPooling1DLayer {
    /// Size of the pooling windows.
    pool_size: usize,
    /// Factor by which to downscale. E.g. 2 will halve the input.
    strides: usize,
    /// Vector of (prefix, suffix) tuples, one for every input dimension,
    /// used for resizing of input, by adding padding as prefix and suffix.
    padding: Vec<(usize, usize)>,
}

impl MaxPooling1DLayer {
    /// Returns a new [`MaxPooling1DLayer`] from predefined parameters.
    ///
    /// # Panics
    /// Will panic if `pool_size` or `strides` are 0 or if the padding is empty.
    #[must_use]
    pub fn new(
        pool_size: usize,
        strides: usize,
        padding: Vec<(usize, usize)>,
    ) -> MaxPooling1DLayer {
        assert!(
            strides > 0 && pool_size > 0,
            "Strides and pool_size should be non-zero!"
        );
        assert!(!padding.is_empty(), "Padding vector should not be empty!");

        MaxPooling1DLayer {
            pool_size,
            strides,
            padding,
        }
    }

    /// Apply max pooling on the input data.
    /// Note: The pooling shape is `(self.pool_size,)` for 1d arrays and `(self.pool_size, 1, 1, ...)` for Nd arrays.
    ///       Similarly, the stride is `(self.strides,)` for 1d arrays and `(self.strides, 1, 1, ...)` for Nd arrays
    ///
    /// # Panics
    /// Pooling cannot be larger than the data.
    #[must_use]
    pub fn apply<D: Dimension>(&self, data: &Array<f32, D>) -> Array<f32, D> {
        // Data must be padded before applying the pooling layer.
        // padding will fail if data.ndim() != padding.len() \
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

            for (elem, window) in out_lane.iter_mut().zip(pool_lane) {
                *elem = window.fold(f32::NEG_INFINITY, |prev, &curr| f32::max(prev, curr));
            }
        }

        result
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
#[allow(clippy::excessive_precision)]
#[allow(clippy::unreadable_literal)]
mod tests {
    use super::*;
    use ndarray::{array, Array1, Array2, Array3};

    #[test]
    fn test_maxpooling1d() {
        let data: Array1<f32> = array![4.16634429, 3.6800784, 7.14640084, 5.70240999, 1.75683464];

        let pooling_layer = MaxPooling1DLayer::new(3, 2, vec![(0, 0)]);
        let result = pooling_layer.apply(&data);
        let expected: Array1<f32> = array![7.146401, 7.146401];

        assert_eq!(expected, result);
    }

    #[test]
    fn test_maxpooling1d_identity() {
        let data: Array1<f32> = array![4.16634429, 3.6800784, 7.14640084, 5.70240999, 1.75683464];

        let pooling_layer = MaxPooling1DLayer::new(1, 1, vec![(0, 0)]);
        let result = pooling_layer.apply(&data);
        // no change should happen
        assert_eq!(data, result);
    }

    #[test]
    fn test_maxpooling2d() {
        let data: Array2<f32> = array![
            [4., 3., 1., 5.],
            [1., 3., 4., 8.],
            [4., 5., 4., 3.],
            [6., 5., 9., 4.]
        ];

        let pooling_layer = MaxPooling1DLayer::new(2, 2, vec![(0, 0), (0, 0)]);
        let result = pooling_layer.apply(&data);
        let expected: Array2<f32> = array![[4.0, 5.0], [3.0, 8.0], [5.0, 4.0], [6.0, 9.0]];

        assert_eq!(expected, result);
    }

    #[test]
    fn test_maxpooling() {
        let data: Array3<f32> = Array3::from_shape_fn([2, 5, 5], |(i, j, k)| {
            if i % 2 == 0 {
                (i + j + k) as f32
            } else {
                -((i + j + k) as f32)
            }
        });
        let maxpooling_layer = MaxPooling1DLayer::new(3, 2, vec![(0, 0), (2, 3), (0, 0)]);
        let result = maxpooling_layer.apply(&data);
        let expected: Array3<f32> = array![
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [2.0, 3.0, 4.0, 5.0, 6.0],
                [4.0, 5.0, 6.0, 7.0, 8.0],
                [4.0, 5.0, 6.0, 7.0, 8.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [-1.0, -2.0, -3.0, -4.0, -5.0],
                [-3.0, -4.0, -5.0, -6.0, -7.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        ];

        assert_eq!(expected, result);
    }

    #[test]
    fn test_maxpooling3d_with_padding() {
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

        let pooling_layer = MaxPooling1DLayer::new(2, 2, vec![(1, 0), (0, 1), (1, 1)]);
        let result = pooling_layer.apply(&data);
        let expected: Array3<f32> = array![
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ],
            [
                [0.0, 7.3036766, 9.564133, 7.146401, 5.70241, 1.7568346, 0.0],
                [
                    0.0,
                    -0.65123755,
                    1.9421113,
                    4.204179,
                    7.8115273,
                    5.85273,
                    0.0
                ]
            ],
            [
                [0.0, 5.975927, 3.1096065, 5.774783, 4.3322973, 3.3641431, 0.0],
                [0.0, 4.976529, 5.649868, 2.2129567, 1.3772863, 4.3095136, 0.0]
            ]
        ];

        assert_eq!(expected, result);
    }

    #[test]
    #[should_panic]
    fn test_maxpooling_panic_higher_pool_size() {
        let data: Array3<f32> = Array3::from_shape_fn([2, 3, 3], |(i, j, k)| {
            if i % 2 == 0 {
                (i + j + k) as f32
            } else {
                -((i + j + k) as f32)
            }
        });
        let maxpooling_layer = MaxPooling1DLayer::new(7, 2, [(0, 0), (2, 1), (0, 0)].to_vec());
        _ = maxpooling_layer.apply(&data);
    }
}
