use crate::activations::Activation;
use crate::padding::padding;
use ndarray::{Array1, Array2, Array3, Axis};
use serde::{Deserialize, Serialize};

/// Defines a one dimensional convolutional layer kernel
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Conv1DLayer {
    weights: Array2<f32>,
    bias: Array1<f32>,
    kernel_size: usize,
    nb_filters: usize,
    no_columns: usize,
    strides: usize,
    padding: Vec<(usize, usize)>,
    dilation_rate: usize,
    groups: usize,
    activation: Activation,
}

impl Conv1DLayer {
    /// Returns a new [`Conv1DLayer`] from predefined parameters.
    ///
    /// # Panics
    /// If `weights` cannot be converted to the output shape.
    #[must_use]
    pub fn new(
        weights: Array3<f32>,
        bias: Array1<f32>,
        strides: usize,
        padding: Vec<(usize, usize)>,
        dilation_rate: usize,
        groups: usize,
        activation: Activation,
    ) -> Conv1DLayer {
        let nb_filters = weights.len_of(Axis(0));
        let kernel_size = weights.len_of(Axis(1));
        let no_columns = weights.len_of(Axis(2));

        // Reformat the weights shape (3D -> 2D) such that each kernel becomes a column
        // in the new 2D matrix.
        let weights: Array2<f32> = {
            // Convert 3D weights into 2D weights by flattening over Axis(0) and transpose the
            // result as it will be called as input.dot(weights)
            let output_shape = [
                weights.raw_dim()[0],
                weights.raw_dim()[1] * weights.raw_dim()[2],
            ];
            weights.into_shape(output_shape).unwrap().reversed_axes()
        };

        Conv1DLayer {
            weights,
            bias,
            kernel_size,
            nb_filters,
            no_columns,
            strides,
            padding,
            dilation_rate,
            groups,
            activation,
        }
    }

    /// Returns a convolution of this kernel with the input data
    ///
    /// # Panics
    /// Input data and Conv1DLayer's weights must have the same number of columns.
    ///
    /// Kernel cannot be larger than the data (this includes padding).
    #[must_use]
    pub fn apply(&self, data: &Array3<f32>) -> Array3<f32> {
        // Data must be padded before applying the pooling layer.
        // E.g. A padding [[0,0], [1,1], [0, 0]] over a 3d array (assume the 2d matrix showed below is a sample
        // from the entire 3d array when iterating over Axis(0)) means:
        //            0 0 0
        // 1 1 1      1 1 1
        // 1 1 1   => 1 1 1
        // 1 1 1      1 1 1
        //            0 0 0

        // First, apply padding to the input.
        let data: Array3<f32> = padding(data, &self.padding);

        // Assert data (after padding) and weights have the same number of columns
        assert_eq!(
            data.len_of(Axis(2)),
            self.no_columns,
            "Input data and Conv1DLayer's weights must have the same number of columns!",
        );

        // Check bounds for the rows (Axis(1))
        assert!(
            data.shape()[1] >= self.kernel_size,
            "Kernel size({}) cannot be larger than the data size({}) (this includes padding)!",
            data.shape()[1],
            self.kernel_size
        );

        // The number of sliding windows of `kernel_size` size, with a slide size of
        // `strides`, that fit into `(data.shape()[axis]`
        let no_sliding_windows = (data.shape()[1] - self.kernel_size) / self.strides + 1;

        // Transform 3D data into 2D data, such that each row will be the flattened image of each
        // window (used in convolution) + one new element for the bias of each kernel
        let intermediate_shape: [usize; 2] = [
            data.shape()[0] * no_sliding_windows,
            self.kernel_size * self.no_columns,
        ];

        // Intermediate 1D vector to store all the flattened windows.
        let mut vector: Vec<f32> =
            Vec::with_capacity(intermediate_shape[0] * intermediate_shape[1]);

        // Iterate over each sample (since data comes in batch of multiple samples),
        // flatten their corresponding windows and add their elements to the vector.
        for data_matrix in data.axis_iter(Axis(0)) {
            // Create an iterator that gives windows of the same sizes as the kernels.
            // Strides represents the steps to skip.
            // E.g. pool_size=3, strides=2, result_matrix.shape() == [8, 10] =>
            // [0-2, :], [2-4, :], [4-6, :] (ranges are inclusive, so 0-2 means 0,1,2)
            // Obs: since [6-8, :] is out of range, it will not be included.
            for window in data_matrix
                .windows([self.kernel_size, data_matrix.len_of(Axis(1))])
                .into_iter()
                .step_by(self.strides)
            {
                // Push the last flattened window
                vector.extend_from_slice(window.as_slice().unwrap());
            }
        }

        // Reshape the 3D input into 2D (to be used in convolution operation).
        let data_intermediate: Array2<f32> =
            Array2::from_shape_vec(intermediate_shape, vector).unwrap();

        // Output shape
        let mut out_shape = data.raw_dim();
        // the number of sliding windows of `kernel_size` size, with a slide size of
        // `strides`, that fit into `(data.shape()[axis]`
        out_shape[1] = no_sliding_windows;
        out_shape[2] = self.nb_filters;

        // Here all the computation happens. Reshape is done to ensure the data
        // is reconstructed back from 2D to 3D.
        let mut result = data_intermediate
            .dot(&self.weights)
            .into_shape(out_shape)
            .unwrap();

        // Apply the bias
        result += &self.bias;

        // Apply in-place activation on the result matrix.
        self.activation.activation_mut(&mut result);
        result
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
mod tests {
    use super::*;
    use ndarray::{arr3, Array, Array1, Array3};

    #[test]
    fn test_conv1d_simple() {
        let data = Array3::from_elem([5, 11, 17], 1.0);

        let conv1d_layer = Conv1DLayer::new(
            Array3::from_elem([29, 3, 17], 1.0),
            Array1::from_elem([29], 1.0),
            1,
            vec![(0, 0), (0, 0), (0, 0)],
            0,
            0,
            Activation::Linear,
        );

        let result = conv1d_layer.apply(&data);
        let expected = Array3::from_elem([5, 9, 29], 52.0);

        assert_eq!(expected, result);
    }

    #[test]
    fn test_conv1d_with_padding() {
        let data = Array3::from_elem([1, 4, 5], 1.0);

        let conv1d_layer = Conv1DLayer::new(
            Array3::from_elem([2, 3, 5], 1.0),
            Array1::from_elem([2], 1.0),
            1,
            vec![(0, 0), (2, 1), (0, 0)],
            0,
            0,
            Activation::Linear,
        );

        let result = conv1d_layer.apply(&data);

        let expected: Array3<f32> = Array3::from_shape_vec(
            [1, 5, 2],
            vec![6.0, 6.0, 11.0, 11.0, 16.0, 16.0, 16.0, 16.0, 11.0, 11.0],
        )
        .unwrap();

        assert_eq!(expected, result);
    }

    #[test]
    fn test_conv1d_complex() {
        let data: Array3<f32> = Array::linspace(1.0, 189.0, 189)
            .into_shape([3, 7, 9])
            .unwrap();

        let weights: Array3<f32> = Array::linspace(1.0, 360.0, 360)
            .into_shape([10, 4, 9])
            .unwrap();
        let bias: Array1<f32> = Array::linspace(1.0, 10.0, 10).into_shape([10]).unwrap();

        let conv1d_layer = Conv1DLayer::new(
            weights,
            bias,
            2,
            vec![(0, 0), (2, 3), (0, 0)],
            0,
            0,
            Activation::Linear,
        );

        let result = conv1d_layer.apply(&data);
        let expected: Array3<f32> = arr3(&[
            [
                [
                    5188.0, 11345.0, 17502.0, 23659.0, 29816.0, 35973.0, 42130.0, 48287.0, 54444.0,
                    60601.0,
                ],
                [
                    16207.0, 40184.0, 64161.0, 88138.0, 112115.0, 136092.0, 160069.0, 184046.0,
                    208023.0, 232000.0,
                ],
                [
                    28195.0, 75500.0, 122805.0, 170110.0, 217415.0, 264720.0, 312025.0, 359330.0,
                    406635.0, 453940.0,
                ],
                [
                    20539.0, 69140.0, 117741.0, 166342.0, 214943.0, 263544.0, 312145.0, 360746.0,
                    409347.0, 457948.0,
                ],
                [
                    2716.0, 21833.0, 40950.0, 60067.0, 79184.0, 98301.0, 117418.0, 136535.0,
                    155652.0, 174769.0,
                ],
            ],
            [
                [
                    36373.0, 83354.0, 130335.0, 177316.0, 224297.0, 271278.0, 318259.0, 365240.0,
                    412221.0, 459202.0,
                ],
                [
                    58165.0, 163790.0, 269415.0, 375040.0, 480665.0, 586290.0, 691915.0, 797540.0,
                    903165.0, 1008790.0,
                ],
                [
                    70153.0, 199106.0, 328059.0, 457012.0, 585965.0, 714918.0, 843871.0, 972824.0,
                    1101777.0, 1230730.0,
                ],
                [
                    44353.0, 154190.0, 264027.0, 373864.0, 483701.0, 593538.0, 703375.0, 813212.0,
                    923049.0, 1032886.0,
                ],
                [
                    5551.0, 45080.0, 84609.0, 124138.0, 163667.0, 203196.0, 242725.0, 282254.0,
                    321783.0, 361312.0,
                ],
            ],
            [
                [
                    67558.0, 155363.0, 243168.0, 330973.0, 418778.0, 506583.0, 594388.0, 682193.0,
                    769998.0, 857803.0,
                ],
                [
                    100123.0, 287396.0, 474669.0, 661942.0, 849215.0, 1036488.0, 1223761.0,
                    1411034.0, 1598307.0, 1785580.0,
                ],
                [
                    112111.0, 322712.0, 533313.0, 743914.0, 954515.0, 1165116.0, 1375717.0,
                    1586318.0, 1796919.0, 2007520.0,
                ],
                [
                    68167.0, 239240.0, 410313.0, 581386.0, 752459.0, 923532.0, 1094605.0,
                    1265678.0, 1436751.0, 1607824.0,
                ],
                [
                    8386.0, 68327.0, 128268.0, 188209.0, 248150.0, 308091.0, 368032.0, 427973.0,
                    487914.0, 547855.0,
                ],
            ],
        ]);

        assert_eq!(expected, result);
    }

    #[test]
    #[should_panic]
    fn test_conv1d_panic_wrong_bias_dimension() {
        let data = Array3::from_elem([17, 10, 5], 1.0);

        let conv1d_layer = Conv1DLayer::new(
            Array3::from_elem([2, 5, 5], 1.0),
            Array1::from_elem([3], 1.0), // instead of 3 it should be a 2
            1,
            vec![(0, 0), (2, 1), (0, 0)],
            0,
            0,
            Activation::Linear,
        );

        _ = conv1d_layer.apply(&data);
    }

    #[test]
    #[should_panic]
    fn test_conv1d_panic_inconsistent_shape_data_shape_weight() {
        let data = Array3::from_elem([17, 10, 5], 1.0);

        let conv1d_layer = Conv1DLayer::new(
            Array3::from_elem([2, 5, 8], 1.0), // instead of 8 it should be a 5
            Array1::from_elem([2], 1.0),
            1,
            vec![(0, 0), (2, 1), (0, 0)],
            0,
            0,
            Activation::Linear,
        );

        _ = conv1d_layer.apply(&data);
    }

    #[test]
    #[should_panic]
    fn test_conv1d_panic_higher_kernel() {
        let data = Array3::from_elem([17, 10, 5], 1.0);

        let conv1d_layer = Conv1DLayer::new(
            // the kernel size (15) must be <= than 10 (data rows per sample) + the 2 paddings (2 + 1) = 13
            Array3::from_elem([2, 15, 5], 1.0),
            Array1::from_elem([2], 1.0),
            1,
            vec![(0, 0), (2, 1), (0, 0)],
            0,
            0,
            Activation::Linear,
        );

        _ = conv1d_layer.apply(&data);
    }
}
