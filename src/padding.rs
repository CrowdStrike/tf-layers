use ndarray::{Array, ArrayBase, Data, Dimension, Slice};
use num_traits::Zero;

use ndarray::Axis;

/// Add zero-padding before and/or after data in each dimension.
///
/// The length of `pad_width` must match the number of dimensions in `data`. For
/// each dimension of `data`, the corresponding entry in `pad_width` is a pair
/// of numbers `[i, j]`. `i` is the amount of zero-padding to insert before the
/// data on the corresponding Axis, while `j` is the padding after.
///
/// E.g., a padding of `[[1,1], [0, 0]]` over a 2d array means to insert 1
/// row of zeroes before and after (axis 0), and to insert 0 columns of zeroes
/// before and after (axis 1):
///
/// ```txt
///            0 0 0
/// 1 1 1      1 1 1
/// 1 1 1   => 1 1 1
/// 1 1 1      1 1 1
///            0 0 0
/// ```
///
/// # Panics
/// Will panic if `ndim` of `data` is not the same length as `pad_with`.
#[must_use]
pub fn padding<A, S, D>(data: &ArrayBase<S, D>, pad_width: &[(usize, usize)]) -> Array<A, D>
where
    A: Clone + Zero,
    S: Data<Elem = A>,
    D: Dimension,
{
    // For each data's dimension there is a list of two usize numbers [i, j] as padding
    // i is the padding before the corresponding Axis, while j is the padding after.
    assert_eq!(
        data.ndim(),
        pad_width.len(),
        "Ndims of data must match the length of pad_with."
    );

    // Compute the output shape
    let mut padded_shape = data.raw_dim();
    for (axis, &(pad_lo, pad_hi)) in pad_width.iter().enumerate() {
        padded_shape[axis] += pad_lo + pad_hi;
    }

    // Create an array full of zeros with this new shape.
    let mut padded = Array::zeros(padded_shape);

    // Transfer data to the padded matrix, taking into consideration the place where this data will be inserted.
    let mut orig_portion = padded.view_mut();
    for (axis, (&axis_size, &(pad_lo, _))) in data.shape().iter().zip(pad_width).enumerate() {
        orig_portion.slice_axis_inplace(Axis(axis), Slice::from(pad_lo..(pad_lo + axis_size)));
    }
    orig_portion.assign(data);

    // Return the new array containing the padding
    padded
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2, Array3};

    #[test]
    fn test_padding_1d() {
        let in_arr = Array1::from(vec![1, 2, 3, 4]);
        let expected = Array1::from(vec![0, 0, 1, 2, 3, 4, 0]);

        assert!(padding(&in_arr, &[(2, 1)]) == expected);
    }

    #[test]
    fn test_padding_2d() {
        let in_arr = Array2::from_shape_vec((2, 2), vec![1, 2, 3, 4]).unwrap();
        let expected = Array2::from_shape_vec(
            (5, 7),
            vec![
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 3, 4, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
            ],
        )
        .unwrap();

        assert!(padding(&in_arr, &[(2, 1), (2, 3)]) == expected);
    }

    #[test]
    fn test_padding_3d() {
        let in_arr = Array3::from_shape_vec((2, 2, 2), vec![1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
        let expected = Array3::from_shape_vec(
            (4, 3, 5),
            vec![
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 3, 4, 0, 0, 0,
                0, 0, 0, 0, 0, 5, 6, 0, 0, 0, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0,
            ],
        )
        .unwrap();

        assert!(padding(&in_arr, &[(1, 1), (0, 1), (3, 0)]) == expected);
    }
}
