use ndarray::{Array, Array2, Axis, Dimension};
use serde::{Deserialize, Serialize};

/// Defines a neural network embedding layer for turning indexes
/// into dense vectors of fixed size.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EmbeddingLayer {
    /// The embedding matrix
    embedding: Array2<f32>,
}

impl EmbeddingLayer {
    /// Returns a new [`EmbeddingLayer`] from an embedding matrix
    #[must_use]
    pub fn new(embedding: Array2<f32>) -> EmbeddingLayer {
        EmbeddingLayer { embedding }
    }

    /// Converts each element of the input data to a 1D array of `f32`.
    /// (the corresponding row in `self.embedding`)
    ///
    /// # Examples
    ///
    /// ```txt
    /// [[1,2,3], [1, 5, 5]] => [[embedding.index_axis(Axis(0), 1), embedding.index_axis(Axis(0), 2), embedding.index_axis(Axis(0), 3)],
    ///                          [embedding.index_axis(Axis(0), 1), embedding.index_axis(Axis(0), 5), embedding.index_axis(Axis(0), 5)]]
    ///
    /// [[[1],[2],[3]], [[1], [5], [5]]] => [[[embedding.index_axis(Axis(0), 1)], [embedding.index_axis(Axis(0), 2)], [embedding.index_axis(Axis(0), 3)]],
    ///                                      [[embedding.index_axis(Axis(0), 1)], [embedding.index_axis(Axis(0), 5)], [embedding.index_axis(Axis(0), 5)]]]
    /// ```
    ///
    /// As a direct consequence, the result matrix has a new dimension added.
    #[must_use]
    pub fn apply<T: Into<usize> + Copy, D: Dimension>(
        &self,
        data: &Array<T, D>,
    ) -> Array<f32, D::Larger> {
        // add a new axis the size of embedding's column count
        let mut dim = data.raw_dim().insert_axis(Axis(data.ndim()));
        dim[data.ndim()] = self.embedding.ncols();

        let mut result: Array<f32, D::Larger> = Array::zeros(dim);

        // Loop over each row in the new `result` array zipped with the associated
        // `elem` from `data`, and populate the row with the corresponding embedding
        for (mut out, &elem) in result.lanes_mut(Axis(data.ndim())).into_iter().zip(data) {
            out.assign(&self.embedding.row(elem.into()));
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr3, array, Array, Array2, Array3, Array4};

    #[test]
    fn test_embedding_2d() {
        let data: Array2<usize> = array![[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 4, 7]];

        let weights: Array2<f32> = Array::linspace(1., 1000., 1000)
            .into_shape([100, 10])
            .unwrap();
        let embedding_layer = EmbeddingLayer::new(weights);

        let result: Array3<f32> = embedding_layer.apply(&data);
        let expected: Array3<f32> = arr3(&[
            [
                [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
                [21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0],
                [31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0],
            ],
            [
                [41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0],
                [51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0],
                [61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0],
            ],
            [
                [71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0],
                [81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0],
                [91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0],
            ],
            [
                [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
                [41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0],
                [71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0],
            ],
        ]);
        assert_eq!(expected, result);
    }

    #[test]
    fn test_embedding_2d_u16_input() {
        let data: Array2<u16> = array![[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 4, 7]];

        let weights: Array2<f32> = Array::linspace(1., 1000., 1000)
            .into_shape([100, 10])
            .unwrap();
        let embedding_layer = EmbeddingLayer::new(weights);

        let result: Array3<f32> = embedding_layer.apply(&data);
        let expected: Array3<f32> = arr3(&[
            [
                [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
                [21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0],
                [31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0],
            ],
            [
                [41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0],
                [51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0],
                [61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0],
            ],
            [
                [71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0],
                [81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0],
                [91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0],
            ],
            [
                [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
                [41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0],
                [71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0],
            ],
        ]);
        assert_eq!(expected, result);
    }

    #[test]
    fn test_embedding_3d() {
        let data: Array3<usize> = array![[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [1, 4, 7]]];

        let weights: Array2<f32> = Array::linspace(1., 1000., 1000)
            .into_shape([100, 10])
            .unwrap();
        let embedding_layer = EmbeddingLayer::new(weights);

        let result: Array4<f32> = embedding_layer.apply(&data);
        let expected: Array4<f32> = Array4::from_shape_vec(
            [2, 2, 3, 10],
            vec![
                11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
                25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0,
                39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0,
                53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0,
                67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0,
                81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0,
                95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
                18.0, 19.0, 20.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 71.0,
                72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0,
            ],
        )
        .unwrap();
        assert_eq!(expected, result);
    }

    #[test]
    #[should_panic]
    fn test_embedding_2d_panic() {
        let data: Array2<usize> = array![[1, 2, 3], [4, 5, 100], [7, 8, 9], [1, 4, 7]];

        let weights: Array2<f32> = Array::linspace(1., 1000., 1000)
            .into_shape([100, 10])
            .unwrap();
        let embedding_layer = EmbeddingLayer::new(weights);

        _ = embedding_layer.apply(&data);
    }

    #[test]
    #[should_panic]
    fn test_embedding_3d_panic() {
        let data: Array3<usize> = array![[[1, 2, 3], [4, 5, 100]], [[7, 8, 9], [1, 4, 7]]];

        let weights: Array2<f32> = Array::linspace(1., 1000., 1000)
            .into_shape([100, 10])
            .unwrap();
        let embedding_layer = EmbeddingLayer::new(weights);

        _ = embedding_layer.apply(&data);
    }
}
