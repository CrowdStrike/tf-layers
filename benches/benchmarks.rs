#![allow(clippy::cast_precision_loss)]
#![allow(clippy::wildcard_imports)]

#[macro_use]
extern crate lazy_static;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::*;
use tensorflow_layers::*;

lazy_static! {
    static ref INPUT3D: Array3<f32> =
        Array::from_shape_fn((10, 25, 4), |(x, y, z)| (x + y + z) as f32);
    static ref INPUT3D_USIZE: Array3<usize> =
        Array::from_shape_fn((10, 25, 4), |(x, y, z)| x + y + z);
    static ref INPUT2D: Array2<f32> = Array::from_shape_fn((250, 4), |(x, y)| (x + y) as f32);
    static ref INPUT2D_USIZE: Array2<usize> = Array::from_shape_fn((250, 4), |(x, y)| x + y);
    static ref INPUT1D: Array1<f32> = Array::from_shape_fn(1000, |i| i as f32);
    static ref INPUT1D_USIZE: Array1<usize> = Array::from_shape_fn(1000, |i| i);
}

fn bench_2d_dense_layer(c: &mut Criterion) {
    let weights = Array2::from_elem([4, 19], 2.7);
    let bias = Array1::from_elem([19], 2.7);
    let activation = Activation::Linear;
    let dense_layer = DenseLayer::new(weights, bias, activation);

    c.bench_function("2d_dense_layer", |b| {
        b.iter(|| dense_layer.apply2d(black_box(&INPUT2D)));
    });
}

fn bench_3d_dense_layer(c: &mut Criterion) {
    let weights = Array2::from_elem([4, 19], 2.7);
    let bias = Array1::from_elem([19], 2.7);
    let activation = Activation::Linear;
    let dense_layer = DenseLayer::new(weights, bias, activation);

    c.bench_function("3d_dense_layer", |b| {
        b.iter(|| dense_layer.apply3d(black_box(&INPUT3D)));
    });
}

fn bench_conv1d_layer(c: &mut Criterion) {
    let conv1d_layer = Conv1DLayer::new(
        Array3::from_elem([10, 3, 4], 2.7),
        Array1::from_elem([10], -1.5),
        1,
        vec![(1, 2), (2, 3), (0, 0)],
        0,
        0,
        Activation::Linear,
    );

    c.bench_function("conv1d_layer", |b| {
        b.iter(|| conv1d_layer.apply(black_box(&INPUT3D)));
    });
}

fn bench_dropout_layer(c: &mut Criterion) {
    let dropout_layer = Dropout::new(0.2);

    c.bench_function("dropout_layer", |b| {
        b.iter(|| dropout_layer.apply(black_box(&INPUT2D)));
    });
}

fn bench_1d_embedding_layer(c: &mut Criterion) {
    let weights: Array2<f32> = Array::linspace(1., 1000., 1000)
        .into_shape([1000, 1])
        .unwrap();
    let embedding_layer = EmbeddingLayer::new(weights);

    c.bench_function("1d_embedding_layer", |b| {
        b.iter(|| embedding_layer.apply(black_box(&INPUT1D_USIZE)));
    });
}

fn bench_2d_embedding_layer(c: &mut Criterion) {
    let weights: Array2<f32> = Array::linspace(1., 1000., 1000)
        .into_shape([500, 2])
        .unwrap();
    let embedding_layer = EmbeddingLayer::new(weights);

    c.bench_function("2d_embedding_layer", |b| {
        b.iter(|| embedding_layer.apply(black_box(&INPUT2D_USIZE)));
    });
}

fn bench_3d_embedding_layer(c: &mut Criterion) {
    let weights: Array2<f32> = Array::linspace(1., 1000., 1000)
        .into_shape([500, 2])
        .unwrap();
    let embedding_layer = EmbeddingLayer::new(weights);

    c.bench_function("3d_embedding_layer", |b| {
        b.iter(|| embedding_layer.apply(black_box(&INPUT3D_USIZE)));
    });
}

fn bench_1d_padding_layer(c: &mut Criterion) {
    c.bench_function("1d_padding_layer", |b| {
        b.iter(|| padding(black_box(&INPUT1D_USIZE), &[(3, 2)]));
    });
}

fn bench_2d_padding_layer(c: &mut Criterion) {
    c.bench_function("2d_padding_layer", |b| {
        b.iter(|| padding(black_box(&INPUT2D_USIZE), &[(2, 3), (2, 3)]));
    });
}

fn bench_3d_padding_layer(c: &mut Criterion) {
    c.bench_function("3d_padding_layer", |b| {
        b.iter(|| padding(black_box(&INPUT3D_USIZE), &[(2, 1), (2, 3), (0, 1)]));
    });
}

fn bench_1d_max_pooling1d(c: &mut Criterion) {
    let pooling_layer = MaxPooling1DLayer::new(3, 2, vec![(3, 2)]);
    c.bench_function("1d_max_pooling1d", |b| {
        b.iter(|| pooling_layer.apply(black_box(&INPUT1D)));
    });
}

fn bench_2d_max_pooling1d(c: &mut Criterion) {
    let pooling_layer = MaxPooling1DLayer::new(3, 2, vec![(2, 3), (2, 3)]);
    c.bench_function("2d_max_pooling1d", |b| {
        b.iter(|| pooling_layer.apply(black_box(&INPUT2D)));
    });
}

fn bench_3d_max_pooling1d(c: &mut Criterion) {
    let pooling_layer = MaxPooling1DLayer::new(3, 2, vec![(2, 1), (2, 3), (0, 1)]);
    c.bench_function("3d_max_pooling1d", |b| {
        b.iter(|| pooling_layer.apply(black_box(&INPUT3D)));
    });
}

fn bench_1d_avg_pooling1d(c: &mut Criterion) {
    let pooling_layer = AveragePooling1DLayer::new(3, 2, vec![(2, 5)]);
    c.bench_function("1d_avg_pooling1d", |b| {
        b.iter(|| pooling_layer.apply(black_box(&INPUT1D)));
    });
}

fn bench_2d_avg_pooling1d(c: &mut Criterion) {
    let pooling_layer = AveragePooling1DLayer::new(3, 2, vec![(2, 3), (2, 3)]);
    c.bench_function("2d_avg_pooling1d", |b| {
        b.iter(|| pooling_layer.apply(black_box(&INPUT2D)));
    });
}

fn bench_3d_avg_pooling1d(c: &mut Criterion) {
    let pooling_layer = AveragePooling1DLayer::new(3, 2, vec![(2, 1), (2, 3), (0, 1)]);
    c.bench_function("3d_avg_pooling1d", |b| {
        b.iter(|| pooling_layer.apply(black_box(&INPUT3D)));
    });
}

fn bench_1d_batch_normalization(c: &mut Criterion) {
    let gamma = Array::from_shape_vec(2, vec![2.25, 2.25]).unwrap();
    let epsilon = 0.0001;
    let batch_normalization_layer = BatchNormalization::new(
        gamma,
        INPUT1D.clone(),
        INPUT1D.clone(),
        INPUT1D.clone(),
        epsilon,
    );

    c.bench_function("1d_batch_normalization", |b| {
        b.iter(|| batch_normalization_layer.apply(black_box(&INPUT1D)));
    });
}

fn bench_2d_batch_normalization(c: &mut Criterion) {
    let gamma = Array::from_shape_vec(2, vec![2.25, 2.25]).unwrap();
    let beta = Array::from_shape_vec(4, vec![3.0, 4.0, 2.5, 3.5]).unwrap();
    let moving_mean = Array::from_shape_vec(4, vec![1.0, 1.5, -1.0, -1.5]).unwrap();
    let moving_variance = Array::from_shape_vec(4, vec![0.5, 0.7, 0.2, 0.7]).unwrap();
    let epsilon = 0.0001;

    let batch_normalization_layer =
        BatchNormalization::new(gamma, beta, moving_mean, moving_variance, epsilon);

    c.bench_function("2d_batch_normalization", |b| {
        b.iter(|| batch_normalization_layer.apply(black_box(&INPUT2D)));
    });
}

fn bench_3d_batch_normalization(c: &mut Criterion) {
    let gamma = Array::from_shape_vec(2, vec![2.25, 2.25]).unwrap();
    let beta = Array::from_shape_vec(4, vec![3.0, 4.0, 2.5, 3.5]).unwrap();
    let moving_mean = Array::from_shape_vec(4, vec![1.0, 1.5, -1.0, -1.5]).unwrap();
    let moving_variance = Array::from_shape_vec(4, vec![0.5, 0.7, 0.2, 0.7]).unwrap();
    let epsilon = 0.0001;

    let batch_normalization_layer =
        BatchNormalization::new(gamma, beta, moving_mean, moving_variance, epsilon);

    c.bench_function("3d_batch_normalization", |b| {
        b.iter(|| batch_normalization_layer.apply(black_box(&INPUT3D)));
    });
}

fn bench_1d_softmax_activation(c: &mut Criterion) {
    c.bench_function("1d_softmax_activation", |b| {
        b.iter(|| Activation::Softmax.activation(black_box(&INPUT1D)));
    });
}
fn bench_2d_softmax_activation(c: &mut Criterion) {
    c.bench_function("2d_softmax_activation", |b| {
        b.iter(|| Activation::Softmax.activation(black_box(&INPUT2D)));
    });
}
fn bench_3d_softmax_activation(c: &mut Criterion) {
    c.bench_function("3d_softmax_activation", |b| {
        b.iter(|| Activation::Softmax.activation(black_box(&INPUT3D)));
    });
}

criterion_group!(
    benches,
    bench_2d_dense_layer,
    bench_3d_dense_layer,
    bench_conv1d_layer,
    bench_dropout_layer,
    bench_1d_embedding_layer,
    bench_2d_embedding_layer,
    bench_3d_embedding_layer,
    bench_1d_padding_layer,
    bench_2d_padding_layer,
    bench_3d_padding_layer,
    bench_1d_max_pooling1d,
    bench_2d_max_pooling1d,
    bench_3d_max_pooling1d,
    bench_1d_avg_pooling1d,
    bench_2d_avg_pooling1d,
    bench_3d_avg_pooling1d,
    bench_1d_batch_normalization,
    bench_2d_batch_normalization,
    bench_3d_batch_normalization,
    bench_1d_softmax_activation,
    bench_2d_softmax_activation,
    bench_3d_softmax_activation,
);

criterion_main!(benches);
