//! Crate for Neural Network Layer Operations

#![deny(missing_docs)]
#![warn(
    deprecated_in_future,
    elided_lifetimes_in_paths,
    keyword_idents,
    missing_copy_implementations,
    missing_debug_implementations,
    non_ascii_idents,
    rustdoc::private_doc_tests,
    single_use_lifetimes,
    trivial_casts,
    trivial_numeric_casts,
    unreachable_pub,
    unused_extern_crates
)]
// clippy doesn't like that all our types below are named after the modules
// they're in, but those repeated names are not part of our public API (because
// the modules are private and we re-export the types from here). disable this
// lint.
#![allow(clippy::module_name_repetitions)]

mod activations;
mod average_pooling;
mod batch_normalization;
mod conv1d;
mod dense;
mod embedding;
mod max_pooling;
mod padding;

pub use activations::Activation;
pub use average_pooling::AveragePooling1DLayer;
pub use batch_normalization::BatchNormalization;
pub use conv1d::Conv1DLayer;
pub use dense::DenseLayer;
pub use embedding::EmbeddingLayer;
pub use max_pooling::MaxPooling1DLayer;
pub use padding::padding;
