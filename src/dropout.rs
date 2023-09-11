use ndarray::{Array2, Zip};
use ndarray_rand::{rand, rand_distr::Bernoulli, RandomExt};
use serde::{Deserialize, Serialize};

/// The implementation of Dropout layer
#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct Dropout {
    /// Between 0 an 1
    rate: f32,
}

impl Dropout {
    /// This creates a new Dropout layer with the rate set
    ///
    /// # Panics
    /// If `rate` is not in 0.0 to 1.0 range
    #[must_use]
    pub fn new(rate: f32) -> Dropout {
        assert!((0.0..=1.0).contains(&rate));
        Dropout { rate }
    }

    /// Randomly sets the input units to 0 with a frequency of rate
    /// Input not set to 0 are scaled up by 1/(1-rate) such that the expected value is unchanged
    ///
    /// # Panics
    /// It could panic if `rate` is out of range, but the check is performed beforehand.
    #[must_use]
    pub fn apply(&self, data: &Array2<f32>) -> Array2<f32> {
        if (self.rate - 1.0).abs() < 0.000_001 {
            return Array2::zeros(data.dim());
        }

        if self.rate == 0_f32 {
            return data.clone();
        }

        let mut res: Array2<f32> = data.clone();
        let mut rng = rand::thread_rng();

        let mask2 = Array2::random_using(
            data.dim(),
            // This unwrap is safe as the code asserts range before.
            Bernoulli::new(f64::from(1.0 - self.rate)).unwrap(),
            &mut rng,
        );

        Zip::from(&mut res).and(&mask2).for_each(|x, mask| {
            if *mask {
                *x /= 1.0 - self.rate;
            } else {
                *x = 0_f32;
            }
        });

        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dropout_simple() {
        let dims = (5, 2);
        let data =
            Array2::from_shape_vec(dims, vec![1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
                .unwrap();
        let dropout = Dropout::new(0.2);
        let result = dropout.apply(&data);
        assert_ne!(data, result);
    }

    #[test]
    fn dropout_one_rate() {
        let dims = (5, 2);
        let data =
            Array2::from_shape_vec(dims, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
                .unwrap();
        let dropout = Dropout::new(1.0);

        let result = dropout.apply(&data);
        assert_eq!(result, Array2::zeros(dims));
    }

    #[test]
    fn dropout_zero_rate() {
        let dims = (5, 2);
        let data =
            Array2::from_shape_vec(dims, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
                .unwrap();
        let dropout = Dropout::new(0.0);

        let result = dropout.apply(&data);
        assert_eq!(result, data);
    }
}
