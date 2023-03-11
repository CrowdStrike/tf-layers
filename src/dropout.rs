use ndarray::{Array2};
use serde::{Deserialize, Serialize};
use ndarray_rand::rand;

/// The implementation of Dropout layer
#[derive(Serialize, Deserialize, Debug, Clone,Copy)]
pub struct Dropout{
    /// Between 0 an 1
    rate:f32,
}

impl Dropout {
    /// This creates a new Dropout layer with the rate set
    #[must_use]
    pub fn new(rate:f32) -> Dropout{
        Dropout { rate: rate}
    }

    /// Randomly sets the input units to 0 with a frequency of rate
    /// Input not set to 0 are scaled up by 1/(1-rate) such that the sum is unchanged
    #[must_use]
    pub fn apply(&self, data: &Array2<f32>) -> Array2<f32>{
        let mut data: Array2<f32> = data.clone();

        let nb_dropouts = (data.len() as f32 * self.rate) as usize;

        if self.rate == 1 as f32 {
            return Array2::zeros(data.dim())
        }else{
            let mut rng = rand::thread_rng();
            let idx_to_drop = rand::seq::index::sample(&mut rng, data.len(), nb_dropouts).into_vec();
            for (x,y) in data.iter_mut().enumerate(){
                if idx_to_drop.contains(&x){
                    *y = 0.0;
                }
                else{
                    *y *= 1.0 / (1.0 - self.rate);
                }
            }
        }
        data
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn test_dropout_simple(){
        let dims = (5,2);
        let data = Array2::from_shape_vec(dims, vec![1.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0]).unwrap();
        let dropout = Dropout::new(0.2);
        let result = dropout.apply(&data);
        let nb_od_zeros = result.iter().filter(|&&x| x == 0.0).count();
        assert_eq!(nb_od_zeros, 2);
    }

    #[test]
    fn test_one_rate(){
        let dims = (5,2);
        let data = Array2::from_shape_vec(dims, vec![0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0]).unwrap();
        let dropout = Dropout::new(1.0);

        let result = dropout.apply(&data);
        assert_eq!(result, Array2::zeros(dims));
    }

    #[test]
    fn test_zero_rate(){
        let dims = (5,2);
        let data = Array2::from_shape_vec(dims, vec![0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0]).unwrap();
        let dropout = Dropout::new(0.0);

        let result = dropout.apply(&data);
        assert_eq!(result, data);
    }
}