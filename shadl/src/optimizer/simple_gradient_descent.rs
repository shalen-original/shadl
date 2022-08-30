use crate::matrix::FMatrix;

use super::Optimizer;

pub struct SimpleGradientDescent {
    learning_rate: f32,
}

impl Default for SimpleGradientDescent {
    fn default() -> Self {
        SimpleGradientDescent::new(10e-2)
    }
}

impl SimpleGradientDescent {
    fn new(learning_rate: f32) -> Self {
        SimpleGradientDescent { learning_rate }
    }
}

impl Optimizer for SimpleGradientDescent {
    fn update_weights<const R: usize, const C: usize>(
        &mut self,
        weights: &mut FMatrix<R, C>,
        mut dweights: FMatrix<R, C>,
    ) {
        dweights.scalar_mul_ip(-self.learning_rate);
        weights.add_ip(dweights);
    }
}

#[cfg(test)]
mod tests {
    use crate::matrix::FMatrix;

    use super::{Optimizer, SimpleGradientDescent};

    #[test]
    fn sgd_works() {
        let mut opt = SimpleGradientDescent::new(10.0);

        let mut a = FMatrix::from([[1.0], [3.0]]);

        let b = FMatrix::from([[5.6], [6.2]]);

        opt.update_weights(&mut a, b);

        assert_eq!(a[(0, 0)], -55.0);
        assert_eq!(a[(1, 0)], -59.0);
    }
}
