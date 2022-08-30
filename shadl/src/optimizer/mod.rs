mod simple_gradient_descent;

use crate::matrix::FMatrix;

pub trait Optimizer {
    fn update_weights<const R: usize, const C: usize>(
        &mut self,
        weights: &mut FMatrix<R, C>,
        dweights: FMatrix<R, C>,
    );
}

pub use simple_gradient_descent::SimpleGradientDescent;
