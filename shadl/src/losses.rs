use crate::matrix::FMatrix;

pub trait Loss<I> {
    fn loss(&self, net_output: &I, labels: &I) -> f32;
    fn backward(&self, net_output: &I, labels: &I, net_output_gradient: &mut I);
}

pub struct MeanSquaredError {}

impl<const R: usize, const BATCH_SIZE: usize> Loss<FMatrix<R, BATCH_SIZE>> for MeanSquaredError {
    fn loss(&self, net_output: &FMatrix<R, BATCH_SIZE>, labels: &FMatrix<R, BATCH_SIZE>) -> f32 {
        let mut loss = 0.0;

        for b in 0..BATCH_SIZE {
            for r in 0..R {
                loss += (labels[(r, b)] - net_output[(r, b)]).powi(2);
            }
        }

        loss /= (BATCH_SIZE as f32) * (R as f32);
        loss
    }

    fn backward(
        &self,
        net_output: &FMatrix<R, BATCH_SIZE>,
        labels: &FMatrix<R, BATCH_SIZE>,
        net_output_gradient: &mut FMatrix<R, BATCH_SIZE>,
    ) {
        for b in 0..BATCH_SIZE {
            for r in 0..R {
                net_output_gradient[(r, b)] = (1.0 / ((BATCH_SIZE as f32) * (R as f32)))
                    * (labels[(r, b)] - net_output[(r, b)]);
            }
        }
    }
}

// TODO add tests for MSE loss

pub struct CrossEntropyLoss {}

impl<const R: usize, const BATCH_SIZE: usize> Loss<FMatrix<R, BATCH_SIZE>> for CrossEntropyLoss {
    fn loss(&self, net_output: &FMatrix<R, BATCH_SIZE>, labels: &FMatrix<R, BATCH_SIZE>) -> f32 {
        let mut loss = 0.0;

        for b in 0..BATCH_SIZE {
            for r in 0..R {
                loss += -labels[(r, b)] * net_output[(r, b)].ln();
            }
        }

        loss /= BATCH_SIZE as f32;
        loss
    }

    fn backward(
        &self,
        net_output: &FMatrix<R, BATCH_SIZE>,
        labels: &FMatrix<R, BATCH_SIZE>,
        net_output_gradient: &mut FMatrix<R, BATCH_SIZE>,
    ) {
        for r in 0..R {
            for b in 0..BATCH_SIZE {
                if labels[(r, b)] != 0.0 {
                    net_output_gradient[(r, b)] = -1.0 / (BATCH_SIZE as f32 * net_output[(r, b)]);
                } else {
                    net_output_gradient[(r, b)] = 0.0;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        losses::{CrossEntropyLoss, Loss},
        matrix::FMatrix,
    };

    #[test]
    fn cross_entropy_works() {
        let net_output =
            FMatrix::<4, 2>::from([[0.25, 0.01], [0.25, 0.01], [0.25, 0.01], [0.25, 0.96]]);
        let labels = FMatrix::<4, 2>::from([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0]]);
        let loss_fn = CrossEntropyLoss {};

        let l = loss_fn.loss(&net_output, &labels);
        let expected_l = 0.71355817782;

        assert_eq!(l, expected_l);

        let mut dx_computed = Default::default();
        loss_fn.backward(&net_output, &labels, &mut dx_computed);

        let dx_expected = FMatrix::from([[0., 0.], [0., 0.], [0., 0.], [-2., -0.5208334]]);

        assert_eq!(dx_computed, dx_expected);
    }
}
