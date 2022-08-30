mod affine;
mod relu;
mod sigmoid;
mod softmax;

use crate::optimizer::Optimizer;
use rand::Rng;

// TODO, remove this trait. crate::matrix::Matrix should be a trait, not
// a struct, and Layer could probably be aware of Matrix.
pub trait Resettable {
    fn reset(&mut self);
}

pub trait Layer<Input> {
    type Output: Default + Resettable;

    // TODO: rethink Layer a bit better so it can more easily
    // be used as `dyn Layer`. This has currently been achieved
    // by the `where Self: Sized` type bounds, but especially for
    // the `backward` method this doesn't look like the best
    // solution

    fn new<R: Rng + ?Sized>(rng: &mut R) -> Self
    where
        Self: Sized;
    fn forward(&mut self, input: &Input, output: &mut Self::Output);
    fn backward<Opt: Optimizer>(
        &mut self,
        gradient: &Self::Output,
        inputs_gradient: &mut Input,
        optimizer: &mut Opt,
    ) where
        Self: Sized;
}

pub use affine::AffineLayer;
pub use relu::ReluLayer;
pub use sigmoid::SigmoidLayer;
pub use softmax::SoftmaxLayer;
