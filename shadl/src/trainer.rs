use rand::Rng;

use crate::{
    dataset::{Dataloader, Dataset},
    matrix::FMatrix,
    nn::NeuralNetwork,
};

pub struct KFoldTrainer<'a, DS, R, const XI: usize, const LI: usize, const BS: usize>
where
    DS: Dataset<'a, XI, LI>,
    R: Rng + ?Sized,
{
    dataset: &'a DS,
    dataloader: Dataloader<XI, LI, BS>,
    rng: &'a mut R,
    process_train_batch: &'a dyn Fn(&FMatrix<XI, BS>) -> f32,
}

impl<'b, DS, R, const XI: usize, const LI: usize, const BS: usize>
    KFoldTrainer<'b, DS, R, XI, LI, BS>
where
    DS: Dataset<'b, XI, LI>,
    R: Rng + ?Sized,
{
    fn new(ds: &'b DS, rng: &'b mut R) -> Self {
        let mut ans = KFoldTrainer {
            dataset: ds,
            dataloader: Dataloader::new(ds, rng),
            rng: rng,
        };

        ans
    }
}
