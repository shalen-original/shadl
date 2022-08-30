use std::path::Path;

use crate::matrix::FMatrix;
use rand::{prelude::SliceRandom, Rng};

#[cfg(feature = "easy_datasets")]
mod mnist;

#[cfg(feature = "easy_datasets")]
mod utils;

#[cfg(feature = "easy_datasets")]
pub use mnist::MNISTDataset;

use self::augment::DataAugmenter;

pub mod augment;

pub trait Dataset<'a, const XI: usize, const LI: usize> {
    fn init<P: AsRef<Path>>(root_folder: &'a P) -> Self;
    fn get_count(&self) -> usize;
    fn get_item(&'a self, index: usize) -> (&'a FMatrix<XI, 1>, &'a FMatrix<LI, 1>);
}

pub struct Dataloader<const XI: usize, const LI: usize, const BATCH_SIZE: usize> {
    data: Vec<FMatrix<XI, BATCH_SIZE>>,
    labels: Vec<FMatrix<LI, BATCH_SIZE>>,
    spare_items_count: usize,
}

impl<const XI: usize, const LI: usize, const BATCH_SIZE: usize> Dataloader<XI, LI, BATCH_SIZE> {
    pub fn new<'a, DS, R>(
        ds: &'a DS,
        rng: &mut R,
        augmenter: &impl DataAugmenter<FMatrix<XI, 1>, R>,
    ) -> Dataloader<XI, LI, BATCH_SIZE>
    where
        DS: Dataset<'a, XI, LI>,
        R: Rng + ?Sized,
    {
        let batch_in_ds = ds.get_count() / BATCH_SIZE;

        let mut dl = Dataloader {
            data: Default::default(),
            labels: Default::default(),
            spare_items_count: ds.get_count() - (batch_in_ds * BATCH_SIZE),
        };

        let mut random_idxs: Vec<usize> = (0..ds.get_count()).collect();
        random_idxs.shuffle(rng);

        for batch_idx in 0..batch_in_ds {
            let mut curr_batch_data: FMatrix<XI, BATCH_SIZE> = Default::default();
            let mut curr_batch_lbl: FMatrix<LI, BATCH_SIZE> = Default::default();

            for i in 0..BATCH_SIZE {
                let (curr_d, curr_lbl) = ds.get_item(random_idxs[batch_idx * BATCH_SIZE + i]);
                let augmented_curr_d = augmenter.augment(&curr_d, rng);

                for r in 0..XI {
                    curr_batch_data[(r, i)] = augmented_curr_d[(r, 0)];
                }

                for r in 0..LI {
                    curr_batch_lbl[(r, i)] = curr_lbl[(r, 0)];
                }
            }

            dl.data.push(curr_batch_data);
            dl.labels.push(curr_batch_lbl);
        }

        assert_eq!(dl.data.len(), dl.labels.len());
        assert_eq!(ds.get_count(), dl.data.len() * BATCH_SIZE);

        dl
    }

    pub fn batches_count(&self) -> usize {
        self.data.len()
    }

    pub fn spare_items_count(&self) -> usize {
        self.spare_items_count
    }

    pub fn get_batch(
        &self,
        batch_idx: usize,
    ) -> (&FMatrix<XI, BATCH_SIZE>, &FMatrix<LI, BATCH_SIZE>) {
        let d = &self.data[batch_idx];
        let l = &self.labels[batch_idx];

        (d, l)
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;

    use crate::{
        dataset::augment::{AffineAugmenter, NoopAugmenter},
        matrix::FMatrix,
    };

    use super::{Dataloader, Dataset};

    struct TestDataset {
        data: Vec<FMatrix<4, 1>>,
        lbls: Vec<FMatrix<2, 1>>,
    }

    impl<'a> Dataset<'a, 4, 2> for TestDataset {
        fn init<P: AsRef<std::path::Path>>(_root_folder: &'a P) -> Self {
            TestDataset {
                data: vec![
                    FMatrix::from([[1.0], [2.0], [3.0], [13.0]]),
                    FMatrix::from([[4.0], [5.0], [6.0], [14.0]]),
                    FMatrix::from([[7.0], [8.0], [9.0], [15.0]]),
                    FMatrix::from([[10.0], [11.0], [12.0], [16.0]]),
                ],
                lbls: vec![
                    FMatrix::from([[1.0], [2.0]]),
                    FMatrix::from([[3.0], [4.0]]),
                    FMatrix::from([[5.0], [6.0]]),
                    FMatrix::from([[7.0], [8.0]]),
                ],
            }
        }

        fn get_count(&self) -> usize {
            self.data.len()
        }

        fn get_item(&'a self, index: usize) -> (&'a FMatrix<4, 1>, &'a FMatrix<2, 1>) {
            (&self.data[index], &self.lbls[index])
        }
    }

    #[test]
    fn dataloader_works() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(89);

        let ds = TestDataset::init(&"unused");
        let dl = Dataloader::<_, _, 2>::new(&ds, &mut rng, &NoopAugmenter::new());

        assert_eq!(dl.batches_count(), 2);

        let (bd, bl) = dl.get_batch(1);

        let expected_bd = FMatrix::from([[4.0, 10.0], [5.0, 11.0], [6.0, 12.0], [14.0, 16.0]]);

        let expected_bl = FMatrix::from([[3.0, 7.0], [4.0, 8.0]]);

        assert_eq!(bd, &expected_bd);
        assert_eq!(bl, &expected_bl);
    }

    #[test]
    fn dataloader_data_augmentation_changes_data() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(89);

        let ds = TestDataset::init(&"unused");
        let aug = AffineAugmenter::new(-5.0..5.0, -40.0..40.0, 0.9..1.1, 2, 2);
        let dl = Dataloader::<_, _, 2>::new(&ds, &mut rng, &aug);

        assert_eq!(dl.batches_count(), 2);

        let (bd, bl) = dl.get_batch(1);

        let expected_bd_without_aug =
            FMatrix::from([[4.0, 10.0], [5.0, 11.0], [6.0, 12.0], [14.0, 16.0]]);

        let expected_bl = FMatrix::from([[3.0, 7.0], [4.0, 8.0]]);

        assert_ne!(bd, &expected_bd_without_aug);
        assert_eq!(bl, &expected_bl);
    }
}
