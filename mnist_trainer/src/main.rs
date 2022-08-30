#![feature(generic_arg_infer)]

use std::{path::Path, time::Instant};

use rand::SeedableRng;
use shadl::dataset::augment::{AffineAugmenter, NoopAugmenter};
use shadl::*;
use shadl::{
    dataset::{self, Dataset},
    layer::{AffineLayer, Layer, SigmoidLayer, SoftmaxLayer},
    losses::{CrossEntropyLoss, Loss},
    matrix::FMatrix,
    optimizer::SimpleGradientDescent,
    serialization,
};

const MNIST_ROOT: &str = "./datasets/mnist";
const OUT_ROOT: &str = "./out";

fn main() {
    if !Path::new(OUT_ROOT).exists() {
        std::fs::create_dir_all(OUT_ROOT).unwrap();
    }

    const BATCH_SIZE: usize = 32;
    const BATCHES_PER_FOLD: usize = 180;
    const FOLD_ITEM_COUNT: usize = BATCHES_PER_FOLD * BATCH_SIZE;

    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(89);
    let ds = dataset::MNISTDataset::init(&MNIST_ROOT);
    let dl = dataset::Dataloader::<_, _, BATCH_SIZE>::new(&ds, &mut rng, &NoopAugmenter::new());

    let mut network;

    let file_name = format!("{}/continue", OUT_ROOT);
    if Path::new(&file_name).exists() {
        println!("Loading from file {}", file_name);
        network = serialization::load_from_file(&file_name);
    } else {
        network = shadl::nn![
            rng,
            AffineLayer<784, 512, BATCH_SIZE>,
            SigmoidLayer<512, BATCH_SIZE>,
            AffineLayer<512, 128, BATCH_SIZE>,
            SigmoidLayer<128, BATCH_SIZE>,
            AffineLayer<128, 32, BATCH_SIZE>,
            SigmoidLayer<32, BATCH_SIZE>,
            AffineLayer<32, 10, BATCH_SIZE>,
            SoftmaxLayer<10, BATCH_SIZE>
        ];
    }

    let folds_per_dataset = dl.batches_count() / BATCHES_PER_FOLD;

    println!("Total dataset items: {}", ds.get_count());
    println!(
        "Given batch size {}, the dataset contains {} batches ({} spare items)",
        BATCH_SIZE,
        dl.batches_count(),
        dl.spare_items_count()
    );
    println!(
        "Given {} batches per fold, you are performing {}-cross validation",
        BATCHES_PER_FOLD, folds_per_dataset
    );
    println!(
        "At each cross validation step, {}% of the dataset is used for training and {}% for validation",
        (((folds_per_dataset as f32) - 1.0) * (FOLD_ITEM_COUNT as f32) * 100.0) / (ds.get_count() as f32),
        (1.0 * (FOLD_ITEM_COUNT as f32) * 100.0) / (ds.get_count() as f32)
    );

    println!("Begin training:");

    let mut network_output = FMatrix::<10, BATCH_SIZE>::default();
    let mut network_input_gradient = FMatrix::<784, BATCH_SIZE>::default();
    let mut dout = FMatrix::<10, BATCH_SIZE>::default();
    let loss_fn = CrossEntropyLoss {};
    let mut opt = SimpleGradientDescent::default();
    let data_augmenter = AffineAugmenter::new(-3.0..3.0, -15.0..15.0, 0.9..1.1, 28, 28);

    for epoch in 0..5 {
        println!("Preparing DataLoader");
        let dl = dataset::Dataloader::<_, _, BATCH_SIZE>::new(&ds, &mut rng, &data_augmenter);
        println!(
            "========= STARTING EPOCH {} ==========================",
            epoch
        );
        for train_fold_idx in 0..folds_per_dataset {
            println!("Using fold {} for validation", train_fold_idx);

            let start = Instant::now();

            let mut train_loss = 0.0;

            for fold_idx in 0..folds_per_dataset {
                if fold_idx == train_fold_idx {
                    continue;
                }

                let fold_start_batch = fold_idx * BATCHES_PER_FOLD;
                let fold_end_batch = fold_idx * BATCHES_PER_FOLD + BATCHES_PER_FOLD;

                for batch_idx in fold_start_batch..fold_end_batch {
                    if batch_idx % 250 == 0 {
                        println!("\t Training on batch {}", batch_idx);
                    }

                    let (curr_b, curr_l) = dl.get_batch(batch_idx);

                    network.forward(curr_b, &mut network_output);
                    let batch_loss = loss_fn.loss(&network_output, curr_l);
                    loss_fn.backward(&network_output, curr_l, &mut dout);
                    network.backward(&dout, &mut network_input_gradient, &mut opt);

                    train_loss += batch_loss;
                }
            }

            let duration = Instant::now() - start;

            println!("Computing validation loss");

            let mut val_loss = 0.0;
            let val_fold_start_batch = train_fold_idx * BATCHES_PER_FOLD;
            let val_fold_end_batch = train_fold_idx * BATCHES_PER_FOLD + BATCHES_PER_FOLD;

            for val_batch_idx in val_fold_start_batch..val_fold_end_batch {
                let (curr_b, curr_l) = dl.get_batch(val_batch_idx);
                network.forward(curr_b, &mut network_output);
                let batch_loss = loss_fn.loss(&network_output, curr_l);

                val_loss += batch_loss;
            }

            let file_name = format!("{}/mnist-epoch-{}-tf-{}", OUT_ROOT, epoch, train_fold_idx);
            serialization::save_to_file(&file_name, &network);

            println!(
                "Done. Iteration took {:.3}s ({:.3} batch/s). Train loss: {}. Val loss: {}",
                duration.as_secs_f32(),
                (BATCHES_PER_FOLD as f32) / duration.as_secs_f32(),
                train_loss,
                val_loss
            );

            println!();

            assert!(!train_loss.is_nan());
        }
    }
}
