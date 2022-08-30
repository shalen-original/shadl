use std::path::Path;

use egui::ColorImage;
use rand::SeedableRng;
use shadl::*;
use shadl::{
    dataset::{self, Dataset},
    layer::{AffineLayer, Layer, SigmoidLayer, SoftmaxLayer},
    matrix::FMatrix,
    serialization,
};

struct MyApp<'a, T> {
    current_image: usize,
    ds: dataset::MNISTDataset<'a>,
    network: T,
}

const MNIST_ROOT: &str = "./datasets/mnist";

fn main() {
    let file_name = format!("./out/trained-model-sigmoid");
    const BATCH_SIZE: usize = 32;

    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(89);

    let network = if Path::new(&file_name).exists() {
        println!("RIGHT");
        serialization::load_from_file(&file_name)
    } else {
        shadl::nn![
            rng,
            AffineLayer<784, 512, BATCH_SIZE>,
            SigmoidLayer<512, BATCH_SIZE>,
            AffineLayer<512, 128, BATCH_SIZE>,
            SigmoidLayer<128, BATCH_SIZE>,
            AffineLayer<128, 32, BATCH_SIZE>,
            SigmoidLayer<32, BATCH_SIZE>,
            AffineLayer<32, 10, BATCH_SIZE>,
            SoftmaxLayer<10, BATCH_SIZE>
        ]
    };

    let app = MyApp {
        current_image: 0,
        ds: dataset::MNISTDataset::init(&MNIST_ROOT),
        network,
    };

    let options = eframe::NativeOptions {
        initial_window_size: Some(egui::vec2(550.0, 550.0)),
        ..Default::default()
    };
    eframe::run_native("Test", options, Box::new(|_cc| Box::new(app)));
}

fn from_one_hot<const R: usize>(v: &FMatrix<R, 1>) -> usize {
    for i in 0..R {
        if v[(i, 0)] == 1.0 {
            return i;
        }
    }
    panic!();
}

impl<T: Layer<FMatrix<784, 32>, Output = FMatrix<10, 32>>> eframe::App for MyApp<'_, T> {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("This is an image:");

            let (img, lbl) = self.ds.get_item(self.current_image);

            ui.horizontal(|ui| {
                if ui.add(egui::Button::new("Next image")).clicked() {
                    self.current_image += 1;
                }
            });

            ui.horizontal(|ui| {
                ui.add(egui::Label::new(format!(
                    "Current label: {:.2?} ({})",
                    lbl,
                    from_one_hot(lbl)
                )));
            });

            ui.horizontal(|ui| {
                let mut img_batch = FMatrix::<784, 32>::default();
                for i in 0..784 {
                    img_batch[(i, 0)] = img[(i, 0)];
                }

                let mut net_out = FMatrix::<10, 32>::default();
                self.network.forward(&img_batch, &mut net_out);

                let mut probs = FMatrix::<10, 1>::default();
                for i in 0..10 {
                    probs[(i, 0)] = net_out[(i, 0)];
                }

                ui.add(egui::Label::new(format!(
                    "Probs              : {:.2?}",
                    probs
                )));
            });

            let mut rgba = vec![0_u8; img.shape().0 * 4];

            // Testing data augmentation
            use shadl::dataset::augment::{AffineAugmenter, DataAugmenter};
            let mut rng = rand_chacha::ChaCha8Rng::from_entropy();
            let aug = AffineAugmenter::new(-3.0..3.0, -15.0..15.0, 0.9..1.1, 28, 28);
            let img = aug.augment(img, &mut rng);

            for i in 0..img.shape().0 {
                rgba[i * 4] = (img[(i, 0)] * 255.0).round() as u8;
                rgba[i * 4 + 1] = (img[(i, 0)] * 255.0).round() as u8;
                rgba[i * 4 + 2] = (img[(i, 0)] * 255.0).round() as u8;
                rgba[i * 4 + 3] = 255;
            }
            let image = ColorImage::from_rgba_unmultiplied([28, 28], &rgba);
            let texture = ui.ctx().load_texture("image", image);

            ui.image(&texture, ui.available_size());
        });
    }
}
