use std::path::Path;

use crate::{dataset::utils::download_and_decompress, matrix::FMatrix};
use reqwest::{self, Url};

use super::Dataset;

pub struct MNISTDataset<'b> {
    root_folder: &'b Path,
    train_images: Vec<FMatrix<784, 1>>,
    train_labels: Vec<FMatrix<10, 1>>,
}

impl<'c> Dataset<'c, 784, 10> for MNISTDataset<'c> {
    fn init<P: AsRef<Path>>(root_folder: &'c P) -> Self {
        let mut ds = MNISTDataset {
            root_folder: root_folder.as_ref(),
            train_images: Vec::default(),
            train_labels: Vec::default(),
        };

        let files = vec![
            "train-images-idx3-ubyte",
            "train-labels-idx1-ubyte",
            "t10k-images-idx3-ubyte",
            "t10k-labels-idx1-ubyte",
        ];

        if !ds.root_folder.exists() {
            std::fs::create_dir_all(ds.root_folder).unwrap();

            const MNIST_BASE_URL: &str = "http://yann.lecun.com/exdb/mnist/";

            for file_name in &files {
                let url_str = format!("{}{}.gz", MNIST_BASE_URL, file_name);
                let url = Url::parse(&url_str).unwrap();
                download_and_decompress(url, &ds.root_folder.join(file_name))
            }
        }

        let tr_imgs_path = ds.root_folder.join(files[0]);
        let train_imgs_bytes = std::fs::read(tr_imgs_path).unwrap();

        let magic_number_imgs = u32::from_be_bytes(train_imgs_bytes[0..4].try_into().unwrap());
        assert!(magic_number_imgs == 2051);

        let number_of_images = usize::try_from(u32::from_be_bytes(
            train_imgs_bytes[4..8].try_into().unwrap(),
        ))
        .unwrap();
        let image_rows = usize::try_from(u32::from_be_bytes(
            train_imgs_bytes[8..12].try_into().unwrap(),
        ))
        .unwrap();
        let image_cols = usize::try_from(u32::from_be_bytes(
            train_imgs_bytes[12..16].try_into().unwrap(),
        ))
        .unwrap();

        let image_pixel_count = image_rows * image_cols;

        for i in 0..number_of_images {
            let current_image_start = 16 + i * image_pixel_count;
            let current_image_indices =
                current_image_start..current_image_start + image_pixel_count;

            let mut current_image = FMatrix::<784, 1>::default();
            let current_image_bytes = &train_imgs_bytes[current_image_indices];
            for i in 0..current_image_bytes.len() {
                current_image[(i, 0)] = (current_image_bytes[i] as f32) / 255.0;
            }

            ds.train_images.push(current_image);
        }

        let tr_lbls_path = ds.root_folder.join(files[1]);
        let train_lbls_bytes = std::fs::read(tr_lbls_path).unwrap();

        let magic_number_lbls = u32::from_be_bytes(train_lbls_bytes[0..4].try_into().unwrap());
        assert!(magic_number_lbls == 2049);

        let number_of_labels = usize::try_from(u32::from_be_bytes(
            train_lbls_bytes[4..8].try_into().unwrap(),
        ))
        .unwrap();
        assert!(number_of_images == number_of_labels);
        for i in 0..number_of_labels {
            let current_lbl =
                u8::from_be_bytes(train_lbls_bytes[i + 8..i + 8 + 1].try_into().unwrap());
            let mut current_lbl_one_hot = FMatrix::<10, 1>::default();
            current_lbl_one_hot[(current_lbl as usize, 0)] = 1.0;
            ds.train_labels.push(current_lbl_one_hot);
        }

        assert!(ds.train_images.len() == ds.train_labels.len());

        ds
    }

    fn get_count(&self) -> usize {
        self.train_images.len()
    }

    fn get_item(&'c self, index: usize) -> (&'c FMatrix<784, 1>, &'c FMatrix<10, 1>) {
        let img = self.train_images.get(index).unwrap();
        let lbl = self.train_labels.get(index).unwrap();

        (img, lbl)
    }
}
