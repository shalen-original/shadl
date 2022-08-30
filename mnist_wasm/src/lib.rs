use shadl::NeuralNetwork;

use rand::SeedableRng;
use shadl::{
    layer::{AffineLayer, Layer, SigmoidLayer, SoftmaxLayer},
    matrix::FMatrix,
    serialization,
};
use wasm_bindgen::prelude::*;

#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

pub fn set_panic_hook() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

fn normalize_color(input: u8) -> f32 {
    (input as f32) / 255.0
}

const INPUT_SIZE_LENGTH: usize = 784;
const OUTPUT_SIZE_LENGTH: usize = 10;
const BATCH_SIZE: usize = 32;

#[wasm_bindgen]
#[repr(C)]
pub struct MnistNetwork {
    input: Vec<u8>,
    output: Vec<f32>,
    network:
        Box<dyn Layer<FMatrix<INPUT_SIZE_LENGTH, BATCH_SIZE>, Output = FMatrix<10, BATCH_SIZE>>>,
}

#[wasm_bindgen]
impl MnistNetwork {
    pub fn new() -> MnistNetwork {
        set_panic_hook();

        let network_bytes = include_bytes!("../../out/trained-model-sigmoid-da");
        let network = if true {
            serialization::load_from_bytes(network_bytes)
        } else {
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(89);
            shadl::nn![
                rng,
                AffineLayer<INPUT_SIZE_LENGTH, 512, BATCH_SIZE>,
                SigmoidLayer<512, BATCH_SIZE>,
                AffineLayer<512, 128, BATCH_SIZE>,
                SigmoidLayer<128, BATCH_SIZE>,
                AffineLayer<128, 32, BATCH_SIZE>,
                SigmoidLayer<32, BATCH_SIZE>,
                AffineLayer<32, 10, BATCH_SIZE>,
                SoftmaxLayer<OUTPUT_SIZE_LENGTH, BATCH_SIZE>
            ]
        };

        MnistNetwork {
            input: vec![0; INPUT_SIZE_LENGTH],
            output: vec![0.0; OUTPUT_SIZE_LENGTH],
            network: Box::new(network),
        }
    }

    pub fn input_array_begin(&mut self) -> *mut u8 {
        self.input.as_mut_ptr()
    }

    pub fn output_array_begin(&mut self) -> *mut f32 {
        self.output.as_mut_ptr()
    }

    pub fn forward(&mut self) {
        let mut input_batch = FMatrix::<INPUT_SIZE_LENGTH, BATCH_SIZE>::default();

        for r in 0..INPUT_SIZE_LENGTH {
            input_batch[(r, 0)] = normalize_color(self.input[r]);
        }

        let mut output_batch = FMatrix::<OUTPUT_SIZE_LENGTH, BATCH_SIZE>::default();
        self.network
            .as_mut()
            .forward(&input_batch, &mut output_batch);

        for r in 0..OUTPUT_SIZE_LENGTH {
            self.output[r] = output_batch[(r, 0)];
        }
    }
}
