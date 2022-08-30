use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{matrix::FMatrix, optimizer::Optimizer};

use super::Layer;

// XI = features per input
// M = outputs, neurons in the layer
#[derive(Serialize, Deserialize, Debug)]
pub struct AffineLayer<const XI: usize, const M: usize, const BATCH_SIZE: usize> {
    weights: FMatrix<M, XI>,
    biases: FMatrix<M, 1>,
    x_transposed_cache: FMatrix<BATCH_SIZE, XI>,
}

impl<const XI: usize, const M: usize, const BATCH_SIZE: usize> Layer<FMatrix<XI, BATCH_SIZE>>
    for AffineLayer<XI, M, BATCH_SIZE>
{
    type Output = FMatrix<M, BATCH_SIZE>;

    fn new<R: Rng + ?Sized>(rng: &mut R) -> Self {
        let mut ans = AffineLayer {
            weights: FMatrix::<M, XI>::default(),
            biases: FMatrix::<M, 1>::default(),
            x_transposed_cache: FMatrix::<BATCH_SIZE, XI>::default(),
        };

        for r in 0..M {
            for c in 0..XI {
                ans.weights[(r, c)] = rng.gen_range(-1.0..=1.0);
            }
        }

        for r in 0..M {
            ans.biases[(r, 0)] = rng.gen_range(-1.0..=1.0);
        }

        ans
    }

    fn forward(&mut self, input: &FMatrix<XI, BATCH_SIZE>, output: &mut Self::Output) {
        input.transpose_mut(&mut self.x_transposed_cache);

        self.weights.mul_mut(input, output);

        output.add_column_ip(&self.biases);
    }

    fn backward<Opt: Optimizer>(
        &mut self,
        gradient: &Self::Output,
        inputs_gradient: &mut FMatrix<XI, BATCH_SIZE>,
        optimizer: &mut Opt,
    ) {
        // Compute dX
        // gradient
        //     .transpose()
        //     .mul(&self.weights)
        //     .transpose_mut(inputs_gradient);

        self.weights.transpose().mul_mut(gradient, inputs_gradient);

        let mut d_biases = FMatrix::<M, 1>::default();
        gradient.sum_rows(&mut d_biases);

        let d_weights = gradient.mul(&self.x_transposed_cache);

        // Although these two are part of the symbolically computed
        // derivative, they are a constant factor which can be
        // ignored here and incorporated inside the learning rate.

        //d_biases.scalar_mul_ip(1.0 / (BATCH_SIZE as f32));
        //d_weights.scalar_mul_ip(1.0 / (BATCH_SIZE as f32));

        optimizer.update_weights(&mut self.biases, d_biases);
        optimizer.update_weights(&mut self.weights, d_weights)
    }
}

#[cfg(test)]
mod tests {
    use std::any::Any;

    use super::{AffineLayer, Layer};
    use crate::{matrix::FMatrix, test_utils::*};
    use proptest::prelude::*;
    use rand::SeedableRng;
    use test::Bencher;

    #[test]
    fn affine_layer_works() {
        const BS: usize = 2;
        let x = FMatrix::from([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]);
        let w = FMatrix::from([[9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]);
        let b = FMatrix::from([[17.0], [18.0]]);

        let correct_out = FMatrix::from([[195.0, 237.0], [260.0, 318.0]]);

        let mut out = FMatrix::<2, BS>::default();

        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(52);

        let mut ll = AffineLayer::<4, 2, BS>::new(&mut rng);
        ll.weights = w;
        ll.biases = b;
        ll.forward(&x, &mut out);

        assert_eq!(out, correct_out);

        let dout = FMatrix::<2, BS>::from([[453.2, 584.2], [234.1, 482.1]]);

        let dx_correct = FMatrix::from([
            [7122.1, 11525.101],
            [7809.4004, 12591.4],
            [8496.7, 13657.7],
            [9184.0, 14724.0],
        ]);

        let mut dw_correct = FMatrix::from([
            [810.80005, 1848.2001, 2885.60, 3923.00],
            [599.15, 1315.3501, 2031.55, 2747.75],
        ]);
        dw_correct.scalar_mul_ip(BS as f32);

        let mut db_correct = FMatrix::from([[518.7], [358.1]]);
        db_correct.scalar_mul_ip(BS as f32);

        let mut dx_computed = FMatrix::<4, BS>::default();
        let opt_fn = |_w: &mut dyn Any, d: &mut dyn Any| {
            if let Some(db) = d.downcast_ref::<FMatrix<2, 1>>() {
                assert_eq!(db, &db_correct);
            }

            if let Some(dw) = d.downcast_ref::<FMatrix<2, 4>>() {
                assert_eq!(dw, &dw_correct);
            }
        };
        let mut opt = TestOptimizer::new(&opt_fn);

        ll.backward(&dout, &mut dx_computed, &mut opt);

        assert_eq!(dx_computed, dx_correct);
    }

    #[bench]
    fn bench_affine_forward(bench: &mut Bencher) {
        const R: usize = 2000;
        const C: usize = 32;
        const O: usize = 1500;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(89);
        let a = random_fmatrix2::<_, R, C>(&mut rng);
        let mut b = FMatrix::<O, 32>::default();

        let mut layer = AffineLayer::new(&mut rng);

        bench.iter(|| {
            layer.forward(&a, &mut b);
        });
    }

    #[bench]
    fn bench_affine_backward(bench: &mut Bencher) {
        const R: usize = 2000;
        const C: usize = 32;
        const O: usize = 1500;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(89);
        let a = random_fmatrix2::<_, R, C>(&mut rng);
        let mut b = random_fmatrix2::<_, O, 32>(&mut rng);

        let mut layer = AffineLayer::new(&mut rng);
        let mut opt = TestOptimizer::new_noop();

        bench.iter(|| {
            layer.backward(&a, &mut b, &mut opt);
        });
    }

    const RNG_SEED: u64 = 51;
    const T_BATCH_SIZE: usize = 16;
    const T_XI: usize = 16;
    const T_M: usize = 4;

    //const T_BATCH_SIZE: usize = 2;
    //const T_XI: usize = 3;
    //const T_M: usize = 2;

    proptest! {
        #[test]
        fn layer_evaluated_on_single_sample_works(mut input in random_fmatrix::<T_XI, T_BATCH_SIZE>()) {
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(RNG_SEED);
            let mut ll = AffineLayer::<T_XI, T_M, T_BATCH_SIZE>::new(&mut rng);

            let mut out = FMatrix::<T_M, T_BATCH_SIZE>::default();
            let mut out_zeros = FMatrix::<T_M, T_BATCH_SIZE>::default();

            ll.forward(&input, &mut out);

            for i in 0..T_XI {
                for j in 1..T_BATCH_SIZE {
                    input[(i, j)] = 0.0;
                }
            }

            ll.forward(&input, &mut out_zeros);

            for j in 0..T_M {
                prop_assert_eq!(out_zeros[(j, 0)], out[(j, 0)]);
            }
        }

        #[test]
        fn dinput_numerically_correct(
            mut input in small_random_fmatrix::<T_XI, T_BATCH_SIZE>(),
            dout in small_random_fmatrix::<T_M, T_BATCH_SIZE>()
        ){
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(RNG_SEED);
            let mut ll = AffineLayer::<T_XI, T_M, T_BATCH_SIZE>::new(&mut rng);

            let dinput_num = numerical_diff(&mut ll, &mut input, &dout, |layer, curr_i| {
                let mut out = FMatrix::default();
                layer.forward(&curr_i, &mut out);
                out
            });

            let mut dinput = FMatrix::default();
            let mut opt = TestOptimizer::new_noop();
            let mut useless = Default::default();
            ll.forward(&input, &mut useless);
            ll.backward(&dout, &mut dinput, &mut opt);

            //println!("max = {:.3?}, avg = {:.3?}", max_rdp(&dinput, &dinput_num), avg_rdp(&dinput, &dinput_num));
            prop_assert!(max_rdp(&dinput, &dinput_num) < 0.2);
            prop_assert!(avg_rdp(&dinput, &dinput_num) < 0.01);
        }

        #[test]
        fn dweights_numerically_correct(
            input in small_random_fmatrix::<T_XI, T_BATCH_SIZE>(),
            mut weights in small_random_fmatrix::<T_M, T_XI>(),
            dout in small_random_fmatrix::<T_M, T_BATCH_SIZE>()
        ){
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(RNG_SEED);
            let mut ll = AffineLayer::<T_XI, T_M, T_BATCH_SIZE>::new(&mut rng);

            let dw_num = numerical_diff(&mut ll, &mut weights, &dout, |layer, curr_i| {
                let mut out = FMatrix::default();
                layer.weights = curr_i.clone();
                layer.forward(&input, &mut out);
                out
            });

            let opt_fn = |_w: &mut dyn Any, dw: &mut dyn Any| {
                if let Some(dw) = dw.downcast_ref::<FMatrix<T_M, T_XI>>() {
                    assert!(max_rdp(&dw, &dw_num) < 0.01);
                }
            };
            let mut opt = TestOptimizer::new(&opt_fn);

            let mut useless = Default::default();
            let mut useless2 = Default::default();
            ll.forward(&input, &mut useless);
            ll.backward(&dout, &mut useless2, &mut opt);
        }

        #[test]
        fn dbiases_numerically_correct(
            input in small_random_fmatrix::<T_XI, T_BATCH_SIZE>(),
            mut biases in small_random_fmatrix::<T_M, 1>(),
            dout in small_random_fmatrix::<T_M, T_BATCH_SIZE>()
        ){
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(RNG_SEED);
            let mut ll = AffineLayer::<T_XI, T_M, T_BATCH_SIZE>::new(&mut rng);

            let db_num = numerical_diff(&mut ll, &mut biases, &dout, |layer, curr_i| {
                let mut out = FMatrix::default();
                layer.biases = curr_i.clone();
                layer.forward(&input, &mut out);
                out
            });

            let opt_fn = |_b: &mut dyn Any, db: &mut dyn Any| {
                if let Some(db) = db.downcast_ref::<FMatrix<T_M, 1>>() {
                    assert!(max_rdp(&db, &db_num) < 0.01);
                }
            };
            let mut opt = TestOptimizer::new(&opt_fn);

            let mut useless = Default::default();
            let mut useless2 = Default::default();
            ll.forward(&input, &mut useless);
            ll.backward(&dout, &mut useless2, &mut opt);
        }
    }
}
