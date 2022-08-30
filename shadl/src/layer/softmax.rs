use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::matrix::{FMatrix, Matrix};

use super::Layer;

#[derive(Serialize, Deserialize, Debug)]
pub struct SoftmaxLayer<const R: usize, const C: usize> {
    output_cached: FMatrix<R, C>,
}

impl<const R: usize, const C: usize> Layer<FMatrix<R, C>> for SoftmaxLayer<R, C> {
    type Output = FMatrix<R, C>;

    fn new<RNG: Rng + ?Sized>(_rng: &mut RNG) -> Self {
        SoftmaxLayer {
            output_cached: FMatrix::<R, C>::default(),
        }
    }

    fn forward(&mut self, input: &FMatrix<R, C>, output: &mut Self::Output) {
        let mut max = FMatrix::<1, C>::default();

        for r in 0..R {
            for c in 0..C {
                max[(0, c)] = max[(0, c)].max(input[(r, c)]);
            }
        }

        let mut sum = FMatrix::<1, C>::default();

        for r in 0..R {
            for c in 0..C {
                sum[(0, c)] += (input[(r, c)] - max[(0, c)]).exp();
            }
        }

        for r in 0..R {
            for c in 0..C {
                output[(r, c)] = (input[(r, c)] - max[(0, c)]).exp() / sum[(0, c)];
            }
        }

        self.output_cached.copy_from(output);
    }

    fn backward<Opt: crate::optimizer::Optimizer>(
        &mut self,
        gradient: &Self::Output,
        inputs_gradient: &mut FMatrix<R, C>,
        _optimizer: &mut Opt,
    ) {
        // TODO. This backwards computation relies on the fact that
        // the loss function is multiclass cross entropy and on the fact
        // that the "true output" y that is fed to the loss function is
        // one-hot encoded. This invariant should be enforced in the typing system.
        let mut n_stars = Matrix::<usize, 1, C>::default();
        for c in 0..C {
            for r in 0..R {
                if gradient[(r, c)] != 0.0 {
                    n_stars[(0, c)] = r;
                }
            }
        }

        for p in 0..R {
            for q in 0..C {
                let n_star = n_stars[(0, q)];
                let s = &self.output_cached;
                let d = if n_star == p { 1.0 } else { 0.0 };
                inputs_gradient[(p, q)] = gradient[(n_star, q)] * s[(n_star, q)] * (d - s[(p, q)]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SoftmaxLayer;
    use crate::{
        layer::Layer,
        matrix::FMatrix,
        test_utils::{
            max_rdp, numerical_diff, random_fmatrix, random_fmatrix2, small_random_fmatrix,
            TestOptimizer,
        },
    };
    use proptest::prelude::*;
    use rand::SeedableRng;
    use test::Bencher;

    #[test]
    fn softmax_layer_works() {
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(45);

        let x = FMatrix::from([[1.0, 2.0], [-1.0, -2.0], [0.1, -0.1]]);

        let mut forward_out = FMatrix::<3, 2>::default();
        let mut ll = SoftmaxLayer::<3, 2>::new(&mut rng);

        ll.forward(&x, &mut forward_out);
        let expected_forward_out = FMatrix::<3, 2>::from([
            [0.6485484, 0.87659925],
            [0.08777148, 0.016055476],
            [0.2636801, 0.10734522],
        ]);
        assert_eq!(forward_out, expected_forward_out);

        let dout = FMatrix::<3, 2>::from([[4.2, 5.2], [2.1, 4.1], [1.5, 7.3]]);

        let mut dx_computed = FMatrix::<3, 2>::default();
        let mut opt = TestOptimizer::new_noop();

        ll.backward(&dout, &mut dx_computed, &mut opt);

        let dx_correct = FMatrix::from([
            [-0.25651398, -0.6869209],
            [-0.03471539, -0.012581395],
            [0.29122937, 0.6995023],
        ]);
        assert_eq!(dx_computed, dx_correct);
    }

    #[bench]
    fn bench_softmax_forward(bench: &mut Bencher) {
        const R: usize = 2000;
        const C: usize = 32;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(89);
        let a = random_fmatrix2::<_, R, C>(&mut rng);
        let mut b = FMatrix::<R, C>::default();

        let mut layer = SoftmaxLayer::new(&mut rng);

        bench.iter(|| {
            layer.forward(&a, &mut b);
        });
    }

    #[bench]
    fn bench_softmax_backward(bench: &mut Bencher) {
        const R: usize = 2000;
        const C: usize = 32;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(89);
        let a = random_fmatrix2::<_, R, C>(&mut rng);
        let mut b = random_fmatrix2::<_, R, C>(&mut rng);

        let mut layer = SoftmaxLayer::new(&mut rng);
        let mut opt = TestOptimizer::new_noop();

        bench.iter(|| {
            layer.backward(&a, &mut b, &mut opt);
        });
    }

    const RNG_SEED: u64 = 25;
    const T_XI: usize = 8;
    const T_BATCH_SIZE: usize = 16;

    proptest! {

        #[test]
        fn layer_evaluated_on_single_sample_works(mut input in random_fmatrix::<T_XI, T_BATCH_SIZE>()) {
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(RNG_SEED);
            let mut ll = SoftmaxLayer::<T_XI, T_BATCH_SIZE>::new(&mut rng);

            let mut out = FMatrix::<T_XI, T_BATCH_SIZE>::default();
            let mut out_zeros = FMatrix::<T_XI, T_BATCH_SIZE>::default();

            ll.forward(&input, &mut out);

            for i in 0..T_XI {
                for j in 1..T_BATCH_SIZE {
                    input[(i, j)] = 0.0;
                }
            }

            ll.forward(&input, &mut out_zeros);

            for j in 0..T_XI {
                prop_assert_eq!(out_zeros[(j, 0)], out[(j, 0)]);
            }
        }

        #[test]
        fn dinput_numerically_correct(
            mut input in small_random_fmatrix::<T_XI, T_BATCH_SIZE>()
        ){
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(RNG_SEED);
            let mut ll = SoftmaxLayer::<T_XI, T_BATCH_SIZE>::new(&mut rng);

            let mut dout = FMatrix::<T_XI, T_BATCH_SIZE>::default();
            for c in 0..T_BATCH_SIZE {
                let random_row: usize = rng.gen_range(0..T_XI);
                dout[(random_row, c)] = rng.gen_range(-100.0..100.0);
            }

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

            prop_assert!(max_rdp(&dinput, &dinput_num) < 0.01);
        }

    }
}
