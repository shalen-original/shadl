use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::matrix::FMatrix;

use super::Layer;

#[derive(Serialize, Deserialize, Debug)]
pub struct SigmoidLayer<const R: usize, const C: usize> {
    output_cached: FMatrix<R, C>,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl<const R: usize, const C: usize> Layer<FMatrix<R, C>> for SigmoidLayer<R, C> {
    type Output = FMatrix<R, C>;

    fn new<RNG: Rng + ?Sized>(_rng: &mut RNG) -> Self {
        SigmoidLayer {
            output_cached: FMatrix::<R, C>::default(),
        }
    }

    fn forward(&mut self, input: &FMatrix<R, C>, output: &mut Self::Output) {
        for r in 0..R {
            for c in 0..C {
                output[(r, c)] = sigmoid(input[(r, c)]);
                self.output_cached[(r, c)] = output[(r, c)];
            }
        }
    }

    fn backward<Opt: crate::optimizer::Optimizer>(
        &mut self,
        gradient: &Self::Output,
        inputs_gradient: &mut FMatrix<R, C>,
        _optimizer: &mut Opt,
    ) {
        for r in 0..R {
            for c in 0..C {
                inputs_gradient[(r, c)] = self.output_cached[(r, c)]
                    * (1.0 - self.output_cached[(r, c)])
                    * gradient[(r, c)];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SigmoidLayer;
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
    fn sigmoid_layer_works() {
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(45);

        let x = FMatrix::from([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);

        let mut forward_out = FMatrix::<3, 2>::default();
        let mut ll = SigmoidLayer::<3, 2>::new(&mut rng);

        ll.forward(&x, &mut forward_out);

        let expected_forward_out = FMatrix::<3, 2>::from([
            [0.7310586, 0.880797],
            [0.95257413, 0.98201376],
            [0.9933072, 0.9975274],
        ]);
        assert_eq!(forward_out, expected_forward_out);

        let dout = FMatrix::<3, 2>::from([[43.2, 54.2], [24.1, 42.1], [12.5, 87.3]]);

        let mut dx_computed = FMatrix::<3, 2>::default();
        let mut opt = TestOptimizer::new_noop();

        ll.backward(&dout, &mut dx_computed, &mut opt);

        let dx_correct = FMatrix::from([
            [8.493635, 5.6906548],
            [1.0887574, 0.7436011],
            [0.08310041, 0.21532248],
        ]);

        assert_eq!(dx_computed, dx_correct);
    }

    #[bench]
    fn bench_sigmoid(bench: &mut Bencher) {
        const R: usize = 2000;
        const C: usize = 32;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(89);
        let a = random_fmatrix2::<_, R, C>(&mut rng);
        let mut b = FMatrix::<R, C>::default();

        let mut layer = SigmoidLayer::new(&mut rng);

        bench.iter(|| {
            layer.forward(&a, &mut b);
        });
    }

    const RNG_SEED: u64 = 25;
    const T_XI: usize = 8;
    const T_BATCH_SIZE: usize = 16;

    proptest! {

        #[test]
        fn layer_evaluated_on_single_sample_works(mut input in random_fmatrix::<T_XI, T_BATCH_SIZE>()) {
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(RNG_SEED);
            let mut ll = SigmoidLayer::<T_XI, T_BATCH_SIZE>::new(&mut rng);

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
            mut input in small_random_fmatrix::<T_XI, T_BATCH_SIZE>(),
            dout in small_random_fmatrix::<T_XI, T_BATCH_SIZE>()
        ){
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(RNG_SEED);
            let mut ll = SigmoidLayer::<T_XI, T_BATCH_SIZE>::new(&mut rng);

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
