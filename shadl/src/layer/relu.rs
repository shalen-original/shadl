use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::matrix::FMatrix;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use super::Layer;

#[derive(Serialize, Deserialize, Debug)]
pub struct ReluLayer<const R: usize, const C: usize> {
    output_cached: FMatrix<R, C>,
}

impl<const R: usize, const C: usize> Layer<FMatrix<R, C>> for ReluLayer<R, C> {
    type Output = FMatrix<R, C>;

    fn new<RNG: Rng + ?Sized>(_rng: &mut RNG) -> Self {
        ReluLayer {
            output_cached: FMatrix::<R, C>::default(),
        }
    }

    fn forward(&mut self, input: &FMatrix<R, C>, output: &mut Self::Output) {
        output
            .par_iter_mut()
            .zip(input.par_iter())
            .for_each(|(out_elem, in_elem)| {
                *out_elem = if *in_elem >= 0.0 { *in_elem } else { 0.0 };
            });

        self.output_cached.copy_from(output);
    }

    fn backward<Opt: crate::optimizer::Optimizer>(
        &mut self,
        gradient: &Self::Output,
        inputs_gradient: &mut FMatrix<R, C>,
        _optimizer: &mut Opt,
    ) {
        inputs_gradient
            .par_iter_mut()
            .zip(gradient.par_iter())
            .zip(self.output_cached.par_iter())
            .for_each(|((in_grad_elem, grad_elem), out_cached_elem)| {
                *in_grad_elem = if *out_cached_elem > 0.0 { 1.0 } else { 0.0 };
                *in_grad_elem *= grad_elem;
            })
    }
}

#[cfg(test)]
mod tests {
    use super::ReluLayer;
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
    fn relu_layer_works() {
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(45);

        let x = FMatrix::from([[1.0, 2.0], [-1.0, -2.0], [0.1, -0.1]]);

        let mut forward_out = FMatrix::<3, 2>::default();
        let mut ll = ReluLayer::<3, 2>::new(&mut rng);

        ll.forward(&x, &mut forward_out);

        let expected_forward_out = FMatrix::<3, 2>::from([[1.0, 2.0], [0.0, 0.0], [0.1, 0.0]]);
        assert_eq!(forward_out, expected_forward_out);

        let dout = FMatrix::<3, 2>::from([[43.2, 54.2], [24.1, 42.1], [12.5, 87.3]]);

        let mut dx_computed = FMatrix::<3, 2>::default();
        let mut opt = TestOptimizer::new_noop();

        ll.backward(&dout, &mut dx_computed, &mut opt);

        let dx_correct = FMatrix::from([[43.2, 54.2], [0.0, 0.0], [12.5, 0.0]]);

        assert_eq!(dx_computed, dx_correct);
    }

    #[bench]
    fn bench_relu(bench: &mut Bencher) {
        const R: usize = 2000;
        const C: usize = 32;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(89);
        let a = random_fmatrix2::<_, R, C>(&mut rng);
        let mut b = FMatrix::<R, C>::default();

        let mut layer = ReluLayer::new(&mut rng);

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
            let mut ll = ReluLayer::<T_XI, T_BATCH_SIZE>::new(&mut rng);

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
            let mut ll = ReluLayer::<T_XI, T_BATCH_SIZE>::new(&mut rng);

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
