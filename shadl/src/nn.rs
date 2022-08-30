use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::layer::Resettable;
use crate::{layer::Layer, optimizer::Optimizer};

#[derive(Serialize, Deserialize, Debug)]
pub struct NeuralNetwork<N, B> {
    network: N,
    forward_buffers: B,
    backward_buffers: B,
}

macro_rules! impl_layer_for_nn {
    ([$($type_names:ident),+], [$($rev_type_names:ident),+], $last:ident, [$($ids:tt),*], [$($rev_ids:tt),*], $last_id:tt) => {
        impl <
            I,
            $last:
            $(Layer::<$rev_type_names::Output>, $rev_type_names: )+
            Layer::<I>
        > Layer<I> for NeuralNetwork<($($type_names),+), ($($type_names::Output),+)>{
            type Output = $last::Output;

            fn new<RNG: Rng + ?Sized>(rng: &mut RNG) -> Self {
                NeuralNetwork {
                    network: (($($type_names::new(rng)),+)),
                    forward_buffers: Default::default(),
                    backward_buffers: Default::default()
                }
            }

            fn forward(&mut self, input: &I, output: &mut Self::Output) {
                self.network.0.forward(input, &mut self.forward_buffers.0);
                let prev = &self.forward_buffers.0;

                $(
                    self.network.$ids.forward(prev, &mut self.forward_buffers.$ids);
                    let prev = &self.forward_buffers.$ids;
                )*

                self.network.$last_id.forward(prev, output);

                self.forward_buffers.0.reset();
                $(self.forward_buffers.$ids.reset();)*
            }

            fn backward<Opt: Optimizer>(&mut self, gradient: & Self::Output, inputs_gradient: &mut I, optimizer: &mut Opt) {
                let lay = &mut self.network.$last_id;
                let bb = gradient;

                $(
                    lay.backward(&bb, &mut self.backward_buffers.$rev_ids, optimizer);
                    let lay = &mut self.network.$rev_ids;
                    let bb = &self.backward_buffers.$rev_ids;
                )*

                lay.backward(&bb, &mut self.backward_buffers.0, optimizer);
                let lay = &mut self.network.0;
                let bb = &self.backward_buffers.0;

                lay.backward(&bb, inputs_gradient, optimizer);

                self.backward_buffers.0.reset();
                $(self.backward_buffers.$ids.reset();)*
            }

        }
    };
}

impl_layer_for_nn!([A, B], [A], B, [], [], 1);
impl_layer_for_nn!([A, B, C], [B, A], C, [1], [1], 2);
impl_layer_for_nn!([A, B, C, D], [C, B, A], D, [1, 2], [2, 1], 3);
impl_layer_for_nn!([A, B, C, D, E], [D, C, B, A], E, [1, 2, 3], [3, 2, 1], 4);
impl_layer_for_nn!(
    [A, B, C, D, E, F],
    [E, D, C, B, A],
    F,
    [1, 2, 3, 4],
    [4, 3, 2, 1],
    5
);
impl_layer_for_nn!(
    [A, B, C, D, E, F, G],
    [F, E, D, C, B, A],
    G,
    [1, 2, 3, 4, 5],
    [5, 4, 3, 2, 1],
    6
);
impl_layer_for_nn!(
    [A, B, C, D, E, F, G, H],
    [G, F, E, D, C, B, A],
    H,
    [1, 2, 3, 4, 5, 6],
    [6, 5, 4, 3, 2, 1],
    7
);

// TODO this macro should return a type, if at all possible, and not a initialized value
// This would allow composing different nn! instantiations, allowing networks bigger than
// eight layers
#[macro_export]
macro_rules! nn {
    ($rng:tt, $($layers:ty),+) => {
        {
            let res: NeuralNetwork::<(
                $($layers),+
            ), (
                $(<$layers as Layer<_>>::Output),+
            )> = Layer::new(&mut $rng);
            res
        }
    };
}

#[cfg(test)]
mod tests {
    use crate::{
        layer::{AffineLayer, SigmoidLayer},
        matrix::FMatrix,
        test_utils::{
            max_rdp, numerical_diff, random_fmatrix, small_random_fmatrix, TestOptimizer,
        },
    };
    use crate::{nn::*, optimizer};
    use proptest::prelude::*;
    use rand::SeedableRng;

    #[test]
    fn macros_work() {
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(45);

        let mut network = nn!(
            rng,
            AffineLayer<3, 3, 1>,
            AffineLayer<3, 3, 1>,
            SigmoidLayer<3, 1>
        );

        let a = FMatrix::from([[1.0], [2.0], [3.0]]);

        let mut b = FMatrix::default();

        network.forward(&a, &mut b);

        let b_expected = FMatrix::from([[0.63301927], [0.39802882], [0.11884275]]);

        assert_eq!(b, b_expected);

        let mut ig = FMatrix::default();
        let mut opt = optimizer::SimpleGradientDescent::default();

        network.backward(&a, &mut ig, &mut opt);

        let ig_expected = FMatrix::from([[0.375444], [-0.32731295], [0.10375953]]);

        assert_eq!(ig, ig_expected);
    }

    const RNG_SEED: u64 = 51;
    const T_BATCH_SIZE: usize = 16;
    const T_XI_1: usize = 16;
    const T_XI_2: usize = T_XI_1 / 2;
    const T_M: usize = T_XI_2 / 2;

    proptest! {

        #[test]
        fn layer_evaluated_on_single_sample_works(mut input in random_fmatrix::<T_XI_1, T_BATCH_SIZE>()) {
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(RNG_SEED);
            let mut network = nn!{
                rng,
                AffineLayer<T_XI_1, T_XI_2, T_BATCH_SIZE>,
                SigmoidLayer<T_XI_2, T_BATCH_SIZE>,
                AffineLayer<T_XI_2, T_M, T_BATCH_SIZE>,
                SigmoidLayer<T_M, T_BATCH_SIZE>
            };

            let mut out = FMatrix::<T_M, T_BATCH_SIZE>::default();
            let mut out_zeros = FMatrix::<T_M, T_BATCH_SIZE>::default();

            network.forward(&input, &mut out);

            for i in 0..T_XI_1 {
                for j in 1..T_BATCH_SIZE {
                    input[(i, j)] = 0.0;
                }
            }

            network.forward(&input, &mut out_zeros);

            for j in 0..T_M {
                prop_assert_eq!(out_zeros[(j, 0)], out[(j, 0)]);
            }
        }

        #[test]
        fn dinput_numerically_correct(
            mut input in small_random_fmatrix::<T_XI_1, T_BATCH_SIZE>(),
            dout in small_random_fmatrix::<T_M, T_BATCH_SIZE>()
        ){
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(RNG_SEED);
            let mut network = nn!{
                rng,
                AffineLayer<T_XI_1, T_XI_2, T_BATCH_SIZE>,
                SigmoidLayer<T_XI_2, T_BATCH_SIZE>,
                AffineLayer<T_XI_2, T_M, T_BATCH_SIZE>,
                SigmoidLayer<T_M, T_BATCH_SIZE>
            };

            let dinput_num = numerical_diff(&mut network, &mut input, &dout, |network, curr_i| {
                let mut out = FMatrix::default();
                network.forward(&curr_i, &mut out);
                out
            });

            let mut dinput = FMatrix::default();
            let mut opt = TestOptimizer::new_noop();
            let mut useless = Default::default();
            network.forward(&input, &mut useless);
            network.backward(&dout, &mut dinput, &mut opt);

            //println!("max = {:.3?}, avg = {:.3?}", max_rdp(&dinput, &dinput_num), avg_rdp(&dinput, &dinput_num));
            prop_assert!(max_rdp(&dinput, &dinput_num) < 0.05);
        }
    }
}
