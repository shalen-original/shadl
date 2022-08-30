use std::{any::Any, ops::Range};

use proptest::collection::vec;
use proptest::prelude::*;

use crate::{
    layer::Layer,
    matrix::{FMatrix, Matrix},
    optimizer::Optimizer,
};

pub fn random_matrix<TA: Arbitrary + Default + Copy, const R: usize, const C: usize>(
    strategy: impl Strategy<Value = TA>,
) -> impl Strategy<Value = Matrix<TA, R, C>> {
    vec(strategy, R * C).prop_map(|v| {
        let mut m = Matrix::<TA, R, C>::default();
        for i in 0..R {
            for j in 0..C {
                m[(i, j)] = v[i * C + j];
            }
        }
        m
    })
}

pub fn random_fmatrix<const R: usize, const C: usize>() -> impl Strategy<Value = FMatrix<R, C>> {
    let r: Range<f32> = 1e-5..1e5;
    random_matrix::<f32, R, C>(r)
}

pub fn random_fmatrix2<RNG: Rng + ?Sized, const R: usize, const C: usize>(
    rng: &mut RNG,
) -> FMatrix<R, C> {
    let mut a = FMatrix::<R, C>::default();

    for r in 0..R {
        for c in 0..C {
            a[(r, c)] = rng.gen();
        }
    }

    a
}

pub fn small_random_fmatrix<const R: usize, const C: usize>() -> impl Strategy<Value = FMatrix<R, C>>
{
    let r: Range<f32> = 1e-1..1e1;
    random_matrix::<f32, R, C>(r)
}

fn rdp_with_epsilon(a: f32, b: f32) -> f32 {
    (a - b).abs() / (a.abs() + b.abs()).max(1e-2)
}

pub fn max_rdp<const R: usize, const C: usize>(a: &FMatrix<R, C>, b: &FMatrix<R, C>) -> f32 {
    //https://stats.stackexchange.com/a/201864
    let mut max_rdp = 0.0_f32;

    for r in 0..R {
        for c in 0..C {
            let curr_rdp = rdp_with_epsilon(a[(r, c)], b[(r, c)]);
            max_rdp = max_rdp.max(curr_rdp);
        }
    }

    max_rdp
}

pub fn avg_rdp<const R: usize, const C: usize>(a: &FMatrix<R, C>, b: &FMatrix<R, C>) -> f32 {
    let mut avg_rdp = 0.0_f32;

    for r in 0..R {
        for c in 0..C {
            let curr_rdp = rdp_with_epsilon(a[(r, c)], b[(r, c)]);
            avg_rdp = avg_rdp + curr_rdp;
        }
    }

    avg_rdp / (R as f32 * C as f32)
}

pub fn numerical_diff<
    L,
    const RI: usize,
    const CI: usize,
    const RO: usize,
    const CO: usize,
    const LRI: usize,
    const LCI: usize,
>(
    ll: &mut L,
    og_input: &mut FMatrix<RI, CI>,
    dout: &FMatrix<RO, CO>,
    fw: impl Fn(&mut L, &FMatrix<RI, CI>) -> FMatrix<RO, CO>,
) -> FMatrix<RI, CI>
where
    L: Layer<FMatrix<LRI, LCI>, Output = FMatrix<RO, CO>>,
{
    const H: f32 = 0.1;

    let mut dinput_num = FMatrix::default();

    for ri in 0..RI {
        for ci in 0..CI {
            let original_value = og_input[(ri, ci)];

            og_input[(ri, ci)] = original_value + H;
            let fwd_out = fw(ll, &og_input);

            og_input[(ri, ci)] = original_value - H;
            let bkw_out = fw(ll, &og_input);

            og_input[(ri, ci)] = original_value;

            for ro in 0..RO {
                for co in 0..CO {
                    let d = (fwd_out[(ro, co)] - bkw_out[(ro, co)]) * dout[(ro, co)];

                    // TODO: the reason for this summation is not clear to me
                    dinput_num[(ri, ci)] += d;
                }
            }

            dinput_num[(ri, ci)] /= 2.0 * H;
        }
    }

    dinput_num
}

// TODO RDP TEST

pub struct TestOptimizer<'a> {
    foreach_callback: &'a dyn Fn(&mut dyn Any, &mut dyn Any),
}

impl<'a> TestOptimizer<'a> {
    pub fn new(cb: &'a dyn Fn(&mut dyn Any, &mut dyn Any)) -> TestOptimizer<'a> {
        TestOptimizer {
            foreach_callback: cb,
        }
    }

    pub fn new_noop() -> TestOptimizer<'a> {
        TestOptimizer {
            foreach_callback: &|_a, _b| {},
        }
    }
}

impl Optimizer for TestOptimizer<'_> {
    fn update_weights<const R: usize, const C: usize>(
        &mut self,
        weights: &mut FMatrix<R, C>,
        mut dweights: FMatrix<R, C>,
    ) {
        (self.foreach_callback)(weights, &mut dweights);
    }
}
