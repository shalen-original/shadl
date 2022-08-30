use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::ops::{Add, MulAssign};
use std::{
    borrow::Borrow,
    ops::{AddAssign, Index, IndexMut, Mul},
};

#[cfg(target_arch = "wasm32")]
use crate::no_rayon::prelude::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use crate::layer::Resettable;

// TODO for sure this can be done better
#[cfg(not(target_arch = "wasm32"))]
pub type MatrixIterMut<'a, T> = rayon::slice::IterMut<'a, T>;

#[cfg(not(target_arch = "wasm32"))]
pub type MatrixIter<'a, T> = rayon::slice::Iter<'a, T>;

#[cfg(target_arch = "wasm32")]
pub type MatrixIterMut<'a, T> = core::slice::IterMut<'a, T>;

#[cfg(target_arch = "wasm32")]
pub type MatrixIter<'a, T> = core::slice::Iter<'a, T>;

#[derive(Serialize, Deserialize, Clone)]
pub struct Matrix<T, const R: usize, const C: usize> {
    data: Vec<T>,
}

pub type FMatrix<const R: usize, const C: usize> = Matrix<f32, R, C>;

impl<TA: Debug, const RA: usize, const CA: usize> Debug for Matrix<TA, RA, CA> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut dbg = f.debug_list();
        for i in 0..RA {
            dbg.entry(&&self.data[i * CA..(i * CA) + CA]);
        }
        dbg.finish()
    }
}

impl<TA: Default + Copy, const RA: usize, const CA: usize> Default for Matrix<TA, RA, CA> {
    fn default() -> Self {
        Self {
            data: vec![TA::default(); RA * CA],
        }
    }
}

impl<TA, const RA: usize, const CA: usize> Index<(usize, usize)> for Matrix<TA, RA, CA> {
    type Output = TA;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (r, c) = index;
        assert!(r < RA);
        assert!(c < CA);

        &self.data[r * CA + c]
    }
}

impl<TA, const RA: usize, const CA: usize> IndexMut<(usize, usize)> for Matrix<TA, RA, CA> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (r, c) = index;
        assert!(r < RA);
        assert!(c < CA);

        &mut self.data[r * CA + c]
    }
}

impl<TA: Default + Copy, const RA: usize, const CA: usize> From<[[TA; CA]; RA]>
    for Matrix<TA, RA, CA>
{
    fn from(value: [[TA; CA]; RA]) -> Self {
        let mut out = Self::default();

        for r in 0..RA {
            for c in 0..CA {
                out[(r, c)] = value[r][c];
            }
        }

        out
    }
}

impl<
        TA: Default + Copy + PartialEq,
        const RA: usize,
        const CA: usize,
        const R: usize,
        const C: usize,
    > PartialEq<Matrix<TA, R, C>> for Matrix<TA, RA, CA>
{
    fn eq(&self, other: &Matrix<TA, R, C>) -> bool {
        if R != RA || C != CA {
            return false;
        }

        let mut res = true;

        for i in 0..RA * CA {
            res = res && (self.data[i] == other.data[i]);
        }

        res
    }
}

impl<TA: Default + Copy + PartialEq, const RA: usize, const CA: usize> Eq for Matrix<TA, RA, CA> {}

impl<TA: Default + Send, const RA: usize, const CA: usize> Resettable for Matrix<TA, RA, CA> {
    fn reset(&mut self) {
        self.data.par_iter_mut().for_each(|elem| {
            *elem = TA::default();
        });
    }
}

impl<
        TA: Default + Copy + Add<Output = TA> + Mul<Output = TA> + AddAssign + MulAssign + Send + Sync,
        const RA: usize,
        const CA: usize,
    > Matrix<TA, RA, CA>
{
    pub fn add_ip<RHS: Borrow<Self>>(&mut self, rhs: RHS) {
        self.data
            .par_iter_mut()
            .zip(&rhs.borrow().data)
            .for_each(|(lhs_elem, rhs_elem)| {
                *lhs_elem += *rhs_elem;
            });
    }

    pub fn scalar_mul_ip(&mut self, scalar: TA) {
        self.data.par_iter_mut().for_each(|el| {
            *el *= scalar;
        });
    }

    pub fn mul<const D: usize, RHS: Borrow<Matrix<TA, CA, D>>>(
        &self,
        rhs: RHS,
    ) -> Matrix<TA, RA, D> {
        let mut out = Matrix::<TA, RA, D>::default();
        self.mul_mut(rhs, &mut out);
        out
    }

    pub fn mul_nopar<const D: usize, RHS: Borrow<Matrix<TA, CA, D>>>(
        &self,
        rhs: RHS,
    ) -> Matrix<TA, RA, D> {
        let mut out = Matrix::<TA, RA, D>::default();
        self.mul_mut_nopar(rhs, &mut out);
        out
    }

    pub fn mul_mut<const D: usize, RHS: Borrow<Matrix<TA, CA, D>>>(
        &self,
        rhs: RHS,
        out: &mut Matrix<TA, RA, D>,
    ) {
        let out_row_iterator = out.data.par_chunks_mut(D);
        let self_row_iterator = self.data.par_chunks(CA);

        let rhsdata = &rhs.borrow().data;

        out_row_iterator
            .zip(self_row_iterator)
            .for_each(|(out_row, self_row)| {
                self_row
                    .iter()
                    .zip(rhsdata.chunks_exact(D))
                    .for_each(|(self_elem, rhs_row)| {
                        out_row
                            .iter_mut()
                            .zip(rhs_row.iter())
                            .for_each(|(out_elem, rhs_elem)| {
                                *out_elem += (*self_elem) * (*rhs_elem);
                            });
                    });
            });
    }

    pub fn mul_mut_nopar<const D: usize, RHS: Borrow<Matrix<TA, CA, D>>>(
        &self,
        rhs: RHS,
        out: &mut Matrix<TA, RA, D>,
    ) {
        // TODO this is hacky as hell and should be fixed, but I don't have much
        // more spare time to spend on this. Matrix should be a trait, and there should
        // be different implementations: SimpleMatrix, CPUParallelMatrix, GPUMatrix, ecc.
        let out_row_iterator = out.data.chunks_mut(D);
        let self_row_iterator = self.data.chunks(CA);

        let rhsdata = &rhs.borrow().data;

        out_row_iterator
            .zip(self_row_iterator)
            .for_each(|(out_row, self_row)| {
                self_row
                    .iter()
                    .zip(rhsdata.chunks_exact(D))
                    .for_each(|(self_elem, rhs_row)| {
                        out_row
                            .iter_mut()
                            .zip(rhs_row.iter())
                            .for_each(|(out_elem, rhs_elem)| {
                                *out_elem += (*self_elem) * (*rhs_elem);
                            });
                    });
            });
    }

    pub fn transpose(&self) -> Matrix<TA, CA, RA> {
        let mut out = Matrix::<TA, CA, RA>::default();
        self.transpose_mut(&mut out);
        out
    }

    pub fn transpose_mut(&self, out: &mut Matrix<TA, CA, RA>) {
        const BLOCK_SIZE: usize = 128; //Completely arbitrary

        for ii in (0..RA).step_by(BLOCK_SIZE) {
            for jj in (0..CA).step_by(BLOCK_SIZE) {
                for i in ii..(ii + BLOCK_SIZE).min(RA) {
                    for j in jj..(jj + BLOCK_SIZE).min(CA) {
                        out[(j, i)] = self[(i, j)];
                    }
                }
            }
        }
    }

    pub fn shape(&self) -> (usize, usize) {
        (RA, CA)
    }

    pub fn copy_from(&mut self, other: &Matrix<TA, RA, CA>) {
        self.data.copy_from_slice(other.data.as_slice());
    }

    pub fn par_iter_mut(&mut self) -> MatrixIterMut<'_, TA> {
        self.data.par_iter_mut()
    }

    pub fn par_iter(&self) -> MatrixIter<'_, TA> {
        self.data.par_iter()
    }

    pub fn add_column_ip(&mut self, other: &Matrix<TA, RA, 1>) {
        self.data
            .par_chunks_mut(CA)
            .zip(other.data.par_iter())
            .for_each(|(self_row, other_elem)| {
                self_row.iter_mut().for_each(|self_elem| {
                    *self_elem += *other_elem;
                })
            });
    }

    pub fn sum_rows(&self, output: &mut Matrix<TA, RA, 1>) {
        self.data
            .par_chunks(CA)
            .zip(output.data.par_iter_mut())
            .for_each(|(self_row, output_elem)| {
                self_row.iter().for_each(|self_row_elem| {
                    *output_elem += *self_row_elem;
                })
            });
    }
}

impl<const R: usize, const C: usize> FMatrix<R, C> {
    pub fn identity() -> FMatrix<R, C> {
        let mut ans = FMatrix::<R, C>::default();

        for i in 0..R.min(C) {
            ans[(i, i)] = 1.0;
        }

        ans
    }
}

#[cfg(test)]
mod tests {
    use crate::test_utils::*;
    use crate::{layer::Resettable, matrix::FMatrix};
    use rand::SeedableRng;
    use test::Bencher;

    #[test]
    fn index_and_index_mut_work() {
        let mut m = FMatrix::<3, 2>::default();

        m[(0, 1)] = 4.0;

        assert_eq!(m[(0, 0)], f32::default());
        assert_eq!(m[(0, 1)], 4.0);
    }

    #[test]
    #[should_panic]
    fn accessing_out_of_bound_panics() {
        let m = FMatrix::<2, 3>::default();
        m[(4, 2)];
    }

    #[test]
    #[should_panic]
    fn accessing_mut_out_of_bound_panics() {
        let mut m = FMatrix::<2, 3>::default();
        m[(1, 5)] = 3.0;
    }

    #[test]
    fn from_2d_array_works() {
        let m = FMatrix::from([[1.0], [2.0]]);

        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(1, 0)], 2.0);
    }

    #[test]
    fn sum_works() {
        let mut a = FMatrix::from([[1.0, 2.0]]);

        let b = FMatrix::from([[3.0, 4.0]]);

        a.add_ip(b);

        assert_eq!(a[(0, 0)], 4.0);
        assert_eq!(a[(0, 1)], 6.0);
    }

    #[test]
    fn mul_by_scalar_works() {
        let mut a = FMatrix::from([[1.0, 2.0]]);

        a.scalar_mul_ip(5.0);

        assert_eq!(a[(0, 0)], 5.0);
        assert_eq!(a[(0, 1)], 10.0);
    }

    #[test]
    fn horiz_transpose_works() {
        let a = FMatrix::from([[1.0, 2.0]]).transpose();

        assert_eq!(a[(0, 0)], 1.0);
        assert_eq!(a[(1, 0)], 2.0);
    }

    #[test]
    #[should_panic]
    fn horiz_transpose_panics_if_out_of_bound() {
        let mut a = FMatrix::from([[1.0, 2.0]]).transpose();
        a[(0, 1)] = 3.0;
    }

    #[test]
    fn transpose_rect_works() {
        let a = FMatrix::from([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).transpose();

        assert_eq!(a[(1, 0)], 2.0);
        assert_eq!(a[(1, 2)], 6.0);
    }

    #[test]
    fn multiply_works() {
        let a = FMatrix::from([[1.0, 2.0]]);

        let b = FMatrix::from([[3.0], [4.0]]);

        let c = a.mul(b);

        assert_eq!(c[(0, 0)], 11.0);
    }

    #[test]
    fn debug_works() {
        let a = FMatrix::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        assert_eq!(format!("{:?}", a), "[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]",);
    }

    #[test]
    fn eq_works() {
        let a = FMatrix::from([[2.0], [810.80000026]]);

        let b = FMatrix::from([[2.0], [810.80000000]]);

        let c = FMatrix::from([[2.0], [810.80010000]]);

        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn reset_works() {
        let mut a = FMatrix::from([[123.0, 321.0], [73.0, 34.5]]);
        a.reset();

        let b = FMatrix::<2, 2>::default();

        assert_eq!(a, b);
    }

    #[test]
    fn multiply_works_2() {
        let a = FMatrix::<3, 2>::from([
            [3.464489, 3.06411034],
            [3.32497139, 2.23945532],
            [2.81179517, 2.17634719],
        ]);

        let b = FMatrix::<2, 4>::from([
            [3.01196111, 3.27291706, 0.17137147, 0.26859069],
            [4.52207883, 3.94724842, 0.05556172, 2.77570207],
        ]);

        let expected = FMatrix::<3, 4>::from([
            [24.29105463, 23.43379, 0.76396185, 9.43558688],
            [20.14167803, 19.72204206, 0.69423321, 7.1091166],
            [18.31063, 17.793356, 0.6027831, 6.7961135],
        ]);

        let mut c = Default::default();

        a.mul_mut(b, &mut c);

        assert_eq!(c, expected);
    }

    #[test]
    fn copy_from_works() {
        let a = FMatrix::<2, 1>::from([[3.464], [3.324]]);
        let mut b = FMatrix::<2, 1>::default();

        assert_ne!(a, b);

        b.copy_from(&a);

        assert_eq!(a, b);
    }

    #[test]
    fn add_column_ip_works() {
        let mut a = FMatrix::<2, 3>::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let b = FMatrix::<2, 1>::from([[1.0], [2.0]]);

        a.add_column_ip(&b);

        let expected_a = FMatrix::<2, 3>::from([[2.0, 3.0, 4.0], [6.0, 7.0, 8.0]]);

        assert_eq!(a, expected_a);
    }

    #[test]
    fn sum_rows_works() {
        let a = FMatrix::<2, 3>::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let mut b = FMatrix::<2, 1>::default();

        a.sum_rows(&mut b);

        let expected_b = FMatrix::<2, 1>::from([[6.0], [15.0]]);

        assert_eq!(b, expected_b);
    }

    #[test]
    fn identity_works() {
        let a = FMatrix::<2, 3>::identity();
        let a_expected = FMatrix::from([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);

        assert_eq!(a, a_expected);
    }

    #[bench]
    fn bench_mul(bench: &mut Bencher) {
        const A: usize = 100;
        const B: usize = 500;
        const C: usize = 200;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(89);
        let a = random_fmatrix2::<_, A, B>(&mut rng);
        let b = random_fmatrix2::<_, B, C>(&mut rng);

        bench.iter(|| {
            let mut c = Default::default();
            a.mul_mut(&b, &mut c);
        });
    }

    #[bench]
    fn bench_add_ip(bench: &mut Bencher) {
        const R: usize = 1000;
        const C: usize = 800;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(89);
        let mut a = random_fmatrix2::<_, R, C>(&mut rng);
        let b = random_fmatrix2::<_, R, C>(&mut rng);

        bench.iter(|| {
            a.add_ip(&b);
        });
    }

    #[bench]
    fn bench_add_column_ip(bench: &mut Bencher) {
        const R: usize = 1000;
        const C: usize = 800;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(89);
        let mut a = random_fmatrix2::<_, R, C>(&mut rng);
        let b = random_fmatrix2::<_, R, 1>(&mut rng);

        bench.iter(|| {
            a.add_column_ip(&b);
        });
    }

    #[bench]
    fn bench_scalar_mul_ip(bench: &mut Bencher) {
        const R: usize = 1000;
        const C: usize = 800;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(89);
        let mut a = random_fmatrix2::<_, R, C>(&mut rng);

        bench.iter(|| {
            a.scalar_mul_ip(5.0);
        });
    }

    #[bench]
    fn bench_reset(bench: &mut Bencher) {
        const R: usize = 2000;
        const C: usize = 2500;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(89);
        let mut a = random_fmatrix2::<_, R, C>(&mut rng);

        bench.iter(|| {
            a.reset();
        });
    }

    #[bench]
    fn bench_transpose(bench: &mut Bencher) {
        const R: usize = 2000;
        const C: usize = 2500;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(89);
        let a = random_fmatrix2::<_, R, C>(&mut rng);
        let mut b = FMatrix::<C, R>::default();

        bench.iter(|| {
            a.transpose_mut(&mut b);
        });
    }
}
