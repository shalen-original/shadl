use std::{f32::consts::PI, ops::Range};

use rand::Rng;

use crate::matrix::FMatrix;

pub trait DataAugmenter<I, R: Rng + ?Sized> {
    fn augment(&self, input: &I, rng: &mut R) -> I;
}

pub struct NoopAugmenter;

impl NoopAugmenter {
    pub fn new() -> Self {
        NoopAugmenter {}
    }
}

impl<A, R: Rng + ?Sized> DataAugmenter<A, R> for NoopAugmenter
where
    A: Clone,
{
    fn augment(&self, a: &A, _: &mut R) -> A {
        a.clone()
    }
}

pub struct AffineAugmenter {
    translation_range: Range<f32>,
    rotation_degrees_range: Range<f32>,
    scale_range: Range<f32>,
    image_rows: usize,
    image_cols: usize,
}

impl AffineAugmenter {
    pub fn new(
        translation: Range<f32>,
        rotation_degrees: Range<f32>,
        scale: Range<f32>,
        image_rows: usize,
        image_cols: usize,
    ) -> Self {
        AffineAugmenter {
            translation_range: translation,
            rotation_degrees_range: rotation_degrees,
            scale_range: scale,
            image_rows: image_rows,
            image_cols: image_cols,
        }
    }

    fn generate_transform(&self, rng: &mut (impl Rng + ?Sized)) -> FMatrix<3, 3> {
        let tr = FMatrix::from([
            [1.0, 0.0, rng.gen_range(self.translation_range.clone())],
            [0.0, 1.0, rng.gen_range(self.translation_range.clone())],
            [0.0, 0.0, 1.0],
        ]);

        let mut sc = FMatrix::<3, 3>::identity();
        sc.scalar_mul_ip(rng.gen_range(self.scale_range.clone()));

        let theta = rng.gen_range(self.rotation_degrees_range.clone()) * PI / 180.0;
        let center_r = self.image_rows as f32 / 2.0;
        let center_c = self.image_cols as f32 / 2.0;
        let tr_rot_before = FMatrix::from([
            [1.0, 0.0, -center_r],
            [0.0, 1.0, -center_c],
            [0.0, 0.0, 1.0],
        ]);
        let rot = FMatrix::from([
            [theta.cos(), theta.sin(), 0.0],
            [-theta.sin(), theta.cos(), 0.0],
            [0.0, 0.0, 1.0],
        ]);
        let tr_rot_after =
            FMatrix::from([[1.0, 0.0, center_r], [0.0, 1.0, center_c], [0.0, 0.0, 1.0]]);

        let center_rot = tr_rot_after.mul_nopar(rot).mul_nopar(tr_rot_before);

        tr.mul_nopar(sc).mul_nopar(center_rot)
    }
}

impl<const XI: usize, R: Rng + ?Sized> DataAugmenter<FMatrix<XI, 1>, R> for AffineAugmenter {
    fn augment(&self, input: &FMatrix<XI, 1>, rng: &mut R) -> FMatrix<XI, 1> {
        let rows = self.image_rows;
        let cols = self.image_cols;

        assert_eq!(rows * cols, XI);

        let transform = self.generate_transform(rng);

        let mut out = FMatrix::<XI, 1>::default();

        for r in 0..rows {
            for c in 0..cols {
                let p = FMatrix::<3, 1>::from([[r as f32], [c as f32], [1.0]]);
                let p_tr = transform.mul_nopar(p);

                let new_r = p_tr[(0, 0)].round() as usize;
                let new_c = p_tr[(1, 0)].round() as usize;

                if new_r < rows && new_c < cols {
                    let old_point = r * cols + c;
                    let new_point = new_r * cols + new_c;
                    out[(new_point, 0)] = input[(old_point, 0)];
                }
            }
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use test::Bencher;

    use crate::{matrix::FMatrix, test_utils::random_fmatrix2};

    use super::{AffineAugmenter, DataAugmenter, NoopAugmenter};

    #[test]
    fn noop_augmenter_works() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(89);
        let aug = NoopAugmenter::new();
        let m: FMatrix<5, 5> = random_fmatrix2(&mut rng);

        let augmented_m = aug.augment(&m, &mut rng);

        assert_eq!(m, augmented_m);
    }

    #[test]
    #[should_panic]
    fn affine_agumenter_panics_if_img_size_is_wrong() {
        const IMAGE_SIDE: usize = 6;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(89);
        let aug = AffineAugmenter::new(-0.3..0.5, 10.0..40.0, 0.9..1.1, IMAGE_SIDE, IMAGE_SIDE);

        let m: FMatrix<5, 1> = random_fmatrix2(&mut rng);
        aug.augment(&m, &mut rng);
    }

    #[test]
    fn affine_agumenter_works_ish() {
        const IMAGE_SIDE: usize = 6;
        const RS: usize = IMAGE_SIDE * IMAGE_SIDE;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(89);
        let aug = AffineAugmenter::new(-0.3..0.5, 10.0..40.0, 0.9..1.1, IMAGE_SIDE, IMAGE_SIDE);

        let m: FMatrix<RS, 1> = random_fmatrix2(&mut rng);

        let augmented_m = aug.augment(&m, &mut rng);

        let expected_m = FMatrix::from([
            [0.0],
            [0.06945938],
            [0.45040977],
            [0.802772],
            [0.6711951],
            [0.0],
            [0.06548977],
            [0.0],
            [0.58278304],
            [0.3597231],
            [0.0],
            [0.8125966],
            [0.6408051],
            [0.8306861],
            [0.84158474],
            [0.3208447],
            [0.19795012],
            [0.08545804],
            [0.26273364],
            [0.3762009],
            [0.9644889],
            [0.34250146],
            [0.038368344],
            [0.8859633],
            [0.6565781],
            [0.0],
            [0.30020088],
            [0.78966963],
            [0.0],
            [0.4738496],
            [0.0],
            [0.8233536],
            [0.8293104],
            [0.45905232],
            [0.47497463],
            [0.0],
        ]);

        assert_eq!(augmented_m, expected_m);
    }

    #[bench]
    fn bench_affine_augment(bench: &mut Bencher) {
        const IMAGE_SIDE: usize = 6;
        const RS: usize = IMAGE_SIDE * IMAGE_SIDE;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(89);
        let aug = AffineAugmenter::new(-10.0..10.0, 10.0..40.0, 0.9..1.1, IMAGE_SIDE, IMAGE_SIDE);

        let m: FMatrix<RS, 1> = random_fmatrix2(&mut rng);

        bench.iter(|| aug.augment(&m, &mut rng));
    }
}
