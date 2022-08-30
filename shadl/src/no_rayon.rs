//https://github.com/rayon-rs/rayon/issues/861

// #[cfg(target_arch = "wasm32")]
// mod no_rayon;

// #[cfg(not(target_arch = "wasm32"))]
// use crate::no_rayon::prelude::*;
// #[cfg(target_arch = "wasm32")]
// use no_rayon::prelude::*;

pub mod prelude {
    pub use super::{NoRayonSlice, NoRayonSliceMut};
}

pub trait NoRayonSlice<T> {
    fn par_iter(&self) -> core::slice::Iter<'_, T>;
    fn par_chunks_exact(&self, chunk_size: usize) -> core::slice::ChunksExact<'_, T>;
    fn par_chunks(&self, chunk_size: usize) -> core::slice::Chunks<'_, T>;
}
impl<T> NoRayonSlice<T> for [T] {
    fn par_iter(&self) -> core::slice::Iter<'_, T> {
        self.iter()
    }
    fn par_chunks_exact(&self, chunk_size: usize) -> core::slice::ChunksExact<'_, T> {
        self.chunks_exact(chunk_size)
    }
    fn par_chunks(&self, chunk_size: usize) -> core::slice::Chunks<'_, T> {
        self.chunks(chunk_size)
    }
}

pub trait NoRayonSliceMut<T> {
    fn par_chunks_exact_mut(&mut self, chunk_size: usize) -> core::slice::ChunksExactMut<'_, T>;
    fn par_chunks_mut(&mut self, chunk_size: usize) -> core::slice::ChunksMut<'_, T>;
    fn par_iter_mut(&mut self) -> core::slice::IterMut<'_, T>;
}
impl<T> NoRayonSliceMut<T> for [T] {
    fn par_chunks_exact_mut(&mut self, chunk_size: usize) -> core::slice::ChunksExactMut<'_, T> {
        self.chunks_exact_mut(chunk_size)
    }
    fn par_chunks_mut(&mut self, chunk_size: usize) -> core::slice::ChunksMut<'_, T> {
        self.chunks_mut(chunk_size)
    }
    fn par_iter_mut(&mut self) -> core::slice::IterMut<'_, T> {
        self.iter_mut()
    }
}
