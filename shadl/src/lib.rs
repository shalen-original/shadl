#![feature(test)]
#![feature(generic_arg_infer)]

extern crate test;

#[cfg(target_arch = "wasm32")]
mod no_rayon;

pub mod dataset;
pub mod layer;
pub mod losses;
pub mod matrix;
pub mod optimizer;
pub mod serialization;

#[macro_use]
mod nn;
pub use nn::*;

#[cfg(test)]
mod test_utils;
