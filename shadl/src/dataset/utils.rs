use std::path::Path;

use flate2::{self, bufread::GzDecoder};
use reqwest::{self, Url};
use std::fs::OpenOptions;

pub fn download_and_decompress(url: Url, destination: &Path) {
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .open(destination)
        .unwrap();

    let response = reqwest::blocking::get(url).unwrap();
    assert!(response.status().is_success());

    let response_buffer = std::io::BufReader::new(response);
    let mut decompressed = GzDecoder::new(response_buffer);
    std::io::copy(&mut decompressed, &mut file).unwrap();
}
