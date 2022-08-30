use std::{
    fs::File,
    io::{BufReader, BufWriter, Write},
};

use serde::Serialize;

pub fn save_to_file<T: Serialize>(file_path: &str, obj: &T) {
    let mut w = BufWriter::new(File::create(file_path).unwrap());
    bincode::serialize_into(&mut w, &obj).unwrap();
    w.flush().unwrap();
}

pub fn load_from_file<T: serde::de::DeserializeOwned>(file_path: &str) -> T {
    let file = File::open(file_path).unwrap();
    bincode::deserialize_from(BufReader::new(file)).unwrap()
}

pub fn load_from_bytes<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> T {
    bincode::deserialize(bytes).unwrap()
}
