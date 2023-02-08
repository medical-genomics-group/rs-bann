use std::fs::File;
use std::io::{BufRead, BufReader};
use std::marker::PhantomData;
use std::path::Path;

pub trait IndexedEntry {
    fn from_str(s: &str, ix: usize) -> Self;
}

pub struct IndexedReader<T: IndexedEntry> {
    num_read: usize,
    reader: BufReader<File>,
    buffer: String,
    _phantom: PhantomData<T>,
}

impl<T: IndexedEntry> IndexedReader<T> {
    pub fn num_lines(fam_path: &Path) -> usize {
        let mut reader = IndexedReader::<T>::new(fam_path);
        while reader.next_entry().is_some() {}
        reader.num_read
    }

    pub fn new(fam_path: &Path) -> Self {
        Self {
            num_read: 0,
            reader: BufReader::new(File::open(fam_path).unwrap()),
            buffer: String::new(),
            _phantom: PhantomData,
        }
    }

    pub fn next_entry(&mut self) -> Option<T> {
        self.buffer.clear();
        if let Ok(bytes_read) = self.reader.read_line(&mut self.buffer) {
            if bytes_read > 0 {
                self.num_read += 1;
                return Some(T::from_str(&self.buffer, self.last_entry_ix()));
            }
        }
        None
    }

    pub fn last_entry_ix(&self) -> usize {
        self.num_read - 1
    }
}
