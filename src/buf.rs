//! A slice that is either an owned `Vec` or a window into a memory-mapped file.
//!
//! Why this exists: loading a 5.6 GB array by reading the file into a `Vec` makes the data live in
//! RAM twice for a while (the file's page cache plus the copy). On an 11 GB machine that pushes
//! Linux into swap, and lookups into a partly swapped-out array take tens of milliseconds.
//! Mapping the file instead means each byte lives in RAM exactly once, as clean file pages that
//! Linux can drop and re-read but never needs to write to swap.

use crate::ivf::Pod;
use memmap2::Mmap;
use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::Arc;

pub enum Buf<T: Pod> {
    Owned(Vec<T>),
    Mapped {
        map: Arc<Mmap>,
        off: usize,
        len: usize,
        _t: PhantomData<T>,
    },
}

impl<T: Pod> Buf<T> {
    /// A window of `len` elements starting at byte `off` of `map`. Falls back to a copy if the
    /// offset isn't aligned for `T` (only possible for unusual dimensions).
    pub fn mapped(map: &Arc<Mmap>, off: usize, len: usize) -> Self {
        let bytes = len * std::mem::size_of::<T>();
        assert!(
            off + bytes <= map.len(),
            "file is shorter than its header says"
        );
        let ptr = map[off..].as_ptr();
        if (ptr as usize).is_multiple_of(std::mem::align_of::<T>()) {
            Buf::Mapped {
                map: Arc::clone(map),
                off,
                len,
                _t: PhantomData,
            }
        } else {
            let mut v = vec![T::default(); len];
            // SAFETY: Pod types have no invalid bit patterns; sizes checked above.
            unsafe {
                std::ptr::copy_nonoverlapping(ptr, v.as_mut_ptr() as *mut u8, bytes);
            }
            Buf::Owned(v)
        }
    }

    /// A copy of the window, for small arrays that are cheaper to own.
    pub fn copied(map: &Arc<Mmap>, off: usize, len: usize) -> Self {
        let b: Buf<T> = Buf::mapped(map, off, len);
        Buf::Owned(b.to_vec())
    }
}

impl<T: Pod> Deref for Buf<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        match self {
            Buf::Owned(v) => v,
            Buf::Mapped { map, off, len, .. } => {
                // SAFETY: alignment and bounds checked in `mapped`; the map outlives the slice.
                unsafe { std::slice::from_raw_parts(map[*off..].as_ptr() as *const T, *len) }
            }
        }
    }
}

impl<T: Pod> From<Vec<T>> for Buf<T> {
    fn from(v: Vec<T>) -> Self {
        Buf::Owned(v)
    }
}

/// Map a whole file. `populate` pre-reads every page (MAP_POPULATE), so the first queries
/// don't pay for page faults: the index is meant to be in RAM, only the base vectors on disk.
pub fn map_file(path: &std::path::Path, populate: bool) -> std::io::Result<Arc<Mmap>> {
    let f = std::fs::File::open(path)?;
    let mut opts = memmap2::MmapOptions::new();
    if populate {
        opts.populate();
    }
    // SAFETY: the file is not modified while mapped (index files are written once).
    let map = unsafe { opts.map(&f)? };
    Ok(Arc::new(map))
}
