//! File formats from the NeurIPS'21 Big-ANN benchmark.
//!
//! `.fbin`  : u32 n, u32 d, then n*d little-endian f32 (row-major)
//! ground truth: u32 nq, u32 k, then nq*k u32 ids, then nq*k f32 distances
//!
//! Base files are memory-mapped, not read into RAM: a 100M x 96 file is ~38 GB, larger than
//! memory. The OS pages in only the vectors we touch (during reranking, a few hundred per query).

use memmap2::Mmap;
use std::fs::File;
use std::io::{BufWriter, Read, Result, Write};
use std::path::Path;

pub struct Fbin {
    mmap: Mmap,
    pub n: usize,
    pub d: usize,
}

impl Fbin {
    /// Open an .fbin file; `limit` uses only the first `limit` vectors (e.g. the 10M prefix of a
    /// larger download), which is how Big-ANN defines its 10M / 100M subsets.
    pub fn open(path: &Path, limit: Option<usize>) -> Result<Self> {
        let f = File::open(path)?;
        let mmap = unsafe { Mmap::map(&f)? };
        if mmap.len() < 8 {
            return Err(std::io::Error::other("file too small for fbin header"));
        }
        let n_hdr = u32::from_le_bytes(mmap[0..4].try_into().unwrap()) as usize;
        let d = u32::from_le_bytes(mmap[4..8].try_into().unwrap()) as usize;
        // A partially downloaded file holds fewer vectors than its header says; trust the bytes.
        let n_avail = (mmap.len() - 8) / (d * 4);
        let mut n = n_hdr.min(n_avail);
        if let Some(l) = limit {
            if l > n {
                return Err(std::io::Error::other(format!(
                    "asked for {l} vectors but file only has {n}"
                )));
            }
            n = l;
        }
        // Random access pattern: tell the kernel not to read ahead.
        #[cfg(unix)]
        let _ = mmap.advise(memmap2::Advice::Random);
        Ok(Fbin { mmap, n, d })
    }

    /// All vectors as one flat slice (n*d floats).
    pub fn data(&self) -> &[f32] {
        let bytes = &self.mmap[8..8 + self.n * self.d * 4];
        assert_eq!(
            bytes.as_ptr() as usize % 4,
            0,
            "fbin data not 4-byte aligned"
        );
        // SAFETY: alignment checked above; f32 has no invalid bit patterns; the mapping lives
        // as long as `self`.
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, self.n * self.d) }
    }

    /// Tell the kernel we'll read the file front to back (build, verify): it then reads ahead in
    /// large blocks. The default set in `open` is Random, right for search but very slow for a
    /// full scan of a file bigger than RAM.
    pub fn advise_sequential(&self) {
        #[cfg(unix)]
        let _ = self.mmap.advise(memmap2::Advice::Sequential);
    }

    /// Ask the kernel to start reading vector `i` from disk now, without waiting for it.
    /// Issuing this for every rerank candidate before touching any of them lets the SSD serve
    /// hundreds of reads in parallel instead of one page fault at a time.
    #[inline]
    pub fn prefetch(&self, i: usize) {
        #[cfg(unix)]
        let _ = self
            .mmap
            .advise_range(memmap2::Advice::WillNeed, 8 + i * self.d * 4, self.d * 4);
    }

    #[inline]
    pub fn vector(&self, i: usize) -> &[f32] {
        &self.data()[i * self.d..(i + 1) * self.d]
    }
}

pub struct GroundTruth {
    pub nq: usize,
    pub k: usize,
    pub ids: Vec<u32>,
}

impl GroundTruth {
    pub fn read(path: &Path) -> Result<Self> {
        let mut f = File::open(path)?;
        let mut hdr = [0u8; 8];
        f.read_exact(&mut hdr)?;
        let nq = u32::from_le_bytes(hdr[0..4].try_into().unwrap()) as usize;
        let k = u32::from_le_bytes(hdr[4..8].try_into().unwrap()) as usize;
        let mut buf = vec![0u8; nq * k * 4];
        f.read_exact(&mut buf)?;
        let ids = buf
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        Ok(GroundTruth { nq, k, ids })
    }

    pub fn row(&self, q: usize) -> &[u32] {
        &self.ids[q * self.k..(q + 1) * self.k]
    }
}

pub fn write_fbin(path: &Path, n: usize, d: usize, data: &[f32]) -> Result<()> {
    assert_eq!(data.len(), n * d);
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(&(n as u32).to_le_bytes())?;
    w.write_all(&(d as u32).to_le_bytes())?;
    for x in data {
        w.write_all(&x.to_le_bytes())?;
    }
    w.flush()
}

pub fn write_gt(path: &Path, nq: usize, k: usize, ids: &[u32], dists: &[f32]) -> Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(&(nq as u32).to_le_bytes())?;
    w.write_all(&(k as u32).to_le_bytes())?;
    for x in ids {
        w.write_all(&x.to_le_bytes())?;
    }
    for x in dists {
        w.write_all(&x.to_le_bytes())?;
    }
    w.flush()
}
