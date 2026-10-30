//! binvec — approximate nearest neighbour search that fits a large vector set on one machine.
//!
//! Pipeline for one query (see `ivf.rs`):
//! 1. compare the query with every cluster centroid, keep the `nprobe` closest clusters;
//! 2. inside those clusters, score every vector cheaply from its 1-bit-per-dimension code;
//! 3. keep the best `rerank` candidates and rescore them exactly from the full vectors.

pub mod buf;
pub mod dist;
pub mod fastscan;
pub mod filter;
pub mod io;
pub mod ivf;
pub mod kmeans;
pub mod refine;
pub mod rng;
pub mod topk;
