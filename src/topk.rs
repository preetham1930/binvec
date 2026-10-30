//! Keep the k smallest (distance, id) pairs seen so far.
//!
//! A max-heap of size k: the root is the worst of the current best k, so a new candidate
//! only needs one comparison with the root to be rejected — the common case once the heap
//! is warm.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Clone, Copy, Debug)]
pub struct Hit {
    pub dist: f32,
    pub id: u32,
}

impl PartialEq for Hit {
    fn eq(&self, o: &Self) -> bool {
        self.dist == o.dist && self.id == o.id
    }
}
impl Eq for Hit {}
impl PartialOrd for Hit {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Hit {
    fn cmp(&self, o: &Self) -> Ordering {
        self.dist.total_cmp(&o.dist).then(self.id.cmp(&o.id))
    }
}

pub struct TopK {
    k: usize,
    heap: BinaryHeap<Hit>,
}

impl TopK {
    pub fn new(k: usize) -> Self {
        TopK {
            k,
            heap: BinaryHeap::with_capacity(k + 1),
        }
    }

    /// Distance a new candidate must beat to get in (infinity until the heap is full).
    #[inline]
    pub fn threshold(&self) -> f32 {
        if self.heap.len() < self.k {
            f32::INFINITY
        } else {
            self.heap.peek().map_or(f32::INFINITY, |h| h.dist)
        }
    }

    #[inline]
    pub fn push(&mut self, dist: f32, id: u32) {
        if self.heap.len() < self.k {
            self.heap.push(Hit { dist, id });
        } else if dist < self.heap.peek().unwrap().dist {
            *self.heap.peek_mut().unwrap() = Hit { dist, id };
        }
    }

    /// Results sorted from nearest to farthest.
    pub fn into_sorted(self) -> Vec<Hit> {
        self.heap.into_sorted_vec()
    }
}

/// Keep roughly the k smallest (distance, id) pairs, for large k.
///
/// A heap pays O(log k) and an unpredictable branch on every accepted candidate; with k in the
/// thousands, that cost showed up in the scan stage once the SIMD kernel made scoring cheap.
/// A reservoir just appends candidates below the current threshold into a buffer of 2k, and when
/// it fills, keeps the best k with one `select_nth_unstable` (O(k)) and tightens the threshold.
/// Amortised O(1) per accepted candidate. (The same idea as FAISS's reservoir collectors.)
pub struct Reservoir {
    k: usize,
    cap: usize,
    thr: f32,
    buf: Vec<Hit>,
}

impl Reservoir {
    pub fn new(k: usize) -> Self {
        let cap = (2 * k).max(64);
        Reservoir {
            k,
            cap,
            thr: f32::INFINITY,
            buf: Vec::with_capacity(cap),
        }
    }

    /// Candidates at or above this can be skipped.
    #[inline]
    pub fn threshold(&self) -> f32 {
        self.thr
    }

    #[inline]
    pub fn push(&mut self, dist: f32, id: u32) {
        if dist < self.thr {
            self.buf.push(Hit { dist, id });
            if self.buf.len() == self.cap {
                self.shrink();
            }
        }
    }

    fn shrink(&mut self) {
        if self.buf.len() > self.k {
            self.buf.select_nth_unstable_by(self.k - 1, |a, b| a.cmp(b));
            self.buf.truncate(self.k);
            // After the select, buf[k-1] is the k-th smallest and everything before it is ≤ it.
            self.thr = self.buf[self.k - 1].dist;
        }
    }

    /// The k best, in no particular order.
    pub fn into_unsorted(mut self) -> Vec<Hit> {
        self.shrink();
        self.buf
    }

    /// The k best, nearest first.
    pub fn into_sorted(mut self) -> Vec<Hit> {
        self.shrink();
        self.buf.sort_unstable();
        self.buf.truncate(self.k);
        self.buf
    }
}

/// Either collector, chosen at run time (the heap is kept for the before/after benchmark).
pub enum Collector {
    Heap(TopK),
    Reservoir(Reservoir),
}

impl Collector {
    pub fn new(k: usize, heap: bool) -> Self {
        if heap {
            Collector::Heap(TopK::new(k))
        } else {
            Collector::Reservoir(Reservoir::new(k))
        }
    }

    #[inline]
    pub fn threshold(&self) -> f32 {
        match self {
            Collector::Heap(h) => h.threshold(),
            Collector::Reservoir(r) => r.threshold(),
        }
    }

    #[inline]
    pub fn push(&mut self, dist: f32, id: u32) {
        match self {
            Collector::Heap(h) => h.push(dist, id),
            Collector::Reservoir(r) => r.push(dist, id),
        }
    }

    pub fn into_unsorted(self) -> Vec<Hit> {
        match self {
            Collector::Heap(h) => h.into_sorted(),
            Collector::Reservoir(r) => r.into_unsorted(),
        }
    }

    pub fn into_sorted(self) -> Vec<Hit> {
        match self {
            Collector::Heap(h) => h.into_sorted(),
            Collector::Reservoir(r) => r.into_sorted(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reservoir_matches_heap() {
        let mut rng = crate::rng::Rng::new(11);
        for &k in &[1usize, 10, 100, 1000] {
            let mut h = TopK::new(k);
            let mut r = Reservoir::new(k);
            for id in 0..20_000u32 {
                let d = rng.unit();
                h.push(d, id);
                r.push(d, id);
            }
            let a: Vec<u32> = h.into_sorted().iter().map(|x| x.id).collect();
            let mut u: Vec<u32> = r.into_unsorted().iter().map(|x| x.id).collect();
            let mut b = a.clone();
            b.sort_unstable();
            u.sort_unstable();
            assert_eq!(b, u, "k = {k} (unsorted set)");
        }
    }

    #[test]
    fn keeps_smallest() {
        let mut t = TopK::new(3);
        for (i, d) in [5.0, 1.0, 4.0, 2.0, 9.0, 0.5].iter().enumerate() {
            t.push(*d, i as u32);
        }
        let ids: Vec<u32> = t.into_sorted().iter().map(|h| h.id).collect();
        assert_eq!(ids, vec![5, 1, 3]);
    }
}
