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

#[cfg(test)]
mod tests {
    use super::*;

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
