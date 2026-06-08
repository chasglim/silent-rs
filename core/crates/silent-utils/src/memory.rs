//! Memory pool implementation inspired by Microsoft SEAL.
//!
//! This module provides a thread-safe memory pool for managing allocations of `u64` vectors.
//! It aims to reduce the overhead of frequent allocations and deallocations in heavy
//! arithmetic operations.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
// Not used but good to have

thread_local! {
    static GLOBAL_POOL: RefCell<MemoryPool> = RefCell::new(MemoryPool::new());
}

/// A handle to the global thread-local memory pool.
#[derive(Clone, Debug)]
pub struct MemoryPoolHandle;

impl MemoryPoolHandle {
    /// Returns a handle to the global memory pool.
    pub fn new() -> Self {
        Self
    }

    /// Acquires a pointer to a block of memory.
    pub fn acquire(&self, size: usize) -> Pointer {
        GLOBAL_POOL.with(|pool| {
            let mut pool = pool.borrow_mut();
            let buffer = pool.get(size);
            Pointer { data: Some(buffer) }
        })
    }
}

impl Default for MemoryPoolHandle {
    fn default() -> Self {
        Self::new()
    }
}

/// A smart pointer that wraps a vector allocation and returns it to the pool on drop.
#[derive(Debug)]
pub struct Pointer {
    data: Option<Vec<u64>>,
}

impl Pointer {
    /// extract the inner vector, consuming the pointer (no return to pool).
    pub fn take(mut self) -> Vec<u64> {
        self.data.take().unwrap()
    }
}

impl Deref for Pointer {
    type Target = Vec<u64>;

    fn deref(&self) -> &Self::Target {
        self.data.as_ref().unwrap()
    }
}

impl DerefMut for Pointer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.data.as_mut().unwrap()
    }
}

impl Drop for Pointer {
    fn drop(&mut self) {
        if let Some(data) = self.data.take() {
            // Return to CURRENT thread's pool
            GLOBAL_POOL.with(|pool| {
                let mut pool = pool.borrow_mut();
                // Optimization: Don't clear. Keep data "dirty" to avoid re-initialization cost on reuse.
                // data.clear();
                pool.return_to_pool(data);
            });
        }
    }
}

#[derive(Debug)]
struct MemoryPool {
    pools: HashMap<usize, Vec<Vec<u64>>>,
}

impl MemoryPool {
    fn new() -> Self {
        Self {
            pools: HashMap::new(),
        }
    }

    fn get(&mut self, size: usize) -> Vec<u64> {
        // Optimization: Use `size` as the key for simple bucketing.
        // We want to return a vector with `len() == size` but *uninitialized* content (dirty).

        if let Some(bucket) = self.pools.get_mut(&size) {
            if let Some(mut vec) = bucket.pop() {
                // Found a cached vector.
                // Optimistically assume capacity is sufficient (it should be if key matches).
                // Ensure length is correct.
                unsafe {
                    // SAFETY: u64 Is POD. We don't care about the content.
                    // If cached, capacity >= size.
                    // If newly allocated below, capacity == size.
                    if vec.capacity() < size {
                        // This case shouldn't happen if we bucket strictly, but handle it safe-ish fallback?
                        // Just realloc if needed (slow path).
                        vec.reserve(size - vec.len());
                    }
                    vec.set_len(size);
                }
                return vec;
            }
        }

        // Alloc new without zeroing
        let mut vec = Vec::with_capacity(size);
        unsafe {
            vec.set_len(size);
        }
        vec
    }

    fn return_to_pool(&mut self, vec: Vec<u64>) {
        // When returning, we keep the allocation "hot".
        // Use capacity as key? Or original size?
        // Our 'get' looks up by requested 'size'.
        // If we bucket by `capacity`, we might give a 1000-cap vec to a 10-size request, which is fine but maybe wasteful?
        // SEAL buckets by "byte count".
        // Let's bucket by *current capacity* (rounded?) or just `len` (requested size)?
        // If we bucket by `vec.len()` (the size it was used for), it's perfect for exact reuse.
        // But if we used a large vec for a small task (len was shrunk), and put it back, it goes to small bucket.
        // Next time 'get(small)' picks it up (good).
        // But if 'get(large)' needs it, it won't find it.
        // Optimizing this perfectly requires "best fit".
        // For now, let's just bucket by `vec.len()` (the released size).
        // Assuming repetitive workloads (like NTT) ask for same size repeatedly.

        // IMPORTANT: Don't shrink capacity!
        // Just store it.
        let key = vec.len();
        self.pools.entry(key).or_insert_with(Vec::new).push(vec);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool_reuse() {
        let pool = MemoryPoolHandle::new();
        {
            let mut ptr1 = pool.acquire(100);
            ptr1[0] = 42;
            assert_eq!(ptr1.len(), 100);
            let _ptr = ptr1.as_ptr();

            // ptr1 drops here, returning to pool
        }

        {
            let ptr2 = pool.acquire(100);
            assert_eq!(ptr2.len(), 100);
            // We can't easily assert reuse without introspection, but compilation ensures it runs.
        }
    }

    #[test]
    fn test_multiple_allocations() {
        let pool = MemoryPoolHandle::new();
        let mut ptrs = Vec::new();
        for _ in 0..10 {
            ptrs.push(pool.acquire(50));
        }
        // All active.
    }
}
