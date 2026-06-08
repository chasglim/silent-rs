//! Simple bump arena for scratch allocations.

#[derive(Debug)]
pub struct Arena<T: Copy + Default> {
    buffer: Vec<T>,
    offset: usize,
}

impl<T: Copy + Default> Arena<T> {
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            offset: 0,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: vec![T::default(); capacity],
            offset: 0,
        }
    }

    pub fn reset(&mut self) {
        self.offset = 0;
    }

    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }

    pub fn alloc(&mut self, len: usize) -> &mut [T] {
        if len == 0 {
            return &mut [];
        }
        let start = self.offset;
        let end = start + len;
        if end > self.buffer.len() {
            self.buffer.resize(end, T::default());
        }
        self.offset = end;
        let ptr = self.buffer.as_mut_ptr();
        // Safety: bump allocation ensures disjoint slices until `reset`.
        unsafe { std::slice::from_raw_parts_mut(ptr.add(start), len) }
    }

    pub fn alloc_zeroed(&mut self, len: usize) -> &mut [T] {
        let slice = self.alloc(len);
        slice.fill(T::default());
        slice
    }
}

impl<T: Copy + Default> Default for Arena<T> {
    fn default() -> Self {
        Self::new()
    }
}

use crate::memory::{MemoryPoolHandle, Pointer};

/// A bump arena that uses a memory pool for the underlying buffer.
/// Specialized for u64 since simple MemoryPool is u64-only.
#[derive(Debug)]
pub struct PooledArena {
    ptr: Option<Pointer>,
    offset: usize,
    capacity: usize,
    pool: MemoryPoolHandle,
}

impl PooledArena {
    pub fn new(pool: MemoryPoolHandle, capacity: usize) -> Self {
        let ptr = pool.acquire(capacity);
        Self {
            ptr: Some(ptr),
            offset: 0,
            capacity,
            pool,
        }
    }

    pub fn reset(&mut self) {
        self.offset = 0;
    }

    pub fn alloc(&mut self, len: usize) -> &mut [u64] {
        if len == 0 {
            return &mut [];
        }
        let start = self.offset;
        let end = start + len;

        // If out of capacity, we have to reallocate.
        // Simple strategy: acquire new larger buffer, copy old data?
        // Or just panic/fail? Scratch allocators usually expect strict bounds.
        // But the original Arena resizes.
        if end > self.capacity {
            let new_cap = end.max(self.capacity * 2);
            let mut new_ptr = self.pool.acquire(new_cap);
            // Copy existing data if needed (scratch usually doesn't need preservation across allocs but alloc expects returned slice to be valid... wait, previous slices become invalid if we move buffer?)
            // The original Arena returns `&mut [T]` which borrows from `self`.
            // So if `self.buffer` moves, it's a compile error due to borrow checker... effectively preventing realloc while slices are out?
            // Actually, `alloc` takes `&mut self`. So you can't hold a reference to a previous alloc and call `alloc` again?
            // Yes, standard Rust borrow rules.
            // But wait, `alloc_two` allows getting two slices.
            // If I call `alloc`, I get `&mut []`. I can't call `alloc` again until that borrow ends.
            // So scratch is sequential.
            // BUT `RnsToolScratch` uses `alloc` and `alloc_two`.
            // If I resize, `new_ptr` is fresh.
            // I should copy data? Usually scratch doesn't rely on persistence *between* alloc calls if it's strictly bump.
            // But if I reallocate, the OLD `Pointer` is dropped and returned to pool.

            // For now, let's implement resize by copy to be safe.
            if let Some(old_ptr) = &self.ptr {
                new_ptr[..self.offset].copy_from_slice(&old_ptr[..self.offset]);
            }
            self.ptr = Some(new_ptr);
            self.capacity = new_cap;
        }

        self.offset = end;
        // Safety: we just ensured capacity.
        // We need to return a slice from `self.ptr`.
        let slice = &mut self.ptr.as_mut().unwrap()[start..end];
        // We need to extend the lifetime to match `&mut self`.
        // The borrow checker handles this naturally.
        unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr(), len) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_allocates_non_overlapping() {
        let mut arena = Arena::<u64>::new();
        let slice = arena.alloc(8);
        let (a, b) = slice.split_at_mut(4);
        a.fill(1);
        b.fill(2);
        assert_eq!(a, &[1, 1, 1, 1]);
        assert_eq!(b, &[2, 2, 2, 2]);
    }

    #[test]
    fn arena_reset_reuses_capacity() {
        let mut arena = Arena::<u64>::with_capacity(8);
        let _ = arena.alloc(4);
        arena.reset();
        let _ = arena.alloc(6);
        assert!(arena.capacity() >= 8);
    }
}
