//! Arena and slab allocators for temporary tensor buffers.
//!
//! Provides `Arena` for bump-style allocation of temporary numeric slices,
//! and `SlabPool` for recycling fixed-size buffers. Both are designed to
//! eliminate per-allocation heap overhead in hot computation loops.

use std::cell::RefCell;
use std::marker::PhantomData;
use std::mem;

/// A simple arena allocator for temporary numeric buffers.
///
/// Allocates from a pre-allocated block of memory, avoiding per-allocation
/// heap overhead. Suitable for temporary tensors in computation loops.
///
/// # Examples
///
/// ```
/// use scivex_core::arena::Arena;
///
/// let arena = Arena::new(1024); // 1024 f64 slots
/// let buf1 = arena.alloc::<f64>(100).unwrap();
/// let buf2 = arena.alloc::<f64>(200).unwrap();
/// assert_eq!(buf1.len(), 100);
/// assert_eq!(buf2.len(), 200);
/// arena.reset(); // reclaim all memory instantly
/// ```
pub struct Arena {
    /// Raw byte storage.
    data: RefCell<Vec<u8>>,
    /// Current offset into the data buffer.
    offset: RefCell<usize>,
    /// Total capacity in bytes.
    capacity: usize,
}

/// A borrowed slice from the arena.
///
/// The arena must outlive this handle. Dereferences to `&[T]` / `&mut [T]`
/// for ergonomic access.
pub struct ArenaSlice<'a, T> {
    ptr: *mut T,
    len: usize,
    _marker: PhantomData<&'a mut T>,
}

// SAFETY: ArenaSlice does not implement Send/Sync because Arena uses RefCell,
// which is !Sync. The PhantomData<&'a mut T> correctly limits the lifetime.

impl Arena {
    /// Create a new arena with capacity for `n_elements` `f64` values.
    ///
    /// The actual byte capacity is `n_elements * size_of::<f64>()`.
    pub fn new(n_elements: usize) -> Self {
        Self::with_byte_capacity(n_elements * mem::size_of::<f64>())
    }

    /// Create an arena with the given byte capacity.
    pub fn with_byte_capacity(bytes: usize) -> Self {
        let data = vec![0u8; bytes];
        Self {
            data: RefCell::new(data),
            offset: RefCell::new(0),
            capacity: bytes,
        }
    }

    /// Allocate a slice of `count` elements of type `T` from the arena.
    ///
    /// Returns `None` if the arena doesn't have enough space.
    /// Elements are zero-initialized.
    pub fn alloc<T: Default + Copy>(&self, count: usize) -> Option<ArenaSlice<'_, T>> {
        let size = mem::size_of::<T>();
        let align = mem::align_of::<T>();
        let total_bytes = size * count;

        let mut offset = self.offset.borrow_mut();

        // Round up to the required alignment.
        let aligned = (*offset + align - 1) & !(align - 1);

        if aligned + total_bytes > self.capacity {
            return None;
        }

        let data = self.data.borrow_mut();
        let base_ptr = data.as_ptr();

        // SAFETY: `aligned + total_bytes <= self.capacity` is checked above.
        // The data vec has been allocated and zeroed to `self.capacity` bytes.
        // The pointer arithmetic stays within the vec's allocation.
        // We ensure proper alignment by rounding `aligned` up to `align_of::<T>()`.
        let ptr = unsafe { base_ptr.add(aligned) as *mut T };

        // Zero-initialize the allocated region (may have stale data after reset).
        // SAFETY: Same bounds justification as above. The region
        // [aligned..aligned+total_bytes) is within the vec's allocation.
        unsafe {
            std::ptr::write_bytes(ptr, 0, count);
        }

        *offset = aligned + total_bytes;

        Some(ArenaSlice {
            ptr,
            len: count,
            _marker: PhantomData,
        })
    }

    /// Allocate a slice and initialize it from an existing slice.
    ///
    /// Returns `None` if the arena doesn't have enough space.
    pub fn alloc_copy<T: Copy>(&self, src: &[T]) -> Option<ArenaSlice<'_, T>> {
        let _size = mem::size_of::<T>();
        let align = mem::align_of::<T>();
        let total_bytes = std::mem::size_of_val(src);

        let mut offset = self.offset.borrow_mut();

        let aligned = (*offset + align - 1) & !(align - 1);

        if aligned + total_bytes > self.capacity {
            return None;
        }

        let data = self.data.borrow_mut();
        let base_ptr = data.as_ptr();

        // SAFETY: `aligned + total_bytes <= self.capacity` is checked above.
        // The data vec is allocated to at least `self.capacity` bytes.
        // Alignment is ensured by rounding up to `align_of::<T>()`.
        let ptr = unsafe { base_ptr.add(aligned) as *mut T };

        // SAFETY: `ptr` points to `src.len()` elements worth of valid,
        // aligned memory within the arena's allocation. Source and destination
        // do not overlap because `src` is an external slice.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), ptr, src.len());
        }

        *offset = aligned + total_bytes;

        Some(ArenaSlice {
            ptr,
            len: src.len(),
            _marker: PhantomData,
        })
    }

    /// Reset the arena, making all previously allocated memory available again.
    ///
    /// This is O(1) — just resets the offset pointer. Previously returned
    /// `ArenaSlice` handles become invalid (enforced by lifetimes).
    pub fn reset(&self) {
        *self.offset.borrow_mut() = 0;
    }

    /// Returns the number of bytes currently in use.
    pub fn used_bytes(&self) -> usize {
        *self.offset.borrow()
    }

    /// Returns the total capacity in bytes.
    pub fn capacity_bytes(&self) -> usize {
        self.capacity
    }

    /// Returns the number of bytes remaining.
    pub fn remaining_bytes(&self) -> usize {
        self.capacity - *self.offset.borrow()
    }
}

impl<T> ArenaSlice<'_, T> {
    /// Returns an immutable slice view of the arena allocation.
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: The pointer was obtained from a valid arena allocation of
        // `self.len` elements. The arena's lifetime is tied to this slice
        // through the `'a` lifetime parameter, ensuring the memory remains
        // valid. The elements were properly initialized (zeroed or copied).
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Returns a mutable slice view of the arena allocation.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: Same as `as_slice`, plus we hold `&mut self` guaranteeing
        // exclusive access to this allocation region.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// Returns the number of elements in this slice.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the slice has zero elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T> std::ops::Deref for ArenaSlice<'_, T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> std::ops::DerefMut for ArenaSlice<'_, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

// ---------------------------------------------------------------------------
// SlabPool
// ---------------------------------------------------------------------------

/// Default slab size classes (element counts).
const DEFAULT_SIZE_CLASSES: &[usize] = &[64, 256, 1024, 4096, 16384, 65536];

/// A pool of fixed-size slabs for common tensor sizes.
///
/// Pre-allocates slabs of common sizes (e.g., 64, 256, 1024, 4096 elements)
/// and hands them out on request, recycling them when returned.
///
/// # Examples
///
/// ```
/// use scivex_core::arena::SlabPool;
///
/// let pool: SlabPool<f64> = SlabPool::new();
/// let mut buf = pool.acquire(100); // gets a 256-element slab
/// assert!(buf.capacity() >= 100);
/// buf[0] = 42.0;
/// pool.release(buf);
/// // The next acquire of a similar size reuses the released buffer.
/// let buf2 = pool.acquire(100);
/// assert!(buf2.capacity() >= 100);
/// ```
pub struct SlabPool<T: Copy> {
    /// `slabs[i]` holds available buffers for `size_classes[i]`.
    slabs: RefCell<Vec<Vec<Vec<T>>>>,
    /// Sorted list of slab element counts.
    size_classes: Vec<usize>,
}

impl<T: Copy + Default> SlabPool<T> {
    /// Create a new slab pool with default size classes
    /// (64, 256, 1024, 4096, 16384, 65536).
    pub fn new() -> Self {
        Self::with_sizes(DEFAULT_SIZE_CLASSES)
    }

    /// Create a pool with custom size classes.
    ///
    /// The provided sizes are sorted internally.
    pub fn with_sizes(sizes: &[usize]) -> Self {
        let mut size_classes: Vec<usize> = sizes.to_vec();
        size_classes.sort_unstable();

        let slabs = RefCell::new(size_classes.iter().map(|_| Vec::new()).collect());

        Self {
            slabs,
            size_classes,
        }
    }

    /// Acquire a buffer of at least `min_size` elements.
    ///
    /// If a recycled slab of suitable size is available, it is returned
    /// (zero-cleared). Otherwise a fresh `Vec` is allocated.
    pub fn acquire(&self, min_size: usize) -> Vec<T> {
        let class_idx = self.size_class_index(min_size);

        let mut slabs = self.slabs.borrow_mut();

        if let Some(idx) = class_idx {
            if let Some(mut buf) = slabs[idx].pop() {
                // Clear for reuse.
                buf.clear();
                buf.resize(self.size_classes[idx], T::default());
                return buf;
            }
            // No recycled slab — allocate a fresh one at this size class.
            let cap = self.size_classes[idx];
            let mut buf = Vec::with_capacity(cap);
            buf.resize(cap, T::default());
            buf
        } else {
            // Requested size exceeds all size classes — allocate exactly.
            let mut buf = Vec::with_capacity(min_size);
            buf.resize(min_size, T::default());
            buf
        }
    }

    /// Return a buffer to the pool for reuse.
    ///
    /// If the buffer's capacity doesn't match any size class it is simply
    /// dropped.
    pub fn release(&self, buf: Vec<T>) {
        let cap = buf.capacity();
        let class_idx = self.size_classes.iter().position(|&s| s == cap);

        if let Some(idx) = class_idx {
            self.slabs.borrow_mut()[idx].push(buf);
        }
        // If it doesn't match a size class, just drop it.
    }

    /// Number of available (recycled) slabs across all size classes.
    pub fn available_count(&self) -> usize {
        self.slabs.borrow().iter().map(std::vec::Vec::len).sum()
    }

    /// Find the index of the smallest size class >= `min_size`.
    fn size_class_index(&self, min_size: usize) -> Option<usize> {
        self.size_classes.iter().position(|&s| s >= min_size)
    }
}

impl<T: Copy + Default> Default for SlabPool<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_basic() {
        let arena = Arena::new(256);
        let mut buf = arena.alloc::<f64>(10).expect("allocation should succeed");
        assert_eq!(buf.len(), 10);

        // Write and read back.
        for i in 0..10 {
            buf[i] = i as f64;
        }
        for i in 0..10 {
            assert!((buf[i] - i as f64).abs() < 1e-15);
        }
    }

    #[test]
    fn test_arena_multiple_allocs() {
        let arena = Arena::new(512);
        let a = arena.alloc::<f64>(100).expect("first alloc");
        let b = arena.alloc::<f64>(100).expect("second alloc");
        let c = arena.alloc::<f64>(100).expect("third alloc");

        assert_eq!(a.len(), 100);
        assert_eq!(b.len(), 100);
        assert_eq!(c.len(), 100);

        // All should be zero-initialized.
        for &v in a.as_slice() {
            assert!((v - 0.0_f64).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_arena_overflow() {
        let arena = Arena::new(10); // 10 f64 = 80 bytes
        let ok = arena.alloc::<f64>(10);
        assert!(ok.is_some());

        // Arena is full — next allocation should fail.
        let fail = arena.alloc::<f64>(1);
        assert!(fail.is_none());
    }

    #[test]
    fn test_arena_reset() {
        let arena = Arena::new(64);

        let _ = arena.alloc::<f64>(60).expect("alloc before reset");
        assert!(arena.alloc::<f64>(10).is_none(), "should be full");

        arena.reset();
        assert_eq!(arena.used_bytes(), 0);

        let buf = arena.alloc::<f64>(60).expect("alloc after reset");
        assert_eq!(buf.len(), 60);
    }

    #[test]
    fn test_slab_pool_acquire_release() {
        let pool: SlabPool<f64> = SlabPool::new();

        let mut buf = pool.acquire(100);
        assert!(buf.capacity() >= 100);
        buf[0] = 99.0;

        let cap = buf.capacity();
        pool.release(buf);
        assert_eq!(pool.available_count(), 1);

        // Re-acquire should reuse the released buffer.
        let buf2 = pool.acquire(100);
        assert_eq!(buf2.capacity(), cap);
        // Should be cleared.
        assert!((buf2[0] - 0.0_f64).abs() < f64::EPSILON);
    }

    #[test]
    fn test_slab_pool_size_class_selection() {
        let pool: SlabPool<f64> = SlabPool::new();

        // Requesting 100 elements should yield a slab from the 256 size class.
        let buf = pool.acquire(100);
        assert_eq!(buf.len(), 256);
        assert!(buf.capacity() >= 256);

        // Requesting 64 should yield exactly 64.
        let buf2 = pool.acquire(64);
        assert_eq!(buf2.len(), 64);

        // Requesting more than max size class gets exact allocation.
        let buf3 = pool.acquire(100_000);
        assert_eq!(buf3.len(), 100_000);
    }

    #[test]
    fn test_arena_alloc_copy() {
        let arena = Arena::new(256);
        let src = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let buf = arena.alloc_copy(&src).expect("alloc_copy");
        assert_eq!(buf.len(), 5);
        assert_eq!(buf.as_slice(), &src);
    }

    #[test]
    fn test_arena_alignment() {
        let arena = Arena::with_byte_capacity(256);

        // Allocate a u8, then a f64 — the f64 must be properly aligned.
        let _ = arena.alloc::<u8>(1).unwrap();
        let f = arena.alloc::<f64>(1).unwrap();
        let ptr = f.as_slice().as_ptr() as usize;
        assert_eq!(
            ptr % mem::align_of::<f64>(),
            0,
            "f64 must be 8-byte aligned"
        );
    }
}
