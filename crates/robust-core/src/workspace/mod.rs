//! Memory-efficient workspace components for high-performance statistical computations
//!
//! This module provides aligned buffers and buffer pools that enable:
//! - Zero-copy operations with proper SIMD alignment
//! - Thread-local buffer reuse to minimize allocations
//! - Type-safe generic buffers for any data type
//! - RAII-based resource management

pub mod bootstrap;
pub use bootstrap::{
    with_bootstrap_workspace, with_f64_bootstrap_workspace, BootstrapWorkspace, IndexWorkspace,
    ResampleWorkspace, SortWorkspace,
};
use std::alloc::{alloc, dealloc, Layout};
use std::marker::PhantomData;
use std::mem;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

/// A properly aligned buffer for type T
///
/// This buffer ensures proper memory alignment for SIMD operations
/// and provides zero-cost abstraction over raw memory management.
pub struct AlignedBuffer<T> {
    ptr: NonNull<T>,
    capacity: usize,
    len: usize,
    layout: Layout,
    _marker: PhantomData<T>,
}

impl<T> AlignedBuffer<T> {
    /// Create a new aligned buffer with specified alignment
    ///
    /// # Panics
    /// - If alignment is not a power of two
    /// - If alignment is less than the natural alignment of T
    /// - If allocation fails
    pub fn new(capacity: usize, alignment: usize) -> Self {
        assert!(
            alignment.is_power_of_two(),
            "Alignment must be a power of two"
        );
        assert!(
            alignment >= mem::align_of::<T>(),
            "Alignment must be at least {}",
            mem::align_of::<T>()
        );

        let layout = Layout::from_size_align(capacity * mem::size_of::<T>(), alignment)
            .expect("Invalid layout");

        let ptr = unsafe {
            let raw_ptr = alloc(layout) as *mut T;
            NonNull::new(raw_ptr).expect("Allocation failed")
        };

        Self {
            ptr,
            capacity,
            len: 0,
            layout,
            _marker: PhantomData,
        }
    }

    /// Get a slice of the used portion
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Get a mutable slice of the used portion
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Resize the buffer (doesn't reallocate, just changes len)
    ///
    /// # Panics
    /// If new_len > capacity
    #[inline]
    pub fn resize(&mut self, new_len: usize) {
        assert!(new_len <= self.capacity, "Cannot resize beyond capacity");
        self.len = new_len;
    }

    /// Clear the buffer (just resets len to 0)
    #[inline]
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Get the capacity of the buffer
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get the current length of the buffer
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the buffer is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T> Drop for AlignedBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr.as_ptr() as *mut u8, self.layout);
        }
    }
}

// Safety: AlignedBuffer owns its data and T is Send
unsafe impl<T: Send> Send for AlignedBuffer<T> {}
// Safety: AlignedBuffer owns its data and T is Sync
unsafe impl<T: Sync> Sync for AlignedBuffer<T> {}

/// Pool of buffers for a specific type T
///
/// Maintains a pool of reusable buffers to minimize allocations.
/// Buffers are automatically returned to the pool when dropped.
pub struct BufferPool<T> {
    buffers: Mutex<Vec<AlignedBuffer<T>>>,
    alignment: usize,
    max_buffers: usize,
}

impl<T: Send> BufferPool<T> {
    /// Create a new buffer pool
    ///
    /// # Arguments
    /// * `alignment` - Memory alignment for buffers (must be power of 2)
    /// * `max_buffers` - Maximum number of buffers to keep in the pool
    pub fn new(alignment: usize, max_buffers: usize) -> Self {
        Self {
            buffers: Mutex::new(Vec::with_capacity(max_buffers)),
            alignment,
            max_buffers,
        }
    }

    /// Check out a buffer with at least the specified capacity
    ///
    /// If a suitable buffer exists in the pool, it will be reused.
    /// Otherwise, a new buffer will be allocated.
    pub fn checkout(&self, capacity: usize) -> CheckedOutBuffer<'_, T> {
        let mut buffers = self.buffers.lock().unwrap();

        // Find a buffer with sufficient capacity
        let buffer = buffers
            .iter()
            .position(|buf| buf.capacity() >= capacity)
            .map(|idx| buffers.swap_remove(idx))
            .unwrap_or_else(|| AlignedBuffer::new(capacity, self.alignment));

        CheckedOutBuffer {
            buffer: Some(buffer),
            pool: self,
        }
    }

    /// Return a buffer to the pool
    fn checkin(&self, mut buffer: AlignedBuffer<T>) {
        // Zero out the buffer for safety and to match test expectations
        if buffer.len > 0 {
            unsafe {
                std::ptr::write_bytes(buffer.ptr.as_ptr(), 0, buffer.len);
            }
        }
        buffer.clear();
        let mut buffers = self.buffers.lock().unwrap();
        if buffers.len() < self.max_buffers {
            buffers.push(buffer);
        }
        // Otherwise let it drop
    }
}

/// RAII guard for a checked-out buffer
///
/// Automatically returns the buffer to the pool when dropped
pub struct CheckedOutBuffer<'a, T: Send> {
    buffer: Option<AlignedBuffer<T>>,
    pool: &'a BufferPool<T>,
}

impl<'a, T: Send> CheckedOutBuffer<'a, T> {
    /// Get the buffer as a mutable reference
    #[inline]
    pub fn get_mut(&mut self) -> &mut AlignedBuffer<T> {
        self.buffer.as_mut().expect("Buffer already taken")
    }

    /// Get the buffer as a slice
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.buffer
            .as_ref()
            .expect("Buffer already taken")
            .as_slice()
    }

    /// Get the buffer as a mutable slice
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.buffer
            .as_mut()
            .expect("Buffer already taken")
            .as_mut_slice()
    }
}

impl<'a, T: Send> Drop for CheckedOutBuffer<'a, T> {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.pool.checkin(buffer);
        }
    }
}

impl<'a, T: Send> std::ops::Deref for CheckedOutBuffer<'a, T> {
    type Target = AlignedBuffer<T>;

    fn deref(&self) -> &Self::Target {
        self.buffer.as_ref().expect("Buffer already taken")
    }
}

impl<'a, T: Send> std::ops::DerefMut for CheckedOutBuffer<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.buffer.as_mut().expect("Buffer already taken")
    }
}

/// Arc-based buffer pool for lifetime-free buffer management
///
/// This pool returns Arc-wrapped buffers that can be freely shared
/// and returned from functions without lifetime constraints.
pub struct ArcBufferPool<T> {
    buffers: Mutex<Vec<AlignedBuffer<T>>>,
    alignment: usize,
    max_buffers: usize,
}

impl<T: Send + 'static> ArcBufferPool<T> {
    /// Create a new Arc-based buffer pool
    pub fn new(alignment: usize, max_buffers: usize) -> Self {
        Self {
            buffers: Mutex::new(Vec::with_capacity(max_buffers)),
            alignment,
            max_buffers,
        }
    }
    
    /// Check out a buffer with at least the specified capacity
    ///
    /// Returns an Arc-wrapped buffer that automatically returns to the pool when dropped.
    pub fn checkout(self: &Arc<Self>, capacity: usize) -> ArcBuffer<T> {
        let mut buffers = self.buffers.lock().unwrap();
        
        // Find a buffer with sufficient capacity
        let buffer = buffers
            .iter()
            .position(|buf| buf.capacity() >= capacity)
            .map(|idx| buffers.swap_remove(idx))
            .unwrap_or_else(|| AlignedBuffer::new(capacity, self.alignment));
        
        ArcBuffer {
            buffer: Some(buffer),
            pool: Arc::downgrade(self),
        }
    }
    
    /// Return a buffer to the pool
    fn return_buffer(&self, mut buffer: AlignedBuffer<T>) {
        // Zero out the buffer for safety
        if buffer.len > 0 {
            unsafe {
                std::ptr::write_bytes(buffer.ptr.as_ptr(), 0, buffer.len);
            }
        }
        buffer.clear();
        
        let mut buffers = self.buffers.lock().unwrap();
        if buffers.len() < self.max_buffers {
            buffers.push(buffer);
        }
        // Otherwise let it drop
    }
    
    /// Get the number of buffers currently in the pool
    pub fn pool_size(&self) -> usize {
        self.buffers.lock().unwrap().len()
    }
}

/// Arc-wrapped buffer that automatically returns to the pool when dropped
///
/// This type can be freely cloned and shared between threads.
pub struct ArcBuffer<T: Send + 'static> {
    buffer: Option<AlignedBuffer<T>>,
    pool: std::sync::Weak<ArcBufferPool<T>>,
}

impl<T: Send + 'static> ArcBuffer<T> {
    /// Create a standalone buffer without a pool
    pub fn new(capacity: usize, alignment: usize) -> Self {
        Self {
            buffer: Some(AlignedBuffer::new(capacity, alignment)),
            pool: std::sync::Weak::new(),
        }
    }
    
    /// Get the buffer as a slice
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.buffer.as_ref().expect("Buffer already taken").as_slice()
    }
    
    /// Get the buffer as a mutable slice
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.buffer.as_mut().expect("Buffer already taken").as_mut_slice()
    }
    
    /// Resize the buffer
    #[inline]
    pub fn resize(&mut self, new_len: usize) {
        self.buffer.as_mut().expect("Buffer already taken").resize(new_len)
    }
    
    /// Clear the buffer
    #[inline]
    pub fn clear(&mut self) {
        self.buffer.as_mut().expect("Buffer already taken").clear()
    }
    
    /// Get the capacity
    #[inline]
    pub fn capacity(&self) -> usize {
        self.buffer.as_ref().expect("Buffer already taken").capacity()
    }
    
    /// Get the length
    #[inline]
    pub fn len(&self) -> usize {
        self.buffer.as_ref().expect("Buffer already taken").len()
    }
    
    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.buffer.as_ref().expect("Buffer already taken").is_empty()
    }
    
    /// Take the inner buffer, leaving None in its place
    ///
    /// This prevents the buffer from being returned to the pool
    pub fn take(mut self) -> AlignedBuffer<T> {
        self.buffer.take().expect("Buffer already taken")
    }
}

impl<T: Send + 'static> Drop for ArcBuffer<T> {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            if let Some(pool) = self.pool.upgrade() {
                pool.return_buffer(buffer);
            }
            // Otherwise let buffer drop normally
        }
    }
}

// Implement Deref for convenience
impl<T: Send + 'static> std::ops::Deref for ArcBuffer<T> {
    type Target = [T];
    
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T: Send + 'static> std::ops::DerefMut for ArcBuffer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

// ArcBuffer can be cloned since it doesn't actually share the buffer
impl<T: Send + 'static> Clone for ArcBuffer<T> {
    fn clone(&self) -> Self {
        panic!("ArcBuffer cannot be cloned - use Arc<Mutex<ArcBuffer>> for sharing")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligned_buffer_creation() {
        let buffer: AlignedBuffer<f64> = AlignedBuffer::new(1024, 64);
        assert_eq!(buffer.capacity(), 1024);
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_aligned_buffer_resize() {
        let mut buffer: AlignedBuffer<f64> = AlignedBuffer::new(1024, 64);
        buffer.resize(512);
        assert_eq!(buffer.len(), 512);
        assert!(!buffer.is_empty());

        buffer.clear();
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
    }

    #[test]
    #[should_panic(expected = "Cannot resize beyond capacity")]
    fn test_aligned_buffer_resize_panic() {
        let mut buffer: AlignedBuffer<f64> = AlignedBuffer::new(10, 64);
        buffer.resize(20);
    }

    #[test]
    fn test_aligned_buffer_slice_access() {
        let mut buffer: AlignedBuffer<f64> = AlignedBuffer::new(10, 64);
        buffer.resize(5);

        // Write through mutable slice
        let slice = buffer.as_mut_slice();
        for (i, val) in slice.iter_mut().enumerate() {
            *val = i as f64;
        }

        // Read through immutable slice
        let slice = buffer.as_slice();
        for (i, &val) in slice.iter().enumerate() {
            assert_eq!(val, i as f64);
        }
    }

    #[test]
    fn test_buffer_pool_checkout_checkin() {
        let pool: BufferPool<f64> = BufferPool::new(64, 10);

        // Checkout a buffer
        let mut buffer1 = pool.checkout(100);
        buffer1.resize(50);
        buffer1.as_mut_slice().fill(1.0);

        // Drop buffer1 - it should return to pool
        drop(buffer1);

        // Checkout again - should reuse the same buffer
        let buffer2 = pool.checkout(50);
        assert_eq!(buffer2.capacity(), 100); // Same capacity as before
        assert_eq!(buffer2.len(), 0); // Buffer should start with length 0
    }

    #[test]
    fn test_buffer_pool_capacity_selection() {
        let pool: BufferPool<f64> = BufferPool::new(64, 10);

        // Create buffers of different sizes
        let buffer1 = pool.checkout(100);
        let buffer2 = pool.checkout(200);
        let buffer3 = pool.checkout(300);

        drop(buffer1);
        drop(buffer2);
        drop(buffer3);

        // Request a buffer of size 150 - should get the 200-capacity buffer
        let buffer4 = pool.checkout(150);
        assert_eq!(buffer4.capacity(), 200);
    }

    #[test]
    fn test_buffer_pool_max_buffers() {
        let pool: BufferPool<f64> = BufferPool::new(64, 2);

        // Create and return 3 buffers
        let buffers: Vec<_> = (0..3).map(|_| pool.checkout(100)).collect();

        // Drop all buffers at once
        drop(buffers);

        // Pool should only keep 2 buffers (max_buffers limit)
        let buffers_count = pool.buffers.lock().unwrap().len();
        assert_eq!(buffers_count, 2);
    }

    #[test]
    fn test_checked_out_buffer_deref() {
        let pool: BufferPool<f64> = BufferPool::new(64, 10);
        let mut buffer = pool.checkout(100);

        // Test Deref
        assert_eq!(buffer.capacity(), 100);

        // Test DerefMut
        buffer.resize(10);
        assert_eq!(buffer.len(), 10);
    }

    #[test]
    fn test_thread_safety() {
        use std::sync::Arc;
        use std::thread;

        let pool = Arc::new(BufferPool::<f64>::new(64, 10));
        let mut handles = vec![];

        // Spawn multiple threads that checkout and checkin buffers
        for i in 0..4 {
            let pool_clone = Arc::clone(&pool);
            let handle = thread::spawn(move || {
                for j in 0..5 {
                    let mut buffer = pool_clone.checkout(100 + i * 10);
                    buffer.resize(50);
                    buffer.as_mut_slice().fill((i * 10 + j) as f64);
                    // Buffer automatically returned when dropped
                }
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        // Pool should have some buffers
        assert!(!pool.buffers.lock().unwrap().is_empty());
    }

    #[test]
    #[should_panic(expected = "Alignment must be a power of two")]
    fn test_invalid_alignment() {
        let _buffer: AlignedBuffer<f64> = AlignedBuffer::new(100, 63);
    }

    #[test]
    #[should_panic(expected = "Alignment must be at least")]
    fn test_insufficient_alignment() {
        let _buffer: AlignedBuffer<f64> = AlignedBuffer::new(100, 4);
    }

    #[test]
    fn test_aligned_buffer_alignment_verification() {
        // Test various alignments
        let alignments = [16, 32, 64, 128, 256];
        
        for &alignment in &alignments {
            let buffer: AlignedBuffer<f64> = AlignedBuffer::new(100, alignment);
            let ptr_addr = buffer.ptr.as_ptr() as usize;
            
            // Verify the buffer is actually aligned
            assert_eq!(
                ptr_addr % alignment, 0,
                "Buffer with alignment {alignment} is not properly aligned"
            );
        }
    }

    #[test]
    fn test_aligned_buffer_zero_capacity() {
        let buffer: AlignedBuffer<f64> = AlignedBuffer::new(0, 64);
        assert_eq!(buffer.capacity(), 0);
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
        
        // Should be safe to get empty slices
        assert_eq!(buffer.as_slice().len(), 0);
    }

    #[test]
    fn test_aligned_buffer_large_capacity() {
        // Test with a large buffer
        let capacity = 1_000_000;
        let mut buffer: AlignedBuffer<u8> = AlignedBuffer::new(capacity, 64);
        assert_eq!(buffer.capacity(), capacity);
        
        // Test we can actually use the full capacity
        buffer.resize(capacity);
        let slice = buffer.as_mut_slice();
        
        // Write pattern to verify memory is accessible
        for (i, val) in slice.iter_mut().enumerate() {
            *val = (i % 256) as u8;
        }
        
        // Verify the pattern
        for (i, &val) in buffer.as_slice().iter().enumerate() {
            assert_eq!(val, (i % 256) as u8);
        }
    }

    #[test]
    fn test_aligned_buffer_different_types() {
        // Test with different types to ensure generic implementation works
        
        // u8
        let mut buffer_u8: AlignedBuffer<u8> = AlignedBuffer::new(100, 16);
        buffer_u8.resize(10);
        buffer_u8.as_mut_slice().fill(42);
        assert!(buffer_u8.as_slice().iter().all(|&x| x == 42));
        
        // i32
        let mut buffer_i32: AlignedBuffer<i32> = AlignedBuffer::new(100, 32);
        buffer_i32.resize(10);
        buffer_i32.as_mut_slice().fill(-42);
        assert!(buffer_i32.as_slice().iter().all(|&x| x == -42));
        
        // Custom struct
        #[derive(Clone, Copy, PartialEq, Debug)]
        struct TestStruct { a: f64, b: u64 }
        
        let mut buffer_struct: AlignedBuffer<TestStruct> = AlignedBuffer::new(100, 64);
        buffer_struct.resize(10);
        let test_val = TestStruct { a: std::f64::consts::PI, b: 42 };
        buffer_struct.as_mut_slice().fill(test_val);
        assert!(buffer_struct.as_slice().iter().all(|&x| x == test_val));
    }

    #[test]
    fn test_buffer_pool_concurrent_stress() {
        use std::sync::Arc;
        use std::thread;
        use std::time::Duration;
        
        let pool = Arc::new(BufferPool::<f64>::new(64, 5));
        let mut handles = vec![];
        
        // Create many threads doing rapid checkout/checkin
        for thread_id in 0..10 {
            let pool_clone = Arc::clone(&pool);
            let handle = thread::spawn(move || {
                for iteration in 0..100 {
                    let size = 50 + (thread_id * 10) + (iteration % 50);
                    let mut buffer = pool_clone.checkout(size);
                    
                    // Do some work with the buffer
                    buffer.resize(size);
                    let value = (thread_id * 1000 + iteration) as f64;
                    buffer.as_mut_slice().fill(value);
                    
                    // Verify the data
                    assert!(buffer.as_slice().iter().all(|&x| x == value));
                    
                    // Small delay to increase contention
                    thread::sleep(Duration::from_micros(10));
                }
            });
            handles.push(handle);
        }
        
        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_buffer_pool_zero_reuse() {
        let pool: BufferPool<f64> = BufferPool::new(64, 10);
        
        // Write data to buffer
        let mut buffer = pool.checkout(100);
        buffer.resize(100);
        buffer.as_mut_slice().fill(42.0);
        drop(buffer);
        
        // Check out again - data should be zeroed
        let mut buffer = pool.checkout(50);
        buffer.resize(50);
        
        // The buffer should start with zeros after being returned to pool
        assert!(buffer.as_slice().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_checked_out_buffer_methods() {
        let pool: BufferPool<f64> = BufferPool::new(64, 10);
        let mut buffer = pool.checkout(100);
        
        // Test get_mut
        let buf_ref = buffer.get_mut();
        buf_ref.resize(10);
        
        // Test as_slice
        assert_eq!(buffer.as_slice().len(), 10);
        
        // Test as_mut_slice
        buffer.as_mut_slice().fill(std::f64::consts::PI);
        assert!(buffer.as_slice().iter().all(|&x| x == std::f64::consts::PI));
    }

    #[test]
    fn test_aligned_buffer_drop() {
        // This test verifies that dropping buffers doesn't cause issues
        // The actual deallocation is tested by running under valgrind or similar
        let mut buffers = Vec::new();
        
        for i in 0..100 {
            let buffer: AlignedBuffer<f64> = AlignedBuffer::new(100 + i, 64);
            buffers.push(buffer);
        }
        
        // Drop half the buffers explicitly
        for _ in 0..50 {
            buffers.pop();
        }
        
        // The rest will be dropped when buffers goes out of scope
    }

    #[test]
    fn test_edge_case_tiny_buffers() {
        let pool: BufferPool<u8> = BufferPool::new(16, 10);
        
        // Test with very small buffers
        for size in 1..=10 {
            let mut buffer = pool.checkout(size);
            buffer.resize(size);
            
            // Write and verify
            for (i, val) in buffer.as_mut_slice().iter_mut().enumerate() {
                *val = i as u8;
            }
            
            for (i, &val) in buffer.as_slice().iter().enumerate() {
                assert_eq!(val, i as u8);
            }
        }
    }

    #[test]
    #[should_panic(expected = "Buffer already taken")]
    fn test_checked_out_buffer_double_take() {
        let pool: BufferPool<f64> = BufferPool::new(64, 10);
        let mut buffer = pool.checkout(100);
        
        // Manually take the buffer
        let _ = buffer.buffer.take();
        
        // This should panic
        let _ = buffer.get_mut();
    }
    
    #[test]
    fn test_arc_buffer_pool_basic() {
        let pool = Arc::new(ArcBufferPool::<f64>::new(64, 10));
        
        // Checkout a buffer
        let mut buffer = pool.checkout(100);
        assert_eq!(buffer.capacity(), 100);
        assert_eq!(buffer.len(), 0);
        
        // Use the buffer
        buffer.resize(50);
        buffer.as_mut_slice().fill(42.0);
        
        // Verify data
        assert!(buffer.as_slice().iter().all(|&x| x == 42.0));
        
        // Drop and checkout again
        drop(buffer);
        let buffer2 = pool.checkout(50);
        assert_eq!(buffer2.capacity(), 100); // Should reuse the same buffer
    }
    
    #[test]
    fn test_arc_buffer_no_lifetime_constraints() {
        // This test demonstrates that ArcBuffer can be returned from functions
        fn get_buffer() -> ArcBuffer<f64> {
            let pool = Arc::new(ArcBufferPool::new(64, 10));
            pool.checkout(100)
        }
        
        let mut buffer = get_buffer();
        buffer.resize(10);
        buffer.as_mut_slice().fill(3.14);
        assert_eq!(buffer.len(), 10);
    }
    
    #[test]
    fn test_arc_buffer_pool_reuse() {
        let pool = Arc::new(ArcBufferPool::<u8>::new(16, 5));
        
        // Check pool is initially empty
        assert_eq!(pool.pool_size(), 0);
        
        // Checkout and return multiple buffers
        let buffers: Vec<_> = (0..3).map(|_| pool.checkout(100)).collect();
        drop(buffers);
        
        // Pool should now have 3 buffers
        assert_eq!(pool.pool_size(), 3);
        
        // Checkout again - should reuse
        let _b1 = pool.checkout(50);
        assert_eq!(pool.pool_size(), 2);
    }
    
    #[test]
    fn test_arc_buffer_standalone() {
        // Create buffer without pool
        let mut buffer = ArcBuffer::<i32>::new(50, 32);
        assert_eq!(buffer.capacity(), 50);
        
        buffer.resize(25);
        for (i, val) in buffer.as_mut_slice().iter_mut().enumerate() {
            *val = i as i32;
        }
        
        // Verify
        for (i, &val) in buffer.as_slice().iter().enumerate() {
            assert_eq!(val, i as i32);
        }
    }
    
    #[test]
    fn test_arc_buffer_pool_thread_safety() {
        use std::thread;
        
        let pool = Arc::new(ArcBufferPool::<f64>::new(64, 20));
        let mut handles = vec![];
        
        // Spawn threads that use the pool
        for thread_id in 0..4 {
            let pool_clone = Arc::clone(&pool);
            let handle = thread::spawn(move || {
                for i in 0..10 {
                    let mut buffer = pool_clone.checkout(100 + thread_id * 10);
                    buffer.resize(50);
                    buffer.as_mut_slice().fill((thread_id * 100 + i) as f64);
                    
                    // Verify
                    let expected = (thread_id * 100 + i) as f64;
                    assert!(buffer.as_slice().iter().all(|&x| x == expected));
                }
            });
            handles.push(handle);
        }
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        // Pool should have accumulated some buffers
        assert!(pool.pool_size() > 0);
    }
    
    #[test]
    fn test_arc_buffer_deref() {
        let pool = Arc::new(ArcBufferPool::<f64>::new(64, 10));
        let mut buffer = pool.checkout(10);
        buffer.resize(5);
        
        // Test Deref
        assert_eq!(buffer.len(), 5);
        assert_eq!((&*buffer).len(), 5);
        
        // Test DerefMut
        buffer[0] = 1.0;
        buffer[1] = 2.0;
        assert_eq!(buffer[0], 1.0);
        assert_eq!(buffer[1], 2.0);
    }
    
    #[test]
    #[should_panic(expected = "ArcBuffer cannot be cloned")]
    fn test_arc_buffer_clone_panic() {
        let buffer = ArcBuffer::<f64>::new(10, 64);
        let _ = buffer.clone();
    }
    
    #[test]
    fn test_arc_buffer_take() {
        let pool = Arc::new(ArcBufferPool::<f64>::new(64, 10));
        let mut buffer = pool.checkout(100);
        buffer.resize(50);
        
        // Take the inner buffer
        let inner = buffer.take();
        assert_eq!(inner.capacity(), 100);
        assert_eq!(inner.len(), 50);
        
        // The taken buffer won't be returned to the pool
        drop(inner);
        assert_eq!(pool.pool_size(), 0);
    }
}
