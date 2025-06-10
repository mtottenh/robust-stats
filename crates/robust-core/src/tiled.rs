//! Tiled sparse matrix representations for cache-efficient computation
//!
//! This module provides tiled data structures that enable cache-efficient
//! processing of sparse matrices. By dividing matrices into tiles, we can
//! process data in cache-friendly chunks and enable efficient SIMD operations.

use crate::sparse::SparseWeights;
use crate::Numeric;
use num_traits::Zero;
use std::alloc::{alloc, dealloc, Layout};
use std::marker::PhantomData;
use std::ops::Range;
use std::ptr;

/// Trait for accessing sparse tile data in SoA format
///
/// This trait allows both SparseTile and OptimizedSparseTile to be used
/// interchangeably in performance-critical code.
pub trait SparseTileData<T: Numeric = f64> {
    /// Get the local row indices
    fn local_rows(&self) -> &[u16];

    /// Get the local column indices
    fn local_cols(&self) -> &[u16];

    /// Get the weights
    fn weights(&self) -> &[T];

    /// Get the tile row coordinate
    fn tile_row(&self) -> usize;

    /// Get the tile column coordinate
    fn tile_col(&self) -> usize;

    /// Get the row start index
    fn row_start(&self) -> usize;

    /// Get the row end index
    fn row_end(&self) -> usize;

    /// Get the column start index
    fn col_start(&self) -> usize;

    /// Get the column end index
    fn col_end(&self) -> usize;

    /// Get the number of non-zero entries
    fn nnz(&self) -> usize {
        self.weights().len()
    }

    /// Check if the tile is empty
    fn is_empty(&self) -> bool {
        self.weights().is_empty()
    }

    /// Get the row starts array for row-grouped processing
    /// Returns None if the implementation doesn't support row grouping
    fn row_starts(&self) -> Option<&[u16]> {
        None
    }
}

/// A specialized buffer for sparse tile data with optimal memory layout
///
/// This buffer allocates all arrays (row_starts, local_rows, local_cols, weights) in a
/// single contiguous memory block. This provides:
/// - Better cache locality (all data in consecutive cache lines)
/// - Reduced memory fragmentation
/// - Optimal alignment for SIMD operations
/// - O(1) access to each row's entries via row_starts array
///
/// Memory layout:
/// ```text
/// [padding][row_starts (u16)][padding][local_rows (u16)][padding][local_cols (u16)][padding][weights (T)]
/// ```
/// Each section is aligned to 32 bytes for AVX2 operations.
///
/// The row_starts array enables efficient row-grouped processing:
/// - Placed first for optimal cache access pattern
/// - Length: n_rows + 1 (includes sentinel)
/// - row_starts[i] = index of first entry for row i
/// - row_starts[n_rows] = total number of entries (sentinel)
/// - Empty rows have row_starts[i] == row_starts[i+1]
/// - Uses u16 since tile sizes are typically small (max ~65k entries)
pub struct SoaTileBuffer<T: Numeric = f64> {
    /// Base pointer to the allocated memory
    ptr: *mut u8,
    /// Total size of allocated memory
    total_size: usize,
    /// Layout used for allocation (needed for deallocation)
    layout: Layout,
    /// Number of entries
    n_entries: usize,
    // Number of rows in this tile
    n_rows: usize,
    /// Offsets to each array within the buffer
    row_starts_offset: usize,
    rows_offset: usize,
    cols_offset: usize,
    weights_offset: usize,
    /// Phantom data to ensure !Send and !Sync and to track type T
    _marker: PhantomData<(*mut u8, T)>,
}

impl<T: Numeric> SoaTileBuffer<T> {
    /// Create a new buffer with the given capacity
    ///
    /// Allocates a single contiguous block of memory for all three arrays,
    /// with proper alignment for SIMD operations.
    pub fn new(n_entries: usize, n_rows: usize) -> Self {
        if n_entries == 0 {
            return Self {
                ptr: ptr::null_mut(),
                total_size: 0,
                layout: Layout::new::<u8>(),
                n_entries: 0,
                n_rows: 0,
                row_starts_offset: 0,
                rows_offset: 0,
                cols_offset: 0,
                weights_offset: 0,
                _marker: PhantomData,
            };
        }

        // Calculate sizes for each array
        let row_starts_size = (n_rows + 1) * std::mem::size_of::<u16>(); // +1 for end sentinel
        let rows_size = n_entries * std::mem::size_of::<u16>();
        let cols_size = n_entries * std::mem::size_of::<u16>();
        let weights_size = n_entries * std::mem::size_of::<T>();

        // Alignment for AVX2 (32 bytes)
        const ALIGNMENT: usize = 32;

        // Calculate aligned offsets - row_starts comes first
        let row_starts_offset = 0;
        let rows_offset = (row_starts_offset + row_starts_size + ALIGNMENT - 1) & !(ALIGNMENT - 1);
        let cols_offset = (rows_offset + rows_size + ALIGNMENT - 1) & !(ALIGNMENT - 1);
        let weights_offset = (cols_offset + cols_size + ALIGNMENT - 1) & !(ALIGNMENT - 1);
        let total_size = weights_offset + weights_size;

        // Create layout with proper alignment
        let layout = Layout::from_size_align(total_size, ALIGNMENT).expect("Invalid layout");

        // Allocate memory
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            panic!("Failed to allocate memory for SoaTileBuffer");
        }

        // Initialize the allocated memory to zero
        unsafe {
            std::ptr::write_bytes(ptr, 0, total_size);
        }

        Self {
            ptr,
            total_size,
            layout,
            n_entries,
            n_rows,
            row_starts_offset,
            rows_offset,
            cols_offset,
            weights_offset,
            _marker: PhantomData,
        }
    }
    /// Get the row starts array (length = n_rows + 1)
    /// row_starts[i] gives the index of the first entry for row i
    /// row_starts[n_rows] gives the total number of entries
    pub fn row_starts(&self) -> &[u16] {
        if self.n_rows == 0 {
            return &[];
        }
        unsafe {
            let ptr = self.ptr.add(self.row_starts_offset) as *const u16;
            std::slice::from_raw_parts(ptr, self.n_rows + 1)
        }
    }

    /// Get a mutable slice to the row starts array
    pub fn row_starts_mut(&mut self) -> &mut [u16] {
        if self.n_rows == 0 {
            return &mut [];
        }
        unsafe {
            let ptr = self.ptr.add(self.row_starts_offset) as *mut u16;
            std::slice::from_raw_parts_mut(ptr, self.n_rows + 1)
        }
    }

    /// Get a slice to the local_rows array
    pub fn local_rows(&self) -> &[u16] {
        if self.n_entries == 0 {
            return &[];
        }
        unsafe {
            let ptr = self.ptr.add(self.rows_offset) as *const u16;
            std::slice::from_raw_parts(ptr, self.n_entries)
        }
    }

    /// Get a mutable slice to the local_rows array
    pub fn local_rows_mut(&mut self) -> &mut [u16] {
        if self.n_entries == 0 {
            return &mut [];
        }
        unsafe {
            let ptr = self.ptr.add(self.rows_offset) as *mut u16;
            std::slice::from_raw_parts_mut(ptr, self.n_entries)
        }
    }

    /// Get a slice to the local_cols array
    pub fn local_cols(&self) -> &[u16] {
        if self.n_entries == 0 {
            return &[];
        }
        unsafe {
            let ptr = self.ptr.add(self.cols_offset) as *const u16;
            std::slice::from_raw_parts(ptr, self.n_entries)
        }
    }

    /// Get a mutable slice to the local_cols array
    pub fn local_cols_mut(&mut self) -> &mut [u16] {
        if self.n_entries == 0 {
            return &mut [];
        }
        unsafe {
            let ptr = self.ptr.add(self.cols_offset) as *mut u16;
            std::slice::from_raw_parts_mut(ptr, self.n_entries)
        }
    }

    /// Get a slice to the weights array
    pub fn weights(&self) -> &[T] {
        if self.n_entries == 0 {
            return &[];
        }
        unsafe {
            let ptr = self.ptr.add(self.weights_offset) as *const T;
            std::slice::from_raw_parts(ptr, self.n_entries)
        }
    }

    /// Get a mutable slice to the weights array
    pub fn weights_mut(&mut self) -> &mut [T] {
        if self.n_entries == 0 {
            return &mut [];
        }
        unsafe {
            let ptr = self.ptr.add(self.weights_offset) as *mut T;
            std::slice::from_raw_parts_mut(ptr, self.n_entries)
        }
    }

    /// Copy data from TileEntry array into this buffer
    pub fn copy_from_entries(&mut self, entries: &[TileEntry<T>]) {
        assert_eq!(entries.len(), self.n_entries, "Entry count mismatch");

        // Use unsafe to get all three mutable pointers at once
        unsafe {
            let rows_ptr = self.ptr.add(self.rows_offset) as *mut u16;
            let cols_ptr = self.ptr.add(self.cols_offset) as *mut u16;
            let weights_ptr = self.ptr.add(self.weights_offset) as *mut T;

            // Single pass through entries - better cache locality
            for (i, entry) in entries.iter().enumerate() {
                *rows_ptr.add(i) = entry.local_row;
                *cols_ptr.add(i) = entry.local_col;
                *weights_ptr.add(i) = entry.weight;
            }
        }
    }

    /// Build the buffer directly without intermediate TileEntry allocation
    /// This is the most efficient way when building from sparse rows
    pub fn fill_direct<F>(&mut self, fill_fn: F)
    where
        F: FnOnce(*mut u16, *mut u16, *mut T, usize),
    {
        unsafe {
            let rows_ptr = self.ptr.add(self.rows_offset) as *mut u16;
            let cols_ptr = self.ptr.add(self.cols_offset) as *mut u16;
            let weights_ptr = self.ptr.add(self.weights_offset) as *mut T;

            // Let the caller fill the arrays directly
            fill_fn(rows_ptr, cols_ptr, weights_ptr, self.n_entries);
        }
    }
    pub fn build_row_starts(&mut self) {
        // Cache values to avoid borrow issues
        let n_rows = self.n_rows;
        let n_entries = self.n_entries;

        if n_entries == 0 {
            return;
        }

        // Fast path: Check if data is already sorted by row
        let local_rows = self.local_rows();
        let mut is_sorted = true;
        for i in 1..n_entries {
            if local_rows[i] < local_rows[i - 1] {
                is_sorted = false;
                break;
            }
        }

        // Only sort if necessary
        if !is_sorted {
            let local_cols = self.local_cols();
            let weights = self.weights();

            // Create index array and sort it
            let mut indices: Vec<usize> = (0..n_entries).collect();
            indices.sort_by_key(|&i| (local_rows[i], local_cols[i]));

            // Create temporary sorted arrays
            let mut sorted_rows = vec![0u16; n_entries];
            let mut sorted_cols = vec![0u16; n_entries];
            let mut sorted_weights = vec![<T as crate::Numeric>::zero(); n_entries];

            for (new_idx, &old_idx) in indices.iter().enumerate() {
                sorted_rows[new_idx] = local_rows[old_idx];
                sorted_cols[new_idx] = local_cols[old_idx];
                sorted_weights[new_idx] = weights[old_idx];
            }

            // Copy sorted data back
            self.local_rows_mut().copy_from_slice(&sorted_rows);
            self.local_cols_mut().copy_from_slice(&sorted_cols);
            self.weights_mut().copy_from_slice(&sorted_weights);
        }

        // Check if we can use u16 indexing
        if n_entries > u16::MAX as usize {
            // This should not happen with our tile splitting logic, but handle gracefully
            eprintln!(
                "Warning: Tile has {n_entries} entries, which exceeds u16 capacity. \
                 Row grouping will not be available for this tile."
            );
            return;
        }

        // First pass: collect row information
        let mut row_info = Vec::new();
        if n_entries > 0 {
            let local_rows = self.local_rows();
            let mut current_row = local_rows[0];
            let mut start_idx = 0;

            #[allow(clippy::needless_range_loop)]
            for i in 1..n_entries {
                let row = local_rows[i];
                if row != current_row {
                    // Record info for the current row
                    row_info.push((current_row as usize, start_idx, i));
                    current_row = row;
                    start_idx = i;
                }
            }
            // Don't forget the last row
            row_info.push((current_row as usize, start_idx, n_entries));
        }

        // Second pass: update row_starts array
        let row_starts = self.row_starts_mut();

        // Initialize all to n_entries (for empty rows)
        for start in row_starts.iter_mut().take(n_rows + 1) {
            *start = n_entries as u16;
        }

        // Fill in the actual row starts
        let mut last_end = 0;
        for (row_idx, start, _end) in row_info {
            // Fill any gap between last row and this row
            for empty_start in &mut row_starts[last_end..row_idx] {
                *empty_start = start as u16;
            }
            row_starts[row_idx] = start as u16;
            last_end = row_idx + 1;
        }
    }
    /// Copy data from separate arrays into this buffer
    pub fn copy_from_arrays(&mut self, local_rows: &[u16], local_cols: &[u16], weights: &[T]) {
        assert_eq!(local_rows.len(), self.n_entries, "Row count mismatch");
        assert_eq!(local_cols.len(), self.n_entries, "Col count mismatch");
        assert_eq!(weights.len(), self.n_entries, "Weight count mismatch");

        self.local_rows_mut().copy_from_slice(local_rows);
        self.local_cols_mut().copy_from_slice(local_cols);
        self.weights_mut().copy_from_slice(weights);
    }

    /// Get the number of entries
    pub fn len(&self) -> usize {
        self.n_entries
    }

    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.n_entries == 0
    }

    /// Get the number of rows in this tile
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Get the total memory size of the buffer
    pub fn memory_size(&self) -> usize {
        self.total_size
    }
}

impl<T: Numeric> Drop for SoaTileBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                dealloc(self.ptr, self.layout);
            }
        }
    }
}

// Safety: SoaTileBuffer can be sent between threads (the memory is owned)
unsafe impl<T: Numeric> Send for SoaTileBuffer<T> {}
// Safety: SoaTileBuffer can be shared between threads (immutable access is safe)
unsafe impl<T: Numeric> Sync for SoaTileBuffer<T> {}

/// A single entry in a sparse tile
///
/// Uses local coordinates within the tile for memory efficiency.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TileEntry<T: Numeric = f64> {
    /// Local row within tile (0..tile_row_size)
    pub local_row: u16,
    /// Local column within tile (0..tile_col_size)
    pub local_col: u16,
    /// The weight value
    pub weight: T,
}

/// A sparse tile containing a subset of the matrix
///
/// Each tile represents a rectangular region of the full matrix,
/// storing only non-zero entries within that region.
///
/// Uses an optimized SoA (Structure of Arrays) layout with all data
/// in a single contiguous memory block for better cache locality.
pub struct SparseTile<T: Numeric = f64> {
    /// Which tile this is in the grid
    pub tile_row: usize,
    pub tile_col: usize,
    /// The actual bounds of this tile in the original matrix
    pub row_start: usize,
    pub row_end: usize,
    pub col_start: usize,
    pub col_end: usize,
    /// Optimized buffer containing all sparse data
    buffer: SoaTileBuffer<T>,
}

impl<T: Numeric> Clone for SparseTile<T> {
    fn clone(&self) -> Self {
        Self {
            tile_row: self.tile_row,
            tile_col: self.tile_col,
            row_start: self.row_start,
            row_end: self.row_end,
            col_start: self.col_start,
            col_end: self.col_end,
            buffer: {
                let mut new_buffer =
                    SoaTileBuffer::new(self.buffer.len(), self.row_end - self.row_start);
                new_buffer.copy_from_arrays(
                    self.buffer.local_rows(),
                    self.buffer.local_cols(),
                    self.buffer.weights(),
                );
                new_buffer
            },
        }
    }
}

impl<T: Numeric> std::fmt::Debug for SparseTile<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SparseTile")
            .field("tile_row", &self.tile_row)
            .field("tile_col", &self.tile_col)
            .field("row_start", &self.row_start)
            .field("row_end", &self.row_end)
            .field("col_start", &self.col_start)
            .field("col_end", &self.col_end)
            .field("n_entries", &self.buffer.len())
            .finish()
    }
}

impl<T: Numeric> SparseTile<T> {
    /// Create a new sparse tile from entries
    /// Converts from AoS (Array of Structs) to SoA (Structure of Arrays)
    /// with all data in a single contiguous memory block
    pub fn new(
        tile_row: usize,
        tile_col: usize,
        row_start: usize,
        row_end: usize,
        col_start: usize,
        col_end: usize,
        entries: Vec<TileEntry<T>>,
    ) -> Self {
        let n_rows = row_end - row_start;
        let mut buffer = SoaTileBuffer::new(entries.len(), n_rows);
        buffer.copy_from_entries(&entries);
        buffer.build_row_starts();
        Self {
            tile_row,
            tile_col,
            row_start,
            row_end,
            col_start,
            col_end,
            buffer,
        }
    }

    /// Create a new sparse tile from entries that are already sorted by row
    /// This avoids the sorting overhead in build_row_starts
    pub fn new_sorted(
        tile_row: usize,
        tile_col: usize,
        row_start: usize,
        row_end: usize,
        col_start: usize,
        col_end: usize,
        entries: Vec<TileEntry<T>>,
    ) -> Self {
        let n_rows = row_end - row_start;
        let mut buffer = SoaTileBuffer::new(entries.len(), n_rows);
        buffer.copy_from_entries(&entries);
        buffer.build_row_starts(); // Will detect sorted data and skip sorting
        Self {
            tile_row,
            tile_col,
            row_start,
            row_end,
            col_start,
            col_end,
            buffer,
        }
    }

    /// Create a sparse tile by directly filling the buffer
    /// This is the most efficient method - no intermediate allocations
    pub fn new_direct<F>(
        tile_row: usize,
        tile_col: usize,
        row_start: usize,
        row_end: usize,
        col_start: usize,
        col_end: usize,
        n_entries: usize,
        fill_fn: F,
    ) -> Self
    where
        F: FnOnce(*mut u16, *mut u16, *mut T, usize),
    {
        let n_rows = row_end - row_start;
        let mut buffer = SoaTileBuffer::new(n_entries, n_rows);

        // Fill directly into the buffer
        buffer.fill_direct(fill_fn);
        buffer.build_row_starts();

        Self {
            tile_row,
            tile_col,
            row_start,
            row_end,
            col_start,
            col_end,
            buffer,
        }
    }

    /// Create from existing arrays
    #[allow(clippy::too_many_arguments)]
    pub fn from_arrays(
        tile_row: usize,
        tile_col: usize,
        row_start: usize,
        row_end: usize,
        col_start: usize,
        col_end: usize,
        local_rows: &[u16],
        local_cols: &[u16],
        weights: &[T],
    ) -> Self {
        assert_eq!(local_rows.len(), local_cols.len());
        assert_eq!(local_rows.len(), weights.len());
        let n_rows = row_end - row_start;
        let mut buffer = SoaTileBuffer::new(local_rows.len(), n_rows);
        buffer.copy_from_arrays(local_rows, local_cols, weights);
        buffer.build_row_starts();
        Self {
            tile_row,
            tile_col,
            row_start,
            row_end,
            col_start,
            col_end,
            buffer,
        }
    }

    /// Create from an existing SoaTileBuffer, taking ownership to avoid copying
    pub fn from_buffer(
        tile_row: usize,
        tile_col: usize,
        row_start: usize,
        row_end: usize,
        col_start: usize,
        col_end: usize,
        buffer: SoaTileBuffer<T>,
    ) -> Self {
        // Validate buffer has correct number of rows
        assert_eq!(
            buffer.n_rows(),
            row_end - row_start,
            "Buffer n_rows must match tile row range"
        );
        
        Self {
            tile_row,
            tile_col,
            row_start,
            row_end,
            col_start,
            col_end,
            buffer,
        }
    }

    /// Get the local row indices
    pub fn local_rows(&self) -> &[u16] {
        self.buffer.local_rows()
    }

    /// Get the local column indices
    pub fn local_cols(&self) -> &[u16] {
        self.buffer.local_cols()
    }

    /// Get the weights
    pub fn weights(&self) -> &[T] {
        self.buffer.weights()
    }

    /// Get mutable access to weights
    pub fn weights_mut(&mut self) -> &mut [T] {
        self.buffer.weights_mut()
    }

    /// Get the density of this tile (fraction of non-zero entries)
    pub fn density(&self) -> f64 {
        let tile_size = (self.row_end - self.row_start) * (self.col_end - self.col_start);
        if tile_size == 0 {
            0.0
        } else {
            self.buffer.len() as f64 / tile_size as f64
        }
    }

    /// Get the number of non-zero entries
    pub fn nnz(&self) -> usize {
        self.buffer.len()
    }

    /// Check if tile is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Get tile dimensions
    pub fn shape(&self) -> (usize, usize) {
        (self.row_end - self.row_start, self.col_end - self.col_start)
    }

    /// Get memory size of this tile
    pub fn memory_size(&self) -> usize {
        std::mem::size_of::<Self>() + self.buffer.memory_size()
    }

    /// Apply this tile using ComputePrimitives (Layer 1)
    pub fn apply<P: crate::ComputePrimitives<T>>(
        &self,
        tile_data: &[T],
        result: &mut [T::Aggregate],
        primitives: &P,
    ) {
        primitives.apply_sparse_tile(tile_data, self, result)
    }

    /// Apply this tile without bounds checks
    ///
    /// # Safety
    /// The caller must ensure:
    /// - All local_cols indices are within bounds of tile_data
    /// - All local_rows indices are within bounds of result
    pub unsafe fn apply_unchecked<P: crate::ComputePrimitives<T>>(
        &self,
        tile_data: &[T],
        result: &mut [T::Aggregate],
        primitives: &P,
    ) {
        primitives.apply_sparse_tile_unchecked(tile_data, self, result)
    }

    /// Check if a global coordinate falls within this tile
    pub fn contains(&self, row: usize, col: usize) -> bool {
        row >= self.row_start && row < self.row_end && col >= self.col_start && col < self.col_end
    }

    /// Convert global coordinates to local tile coordinates
    pub fn global_to_local(&self, row: usize, col: usize) -> Option<(u16, u16)> {
        if self.contains(row, col) {
            Some(((row - self.row_start) as u16, (col - self.col_start) as u16))
        } else {
            None
        }
    }

    /// Convert local tile coordinates to global coordinates
    pub fn local_to_global(&self, local_row: u16, local_col: u16) -> (usize, usize) {
        (
            self.row_start + local_row as usize,
            self.col_start + local_col as usize,
        )
    }
}

// Implement SparseTileData for SparseTile
impl<T: Numeric> SparseTileData<T> for SparseTile<T> {
    fn local_rows(&self) -> &[u16] {
        self.local_rows()
    }

    fn local_cols(&self) -> &[u16] {
        self.local_cols()
    }

    fn weights(&self) -> &[T] {
        self.weights()
    }

    fn tile_row(&self) -> usize {
        self.tile_row
    }

    fn tile_col(&self) -> usize {
        self.tile_col
    }

    fn row_start(&self) -> usize {
        self.row_start
    }

    fn row_end(&self) -> usize {
        self.row_end
    }

    fn col_start(&self) -> usize {
        self.col_start
    }

    fn col_end(&self) -> usize {
        self.col_end
    }

    fn row_starts(&self) -> Option<&[u16]> {
        Some(self.buffer.row_starts())
    }
}

/// A tiled sparse matrix optimized for cache-efficient access
///
/// This structure divides a sparse matrix into rectangular tiles,
/// storing only non-empty tiles. This enables:
/// - Cache-efficient processing by keeping working sets small
/// - SIMD operations on dense tiles
/// - Parallel processing of independent tiles
#[derive(Clone, Debug)]
pub struct TiledSparseMatrix<T: Numeric = f64> {
    /// All tiles in row-major order (only non-empty tiles are stored)
    pub tiles: Vec<SparseTile<T>>,
    /// Number of tile rows and columns
    pub n_tile_rows: usize,
    pub n_tile_cols: usize,
    /// Tile dimensions
    pub tile_row_size: usize,
    pub tile_col_size: usize,
    /// Original matrix dimensions
    pub n_rows: usize,
    pub n_cols: usize,
}

impl<T: Numeric> TiledSparseMatrix<T> {
    /// Maximum entries per tile to ensure u16 indexing works
    /// We support up to 65,535 entries (u16::MAX)
    /// This allows tiles up to 255×255 dense, or larger sparse tiles
    const MAX_ENTRIES_PER_TILE: usize = u16::MAX as usize;

    /// Create a new TiledSparseMatrix directly from tiles
    pub fn new(
        n_rows: usize,
        n_cols: usize,
        tile_row_size: usize,
        tile_col_size: usize,
        tiles: Vec<SparseTile<T>>,
    ) -> Self {
        let n_tile_rows = n_rows.div_ceil(tile_row_size);
        let n_tile_cols = n_cols.div_ceil(tile_col_size);
        
        Self {
            tiles,
            n_tile_rows,
            n_tile_cols,
            tile_row_size,
            tile_col_size,
            n_rows,
            n_cols,
        }
    }

    /// Create a tiled sparse matrix from row-based sparse weights
    ///
    /// # Arguments
    /// * `sparse_rows` - Each element represents a sparse row of the matrix
    /// * `tile_row_size` - Number of rows per tile
    /// * `tile_col_size` - Number of columns per tile
    ///
    /// # Panics
    /// - If sparse_rows is empty
    /// - If tile sizes are zero
    /// - If rows have inconsistent column counts
    ///
    /// # Automatic Tile Splitting
    /// If a tile would have more than MAX_ENTRIES_PER_TILE entries,
    /// it will be automatically split into smaller tiles to ensure
    /// compatibility with u16 indexing.
    pub fn from_sparse_rows(
        sparse_rows: Vec<SparseWeights<T>>,
        mut tile_row_size: usize,
        mut tile_col_size: usize,
    ) -> Self {
        assert!(
            !sparse_rows.is_empty(),
            "Cannot create matrix from empty rows"
        );
        assert!(tile_row_size > 0, "Tile row size must be positive");
        assert!(tile_col_size > 0, "Tile column size must be positive");

        let n_rows = sparse_rows.len();
        let n_cols = sparse_rows[0].n;

        // Validate all rows have same column count
        for (i, row) in sparse_rows.iter().enumerate() {
            assert_eq!(
                row.n, n_cols,
                "Row {} has different column count: {} vs {}",
                i, row.n, n_cols
            );
        }

        // Adjust tile sizes if they could lead to too many entries
        // We support up to u16::MAX entries, but for dense tiles we need to be careful
        let max_possible_entries = tile_row_size * tile_col_size;
        if max_possible_entries > Self::MAX_ENTRIES_PER_TILE {
            // Calculate new tile sizes that respect the limit
            // For dense tiles, we can support up to 255×255 = 65,025 entries
            // or any rectangular variant that doesn't exceed u16::MAX
            let original_row_size = tile_row_size;
            let original_col_size = tile_col_size;

            // Try to maintain aspect ratio while respecting limit
            let aspect_ratio = tile_row_size as f64 / tile_col_size as f64;
            let max_area = Self::MAX_ENTRIES_PER_TILE as f64;

            // Solve for dimensions that maintain aspect ratio
            // area = rows * cols, rows = aspect_ratio * cols
            // area = aspect_ratio * cols^2
            let new_cols = (max_area / aspect_ratio).sqrt() as usize;
            let new_rows = (new_cols as f64 * aspect_ratio) as usize;

            tile_col_size = new_cols.min(tile_col_size);
            tile_row_size = new_rows.min(tile_row_size);

            // Ensure we don't exceed the limit
            while tile_row_size * tile_col_size > Self::MAX_ENTRIES_PER_TILE {
                if tile_row_size > tile_col_size {
                    tile_row_size -= 1;
                } else {
                    tile_col_size -= 1;
                }
            }

            eprintln!(
                "Warning: Requested tile size {}×{} ({} entries) exceeds u16 capacity. \
                 Adjusted to {}×{} ({} entries).",
                original_row_size,
                original_col_size,
                original_row_size * original_col_size,
                tile_row_size,
                tile_col_size,
                tile_row_size * tile_col_size
            );
        }

        // Calculate number of tiles needed
        let n_tile_rows = n_rows.div_ceil(tile_row_size);
        let n_tile_cols = n_cols.div_ceil(tile_col_size);

        // Create all tiles
        let mut tiles = Vec::new();

        for tile_row in 0..n_tile_rows {
            for tile_col in 0..n_tile_cols {
                // Determine bounds for this tile
                let row_start = tile_row * tile_row_size;
                let row_end = ((tile_row + 1) * tile_row_size).min(n_rows);
                let col_start = tile_col * tile_col_size;
                let col_end = ((tile_col + 1) * tile_col_size).min(n_cols);

                // Collect all entries that fall within this tile
                let mut entries = Vec::new();

                // Check each row in this tile's row range
                for (row_idx, sparse_row) in sparse_rows[row_start..row_end].iter().enumerate() {
                    let row = row_start + row_idx;

                    // Find entries that fall within this tile's column range
                    // Use binary search since indices are sorted
                    let start_idx = sparse_row
                        .indices
                        .binary_search(&col_start)
                        .unwrap_or_else(|x| x);

                    for i in start_idx..sparse_row.indices.len() {
                        let col_idx = sparse_row.indices[i];
                        if col_idx >= col_end {
                            break;
                        }

                        entries.push(TileEntry {
                            local_row: (row - row_start) as u16,
                            local_col: (col_idx - col_start) as u16,
                            weight: sparse_row.weights[i],
                        });
                    }
                }

                // Only create tile if it has non-zero entries
                if !entries.is_empty() {
                    // Check if this tile has too many entries
                    if entries.len() > Self::MAX_ENTRIES_PER_TILE {
                        // Split the tile into smaller sub-tiles
                        Self::split_large_tile(
                            &mut tiles,
                            tile_row,
                            tile_col,
                            row_start,
                            row_end,
                            col_start,
                            col_end,
                            entries,
                            &sparse_rows[row_start..row_end],
                        );
                    } else {
                        // Entries are already sorted by row since we iterate row by row
                        tiles.push(SparseTile::new_sorted(
                            tile_row, tile_col, row_start, row_end, col_start, col_end, entries,
                        ));
                    }
                }
            }
        }

        TiledSparseMatrix {
            tiles,
            n_tile_rows,
            n_tile_cols,
            tile_row_size,
            tile_col_size,
            n_rows,
            n_cols,
        }
    }

    /// Get all tiles that overlap with the given row range
    pub fn get_tiles_for_rows(&self, row_range: Range<usize>) -> Vec<&SparseTile<T>> {
        self.tiles
            .iter()
            .filter(|tile| tile.row_start < row_range.end && tile.row_end > row_range.start)
            .collect()
    }

    /// Get all tiles that overlap with the given column range
    pub fn get_tiles_for_cols(&self, col_range: Range<usize>) -> Vec<&SparseTile<T>> {
        self.tiles
            .iter()
            .filter(|tile| tile.col_start < col_range.end && tile.col_end > col_range.start)
            .collect()
    }

    /// Get a specific tile by its coordinates
    pub fn get_tile(&self, tile_row: usize, tile_col: usize) -> Option<&SparseTile<T>> {
        self.tiles
            .iter()
            .find(|t| t.tile_row == tile_row && t.tile_col == tile_col)
    }

    /// Get total number of non-zero entries across all tiles
    pub fn nnz(&self) -> usize {
        self.tiles.iter().map(|t| t.nnz()).sum()
    }

    /// Get overall sparsity (fraction of zero entries)
    pub fn sparsity(&self) -> f64 {
        let total_entries = self.n_rows * self.n_cols;
        if total_entries == 0 {
            0.0
        } else {
            1.0 - (self.nnz() as f64 / total_entries as f64)
        }
    }

    /// Get memory usage in bytes
    pub fn memory_size(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.tiles.capacity() * std::mem::size_of::<SparseTile<T>>()
            + self.tiles.iter().map(|t| t.memory_size()).sum::<usize>()
    }

    /// Split a large tile into smaller sub-tiles to respect entry limits
    fn split_large_tile(
        tiles: &mut Vec<SparseTile<T>>,
        tile_row: usize,
        tile_col: usize,
        row_start: usize,
        row_end: usize,
        col_start: usize,
        col_end: usize,
        entries: Vec<TileEntry<T>>,
        sparse_rows: &[SparseWeights<T>],
    ) {
        // Strategy: Split the tile into quadrants (or smaller if needed)
        let n_rows = row_end - row_start;
        let n_cols = col_end - col_start;

        // Determine split dimensions
        let row_splits = if n_rows > 1 { 2 } else { 1 };
        let col_splits = if n_cols > 1 { 2 } else { 1 };

        // If we can't split further, we have to accept the large tile
        if row_splits == 1 && col_splits == 1 {
            eprintln!(
                "Warning: Tile at ({},{}) has {} entries (exceeds limit of {}). \
                 Cannot split further due to size constraints.",
                tile_row,
                tile_col,
                entries.len(),
                Self::MAX_ENTRIES_PER_TILE
            );
            tiles.push(SparseTile::new_sorted(
                tile_row, tile_col, row_start, row_end, col_start, col_end, entries,
            ));
            return;
        }

        // Calculate sub-tile boundaries
        let row_mid = row_start + n_rows / row_splits;
        let col_mid = col_start + n_cols / col_splits;

        // Create sub-tiles
        for sub_row in 0..row_splits {
            for sub_col in 0..col_splits {
                let sub_row_start = if sub_row == 0 { row_start } else { row_mid };
                let sub_row_end = if sub_row == 0 { row_mid } else { row_end };
                let sub_col_start = if sub_col == 0 { col_start } else { col_mid };
                let sub_col_end = if sub_col == 0 { col_mid } else { col_end };

                // Collect entries for this sub-tile
                let mut sub_entries = Vec::new();
                for entry in &entries {
                    let global_row = row_start + entry.local_row as usize;
                    let global_col = col_start + entry.local_col as usize;

                    if global_row >= sub_row_start
                        && global_row < sub_row_end
                        && global_col >= sub_col_start
                        && global_col < sub_col_end
                    {
                        sub_entries.push(TileEntry {
                            local_row: (global_row - sub_row_start) as u16,
                            local_col: (global_col - sub_col_start) as u16,
                            weight: entry.weight,
                        });
                    }
                }

                // Recursively check if this sub-tile needs further splitting
                if !sub_entries.is_empty() {
                    if sub_entries.len() > Self::MAX_ENTRIES_PER_TILE {
                        Self::split_large_tile(
                            tiles,
                            tile_row * row_splits + sub_row,
                            tile_col * col_splits + sub_col,
                            sub_row_start,
                            sub_row_end,
                            sub_col_start,
                            sub_col_end,
                            sub_entries,
                            &sparse_rows[sub_row_start - row_start..sub_row_end - row_start],
                        );
                    } else {
                        tiles.push(SparseTile::new_sorted(
                            tile_row * row_splits + sub_row,
                            tile_col * col_splits + sub_col,
                            sub_row_start,
                            sub_row_end,
                            sub_col_start,
                            sub_col_end,
                            sub_entries,
                        ));
                    }
                }
            }
        }
    }

    /// Normalize rows so that each row sums to 1.0
    /// This is crucial for quantile weight matrices
    pub fn normalize_rows(&mut self) 
    where
        T: num_traits::NumCast,
    {
        use num_traits::NumCast;
        
        // First, compute row sums
        let mut row_sums = vec![<T::Float as num_traits::Zero>::zero(); self.n_rows];
        
        // Accumulate sums for each row across all tiles
        for tile in &self.tiles {
            let weights = tile.weights();
            let local_rows = tile.local_rows();
            
            for (&local_row, &weight) in local_rows.iter().zip(weights.iter()) {
                let global_row = tile.row_start + local_row as usize;
                row_sums[global_row] = row_sums[global_row] + weight.to_float();
            }
        }
        
        // Now normalize each tile's weights
        for tile in &mut self.tiles {
            // Get needed values before the mutable borrow
            let row_start = tile.row_start;
            let local_rows = tile.local_rows().to_vec(); // Clone to avoid borrow issues
            let weights = tile.weights_mut();
            
            for (i, &local_row) in local_rows.iter().enumerate() {
                let global_row = row_start + local_row as usize;
                let row_sum = row_sums[global_row];
                
                if row_sum > <T::Float as num_traits::Zero>::zero() {
                    // Normalize the weight
                    let normalized = weights[i].to_float() / row_sum;
                    weights[i] = <T as NumCast>::from(normalized).unwrap();
                }
            }
        }
    }

    /// Get statistics about tile distribution
    pub fn tile_stats(&self) -> TileStats {
        if self.tiles.is_empty() {
            return TileStats {
                n_tiles: 0,
                n_empty_tiles: self.n_tile_rows * self.n_tile_cols,
                avg_entries_per_tile: 0.0,
                min_entries_per_tile: 0,
                max_entries_per_tile: 0,
                avg_density: 0.0,
            };
        }

        let n_tiles = self.tiles.len();
        let n_empty_tiles = self.n_tile_rows * self.n_tile_cols - n_tiles;
        let entries_per_tile: Vec<usize> = self.tiles.iter().map(|t| t.nnz()).collect();
        let total_entries: usize = entries_per_tile.iter().sum();
        let avg_entries = total_entries as f64 / n_tiles as f64;
        let min_entries = *entries_per_tile.iter().min().unwrap_or(&0);
        let max_entries = *entries_per_tile.iter().max().unwrap_or(&0);
        let avg_density = self.tiles.iter().map(|t| t.density()).sum::<f64>() / n_tiles as f64;

        TileStats {
            n_tiles,
            n_empty_tiles,
            avg_entries_per_tile: avg_entries,
            min_entries_per_tile: min_entries,
            max_entries_per_tile: max_entries,
            avg_density,
        }
    }
}
/// Statistics about tile distribution
#[derive(Debug, Clone)]
pub struct TileStats {
    pub n_tiles: usize,
    pub n_empty_tiles: usize,
    pub avg_entries_per_tile: f64,
    pub min_entries_per_tile: usize,
    pub max_entries_per_tile: usize,
    pub avg_density: f64,
}

/// Compute matrix-vector multiplication using tiled representation
///
/// Uses ComputePrimitives (Layer 1) for the actual computation.
pub fn tiled_sparse_matvec<T: Numeric, P: crate::ComputePrimitives<T>>(
    matrix: &TiledSparseMatrix<T>,
    vector: &[T],
    primitives: &P,
) -> Vec<T::Aggregate> {
    assert_eq!(
        vector.len(),
        matrix.n_cols,
        "Vector length must match matrix columns"
    );

    let mut result = vec![<T::Aggregate as Zero>::zero(); matrix.n_rows];

    for tile in &matrix.tiles {
        // Extract the relevant portion of the vector for this tile
        let tile_data = &vector[tile.col_start..tile.col_end];
        let result_slice = &mut result[tile.row_start..tile.row_end];

        // Use Layer 1 (ComputePrimitives) for the computation
        unsafe {
            primitives.apply_sparse_tile_unchecked(tile_data, tile, result_slice);
        }
    }

    result
}

/// Compute matrix-vector multiplication for multiple vectors in batch
pub fn tiled_sparse_matvec_batch<T: Numeric, P: crate::ComputePrimitives<T>>(
    matrix: &TiledSparseMatrix<T>,
    vectors: &[&[T]],
    primitives: &P,
) -> Vec<Vec<T::Aggregate>> {
    vectors
        .iter()
        .map(|v| tiled_sparse_matvec(matrix, v, primitives))
        .collect()
}

/// Alternative implementations for benchmarking comparisons
#[cfg(feature = "benchmark-variants")]
pub mod benchmark_variants {
    use super::*;

    /// Original Array of Structs (AoS) implementation
    pub struct SparseTileAoS<T: Numeric = f64> {
        pub tile_row: usize,
        pub tile_col: usize,
        pub row_start: usize,
        pub row_end: usize,
        pub col_start: usize,
        pub col_end: usize,
        /// Original AoS storage
        pub entries: Vec<TileEntry<T>>,
    }

    impl<T: Numeric> SparseTileAoS<T> {
        pub fn new(
            tile_row: usize,
            tile_col: usize,
            row_start: usize,
            row_end: usize,
            col_start: usize,
            col_end: usize,
            entries: Vec<TileEntry<T>>,
        ) -> Self {
            Self {
                tile_row,
                tile_col,
                row_start,
                row_end,
                col_start,
                col_end,
                entries,
            }
        }

        pub fn nnz(&self) -> usize {
            self.entries.len()
        }

        /// Apply operation with bounds checks (original implementation)
        pub fn apply(&self, tile_data: &[T], result: &mut [T::Aggregate]) {
            for entry in &self.entries {
                let row_offset = entry.local_row as usize;
                let col_offset = entry.local_col as usize;
                result[row_offset] = result[row_offset]
                    + <T::Aggregate as From<T>>::from(tile_data[col_offset])
                        * <T::Aggregate as From<T>>::from(entry.weight);
            }
        }

        /// Apply operation without bounds checks
        ///
        /// # Safety
        /// The caller must ensure:
        /// - All entry local_row indices are within bounds of result slice
        /// - All entry local_col indices are within bounds of tile_data slice
        pub unsafe fn apply_unchecked(&self, tile_data: &[T], result: &mut [T::Aggregate]) {
            for entry in &self.entries {
                let row_offset = entry.local_row as usize;
                let col_offset = entry.local_col as usize;
                let val = *tile_data.get_unchecked(col_offset);
                let w = entry.weight;
                let result_ref = result.get_unchecked_mut(row_offset);
                *result_ref = *result_ref
                    + <T::Aggregate as From<T>>::from(val) * <T::Aggregate as From<T>>::from(w);
            }
        }
    }

    /// Naive Structure of Arrays (SoA) with separate Vecs
    pub struct SparseTileVecSoA<T: Numeric = f64> {
        pub tile_row: usize,
        pub tile_col: usize,
        pub row_start: usize,
        pub row_end: usize,
        pub col_start: usize,
        pub col_end: usize,
        /// Separate vectors for each array
        pub local_rows: Vec<u16>,
        pub local_cols: Vec<u16>,
        pub weights: Vec<T>,
    }

    impl<T: Numeric> SparseTileVecSoA<T> {
        pub fn new(
            tile_row: usize,
            tile_col: usize,
            row_start: usize,
            row_end: usize,
            col_start: usize,
            col_end: usize,
            entries: Vec<TileEntry<T>>,
        ) -> Self {
            let n_entries = entries.len();

            let mut local_rows = Vec::with_capacity(n_entries);
            let mut local_cols = Vec::with_capacity(n_entries);
            let mut weights = Vec::with_capacity(n_entries);

            for entry in entries {
                local_rows.push(entry.local_row);
                local_cols.push(entry.local_col);
                weights.push(entry.weight);
            }

            Self {
                tile_row,
                tile_col,
                row_start,
                row_end,
                col_start,
                col_end,
                local_rows,
                local_cols,
                weights,
            }
        }

        pub fn nnz(&self) -> usize {
            self.weights.len()
        }

        /// Apply operation with bounds checks
        pub fn apply(&self, tile_data: &[T], result: &mut [T::Aggregate]) {
            for i in 0..self.weights.len() {
                let row_offset = self.local_rows[i] as usize;
                let col_offset = self.local_cols[i] as usize;
                result[row_offset] = result[row_offset]
                    + <T::Aggregate as From<T>>::from(tile_data[col_offset])
                        * <T::Aggregate as From<T>>::from(self.weights[i]);
            }
        }

        /// Apply operation without bounds checks
        ///
        /// # Safety
        /// The caller must ensure:
        /// - All local_rows indices are within bounds of result slice
        /// - All local_cols indices are within bounds of tile_data slice
        /// - weights, local_rows, and local_cols have the same length
        pub unsafe fn apply_unchecked(&self, tile_data: &[T], result: &mut [T::Aggregate]) {
            for i in 0..self.weights.len() {
                let row_offset = *self.local_rows.get_unchecked(i) as usize;
                let col_offset = *self.local_cols.get_unchecked(i) as usize;
                let weight = *self.weights.get_unchecked(i);

                let val = *tile_data.get_unchecked(col_offset);
                let result_ref = result.get_unchecked_mut(row_offset);
                *result_ref = *result_ref
                    + <T::Aggregate as From<T>>::from(val)
                        * <T::Aggregate as From<T>>::from(weight);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soa_tile_buffer_basic() {
        // Test creating and using SoaTileBuffer with a simple 4x4 tile
        let entries = vec![
            // Row 0: entries at columns 0, 2
            TileEntry {
                local_row: 0,
                local_col: 0,
                weight: 1.0,
            },
            TileEntry {
                local_row: 0,
                local_col: 2,
                weight: 2.0,
            },
            // Row 1: entry at column 1
            TileEntry {
                local_row: 1,
                local_col: 1,
                weight: 3.0,
            },
            // Row 2: empty
            // Row 3: entries at columns 0, 1, 2, 3
            TileEntry {
                local_row: 3,
                local_col: 0,
                weight: 4.0,
            },
            TileEntry {
                local_row: 3,
                local_col: 1,
                weight: 5.0,
            },
            TileEntry {
                local_row: 3,
                local_col: 2,
                weight: 6.0,
            },
            TileEntry {
                local_row: 3,
                local_col: 3,
                weight: 7.0,
            },
        ];

        let n_rows = 4;
        let mut buffer = SoaTileBuffer::new(entries.len(), n_rows);

        // Check initial state
        assert_eq!(buffer.n_entries, 7);
        assert_eq!(buffer.n_rows, 4);

        // Copy entries
        buffer.copy_from_entries(&entries);

        // Verify the data was copied correctly
        let rows = buffer.local_rows();
        let cols = buffer.local_cols();
        let weights = buffer.weights();

        assert_eq!(rows.len(), 7);
        assert_eq!(cols.len(), 7);
        assert_eq!(weights.len(), 7);

        assert_eq!(rows, &[0, 0, 1, 3, 3, 3, 3]);
        assert_eq!(cols, &[0, 2, 1, 0, 1, 2, 3]);
        assert_eq!(weights, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

        // Build row starts
        buffer.build_row_starts();

        // Verify row_starts array
        let row_starts = buffer.row_starts();
        assert_eq!(row_starts.len(), 5); // n_rows + 1

        // Expected: [0, 2, 3, 3, 7]
        // Row 0 starts at index 0 (has 2 entries)
        // Row 1 starts at index 2 (has 1 entry)
        // Row 2 starts at index 3 (has 0 entries)
        // Row 3 starts at index 3 (has 4 entries)
        // Sentinel at index 7 (total entries)
        assert_eq!(row_starts[0], 0, "Row 0 should start at index 0");
        assert_eq!(row_starts[1], 2, "Row 1 should start at index 2");
        assert_eq!(row_starts[2], 3, "Row 2 should start at index 3");
        assert_eq!(row_starts[3], 3, "Row 3 should start at index 3");
        assert_eq!(row_starts[4], 7, "Sentinel should be at index 7");
    }

    #[test]
    fn test_soa_tile_buffer_edge_cases() {
        // Test with entries that could trigger the AVX2 issue
        // Create a pattern where we have exactly 4 entries per row
        let mut entries = Vec::new();
        let n_rows = 4;
        let n_cols = 4;

        // Fill each row with exactly 4 entries
        for row in 0..n_rows {
            for col in 0..n_cols {
                entries.push(TileEntry {
                    local_row: row as u16,
                    local_col: col as u16,
                    weight: (row * n_cols + col) as f64,
                });
            }
        }

        assert_eq!(entries.len(), 16);

        let mut buffer = SoaTileBuffer::new(entries.len(), n_rows);
        buffer.copy_from_entries(&entries);
        buffer.build_row_starts();

        // Verify row_starts
        let row_starts = buffer.row_starts();
        assert_eq!(row_starts[0], 0); // Row 0 starts at 0
        assert_eq!(row_starts[1], 4); // Row 1 starts at 4
        assert_eq!(row_starts[2], 8); // Row 2 starts at 8
        assert_eq!(row_starts[3], 12); // Row 3 starts at 12
        assert_eq!(row_starts[4], 16); // Sentinel

        // Verify we can access all elements safely
        let cols = buffer.local_cols();
        let weights = buffer.weights();

        for row in 0..n_rows {
            let start = row_starts[row] as usize;
            let end = row_starts[row + 1] as usize;

            assert!(
                end <= entries.len(),
                "End index {} exceeds entries {}",
                end,
                entries.len()
            );

            // Check that we can read 4 elements at once if aligned
            if end - start >= 4 {
                for chunk_start in (start..end).step_by(4) {
                    if chunk_start + 3 < end {
                        // Verify we can access these indices
                        let _col0 = cols[chunk_start];
                        let _col1 = cols[chunk_start + 1];
                        let _col2 = cols[chunk_start + 2];
                        let _col3 = cols[chunk_start + 3];

                        let _w0 = weights[chunk_start];
                        let _w1 = weights[chunk_start + 1];
                        let _w2 = weights[chunk_start + 2];
                        let _w3 = weights[chunk_start + 3];
                    }
                }
            }
        }
    }

    #[test]
    fn test_soa_tile_buffer_memory_layout() {
        // Test memory alignment and layout
        let entries = vec![
            TileEntry {
                local_row: 0,
                local_col: 1,
                weight: 0.5,
            },
            TileEntry {
                local_row: 1,
                local_col: 2,
                weight: 0.3,
            },
            TileEntry {
                local_row: 2,
                local_col: 0,
                weight: 0.7,
            },
        ];

        let mut buffer = SoaTileBuffer::new(entries.len(), 3);
        buffer.copy_from_entries(&entries);
        buffer.build_row_starts();

        // Check memory alignment (should be 32-byte aligned for AVX2)
        let row_starts_ptr = buffer.row_starts().as_ptr() as usize;
        let rows_ptr = buffer.local_rows().as_ptr() as usize;
        let cols_ptr = buffer.local_cols().as_ptr() as usize;
        let weights_ptr = buffer.weights().as_ptr() as usize;

        // Row starts should be at the beginning
        assert_eq!(
            row_starts_ptr,
            buffer.ptr as usize + buffer.row_starts_offset
        );

        // Each section should start at a 32-byte aligned offset
        assert_eq!(
            buffer.rows_offset % 32,
            0,
            "Rows offset should be 32-byte aligned"
        );
        assert_eq!(
            buffer.cols_offset % 32,
            0,
            "Cols offset should be 32-byte aligned"
        );
        assert_eq!(
            buffer.weights_offset % 32,
            0,
            "Weights offset should be 32-byte aligned"
        );

        // Verify no overlap between arrays
        let row_starts_end = row_starts_ptr + 4 * 2; // 4 u16s
        let rows_end = rows_ptr + 3 * 2; // 3 u16s
        let cols_end = cols_ptr + 3 * 2; // 3 u16s
        let weights_end = weights_ptr + 3 * 8; // 3 f64s

        assert!(row_starts_end <= rows_ptr, "row_starts overlaps with rows");
        assert!(rows_end <= cols_ptr, "rows overlaps with cols");
        assert!(cols_end <= weights_ptr, "cols overlaps with weights");
        assert!(
            weights_end <= buffer.ptr as usize + buffer.total_size,
            "weights exceeds allocation"
        );
    }

    #[test]
    fn test_avx2_benchmark_scenario() {
        // Replicate the exact scenario from the benchmark
        // 16x16 tile with 50% density = 128 entries
        let rows = 16;
        let cols = 16;
        let density = 0.5;
        let total_entries = rows * cols;
        let n_entries = (total_entries as f64 * density) as usize;

        assert_eq!(n_entries, 128);

        let mut entries = Vec::with_capacity(n_entries);
        for i in 0..n_entries {
            // Use the same pattern as the benchmark
            let row = (i * 7) % rows;
            let col = (i * 13) % cols;
            let weight = 0.1 + (i as f64 * 0.01).sin().abs();

            entries.push(TileEntry {
                local_row: row as u16,
                local_col: col as u16,
                weight,
            });
        }

        // Create the tile
        let tile = SparseTile::new(0, 0, 0, rows, 0, cols, entries.clone());

        // Verify the buffer was created correctly
        assert_eq!(tile.nnz(), 128);
        assert_eq!(tile.local_rows().len(), 128);
        assert_eq!(tile.local_cols().len(), 128);
        assert_eq!(tile.weights().len(), 128);

        // Check row_starts
        let row_starts = tile.row_starts().unwrap();
        assert_eq!(row_starts.len(), 17); // 16 rows + 1 sentinel

        // Verify all values in local_cols are valid
        for (i, &col) in tile.local_cols().iter().enumerate() {
            assert!(col < 16, "Entry {i} has invalid col value: {col}");
        }

        // Simulate what the AVX2 code does
        let tile_data: Vec<f64> = (0..cols).map(|i| i as f64 * 0.1).collect();
        let _result = vec![0.0; rows];

        // Test accessing data the way AVX2 does
        for row in 0..16 {
            let start = row_starts[row] as usize;
            let end = row_starts[row + 1] as usize;

            if start >= end {
                continue; // Empty row
            }

            let n_entries = end - start;
            let chunks = n_entries / 4;

            // Process in chunks of 4 like AVX2 does
            for chunk in 0..chunks {
                let base = start + chunk * 4;

                // This is where the segfault happens - check if we can access these
                assert!(
                    base + 3 < tile.local_cols().len(),
                    "Trying to access cols[{}..{}] but len={}",
                    base,
                    base + 3,
                    tile.local_cols().len()
                );

                let col0 = tile.local_cols()[base] as usize;
                let col1 = tile.local_cols()[base + 1] as usize;
                let col2 = tile.local_cols()[base + 2] as usize;
                let col3 = tile.local_cols()[base + 3] as usize;

                // Verify column indices are valid
                assert!(
                    col0 < tile_data.len(),
                    "col0={} >= tile_data.len={}",
                    col0,
                    tile_data.len()
                );
                assert!(
                    col1 < tile_data.len(),
                    "col1={} >= tile_data.len={}",
                    col1,
                    tile_data.len()
                );
                assert!(
                    col2 < tile_data.len(),
                    "col2={} >= tile_data.len={}",
                    col2,
                    tile_data.len()
                );
                assert!(
                    col3 < tile_data.len(),
                    "col3={} >= tile_data.len={}",
                    col3,
                    tile_data.len()
                );

                // Access the data
                let _d0 = tile_data[col0];
                let _d1 = tile_data[col1];
                let _d2 = tile_data[col2];
                let _d3 = tile_data[col3];

                let _w0 = tile.weights()[base];
                let _w1 = tile.weights()[base + 1];
                let _w2 = tile.weights()[base + 2];
                let _w3 = tile.weights()[base + 3];
            }
        }
    }

    #[test]
    fn test_sparse_tile_with_buffer() {
        let entries = vec![
            TileEntry {
                local_row: 0,
                local_col: 1,
                weight: 0.5,
            },
            TileEntry {
                local_row: 1,
                local_col: 2,
                weight: 0.3,
            },
            TileEntry {
                local_row: 2,
                local_col: 0,
                weight: 0.7,
            },
        ];

        let tile = SparseTile::new(0, 0, 0, 3, 0, 3, entries);

        // Verify we can access the data
        assert_eq!(tile.nnz(), 3);
        assert_eq!(tile.local_rows(), &[0, 1, 2]);
        assert_eq!(tile.local_cols(), &[1, 2, 0]);
        assert_eq!(tile.weights(), &[0.5, 0.3, 0.7]);

        // Test cloning
        let cloned = tile.clone();
        assert_eq!(cloned.nnz(), tile.nnz());
        assert_eq!(cloned.local_rows(), tile.local_rows());
        assert_eq!(cloned.local_cols(), tile.local_cols());
        assert_eq!(cloned.weights(), tile.weights());
    }

    fn create_test_sparse_weights() -> Vec<SparseWeights<f64>> {
        vec![
            SparseWeights::new(vec![1, 3, 5, 7], vec![0.1, 0.4, 0.3, 0.2], 10),
            SparseWeights::new(vec![0, 2, 4, 6, 8], vec![0.2, 0.2, 0.2, 0.2, 0.2], 10),
            SparseWeights::new(vec![3, 5, 9], vec![0.5, 0.3, 0.2], 10),
        ]
    }

    #[test]
    fn test_tile_entry_creation() {
        let entry = TileEntry {
            local_row: 5,
            local_col: 10,
            weight: 0.5,
        };
        assert_eq!(entry.local_row, 5);
        assert_eq!(entry.local_col, 10);
        assert_eq!(entry.weight, 0.5);
    }

    #[test]
    fn test_sparse_tile_contains() {
        let tile = SparseTile::<f64>::new(0, 1, 0, 4, 4, 8, vec![]);

        assert!(tile.contains(2, 6));
        assert!(!tile.contains(5, 6)); // row out of bounds
        assert!(!tile.contains(2, 9)); // col out of bounds
    }

    #[test]
    fn test_sparse_tile_coordinate_conversion() {
        let tile = SparseTile::<f64>::new(1, 2, 4, 8, 8, 12, vec![]);

        // Test global to local
        assert_eq!(tile.global_to_local(5, 10), Some((1, 2)));
        assert_eq!(tile.global_to_local(3, 10), None); // outside tile

        // Test local to global
        assert_eq!(tile.local_to_global(1, 2), (5, 10));
    }

    #[test]
    fn test_tiled_sparse_matrix_creation() {
        let sparse_rows = create_test_sparse_weights();
        let matrix = TiledSparseMatrix::from_sparse_rows(sparse_rows, 2, 4);

        assert_eq!(matrix.n_rows, 3);
        assert_eq!(matrix.n_cols, 10);
        assert_eq!(matrix.n_tile_rows, 2); // ceil(3/2)
        assert_eq!(matrix.n_tile_cols, 3); // ceil(10/4)
        assert_eq!(matrix.tile_row_size, 2);
        assert_eq!(matrix.tile_col_size, 4);
    }

    #[test]
    fn test_tiled_sparse_matrix_tiles() {
        let sparse_rows = create_test_sparse_weights();
        let matrix = TiledSparseMatrix::from_sparse_rows(sparse_rows, 2, 4);

        // Check that we have the right number of non-empty tiles
        assert!(!matrix.tiles.is_empty());
        assert!(matrix.tiles.len() <= 6); // At most 2x3 tiles

        // Verify all entries are preserved
        let total_entries = matrix.nnz();
        assert_eq!(total_entries, 12); // 4 + 5 + 3 entries
    }

    #[test]
    fn test_get_tiles_for_rows() {
        let sparse_rows = create_test_sparse_weights();
        let matrix = TiledSparseMatrix::from_sparse_rows(sparse_rows, 2, 4);

        let tiles_for_first_row = matrix.get_tiles_for_rows(0..1);
        assert!(tiles_for_first_row.iter().all(|t| t.row_start == 0));

        let tiles_for_all_rows = matrix.get_tiles_for_rows(0..3);
        assert_eq!(tiles_for_all_rows.len(), matrix.tiles.len());
    }

    #[test]
    fn test_get_tiles_for_cols() {
        let sparse_rows = create_test_sparse_weights();
        let matrix = TiledSparseMatrix::from_sparse_rows(sparse_rows, 2, 4);

        let tiles_for_first_cols = matrix.get_tiles_for_cols(0..4);
        assert!(tiles_for_first_cols.iter().all(|t| t.col_start < 4));
    }

    #[test]
    fn test_tiled_matvec() {
        use crate::ScalarBackend;

        let sparse_rows = create_test_sparse_weights();
        let matrix = TiledSparseMatrix::from_sparse_rows(sparse_rows.clone(), 2, 4);
        let primitives = ScalarBackend::new();

        let vector = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result = tiled_sparse_matvec(&matrix, &vector, &primitives);

        // Compute expected result manually
        // Row 0: 0.1*2 + 0.4*4 + 0.3*6 + 0.2*8 = 0.2 + 1.6 + 1.8 + 1.6 = 5.2
        // Row 1: 0.2*1 + 0.2*3 + 0.2*5 + 0.2*7 + 0.2*9 = 0.2 + 0.6 + 1.0 + 1.4 + 1.8 = 5.0
        // Row 2: 0.5*4 + 0.3*6 + 0.2*10 = 2.0 + 1.8 + 2.0 = 5.8

        assert_eq!(result.len(), 3);
        assert!((result[0] - 5.2).abs() < 1e-10);
        assert!((result[1] - 5.0).abs() < 1e-10);
        assert!((result[2] - 5.8).abs() < 1e-10);
    }

    #[test]
    fn test_tile_stats() {
        let sparse_rows = create_test_sparse_weights();
        let matrix = TiledSparseMatrix::from_sparse_rows(sparse_rows, 2, 4);

        let stats = matrix.tile_stats();
        assert!(stats.n_tiles > 0);
        assert!(stats.avg_entries_per_tile > 0.0);
        assert!(stats.min_entries_per_tile > 0);
        assert!(stats.max_entries_per_tile >= stats.min_entries_per_tile);
    }

    #[test]
    fn test_edge_tiles() {
        // Create a matrix where the last tile isn't full-sized
        let sparse_rows = vec![
            SparseWeights::new(vec![0, 5, 9], vec![0.3, 0.4, 0.3], 10),
            SparseWeights::new(vec![1, 6, 8], vec![0.2, 0.5, 0.3], 10),
            SparseWeights::new(vec![2, 4, 7], vec![0.4, 0.3, 0.3], 10),
        ];

        let matrix = TiledSparseMatrix::from_sparse_rows(sparse_rows, 2, 3);

        // Check edge tile dimensions
        let last_row_tiles = matrix.get_tiles_for_rows(2..3);
        for tile in last_row_tiles {
            assert_eq!(tile.row_end - tile.row_start, 1); // Only 1 row
        }

        let last_col_tiles = matrix.get_tiles_for_cols(9..10);
        for tile in last_col_tiles {
            assert_eq!(tile.col_end - tile.col_start, 1); // Only 1 column
        }
    }

    #[test]
    fn test_empty_tiles_not_stored() {
        // Create sparse rows with a gap - no entries in columns 4-7
        let sparse_rows = vec![
            SparseWeights::new(
                vec![0, 1, 2, 3, 8, 9],
                vec![0.1, 0.1, 0.1, 0.1, 0.3, 0.3],
                10,
            ),
            SparseWeights::new(vec![0, 2, 8, 9], vec![0.25, 0.25, 0.25, 0.25], 10),
        ];

        let matrix = TiledSparseMatrix::from_sparse_rows(sparse_rows, 2, 4);

        // Should not have a tile for columns 4-7
        let middle_tile = matrix.get_tile(0, 1);
        assert!(middle_tile.is_none());

        // Verify using stats
        let stats = matrix.tile_stats();
        assert!(stats.n_empty_tiles > 0);
    }

    #[test]
    fn test_sparsity_calculation() {
        let sparse_rows = create_test_sparse_weights();
        let matrix = TiledSparseMatrix::from_sparse_rows(sparse_rows, 2, 4);

        let sparsity = matrix.sparsity();
        // 12 non-zeros out of 30 total = 18/30 = 0.6 sparsity
        assert!((sparsity - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_batch_matvec() {
        use crate::ScalarBackend;

        let sparse_rows = create_test_sparse_weights();
        let matrix = TiledSparseMatrix::from_sparse_rows(sparse_rows, 2, 4);
        let primitives = ScalarBackend::new();

        let vectors = [vec![1.0; 10], vec![2.0; 10], vec![0.5; 10]];
        let vector_refs: Vec<&[f64]> = vectors.iter().map(|v| v.as_slice()).collect();

        let results = tiled_sparse_matvec_batch(&matrix, &vector_refs, &primitives);
        assert_eq!(results.len(), 3);

        // Each result should be the same pattern scaled by the constant
        for result in &results {
            assert_eq!(result.len(), 3);
        }
    }

    #[test]
    #[should_panic(expected = "Cannot create matrix from empty rows")]
    fn test_empty_rows_panic() {
        let sparse_rows: Vec<SparseWeights<f64>> = vec![];
        TiledSparseMatrix::from_sparse_rows(sparse_rows, 2, 4);
    }

    #[test]
    #[should_panic(expected = "Tile row size must be positive")]
    fn test_zero_tile_size_panic() {
        let sparse_rows = create_test_sparse_weights();
        TiledSparseMatrix::from_sparse_rows(sparse_rows, 0, 4);
    }

    #[test]
    #[should_panic(expected = "Vector length must match matrix columns")]
    fn test_matvec_dimension_mismatch() {
        use crate::ScalarBackend;

        let sparse_rows = create_test_sparse_weights();
        let matrix = TiledSparseMatrix::from_sparse_rows(sparse_rows, 2, 4);
        let vector = vec![1.0; 5]; // Wrong size
        let primitives = ScalarBackend::new();
        tiled_sparse_matvec(&matrix, &vector, &primitives);
    }
}
