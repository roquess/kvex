//! # sied - SIMD operations for Erlang
//!
//! This crate provides high-performance vectorized operations for Erlang
//! through Rustler NIFs. All implementations use simdeez for portable
//! SIMD abstractions with runtime dispatch.
//!
//! ## Features
//! - Runtime SIMD detection (SSE2, SSE4.1, AVX2, NEON)
//! - Comprehensive mathematical operations
//! - Safe Rust with automatic SIMD optimization
//! - Zero unsafe code
use rustler::{Binary, Encoder, Env, Error, OwnedBinary, Term};
use simdeez::prelude::*;

/// Atom definitions for consistent error and success responses
mod atoms {
    rustler::atoms! {
        ok,
        error,
        length_mismatch,
        invalid_input,
        empty_vector,
        zero_vector
    }
}

rustler::init!("sied");

//==============================================================================
// Basic Arithmetic Operations
//==============================================================================
/// Element-wise addition of two f32 vectors
#[rustler::nif]
fn add_f32(env: Env, a: Vec<f32>, b: Vec<f32>) -> Result<Term, Error> {
    if a.len() != b.len() {
        return Ok((atoms::error(), atoms::length_mismatch()).encode(env));
    }
    let result = add_f32_simd(&a, &b);
    Ok((atoms::ok(), result).encode(env))
}

/// Element-wise addition of two f64 vectors
#[rustler::nif]
fn add_f64(env: Env, a: Vec<f64>, b: Vec<f64>) -> Result<Term, Error> {
    if a.len() != b.len() {
        return Ok((atoms::error(), atoms::length_mismatch()).encode(env));
    }
    let result = add_f64_simd(&a, &b);
    Ok((atoms::ok(), result).encode(env))
}

/// Element-wise subtraction of two f32 vectors
#[rustler::nif]
fn subtract_f32(env: Env, a: Vec<f32>, b: Vec<f32>) -> Result<Term, Error> {
    if a.len() != b.len() {
        return Ok((atoms::error(), atoms::length_mismatch()).encode(env));
    }
    let result = subtract_f32_simd(&a, &b);
    Ok((atoms::ok(), result).encode(env))
}

/// Element-wise subtraction of two f64 vectors
#[rustler::nif]
fn subtract_f64(env: Env, a: Vec<f64>, b: Vec<f64>) -> Result<Term, Error> {
    if a.len() != b.len() {
        return Ok((atoms::error(), atoms::length_mismatch()).encode(env));
    }
    let result = subtract_f64_simd(&a, &b);
    Ok((atoms::ok(), result).encode(env))
}

/// Element-wise multiplication of two f32 vectors
#[rustler::nif]
fn multiply_f32(env: Env, a: Vec<f32>, b: Vec<f32>) -> Result<Term, Error> {
    if a.len() != b.len() {
        return Ok((atoms::error(), atoms::length_mismatch()).encode(env));
    }
    let result = multiply_f32_simd(&a, &b);
    Ok((atoms::ok(), result).encode(env))
}

/// Element-wise multiplication of two f64 vectors
#[rustler::nif]
fn multiply_f64(env: Env, a: Vec<f64>, b: Vec<f64>) -> Result<Term, Error> {
    if a.len() != b.len() {
        return Ok((atoms::error(), atoms::length_mismatch()).encode(env));
    }
    let result = multiply_f64_simd(&a, &b);
    Ok((atoms::ok(), result).encode(env))
}

/// Element-wise division of two f32 vectors
#[rustler::nif]
fn divide_f32(env: Env, a: Vec<f32>, b: Vec<f32>) -> Result<Term, Error> {
    if a.len() != b.len() {
        return Ok((atoms::error(), atoms::length_mismatch()).encode(env));
    }
    let result = divide_f32_simd(&a, &b);
    Ok((atoms::ok(), result).encode(env))
}

/// Element-wise division of two f64 vectors
#[rustler::nif]
fn divide_f64(env: Env, a: Vec<f64>, b: Vec<f64>) -> Result<Term, Error> {
    if a.len() != b.len() {
        return Ok((atoms::error(), atoms::length_mismatch()).encode(env));
    }
    let result = divide_f64_simd(&a, &b);
    Ok((atoms::ok(), result).encode(env))
}

//==============================================================================
// Reduction Operations
//==============================================================================
/// Dot product of two f32 vectors
#[rustler::nif]
fn dot_product_f32(env: Env, a: Vec<f32>, b: Vec<f32>) -> Result<Term, Error> {
    if a.len() != b.len() {
        return Ok((atoms::error(), atoms::length_mismatch()).encode(env));
    }
    let result = dot_product_f32_simd(&a, &b);
    Ok((atoms::ok(), result).encode(env))
}

/// Dot product of two f64 vectors
#[rustler::nif]
fn dot_product_f64(env: Env, a: Vec<f64>, b: Vec<f64>) -> Result<Term, Error> {
    if a.len() != b.len() {
        return Ok((atoms::error(), atoms::length_mismatch()).encode(env));
    }
    let result = dot_product_f64_simd(&a, &b);
    Ok((atoms::ok(), result).encode(env))
}

/// Sum of all elements in an f32 vector
#[rustler::nif]
fn sum_f32(env: Env, a: Vec<f32>) -> Result<Term, Error> {
    let result = sum_f32_simd(&a);
    Ok((atoms::ok(), result).encode(env))
}

/// Sum of all elements in an f64 vector
#[rustler::nif]
fn sum_f64(env: Env, a: Vec<f64>) -> Result<Term, Error> {
    let result = sum_f64_simd(&a);
    Ok((atoms::ok(), result).encode(env))
}

//==============================================================================
// Statistical Operations
//==============================================================================
/// Mean (average) of an f32 vector
#[rustler::nif]
fn mean_f32(env: Env, a: Vec<f32>) -> Result<Term, Error> {
    if a.is_empty() {
        return Ok((atoms::error(), atoms::empty_vector()).encode(env));
    }
    let result = sum_f32_simd(&a) / a.len() as f32;
    Ok((atoms::ok(), result).encode(env))
}

/// Mean (average) of an f64 vector
#[rustler::nif]
fn mean_f64(env: Env, a: Vec<f64>) -> Result<Term, Error> {
    if a.is_empty() {
        return Ok((atoms::error(), atoms::empty_vector()).encode(env));
    }
    let result = sum_f64_simd(&a) / a.len() as f64;
    Ok((atoms::ok(), result).encode(env))
}

/// Variance of an f32 vector
#[rustler::nif]
fn variance_f32(env: Env, a: Vec<f32>) -> Result<Term, Error> {
    if a.is_empty() {
        return Ok((atoms::error(), atoms::empty_vector()).encode(env));
    }
    let mean = sum_f32_simd(&a) / a.len() as f32;
    let mean_vec = vec![mean; a.len()];
    let diff = subtract_f32_simd(&a, &mean_vec);
    let squared = multiply_f32_simd(&diff, &diff);
    let result = sum_f32_simd(&squared) / a.len() as f32;
    Ok((atoms::ok(), result).encode(env))
}

/// Variance of an f64 vector
#[rustler::nif]
fn variance_f64(env: Env, a: Vec<f64>) -> Result<Term, Error> {
    if a.is_empty() {
        return Ok((atoms::error(), atoms::empty_vector()).encode(env));
    }
    let mean = sum_f64_simd(&a) / a.len() as f64;
    let mean_vec = vec![mean; a.len()];
    let diff = subtract_f64_simd(&a, &mean_vec);
    let squared = multiply_f64_simd(&diff, &diff);
    let result = sum_f64_simd(&squared) / a.len() as f64;
    Ok((atoms::ok(), result).encode(env))
}

/// Standard deviation of an f32 vector
#[rustler::nif]
fn std_dev_f32(env: Env, a: Vec<f32>) -> Result<Term, Error> {
    if a.is_empty() {
        return Ok((atoms::error(), atoms::empty_vector()).encode(env));
    }
    let mean = sum_f32_simd(&a) / a.len() as f32;
    let mean_vec = vec![mean; a.len()];
    let diff = subtract_f32_simd(&a, &mean_vec);
    let squared = multiply_f32_simd(&diff, &diff);
    let variance = sum_f32_simd(&squared) / a.len() as f32;
    let result = variance.sqrt();
    Ok((atoms::ok(), result).encode(env))
}

/// Standard deviation of an f64 vector
#[rustler::nif]
fn std_dev_f64(env: Env, a: Vec<f64>) -> Result<Term, Error> {
    if a.is_empty() {
        return Ok((atoms::error(), atoms::empty_vector()).encode(env));
    }
    let mean = sum_f64_simd(&a) / a.len() as f64;
    let mean_vec = vec![mean; a.len()];
    let diff = subtract_f64_simd(&a, &mean_vec);
    let squared = multiply_f64_simd(&diff, &diff);
    let variance = sum_f64_simd(&squared) / a.len() as f64;
    let result = variance.sqrt();
    Ok((atoms::ok(), result).encode(env))
}

//==============================================================================
// Min/Max Operations
//==============================================================================
/// Minimum value in an f32 vector
#[rustler::nif]
fn min_f32(env: Env, a: Vec<f32>) -> Result<Term, Error> {
    if a.is_empty() {
        return Ok((atoms::error(), atoms::empty_vector()).encode(env));
    }
    let result = min_f32_simd(&a);
    Ok((atoms::ok(), result).encode(env))
}

/// Minimum value in an f64 vector
#[rustler::nif]
fn min_f64(env: Env, a: Vec<f64>) -> Result<Term, Error> {
    if a.is_empty() {
        return Ok((atoms::error(), atoms::empty_vector()).encode(env));
    }
    let result = min_f64_simd(&a);
    Ok((atoms::ok(), result).encode(env))
}

/// Maximum value in an f32 vector
#[rustler::nif]
fn max_f32(env: Env, a: Vec<f32>) -> Result<Term, Error> {
    if a.is_empty() {
        return Ok((atoms::error(), atoms::empty_vector()).encode(env));
    }
    let result = max_f32_simd(&a);
    Ok((atoms::ok(), result).encode(env))
}

/// Maximum value in an f64 vector
#[rustler::nif]
fn max_f64(env: Env, a: Vec<f64>) -> Result<Term, Error> {
    if a.is_empty() {
        return Ok((atoms::error(), atoms::empty_vector()).encode(env));
    }
    let result = max_f64_simd(&a);
    Ok((atoms::ok(), result).encode(env))
}

/// Element-wise minimum of two f32 vectors
#[rustler::nif]
fn min_elementwise_f32(env: Env, a: Vec<f32>, b: Vec<f32>) -> Result<Term, Error> {
    if a.len() != b.len() {
        return Ok((atoms::error(), atoms::length_mismatch()).encode(env));
    }
    let result = min_elementwise_f32_simd(&a, &b);
    Ok((atoms::ok(), result).encode(env))
}

/// Element-wise minimum of two f64 vectors
#[rustler::nif]
fn min_elementwise_f64(env: Env, a: Vec<f64>, b: Vec<f64>) -> Result<Term, Error> {
    if a.len() != b.len() {
        return Ok((atoms::error(), atoms::length_mismatch()).encode(env));
    }
    let result = min_elementwise_f64_simd(&a, &b);
    Ok((atoms::ok(), result).encode(env))
}

/// Element-wise maximum of two f32 vectors
#[rustler::nif]
fn max_elementwise_f32(env: Env, a: Vec<f32>, b: Vec<f32>) -> Result<Term, Error> {
    if a.len() != b.len() {
        return Ok((atoms::error(), atoms::length_mismatch()).encode(env));
    }
    let result = max_elementwise_f32_simd(&a, &b);
    Ok((atoms::ok(), result).encode(env))
}

/// Element-wise maximum of two f64 vectors
#[rustler::nif]
fn max_elementwise_f64(env: Env, a: Vec<f64>, b: Vec<f64>) -> Result<Term, Error> {
    if a.len() != b.len() {
        return Ok((atoms::error(), atoms::length_mismatch()).encode(env));
    }
    let result = max_elementwise_f64_simd(&a, &b);
    Ok((atoms::ok(), result).encode(env))
}

//==============================================================================
// Unary Operations
//==============================================================================
/// Absolute value of an f32 vector
#[rustler::nif]
fn abs_f32(env: Env, a: Vec<f32>) -> Result<Term, Error> {
    let result = abs_f32_simd(&a);
    Ok((atoms::ok(), result).encode(env))
}

/// Absolute value of an f64 vector
#[rustler::nif]
fn abs_f64(env: Env, a: Vec<f64>) -> Result<Term, Error> {
    let result = abs_f64_simd(&a);
    Ok((atoms::ok(), result).encode(env))
}

/// Square root of an f32 vector
#[rustler::nif]
fn sqrt_f32(env: Env, a: Vec<f32>) -> Result<Term, Error> {
    let result = sqrt_f32_simd(&a);
    Ok((atoms::ok(), result).encode(env))
}

/// Square root of an f64 vector
#[rustler::nif]
fn sqrt_f64(env: Env, a: Vec<f64>) -> Result<Term, Error> {
    let result = sqrt_f64_simd(&a);
    Ok((atoms::ok(), result).encode(env))
}

/// Negate an f32 vector
#[rustler::nif]
fn negate_f32(env: Env, a: Vec<f32>) -> Result<Term, Error> {
    let result = negate_f32_simd(&a);
    Ok((atoms::ok(), result).encode(env))
}

/// Negate an f64 vector
#[rustler::nif]
fn negate_f64(env: Env, a: Vec<f64>) -> Result<Term, Error> {
    let result = negate_f64_simd(&a);
    Ok((atoms::ok(), result).encode(env))
}

//==============================================================================
// Batch Operations (for vector search)
//==============================================================================

/// Batch dot product: compute dot product of one query against many f32 vectors
/// Returns {ok, [Score]} in the same order as the input list.
/// One NIF call instead of N — critical for kvex search performance.
#[rustler::nif(schedule = "DirtyCpu")]
fn dot_product_batch_f32(env: Env, query: Vec<f32>, vecs: Vec<Vec<f32>>) -> Result<Term, Error> {
    if vecs.is_empty() {
        return Ok((atoms::ok(), Vec::<f32>::new()).encode(env));
    }
    let dim = query.len();
    for v in &vecs {
        if v.len() != dim {
            return Ok((atoms::error(), atoms::length_mismatch()).encode(env));
        }
    }
    let scores: Vec<f32> = vecs.iter().map(|v| dot_product_f32_simd(&query, v)).collect();
    Ok((atoms::ok(), scores).encode(env))
}

/// Batch dot product: query (f32 binary) against many f32 binaries.
/// Avoids Erlang list-of-floats marshalling — binaries are zero-copy in the NIF.
/// Each binary must be 4*dim bytes (little-endian f32).
/// Not marked DirtyCpu — for K*OVERSAMPLE=100 candidates of 512 bytes each
/// the NIF completes in <200μs, acceptable to run on a normal scheduler.
#[rustler::nif]
fn dot_product_batch_f32_bin<'a>(
    env: Env<'a>,
    query: Binary<'a>,
    vecs: Vec<Binary<'a>>,
) -> Result<Term<'a>, Error> {
    if vecs.is_empty() {
        return Ok((atoms::ok(), Vec::<f32>::new()).encode(env));
    }
    let q = binary_to_f32_slice(query.as_slice());
    let scores: Vec<f32> = vecs
        .iter()
        .map(|v| dot_product_f32_simd(&q, &binary_to_f32_slice(v.as_slice())))
        .collect();
    Ok((atoms::ok(), scores).encode(env))
}

fn binary_to_f32_slice(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Batch dot product: compute dot product of one query against many f64 vectors
#[rustler::nif(schedule = "DirtyCpu")]
fn dot_product_batch_f64(env: Env, query: Vec<f64>, vecs: Vec<Vec<f64>>) -> Result<Term, Error> {
    if vecs.is_empty() {
        return Ok((atoms::ok(), Vec::<f64>::new()).encode(env));
    }
    let dim = query.len();
    for v in &vecs {
        if v.len() != dim {
            return Ok((atoms::error(), atoms::length_mismatch()).encode(env));
        }
    }
    let scores: Vec<f64> = vecs.iter().map(|v| dot_product_f64_simd(&query, v)).collect();
    Ok((atoms::ok(), scores).encode(env))
}

//==============================================================================
// Vector Norm and Normalization
//==============================================================================

/// L2 norm (Euclidean length) of an f32 vector
#[rustler::nif]
fn l2_norm_f32(env: Env, a: Vec<f32>) -> Result<Term, Error> {
    if a.is_empty() {
        return Ok((atoms::error(), atoms::empty_vector()).encode(env));
    }
    let result = dot_product_f32_simd(&a, &a).sqrt();
    Ok((atoms::ok(), result).encode(env))
}

/// L2 norm (Euclidean length) of an f64 vector
#[rustler::nif]
fn l2_norm_f64(env: Env, a: Vec<f64>) -> Result<Term, Error> {
    if a.is_empty() {
        return Ok((atoms::error(), atoms::empty_vector()).encode(env));
    }
    let result = dot_product_f64_simd(&a, &a).sqrt();
    Ok((atoms::ok(), result).encode(env))
}

/// L2-normalize an f32 vector to unit length.
/// Returns the original vector unchanged if its norm is zero.
#[rustler::nif]
fn l2_normalize_f32(env: Env, a: Vec<f32>) -> Result<Term, Error> {
    if a.is_empty() {
        return Ok((atoms::error(), atoms::empty_vector()).encode(env));
    }
    let norm = dot_product_f32_simd(&a, &a).sqrt();
    if norm == 0.0 {
        return Ok((atoms::ok(), a).encode(env));
    }
    let result = scale_f32_simd(&a, 1.0 / norm);
    Ok((atoms::ok(), result).encode(env))
}

/// L2-normalize an f64 vector to unit length.
#[rustler::nif]
fn l2_normalize_f64(env: Env, a: Vec<f64>) -> Result<Term, Error> {
    if a.is_empty() {
        return Ok((atoms::error(), atoms::empty_vector()).encode(env));
    }
    let norm = dot_product_f64_simd(&a, &a).sqrt();
    if norm == 0.0 {
        return Ok((atoms::ok(), a).encode(env));
    }
    let result = scale_f64_simd(&a, 1.0 / norm);
    Ok((atoms::ok(), result).encode(env))
}

/// L2-normalize a batch of f32 vectors
#[rustler::nif(schedule = "DirtyCpu")]
fn l2_normalize_batch_f32(env: Env, vecs: Vec<Vec<f32>>) -> Result<Term, Error> {
    let result: Vec<Vec<f32>> = vecs.iter().map(|v| {
        if v.is_empty() { return v.clone(); }
        let norm = dot_product_f32_simd(v, v).sqrt();
        if norm == 0.0 { v.clone() } else { scale_f32_simd(v, 1.0 / norm) }
    }).collect();
    Ok((atoms::ok(), result).encode(env))
}

/// L2-normalize a batch of f64 vectors
#[rustler::nif(schedule = "DirtyCpu")]
fn l2_normalize_batch_f64(env: Env, vecs: Vec<Vec<f64>>) -> Result<Term, Error> {
    let result: Vec<Vec<f64>> = vecs.iter().map(|v| {
        if v.is_empty() { return v.clone(); }
        let norm = dot_product_f64_simd(v, v).sqrt();
        if norm == 0.0 { v.clone() } else { scale_f64_simd(v, 1.0 / norm) }
    }).collect();
    Ok((atoms::ok(), result).encode(env))
}

//==============================================================================
// Cosine Similarity
//==============================================================================

/// Cosine similarity between two f32 vectors: dot(A,B) / (|A| * |B|)
#[rustler::nif]
fn cosine_similarity_f32(env: Env, a: Vec<f32>, b: Vec<f32>) -> Result<Term, Error> {
    if a.len() != b.len() {
        return Ok((atoms::error(), atoms::length_mismatch()).encode(env));
    }
    if a.is_empty() {
        return Ok((atoms::error(), atoms::empty_vector()).encode(env));
    }
    let dot = dot_product_f32_simd(&a, &b);
    let norm_a = dot_product_f32_simd(&a, &a).sqrt();
    let norm_b = dot_product_f32_simd(&b, &b).sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return Ok((atoms::error(), atoms::zero_vector()).encode(env));
    }
    Ok((atoms::ok(), dot / (norm_a * norm_b)).encode(env))
}

/// Cosine similarity between two f64 vectors
#[rustler::nif]
fn cosine_similarity_f64(env: Env, a: Vec<f64>, b: Vec<f64>) -> Result<Term, Error> {
    if a.len() != b.len() {
        return Ok((atoms::error(), atoms::length_mismatch()).encode(env));
    }
    if a.is_empty() {
        return Ok((atoms::error(), atoms::empty_vector()).encode(env));
    }
    let dot = dot_product_f64_simd(&a, &b);
    let norm_a = dot_product_f64_simd(&a, &a).sqrt();
    let norm_b = dot_product_f64_simd(&b, &b).sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return Ok((atoms::error(), atoms::zero_vector()).encode(env));
    }
    Ok((atoms::ok(), dot / (norm_a * norm_b)).encode(env))
}

/// Batch cosine similarity: one f32 query against many f32 vectors
#[rustler::nif(schedule = "DirtyCpu")]
fn cosine_similarity_batch_f32(env: Env, query: Vec<f32>, vecs: Vec<Vec<f32>>) -> Result<Term, Error> {
    if vecs.is_empty() {
        return Ok((atoms::ok(), Vec::<f32>::new()).encode(env));
    }
    let dim = query.len();
    let query_norm = dot_product_f32_simd(&query, &query).sqrt();
    if query_norm == 0.0 {
        return Ok((atoms::error(), atoms::zero_vector()).encode(env));
    }
    let mut scores = Vec::with_capacity(vecs.len());
    for v in &vecs {
        if v.len() != dim {
            return Ok((atoms::error(), atoms::length_mismatch()).encode(env));
        }
        let norm_v = dot_product_f32_simd(v, v).sqrt();
        if norm_v == 0.0 {
            scores.push(0.0f32);
        } else {
            scores.push(dot_product_f32_simd(&query, v) / (query_norm * norm_v));
        }
    }
    Ok((atoms::ok(), scores).encode(env))
}

/// Batch cosine similarity: one f64 query against many f64 vectors
#[rustler::nif(schedule = "DirtyCpu")]
fn cosine_similarity_batch_f64(env: Env, query: Vec<f64>, vecs: Vec<Vec<f64>>) -> Result<Term, Error> {
    if vecs.is_empty() {
        return Ok((atoms::ok(), Vec::<f64>::new()).encode(env));
    }
    let dim = query.len();
    let query_norm = dot_product_f64_simd(&query, &query).sqrt();
    if query_norm == 0.0 {
        return Ok((atoms::error(), atoms::zero_vector()).encode(env));
    }
    let mut scores = Vec::with_capacity(vecs.len());
    for v in &vecs {
        if v.len() != dim {
            return Ok((atoms::error(), atoms::length_mismatch()).encode(env));
        }
        let norm_v = dot_product_f64_simd(v, v).sqrt();
        if norm_v == 0.0 {
            scores.push(0.0f64);
        } else {
            scores.push(dot_product_f64_simd(&query, v) / (query_norm * norm_v));
        }
    }
    Ok((atoms::ok(), scores).encode(env))
}

//==============================================================================
// Binary Quantization (1-bit) and Hamming Distance
//==============================================================================

/// 1-bit quantize an f32 binary: accepts little-endian f32 bytes instead of a float list.
/// Avoids Erlang list marshalling when the vector is already stored as a binary.
#[rustler::nif]
fn to_binary_f32_bin<'a>(env: Env<'a>, data: Binary<'a>) -> Result<Term<'a>, Error> {
    if data.is_empty() {
        return Ok((atoms::error(), atoms::empty_vector()).encode(env));
    }
    let vec = binary_to_f32_slice(data.as_slice());
    quantize_to_binary_f32(env, &vec)
}

/// 1-bit quantize an f32 vector: each element becomes 1 if above mean, else 0.
/// Returns a packed Erlang binary. A 128-dim vector becomes 16 bytes (8x compression
/// vs f32, 32x vs full precision).
#[rustler::nif]
fn to_binary_f32<'a>(env: Env<'a>, vec: Vec<f32>) -> Result<Term<'a>, Error> {
    if vec.is_empty() {
        return Ok((atoms::error(), atoms::empty_vector()).encode(env));
    }
    quantize_to_binary_f32(env, &vec)
}

fn quantize_to_binary_f32<'a>(env: Env<'a>, vec: &[f32]) -> Result<Term<'a>, Error> {
    let mean = sum_f32_simd(vec) / vec.len() as f32;
    let n_bytes = (vec.len() + 7) / 8;
    let mut bytes = vec![0u8; n_bytes];
    for (i, &val) in vec.iter().enumerate() {
        if val > mean {
            bytes[i / 8] |= 1u8 << (i % 8);
        }
    }
    let mut owned = match OwnedBinary::new(n_bytes) {
        Some(b) => b,
        None => return Ok((atoms::error(), atoms::invalid_input()).encode(env)),
    };
    owned.as_mut_slice().copy_from_slice(&bytes);
    Ok((atoms::ok(), owned.release(env)).encode(env))
}

/// Batch hamming distance: compute hamming distance between a query binary
/// and a list of stored binaries. Lower distance = more similar.
/// Uses u64 POPCNT instructions for speed (auto-vectorized by LLVM).
/// Intended for the first phase of two-phase ANN search in kvex.
/// Not marked DirtyCpu — 10k × 16-byte POPCNT completes in <500μs,
/// so blocking a normal scheduler briefly is acceptable.
#[rustler::nif]
fn hamming_distance_batch<'a>(
    env: Env<'a>,
    query: Binary<'a>,
    vecs: Vec<Binary<'a>>,
) -> Result<Term<'a>, Error> {
    let q = query.as_slice();
    let distances: Vec<u32> = vecs.iter()
        .map(|v| hamming_distance_impl(q, v.as_slice()))
        .collect();
    Ok((atoms::ok(), distances).encode(env))
}

fn hamming_distance_impl(a: &[u8], b: &[u8]) -> u32 {
    let n = a.len().min(b.len());
    let mut dist = 0u32;
    let full_chunks = n / 8;
    for i in 0..full_chunks {
        let ai = u64::from_le_bytes(a[i * 8..i * 8 + 8].try_into().unwrap());
        let bi = u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap());
        dist += (ai ^ bi).count_ones();
    }
    for i in (full_chunks * 8)..n {
        dist += (a[i] ^ b[i]).count_ones();
    }
    dist
}

/// Hamming top-K from a flat binary buffer.
/// flat_vecs: all binary-quantized vectors concatenated (vec_len bytes each).
/// Returns the indices (0-based, u32) of the top_k closest vectors sorted by
/// ascending Hamming distance.
///
/// Uses a max-heap of size K — allocates O(K) instead of O(N), avoids
/// the N-element Vec and select_nth_unstable overhead.
#[rustler::nif]
fn hamming_topk_flat<'a>(
    env: Env<'a>,
    query: Binary<'a>,
    flat_vecs: Binary<'a>,
    vec_len: usize,
    top_k: usize,
) -> Result<Term<'a>, Error> {
    if vec_len == 0 {
        return Ok((atoms::error(), atoms::invalid_input()).encode(env));
    }
    let data = flat_vecs.as_slice();
    let n = data.len() / vec_len;
    if n == 0 || top_k == 0 {
        return Ok((atoms::ok(), Vec::<u32>::new()).encode(env));
    }
    let q   = query.as_slice();
    let k   = top_k.min(n);
    // Max-heap of (dist, idx): we keep the K smallest distances.
    // When full, peek the worst (largest dist) and replace if current is better.
    let mut heap: std::collections::BinaryHeap<(u32, u32)> =
        std::collections::BinaryHeap::with_capacity(k + 1);
    for i in 0..n as u32 {
        let start = i as usize * vec_len;
        let dist  = hamming_distance_impl(q, &data[start..start + vec_len]);
        if heap.len() < k {
            heap.push((dist, i));
        } else if let Some(&(worst, _)) = heap.peek() {
            if dist < worst {
                heap.pop();
                heap.push((dist, i));
            }
        }
    }
    // into_sorted_vec returns ascending order for a max-heap (pops largest first, reverses)
    let sorted = heap.into_sorted_vec();
    let indices: Vec<u32> = sorted.iter().map(|(_, idx)| *idx).collect();
    Ok((atoms::ok(), indices).encode(env))
}

/// Zero-copy reinterpretation of a LE f32 byte slice as &[f32].
/// SAFETY: Erlang stores f32 binaries as IEEE 754 little-endian, matching
/// native layout on all BEAM-supported architectures (x86-64, ARM64, RISC-V LE).
#[inline]
fn bytes_as_f32(bytes: &[u8]) -> &[f32] {
    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4) }
}

/// Dot-product scoring of selected vectors from a flat f32 binary.
/// flat_f32: all f32 vectors concatenated (vec_byte_len bytes = 4*dim each).
/// indices: which vectors to score (produced by hamming_topk_flat).
/// Returns {ok, [{Score, Idx}]} sorted by descending score.
///
/// Zero-copy: reinterprets the flat binary as &[f32] without allocation.
#[rustler::nif]
fn dot_product_topk_flat<'a>(
    env: Env<'a>,
    query: Binary<'a>,
    flat_f32: Binary<'a>,
    vec_byte_len: usize,
    indices: Vec<u32>,
) -> Result<Term<'a>, Error> {
    if vec_byte_len == 0 || vec_byte_len % 4 != 0 {
        return Ok((atoms::error(), atoms::invalid_input()).encode(env));
    }
    let q    = bytes_as_f32(query.as_slice());
    let data = flat_f32.as_slice();
    let dim  = vec_byte_len / 4;
    let mut scored: Vec<(f32, u32)> = indices
        .iter()
        .filter_map(|&idx| {
            let start = idx as usize * dim;
            let end   = start + dim;
            let all   = bytes_as_f32(data);
            if end > all.len() { return None; }
            Some((dot_product_f32_simd(q, &all[start..end]), idx))
        })
        .collect();
    scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    Ok((atoms::ok(), scored).encode(env))
}

//==============================================================================
// SIMD Implementation Functions using simdeez
//==============================================================================
simd_runtime_generate! {
    fn add_f32_simd(a: &[f32], b: &[f32]) -> Vec<f32> {
        let mut result = Vec::with_capacity(a.len());
        unsafe { result.set_len(a.len()); }
        let mut a_slice = &a[..];
        let mut b_slice = &b[..];
        let mut res_slice = &mut result[..];
        while a_slice.len() >= S::Vf32::WIDTH {
            let va = S::Vf32::load_from_slice(a_slice);
            let vb = S::Vf32::load_from_slice(b_slice);
            let vr = va + vb;
            vr.copy_to_slice(res_slice);
            a_slice = &a_slice[S::Vf32::WIDTH..];
            b_slice = &b_slice[S::Vf32::WIDTH..];
            res_slice = &mut res_slice[S::Vf32::WIDTH..];
        }
        for i in 0..a_slice.len() {
            res_slice[i] = a_slice[i] + b_slice[i];
        }
        result
    }
}

simd_runtime_generate! {
    fn add_f64_simd(a: &[f64], b: &[f64]) -> Vec<f64> {
        let mut result = Vec::with_capacity(a.len());
        unsafe { result.set_len(a.len()); }
        let mut a_slice = &a[..];
        let mut b_slice = &b[..];
        let mut res_slice = &mut result[..];
        while a_slice.len() >= S::Vf64::WIDTH {
            let va = S::Vf64::load_from_slice(a_slice);
            let vb = S::Vf64::load_from_slice(b_slice);
            let vr = va + vb;
            vr.copy_to_slice(res_slice);
            a_slice = &a_slice[S::Vf64::WIDTH..];
            b_slice = &b_slice[S::Vf64::WIDTH..];
            res_slice = &mut res_slice[S::Vf64::WIDTH..];
        }
        for i in 0..a_slice.len() {
            res_slice[i] = a_slice[i] + b_slice[i];
        }
        result
    }
}

simd_runtime_generate! {
    fn subtract_f32_simd(a: &[f32], b: &[f32]) -> Vec<f32> {
        let mut result = Vec::with_capacity(a.len());
        unsafe { result.set_len(a.len()); }
        let mut a_slice = &a[..];
        let mut b_slice = &b[..];
        let mut res_slice = &mut result[..];
        while a_slice.len() >= S::Vf32::WIDTH {
            let va = S::Vf32::load_from_slice(a_slice);
            let vb = S::Vf32::load_from_slice(b_slice);
            let vr = va - vb;
            vr.copy_to_slice(res_slice);
            a_slice = &a_slice[S::Vf32::WIDTH..];
            b_slice = &b_slice[S::Vf32::WIDTH..];
            res_slice = &mut res_slice[S::Vf32::WIDTH..];
        }
        for i in 0..a_slice.len() {
            res_slice[i] = a_slice[i] - b_slice[i];
        }
        result
    }
}

simd_runtime_generate! {
    fn subtract_f64_simd(a: &[f64], b: &[f64]) -> Vec<f64> {
        let mut result = Vec::with_capacity(a.len());
        unsafe { result.set_len(a.len()); }
        let mut a_slice = &a[..];
        let mut b_slice = &b[..];
        let mut res_slice = &mut result[..];
        while a_slice.len() >= S::Vf64::WIDTH {
            let va = S::Vf64::load_from_slice(a_slice);
            let vb = S::Vf64::load_from_slice(b_slice);
            let vr = va - vb;
            vr.copy_to_slice(res_slice);
            a_slice = &a_slice[S::Vf64::WIDTH..];
            b_slice = &b_slice[S::Vf64::WIDTH..];
            res_slice = &mut res_slice[S::Vf64::WIDTH..];
        }
        for i in 0..a_slice.len() {
            res_slice[i] = a_slice[i] - b_slice[i];
        }
        result
    }
}

simd_runtime_generate! {
    fn multiply_f32_simd(a: &[f32], b: &[f32]) -> Vec<f32> {
        let mut result = Vec::with_capacity(a.len());
        unsafe { result.set_len(a.len()); }
        let mut a_slice = &a[..];
        let mut b_slice = &b[..];
        let mut res_slice = &mut result[..];
        while a_slice.len() >= S::Vf32::WIDTH {
            let va = S::Vf32::load_from_slice(a_slice);
            let vb = S::Vf32::load_from_slice(b_slice);
            let vr = va * vb;
            vr.copy_to_slice(res_slice);
            a_slice = &a_slice[S::Vf32::WIDTH..];
            b_slice = &b_slice[S::Vf32::WIDTH..];
            res_slice = &mut res_slice[S::Vf32::WIDTH..];
        }
        for i in 0..a_slice.len() {
            res_slice[i] = a_slice[i] * b_slice[i];
        }
        result
    }
}

simd_runtime_generate! {
    fn multiply_f64_simd(a: &[f64], b: &[f64]) -> Vec<f64> {
        let mut result = Vec::with_capacity(a.len());
        unsafe { result.set_len(a.len()); }
        let mut a_slice = &a[..];
        let mut b_slice = &b[..];
        let mut res_slice = &mut result[..];
        while a_slice.len() >= S::Vf64::WIDTH {
            let va = S::Vf64::load_from_slice(a_slice);
            let vb = S::Vf64::load_from_slice(b_slice);
            let vr = va * vb;
            vr.copy_to_slice(res_slice);
            a_slice = &a_slice[S::Vf64::WIDTH..];
            b_slice = &b_slice[S::Vf64::WIDTH..];
            res_slice = &mut res_slice[S::Vf64::WIDTH..];
        }
        for i in 0..a_slice.len() {
            res_slice[i] = a_slice[i] * b_slice[i];
        }
        result
    }
}

simd_runtime_generate! {
    fn divide_f32_simd(a: &[f32], b: &[f32]) -> Vec<f32> {
        let mut result = Vec::with_capacity(a.len());
        unsafe { result.set_len(a.len()); }
        let mut a_slice = &a[..];
        let mut b_slice = &b[..];
        let mut res_slice = &mut result[..];
        while a_slice.len() >= S::Vf32::WIDTH {
            let va = S::Vf32::load_from_slice(a_slice);
            let vb = S::Vf32::load_from_slice(b_slice);
            let vr = va / vb;
            vr.copy_to_slice(res_slice);
            a_slice = &a_slice[S::Vf32::WIDTH..];
            b_slice = &b_slice[S::Vf32::WIDTH..];
            res_slice = &mut res_slice[S::Vf32::WIDTH..];
        }
        for i in 0..a_slice.len() {
            res_slice[i] = a_slice[i] / b_slice[i];
        }
        result
    }
}

simd_runtime_generate! {
    fn divide_f64_simd(a: &[f64], b: &[f64]) -> Vec<f64> {
        let mut result = Vec::with_capacity(a.len());
        unsafe { result.set_len(a.len()); }
        let mut a_slice = &a[..];
        let mut b_slice = &b[..];
        let mut res_slice = &mut result[..];
        while a_slice.len() >= S::Vf64::WIDTH {
            let va = S::Vf64::load_from_slice(a_slice);
            let vb = S::Vf64::load_from_slice(b_slice);
            let vr = va / vb;
            vr.copy_to_slice(res_slice);
            a_slice = &a_slice[S::Vf64::WIDTH..];
            b_slice = &b_slice[S::Vf64::WIDTH..];
            res_slice = &mut res_slice[S::Vf64::WIDTH..];
        }
        for i in 0..a_slice.len() {
            res_slice[i] = a_slice[i] / b_slice[i];
        }
        result
    }
}

simd_runtime_generate! {
    fn dot_product_f32_simd(a: &[f32], b: &[f32]) -> f32 {
        let mut sum = S::Vf32::set1(0.0);
        let mut a_slice = &a[..];
        let mut b_slice = &b[..];
        while a_slice.len() >= S::Vf32::WIDTH {
            let va = S::Vf32::load_from_slice(a_slice);
            let vb = S::Vf32::load_from_slice(b_slice);
            sum = sum + (va * vb);
            a_slice = &a_slice[S::Vf32::WIDTH..];
            b_slice = &b_slice[S::Vf32::WIDTH..];
        }
        let mut result = sum.horizontal_add();
        for i in 0..a_slice.len() {
            result += a_slice[i] * b_slice[i];
        }
        result
    }
}

simd_runtime_generate! {
    fn dot_product_f64_simd(a: &[f64], b: &[f64]) -> f64 {
        let mut sum = S::Vf64::set1(0.0);
        let mut a_slice = &a[..];
        let mut b_slice = &b[..];
        while a_slice.len() >= S::Vf64::WIDTH {
            let va = S::Vf64::load_from_slice(a_slice);
            let vb = S::Vf64::load_from_slice(b_slice);
            sum = sum + (va * vb);
            a_slice = &a_slice[S::Vf64::WIDTH..];
            b_slice = &b_slice[S::Vf64::WIDTH..];
        }
        let mut result = sum.horizontal_add();
        for i in 0..a_slice.len() {
            result += a_slice[i] * b_slice[i];
        }
        result
    }
}

simd_runtime_generate! {
    fn sum_f32_simd(a: &[f32]) -> f32 {
        let mut sum = S::Vf32::set1(0.0);
        let mut a_slice = &a[..];
        while a_slice.len() >= S::Vf32::WIDTH {
            let va = S::Vf32::load_from_slice(a_slice);
            sum = sum + va;
            a_slice = &a_slice[S::Vf32::WIDTH..];
        }
        let mut result = sum.horizontal_add();
        for i in 0..a_slice.len() {
            result += a_slice[i];
        }
        result
    }
}

simd_runtime_generate! {
    fn sum_f64_simd(a: &[f64]) -> f64 {
        let mut sum = S::Vf64::set1(0.0);
        let mut a_slice = &a[..];
        while a_slice.len() >= S::Vf64::WIDTH {
            let va = S::Vf64::load_from_slice(a_slice);
            sum = sum + va;
            a_slice = &a_slice[S::Vf64::WIDTH..];
        }
        let mut result = sum.horizontal_add();
        for i in 0..a_slice.len() {
            result += a_slice[i];
        }
        result
    }
}

simd_runtime_generate! {
    fn min_f32_simd(a: &[f32]) -> f32 {
        let mut min_val = S::Vf32::set1(f32::INFINITY);
        let mut a_slice = &a[..];
        while a_slice.len() >= S::Vf32::WIDTH {
            let va = S::Vf32::load_from_slice(a_slice);
            min_val = min_val.min(va);
            a_slice = &a_slice[S::Vf32::WIDTH..];
        }
        let mut result = min_val.horizontal_add();
        for &val in a_slice {
            if val < result {
                result = val;
            }
        }
        result
    }
}

simd_runtime_generate! {
    fn min_f64_simd(a: &[f64]) -> f64 {
        let mut min_val = S::Vf64::set1(f64::INFINITY);
        let mut a_slice = &a[..];
        while a_slice.len() >= S::Vf64::WIDTH {
            let va = S::Vf64::load_from_slice(a_slice);
            min_val = min_val.min(va);
            a_slice = &a_slice[S::Vf64::WIDTH..];
        }
        let mut result = min_val.horizontal_add();
        for &val in a_slice {
            if val < result {
                result = val;
            }
        }
        result
    }
}

simd_runtime_generate! {
    fn max_f32_simd(a: &[f32]) -> f32 {
        let mut max_val = S::Vf32::set1(f32::NEG_INFINITY);
        let mut a_slice = &a[..];
        while a_slice.len() >= S::Vf32::WIDTH {
            let va = S::Vf32::load_from_slice(a_slice);
            max_val = max_val.max(va);
            a_slice = &a_slice[S::Vf32::WIDTH..];
        }
        let mut result = max_val.horizontal_add();
        for &val in a_slice {
            if val > result {
                result = val;
            }
        }
        result
    }
}

simd_runtime_generate! {
    fn max_f64_simd(a: &[f64]) -> f64 {
        let mut max_val = S::Vf64::set1(f64::NEG_INFINITY);
        let mut a_slice = &a[..];
        while a_slice.len() >= S::Vf64::WIDTH {
            let va = S::Vf64::load_from_slice(a_slice);
            max_val = max_val.max(va);
            a_slice = &a_slice[S::Vf64::WIDTH..];
        }
        let mut result = max_val.horizontal_add();
        for &val in a_slice {
            if val > result {
                result = val;
            }
        }
        result
    }
}

simd_runtime_generate! {
    fn min_elementwise_f32_simd(a: &[f32], b: &[f32]) -> Vec<f32> {
        let mut result = Vec::with_capacity(a.len());
        unsafe { result.set_len(a.len()); }
        let mut a_slice = &a[..];
        let mut b_slice = &b[..];
        let mut res_slice = &mut result[..];
        while a_slice.len() >= S::Vf32::WIDTH {
            let va = S::Vf32::load_from_slice(a_slice);
            let vb = S::Vf32::load_from_slice(b_slice);
            let vr = va.min(vb);
            vr.copy_to_slice(res_slice);
            a_slice = &a_slice[S::Vf32::WIDTH..];
            b_slice = &b_slice[S::Vf32::WIDTH..];
            res_slice = &mut res_slice[S::Vf32::WIDTH..];
        }
        for i in 0..a_slice.len() {
            res_slice[i] = a_slice[i].min(b_slice[i]);
        }
        result
    }
}

simd_runtime_generate! {
    fn min_elementwise_f64_simd(a: &[f64], b: &[f64]) -> Vec<f64> {
        let mut result = Vec::with_capacity(a.len());
        unsafe { result.set_len(a.len()); }
        let mut a_slice = &a[..];
        let mut b_slice = &b[..];
        let mut res_slice = &mut result[..];
        while a_slice.len() >= S::Vf64::WIDTH {
            let va = S::Vf64::load_from_slice(a_slice);
            let vb = S::Vf64::load_from_slice(b_slice);
            let vr = va.min(vb);
            vr.copy_to_slice(res_slice);
            a_slice = &a_slice[S::Vf64::WIDTH..];
            b_slice = &b_slice[S::Vf64::WIDTH..];
            res_slice = &mut res_slice[S::Vf64::WIDTH..];
        }
        for i in 0..a_slice.len() {
            res_slice[i] = a_slice[i].min(b_slice[i]);
        }
        result
    }
}

simd_runtime_generate! {
    fn max_elementwise_f32_simd(a: &[f32], b: &[f32]) -> Vec<f32> {
        let mut result = Vec::with_capacity(a.len());
        unsafe { result.set_len(a.len()); }
        let mut a_slice = &a[..];
        let mut b_slice = &b[..];
        let mut res_slice = &mut result[..];
        while a_slice.len() >= S::Vf32::WIDTH {
            let va = S::Vf32::load_from_slice(a_slice);
            let vb = S::Vf32::load_from_slice(b_slice);
            let vr = va.max(vb);
            vr.copy_to_slice(res_slice);
            a_slice = &a_slice[S::Vf32::WIDTH..];
            b_slice = &b_slice[S::Vf32::WIDTH..];
            res_slice = &mut res_slice[S::Vf32::WIDTH..];
        }
        for i in 0..a_slice.len() {
            res_slice[i] = a_slice[i].max(b_slice[i]);
        }
        result
    }
}

simd_runtime_generate! {
    fn max_elementwise_f64_simd(a: &[f64], b: &[f64]) -> Vec<f64> {
        let mut result = Vec::with_capacity(a.len());
        unsafe { result.set_len(a.len()); }
        let mut a_slice = &a[..];
        let mut b_slice = &b[..];
        let mut res_slice = &mut result[..];
        while a_slice.len() >= S::Vf64::WIDTH {
            let va = S::Vf64::load_from_slice(a_slice);
            let vb = S::Vf64::load_from_slice(b_slice);
            let vr = va.max(vb);
            vr.copy_to_slice(res_slice);
            a_slice = &a_slice[S::Vf64::WIDTH..];
            b_slice = &b_slice[S::Vf64::WIDTH..];
            res_slice = &mut res_slice[S::Vf64::WIDTH..];
        }
        for i in 0..a_slice.len() {
            res_slice[i] = a_slice[i].max(b_slice[i]);
        }
        result
    }
}

simd_runtime_generate! {
    fn abs_f32_simd(a: &[f32]) -> Vec<f32> {
        let mut result = Vec::with_capacity(a.len());
        unsafe { result.set_len(a.len()); }
        let mut a_slice = &a[..];
        let mut res_slice = &mut result[..];
        while a_slice.len() >= S::Vf32::WIDTH {
            let va = S::Vf32::load_from_slice(a_slice);
            let vr = va.abs();
            vr.copy_to_slice(res_slice);
            a_slice = &a_slice[S::Vf32::WIDTH..];
            res_slice = &mut res_slice[S::Vf32::WIDTH..];
        }
        for i in 0..a_slice.len() {
            res_slice[i] = a_slice[i].abs();
        }
        result
    }
}

simd_runtime_generate! {
    fn abs_f64_simd(a: &[f64]) -> Vec<f64> {
        let mut result = Vec::with_capacity(a.len());
        unsafe { result.set_len(a.len()); }
        let mut a_slice = &a[..];
        let mut res_slice = &mut result[..];
        while a_slice.len() >= S::Vf64::WIDTH {
            let va = S::Vf64::load_from_slice(a_slice);
            let vr = va.abs();
            vr.copy_to_slice(res_slice);
            a_slice = &a_slice[S::Vf64::WIDTH..];
            res_slice = &mut res_slice[S::Vf64::WIDTH..];
        }
        for i in 0..a_slice.len() {
            res_slice[i] = a_slice[i].abs();
        }
        result
    }
}

simd_runtime_generate! {
    fn sqrt_f32_simd(a: &[f32]) -> Vec<f32> {
        let mut result = Vec::with_capacity(a.len());
        unsafe { result.set_len(a.len()); }
        let mut a_slice = &a[..];
        let mut res_slice = &mut result[..];
        while a_slice.len() >= S::Vf32::WIDTH {
            let va = S::Vf32::load_from_slice(a_slice);
            let vr = va.sqrt();
            vr.copy_to_slice(res_slice);
            a_slice = &a_slice[S::Vf32::WIDTH..];
            res_slice = &mut res_slice[S::Vf32::WIDTH..];
        }
        for i in 0..a_slice.len() {
            res_slice[i] = a_slice[i].sqrt();
        }
        result
    }
}

simd_runtime_generate! {
    fn sqrt_f64_simd(a: &[f64]) -> Vec<f64> {
        let mut result = Vec::with_capacity(a.len());
        unsafe { result.set_len(a.len()); }
        let mut a_slice = &a[..];
        let mut res_slice = &mut result[..];
        while a_slice.len() >= S::Vf64::WIDTH {
            let va = S::Vf64::load_from_slice(a_slice);
            let vr = va.sqrt();
            vr.copy_to_slice(res_slice);
            a_slice = &a_slice[S::Vf64::WIDTH..];
            res_slice = &mut res_slice[S::Vf64::WIDTH..];
        }
        for i in 0..a_slice.len() {
            res_slice[i] = a_slice[i].sqrt();
        }
        result
    }
}

simd_runtime_generate! {
    fn negate_f32_simd(a: &[f32]) -> Vec<f32> {
        let mut result = Vec::with_capacity(a.len());
        unsafe { result.set_len(a.len()); }
        let zero = S::Vf32::set1(0.0);
        let mut a_slice = &a[..];
        let mut res_slice = &mut result[..];
        while a_slice.len() >= S::Vf32::WIDTH {
            let va = S::Vf32::load_from_slice(a_slice);
            let vr = zero - va;
            vr.copy_to_slice(res_slice);
            a_slice = &a_slice[S::Vf32::WIDTH..];
            res_slice = &mut res_slice[S::Vf32::WIDTH..];
        }
        for i in 0..a_slice.len() {
            res_slice[i] = -a_slice[i];
        }
        result
    }
}

simd_runtime_generate! {
    fn negate_f64_simd(a: &[f64]) -> Vec<f64> {
        let mut result = Vec::with_capacity(a.len());
        unsafe { result.set_len(a.len()); }
        let zero = S::Vf64::set1(0.0);
        let mut a_slice = &a[..];
        let mut res_slice = &mut result[..];
        while a_slice.len() >= S::Vf64::WIDTH {
            let va = S::Vf64::load_from_slice(a_slice);
            let vr = zero - va;
            vr.copy_to_slice(res_slice);
            a_slice = &a_slice[S::Vf64::WIDTH..];
            res_slice = &mut res_slice[S::Vf64::WIDTH..];
        }
        for i in 0..a_slice.len() {
            res_slice[i] = -a_slice[i];
        }
        result
    }
}

simd_runtime_generate! {
    fn scale_f32_simd(a: &[f32], scalar: f32) -> Vec<f32> {
        let vs = S::Vf32::set1(scalar);
        let mut result = Vec::with_capacity(a.len());
        unsafe { result.set_len(a.len()); }
        let mut a_slice = &a[..];
        let mut res_slice = &mut result[..];
        while a_slice.len() >= S::Vf32::WIDTH {
            let va = S::Vf32::load_from_slice(a_slice);
            let vr = va * vs;
            vr.copy_to_slice(res_slice);
            a_slice = &a_slice[S::Vf32::WIDTH..];
            res_slice = &mut res_slice[S::Vf32::WIDTH..];
        }
        for i in 0..a_slice.len() {
            res_slice[i] = a_slice[i] * scalar;
        }
        result
    }
}

simd_runtime_generate! {
    fn scale_f64_simd(a: &[f64], scalar: f64) -> Vec<f64> {
        let vs = S::Vf64::set1(scalar);
        let mut result = Vec::with_capacity(a.len());
        unsafe { result.set_len(a.len()); }
        let mut a_slice = &a[..];
        let mut res_slice = &mut result[..];
        while a_slice.len() >= S::Vf64::WIDTH {
            let va = S::Vf64::load_from_slice(a_slice);
            let vr = va * vs;
            vr.copy_to_slice(res_slice);
            a_slice = &a_slice[S::Vf64::WIDTH..];
            res_slice = &mut res_slice[S::Vf64::WIDTH..];
        }
        for i in 0..a_slice.len() {
            res_slice[i] = a_slice[i] * scalar;
        }
        result
    }
}
