//! Pure-Rust vector math utilities for embedding operations.
//!
//! These are used by the vector search module for brute-force similarity.
//! When the `local-embeddings` feature is enabled, candle-based pooling
//! operates on Tensors directly (see CandleNativeBackend).

/// Compute dot product of two vectors.
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Compute cosine similarity between two L2-normalized vectors.
///
/// For normalized vectors, cosine similarity equals dot product.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    dot_product(a, b)
}

/// L2-normalize a vector in place.
pub fn normalize_l2_inplace(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// L2-normalize a vector, returning a new vector.
pub fn normalize_l2(v: &[f32]) -> Vec<f32> {
    let mut result = v.to_vec();
    normalize_l2_inplace(&mut result);
    result
}

/// Serialize a float vector to little-endian bytes for SQLite BLOB storage.
pub fn vec_to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Deserialize a float vector from little-endian bytes.
pub fn bytes_to_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_product_basic() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let result = dot_product(&a, &b);
        assert!((result - 32.0).abs() < f32::EPSILON); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn dot_product_zero() {
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        assert!((dot_product(&a, &b)).abs() < f32::EPSILON); // orthogonal
    }

    #[test]
    fn normalize_l2_unit_vector() {
        let v = normalize_l2(&[3.0, 4.0]);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);
        assert!((v[0] - 0.6).abs() < 0.001);
        assert!((v[1] - 0.8).abs() < 0.001);
    }

    #[test]
    fn normalize_zero_vector() {
        let v = normalize_l2(&[0.0, 0.0, 0.0]);
        assert!(v.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn vec_bytes_roundtrip() {
        let original = vec![1.0_f32, -2.5, 3.14159, 0.0];
        let bytes = vec_to_bytes(&original);
        let recovered = bytes_to_vec(&bytes);
        assert_eq!(original.len(), recovered.len());
        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn cosine_similarity_normalized() {
        let a = normalize_l2(&[1.0, 0.0]);
        let b = normalize_l2(&[1.0, 0.0]);
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 0.001); // identical vectors

        let c = normalize_l2(&[0.0, 1.0]);
        let sim_orth = cosine_similarity(&a, &c);
        assert!(sim_orth.abs() < 0.001); // orthogonal
    }
}
