//! Cosine similarity and L2 normalization.

/// In-place L2 normalize. Zero vectors are left unchanged.
pub fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Cosine similarity. Returns 0.0 for empty or length-mismatched inputs.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_unit_vectors_score_one() {
        let a = [0.0f32, 1.0, 0.0];
        assert!((cosine(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn orthogonal_score_zero() {
        let a = [1.0f32, 0.0];
        let b = [0.0f32, 1.0];
        assert!(cosine(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn normalize_then_cosine() {
        let mut a = vec![3.0f32, 0.0, 4.0];
        let mut b = vec![6.0f32, 0.0, 8.0];
        l2_normalize(&mut a);
        l2_normalize(&mut b);
        assert!((cosine(&a, &b) - 1.0).abs() < 1e-5);
    }
}
