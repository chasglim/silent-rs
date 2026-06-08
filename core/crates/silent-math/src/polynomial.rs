//! Polynomial storage helpers.

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Polynomial {
    coeffs: Vec<u64>,
}

impl Polynomial {
    pub fn new(coeffs: Vec<u64>) -> Self {
        Self { coeffs }
    }

    pub fn zeros(degree: usize) -> Self {
        Self {
            coeffs: vec![0u64; degree],
        }
    }

    pub fn len(&self) -> usize {
        self.coeffs.len()
    }

    pub fn coeffs(&self) -> &[u64] {
        &self.coeffs
    }

    pub fn coeffs_mut(&mut self) -> &mut [u64] {
        &mut self.coeffs
    }

    pub fn view(&self) -> PolyView<'_> {
        PolyView {
            coeffs: &self.coeffs,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PolyView<'a> {
    coeffs: &'a [u64],
}

impl<'a> PolyView<'a> {
    pub fn coeffs(&self) -> &'a [u64] {
        self.coeffs
    }

    pub fn len(&self) -> usize {
        self.coeffs.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polynomial_view_roundtrip() {
        let poly = Polynomial::new(vec![1, 2, 3]);
        let view = poly.view();
        assert_eq!(view.len(), 3);
        assert_eq!(view.coeffs(), &[1, 2, 3]);
    }
}
