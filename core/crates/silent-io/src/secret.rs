//! A byte buffer that zeroizes its contents on drop.
//!
//! Used to wrap serialized secret-key material so that plaintext key bytes
//! are erased from memory after use.  Implements [`std::io::Write`] so it can
//! serve as the target for [`crate::CanonicalEncode::encode_to`].

use std::fmt;
use std::io;
use zeroize::Zeroize;

/// A byte buffer whose contents are securely erased when dropped.
///
/// # Safety properties
///
/// - [`Drop`] calls [`Zeroize::zeroize`] on the internal buffer.
/// - [`Debug`](fmt::Debug) and [`Display`](fmt::Display) output a redacted
///   marker; the actual bytes are never formatted.
/// - `Clone` is deliberately not implemented.
/// - [`as_ref()`](SecretBytes::as_ref) provides read-only access for encoding
///   to an external writer.
pub struct SecretBytes(Vec<u8>);

impl SecretBytes {
    /// An empty `SecretBytes`.
    pub const fn new() -> Self {
        Self(Vec::new())
    }
}

impl Default for SecretBytes {
    fn default() -> Self {
        Self::new()
    }
}

impl AsRef<[u8]> for SecretBytes {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl SecretBytes {
    /// Create from a `Vec<u8>`.
    pub fn from_vec(v: Vec<u8>) -> Self {
        Self(v)
    }

    /// Number of bytes stored.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns `true` when the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl fmt::Debug for SecretBytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("SecretBytes")
            .field(&format_args!("<{} bytes>", self.0.len()))
            .finish()
    }
}

impl fmt::Display for SecretBytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SecretBytes(<redacted>)")
    }
}

impl Drop for SecretBytes {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}

impl io::Write for SecretBytes {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.0.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn debug_does_not_leak() {
        let sb = SecretBytes::from_vec(vec![0xde, 0xad, 0xbe, 0xef]);
        let dbg = format!("{sb:?}");
        let disp = format!("{sb}");
        assert!(!dbg.contains("dead"));
        assert!(!disp.contains("dead"));
        assert!(dbg.contains("4 bytes"));
    }

    #[test]
    fn write_trait_works() {
        let mut sb = SecretBytes::new();
        write!(sb, "hello").unwrap();
        assert_eq!(sb.as_ref(), b"hello");
    }

    #[test]
    fn write_trait_multiple() {
        let mut sb = SecretBytes::from_vec(Vec::new());
        sb.write_all(b"abc").unwrap();
        sb.write_all(b"def").unwrap();
        assert_eq!(sb.as_ref(), b"abcdef");
    }
}
