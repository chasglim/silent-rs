//! Serialization primitives using rkyv (Zero-Copy).

use rkyv::ser::serializers::AllocSerializer;
pub use rkyv::{Archive, Deserialize, Serialize};

/// Serializes the given value to a `Vec<u8>`.
pub fn to_bytes<T>(value: &T) -> Result<Vec<u8>, String>
where
    T: Serialize<AllocSerializer<256>>,
{
    rkyv::to_bytes::<T, 256>(value)
        .map(|aligned| aligned.into_vec())
        .map_err(|e| e.to_string())
}

/// deserializes a `Vec<u8>` into a checked archived reference.
pub fn from_bytes<'a, T>(bytes: &'a [u8]) -> Result<&'a T::Archived, String>
where
    T: Archive,
    T::Archived: rkyv::CheckBytes<rkyv::validation::validators::DefaultValidator<'a>>,
{
    rkyv::check_archived_root::<T>(bytes).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Archive, Deserialize, Serialize, Debug, PartialEq)]
    #[archive(check_bytes)] // Required for check_archived_root
    struct TestStruct {
        a: u32,
        b: Vec<u64>,
    }

    #[test]
    fn rkyv_round_trip() {
        let original = TestStruct {
            a: 42,
            b: vec![1, 2, 3],
        };

        let bytes = to_bytes(&original).expect("serialize");
        let archived = from_bytes::<TestStruct>(&bytes).expect("deserialize check");

        // Zero-copy access
        assert_eq!(archived.a, 42);
        // rkyv 0.7: ArchivedVec derefs to [T].
        assert_eq!(&archived.b[..], &[1, 2, 3]);

        // Full deserialize
        let deserialized: TestStruct = archived.deserialize(&mut rkyv::Infallible).unwrap();
        assert_eq!(deserialized, original);
    }
}
