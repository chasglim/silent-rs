#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DistributionType {
    Ternary = 0,
    GaussianError = 1,
    Uniform = 2,
}
