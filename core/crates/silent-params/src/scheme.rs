#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SchemeFamily {
    Rlwe = 0,
    Bfv = 1,
    Ckks = 2,
    Hss = 3,
    PqcKem = 4,
    PqcSig = 5,
    Tfhe = 6,
    Bgv = 7,
}
