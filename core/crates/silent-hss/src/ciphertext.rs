use silent_rlwe::Ciphertext;

#[derive(Clone, Debug)]
pub struct HssCiphertext {
    pub enc_m: Ciphertext,
    pub enc_m_times_s: Ciphertext,
}
