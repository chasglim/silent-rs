use crate::bridge::keys::b2t_ksk::BfvToTfheKsk;
use crate::schemes::tfhe::keyswitch::LweKeyswitcher;
use silent_rlwe::LweCiphertext;

/// Cross key switching using `KSK_B2T`.
/// `LWE'_{s_BFV, q_T} -> TLWE_{s_TFHE, q_T}`
pub fn keyswitch_to_tfhe(lwe: &LweCiphertext, ksk: &BfvToTfheKsk) -> LweCiphertext {
    let switcher = LweKeyswitcher::new(&ksk.ksk);
    switcher.keyswitch(lwe)
}
