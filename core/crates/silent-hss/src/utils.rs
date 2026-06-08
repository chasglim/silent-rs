use rand::RngCore;
use silent_math::modulus::Modulus;
use silent_ring::Poly;
use silent_utils::rng::SecureRng;

/// Samples a polynomial uniformly in R_q in coefficient domain.
pub fn sample_uniform_poly(
    rng: &mut SecureRng,
    degree: usize,
    num_moduli: usize,
    moduli: &[Modulus],
) -> Poly {
    let mut poly = unsafe { Poly::new_uninit(degree, num_moduli) };
    let mut seed = [0u8; 64];
    rng.fill_bytes(&mut seed);
    let mut local_rng = silent_utils::rng::Blake2xbRng::from_seed_bytes(seed);

    for i in 0..num_moduli {
        let limb = poly.limb_mut(i);
        let modulus = moduli[i].value();
        let max_multiple = u64::MAX - (u64::MAX % modulus);
        for coeff in limb.iter_mut() {
            loop {
                let v = local_rng.next_u64();
                if v < max_multiple {
                    *coeff = v % modulus;
                    break;
                }
            }
        }
    }

    poly
}

/// Samples a polynomial from a ternary distribution {-1, 0, 1} in coefficient domain.
pub fn sample_ternary_poly(
    rng: &mut SecureRng,
    degree: usize,
    num_moduli: usize,
    moduli: &[Modulus],
) -> Poly {
    let mut poly = unsafe { Poly::new_uninit(degree, num_moduli) };
    let mut values = vec![0i64; degree];

    // Fill values with -1, 0, 1 using rejection sampling
    let mut seed = [0u8; 64];
    rng.fill_bytes(&mut seed);
    // use rand::SeedableRng; // Not needed for from_seed_bytes inherent method
    let mut local_rng = silent_utils::rng::Blake2xbRng::from_seed_bytes(seed);

    let max_multiple = u32::MAX - (u32::MAX % 3);
    for val in values.iter_mut() {
        loop {
            let v = local_rng.next_u32();
            if v < max_multiple {
                *val = (v % 3) as i64 - 1;
                break;
            }
        }
    }

    // Assign to limbs
    for i in 0..num_moduli {
        let limb = poly.limb_mut(i);
        let m = moduli[i].value();
        for j in 0..degree {
            let v = values[j];
            if v < 0 {
                limb[j] = m.wrapping_sub(v.abs() as u64);
            } else {
                limb[j] = v as u64;
            }
        }
    }
    poly
}

/// Samples a polynomial from a Centered Binomial Distribution (CBD) in coefficient domain.
/// Following SEAL's default CBD logic (sigma=3.2 approx).
pub fn sample_cbd_poly(
    rng: &mut SecureRng,
    degree: usize,
    num_moduli: usize,
    moduli: &[Modulus],
) -> Poly {
    let mut poly = unsafe { Poly::new_uninit(degree, num_moduli) };

    // Generate 6 bytes per coefficient (SEAL standard for sigma=3.2)
    let total_bytes = degree * 6;
    let mut bytes = vec![0u8; total_bytes];
    let mut seed = [0u8; 64];
    rng.fill_bytes(&mut seed);
    // use rand::SeedableRng;
    let mut local_rng = silent_utils::rng::Blake2xbRng::from_seed_bytes(seed);
    local_rng.fill_bytes(&mut bytes);

    for i in 0..degree {
        let base = i * 6;
        let x0 = bytes[base];
        let x1 = bytes[base + 1];
        let mut x2 = bytes[base + 2];
        let x3 = bytes[base + 3];
        let x4 = bytes[base + 4];
        let mut x5 = bytes[base + 5];
        x2 &= 0x1F;
        x5 &= 0x1F;

        let pos = x0.count_ones() + x1.count_ones() + x2.count_ones();
        let neg = x3.count_ones() + x4.count_ones() + x5.count_ones();
        let noise = pos as i32 - neg as i32;

        let flag = if noise < 0 { u64::MAX } else { 0 };
        let noise_u = noise as u64;

        for k in 0..num_moduli {
            let m = moduli[k].value();
            poly.limb_mut(k)[i] = noise_u.wrapping_add(flag & m);
        }
    }
    poly
}
