//! Ring context binding degree and RNS base.

use silent_math::ntt::NttTables;
use silent_math::rns::RnsBase;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug)]
pub struct RingContext {
    degree: usize,
    rns: RnsBase,
    ntt_tables: Arc<Vec<NttTables>>,
    automorphism_maps: Arc<Mutex<HashMap<u64, Arc<Vec<usize>>>>>,
    coeff_galois_maps: Arc<Mutex<HashMap<u64, Arc<Vec<u32>>>>>,
}

impl RingContext {
    pub fn new(degree: usize, rns: RnsBase) -> Self {
        let mut tables = Vec::with_capacity(rns.len());
        for modulus in rns.moduli() {
            // We typically use Negacyclic NTT for RLWE (x^N + 1)
            let table = NttTables::new_negacyclic(degree, *modulus)
                .expect("Failed to create NTT tables for modulus");
            tables.push(table);
        }

        Self {
            degree,
            rns,
            ntt_tables: Arc::new(tables),
            automorphism_maps: Arc::new(Mutex::new(HashMap::new())),
            coeff_galois_maps: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn degree(&self) -> usize {
        self.degree
    }

    pub fn rns(&self) -> &RnsBase {
        &self.rns
    }

    pub fn ntt_tables(&self) -> &[NttTables] {
        &self.ntt_tables
    }

    pub fn automorphism_map(&self, k: u64) -> Arc<Vec<usize>> {
        assert!(k % 2 == 1, "Galois element must be odd");

        if let Some(map) = self.automorphism_maps.lock().unwrap().get(&k) {
            return Arc::clone(map);
        }

        let computed = Arc::new(self.gen_automorphism_map(k));
        let mut cache = self.automorphism_maps.lock().unwrap();
        let entry = cache.entry(k).or_insert_with(|| Arc::clone(&computed));
        Arc::clone(entry)
    }

    pub fn gen_automorphism_map(&self, k: u64) -> Vec<usize> {
        assert!(k % 2 == 1, "Galois element must be odd");
        let n = self.degree;
        let log_n = n.trailing_zeros();
        let mut map = vec![0usize; n];

        // NTT form is stored in bit-reversed order (SEAL-style).
        fn reverse_bits(value: u64, bits: u32) -> u64 {
            if bits == 0 {
                return 0;
            }
            value.reverse_bits() >> (64 - bits)
        }

        let coeff_count_minus_one = n as u64 - 1;
        for i in n..(n << 1) {
            let reversed = reverse_bits(i as u64, log_n + 1);
            let mut index_raw = (k * reversed) >> 1;
            index_raw &= coeff_count_minus_one;
            let index = reverse_bits(index_raw, log_n) as usize;
            map[i - n] = index;
        }
        map
    }

    pub fn coeff_galois_map(&self, k: u64) -> Arc<Vec<u32>> {
        assert!(k % 2 == 1, "Galois element must be odd");

        if let Some(map) = self.coeff_galois_maps.lock().unwrap().get(&k) {
            return Arc::clone(map);
        }

        let computed = Arc::new(self.gen_coeff_galois_map(k));
        let mut cache = self.coeff_galois_maps.lock().unwrap();
        let entry = cache.entry(k).or_insert_with(|| Arc::clone(&computed));
        Arc::clone(entry)
    }

    pub fn gen_coeff_galois_map(&self, k: u64) -> Vec<u32> {
        assert!(k % 2 == 1, "Galois element must be odd");
        let n = self.degree as u64;
        let m = n * 2;
        let mask = m - 1;
        let mut map = vec![0u32; n as usize];
        for i in 0..n {
            let index = (i * k) & mask;
            map[i as usize] = index as u32;
        }
        map
    }
}
