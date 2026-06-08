use std::error::Error;

use silent_params::{RegisteredParameterSet, find_preset};

fn main() -> Result<(), Box<dyn Error>> {
    println!(
        "section,name,scheme,security,params_id,ring_dim,log_n,ciphertext_modulus_bits,plaintext_modulus,details"
    );

    for name in [
        "current-bfv-v1",
        "current-hss-v1",
        "mlkem512-v1",
        "mlkem768-v1",
        "mlkem1024-v1",
        "mldsa44-v1",
        "mldsa65-v1",
        "mldsa87-v1",
    ] {
        let preset = find_preset(name).ok_or_else(|| format!("missing parameter preset {name}"))?;
        print_preset("audited_preset", preset)?;
    }

    print_operator_row(
        "runtime_smoke",
        "silent_lookup_small_prime",
        "PrivateLookupConfig",
        "Toy",
        "p=97|q_bits=50|matrix_rows=16|gadget_cols_t=6|max_domain=128|radix_bits=5",
    );
    print_operator_row(
        "runtime_smoke",
        "silent_hss_slot_smoke",
        "HssContext",
        "Toy",
        "degree=16|p=97|rns_chain=4x50bit|scope=correctness_and_microbenchmark_only",
    );
    print_operator_row(
        "runtime_full",
        "silent_lookup_full_prime",
        "PrivateLookupConfig",
        "Classical128-compatible metadata",
        "p=65537|q_bits=62|matrix_rows=16|gadget_cols_t=8|max_domain=65537|radix_bits=15",
    );
    print_operator_row(
        "runtime_full",
        "silent_hss_current_hss_v1",
        "HssContext",
        "Classical128",
        "preset=current-hss-v1|degree=4096|p=65537|rns_bits=54+55|share_bits=72",
    );
    print_operator_row(
        "protocol_model",
        "bert_base_refresh_barriers",
        "LatencyModel",
        "N/A",
        "layers=12|barriers_per_layer=2|R=24|LAN_RTT_ms=0.5|WAN_RTT_ms=4",
    );

    Ok(())
}

fn print_preset(section: &str, preset: RegisteredParameterSet) -> Result<(), Box<dyn Error>> {
    preset.validate()?;
    let params_id = preset.params_id().to_hex();
    match preset {
        RegisteredParameterSet::Bfv(params) => {
            println!(
                "{section},{},{},{:?},{},{},{},{},{},depth={}",
                params.name,
                preset_scheme("Bfv"),
                params.rlwe.security_level,
                params_id,
                params.rlwe.ring.ring_dim.0,
                params.rlwe.ring.log_n.0,
                format_bits(&params.rlwe.ciphertext_modulus_bits),
                params.plaintext_modulus.0,
                params.multiplicative_depth.0
            );
        }
        RegisteredParameterSet::Hss(params) => {
            println!(
                "{section},{},{},{:?},{},{},{},{},{},share_bits={}|max_linear_terms={}|max_rmult_depth={}|scale_bits={}|reconstruction_bits={}|correctness_margin_bits={}|required_budget_bits={}",
                params.name,
                preset_scheme("Hss"),
                params.rlwe.security_level,
                params_id,
                params.rlwe.ring.ring_dim.0,
                params.rlwe.ring.log_n.0,
                format_bits(&params.rlwe.ciphertext_modulus_bits),
                params.plaintext_modulus.0,
                params.share_modulus_bits.0,
                params.max_linear_terms.0,
                params.max_rmult_depth.0,
                params
                    .fixed_point_scale_bits
                    .map(|bits| bits.0.to_string())
                    .unwrap_or_else(|| "none".to_string()),
                params
                    .reconstruction_bound_bits
                    .map(|bits| bits.0.to_string())
                    .unwrap_or_else(|| "none".to_string()),
                params
                    .correctness_margin_bits
                    .map(|bits| bits.0.to_string())
                    .unwrap_or_else(|| "none".to_string()),
                params.estimated_required_budget_bits()
            );
        }
        RegisteredParameterSet::PqcKem(params) => {
            println!(
                "{section},{},{},{:?},{},,,,,algorithm={}|pk_bytes={}|sk_bytes={}|ct_bytes={}|ss_bytes={}",
                params.name,
                preset_scheme("PqcKem"),
                params.security_level,
                params_id,
                params.algorithm,
                params.public_key_bytes,
                params.secret_key_bytes,
                params.ciphertext_bytes,
                params.shared_secret_bytes
            );
        }
        RegisteredParameterSet::PqcSig(params) => {
            println!(
                "{section},{},{},{:?},{},,,,,algorithm={}|pk_bytes={}|sk_bytes={}|sig_bytes={}",
                params.name,
                preset_scheme("PqcSig"),
                params.security_level,
                params_id,
                params.algorithm,
                params.public_key_bytes,
                params.secret_key_bytes,
                params.signature_bytes
            );
        }
        other => {
            println!(
                "{section},{},{:?},{:?},{},,,,,validated=true",
                other.name(),
                other.scheme_family(),
                other.security_level(),
                params_id
            );
        }
    }
    Ok(())
}

fn print_operator_row(section: &str, name: &str, scheme: &str, security: &str, details: &str) {
    println!("{section},{name},{scheme},{security},,,,,,{details}");
}

fn format_bits(bits: &[silent_params::ModulusBits]) -> String {
    bits.iter()
        .map(|bits| bits.0.to_string())
        .collect::<Vec<_>>()
        .join("+")
}

fn preset_scheme(name: &'static str) -> &'static str {
    name
}
