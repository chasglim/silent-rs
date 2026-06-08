use std::env;
use std::path::{Path, PathBuf};

fn main() {
    if env::var_os("CARGO_FEATURE_HEXL").is_none() {
        return;
    }

    println!("cargo:rerun-if-changed=hexl/hexl_bridge.cc");
    println!("cargo:rerun-if-env-changed=HEXL_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=HEXL_LIB_DIR");
    println!("cargo:rerun-if-env-changed=HEXL_LIB_NAME");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");

    let mut include_dirs = Vec::new();
    if let Ok(include_dir) = env::var("HEXL_INCLUDE_DIR") {
        include_dirs.push(PathBuf::from(include_dir));
    }

    let mut link_dirs = Vec::new();
    if let Ok(lib_dir) = env::var("HEXL_LIB_DIR") {
        link_dirs.push(PathBuf::from(lib_dir));
    }

    let env_lib_name = env::var("HEXL_LIB_NAME").ok();
    let mut link_libs = Vec::new();
    if let Some(lib) = env_lib_name.clone() {
        link_libs.push(lib);
    }

    if include_dirs.is_empty() {
        if let Ok(pkg) = pkg_config::Config::new().probe("hexl") {
            include_dirs.extend(pkg.include_paths);
            link_dirs.extend(pkg.link_paths);
            if env_lib_name.is_none() {
                link_libs.extend(pkg.libs);
            }
        } else if let Ok(pkg) = pkg_config::Config::new().probe("intel-hexl") {
            include_dirs.extend(pkg.include_paths);
            link_dirs.extend(pkg.link_paths);
            if env_lib_name.is_none() {
                link_libs.extend(pkg.libs);
            }
        }
    }

    let mut resolved_includes = Vec::new();
    for dir in include_dirs {
        if has_hexl_header(&dir) {
            resolved_includes.push(dir);
        } else if dir.join("hexl.hpp").exists()
            && let Some(parent) = dir.parent()
        {
            resolved_includes.push(parent.to_path_buf());
        }
    }

    if resolved_includes.is_empty() {
        panic!(
            "HEXL headers not found. Set HEXL_INCLUDE_DIR to the directory containing hexl/hexl.hpp or install HEXL with pkg-config."
        );
    }

    if link_libs.is_empty() {
        link_libs.push("hexl".to_string());
    }

    let mut build = cc::Build::new();
    build.cpp(true).file("hexl/hexl_bridge.cc");
    for include_dir in &resolved_includes {
        build.include(include_dir);
    }
    build.flag_if_supported("-std=c++17").compile("hexl_bridge");

    for dir in link_dirs {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
    for lib in link_libs {
        println!("cargo:rustc-link-lib={}", lib);
    }
}

fn has_hexl_header(dir: &Path) -> bool {
    dir.join("hexl/hexl.hpp").exists()
}
