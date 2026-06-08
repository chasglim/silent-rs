fn main() {
    #[cfg(feature = "proto")]
    {
        // SAFETY: Cargo build scripts run this setup before invoking prost-build
        // and do not spawn concurrent threads that read the process environment.
        unsafe {
            std::env::set_var(
                "PROTOC",
                protoc_bin_vendored::protoc_bin_path()
                    .expect("protoc binary not found in vendored package"),
            );
        }
        prost_build::compile_protos(&["proto/message.proto"], &["proto/"])
            .expect("protobuf code generation failed");
    }
}
