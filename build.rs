extern crate bindgen;

use std::env;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Eq, PartialEq)]
enum OS {
    Linux,
    #[allow(clippy::enum_variant_names)]
    MacOS,
    Windows,
}

impl OS {
    fn get() -> Self {
        let os = env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS");
        match os.as_str() {
            "linux" => Self::Linux,
            "macos" => Self::MacOS,
            "windows" => Self::Windows,
            os => panic!("Unsupported system {os}"),
        }
    }
}

fn make_shared_lib<P: AsRef<Path>>(os: OS, xla_dir: P) {
    println!("cargo:rerun-if-changed=xla_rs/xla_rs.cc");
    println!("cargo:rerun-if-changed=xla_rs/xla_rs.h");
    match os {
        OS::Linux | OS::MacOS => {
            cc::Build::new()
                .cpp(true)
                .pic(true)
                .warnings(false)
                .include(xla_dir.as_ref().join("include"))
                .flag("-std=c++17")
                .flag("-Wno-deprecated-declarations")
                .flag("-Wno-missing-template-arg-list-after-template-kw")
                .flag("-Wno-macro-redefined")
                .flag("-Wno-defaulted-function-deleted")
                .flag("-w")
                .flag("-DLLVM_ON_UNIX=1")
                .flag("-DLLVM_VERSION_STRING=")
                .file("xla_rs/xla_rs.cc")
                .compile("xla_rs");
        }
        OS::Windows => {
            cc::Build::new()
                .cpp(true)
                .pic(true)
                .warnings(false)
                .include(xla_dir.as_ref().join("include"))
                .file("xla_rs/xla_rs.cc")
                .compile("xla_rs");
        }
    };
}

#[path = "scripts/build_helper.rs"]
mod build_helper;

use build_helper::ensure_xla_installation;

fn main() {
    let os = OS::get();

    // Automatically install XLA extension using build_helper
    let xla_dir = ensure_xla_installation().expect("Failed to install XLA extension");

    // Using XLA extension silently

    println!("cargo:rerun-if-changed=xla_rs/xla_rs.h");
    println!("cargo:rerun-if-changed=xla_rs/xla_rs.cc");
    let bindings = bindgen::Builder::default()
        .header("xla_rs/xla_rs.h")
        .clang_arg(format!("-I{}", xla_dir.join("include").display()))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings.write_to_file(out_path.join("c_xla.rs")).expect("Couldn't write bindings!");

    // Exit early on docs.rs as the C++ library would not be available.
    if std::env::var("DOCS_RS").is_ok() {
        return;
    }
    make_shared_lib(os, &xla_dir);
    // The --copy-dt-needed-entries -lstdc++ are helpful to get around some
    // "DSO missing from command line" error
    // undefined reference to symbol '_ZStlsIcSt11char_traitsIcESaIcEERSt13basic_ostreamIT_T0_ES7_RKNSt7__cxx1112basic_stringIS4_S5_T1_EE@@GLIBCXX_3.4.21'
    if os == OS::Linux {
        // println!("cargo:rustc-link-arg=-Wl,--copy-dt-needed-entries");
        println!("cargo:rustc-link-arg=-Wl,-lstdc++");
    }
    println!("cargo:rustc-link-search=native={}", xla_dir.join("lib").display());
    println!("cargo:rustc-link-lib=static=xla_rs");
    if os == OS::MacOS {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", xla_dir.join("lib").display());
    } else {
        println!("cargo:rustc-link-arg=-Wl,-rpath={}", xla_dir.join("lib").display());
    }
    println!("cargo:rustc-link-lib=xla_extension");
}
