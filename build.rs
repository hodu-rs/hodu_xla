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
    println!("cargo:rerun-if-changed=c/xla_wrapper.cc");
    println!("cargo:rerun-if-changed=c/xla_wrapper.h");
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
                .file("c/xla_wrapper.cc")
                .compile("xla_wrapper");
        },
        OS::Windows => {
            cc::Build::new()
                .cpp(true)
                .pic(true)
                .warnings(false)
                .include(xla_dir.as_ref().join("include"))
                .file("c/xla_wrapper.cc")
                .compile("xla_wrapper");
        },
    };
}

#[path = "scripts/build_helper.rs"]
mod build_helper;

use build_helper::ensure_xla_installation;

fn main() {
    let os = OS::get();

    // Check if XLA_EXTENSION_DIR is set, otherwise install XLA extension
    let xla_dir = if let Ok(dir) = env::var("XLA_EXTENSION_DIR") {
        PathBuf::from(dir)
    } else {
        ensure_xla_installation().expect("Failed to install XLA extension")
    };

    // Using XLA extension silently

    println!("cargo:rerun-if-changed=c/xla_wrapper.h");
    println!("cargo:rerun-if-changed=c/xla_wrapper.cc");
    let bindings = bindgen::Builder::default()
        .header("c/xla_wrapper.h")
        .clang_arg(format!("-I{}", xla_dir.join("include").display()))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("c_xla.rs"))
        .expect("Couldn't write bindings!");

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
    println!("cargo:rustc-link-lib=static=xla_wrapper");

    let lib_path = xla_dir.join("lib");

    let out_dir = env::var("OUT_DIR").unwrap();
    let target_profile_dir = PathBuf::from(&out_dir).ancestors().nth(3).unwrap().to_path_buf();

    let lib_extension = match os {
        OS::Linux => "so",
        OS::MacOS => "so",
        OS::Windows => "dll",
    };

    let src_lib = lib_path.join(format!("libxla_extension.{}", lib_extension));
    let dst_lib = target_profile_dir.join(format!("libxla_extension.{}", lib_extension));

    if src_lib.exists() {
        let _ = std::fs::copy(&src_lib, &dst_lib);
    }

    if os == OS::MacOS {
        println!("cargo:rustc-link-arg=-Wl,-headerpad_max_install_names");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_path.display());
        println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path");
    } else if os == OS::Linux {
        println!("cargo:rustc-link-arg=-Wl,-rpath={}", lib_path.display());
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
    }
    println!("cargo:rustc-link-lib=xla_extension");
}
