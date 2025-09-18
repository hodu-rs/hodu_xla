// Build helper for automatic XLA extension installation
// This module provides functions to automatically download and install XLA extension
// when building xla-rs projects.

#![allow(clippy::enum_variant_names)]
#![allow(dead_code)]

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OS {
    Linux,
    MacOS,
    Windows,
}

impl OS {
    pub fn detect() -> Result<Self, String> {
        match env::consts::OS {
            "linux" => Ok(OS::Linux),
            "macos" => Ok(OS::MacOS),
            "windows" => Ok(OS::Windows),
            os => Err(format!("Unsupported OS: {}", os)),
        }
    }

    pub fn platform_name(&self) -> &'static str {
        match self {
            OS::Linux => "linux",
            OS::MacOS => "darwin",
            OS::Windows => "windows",
        }
    }

    pub fn file_extension(&self) -> &'static str {
        match self {
            OS::Linux | OS::MacOS => "tar.gz",
            OS::Windows => "zip",
        }
    }

    pub fn script_name(&self) -> &'static str {
        match self {
            OS::Linux | OS::MacOS => "install.sh",
            OS::Windows => "install.ps1",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Arch {
    X86_64,
    Aarch64,
}

impl Arch {
    pub fn detect() -> Result<Self, String> {
        match env::consts::ARCH {
            "x86_64" => Ok(Arch::X86_64),
            "aarch64" => Ok(Arch::Aarch64),
            arch => Err(format!("Unsupported architecture: {}", arch)),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Arch::X86_64 => "x86_64",
            Arch::Aarch64 => "aarch64",
        }
    }
}

/// Configuration for XLA extension installation
pub struct XlaInstaller {
    version: String,
    install_dir: PathBuf,
    os: OS,
    arch: Arch,
    force_reinstall: bool,
}

impl XlaInstaller {
    pub fn new() -> Result<Self, String> {
        let version = env::var("ELIXIR_NX_XLA_VERSION").unwrap_or_else(|_| "0.8.0".to_string());
        let install_dir = Self::get_install_dir();
        let os = OS::detect()?;
        let arch = Arch::detect()?;

        Ok(Self { version, install_dir, os, arch, force_reinstall: false })
    }

    pub fn with_version(version: String) -> Result<Self, String> {
        let mut installer = Self::new()?;
        installer.version = version;
        Ok(installer)
    }

    pub fn with_install_dir(mut self, dir: PathBuf) -> Self {
        self.install_dir = dir;
        self
    }

    pub fn with_force_reinstall(mut self, force: bool) -> Self {
        self.force_reinstall = force;
        self
    }

    fn get_install_dir() -> PathBuf {
        if let Ok(dir) = env::var("XLA_EXTENSION_DIR") {
            PathBuf::from(dir)
        } else {
            // Use $HOME/.hodu/xla-rs/ as the default installation directory
            let home = env::var("HOME").unwrap_or_else(|_| ".".to_string());
            PathBuf::from(home)
                .join(".hodu")
                .join("xla-rs")
                .join("xla_extension")
        }
    }

    pub fn install_dir(&self) -> &Path {
        &self.install_dir
    }

    /// Check if XLA extension is properly installed
    pub fn is_installed(&self) -> bool {
        let lib_dir = self.install_dir.join("lib");
        let include_dir = self.install_dir.join("include");
        let pjrt_header = include_dir.join("xla").join("pjrt").join("c").join("pjrt_c_api.h");

        lib_dir.exists() && include_dir.exists() && pjrt_header.exists() && self.has_library_files()
    }

    fn has_library_files(&self) -> bool {
        let lib_dir = self.install_dir.join("lib");
        if !lib_dir.exists() {
            return false;
        }

        // Check for XLA extension library
        let extensions = match self.os {
            OS::Linux => vec!["so"],
            OS::MacOS => vec!["so", "dylib"],
            OS::Windows => vec!["dll", "lib"],
        };

        for ext in extensions {
            let lib_file = lib_dir.join(format!("libxla_extension.{}", ext));
            if lib_file.exists() {
                return true;
            }
        }

        false
    }

    /// Install XLA extension if needed
    pub fn install_if_needed(&self) -> Result<(), String> {
        if self.is_installed() && !self.force_reinstall {
            return Ok(());
        }

        self.install()
    }

    /// Install XLA extension
    pub fn install(&self) -> Result<(), String> {
        // Create installation directory
        fs::create_dir_all(&self.install_dir)
            .map_err(|e| format!("Failed to create install directory: {}", e))?;

        // Try script-based installation first
        if let Ok(()) = self.try_script_installation() {
            return Ok(());
        }

        // Fallback to manual installation
        self.manual_install()
    }

    fn try_script_installation(&self) -> Result<(), String> {
        let script_dir =
            env::current_dir().map_err(|_| "Cannot get current directory")?.join("scripts");

        let script_path = script_dir.join(self.os.script_name());

        if !script_path.exists() {
            return Err("Installation script not found".to_string());
        }

        // Running installation script silently

        let mut cmd = match self.os {
            OS::Linux | OS::MacOS => {
                let mut cmd = Command::new("bash");
                cmd.arg(&script_path);
                cmd
            }
            OS::Windows => {
                let mut cmd = Command::new("powershell");
                cmd.args(["-ExecutionPolicy", "Bypass", "-File", script_path.to_str().unwrap()]);
                cmd
            }
        };

        cmd.env("ELIXIR_NX_XLA_VERSION", &self.version);
        cmd.env("XLA_EXTENSION_DIR", &self.install_dir);

        let output =
            cmd.output().map_err(|e| format!("Failed to execute installation script: {}", e))?;

        if !output.status.success() {
            return Err(format!(
                "Installation script failed with exit code: {}\nStderr: {}",
                output.status.code().unwrap_or(-1),
                String::from_utf8_lossy(&output.stderr)
            ));
        }

        if !self.is_installed() {
            return Err("Installation script completed but XLA extension not found".to_string());
        }

        // XLA extension installed successfully via script
        Ok(())
    }

    fn manual_install(&self) -> Result<(), String> {
        let filename = self.get_archive_filename();
        let download_url = self.get_download_url(&filename);
        let archive_path = self.install_dir.join(&filename);

        // Download if not already present
        if !archive_path.exists() {
            self.download_file(&download_url, &archive_path)?;
        }

        // Extract archive
        self.extract_archive(&archive_path)?;

        // Verify installation
        if !self.is_installed() {
            return Err(
                "Manual installation failed - XLA extension not found after extraction".to_string()
            );
        }

        Ok(())
    }

    fn get_archive_filename(&self) -> String {
        format!(
            "xla_extension-{}-{}-{}-cpu.{}",
            self.version,
            self.arch.name(),
            self.os.platform_name(),
            self.os.file_extension()
        )
    }

    fn get_download_url(&self, filename: &str) -> String {
        format!("https://github.com/elixir-nx/xla/releases/download/v{}/{}", self.version, filename)
    }

    fn download_file(&self, url: &str, output_path: &Path) -> Result<(), String> {
        // Downloading XLA extension silently

        // Try curl first
        let curl_result =
            Command::new("curl").args(["-L", "-o", output_path.to_str().unwrap(), url]).output();

        if let Ok(output) = curl_result {
            if output.status.success() {
                return Ok(());
            }
        }

        // Try wget as fallback
        let wget_result =
            Command::new("wget").args(["-O", output_path.to_str().unwrap(), url]).output();

        if let Ok(output) = wget_result {
            if output.status.success() {
                return Ok(());
            }
        }

        Err("Neither curl nor wget available or download failed".to_string())
    }

    fn extract_archive(&self, archive_path: &Path) -> Result<(), String> {
        // Extracting archive silently

        // Clean up existing installation
        let lib_dir = self.install_dir.join("lib");
        let include_dir = self.install_dir.join("include");
        if lib_dir.exists() {
            let _ = fs::remove_dir_all(&lib_dir);
        }
        if include_dir.exists() {
            let _ = fs::remove_dir_all(&include_dir);
        }

        match self.os {
            OS::Linux | OS::MacOS => {
                let status = Command::new("tar")
                    .args(["-xzf", archive_path.to_str().unwrap()])
                    .current_dir(&self.install_dir)
                    .status()
                    .map_err(|e| format!("Failed to run tar: {}", e))?;

                if !status.success() {
                    return Err("Tar extraction failed".to_string());
                }

                // Handle subdirectory structure (move from xla_extension/ to current dir)
                let extracted_subdir = self.install_dir.join("xla_extension");
                if extracted_subdir.exists() {
                    let lib_src = extracted_subdir.join("lib");
                    let include_src = extracted_subdir.join("include");

                    if lib_src.exists() && include_src.exists() {
                        fs::rename(lib_src, &lib_dir)
                            .map_err(|e| format!("Failed to move lib directory: {}", e))?;
                        fs::rename(include_src, &include_dir)
                            .map_err(|e| format!("Failed to move include directory: {}", e))?;
                        let _ = fs::remove_dir_all(extracted_subdir);
                    }
                }
            }
            OS::Windows => {
                return Err(
                    "ZIP extraction not implemented. Please use PowerShell script instead."
                        .to_string(),
                );
            }
        }

        Ok(())
    }

    /// Set up build environment variables for linking
    pub fn setup_build_env(&self) -> Result<(), String> {
        if !self.is_installed() {
            return Err("XLA extension not installed. Run install_if_needed() first.".to_string());
        }

        let lib_dir = self.install_dir.join("lib");
        let include_dir = self.install_dir.join("include");

        // Add library search path
        println!("cargo:rustc-link-search=native={}", lib_dir.display());

        // Add rpath for runtime linking
        match self.os {
            OS::MacOS => {
                println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
            }
            OS::Linux => {
                println!("cargo:rustc-link-arg=-Wl,-rpath={}", lib_dir.display());
            }
            OS::Windows => {
                // Windows uses PATH instead of rpath
            }
        }

        // Link XLA extension library
        println!("cargo:rustc-link-lib=xla_extension");

        // Add include directory for C++ compilation
        if let Ok(include_canonical) = include_dir.canonicalize() {
            println!("cargo:include={}", include_canonical.display());
        }

        // Build environment configured for XLA extension
        Ok(())
    }

    /// Validate installation and provide diagnostic information
    pub fn validate_installation(&self) -> Result<(), String> {
        if !self.is_installed() {
            return Err("XLA extension is not properly installed".to_string());
        }

        let lib_dir = self.install_dir.join("lib");
        let include_dir = self.install_dir.join("include");

        // Check library files
        if let Ok(entries) = fs::read_dir(&lib_dir) {
            let mut found_libs = Vec::new();
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if name_str.contains("xla") {
                    found_libs.push(name_str.to_string());
                }
            }

            if found_libs.is_empty() {
                return Err("No XLA library files found".to_string());
            }

            // Found XLA libraries silently
        }

        // Check include files
        let pjrt_header = include_dir.join("xla").join("pjrt").join("c").join("pjrt_c_api.h");
        if !pjrt_header.exists() {
            return Err("PJRT C API header not found".to_string());
        }

        // XLA extension installation validated successfully
        Ok(())
    }

    /// Get installation info for debugging
    pub fn get_info(&self) -> InstallationInfo {
        InstallationInfo {
            version: self.version.clone(),
            install_dir: self.install_dir.clone(),
            os: self.os,
            arch: self.arch,
            is_installed: self.is_installed(),
        }
    }
}

impl Default for XlaInstaller {
    fn default() -> Self {
        Self::new().expect("Failed to create default XLA installer")
    }
}

#[derive(Debug)]
pub struct InstallationInfo {
    pub version: String,
    pub install_dir: PathBuf,
    pub os: OS,
    pub arch: Arch,
    pub is_installed: bool,
}

impl std::fmt::Display for InstallationInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "XLA Extension Info:\n  Version: {}\n  Install Dir: {}\n  OS: {:?}\n  Arch: {:?}\n  Installed: {}",
            self.version,
            self.install_dir.display(),
            self.os,
            self.arch,
            self.is_installed
        )
    }
}

/// Convenience function for use in build.rs
/// This function handles the complete XLA installation process
pub fn ensure_xla_installation() -> Result<PathBuf, String> {
    let installer = XlaInstaller::new()?;

    installer.install_if_needed()?;
    installer.validate_installation()?;
    installer.setup_build_env()?;

    Ok(installer.install_dir().to_path_buf())
}

/// Convenience function with custom version
pub fn ensure_xla_installation_with_version(version: String) -> Result<PathBuf, String> {
    let installer = XlaInstaller::with_version(version)?;

    println!("cargo:warning={}", installer.get_info());

    installer.install_if_needed()?;
    installer.validate_installation()?;
    installer.setup_build_env()?;

    Ok(installer.install_dir().to_path_buf())
}

/// Clean up XLA installation (for testing or troubleshooting)
pub fn clean_xla_installation() -> Result<(), String> {
    let installer = XlaInstaller::new()?;
    let install_dir = installer.install_dir();

    if install_dir.exists() {
        fs::remove_dir_all(install_dir)
            .map_err(|e| format!("Failed to remove installation directory: {}", e))?;
        println!("cargo:warning=XLA installation cleaned successfully");
    } else {
        println!("cargo:warning=No XLA installation found to clean");
    }

    Ok(())
}
