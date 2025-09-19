# xla-rs

Experimentation using the xla compiler from rust

The XLA extension is automatically downloaded and installed when you build the project:

```bash
cargo build
```

The build system automatically downloads pre-compiled XLA binaries from the [hodu-rs/xla-rs repo](https://github.com/hodu-rs/xla-rs/releases). This downloads the XLA binaries packaged by the hodu-rs project.

You can customize the installation:
- `XLA_VERSION`: Set the XLA extension version (default: elixirnxxla0.9.1)
- `XLA_EXTENSION_DIR`: Set custom installation directory (default: ~/.hodu/xla-rs/xla_extension)

## Manual Installation

For manual installation, use the provided scripts:

### CPU Version (Default)
```bash
./scripts/install.sh    # Unix/Linux/macOS
./scripts/install.ps1   # Windows PowerShell
```

### CUDA Version
```bash
./scripts/install-linux-cuda.sh    # Linux with CUDA
./scripts/install-windows-cuda.ps1 # Windows with CUDA
```

## CUDA Support

To enable CUDA support, use the `cuda` feature:

```bash
cargo build --features cuda
```

This will automatically download and install CUDA-enabled XLA binaries on supported platforms (Linux and Windows). macOS does not support CUDA.
