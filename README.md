# hodu_xla

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
