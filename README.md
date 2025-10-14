# hodu_xla

Experimentation using the xla compiler from rust

The XLA extension is automatically downloaded and installed when you build the project:

```bash
cargo build
```

The build system automatically downloads pre-compiled XLA binaries from the [hodu-rs/hodu_xla repo](https://github.com/hodu-rs/hodu_xla/releases). This downloads the XLA binaries packaged by the hodu-rs project.

You can customize the installation:
- `XLA_EXTENSION_DIR`: Set custom installation directory (default: ~/.hodu/hodu_xla/extensions)

## Manual Installation

For manual installation, use the provided scripts:

### CPU Version (Default)
```bash
./scripts/install.sh    # Unix/Linux/macOS
```

## Supported Platforms

| Target Triple | Device | Status |
|--------------|----------|--------|
| x86_64-unknown-linux-gnu | CPU | ✅ Stable |
| aarch64-unknown-linux-gnu | CPU | ✅ Stable |
| x86_64-apple-darwin | CPU | ❌ Not Supported |
| aarch64-apple-darwin | CPU | ✅ Stable |
| x86_64-pc-windows-msvc | CPU | ❌ Not Supported |
