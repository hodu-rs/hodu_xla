# xla-rs

Experimentation using the xla compiler from rust

The XLA extension is automatically downloaded and installed when you build the project:

```bash
cargo build
```

The build system automatically downloads pre-compiled XLA binaries from the [elixir-nx/xla repo](https://github.com/elixir-nx/xla/releases). This downloads the XLA binaries packaged by the elixir-nx project, not the original XLA source. The currently supported elixir-nx/xla version is 0.6.0.

You can customize the installation:
- `ELIXIR_NX_XLA_VERSION`: Set the elixir-nx/xla release version (default: 0.6.0)
- `XLA_EXTENSION_DIR`: Set custom installation directory (default: ./xla_extension)

For manual installation, use the provided scripts:
```bash
./scripts/install.sh    # Unix/Linux/macOS
./scripts/install.ps1   # Windows PowerShell
```
