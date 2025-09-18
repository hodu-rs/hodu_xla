#!/bin/bash

# elixir-nx/xla Binary Installation Script
# This script downloads and installs pre-compiled XLA binaries from the elixir-nx/xla project
# Note: This downloads XLA binaries packaged by elixir-nx, not the original Google XLA

set -euo pipefail

# Configuration
ELIXIR_NX_XLA_VERSION="${ELIXIR_NX_XLA_VERSION:-0.8.0}"
INSTALL_DIR="${XLA_EXTENSION_DIR:-${HOME}/.hodu/xla-rs/xla_extension}"

# Detect OS and architecture
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

echo "🚀 Installing elixir-nx/xla v${ELIXIR_NX_XLA_VERSION} (pre-compiled XLA binaries)"
echo "📁 Installation directory: ${INSTALL_DIR}"
echo "🖥️  Detected OS: ${OS}, Architecture: ${ARCH}"

# Map architecture names
case "${ARCH}" in
    "x86_64")
        ARCH="x86_64"
        ;;
    "aarch64"|"arm64")
        ARCH="aarch64"
        ;;
    *)
        echo "❌ Unsupported architecture: ${ARCH}"
        exit 1
        ;;
esac

# Map OS names and determine file extension
case "${OS}" in
    "linux")
        PLATFORM="linux"
        EXT="tar.gz"
        ;;
    "darwin")
        PLATFORM="darwin"
        EXT="tar.gz"
        ;;
    *)
        echo "❌ Unsupported OS: ${OS}"
        exit 1
        ;;
esac

# Build download URL
BASE_URL="https://github.com/elixir-nx/xla/releases/download"
FILENAME="xla_extension-${ELIXIR_NX_XLA_VERSION}-${ARCH}-${PLATFORM}-cpu.${EXT}"
DOWNLOAD_URL="${BASE_URL}/v${ELIXIR_NX_XLA_VERSION}/${FILENAME}"

echo "🌐 Download URL: ${DOWNLOAD_URL}"

# Create installation directory
mkdir -p "${INSTALL_DIR}"
cd "${INSTALL_DIR}"

# Download if not already present
if [ ! -f "${FILENAME}" ]; then
    echo "📥 Downloading XLA extension..."
    if command -v curl >/dev/null 2>&1; then
        curl -L -o "${FILENAME}" "${DOWNLOAD_URL}"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "${FILENAME}" "${DOWNLOAD_URL}"
    else
        echo "❌ Neither curl nor wget found. Please install one of them."
        exit 1
    fi
else
    echo "✅ ${FILENAME} already exists, skipping download"
fi

# Extract archive
echo "📦 Extracting archive..."
if [ -d "lib" ] && [ -d "include" ]; then
    echo "⚠️  Installation directories already exist, removing..."
    rm -rf lib include
fi

tar -xzf "${FILENAME}"

# Move contents from extracted subdirectory if needed
if [ -d "xla_extension/lib" ] && [ -d "xla_extension/include" ]; then
    echo "📁 Moving contents from xla_extension subdirectory..."
    mv xla_extension/lib .
    mv xla_extension/include .
    rmdir xla_extension
fi

# Verify installation
if [ -d "lib" ] && [ -d "include" ]; then
    echo "✅ Extraction completed successfully"

    # List library files
    echo "📚 Library files:"
    ls -la lib/

    # Check for critical headers
    echo "🔍 Checking critical headers:"
    if [ -f "include/xla/pjrt/c/pjrt_c_api.h" ]; then
        echo "✅ PJRT C API header found"
    else
        echo "⚠️ PJRT C API header not found"
    fi

    if [ -d "include/xla" ]; then
        echo "✅ XLA headers found"
    else
        echo "⚠️ XLA headers not found"
    fi
else
    echo "❌ Extraction failed - lib or include directory not found"
    exit 1
fi

# Set up environment
echo ""
echo "🎉 Installation completed successfully!"
echo ""
echo "📝 To use this installation, set the environment variable:"
echo "   export XLA_EXTENSION_DIR=${INSTALL_DIR}"
echo ""
echo "🔧 Or use it directly in your build:"
echo "   XLA_EXTENSION_DIR=${INSTALL_DIR} cargo build"
echo ""

# Clean up downloaded archive (optional)
read -p "🗑️  Remove downloaded archive ${FILENAME}? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm "${FILENAME}"
    echo "✅ Archive removed"
else
    echo "📦 Archive kept at ${INSTALL_DIR}/${FILENAME}"
fi

echo ""
echo "✨ Ready to build Rust projects with XLA support!"
