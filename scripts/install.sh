#!/bin/bash

# elixir-nx/xla Binary Installation Script
# This script downloads and installs pre-compiled XLA binaries from the elixir-nx/xla project
# Note: This downloads XLA binaries packaged by elixir-nx, not the original Google XLA

set -euo pipefail

# Configuration
LIB_VERSION="${LIB_VERSION:-0.4.0}"
INSTALL_DIR="${XLA_EXTENSION_DIR:-${HOME}/.hodu/hodu_xla/extensions}"

# Detect OS and architecture
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

echo "🚀 Installing hodu-rs/hodu_xla v${LIB_VERSION} (pre-compiled XLA binaries)"
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
        PLATFORM="unknown-linux-gnu"
        EXT="tar.gz"
        ;;
    "darwin")
        PLATFORM="apple-darwin"
        EXT="tar.gz"
        ;;
    *)
        echo "❌ Unsupported OS: ${OS}"
        exit 1
        ;;
esac

# Build download URL
BASE_URL="https://github.com/hodu-rs/hodu_xla/releases/download"
FILENAME="xla_extension-${LIB_VERSION}-${ARCH}-${PLATFORM}-cpu.${EXT}"
DOWNLOAD_URL="${BASE_URL}/v${LIB_VERSION}/${FILENAME}"

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

# Create version-specific directory
EXTRACT_DIR="${FILENAME%.tar.gz}"
VERSION_DIR="${INSTALL_DIR}/${EXTRACT_DIR}"

# Check if this version is already installed
if [ -d "${VERSION_DIR}/lib" ] && [ -d "${VERSION_DIR}/include" ]; then
    echo "✅ ${EXTRACT_DIR} already installed, skipping extraction"
else
    echo "📦 Extracting archive to ${VERSION_DIR}..."

    # Remove old installation if exists
    if [ -d "${VERSION_DIR}" ]; then
        echo "⚠️  Directory ${VERSION_DIR} already exists, removing..."
        rm -rf "${VERSION_DIR}"
    fi

    mkdir -p "${VERSION_DIR}"
    tar -xzf "${FILENAME}" -C "${VERSION_DIR}"

    # Move contents from extracted subdirectory if needed
    if [ -d "${VERSION_DIR}/xla_extension/lib" ] && [ -d "${VERSION_DIR}/xla_extension/include" ]; then
        echo "📁 Moving contents from xla_extension subdirectory..."
        mv "${VERSION_DIR}/xla_extension/lib" "${VERSION_DIR}/"
        mv "${VERSION_DIR}/xla_extension/include" "${VERSION_DIR}/"
        rmdir "${VERSION_DIR}/xla_extension"
    fi
fi

# Verify installation
if [ -d "${VERSION_DIR}/lib" ] && [ -d "${VERSION_DIR}/include" ]; then
    echo "✅ Installation verified at ${VERSION_DIR}"

    # List library files
    echo "📚 Library files:"
    ls -la "${VERSION_DIR}/lib/"

    # Check for critical headers
    echo "🔍 Checking critical headers:"
    if [ -f "${VERSION_DIR}/include/xla/pjrt/c/pjrt_c_api.h" ]; then
        echo "✅ PJRT C API header found"
    else
        echo "⚠️ PJRT C API header not found"
    fi

    if [ -d "${VERSION_DIR}/include/xla" ]; then
        echo "✅ XLA headers found"
    else
        echo "⚠️ XLA headers not found"
    fi
else
    echo "❌ Installation failed - lib or include directory not found"
    exit 1
fi

# Set up environment
echo ""
echo "🎉 Installation completed successfully!"
echo ""
echo "📝 To use this installation, set the environment variable:"
echo "   export XLA_EXTENSION_DIR=${VERSION_DIR}"
echo ""
echo "🔧 Or use it directly in your build:"
echo "   XLA_EXTENSION_DIR=${VERSION_DIR} cargo build"
echo ""
echo "📁 Installed at: ${VERSION_DIR}"
echo ""

# Clean up downloaded archive (optional)
ARCHIVE_PATH="${INSTALL_DIR}/${FILENAME}"
if [ -f "${ARCHIVE_PATH}" ]; then
    read -p "🗑️  Remove downloaded archive ${FILENAME}? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm "${ARCHIVE_PATH}"
        echo "✅ Archive removed"
    else
        echo "📦 Archive kept at ${ARCHIVE_PATH}"
    fi
fi

echo ""
echo "✨ Ready to build Rust projects with XLA support!"
