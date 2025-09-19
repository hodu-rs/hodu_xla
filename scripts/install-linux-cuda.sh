#!/bin/bash

# hodu-rs/xla-rs CUDA Binary Installation Script for Linux
# This script downloads and installs pre-compiled XLA CUDA binaries from the hodu-rs/xla-rs project

set -euo pipefail

# Configuration
XLA_VERSION="${XLA_VERSION:-0.9.1}"
LIB_VERSION="${LIB_VERSION:-0.3.0}"
INSTALL_DIR="${XLA_EXTENSION_DIR:-${HOME}/.hodu/xla-rs/xla_extension}"

# Detect architecture
ARCH=$(uname -m)

echo "🚀 Installing hodu-rs/xla-rs v${XLA_VERSION} CUDA (pre-compiled XLA binaries)"
echo "📁 Installation directory: ${INSTALL_DIR}"
echo "🎯 Target: CUDA"
echo "🖥️  Detected Architecture: ${ARCH}"

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

# Build download URL for CUDA version
BASE_URL="https://github.com/hodu-rs/xla-rs/releases/download"
FILENAME="xla_extension-${XLA_VERSION}-${ARCH}-linux-gnu-cuda12.tar.gz"
DOWNLOAD_URL="${BASE_URL}/${LIB_VERSION}/${FILENAME}"

echo "🌐 Download URL: ${DOWNLOAD_URL}"

# Create installation directory
mkdir -p "${INSTALL_DIR}"
cd "${INSTALL_DIR}"

# Download if not already present
if [ ! -f "${FILENAME}" ]; then
    echo "📥 Downloading XLA CUDA extension..."
    if command -v curl >/dev/null 2>&1; then
        curl -L -o "${FILENAME}" "${DOWNLOAD_URL}"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "${FILENAME}" "${DOWNLOAD_URL}"
    else
        echo "❌ Neither curl nor wget found. Please install one of them."
        exit 1
    fi

    if [ $? -ne 0 ]; then
        echo "❌ Download failed. Please check your internet connection and try again."
        rm -f "${FILENAME}"
        exit 1
    fi

    echo "✅ Download completed"
else
    echo "✅ ${FILENAME} already exists, skipping download"
fi

# Extract archive
echo "📦 Extracting archive..."

# Remove existing directories if they exist
if [ -d "lib" ] || [ -d "include" ]; then
    echo "⚠️  Installation directories already exist, removing..."
    rm -rf lib include
fi

tar -xzf "${FILENAME}"

# Move contents from extracted subdirectory if needed
if [ -d "xla_extension/lib" ] && [ -d "xla_extension/include" ]; then
    echo "📁 Moving contents from xla_extension subdirectory..."
    mv xla_extension/lib .
    mv xla_extension/include .
    rm -rf xla_extension
fi

echo "✅ Extraction completed successfully"

# Verify installation
if [ -d "lib" ] && [ -d "include" ]; then
    echo "📚 Library files:"
    ls -la lib/

    echo "🔍 Checking critical headers:"
    PJRT_HEADER="include/xla/pjrt/c/pjrt_c_api.h"

    if [ -f "${PJRT_HEADER}" ]; then
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

echo ""
echo "🎉 XLA CUDA extension installation completed successfully!"
echo ""
echo "📝 Installation directory: ${INSTALL_DIR}"
echo "🔧 The extension is now ready for use with CUDA-enabled applications."
echo ""

# Optional: Clean up downloaded archive
read -p "🗑️  Remove downloaded archive ${FILENAME}? (y/N): " -r
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -f "${FILENAME}"
    echo "✅ Archive removed"
else
    echo "📦 Archive kept at ${INSTALL_DIR}/${FILENAME}"
fi

echo ""
echo "✨ Ready to build Rust projects with XLA CUDA support!"