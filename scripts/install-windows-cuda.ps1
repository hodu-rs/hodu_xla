# hodu-rs/xla-rs CUDA Binary Installation Script for Windows PowerShell
# This script downloads and installs pre-compiled XLA CUDA binaries from the hodu-rs/xla-rs project

param(
    [string]$XlaVersion = "0.9.1",
    [string]$LibVersion = "0.3.0",
    [string]$InstallDir = $null
)

# Set error handling
$ErrorActionPreference = "Stop"

# Configuration
if (-not $InstallDir) {
    $InstallDir = Join-Path $env:USERPROFILE ".hodu\xla-rs\xla_extension"
}

# Use environment variables if available
if ($env:XLA_VERSION) { $XlaVersion = $env:XLA_VERSION }
if ($env:LIB_VERSION) { $LibVersion = $env:LIB_VERSION }

Write-Host "🚀 Installing XLA CUDA Extension v$XlaVersion" -ForegroundColor Green
Write-Host "📁 Installation directory: $InstallDir" -ForegroundColor Cyan
Write-Host "🎯 Target: CUDA" -ForegroundColor Cyan

# Detect architecture
$Arch = if ([System.Environment]::Is64BitOperatingSystem) { "x86_64" } else { "x86" }
Write-Host "🖥️  Detected Architecture: $Arch" -ForegroundColor Cyan

# Note: Currently only x86_64 CUDA binaries are available for Windows
if ($Arch -ne "x86_64") {
    Write-Host "❌ CUDA binaries are only available for x86_64 architecture" -ForegroundColor Red
    exit 1
}

# Build download URL for CUDA version
$BaseUrl = "https://github.com/hodu-rs/xla-rs/releases/download"
$Filename = "xla_extension-$XlaVersion-$Arch-windows-cuda12.zip"
$DownloadUrl = "$BaseUrl/$LibVersion/$Filename"

Write-Host "🌐 Download URL: $DownloadUrl" -ForegroundColor Yellow

# Create installation directory
if (-not (Test-Path $InstallDir)) {
    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
}

Set-Location $InstallDir

# Download if not already present
$FilePath = Join-Path $InstallDir $Filename
if (-not (Test-Path $FilePath)) {
    Write-Host "📥 Downloading XLA CUDA extension..." -ForegroundColor Yellow

    try {
        # Use Invoke-WebRequest for downloading
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -Uri $DownloadUrl -OutFile $FilePath -UseBasicParsing
        Write-Host "✅ Download completed" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Download failed: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "Note: CUDA binaries may not be available for Windows yet. Check the releases page." -ForegroundColor Yellow
        exit 1
    }
}
else {
    Write-Host "✅ $Filename already exists, skipping download" -ForegroundColor Green
}

# Extract archive
Write-Host "📦 Extracting archive..." -ForegroundColor Yellow

# Remove existing directories if they exist
$LibDir = Join-Path $InstallDir "lib"
$IncludeDir = Join-Path $InstallDir "include"

if ((Test-Path $LibDir) -or (Test-Path $IncludeDir)) {
    Write-Host "⚠️  Installation directories already exist, removing..." -ForegroundColor Yellow
    if (Test-Path $LibDir) { Remove-Item $LibDir -Recurse -Force }
    if (Test-Path $IncludeDir) { Remove-Item $IncludeDir -Recurse -Force }
}

try {
    # Extract using .NET classes (available in PowerShell 5.0+)
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::ExtractToDirectory($FilePath, $InstallDir)

    # Move contents from extracted subdirectory if needed
    $ExtractedXlaDir = Join-Path $InstallDir "xla_extension"
    $ExtractedLibDir = Join-Path $ExtractedXlaDir "lib"
    $ExtractedIncludeDir = Join-Path $ExtractedXlaDir "include"

    if ((Test-Path $ExtractedLibDir) -and (Test-Path $ExtractedIncludeDir)) {
        Write-Host "📁 Moving contents from xla_extension subdirectory..." -ForegroundColor Yellow
        Move-Item $ExtractedLibDir $InstallDir -Force
        Move-Item $ExtractedIncludeDir $InstallDir -Force
        Remove-Item $ExtractedXlaDir -Recurse -Force
    }

    Write-Host "✅ Extraction completed successfully" -ForegroundColor Green
}
catch {
    Write-Host "❌ Extraction failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Verify installation
if ((Test-Path $LibDir) -and (Test-Path $IncludeDir)) {
    Write-Host "📚 Library files:" -ForegroundColor Cyan
    Get-ChildItem $LibDir | ForEach-Object { Write-Host "  $($_.Name)" }

    Write-Host "🔍 Checking critical headers:" -ForegroundColor Cyan
    $PjrtHeader = Join-Path $IncludeDir "xla\pjrt\c\pjrt_c_api.h"
    $XlaDir = Join-Path $IncludeDir "xla"

    if (Test-Path $PjrtHeader) {
        Write-Host "✅ PJRT C API header found" -ForegroundColor Green
    } else {
        Write-Host "⚠️ PJRT C API header not found" -ForegroundColor Yellow
    }

    if (Test-Path $XlaDir) {
        Write-Host "✅ XLA headers found" -ForegroundColor Green
    } else {
        Write-Host "⚠️ XLA headers not found" -ForegroundColor Yellow
    }
}
else {
    Write-Host "❌ Extraction failed - lib or include directory not found" -ForegroundColor Red
    exit 1
}

# Set up environment
Write-Host ""
Write-Host "🎉 XLA CUDA extension installation completed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "📝 Installation directory: $InstallDir" -ForegroundColor Cyan
Write-Host "🔧 The extension is now ready for use with CUDA-enabled applications." -ForegroundColor Cyan
Write-Host ""

# Clean up downloaded archive (optional)
$CleanupChoice = Read-Host "🗑️  Remove downloaded archive $Filename? (y/N)"
if ($CleanupChoice -match "^[Yy]$") {
    Remove-Item $FilePath -Force
    Write-Host "✅ Archive removed" -ForegroundColor Green
}
else {
    Write-Host "📦 Archive kept at $FilePath" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "✨ Ready to build Rust projects with XLA CUDA support!" -ForegroundColor Green