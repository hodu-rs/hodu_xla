#!/bin/bash

set -e

# Stay in build directory
cd "$(dirname "$0")"

print_usage_and_exit() {
  echo "Usage: $0 <target> <xla_dir>"
  echo ""
  echo "Compiles the project using local XLA source. Available targets: cpu, cuda12, tpu, rocm."
  echo ""
  echo "Arguments:"
  echo "  target    - Build target (cpu, cuda12, tpu, rocm)"
  echo "  xla_dir   - Path to XLA source directory"
  echo ""
  echo "Example:"
  echo "  $0 cuda12 /path/to/xla"
  exit 1
}

if [ $# -ne 2 ]; then
  print_usage_and_exit
fi

target="$1"
xla_dir="$2"

# Validate XLA directory
if [ ! -d "$xla_dir" ]; then
  echo "Error: XLA directory does not exist: $xla_dir"
  exit 1
fi

# Convert to absolute path
xla_dir=$(cd "$xla_dir" && pwd)

echo "Building XLA extension for target: $target"
echo "Using XLA source directory: $xla_dir"

# Set environment variables based on target
case "$target" in
  "cpu")
    export XLA_TARGET=cpu
    BAZEL_FLAGS=""
  ;;

  "tpu")
    export XLA_TARGET=tpu
    BAZEL_FLAGS="--define=with_tpu_support=true"
  ;;

  "cuda12")
    export XLA_TARGET=cuda12
    BAZEL_FLAGS="--config=cuda"
    BAZEL_FLAGS="$BAZEL_FLAGS --repo_env=HERMETIC_CUDA_VERSION=\"12.8.0\""
    BAZEL_FLAGS="$BAZEL_FLAGS --repo_env=HERMETIC_CUDNN_VERSION=\"9.8.0\""
    BAZEL_FLAGS="$BAZEL_FLAGS --action_env=HERMETIC_CUDA_COMPUTE_CAPABILITIES=\"sm_50,sm_60,sm_70,sm_80,sm_90,sm_100,compute_120\""
    BAZEL_FLAGS="$BAZEL_FLAGS --action_env=TF_NVCC_CLANG=\"1\""
    BAZEL_FLAGS="$BAZEL_FLAGS --@local_config_cuda//:cuda_compiler=nvcc"
  ;;

  "rocm")
    export XLA_TARGET=rocm
    BAZEL_FLAGS="--config=rocm"
    BAZEL_FLAGS="$BAZEL_FLAGS --action_env=HIP_PLATFORM=hcc"
    BAZEL_FLAGS="$BAZEL_FLAGS --action_env=TF_ROCM_AMDGPU_TARGETS=\"gfx900,gfx906,gfx908,gfx90a,gfx940,gfx941,gfx942,gfx1030,gfx1100,gfx1200,gfx1201\""
  ;;

  *)
    print_usage_and_exit
  ;;
esac

# Common Bazel flags
BAZEL_FLAGS="$BAZEL_FLAGS --define \"framework_shared_object=false\""
BAZEL_FLAGS="$BAZEL_FLAGS -c opt"
BAZEL_FLAGS="$BAZEL_FLAGS --repo_env=CC=clang"
BAZEL_FLAGS="$BAZEL_FLAGS --repo_env=CXX=clang++"
BAZEL_FLAGS="$BAZEL_FLAGS --copt=-Wno-error=unused-command-line-argument"
BAZEL_FLAGS="$BAZEL_FLAGS --copt=-Wno-gnu-offsetof-extensions"
BAZEL_FLAGS="$BAZEL_FLAGS --copt=-Qunused-arguments"
BAZEL_FLAGS="$BAZEL_FLAGS --copt=-Wno-error=c23-extensions"

# Get current build directory (absolute path)
build_dir="$(pwd)"

# Setup extension directory symlink
extension_link="$xla_dir/xla/extension"
echo "Creating extension symlink: $extension_link -> $build_dir"
rm -f "$extension_link"
ln -s "$build_dir" "$extension_link"

# Build
echo "Building with Bazel..."
echo "Bazel flags: $BAZEL_FLAGS"
cd "$xla_dir"
bazel build $BAZEL_FLAGS //xla/extension:xla_extension

# Copy output
output_dir="$build_dir/output/$target"
mkdir -p "$output_dir"
cp -f "$xla_dir/bazel-bin/xla/extension/xla_extension.tar.gz" "$output_dir/"

echo ""
echo "Build completed successfully!"
echo "Output: $output_dir/xla_extension.tar.gz"
