use hodu_xla::{ArrayElement, Result};

/// Helper function to create a literal from typed data
fn create_literal_f32(dims: &[usize], data: &[f32]) -> Result<hodu_xla::Literal> {
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    hodu_xla::Literal::create_from_shape_and_untyped_data(f32::TY, dims, bytes)
}

/// Test Rev (reverse) operation on 1D tensor
#[test]
fn test_rev_1d() -> Result<()> {
    let client = hodu_xla::PjRtClient::cpu()?;
    let builder = hodu_xla::XlaBuilder::new("rev_1d_test");

    // Input: 1x1x5 (batch=1, channels=1, length=5)
    let input = builder.parameter(0, f32::TY, &[1, 1, 5], "input")?;

    // Reverse along dimension 2 (the length dimension)
    let result = input.rev(&[2])?;

    let computation = result.build()?;
    let executable = client.compile(&computation)?;

    let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let input_literal = create_literal_f32(&[1, 1, 5], &input_data)?;

    let result = executable.execute::<hodu_xla::Literal>(&[input_literal])?;
    let result = result[0][0].to_literal_sync()?;

    // Expected: [5.0, 4.0, 3.0, 2.0, 1.0]
    let output = result.to_vec::<f32>()?;
    let expected: Vec<f32> = vec![5.0, 4.0, 3.0, 2.0, 1.0];

    eprintln!("Input: {:?}", vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    eprintln!("Output: {:?}", output);
    eprintln!("Expected: {:?}", expected);

    for (i, (&out, &exp)) in output.iter().zip(expected.iter()).enumerate() {
        assert!(
            (out - exp).abs() < 1e-5,
            "Mismatch at index {}: got {}, expected {}",
            i,
            out,
            exp
        );
    }

    Ok(())
}

/// Test Rev operation on 2D tensor (reversing spatial dimensions)
#[test]
fn test_rev_2d() -> Result<()> {
    let client = hodu_xla::PjRtClient::cpu()?;
    let builder = hodu_xla::XlaBuilder::new("rev_2d_test");

    // Input: 1x1x2x3 (batch=1, channels=1, height=2, width=3)
    let input = builder.parameter(0, f32::TY, &[1, 1, 2, 3], "input")?;

    // Reverse along dimensions 2 and 3 (both spatial dimensions)
    let result = input.rev(&[2, 3])?;

    let computation = result.build()?;
    let executable = client.compile(&computation)?;

    // Input: [[1, 2, 3],
    //         [4, 5, 6]]
    let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input_literal = create_literal_f32(&[1, 1, 2, 3], &input_data)?;

    let result = executable.execute::<hodu_xla::Literal>(&[input_literal])?;
    let result = result[0][0].to_literal_sync()?;

    // Expected: [[6, 5, 4],
    //            [3, 2, 1]]
    let output = result.to_vec::<f32>()?;
    let expected: Vec<f32> = vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

    eprintln!("Input (2x3): {:?}", input_data);
    eprintln!("Output (2x3): {:?}", output);
    eprintln!("Expected (2x3): {:?}", expected);

    for (i, (&out, &exp)) in output.iter().zip(expected.iter()).enumerate() {
        assert!(
            (out - exp).abs() < 1e-5,
            "Mismatch at index {}: got {}, expected {}",
            i,
            out,
            exp
        );
    }

    Ok(())
}

/// Test Rev operation on kernel for ConvTranspose (reverse only spatial dims)
#[test]
fn test_rev_kernel_1d() -> Result<()> {
    let client = hodu_xla::PjRtClient::cpu()?;
    let builder = hodu_xla::XlaBuilder::new("rev_kernel_1d_test");

    // Kernel: 1x1x3 (in_channels=1, out_channels=1, kernel_size=3)
    let kernel = builder.parameter(0, f32::TY, &[1, 1, 3], "kernel")?;

    // Reverse along dimension 2 (the kernel spatial dimension)
    let result = kernel.rev(&[2])?;

    let computation = result.build()?;
    let executable = client.compile(&computation)?;

    let kernel_data: Vec<f32> = vec![1.0, 0.5, 0.25];
    let kernel_literal = create_literal_f32(&[1, 1, 3], &kernel_data)?;

    let result = executable.execute::<hodu_xla::Literal>(&[kernel_literal])?;
    let result = result[0][0].to_literal_sync()?;

    // Expected: [0.25, 0.5, 1.0]
    let output = result.to_vec::<f32>()?;
    let expected: Vec<f32> = vec![0.25, 0.5, 1.0];

    eprintln!("Kernel: {:?}", kernel_data);
    eprintln!("Reversed: {:?}", output);
    eprintln!("Expected: {:?}", expected);

    for (i, (&out, &exp)) in output.iter().zip(expected.iter()).enumerate() {
        assert!(
            (out - exp).abs() < 1e-5,
            "Mismatch at index {}: got {}, expected {}",
            i,
            out,
            exp
        );
    }

    Ok(())
}
