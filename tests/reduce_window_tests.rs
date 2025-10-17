use hodu_xla::{ArrayElement, Result};

/// Helper function to create a literal from typed data
fn create_literal_f32(dims: &[usize], data: &[f32]) -> Result<hodu_xla::Literal> {
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    hodu_xla::Literal::create_from_shape_and_untyped_data(f32::TY, dims, bytes)
}

/// Test basic max pooling using reduce_window (2D)
#[test]
fn max_pool_2d_basic() -> Result<()> {
    let client = hodu_xla::PjRtClient::cpu()?;
    let builder = hodu_xla::XlaBuilder::new("max_pool_2d");

    // Input: 1x1x4x4 (batch=1, channels=1, height=4, width=4)
    let input = builder.parameter(0, f32::TY, &[1, 1, 4, 4], "input")?;

    // Create max computation
    let max_builder = hodu_xla::XlaBuilder::new("Max");
    let x = max_builder.parameter(0, f32::TY, &[], "x")?;
    let y = max_builder.parameter(1, f32::TY, &[], "y")?;
    let max_comp = x.max(&y)?.build()?;

    // Initial value for max is negative infinity
    let init_value = builder.min_value(f32::TY)?;

    // Apply reduce_window with 2x2 window, stride 2
    let result = input.reduce_window(
        init_value,
        max_comp,
        &[1, 1, 2, 2],                     // window_dimensions (keep batch and channel dims)
        &[1, 1, 2, 2],                     // window_strides
        &[(0, 0), (0, 0), (0, 0), (0, 0)], // no padding
    )?;

    let computation = result.build()?;
    let executable = client.compile(&computation)?;

    // Create input data: 4x4 matrix
    #[rustfmt::skip]
    let input_data: Vec<f32> = vec![
        1.0,  2.0,  3.0,  4.0,
        5.0,  6.0,  7.0,  8.0,
        9.0,  10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ];
    let input_literal = create_literal_f32(&[1, 1, 4, 4], &input_data)?;

    let result = executable.execute::<hodu_xla::Literal>(&[input_literal])?;
    let result = result[0][0].to_literal_sync()?;

    // Output should be 1x1x2x2 (2x2 pooling with stride 2)
    assert_eq!(result.array_shape()?.dims(), [1, 1, 2, 2]);

    let output = result.to_vec::<f32>()?;

    // Expected output (max of each 2x2 block):
    // [max(1,2,5,6), max(3,4,7,8)]      = [6, 8]
    // [max(9,10,13,14), max(11,12,15,16)] = [14, 16]
    #[rustfmt::skip]
    let expected: Vec<f32> = vec![
        6.0, 8.0,
        14.0, 16.0,
    ];

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

/// Test average pooling using reduce_window (2D)
#[test]
fn avg_pool_2d_basic() -> Result<()> {
    let client = hodu_xla::PjRtClient::cpu()?;
    let builder = hodu_xla::XlaBuilder::new("avg_pool_2d");

    // Input: 1x1x4x4 (batch=1, channels=1, height=4, width=4)
    let input = builder.parameter(0, f32::TY, &[1, 1, 4, 4], "input")?;

    // Create sum computation (we'll divide by window size later)
    let add_builder = hodu_xla::XlaBuilder::new("Add");
    let x = add_builder.parameter(0, f32::TY, &[], "x")?;
    let y = add_builder.parameter(1, f32::TY, &[], "y")?;
    let add_comp = x.add_(&y)?.build()?;

    // Initial value for sum is zero
    let init_value = builder.zero(f32::TY)?;

    // Apply reduce_window with 2x2 window, stride 2
    let sum_result = input.reduce_window(
        init_value,
        add_comp,
        &[1, 1, 2, 2],                     // window_dimensions
        &[1, 1, 2, 2],                     // window_strides
        &[(0, 0), (0, 0), (0, 0), (0, 0)], // no padding
    )?;

    // Divide by window size (2*2 = 4) to get average
    let window_size = builder.constant_r0(4f32)?;
    let result = (sum_result / window_size)?;

    let computation = result.build()?;
    let executable = client.compile(&computation)?;

    // Create input data: 4x4 matrix
    #[rustfmt::skip]
    let input_data: Vec<f32> = vec![
        1.0,  2.0,  3.0,  4.0,
        5.0,  6.0,  7.0,  8.0,
        9.0,  10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ];
    let input_literal = create_literal_f32(&[1, 1, 4, 4], &input_data)?;

    let result = executable.execute::<hodu_xla::Literal>(&[input_literal])?;
    let result = result[0][0].to_literal_sync()?;

    // Output should be 1x1x2x2 (2x2 pooling with stride 2)
    assert_eq!(result.array_shape()?.dims(), [1, 1, 2, 2]);

    let output = result.to_vec::<f32>()?;

    // Expected output (average of each 2x2 block):
    // [avg(1,2,5,6), avg(3,4,7,8)]         = [3.5, 5.5]
    // [avg(9,10,13,14), avg(11,12,15,16)]  = [11.5, 13.5]
    #[rustfmt::skip]
    let expected: Vec<f32> = vec![
        3.5, 5.5,
        11.5, 13.5,
    ];

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

/// Test max pooling with stride 1 (overlapping windows)
#[test]
fn max_pool_2d_stride_1() -> Result<()> {
    let client = hodu_xla::PjRtClient::cpu()?;
    let builder = hodu_xla::XlaBuilder::new("max_pool_2d_stride1");

    // Input: 1x1x3x3
    let input = builder.parameter(0, f32::TY, &[1, 1, 3, 3], "input")?;

    // Create max computation
    let max_builder = hodu_xla::XlaBuilder::new("Max");
    let x = max_builder.parameter(0, f32::TY, &[], "x")?;
    let y = max_builder.parameter(1, f32::TY, &[], "y")?;
    let max_comp = x.max(&y)?.build()?;

    let init_value = builder.min_value(f32::TY)?;

    // Apply reduce_window with 2x2 window, stride 1
    let result = input.reduce_window(
        init_value,
        max_comp,
        &[1, 1, 2, 2], // window_dimensions
        &[1, 1, 1, 1], // window_strides (stride 1 = overlapping)
        &[(0, 0), (0, 0), (0, 0), (0, 0)],
    )?;

    let computation = result.build()?;
    let executable = client.compile(&computation)?;

    #[rustfmt::skip]
    let input_data: Vec<f32> = vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    ];
    let input_literal = create_literal_f32(&[1, 1, 3, 3], &input_data)?;

    let result = executable.execute::<hodu_xla::Literal>(&[input_literal])?;
    let result = result[0][0].to_literal_sync()?;

    // Output should be 1x1x2x2 (3x3 input with 2x2 window and stride 1)
    assert_eq!(result.array_shape()?.dims(), [1, 1, 2, 2]);

    let output = result.to_vec::<f32>()?;

    // Expected output:
    // [max(1,2,4,5), max(2,3,5,6)]     = [5, 6]
    // [max(4,5,7,8), max(5,6,8,9)]     = [8, 9]
    #[rustfmt::skip]
    let expected: Vec<f32> = vec![
        5.0, 6.0,
        8.0, 9.0,
    ];

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

/// Test 1D max pooling
#[test]
fn max_pool_1d_basic() -> Result<()> {
    let client = hodu_xla::PjRtClient::cpu()?;
    let builder = hodu_xla::XlaBuilder::new("max_pool_1d");

    // Input: 1x1x8 (batch=1, channels=1, length=8)
    let input = builder.parameter(0, f32::TY, &[1, 1, 8], "input")?;

    // Create max computation
    let max_builder = hodu_xla::XlaBuilder::new("Max");
    let x = max_builder.parameter(0, f32::TY, &[], "x")?;
    let y = max_builder.parameter(1, f32::TY, &[], "y")?;
    let max_comp = x.max(&y)?.build()?;

    let init_value = builder.min_value(f32::TY)?;

    // Apply reduce_window with window size 2, stride 2
    let result = input.reduce_window(
        init_value,
        max_comp,
        &[1, 1, 2],                // window_dimensions
        &[1, 1, 2],                // window_strides
        &[(0, 0), (0, 0), (0, 0)], // no padding
    )?;

    let computation = result.build()?;
    let executable = client.compile(&computation)?;

    let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let input_literal = create_literal_f32(&[1, 1, 8], &input_data)?;

    let result = executable.execute::<hodu_xla::Literal>(&[input_literal])?;
    let result = result[0][0].to_literal_sync()?;

    // Output should be 1x1x4
    assert_eq!(result.array_shape()?.dims(), [1, 1, 4]);

    let output = result.to_vec::<f32>()?;

    // Expected output: [max(1,2), max(3,4), max(5,6), max(7,8)] = [2, 4, 6, 8]
    let expected: Vec<f32> = vec![2.0, 4.0, 6.0, 8.0];

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

/// Test 3D max pooling
#[test]
fn max_pool_3d_basic() -> Result<()> {
    let client = hodu_xla::PjRtClient::cpu()?;
    let builder = hodu_xla::XlaBuilder::new("max_pool_3d");

    // Input: 1x1x2x2x2 (batch=1, channels=1, depth=2, height=2, width=2)
    let input = builder.parameter(0, f32::TY, &[1, 1, 2, 2, 2], "input")?;

    // Create max computation
    let max_builder = hodu_xla::XlaBuilder::new("Max");
    let x = max_builder.parameter(0, f32::TY, &[], "x")?;
    let y = max_builder.parameter(1, f32::TY, &[], "y")?;
    let max_comp = x.max(&y)?.build()?;

    let init_value = builder.min_value(f32::TY)?;

    // Apply reduce_window with 2x2x2 window, stride 2
    let result = input.reduce_window(
        init_value,
        max_comp,
        &[1, 1, 2, 2, 2],                          // window_dimensions
        &[1, 1, 2, 2, 2],                          // window_strides
        &[(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)], // no padding
    )?;

    let computation = result.build()?;
    let executable = client.compile(&computation)?;

    // 2x2x2 cube with values 1-8
    let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let input_literal = create_literal_f32(&[1, 1, 2, 2, 2], &input_data)?;

    let result = executable.execute::<hodu_xla::Literal>(&[input_literal])?;
    let result = result[0][0].to_literal_sync()?;

    // Output should be 1x1x1x1x1 (single max value)
    assert_eq!(result.array_shape()?.dims(), [1, 1, 1, 1, 1]);

    let output = result.to_vec::<f32>()?;

    // Expected output: max of all 8 values = 8
    let expected: Vec<f32> = vec![8.0];

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
