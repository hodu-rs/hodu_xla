use hodu_xla::{ArrayElement, Result};

/// Helper function to create a literal from typed data
fn create_literal_f32(dims: &[usize], data: &[f32]) -> Result<hodu_xla::Literal> {
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    hodu_xla::Literal::create_from_shape_and_untyped_data(f32::TY, dims, bytes)
}

/// Test a simple 2D convolution with NCHW layout (batch, channels, height, width)
#[test]
fn conv2d_basic() -> Result<()> {
    let client = hodu_xla::PjRtClient::cpu()?;
    let builder = hodu_xla::XlaBuilder::new("conv2d_test");

    // Input: 1x1x4x4 (batch=1, channels=1, height=4, width=4)
    let input = builder.parameter(0, f32::TY, &[1, 1, 4, 4], "input")?;

    // Kernel: 1x1x2x2 (out_channels=1, in_channels=1, height=2, width=2)
    let kernel = builder.parameter(1, f32::TY, &[1, 1, 2, 2], "kernel")?;

    // Perform convolution
    // NCHW format: input_batch=0, input_feature=1, input_spatial=[2,3]
    // OIHW format: kernel_out=0, kernel_in=1, kernel_spatial=[2,3]
    let result = input.conv_general_dilated(
        &kernel,
        &[1, 1],           // window_strides
        &[(0, 0), (0, 0)], // padding
        &[1, 1],           // lhs_dilation (no dilation)
        &[1, 1],           // rhs_dilation (no dilation)
        0,                 // input_batch_dimension
        1,                 // input_feature_dimension
        &[2, 3],           // input_spatial_dimensions
        1,                 // kernel_input_feature_dimension
        0,                 // kernel_output_feature_dimension
        &[2, 3],           // kernel_spatial_dimensions
        1,                 // feature_group_count
        1,                 // batch_group_count
    )?;

    let computation = result.build()?;
    let executable = client.compile(&computation)?;

    // Create input data: 4x4 matrix with values 1-16
    #[rustfmt::skip]
    let input_data: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ];
    let input_literal = create_literal_f32(&[1, 1, 4, 4], &input_data)?;

    // Create kernel: 2x2 identity-like kernel
    #[rustfmt::skip]
    let kernel_data: Vec<f32> = vec![
        1.0, 0.0,
        0.0, 1.0,
    ];
    let kernel_literal = create_literal_f32(&[1, 1, 2, 2], &kernel_data)?;

    let result = executable.execute::<hodu_xla::Literal>(&[input_literal, kernel_literal])?;
    let result = result[0][0].to_literal_sync()?;

    // Output should be 1x1x3x3 (stride=1, no padding)
    assert_eq!(result.array_shape()?.dims(), [1, 1, 3, 3]);

    let output = result.to_vec::<f32>()?;
    // Expected output:
    // [1+6, 2+7, 3+8]
    // [5+10, 6+11, 7+12]
    // [9+14, 10+15, 11+16]
    #[rustfmt::skip]
    let expected: Vec<f32> = vec![
        7.0, 9.0, 11.0,
        15.0, 17.0, 19.0,
        23.0, 25.0, 27.0,
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

/// Test convolution with stride
#[test]
fn conv2d_with_stride() -> Result<()> {
    let client = hodu_xla::PjRtClient::cpu()?;
    let builder = hodu_xla::XlaBuilder::new("conv2d_stride_test");

    // Input: 1x1x5x5
    let input = builder.parameter(0, f32::TY, &[1, 1, 5, 5], "input")?;
    // Kernel: 1x1x3x3
    let kernel = builder.parameter(1, f32::TY, &[1, 1, 3, 3], "kernel")?;

    // Convolution with stride 2
    let result = input.conv_general_dilated(
        &kernel,
        &[2, 2],           // window_strides (stride=2)
        &[(0, 0), (0, 0)], // padding
        &[1, 1],           // lhs_dilation
        &[1, 1],           // rhs_dilation
        0,                 // input_batch_dimension
        1,                 // input_feature_dimension
        &[2, 3],           // input_spatial_dimensions
        1,                 // kernel_input_feature_dimension
        0,                 // kernel_output_feature_dimension
        &[2, 3],           // kernel_spatial_dimensions
        1,                 // feature_group_count
        1,                 // batch_group_count
    )?;

    let computation = result.build()?;
    let executable = client.compile(&computation)?;

    // Create simple input
    let input_data: Vec<f32> = (1..=25).map(|x| x as f32).collect();
    let input_literal = create_literal_f32(&[1, 1, 5, 5], &input_data)?;

    // Simple averaging kernel
    let kernel_data: Vec<f32> = vec![1.0; 9];
    let kernel_literal = create_literal_f32(&[1, 1, 3, 3], &kernel_data)?;

    let result = executable.execute::<hodu_xla::Literal>(&[input_literal, kernel_literal])?;
    let result = result[0][0].to_literal_sync()?;

    // With stride 2, output should be 1x1x2x2
    assert_eq!(result.array_shape()?.dims(), [1, 1, 2, 2]);

    Ok(())
}

/// Test convolution with padding
#[test]
fn conv2d_with_padding() -> Result<()> {
    let client = hodu_xla::PjRtClient::cpu()?;
    let builder = hodu_xla::XlaBuilder::new("conv2d_padding_test");

    // Input: 1x1x4x4
    let input = builder.parameter(0, f32::TY, &[1, 1, 4, 4], "input")?;
    // Kernel: 1x1x3x3
    let kernel = builder.parameter(1, f32::TY, &[1, 1, 3, 3], "kernel")?;

    // Convolution with padding to maintain size (same padding)
    let result = input.conv_general_dilated(
        &kernel,
        &[1, 1],           // window_strides
        &[(1, 1), (1, 1)], // padding (1 on each side)
        &[1, 1],           // lhs_dilation
        &[1, 1],           // rhs_dilation
        0,                 // input_batch_dimension
        1,                 // input_feature_dimension
        &[2, 3],           // input_spatial_dimensions
        1,                 // kernel_input_feature_dimension
        0,                 // kernel_output_feature_dimension
        &[2, 3],           // kernel_spatial_dimensions
        1,                 // feature_group_count
        1,                 // batch_group_count
    )?;

    let computation = result.build()?;
    let executable = client.compile(&computation)?;

    let input_data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let input_literal = create_literal_f32(&[1, 1, 4, 4], &input_data)?;

    let kernel_data: Vec<f32> = vec![1.0; 9];
    let kernel_literal = create_literal_f32(&[1, 1, 3, 3], &kernel_data)?;

    let result = executable.execute::<hodu_xla::Literal>(&[input_literal, kernel_literal])?;
    let result = result[0][0].to_literal_sync()?;

    // With padding (1,1) and kernel 3x3, output should be same size: 1x1x4x4
    assert_eq!(result.array_shape()?.dims(), [1, 1, 4, 4]);

    Ok(())
}

/// Test dilated convolution (atrous convolution)
#[test]
fn conv2d_with_dilation() -> Result<()> {
    let client = hodu_xla::PjRtClient::cpu()?;
    let builder = hodu_xla::XlaBuilder::new("conv2d_dilation_test");

    // Input: 1x1x5x5
    let input = builder.parameter(0, f32::TY, &[1, 1, 5, 5], "input")?;
    // Kernel: 1x1x2x2
    let kernel = builder.parameter(1, f32::TY, &[1, 1, 2, 2], "kernel")?;

    // Convolution with dilation rate 2 (effective kernel becomes 3x3 with gaps)
    let result = input.conv_general_dilated(
        &kernel,
        &[1, 1],           // window_strides
        &[(0, 0), (0, 0)], // padding
        &[1, 1],           // lhs_dilation
        &[2, 2],           // rhs_dilation (dilated kernel)
        0,                 // input_batch_dimension
        1,                 // input_feature_dimension
        &[2, 3],           // input_spatial_dimensions
        1,                 // kernel_input_feature_dimension
        0,                 // kernel_output_feature_dimension
        &[2, 3],           // kernel_spatial_dimensions
        1,                 // feature_group_count
        1,                 // batch_group_count
    )?;

    let computation = result.build()?;
    let executable = client.compile(&computation)?;

    let input_data: Vec<f32> = (1..=25).map(|x| x as f32).collect();
    let input_literal = create_literal_f32(&[1, 1, 5, 5], &input_data)?;

    let kernel_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
    let kernel_literal = create_literal_f32(&[1, 1, 2, 2], &kernel_data)?;

    let result = executable.execute::<hodu_xla::Literal>(&[input_literal, kernel_literal])?;
    let result = result[0][0].to_literal_sync()?;

    // With dilation 2, a 2x2 kernel becomes effectively 3x3, so output is 1x1x3x3
    assert_eq!(result.array_shape()?.dims(), [1, 1, 3, 3]);

    Ok(())
}

/// Test multi-channel convolution
#[test]
fn conv2d_multi_channel() -> Result<()> {
    let client = hodu_xla::PjRtClient::cpu()?;
    let builder = hodu_xla::XlaBuilder::new("conv2d_multichannel_test");

    // Input: 1x3x4x4 (batch=1, in_channels=3, height=4, width=4)
    let input = builder.parameter(0, f32::TY, &[1, 3, 4, 4], "input")?;

    // Kernel: 2x3x2x2 (out_channels=2, in_channels=3, height=2, width=2)
    let kernel = builder.parameter(1, f32::TY, &[2, 3, 2, 2], "kernel")?;

    let result = input.conv_general_dilated(
        &kernel,
        &[1, 1],           // window_strides
        &[(0, 0), (0, 0)], // padding
        &[1, 1],           // lhs_dilation
        &[1, 1],           // rhs_dilation
        0,                 // input_batch_dimension
        1,                 // input_feature_dimension
        &[2, 3],           // input_spatial_dimensions
        1,                 // kernel_input_feature_dimension
        0,                 // kernel_output_feature_dimension
        &[2, 3],           // kernel_spatial_dimensions
        1,                 // feature_group_count
        1,                 // batch_group_count
    )?;

    let computation = result.build()?;
    let executable = client.compile(&computation)?;

    // Create input with 3 channels
    let input_data: Vec<f32> = (1..=48).map(|x| x as f32).collect();
    let input_literal = create_literal_f32(&[1, 3, 4, 4], &input_data)?;

    // Create kernel with 2 output channels and 3 input channels
    let kernel_data: Vec<f32> = vec![1.0; 24]; // 2*3*2*2 = 24
    let kernel_literal = create_literal_f32(&[2, 3, 2, 2], &kernel_data)?;

    let result = executable.execute::<hodu_xla::Literal>(&[input_literal, kernel_literal])?;
    let result = result[0][0].to_literal_sync()?;

    // Output should be 1x2x3x3 (batch=1, out_channels=2, height=3, width=3)
    assert_eq!(result.array_shape()?.dims(), [1, 2, 3, 3]);

    Ok(())
}
