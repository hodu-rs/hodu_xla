use hodu_xla::{ArrayElement, Result};

#[test]
fn scatter_add_1d() -> Result<()> {
    let client = hodu_xla::PjRtClient::cpu()?;
    let builder = hodu_xla::XlaBuilder::new("scatter_add_1d");

    // Create operand: [0, 0, 0, 0, 0]
    let operand = builder.constant_r1(&[0f32, 0., 0., 0., 0.])?;

    // Scatter indices: [0, 2, 4]
    let scatter_indices = builder.constant_r1(&[0i32, 2, 4])?;

    // Updates: [1, 2, 3]
    let updates = builder.constant_r1(&[1f32, 2., 3.])?;

    // Create update computation (addition)
    let comp_builder = hodu_xla::XlaBuilder::new("add");
    let p0 = comp_builder.parameter(0, f32::TY, &[], "p0")?;
    let p1 = comp_builder.parameter(1, f32::TY, &[], "p1")?;
    let add_comp = (p0 + p1)?.build()?;

    // For 1D scatter, we need to properly shape indices and updates
    // Indices: [3] -> [3, 1] (each index is a 1-element vector)
    // Updates: [3] -> [3, 1] (must match the expected shape)
    let scatter_indices = scatter_indices.reshape(&[3, 1])?;
    let updates = updates.reshape(&[3, 1])?;

    // Scatter with:
    // - update_window_dims: [1] (the last dimension of updates is the window)
    // - inserted_window_dims: [] (no inserted dimensions)
    // - scatter_dims_to_operand_dims: [0] (index dimension 0 maps to operand dimension 0)
    // - index_vector_dim: 1 (indices are shaped [3, 1], so index_vector_dim is 1)
    let result = operand.scatter(
        &scatter_indices,
        &updates,
        add_comp,
        &[1],    // update_window_dims
        &[],     // inserted_window_dims
        &[0],    // scatter_dims_to_operand_dims
        Some(1), // index_vector_dim
        false,   // indices_are_sorted
        false,   // unique_indices
    )?;

    let computation = result.build()?;
    let result = client.compile(&computation)?;
    let result = result.execute::<hodu_xla::Literal>(&[])?;
    let result = result[0][0].to_literal_sync()?;

    assert_eq!(result.to_vec::<f32>()?, [1., 0., 2., 0., 3.]);
    Ok(())
}

#[test]
fn scatter_update_2d() -> Result<()> {
    let client = hodu_xla::PjRtClient::cpu()?;
    let builder = hodu_xla::XlaBuilder::new("scatter_update_2d");

    // Create operand: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    let operand_data = vec![1f32, 2., 3., 4., 5., 6., 7., 8., 9.];
    let operand = builder.parameter(0, f32::TY, &[3, 3], "operand")?;

    // Scatter indices: [[0, 0], [1, 1], [2, 2]]
    let indices_data = vec![0i32, 0, 1, 1, 2, 2];
    let scatter_indices = builder.parameter(1, i32::TY, &[3, 2], "indices")?;

    // Updates: [10, 20, 30]
    let updates_data = vec![10f32, 20., 30.];
    let updates = builder.parameter(2, f32::TY, &[3], "updates")?;

    // Create update computation (replace)
    let comp_builder = hodu_xla::XlaBuilder::new("replace");
    let _p0 = comp_builder.parameter(0, f32::TY, &[], "p0")?;
    let p1 = comp_builder.parameter(1, f32::TY, &[], "p1")?;
    let replace_comp = p1.build()?; // Just return the update value

    // Scatter with:
    // - update_window_dims: [] (scalar updates)
    // - inserted_window_dims: [0, 1] (insert into both dimensions)
    // - scatter_dims_to_operand_dims: [0, 1] (map both index components)
    // - index_vector_dim: 1 (the last dimension of indices)
    let result = operand.scatter(
        &scatter_indices,
        &updates,
        replace_comp,
        &[],     // update_window_dims
        &[0, 1], // inserted_window_dims
        &[0, 1], // scatter_dims_to_operand_dims
        Some(1), // index_vector_dim
        false,   // indices_are_sorted
        false,   // unique_indices
    )?;

    let computation = result.build()?;
    let compiled = client.compile(&computation)?;

    let operand_literal = {
        let bytes = unsafe { std::slice::from_raw_parts(operand_data.as_ptr() as *const u8, operand_data.len() * 4) };
        hodu_xla::Literal::create_from_shape_and_untyped_data(f32::TY, &[3, 3], bytes)?
    };
    let indices_literal = {
        let bytes = unsafe { std::slice::from_raw_parts(indices_data.as_ptr() as *const u8, indices_data.len() * 4) };
        hodu_xla::Literal::create_from_shape_and_untyped_data(i32::TY, &[3, 2], bytes)?
    };
    let updates_literal = {
        let bytes = unsafe { std::slice::from_raw_parts(updates_data.as_ptr() as *const u8, updates_data.len() * 4) };
        hodu_xla::Literal::create_from_shape_and_untyped_data(f32::TY, &[3], bytes)?
    };

    let result = compiled.execute::<hodu_xla::Literal>(&[operand_literal, indices_literal, updates_literal])?;
    let result = result[0][0].to_literal_sync()?;

    // Expected: [[10, 2, 3], [4, 20, 6], [7, 8, 30]]
    assert_eq!(result.to_vec::<f32>()?, [10., 2., 3., 4., 20., 6., 7., 8., 30.]);
    Ok(())
}

#[test]
fn scatter_max_1d() -> Result<()> {
    let client = hodu_xla::PjRtClient::cpu()?;
    let builder = hodu_xla::XlaBuilder::new("scatter_max_1d");

    // Create operand: [1, 2, 3, 4, 5]
    let operand = builder.constant_r1(&[1f32, 2., 3., 4., 5.])?;

    // Scatter indices: [0, 2, 4]
    let scatter_indices = builder.constant_r1(&[0i32, 2, 4])?;

    // Updates: [10, 1, 8] - note 1 is less than 3 at index 2
    let updates = builder.constant_r1(&[10f32, 1., 8.])?;

    // Create update computation (max)
    let comp_builder = hodu_xla::XlaBuilder::new("max");
    let p0 = comp_builder.parameter(0, f32::TY, &[], "p0")?;
    let p1 = comp_builder.parameter(1, f32::TY, &[], "p1")?;
    let max_comp = p0.max(&p1)?.build()?;

    // Reshape indices and updates to match scatter requirements
    let scatter_indices = scatter_indices.reshape(&[3, 1])?;
    let updates = updates.reshape(&[3, 1])?;

    let result = operand.scatter(
        &scatter_indices,
        &updates,
        max_comp,
        &[1],    // update_window_dims
        &[],     // inserted_window_dims
        &[0],    // scatter_dims_to_operand_dims
        Some(1), // index_vector_dim
        false,   // indices_are_sorted
        false,   // unique_indices
    )?;

    let computation = result.build()?;
    let result = client.compile(&computation)?;
    let result = result.execute::<hodu_xla::Literal>(&[])?;
    let result = result[0][0].to_literal_sync()?;

    // Expected: [10, 2, 3, 4, 8] - max(1,10)=10, max(3,1)=3, max(5,8)=8
    assert_eq!(result.to_vec::<f32>()?, [10., 2., 3., 4., 8.]);
    Ok(())
}

#[test]
fn select_and_scatter_max_pool_grad() -> Result<()> {
    let client = hodu_xla::PjRtClient::cpu()?;
    let builder = hodu_xla::XlaBuilder::new("select_and_scatter");

    // Simulate a max pooling gradient operation
    // Input: [[1, 2], [3, 4]]
    let operand = builder.constant_r1(&[1f32, 2., 3., 4.])?;
    let operand = operand.reshape(&[1, 2, 2, 1])?; // NHWC format

    // Gradient from output: [[1]]
    let source = builder.constant_r1(&[1f32])?;
    let source = source.reshape(&[1, 1, 1, 1])?;

    // Init value: 0
    let init_value = builder.constant_r0(0f32)?;

    // Create select computation (>= for max pooling)
    let select_builder = hodu_xla::XlaBuilder::new("select");
    let lhs = select_builder.parameter(0, f32::TY, &[], "lhs")?;
    let rhs = select_builder.parameter(1, f32::TY, &[], "rhs")?;
    let select_comp = lhs.ge(&rhs)?.build()?;

    // Create scatter computation (addition for gradient accumulation)
    let scatter_builder = hodu_xla::XlaBuilder::new("scatter");
    let lhs = scatter_builder.parameter(0, f32::TY, &[], "lhs")?;
    let rhs = scatter_builder.parameter(1, f32::TY, &[], "rhs")?;
    let scatter_comp = (lhs + rhs)?.build()?;

    // Window dimensions: [2, 2] (pool over 2x2 window)
    // Window strides: [2, 2]
    // Padding: [(0, 0), (0, 0), (0, 0), (0, 0)] for NHWC
    let result = operand.select_and_scatter(
        select_comp,
        &[1, 2, 2, 1],                     // window_dimensions (NHWC)
        &[1, 2, 2, 1],                     // window_strides (NHWC)
        &[(0, 0), (0, 0), (0, 0), (0, 0)], // padding
        &source,
        &init_value,
        scatter_comp,
    )?;

    let computation = result.build()?;
    let result = client.compile(&computation)?;
    let result = result.execute::<hodu_xla::Literal>(&[])?;
    let result = result[0][0].to_literal_sync()?;

    // The gradient should go to the maximum element (4 at position [1, 1])
    let output = result.to_vec::<f32>()?;
    assert_eq!(output.len(), 4);
    // The last element should get the gradient since 4 is the max
    assert_eq!(output[3], 1.0);
    Ok(())
}

#[test]
fn scatter_duplicate_indices() -> Result<()> {
    let client = hodu_xla::PjRtClient::cpu()?;
    let builder = hodu_xla::XlaBuilder::new("scatter_duplicate_indices");

    // Test scatter with duplicate indices
    // Create operand: [0, 0, 0]
    let operand = builder.constant_r1(&[0f32, 0., 0.])?;

    // Scatter indices with duplicates: [1, 1, 2]
    let scatter_indices = builder.constant_r1(&[1i32, 1, 2])?;

    // Updates: [1, 2, 3]
    let updates = builder.constant_r1(&[1f32, 2., 3.])?;

    // Create update computation (addition)
    let comp_builder = hodu_xla::XlaBuilder::new("add");
    let p0 = comp_builder.parameter(0, f32::TY, &[], "p0")?;
    let p1 = comp_builder.parameter(1, f32::TY, &[], "p1")?;
    let add_comp = (p0 + p1)?.build()?;

    // Reshape indices and updates to match scatter requirements
    let scatter_indices = scatter_indices.reshape(&[3, 1])?;
    let updates = updates.reshape(&[3, 1])?;

    // Scatter - with duplicate indices, both updates should be applied
    let result = operand.scatter(
        &scatter_indices,
        &updates,
        add_comp,
        &[1],    // update_window_dims
        &[],     // inserted_window_dims
        &[0],    // scatter_dims_to_operand_dims
        Some(1), // index_vector_dim
        false,   // indices_are_sorted
        false,   // unique_indices (explicitly false)
    )?;

    let computation = result.build()?;
    let result = client.compile(&computation)?;
    let result = result.execute::<hodu_xla::Literal>(&[])?;
    let result = result[0][0].to_literal_sync()?;

    // Expected: [0, 3, 3] - index 1 gets 1+2=3, index 2 gets 3
    assert_eq!(result.to_vec::<f32>()?, [0., 3., 3.]);
    Ok(())
}
