use hodu_xla::{ArrayElement, Result};

#[test]
fn while_op() -> Result<()> {
    let client = hodu_xla::PjRtClient::cpu()?;
    let builder = hodu_xla::XlaBuilder::new("test");
    let cond = {
        let builder = hodu_xla::XlaBuilder::new("cond");
        let x = builder.parameter(0, i32::TY, &[], "x")?;
        x.le(&builder.constant_r0(10i32)?)?.build()?
    };
    let body = {
        let builder = hodu_xla::XlaBuilder::new("cond");
        let x = builder.parameter(0, i32::TY, &[], "x")?;
        (x + builder.constant_r0(1i32)?)?.build()?
    };
    let init = builder.constant_r0(0i32)?;
    let w = hodu_xla::XlaOp::while_(cond, body, init)?;
    let computation = w.build()?;
    let result = client.compile(&computation)?;
    let result = result.execute::<hodu_xla::Literal>(&[])?;
    let result = result[0][0].to_literal_sync()?;
    assert_eq!(result.element_count(), 1);
    assert_eq!(result.shape()?, hodu_xla::Shape::array::<i32>(vec![]));
    assert_eq!(result.to_vec::<i32>()?, [11]);
    Ok(())
}

#[test]
fn while_op2() -> Result<()> {
    let client = hodu_xla::PjRtClient::cpu()?;
    let builder = hodu_xla::XlaBuilder::new("test");
    let state_shape = hodu_xla::Shape::tuple(vec![
        hodu_xla::Shape::array::<i32>(vec![]),
        hodu_xla::Shape::array::<f32>(vec![2]),
    ]);
    let cond = {
        let builder = hodu_xla::XlaBuilder::new("cond");
        let x = builder.parameter_s(0, &state_shape, "x")?;
        x.get_tuple_element(0)?.le(&builder.constant_r0(10i32)?)?.build()?
    };
    let body = {
        let builder = hodu_xla::XlaBuilder::new("cond");
        let x = builder.parameter_s(0, &state_shape, "x")?;
        let x0 = (x.get_tuple_element(0)? + builder.constant_r0(1i32)?)?;
        let x1 = (x.get_tuple_element(1)? + builder.constant_r1(&[0f32, 1f32])?)?;
        let x = builder.tuple(&[x0, x1])?;
        x.build()?
    };
    let init_x0 = builder.constant_r0(0i32)?;
    let init_x1 = builder.constant_r1(&[1.2f32, 2.3f32])?;
    let init = builder.tuple(&[init_x0, init_x1])?;
    let w = hodu_xla::XlaOp::while_(cond, body, init)?;
    let computation = w.build()?;
    let result = client.compile(&computation)?;
    let result = result.execute::<hodu_xla::Literal>(&[])?;
    assert_eq!(result[0].len(), 2);
    let result0 = result[0][0].to_literal_sync()?;
    let result1 = result[0][1].to_literal_sync()?;
    assert_eq!(result0.element_count(), 1);
    assert_eq!(result0.shape()?, hodu_xla::Shape::array::<i32>(vec![]));
    assert_eq!(result0.to_vec::<i32>()?, [11]);
    assert_eq!(result1.element_count(), 2);
    assert_eq!(result1.shape()?, hodu_xla::Shape::array::<f32>(vec![2]));
    assert_eq!(result1.to_vec::<f32>()?, [1.2, 13.3]);
    Ok(())
}
