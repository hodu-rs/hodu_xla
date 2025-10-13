use hodu_xla::Result;

#[test]
fn tuple_op() -> Result<()> {
    let client = hodu_xla::PjRtClient::cpu()?;
    let builder = hodu_xla::XlaBuilder::new("test");
    let cst42 = builder.constant_r0(42f32)?;
    let cst43 = builder.constant_r1c(43f32, 2)?;
    let computation = builder.tuple(&[cst42, cst43])?.build()?;
    let result = client.compile(&computation)?;
    let result = result.execute::<hodu_xla::Literal>(&[])?;

    assert_eq!(result[0].len(), 2);
    let result0 = result[0][0].to_literal_sync()?;
    let result1 = result[0][1].to_literal_sync()?;

    assert_eq!(result0.array_shape()?, hodu_xla::ArrayShape::new::<f32>(vec![]));
    assert_eq!(result1.array_shape()?, hodu_xla::ArrayShape::new::<f32>(vec![2]));
    assert_eq!(result1.to_vec::<f32>()?, vec![43f32, 43f32]);
    Ok(())
}
