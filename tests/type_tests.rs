use half::{bf16, f16};
use xla::Result;

#[test]
fn test_i8_support() -> Result<()> {
    let client = xla::PjRtClient::cpu()?;
    let builder = xla::XlaBuilder::new("test_i8");
    let literal = xla::Literal::scalar(42i8);
    let constant = builder.constant_literal(&literal)?;
    let computation = constant.build()?;
    let executable = client.compile(&computation)?;
    let result = executable.execute::<xla::Literal>(&[])?;

    let output = result[0][0].to_literal_sync()?;
    assert_eq!(output.to_vec::<i8>()?, vec![42i8]);
    Ok(())
}

#[test]
fn test_i16_support() -> Result<()> {
    let client = xla::PjRtClient::cpu()?;
    let builder = xla::XlaBuilder::new("test_i16");
    let literal = xla::Literal::scalar(1234i16);
    let constant = builder.constant_literal(&literal)?;
    let computation = constant.build()?;
    let executable = client.compile(&computation)?;
    let result = executable.execute::<xla::Literal>(&[])?;

    let output = result[0][0].to_literal_sync()?;
    assert_eq!(output.to_vec::<i16>()?, vec![1234i16]);
    Ok(())
}

#[test]
fn test_i32_support() -> Result<()> {
    let client = xla::PjRtClient::cpu()?;
    let builder = xla::XlaBuilder::new("test_i32");
    let literal = xla::Literal::scalar(123456i32);
    let constant = builder.constant_literal(&literal)?;
    let computation = constant.build()?;
    let executable = client.compile(&computation)?;
    let result = executable.execute::<xla::Literal>(&[])?;

    let output = result[0][0].to_literal_sync()?;
    assert_eq!(output.to_vec::<i32>()?, vec![123456i32]);
    Ok(())
}

#[test]
fn test_i64_support() -> Result<()> {
    let client = xla::PjRtClient::cpu()?;
    let builder = xla::XlaBuilder::new("test_i64");
    let literal = xla::Literal::scalar(1234567890i64);
    let constant = builder.constant_literal(&literal)?;
    let computation = constant.build()?;
    let executable = client.compile(&computation)?;
    let result = executable.execute::<xla::Literal>(&[])?;

    let output = result[0][0].to_literal_sync()?;
    assert_eq!(output.to_vec::<i64>()?, vec![1234567890i64]);
    Ok(())
}

#[test]
fn test_u8_support() -> Result<()> {
    let client = xla::PjRtClient::cpu()?;
    let builder = xla::XlaBuilder::new("test_u8");
    let literal = xla::Literal::scalar(255u8);
    let constant = builder.constant_literal(&literal)?;
    let computation = constant.build()?;
    let executable = client.compile(&computation)?;
    let result = executable.execute::<xla::Literal>(&[])?;

    let output = result[0][0].to_literal_sync()?;
    assert_eq!(output.to_vec::<u8>()?, vec![255u8]);
    Ok(())
}

#[test]
fn test_u16_support() -> Result<()> {
    let client = xla::PjRtClient::cpu()?;
    let builder = xla::XlaBuilder::new("test_u16");
    let literal = xla::Literal::scalar(65535u16);
    let constant = builder.constant_literal(&literal)?;
    let computation = constant.build()?;
    let executable = client.compile(&computation)?;
    let result = executable.execute::<xla::Literal>(&[])?;

    let output = result[0][0].to_literal_sync()?;
    assert_eq!(output.to_vec::<u16>()?, vec![65535u16]);
    Ok(())
}

#[test]
fn test_u32_support() -> Result<()> {
    let client = xla::PjRtClient::cpu()?;
    let builder = xla::XlaBuilder::new("test_u32");
    let literal = xla::Literal::scalar(4294967295u32);
    let constant = builder.constant_literal(&literal)?;
    let computation = constant.build()?;
    let executable = client.compile(&computation)?;
    let result = executable.execute::<xla::Literal>(&[])?;

    let output = result[0][0].to_literal_sync()?;
    assert_eq!(output.to_vec::<u32>()?, vec![4294967295u32]);
    Ok(())
}

#[test]
fn test_u64_support() -> Result<()> {
    let client = xla::PjRtClient::cpu()?;
    let builder = xla::XlaBuilder::new("test_u64");
    let literal = xla::Literal::scalar(18446744073709551615u64);
    let constant = builder.constant_literal(&literal)?;
    let computation = constant.build()?;
    let executable = client.compile(&computation)?;
    let result = executable.execute::<xla::Literal>(&[])?;

    let output = result[0][0].to_literal_sync()?;
    assert_eq!(output.to_vec::<u64>()?, vec![18446744073709551615u64]);
    Ok(())
}

#[test]
fn test_f32_support() -> Result<()> {
    let client = xla::PjRtClient::cpu()?;
    let builder = xla::XlaBuilder::new("test_f32");
    let literal = xla::Literal::scalar(std::f32::consts::PI);
    let constant = builder.constant_literal(&literal)?;
    let computation = constant.build()?;
    let executable = client.compile(&computation)?;
    let result = executable.execute::<xla::Literal>(&[])?;

    let output = result[0][0].to_literal_sync()?;
    let result_vec = output.to_vec::<f32>()?;
    assert!((result_vec[0] - std::f32::consts::PI).abs() < f32::EPSILON);
    Ok(())
}

#[test]
fn test_f64_support() -> Result<()> {
    let client = xla::PjRtClient::cpu()?;
    let builder = xla::XlaBuilder::new("test_f64");
    let literal = xla::Literal::scalar(std::f64::consts::E);
    let constant = builder.constant_literal(&literal)?;
    let computation = constant.build()?;
    let executable = client.compile(&computation)?;
    let result = executable.execute::<xla::Literal>(&[])?;

    let output = result[0][0].to_literal_sync()?;
    let result_vec = output.to_vec::<f64>()?;
    assert!((result_vec[0] - std::f64::consts::E).abs() < f64::EPSILON);
    Ok(())
}

#[test]
fn test_f16_support() -> Result<()> {
    let client = xla::PjRtClient::cpu()?;
    let builder = xla::XlaBuilder::new("test_f16");
    let value = f16::from_f32(std::f32::consts::PI);
    let literal = xla::Literal::scalar(value);
    let constant = builder.constant_literal(&literal)?;
    let computation = constant.build()?;
    let executable = client.compile(&computation)?;
    let result = executable.execute::<xla::Literal>(&[])?;

    let output = result[0][0].to_literal_sync()?;
    let result_vec = output.to_vec::<f16>()?;
    assert!((result_vec[0].to_f32() - std::f32::consts::PI).abs() < 0.01); // F16 has lower precision
    Ok(())
}

#[test]
fn test_bf16_support() -> Result<()> {
    let client = xla::PjRtClient::cpu()?;
    let builder = xla::XlaBuilder::new("test_bf16");
    let value = bf16::from_f32(std::f32::consts::E);
    let literal = xla::Literal::scalar(value);
    let constant = builder.constant_literal(&literal)?;
    let computation = constant.build()?;
    let executable = client.compile(&computation)?;
    let result = executable.execute::<xla::Literal>(&[])?;

    let output = result[0][0].to_literal_sync()?;
    let result_vec = output.to_vec::<bf16>()?;
    assert!((result_vec[0].to_f32() - std::f32::consts::E).abs() < 0.01); // BF16 has lower precision
    Ok(())
}

#[test]
fn test_vec1_types() -> Result<()> {
    let client = xla::PjRtClient::cpu()?;

    // Test i8 vector
    let builder = xla::XlaBuilder::new("test_i8_vec");
    let data_i8 = vec![1i8, 2i8, 3i8, 4i8];
    let literal = xla::Literal::vec1(&data_i8);
    let constant = builder.constant_literal(&literal)?;
    let computation = constant.build()?;
    let executable = client.compile(&computation)?;
    let result = executable.execute::<xla::Literal>(&[])?;
    let output = result[0][0].to_literal_sync()?;
    assert_eq!(output.to_vec::<i8>()?, data_i8);

    // Test u8 vector
    let builder = xla::XlaBuilder::new("test_u8_vec");
    let data_u8 = vec![10u8, 20u8, 30u8, 40u8];
    let literal = xla::Literal::vec1(&data_u8);
    let constant = builder.constant_literal(&literal)?;
    let computation = constant.build()?;
    let executable = client.compile(&computation)?;
    let result = executable.execute::<xla::Literal>(&[])?;
    let output = result[0][0].to_literal_sync()?;
    assert_eq!(output.to_vec::<u8>()?, data_u8);

    // Test f16 vector
    let builder = xla::XlaBuilder::new("test_f16_vec");
    let data_f16 = vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)];
    let literal = xla::Literal::vec1(&data_f16);
    let constant = builder.constant_literal(&literal)?;
    let computation = constant.build()?;
    let executable = client.compile(&computation)?;
    let result = executable.execute::<xla::Literal>(&[])?;
    let output = result[0][0].to_literal_sync()?;
    let result_vec = output.to_vec::<f16>()?;
    for (i, &expected) in data_f16.iter().enumerate() {
        assert!((result_vec[i].to_f32() - expected.to_f32()).abs() < 0.01);
    }

    Ok(())
}
