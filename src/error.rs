/// Main library error type.
#[derive(Debug)]
pub enum Error {
    /// Incorrect number of elements.
    WrongElementCount {
        dims: Vec<usize>,
        element_count: usize,
    },

    /// Error from the xla C++ library.
    XlaError {
        msg: String,
        backtrace: String,
    },

    UnexpectedElementType(i32),

    UnexpectedNumberOfDims {
        expected: usize,
        got: usize,
        dims: Vec<i64>,
    },

    NotAnElementType {
        got: crate::PrimitiveType,
    },

    NotAnArray {
        expected: Option<usize>,
        got: crate::Shape,
    },

    UnsupportedShape {
        shape: crate::Shape,
    },

    UnexpectedNumberOfElemsInTuple {
        expected: usize,
        got: usize,
    },

    ElementTypeMismatch {
        on_device: crate::ElementType,
        on_host: crate::ElementType,
    },

    UnsupportedElementType {
        ty: crate::PrimitiveType,
        op: &'static str,
    },

    TargetBufferIsTooLarge {
        offset: usize,
        shape: crate::ArrayShape,
        buffer_len: usize,
    },

    BinaryBufferIsTooLarge {
        element_count: usize,
        buffer_len: usize,
    },

    EmptyLiteral,

    IndexOutOfBounds {
        index: i64,
        rank: usize,
    },

    Npy(String),

    /// I/O error.
    Io(std::io::Error),

    /// Zip file format error.
    Zip(zip::result::ZipError),

    /// Integer parse error.
    ParseInt(std::num::ParseIntError),

    CannotCreateLiteralWithData {
        data_len_in_bytes: usize,
        ty: crate::PrimitiveType,
        dims: Vec<usize>,
    },

    MatMulIncorrectDims {
        lhs_dims: Vec<i64>,
        rhs_dims: Vec<i64>,
        msg: &'static str,
    },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::WrongElementCount { dims, element_count } => {
                write!(f, "wrong element count {} for dims {:?}", element_count, dims)
            },
            Error::XlaError { msg, backtrace } => {
                write!(f, "xla error {}\n{}", msg, backtrace)
            },
            Error::UnexpectedElementType(ty) => {
                write!(f, "unexpected element type {}", ty)
            },
            Error::UnexpectedNumberOfDims { expected, got, dims } => {
                write!(
                    f,
                    "unexpected number of dimensions, expected: {}, got: {} ({:?})",
                    expected, got, dims
                )
            },
            Error::NotAnElementType { got } => {
                write!(f, "not an element type, got: {:?}", got)
            },
            Error::NotAnArray { expected, got } => {
                write!(f, "not an array, expected: {:?}, got: {:?}", expected, got)
            },
            Error::UnsupportedShape { shape } => {
                write!(f, "cannot handle unsupported shapes {:?}", shape)
            },
            Error::UnexpectedNumberOfElemsInTuple { expected, got } => {
                write!(
                    f,
                    "unexpected number of tuple elements, expected: {}, got: {}",
                    expected, got
                )
            },
            Error::ElementTypeMismatch { on_device, on_host } => {
                write!(
                    f,
                    "element type mismatch, on-device: {:?}, on-host: {:?}",
                    on_device, on_host
                )
            },
            Error::UnsupportedElementType { ty, op } => {
                write!(f, "unsupported element type for {}: {:?}", op, ty)
            },
            Error::TargetBufferIsTooLarge {
                offset,
                shape,
                buffer_len,
            } => {
                write!(
                    f,
                    "target buffer is too large, offset {}, shape {:?}, buffer_len: {}",
                    offset, shape, buffer_len
                )
            },
            Error::BinaryBufferIsTooLarge {
                element_count,
                buffer_len,
            } => {
                write!(
                    f,
                    "binary buffer is too large, element count {}, buffer_len: {}",
                    element_count, buffer_len
                )
            },
            Error::EmptyLiteral => write!(f, "empty literal"),
            Error::IndexOutOfBounds { index, rank } => {
                write!(f, "index out of bounds {}, rank {}", index, rank)
            },
            Error::Npy(msg) => write!(f, "npy/npz error {}", msg),
            Error::Io(err) => write!(f, "I/O error: {}", err),
            Error::Zip(err) => write!(f, "Zip error: {}", err),
            Error::ParseInt(err) => write!(f, "Parse int error: {}", err),
            Error::CannotCreateLiteralWithData {
                data_len_in_bytes,
                ty,
                dims,
            } => {
                write!(
                    f,
                    "cannot create literal with shape {:?} {:?} from bytes data with len {}",
                    ty, dims, data_len_in_bytes
                )
            },
            Error::MatMulIncorrectDims {
                lhs_dims,
                rhs_dims,
                msg,
            } => {
                write!(
                    f,
                    "invalid dimensions in matmul, lhs: {:?}, rhs: {:?}, {}",
                    lhs_dims, rhs_dims, msg
                )
            },
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(err) => Some(err),
            Error::Zip(err) => Some(err),
            Error::ParseInt(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::Io(err)
    }
}

impl From<zip::result::ZipError> for Error {
    fn from(err: zip::result::ZipError) -> Self {
        Error::Zip(err)
    }
}

impl From<std::num::ParseIntError> for Error {
    fn from(err: std::num::ParseIntError) -> Self {
        Error::ParseInt(err)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
