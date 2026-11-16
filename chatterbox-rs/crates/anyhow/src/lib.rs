use std::fmt;

pub struct Error {
    inner: Box<dyn std::error::Error + Send + Sync>,
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.inner)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.inner)
    }
}

#[derive(Debug)]
struct StringError(String);

impl fmt::Display for StringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for StringError {}

impl Error {
    pub fn msg<M: fmt::Display>(message: M) -> Self {
        Self {
            inner: Box::new(StringError(message.to_string())),
        }
    }
}

impl<E: std::error::Error + Send + Sync + 'static> From<E> for Error {
    fn from(err: E) -> Self {
        Self {
            inner: Box::new(err),
        }
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[macro_export]
macro_rules! anyhow {
    ($msg:literal $(,)?) => {
        $crate::Error::msg($msg)
    };
    ($err:expr $(,)?) => {
        $crate::Error::msg($err)
    };
    ($fmt:expr, $($arg:tt)+) => {
        $crate::Error::msg(format!($fmt, $($arg)+))
    };
}

#[macro_export]
macro_rules! bail {
    ($msg:literal $(,)?) => {
        return Err($crate::anyhow!($msg))
    };
    ($err:expr $(,)?) => {
        return Err($crate::anyhow!($err))
    };
    ($fmt:expr, $($arg:tt)+) => {
        return Err($crate::anyhow!($fmt, $($arg)+))
    };
}
