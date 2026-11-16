use std::fs::File;
use std::io::{self, BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SampleFormat {
    Float,
    Int,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct WavSpec {
    pub channels: u16,
    pub sample_rate: u32,
    pub bits_per_sample: u16,
    pub sample_format: SampleFormat,
}

#[derive(Debug)]
pub enum Error {
    IoError(io::Error),
    FormatError(&'static str),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::IoError(e) => write!(f, "IO error: {}", e),
            Error::FormatError(msg) => write!(f, "Format error: {}", msg),
        }
    }
}

impl std::error::Error for Error {}

impl From<io::Error> for Error {
    fn from(err: io::Error) -> Self {
        Error::IoError(err)
    }
}

pub type Result<T> = std::result::Result<T, Error>;

pub struct WavWriter<W: Write + Seek> {
    writer: W,
    spec: WavSpec,
    bytes_written: u32,
    header_written: bool,
}

impl WavWriter<BufWriter<File>> {
    pub fn create<P: AsRef<Path>>(filename: P, spec: WavSpec) -> Result<Self> {
        let file = File::create(filename)?;
        let writer = BufWriter::new(file);
        let mut wav_writer = WavWriter {
            writer,
            spec,
            bytes_written: 0,
            header_written: false,
        };
        wav_writer.write_header()?;
        Ok(wav_writer)
    }
}

impl<W: Write + Seek> WavWriter<W> {
    fn write_header(&mut self) -> Result<()> {
        let w = &mut self.writer;
        w.write_all(b"RIFF")?;
        w.write_all(&0u32.to_le_bytes())?; // Placeholder for riff chunk size
        w.write_all(b"WAVE")?;

        // fmt chunk
        w.write_all(b"fmt ")?;
        w.write_all(&16u32.to_le_bytes())?; // Chunk size 16 for PCM/Float
        let format_tag: u16 = match self.spec.sample_format {
            SampleFormat::Float => 3, // IEEE Float
            SampleFormat::Int => 1,   // PCM
        };
        w.write_all(&format_tag.to_le_bytes())?;
        w.write_all(&self.spec.channels.to_le_bytes())?;
        w.write_all(&self.spec.sample_rate.to_le_bytes())?;
        let byte_rate = self.spec.sample_rate
            * self.spec.channels as u32
            * (self.spec.bits_per_sample as u32 / 8);
        w.write_all(&byte_rate.to_le_bytes())?;
        let block_align = self.spec.channels * (self.spec.bits_per_sample / 8);
        w.write_all(&block_align.to_le_bytes())?;
        w.write_all(&self.spec.bits_per_sample.to_le_bytes())?;

        // data chunk header
        w.write_all(b"data")?;
        w.write_all(&0u32.to_le_bytes())?; // Placeholder for data size
        self.header_written = true;
        Ok(())
    }

    pub fn write_sample(&mut self, sample: f32) -> Result<()> {
        match self.spec.sample_format {
            SampleFormat::Float => {
                let bytes = sample.to_le_bytes();
                self.writer.write_all(&bytes)?;
                self.bytes_written += 4;
            }
            SampleFormat::Int => {
                let clamped = sample.max(-1.0).min(1.0);
                let pcm16 = (clamped * 32767.0) as i16;
                let bytes = pcm16.to_le_bytes();
                self.writer.write_all(&bytes)?;
                self.bytes_written += 2;
            }
        }
        Ok(())
    }

    pub fn finalize(mut self) -> Result<()> {
        self.writer.flush()?;
        // Update data chunk size (offset 40)
        self.writer.seek(SeekFrom::Start(40))?;
        self.writer.write_all(&self.bytes_written.to_le_bytes())?;

        // Update RIFF chunk size (offset 4: total file size - 8)
        let total_size = 36 + self.bytes_written;
        self.writer.seek(SeekFrom::Start(4))?;
        self.writer.write_all(&total_size.to_le_bytes())?;
        self.writer.flush()?;
        Ok(())
    }
}
