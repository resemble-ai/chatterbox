use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Clone, Debug)]
pub struct AddedToken {
    pub content: String,
    pub special: bool,
}

impl AddedToken {
    pub fn from<S: Into<String>>(content: S, special: bool) -> Self {
        Self {
            content: content.into(),
            special,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Encoding {
    ids: Vec<u32>,
}

impl Encoding {
    pub fn get_ids(&self) -> &[u32] {
        &self.ids
    }
}

pub mod models {
    pub mod wordpiece {
        use std::collections::HashMap;
        use std::fs::File;
        use std::io::{BufRead, BufReader};
        use std::path::Path;

        #[derive(Default)]
        pub struct WordPieceBuilder {
            vocab: HashMap<String, u32>,
            unk_token: String,
        }

        impl WordPieceBuilder {
            pub fn new() -> Self {
                Self {
                    vocab: HashMap::new(),
                    unk_token: "[UNK]".to_string(),
                }
            }

            pub fn vocab(mut self, vocab: HashMap<String, u32>) -> Self {
                self.vocab = vocab;
                self
            }

            pub fn files<P: AsRef<Path>>(mut self, path: P) -> Result<Self, String> {
                let file = File::open(path).map_err(|e| e.to_string())?;
                let reader = BufReader::new(file);
                for (idx, line) in reader.lines().enumerate() {
                    let line = line.map_err(|e| e.to_string())?;
                    self.vocab.insert(line.trim().to_string(), idx as u32);
                }
                Ok(self)
            }

            pub fn unk_token(mut self, token: String) -> Self {
                self.unk_token = token;
                self
            }

            pub fn build(self) -> Result<WordPiece, String> {
                Ok(WordPiece {
                    vocab: self.vocab,
                    unk_token: self.unk_token,
                })
            }
        }

        pub struct WordPiece {
            pub vocab: HashMap<String, u32>,
            pub unk_token: String,
        }

        impl WordPiece {
            pub fn builder() -> WordPieceBuilder {
                WordPieceBuilder::new()
            }
        }
    }
}

pub struct Tokenizer {
    vocab: HashMap<String, u32>,
    id_to_token: HashMap<u32, String>,
}

impl Tokenizer {
    pub fn new(model: models::wordpiece::WordPiece) -> Self {
        let mut id_to_token = HashMap::new();
        for (k, v) in &model.vocab {
            id_to_token.insert(*v, k.clone());
        }
        Self {
            vocab: model.vocab,
            id_to_token,
        }
    }

    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| e.to_string())?;
        let reader = BufReader::new(file);
        let mut vocab = HashMap::new();
        let mut id_to_token = HashMap::new();
        for (idx, line) in reader.lines().enumerate() {
            let line = line.map_err(|e| e.to_string())?;
            let token = line.trim().to_string();
            vocab.insert(token.clone(), idx as u32);
            id_to_token.insert(idx as u32, token);
        }
        Ok(Self { vocab, id_to_token })
    }

    pub fn get_vocab(&self, _with_added: bool) -> HashMap<String, u32> {
        self.vocab.clone()
    }

    pub fn add_special_tokens(&mut self, tokens: &[AddedToken]) -> usize {
        let mut added = 0;
        for token in tokens {
            if !self.vocab.contains_key(&token.content) {
                let id = self.vocab.len() as u32;
                self.vocab.insert(token.content.clone(), id);
                self.id_to_token.insert(id, token.content.clone());
                added += 1;
            }
        }
        added
    }

    pub fn encode<S: AsRef<str>>(&self, text: S, _add_special_tokens: bool) -> Result<Encoding, String> {
        let text_ref = text.as_ref();
        let mut ids = Vec::new();
        let unk_id = *self.vocab.get("[UNK]").unwrap_or(&0);

        for word in text_ref.split_whitespace() {
            if word.is_empty() {
                continue;
            }

            let mut start = 0;
            let chars: Vec<char> = word.chars().collect();
            let mut matched_any = false;

            while start < chars.len() {
                let mut end = chars.len();
                let mut found = false;
                while start < end {
                    let sub: String = chars[start..end].iter().collect();
                    let prefix = if start > 0 {
                        format!("##{}", sub)
                    } else {
                        sub.clone()
                    };

                    if let Some(&id) = self.vocab.get(&prefix) {
                        ids.push(id);
                        found = true;
                        matched_any = true;
                        start = end;
                        break;
                    }
                    end -= 1;
                }

                if !found {
                    ids.push(unk_id);
                    break;
                }
            }

            if !matched_any {
                ids.push(unk_id);
            }
        }

        Ok(Encoding { ids })
    }

    pub fn decode(&self, seq: &[u32], _skip_special_tokens: bool) -> Result<String, String> {
        let mut tokens = Vec::new();
        for &id in seq {
            if let Some(tok) = self.id_to_token.get(&id) {
                if tok.starts_with("##") {
                    if let Some(last) = tokens.last_mut() {
                        let sub = &tok[2..];
                        let combined = format!("{}{}", last, sub);
                        *last = combined;
                    } else {
                        tokens.push(tok.clone());
                    }
                } else {
                    tokens.push(tok.clone());
                }
            }
        }
        Ok(tokens.join(" "))
    }
}
