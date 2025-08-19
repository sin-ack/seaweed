use thiserror::Error;

pub mod ast {
    /// List of keywords in the language in use.
    #[derive(Debug, Eq, PartialEq)]
    pub enum Keyword {
        Abstract,
        Class,
        Else,
        Extends,
        For,
        Function,
        If,
        Import,
        Let,
        Local,
        Module,
        New,
        Open,
        Outer,
        Super,
        This,
        Typealias,
        When,
        // The following keywords only make sense in the context of builtin calls.
        Trace,        // `trace()`
        Throw,        // `throw()`
        ImportGlob,   // `import*()`
        Read,         // `read()`
        ReadNullable, // `read?()`
        ReadGlob,     // `read*()`
        // The following keywords are reserved, but do not have any special meaning
        // in the language.
        Protected,
        Override,
        Record,
        Delete,
        Case,
        Switch,
        Vararg,
    }

    impl Keyword {
        pub(crate) fn from_str(identifier: &str) -> Option<Keyword> {
            match identifier {
                "abstract" => Some(Keyword::Abstract),
                "class" => Some(Keyword::Class),
                "else" => Some(Keyword::Else),
                "extends" => Some(Keyword::Extends),
                "for" => Some(Keyword::For),
                "function" => Some(Keyword::Function),
                "if" => Some(Keyword::If),
                "import" => Some(Keyword::Import),
                "let" => Some(Keyword::Let),
                "local" => Some(Keyword::Local),
                "module" => Some(Keyword::Module),
                "new" => Some(Keyword::New),
                "open" => Some(Keyword::Open),
                "outer" => Some(Keyword::Outer),
                "super" => Some(Keyword::Super),
                "this" => Some(Keyword::This),
                "typealias" => Some(Keyword::Typealias),
                "when" => Some(Keyword::When),
                // Builtin calls
                "trace" => Some(Keyword::Trace),
                "throw" => Some(Keyword::Throw),
                "import*" => Some(Keyword::ImportGlob),
                "read" => Some(Keyword::Read),
                "read?" => Some(Keyword::ReadNullable),
                "read*" => Some(Keyword::ReadGlob),
                _ => None,
            }
        }

        /// Return whether the keyword is a builtin call.
        pub fn is_builtin_call(&self) -> bool {
            matches!(
                self,
                Keyword::Trace
                    | Keyword::Throw
                    | Keyword::ImportGlob
                    | Keyword::Read
                    | Keyword::ReadNullable
                    | Keyword::ReadGlob
            )
        }

        /// Return whether the keyword is a reserved keyword.
        pub fn is_reserved(&self) -> bool {
            matches!(
                self,
                Keyword::Protected
                    | Keyword::Override
                    | Keyword::Record
                    | Keyword::Delete
                    | Keyword::Case
                    | Keyword::Switch
                    | Keyword::Vararg
            )
        }
    }

    /// List of symbols in the language.
    #[derive(Eq, PartialEq)]
    pub enum Symbol {
        Backslash, // `\`
        Backtick,  // `` ` ``
        Bang,      // `!`
        Comma,     // `,`
        Equals,    // `=`
        Semicolon, // `;`
        Newline,   // `\n`, alternative to `Semicolon`

        DoubleQuote,       // `"`
        TripleDoubleQuote, // `"""`

        Plus,         // `+`
        Minus,        // `-`
        Asterisk,     // `*`
        ForwardSlash, // `/`
        Percent,      // `%`

        Period,     // `.`
        Question,   // `?`
        RightArrow, // `->`
        Colon,      // `:`

        GreaterThan,        // `>`
        GreaterThanOrEqual, // `>=`
        LessThan,           // `<`
        LessThanOrEqual,    // `<=`
        DoubleEquals,       // `==`
        NotEquals,          // `!=`
        DoubleAmpersand,    // `&&`
        DoublePipe,         // `||`

        OpenParen,    // `(`
        CloseParen,   // `)`
        OpenBracket,  // `[`
        CloseBracket, // `]`
        OpenBrace,    // `{`
        CloseBrace,   // `}`
    }
    impl Symbol {
        pub(crate) fn is_symbol_char(ch: char) -> bool {
            matches!(
                ch,
                '\\' | '`'
                    | '!'
                    | ','
                    | '='
                    | ';'
                    | '"'
                    | '+'
                    | '-'
                    | '*'
                    | '/'
                    | '%'
                    | '.'
                    | '?'
                    | '>'
                    | '<'
                    | ':'
                    | '&'
                    | '|'
                    | '('
                    | ')'
                    | '['
                    | ']'
                    | '{'
                    | '}'
            )
        }

        pub(crate) fn from_str(symbol: &str) -> Option<Symbol> {
            match symbol {
                "\\" => Some(Symbol::Backslash),
                "`" => Some(Symbol::Backtick),
                "!" => Some(Symbol::Bang),
                "," => Some(Symbol::Comma),
                "=" => Some(Symbol::Equals),
                ";" => Some(Symbol::Semicolon),
                "\"" => Some(Symbol::DoubleQuote),
                "\"\"\"" => Some(Symbol::TripleDoubleQuote),
                "+" => Some(Symbol::Plus),
                "-" => Some(Symbol::Minus),
                "*" => Some(Symbol::Asterisk),
                "/" => Some(Symbol::ForwardSlash),
                "%" => Some(Symbol::Percent),
                "." => Some(Symbol::Period),
                "?" => Some(Symbol::Question),
                "->" => Some(Symbol::RightArrow),
                ":" => Some(Symbol::Colon),
                ">" => Some(Symbol::GreaterThan),
                ">=" => Some(Symbol::GreaterThanOrEqual),
                "<" => Some(Symbol::LessThan),
                "<=" => Some(Symbol::LessThanOrEqual),
                "==" => Some(Symbol::DoubleEquals),
                "!=" => Some(Symbol::NotEquals),
                "&&" => Some(Symbol::DoubleAmpersand),
                "||" => Some(Symbol::DoublePipe),
                "(" => Some(Symbol::OpenParen),
                ")" => Some(Symbol::CloseParen),
                "[" => Some(Symbol::OpenBracket),
                "]" => Some(Symbol::CloseBracket),
                "{" => Some(Symbol::OpenBrace),
                "}" => Some(Symbol::CloseBrace),
                _ => None,
            }
        }
    }

    impl std::fmt::Debug for Symbol {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Symbol::Backslash => write!(f, "\\"),
                Symbol::Backtick => write!(f, "`"),
                Symbol::Bang => write!(f, "!"),
                Symbol::Comma => write!(f, ","),
                Symbol::Equals => write!(f, "="),
                Symbol::Semicolon => write!(f, ";"),
                Symbol::Newline => write!(f, "\\n"),
                Symbol::DoubleQuote => write!(f, "\""),
                Symbol::TripleDoubleQuote => write!(f, "\"\"\""),
                Symbol::Plus => write!(f, "+"),
                Symbol::Minus => write!(f, "-"),
                Symbol::Asterisk => write!(f, "*"),
                Symbol::ForwardSlash => write!(f, "/"),
                Symbol::Percent => write!(f, "%"),
                Symbol::Period => write!(f, "."),
                Symbol::Question => write!(f, "?"),
                Symbol::RightArrow => write!(f, "->"),
                Symbol::Colon => write!(f, ":"),
                Symbol::GreaterThan => write!(f, ">"),
                Symbol::GreaterThanOrEqual => write!(f, ">="),
                Symbol::LessThan => write!(f, "<"),
                Symbol::LessThanOrEqual => write!(f, "<="),
                Symbol::DoubleEquals => write!(f, "=="),
                Symbol::NotEquals => write!(f, "!="),
                Symbol::DoubleAmpersand => write!(f, "&&"),
                Symbol::DoublePipe => write!(f, "||"),
                Symbol::OpenParen => write!(f, "("),
                Symbol::CloseParen => write!(f, ")"),
                Symbol::OpenBracket => write!(f, "["),
                Symbol::CloseBracket => write!(f, "]"),
                Symbol::OpenBrace => write!(f, "{{"),
                Symbol::CloseBrace => write!(f, "}}"),
            }
        }
    }

    #[derive(Debug, PartialEq)]
    pub enum Token {
        /// A keyword token.
        Keyword(Keyword),
        /// An UTF-8 string literal.
        StringLiteral(String),
        /// An identifier.
        Identifier(String),
        /// The blank identifier, `_`, used to ignore values.
        BlankIdentifier,
        /// A 64-bit integer literal.
        Integer(i64),
        /// A double-precision floating point literal.
        Float(f64),
        /// A null literal.
        Null,
        /// A boolean literal.
        Boolean(bool),
        /// A symbol.
        Symbol(Symbol),
        /// A line of documentation comment.
        DocComment(String),
        /// End of file token.
        Eof,
    }
}

/// A simple tokenizer.
pub struct Tokenizer<'a> {
    /// The input buffer being parsed.
    input: &'a str,
    /// The offset, in bytes, of the next character to be parsed.
    offset: usize,
}

enum TokenizerState {
    /// Default parser state. Scans for keywords, otherwise looks for identifiers and literals.
    Normal,
    /// Parsing a decimal integer, or the integer part of a floating point number.
    Decimal,
    /// Parsing the fractional part of a floating point number.
    Fractional,
    /// Parsing the exponent part of a floating point number.
    FloatExponent,
    /// Parsing a binary integer.
    Binary,
    /// Parsing an octal integer.
    Octal,
    /// Parsing a hexadecimal integer.
    Hexadecimal,
    /// In a string.
    String,
    /// In a multiline string.
    MultilineString,
    /// In a comment.
    Comment,
}

#[derive(Error, Debug)]
pub enum TokenizerError {
    #[error("Unexpected character {0}.")]
    UnexpectedCharacter(String),
    #[error("Int literal {0} is too large.")]
    IntLiteralTooLarge(String),
}

impl<'a> Tokenizer<'a> {
    /// Create a new tokenizer with the given input.
    pub fn new(input: &'a str) -> Self {
        Tokenizer { input, offset: 0 }
    }

    /// Peek the current character without consuming it.
    fn peek(&self) -> Option<char> {
        self.input[self.offset..].chars().next()
    }

    /// Consume the current character and return it.
    fn consume(&mut self) -> Option<char> {
        let ch = self.peek();
        if let Some(ch) = ch {
            self.offset += ch.len_utf8();
        }
        ch
    }

    /// Rewind the tokenizer by the given character.
    fn rewind(&mut self, ch: char) {
        self.offset -= ch.len_utf8();
    }

    /// Consume the next character while it matches the given predicate.
    fn consume_while<F>(&mut self, predicate: F) -> Option<char>
    where
        F: Fn(char) -> bool,
    {
        let mut ch = self.consume();
        while let Some(c) = ch {
            if !predicate(c) {
                self.rewind(c);
                break;
            }
            ch = self.consume();
        }
        ch
    }

    // TODO: Option<ast::Token> to make it streaming?
    // TODO: Pkl documentation says identifiers have to follow Unicode UAX31-R1-1 syntax,
    //       which reads like nonsense to me so we'll just accept any alphanumeric
    //       character or underscore for now. This should be tightened up later.
    /// Get the next token from the input.
    pub fn next_token(&'_ mut self) -> Result<ast::Token, TokenizerError> {
        // Skip whitespace
        self.consume_while(|c| c.is_whitespace());

        if self.peek() == None {
            // Out of tokens! We're done here.
            return Ok(ast::Token::Eof);
        }

        let mut state = TokenizerState::Normal;
        let mut number_buffer = String::new();
        'next_state: loop {
            let mut c = self.consume();
            match state {
                TokenizerState::Normal => match c {
                    None => return Ok(ast::Token::Eof),
                    // Keyword, or identifier otherwise.
                    Some('a'..='z') | Some('A'..='Z') | Some('_') => {
                        let mut identifier = String::new();
                        identifier.push(c.unwrap());
                        let mut ch = self.consume();
                        while let Some(c) = ch {
                            if !(c.is_alphanumeric() || c == '_') {
                                self.rewind(c);
                                break;
                            }

                            identifier.push(c);
                            ch = self.consume();
                        }

                        return if identifier == "_" {
                            Ok(ast::Token::BlankIdentifier)
                        } else if let Some(keyword) = ast::Keyword::from_str(&identifier) {
                            Ok(ast::Token::Keyword(keyword))
                        } else {
                            Ok(ast::Token::Identifier(identifier))
                        };
                    }
                    Some('0') => {
                        // We don't know what this means yet:
                        // - Leading 0 for a decimal (Pkl allows a leading 0 weirdly enough, whereas languages like Python do not).
                        // - 0b/0x/0o prefix for binary, hexadecimal or octal integer.
                        c = self.consume();
                        match c {
                            Some('b') => state = TokenizerState::Binary,
                            Some('x') => state = TokenizerState::Hexadecimal,
                            Some('o') => state = TokenizerState::Octal,
                            // Decimal integer. `_` here to allow for 0_123 and so on.
                            Some('0'..='9') | Some('_') => {
                                state = TokenizerState::Decimal;
                                number_buffer.push('0');
                            }
                            // Decimal floating point number.
                            Some('.') => {
                                state = TokenizerState::Fractional;
                                number_buffer.push('0');
                                number_buffer.push('.');
                            }
                            // Just a zero, no prefix.
                            Some(_) | None => {
                                self.rewind(c.unwrap());
                                return Ok(ast::Token::Integer(0));
                            }
                        }
                    }
                    Some('1'..='9') => {
                        state = TokenizerState::Decimal;
                        number_buffer.push(c.unwrap());
                    }
                    // Try to parse a symbol.
                    Some(ch) => {
                        match ch {
                            // If a period is immediately followed by a digit, it is treated as a decimal point.
                            '.' => {
                                let next_ch = self.peek();
                                match next_ch {
                                    Some('0'..='9') => {
                                        state = TokenizerState::Fractional;
                                        number_buffer.push('0');
                                        number_buffer.push('.');
                                        continue 'next_state;
                                    }
                                    // Otherwise, it is a symbol. Carry on with the symbol parsing.
                                    _ => {}
                                }
                            }
                            // If a minus sign is immediately followed by a digit, it is treated as a negative number.
                            '-' => {
                                let next_ch = self.peek();

                                match next_ch {
                                    Some('0'..='9') => {
                                        number_buffer.push('-');
                                        // Let Normal state dispatch decide on what to do next.
                                        continue 'next_state;
                                    }
                                    Some('.') => {
                                        // This might be gibberish or it might be a negative float;
                                        // check one more character.
                                        self.consume();
                                        let next_next_ch = self.peek();
                                        match next_next_ch {
                                            Some('0'..='9') => {
                                                // Negative float.
                                                number_buffer.push('-');
                                                number_buffer.push('0');
                                                number_buffer.push('.');
                                                state = TokenizerState::Fractional;
                                                continue 'next_state;
                                            }
                                            // Nope, something else.
                                            _ => self.rewind('.'),
                                        }
                                    }
                                    // Otherwise, it is a symbol. Carry on with the symbol parsing.
                                    _ => {}
                                }
                            }
                            _ => {}
                        }

                        self.rewind(ch);
                        let symbol_start_offset = self.offset;

                        if !ast::Symbol::is_symbol_char(ch) {
                            return Err(TokenizerError::UnexpectedCharacter(ch.to_string()));
                        }

                        let mut symbol = String::new();
                        let mut ch = self.consume();
                        while let Some(c) = ch {
                            if !ast::Symbol::is_symbol_char(c) {
                                self.offset = symbol_start_offset;
                                return Err(TokenizerError::UnexpectedCharacter(c.to_string()));
                            }

                            symbol.push(c);
                            if let Some(symbol) = ast::Symbol::from_str(&symbol) {
                                return Ok(ast::Token::Symbol(symbol));
                            }

                            ch = self.consume();
                        }
                    }
                },
                TokenizerState::Decimal => match c {
                    Some('0'..='9') | Some('_') => {
                        if c != Some('_') {
                            number_buffer.push(c.unwrap());
                        }
                    }
                    Some('.') => {
                        state = TokenizerState::Fractional;
                        number_buffer.push('.');
                    }
                    None | Some(_) => {
                        if let Some(ch) = c {
                            self.rewind(ch);
                        }

                        let value =
                            i64::from_str_radix(&number_buffer, 10).map_err(|err| {
                                match err.kind() {
                                    std::num::IntErrorKind::PosOverflow
                                    | std::num::IntErrorKind::NegOverflow => {
                                        return TokenizerError::IntLiteralTooLarge(
                                            number_buffer.clone(),
                                        );
                                    }
                                    _ => panic!("Unexpected error while parsing integer: {err}"),
                                }
                            })?;
                        return Ok(ast::Token::Integer(value));
                    }
                },
                TokenizerState::Fractional => match c {
                    Some('0'..='9') | Some('_') => {
                        if c != Some('_') {
                            number_buffer.push(c.unwrap());
                        }
                    }
                    Some('e') | Some('E') => {
                        number_buffer.push('e');
                        match self.peek() {
                            Some('+') | Some('-') => {
                                number_buffer.push(self.consume().unwrap());
                            }
                            _ => {}
                        }

                        state = TokenizerState::FloatExponent;
                    }
                    None | Some(_) => {
                        if let Some(ch) = c {
                            self.rewind(ch);
                        }

                        let value = number_buffer
                            .parse::<f64>()
                            .map_err(|_| panic!("TODO: Not sure how we could get here."))?;
                        return Ok(ast::Token::Float(value));
                    }
                },
                TokenizerState::FloatExponent => match c {
                    Some('0'..='9') | Some('_') => {
                        if c != Some('_') {
                            number_buffer.push(c.unwrap());
                        }
                    }
                    None | Some(_) => {
                        if let Some(ch) = c {
                            self.rewind(ch);
                        }

                        let value = number_buffer
                            .parse::<f64>()
                            .map_err(|_| panic!("TODO: Not sure how we could get here."))?;
                        return Ok(ast::Token::Float(value));
                    }
                },
                TokenizerState::Binary => match c {
                    Some('0'..='1') | Some('_') => {
                        if c != Some('_') {
                            number_buffer.push(c.unwrap());
                        }
                    }
                    None | Some(_) => {
                        if let Some(ch) = c {
                            self.rewind(ch);
                        }

                        let value =
                            i64::from_str_radix(&number_buffer, 2).map_err(|err| {
                                match err.kind() {
                                    std::num::IntErrorKind::PosOverflow
                                    | std::num::IntErrorKind::NegOverflow => {
                                        return TokenizerError::IntLiteralTooLarge(
                                            number_buffer.clone(),
                                        );
                                    }
                                    _ => panic!("Unexpected error while parsing integer: {err}"),
                                }
                            })?;
                        return Ok(ast::Token::Integer(value));
                    }
                },
                TokenizerState::Octal => match c {
                    Some('0'..='7') | Some('_') => {
                        if c != Some('_') {
                            // Skip underscores in the number.
                            number_buffer.push(c.unwrap());
                        }
                    }
                    None | Some(_) => {
                        if let Some(ch) = c {
                            self.rewind(ch);
                        }

                        let value =
                            i64::from_str_radix(&number_buffer, 8).map_err(|err| {
                                match err.kind() {
                                    std::num::IntErrorKind::PosOverflow
                                    | std::num::IntErrorKind::NegOverflow => {
                                        return TokenizerError::IntLiteralTooLarge(
                                            number_buffer.clone(),
                                        );
                                    }
                                    _ => panic!("Unexpected error while parsing integer: {err}"),
                                }
                            })?;
                        return Ok(ast::Token::Integer(value));
                    }
                },
                TokenizerState::Hexadecimal => match c {
                    Some('0'..='9') | Some('a'..='f') | Some('A'..='F') | Some('_') => {
                        if c != Some('_') {
                            // Skip underscores in the number.
                            number_buffer.push(c.unwrap());
                        }
                    }
                    None | Some(_) => {
                        if let Some(ch) = c {
                            self.rewind(ch);
                        }

                        let value =
                            i64::from_str_radix(&number_buffer, 16).map_err(|err| {
                                match err.kind() {
                                    std::num::IntErrorKind::PosOverflow
                                    | std::num::IntErrorKind::NegOverflow => {
                                        return TokenizerError::IntLiteralTooLarge(
                                            number_buffer.clone(),
                                        );
                                    }
                                    _ => panic!("Unexpected error while parsing integer: {err}"),
                                }
                            })?;
                        return Ok(ast::Token::Integer(value));
                    }
                },
                TokenizerState::String => todo!(),
                TokenizerState::MultilineString => todo!(),
                TokenizerState::Comment => todo!(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integer_tokenization() {
        let mut tokenizer = Tokenizer::new("123 0 0x1a 0b101 0o77");
        assert_eq!(tokenizer.next_token().unwrap(), ast::Token::Integer(123));
        assert_eq!(tokenizer.next_token().unwrap(), ast::Token::Integer(0));
        assert_eq!(tokenizer.next_token().unwrap(), ast::Token::Integer(26));
        assert_eq!(tokenizer.next_token().unwrap(), ast::Token::Integer(5));
        assert_eq!(tokenizer.next_token().unwrap(), ast::Token::Integer(63));
    }

    #[test]
    fn test_integer_limits() {
        let mut tokenizer =
            Tokenizer::new("9223372036854775807 -9223372036854775808 999999999999999999999999999");
        assert_eq!(
            tokenizer.next_token().unwrap(),
            ast::Token::Integer(9223372036854775807)
        );
        assert_eq!(
            tokenizer.next_token().unwrap(),
            ast::Token::Integer(-9223372036854775808)
        );

        // This should fail due to overflow.
        let result = tokenizer.next_token();
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            TokenizerError::IntLiteralTooLarge(_)
        ));
    }

    #[test]
    fn test_float_tokenization() {
        let mut tokenizer = Tokenizer::new("123.456 0.0 1.23e4 5.67E-8 .1 -.5");
        assert_eq!(tokenizer.next_token().unwrap(), ast::Token::Float(123.456));
        assert_eq!(tokenizer.next_token().unwrap(), ast::Token::Float(0.0));
        assert_eq!(tokenizer.next_token().unwrap(), ast::Token::Float(12300.0));
        assert_eq!(tokenizer.next_token().unwrap(), ast::Token::Float(5.67e-8));
        assert_eq!(tokenizer.next_token().unwrap(), ast::Token::Float(0.1));
        assert_eq!(tokenizer.next_token().unwrap(), ast::Token::Float(-0.5));
    }

    #[test]
    fn test_multiple_float_quirk() {
        let mut tokenizer = Tokenizer::new(".1.2.3");
        assert_eq!(tokenizer.next_token().unwrap(), ast::Token::Float(0.1));
        assert_eq!(tokenizer.next_token().unwrap(), ast::Token::Float(0.2));
        assert_eq!(tokenizer.next_token().unwrap(), ast::Token::Float(0.3));
    }

    #[test]
    fn test_shouldnt_just_match_on_keyword_prefix() {
        // Not Keyword(Function) Identifier("ality"), but Identifier("functionality")
        let mut tokenizer = Tokenizer::new("function functionality");
        assert_eq!(
            tokenizer.next_token().unwrap(),
            ast::Token::Keyword(ast::Keyword::Function)
        );
        assert_eq!(
            tokenizer.next_token().unwrap(),
            ast::Token::Identifier("functionality".to_string())
        );
    }
}
