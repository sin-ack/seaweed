use thiserror::Error;

macro_rules! string_enum {
    (
        $(#[$meta:meta])*
        $name:ident, {
            $($variant:ident => $val:expr),* $(,)?
        }
    ) => {
        $(#[$meta])*
        #[derive(Eq, PartialEq)]
        pub enum $name {
            $($variant),*
        }

        impl $name {
            pub(crate) fn from_str(s: &str) -> Option<$name> {
                match s {
                    $(
                        $val => Some($name::$variant),
                    )*
                    _ => None,
                }
            }

            pub(crate) fn as_str(&self) -> &str {
                match self {
                    $(
                        $name::$variant => $val,
                    )*
                }
            }
        }

        impl std::fmt::Debug for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "\"{}\"", self.as_str())
            }
        }
    };
}

pub mod ast {
    string_enum!(
        /// List of keywords in the language in use.
        Keyword, {
        Abstract => "abstract",
        Class => "class",
        Else => "else",
        Extends => "extends",
        For => "for",
        Function => "function",
        If => "if",
        Import => "import",
        Let => "let",
        Local => "local",
        Module => "module",
        New => "new",
        Open => "open",
        Outer => "outer",
        Super => "super",
        This => "this",
        Typealias => "typealias",
        When => "when",

        // The following keywords only make sense in the context of builtin calls.
        Trace => "trace", // `trace()`
        Throw => "throw", // `throw()`
        ImportGlob => "import*", // `import*()`
        Read => "read", // `read()`
        ReadNullable => "read?", // `read?()`
        ReadGlob => "read*", // `read*()`

        // The following keywords are reserved, but do not have any special meaning
        // in the language.
        Protected => "protected",
        Override => "override",
        Record => "record",
        Delete => "delete",
        Case => "case",
        Switch => "switch",
        Vararg => "vararg"
    });

    impl Keyword {
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

    string_enum!(
        /// List of symbols in the language.
        Symbol, {
            Backslash => "\\",
            Backtick => "`",
            Bang => "!",
            Comma => ",",
            Equals => "=",
            Semicolon => ";",
            Newline => "\n", //alternative to "Semicolon"

            Plus => "+",
            Minus => "-",
            Asterisk => "*",
            ForwardSlash => "/",
            Percent => "%",

            Period => ".",
            Question => "?",
            RightArrow => "->",
            Colon => ":",

            GreaterThan => ">",
            GreaterThanOrEqual => ">=",
            LessThan => "<",
            LessThanOrEqual => "<=",
            DoubleEquals => "==",
            NotEquals => "!=",
            DoubleAmpersand => "&&",
            DoublePipe => "||",

            OpenParen => "(",
            CloseParen => ")",
            OpenBracket => "[",
            CloseBracket => "]",
            OpenBrace => "{",
            CloseBrace => "}",
        }
    );
    impl Symbol {
        pub(crate) fn is_symbol_char(ch: char) -> bool {
            matches!(
                ch,
                '\\' | '`'
                    | '!'
                    | ','
                    | '='
                    | ';'
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
    }

    #[derive(Debug, PartialEq)]
    pub enum Token {
        /// A keyword token.
        Keyword(Keyword),
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

        // --- String literals ---
        /// A literal part of a string literal.
        StringPart(String),
        /// Indicator for the start of an interpolated expression in a string literal.
        InterpolatedExpressionStart,
        /// An end of an interpolated expression in a string literal.
        InterpolatedExpressionEnd,
    }
}

/// A context that can be pushed onto the tokenizer stack to change its behavior.
enum TokenizerContext {
    /// The tokenizer is currently inside a string literal, and is tokenizing
    /// an interpolated expression (`\(...)`).
    /// When a `)` is encountered, the tokenizer will reset to the `String` state.
    InsideInterpolatedExpression,
}

/// A simple tokenizer.
pub struct Tokenizer<'a> {
    /// The input buffer being parsed.
    input: &'a str,
    /// The offset, in bytes, of the next character to be parsed.
    offset: usize,
    /// The stack of contexts that the tokenizer is currently in.
    contexts: Vec<TokenizerContext>,
    /// The initial state that the next `next_token` call will start in.
    next_initial_state: TokenizerState,
}

#[derive(Copy, Clone, Debug)]
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
    /// Just encountered an interpolated expression in a string literal.
    EncounteredInterpolatedExpression,
    /// In a comment.
    Comment,
}

#[derive(Error, Debug)]
pub enum TokenizerError {
    #[error("Unexpected character {0}.")]
    UnexpectedCharacter(String),
    #[error("Int literal {0} is too large.")]
    IntLiteralTooLarge(String),
    #[error("Invalid character escape sequence \\{0}.")]
    InvalidCharacterEscapeSequence(char),
    #[error("Unexpected end of file.")]
    UnexpectedEndOfFile,
    #[error("Missing {0} delimiter.")]
    MissingDelimiter(&'static str),
}

impl<'a> Tokenizer<'a> {
    /// Create a new tokenizer with the given input.
    pub fn new(input: &'a str) -> Self {
        Tokenizer {
            input,
            offset: 0,
            contexts: Vec::new(),
            next_initial_state: TokenizerState::Normal,
        }
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
        let mut state = self.next_initial_state;
        self.next_initial_state = TokenizerState::Normal;

        // Perform initial actions for this state.
        match state {
            TokenizerState::Normal => {
                // Skip whitespace
                self.consume_while(|c| c.is_whitespace());
            }
            TokenizerState::EncounteredInterpolatedExpression => {
                // In the previous call to `next_token`, we encountered an
                // interpolated expression start.
                // Then we emitted a `StringPart` token, and now we have to emit
                // an `InterpolatedExpressionStart` token so that the parser
                // knows that it has to expect an expression next.
                self.contexts
                    .push(TokenizerContext::InsideInterpolatedExpression);
                return Ok(ast::Token::InterpolatedExpressionStart);
            }
            TokenizerState::String => {}
            _ => panic!("Unexpected initial state: {:?}", state),
        }

        if self.peek() == None {
            // Out of tokens! If our context is clear, return an EOF token;
            // otherwise we shouldn't have hit this point.
            return match self.contexts.last() {
                Some(TokenizerContext::InsideInterpolatedExpression) => {
                    Err(TokenizerError::UnexpectedEndOfFile)
                }
                None => Ok(ast::Token::Eof),
            };
        }

        let mut number_buffer = String::new();
        let mut string_buffer = String::new();
        loop {
            let mut c = self.consume();
            match state {
                TokenizerState::Normal => match c {
                    None => return Ok(ast::Token::Eof),
                    // Keyword, or identifier otherwise.
                    Some('a'..='z') | Some('A'..='Z') | Some('_') => {
                        let mut identifier = String::new();
                        identifier.push(c.unwrap());
                        while let Some(c) = self.consume() {
                            if !(c.is_alphanumeric() || c == '_') {
                                self.rewind(c);
                                break;
                            }

                            identifier.push(c);
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

                    Some('"') => {
                        state = TokenizerState::String;
                    }

                    // A `)` might be a symbol, or it might be the end of an interpolated expression.
                    Some(')') => {
                        if let Some(TokenizerContext::InsideInterpolatedExpression) =
                            self.contexts.last()
                        {
                            // This is the end of an interpolated expression.
                            self.contexts.pop();
                            self.next_initial_state = TokenizerState::String;
                            return Ok(ast::Token::InterpolatedExpressionEnd);
                        } else {
                            // This is just a symbol.
                            return Ok(ast::Token::Symbol(ast::Symbol::CloseParen));
                        }
                    }

                    // If a period is immediately followed by a digit, it is treated as a decimal point.
                    Some('.') => {
                        let next_ch = self.peek();
                        match next_ch {
                            Some('0'..='9') => {
                                state = TokenizerState::Fractional;
                                number_buffer.push('0');
                                number_buffer.push('.');
                            }
                            // Otherwise, it is a symbol.
                            _ => return Ok(ast::Token::Symbol(ast::Symbol::Period)),
                        }
                    }
                    // If a minus sign is immediately followed by a digit, it is treated as a negative number.
                    Some('-') => {
                        let next_ch = self.peek();

                        match next_ch {
                            Some('0'..='9') => {
                                number_buffer.push('-');
                                // Let Normal state dispatch decide on what to do next.
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
                                    }
                                    // Nope, something else.
                                    _ => self.rewind('.'),
                                }
                            }
                            // Otherwise, it is a symbol.
                            _ => return Ok(ast::Token::Symbol(ast::Symbol::Minus)),
                        }
                    }
                    // Try to parse a symbol.
                    Some(ch) => {
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

                TokenizerState::String => {
                    match c {
                        Some('"') => {
                            // End of string literal.
                            return Ok(ast::Token::StringPart(string_buffer));
                        }
                        Some('\\') => {
                            // Escape sequence, consume the next character.
                            let next_ch = self.consume();
                            match next_ch {
                                Some('n') => string_buffer.push('\n'),
                                Some('r') => string_buffer.push('\r'),
                                Some('t') => string_buffer.push('\t'),
                                Some('"') => string_buffer.push('"'),
                                Some('\\') => string_buffer.push('\\'),

                                Some('(') => {
                                    // Start of an interpolated expression.
                                    self.next_initial_state =
                                        TokenizerState::EncounteredInterpolatedExpression;
                                    return Ok(ast::Token::StringPart(string_buffer));
                                }

                                Some(ch) => {
                                    self.rewind(ch);
                                    self.rewind('\\');
                                    return Err(TokenizerError::InvalidCharacterEscapeSequence(ch));
                                }
                                None => {
                                    // Unexpected end of input in a string literal escape sequence.
                                    return Err(TokenizerError::UnexpectedEndOfFile);
                                }
                            }
                        }
                        // TODO: Implement multiline strings (`"""..."""`).
                        Some('\n') | None => {
                            return Err(TokenizerError::MissingDelimiter("\""));
                        }

                        Some(ch) => {
                            // Just a regular character, add it to the string buffer.
                            string_buffer.push(ch);
                        }
                    }
                }
                TokenizerState::EncounteredInterpolatedExpression => unreachable!(),
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
    fn integer_tokenization() {
        let mut tokenizer = Tokenizer::new("123 0 0x1a 0b101 0o77");
        assert_eq!(tokenizer.next_token().unwrap(), ast::Token::Integer(123));
        assert_eq!(tokenizer.next_token().unwrap(), ast::Token::Integer(0));
        assert_eq!(tokenizer.next_token().unwrap(), ast::Token::Integer(26));
        assert_eq!(tokenizer.next_token().unwrap(), ast::Token::Integer(5));
        assert_eq!(tokenizer.next_token().unwrap(), ast::Token::Integer(63));
    }

    #[test]
    fn integer_limits() {
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
    fn float_tokenization() {
        let mut tokenizer = Tokenizer::new("123.456 0.0 1.23e4 5.67E-8 .1 -.5");
        assert_eq!(tokenizer.next_token().unwrap(), ast::Token::Float(123.456));
        assert_eq!(tokenizer.next_token().unwrap(), ast::Token::Float(0.0));
        assert_eq!(tokenizer.next_token().unwrap(), ast::Token::Float(12300.0));
        assert_eq!(tokenizer.next_token().unwrap(), ast::Token::Float(5.67e-8));
        assert_eq!(tokenizer.next_token().unwrap(), ast::Token::Float(0.1));
        assert_eq!(tokenizer.next_token().unwrap(), ast::Token::Float(-0.5));
    }

    #[test]
    fn multiple_float_quirk() {
        let mut tokenizer = Tokenizer::new(".1.2.3");
        assert_eq!(tokenizer.next_token().unwrap(), ast::Token::Float(0.1));
        assert_eq!(tokenizer.next_token().unwrap(), ast::Token::Float(0.2));
        assert_eq!(tokenizer.next_token().unwrap(), ast::Token::Float(0.3));
    }

    #[test]
    fn shouldnt_just_match_on_keyword_prefix() {
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

    #[test]
    fn basic_string_tokenization() {
        let mut tokenizer = Tokenizer::new(r#""hello" "world""#);
        assert_eq!(
            tokenizer.next_token().unwrap(),
            ast::Token::StringPart("hello".to_string())
        );

        assert_eq!(
            tokenizer.next_token().unwrap(),
            ast::Token::StringPart("world".to_string())
        );
    }

    #[test]
    fn eof_in_string_is_error() {
        let mut tokenizer = Tokenizer::new(r#""hello"#);
        let result = tokenizer.next_token();
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            TokenizerError::MissingDelimiter("\"")
        ));
    }

    #[test]
    fn newline_in_string_is_error() {
        let mut tokenizer = Tokenizer::new(
            r#""hello
world""#,
        );
        let result = tokenizer.next_token();
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            TokenizerError::MissingDelimiter("\"")
        ));
    }

    #[test]
    fn interpolated_expression() {
        let mut tokenizer = Tokenizer::new(r#""hello \(world)""#);
        assert_eq!(
            tokenizer.next_token().unwrap(),
            ast::Token::StringPart("hello ".to_string())
        );
        assert_eq!(
            tokenizer.next_token().unwrap(),
            ast::Token::InterpolatedExpressionStart
        );
        assert_eq!(
            tokenizer.next_token().unwrap(),
            ast::Token::Identifier("world".to_string())
        );
        assert_eq!(
            tokenizer.next_token().unwrap(),
            ast::Token::InterpolatedExpressionEnd
        );
        assert_eq!(
            tokenizer.next_token().unwrap(),
            ast::Token::StringPart("".to_string())
        );
    }

    #[test]
    fn nested_interpolated_expression() {
        let mut tokenizer = Tokenizer::new(r#""hello \(world "this is a \(nested) expr") waow""#);
        assert_eq!(
            tokenizer.next_token().unwrap(),
            ast::Token::StringPart("hello ".to_string())
        );
        assert_eq!(
            tokenizer.next_token().unwrap(),
            ast::Token::InterpolatedExpressionStart
        );
        assert_eq!(
            tokenizer.next_token().unwrap(),
            ast::Token::Identifier("world".to_string())
        );
        assert_eq!(
            tokenizer.next_token().unwrap(),
            ast::Token::StringPart("this is a ".to_string())
        );
        assert_eq!(
            tokenizer.next_token().unwrap(),
            ast::Token::InterpolatedExpressionStart
        );
        assert_eq!(
            tokenizer.next_token().unwrap(),
            ast::Token::Identifier("nested".to_string())
        );
        assert_eq!(
            tokenizer.next_token().unwrap(),
            ast::Token::InterpolatedExpressionEnd
        );
        assert_eq!(
            tokenizer.next_token().unwrap(),
            ast::Token::StringPart(" expr".to_string())
        );
        assert_eq!(
            tokenizer.next_token().unwrap(),
            ast::Token::InterpolatedExpressionEnd
        );
        assert_eq!(
            tokenizer.next_token().unwrap(),
            ast::Token::StringPart(" waow".to_string())
        );
    }

    #[test]
    fn eof_inside_interpolated_expression_should_error() {
        let mut tokenizer = Tokenizer::new(r#""hello \(world"#);
        assert_eq!(
            tokenizer.next_token().unwrap(),
            ast::Token::StringPart("hello ".to_string())
        );
        assert_eq!(
            tokenizer.next_token().unwrap(),
            ast::Token::InterpolatedExpressionStart
        );
        assert_eq!(
            tokenizer.next_token().unwrap(),
            ast::Token::Identifier("world".to_string())
        );

        let result = tokenizer.next_token();
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            TokenizerError::UnexpectedEndOfFile
        ));
    }
}
