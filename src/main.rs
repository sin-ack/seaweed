use std::io::Read;

use clap::Parser;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// The expression to tokenize. If not provided, stdin will be used as input.
    #[arg(short = 'x', long, required = false)]
    expression: Option<String>,
}

fn main() {
    let args = Args::parse();

    let expression = match args.expression {
        Some(expr) => expr,
        None => {
            let mut buffer = Vec::new();
            std::io::stdin()
                .read_to_end(&mut buffer)
                .expect("Failed to read from stdin");
            String::from_utf8(buffer).expect("Invalid UTF-8 input")
        }
    };

    let mut tokenizer = seaweed::Tokenizer::new(&expression);

    loop {
        match tokenizer.next_token() {
            Ok(t) => {
                if t == seaweed::ast::Token::Eof {
                    break;
                }
                println!("{:?}", t);
            }
            Err(e) => {
                eprintln!("Error: {}", e);
                break;
            }
        }
    }
}
