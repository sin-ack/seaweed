use clap::Parser;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// The expression to tokenize
    #[arg(short = 'x', long)]
    expression: String,
}

fn main() {
    let args = Args::parse();

    let mut tokenizer = seaweed::Tokenizer::new(&args.expression);

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
