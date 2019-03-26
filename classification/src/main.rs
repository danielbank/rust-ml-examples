extern crate csv;
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate rusty_machine;

use std::env;
use std::error::Error;
use std::ffi::OsString;
use std::fs::File;
use std::process;

use rusty_machine::learning::svm::SVM;
use rusty_machine::learning::SupModel;
use rusty_machine::linalg::Matrix;
use rusty_machine::linalg::Vector;
use rusty_machine::learning::toolkit::kernel;
use rusty_machine::learning::toolkit::kernel::Polynomial;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
struct Record {
    x1: f64,
    x2: f64,
    y: f64
}

fn run() -> Result<(), Box<Error>> {
    let file_path = get_first_arg()?;
    let file = File::open(file_path)?;

    let mut x1_vec: Vec<f64> = Vec::new();
    let mut x2_vec: Vec<f64> = Vec::new();
    let mut targets_vec: Vec<f64> = Vec::new();

    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(file);

    for result in rdr.deserialize() {
        let record: Record = result?;
        x1_vec.push(record.x1);
        x2_vec.push(record.x2);
        if record.y == 0.0 {
            targets_vec.push(-1.0);
        } else {
            targets_vec.push(1.0);
        }
    }

    let x_vec = [&x1_vec[..], &x2_vec[..]].concat();
    let inputs = Matrix::new(x1_vec.len(), 2, x_vec);
    let targets = Vector::new(targets_vec);

    let model = train_model(inputs, targets);

    for x in -20..20 {
        let mut x = x as f64;
        x = x / 10.0;
        let new_point = Matrix::new(1, 2, vec![x, x]);
        let output = model.predict(&new_point).unwrap();
        println!("Prediction for {}: {} ", new_point, output[0]);
    }

    Ok(())
}

fn train_model(inputs: Matrix<f64>, targets: Vector<f64>) -> SVM<Polynomial> {
    // Constructs a new polynomial with alpha = 1, c = 0, d = 2.
    let ker = kernel::Polynomial::new(1e-8, 0.0, 2.0);
    let mut svm_mod = SVM::new(ker, 0.3);
    svm_mod.train(&inputs, &targets).unwrap();
    svm_mod
}

fn get_first_arg() -> Result<OsString, Box<Error>> {
    match env::args_os().nth(1) {
        None => Err(From::from("expected 1 argument, but got none")),
        Some(file_path) => Ok(file_path),
    }
}

fn main() {
    if let Err(err) = run() {
        println!("{}", err);
        process::exit(1);
    }
}