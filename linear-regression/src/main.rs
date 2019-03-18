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

use rusty_machine::learning::lin_reg::LinRegressor;
use rusty_machine::learning::SupModel;
use rusty_machine::linalg::Matrix;
use rusty_machine::linalg::Vector;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
struct Record {
    bmi: f64,
    life_expectancy: f64,
    country: String
}

fn run() -> Result<(), Box<Error>> {
    let file_path = get_first_arg()?;
    let file = File::open(file_path)?;
    let mut rdr = csv::Reader::from_reader(file);

    let mut inputs_vec: Vec<f64> = Vec::new();
    let mut targets_vec: Vec<f64> = Vec::new();

    for result in rdr.deserialize() {
        let record: Record = result?;
        inputs_vec.push(record.bmi);
        targets_vec.push(record.life_expectancy);
    }

    let inputs = Matrix::new(inputs_vec.len(), 1, inputs_vec);
    let targets = Vector::new(targets_vec);

    let model = train_model(inputs, targets);

    let new_point = Matrix::new(1,1,vec![21.07931]);
    let output = model.predict(&new_point).unwrap();
    println!("Life Expectancy Prediction for {}: {} ", new_point[[0,0]], output[0]);

    Ok(())
}

fn train_model(inputs: Matrix<f64>, targets: Vector<f64>) -> LinRegressor {
    let mut lin_mod = LinRegressor::default();
    lin_mod.train(&inputs, &targets).unwrap();
    lin_mod
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