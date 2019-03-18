extern crate csv;
extern crate rusty_machine;
extern crate serde;
// This lets us write `#[derive(Deserialize)]`.
#[macro_use]
extern crate serde_derive;

use std::env;
use std::error::Error;
use std::ffi::OsString;
use std::fs::File;
use std::process;

// use rusty_machine::learning::lin_reg::LinRegressor;
// use rusty_machine::learning::SupModel;
// use rusty_machine::linalg::Matrix;
// use rusty_machine::linalg::Vector;

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

    // TODO: 
    // not quite sure how to get the count of rdr.deserialize (lazy iterator?)
    // use vectors to accumulate results for now
    let mut inputs: Vec<f64> = Vec::new();
    let mut targets: Vec<f64> = Vec::new();

    for result in rdr.deserialize() {
        let record: Record = result?;
        inputs.push(record.bmi);
        targets.push(record.life_expectancy);
    }
    
    Ok(())
}

fn get_first_arg() -> Result<OsString, Box<Error>> {
    match env::args_os().nth(1) {
        None => Err(From::from("expected 1 argument, but got none")),
        Some(file_path) => Ok(file_path),
    }
}

/*
fn train_model() {
    let inputs = Matrix::new(4,1,vec![1.0,3.0,5.0,7.0]);
    let targets = Vector::new(vec![1.,5.,9.,13.]);

    let mut lin_mod = LinRegressor::default();

    // Train the model
    lin_mod.train(&inputs, &targets).unwrap();

    // Now we'll predict a new point
    let new_point = Matrix::new(1,1,vec![10.]);
    let output = lin_mod.predict(&new_point).unwrap();

    println!("prediction {} ", output[0]);

    // Hopefully we classified our new point correctly!
    assert!(output[0] > 17f64, "Our regressor isn't very good!");
}
*/

fn main() {
    if let Err(err) = run() {
        println!("{}", err);
        process::exit(1);
    }
}