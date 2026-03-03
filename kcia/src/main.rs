use candle_datasets::hub::from_hub as load_dataset;
use hf_hub::api::sync::ApiBuilder; // 変更
use ndarray::{Array1, Array2, Axis, array, concatenate};
use ndarray_linalg::Solve;
use parquet::file::reader::FileReader;
use parquet::record::RowAccessor;
use rand;
use std::path::PathBuf; // 追加。パスの指定用
use tqdm::tqdm;

fn ascii_printable() -> Vec<char> {
    (32..127).map(|i| i as u8 as char).collect()
}

fn ascii_count(s: String) -> Array1<f64> {
    let chars = ascii_printable();
    let mut counts = Array1::<f64>::zeros(chars.len());
    for c in s.chars() {
        if let Some(pos) = chars.iter().position(|&x| x == c) {
            counts[pos] += 1.0;
        }
    }
    counts / s.chars().count() as f64
}

fn linear_regression(xs: Array2<f64>, ys: Array1<f64>) -> Array1<f64> {
    let ones = Array2::<f64>::ones((xs.nrows(), 1));
    let xs_biased = concatenate![Axis(1), ones, xs];
    let n_features = xs_biased.shape()[1];
    let mut xtx = xs_biased.t().dot(&xs_biased);
    let ridge = 1e-12;
    xtx = xtx + Array2::<f64>::eye(n_features) * ridge;
    xtx.solve_into(xs_biased.t().dot(&ys)).unwrap()
}

fn eval_r2(weights: Array1<f64>, xs: Array2<f64>, ys: Array1<f64>) -> f64 {
    let ones = Array2::<f64>::ones((xs.nrows(), 1));
    let xs_biased = concatenate![Axis(1), ones, xs];
    let y_pred = xs_biased.dot(&weights);
    let ss_res = (ys.clone() - y_pred).mapv(|x| x * x).sum();
    let ss_tot = (ys.clone() - ys.mean().unwrap()).mapv(|x| x * x).sum();
    1.0 - ss_res / ss_tot
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_linear_regression_with_bias() {
        // y = 2x + 1
        let xs = array![[1.0], [2.0], [3.0], [4.0]];
        let ys = array![3.0, 5.0, 7.0, 9.0];
        let weights = linear_regression(xs, ys);

        // weights[0] is bias (1.0), weights[1] is slope (2.0)
        assert!((weights[0] - 1.0).abs() < 1e-6);
        assert!((weights[1] - 2.0).abs() < 1e-6);
    }
}

fn random_replace(s: String) -> (String, f64) {
    let chars = ascii_printable();
    let mut r = 1.0;
    let mut s = s.chars().collect::<Vec<char>>();
    let kkrt = rand::random::<f64>();
    for i in 0..s.len() {
        if rand::random::<f64>() < kkrt {
            s[i] = chars[rand::random::<usize>() % chars.len()];
            r -= 1.0 / s.len() as f64;
        }
    }
    (s.into_iter().collect(), r)
}

fn main() {
    let builder = ApiBuilder::new();
    let builder = builder.with_cache_dir(PathBuf::from("./cache"));

    let n_chars = ascii_printable().len();
    let mut xs = Array2::<f64>::zeros((0, n_chars));
    let mut ys = Array1::<f64>::zeros(0);

    let api = builder.build().unwrap();
    let repo_name = "ronantakizawa/github-top-code".to_string();
    let ds = load_dataset(&api, repo_name).unwrap();

    while let Some(file) = ds.iter().next() {
        let schema = file.metadata().file_metadata().schema();
        if let Ok(row_iter) = file.get_row_iter(Some(schema.clone())) {
            let input_index = schema
                .get_fields()
                .iter()
                .position(|f| f.name() == "content")
                .expect("column 'content' not found");
            for row in tqdm(row_iter) {
                if let Ok(row) = row {
                    let (s, r) = random_replace(row.get_string(input_index).unwrap().to_string());
                    xs = concatenate![Axis(0), xs, ascii_count(s).insert_axis(Axis(0))];
                    ys = concatenate![Axis(0), ys, array![r]];
                }
            }
        }
        break;
    }

    let weights = linear_regression(xs.clone(), ys.clone());
    println!("{:?}", weights);
    println!("R^2: {:?}", eval_r2(weights, xs, ys));
}
