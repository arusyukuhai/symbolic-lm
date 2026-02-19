use ndarray::prelude::*;
use numpy::{PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, ToPyArray};
use pyo3::prelude::*;
use std::f32::consts::PI;

#[pyfunction]
fn exec_grid_rnn_rust(
    gene1: PyReadonlyArray2<i32>,
    gene2: PyReadonlyArray2<i32>,
    active_nodes: PyReadonlyArray1<i32>,
    image: PyReadonlyArray3<f32>,
    hidden_dim: usize,
    out_dim: usize,
) -> PyResult<Py<PyArray3<f32>>> {
    let gene1 = gene1.as_array();
    let gene2 = gene2.as_array();
    let active_nodes = active_nodes.as_array();
    let image = image.as_array();

    let h = image.shape()[0];
    let w = image.shape()[1];
    let c = image.shape()[2];
    let n_nodes = gene1.shape()[0];
    let n_inputs = c + 2 * hidden_dim;
    let total_calc = n_inputs + n_nodes;
    let last_k = out_dim + hidden_dim;

    let mut hidden = Array3::<f32>::zeros((h + 1, w + 1, hidden_dim));
    let mut output = Array3::<f32>::zeros((h, w, out_dim));
    let mut calculated = Array1::<f32>::zeros(total_calc);

    for y in 0..h {
        for x in 0..w {
            // Set Inputs
            for i in 0..c {
                calculated[i] = image[[y, x, i]];
            }
            for i in 0..hidden_dim {
                calculated[c + i] = hidden[[y, x + 1, i]]; // top
                calculated[c + hidden_dim + i] = hidden[[y + 1, x, i]]; // left
            }

            // Exec Active Nodes
            for &n_val in active_nodes.iter() {
                let n = n_val as usize;
                if n < n_inputs {
                    continue;
                }

                let local_n = n - n_inputs;
                let fid = gene2[[local_n, 0]];
                let val_x = calculated[gene1[[local_n, 0]] as usize];
                let val_y = calculated[gene1[[local_n, 1]] as usize];
                let val_z = calculated[gene1[[local_n, 2]] as usize];

                calculated[n] = exec_node_rs(fid, val_x, val_y, val_z);
            }

            // Store Results
            for i in 0..out_dim {
                output[[y, x, i]] = calculated[total_calc - last_k + i];
            }
            for i in 0..hidden_dim {
                hidden[[y + 1, x + 1, i]] = calculated[total_calc - last_k + out_dim + i];
            }
        }
    }

    Ok(Python::with_gil(|py| output.to_pyarray(py).to_owned()))
}

fn exec_node_rs(fid: i32, x: f32, y: f32, z: f32) -> f32 {
    match fid {
        // i0__
        0 => x.sin(),
        1 => x.cos(),
        2 => (x * PI).sin(),
        3 => (x * PI).cos(),
        4 => x * 2.0,
        5 => x * 10.0,
        6 => x * 0.1,
        7 => x * 0.5,
        8 => x * 0.9,
        9 => x + 1.0,
        10 => x - 1.0,
        11 => x + 10.0,
        12 => x - 10.0,
        13 => -x,
        14 => x / 2.0,
        15 => x.tanh(),
        16 => x * x,
        17 => x.abs(),
        18 => x.abs().sqrt(),
        19 => x.abs().powf(1.0 / 3.0) * x.signum(),
        20 => (x * x + 1e-12).ln(),
        21 => x,
        22 => x.max(0.0),
        23 => x.min(0.0),
        24 => x + x.sin().powi(2),

        // i1__
        25 => x + y,
        26 => x - y,
        27 => x * y,
        28 => x + y * 0.5,
        29 => x + y * 0.1,
        30 => x + y * 0.9,
        31 => x / (y * y + 1e-12),
        32 => x.max(y),
        33 => x.min(y),
        34 => (x * x + y * y).sqrt(),
        35 => (x + y) / 2.0,
        36 => (x - y).powi(2),
        37 => x * y.tanh(),
        38 => x * (y.tanh() + 1.0),
        39 => (x * PI * y).sin(),
        40 => (x * PI * y).cos(),

        // i2__
        41 => x + y + z,
        42 => x + y - z,
        43 => x - y - z,
        44 => x * y * z,
        45 => x * (y * y + 1e-12) / (z * z + 1e-12),
        46 => x / (y * y + z * z + 1e-12),
        47 => x.max(y).max(z),
        48 => x.min(y).min(z),
        49 => x.max(y.min(z)),
        50 => (x * x + y * y + z * z).sqrt(),
        51 => ((x - y).powi(2) + (y - z).powi(2) + (z - x).powi(2)).sqrt(),
        52 => {
            let m = x * y * z;
            m.abs().powf(1.0 / 3.0) * m.signum()
        }
        53 => {
            let m = (x - y) * (y - z) * (z - x);
            m.abs().powf(1.0 / 3.0) * m.signum()
        }
        54 => (x + y + z) / 3.0,
        55 => (x * PI).sin() * (1.0 + y.tanh()) + (z * PI).cos() * (1.0 - x.tanh()),

        _ => 0.0,
    }
}

#[pymodule]
fn rs_cgp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(exec_grid_rnn_rust, m)?)?;
    Ok(())
}
