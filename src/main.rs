use anyhow::Result;
use hashbrown::HashMap;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use rustc_hash::FxBuildHasher;
use rustfft::num_traits::ToPrimitive;
use rustfft::{num_complex::Complex32, FftPlanner};

use std::time::Instant;

// =========================
// Data gen (same spirit)
// =========================

fn generatetekito(rng: &mut impl Rng) -> String {
    // Pythonの「1..2^(randint(4,32))」に近いが、オーバーフロー回避のためu64で
    let pow = rng.gen_range(4..32);
    let s1: u64 = rng.gen_range(1..(1u64 << pow));
    let pow2 = rng.gen_range(4..32);
    let s2: u64 = rng.gen_range(1..(1u64 << pow2));
    let pow3 = rng.gen_range(4..32);
    let s3: u64 = rng.gen_range(1..(1u64 << pow3));

    let enz = rng.gen_range(0..13);
    match enz {
        0 => format!("{s1}+{s2}={}", s1 + s2),
        1 => format!("{s1}-{s2}={}", s1.wrapping_sub(s2)),
        2 => format!("{s1}*{s2}={}", s1.saturating_mul(s2)),
        3 => format!("{s1}/{s2}={}", s1 / s2),
        4 => format!("{s1}+{s2}+{s3}={}", s1 + s2 + s3),
        5 => format!("{s1}+{s2}-{s3}={}", (s1 + s2).wrapping_sub(s3)),
        6 => format!("{s1}+{s2}*{s3}={}", s1 + s2.saturating_mul(s3)),
        7 => format!("{s1}*{s2}+{s3}={}", s1.saturating_mul(s2) + s3),
        8 => format!(
            "{s1}*{s2}-{s3}={}",
            (s1.saturating_mul(s2)).wrapping_sub(s3)
        ),
        9 => format!("{s1}-{s2}*{s3}={}", s1.wrapping_sub(s2.saturating_mul(s3))),
        10 => format!("{s1}*{s2}/{s3}={}", (s1.saturating_mul(s2)) / s3.max(1)),
        11 => format!("{s1}-{s2}/{s3}={}", s1.wrapping_sub(s2 / s3.max(1))),
        _ => format!("{s1}/{s2}*{s3}={}", (s1 / s2.max(1)).saturating_mul(s3)),
    }
}

// token mapping: 0..9 + + - * / =  => 15
fn char_to_tok(c: char) -> Option<usize> {
    match c {
        '0'..='9' => Some((c as u8 - b'0') as usize),
        '+' => Some(10),
        '-' => Some(11),
        '*' => Some(12),
        '/' => Some(13),
        '=' => Some(14),
        _ => None,
    }
}

fn gendata(rng: &mut impl Rng, g: &str, tt: usize) -> (Vec<Vec<f32>>, f32) {
    // Python同様：文字列をトークン列にして、tt箇所をランダム置換し、one-hot(15,L) を返す
    let toks: Vec<usize> = g.chars().filter_map(char_to_tok).collect();
    let l = toks.len().max(1);
    let score = ((l.saturating_sub(tt)) as f32) / (l as f32);

    let mut at = toks.clone();
    let mut perm: Vec<usize> = (0..l).collect();
    perm.shuffle(rng);

    for i in 0..tt.min(l) {
        let idx = perm[i];
        let old = at[idx];
        let mut t = rng.gen_range(0..15);
        while t == old {
            t = rng.gen_range(0..15);
        }
        at[idx] = t;
    }

    let mut gt = vec![vec![0f32; l]; 15];
    for (j, &tok) in at.iter().enumerate() {
        gt[tok][j] = 1.0;
    }
    (gt, score)
}

// =========================
// Correlations (Pearson / Spearman / Chatterjee xi)
// =========================

fn pearson_abs(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    if n < 2 {
        return 0.0;
    }
    let mut ma = 0.0f64;
    let mut mb = 0.0f64;
    for i in 0..n {
        ma += a[i] as f64;
        mb += b[i] as f64;
    }
    ma /= n as f64;
    mb /= n as f64;
    let mut cov = 0.0f64;
    let mut va = 0.0f64;
    let mut vb = 0.0f64;
    for i in 0..n {
        let da = a[i] as f64 - ma;
        let db = b[i] as f64 - mb;
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    if va <= 1e-18 || vb <= 1e-18 {
        return 0.0;
    }
    let c = cov / (va.sqrt() * vb.sqrt());
    if c.is_finite() {
        c.abs()
    } else {
        0.0
    }
}

fn argsort_ordinal(values: &[f32], idx: &mut Vec<usize>) {
    idx.clear();
    idx.extend(0..values.len());
    idx.sort_by(|&i, &j| values[i].total_cmp(&values[j]));
}

// ordinal ranks (1..n), tieは順序依存（Python rankdata(method="ordinal") に近い）
fn ordinal_ranks(values: &[f32], idx_buf: &mut Vec<usize>, out_rank: &mut Vec<i32>) {
    let n = values.len();
    argsort_ordinal(values, idx_buf);
    out_rank.clear();
    out_rank.resize(n, 0);
    for (r, &i) in idx_buf.iter().enumerate() {
        out_rank[i] = (r + 1) as i32;
    }
}

fn spearman_abs(
    a: &[f32],
    b: &[f32],
    idx_buf: &mut Vec<usize>,
    ra: &mut Vec<i32>,
    rb: &mut Vec<i32>,
) -> f64 {
    let n = a.len().min(b.len());
    if n < 2 {
        return 0.0;
    }
    ordinal_ranks(&a[..n], idx_buf, ra);
    ordinal_ranks(&b[..n], idx_buf, rb);

    // Pearson on ranks
    let mut fa = vec![0f32; n];
    let mut fb = vec![0f32; n];
    for i in 0..n {
        fa[i] = ra[i] as f32;
        fb[i] = rb[i] as f32;
    }
    pearson_abs(&fa, &fb)
}

// Chatterjee xi: sort by x, compute ordinal ranks of y_sorted, sum |diff|
fn chatterjee_xi(x: &[f32], y: &[f32], idx_buf: &mut Vec<usize>, r: &mut Vec<i32>) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }

    // sort indices by x
    idx_buf.clear();
    idx_buf.extend(0..n);
    idx_buf.sort_by(|&i, &j| x[i].total_cmp(&x[j]));

    // y_sorted
    let mut y_sorted = vec![0f32; n];
    for (k, &i) in idx_buf.iter().enumerate() {
        y_sorted[k] = y[i];
    }

    // ranks of y_sorted (ordinal)
    ordinal_ranks(&y_sorted, idx_buf, r);

    let mut diff_sum: i64 = 0;
    for i in 0..(n - 1) {
        diff_sum += (r[i + 1] - r[i]).abs() as i64;
    }

    let denom = (n as i64 * n as i64 - 1) as f64;
    if denom <= 0.0 {
        return 0.0;
    }
    let xi = 1.0 - (3.0 * (diff_sum as f64)) / denom;
    if xi.is_finite() {
        xi
    } else {
        0.0
    }
}

fn safe_corr(a: &[f32], b: &[f32]) -> f64 {
    let mut idx = Vec::new();
    let mut ra = Vec::new();
    let mut rb = Vec::new();
    let mut r1 = Vec::new();
    let mut r2 = Vec::new();

    let p = pearson_abs(a, b);
    let s = spearman_abs(a, b, &mut idx, &mut ra, &mut rb);
    let xi_ab = chatterjee_xi(a, b, &mut idx, &mut r1).max(0.0);
    let xi_ba = chatterjee_xi(b, a, &mut idx, &mut r2).max(0.0);

    let v = s * p * xi_ab * xi_ba;
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

// =========================
// Ops (subset+拡張しやすい設計)
// =========================

#[derive(Clone, Debug)]
enum Op1 {
    Add(f32),
    Mul(f32),
    Pow2,
    Pow3,
    Neg,
    InvSqEps(f32),
    Abs,
    Relu,
    Tanh,
    SinPi,
    Sort,
    ArgsortAsF32, // argsort -> float
    RankAsF32,    // argsort(argsort)
    FftRealScaled,
    FftImagScaled,
    IfftPowSpec, // ifft(|fft|^2).real (近似)
    TT,
    TT2,
    Flip,
    Rotate(isize),       // +: right, -: left
    EvenOddConcat(bool), // false: [even,odd], true:[odd,even]
    MeanConst,
    StdConst,
    Center,
    AddMean,
    MulMean,
    MulStd,
    LogStdConst,
    ZScore,
    L2Norm,
    Cumsum,
    CumprodNorm,
    ScaleByLen,
    DivByLen,
}

#[derive(Clone, Debug)]
enum Op2 {
    Add,
    Sub,
    Mul,
    DivSqEps(f32),
    DivAbsEps(f32),
    SqDiff,
    Mean,
    Mix(f32), // x*(1-a)+y*a
    Hypot,
    Max,
    Min,
    TakeByArgsortY,          // take(x, argsort(y))
    TakeTTByArgsortY,        // take(TT(take(x,argsort(y))), inv_argsort(y))
    Convolve,                // ifft(fft(x)*fft(y))/L
    Xcorr,                   // ifft(fft(x)*conj(fft(y)))/L
    SinPiMul,                // sin(pi*x*y)
    PolyAttn { deg: usize }, // poly regression attention (k=x, v=y, q=y)
}

#[derive(Clone, Debug)]
enum Op3 {
    Mean3,
    PairwiseDist,
    GeomMeanSigned,
    XPlusYMinusZ,
    XPlusHalfYMinusZ,
    Norm3,
    ConvolveDivZeps(f32),     // ifft(fft(x)*fft(y)/ (fft(z)+eps)) (近似)
    PolyAttn { deg: usize },  // k=x,v=y,q=z
    TakeTakeSort,             // take(take(x, argsort(y)), inv_argsort(z))
    PolyAttnB { deg: usize }, // k=x,v=y,q=z
}

// Workspace: 使い回すバッファ類
struct Workspace {
    idx: Vec<usize>,
    rank_i32: Vec<i32>,
    tmp: Vec<f32>,
    fft_planner: FftPlanner<f32>,
    fft_buf: Vec<Complex32>,
    fft_buf2: Vec<Complex32>,
}

impl Workspace {
    fn new() -> Self {
        Self {
            idx: Vec::new(),
            rank_i32: Vec::new(),
            tmp: Vec::new(),
            fft_planner: FftPlanner::new(),
            fft_buf: Vec::new(),
            fft_buf2: Vec::new(),
        }
    }
}

fn ensure_len(dst: &mut Vec<f32>, l: usize) {
    dst.clear();
    dst.resize(l, 0.0);
}

fn tt(src: &[f32], dst: &mut Vec<f32>) {
    let l = src.len();
    ensure_len(dst, l);
    if l < 3 {
        dst.copy_from_slice(src);
        return;
    }
    dst[0] = 1.0;
    dst[l - 1] = 1.0;
    for i in 1..(l - 1) {
        dst[i] = 0.25 * (src[i - 1] + 2.0 * src[i] + src[i + 1]);
    }
}

fn tt2(src: &[f32], dst: &mut Vec<f32>) {
    let l = src.len();
    ensure_len(dst, l);
    if l < 3 {
        dst.copy_from_slice(src);
        return;
    }
    dst[0] = 1.0;
    dst[l - 1] = 1.0;
    for i in 1..(l - 1) {
        let a = src[i - 1] - src[i];
        let b = src[i] - src[i + 1];
        dst[i] = 0.5 * (a * a + b * b);
    }
}

fn rotate(src: &[f32], dst: &mut Vec<f32>, shift: isize) {
    let l = src.len();
    ensure_len(dst, l);
    if l == 0 {
        return;
    }
    let s = ((shift % l as isize) + l as isize) % l as isize;
    let s = s as usize;
    // right rotate by s
    for i in 0..l {
        let j = (i + l - s) % l;
        dst[i] = src[j];
    }
}

fn even_odd_concat(src: &[f32], dst: &mut Vec<f32>, swap: bool) {
    let l = src.len();
    ensure_len(dst, l);
    let mut k = 0;
    if !swap {
        for i in (0..l).step_by(2) {
            dst[k] = src[i];
            k += 1;
        }
        for i in (1..l).step_by(2) {
            dst[k] = src[i];
            k += 1;
        }
    } else {
        for i in (1..l).step_by(2) {
            dst[k] = src[i];
            k += 1;
        }
        for i in (0..l).step_by(2) {
            dst[k] = src[i];
            k += 1;
        }
    }
}

fn argsort_as_f32(src: &[f32], ws: &mut Workspace, dst: &mut Vec<f32>) {
    let l = src.len();
    ensure_len(dst, l);
    argsort_ordinal(src, &mut ws.idx);
    for (r, &i) in ws.idx.iter().enumerate() {
        dst[r] = i as f32;
    }
}

fn rank_as_f32(src: &[f32], ws: &mut Workspace, dst: &mut Vec<f32>) {
    let l = src.len();
    ensure_len(dst, l);
    ordinal_ranks(src, &mut ws.idx, &mut ws.rank_i32);
    for i in 0..l {
        dst[i] = ws.rank_i32[i] as f32;
    }
}

fn fft_real_imag_scaled(src: &[f32], ws: &mut Workspace, dst: &mut Vec<f32>, imag: bool) {
    let l = src.len();
    ensure_len(dst, l);
    ws.fft_buf.clear();
    ws.fft_buf.resize(l, Complex32::new(0.0, 0.0));
    for i in 0..l {
        ws.fft_buf[i].re = src[i];
    }
    let fft = ws.fft_planner.plan_fft_forward(l);
    fft.process(&mut ws.fft_buf);
    let scale = (l as f32).max(1.0);
    for i in 0..l {
        dst[i] = if imag {
            ws.fft_buf[i].im / scale
        } else {
            ws.fft_buf[i].re / scale
        };
    }
}

// numpy irfft(|rfft(x)|^2) の近似として ifft(|fft(x)|^2).real を使う（スケールは厳密一致しない）
fn ifft_pow_spec(src: &[f32], ws: &mut Workspace, dst: &mut Vec<f32>) {
    let l = src.len();
    ensure_len(dst, l);
    ws.fft_buf.clear();
    ws.fft_buf.resize(l, Complex32::new(0.0, 0.0));
    for i in 0..l {
        ws.fft_buf[i].re = src[i];
    }
    let fft = ws.fft_planner.plan_fft_forward(l);
    fft.process(&mut ws.fft_buf);

    for z in &mut ws.fft_buf {
        // |F|^2 as complex (real)
        let mag2 = z.re * z.re + z.im * z.im;
        *z = Complex32::new(mag2, 0.0);
    }

    let ifft = ws.fft_planner.plan_fft_inverse(l);
    ifft.process(&mut ws.fft_buf);

    let scale = (l as f32).max(1.0);
    for i in 0..l {
        dst[i] = ws.fft_buf[i].re / scale;
    }
}

fn convolve(x: &[f32], y: &[f32], ws: &mut Workspace, dst: &mut Vec<f32>, conj_y: bool) {
    let l = x.len().min(y.len());
    ensure_len(dst, l);
    ws.fft_buf.clear();
    ws.fft_buf.resize(l, Complex32::new(0.0, 0.0));
    ws.fft_buf2.clear();
    ws.fft_buf2.resize(l, Complex32::new(0.0, 0.0));

    for i in 0..l {
        ws.fft_buf[i].re = x[i];
        ws.fft_buf2[i].re = y[i];
    }
    let fft = ws.fft_planner.plan_fft_forward(l);
    fft.process(&mut ws.fft_buf);
    fft.process(&mut ws.fft_buf2);

    for i in 0..l {
        let a = ws.fft_buf[i];
        let mut b = ws.fft_buf2[i];
        if conj_y {
            b.im = -b.im;
        }
        ws.fft_buf[i] = a * b;
    }
    let ifft = ws.fft_planner.plan_fft_inverse(l);
    ifft.process(&mut ws.fft_buf);

    let scale = (l as f32).max(1.0);
    for i in 0..l {
        dst[i] = ws.fft_buf[i].re / scale;
    }
}

// poly attention (Pythonの _attn_poly_fast 相当の近似)
fn solve_gauss(a: &mut [f64], b: &mut [f64], n: usize) -> bool {
    // a: row-major n*n
    for i in 0..n {
        // pivot
        let mut piv = i;
        let mut best = a[i * n + i].abs();
        for r in (i + 1)..n {
            let v = a[r * n + i].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best < 1e-18 {
            return false;
        }
        if piv != i {
            for c in 0..n {
                a.swap(i * n + c, piv * n + c);
            }
            b.swap(i, piv);
        }
        // normalize row i
        let diag = a[i * n + i];
        for c in i..n {
            a[i * n + c] /= diag;
        }
        b[i] /= diag;
        // eliminate
        for r in 0..n {
            if r == i {
                continue;
            }
            let f = a[r * n + i];
            if f.abs() < 1e-18 {
                continue;
            }
            for c in i..n {
                a[r * n + c] -= f * a[i * n + c];
            }
            b[r] -= f * b[i];
        }
    }
    true
}

fn poly_attn(k: &[f32], v: &[f32], q: &[f32], deg: usize, dst: &mut Vec<f32>) {
    let l = k.len().min(v.len()).min(q.len());
    ensure_len(dst, l);
    if l == 0 {
        return;
    }

    let mut scale = 0.0f32;
    for &x in &k[..l] {
        scale = scale.max(x.abs());
    }
    scale = (scale + 1e-12).max(1e-12);

    let size = deg + 1;
    let max_pow = 2 * deg;

    // moments s[m] = sum kn^m
    let mut s = vec![0f64; max_pow + 1];
    let mut bvec = vec![0f64; size];

    for i in 0..l {
        let kn = (k[i] / scale) as f64;
        let mut p = 1.0f64;
        for m in 0..=max_pow {
            s[m] += p;
            if m <= deg {
                bvec[m] += (v[i] as f64) * p;
            }
            p *= kn;
        }
    }

    // A[i,j] = s[i+j] + ridge*I
    let mut amat = vec![0f64; size * size];
    for i in 0..size {
        for j in 0..size {
            amat[i * size + j] = s[i + j];
        }
    }
    let mut tr = 0.0;
    for i in 0..size {
        tr += amat[i * size + i];
    }
    let ridge = (1e-8 * tr / (size as f64)).max(1e-10);
    for i in 0..size {
        amat[i * size + i] += ridge;
    }

    // solve
    let mut a2 = amat.clone();
    let mut coef = bvec.clone();
    if !solve_gauss(&mut a2, &mut coef, size) {
        // fallback: zeros
        return;
    }

    // evaluate at qn
    for i in 0..l {
        let qn = (q[i] / scale) as f64;
        let mut p = 1.0f64;
        let mut y = 0.0f64;
        for j in 0..size {
            y += coef[j] * p;
            p *= qn;
        }
        dst[i] = y as f32;
    }
}

fn apply_op1(op: &Op1, x: &[f32], ws: &mut Workspace, dst: &mut Vec<f32>) {
    let l = x.len();
    match *op {
        Op1::Add(a) => {
            ensure_len(dst, l);
            for i in 0..l {
                dst[i] = x[i] + a;
            }
        }
        Op1::Mul(a) => {
            ensure_len(dst, l);
            for i in 0..l {
                dst[i] = x[i] * a;
            }
        }
        Op1::Pow2 => {
            ensure_len(dst, l);
            for i in 0..l {
                dst[i] = x[i] * x[i];
            }
        }
        Op1::Pow3 => {
            ensure_len(dst, l);
            for i in 0..l {
                dst[i] = x[i] * x[i] * x[i];
            }
        }
        Op1::Neg => {
            ensure_len(dst, l);
            for i in 0..l {
                dst[i] = -x[i];
            }
        }
        Op1::InvSqEps(eps) => {
            ensure_len(dst, l);
            for i in 0..l {
                dst[i] = 1.0 / (x[i] * x[i] + eps);
            }
        }
        Op1::Abs => {
            ensure_len(dst, l);
            for i in 0..l {
                dst[i] = x[i].abs();
            }
        }
        Op1::Relu => {
            ensure_len(dst, l);
            for i in 0..l {
                dst[i] = x[i].max(0.0);
            }
        }
        Op1::Tanh => {
            ensure_len(dst, l);
            for i in 0..l {
                dst[i] = x[i].tanh();
            }
        }
        Op1::SinPi => {
            ensure_len(dst, l);
            for i in 0..l {
                dst[i] = (std::f32::consts::PI * x[i]).sin();
            }
        }
        Op1::Sort => {
            dst.clear();
            dst.extend_from_slice(x);
            dst.sort_by(|a, b| a.total_cmp(b));
        }
        Op1::ArgsortAsF32 => argsort_as_f32(x, ws, dst),
        Op1::RankAsF32 => rank_as_f32(x, ws, dst),
        Op1::FftRealScaled => fft_real_imag_scaled(x, ws, dst, false),
        Op1::FftImagScaled => fft_real_imag_scaled(x, ws, dst, true),
        Op1::IfftPowSpec => ifft_pow_spec(x, ws, dst),
        Op1::TT => tt(x, dst),
        Op1::TT2 => tt2(x, dst),
        Op1::Flip => {
            ensure_len(dst, l);
            for i in 0..l {
                dst[i] = x[l - 1 - i];
            }
        }
        Op1::Rotate(s) => rotate(x, dst, s),
        Op1::EvenOddConcat(swap) => even_odd_concat(x, dst, swap),
        Op1::MeanConst => {
            let mut m = 0.0f64;
            for &v in x {
                m += v as f64;
            }
            m /= (l as f64).max(1.0);
            ensure_len(dst, l);
            for i in 0..l {
                dst[i] = m as f32;
            }
        }
        Op1::StdConst => {
            let mut m = 0.0f64;
            for &v in x {
                m += v as f64;
            }
            m /= (l as f64).max(1.0);
            let mut v = 0.0f64;
            for &t in x {
                let d = t as f64 - m;
                v += d * d;
            }
            v /= (l as f64).max(1.0);
            let s = v.sqrt() as f32;
            ensure_len(dst, l);
            for i in 0..l {
                dst[i] = s;
            }
        }
        Op1::Center => {
            let mut m = 0.0f64;
            for &v in x {
                m += v as f64;
            }
            m /= (l as f64).max(1.0);
            ensure_len(dst, l);
            for i in 0..l {
                dst[i] = x[i] - m as f32;
            }
        }
        Op1::AddMean => {
            let mut m = 0.0f64;
            for &v in x {
                m += v as f64;
            }
            m /= (l as f64).max(1.0);
            ensure_len(dst, l);
            for i in 0..l {
                dst[i] = x[i] + m as f32;
            }
        }
        Op1::MulMean => {
            let mut m = 0.0f64;
            for &v in x {
                m += v as f64;
            }
            m /= (l as f64).max(1.0);
            ensure_len(dst, l);
            for i in 0..l {
                dst[i] = x[i] * m as f32;
            }
        }
        Op1::MulStd => {
            let mut m = 0.0f64;
            for &v in x {
                m += v as f64;
            }
            m /= (l as f64).max(1.0);
            let mut v = 0.0f64;
            for &t in x {
                let d = t as f64 - m;
                v += d * d;
            }
            v /= (l as f64).max(1.0);
            let s = v.sqrt() as f32;
            ensure_len(dst, l);
            for i in 0..l {
                dst[i] = x[i] * s;
            }
        }
        Op1::LogStdConst => {
            let mut m = 0.0f64;
            for &v in x {
                m += v as f64;
            }
            m /= (l as f64).max(1.0);
            let mut v = 0.0f64;
            for &t in x {
                let d = t as f64 - m;
                v += d * d;
            }
            v /= (l as f64).max(1.0);
            let s = (v.sqrt() + 1e-12).ln() as f32;
            ensure_len(dst, l);
            for i in 0..l {
                dst[i] = s;
            }
        }
        Op1::ZScore => {
            let mut m = 0.0f64;
            for &v in x {
                m += v as f64;
            }
            m /= (l as f64).max(1.0);
            let mut v = 0.0f64;
            for &t in x {
                let d = t as f64 - m;
                v += d * d;
            }
            v /= (l as f64).max(1.0);
            let s = (v.sqrt() as f32 + 1e-12);
            ensure_len(dst, l);
            for i in 0..l {
                dst[i] = (x[i] - m as f32) / s;
            }
        }
        Op1::L2Norm => {
            let mut ss = 0.0f64;
            for &v in x {
                ss += (v as f64) * (v as f64);
            }
            let nrm = (ss.sqrt() as f32 + 1e-12);
            ensure_len(dst, l);
            for i in 0..l {
                dst[i] = x[i] / nrm;
            }
        }
        Op1::Cumsum => {
            ensure_len(dst, l);
            let mut s = 0.0f32;
            for i in 0..l {
                s += x[i];
                dst[i] = s;
            }
        }
        Op1::CumprodNorm => {
            // Pythonの「cumprod(x / sqrt(mean(x^2)))」に近い
            let mut ss = 0.0f64;
            for &v in x {
                ss += (v as f64) * (v as f64);
            }
            let denom = (ss / (l as f64).max(1.0)).sqrt() as f32 + 1e-12;
            ensure_len(dst, l);
            let mut p = 1.0f32;
            for i in 0..l {
                p *= x[i] / denom;
                dst[i] = p;
            }
        }
        Op1::ScaleByLen => {
            ensure_len(dst, l);
            let a = l as f32;
            for i in 0..l {
                dst[i] = x[i] * a;
            }
        }
        Op1::DivByLen => {
            ensure_len(dst, l);
            let a = (l as f32).max(1.0);
            for i in 0..l {
                dst[i] = x[i] / a;
            }
        }
    }
}

fn apply_op2(op: &Op2, x: &[f32], y: &[f32], ws: &mut Workspace, dst: &mut Vec<f32>) {
    let l = x.len().min(y.len());
    match *op {
        Op2::Add => {
            ensure_len(dst, l);
            for i in 0..l {
                dst[i] = x[i] + y[i];
            }
        }
        Op2::Sub => {
            ensure_len(dst, l);
            for i in 0..l {
                dst[i] = x[i] - y[i];
            }
        }
        Op2::Mul => {
            ensure_len(dst, l);
            for i in 0..l {
                dst[i] = x[i] * y[i];
            }
        }
        Op2::DivSqEps(eps) => {
            ensure_len(dst, l);
            for i in 0..l {
                dst[i] = x[i] / (y[i] * y[i] + eps);
            }
        }
        Op2::DivAbsEps(eps) => {
            ensure_len(dst, l);
            for i in 0..l {
                dst[i] = x[i] / (y[i].abs() + eps);
            }
        }
        Op2::SqDiff => {
            ensure_len(dst, l);
            for i in 0..l {
                let d = x[i] - y[i];
                dst[i] = d * d;
            }
        }
        Op2::Mean => {
            ensure_len(dst, l);
            for i in 0..l {
                dst[i] = 0.5 * (x[i] + y[i]);
            }
        }
        Op2::Mix(a) => {
            ensure_len(dst, l);
            for i in 0..l {
                dst[i] = x[i] * (1.0 - a) + y[i] * a;
            }
        }
        Op2::Hypot => {
            ensure_len(dst, l);
            for i in 0..l {
                dst[i] = (x[i] * x[i] + y[i] * y[i]).sqrt();
            }
        }
        Op2::Max => {
            ensure_len(dst, l);
            for i in 0..l {
                dst[i] = x[i].max(y[i]);
            }
        }
        Op2::Min => {
            ensure_len(dst, l);
            for i in 0..l {
                dst[i] = x[i].min(y[i]);
            }
        }
        Op2::TakeByArgsortY => {
            ws.idx.clear();
            ws.idx.extend(0..l);
            ws.idx.sort_by(|&i, &j| y[i].total_cmp(&y[j]));
            ensure_len(dst, l);
            for (k, &i) in ws.idx.iter().enumerate() {
                dst[k] = x[i];
            }
        }
        Op2::TakeTTByArgsortY => {
            // take(x,argsort(y)) -> TT -> inv_argsort で戻す（Pythonの雰囲気）
            ws.idx.clear();
            ws.idx.extend(0..l);
            ws.idx.sort_by(|&i, &j| y[i].total_cmp(&y[j]));
            ws.tmp.clear();
            ws.tmp.resize(l, 0.0);
            for (k, &i) in ws.idx.iter().enumerate() {
                ws.tmp[k] = x[i];
            }
            let mut tmp2 = Vec::new();
            tt(&ws.tmp, &mut tmp2);
            // inv_argsort
            let mut inv = vec![0usize; l];
            for (k, &i) in ws.idx.iter().enumerate() {
                inv[i] = k;
            }
            ensure_len(dst, l);
            for i in 0..l {
                dst[i] = tmp2[inv[i]];
            }
        }
        Op2::Convolve => convolve(&x[..l], &y[..l], ws, dst, false),
        Op2::Xcorr => convolve(&x[..l], &y[..l], ws, dst, true),
        Op2::SinPiMul => {
            ensure_len(dst, l);
            for i in 0..l {
                dst[i] = (std::f32::consts::PI * x[i] * y[i]).sin();
            }
        }
        Op2::PolyAttn { deg } => {
            poly_attn(&x[..l], &y[..l], &y[..l], deg, dst);
        }
    }
}

fn apply_op3(op: &Op3, x: &[f32], y: &[f32], z: &[f32], ws: &mut Workspace, dst: &mut Vec<f32>) {
    let l = x.len().min(y.len()).min(z.len());
    match *op {
        Op3::Mean3 => {
            ensure_len(dst, l);
            for i in 0..l {
                dst[i] = (x[i] + y[i] + z[i]) / 3.0;
            }
        }
        Op3::PairwiseDist => {
            ensure_len(dst, l);
            for i in 0..l {
                let a = x[i] - y[i];
                let b = y[i] - z[i];
                let c = z[i] - x[i];
                dst[i] = (a * a + b * b + c * c).sqrt();
            }
        }
        Op3::GeomMeanSigned => {
            ensure_len(dst, l);
            for i in 0..l {
                let p = x[i] * y[i] * z[i];
                dst[i] = p.signum() * p.abs().powf(1.0 / 3.0);
            }
        }
        Op3::XPlusYMinusZ => {
            ensure_len(dst, l);
            for i in 0..l {
                dst[i] = x[i] + y[i] - z[i];
            }
        }
        Op3::XPlusHalfYMinusZ => {
            ensure_len(dst, l);
            for i in 0..l {
                dst[i] = x[i] + 0.5 * (y[i] - z[i]);
            }
        }
        Op3::Norm3 => {
            ensure_len(dst, l);
            for i in 0..l {
                dst[i] = (x[i] * x[i] + y[i] * y[i] + z[i] * z[i]).sqrt();
            }
        }
        Op3::ConvolveDivZeps(eps) => {
            // fft(x)*fft(y)/(fft(z)+eps) 近似
            let l = l;
            ensure_len(dst, l);
            ws.fft_buf.clear();
            ws.fft_buf.resize(l, Complex32::new(0.0, 0.0));
            ws.fft_buf2.clear();
            ws.fft_buf2.resize(l, Complex32::new(0.0, 0.0));
            let mut bufz = vec![Complex32::new(0.0, 0.0); l];
            for i in 0..l {
                ws.fft_buf[i].re = x[i];
                ws.fft_buf2[i].re = y[i];
                bufz[i].re = z[i];
            }
            let fft = ws.fft_planner.plan_fft_forward(l);
            fft.process(&mut ws.fft_buf);
            fft.process(&mut ws.fft_buf2);
            fft.process(&mut bufz);

            for i in 0..l {
                let num = ws.fft_buf[i] * ws.fft_buf2[i];
                let den = Complex32::new(bufz[i].re + eps, bufz[i].im);
                ws.fft_buf[i] = num / den;
            }
            let ifft = ws.fft_planner.plan_fft_inverse(l);
            ifft.process(&mut ws.fft_buf);
            let scale = (l as f32).max(1.0);
            for i in 0..l {
                dst[i] = ws.fft_buf[i].re / scale;
            }
        }
        Op3::PolyAttn { deg } => {
            poly_attn(&x[..l], &y[..l], &z[..l], deg, dst);
        }
        Op3::PolyAttnB { deg } => {
            let fdeg = deg as f32;
            let tk: Vec<_> = x[..l].iter().map(|&v| v.powf(fdeg)).collect();
            let tv: Vec<_> = y[..l].iter().map(|&v| v.powf(fdeg)).collect();
            let tq: Vec<_> = z[..l].iter().map(|&v| v.powf(fdeg)).collect();
            poly_attn(&tk, &tv, &tq, deg, dst);
            for v in dst.iter_mut() {
                *v = v.signum() * v.abs().powf(1.0 / fdeg);
            }
        }
        Op3::TakeTakeSort => {
            // take(take(x, argsort(y)), inv_argsort(z)) の雰囲気（厳密一致ではない）
            ws.idx.clear();
            ws.idx.extend(0..l);
            ws.idx.sort_by(|&i, &j| y[i].total_cmp(&y[j]));
            ws.tmp.clear();
            ws.tmp.resize(l, 0.0);
            for (k, &i) in ws.idx.iter().enumerate() {
                ws.tmp[k] = x[i];
            }

            let mut idxz = (0..l).collect::<Vec<_>>();
            idxz.sort_by(|&i, &j| z[i].total_cmp(&z[j]));
            let mut invz = vec![0usize; l];
            for (k, &i) in idxz.iter().enumerate() {
                invz[i] = k;
            }

            ensure_len(dst, l);
            for i in 0..l {
                dst[i] = ws.tmp[invz[i]];
            }
        }
    }
}

fn make_ops() -> (Vec<Op1>, Vec<Op2>, Vec<Op3>) {
    // Pythonの funcs_1 / funcs_2 / funcs_3 を “近い雰囲気” で列挙
    // 必要ならここに追加していく（op_id の安定性が必要なら順序固定）
    let mut u = Vec::new();
    u.push(Op1::Add(1.0));
    u.push(Op1::Mul(2.0));
    u.push(Op1::Pow2);
    u.push(Op1::Neg);
    u.push(Op1::InvSqEps(1e-12));
    u.push(Op1::Sort);
    u.push(Op1::ArgsortAsF32);
    u.push(Op1::RankAsF32);
    u.push(Op1::FftRealScaled);
    u.push(Op1::FftImagScaled);
    u.push(Op1::TT);
    u.push(Op1::TT2);
    // TT2(TT2(x)) / TT(TT(x)) 等は GA が作れるので省略してもよいが、入れておく
    u.push(Op1::TT2);
    u.push(Op1::TT);
    u.push(Op1::Mul(0.5));
    u.push(Op1::Flip);
    u.push(Op1::Rotate(1));
    u.push(Op1::Rotate(-1));
    u.push(Op1::Rotate(2));
    u.push(Op1::Rotate(-2));
    u.push(Op1::Rotate(4));
    u.push(Op1::Rotate(-4));
    u.push(Op1::Rotate(8));
    u.push(Op1::Rotate(-8));
    u.push(Op1::Rotate(16));
    u.push(Op1::Rotate(-16));
    u.push(Op1::Rotate(32));
    u.push(Op1::Rotate(-32));
    u.push(Op1::EvenOddConcat(false));
    u.push(Op1::EvenOddConcat(true));
    u.push(Op1::MeanConst);
    u.push(Op1::StdConst);
    u.push(Op1::Center);
    u.push(Op1::AddMean);
    u.push(Op1::MulMean);
    u.push(Op1::MulStd);
    u.push(Op1::LogStdConst);
    u.push(Op1::Mul(0.1));
    u.push(Op1::Mul(0.01));
    u.push(Op1::Mul(0.001));
    u.push(Op1::ZScore);
    u.push(Op1::L2Norm);
    u.push(Op1::Mul(10.0));
    u.push(Op1::Pow3);
    u.push(Op1::SinPi);
    u.push(Op1::Add(-1.0));
    u.push(Op1::Mul(0.9));
    u.push(Op1::Mul(1.1));
    u.push(Op1::Tanh);
    u.push(Op1::IfftPowSpec);
    u.push(Op1::Cumsum);
    u.push(Op1::CumprodNorm);
    u.push(Op1::Abs);
    u.push(Op1::Relu);
    u.push(Op1::ScaleByLen);
    u.push(Op1::DivByLen);

    let mut b = Vec::new();
    b.push(Op2::Add);
    b.push(Op2::Sub);
    b.push(Op2::Mul);
    b.push(Op2::DivSqEps(1e-12));
    b.push(Op2::DivAbsEps(1e-12));
    b.push(Op2::SqDiff);
    b.push(Op2::Mean);
    b.push(Op2::Mix(0.9)); // x*0.1+y*0.9 の代替
    b.push(Op2::Mix(0.1)); // x*0.9+y*0.1 の代替
    b.push(Op2::Hypot);
    b.push(Op2::Convolve);
    b.push(Op2::Xcorr);
    b.push(Op2::TakeByArgsortY);
    b.push(Op2::TakeTTByArgsortY);
    b.push(Op2::Max);
    b.push(Op2::Min);
    b.push(Op2::SinPiMul);
    b.push(Op2::PolyAttn { deg: 3 });
    b.push(Op2::PolyAttn { deg: 5 });
    b.push(Op2::PolyAttn { deg: 7 });
    b.push(Op2::PolyAttn { deg: 11 });

    let mut t = Vec::new();
    t.push(Op3::PairwiseDist);
    t.push(Op3::Mean3);
    t.push(Op3::GeomMeanSigned);
    t.push(Op3::XPlusYMinusZ);
    t.push(Op3::XPlusHalfYMinusZ);
    t.push(Op3::Norm3);
    t.push(Op3::ConvolveDivZeps(1e-12));
    t.push(Op3::PolyAttn { deg: 3 });
    t.push(Op3::PolyAttn { deg: 5 });
    t.push(Op3::PolyAttn { deg: 7 });
    t.push(Op3::PolyAttn { deg: 11 });
    t.push(Op3::PolyAttnB { deg: 3 });
    t.push(Op3::PolyAttnB { deg: 5 });
    t.push(Op3::PolyAttnB { deg: 7 });
    t.push(Op3::PolyAttnB { deg: 11 });
    t.push(Op3::TakeTakeSort);

    (u, b, t)
}

// =========================
// Gene / struct graph
// =========================

#[derive(Clone)]
struct Gene {
    // refs[node][k] in [0,node)
    refs: Vec<[u16; 3]>,
    // op id in [0, total_ops)
    ops: Vec<u16>,
}

fn sample_categorical(rng: &mut impl Rng, probs: &[f32]) -> usize {
    let mut s = 0.0f32;
    let r: f32 = rng.gen::<f32>();
    for (i, &p) in probs.iter().enumerate() {
        s += p;
        if r <= s {
            return i;
        }
    }
    probs.len().saturating_sub(1)
}

fn make_uniform_probs(n: usize) -> Vec<f32> {
    let p = 1.0 / (n as f32).max(1.0);
    vec![p; n]
}

fn random_gene(rng: &mut impl Rng, modelllen: usize, num_inputs: usize, op_probs: &[f32]) -> Gene {
    let total_ops = op_probs.len();
    let mut refs = vec![[0u16; 3]; modelllen];
    let mut ops = vec![0u16; modelllen];

    for node in 0..modelllen {
        if node < num_inputs {
            refs[node] = [0, 0, 0];
            ops[node] = 0;
            continue;
        }
        let hi = node.min(65535).max(1);
        refs[node] = [
            rng.gen_range(0..hi) as u16,
            rng.gen_range(0..hi) as u16,
            rng.gen_range(0..hi) as u16,
        ];
        let oid = sample_categorical(rng, op_probs).min(total_ops - 1);
        ops[node] = oid as u16;
    }
    Gene { refs, ops }
}

#[derive(Clone, Copy, Debug)]
struct StructNode {
    stype: u8, // 0 input, 1 unary, 2 binary, 3 ternary
    func: u16, // local index
    c1: u32,
    c2: u32,
    c3: u32,
}

#[derive(Hash, PartialEq, Eq)]
struct Key {
    stype: u8,
    func: u16,
    c1: u32,
    c2: u32,
    c3: u32,
}

struct StructGraph {
    structs: Vec<StructNode>, // sid = index
    // node_sids[ind*MODELLEN + node] = sid or u32::MAX if unused
    node_sids: Vec<u32>,
    // last_sids[ind*last_k + j]
    last_sids: Vec<u32>,
    needed: Vec<bool>,
    num_inputs: usize,
    modelllen: usize,
    pop: usize,
    last_k: usize,
    len_u: usize,
    len_b: usize,
    len_t: usize,
}

fn op_info(oid: usize, len_u: usize, len_b: usize, len_t: usize) -> (u8, u16) {
    if oid < len_u {
        (1, oid as u16)
    } else if oid < len_u + len_b {
        (2, (oid - len_u) as u16)
    } else {
        let o = (oid - len_u - len_b).min(len_t.saturating_sub(1));
        (3, o as u16)
    }
}

fn build_struct_graph(
    genes: &[Gene],
    num_inputs: usize,
    last_k: usize,
    len_u: usize,
    len_b: usize,
    len_t: usize,
) -> StructGraph {
    let pop = genes.len();
    let modelllen = genes[0].ops.len();

    let mut structs: Vec<StructNode> = Vec::new();
    // sid 0..num_inputs-1 are inputs
    for _ in 0..num_inputs {
        structs.push(StructNode {
            stype: 0,
            func: 0,
            c1: 0,
            c2: 0,
            c3: 0,
        });
    }

    let mut map: HashMap<Key, u32, FxBuildHasher> =
        HashMap::with_capacity_and_hasher(pop * last_k * 4 + 1024, FxBuildHasher::default());

    let mut node_sids = vec![u32::MAX; pop * modelllen];
    let mut last_sids = vec![0u32; pop * last_k];

    let mut used = vec![false; modelllen];
    let mut stack: Vec<usize> = Vec::new();

    for (ind, g) in genes.iter().enumerate() {
        used.fill(false);
        stack.clear();

        let start = modelllen.saturating_sub(last_k).max(num_inputs);
        for n in start..modelllen {
            stack.push(n);
        }

        while let Some(n) = stack.pop() {
            if used[n] {
                continue;
            }
            used[n] = true;
            if n < num_inputs {
                continue;
            }
            let oid = (g.ops[n] as usize).min(len_u + len_b + len_t - 1);
            let (stype, _f) = op_info(oid, len_u, len_b, len_t);
            let r = g.refs[n];
            let a = r[0] as usize;
            let b = r[1] as usize;
            let c = r[2] as usize;
            match stype {
                1 => {
                    if a < n {
                        stack.push(a);
                    }
                }
                2 => {
                    if a < n {
                        stack.push(a);
                    }
                    if b < n {
                        stack.push(b);
                    }
                }
                _ => {
                    if a < n {
                        stack.push(a);
                    }
                    if b < n {
                        stack.push(b);
                    }
                    if c < n {
                        stack.push(c);
                    }
                }
            }
        }
        for i in 0..num_inputs {
            used[i] = true;
        }

        // inputs
        for i in 0..num_inputs {
            node_sids[ind * modelllen + i] = i as u32;
        }

        // build sids in node order (children < node 前提)
        for node in num_inputs..modelllen {
            if !used[node] {
                continue;
            }
            let oid = (g.ops[node] as usize).min(len_u + len_b + len_t - 1);
            let (stype, func) = op_info(oid, len_u, len_b, len_t);
            let r = g.refs[node];

            let a = r[0] as usize;
            let b = r[1] as usize;
            let c = r[2] as usize;

            let c1 = node_sids[ind * modelllen + a];
            let c2 = node_sids[ind * modelllen + b];
            let c3 = node_sids[ind * modelllen + c];

            let key = match stype {
                1 => Key {
                    stype,
                    func,
                    c1,
                    c2: 0,
                    c3: 0,
                },
                2 => Key {
                    stype,
                    func,
                    c1,
                    c2,
                    c3: 0,
                },
                _ => Key {
                    stype,
                    func,
                    c1,
                    c2,
                    c3,
                },
            };

            let sid = if let Some(&sid) = map.get(&key) {
                sid
            } else {
                let sid = structs.len() as u32;
                structs.push(StructNode {
                    stype,
                    func,
                    c1: key.c1,
                    c2: key.c2,
                    c3: key.c3,
                });
                map.insert(key, sid);
                sid
            };

            node_sids[ind * modelllen + node] = sid;
        }

        // last sids
        for j in 0..last_k {
            let node = modelllen - last_k + j;
            let sid = node_sids[ind * modelllen + node];
            last_sids[ind * last_k + j] = sid;
        }
    }

    // needed propagation: last_sids を起点に、sid降順で子をneededにする
    let s = structs.len();
    let mut needed = vec![false; s];
    for &sid in &last_sids {
        if sid != u32::MAX && (sid as usize) < s {
            needed[sid as usize] = true;
        }
    }
    for sid in (num_inputs..s).rev() {
        if !needed[sid] {
            continue;
        }
        let st = structs[sid];
        if st.stype >= 1 {
            needed[st.c1 as usize] = true;
            if st.stype >= 2 {
                needed[st.c2 as usize] = true;
            }
            if st.stype >= 3 {
                needed[st.c3 as usize] = true;
            }
        }
    }

    StructGraph {
        structs,
        node_sids,
        last_sids,
        needed,
        num_inputs,
        modelllen,
        pop,
        last_k,
        len_u,
        len_b,
        len_t,
    }
}

// =========================
// Execution
// =========================

struct ExecCtx<'a> {
    graph: &'a StructGraph,
    ops_u: &'a [Op1],
    ops_b: &'a [Op2],
    ops_t: &'a [Op3],
}

fn mean_f32(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    let mut s = 0.0f64;
    for &v in x {
        s += v as f64;
    }
    (s / x.len() as f64) as f32
}

// x_inputs: Vec<Vec<f32>> shape (num_inputs, L)
// returns logits[pop] for last_k=1 (必要なら拡張)
fn eval_one_sample(ctx: &ExecCtx, ws: &mut Workspace, x_inputs: &[Vec<f32>]) -> Vec<f32> {
    let g = ctx.graph;
    let s = g.structs.len();
    let num_inputs = g.num_inputs;

    let l = x_inputs[0].len();
    let mut outputs: Vec<Vec<f32>> = vec![Vec::new(); s];
    let mut means: Vec<f32> = vec![0.0; s];

    for i in 0..num_inputs {
        outputs[i] = x_inputs[i].clone();
        means[i] = mean_f32(&outputs[i]);
    }

    let mut out_tmp: Vec<f32> = Vec::new(); // <- 追加：一時出力

    for sid in num_inputs..s {
        if !g.needed[sid] {
            continue;
        }
        let st = g.structs[sid];

        out_tmp.clear();

        match st.stype {
            1 => {
                let c1 = st.c1 as usize;
                let x = &outputs[c1];
                apply_op1(&ctx.ops_u[st.func as usize], x, ws, &mut out_tmp);
            }
            2 => {
                let c1 = st.c1 as usize;
                let c2 = st.c2 as usize;
                let x = &outputs[c1];
                let y = &outputs[c2];
                apply_op2(&ctx.ops_b[st.func as usize], x, y, ws, &mut out_tmp);
            }
            _ => {
                let c1 = st.c1 as usize;
                let c2 = st.c2 as usize;
                let c3 = st.c3 as usize;
                let x = &outputs[c1];
                let y = &outputs[c2];
                let z = &outputs[c3];
                apply_op3(&ctx.ops_t[st.func as usize], x, y, z, ws, &mut out_tmp);
            }
        }

        if out_tmp.len() != l {
            out_tmp.resize(l, 0.0);
        }

        means[sid] = mean_f32(&out_tmp);

        // outputs[sid] にムーブ（swapで再利用）
        std::mem::swap(&mut outputs[sid], &mut out_tmp);
        // out_tmp は次ループで clear されるので outputs[sid] は保持される
    }

    let mut logits = vec![0f32; g.pop];
    for ind in 0..g.pop {
        let sid0 = g.last_sids[ind * g.last_k + 0];
        logits[ind] = if sid0 != u32::MAX {
            means[sid0 as usize]
        } else {
            0.0
        };
    }
    logits
}

// =========================
// GA (最小: elitism + tournament + crossover + mutation)
// =========================

fn mutate_gene(
    rng: &mut impl Rng,
    g: &mut Gene,
    num_inputs: usize,
    op_probs: &[f32],
    p_ref: f32,
    p_op: f32,
    p_seg: f32,
) {
    let modelllen = g.ops.len();
    let total_ops = op_probs.len();

    // point mutations
    for node in num_inputs..modelllen {
        if rng.gen::<f32>() < p_ref {
            let hi = node.min(65535).max(1);
            let which = rng.gen_range(0..3);
            g.refs[node][which] = rng.gen_range(0..hi) as u16;
        }
        if rng.gen::<f32>() < p_op {
            let oid = sample_categorical(rng, op_probs).min(total_ops - 1);
            g.ops[node] = oid as u16;
        }
    }

    // segment shuffle-ish
    if rng.gen::<f32>() < p_seg {
        let a = rng.gen_range(num_inputs..modelllen - 1);
        let b = rng.gen_range(a + 1..modelllen);
        for node in a..b {
            let oid = sample_categorical(rng, op_probs).min(total_ops - 1);
            g.ops[node] = oid as u16;
            let hi = node.min(65535).max(1);
            g.refs[node] = [
                rng.gen_range(0..hi) as u16,
                rng.gen_range(0..hi) as u16,
                rng.gen_range(0..hi) as u16,
            ];
        }
    }
}

fn crossover(rng: &mut impl Rng, a: &Gene, b: &Gene, num_inputs: usize) -> Gene {
    let modelllen = a.ops.len();
    let mut child = a.clone();
    let l = modelllen;
    let i = rng.gen_range(num_inputs..l - 1);
    let j = rng.gen_range(i + 1..l);
    child.ops[i..j].copy_from_slice(&b.ops[i..j]);
    child.refs[i..j].copy_from_slice(&b.refs[i..j]);
    child
}

fn tournament_select(rng: &mut impl Rng, scores: &[f64], k: usize) -> usize {
    let mut best = rng.gen_range(0..scores.len());
    for _ in 1..k {
        let j = rng.gen_range(0..scores.len());
        if scores[j] > scores[best] {
            best = j;
        }
    }
    best
}

// =========================
// Demo loop
// =========================

fn main() -> Result<()> {
    // parameters (あなたの超巨大設定は Rust 側でも可能だが、まずは小さく検証推奨)
    let modelllen = 100000usize;
    let pop = 512usize;
    let iters = 200000usize;
    let samples = 16384usize;
    let dataset = 1048576usize;
    let last_k = 1usize;
    let num_inputs = 15usize;

    let (ops_u, ops_b, ops_t) = make_ops();
    let total_ops = ops_u.len() + ops_b.len() + ops_t.len();
    let op_probs = make_uniform_probs(total_ops);

    let mut rng = ChaCha8Rng::seed_from_u64(0);

    // dataset
    println!("generating dataset...");
    let mut traindats: Vec<(Vec<Vec<f32>>, f32)> = Vec::with_capacity(dataset);
    for _ in 0..dataset {
        let gp = generatetekito(&mut rng);
        let tt = rng.gen_range(1..(gp.len() / 2).max(2));
        let (gt, score) = gendata(&mut rng, &gp, tt);
        traindats.push((gt, score));
    }

    // init population
    println!("init population...");
    let mut genes: Vec<Gene> = (0..pop)
        .map(|_| random_gene(&mut rng, modelllen, num_inputs, &op_probs))
        .collect();

    let mut best_score_ema = 0.0f64;

    for step in 0..iters {
        // pick batch
        let batch_idx = step % (dataset / samples).max(1);
        let start = batch_idx * samples;
        let end = (start + samples).min(dataset);
        let batch = &traindats[start..end];

        // build struct graph (世代ごとに構造共有)
        let t0 = Instant::now();
        let graph = build_struct_graph(
            &genes,
            num_inputs,
            last_k,
            ops_u.len(),
            ops_b.len(),
            ops_t.len(),
        );
        let t_build = t0.elapsed().as_secs_f64();

        let ctx = ExecCtx {
            graph: &graph,
            ops_u: &ops_u,
            ops_b: &ops_b,
            ops_t: &ops_t,
        };

        // evaluate logits matrix in parallel (samples x pop)
        let mut logits = vec![0f32; batch.len() * pop];
        let mut targets = vec![0f32; batch.len()];

        let t1 = Instant::now();
        logits
            .par_chunks_mut(pop)
            .zip(batch.par_iter().enumerate())
            .for_each(|(row, (si, (gt, score)))| {
                // per-thread workspace
                let mut ws = Workspace::new();
                // x_inputs: (num_inputs,L)
                // gt is (15,L) already
                let row_logits = eval_one_sample(&ctx, &mut ws, gt);
                row.copy_from_slice(&row_logits);
                // targets
                // SAFETY: targets は並列書き込みになるので、ここでは原子的に書けない
                // → enumerateのsiを使って後で単スレで埋める
                // なので score は無視しておく（下でtargets埋める）
                let _ = si;
                let _ = score;
            });
        for (i, (_, sc)) in batch.iter().enumerate() {
            targets[i] = *sc;
        }
        let t_eval = t1.elapsed().as_secs_f64();

        // compute corr per individual (parallel)
        let t2 = Instant::now();
        let scores: Vec<f64> = (0..pop)
            .into_par_iter()
            .map(|i| {
                let mut col = vec![0f32; batch.len()];
                for r in 0..batch.len() {
                    col[r] = logits[r * pop + i];
                }
                safe_corr(&col, &targets)
            })
            .collect();
        let t_corr = t2.elapsed().as_secs_f64();

        // select best
        let mut best_i = 0usize;
        for i in 1..pop {
            if scores[i] > scores[best_i] {
                best_i = i;
            }
        }
        let best = scores[best_i];
        if step == 0 {
            best_score_ema = best;
        }
        best_score_ema = 0.9 * best_score_ema + 0.1 * best;

        println!(
            "{step}, {best:.6}, {best_score_ema:.6}, {}, {:.3}s, {:.3}s, {:.3}s",
            graph.structs.len(),
            t_build,
            t_eval,
            t_corr
        );

        // next generation
        // elitism: top 4
        let mut order: Vec<usize> = (0..pop).collect();
        order.sort_by(|&i, &j| {
            scores[j]
                .partial_cmp(&scores[i])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut new_genes: Vec<Gene> = Vec::with_capacity(pop);
        for k in 0..32.min(pop) {
            new_genes.push(genes[order[k]].clone());
        }

        // rest: tournament + crossover + mutation
        while new_genes.len() < pop {
            let p1 = tournament_select(&mut rng, &scores, 16);
            let p2 = tournament_select(&mut rng, &scores, 16);
            let mut c = crossover(&mut rng, &genes[p1], &genes[p2], num_inputs);
            mutate_gene(&mut rng, &mut c, num_inputs, &op_probs, 0.005, 0.005, 0.005);
            new_genes.push(c);
        }

        genes = new_genes;
    }

    Ok(())
}
