//! Rust accelerator for HoloGraph HDC kernels.
//!
//! This crate mirrors the Python `RealKernel` and `TernaryKernel` interfaces.
//! It exposes them through PyO3 so the Python loader can hot-swap to the Rust
//! implementation when the extension is compiled.  The headline win is the
//! ternary kernel's *bitsliced* binding, which packs trits into pairs of
//! u64 words and binds them with bitwise ops — closer to what a future
//! hardware accelerator would actually do.
//!
//! Build: `cd rust-kernel && maturin develop --release`
//! Then in Python: `from holograph._native import RealKernelRs, TernaryKernelRs`

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rand::SeedableRng;
use rand::distributions::{Distribution, Uniform};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

/// Tanh applied element-wise.
fn tanh_inplace(v: &mut [f32]) {
    v.par_iter_mut().for_each(|x| *x = x.tanh());
}

// -------------------------------------------------------------------------
// Bitsliced ternary representation
// -------------------------------------------------------------------------
//
// Each trit gets two bits across two parallel bit-planes:
//     pos[k] = 1  iff  trit_k == +1
//     neg[k] = 1  iff  trit_k == -1
//     zero    iff  both bits are 0
//     (1,1) is reserved and treated as 0 on decode.
//
// In this scheme:
//     bind(a, b) = (pos := (a.pos & b.pos) | (a.neg & b.neg),
//                   neg := (a.pos & b.neg) | (a.neg & b.pos))
//     sign(sum) for bundling falls out of int counters (we add lane-wise).
//     similarity(a, b) = (matches - mismatches) / active_dims where:
//         active  = (a.pos | a.neg) & (b.pos | b.neg)
//         matches = popcount(active & ((a.pos & b.pos) | (a.neg & b.neg)))
//         mismatches = popcount(active & ((a.pos & b.neg) | (a.neg & b.pos)))

#[derive(Clone)]
struct BitsliceTernary {
    dim: usize,
    pos: Vec<u64>,
    neg: Vec<u64>,
}

impl BitsliceTernary {
    fn new(dim: usize) -> Self {
        let words = (dim + 63) / 64;
        BitsliceTernary { dim, pos: vec![0u64; words], neg: vec![0u64; words] }
    }

    fn from_int8(slice: &[i8]) -> Self {
        let dim = slice.len();
        let words = (dim + 63) / 64;
        let mut pos = vec![0u64; words];
        let mut neg = vec![0u64; words];
        for (i, &t) in slice.iter().enumerate() {
            let w = i / 64;
            let b = i % 64;
            match t {
                1 => pos[w] |= 1u64 << b,
                -1 => neg[w] |= 1u64 << b,
                _ => {}
            }
        }
        BitsliceTernary { dim, pos, neg }
    }

    fn to_int8(&self) -> Vec<i8> {
        let mut out = vec![0i8; self.dim];
        for i in 0..self.dim {
            let w = i / 64;
            let b = i % 64;
            let p = (self.pos[w] >> b) & 1;
            let n = (self.neg[w] >> b) & 1;
            out[i] = if p == 1 && n == 0 { 1 }
                     else if n == 1 && p == 0 { -1 }
                     else { 0 };
        }
        out
    }

    fn bind(&self, other: &BitsliceTernary) -> BitsliceTernary {
        assert_eq!(self.dim, other.dim);
        let words = self.pos.len();
        let mut out = BitsliceTernary::new(self.dim);
        for w in 0..words {
            // The bit-rule for trit multiplication:
            //   result_pos = (a.pos & b.pos) | (a.neg & b.neg)
            //   result_neg = (a.pos & b.neg) | (a.neg & b.pos)
            out.pos[w] = (self.pos[w] & other.pos[w]) | (self.neg[w] & other.neg[w]);
            out.neg[w] = (self.pos[w] & other.neg[w]) | (self.neg[w] & other.pos[w]);
        }
        out
    }

    fn similarity(&self, other: &BitsliceTernary) -> f64 {
        assert_eq!(self.dim, other.dim);
        let words = self.pos.len();
        let mut matches: u64 = 0;
        let mut mismatches: u64 = 0;
        let mut active: u64 = 0;
        for w in 0..words {
            let a_act = self.pos[w] | self.neg[w];
            let b_act = other.pos[w] | other.neg[w];
            let act = a_act & b_act;
            let m = act & ((self.pos[w] & other.pos[w]) | (self.neg[w] & other.neg[w]));
            let mm = act & ((self.pos[w] & other.neg[w]) | (self.neg[w] & other.pos[w]));
            matches += m.count_ones() as u64;
            mismatches += mm.count_ones() as u64;
            active += act.count_ones() as u64;
        }
        if active == 0 { 0.0 }
        else { (matches as f64 - mismatches as f64) / (active as f64) }
    }
}

// -------------------------------------------------------------------------
// PyO3 classes
// -------------------------------------------------------------------------

/// Float32 + tanh + cosine implementation.
#[pyclass]
struct RealKernelRs {
    dim: usize,
}

#[pymethods]
impl RealKernelRs {
    #[new]
    fn new(dim: usize) -> Self {
        RealKernelRs { dim }
    }

    #[getter]
    fn dim(&self) -> usize { self.dim }

    #[getter]
    fn name(&self) -> &'static str { "real" }

    /// Generate a row-normalised Gaussian random basis matrix.
    fn random_basis<'py>(&self, py: Python<'py>, n_rows: usize, seed: u64) -> Bound<'py, PyArray2<f32>> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let std_normal = rand_distr_standard_normal();
        let mut data: Vec<f32> = Vec::with_capacity(n_rows * self.dim);
        for _ in 0..(n_rows * self.dim) {
            data.push(std_normal.sample(&mut rng));
        }
        // Row-normalize.
        for row in 0..n_rows {
            let start = row * self.dim;
            let end = start + self.dim;
            let mut norm: f32 = 0.0;
            for x in &data[start..end] {
                norm += x * x;
            }
            norm = norm.sqrt();
            if norm > 0.0 {
                for x in &mut data[start..end] {
                    *x /= norm;
                }
            }
        }
        let arr = ndarray_from_vec_2d(data, n_rows, self.dim);
        arr.into_pyarray_bound(py)
    }

    fn bind<'py>(&self, py: Python<'py>,
                  a: PyReadonlyArray1<'_, f32>,
                  b: PyReadonlyArray1<'_, f32>) -> Bound<'py, PyArray1<f32>> {
        let av = a.as_slice().unwrap();
        let bv = b.as_slice().unwrap();
        let mut out: Vec<f32> = av.iter().zip(bv.iter()).map(|(x, y)| x * y).collect();
        let _ = &mut out;
        out.into_pyarray_bound(py)
    }

    fn bundle<'py>(&self, py: Python<'py>, vs: Vec<PyReadonlyArray1<'_, f32>>) -> Bound<'py, PyArray1<f32>> {
        if vs.is_empty() {
            return vec![0.0f32; self.dim].into_pyarray_bound(py);
        }
        let mut acc = vec![0.0f32; self.dim];
        for v in &vs {
            let s = v.as_slice().unwrap();
            for (i, x) in s.iter().enumerate() {
                acc[i] += x;
            }
        }
        let mut norm = 0.0f32;
        for x in &acc { norm += x * x; }
        norm = norm.sqrt();
        if norm > 0.0 {
            for x in &mut acc { *x /= norm; }
        }
        acc.into_pyarray_bound(py)
    }

    fn similarity(&self, a: PyReadonlyArray1<'_, f32>, b: PyReadonlyArray1<'_, f32>) -> f32 {
        let av = a.as_slice().unwrap();
        let bv = b.as_slice().unwrap();
        let mut dot = 0.0f32;
        let mut na = 0.0f32;
        let mut nb = 0.0f32;
        for i in 0..av.len() {
            dot += av[i] * bv[i];
            na += av[i] * av[i];
            nb += bv[i] * bv[i];
        }
        let na = na.sqrt();
        let nb = nb.sqrt();
        if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
    }

    /// PSP-HDC encode: h = tanh((scaled * e) @ B)
    fn encode_scalar<'py>(&self, py: Python<'py>,
                           scaled: f32,
                           embedding: PyReadonlyArray1<'_, f32>,
                           basis: PyReadonlyArray2<'_, f32>) -> Bound<'py, PyArray1<f32>> {
        let e = embedding.as_slice().unwrap();
        let b = basis.as_array();
        let d = self.dim;
        let mut out = vec![0.0f32; d];
        for j in 0..d {
            let mut acc = 0.0f32;
            for k in 0..e.len() {
                acc += scaled * e[k] * b[[k, j]];
            }
            out[j] = acc;
        }
        tanh_inplace(&mut out);
        out.into_pyarray_bound(py)
    }

    fn pack<'py>(&self, py: Python<'py>, hv: PyReadonlyArray1<'_, f32>) -> Py<PyBytes> {
        let s = hv.as_slice().unwrap();
        let bytes: Vec<u8> = s.iter().flat_map(|x| x.to_le_bytes()).collect();
        PyBytes::new_bound(py, &bytes).into()
    }

    fn unpack<'py>(&self, py: Python<'py>, blob: &[u8]) -> Bound<'py, PyArray1<f32>> {
        let dim = self.dim;
        let mut out = vec![0.0f32; dim];
        for i in 0..dim {
            let off = i * 4;
            let arr: [u8; 4] = [blob[off], blob[off+1], blob[off+2], blob[off+3]];
            out[i] = f32::from_le_bytes(arr);
        }
        out.into_pyarray_bound(py)
    }
}


/// Bitsliced ternary kernel.
#[pyclass]
struct TernaryKernelRs {
    dim: usize,
    deadband: f32,
}

#[pymethods]
impl TernaryKernelRs {
    #[new]
    fn new(dim: usize, deadband: f32) -> Self {
        TernaryKernelRs { dim, deadband }
    }

    #[getter]
    fn dim(&self) -> usize { self.dim }

    #[getter]
    fn name(&self) -> &'static str { "ternary" }

    fn random_basis<'py>(&self, py: Python<'py>, n_rows: usize, seed: u64) -> Bound<'py, PyArray2<f32>> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let sn = rand_distr_standard_normal();
        let mut data: Vec<f32> = Vec::with_capacity(n_rows * self.dim);
        for _ in 0..(n_rows * self.dim) {
            data.push(sn.sample(&mut rng));
        }
        for row in 0..n_rows {
            let start = row * self.dim;
            let end = start + self.dim;
            let mut norm: f32 = 0.0;
            for x in &data[start..end] { norm += x * x; }
            norm = norm.sqrt();
            if norm > 0.0 {
                for x in &mut data[start..end] { *x /= norm; }
            }
        }
        let arr = ndarray_from_vec_2d(data, n_rows, self.dim);
        arr.into_pyarray_bound(py)
    }

    fn encode_scalar<'py>(&self, py: Python<'py>,
                           scaled: f32,
                           embedding: PyReadonlyArray1<'_, f32>,
                           basis: PyReadonlyArray2<'_, f32>) -> Bound<'py, PyArray1<i8>> {
        let e = embedding.as_slice().unwrap();
        let b = basis.as_array();
        let d = self.dim;
        let mut out = vec![0i8; d];
        for j in 0..d {
            let mut acc = 0.0f32;
            for k in 0..e.len() {
                acc += scaled * e[k] * b[[k, j]];
            }
            out[j] = if acc.abs() <= self.deadband {
                0
            } else if acc > 0.0 { 1 } else { -1 };
        }
        out.into_pyarray_bound(py)
    }

    fn bind<'py>(&self, py: Python<'py>,
                  a: PyReadonlyArray1<'_, i8>,
                  b: PyReadonlyArray1<'_, i8>) -> Bound<'py, PyArray1<i8>> {
        let av = a.as_slice().unwrap();
        let bv = b.as_slice().unwrap();
        let bs_a = BitsliceTernary::from_int8(av);
        let bs_b = BitsliceTernary::from_int8(bv);
        let result = bs_a.bind(&bs_b);
        result.to_int8().into_pyarray_bound(py)
    }

    fn bundle<'py>(&self, py: Python<'py>, vs: Vec<PyReadonlyArray1<'_, i8>>) -> Bound<'py, PyArray1<i8>> {
        if vs.is_empty() {
            return vec![0i8; self.dim].into_pyarray_bound(py);
        }
        let n = vs.len() as f32;
        let mut acc = vec![0i32; self.dim];
        for v in &vs {
            let s = v.as_slice().unwrap();
            for (i, x) in s.iter().enumerate() {
                acc[i] += *x as i32;
            }
        }
        let threshold = 0.5_f32 * n.sqrt();
        let threshold = if threshold > self.deadband { threshold } else { self.deadband };
        let mut out = vec![0i8; self.dim];
        for i in 0..self.dim {
            let a = acc[i] as f32;
            out[i] = if a.abs() <= threshold {
                0
            } else if a > 0.0 { 1 } else { -1 };
        }
        out.into_pyarray_bound(py)
    }

    fn similarity(&self, a: PyReadonlyArray1<'_, i8>, b: PyReadonlyArray1<'_, i8>) -> f64 {
        let bs_a = BitsliceTernary::from_int8(a.as_slice().unwrap());
        let bs_b = BitsliceTernary::from_int8(b.as_slice().unwrap());
        bs_a.similarity(&bs_b)
    }

    fn quantize<'py>(&self, py: Python<'py>, real_hv: PyReadonlyArray1<'_, f32>) -> Bound<'py, PyArray1<i8>> {
        let s = real_hv.as_slice().unwrap();
        let mut out = vec![0i8; self.dim];
        for i in 0..self.dim {
            let x = s[i];
            out[i] = if x.abs() <= self.deadband {
                0
            } else if x > 0.0 { 1 } else { -1 };
        }
        out.into_pyarray_bound(py)
    }

    fn pack<'py>(&self, py: Python<'py>, hv: PyReadonlyArray1<'_, i8>) -> Py<PyBytes> {
        let s = hv.as_slice().unwrap();
        let mut codes: Vec<u8> = vec![0u8; self.dim];
        for i in 0..self.dim {
            codes[i] = match s[i] {
                1 => 1, -1 => 2, _ => 0,
            };
        }
        let pad = (4 - (self.dim % 4)) % 4;
        if pad > 0 { codes.extend(std::iter::repeat(0u8).take(pad)); }
        let mut out: Vec<u8> = Vec::with_capacity(codes.len() / 4);
        for chunk in codes.chunks(4) {
            out.push(chunk[0] | (chunk[1] << 2) | (chunk[2] << 4) | (chunk[3] << 6));
        }
        PyBytes::new_bound(py, &out).into()
    }

    fn unpack<'py>(&self, py: Python<'py>, blob: &[u8]) -> Bound<'py, PyArray1<i8>> {
        let mut out: Vec<i8> = Vec::with_capacity(self.dim);
        let mut emitted: usize = 0;
        'outer: for byte in blob {
            for shift in [0u32, 2, 4, 6] {
                if emitted == self.dim { break 'outer; }
                let code = (byte >> shift) & 0b11;
                out.push(match code {
                    1 => 1, 2 => -1, _ => 0,
                });
                emitted += 1;
            }
        }
        while out.len() < self.dim { out.push(0); }
        out.into_pyarray_bound(py)
    }
}


// -------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------

fn ndarray_from_vec_2d(data: Vec<f32>, n_rows: usize, n_cols: usize) -> ndarray::Array2<f32> {
    ndarray::Array2::from_shape_vec((n_rows, n_cols), data).expect("shape mismatch")
}

/// Standard-normal sampler using Box-Muller (no external dependency).
struct StandardNormal {
    uniform: Uniform<f32>,
    cached: std::cell::Cell<Option<f32>>,
}

fn rand_distr_standard_normal() -> StandardNormal {
    StandardNormal {
        uniform: Uniform::new(f32::EPSILON, 1.0_f32),
        cached: std::cell::Cell::new(None),
    }
}

impl StandardNormal {
    fn sample(&self, rng: &mut ChaCha8Rng) -> f32 {
        if let Some(v) = self.cached.take() {
            return v;
        }
        let u1 = self.uniform.sample(rng);
        let u2 = self.uniform.sample(rng);
        let r = (-2.0_f32 * u1.ln()).sqrt();
        let theta = 2.0_f32 * std::f32::consts::PI * u2;
        let z0 = r * theta.cos();
        let z1 = r * theta.sin();
        self.cached.set(Some(z1));
        z0
    }
}


// -------------------------------------------------------------------------
// Module entry
// -------------------------------------------------------------------------

#[pymodule]
fn holograph_hdc_rs(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RealKernelRs>()?;
    m.add_class::<TernaryKernelRs>()?;
    m.add("__doc__", "HoloGraph HDC kernels: Rust accelerator with bitsliced ternary binding.")?;
    Ok(())
}
