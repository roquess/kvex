use rustler::types::binary::Binary;
use rustler::{Decoder, Encoder, Env, Error as NifError, NifResult, ResourceArc, Term};
use std::sync::RwLock;
use turbovec::TurboQuantIndex;

mod atoms {
    rustler::atoms! {
        ok,
        error,
        bad_dim,
        bad_option,
        bits,
        dim_mismatch,
        empty_index,
    }
}

pub enum Id {
    Int(u64),
    Bin(Vec<u8>),
}

impl<'a> Decoder<'a> for Id {
    fn decode(term: Term<'a>) -> NifResult<Self> {
        if let Ok(n) = term.decode::<u64>() {
            return Ok(Id::Int(n));
        }
        if let Ok(b) = term.decode::<Binary>() {
            return Ok(Id::Bin(b.as_slice().to_vec()));
        }
        Err(NifError::BadArg)
    }
}

/// Wrapper accepted as either a list of floats or a little-endian f32
/// binary. Always materialises to `Vec<f32>`.
pub struct VecF32(pub Vec<f32>);

impl<'a> Decoder<'a> for VecF32 {
    fn decode(term: Term<'a>) -> NifResult<Self> {
        if let Ok(v) = term.decode::<Vec<f32>>() {
            return Ok(VecF32(v));
        }
        if let Ok(b) = term.decode::<Binary>() {
            let bytes = b.as_slice();
            if bytes.len() % 4 != 0 {
                return Err(NifError::BadArg);
            }
            let mut out = Vec::with_capacity(bytes.len() / 4);
            for chunk in bytes.chunks_exact(4) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            return Ok(VecF32(out));
        }
        Err(NifError::BadArg)
    }
}

pub struct KvexIndex {
    pub dim: usize,
    pub bits: usize,
    pub ids: Vec<Id>,
    pub inner: TurboQuantIndex,
}

pub struct IndexResource(pub RwLock<KvexIndex>);

fn load(env: Env, _info: Term) -> bool {
    rustler::resource!(IndexResource, env);
    true
}

#[rustler::nif]
fn new_index<'a>(env: Env<'a>, dim: usize, bits: usize) -> Term<'a> {
    if dim == 0 || dim % 8 != 0 {
        return (atoms::error(), (atoms::bad_dim(), dim)).encode(env);
    }
    if !(2..=4).contains(&bits) {
        return (atoms::error(), (atoms::bad_option(), atoms::bits(), bits)).encode(env);
    }
    let idx = KvexIndex {
        dim,
        bits,
        ids: Vec::new(),
        inner: TurboQuantIndex::new(dim, bits),
    };
    let resource = ResourceArc::new(IndexResource(RwLock::new(idx)));
    (atoms::ok(), resource).encode(env)
}

#[rustler::nif]
fn size(resource: ResourceArc<IndexResource>) -> usize {
    resource.0.read().unwrap().inner.len()
}

#[rustler::nif]
fn add_vec<'a>(
    env: Env<'a>,
    resource: ResourceArc<IndexResource>,
    id: Id,
    vector: VecF32,
) -> Term<'a> {
    let vector = vector.0;
    let mut guard = resource.0.write().unwrap();
    if vector.len() != guard.dim {
        return (
            atoms::error(),
            (atoms::dim_mismatch(), guard.dim, vector.len()),
        )
            .encode(env);
    }
    guard.inner.add(&vector);
    guard.ids.push(id);
    atoms::ok().encode(env)
}

#[rustler::nif(schedule = "DirtyCpu")]
fn search_vec<'a>(
    env: Env<'a>,
    resource: ResourceArc<IndexResource>,
    query: VecF32,
    k: usize,
) -> Term<'a> {
    let query = query.0;
    let guard = resource.0.read().unwrap();
    if guard.inner.len() == 0 {
        return (atoms::error(), atoms::empty_index()).encode(env);
    }
    if query.len() != guard.dim {
        return (
            atoms::error(),
            (atoms::dim_mismatch(), guard.dim, query.len()),
        )
            .encode(env);
    }
    let results = guard.inner.search(&query, k);
    let indices = results.indices_for_query(0);
    let scores = results.scores_for_query(0);
    let mut out: Vec<(Term<'a>, f32)> = Vec::with_capacity(results.k);
    for (pos, &ix) in indices.iter().enumerate() {
        let id_term = match &guard.ids[ix as usize] {
            Id::Int(n) => n.encode(env),
            Id::Bin(b) => {
                let mut owned = rustler::types::binary::OwnedBinary::new(b.len())
                    .expect("oom allocating binary");
                owned.as_mut_slice().copy_from_slice(b);
                owned.release(env).to_term(env)
            }
        };
        out.push((id_term, scores[pos]));
    }
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let list: Vec<Term<'a>> = out
        .into_iter()
        .map(|(id, s)| (id, s).encode(env))
        .collect();
    (atoms::ok(), list).encode(env)
}

#[rustler::nif(schedule = "DirtyCpu")]
fn add_batch<'a>(
    env: Env<'a>,
    resource: ResourceArc<IndexResource>,
    pairs: Vec<(Id, VecF32)>,
) -> Term<'a> {
    let mut guard = resource.0.write().unwrap();
    for (i, (_id, v)) in pairs.iter().enumerate() {
        if v.0.len() != guard.dim {
            return (
                atoms::error(),
                (atoms::dim_mismatch(), i, guard.dim, v.0.len()),
            )
                .encode(env);
        }
    }
    let mut flat = Vec::with_capacity(pairs.len() * guard.dim);
    for (_id, v) in &pairs {
        flat.extend_from_slice(&v.0);
    }
    guard.inner.add(&flat);
    for (id, _) in pairs.into_iter() {
        guard.ids.push(id);
    }
    atoms::ok().encode(env)
}

rustler::init!(
    "kvex_nif",
    [new_index, size, add_vec, search_vec, add_batch],
    load = load
);
