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
    vector: Vec<f32>,
) -> Term<'a> {
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

rustler::init!("kvex_nif", [new_index, size, add_vec], load = load);
