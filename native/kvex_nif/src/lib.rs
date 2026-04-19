use rustler::{Encoder, Env, ResourceArc, Term};
use std::sync::RwLock;
use turbovec::TurboQuantIndex;

mod atoms {
    rustler::atoms! {
        ok,
        error,
        bad_dim,
        bad_option,
        bits,
    }
}

pub enum Id {
    Int(u64),
    Bin(Vec<u8>),
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

rustler::init!("kvex_nif", [new_index, size], load = load);
