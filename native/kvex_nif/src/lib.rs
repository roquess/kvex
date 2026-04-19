use rustler::{Atom, Env, Term};

mod atoms {
    rustler::atoms! {
        ok,
        error,
    }
}

#[rustler::nif]
fn hello() -> Atom {
    atoms::ok()
}

fn load(_env: Env, _info: Term) -> bool {
    true
}

rustler::init!("kvex_nif", [hello], load = load);
