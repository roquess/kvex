# kvex

[![Hex.pm](https://img.shields.io/hexpm/v/kvex.svg)](https://hex.pm/packages/kvex)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Approximate k-nearest-neighbour vector search for Erlang/OTP, powered by
[TurboQuant](https://github.com/RyanCodrai/turbovec) (Google Research,
ICLR 2026).

- **Data-oblivious quantization** — no training phase, no codebook
  retraining when data changes, vectors can be added at any time.
- **2–4 bit compression** per coordinate with near-optimal distortion.
- **SIMD-accelerated search** (NEON / AVX2 / AVX-512BW) via the
  `turbovec` Rust crate, exposed through a `rustler` NIF.
- **Parallel reads** — search runs under a read lock; concurrent BEAM
  processes query the same index in parallel.

## Installation

Add `kvex` to your `rebar.config`:

```erlang
{deps, [
    {kvex, "0.1.0"}
]}.
```

## Quick start

```erlang
%% Create an index of 128-dim vectors, compressed to 4 bits per coord.
{ok, Ix} = kvex:new(128),

%% Insert a single vector.
Vec = [rand:uniform() || _ <- lists:seq(1, 128)],
ok = kvex:add(Ix, 42, Vec),

%% Batch insert is more efficient for large loads.
Pairs = [{I, [rand:uniform() || _ <- lists:seq(1, 128)]}
         || I <- lists:seq(1, 10000)],
ok = kvex:add_batch(Ix, Pairs),

%% Top-10 nearest neighbours.
{ok, Results} = kvex:search(Ix, Vec, 10).
%% Results :: [{Id, Score}], sorted descending by score.
```

Vectors can also be passed as little-endian f32 binaries of length
`4 * Dim`, which avoids list traversal on hot paths:

```erlang
VBin = << <<X:32/float-little>> || X <- Vec >>,
ok   = kvex:add(Ix, 43, VBin),
{ok, Top} = kvex:search(Ix, VBin, 10).
```

## API at a glance

| Function | Purpose |
|---|---|
| `new(Dim)` / `new(Dim, Opts)` | Create an empty index. `Dim` must be a multiple of 8. |
| `add(Ix, Id, Vector)` | Insert a single vector. |
| `add_batch(Ix, Pairs)` | Atomic batch insert. |
| `search(Ix, Query, K)` | Top-K most similar vectors. |
| `size(Ix)` | Number of indexed vectors. |
| `delete(Ix)` | Explicit release (also happens automatically on GC). |

Options accepted by `new/2`:

| Option | Values | Default | Description |
|---|---|---|---|
| `bits` | `2 \| 3 \| 4` | `4` | TurboQuant bit-width per coordinate. |

Full documentation on [hexdocs.pm/kvex](https://hexdocs.pm/kvex).

## Notes on scoring

`turbovec` returns raw similarity scores (higher = more similar). They
are monotone with the inner product on the rotated / quantized
representation and are comparable within a single index, but are
**not** calibrated to L2 or cosine. For cosine similarity, L2-normalize
your vectors before `add` and before `search` — the inner product on
unit vectors equals cosine.

## Platform notes

- **Linux / macOS:** tested on stable toolchains. Linux builds pull
  OpenBLAS via `turbovec`'s dependency tree; install
  `libopenblas-dev` (Debian / Ubuntu) or equivalent.
- **Windows:** build requires the MSVC toolchain and a recent `rustc`.

## Building from source

```bash
rebar3 compile
rebar3 ct
rebar3 edoc
```

The `rebar3_rustler` plugin drives cargo under `native/kvex_nif/` and
copies the resulting shared library into `priv/crates/kvex_nif/`.

## License

Apache License 2.0 — see [LICENSE](LICENSE). `turbovec` is
MIT-licensed.

## Acknowledgements

- TurboQuant algorithm: Google Research, ICLR 2026.
- Rust implementation: [RyanCodrai/turbovec](https://github.com/RyanCodrai/turbovec).
