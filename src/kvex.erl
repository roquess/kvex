%%% @doc
%%% kvex — approximate k-NN vector search on the BEAM.
%%%
%%% Backed by the TurboQuant algorithm (Google Research, ICLR 2026) via
%%% the `turbovec' Rust crate. Vectors are compressed to 2–4 bits per
%%% coordinate; no training phase, incremental inserts supported.
%%% @end
-module(kvex).

-export([version/0, hello/0]).

-spec version() -> binary().
%% @doc Returns the library version, e.g. `<<"0.1.0">>'.
version() ->
    <<"0.1.0">>.

-spec hello() -> ok.
%% @private
%% @doc Smoke-test entry point that confirms the NIF library is loaded.
%% Returns the atom `ok' when the native code is reachable.
hello() ->
    kvex_nif:hello().
