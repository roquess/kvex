%%% @doc
%%% kvex — approximate k-NN vector search on the BEAM.
%%%
%%% Backed by the TurboQuant algorithm (Google Research, ICLR 2026) via
%%% the `turbovec' Rust crate. Vectors are compressed to 2–4 bits per
%%% coordinate; no training phase, incremental inserts supported.
%%%
%%% See the module documentation of each exported function for usage.
%%% @end
-module(kvex).

-export([version/0]).

-spec version() -> binary().
%% @doc Returns the library version as a binary, e.g. `<<"0.1.0">>'.
version() ->
    <<"0.1.0">>.
