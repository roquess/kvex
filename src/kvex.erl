%%% @doc
%%% kvex — approximate k-NN vector search on the BEAM.
%%%
%%% Backed by the TurboQuant algorithm (Google Research, ICLR 2026) via
%%% the `turbovec' Rust crate. Vectors are compressed to 2–4 bits per
%%% coordinate; no training phase, incremental inserts supported.
%%%
%%% == Quick start ==
%%% ```
%%% {ok, Ix} = kvex:new(128),
%%% 0        = kvex:size(Ix),
%%% ok       = kvex:delete(Ix).
%%% '''
%%% @end
-module(kvex).

-export([version/0, new/1, new/2, delete/1, size/1, add/3, add_batch/2, search/3]).

-type index()  :: reference().
-type opts()   :: #{bits => 2 | 3 | 4}.
-type id()     :: non_neg_integer() | binary().
-type vector() :: [float()] | binary().

-export_type([index/0, opts/0, id/0, vector/0]).

-define(DEFAULT_BITS, 4).

-spec version() -> binary().
%% @doc Returns the library version.
%%
%% Example:
%% ```
%% <<"0.1.0">> = kvex:version().
%% '''
version() ->
    <<"0.1.0">>.

-spec new(Dim :: pos_integer()) -> {ok, index()} | {error, term()}.
%% @doc Equivalent to `new(Dim, #{bits => 4})'.
%%
%% Example:
%% ```
%% {ok, Ix} = kvex:new(128).
%% '''
new(Dim) ->
    new(Dim, #{}).

-spec new(Dim :: pos_integer(), opts()) -> {ok, index()} | {error, term()}.
%% @doc Creates an empty index of dimension `Dim'.
%%
%% `Dim' must be a positive integer multiple of 8. The `bits' option
%% selects TurboQuant's bit-width per coordinate (2, 3 or 4 — default 4).
%%
%% Returns `{ok, Ref}' where `Ref' is an opaque handle, or
%% `{error, {bad_dim, Dim}}' / `{error, {bad_option, bits, Value}}'.
%%
%% Example:
%% ```
%% {ok, Ix}               = kvex:new(128, #{bits => 2}),
%% {error, {bad_dim, 7}}  = kvex:new(7).
%% '''
new(Dim, Opts) when is_integer(Dim), is_map(Opts) ->
    Bits = maps:get(bits, Opts, ?DEFAULT_BITS),
    kvex_nif:new_index(Dim, Bits).

-spec size(index()) -> non_neg_integer().
%% @doc Returns the number of indexed vectors.
%%
%% Example:
%% ```
%% {ok, Ix} = kvex:new(128),
%% 0        = kvex:size(Ix).
%% '''
size(Ref) ->
    kvex_nif:size(Ref).

-spec delete(index()) -> ok.
%% @doc Explicitly drops the index reference. Equivalent to letting the
%% reference be garbage-collected; useful for tests and for predictable
%% native memory release.
%%
%% Example:
%% ```
%% {ok, Ix} = kvex:new(128),
%% ok       = kvex:delete(Ix).
%% '''
delete(_Ref) ->
    ok.

-spec add(index(), id(), vector()) -> ok | {error, term()}.
%% @doc Inserts a single vector under the given id.
%%
%% Ids may be non-negative integers or binaries (e.g. UUIDs). Vectors
%% may be passed as a list of floats or as a little-endian f32 binary
%% of length `4 * Dim'.
%%
%% Returns `ok' on success, or `{error, {dim_mismatch, Expected, Got}}'.
%%
%% Example:
%% ```
%% {ok, Ix} = kvex:new(128),
%% Vec      = [rand:uniform() || _ <- lists:seq(1, 128)],
%% ok       = kvex:add(Ix, 42, Vec).
%% '''
add(Ref, Id, Vec) when is_list(Vec); is_binary(Vec) ->
    kvex_nif:add_vec(Ref, Id, Vec).

-spec add_batch(index(), [{id(), vector()}]) -> ok | {error, term()}.
%% @doc Inserts a list of `{Id, Vector}' pairs atomically.
%%
%% If any vector has the wrong dimension the whole batch is rejected
%% with `{error, {dim_mismatch, Position, Expected, Got}}' where
%% `Position' is the 0-based index of the offending entry.
%%
%% Runs on a dirty CPU scheduler — safe to call with large batches
%% without blocking a BEAM scheduler thread.
%%
%% Example:
%% ```
%% {ok, Ix} = kvex:new(128),
%% Pairs    = [{I, [rand:uniform() || _ <- lists:seq(1, 128)]}
%%             || I <- lists:seq(1, 10000)],
%% ok       = kvex:add_batch(Ix, Pairs).
%% '''
add_batch(Ref, Pairs) when is_list(Pairs) ->
    kvex_nif:add_batch(Ref, Pairs).

-spec search(index(), Query :: vector(), K :: pos_integer()) ->
        {ok, [{id(), Score :: float()}]} | {error, term()}.
%% @doc Returns up to `K' most-similar vectors to `Query', sorted
%% descending by score (higher = more similar).
%%
%% Scores are raw TurboQuant similarity scores — monotone with inner
%% product on the rotated / quantized representation. They are
%% comparable within a single index but not calibrated to any specific
%% metric. To get cosine similarity, L2-normalize vectors before `add'
%% and before `search'.
%%
%% Errors: `{error, empty_index}' if the index has no vectors,
%% `{error, {dim_mismatch, Expected, Got}}' on size mismatch.
%%
%% Example:
%% ```
%% {ok, Ix}      = kvex:new(128),
%% ok            = kvex:add(Ix, 1, Vec),
%% {ok, Results} = kvex:search(Ix, Query, 10).
%% '''
search(Ref, Query, K) when (is_list(Query) orelse is_binary(Query)),
                           is_integer(K), K > 0 ->
    kvex_nif:search_vec(Ref, Query, K).
