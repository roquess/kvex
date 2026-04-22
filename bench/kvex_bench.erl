-module(kvex_bench).
-export([run/0, run/1]).

-define(REPS, 50).

-type config() :: #{
    dim  := pos_integer(),
    n    := pos_integer(),
    k    := pos_integer(),
    reps := pos_integer()
}.

-spec run() -> ok.
%% @doc Run benchmark with default settings (128-dim, 10k vectors, k=10).
run() ->
    run(#{dim => 128, n => 10000, k => 10, reps => ?REPS}).

-spec run(config()) -> ok.
run(#{dim := Dim, n := N, k := K, reps := Reps}) ->
    io:format("~n=== kvex benchmark — ~s ===~n", [kvex:version()]),
    io:format("dim=~w  n=~w  k=~w  reps=~w~n~n", [Dim, N, K, Reps]),

    %% Fixed seed for reproducible data across versions
    rand:seed(exsss, {42, 0, 0}),
    Vecs  = [{I, rand_vec(Dim)} || I <- lists:seq(1, N)],
    Query = rand_vec(Dim),

    %% --- new/1 ---
    {T_new, {ok, Ix}} = timer:tc(kvex, new, [Dim]),
    report("new/1           ", T_new, 1),

    %% --- add_batch/2 ---
    {T_add, ok} = timer:tc(kvex, add_batch, [Ix, Vecs]),
    report("add_batch/2     ", T_add, N),

    %% --- search/3 single (cold) ---
    {T_cold, {ok, _}} = timer:tc(kvex, search, [Ix, Query, K]),
    report("search/3 (cold) ", T_cold, 1),

    %% --- search/3 average over Reps ---
    {T_total, _} = timer:tc(fun() ->
        [kvex:search(Ix, Query, K) || _ <- lists:seq(1, Reps)]
    end),
    report("search/3 (avg)  ", T_total div Reps, 1),

    %% --- search/3 throughput ---
    Qps = trunc(1_000_000 / max(1, T_total div Reps)),
    io:format("  throughput      ~8w queries/s~n", [Qps]),

    %% --- size/1 sanity ---
    N = kvex:size(Ix),
    io:format("  size            ~8w vectors~n", [N]),

    %% --- delete/1 ---
    {T_del, ok} = timer:tc(kvex, delete, [Ix]),
    report("delete/1        ", T_del, 1),

    io:format("~n"),
    ok.

%%%===================================================================
%%% Helpers
%%%===================================================================

report(Label, UsTotal, Count) ->
    Us   = UsTotal div max(1, Count),
    Ms   = UsTotal / 1000.0,
    io:format("  ~s ~8w μs  (~.2f ms total)~n", [Label, Us, Ms]).

rand_vec(Dim) ->
    [rand:uniform() * 2.0 - 1.0 || _ <- lists:seq(1, Dim)].
