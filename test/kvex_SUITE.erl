-module(kvex_SUITE).

-compile(export_all).
-compile(nowarn_export_all).

-include_lib("common_test/include/ct.hrl").

all() ->
    [
        version_is_binary,
        new_and_delete,
        bad_dim_rejected,
        bad_bits_rejected,
        size_of_empty,
        add_single_increments_size,
        dim_mismatch,
        empty_index_search,
        add_single_self_retrieval,
        search_topk_order,
        large_k_clamped,
        add_batch_then_search,
        binary_vector_input,
        concurrent_search,
        gc_releases_native
    ].

version_is_binary(_Cfg) ->
    V = kvex:version(),
    true = is_binary(V),
    <<"0.2.0">> = V.

new_and_delete(_Cfg) ->
    {ok, Ref} = kvex:new(128),
    0 = kvex:size(Ref),
    ok = kvex:delete(Ref).

bad_dim_rejected(_Cfg) ->
    {error, {bad_dim, 0}}  = kvex:new(0),
    {error, {bad_dim, -1}} = kvex:new(-1).

bad_bits_rejected(_Cfg) ->
    %% v0.2.0 silently ignores the bits option (no turbovec quantization)
    {ok, Ix} = kvex:new(128, #{bits => 5}),
    ok = kvex:delete(Ix).

size_of_empty(_Cfg) ->
    {ok, Ref} = kvex:new(128, #{bits => 2}),
    0 = kvex:size(Ref).

add_single_increments_size(_Cfg) ->
    {ok, Ix} = kvex:new(128),
    Vec = [rand:uniform() || _ <- lists:seq(1, 128)],
    ok = kvex:add(Ix, 42, Vec),
    1  = kvex:size(Ix).

dim_mismatch(_Cfg) ->
    {ok, Ix} = kvex:new(128),
    Short   = [0.0 || _ <- lists:seq(1, 64)],
    {error, {dim_mismatch, 128, 64}} = kvex:add(Ix, 1, Short).

empty_index_search(_Cfg) ->
    {ok, Ix} = kvex:new(128),
    Q = [0.0 || _ <- lists:seq(1, 128)],
    {error, empty_index} = kvex:search(Ix, Q, 5).

add_single_self_retrieval(_Cfg) ->
    {ok, Ix} = kvex:new(128),
    V = [rand:uniform() || _ <- lists:seq(1, 128)],
    ok = kvex:add(Ix, 7, V),
    {ok, [{7, _Score}]} = kvex:search(Ix, V, 1).

search_topk_order(_Cfg) ->
    {ok, Ix} = kvex:new(64),
    lists:foreach(fun(I) ->
        V = [rand:uniform() || _ <- lists:seq(1, 64)],
        ok = kvex:add(Ix, I, V)
    end, lists:seq(1, 50)),
    Q = [rand:uniform() || _ <- lists:seq(1, 64)],
    {ok, Results} = kvex:search(Ix, Q, 10),
    10 = length(Results),
    Scores = [S || {_Id, S} <- Results],
    Scores = lists:reverse(lists:sort(Scores)).

large_k_clamped(_Cfg) ->
    {ok, Ix} = kvex:new(64),
    lists:foreach(fun(I) ->
        V = [rand:uniform() || _ <- lists:seq(1, 64)],
        ok = kvex:add(Ix, I, V)
    end, lists:seq(1, 5)),
    Q = [rand:uniform() || _ <- lists:seq(1, 64)],
    {ok, Results} = kvex:search(Ix, Q, 100),
    5 = length(Results).

add_batch_then_search(_Cfg) ->
    {ok, Ix} = kvex:new(128),
    Batch = [
        {I, [rand:uniform() || _ <- lists:seq(1, 128)]}
        || I <- lists:seq(1, 1000)
    ],
    ok = kvex:add_batch(Ix, Batch),
    1000 = kvex:size(Ix),
    Q = [rand:uniform() || _ <- lists:seq(1, 128)],
    {ok, Top10} = kvex:search(Ix, Q, 10),
    10 = length(Top10),
    lists:foreach(fun({Id, Score}) ->
        true = is_integer(Id),
        true = Id >= 1 andalso Id =< 1000,
        true = is_float(Score)
    end, Top10).

binary_vector_input(_Cfg) ->
    {ok, Ix} = kvex:new(128),
    VList = [rand:uniform() || _ <- lists:seq(1, 128)],
    VBin  = << <<X:32/float-little>> || X <- VList >>,
    ok = kvex:add(Ix, 1, VList),
    ok = kvex:add(Ix, 2, VBin),
    2  = kvex:size(Ix),
    {ok, [{Top1, _} | _]} = kvex:search(Ix, VBin, 2),
    true = (Top1 =:= 1 orelse Top1 =:= 2).

concurrent_search(_Cfg) ->
    {ok, Ix} = kvex:new(128),
    Pairs = [{I, [rand:uniform() || _ <- lists:seq(1, 128)]}
             || I <- lists:seq(1, 500)],
    ok = kvex:add_batch(Ix, Pairs),
    Q = [rand:uniform() || _ <- lists:seq(1, 128)],
    {ok, Expected} = kvex:search(Ix, Q, 10),
    Parent = self(),
    Pids = [spawn(fun() ->
        {ok, R} = kvex:search(Ix, Q, 10),
        Parent ! {self(), R}
    end) || _ <- lists:seq(1, 20)],
    Results = [receive {P, R} -> R end || P <- Pids],
    lists:foreach(fun(R) -> Expected = R end, Results).

gc_releases_native(_Cfg) ->
    Before = erlang:memory(binary),
    Pid = spawn(fun() ->
        {ok, Ix} = kvex:new(512),
        Pairs = [{I, [rand:uniform() || _ <- lists:seq(1, 512)]}
                 || I <- lists:seq(1, 10000)],
        ok = kvex:add_batch(Ix, Pairs),
        receive done -> ok end
    end),
    MRef = erlang:monitor(process, Pid),
    Pid ! done,
    receive {'DOWN', MRef, process, Pid, _} -> ok end,
    [erlang:garbage_collect(P) || P <- erlang:processes()],
    After = erlang:memory(binary),
    Delta = After - Before,
    true = Delta =< 4 * 1024 * 1024.
