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
        dim_mismatch
    ].

version_is_binary(_Cfg) ->
    V = kvex:version(),
    true = is_binary(V),
    <<"0.1.0">> = V.

new_and_delete(_Cfg) ->
    {ok, Ref} = kvex:new(128),
    0 = kvex:size(Ref),
    ok = kvex:delete(Ref).

bad_dim_rejected(_Cfg) ->
    {error, {bad_dim, 7}} = kvex:new(7),
    {error, {bad_dim, 0}} = kvex:new(0).

bad_bits_rejected(_Cfg) ->
    {error, {bad_option, bits, 5}} = kvex:new(128, #{bits => 5}),
    {error, {bad_option, bits, 1}} = kvex:new(128, #{bits => 1}).

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
