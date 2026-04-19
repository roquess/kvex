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
        size_of_empty
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
