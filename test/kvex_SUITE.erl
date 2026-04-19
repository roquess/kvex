-module(kvex_SUITE).

-compile(export_all).
-compile(nowarn_export_all).

-include_lib("common_test/include/ct.hrl").

all() -> [version_is_binary, hello_returns_ok].

version_is_binary(_Cfg) ->
    V = kvex:version(),
    true = is_binary(V),
    <<"0.1.0">> = V.

hello_returns_ok(_Cfg) ->
    ok = kvex:hello().
