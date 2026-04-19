%%% @private
%%% @doc
%%% Internal module. Loads the `kvex_nif' shared library and exposes the
%%% raw NIF functions. All public API goes through {@link kvex}.
%%% @end
-module(kvex_nif).

-export([new_index/2, size/1, add_vec/3, search_vec/3, add_batch/2]).

-on_load(init/0).

-define(APPNAME, kvex).
-define(LIBNAME, "kvex_nif").

init() ->
    PrivDir = case code:priv_dir(?APPNAME) of
        {error, bad_name} ->
            case filelib:is_dir(filename:join(["..", priv])) of
                true  -> filename:join(["..", priv]);
                false -> "priv"
            end;
        Dir ->
            Dir
    end,
    SoName = filename:join([PrivDir, "crates", ?LIBNAME, ?LIBNAME]),
    erlang:load_nif(SoName, 0).

new_index(_Dim, _Bits) ->
    erlang:nif_error(nif_not_loaded).

size(_Ref) ->
    erlang:nif_error(nif_not_loaded).

add_vec(_Ref, _Id, _Vec) ->
    erlang:nif_error(nif_not_loaded).

search_vec(_Ref, _Query, _K) ->
    erlang:nif_error(nif_not_loaded).

add_batch(_Ref, _Pairs) ->
    erlang:nif_error(nif_not_loaded).
