%%% @private
%%% @doc
%%% Internal module. Loads the `kvex_nif' shared library and exposes the
%%% raw NIF functions. All public API goes through {@link kvex}.
%%% @end
-module(kvex_nif).

-export([hello/0]).

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

hello() ->
    erlang:nif_error(nif_not_loaded).
