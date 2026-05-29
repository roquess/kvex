.PHONY: build build_if_needed test publish clean docs

# Auto-detect platform
ifeq ($(OS),Windows_NT)
    DLL_EXT := .dll
    NIF_NAME := sied.dll
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Linux)
        DLL_EXT := .so
        NIF_NAME := sied.so
    endif
    ifeq ($(UNAME_S),Darwin)
        DLL_EXT := .dylib
        NIF_NAME := sied.dylib
    endif
endif

# Locate cargo: prefer PATH, fall back to the default rustup location.
CARGO := $(shell command -v cargo 2>/dev/null || echo $(HOME)/.cargo/bin/cargo)

build:
	cd native/sied && $(CARGO) build --release
	mkdir -p priv
	cp native/sied/target/release/*$(DLL_EXT) priv/$(NIF_NAME)

build_if_needed:
	@if [ ! -f "priv/$(NIF_NAME)" ]; then $(MAKE) build; fi

test: build
	rebar3 do eunit, ct

publish: test docs
	rebar3 hex publish

docs:
	rebar3 edoc

clean:
	rm -rf _build native/sied/target priv/*

