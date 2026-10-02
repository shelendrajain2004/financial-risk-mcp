.PHONY: help all build-cpp test bench-cpp run-demo clean

all: build-cpp test

help:
	@echo "Financial Risk MCP Server - Commands:"
	@echo "  make all         : Build C++ shared library and run tests"
	@echo "  make build-cpp   : Compile native C++ shared library and benchmark CLI"
	@echo "  make test        : Run full unit test suite"
	@echo "  make bench-cpp   : Execute C++ multithreaded Monte Carlo benchmark"
	@echo "  make run-demo    : Run end-to-end Python client demo over stdio"
	@echo "  make clean       : Clean build artifacts and binaries"

build-cpp:
	g++ -O3 -std=c++17 -fPIC -shared -DBUILD_SHARED_LIB cpp_engine/monte_carlo_pfe_engine.cpp -pthread -o cpp_engine/libmontecarlo_pfe.so
	g++ -O3 -std=c++17 cpp_engine/monte_carlo_pfe_engine.cpp -pthread -o cpp_engine/pfe_benchmark

test: build-cpp
	PYTHONPATH=src python3 -m unittest discover tests -v

bench-cpp: build-cpp
	./cpp_engine/pfe_benchmark 100000

run-demo: build-cpp
	PYTHONPATH=src python3 examples/sample_client.py

clean:
	rm -f cpp_engine/pfe_benchmark cpp_engine/libmontecarlo_pfe.so
	find . -type d -name "__pycache__" -exec rm -rf {} +
