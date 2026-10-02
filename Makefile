.PHONY: help test bench-cpp run-demo clean

help:
	@echo "Financial Risk MCP Server - Commands:"
	@echo "  make test        : Run full unit test suite"
	@echo "  make bench-cpp   : Build and execute C++ multithreaded Monte Carlo benchmark"
	@echo "  make run-demo    : Run end-to-end Python client demo over stdio"
	@echo "  make clean       : Clean build artifacts and binaries"

test:
	PYTHONPATH=src python3 -m unittest discover tests -v

bench-cpp:
	g++ -O3 -std=c++17 cpp_engine/monte_carlo_pfe_engine.cpp -pthread -o cpp_engine/pfe_benchmark
	./cpp_engine/pfe_benchmark 100000

run-demo:
	PYTHONPATH=src python3 examples/sample_client.py

clean:
	rm -f cpp_engine/pfe_benchmark
	find . -type d -name "__pycache__" -exec rm -rf {} +
