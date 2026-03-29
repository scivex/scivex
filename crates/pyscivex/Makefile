.PHONY: build develop test lint fmt clean

# Build the extension module in release mode
build:
	maturin build --release

# Build in-place for development (fast iteration)
develop:
	maturin develop --release

# Run all tests
test: develop
	pytest tests/ -v

# Run linter
lint:
	ruff check python/ tests/

# Format code
fmt:
	ruff format python/ tests/

# Clean build artifacts
clean:
	rm -rf target/wheels/ dist/ build/ *.egg-info
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
