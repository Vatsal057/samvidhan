.PHONY: install ingest run eval test lint docker clean

install:        ## Install the package (editable) + dev dependencies
	pip install -r requirements-dev.txt
	pip install -e .

ingest:         ## Build the vector store from the bundled sample corpus
	python -m samvidhan.ingest --text data/sample

run:            ## Launch the Streamlit app
	streamlit run app.py

eval:           ## Run the retrieval evaluation harness
	python eval/run_eval.py

test:           ## Run the unit test suite
	pytest -q

lint:           ## Lint with ruff
	ruff check src tests eval

docker:         ## Build and run the container
	docker build -t samvidhan . && docker run --rm -p 8501:8501 samvidhan

clean:
	rm -rf chroma_db .pytest_cache .ruff_cache **/__pycache__
