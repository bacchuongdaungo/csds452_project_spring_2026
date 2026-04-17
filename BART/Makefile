ENV = bart_ihdp

install:
	conda create -n $(ENV) python=3.10 -y
	conda install -n $(ENV) -c conda-forge pymc pymc-bart scikit-learn pandas "numpy<2.2" -y
	conda run --no-capture-output -n $(ENV) pip install kagglehub

test:
	conda run --no-capture-output -n $(ENV) python3 bart_ihdp.py --test

run:
	conda run --no-capture-output -n $(ENV) python3 bart_ihdp.py --trees 200 --draws 1000 --tune 1000

clean:
	rm -f bart_results.csv bart_results_test.csv