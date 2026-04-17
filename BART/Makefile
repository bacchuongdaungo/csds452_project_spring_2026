ENV = bart_ihdp

install:
	conda create -n $(ENV) python=3.10 -y
	conda install -n $(ENV) -c conda-forge pymc pymc-bart scikit-learn pandas "numpy<2.2" -y
	conda run --no-capture-output -n $(ENV) pip install kagglehub
	conda run --no-capture-output -n $(ENV) pip install optuna

test:
	conda run --no-capture-output -n $(ENV) python3 bart_ihdp.py --test

run:
    conda run --no-capture-output -n $(ENV) python3 bart_ihdp.py \
        --trees 200 --draws 1000 --tune 1000 --alpha 0.87 --beta 1.92

tune:
	conda run --no-capture-output -n bart_ihdp python3 bart_tune.py

clean:
	rm -f bart_results.csv bart_results_test.csv