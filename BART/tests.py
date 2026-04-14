import numpy as np

# Quick check of the raw data for one of the IHDP replicas (the same source as the CEVAE) one seems to be much worse, why?
data = np.loadtxt("data/ihdp_dataset/csv/ihdp_npci_9.csv", delimiter=",")
print("treatment rate:", data[:,0].mean())
print("outcome mean:", data[:,1].mean())
print("outcome std:", data[:,1].std())
print("\n")

data = np.loadtxt("data/ihdp_dataset/csv/ihdp_npci_10.csv", delimiter=",")
print("treatment rate:", data[:,0].mean())
print("outcome mean:", data[:,1].mean())
print("outcome std:", data[:,1].std())