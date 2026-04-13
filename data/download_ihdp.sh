echo "Starting download of IHDP data..."
mkdir -p ihdp_dataset/csv

base="https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/9081f863e24ce21bd34c8d6a41bf0edc7d1b65dd/datasets/IHDP"

curl -L "$base/columns.txt" -o ihdp_dataset/columns.txt

for i in 1 2 3 4 5 6 7 8 9 10
do
  curl -L "$base/csv/ihdp_npci_${i}.csv" -o "ihdp_dataset/csv/ihdp_npci_${i}.csv"
done

echo "Finished downloading IHDP data."