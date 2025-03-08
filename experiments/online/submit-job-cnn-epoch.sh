# for file in configs/$1/*; do

for file in $1/*; do
    echo "Submitting job for $file"
    sbatch job-cnn-epoch.sh $file "../cnn/checkpoints/"
done