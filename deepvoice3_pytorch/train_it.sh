if [ -z "$1" ]; then
    echo "Usage: train_it.sh path/to/base/data/dir"
    exit
fi
python train.py --preset=presets/nyanko_ljspeech.json --data-root="$1"
