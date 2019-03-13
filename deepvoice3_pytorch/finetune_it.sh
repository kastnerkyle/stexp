if [ -z "$1" ]; then
    echo "Usage: finetune_it.sh path/to/checkpoint.pth path/to/dataset"
    exit
fi
if [ -z "$2" ]; then
    echo "Usage: finetune_it.sh path/to/checkpoint.pth path/to/dataset"
    exit
fi
python finetune.py --preset=presets/nyanko_finetune.json --data-root="$2" --checkpoint="$1" --reset-optimizer
#python train.py --preset=presets/nyanko_finetune.json --data-root=nyanko_obama_amazon
