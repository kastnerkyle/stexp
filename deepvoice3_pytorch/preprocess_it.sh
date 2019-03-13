if [ -z "$1" ]; then
    echo "Usage: preprocess_it.sh name_of_dataset path/to/base/data/dir output/data/path"
    exit
fi
if [ -z "$2" ]; then
    echo "Usage: preprocess_it.sh name_of_dataset path/to/base/data/dir output/data/path"
    exit
fi
if [ -z "$3" ]; then
    echo "Usage: preprocess_it.sh name_of_dataset path/to/base/data/dir output/data/path"
    exit
fi
#python preprocess.py --preset=presets/nyanko_ljspeech.json obama_amazon /home/ubuntu/obama_amazon_data nyanko_obama_amazon
python preprocess.py --preset=presets/nyanko_ljspeech.json $1 $2 $3
