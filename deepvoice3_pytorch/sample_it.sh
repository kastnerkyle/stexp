if [ -z "$1" ]; then
      echo "Usage: script.sh path/to/checkpoint.pth"
fi
#python synthesis.py --preset=presets/nyanko_ljspeech.json /home/ubuntu/deepvoice3_models/20171129_nyanko_checkpoint_step000585000.pth sentences.txt output_dir
python synthesis.py --preset=presets/nyanko_ljspeech.json $1 sentences.txt output_dir
