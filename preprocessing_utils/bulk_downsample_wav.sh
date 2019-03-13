wav_dir=raw_wav
hq_dir=hq_wav
base_dir=$(pwd)
mkdir -p $wav_dir 
for f in `ls "$hq_dir"/*.wav`; do
    fbase=$(basename $f)
    basenoext=${fbase%.*}
    echo ffmpeg -y -i "$hq_dir"/"$basenoext".wav -acodec pcm_s16le -ac 1 -ar 22050 "$wav_dir"/"$basenoext".wav
    ffmpeg -y -i "$hq_dir"/"$basenoext".wav -acodec pcm_s16le -ac 1 -ar 22050 "$wav_dir"/"$basenoext".wav
done
