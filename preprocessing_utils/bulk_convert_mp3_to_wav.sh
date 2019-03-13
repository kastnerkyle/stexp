wav_dir=raw_wav
mp3_dir=raw_mp3
base_dir=$(pwd)
for f in "$mp3_dir"/*.mp3; do
    fbase=$(basename $f)
    basenoext=${fbase%.*}
    echo ffmpeg -y -i "$mp3_dir"/"$basenoext".mp3 -acodec pcm_s16le -ac 1 -ar 22050 "$wav_dir"/"$basenoext".wav
    ffmpeg -y -i "$mp3_dir"/"$basenoext".mp3 -acodec pcm_s16le -ac 1 -ar 22050 "$wav_dir"/"$basenoext".wav
done
