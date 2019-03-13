wav_dir=sample_stage1_wav
hq_dir=sample_stage1_wav
base_dir=$(pwd)
mkdir -p $wav_dir 
for f in `ls "$hq_dir"/*/*.wav`; do
    q="tmp.wav"
    cp $f $q
    echo ffmpeg -y -i "$q" -acodec pcm_s16le -ac 1 -ar 22050 "$f"
    ffmpeg -y -i "$q" -acodec pcm_s16le -ac 1 -ar 22050 "$f"
    rm "$q"
done
