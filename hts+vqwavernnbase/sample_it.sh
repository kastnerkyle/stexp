SPKR_DIR=sample_stage1_wav/p1
DATA_DIR=sample_data_dir
OUT_DIR=model_outputs
rm -rf "$DATA_DIR"
rm -rf "$SPKR_DIR"
rm -rf "$OUT_DIR"
mkdir -p $SPKR_DIR

for i in `seq 109`; do 
    echo $i
    bash say_it.sh "The only A I you can truly trust is you " "$SPKR_DIR"/out$i.wav
done

bash bulk_downsample_wav.sh
python preprocess_multispeaker.py sample_stage1_wav/ sample_data_dir
python sample_wavernn.py -g
