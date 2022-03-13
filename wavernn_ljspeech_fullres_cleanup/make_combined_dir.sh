# this script makes a folder data_dir such that
# WaveRNN will train to go from dirty mel -> clean wav
SCRIPT=`readlink -f ${BASH_SOURCE[0]} || greadlink -f ${BASH_SOURCE[0]}`
SCRIPTPATH=`dirname "$SCRIPT"`
mkdir -p $SCRIPTPATH/data_dir
mkdir -p $SCRIPTPATH/data_dir/test
ln -s $SCRIPTPATH/clean_data_dir/wav $SCRIPTPATH/data_dir/wav
# these should be the same!
ln -s $SCRIPTPATH/clean_data_dir/dataset_ids.pkl $SCRIPTPATH/data_dir/dataset_ids.pkl
ln -s $SCRIPTPATH/dirty_data_dir/mel $SCRIPTPATH/data_dir/mel
pushd .
cd data_dir/test
for f in "$SCRIPTPATH/clean_data_dir/test/*wav*"; do
    ln -s $f .
done

for f in "$SCRIPTPATH/dirty_data_dir/test/*mel*"; do
    ln -s $f .
done
popd
