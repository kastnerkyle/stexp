# example command to convert from mp3 to 22050 kHz wav
ffmpeg -y -i blah.mp3 -acodec pcm_s16le -ac 1 -ar 22050 blah.wav
