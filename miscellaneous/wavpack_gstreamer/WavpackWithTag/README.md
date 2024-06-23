# Tag metadata writer to wavpack-compressed audio chunks

This is a Gstreamer plugin which is in charged of inserting metadata tags to audio chunks compressed with [wavpack](http://www.wavpack.com/). This plugin can be compiled and ran in a normal Gstreamer pipeline where all the user has to provide is the audio chunk and the metadata to be added at the end of each chunk. The chunks could be later decoded with WavpackDec API in order to extract the metadata.

### CONTRIBUTORS

Josue R. Cuevas

josuercuevas@gmail.com
