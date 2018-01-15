# Wavpack sink plugin

A simple Gstreamer plugin for writing [wavpack](http://www.wavpack.com/) files containing multiple audio chunks with a length defined by the user. The main difference with the normal plugin is that once a certain number of chunks, for example 5, have been buffered they are written on disk and a new wavpack file is created.

### CONTRIBUTORS

Josue R. Cuevas

josuercuevas@gmail.com
