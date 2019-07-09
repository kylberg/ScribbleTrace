# ScribbleTrace

_The repository is in a very early prototyping stage._

The aim with this repo is to design a few algorithms for tracing bitmaps into patterns reflecting intensities that can be exported as svg to be used with e.g. pen plotters. If time allows a GUI in pyqt5 might be added for increased usability.

## Planned tracing styles

* Scribble squares
* Scribble lines
* Scribble curves
* Continious scribble curve


## Preliminary results

The first test of the scribble squares style:
![Screenshot](https://github.com/kylberg/ScribbleTrace/blob/personal/kylberg/PoC-scribble-squares/examples.png)

__Top left__ Original image. __Top right__ gray scale, quantized and downsampled version. __Middle left__ The _scribble squares_ style, drawing squares from outside in.  __Middle right__ The _scribble squares_ style, drawing squares from inside out. __Bottom left__ The _scribble lines_ style with straight lines along gradients.  __Bottom right__ The _Scribble curves_ style, drawing short Bezier curves along gradients.

## Background and inspiration
Around 2014 I stumbled upon the inspiring work by Sandy Noble on his Polargraph drawing machine (http://www.polargraph.co.uk/). Drawing machines and plotters preexisting inkjet printers have always interested me but Sandy's work was something else.

The Polagraph control software implements several interesting strategies of halftoneing or if you like tracing of bitmaps by drawing different patterns rather than just placing dots, with their distribution reflecting the bitmap intensity, which is a common strategy.

I like the idea of replacing pixels with patterns and I believe it all started with noticing halftoneing in print. Another moment of inspiration was during my PhD when a colleague wrote a software converting a bitmap into picture of an embroidery with cross stitches. Among the tracing styles on the Polargraph software there is one called Norwegian pixels that like and suddenly I stumbeled upon an excellent small software caled Zebra Trace by Maxim Barabash (https://github.com/maxim-s-barabash/ZebraTrace/). 

These moments of inspiration combined led me to create this repository. 


