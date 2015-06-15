# smartEYE

This is an implemetation of the [VADNN](http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=7086900) article.
This collection of scripts for [Torch7](http://torch.ch/) can compute both *bottom-up* and *top-down* visual attention saliency maps.


## You need to fetch some data

You need to fetch the trained model, at least, to be able to run the live demo.

```bash
./getData-public.sh
```


## Dependency

If you want to use an USB camera, you will need to install the `camera` package

```bash
luarocks install camera
```


## How to run a live person top-down saliency demo

There are available three different methods with which the final top-down saliency map is displayed.
Each method can be selected by `--mode #`, where `#` can be 1, 2 or 3.
The source can be a USB webcam, which is the default option, or an Ethernet camera, which can be chosen with `--eth`.
For example, we can run on USB, mode 3 by typing

```bash
cd dev
qlua saliencyCam.lua --mode 3
```

or, we could run on Ethernet and using method 2 with

```bash
qlua saliencyCam.lua --mode 2 --eth
```
