# smartEYE saliency project

This collection of scripts computes both *bottom-up* and *top-down* saliency map.
A theoretical explanation can be found in the VADNN corresponding article.

## How to run a live person top-down saliency demo

There are available three different methods with which the final top-down saliency map is displayed.
Each method can be selected by `--mode #`, where `#` can be 1, 2 or 3.
The source can be a USB webcam, which is the default option, or an Ethernet camera, which can be chosen with `--eth`.
For example, we can run on USB, mode 3 by typing

```bash
qlua saliencyCam.lua --mode 3
```

or, we could run on Ethernet and using method 2 with

```bash
qlua saliencyCam.lua --mode 2 --eth
```
