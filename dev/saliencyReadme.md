# Saliency README file

This file contains some information about the developing of the saliency project code

## Files

### `predictError.lua`

Reproduces the *testing* step of the *network training* procedure (just to be sure I am not doing weird stuff).

### `TestDataset.t7`

The whole *test dataset* has been extracted and saved into a Lua table `{}` under the entry with the labels's name. The structure is the following: each entry (`barcode`, `bootle`, `dog`, etc...) is an *array* of `{error, image}` table. `error` is the output of the *cost function*, i.e. the `LogSoftMax()` of the *correct output* and `image` contains a `float` version of the *input image*.

This file contains also a `classes` entry which contains an *array* of the available classes: each *index* correspond to the correct *class index*.

### `Top10TestData.t7`

The top-10 predictions per each testing class have been stored into this file. The structure is alike the `TestDataset.t7` file, but the `classes` entry is missing in this case.

### `getTop10.lua`

Extracts `Top10TestData.t7` from `TestDataset.t7`.

### `dispTop10.lua`

Displays all (17) categories top-10 test images and corresponding label.
