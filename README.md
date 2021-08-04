# pyimgdata

A tool for collecting CC and public domain images from flickr based of [Jeff Heaton's pyimgdata](https://github.com/jeffheaton/pyimgdata) project.

This tool is barely prototype stage, but was useable enough for my personal use for the [Sound Escapes](http://www.ryankelln.com/project/sound-escapes/) concert.


## Install

### Using conda 
```bash
$ conda create --prefix envs/ python=3.8
$ conda activate envs/
$ pip install -r requirements.txt
```

There needs to be a fix in `flickrapi` to support paging:

In your conda environment, e.g. `lib/python3.9/site-packages/flickrapi/core.py`
there is a small change to `data_walker` function:

```python
def data_walker(self, method, searchstring='*/photo', **params):

    page = params.pop('page', 1)
    total = page  # We don't know that yet, update when needed
```

In addition with recent versions of python you need to change:

```diff
def data_walker(self, method, searchstring='*/photo', **params):

-    photoset = rsp.getchildren()[0]
+    photoset = list(rsp)[0]

```

## Running

Create your own ini settings file. Copy `example.ini` and put in your own settings:

```
[FLICKR]
id = <your flickr id>
secret = <your flickr api secret>
[Download]
path = images/saved/to/path
search = example keywords
prefix = example                 # prefix added to images
update_minutes = 1
license = 0,1,2,3,4,5,6,7,8,9,10
image_count = 500
sources_file = my_sources.csv    # save urls, prevent duplicates
image_size = l                   # flickr image size search restrictions, see below
orientation = landscape          # [optional] flickr orientation (landscape/portrait)
tag_mode = all                   # [optional] flickr tag mode (all/any)
start_page = 0                   # [optional] starting page, defaults to automatic paging
[Process]
process = True
crop_square = True
min_width = 256
min_height = 256
scale_width = 256
scale_height = 256
image_format = jpg
[ImageHash]
algorithm = crop_resistant      # (phash, dhash, average_hash, crop_resistant)
threshold = 3                   # [default : 4] percentage of difference (<1.0 or > 1)
```

Running:

```bash
$ python flickr-download.py example.ini

```

If opencv is installed, which I recommend, then each downlaoded image will display in a window to be validated and cropped. Select the window, then use the keyboard commands:

* `y`: yes, accept the image (hitting enter/return will also accept)
* `n`: no, reject the image
* `c`: crop the image: using the left mouse button select top left corner and drag and release at the bottom right corner (sorry, no display of the selection rect yet)
* `q`: quit

If the config option `sources_file` is set then it is safe stop and restart where you left off. 

The [ImageHash](https://github.com/JohannesBuchner/imagehash) project is used to reject images that are too similar to existing images. 


# Flickr API info

## Photo Source URLs

See flickr's docs:
* https://www.flickr.com/services/api/flickr.photos.search.html
* https://www.flickr.com/services/api/misc.urls.html

The list looks like:
`url_sq, url_t, url_s, url_q, url_m, url_n, url_z, url_c, url_l, url_o`

Where:
`url_l` = best option for 1024 sized images?

```
s   small square 75x75
q   large square 150x150
t   thumbnail, 100 on longest side
m   small, 240 on longest side
n   small, 320 on longest side
-   medium, 500 on longest side
z   medium 640, 640 on longest side
c   medium 800, 800 on longest side†
b   large, 1024 on longest side*
h   large 1600, 1600 on longest side†
k   large 2048, 2048 on longest side†
o   original image, either a jpg, gif or png, depending on source format
```

## Flickr licence info

These are the Flickr License identifiers for images.

* 0 All Rights Reserved
* 1 Attribution-NonCommercial-ShareAlike License
* 2 Attribution-NonCommercial License
* 3 Attribution-NonCommercial-NoDerivs License
* 4 Attribution License
* 5 Attribution-ShareAlike License
* 6 Attribution-NoDerivs License
* 7 No known copyright restrictions
* 8 United States Government Work
* 9 Public Domain Dedication (CC0)
* 10 Public Domain Mark