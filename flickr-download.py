# Flickr Download, by Jeff Heaton (http://www.heatonresearch.com)
# https://github.com/jeffheaton/pyimgdata
# Copyright 2020, MIT License
import configparser
import csv
import logging
import logging.config
import math
import os
import re
import sys
import time
from hashlib import sha256
from io import BytesIO
from urllib.parse import urlparse
from urllib.request import urlretrieve

import cv2
import flickrapi
import imagehash
import numpy as np
import requests
from PIL import Image
from tqdm import tqdm

# sources column format
SEARCH_TERM = 0
URL = 1
PATH = 2
HASH = 3

# https://code.flickr.net/2008/08/19/standard-photos-response-apis-for-civilized-age/

# Nicely formatted time string
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m:>02}:{s:>05.2f}"
        
def is_true(str):
    return str.lower()[0] == 't'

"""
Config:

[FLICKR]
id = [your api id]
secret = [your secret/password]
[Download]
path = images/saved/to/path
search = christmas decorations
prefix = christmas               # prefix added to images
update_minutes = 1
license = 0,1,2,3,4,5,6,7,8,9,10
image_count = 100000
sources_file = sources.csv       # save urls, prevent duplicates
image_size = l                   # flickr image size search restrictions, see below
orientation = landscape          # [optional] flickr orientation (landscape/portrait)
tag_mode = all                   # [optional] flickr tag mode (all/any)
start_page = 1                   # [optional] starting page, defaults to automatic paging
[Process]
process = True
crop_square = True
min_width = 256
min_height = 256
scale_width = 256
scale_height = 256
image_format = jpg
[ImageHash]
algorithm = crop_resistant
threshold = 3                   # [default : 4] percentage of difference (<1.0 or > 1)


Photo Source URLs
https://www.flickr.com/services/api/flickr.photos.search.html
https://www.flickr.com/services/api/misc.urls.html

url_sq, url_t, url_s, url_q, url_m, url_n, url_z, url_c, url_l, url_o

url_l = best option for 1024 sized images?

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

"""

def url_to_id(url):
    # could be a csv list item, get last col
    if isinstance(url, list):
        url = url[-1]

    if url.isdigit():
        return url

    result = urlparse(url)
    m = re.search("/\d+/(\d+)_", result.path)
    if m is None:
        raise RuntimeError(f"Cannot find id from {url}")
    #print("{} to id: {}".format(result.path, m.group(1)))
    return m.group(1)

class FlickrImageDownload:
    def __init__(self, config='config_flickr.ini'):
        self.config = configparser.ConfigParser()
        self.config.read(config)
        logging.config.fileConfig("logging.properties")
        
        self.config_path = self.config['Download']['path']
        self.config_prefix = self.config['Download']['prefix']
        self.config_search = self.config['Download']['search']
        self.config_update_minutes = int(self.config['Download']['update_minutes'])
        self.config_image_count = int(self.config['Download']['image_count'])
        self.config_license_allowed = [int(e) if e.isdigit() else e 
            for e in self.config['Download']['license'].split(',')]
        self.config_format = self.config['Process']['image_format']
        self.config_process = is_true(self.config['Process']['process'])
        self.config_crop_square = is_true(self.config['Process']['crop_square'])
        self.config_scale_width = int(self.config['Process']['scale_width'])
        self.config_scale_height = int(self.config['Process']['scale_height'])
        self.config_min_width = int(self.config['Process']['min_width'])
        self.config_min_height = int(self.config['Process']['min_height'])
        
        # optional config:
        self.config_flickr_tag_mode = self.config['Download'].get('tag_mode', 'all')
        self.config_flickr_size = self.config['Download'].get('image_size', 'c')
        self.config_sources_file = self.config['Download'].get('sources_file', None)
        self.config_flickr_orientation = self.config['Download'].get('orientation', None)
        self.config_start_page = self.config['Download'].get('start_page', None)
        if self.config_start_page is not None:
            self.config_start_page = int(self.config_start_page)

        # image hash config
        if self.config.has_section('ImageHash'):
            hashmethod = self.config['ImageHash'].get('algorithm', 'ahash')
            if hashmethod == 'ahash' or hashmethod == 'average':
                hashfunc = imagehash.average_hash
            elif hashmethod == 'phash' or hashmethod == 'perceptual':
                hashfunc = imagehash.phash
            elif hashmethod == 'dhash' or hashmethod == 'difference':
                hashfunc = imagehash.dhash
            elif hashmethod == 'whash-haar' or hashmethod == 'haar_wavelet':
                hashfunc = imagehash.whash
            elif hashmethod == 'whash-db4' or hashmethod == 'daubechies_wavelet':
                hashfunc = lambda img: imagehash.whash(img, mode='db4')
            elif hashmethod == 'colorhash' or hashmethod == 'color':
                hashfunc = imagehash.colorhash
            elif hashmethod == 'crop-resistant' or hashmethod == 'crop_resistant':
                hashfunc = imagehash.crop_resistant_hash
            self.config_hash_func = hashfunc
            self.config_hash_threshold = float(self.config['ImageHash'].get('threshold', 4))
            if self.config_hash_threshold >= 1.0:
                self.config_hash_threshold /= 100.0
        else:
            self.config_hash_func = None

        self.flickr=flickrapi.FlickrAPI(
            self.config['FLICKR']['id'], 
            self.config['FLICKR']['secret'], 
            cache=True)
        print(self.flickr)
        
        # ensure paths exist
        os.makedirs(self.config_path, exist_ok=True)
    

    def reset_counts(self):
        self.start_time = time.time()
        self.count = 0
        self.cached = 0
        self.download_count = 0
        self.last_update = 0
        self.error_count = 0
        self.sources = []
        self.hashes = set()

        
    def load_image(self, url):
        """Returns a PIL image and and openCV image"""
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content)).convert('RGB')
            img.load()
            try:
                cv_img = cv2.imdecode(np.asarray(bytearray(response.content)), 1)
            except Exception as e:
                print("Failed to decode cv image: ", e)
                return img, None
            return img, cv_img
        except KeyboardInterrupt:
            logging.info("Keyboard interrupt, stopping")
            sys.exit(0)
        except:
            logging.warning(f"Unexpected exception while downloading image: {url}" , exc_info=True)
            return None, None
        

    def obtain_photo(self, photo):
        """Returns a PIL image and and openCV image"""
        url = photo.get('url_c')
        license = photo.get('license')

        if int(license) in self.config_license_allowed and url:
            image, cv_image = self.load_image(url) 
            
            if image:
                return image, cv_image
            else:
                self.error_count += 1
                
        return None, None
    
    def check_to_keep_photo(self, url, image):
        if self.config_hash_func:
            h = self.config_hash_func(image)
        else:
            h = sha256(image.tobytes()).hexdigest()

        file_path = f"{self.config_prefix}-{h}.{self.config_format}"
        p = os.path.join(self.config_path, file_path)
        exists = os.path.exists(p)

        # see if hash exists in sources already
        if h in self.hashes:
            if exists:
                self.cached += 1
                logging.debug(f"Image already exists: {url}")
                return None
            else:
                logging.debug(f"Image already retrieved, but rejected: {url}")
                return None

        # check for closeness of hash match
        if self.config_hash_func is not None:
            closest_match = 1000000
            bits_threshold = max(1, round(float(len(h)) * self.config_hash_threshold))
            for other_hash in self.hashes:
                diff = h - other_hash
                if diff < closest_match:
                    closest_match = diff
                if diff <= bits_threshold:
                    logging.debug(f"Image too similar to existing (diff: {diff} <= {bits_threshold} ({self.config_hash_threshold}%)): {url}")
                    return None
            closest_percent = round(float(closest_match) / float(len(h)) * 100.0, 1)
            logging.info(f"Closest match: {closest_match} ({closest_percent}%)")

        if not exists:
            self.sources.append([self.config_search, url, file_path, h])
            self.hashes.add(h)
            self.download_count += 1
            logging.debug(f"Downloaded: {url} to {p}")
            return p
        else:
            self.cached += 1
            logging.debug(f"Image already exists: {url}")
            return None
        

    def process_image(self, image, path):        
        width, height = image.size
        
        if self.config_process:
            # Crop the image, centered
            if self.config_crop_square:
                new_width = min(width,height)
                new_height = new_width
                left = (width - new_width)/2
                top = (height - new_height)/2
                right = (width + new_width)/2
                bottom = (height + new_height)/2
                image = image.crop((left, top, right, bottom))
                
            # Scale the image
            if self.config_scale_width > 0:
                image = image.resize((
                    self.config_scale_width, 
                    self.config_scale_height), 
                    Image.ANTIALIAS)


        # Convert to full color (no grayscale, no transparent)
        if image.mode not in ('RGB'):
            logging.debug(f"Grayscale to RGB: {path}")
            rgbimg = Image.new("RGB", image.size)
            rgbimg.paste(image)
            image = rgbimg
            
        return image


    def crop(self, image, edge, percent):
        w, h = image.width, image.height
        box =  (0, 0, w, h)
        if edge == '':
            edge = 'a'
        edge = edge.lower()[0]

        if percent >= 1.0:
            percent /= 100.0

        if edge == 't':
            box = (0, math.floor(h * percent), w, h)
        if edge == 'b':
            box = (0, 0, w, math.floor(h * (1.0 - percent)))
        if edge == 'l':
            box =(math.floor(w * percent), 0, w, h)
        if edge == 'r':
            box = (0, 0, math.floor(w * (1.0 - percent)), h)
        if edge == 'a':
            box = (
                math.floor(w * percent),
                math.floor(h * percent),
                math.floor(w * (1.0 - percent)),
                math.floor(h * (1.0 - percent))
            )
        
        return image.crop(box)


    def track_progress(self):
        elapsed_min = int((time.time() - self.start_time)/60)
        self.since_last_update = elapsed_min - self.last_update
        if self.since_last_update >= self.config_update_minutes:
            logging.info(f"Update for {elapsed_min}: images={self.download_count:,}; errors={self.error_count:,}; cached={self.cached:,}")
            self.last_update = elapsed_min

        if self.count >= self.config_image_count:
            logging.info("Reached max download count")
            return True
        
        return False
    

    def write_sources(self):
        if self.config_sources_file:
            logging.info("Writing sources file.")
            filename = os.path.join(self.config_path, self.config_sources_file)
            with open(filename, 'w', newline='') as csvfile:  
                csvwriter = csv.writer(csvfile)  
                csvwriter.writerow(['search', 'url', 'file', 'hash'])  
                csvwriter.writerows(self.sources)


    def load_sources(self):
        if self.config_sources_file:
            filename = os.path.join(self.config_path, self.config_sources_file)
            if os.path.exists(filename):
                with open(filename) as f:
                    reader = csv.reader(f)
                    self.sources = [row for row in reader] 
                self.sources.pop(0) # remove header
                if self.config_hash_func is not None:
                    self.hashes.update([imagehash.hex_to_hash(row[HASH]) for row in self.sources])
                # find existing count of images
                # files in sources with existing files
                for row in self.sources:
                    if row[PATH] != '':
                        if os.path.exists( os.path.join(self.config_path, row[PATH]) ):
                            self.count += 1
                        else:
                            logging.debug(f"Image path doesn't exist: {row[PATH]}")
                            row[PATH] = ''  # remove it

    def run(self):
        logging.info("Starting...")

        self.reset_counts() 

        # load existing sources
        self.load_sources()
        ignore_ids = [url_to_id(url) for search, url, f, hash in self.sources]

        if self.count >= self.config_image_count:
            logging.info(f"Already obtained {self.config_image_count} images in {self.config_path}")
            return

        # get flickr urls
        flickr_url = f"url_{self.config_flickr_size}"
        
        # paging
        per_page = 50
        if self.config_start_page is None:
            page_num = 0
            if self.count > 0:
                page_num = math.floor(self.count / per_page)
        else:
            page_num = self.config_start_page

        params = dict(
            text=self.config_search,
            #tag_mode=self.config_flickr_tag_mode,
            #tags=self.config_search,
            extras=f"{flickr_url},license",
            per_page=per_page,           
            sort='relevance',
            content_type=1,   # photos only
            orientation=self.config_flickr_orientation,
            media='photos',
            page=page_num
        )
        print(params)
        photos = self.flickr.walk(**params)

        # opencv image window
        cv2.namedWindow('image')
        global posList
        global drawing
        posList = []
        drawing = False

        def getMouseHandler(img):
            def onMouse(event, x, y, flags, param):
                global posList, drawing
                if event == cv2.EVENT_LBUTTONDOWN:
                    posList = [(x,y)]
                    drawing = True
                    print("start drwawing", x, y)
                elif event == cv2.EVENT_MOUSEMOVE:
                    if drawing == True:
                        cv2.rectangle(img, pt1=posList[0], pt2=(x, y), color=(255,0,0),thickness=1)
                        print(x, y)
                elif event == cv2.EVENT_LBUTTONUP:
                    posList.append((x, y))
                    cv2.rectangle(img, pt1=posList[0], pt2=(x, y), color=(0,0,0),thickness=1)
                    drawing = False
                    print("stop drawing", x, y)
            return onMouse

        images_bar = tqdm(desc="Images", unit="", initial=self.count, total=self.config_image_count)
        urls_bar = tqdm(photos, desc="Urls processed", unit="")

        # input closure
        def get_input(text):
            tqdm.write(text)
            urls_bar.clear()
            images_bar.clear()
            return input()

        quit = False
        for photo in urls_bar:
            try:
                url = photo.get(flickr_url)
                logging.debug(f"Retrieved url: {url}")
                if url == '' or url == None:
                    logging.debug("Url could not be fetched")
                    tqdm.write("Url could not be fetched")
                    time.sleep(0.5) # slow down
                    pass
                else:
                    photo_id = url_to_id(url)
                    if photo_id not in ignore_ids:
                        img, cv_img = self.obtain_photo(photo)
                        if img: 
                            path = self.check_to_keep_photo(url, img)
                            if path:
                                tqdm.write("Found new image")
                                img = self.process_image(img, path)
                                while True:
                                    tqdm.write("Keep image? Y/n/[c]rop)/[q]uit: ")
                                    if cv_img is not None:
                                        tqdm.write("Select image window before keyboard input, left click drag to crop")
                                        posList = []
                                        def onMouse(event, x, y, flags, param):
                                            global posList
                                            if event == cv2.EVENT_LBUTTONDOWN:
                                                posList = [(x,y)]
                                            elif event == cv2.EVENT_LBUTTONUP:
                                                posList.append((x, y))
                                        cv2.setMouseCallback('image', onMouse)
                                       
                                        cv2.imshow('image', cv_img)
                                        k = cv2.waitKey(0)
                                    else:
                                        img.show()
                                        k = get_input()

                                    if isinstance(k, int):
                                        k = chr(k)
                                    if isinstance(k, str):
                                        if k.strip() == '' or k == "\r" or k == "\n":
                                            k = 'y'
                                        k = k.lower()[0]
                                    else:
                                        logging.error("Invalid input")
                                        sys.exit(1)

                                    tqdm.write(f"Input: {k}")
                                    if k == 'c':
                                        if len(posList) == 2:
                                            box = (
                                                posList[0][0],
                                                posList[0][1],
                                                posList[1][0],
                                                posList[1][1],
                                            )
                                            img = img.crop(box)
                                        else:
                                            crop = get_input("Crop [t]op/[b]ottom/[l]eft/[r]ight/[a]ll: ")
                                            percent = get_input("Crop percent (0-1.0): ")
                                            img = self.crop(img, crop, float(percent))
                                        cv_img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
                                    elif k == 'n':
                                        tqdm.write("Rejected image")
                                        # remove filename from sources
                                        self.sources[-1][PATH] = ""
                                        break
                                    elif k == 'q':
                                        quit = True
                                        break
                                    else:
                                        img.save(path)
                                        self.count += 1
                                        images_bar.update(1)
                                        images_bar.refresh()
                                        urls_bar.refresh()
                                        break
            except Exception as e:
                logging.warn("Url fetch failed: ", e)
                tqdm.write("Url fetch failed")
                time.sleep(0.5) # slow down
            if self.track_progress() or quit:
                break
        
        urls_bar.close()
        images_bar.close()
        cv2.destroyAllWindows()

        self.write_sources()
        elapsed_time = time.time() - self.start_time
        logging.info("Complete, elapsed time: {}".format(hms_string(elapsed_time)))



if __name__=='__main__':
    if len(sys.argv) == 2:
        task = FlickrImageDownload(sys.argv[1])
    else:
        task = FlickrImageDownload()
    task.run()
