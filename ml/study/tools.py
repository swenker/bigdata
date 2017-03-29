__author__ = 'wenjusun'

from PIL import Image

def get_bitmap(bitmap_path):
    img = Image.open(bitmap_path)

    print type(img.getdata())
    pix_val = list(img.getdata())
    print len(pix_val)
    print pix_val[0] #it's a rgb tuple?
    print img.getdata()

    return pix_val


get_bitmap(r'C:\ZZZZZ\0-sunwj\bigdata\data\bmp\2-1.bmp')