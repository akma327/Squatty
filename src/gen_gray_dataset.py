# gen_gray_dataset.py


import glob
from PIL import Image

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


input_files = glob.glob("/scratch/PI/rondror/akma327/classes/CS231A/project/data/images_scaled/*jpg")

for i, ifile in enumerate(input_files):
	if(i %50 == 0): print(i)
	img = Image.open(ifile).convert('L')
	outfile = "/scratch/PI/rondror/akma327/classes/CS231A/project/data/images_scaled_gray/" + ifile.split("/")[-1]
	img.save(outfile)
