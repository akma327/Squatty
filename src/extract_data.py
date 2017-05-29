import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import PIL

def resize_images(data):
  f = open('../../data/images/joint_annotation_data_scaled.txt', 'w')
  for i in range(len(data)):
    try:
      line = data[i].split()
      image_name = line[0]
      img = mpimg.imread("../../data/images/" + image_name)
      line = line[1:]
      x = line[:len(line) / 2]
      y = line[len(line) / 2:]
      
      im = Image.open("../../data/images/" + image_name)
      size = 480, 640
      im = im.resize(size, PIL.Image.ANTIALIAS)
      im.save("../../data/images_scaled/" + image_name)

      newline = [image_name]
      xdim, ydim, zdim = img.shape
      for xval in x:
        newline.append(str(float(xval) * size[0] / xdim))
      for yval in y:
        newline.append(str(float(yval) * size[1] / ydim))
      newline = "\t".join(newline)
      f.write(newline + "\n")
    except:
      print "Invalid file"
  f.close()

def plot_images(data, num_to_plot=5):
  for i in range(num_to_plot):
    line = data[i].split()
    image_name = line[0]
    img = mpimg.imread("../../data/images/" + image_name)
    line = line[1:]
    x = line[:len(line) / 2]
    y = line[len(line) / 2:]

    plt.imshow(img)
    plt.scatter(x, y)
    plt.show()

def extract_data():
  with open("../../data/images/joint_annotation_data.txt") as f:
    data = f.readlines()
  resize_images(data)

if __name__ == "__main__":
  extract_data()
