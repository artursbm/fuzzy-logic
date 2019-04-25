# Artur Mello
# Fuzzy C Means - Image Segmentation
# TP 1 - Sistemas Nebulosos
import os
import numpy as np
from fuzzy_c_means import fuzzy_c_means
from PIL import Image


def main():
    dire = "ImagensTeste"
    images_in = os.listdir(dire)
    
    for image in images_in:
        k = 8
        print("Segmentando imagem {}".format(image))
        img = Image.open(os.path.join(dire, image))
        width, height = img.size
        _pixels = np.asarray(img.getdata())

        pixels, centroids, data_cluster, it = fuzzy_c_means(_pixels, k)

        new_pixels = np.zeros([len(data_cluster), 3])
        for i in range(len(data_cluster)):
            new_pixels[i] = centroids[int(data_cluster[i])]

        new_pixels = new_pixels.reshape(height, width, 3)
        new_image = Image.fromarray(new_pixels.astype('uint8'))
        new_image_name = 'cluster_' + image
        new_image.save('ImagensSegmentadas/'+new_image_name, 'JPEG')


if __name__ == "__main__":
    main()
