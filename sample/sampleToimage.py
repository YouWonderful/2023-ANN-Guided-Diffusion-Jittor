import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import argparse
import shutil

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_file', default='./samples_4x256x256x3.npz', type = str, help = 'sample file name')
    args = parser.parse_args()
    
    sample = np.load(args.sample_file)
    images = sample['arr_0']
    labels = sample['arr_1']
    
    if not os.path.exists('./image/'):
        os.makedirs('./image/')
    else:
        shutil.rmtree('./image/')
        os.makedirs('./image/')
        
    for i in range(images.shape[0]):
        image = images[i]
        im = Image.fromarray(image)
        im.save("./image/sample_" + str(i) + ".jpeg")
        # plt.imshow(image)
        # plt.show()
    print("Successfully executed")
    
if __name__=='__main__':
    main()