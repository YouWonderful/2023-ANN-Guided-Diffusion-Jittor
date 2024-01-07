import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', default=1, type = int, help = 'scale')
    args = parser.parse_args()
    
    if args.scale == 1:
        for i in range(10):
            sample = np.load('samples_1000x256x256x3_'+str(i)+'.npz')
            if i == 0:
                images = sample['arr_0']
                lables = sample['arr_1']
            else:
                images = np.concatenate((images, sample['arr_0']), axis=0)
                lables = np.concatenate((lables, sample['arr_1']), axis=0)
            print(images.shape)
            print(lables.shape)
            
        # sample = np.load('samples_1000x256x256x3_'+str(8)+'.npz')
        # images = np.concatenate((images, sample['arr_0'].repeat(2, axis=0)), axis=0)
        # lables = np.concatenate((lables, sample['arr_1'].repeat(2, axis=0)), axis=0)
        # print(images.shape)
        # print(lables.shape)
        
        save_path = 'samples_10000x256x256x3.npz'
        np.savez(save_path, images, lables)
    elif args.scale == 10:
        for i in range(5):
            sample = np.load('10_samples_1000x256x256x3_'+str(i)+'.npz')
            if i == 0:
                images = sample['arr_0']
                lables = sample['arr_1']
            else:
                images = np.concatenate((images, sample['arr_0']), axis=0)
                lables = np.concatenate((lables, sample['arr_1']), axis=0)
            print(images.shape)
            print(lables.shape)
            
        sample = np.load('10_samples_5000x256x256x3'+'.npz')
        images = np.concatenate((images, sample['arr_0']), axis=0)
        lables = np.concatenate((lables, sample['arr_1']), axis=0)
        print(images.shape)
        print(lables.shape)
        
        save_path = '10_samples_10000x256x256x3.npz'
        np.savez(save_path, images, lables)
    
    # saved = np.load(save_path)
    # print(saved)
        
    print("Successfully executed")
    
if __name__=='__main__':
    main()