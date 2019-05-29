from init import *

import numpy as np
import matplotlib.pyplot as plt

rgb_indianpines = [[255,255,255], [255,254,137], [3,28,241], [255,89,1], [5,255,133], [255,2,251], [89,1,255], [3,171,255], [12,255,7], [172,175,84], [160,78,158], [101,173,255], [60,91,112], [104,192,63], [139,69,46], [119,255,172], [254,255,3]]

# CHANGED a lot xd
def plot_img_and_mask(img, target, mask, loss, dir, tb_val_writer=None): # CHANGED suposant format chw i mask i target amb valors de 0 a NET_CLASSES
    #print("max img", np.max(img), "unique target", np.unique(target), "unique mask", np.unique(mask))

    fig = plt.figure()
    fig.suptitle(str(loss))
    a = fig.add_subplot(1, 3, 1)
    img = np.transpose(img, axes=[1,2,0])
    a.set_title('Input image')
    # tret del document "Calibration_Information_for_220_Channel_Data_Band_Set"
    img[:,:,0] = img[:,:,30]
    img[:,:,1] = img[:,:,14]
    img[:,:,2] = img[:,:,8]
    img = img[:,:,0:3]
    img = (img/np.max(img)*255).astype(int)
    plt.imshow(img)

    #print(target.shape, mask.shape)
    rgb_target = np.zeros((3, target.shape[1], target.shape[2]))

    for (ch,j,k), cl in np.ndenumerate(target):
      rgb_target[:,j,k] = rgb_indianpines[cl]

    b = fig.add_subplot(1, 3, 2)
    b.set_title('Target mask')
    rgb_target = np.transpose((rgb_target/np.max(rgb_target)*255).astype(int), axes=[1,2,0])
    plt.imshow(rgb_target)

    rgb_mask = np.zeros((3, mask.shape[0], mask.shape[1]))

    for (j,k), cl in np.ndenumerate(mask):
      rgb_mask[:,j,k] = rgb_indianpines[cl]

    c = fig.add_subplot(1, 3, 3)
    c.set_title('Output mask')
    rgb_mask = np.transpose((rgb_mask/np.max(rgb_mask)*255).astype(int), axes=[1,2,0])
    plt.imshow(rgb_mask)

    #plt.colorbar()
    plt.show()

    if tb_val_writer == None:
        dir = '/'.join(dir.split('/')[:-1]) + '/'
        plt.savefig(dir + 'predict' + dir.split('/')[-2] + '.png', bbox_inches='tight')
    else:
        tb_val_writer.add_figure('example val patch', fig)
    plt.close()
