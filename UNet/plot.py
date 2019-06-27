from init import *

import numpy as np
import matplotlib.pyplot as plt

# tret del document "Calibration_Information_for_220_Channel_Data_Band_Set"
rgb_indianpines = [[255,255,255], [255,254,137], [3,28,241], [255,89,1], [5,255,133], [255,2,251], [89,1,255], [3,171,255], [12,255,7], [172,175,84], [160,78,158], [101,173,255], [60,91,112], [104,192,63], [139,69,46], [119,255,172], [254,255,3]]
# si CASI 396,9-1039 nm i r,g,b es 700,520,460 nm
rgb_maspalomas = [[0,0,0], [72,255,72], [0,139,0], [255,255,0], [139,71,38], [0,255,0], [145,44,238], [0,0,255], [255,255,255], [160,32,240], [0,255,255], [205,0,0], [205,205,0]]

# CHANGED a lot xd
def plot_img_and_mask(img, target, mask, loss, dir, dir2="", tb_val_writer=None): # CHANGED suposant format img chw i mask i target amb valors de 0 a NET_CLASSES
    #print("max img", np.max(img), "unique target", np.unique(target), "unique mask", np.unique(mask))

    fig = plt.figure()
    fig.suptitle(str(loss))
    a = fig.add_subplot(1, 3, 1)
    img = np.transpose(img, axes=[1,2,0])
    a.set_title('Input image')

    if INPUT_BANDS > 200:
        img[:,:,0] = img[:,:,30]
        img[:,:,1] = img[:,:,14]
        img[:,:,2] = img[:,:,8]
    else:
        img[:,:,0] = img[:,:,32]
        img[:,:,1] = img[:,:,13]
        img[:,:,2] = img[:,:,7]

    img = img[:,:,0:3]
    img = (img/np.max(img)*255).astype(int)
    plt.imshow(img)

    #print(target.shape, mask.shape)
    rgb_target = np.zeros((3, target.shape[0], target.shape[1]))

    for (j,k), cl in np.ndenumerate(target):
        if cl == -1:
            rgb_target[:,j,k] = [255,137,20]
        else:
            rgb_target[:,j,k] = rgb_indianpines[cl] if INPUT_BANDS > 200 else rgb_maspalomas[cl]


    b = fig.add_subplot(1, 3, 2)
    b.set_title('Target mask')
    if (target == 0).sum() == target.size:
        rgb_target = np.transpose(rgb_target.astype(int), axes=[1,2,0])
    else:
        rgb_target = np.transpose((rgb_target/np.max(rgb_target)*255).astype(int), axes=[1,2,0])
    plt.imshow(rgb_target)

    rgb_mask = np.zeros((3, mask.shape[0], mask.shape[1]))

    for (j,k), cl in np.ndenumerate(mask):
      rgb_mask[:,j,k] = rgb_indianpines[cl] if INPUT_BANDS > 200 else rgb_maspalomas[cl]

    c = fig.add_subplot(1, 3, 3)
    c.set_title('Output mask')
    if (mask == 0).sum() == mask.size:
        rgb_mask = np.transpose(rgb_mask.astype(int), axes=[1,2,0])
    else:
        rgb_mask = np.transpose((rgb_mask/np.max(rgb_mask)*255).astype(int), axes=[1,2,0])
    plt.imshow(rgb_mask)

    #plt.colorbar()
    # plt.show()

    if tb_val_writer == None:
        model_name = dir.split('/')[-1].split('.')[0]
        dir = '/'.join(dir.split('/')[:-1]) + '/'
        plt.savefig(dir + 'predict' + dir2 + model_name + '.png', bbox_inches='tight')
        np.save(dir + "pred.npy", mask)
    else:
        tb_val_writer.add_figure('example val patch', fig)
    plt.close()
