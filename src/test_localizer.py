import numpy as np

from fish8 import load_test_data
from train_localizer import localize_net

best_model_fname = '../localizer_fold{}.h5'
picture_dir = '../input/test/test_stg1/'
buckets = 20
nfolds = 5

Xte, test_id = load_test_data(directory=picture_dir)
localizer = localize_net(buckets)

pred_left, pred_right = None, None
for fold in xrange(nfolds):
    localizer.load_weights(best_model_fname.format(fold+1))

    if not fold:
        pred_left, pred_right = localizer.predict(Xte, batch_size=64)
    else:
        pred_left_tmp, pred_right_tmp = localizer.predict(Xte, batch_size=64, verbose=1)
        pred_left += pred_left_tmp
        pred_right += pred_right_tmp

pred_left /= float(nfolds)
pred_right /= float(nfolds)


import matplotlib.pyplot as plt
import matplotlib.patches as patches


def display_with_rect(im, rects):
    colors = ('r', 'b', 'g')
    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    # print im.transpose(2, 0, 1).shape
    # print np.max(im)
    ax.imshow(im)

    for i, rect in enumerate(rects):
        edgecolor = colors[i]
        # Create a Rectangle patch
        rect = patches.Rectangle((rect[0], rect[1]), rect[2] - rect[0], rect[3] - rect[1], linewidth=1,
                                 edgecolor=edgecolor, facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()

for i, img in enumerate(test_id):
    print('Showing prediction for IMG {}'.format(img))
    ndimg = Xte[i]
    h, w = ndimg.shape

    y_left_coord = np.unravel_index(np.argmax(pred_left[i]), (buckets, buckets))
    y_right_coord = np.unravel_index(np.argmax(pred_right[i]), (buckets, buckets))
    y_left_coord = (y_left_coord[0][0] * (h / float(buckets)), y_left_coord[1][0] * (w / float(buckets)))
    y_right_coord = (y_right_coord[0][0] * (h / float(buckets)), y_right_coord[1][0] * (w / float(buckets)))
    print y_left_coord
    print y_right_coord
    display_with_rect(ndimg, [(y_left_coord[1], y_left_coord[0], y_right_coord[1], y_right_coord[0])])
