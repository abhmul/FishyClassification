import numpy as np

from fish8 import load_test_data
from train_localizer import localize_net

best_model_fname = '../localizer_fold{}.h5'
picture_dir = '../input/test/test_stg1/'
buckets = 20
nfolds = 5

Xte, test_id = load_test_data(directory=picture_dir)
Xte = Xte[:32]
test_id = test_id[:32]
localizer = localize_net(buckets)

pred_left, pred_right = None, None
for fold in xrange(nfolds):
    localizer.load_weights(best_model_fname.format(fold))

    pred_left_tmp, pred_right_tmp = localizer.predict(Xte, batch_size=32, verbose=1)

    if not fold:
        pred_left, pred_right = pred_left_tmp, pred_right_tmp
    else:
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
    h, w = ndimg.shape[1:3]

    y_left_coord1 = np.unravel_index(np.argmax(pred_left[i]), (buckets, buckets))
    right_box = pred_right[i, y_left_coord1[0]:, y_left_coord1[0]:]
    y_right_coord1 = np.unravel_index(np.argmax(right_box), right_box.shape)
    y_right_coord1 = (y_right_coord1[0] + y_left_coord1[0], y_right_coord1[1] + y_left_coord1[1])
    print 'Bucket Box 1: ', (tuple(y_left_coord1), y_right_coord1)
    print 'Bucket Box 1 Prob', (pred_left[i, y_left_coord1[0], y_left_coord1[1]], pred_right[i, y_right_coord1[0], y_right_coord1[1]])

    y_right_coord2 = np.unravel_index(np.argmax(pred_right[i]), (buckets, buckets))
    left_box = pred_left[i, :y_right_coord2[0]-1, :y_right_coord2[0]-1]
    y_left_coord2 = np.unravel_index(np.argmax(left_box),
                                      left_box.shape)
    print 'Bucket Box 2: ', (tuple(y_left_coord2), tuple(y_right_coord2))
    print 'Bucket Box 2 Prob', (
    pred_left[i, y_left_coord2[0], y_left_coord2[1]], pred_right[i, y_right_coord2[0], y_right_coord2[1]])


    coords = [y_left_coord1, y_right_coord1, y_left_coord2, y_right_coord2]

    for i in xrange(0, len(coords), 2):
        y_left_coord = coords[i]
        y_right_coord = coords[i+1]

        coords[i] = (y_left_coord[0] * (h / float(buckets)), y_left_coord[1] * (w / float(buckets)))
        coords[i+1] = (y_right_coord[0] * (h / float(buckets)), y_right_coord[1] * (w / float(buckets)))
        print 'Bounding Box: ', tuple(coords[i:i+2])
    display_with_rect(ndimg, [(coords[0][1], coords[0][0], coords[1][1], coords[1][0]),
                              (coords[2][1], coords[2][0], coords[3][1], coords[3][0])])
