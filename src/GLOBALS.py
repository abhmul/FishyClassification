from math import log

# Loading data
FOLDERS = ['NoF', 'ALB', 'BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT']
IMGSIZE = (512, 512)

# Paths
PATH_TRAIN_CROPPED = ''
PATH_VAL_CROPPED = ''

# Input Data
INPUT_IMGSIZE = (64, 64)

# Sliding Window
SLIDING_WINDOW_RATIO = .25
WINDOW_MIN_RATIO = .0625
XSTRIDE = 1
YSTRIDE = 1
