# Loading data
FOLDERS = ['NoF', 'ALB', 'BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT']
IMGSIZE = (512, 512)

# Paths
PATH_TRAIN_CROPPED = '../input/cropped_train/'
PATH_VAL_CROPPED = '../input/cropped_validation/'
PATH_TRAIN = ''

# Input Data
INPUT_IMGSIZE = (64, 64)
MODE = 'rgb'
if MODE == 'grayscale':
    CHANNELS = 1
elif MODE == 'rgb':
    CHANNELS = 3
else:
    NameError("MODE must be \'rgb\' or \'grayscale\'")

# Model
MODEL = 'vgg'

# Plotting
PLOT = True

# Sliding Window
SLIDING_WINDOW_RATIO = .25
WINDOW_MIN_RATIO = .0625
XSTRIDE = 1
YSTRIDE = 1
