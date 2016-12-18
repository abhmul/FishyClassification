# Loading data
# FOLDERS = ['NoF', 'ALB', 'BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT']
FOLDERS = ['NoF']
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

# Augmentation
ROTATION = .4
ZOOM = .3
SHEAR = .3
HORIZONTAL = True
VERTICAL = True
FILL_MODE = 'nearest'

# Training
NUM_EPOCHS = 100
SAMPLES = 256  # 1224

# Plotting
PLOT = True

# Sliding Window
SLIDING_WINDOW_RATIO = .25
WINDOW_MIN_RATIO = .0625
XSTRIDE = 1
YSTRIDE = 1
