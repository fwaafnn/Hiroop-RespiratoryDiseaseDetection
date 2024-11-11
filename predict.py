import numpy as np
import librosa
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=UserWarning, module='keras')
import tensorflow as tf
tf.get_logger().setLevel('ERROR')


