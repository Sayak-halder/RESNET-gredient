import torch

class Config:
    # Model Settings
    NUM_CLASSES = 3 # Rock, Paper, Scissors
    
    # Training Settings
    BATCH_SIZE = 32
    EPOCHS = 25
    LEARNING_RATE = 0.01
    MOMENTUM = 0.9
    
    # Data Settings
    # UPDATE THIS PATH to your Kaggle dataset path
    DATA_DIR = '/kaggle/input/rock-paper-scissors-dataset/rock-paper-scissors/rps' 
    
    # Device & Paths
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR = "./outputs/charts"