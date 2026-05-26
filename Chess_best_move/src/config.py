import os

# Base paths
BASE_PATH = "/Users/manyasinha/Documents/Github Projects/Chess_best_move"
DATASET_PATH = os.path.join(BASE_PATH, "dataset")
MODEL_PATH = os.path.join(BASE_PATH, "models", "best.pt")

# Model settings
CONFIDENCE_THRESHOLD = 0.4

# Board settings
BOARD_SIZE = 800

# Mode settings (options: "random", "fixed")
MODE = "random"
FIXED_IMAGE_NAME = "3aafc2d38807dddd1b43a54cb70f500d_jpg.rf.62764e7af18d8f34c7df2b39b699410b.jpg"
#Some Images to choose from:
#3baf85c957b9d28a16c0b65cb2ef0d29_jpg.rf.b0d51768b3d43d7d467e54ab66acafe5.jpg
#3aafc2d38807dddd1b43a54cb70f500d_jpg.rf.62764e7af18d8f34c7df2b39b699410b.jpg
#9453c2097cb4ccc676e273939894b3da_jpg.rf.03d16c5afdd17458100fda4dbd9c28f8.jpg
#495019998442ddf85b59e387d4916cd3_jpg.rf.2e7b9222427c945a7d371024ae7049c2.jpg
#15cc23c777b00d0e123f9df468f2852b_jpg.rf.009a7b65eb55dfb43d7e792790e762e3.jpg
#IMG_0293_JPG_jpg.rf.780141dd9bb16bedd19795e8fa08d0cd.jpg
#c20ca9283ea51ac7707905894a7da703_jpg.rf.2533bd443aba28b547013922ad836118.jpg 
#IMG_0311_JPG_jpg.rf.1fb7e556ab8f284e30ede8078ae64acf.jpg
#IMG_0310_JPG_jpg.rf.af47432c9c1e7eefafa7d15bdaec79aa.jpg
#7e97f49e613a59a70b833e4c0b2c1c04_jpg.rf.cc3bc3739a5f5564049e7dfeeda976a7.jpg 

# Files and ranks mapping
FILES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
RANKS = ['1', '2', '3', '4', '5', '6', '7', '8']

# Piece mapping for FEN
PIECE_MAP = {
    "white_pawn": "P",
    "white_rook": "R",
    "white_knight": "N",
    "white_bishop": "B",
    "white_queen": "Q",
    "white_king": "K",
    "black_pawn": "p",
    "black_rook": "r",
    "black_knight": "n",
    "black_bishop": "b",
    "black_queen": "q",
    "black_king": "k",
}