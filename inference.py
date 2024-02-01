import os
import cv2
import AnimeGANv3_src
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str)
args = parser.parse_args()

if not os.path.exists(args.source):
    raise ValueError(f"Path: {args.source} not exists")
img = cv2.imread(args.source)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
start_time = time.time()
output = AnimeGANv3_src.Convert(img, 'T', False)
cv2.imwrite("./result.jpeg", output[:,:,::-1])
end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds") 
