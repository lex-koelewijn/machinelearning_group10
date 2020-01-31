import pandas as pd
import numpy as np

try: 
	os.remove('combined.txt')
except:
	print("No file to remove")

filelist = ['chroma_prediction1_a=1.txt', 'chroma_prediction2_a=1.txt', 'chroma_prediction3_a=1.txt','chroma_prediction4_a=1.txt']
frame = pd.concat([pd.read_csv(item) for item in filelist], axis=1)

frame = frame.fillna(0)
frame = frame.astype(int)
# tfile = open('combined.txt', 'a')
frame.to_csv('combined.txt', sep='\t', encoding='utf-8', header=None, index=False)
# tfile.write(frame.to_string(index=False))