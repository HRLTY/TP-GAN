import csv
import os
import numpy as np
PATH = '/home/ubuntu3000/pt/TP-GAN/data/45_5pt'
trans_points = np.empty([5,2],dtype=np.int32)
for filename in os.listdir(PATH):
    with open(os.path.join(PATH,filename), 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for ind,row in enumerate(reader):
            trans_points[ind,:] = row 
        print(trans_points)
        break
