import csv
import os

def create_csv(path_):
    PATH = '/home/ubuntu3000/pt/TP-GAN/data/45'
    data=[]
    data_test=[]
    path=os.path.join(path_,'train.csv')
    with open(path, "w") as file:
        csv_file = csv.writer(file)
        for filename in os.listdir(PATH):
            head = [filename]
            if not 'test' in filename:
                data.append(head)
        csv_file.writerows(data)

if __name__== '__main__':

    create_csv('/home/ubuntu3000/pt/TP-GAN/data/')
