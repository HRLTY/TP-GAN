import argparse,cv2
#from mtcnn import MTCNN
import tensorflow as tf
import os
import csv
class MTCNN:

    def __init__(self, model_path, min_size=40, factor=0.709, thresholds=[0.6, 0.7, 0.7]):
        self.min_size = min_size
        self.factor = factor
        self.thresholds = thresholds

        graph = tf.Graph()
        with graph.as_default():
            with open(model_path, 'rb') as f:
                graph_def = tf.GraphDef.FromString(f.read())
                tf.import_graph_def(graph_def, name='')
        self.graph = graph
        config = tf.ConfigProto(
            allow_soft_placement=True,
            intra_op_parallelism_threads=4,
            inter_op_parallelism_threads=4)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=graph, config=config)

    def detect(self, img):
        feeds = {
            self.graph.get_operation_by_name('input').outputs[0]: img,
            self.graph.get_operation_by_name('min_size').outputs[0]: self.min_size,
            self.graph.get_operation_by_name('thresholds').outputs[0]: self.thresholds,
            self.graph.get_operation_by_name('factor').outputs[0]: self.factor
        }
        fetches = [self.graph.get_operation_by_name('prob').outputs[0],
                  self.graph.get_operation_by_name('landmarks').outputs[0],
                  self.graph.get_operation_by_name('box').outputs[0]]
        prob, landmarks, box = self.sess.run(fetches, feeds)
        return box, prob, landmarks
        
def test_image(PATH):
    mtcnn = MTCNN('./mtcnn.pb')
    save_PATH='/home/ubuntu3000/pt/TP-GAN/data/45_5pt'

    for imgpath in os.listdir(PATH):
        path = os.path.join(save_PATH,imgpath.replace('.png','.5pt'))
        csvfile = open(path, "w")
        img = cv2.imread(os.path.join(PATH,imgpath))
        data=[]
        bbox, scores, landmarks = mtcnn.detect(img)
        for box, pts in zip(bbox, landmarks):
            pts = pts.astype('int32')
            for i in range(5):
                row = str(pts[i+5]) + ' '+ str(pts[i])+'\n'
                csvfile.write(row)
        csvfile.close()

def create_img(PATH):
    mtcnn = MTCNN('./mtcnn.pb')
    csvfile = open('test_tem.5pt', "w")
    img_rec = cv2.imread(PATH)
    data=[]
    bbox, scores, landmarks = mtcnn.detect(img_rec)
    for box, pts in zip(bbox, landmarks):
        bbox=bbox.astype('int32')
        pts = pts.astype('int32')
        print([int(box[0]),int(box[2]),int(box[1]),int(box[3])])
        for i in range(5):
            row = str(pts[i+5]) + ' '+ str(pts[i])+'\n'
            #img=cv2.circle(img,(pts[i+5],pts[i]),1,(0,255,0),2)
    img_rec=img_rec[int(box[0])-20:int(box[2])+20,int(box[1])-50:int(box[3])+50]
    img_rec=cv2.resize(img_rec,(128,128))
    cv2.imwrite('test_tem.png',img_rec)
    bbox, scores, landmarks = mtcnn.detect(img_rec)
    for box, pts in zip(bbox, landmarks):
        pts = pts.astype('int32')
        for i in range(5):
            row = str(pts[i+5]) + ' '+ str(pts[i])+'\n'
            img_vis=cv2.circle(img_rec,(pts[i+5],pts[i]),1,(0,255,0),2)
            csvfile.write(row)
    cv2.imwrite('test_tem_vis.png',img_vis)
    csvfile.close()

def cut(PATH):
    img = cv2.imread(PATH)
    #img=img[]
    cv2.imwrite('test_tem.png',img)

if __name__ == '__main__':
    create_img('index.jpeg')
