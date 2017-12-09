import numpy as np

class Detector:
    
    def __init__(self):
        self.stack_boxes=[]
        self.boxes=[]
        self.n_cars=0
        self.dist=[0]
        
    def get_boxes(self):
        return self.boxes
        
    def distance(self, boxes):
        if len(boxes) == self.n_cars and len(self.boxes) == len(boxes):
            for b in range(len(boxes)):
                x=np.array(boxes[b])
                y=np.array(self.boxes[b])
                self.dist=[]
                if self.boxes is not []:
                    dist = abs(x - y) / y
                    dist = sum(dist) / len(dist)
                    self.dist.append(dist)
                else:
                    self.dist = [0]
        else:
            self.n_cars=len(boxes)
            self.stack_boxes=[]
            self.boxes=[]
            #self.dist=[0]
                
    def update(self, boxes, t=20, dist=0.25):
        if len(self.stack_boxes) > t:
            self.stack_boxes.pop(0)
        self.distance(boxes)
        ##print(np.mean(self.dist))
        if  np.mean(self.dist) < 0.25:
            self.stack_boxes.append(boxes)
            temp = np.mean(np.array(self.stack_boxes), axis=0)
            temp=temp.astype(int)
            
            self.boxes=[ ( tuple(b[0]), tuple(b[1]) ) for b in temp ]