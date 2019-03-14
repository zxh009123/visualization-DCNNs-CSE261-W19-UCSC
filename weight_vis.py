import torch, time, copy, sys, os
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
import gconv_test as custom_model
import torch, time, sys
import cv2
import numpy as np
import hiddenlayer as hl


class feedFoward:
    def __init__(self, path):
        # load pre-trained generator here
        self._model = custom_model.res18GConv(200)
        self._model.load_state_dict(torch.load(path)['model_state_dict'])
        self._model.eval()
        x = hl.build_graph(self._model, torch.zeros([1, 3, 64, 64]))
        print(type(x))
        x.save('./t.png')
        # filter_show = np.ones((1,17))
        # for i in range(50):
        #     split = np.ones((4,1))
        #     for j in self._model.state_dict()['conv2dT3.weight'][i*4:i*4+4]:
        #         j = j.numpy()
        #         j = j.reshape(4,4)
        #         j[j<0] = 0
        #         split = np.hstack((split, j))
        #     filter_show = np.vstack((filter_show, split))
        # filter_show = cv2.resize(filter_show, (340, 4000), interpolation=cv2.INTER_NEAREST)
        # cv2.imshow('res', filter_show)
        # cv2.waitKey()

            

            
        


    def Foward(self, pic):
        # transform face by normalizing 
        pic = torch.unsqueeze(self._transform(pic), 0).float()
        result, up1, up2 = self._model.forward(pic)
        # calculate result

        # maskA = mask.detach().numpy()[0]
        # maskA = np.transpose(maskA, (1,2,0))

        return result, up1, up2

if __name__ == '__main__':
    model_path = './models/test-refine/model_53_epoch.pth'
    network = feedFoward(model_path)