

# 本代码单独用以测试
import os
import sys
import torch
import cv2
import numpy as np
import torch.nn as nn
import torchvision.utils as vutils
from utils import decode_segmap
from SegDataFolder import semData
from getSetting import get_yaml, get_criterion, get_optim, get_scheduler, get_net
from metric import AverageMeter, intersectionAndUnion
from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
from modeling.deeplab import *
import matplotlib
matplotlib.use('Agg')
from torch.utils.data import Dataset
import torchvision.transforms.transforms as _transform
import torch


class Solver(object):
    def __init__(self,configs):
        self.configs = configs
        self.cuda = torch.cuda.is_available()

        self.n_classes = self.configs['n_classes']
        self.ignore_index = self.configs['ignore_index']
        self.channels = self.configs['channels']
        self.net = get_net(self.configs)
        self.softmax = nn.Softmax(dim=1)
        self.criterion = get_criterion(self.configs)
        self.optimizer = get_optim(self.configs, self.net)
        self.scheduler = get_scheduler(self.configs, self.optimizer)
        self.batchsize = self.configs['batchsize']
        self.start_epoch = self.configs['start_epoch']
        self.end_epoch = self.configs['end_epoch']
        self.logIterval = self.configs['logIterval']
        self.valIterval = self.configs['valIterval']

        self.resume = self.configs['resume']['flag']
        if self.resume:
            self.resume_state(self.configs['resume']['state_path'])
        
        if self.cuda:
            self.net = self.net.cuda()
            if self.resume:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()

        self.trainSet = semData(train=True, channels=self.channels)
        self.valSet = semData(train=False, channels=self.channels)
        self.train_dataloader = torch.utils.data.DataLoader(self.trainSet, batch_size=self.batchsize, shuffle=True) 
        self.val_dataloader = torch.utils.data.DataLoader(self.valSet, 1, shuffle=False)

        self.best_miou = 0.00
        self.result_dir = self.configs['result_dir']
        os.makedirs(self.result_dir,exist_ok=True)
        self.writer = SummaryWriter(self.result_dir)
        with self.writer:
            if not self.resume:
                inp = torch.randn([1,self.channels,256,256]).cuda() if self.cuda else torch.randn([1,self.channels,256,256])
                self.writer.add_graph(self.net, inp)

    def save_state(self, epoch, path):
        state = {
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(state, path)

    def resume_state(self, path):
        checkpoint = torch.load(path)
        self.net.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.start_epoch = checkpoint['epoch']
        print('Resume from epoch{}...'.format(self.start_epoch))

    def test(self, singleImages=[], outName=''):
        assert len(singleImages) == self.channels
        from PIL import Image
        import transform as T
       # mean = np.array([0.013366839, -1.0583542e-05, -4.3700067e-05, 0.0026632368, 19.917131, 0.6687881, 0.6205375])  # 训练原图的平均值
       # std = np.array([0.20220748, 0.017818337, 0.013740968, 0.006313215, 7.6970053, 0.14615154, 0.15835664])  # 训练原图的方差
        '''predict1 mean = np.array([0.0903603, -2.3486707e-04, -1.4655667e-04, 0.0177745, 15.8372908, 0.6913939, 0.6046386,
                                     0.0943692, -6.2960805e-04, -5.6976004e-04, 0.0175582, 15.4808931, 0.7018849, 0.5963478,
                                    0.1264224, 7.3994452e-05, -5.4768787e-04, 0.0260056, 17.9446983, 0.6487185, 0.6619543,
                         0.0908136, 3.0130215e-04, -5.9409649e-04, 0.0255898, 23.1341686, 0.5398317, 0.7672510,
                         0.1100072, 6.7150162e-04, -2.1540790e-04, 0.0318259, 23.4367485, 0.5363507, 0.7700776])  # 测试原图的平均值

        std = np.array([0.0721300, 0.0088231, 0.0072266, 0.0168974, 5.2004261, 0.1077081, 0.1284363,
                        0.0655980, 0.0091909, 0.0065324, 0.0134588, 4.3799548, 0.0890403, 0.1088782,
                        0.0684424, 0.0078184, 0.0092252, 0.0119413, 4.0890222, 0.0827545, 0.0907580,
                        0.0655120, 0.0083495, 0.0095802, 0.0127735, 5.0186863, 0.1037111, 0.0884352,
                        0.0607109, 0.0080249, 0.0071815, 0.0101059, 5.1788187, 0.1045517, 0.0898428])  # 测试原图的方差
                        '''

        mean = np.array([ 0.0810,  8.9021e-05,  -1.5002e-04,  0.0181, 18.9629,   0.6262,   0.6757,
                          0.0968,  -1.9968e-04, -1.7126e-04,  0.0165, 15.8349,   0.6934,   0.6067,
                          0.1025,  3.0362e-04,  -4.7170e-04,  0.0252, 21.0515,   0.5838,  0.7282,
                          0.0908,  3.0130e-04,  -5.9410e-04,  0.0256,  23.1342,  0.5398,    0.7673,
                          0.1204,  0.0019,      -3.8942e-05,  0.0358,  23.6127,  0.5353,           0.7718     ])  # 测试原图的平均值

        std = np.array([0.0766, 0.0078, 0.0073, 0.0176, 5.9540, 0.1251, 0.1373,
                        0.0902, 0.0086, 0.0075, 0.0141, 4.3090, 0.0890, 0.1122,
                        0.0677, 0.0079, 0.0072, 0.0113, 4.6012, 0.0948, 0.0876,
                        0.0655, 0.0083, 0.0096, 0.0128, 5.0187, 0.1037, 0.0884,
                        0.0692, 0.0090, 0.0096, 0.0113, 5.1345, 0.1009, 0.0870])  # 测试原图的方差

        def get_idx(channels):
            assert channels in [2, 4, 35]
            if channels == 35:
                return list(range(35))
            elif channels == 4:
                return list(range(4))
            elif channels == 2:
                return list(range(6))[-2:]

        mean_, std_ = mean[get_idx(self.channels)], std[get_idx(self.channels)]
        _t = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean_, std=std_)
        ])
        L = []
        for item in singleImages:
            img = Image.open(item)
            img = np.expand_dims(np.array(img), axis=2)
            L.append(img)
        image = np.concatenate(L, axis=-1)
        # img = torch.from_numpy(image.transpose((2, 0, 1))).unsqueeze(0)
        # for t, m, s in zip(img, mean_, std_):
        #         t.sub_(m).div_(s)
        img, _ = _t(image, image[:, :, 0,])
        img = img.unsqueeze(0)
        img = img.cuda() if self.cuda else img

        self.net.eval()
        with torch.no_grad():
            outp = self.net(img)
            score = self.softmax(outp)
            pred = score.max(1)[1]

            saved_1 = pred.squeeze().cpu().numpy()
            saved_255 = 255 * saved_1

            cv2.imwrite('test_output/pred/1/{}.png'.format(outName), saved_1, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            cv2.imwrite('test_output/pred/255/{}.png'.format(outName), saved_255, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    def init_logfile(self):
        self.vallosslog = open(os.path.join(self.result_dir,'valloss.csv'),'w')
        self.vallosslog.writelines('epoch,loss\n')   

        self.valallacclog = open(os.path.join(self.result_dir,'valacc.csv'),'w')
        self.valallacclog.writelines('epoch,acc\n')

        self.trainlosslog = open(os.path.join(self.result_dir,'trainloss.csv'),'w')
        self.trainlosslog.writelines('epoch,loss\n')

        self.trainallacclog = open(os.path.join(self.result_dir,'trainacc.csv'),'w')
        self.trainallacclog.writelines('epoch,acc\n')

        self.precisionlog = open(os.path.join(self.result_dir,'presion.csv'),'w')
        self.precisionlog.writelines('epoch,precision\n')

        self.recalllog = open(os.path.join(self.result_dir,'recall.csv'),'w')
        self.recalllog.writelines('epoch,recall\n')

        self.f1log = open(os.path.join(self.result_dir, 'f1.csv'),'w')
        self.f1log.writelines('epoch,f1\n')

    def close_logfile(self):
        self.vallosslog.close()
        self.valallacclog.close()
        self.trainlosslog.close()
        self.trainallacclog.close()
        self.precisionlog.close()
        self.recalllog.close()
        self.f1log.close()  

    def trainer(self):
        try:
            for _ in range(self.start_epoch):
                self.scheduler.step()
            
            for epoch in range(self.start_epoch, self.end_epoch):
                self.train(epoch)
                self.scheduler.step()
                self.save_state(epoch, '{}/{}-ep{}.pth'.format(self.result_dir,self.configs['net'], epoch))
                if (epoch+1)%self.valIterval == 0:
                    self.val(epoch)
            
        except KeyboardInterrupt:
            print('Saving checkpoints from keyboardInterrupt...')
            self.save_state(epoch, '{}/{}-kb_resume.pth'.format(self.result_dir,self.configs['net']))
        
        finally:
            self.writer.close()


        # self.save_pred()
    def visualize(self, img, label, pred):
        label = label.clone().squeeze(0).cpu().numpy()
        label = decode_segmap(label).transpose((2,0,1))
        label = torch.from_numpy(label).unsqueeze(0)
        label = label.cuda() if self.cuda else label

        pred = pred.clone().squeeze(0).cpu().numpy()
        pred = decode_segmap(pred).transpose((2,0,1))
        pred = torch.from_numpy(pred).unsqueeze(0)
        pred = pred.cuda() if self.cuda else pred

        vis = torch.cat([self.denorm(img), label.float(), pred.float()], dim=0)
        vis_cat = vutils.make_grid(vis,nrow=3,padding=5,pad_value=0.8)
        return vis_cat
    

    def denorm(self, x):
        mean_ = torch.Tensor(mean).view(3,1,1)
        std_ = torch.Tensor(std).view(3,1,1)
        mean_ = mean_.cuda() if self.cuda else mean_
        std_ = std_.cuda() if self.cuda else std_
        out = x * std_ + mean_
        out = out / 255.
        return out.clamp_(0,1)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    config = get_yaml('ConfigFiles/config-attnUnet.yaml')
    print(config)
    path = 'Data/predict/'
    solver = Solver(config)
    for i in range(1,41):
        stri=str(i)+'.tif'
        img_path = []
        for j in range(5,10):
            strj = str(j)
            img_path.append(path+strj+'_1_C11/'+stri)
            img_path.append(path + strj + '_2_C12real/' + stri)
            img_path.append(path + strj + '_3_C12imag/' + stri)
            img_path.append(path + strj + '_4_C22/' + stri)
            img_path.append(path + strj + '_5_alpha/' + stri)
            img_path.append(path + strj + '_6_anisotropy/' + stri)
            img_path.append(path + strj + '_7_entropy/' + stri)
        solver.test(img_path,outName=stri)
'''
    solver.test(['Data/predict/5_1_C11/',
                 'Data/predict/5_2_C12real/',
                 'Data/predict/5_3_C12imag/',
                 'Data/predict/5_4_C22/',
                 'Data/predict/5_5_alpha/',
                 'Data/predict/5_6_anisotropy/',
                 'Data/predict/5_7_entropy/',

                 'Data/predict/6_1_C11/',
                 'Data/predict/6_2_C12real/',
                 'Data/predict/6_3_C12imag/',
                 'Data/predict/6_4_C22/',
                 'Data/predict/6_5_alpha/',
                 'Data/predict/6_6_anisotropy/',
                 'Data/predict/6_7_entropy/',

                 'Data/predict/7_1_C11/',
                 'Data/predict/7_2_C12real/',
                 'Data/predict/7_3_C12imag/',
                 'Data/predict/7_4_C22/',
                 'Data/predict/7_5_alpha/',
                 'Data/predict/7_6_anisotropy/',
                 'Data/predict/7_7_entropy/',

                 'Data/predict/8_1_C11/',
                 'Data/predict/8_2_C12real/',
                 'Data/predict/8_3_C12imag/',
                 'Data/predict/8_4_C22/',
                 'Data/predict/8_5_alpha/',
                 'Data/predict/8_6_anisotropy/',
                 'Data/predict/8_7_entropy/',

                 'Data/predict/9_1_C11/',
                 'Data/predict/9_2_C12real/',
                 'Data/predict/9_3_C12imag/',
                 'Data/predict/9_4_C22/',
                 'Data/predict/9_5_alpha/',
                 'Data/predict/9_6_anisotropy/',
                 'Data/predict/9_7_entropy/'],outName='')
'''
    # solver.trainer()
