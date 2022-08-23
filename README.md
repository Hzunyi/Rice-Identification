# Rice-RMAU-Net
model:RMAU-Net
1.Title： Rice Monitoring Based on Multi-Temporal Sentinel-1 SAR Images and an Attention U-Net Model
2.Key words: Rice monitoring, Polarimetric synthetic aperture radar, Deep convolutional neural network, Transfer mechanism
3.Author: Xiaoshuang Ma, Member, IEEE, Zunyi Huang, Shengyuan Zhu, Wei Fang, Yinglei Wu
4.The slove.py is the master model,it use to change the different model like as Deeplab, U-Net, FCN et.al. And you can use the code to train all model.
5.The SegDataFolder.py is to processing data. Which include channel data, mean and std. If you want to train new data, you should pay attention on the channel data and the value of mean and std.
6. The getSetting.py is to change the hyper-parameter such as optimizer, scheduler, criterion, backbone et.al.

