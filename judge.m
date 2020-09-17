function [bestScore, best]= judge(patchI)
[net, stats] = cnn_toy_data_net5_5000_48_64();
save('D:\余丽仙毕业设计\ylxGradution project 48_64\matconvnet-1.0-beta25\data\charscnn.mat', '-struct', 'net') ;
net=load('D:\余丽仙毕业设计\ylxGradution project 48_64\matconvnet-1.0-beta25\data\toy\net-epoch-45.mat');  %取出我们的最终训练模型，这里只训练了45次  
net=net.net;
net2=load('D:\余丽仙毕业设计\ylxGradution project 48_64\matconvnet-1.0-beta25\data\toy\imdb.mat');
meani=net2.data_mean;
% im = imread(strcat('D:\余丽仙毕业设计\ylxGradution project 48_64\cnn-net_5000_48_64\toy-dataset-net5-5000-48_64\test\person\',Files(i).name)) ;
im_ = single(patchI) ; % note: 255 range
im_ = im_ - meani ;
 net.layers{end}.type = 'softmax';
 % run the CNN
res=vl_simplenn(net,im_);
scores = squeeze(gather(res(end).x)) ;%分属于每个类别的分数  res(end)是最后一层
[bestScore, best] = max(scores) ;
  