function [bestScore, best]= judge(patchI)
[net, stats] = cnn_toy_data_net5_5000_48_64();
save('D:\�����ɱ�ҵ���\ylxGradution project 48_64\matconvnet-1.0-beta25\data\charscnn.mat', '-struct', 'net') ;
net=load('D:\�����ɱ�ҵ���\ylxGradution project 48_64\matconvnet-1.0-beta25\data\toy\net-epoch-45.mat');  %ȡ�����ǵ�����ѵ��ģ�ͣ�����ֻѵ����45��  
net=net.net;
net2=load('D:\�����ɱ�ҵ���\ylxGradution project 48_64\matconvnet-1.0-beta25\data\toy\imdb.mat');
meani=net2.data_mean;
% im = imread(strcat('D:\�����ɱ�ҵ���\ylxGradution project 48_64\cnn-net_5000_48_64\toy-dataset-net5-5000-48_64\test\person\',Files(i).name)) ;
im_ = single(patchI) ; % note: 255 range
im_ = im_ - meani ;
 net.layers{end}.type = 'softmax';
 % run the CNN
res=vl_simplenn(net,im_);
scores = squeeze(gather(res(end).x)) ;%������ÿ�����ķ���  res(end)�����һ��
[bestScore, best] = max(scores) ;
  