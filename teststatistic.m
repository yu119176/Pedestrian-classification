[net, stats] = cnn_toy_data_net5_5000_48_64()

% plot(stats.val.top1err, 'o-') ; hold all ;

%Once training is finished, the model is saved back:
% Save the result for later use

% net.layers(end) = [] ;
% net.imageMean = imageMean ;
save('D:\�����ɱ�ҵ���\ylxGradution project 48_64\matconvnet-1.0-beta25\data\charscnn.mat', '-struct', 'net') ;


%use the model to classify an image
% Load the CNN learned before
% net = load('D:\Graduation Project\matconvnet-1.0-beta25\data\charscnn.mat') ;
net=load('D:\�����ɱ�ҵ���\ylxGradution project 48_64\matconvnet-1.0-beta25\data\toy\net-epoch-45.mat');  %ȡ�����ǵ�����ѵ��ģ�ͣ�����ֻѵ����45��  
net=net.net;
net2=load('D:\�����ɱ�ҵ���\ylxGradution project 48_64\matconvnet-1.0-beta25\data\toy\imdb.mat');
meani=net2.data_mean;

right=0;
wrong=0;
%�˵Ĳ���
Files = dir(strcat('D:\�����ɱ�ҵ���\ylxGradution project 48_64\cnn-net_5000_48_64\toy-dataset-net5-5000-48_64\test\person\','*.png'));
LengthFiles = length(Files);
for i=1:LengthFiles
    
    % obtain and preprocess an image
    im = imread(strcat('D:\�����ɱ�ҵ���\ylxGradution project 48_64\cnn-net_5000_48_64\toy-dataset-net5-5000-48_64\test\person\',Files(i).name)) ;
    im_ = single(im) ; % note: 255 range
    im_ = imresize(im_,[64 48]) ;
    im_ = im_ - meani ;
    net.layers{end}.type = 'softmax';
    
    % run the CNN
    
    res=vl_simplenn(net,im_);
    %As usual, res contains the results of the computation, including all intermediate
    %layers. The last one can be used to perform the classification:
    % show the classification result
    scores = squeeze(gather(res(end).x)) ;%������ÿ�����ķ���  res(end)�����һ��
    [bestScore, best] = max(scores) ;
    if best==1
       right=right+1;
    else
        wrong=wrong+1;
    end
end
%���˵Ĳ���
% Files = dir(strcat('D:\�����ɱ�ҵ���\ylxGradution project 48_64\cnn-net_5000_48_64\toy-dataset-net5-5000-48_64\test\background\','*.png'));
% LengthFiles = length(Files);
% for i=1:LengthFiles
%     
%     % obtain and preprocess an image
%     im = imread(strcat('D:\�����ɱ�ҵ���\ylxGradution project 48_64\cnn-net_5000_48_64\toy-dataset-net5-5000-48_64\test\background\',Files(i).name)) ;
%     im_ = single(im) ; % note: 255 range
%     im_ = imresize(im_,[64 48]) ;
%     im_ = im_ - meani ;
%     net.layers{end}.type = 'softmax';
%     
%     % run the CNN
%     
%     res=vl_simplenn(net,im_);
%     %As usual, res contains the results of the computation, including all intermediate
%     %layers. The last one can be used to perform the classification:
%     % show the classification result
%     scores = squeeze(gather(res(end).x)) ;%������ÿ�����ķ���  res(end)�����һ��
%     [bestScore, best] = max(scores) ;
%     if best==2
%       right=right+1;
%     else
%         wrong=wrong+1;
%     end
% end
wrongrate=wrong/LengthFiles;
rightrate=right/LengthFiles;

% figure(1) ; clf ; imagesc(im) ;

