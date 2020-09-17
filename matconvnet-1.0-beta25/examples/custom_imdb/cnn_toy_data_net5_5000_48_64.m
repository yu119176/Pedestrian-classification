function [net, stats] = cnn_toy_data_net5_5000_48_64(varargin)
%对于图片是宽乘以高即列乘以行，而二维矩阵中是行*列（即高乘以宽）
% CNN_TOY_DATA
% Minimal demonstration of MatConNet training of a CNN on toy data.
%
% It also serves as a short tutorial on creating and using a custom imdb
% (image database).
%
% The task is to distinguish between images of triangles, squares and
% circles.

% Copyright (C) 2017 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

run([fileparts(mfilename('fullpath')) '/../../matlab/vl_setupnn.m']) ;

% Parameter defaults. You can add any custom parameters here (e.g.
% opts.alpha = 1), and change them when calling: cnn_toy_data('alpha', 2).

opts.train.batchSize = 100 ;
opts.train.numEpochs = 45 ;
opts.train.continue = true ;
opts.train.gpus = [] ;
opts.train.learningRate = [0.05*ones(1,30) 0.005*ones(1,10) 0.0005*ones(1,5)] ; 
opts.train.weightDecay = 0.0001 ;
opts.train.expDir = [vl_rootnn '/data/toy'] ;
opts.dataDir = [vl_rootnn '/data/toy-dataset'] ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.imdbPath = [opts.train.expDir '/imdb.mat'] ;
opts = vl_argparse(opts, varargin) ;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

% Generate images if they don't exist (this would be skipped for real data)
if ~exist(opts.dataDir, 'dir')
    mkdir(opts.dataDir) ;
    cnn_toy_data_generator(opts.dataDir) ;
end

% Create image database (imdb struct). It can be cached to a file for speed
if exist(opts.imdbPath, 'file')
    disp('Reloading image database...')
    imdb = load(opts.imdbPath) ;
else
    disp('Creating image database...')
    imdb = getImdb(opts.dataDir) ;
    mkdir(fileparts(opts.imdbPath)) ;
    save(opts.imdbPath, '-struct', 'imdb') ;
end

% Create network (see HELP VL_SIMPLENN)

lr = [.1 2] ; %学习率

% Define network CIFAR10-quick
net.layers = {} ;

%定义卷基层，其中randn生成的矩阵为W矩阵，zeros向量为bias向量
%定义各层参数，type是网络的层属性，stride为步长，pad为填充
%  method中max为最大池化
%第一层卷积核是5*5，输出32个特征特征,1是代表上一层的特征，32表示要输出的特征个数,zeros里代表了偏置
% Block 1
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{0.01*randn(5,5,1,35, 'single'), zeros(1, 35, 'single')}}, ...
                           'learningRate', lr, ...
                           'stride', 1, ... %stride，滤波器平移的步长为1
                           'pad', 2) ; %pad，在图像边缘加的大小，48*64，输出特征面的大小=（64+2+2-5）/stride+1，48*64
net.layers{end+1} = struct('type', 'pool', ... 
                           'method', 'max', ...
                           'pool', [3 3], ... %池化层的滤波器大小=3*3
                           'stride', 2, ...
                           'pad', [0 1 0 1]) ; %上，下，左，右，输出特征面的大小=（64+1-3）/stride+1，24*32
net.layers{end+1} = struct('type', 'relu') ;%激活层（减少冗余）

% Block 2
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{0.05*randn(5,5,35,35, 'single'), zeros(1,35,'single')}}, ...
                           'learningRate', lr, ...
                           'stride', 1, ...
                           'pad', 2) ; %上，下，左，右，输出特征面的大小=（32+4-5）/stride+1，24*32
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'avg', ...
                           'pool', [3 3], ...
                           'stride', 2, ...%上，下，左，右，输出特征面的大小=（32+1-3）/stride+1，12*16
                           'pad', [0 1 0 1]) ; % Emulate caffe   12*16

% Block 3
net.layers{end+1} = struct('type', 'conv', ...
                            'weights', {{0.05*randn(5,5,35,35, 'single'), zeros(1,35,'single')}}, ...
                           'learningRate', lr, ...
                           'stride', 1, ...
                           'pad', 2) ; %上，下，左，右，输出特征面的大小=（16+4-5）/stride+1，12*16
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'avg', ...
                           'pool', [3 3], ...
                           'stride', 2, ...%上，下，左，右，输出特征面的大小=（16+1-3）/stride+1，6*8
                           'pad', [0 1 0 1]) ; % Emulate caffe 6*8

  % Block 4
net.layers{end+1} = struct('type', 'conv', ...
                            'weights', {{0.05*randn(5,5,35,70, 'single'), zeros(1,70,'single')}}, ...
                           'learningRate', lr, ...
                           'stride', 1, ...
                           'pad', 2) ; %上，下，左，右，输出特征面的大小=（8+4-5）/stride+1，6*8
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'avg', ...
                           'pool', [3 3], ...
                           'stride', 2, ...%上，下，左，右，输出特征面的大小=（8+1-3）/stride+1，3*4（输出图片的大小）
                           'pad', [0 1 0 1]) ; % Emulate caffe 3*4                     
                       
% Block 5
net.layers{end+1} = struct('type', 'conv', ...
                            'weights', {{0.05*randn(4,3,70,80, 'single'), zeros(1,80,'single')}}, ...
                           'learningRate', lr, ...
                           'stride', 1, ...
                           'pad', 0) ;%上，下，左，右，输出特征面的大小=（4-4）/stride+1，1*1
net.layers{end+1} = struct('type', 'relu') ;

% Block 6
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{0.05*randn(1,1,80,2, 'single'), zeros(1,2,'single')}}, ...
                           'learningRate', .1*lr, ...
                           'stride', 1, ...
                           'pad', 0) ;  %最后算到和图像一样大小
                       
                       

% Loss layer
net.layers{end+1} = struct('type', 'softmaxloss') ;

% Meta parameters
net.meta.inputSize = [64 48 1] ;
net.meta.trainOpts.learningRate = [0.05*ones(1,30) 0.005*ones(1,10) 0.0005*ones(1,5)] ;
net.meta.trainOpts.weightDecay = 0.0001 ;
net.meta.trainOpts.batchSize = 100 ;
net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate) ;

% Fill in any values we didn't specify explicitly
net = vl_simplenn_tidy(net) ;


% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

use_gpu = ~isempty(opts.train.gpus) ;

% Start training
[net, stats] = cnn_train(net, imdb, @(imdb, batch) getBatch(imdb, batch, use_gpu), ...
    'train', find(imdb.set == 1), 'val', find(imdb.set == 2), opts.train) ;

% Visualize the learned filters
% figure(3) ; vl_tshow(net.layers{1}.weights{1}) ; title('Conv1 filters') ;
% figure(4) ; vl_tshow(net.layers{3}.weights{1}) ; title('Conv2 filters') ;
% figure(5) ; vl_tshow(net.layers{5}.weights{1}) ; title('Conv3 filters') ;


% --------------------------------------------------------------------
function [images, labels] = getBatch(imdb, batch, use_gpu)
% --------------------------------------------------------------------
% This is where we return a given set of images (and their labels) from
% our imdb structure.
% If the dataset was too large to fit in memory, getBatch could load images
% from disk instead (with indexes given in 'batch').

images = imdb.images(:,:,:,batch) ;
labels = imdb.labels(batch) ;

if use_gpu
    images = gpuArray(images) ;
end

% --------------------------------------------------------------------
function imdb = getImdb(dataDir)
% --------------------------------------------------------------------
% Initialize the imdb structure (image database).
% Note the fields are arbitrary: only your getBatch needs to understand it.
% The field imdb.set is used to distinguish between the training and
% validation sets, and is only used in the above call to cnn_train.

% The sets, and number of samples per label in each set
% sets = {'train', 'val'} ;
% numSamples = [1500, 150] ;

% Preallocate memory
totalSamples = 11200 ;  
images = zeros(64, 48, 1, totalSamples, 'single') ;
labels = zeros(totalSamples, 1) ;
set = ones(totalSamples, 1) ;

% Read all samples
% Read all samples贴标签
for i=1:totalSamples
    
    if i<=5000  %正样本，训练样本，5000个
        if i==1
           j=1;
        end
      
        Files = dir(strcat('D:\余丽仙毕业设计\ylxGradution project 48_64\cnn-net_5000_48_64\toy-dataset-net5-5000-48_64\train\1\','*.png'));
        img=imread(strcat('D:\余丽仙毕业设计\ylxGradution project 48_64\cnn-net_5000_48_64\toy-dataset-net5-5000-48_64\train\1\',Files(j).name));
        images(:,:,:,i) = img ;
        labels(i)=1;
        set(i)=1;
        j=j+1;
    elseif i<=10000    %负样本，训练样本，5000个
        if i==5001
            j=1;
        end
        
        Files = dir(strcat('D:\余丽仙毕业设计\ylxGradution project 48_64\cnn-net_5000_48_64\toy-dataset-net5-5000-48_64\train\2\','*.png'));
        img=imread(strcat('D:\余丽仙毕业设计\ylxGradution project 48_64\cnn-net_5000_48_64\toy-dataset-net5-5000-48_64\train\2\',Files(j).name));
        images(:,:,:,i) = img ;
        labels(i)=2;
        set(i)=1;
        j=j+1;
    elseif i<=10600  %正样本，验证样本，600个
        if i==10001
            j=1;
        end
       
        Files = dir(strcat('D:\余丽仙毕业设计\ylxGradution project 48_64\cnn-net_5000_48_64\toy-dataset-net5-5000-48_64\val\1\','*.png'));
        img=imread(strcat('D:\余丽仙毕业设计\ylxGradution project 48_64\cnn-net_5000_48_64\toy-dataset-net5-5000-48_64\val\1\',Files(j).name));
        images(:,:,:,i) = img ;
        labels(i)=1;
        set(i)=2;
        j=j+1;
    elseif i<=11200  %负样本，验证样本，600个
        if i==10601
            j=1;
        end
        
        Files = dir(strcat('D:\余丽仙毕业设计\ylxGradution project 48_64\cnn-net_5000_48_64\toy-dataset-net5-5000-48_64\val\2\','*.png'));
        img=imread(strcat('D:\余丽仙毕业设计\ylxGradution project 48_64\cnn-net_5000_48_64\toy-dataset-net5-5000-48_64\val\2\',Files(j).name));
        images(:,:,:,i) = img ;
        labels(i)=2;
        set(i)=2;
        j=j+1;
    end
   
end

dataMean = mean(images, 4);
imdb.data_mean = dataMean;

% Show some random example images
figure(2) ;
montage(images(:,:,:,randperm(totalSamples, 100))) ;
title('Example images') ;

% Remove mean over whole dataset
images = bsxfun(@minus, images, mean(images, 4)) ;
save('D:\余丽仙毕业设计\ylxGradution project 48_64\matconvnet-1.0-beta25\data\averageImage.mat', 'images') ;

% Store results in thezeros imdb struct
imdb.images = images ;
imdb.labels = labels ;
imdb.set = set ;
