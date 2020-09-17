clear;clc;
imgI=imread('C:\Users\13922\Desktop\毕业设计\未处理的图片\2.png');
[imgsx,imgsy]=size(imgI);%图片本身的大小
patchsx=120;patchsy=40;%从图片中取出来的小图的大小，我们自己所取的图片不能大于这个
strd=7;%每次滑动的步长
j=1;
 for  i=1:5
    for x=1:strd:imgsx-patchsx
        for y=1:strd:imgsy-patchsy
            I1=imgI(x:x+patchsx-1,y:y+patchsy);
            patchI=imresize(I1,[64 48]);%patch为CNN中图像大小
            %减均值，变为single变量
             [bestScore, best]= judge(patchI);
            if best==1 && bestScore==1    %CNN判定为行人
                imgI(x:x+patchsx-1,y)=255;
                imgI(x:x+patchsx-1,y+patchsy-1)=255; 
                imgI(x,y:y+patchsy-1)=255;
                imgI(x+patchsx-1,y:y+patchsy-1)=255;%画出四条边，也就是我们上面选定的那个小框,行人用白色框标记出来
                imshow(imgI);
                imwrite(imgI,strcat('D:\余丽仙毕业设计\ylxGradution project 48_64\imgjudgment\',int2str(j),'.png'));
                j=j+1;
               %只标记一个人
            end
        end
   end
    patchsx=ceil(0.9*patchsx);%每次乘以0.9然后取整
    patchsy=ceil(0.9*patchsy);
end

    