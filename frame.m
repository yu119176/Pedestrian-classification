clear;clc;
imgI=imread('C:\Users\13922\Desktop\��ҵ���\δ�����ͼƬ\2.png');
[imgsx,imgsy]=size(imgI);%ͼƬ����Ĵ�С
patchsx=120;patchsy=40;%��ͼƬ��ȡ������Сͼ�Ĵ�С�������Լ���ȡ��ͼƬ���ܴ������
strd=7;%ÿ�λ����Ĳ���
j=1;
 for  i=1:5
    for x=1:strd:imgsx-patchsx
        for y=1:strd:imgsy-patchsy
            I1=imgI(x:x+patchsx-1,y:y+patchsy);
            patchI=imresize(I1,[64 48]);%patchΪCNN��ͼ���С
            %����ֵ����Ϊsingle����
             [bestScore, best]= judge(patchI);
            if best==1 && bestScore==1    %CNN�ж�Ϊ����
                imgI(x:x+patchsx-1,y)=255;
                imgI(x:x+patchsx-1,y+patchsy-1)=255; 
                imgI(x,y:y+patchsy-1)=255;
                imgI(x+patchsx-1,y:y+patchsy-1)=255;%���������ߣ�Ҳ������������ѡ�����Ǹ�С��,�����ð�ɫ���ǳ���
                imshow(imgI);
                imwrite(imgI,strcat('D:\�����ɱ�ҵ���\ylxGradution project 48_64\imgjudgment\',int2str(j),'.png'));
                j=j+1;
               %ֻ���һ����
            end
        end
   end
    patchsx=ceil(0.9*patchsx);%ÿ�γ���0.9Ȼ��ȡ��
    patchsy=ceil(0.9*patchsy);
end

    