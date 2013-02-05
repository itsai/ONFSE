function [TotalFACE,TotalMeanFACE,PCA,pcaTotalFACE,SW,SB,SWpool,SWvector,SBpool,SBvector,NFSEeig,NFSEeigvalue,latent]= ONFSE_Train_Eig

people=5;          %所有人

individualsample=8; %(15->12, 12->15, 9->19,  6->29, 5_>34, 3->57) %170張挑8張，可以用Random方式

nearestneighbor_sw=4;   % 每一個樣本在組內挑選最近的鄰居數, individualsample=4 (C4取2)

nearestneighbor_sb=8;   % 每一個樣本在組間挑選最近的鄰居數 C8取2

neighbor=6;  % K1  ori 10 抓鄰近的點，組內的線

neighborsb=28; % K3 任意其他類別的鄰居，為了求SB

principlenum=300; % PCA維度 所有資料讀進後降維，例：32x32(一張影像)串成1024，(二維降一維後最大取300)

TotalFACE=[]; %所有樣本影像 1024 x (68x8)

pcaTotalFACE=[]; %降維後剩300


%+++++++  PCA transform ++++++++++++++++++++++++++++++++++++++++++++++++++++++

for i=1:people
    i
FACE=[];


% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   for j=1:1:170
    if (j==1 | j==30 | j==59 | j==88 | j==117 | j==146 | j==170 | j==136) % individualsample=7，以的樣本為基準    
    s=['cmu3232' '\' num2str(i) '\' num2str(j) '.bmp'];
    X=imread(s); %讀路徑影像
    X=double(X); %讀進來是三維資料，用double降維變二維
    [row,col]=size(X); %讀此筆資料的寬和高 32x32
    %figure(1);
    %imshow(X,map);

    %--
    face0=X;
    tempface0=[];
    %--排成一個 row(1024)
    for k=1:row
        tempface0=[tempface0,face0(k,:)];
    end
    %FACE=[FACE;tempface0];                   %暫存單一類別原空間訓練影像
    TotalFACE=[TotalFACE;tempface0];         %所有原空間訓練影像  (串成行)  
    end % end of if 
   end % end of j
% ----------------------------------------------------------------------   
   
end % end of i

TotalMeanFACE=mean(TotalFACE); %所有樣本做mean取平均數 (1x1024)
zeromeanTotalFACE=TotalFACE;                %正規化後所有原空間訓練影像(以mean為基準點)

%++++++++++ zero mean ++++++++++++++++++++++++++++++++
for i=1:1:individualsample*people
    for j=1:1:(row)*(col) 
        zeromeanTotalFACE(i,j)=zeromeanTotalFACE(i,j)-TotalMeanFACE(j); %以mean為原點正規化
    end
end
%-----------------------------------------------------

pcaSST=cov(zeromeanTotalFACE);  %求共變異數矩陣 (1024)x(68x8) * cov（x * y) = (1024x544的轉置矩陣是544x1024)(1024x1024)

[PCA,latent,explained] = pcacov(pcaSST);    %PCA主成份分析(Matlab內建函式)(主要作排序) PCA = 1024*1024

projectPCA=PCA(:,1:principlenum);                       %取出主成分的維度(取出第1~300), projectPCA(重要的斜率)=1024x300

for i=1:1:individualsample*people   %投影 68x8
    tempFACE=zeromeanTotalFACE(i,:); %tempFACE暫存用 (i=1~544:1024)=1x1024(1筆)
    tempFACE=projectPCA'*tempFACE';  %'=(矩陣轉置) (300x1024)*(1024x1) = 300x1
    tempFACE=tempFACE'; %300x1'= 1x300 (Row轉Col)
    pcaTotalFACE=[pcaTotalFACE;tempFACE];               %累加儲存所有投影至PCA空間中的訓練影像(降維後的樣本)第一筆有8個資料共68人x一筆資料有300個維度 (pcaTotalFace求出後為544x300)
end
%projectPCA = 投影後的所有維度長度，共有300個維度(每個維度有各自的值)
%------- PCA transform ------------------------------------------------------------------------------
    
    
SW=zeros(principlenum,principlenum);             %  初始化 SW 為 0 principlenum = (300x300)
SB=zeros(principlenum,principlenum);             %  初始化 SB 為 0
%totaldis=0;
samplecount=0; %計數器

    %++++++++ NFL SW ++++++++++++++++++++++++++++++++
    
    
for i=1:individualsample:individualsample*people  %for(i = 1; i < 68 * 8; i += 8)
    FACE=pcaTotalFACE(i:i+individualsample-1,:);   %暫存單一類別PCA空間中訓練影像    i是1,9,17 ...  68個人每個人有170筆資料隨機取的8筆資料，把每個人的同一筆300x1取出
    
    
    
    %------------------------------------------------------------------------------------------------------------ 
    for k=1:individualsample % 每一筆訓練資料要與任何其它一對資料配對 n*c(n-1,2), 4*c(3,2)=12
        combination=FACE;
        combination(k,:)=[]; % k 是一個準備用來投影至向量 mn 上的一個點，把該點從 sample 中拔除後，剩下的就是用來組合成線的 sample

        %++++++ 在 within 中將任一點與其它所有點之間的距離排序，可以避免過遠點組成的線與 FACE(k,:) 之間的垂直距離被選到(此種多為 extrapolation)，此種直觀上較不具代表性    
        nearest_sw_dis=[]; % 一個 column vector，用來暫存 FACE(k,:)與combination(?,:) 之間的距離以找出最鄰近的鄰居們 (距離是用來排序的標準)
        
        for t=1:individualsample-1                              % 去掉 k 一個點，剩下的 within 人臉數
            NormalVector=FACE(k,:)-combination(t,:);            % 暫存向量，用來計算長度用 (算自已與其他點的距離)
            tempdis_0=NormalVector*NormalVector';               %自已乘自已用來算距離長度
            nearest_sw_dis=[nearest_sw_dis;tempdis_0];          %1x7距陣存放7個距離長度值
        end
        
        [junk, index] = sort(nearest_sw_dis);                   %做排序(由小而大)
        combination = combination(index,:);                     %用index重新排序
    %------------------------------------------------------------------------------------------------------------
          
    
    
    
        
        
        %tempdis_0=inf;                                         % 為了計算最短NFL
        
        
        
        SWpool=[]; %  為了暫存每一筆資料的所有點到線的向量，才能以neighbor的概念選擇neighbor數
        
       for m=1:nearestneighbor_sw-1                              % 去掉 k 一個點，以及 combination 的最後一個點
            OA=FACE(k,:)-combination(m,:);                      % 以 m 當作起點的欲投影向量，思考派中的 a 向量
            for n=m+1:nearestneighbor_sw                        % m 與 n 是用來組合成為一條線所使用的 sample 點
                OB=combination(n,:)-combination(m,:);           % 以 m 當作起點的被投影向量，思考派中的 b 向量
                OM=((OA*OB')/(OB*OB'))*OB;                      % 投影點公式，思考派中的 m 向量
                NormalVector=OA-OM;                             % 點到投影點之間的向量(法向量)
                
                if (OB*OB'>0)                                   % 所有的NFL(防呆，避免有0值)
                %    SW=SW+(NormalVector'*NormalVector);          % 計算組內變異
                    tempdis_0=NormalVector*NormalVector';       % (1x300) * (300x1)' = 1x1(tempdis_0)
                %    totaldis=totaldis+tempdis_0;
                %    samplecount=samplecount+1;
                    SWvector=zeros(1,1+principlenum);           %(1x301)，1是放長度tempdis_0，2~301被NormalVector
                    
                    SWvector(1,1)=tempdis_0;
                    SWvector(1,2:1+principlenum)=NormalVector;  
                    SWpool=[SWpool;SWvector];   %疊上去 (for n=m+1:individualsample-1)，所以SWpool = 21 x 301
                end
                
                %if (NormalVector*NormalVector'<tempdis_0)      % 最短的NFL
                %    tempdis_0=NormalVector*NormalVector';
                %    tempSW=NormalVector'*NormalVector;

                %end

            end % end of n
        end % end of m
        %SW=SW+tempSW;
        %totaldis=totaldis+tempdis_0;
        %samplecount=samplecount+1;
        
        %-------- 從 neighbor 中依序從最短者選出 within 的向量來算cov----------
        [SWpool,swindex]=sortrows(SWpool,1);    %根據此矩陣的某一列排序(由小到大)
        for neighborcount=1:1:neighbor   %根據OA-OM長度(NormalVector的距離值不一定是由小到大)
            SW=SW+SWpool(neighborcount,2:1+principlenum)'*SWpool(neighborcount,2:1+principlenum); %(300x1) * (1x300)= 300x300 (為了求cov)
        end
        %----------------------------------------------------------------------
        samplecount=samplecount+1 %68*8
    end % end of k=8
end % end of i=68
    %--------------------------------------------------
%Avgtotaldis=totaldis/samplecount
sbcount=0;  
%++++++++ NFL SB ++++++++++++++++++++++++++++++++
for i=1:individualsample:(people*individualsample)         %for(i=1; i<8*(68*8); i+=8)
    temppcaTotalFACE=pcaTotalFACE;
    FACE=temppcaTotalFACE(i:i+individualsample-1,:);       %暫存單一類別PCA空間中訓練影像
    temppcaTotalFACE(i:i+individualsample-1,:)=[];         %剩下其它所有類別PCA空間中訓練影像 (把自已類別挖掉)
    
    for j=1:1:individualsample %1~8
        
        %SBtempdis_0=inf;
        
         SBcombination=temppcaTotalFACE;    %sb的combination
         
          %++++++ 在 beteen 中將任一點與其它類別所有點之間的距離排序，可以避免過遠點組成的線與 FACE(k,:) 之間的垂直距離被選到(此種多為 extrapolation)，此種直觀上較不具代表性    
        nearest_sb_dis=[]; % 一個 column vector，用來暫存 FACE(k,:)與SBcombination(?,:) 之間的距離以找出最鄰近的鄰居們 (距離是用來排序的標準)
        
        for t=1:individualsample*(people-1)                       % between其它類別的人臉數(8*67=536)
            NormalVector=FACE(j,:)-SBcombination(t,:);            % 暫存向量，用來計算長度用
            tempdis_0=NormalVector*NormalVector';
            nearest_sb_dis=[nearest_sb_dis;tempdis_0];
        end
        
        [junk, index] = sort(nearest_sb_dis);       %由小而大
        SBcombination = SBcombination(index,:);     %根據index重新組合
    %------------------------------------------------------------------------------------------------------------              
            
        
        SBpool=[]; %  為了暫存每一筆資料的所有點到線的向量，才能以neighbor的概念選擇neighbor數
        
        
for m=1:nearestneighbor_sb         
OA=FACE(j,:)-SBcombination(m,:);            
                for n=m+1:nearestneighbor_sb                          %m=1, n=1~7
                    OB=SBcombination(n,:)-SBcombination(m,:);
                    OM=((OA*OB')/(OB*OB'))*OB;
                    NormalVector=OA-OM;
                    
                    if (OB*OB'>0)                                        % 所有的NFL
                    %   tempSB=NormalVector'*NormalVector;                % 計算組間變異
                        SBtempdis_0=NormalVector*NormalVector';         %(1x300) * (300x1) = 1x1
                    %   SB=SB+tempSB*exp(-SBtempdis_0/(2*Avgtotaldis));
                                        
                        SBvector=zeros(1,1+principlenum);               %1~301歸0
                        SBvector(1,1)=SBtempdis_0;                      %把距離填進(1,1)的位置
                        SBvector(1,2:1+principlenum)=NormalVector;      %(2~301)填NormalVector的值
                        SBpool=[SBpool;SBvector];                       %全部的迴圈counter = 1876, SBpool=1876x301
                    end

                    %if (NormalVector*NormalVector'<SBtempdis_0)
                    %   SBtempdis_0=NormalVector*NormalVector';
                    %   tempSB=NormalVector'*NormalVector;

                    %end % end of if
               end % end of n
           end % end of m
       end % end of k
        %SB=SB+tempSB;%*exp(-SBtempdis_0/(2*Avgtotaldis));
        
        %-------- 從 neighbor 中依序從最短者選出 within 的向量來算cov----------
        [SBpool,sbindex]=sortrows(SBpool,1);
        for neighborcount=1:1:neighborsb
            SB=SB+SBpool(neighborcount,2:1+principlenum)'*SBpool(neighborcount,2:1+principlenum);
        end
        %----------------------------------------------------------------------         
        sbcount=sbcount+1
end % end of i
%------------------------------------------------

%[NFLPCA,latent,explained] = pcacov(SB*inv(SW)); 嚴重錯誤
%[NFSEeig,latent] = eig(inv(SW)*SB); 嚴重錯誤

[NFSEeig,latent] = eig(inv(SW)*SB);  %Max = sb/sw 也等於 1/sw * sb，latent = eigen value (NFSEeig=特徵向量, latent=特徵矩陣值) (乘法順序不能變，會有問題) (value越大才是想要的結果)
NFSEeigvalue=diag(latent);           %diag = 取出對角線矩陣(diagonal matrix)的值，單位是距離
[junk, index] = sort(-NFSEeigvalue); % sort 是由小而大，所以加負號 (=由大到小) (eig不會排順序，所以要排)
NFSEeig = NFSEeig(:, index);
NFSEeigvalue=NFSEeigvalue(index);

%======================================
% Generalized Eigen with Orthogonal Constraint 
%======================================

% dimM = size(SB,2)
% rDPrime = chol(SW);
% lDPrime = rDPrime';
% Q0 = rDPrime\(lDPrime\SB);
% ONFSEeig = [];
% ONFSEeigvalue = [];
% tmpD = [];
% Q = Q0;
% 
% for i = 1:100
%      [eigVec, eigv] = eigs(Q,1,'lr');
%      ONFSEeig = [ONFSEeig, eigVec];
%      ONFSEeigvalue = [ONFSEeigvalue;eigv];
%      tmpD = [tmpD, rDPrime\(lDPrime\eigVec)];
%      DTran = ONFSEeig';
%      tmptmpD = DTran*tmpD;
%      tmptmpD = max(tmptmpD,tmptmpD');
%      rtmptmpD = chol(tmptmpD);
%      tmptmpD = rtmptmpD\(rtmptmpD'\DTran);
%      Q = -tmpD*tmptmpD;
%      for j=1:dimM
%          Q(j,j) = Q(j,j) + 1;
%      end
%      Q = Q*Q0;
%      disp([num2str(i),' eigenvector calculated!']);
% end