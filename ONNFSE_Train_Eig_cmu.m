function [TotalFACE,TotalMeanFACE,PCA,pcaTotalFACE,SW,SB,SWpool,SWvector,SBpool,SBvector,NFSEeig,NFSEeigvalue]= ONNFSE_Train_Eig_cmu
%Orthogonal Nearest Neighbor Feature Space Embedding (ONNFSE)

people=5;

individualsample=8;     % 每一個類別訓練樣本數 ()

nearestneighbor_sw=4;    % 每一個樣本在組內挑選最近的鄰居數

nearestneighbor_sb=8;   % 每一個樣本在組間挑選最近的鄰居數

neighbor=6;             % K1, individualsample=8

neighborsb=28;           % K2

principlenum=300; % PCA維度, individualsample=6, 9

TotalFACE=[];

pcaTotalFACE=[];

%+++++++  PCA transform ++++++++++++++++++++++++++++++++++++++++++++++++++++++

for i=1:people
    i
FACE=[];


% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   for j=1:1:170
    if (j==1 | j==30 | j==59 | j==88 | j==117 | j==146 | j==170 | j==136) % individualsample=7，以的樣本為基準    
    s=['cmu3232' '\' num2str(i) '\' num2str(j) '.bmp'];
    X=imread(s);
    X=double(X);
    [row,col]=size(X);
    %figure(1);
    %imshow(X,map);

    %--
    face0=X;
    tempface0=[];
    %--排成一個 row
    for k=1:row
        tempface0=[tempface0,face0(k,:)];
    end
    FACE=[FACE;tempface0];                   %暫存單一類別原空間訓練影像
    TotalFACE=[TotalFACE;tempface0];         %所有原空間訓練影像
    end
   end % end of j
% ----------------------------------------------------------------------   
   
end % end of i

TotalMeanFACE=mean(TotalFACE);
zeromeanTotalFACE=TotalFACE;                %正規化後所有原空間訓練影像

%++++++++++ zero mean ++++++++++++++++++++++++++++++++
for i=1:1:individualsample*people
    for j=1:1:(row)*(col) 
        zeromeanTotalFACE(i,j)=zeromeanTotalFACE(i,j)-TotalMeanFACE(j); %正規化
    end
end
%-----------------------------------------------------

pcaSST=cov(zeromeanTotalFACE);

display('主成分分析')
[PCA,latent,explained] = pcacov(pcaSST);

projectPCA=PCA(:,1:principlenum);                       %取出主成分的維度

for i=1:1:individualsample*people
    tempFACE=zeromeanTotalFACE(i,:);
    tempFACE=projectPCA'*tempFACE';
    tempFACE=tempFACE';
    pcaTotalFACE=[pcaTotalFACE;tempFACE];               %儲存所有投影至PCA空間中的訓練影像
end

%------- PCA transform ------------------------------------------------------------------------------

SW=zeros(principlenum,principlenum);             %  初始化 SW 為 0
SB=zeros(principlenum,principlenum);             %  初始化 SB 為 0
totaldis=0;
samplecount=0;

    %++++++++ NFL SW ++++++++++++++++++++++++++++++++

for i=1:individualsample:individualsample*people
    FACE=pcaTotalFACE(i:i+individualsample-1,:);   %暫存單一類別PCA空間中訓練影像    
    
    for k=1:individualsample % 每一筆訓練資料要與任何其它一對資料配對 n*c(n-1,2), 4*c(3,2)=12
        combination=FACE;
        combination(k,:)=[]; % k 是一個準備用來投影至向量 mn 上的一個點，把該點從 sample 中拔除後，剩下的就是用來組合成線的候選 sample

        
    %++++++ 在 within 中將任一點與其它所有點之間的距離排序，可以避免過遠點組成的線與 FACE(k,:) 之間的垂直距離被選到(此種多為 extrapolation)，此種直觀上較不具代表性    
        nearest_sw_dis=[]; % 一個 column vector，用來暫存 FACE(k,:)與combination(?,:) 之間的距離以找出最鄰近的鄰居們 (距離是用來排序的標準)
        
        for t=1:individualsample-1                              % 去掉 k 一個點，剩下的 within 人臉數
            NormalVector=FACE(k,:)-combination(t,:);            % 暫存向量，用來計算長度用
            tempdis_0=NormalVector*NormalVector';
            nearest_sw_dis=[nearest_sw_dis;tempdis_0];
        end
        
        [junk, index] = sort(nearest_sw_dis);
        combination = combination(index,:);
    %------------------------------------------------------------------------------------------------------------
          
        SWpool=[]; %  為了暫存每一筆資料的所有點到線的向量，才能以neighbor的概念選擇neighbor數
        
        for m=1:nearestneighbor_sw-1                            
            OA=FACE(k,:)-combination(m,:);                      % 以 m 當作起點的欲投影向量，思考派中的 a 向量
            for n=m+1:nearestneighbor_sw                        % m 與 n 是用來組合成為一條線所使用的 sample 點
                OB=combination(n,:)-combination(m,:);           % 以 m 當作起點的被投影向量，思考派中的 b 向量
                OM=((OA*OB')/(OB*OB'))*OB;                      % 投影點公式，思考派中的 m 向量
                NormalVector=OA-OM;                             % 點到投影點之間的向量(法向量)
                
                if (OB*OB'>0)                                   % 防呆，避免分母為0
  
                    tempdis_0=NormalVector*NormalVector';
    
                    SWvector=zeros(1,1+principlenum);
                    
                    SWvector(1,1)=tempdis_0;
                    SWvector(1,2:1+principlenum)=NormalVector;
                    SWpool=[SWpool;SWvector];
                end
                

            end % end of n
        end % end of m

        
        %-------- 從 neighbor 中依序從最短者選出 within 的向量來算cov----------
        [SWpool,swindex]=sortrows(SWpool,1);
        %[junkm,junkn]=size(SWpool);
        
        %if (junkm>neighbor)
           for neighborcount=1:1:neighbor
               SW=SW+SWpool(neighborcount,2:1+principlenum)'*SWpool(neighborcount,2:1+principlenum);
           end
        %end
        
        %if (junkm<neighbor)
        %   for neighborcount=1:1:junkm
        %       SW=SW+SWpool(neighborcount,2:1+principlenum)'*SWpool(neighborcount,2:1+principlenum);
        %   end
        %end
        %----------------------------------------------------------------------
        samplecount=samplecount+1
    end % end of k
end % end of i
    %--------------------------------------------------

sbcount=0;  
%++++++++ NFL SB ++++++++++++++++++++++++++++++++
for i=1:individualsample:(people*individualsample)
    temppcaTotalFACE=pcaTotalFACE;
    FACE=temppcaTotalFACE(i:i+individualsample-1,:);       %暫存單一類別PCA空間中訓練影像
    temppcaTotalFACE(i:i+individualsample-1,:)=[];         %剩下其它所有類別PCA空間中訓練影像
    
    for j=1:1:individualsample
            
        SBcombination=temppcaTotalFACE;

   %++++++ 在 beteen 中將任一點與其它類別所有點之間的距離排序，可以避免過遠點組成的線與 FACE(k,:) 之間的垂直距離被選到(此種多為 extrapolation)，此種直觀上較不具代表性    
        nearest_sb_dis=[]; % 一個 column vector，用來暫存 FACE(k,:)與SBcombination(?,:) 之間的距離以找出最鄰近的鄰居們 (距離是用來排序的標準)
        
        for t=1:individualsample*(people-1)                       % between其它類別的人臉數
            NormalVector=FACE(j,:)-SBcombination(t,:);            % 暫存向量，用來計算長度用
            tempdis_0=NormalVector*NormalVector';
            nearest_sb_dis=[nearest_sb_dis;tempdis_0];
        end
        
        [junk, index] = sort(nearest_sb_dis);
        SBcombination = SBcombination(index,:);
    %------------------------------------------------------------------------------------------------------------              
            
            SBpool=[]; %  為了暫存每一筆資料的所有點到線的向量，才能以neighbor的概念選擇neighbor數
            
            for m=1:nearestneighbor_sb-1
                OA=FACE(j,:)-SBcombination(m,:);
                for n=m+1:nearestneighbor_sb
                    OB=SBcombination(n,:)-SBcombination(m,:);
                    OM=((OA*OB')/(OB*OB'))*OB;
                    NormalVector=OA-OM;
                    
                    if (OB*OB'>0)                                        % 所有的NFL
                    
                        SBtempdis_0=NormalVector*NormalVector';
              
                        SBvector=zeros(1,1+principlenum);
                        SBvector(1,1)=SBtempdis_0;
                        SBvector(1,2:1+principlenum)=NormalVector;
                        SBpool=[SBpool;SBvector];
                    end


               end % end of n
           end % end of m

        %-------- 從 neighbor 中依序從最短者選出 within 的向量來算cov----------
        [SBpool,sbindex]=sortrows(SBpool,1);
        for neighborcount=1:1:neighborsb
            SB=SB+SBpool(neighborcount,2:1+principlenum)'*SBpool(neighborcount,2:1+principlenum);
        end
        %----------------------------------------------------------------------         
        sbcount=sbcount+1
    end % end of j
end % end of i
%------------------------------------------------

%[NFLPCA,latent,explained] = pcacov(SB*inv(SW)); 嚴重錯誤
%[NFSEeig,latent] = eig(inv(SW)*SB); 嚴重錯誤

[NFSEeig,latent] = eig(inv(SW)*SB);
NFSEeigvalue=diag(latent);
[junk, index] = sort(-NFSEeigvalue); % sort 是由小而大，所以加負號
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
% for i = 1:30
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