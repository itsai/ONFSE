function [TotalFACE,TotalMeanFACE,PCA,pcaTotalFACE,SW,SB,SWpool,SWvector,SBpool,SBvector,NFSEeig,NFSEeigvalue,latent]= ONFSE_Train_Eig

people=5;          %�Ҧ��H

individualsample=8; %(15->12, 12->15, 9->19,  6->29, 5_>34, 3->57) %170�i�D8�i�A�i�H��Random�覡

nearestneighbor_sw=4;   % �C�@�Ӽ˥��b�դ��D��̪񪺾F�~��, individualsample=4 (C4��2)

nearestneighbor_sb=8;   % �C�@�Ӽ˥��b�ն��D��̪񪺾F�~�� C8��2

neighbor=6;  % K1  ori 10 ��F���I�A�դ����u

neighborsb=28; % K3 ���N��L���O���F�~�A���F�DSB

principlenum=300; % PCA���� �Ҧ����Ū�i�᭰���A�ҡG32x32(�@�i�v��)�ꦨ1024�A(�G�����@����̤j��300)

TotalFACE=[]; %�Ҧ��˥��v�� 1024 x (68x8)

pcaTotalFACE=[]; %�������300


%+++++++  PCA transform ++++++++++++++++++++++++++++++++++++++++++++++++++++++

for i=1:people
    i
FACE=[];


% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   for j=1:1:170
    if (j==1 | j==30 | j==59 | j==88 | j==117 | j==146 | j==170 | j==136) % individualsample=7�A�H���˥������    
    s=['cmu3232' '\' num2str(i) '\' num2str(j) '.bmp'];
    X=imread(s); %Ū���|�v��
    X=double(X); %Ū�i�ӬO�T����ơA��double�����ܤG��
    [row,col]=size(X); %Ū������ƪ��e�M�� 32x32
    %figure(1);
    %imshow(X,map);

    %--
    face0=X;
    tempface0=[];
    %--�Ʀ��@�� row(1024)
    for k=1:row
        tempface0=[tempface0,face0(k,:)];
    end
    %FACE=[FACE;tempface0];                   %�Ȧs��@���O��Ŷ��V�m�v��
    TotalFACE=[TotalFACE;tempface0];         %�Ҧ���Ŷ��V�m�v��  (�ꦨ��)  
    end % end of if 
   end % end of j
% ----------------------------------------------------------------------   
   
end % end of i

TotalMeanFACE=mean(TotalFACE); %�Ҧ��˥���mean�������� (1x1024)
zeromeanTotalFACE=TotalFACE;                %���W�ƫ�Ҧ���Ŷ��V�m�v��(�Hmean������I)

%++++++++++ zero mean ++++++++++++++++++++++++++++++++
for i=1:1:individualsample*people
    for j=1:1:(row)*(col) 
        zeromeanTotalFACE(i,j)=zeromeanTotalFACE(i,j)-TotalMeanFACE(j); %�Hmean�����I���W��
    end
end
%-----------------------------------------------------

pcaSST=cov(zeromeanTotalFACE);  %�D�@�ܲ��Ưx�} (1024)x(68x8) * cov�]x * y) = (1024x544����m�x�}�O544x1024)(1024x1024)

[PCA,latent,explained] = pcacov(pcaSST);    %PCA�D�������R(Matlab���ب禡)(�D�n�@�Ƨ�) PCA = 1024*1024

projectPCA=PCA(:,1:principlenum);                       %���X�D����������(���X��1~300), projectPCA(���n���ײv)=1024x300

for i=1:1:individualsample*people   %��v 68x8
    tempFACE=zeromeanTotalFACE(i,:); %tempFACE�Ȧs�� (i=1~544:1024)=1x1024(1��)
    tempFACE=projectPCA'*tempFACE';  %'=(�x�}��m) (300x1024)*(1024x1) = 300x1
    tempFACE=tempFACE'; %300x1'= 1x300 (Row��Col)
    pcaTotalFACE=[pcaTotalFACE;tempFACE];               %�֥[�x�s�Ҧ���v��PCA�Ŷ������V�m�v��(�����᪺�˥�)�Ĥ@����8�Ӹ�Ʀ@68�Hx�@����Ʀ�300�Ӻ��� (pcaTotalFace�D�X�ᬰ544x300)
end
%projectPCA = ��v�᪺�Ҧ����ת��סA�@��300�Ӻ���(�C�Ӻ��צ��U�۪���)
%------- PCA transform ------------------------------------------------------------------------------
    
    
SW=zeros(principlenum,principlenum);             %  ��l�� SW �� 0 principlenum = (300x300)
SB=zeros(principlenum,principlenum);             %  ��l�� SB �� 0
%totaldis=0;
samplecount=0; %�p�ƾ�

    %++++++++ NFL SW ++++++++++++++++++++++++++++++++
    
    
for i=1:individualsample:individualsample*people  %for(i = 1; i < 68 * 8; i += 8)
    FACE=pcaTotalFACE(i:i+individualsample-1,:);   %�Ȧs��@���OPCA�Ŷ����V�m�v��    i�O1,9,17 ...  68�ӤH�C�ӤH��170������H������8����ơA��C�ӤH���P�@��300x1���X
    
    
    
    %------------------------------------------------------------------------------------------------------------ 
    for k=1:individualsample % �C�@���V�m��ƭn�P����䥦�@���ưt�� n*c(n-1,2), 4*c(3,2)=12
        combination=FACE;
        combination(k,:)=[]; % k �O�@�ӷǳƥΨӧ�v�ܦV�q mn �W���@���I�A����I�q sample ���ް���A�ѤU���N�O�ΨӲզX���u�� sample

        %++++++ �b within ���N���@�I�P�䥦�Ҧ��I�������Z���ƧǡA�i�H�קK�L���I�զ����u�P FACE(k,:) �����������Z���Q���(���ئh�� extrapolation)�A���ت��[�W������N���    
        nearest_sw_dis=[]; % �@�� column vector�A�ΨӼȦs FACE(k,:)�Pcombination(?,:) �������Z���H��X�̾F�񪺾F�~�� (�Z���O�ΨӱƧǪ��з�)
        
        for t=1:individualsample-1                              % �h�� k �@���I�A�ѤU�� within �H�y��
            NormalVector=FACE(k,:)-combination(t,:);            % �Ȧs�V�q�A�Ψӭp����ץ� (��ۤw�P��L�I���Z��)
            tempdis_0=NormalVector*NormalVector';               %�ۤw���ۤw�ΨӺ�Z������
            nearest_sw_dis=[nearest_sw_dis;tempdis_0];          %1x7�Z�}�s��7�ӶZ�����׭�
        end
        
        [junk, index] = sort(nearest_sw_dis);                   %���Ƨ�(�Ѥp�Ӥj)
        combination = combination(index,:);                     %��index���s�Ƨ�
    %------------------------------------------------------------------------------------------------------------
          
    
    
    
        
        
        %tempdis_0=inf;                                         % ���F�p��̵uNFL
        
        
        
        SWpool=[]; %  ���F�Ȧs�C�@����ƪ��Ҧ��I��u���V�q�A�~��Hneighbor���������neighbor��
        
       for m=1:nearestneighbor_sw-1                              % �h�� k �@���I�A�H�� combination ���̫�@���I
            OA=FACE(k,:)-combination(m,:);                      % �H m ��@�_�I������v�V�q�A��Ҭ����� a �V�q
            for n=m+1:nearestneighbor_sw                        % m �P n �O�ΨӲզX�����@���u�ҨϥΪ� sample �I
                OB=combination(n,:)-combination(m,:);           % �H m ��@�_�I���Q��v�V�q�A��Ҭ����� b �V�q
                OM=((OA*OB')/(OB*OB'))*OB;                      % ��v�I�����A��Ҭ����� m �V�q
                NormalVector=OA-OM;                             % �I���v�I�������V�q(�k�V�q)
                
                if (OB*OB'>0)                                   % �Ҧ���NFL(���b�A�קK��0��)
                %    SW=SW+(NormalVector'*NormalVector);          % �p��դ��ܲ�
                    tempdis_0=NormalVector*NormalVector';       % (1x300) * (300x1)' = 1x1(tempdis_0)
                %    totaldis=totaldis+tempdis_0;
                %    samplecount=samplecount+1;
                    SWvector=zeros(1,1+principlenum);           %(1x301)�A1�O�����tempdis_0�A2~301�QNormalVector
                    
                    SWvector(1,1)=tempdis_0;
                    SWvector(1,2:1+principlenum)=NormalVector;  
                    SWpool=[SWpool;SWvector];   %�|�W�h (for n=m+1:individualsample-1)�A�ҥHSWpool = 21 x 301
                end
                
                %if (NormalVector*NormalVector'<tempdis_0)      % �̵u��NFL
                %    tempdis_0=NormalVector*NormalVector';
                %    tempSW=NormalVector'*NormalVector;

                %end

            end % end of n
        end % end of m
        %SW=SW+tempSW;
        %totaldis=totaldis+tempdis_0;
        %samplecount=samplecount+1;
        
        %-------- �q neighbor ���̧Ǳq�̵u�̿�X within ���V�q�Ӻ�cov----------
        [SWpool,swindex]=sortrows(SWpool,1);    %�ھڦ��x�}���Y�@�C�Ƨ�(�Ѥp��j)
        for neighborcount=1:1:neighbor   %�ھ�OA-OM����(NormalVector���Z���Ȥ��@�w�O�Ѥp��j)
            SW=SW+SWpool(neighborcount,2:1+principlenum)'*SWpool(neighborcount,2:1+principlenum); %(300x1) * (1x300)= 300x300 (���F�Dcov)
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
    FACE=temppcaTotalFACE(i:i+individualsample-1,:);       %�Ȧs��@���OPCA�Ŷ����V�m�v��
    temppcaTotalFACE(i:i+individualsample-1,:)=[];         %�ѤU�䥦�Ҧ����OPCA�Ŷ����V�m�v�� (��ۤw���O����)
    
    for j=1:1:individualsample %1~8
        
        %SBtempdis_0=inf;
        
         SBcombination=temppcaTotalFACE;    %sb��combination
         
          %++++++ �b beteen ���N���@�I�P�䥦���O�Ҧ��I�������Z���ƧǡA�i�H�קK�L���I�զ����u�P FACE(k,:) �����������Z���Q���(���ئh�� extrapolation)�A���ت��[�W������N���    
        nearest_sb_dis=[]; % �@�� column vector�A�ΨӼȦs FACE(k,:)�PSBcombination(?,:) �������Z���H��X�̾F�񪺾F�~�� (�Z���O�ΨӱƧǪ��з�)
        
        for t=1:individualsample*(people-1)                       % between�䥦���O���H�y��(8*67=536)
            NormalVector=FACE(j,:)-SBcombination(t,:);            % �Ȧs�V�q�A�Ψӭp����ץ�
            tempdis_0=NormalVector*NormalVector';
            nearest_sb_dis=[nearest_sb_dis;tempdis_0];
        end
        
        [junk, index] = sort(nearest_sb_dis);       %�Ѥp�Ӥj
        SBcombination = SBcombination(index,:);     %�ھ�index���s�զX
    %------------------------------------------------------------------------------------------------------------              
            
        
        SBpool=[]; %  ���F�Ȧs�C�@����ƪ��Ҧ��I��u���V�q�A�~��Hneighbor���������neighbor��
        
        
for m=1:nearestneighbor_sb         
OA=FACE(j,:)-SBcombination(m,:);            
                for n=m+1:nearestneighbor_sb                          %m=1, n=1~7
                    OB=SBcombination(n,:)-SBcombination(m,:);
                    OM=((OA*OB')/(OB*OB'))*OB;
                    NormalVector=OA-OM;
                    
                    if (OB*OB'>0)                                        % �Ҧ���NFL
                    %   tempSB=NormalVector'*NormalVector;                % �p��ն��ܲ�
                        SBtempdis_0=NormalVector*NormalVector';         %(1x300) * (300x1) = 1x1
                    %   SB=SB+tempSB*exp(-SBtempdis_0/(2*Avgtotaldis));
                                        
                        SBvector=zeros(1,1+principlenum);               %1~301�k0
                        SBvector(1,1)=SBtempdis_0;                      %��Z����i(1,1)����m
                        SBvector(1,2:1+principlenum)=NormalVector;      %(2~301)��NormalVector����
                        SBpool=[SBpool;SBvector];                       %�������j��counter = 1876, SBpool=1876x301
                    end

                    %if (NormalVector*NormalVector'<SBtempdis_0)
                    %   SBtempdis_0=NormalVector*NormalVector';
                    %   tempSB=NormalVector'*NormalVector;

                    %end % end of if
               end % end of n
           end % end of m
       end % end of k
        %SB=SB+tempSB;%*exp(-SBtempdis_0/(2*Avgtotaldis));
        
        %-------- �q neighbor ���̧Ǳq�̵u�̿�X within ���V�q�Ӻ�cov----------
        [SBpool,sbindex]=sortrows(SBpool,1);
        for neighborcount=1:1:neighborsb
            SB=SB+SBpool(neighborcount,2:1+principlenum)'*SBpool(neighborcount,2:1+principlenum);
        end
        %----------------------------------------------------------------------         
        sbcount=sbcount+1
end % end of i
%------------------------------------------------

%[NFLPCA,latent,explained] = pcacov(SB*inv(SW)); �Y�����~
%[NFSEeig,latent] = eig(inv(SW)*SB); �Y�����~

[NFSEeig,latent] = eig(inv(SW)*SB);  %Max = sb/sw �]���� 1/sw * sb�Alatent = eigen value (NFSEeig=�S�x�V�q, latent=�S�x�x�}��) (���k���Ǥ����ܡA�|�����D) (value�V�j�~�O�Q�n�����G)
NFSEeigvalue=diag(latent);           %diag = ���X�﨤�u�x�}(diagonal matrix)���ȡA���O�Z��
[junk, index] = sort(-NFSEeigvalue); % sort �O�Ѥp�Ӥj�A�ҥH�[�t�� (=�Ѥj��p) (eig���|�ƶ��ǡA�ҥH�n��)
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