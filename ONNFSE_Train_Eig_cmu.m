function [TotalFACE,TotalMeanFACE,PCA,pcaTotalFACE,SW,SB,SWpool,SWvector,SBpool,SBvector,NFSEeig,NFSEeigvalue]= ONNFSE_Train_Eig_cmu
%Orthogonal Nearest Neighbor Feature Space Embedding (ONNFSE)

people=5;

individualsample=8;     % �C�@�����O�V�m�˥��� ()

nearestneighbor_sw=4;    % �C�@�Ӽ˥��b�դ��D��̪񪺾F�~��

nearestneighbor_sb=8;   % �C�@�Ӽ˥��b�ն��D��̪񪺾F�~��

neighbor=6;             % K1, individualsample=8

neighborsb=28;           % K2

principlenum=300; % PCA����, individualsample=6, 9

TotalFACE=[];

pcaTotalFACE=[];

%+++++++  PCA transform ++++++++++++++++++++++++++++++++++++++++++++++++++++++

for i=1:people
    i
FACE=[];


% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   for j=1:1:170
    if (j==1 | j==30 | j==59 | j==88 | j==117 | j==146 | j==170 | j==136) % individualsample=7�A�H���˥������    
    s=['cmu3232' '\' num2str(i) '\' num2str(j) '.bmp'];
    X=imread(s);
    X=double(X);
    [row,col]=size(X);
    %figure(1);
    %imshow(X,map);

    %--
    face0=X;
    tempface0=[];
    %--�Ʀ��@�� row
    for k=1:row
        tempface0=[tempface0,face0(k,:)];
    end
    FACE=[FACE;tempface0];                   %�Ȧs��@���O��Ŷ��V�m�v��
    TotalFACE=[TotalFACE;tempface0];         %�Ҧ���Ŷ��V�m�v��
    end
   end % end of j
% ----------------------------------------------------------------------   
   
end % end of i

TotalMeanFACE=mean(TotalFACE);
zeromeanTotalFACE=TotalFACE;                %���W�ƫ�Ҧ���Ŷ��V�m�v��

%++++++++++ zero mean ++++++++++++++++++++++++++++++++
for i=1:1:individualsample*people
    for j=1:1:(row)*(col) 
        zeromeanTotalFACE(i,j)=zeromeanTotalFACE(i,j)-TotalMeanFACE(j); %���W��
    end
end
%-----------------------------------------------------

pcaSST=cov(zeromeanTotalFACE);

display('�D�������R')
[PCA,latent,explained] = pcacov(pcaSST);

projectPCA=PCA(:,1:principlenum);                       %���X�D����������

for i=1:1:individualsample*people
    tempFACE=zeromeanTotalFACE(i,:);
    tempFACE=projectPCA'*tempFACE';
    tempFACE=tempFACE';
    pcaTotalFACE=[pcaTotalFACE;tempFACE];               %�x�s�Ҧ���v��PCA�Ŷ������V�m�v��
end

%------- PCA transform ------------------------------------------------------------------------------

SW=zeros(principlenum,principlenum);             %  ��l�� SW �� 0
SB=zeros(principlenum,principlenum);             %  ��l�� SB �� 0
totaldis=0;
samplecount=0;

    %++++++++ NFL SW ++++++++++++++++++++++++++++++++

for i=1:individualsample:individualsample*people
    FACE=pcaTotalFACE(i:i+individualsample-1,:);   %�Ȧs��@���OPCA�Ŷ����V�m�v��    
    
    for k=1:individualsample % �C�@���V�m��ƭn�P����䥦�@���ưt�� n*c(n-1,2), 4*c(3,2)=12
        combination=FACE;
        combination(k,:)=[]; % k �O�@�ӷǳƥΨӧ�v�ܦV�q mn �W���@���I�A����I�q sample ���ް���A�ѤU���N�O�ΨӲզX���u���Կ� sample

        
    %++++++ �b within ���N���@�I�P�䥦�Ҧ��I�������Z���ƧǡA�i�H�קK�L���I�զ����u�P FACE(k,:) �����������Z���Q���(���ئh�� extrapolation)�A���ت��[�W������N���    
        nearest_sw_dis=[]; % �@�� column vector�A�ΨӼȦs FACE(k,:)�Pcombination(?,:) �������Z���H��X�̾F�񪺾F�~�� (�Z���O�ΨӱƧǪ��з�)
        
        for t=1:individualsample-1                              % �h�� k �@���I�A�ѤU�� within �H�y��
            NormalVector=FACE(k,:)-combination(t,:);            % �Ȧs�V�q�A�Ψӭp����ץ�
            tempdis_0=NormalVector*NormalVector';
            nearest_sw_dis=[nearest_sw_dis;tempdis_0];
        end
        
        [junk, index] = sort(nearest_sw_dis);
        combination = combination(index,:);
    %------------------------------------------------------------------------------------------------------------
          
        SWpool=[]; %  ���F�Ȧs�C�@����ƪ��Ҧ��I��u���V�q�A�~��Hneighbor���������neighbor��
        
        for m=1:nearestneighbor_sw-1                            
            OA=FACE(k,:)-combination(m,:);                      % �H m ��@�_�I������v�V�q�A��Ҭ����� a �V�q
            for n=m+1:nearestneighbor_sw                        % m �P n �O�ΨӲզX�����@���u�ҨϥΪ� sample �I
                OB=combination(n,:)-combination(m,:);           % �H m ��@�_�I���Q��v�V�q�A��Ҭ����� b �V�q
                OM=((OA*OB')/(OB*OB'))*OB;                      % ��v�I�����A��Ҭ����� m �V�q
                NormalVector=OA-OM;                             % �I���v�I�������V�q(�k�V�q)
                
                if (OB*OB'>0)                                   % ���b�A�קK������0
  
                    tempdis_0=NormalVector*NormalVector';
    
                    SWvector=zeros(1,1+principlenum);
                    
                    SWvector(1,1)=tempdis_0;
                    SWvector(1,2:1+principlenum)=NormalVector;
                    SWpool=[SWpool;SWvector];
                end
                

            end % end of n
        end % end of m

        
        %-------- �q neighbor ���̧Ǳq�̵u�̿�X within ���V�q�Ӻ�cov----------
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
    FACE=temppcaTotalFACE(i:i+individualsample-1,:);       %�Ȧs��@���OPCA�Ŷ����V�m�v��
    temppcaTotalFACE(i:i+individualsample-1,:)=[];         %�ѤU�䥦�Ҧ����OPCA�Ŷ����V�m�v��
    
    for j=1:1:individualsample
            
        SBcombination=temppcaTotalFACE;

   %++++++ �b beteen ���N���@�I�P�䥦���O�Ҧ��I�������Z���ƧǡA�i�H�קK�L���I�զ����u�P FACE(k,:) �����������Z���Q���(���ئh�� extrapolation)�A���ت��[�W������N���    
        nearest_sb_dis=[]; % �@�� column vector�A�ΨӼȦs FACE(k,:)�PSBcombination(?,:) �������Z���H��X�̾F�񪺾F�~�� (�Z���O�ΨӱƧǪ��з�)
        
        for t=1:individualsample*(people-1)                       % between�䥦���O���H�y��
            NormalVector=FACE(j,:)-SBcombination(t,:);            % �Ȧs�V�q�A�Ψӭp����ץ�
            tempdis_0=NormalVector*NormalVector';
            nearest_sb_dis=[nearest_sb_dis;tempdis_0];
        end
        
        [junk, index] = sort(nearest_sb_dis);
        SBcombination = SBcombination(index,:);
    %------------------------------------------------------------------------------------------------------------              
            
            SBpool=[]; %  ���F�Ȧs�C�@����ƪ��Ҧ��I��u���V�q�A�~��Hneighbor���������neighbor��
            
            for m=1:nearestneighbor_sb-1
                OA=FACE(j,:)-SBcombination(m,:);
                for n=m+1:nearestneighbor_sb
                    OB=SBcombination(n,:)-SBcombination(m,:);
                    OM=((OA*OB')/(OB*OB'))*OB;
                    NormalVector=OA-OM;
                    
                    if (OB*OB'>0)                                        % �Ҧ���NFL
                    
                        SBtempdis_0=NormalVector*NormalVector';
              
                        SBvector=zeros(1,1+principlenum);
                        SBvector(1,1)=SBtempdis_0;
                        SBvector(1,2:1+principlenum)=NormalVector;
                        SBpool=[SBpool;SBvector];
                    end


               end % end of n
           end % end of m

        %-------- �q neighbor ���̧Ǳq�̵u�̿�X within ���V�q�Ӻ�cov----------
        [SBpool,sbindex]=sortrows(SBpool,1);
        for neighborcount=1:1:neighborsb
            SB=SB+SBpool(neighborcount,2:1+principlenum)'*SBpool(neighborcount,2:1+principlenum);
        end
        %----------------------------------------------------------------------         
        sbcount=sbcount+1
    end % end of j
end % end of i
%------------------------------------------------

%[NFLPCA,latent,explained] = pcacov(SB*inv(SW)); �Y�����~
%[NFSEeig,latent] = eig(inv(SW)*SB); �Y�����~

[NFSEeig,latent] = eig(inv(SW)*SB);
NFSEeigvalue=diag(latent);
[junk, index] = sort(-NFSEeigvalue); % sort �O�Ѥp�Ӥj�A�ҥH�[�t��
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