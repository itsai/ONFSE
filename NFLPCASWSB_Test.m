function [correct,RecognitionRate]=NFLPCASWSB_Test(PCA,NFSEeig,TotalMeanFACE,dimension)

principlenum=300;% PCA維度(注意，只有CMU=15時，才設定為400)

people=68;
withinsample=8;
totalcount=0;
correct=0;

projectPCA=PCA(:,1:principlenum);
projectlaplacian=NFSEeig(:,1:dimension);

FFACE=[];     

 %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

for k=1:1:people
     % !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
     for m=1:1:170                                                  %| m==
         if (m==1 | m==30 | m==59 | m==88 | m==117 | m==146 | m==170 | m==136) % individualsample=7，以的樣本為基準
         matchstring=['cmu3232' '\' num2str(k) '\' num2str(m) '.bmp'];
         matchX=imread(matchstring);
         matchX=double(matchX);
         %if (k==1 & m==1)
         if (k==1 & m==1)    % 取維度用，只有不是從 m=1 開始時需要這行
            [matchrow,matchcol]=size(matchX);
            row=matchrow;
            col=matchcol;
         end         
         
         matchtempF=[];
         %--排成一個 row
         for n=1:matchrow
             matchtempF=[matchtempF,matchX(n,:)];
         end

         matchtempF=matchtempF-TotalMeanFACE; % row
         
         matchtempF=projectPCA'*matchtempF';   % col     

         %matchtempF=matchtempF';              % row
         
         matchtempF=projectlaplacian'*matchtempF; % col
         
         matchtempF=matchtempF';              % row
         
         FFACE=[FFACE;matchtempF];
        end % end of if
     end 
     % !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

 end % end of k=1:1:people
 

 
 %*************************************************************************
 
 for i=1:1:people
   for j=1:1:170
       %if(j~=1 & j~=13 & j~=25 & j~=37 & j~=49 & j~=61 & j~=73 & j~=85 & j~=97 & j~=109 & j~=121 & j~=133 & j~=145 & j~=157 & j~=169)
       %if(j~=1 & j~=16 & j~=31 & j~=46 & j~=61 & j~=76 & j~=91 & j~=106 & j~=121 & j~=136 & j~=151 & j~=166)
       %if(j~=1 & j~=20 & j~=39 & j~=58 & j~=77 & j~=96 & j~=115 & j~=134 & j~=153)
       
       if(j~=1 & j~=30 & j~=59 & j~=88 & j~=117 & j~=146 & j~=170 & j~=136) % individual sample=7
                                                                 %& j~=
       %if(j~=1 & j~=30 & j~=59 & j~=88 & j~=117 & j~=146)
       %if(j~=1 & j~=35 & j~=69 & j~=103 & j~=137)
       
totalcount=totalcount+1;      
       
tempF=[];

s=['cmu3232' '\' num2str(i) '\' num2str(j) '.bmp'];


%---------------------------------------------------------

    test=imread(s);
    [row,col]=size(test);
  
    test=double(test);

    %imshow(test,map)
    
    for k=1:row
        tempF=[tempF,test(k,:)];
    end
    
    tempF=tempF-TotalMeanFACE;

    tempF=projectPCA'*tempF';  % col

    %resultF=resultF';         % row
    
    resultF=projectlaplacian'*tempF; % col
    
    resultF=resultF';         % row

    mindistanceFinal=Inf;   
    
    for k=1:withinsample:people*withinsample

        mindistanceF=Inf;
        
  
     
     % ++++  F  +++++++++  NFL
%     for p=k:(k-1+withinsample-1)
%         OAF=resultF-FFACE(p,:);
%         for q=p+1:(k-1+withinsample)
%             OBF=FFACE(q,:)-FFACE(p,:);
%             OMF=((OAF*OBF')/(OBF*OBF'))*OBF;
%             NormalVector=OAF-OMF;
%             Eucdis=NormalVector*NormalVector';
%             if (Eucdis < mindistanceF)
%                 mindistanceF=Eucdis;
%             end
%         end
%     end
     % ----  F  ---------     
     
     % ++++  F  +++++++++ NN
     for p=k:(k-1+withinsample)
         OAF=resultF-FFACE(p,:);
             Eucdis=OAF*OAF';
             if (Eucdis < mindistanceF)
                 mindistanceF=Eucdis;
             end
     end
     % ----  F  ---------     
       
      Eucdisfinal=mindistanceF;
      
      if (Eucdisfinal < mindistanceFinal)
         mindistanceFinal=Eucdisfinal;
         ID=floor(k/withinsample)+1;    
      end
      
    end % end of k=1:15:people*winthinsample      
    
    if (i==ID)
       correct=correct+1;
       ID;    
    end
RecognitionRate=correct/totalcount

end % if

end % 資料
end % 類別