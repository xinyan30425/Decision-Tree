%First add the attribute label in the csv file, then import it to
%matlab,and seperate workspace variables in to seven 728x1 tables: buying,maint,doors,persons,lug_boot,safety,and label

%Replace attributes index to numerical values for all attributes

% buyingType=struct('high',1,'vhigh',2,'low',3,'med',4);
% maintType=struct('high',1,'vhigh',2,'low',3,'med',4);
% doorsType=struct('2',1,'3',2,'4',3,'5more',4);
% personsType={'2',1,'3',2,'4',3,'5more',4};
% lug_bootType={'big',1,'med',2,'small',3};
% safetyType={'big',1,'med',2,'small',3};
% labelType=struct('acc',1,'good',2,'uncc',3,'vgood',4);


buying=table(buying);
buying.buying=findgroups(buying.buying);
buying.buying(strcmpi(buying.buying,'high')) = {1}; 
buying.buying(strcmpi(buying.buying,'vhigh')) = {2}; 
buying.buying(strcmpi(buying.buying,'low')) = {3}; 
buying.buying(strcmpi(buying.buying,'med')) = {4}; 
 
maint=table(maint);
maint.maint=findgroups(maint.maint);
maint.maint(strcmpi(maint.maint,'high')) = {1}; 
maint.maint(strcmpi(maint.maint,'vhigh')) = {2}; 
maint.maint(strcmpi(maint.maint,'low')) = {3}; 
maint.maint(strcmpi(maint.maint,'med')) = {4}; 
 
doors=table(doors);
doors.doors=findgroups(doors.doors);
doors.doors(strcmpi(doors.doors,'2')) = {1}; 
doors.doors(strcmpi(doors.doors,'3')) = {2}; 
doors.doors(strcmpi(doors.doors,'4')) = {3}; 
doors.doors(strcmpi(doors.doors,'5more')) = {4}; 
 
persons=table(persons);
persons.persons=findgroups(persons.persons);
persons.persons(strcmpi(persons.persons,'2')) = {1}; 
persons.persons(strcmpi(persons.persons,'4')) = {2}; 
persons.persons(strcmpi(persons.persons,'5more')) = {3};
 
lug_boot=table(lug_boot);
lug_boot.lug_boot=findgroups(lug_boot.lug_boot);
lug_boot.lug_boot(strcmpi(lug_boot.lug_boot,'big')) = {1}; 
lug_boot.lug_boot(strcmpi(lug_boot.lug_boot,'med')) = {2}; 
lug_boot.lug_boot(strcmpi(lug_boot.lug_boot,'small')) = {3}; 
 
 
safety=table(safety);
safety.safety=findgroups(safety.safety);
safety.safety(strcmpi(safety.safety,'high')) = {1}; 
safety.safety(strcmpi(safety.safety,'med')) = {2}; 
safety.safety(strcmpi(safety.safety,'low')) = {3}; 
 
label=table(label);
label.label=findgroups(label.label);
label.label(strcmpi(label.label,'acc')) = {1}; 
label.label(strcmpi(label.label,'good')) = {2}; 
label.label(strcmpi(label.label,'unacc')) = {3}; 
label.label(strcmpi(label.label,'vgood')) = {4}; 

%combine seven reconstructed tables into one table


Data=[buying,maint,doors,persons,lug_boot,safety,label];

%remove Nan and nondefined values 
Clean = rmmissing(Data);

%Seperate the Data table variables agian:seven 354x1 tables:
%buying1,maint1,doors1,persons1,lug_boot1,safety1,and label1
%and convert each seperated table to matrix 

buying1=table(buying1);
maint1=table(maint1);
doors1=table(doors1);
persons1=table(persons1);
lug_boot1=table(lug_boot1);
safety1=table(safety1);
label1=table(label1);

Buying1=buying1{:,1};
Maint1=maint1{:,1};
Doors1=doors1{:,1};
Persons1=persons1{:,1};
Lug_boot1=lug_boot1{:,1};
Safety1=safety1{:,1};
Target1=label1{:,1};

%defiine test data:DATA
DATA=[Buying1 Maint1 Doors1 Persons1 Lug_boot1 Safety1];

propertyName={'Buying1','Maint1','Doors1','Persons1','Lug_boot1','Safety1'};
delta=0.1;
decisionTreeModel=decisiontree(DATA,Target1,propertyName,delta);

label=decisionTreeTest(decisionTreeModel,DATA,propertyName);

%build decision tree training model
function decisionTreeModel=decisiontree(data,label,propertyName,delta)
 
global Node;
 
Node=struct('level',-1,'fatherNodeName',[],'EdgeProperty',[],'NodeName',[]);
BuildTree(-1,'root','Stem',data,label,propertyName,delta);
Node(1)=[];
model.Node=Node;
decisionTreeModel=model;
end

%Recursively build the decision tree
function BuildTree(fatherlevel,fatherNodeName,edge,data,label,propertyName,delta)
 
global Node;
sonNode=struct('level',0,'fatherNodeName',[],'EdgeProperty',[],'NodeName',[]);
sonNode.level=fatherlevel+1;
sonNode.fatherNodeName=fatherNodeName;
sonNode.EdgeProperty=edge;
if length(unique(label))==1
    sonNode.NodeName=label(1);
    Node=[Node sonNode];
    return;
end
if length(propertyName)<1
    labelSet=unique(label);
    k=length(labelSet);
    labelNum=zeros(k,1);
    for i=1:k
        labelNum(i)=length(find(label==labelSet(i)));
    end
    [~,labelIndex]=max(labelNum);
    sonNode.NodeName=labelSet(labelIndex);
    Node=[Node sonNode];
    return;
end
[sonIndex,BuildNode]=CalcuteNode(data,label,delta);
if BuildNode
    dataRowIndex=setdiff(1:length(propertyName),sonIndex);
    sonNode.NodeName=propertyName{sonIndex};
    Node=[Node sonNode];
    propertyName(sonIndex)=[];
    sonData=data(:,sonIndex);
    sonEdge=unique(sonData);
 
    for i=1:length(sonEdge)
        edgeDataIndex=find(sonData==sonEdge(i));
        BuildTree(sonNode.level,sonNode.NodeName,sonEdge(i),data(edgeDataIndex,dataRowIndex),label(edgeDataIndex,:),propertyName,delta);
    end
else
    labelSet=unique(label);
    k=length(labelSet);
    labelNum=zeros(k,1);
    for i=1:k
        labelNum(i)=length(find(label==labelSet(i)));
    end
    [~,labelIndex]=max(labelNum);
    sonNode.NodeName=labelSet(labelIndex);
    Node=[Node sonNode];
    return;
end
end

%Calculate entropy
function [NodeIndex,BuildNode]=CalcuteNode(DATA,label,delta)
 
LargeEntropy=CEntropy(label);
[m,n]=size(DATA);
EntropyGain=LargeEntropy*ones(1,n);
BuildNode=true;
for i=1:n
    pData=DATA(:,i);
    itemList=unique(pData);
    for j=1:length(itemList)
        itemIndex=find(pData==itemList(j));
        EntropyGain(i)=EntropyGain(i)-length(itemIndex)/m*CEntropy(label(itemIndex));
    end

end
[maxGainEntropy,NodeIndex]=max(EntropyGain);
if maxGainEntropy<delta
    BuildNode=false;
end
end
function result=CEntropy(propertyList)
 
result=0;
totalLength=length(propertyList);
itemList=unique(propertyList);
pNum=length(itemList);
for i=1:pNum
    itemLength=length(find(propertyList==itemList(i)));
    pItem=itemLength/totalLength;
    result=result-pItem*log2(pItem);
end
end
function label=decisionTreeTest(decisionTreeModel,sampleSet,propertyName)
 
lengthSample=size(sampleSet,1);
label=zeros(lengthSample,1);
for sampleIndex=1:lengthSample
    sample=sampleSet(sampleIndex,:);
    Nodes=decisionTreeModel.Node;
    rootNode=Nodes(1);
    head=rootNode.NodeName;
    index=GetFeatureNum(propertyName,head);
    edge=sample(index);
    k=1;
    level=1;
    while k<length(Nodes)
        k=k+1;
        if Nodes(k).level==level
            if strcmp(Nodes(k).fatherNodeName,head)
                if Nodes(k).EdgeProperty==edge
                    if Nodes(k).NodeName<10
                        label(sampleIndex)=Nodes(k).NodeName;
                        break;
                    else
                        head=Nodes(k).NodeName;
                        index=GetFeatureNum(propertyName,head);
                        edge=sample(index);
                        level=level+1;
                    end
                end
            end
        end
    end
end
end

function result=GetFeatureNum(propertyName,str)
result=0;
for i=1:length(propertyName)
    if strcmp(propertyName{i},str)==1
        result=i;
        break;
    end
end
end


