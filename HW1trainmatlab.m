
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

