import matplotlib.pyplot as plt
from math import log
from collections import Counter
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from matplotlib.font_manager import FontProperties


# 属性值唯一值
def dataProcess(filename, output_file):
    # 读取西瓜数据集文件
    data = pd.read_csv(filename)
    # 删除第一列（序号列）
    data = data.iloc[:, 1:]

    # 使用LabelEncoder将每个属性值转换为唯一标签
    label_encoders = {}
    original_attribute_values = {}  # 存储原始属性值
    transformed_attribute_values = {}  # 存储转换后的属性值
    for column in data.columns[:-3]:  # 最后一列是目标变量，不需要转换，最后三列不用转换
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])
        # 存储原始属性值
        original_attribute_values[column] = sorted(data[column].unique())
        # 存储转换后的属性值
        transformed_attribute_values[column] = sorted(label_encoders[column].classes_)

    # 将目标变量中的"是"改为1，"否"改为0
    data['好瓜'] = data['好瓜'].map({'是': 1, '否': 0})

    # 将数据集划分为训练集和测试集
    x = data.iloc[:, :-1]  # 特征
    y = data.iloc[:, -1]   # 目标变量
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=21)

    # 将训练集和测试集写入文件
    train_data = pd.concat([x_train, y_train], axis=1)  # 将特征和目标变量合并为训练集
    test_data = pd.concat([x_test, y_test], axis=1)  # 将特征和目标变量合并为测试集

    train_data.to_csv("train_data.csv", index=False)  # 将训练集写入到CSV文件，不保存索引
    test_data.to_csv("test_data.csv", index=False)  # 将测试集写入到CSV文件，不保存索引

    # 写入原始属性值和转换后的属性值到文件
    with open(output_file, 'w') as file:
        for column in data.columns[:-3]:  # 最后一列是目标变量，不需要写入
            file.write(f"{column}:\n")
            val_number = 0
            for orig_val, trans_val in zip(original_attribute_values[column], transformed_attribute_values[column]):
                if val_number<4:
                    file.write(f"[{orig_val:2},{trans_val:4}]  ")
                    val_number+=1
                else:
                    file.write("\n")
                    val_number = 0
            file.write("\n")
        # 写入好瓜和坏瓜对应的数值
        file.write("好瓜:\n")
        file.write("[0,否] [1,是]")
    print(f"数据集转换前后对应关系保存到{output_file}")


# 读取数据集文件,读取数据集内容和属性值
def read_dataset(filename):
    data = pd.read_csv(filename)
    labels = data.columns[:-1].tolist()  # 特征列名
    dataset = data.values.tolist()  # 数据集
    return dataset, labels


# 排序
def sort(data):
    for i in range(len(data)-1):
        for j in range(i+1,len(data)):
            if data[i] > data[j]:
                temp = data[i]
                data[i] = data[j]
                data[j] = temp
    return data


# 读取连续值，并返回阈值数据
def lianxu(dataset, axis):
    data_num = []
    for i in range(len(dataset)):
        data = dataset[i][axis]
        data_num.append(data)
    data_num = sort(data_num)
    data_ta = []
    for j in range(0,len(data_num)-1):
        num = (data_num[j] + data_num[j+1])*1.0 / 2
        num_rounded = round(num, 3)  # 将 num 保留三位小数
        data_ta.append(num_rounded)
    return data_ta


# 得到经过连续值阈值二分后的数据集
def doudata(dataset, axis, num):
    subdataset1 = []
    subdataset2 = []
    for featVec in dataset:
        if featVec[axis] <= num:
            subdataset1.append(featVec)
        else:
            subdataset2.append(featVec)
    return subdataset1, subdataset2


# 得到最优划分阈值
def yuzhi(dataset, axis):
    # 计算熵和二分后的熵得到最优熵和最优划分阈值
    baseEnt = cal_entropy(dataset)
    bestInfoGain = 0.0  # 最好的信息增益
    bestTa = 0.0  # 最好的连续值阈值
    # infoGain_ratio = 0.0
    t = lianxu(dataset, axis)
    for i in range(len(t)):
        newEnt = 0.0
        infogain = 0.0
        # IV = 0.0
        subdataset1,subdataset2 = doudata(dataset, axis, t[i])
        p1 = len(subdataset1) / float(len(dataset))
        p2 = len(subdataset2) / float(len(dataset))
        newEnt = p1 * cal_entropy(subdataset1) + p2 * cal_entropy(subdataset2)
        infogain = baseEnt - newEnt
        # IV = - p1 * log(p1, 2) - p2 * log(p2, 2)
        if infogain > bestInfoGain:
            bestInfoGain = infogain
            bestTa = t[i]
            # if IV == 0 :
            #     infoGain_ratio = 0.0
            #     continue
            # infoGain_ratio = bestInfoGain / IV
    return bestInfoGain, bestTa


# 计算熵
def cal_entropy(dataset):
    numEntries = len(dataset)   # 数据集中条目的总数
    labelCounts = Counter(entry[-1] for entry in dataset)  # 计算每个类别的数量
    shannonEnt = 0.0   # 初始化熵为0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries    # 计算每个类别的概率
        shannonEnt -= prob * log(prob, 2)    # 计算熵
    return shannonEnt

# 划分数据集
def splitdataset(dataset, axis, value):
    retDataset = []   # 初始化划分后的数据子集列表
    for featVec in dataset:
        if featVec[axis] == value:   # 如果当前数据条目的特征值与给定特征值相同
            # 去除当前特征值，并将剩余特征值添加到划分后的数据子集中
            reducedFeatVec = featVec[:axis] + featVec[axis+1:]
            retDataset.append(reducedFeatVec)
    return retDataset



# 选择最好的划分特征，标准：信息增益
def chooseBestFeatureToSplit(dataset, labels):
    numFeatures = len(dataset[0]) - 1   # 特征数量，除了密度和含糖率
    baseEnt = cal_entropy(dataset)   # 基础熵，原数据集的熵
    bestInfoGain = 0.0   # 最好的信息增益
    bestFeatureIndex = -1  # 添加一个变量来存储最优特征的索引
    ta = -1
    midu_index = labels.index('密度')
    tang_index = labels.index('含糖率')
    for i in range(numFeatures):
        if i == midu_index:
            infoGain, midu_ta = yuzhi(dataset, midu_index)
        elif i == tang_index:
            infoGain, tang_ta = yuzhi(dataset, tang_index)
        else:
            featList = [example[i] for example in dataset]
            uniqueVals = set(featList)
            newEnt = 0.0
            # 计算根据当前特征划分后的熵和TV
            for value in uniqueVals:
                subdataset = splitdataset(dataset, i, value)
                p = len(subdataset) / float(len(dataset))
                newEnt += p * cal_entropy(subdataset)
            infoGain = baseEnt - newEnt
        print(u"ID3中第%d个特征的信息增益为：%.3f" % ( i, infoGain))
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeatureIndex = i  # 使用属性索引而不是标签
            if i == midu_index:
                ta = midu_ta
            elif i == tang_index:
                ta = tang_ta
            else:
                ta = -1
    return bestFeatureIndex, ta


# 统计出现次数最多的类别
def majorityCnt(classList):
    classCount = Counter(classList)
    sortedClassCount = sorted(classCount.items(), key = lambda x: x[1], reverse=True)
    return sortedClassCount[0][0]


# 构建决策树
def createTree(dataset, labels):
    classList = [example[-1] for example in dataset]
    if classList.count(classList[0]) == len(classList):
        return int(classList[0])
    if len(dataset[0]) == 1:
        return int(majorityCnt(classList))
    bestFeatIndex, ta= chooseBestFeatureToSplit(dataset, labels)
    bestFeatLabel = labels[bestFeatIndex]
    print(u"此时最优索引为：" + str(bestFeatIndex))
    if ta == -1:
        Tree = {bestFeatLabel: {}}
        featValues = set([example[bestFeatIndex] for example in dataset])
        for value in featValues:
            subLabels = labels[:]  # 注意这里需要创建副本，以免影响原始列表
            subDataset = splitdataset(dataset, bestFeatIndex, value)
            subLabels.remove(bestFeatLabel)  # 移除已经选择的特征标签
            Tree[bestFeatLabel][int(value)] = createTree(subDataset, subLabels)
    else:
        Tree = {bestFeatLabel: {}}
        featValues = set([example[bestFeatIndex] for example in dataset])
        for value in featValues:
            subLabels = labels[:]  # 注意这里需要创建副本，以免影响原始列表
            subDataset1,subDataset2 = doudata(dataset, bestFeatIndex, ta)
            if value <= ta :
                Tree[bestFeatLabel][value] = createTree(subDataset1, subLabels)
            else:
                Tree[bestFeatLabel][value] = createTree(subDataset2, subLabels)

    return Tree


# 获取决策树叶子结点的数目
def getNumLeafs(tree):
    numLeafs = 0
    firstStr = next(iter(tree))
    secondDict = tree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs



# 获取决策树的层数
def getTreeDepth(tree):
    maxDepth = 0
    firstStr = next(iter(tree))
    secondDict = tree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == "dict":
            thisDepth = getTreeDepth(secondDict[key]) + 1
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


# 绘制结点
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    arrow_args = dict(arrowstyle="<-")
    font = FontProperties(fname="C:\Windows\Fonts\msyh.ttc", size = 14)
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                              xytext = centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args,fontproperties=font)


# 标注有向边属性值
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid,txtString, va = "center", ha = "center", rotation = 30)


# 决策树可视化（这部分需要根据具体的绘图方法进行修改）
def plotTree(myTree, parentPt, nodeTxt):
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")  # 设置结点格式
    leafNode = dict(boxstyle="round4", fc="0.8")  # 设置叶结点格式
    numLeafs = getNumLeafs(myTree)  # 获取决策树叶结点数目，决定了树的宽度
    depth = getTreeDepth(myTree)  # 获取决策树层数
    firstStr = next(iter(myTree))  # 下个字典
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)  # 中心位置
    plotMidText(cntrPt, parentPt, nodeTxt)  # 标注有向边属性值
    plotNode(firstStr, cntrPt, parentPt, decisionNode)  # 绘制结点
    secondDict = myTree[firstStr]  # 下一个字典，也就是继续绘制子结点
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD  # y偏移
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  # 测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            plotTree(secondDict[key], cntrPt, str(key))  # 不是叶结点，递归调用继续绘制
        else:  # 如果是叶结点，绘制叶结点，并标注有向边属性值
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD



# 创建绘制面板
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')                                                    #创建fig
    fig.clf()                                                                                #清空fig
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)                                #去掉x、y轴
    plotTree.totalW = float(getNumLeafs(inTree))                                            #获取决策树叶结点数目
    plotTree.totalD = float(getTreeDepth(inTree))                                            #获取决策树层数
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;                                  #x偏移
    plotTree(inTree, (0.5, 1.0), '')                                                            #绘制决策树
    plt.show()


# 检验决策树的准确率
def classify(inputTree, Labels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = Labels.index(firstStr)
    Label = None  # 将Label声明为局部变量
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if isinstance(secondDict[key], dict):
                Label = classify(secondDict[key], Labels, testVec)
            else:
                Label = secondDict[key]
    return Label



def accuracy(tree, labels, test_data):
    error_count = 0
    for test_vec in test_data:
        if test_vec[-1] != classify(tree, labels, test_vec):
            error_count += 1
    return 1 - (error_count / len(test_data))


if __name__ == '__main__':
    # 读取训练集和测试集
    dataProcess("xigua_data3.0.csv", "description.txt")

    train_data, labels_train = read_dataset("train_data.csv")
    test_data, labels_test = read_dataset("test_data.csv")

    # 构建决策树
    Tree = createTree(train_data, labels_train)

    print(Tree)

    # 计算并输出准确率
    accuracy_score = accuracy(Tree, labels_train, test_data)
    print("决策树准确率：", accuracy_score)

    # 可视化决策树
    createPlot(Tree)




