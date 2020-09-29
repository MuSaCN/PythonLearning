# Author:Zhang Yuan
from MyPackage import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

#------------------------------------------------------------
__mypath__ = MyPath.MyClass_Path("\\Python神经网络编程")  # 路径类
myfile = MyFile.MyClass_File()  # 文件操作类
mytime = MyTime.MyClass_Time()  # 时间类
myplt = MyPlot.MyClass_Plot()  # 直接绘图类(单个图窗)
mypltpro = MyPlot.MyClass_PlotPro()  # Plot高级图系列
myfig = MyPlot.MyClass_Figure(AddFigure=False)  # 对象式绘图类(可多个图窗)
myfigpro = MyPlot.MyClass_FigurePro(AddFigure=False)  # Figure高级图系列
mynp = MyArray.MyClass_NumPy()  # 多维数组类(整合Numpy)
mypd = MyArray.MyClass_Pandas()  # 矩阵数组类(整合Pandas)
mypdpro = MyArray.MyClass_PandasPro()  # 高级矩阵数组类
myDA = MyDataAnalysis.MyClass_DataAnalysis()  # 数据分析类
# myMql = MyMql.MyClass_MqlBackups() # Mql备份类
# myMT5 = MyMql.MyClass_ConnectMT5(connect=False) # Python链接MetaTrader5客户端类
# myDefault = MyDefault.MyClass_Default_Matplotlib() # matplotlib默认设置
# myBaidu = MyWebCrawler.MyClass_BaiduPan() # Baidu网盘交互类
myWebQD = MyWebCrawler.MyClass_WebQuotesDownload()  # 金融行情下载类
myBT = MyBackTest.MyClass_BackTestEvent()  # 事件驱动型回测类
myBTV = MyBackTest.MyClass_BackTestVector()  # 向量型回测类
#------------------------------------------------------------


# 自定义 BP Neural Network
class My_BPNeuralNetwork:

    # ---initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate = 0.1):
        # ---引入激活函数库
        from scipy import special # 引入特定函数类，scipy.special for the sigmoid function expit()
        # set number of nodes in each input, hidden, output layer
        self.i_nodes = inputnodes
        self.h_nodes = hiddennodes
        self.o_nodes = outputnodes
        # 设置权重矩阵，为计算方便矩阵元素 w[i,j] 表示 j to i；# 以正态分布(0, 1/pow(n,0.5))随机抽样的方式初始化权重
        self.w_ih = np.random.normal(0.0, pow(self.i_nodes, -0.5), (self.h_nodes, self.i_nodes))
        self.w_ho = np.random.normal(0.0, pow(self.h_nodes, -0.5), (self.o_nodes, self.h_nodes))
        # learning rate
        self.lrate = learningrate
        # activation function is the sigmoid function
        self.activation_function = lambda x: special.expit(x)

        pass

    # ---train the neural network，BP算法，调整权重
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        # ---first calculate
        # calculate hidden layer
        hidden_inputs = np.dot(self.w_ih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        # calculate final output layer
        final_inputs = np.dot(self.w_ho, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        # ---calculate error，输出层直接减，隐藏层根据权重计算
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.w_ho.T, output_errors)
        # ---update weights
        self.w_ho += self.lrate * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        np.transpose(hidden_outputs))
        self.w_ih += self.lrate * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        np.transpose(inputs))

        pass

    # ---query查询下网络，输入输入层list，输出输出层。
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.w_ih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        # calculate signals into final output layer
        final_inputs = np.dot(self.w_ho, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        return final_outputs


# number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1

# create instance of neural network
n = My_BPNeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

filepath = __mypath__.current_workpath()+"\\mnist_dataset\\mnist_train_100.csv"

# load the mnist training data CSV file into a list
training_data_file = open(filepath, 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# train the neural network
times = 5
for e in range(times):
    # go through all records in the training data set
    for record in training_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # scale and shift the inputs
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = np.zeros(output_nodes) + 0.01
        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass

# %%
filepath1 = __mypath__.current_workpath() + "\\mnist_dataset\\mnist_test_10.csv"
# load the mnist test data CSV file into a list
test_data_file = open(filepath1, 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# %%

# test the neural network

# scorecard for how well the network performs, initially empty
scorecard = []

# go through all the records in the test data set
for record in test_data_list:
    # split the record by the ',' commas
    all_values = record.split(',')
    # correct answer is first value
    correct_label = int(all_values[0])
    # scale and shift the inputs
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # query the network
    outputs = n.query(inputs)
    # the index of the highest value corresponds to the label
    label = np.argmax(outputs)
    # append correct or incorrect to list
    if (label == correct_label):
        # network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
    else:
        # network's answer doesn't match correct answer, add 0 to scorecard
        scorecard.append(0)
        pass

    pass

# %%

# calculate the performance score, the fraction of correct answers
scorecard_array = np.asarray(scorecard)
print("performance = ", scorecard_array.sum() / scorecard_array.size)




