import random
import torch
from torch import nn
from torch import optim
import torch.utils.data as Data
import numpy as np

def get_train_data():
    def get_data(path, g_cnns):
        with open(path, encoding='utf-8') as f:
            datas = f.readlines()
        for each_data in range(len(datas)):
            datas[each_data] = datas[each_data].split('_!_')
            del datas[each_data][0]
            del datas[each_data][1]
            del datas[each_data][-1]
        for i in range(len(datas)):
            datas[i][0] = int(g_cnns[datas[i][0]])
        
        x = []
        y = []
        for data in datas:
            x.append(data[1])
            y.append(data[0])
        return x, y
            
    path="./Datasets/toutiao-text-classfication-dataset-master/toutiao-text-classfication-dataset-master/toutiao_cat_data.txt"

    g_cnns = {
        '100': '0',
        '101': '1',
        '102': '2',
        '103': '3',
        '104': '4',
        '105': '5',
        '106': '6',
        '107': '7',
        '108': '8',
        '109': '9',
        '110': '10',
        '111': '11',
        '112': '12',
        '113': '13',
        '114': '14',
        '115': '15',
        '116': '16'
    }

    x, y = get_data(path, g_cnns)
    y_label = torch.tensor(y[:514], dtype=torch.long)
    
    def get_x(x):
        n = []
        for each_x in x[:700]:
            for each_word in each_x:
                if each_word not in n:
                    n.append(each_word)
        n_len = len(n)
        random.shuffle(n)

        datas = []
        for each_x in range(514):
            s = np.zeros(n_len)
            for each_word in x[each_x]:
                i = n.index(each_word)
                s[i] = 1
            datas.append(s)
            
        datas = torch.tensor(datas, dtype=torch.float)
        return datas, n_len, n

    x_data, word_len, vocal_dict = get_x(x)

    return x_data, y_label, word_len, vocal_dict

class my_net(nn.Module):
    def __init__(self, word_len):
        super(my_net, self).__init__()
        self.word_len = word_len
        self.input_layer = nn.Linear(word_len, self.word_len)
        self.hidden_layer = nn.Linear(word_len, self.word_len)
        self.output_layer = nn.Linear(word_len, 17)
        self.action_1 = nn.ReLU()

    
    def forward(self, x_data):
        self.input_layer_output = self.input_layer(x_data)
        self.input_layer_output = self.action_1(self.input_layer_output)
        self.hidden_layer_output = self.hidden_layer(self.input_layer_output)
        self.hidden_layer_output = self.action_1(self.hidden_layer_output)
        self.output_layer_output = self.output_layer(self.hidden_layer_output)
        return self.output_layer_output

def train_model(train):
    x_data, y_label, word_len, vocal_dict = get_train_data()
    net = my_net(word_len)
    save_path = './model.pkl'
    save_messages = {
        'net':net.state_dict(),
        'word_len':word_len,
        'vocal_dict':vocal_dict
    }
    
    if train:
        optimizer = optim.RMSprop(net.parameters(), lr=0.005)
        loss_function = nn.CrossEntropyLoss()
        databets = Data.TensorDataset(x_data, y_label)
        data_iter = Data.DataLoader(databets, 10, True)

        num_epochs = 10
        for epoch in range(1, num_epochs + 1):
            for X, y in data_iter:
                output = net(X)
                l = loss_function(output, y)
                optimizer.zero_grad()
                l.backward()
                optimizer.step() 
            print('epoch %d, loss: %f' % (epoch, l.item()))
            if l.item() < 0.000002:
                break
        
        torch.save(save_messages, save_path)
    else:
        net.load_state_dict(torch.load(save_path)['net'])
        word_len = torch.load(save_path)['word_len']
        vocal_dict = torch.load(save_path)['vocal_dict']
        
    return net, word_len, vocal_dict

def test_model(net, vocal_dict, test_news, i2s):
    test_news_array = np.zeros(word_len)
    for each in test_news:
        i = vocal_dict.index(each)
        test_news_array[i] = 1
    test_news_array = torch.tensor(test_news_array, dtype=torch.float)
    result = net(test_news_array)

    i = torch.argmax(result).item()
    print('这个标题是 {} 类新闻'.format(i2s[i]))

idx_to_str = {
    0:'民生',
    1:'文化',
    2:'娱乐',
    3:'体育',
    4:'财经',
    5:'时政',
    6:'房产',
    7:'汽车',
    8:'教育',
    9:'科技',
    10:'军事',
    11:'宗教',
    12:'旅游',
    13:'国际',
    14:'证券',
    15:'农业',
    16:'游戏'
}

net, word_len, vocal_dict = train_model(False)
test_news = '王者荣耀最新英雄分析'
test_model(net, vocal_dict, test_news, idx_to_str)

