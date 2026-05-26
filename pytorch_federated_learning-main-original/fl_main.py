#!/usr/bin/env python
import os
import random
import json
import pickle
import argparse
import yaml
from json import JSONEncoder
from tqdm import tqdm

# 导入联邦学习算法相关的客户端和服务类
from fed_baselines.client_base import FedClient
from fed_baselines.client_fedprox import FedProxClient
from fed_baselines.client_scaffold import ScaffoldClient
from fed_baselines.client_fednova import FedNovaClient
from fed_baselines.server_base import FedServer
from fed_baselines.server_scaffold import ScaffoldServer
from fed_baselines.server_fednova import FedNovaServer

# 导入后处理和预处理模块和工具类
from postprocessing.recorder import Recorder
from preprocessing.baselines_dataloader import divide_data
from utils.models import *

# JSON数据中允许的几种基本类型
json_types = (list, dict, str, int, float, bool, type(None))


# 将python对象序列化
class PythonObjectEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, json_types):
            return super().default(self, obj)
        return {'_python_object': pickle.dumps(obj).decode('latin-1')}    # 不是普通JSON类型，序列化成二进制字节串，包装成一个字典{‘_python_object':……}。


# 将序列化的特殊对象解码回原始对象，恢复成原来的python实例
def as_python_object(dct):
    if '_python_object' in dct:
        return pickle.loads(dct['_python_object'].encode('latin-1'))
    return dct            # 普通JSON照常返回


# 解析命令行参数，获取配置文件路径
def fed_args():
    """
    Arguments for running federated learning baselines
    :return: Arguments for federated learning baselines
    """
    parser = argparse.ArgumentParser()      # 定义和解析命令行参数

    parser.add_argument('--config', type=str, required=True, help='Yaml file for configuration')  # 获取参数配置文件

    args = parser.parse_args()  # 解析命令行参数，将解析后的结果（一个包含属性config的对象）赋给args
    return args
    

# 主函数：联邦学习流程
def fed_run():
    """
    Main function for the baselines of federated learning
    """
    args = fed_args()    # 获取命令行参数：配置文件地址
    with open(args.config, "r", encoding='utf-8') as yaml_file:   # 打开配置文件
        try:
            config = yaml.safe_load(yaml_file)   # 解析内容为字典config
        except yaml.YAMLError as exc:
            print(exc)   # 输出异常

    # 定义支持的联邦算法列表，校验配置文件中客户端使用的算法在支持列表中，否则输出错误信息
    algo_list = ["FedAvg", "SCAFFOLD", "FedProx", "FedNova"]
    assert config["client"]["fed_algo"] in algo_list, "The federated learning algorithm is not supported"

    # 定义支持的数据集，校验配置文件中的数据集是否在支持列表中
    dataset_list = ['MNIST', 'CIFAR10', 'FashionMNIST', 'SVHN', 'CIFAR100']
    assert config["system"]["dataset"] in dataset_list, "The dataset is not supported"

    # 支持的模型
    model_list = ["LeNet", 'AlexCifarNet', "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "CNN"]
    assert config["system"]["model"] in model_list, "The model is not supported"

    # 设置随机种子，保证实验可复现性
    np.random.seed(config["system"]["i_seed"])
    torch.manual_seed(config["system"]["i_seed"])
    random.seed(config["system"]["i_seed"])

    # 定义客户端对象，训练内容对象
    client_dict = {}
    recorder = Recorder()

    # 定义训练集和测试集内容，按照客户端数量，每个客户端数据集类别数，数据集名称和随机种子进行划分
    trainset_config, testset = divide_data(num_client=config["system"]["num_client"], num_local_class=config["system"]["num_local_class"], dataset_name=config["system"]["dataset"],
                                           i_seed=config["system"]["i_seed"])
    max_acc = 0   # 用于记录训练过程中的最高测试准确率
    # 创建对应客户端对象，并为客户端加载训练数据
    # Initialize the clients w.r.t. the federated learning algorithms and the specific federated settings
    for client_id in trainset_config['users']:
        if config["client"]["fed_algo"] == 'FedAvg':
            client_dict[client_id] = FedClient(client_id, dataset_id=config["system"]["dataset"], epoch=config["client"]["num_local_epoch"], model_name=config["system"]["model"])
        elif config["client"]["fed_algo"] == 'SCAFFOLD':
            client_dict[client_id] = ScaffoldClient(client_id, dataset_id=config["system"]["dataset"], epoch=config["client"]["num_local_epoch"], model_name=config["system"]["model"])
        elif config["client"]["fed_algo"] == 'FedProx':
            client_dict[client_id] = FedProxClient(client_id, dataset_id=config["system"]["dataset"], epoch=config["client"]["num_local_epoch"], model_name=config["system"]["model"])
        elif config["client"]["fed_algo"] == 'FedNova':
            client_dict[client_id] = FedNovaClient(client_id, dataset_id=config["system"]["dataset"], epoch=config["client"]["num_local_epoch"], model_name=config["system"]["model"])
        client_dict[client_id].load_trainset(trainset_config['user_data'][client_id])

    # 创建服务器，并加载测试集
    # Initialize the clients w.r.t. the federated learning algorithms and the specific federated settings
    if config["client"]["fed_algo"] == 'FedAvg':
        fed_server = FedServer(trainset_config['users'], dataset_id=config["system"]["dataset"], model_name=config["system"]["model"])
    elif config["client"]["fed_algo"] == 'SCAFFOLD':
        fed_server = ScaffoldServer(trainset_config['users'], dataset_id=config["system"]["dataset"], model_name=config["system"]["model"])
        scv_state = fed_server.scv.state_dict()
    elif config["client"]["fed_algo"] == 'FedProx':
        fed_server = FedServer(trainset_config['users'], dataset_id=config["system"]["dataset"], model_name=config["system"]["model"])
    elif config["client"]["fed_algo"] == 'FedNova':
        fed_server = FedNovaServer(trainset_config['users'], dataset_id=config["system"]["dataset"], model_name=config["system"]["model"])
    fed_server.load_testset(testset)
    global_state_dict = fed_server.state_dict()

    # Main process of federated learning in multiple communication rounds

    # 创建进度条，显示训练轮数
    pbar = tqdm(range(config["system"]["num_round"]))
    for global_round in pbar:     # 遍历每一轮联邦学习
        for client_id in trainset_config['users']:    # 在每一轮中遍历所有客户端
            # Local training 本地训练，客户端更新模型、训练、服务器接收结果
            if config["client"]["fed_algo"] == 'FedAvg':
                client_dict[client_id].update(global_state_dict)
                state_dict, n_data, loss = client_dict[client_id].train()
                fed_server.rec(client_dict[client_id].name, state_dict, n_data, loss)
            elif config["client"]["fed_algo"] == 'SCAFFOLD':
                client_dict[client_id].update(global_state_dict, scv_state)
                state_dict, n_data, loss, delta_ccv_state = client_dict[client_id].train()
                fed_server.rec(client_dict[client_id].name, state_dict, n_data, loss, delta_ccv_state)
            elif config["client"]["fed_algo"] == 'FedProx':
                client_dict[client_id].update(global_state_dict)
                state_dict, n_data, loss = client_dict[client_id].train()
                fed_server.rec(client_dict[client_id].name, state_dict, n_data, loss)
            elif config["client"]["fed_algo"] == 'FedNova':
                client_dict[client_id].update(global_state_dict)
                state_dict, n_data, loss, coeff, norm_grad = client_dict[client_id].train()
                fed_server.rec(client_dict[client_id].name, state_dict, n_data, loss, coeff, norm_grad)

        # 服务器选择要聚合的客户端
        # Global aggregation
        fed_server.select_clients()
        if config["client"]["fed_algo"] == 'FedAvg':
            global_state_dict, avg_loss, _ = fed_server.agg()
        elif config["client"]["fed_algo"] == 'SCAFFOLD':
            global_state_dict, avg_loss, _, scv_state = fed_server.agg()  # scarffold
        elif config["client"]["fed_algo"] == 'FedProx':
            global_state_dict, avg_loss, _ = fed_server.agg()
        elif config["client"]["fed_algo"] == 'FedNova':
            global_state_dict, avg_loss, _ = fed_server.agg()

        # Testing and flushing
        # 服务器测试全局模型准确率
        accuracy = fed_server.test()
        fed_server.flush()

        # Record the results 记录准确率和训练损失
        recorder.res['server']['iid_accuracy'].append(accuracy)
        recorder.res['server']['train_loss'].append(avg_loss)

        # 更新最大准确率
        if max_acc < accuracy:
            max_acc = accuracy

        # 更新进度条描述，显示当前轮数、损失、准确率和最大准确率
        pbar.set_description(
            'Global Round: %d' % global_round +
            '| Train loss: %.4f ' % avg_loss +
            '| Accuracy: %.4f' % accuracy +
            '| Max Acc: %.4f' % max_acc)

        # Save the results 保存结果，如果结果目录不存在，则创建目录
        if not os.path.exists(config["system"]["res_root"]):
            os.makedirs(config["system"]["res_root"])

        # 将记录的服务器端的准确率和训练损失记录到以[算法名，模型名，本地训练轮数，每个客户端数据类别数，随机种子]命名的文件中
        with open(os.path.join(config["system"]["res_root"], '[\'%s\',' % config["client"]["fed_algo"] +
                                        '\'%s\',' % config["system"]["model"] +
                                        str(config["client"]["num_local_epoch"]) + ',' +
                                        str(config["system"]["num_local_class"]) + ',' +
                                        str(config["system"]["i_seed"])) + ']', "w") as jsfile:
            json.dump(recorder.res, jsfile, cls=PythonObjectEncoder)


# 程序入口
if __name__ == "__main__":
    fed_run()
