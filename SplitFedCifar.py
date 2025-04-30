#============================================================================
# SplitfedV1 (SFLV1) learning: ResNet18 on HAM10000
# HAM10000 dataset: Tschandl, P.: The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions (2018), doi:10.7910/DVN/DBW86T

# We have three versions of our implementations
# Version1: without using socket and no DP+PixelDP
# Version2: with using socket but no DP+PixelDP
# Version3: without using socket but with DP+PixelDP

# This program is Version1: Single program simulation 
# ============================================================================
import torch
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import math
import os.path
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from glob import glob
from pandas import DataFrame
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict

import random
import numpy as np
import os


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy


SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    print(torch.cuda.get_device_name(0))    

#===================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# To print in color -------test/train of the client side
def prRed(skk): print("\033[91m {}\033[00m" .format(skk)) 
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))     

#===================================================================
# No. of users
num_users = 5
epochs = 200
frac = 1        # participation of clients; if 1 then 100% clients participate in SFLV1
lr = 0.0001
S = np.zeros((num_users, num_users))  # Cosine similarity matrix
fx_collect = []                 # activation embedding

def embed_fx(fx_batch: torch.Tensor) -> torch.Tensor:
    """
    fx_batch: (B, C, H, W) or (B, F) ...
    返回: (C*H*W,)  先对 batch 求平均，再展平
    """
    with torch.no_grad():
        v = fx_batch.mean(dim=0).view(-1)   # (C*H*W,)
        return v / (v.norm() + 1e-8)        # 做一次 L2-norm，方便后面直接点积得到余弦

def cosine_similarity(a, b):
    return F.cosine_similarity(a, b, dim=0)


def dirichlet_partition(dataset, num_clients, alpha=0.5, seed=42):
    """
    按 Dirichlet(alpha) 将 dataset 索引划分给 num_clients 个客户端
    返回: dict {client_id: set(indices)}
    """
    np.random.seed(seed)
    num_classes = 10                                  # CIFAR-10
    class_indices = [[] for _ in range(num_classes)]  # 每类的样本索引

    # 1) 先按类别把全局索引分桶
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    # 2) 记录客户端得到的索引
    client_dict = defaultdict(list)

    # 3) 对每个类别用 Dirichlet 把样本再划给所有客户端
    for c in range(num_classes):
        idxs = class_indices[c]
        np.random.shuffle(idxs)

        # 3-A 抽取比例
        proportions = np.random.dirichlet(alpha=np.repeat(alpha, num_clients))
        # 为防极端情况，可加一个微小平滑项再正规化
        proportions = (proportions + 1e-6) / proportions.sum()

        # 3-B 按比例切片，分给各客户端
        counts = (proportions * len(idxs)).astype(int)

        # 因四舍五入导致总和≠len(idxs)，再补齐
        while counts.sum() < len(idxs):
            counts[np.random.randint(num_clients)] += 1

        start = 0
        for client_id, cnt in enumerate(counts):
            client_dict[client_id].extend(idxs[start:start+cnt])
            start += cnt

    # 4) 转 set / list（按需）
    return {cid: np.array(idxs) for cid, idxs in client_dict.items()}

def split_test_by_train(train_dict, train_dataset, test_dataset, seed=123):
    """
    根据 train_dict 中每个客户端训练集的类别集合，
    将 test_dataset 分配给拥有该类的客户端。
    返回: {client_id: np.array(测试样本索引)}
    """
    np.random.seed(seed)
    num_classes = 10
    num_clients = len(train_dict)

    # ---------- 1) 统计每个客户端在“训练集”里拥有的类别 ----------
    client_labels = [set() for _ in range(num_clients)]
    for cid, idxs in train_dict.items():
        labels = [train_dataset[i][1] for i in idxs]   # ← 用训练集
        client_labels[cid].update(labels)

    # ---------- 2) 按类别收集“测试集”索引 ----------
    class_indices_test = [[] for _ in range(num_classes)]
    for idx, (_, label) in enumerate(test_dataset):
        class_indices_test[label].append(idx)

    # ---------- 3) 初始化返回字典 ----------
    test_dict = {cid: [] for cid in range(num_clients)}

    # ---------- 4) 逐类把测试样本分给拥有该类的客户端 ----------
    for c in range(num_classes):
        idxs = class_indices_test[c]
        np.random.shuffle(idxs)

        owners = [cid for cid in range(num_clients) if c in client_labels[cid]]
        if not owners:                 # 若极端地没人拥有该类，可全部丢给第 0 个客户端或跳过
            continue                   # 此处简单跳过

        splits = np.array_split(idxs, len(owners))
        for chunk, cid in zip(splits, owners):
            test_dict[cid].extend(chunk)

    return {cid: np.array(idxs) for cid, idxs in test_dict.items()}

#=====================================================================================================
#                           Client-side Model definition
#=====================================================================================================
# Model at client side

# Define ResNet18 Model for Client Side
class ResNet18_Client(nn.Module):
    def __init__(self):
        super().__init__()

        # conv1：为了 CIFAR-10，改为 3×3 stride=1，保留 32×32 分辨率
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)

        # ---------------- ResNet layer1 (2 basic blocks) ----------------
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        # ---------------- ResNet layer2 (2 basic blocks) ----------------
        # 这里 stride=2，把 32×32 → 16×16
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)

        self._weights_init()

    # -------- BasicBlock 定义 --------
    class _BasicBlock(nn.Module):
        expansion = 1
        def __init__(self, in_planes, planes, stride=1, downsample=None):
            super().__init__()
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                                   stride=stride, padding=1, bias=False)
            self.bn1   = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.bn2   = nn.BatchNorm2d(planes)
            self.downsample = downsample
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            identity = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
            return self.relu(out)

    # -------- 生成若干 BasicBlock --------
    def _make_layer(self, in_planes, planes, num_blocks, stride):
        downsample = None
        if stride != 1 or in_planes != planes:
            # 通道数或尺寸发生变化时，用 1×1 卷积匹配
            downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        layers = [self._BasicBlock(in_planes, planes, stride, downsample)]
        for _ in range(1, num_blocks):
            layers.append(self._BasicBlock(planes, planes))
        return nn.Sequential(*layers)

    # -------- 权重初始化 --------
    def _weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # -------- 前向传播 --------
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))   # (B,64,32,32)
        x = self.layer1(x)                       # (B,64,32,32)
        x = self.layer2(x)                       # (B,128,16,16)
        return x
   

#=====================================================================================================
#                           Server-side Model definition
#=====================================================================================================
# Model at server side
class ResNet18_Server(nn.Module):
    """
    接收来自客户端的特征 (B,128,16,16)
    包含 layer3、layer4、全局池化与全连接
    """
    def __init__(self, num_classes=10):
        super().__init__()

        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2) # 16→8
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2) # 8→4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        self._weights_init()

    # BasicBlock 与客户端保持一致
    class _BasicBlock(nn.Module):
        expansion = 1
        def __init__(self, in_planes, planes, stride=1, downsample=None):
            super().__init__()
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                                   stride=stride, padding=1, bias=False)
            self.bn1   = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.bn2   = nn.BatchNorm2d(planes)
            self.downsample = downsample
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            identity = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
            return self.relu(out)

    def _make_layer(self, in_planes, planes, num_blocks, stride):
        downsample = None
        if stride != 1 or in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        layers = [self._BasicBlock(in_planes, planes, stride, downsample)]
        for _ in range(1, num_blocks):
            layers.append(self._BasicBlock(planes, planes))
        return nn.Sequential(*layers)

    def _weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer3(x)                 # (B,256,8,8)
        x = self.layer4(x)                 # (B,512,4,4)
        x = self.avgpool(x)                # (B,512,1,1)
        x = torch.flatten(x, 1)            # (B,512)
        return self.fc(x)                  # (B,10)


net_glob_client = ResNet18_Client().to(device)
net_glob_server = ResNet18_Server().to(device)
  

# -------------------------------------------------
# 编码器 Enc：降维/量化
# 这里示例：3×3 stride=2 卷积 + BN + ReLU -> (B, C/2, H/2, W/2)
# -------------------------------------------------
class Encoder(nn.Module):
    """
    输入: (B,128,16,16)
    输出: (B,64,8,8)          —— 仅降 1 次
    """
    def __init__(self, in_channels=128, ratio=0.5, noise_std=0.0):
        super().__init__()
        mid = int(in_channels * ratio)          # 128 → 64
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, 2, 1, bias=False),  # 16→8
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        self.noise_std = noise_std

    def forward(self, x):
        z = self.conv(x)
        if self.noise_std > 0:
            z = z + torch.randn_like(z) * self.noise_std
        return z                                # (B,64,8,8)


class Decoder(nn.Module):
    """
    输入: (B,64,8,8)
    输出: (B,128,16,16)        —— 仅升 1 次
    """
    def __init__(self, out_channels=128, ratio=0.5):
        super().__init__()
        in_channels = int(out_channels * ratio)  # 64
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size=4, stride=2, padding=1, bias=False),  # 8→16
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, z):
        return self.deconv(z)                   # (B,128,16,16)

def pretrain_autoencoder(client_net, enc, dec, ldr_train,
                         epochs=5, lr=1e-3, device="cuda"):
    client_net.eval()                    # 冻结特征提取网络
    enc.train(); dec.train()
    opt = torch.optim.Adam(
        list(enc.parameters()) + list(dec.parameters()), lr=lr
    )
    mse = nn.MSELoss()

    for ep in range(epochs):
        loss_sum, cnt = 0, 0
        for batch_idx, (images, labels) in enumerate(ldr_train):
            images, labels = images.to(device), labels.to(device)
                
            with torch.no_grad():
                fx = client_net(images)    # (B,128,16,16)

            z  = enc(fx)
            fx_hat = dec(z)

            loss = mse(fx_hat, fx)
            opt.zero_grad(); loss.backward(); opt.step()

            loss_sum += loss.item(); cnt += 1
        print(f"[AE-pre] epoch {ep+1}/{epochs}  loss={loss_sum/cnt:.4f}")

    return enc, dec                      # 训练后的参数


encoder_client = Encoder().to(device)
decoder_client = Decoder().to(device)
#===================================================================================
# For Server Side Loss and Accuracy 
loss_train_collect = []
acc_train_collect = []
loss_test_collect = []
acc_test_collect = []
batch_acc_train = []
batch_loss_train = []
batch_acc_test = []
batch_loss_test = []


criterion = nn.CrossEntropyLoss()
count1 = 0
count2 = 0
#====================================================================================================
#                                  Server Side Program
#====================================================================================================
# Federated averaging: FedAvg
def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 *correct.float()/preds.shape[0]
    return acc

# to print train - test together in each round-- these are made global
acc_avg_all_user_train = 0
loss_avg_all_user_train = 0
loss_train_collect_user = []
acc_train_collect_user = []
loss_test_collect_user = []
acc_test_collect_user = []

w_glob_server = net_glob_server.state_dict()
w_locals_server = []

#client idx collector
idx_collect = []
l_epoch_check = False
fed_check = False
# Initialization of net_model_server and net_server (server-side model)
net_model_server = [net_glob_server for i in range(num_users)]
net_server = copy.deepcopy(net_model_server[0]).to(device)
#optimizer_server = torch.optim.Adam(net_server.parameters(), lr = lr)

def flatten_params(model_state_dict):
    # 将所有的参数展平为一个一维向量
    params = []
    for param in model_state_dict.values():
        params.append(param.view(-1))  # 展平每个参数张量
    return torch.cat(params)  # 将所有参数拼接成一个大的张量

# Server-side function associated with Training 
def train_server(z, y, l_epoch_count, l_epoch, idx, len_batch, decoder):
    global net_model_server, criterion, optimizer_server, device, batch_acc_train, batch_loss_train, l_epoch_check, fed_check
    global loss_train_collect, acc_train_collect, count1, acc_avg_all_user_train, loss_avg_all_user_train, idx_collect, w_locals_server, w_glob_server, net_server
    global loss_train_collect_user, acc_train_collect_user, lr
    
    net_server = copy.deepcopy(net_model_server[idx]).to(device)
    net_server.train()
    optimizer_server = torch.optim.Adam(net_server.parameters(), lr = lr)

    
    # train and update
    optimizer_server.zero_grad()
    
    z = z.to(device)
    y = y.to(device)
    
    #---------forward prop-------------
    fx_hat = decoder(z)
    fx_hat.retain_grad()
    fx_server = net_server(fx_hat)
    
    # calculate loss
    loss = criterion(fx_server, y)
    # calculate accuracy
    acc = calculate_accuracy(fx_server, y)
    
    #--------backward prop--------------
    loss.backward()
    dfx_client = fx_hat.grad.clone().detach()
    optimizer_server.step()
    
    batch_loss_train.append(loss.item())
    batch_acc_train.append(acc.item())
    
    # Update the server-side model for the current batch
    net_model_server[idx] = copy.deepcopy(net_server)
    
    # count1: to track the completion of the local batch associated with one client
    count1 += 1
    if count1 == len_batch:
        acc_avg_train = sum(batch_acc_train)/len(batch_acc_train)           # it has accuracy for one batch
        loss_avg_train = sum(batch_loss_train)/len(batch_loss_train)
        
        batch_acc_train = []
        batch_loss_train = []
        count1 = 0
        
        prRed('Client{} Train => Local Epoch: {} \tAcc: {:.3f} \tLoss: {:.4f}'.format(idx, l_epoch_count, acc_avg_train, loss_avg_train))
        
        # copy the last trained model in the batch       
        w_server = net_server.state_dict()      
        
        # If one local epoch is completed, after this a new client will come
        if l_epoch_count == l_epoch-1:
            fx_embed = embed_fx(fx_hat.cpu())        
            fx_collect.append(fx_embed)                 

            l_epoch_check = True                # to evaluate_server function - to check local epoch has completed or not 
            # We store the state of the net_glob_server() 
            w_locals_server.append(copy.deepcopy(w_server))
            
            # we store the last accuracy in the last batch of the epoch and it is not the average of all local epochs
            # this is because we work on the last trained model and its accuracy (not earlier cases)
            
            #print("accuracy = ", acc_avg_train)
            acc_avg_train_all = acc_avg_train
            loss_avg_train_all = loss_avg_train
                        
            # accumulate accuracy and loss for each new user
            loss_train_collect_user.append(loss_avg_train_all)
            acc_train_collect_user.append(acc_avg_train_all)
            
            # collect the id of each new user                        
            if idx not in idx_collect:
                idx_collect.append(idx) 
                #print(idx_collect)
        
        # This is for federation process--------------------
        if len(idx_collect) == num_users:
            fed_check = True                                                  # to evaluate_server function  - to check fed check has hitted
            # Calculate cosine similarities between all pairs of clients
            for i in range(num_users):
                 for j in range(i, num_users):
                    sim = torch.dot(fx_collect[i], fx_collect[j]).item()
                    S[i, j] = S[j, i] = sim

            # Perform agglomerative clustering to group similar clients
            clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.04, metric='precomputed', linkage='average')
            clustering.fit(1 - S)  # we need to use (1 - similarity) as distance

            # Group clients into clusters based on similarity
            client_groups = {}
            for i, label in enumerate(clustering.labels_):
                if label not in client_groups:
                    client_groups[label] = []
                client_groups[label].append(i)
            
            # Perform federation and model update per group
            for group in client_groups.values():
                # Perform federated learning for each group of clients
                group_w_locals = [w_locals_server[i] for i in group]
                w_glob_server = FedAvg(group_w_locals)
                net_glob_server.load_state_dict(w_glob_server)
                # Distribute the global model to the clients in the group
                for i in group:
                    net_model_server[i] = copy.deepcopy(net_glob_server)
            
            # Reset for the next round
            w_locals_server = []
            idx_collect = []
            
            acc_avg_all_user_train = sum(acc_train_collect_user) / len(acc_train_collect_user)
            loss_avg_all_user_train = sum(loss_train_collect_user) / len(loss_train_collect_user)
            
            loss_train_collect.append(loss_avg_all_user_train)
            acc_train_collect.append(acc_avg_all_user_train)
            
            acc_train_collect_user = []
            loss_train_collect_user = []
            
    # send gradients to the client               
    return dfx_client

# Server-side functions associated with Testing
def evaluate_server(fx_client, y, idx, len_batch, ell):
    global net_model_server, criterion, batch_acc_test, batch_loss_test, check_fed, net_server, net_glob_server 
    global loss_test_collect, acc_test_collect, count2, num_users, acc_avg_train_all, loss_avg_train_all, w_glob_server, l_epoch_check, fed_check
    global loss_test_collect_user, acc_test_collect_user, acc_avg_all_user_train, loss_avg_all_user_train
    
    net = copy.deepcopy(net_model_server[idx]).to(device)
    net.eval()
  
    with torch.no_grad():
        fx_client = fx_client.to(device)
        y = y.to(device) 
        #---------forward prop-------------
        fx_server = net(fx_client)
        
        # calculate loss
        loss = criterion(fx_server, y)
        # calculate accuracy
        acc = calculate_accuracy(fx_server, y)
        
        
        batch_loss_test.append(loss.item())
        batch_acc_test.append(acc.item())
        
               
        count2 += 1
        if count2 == len_batch:
            acc_avg_test = sum(batch_acc_test)/len(batch_acc_test)
            loss_avg_test = sum(batch_loss_test)/len(batch_loss_test)
            
            batch_acc_test = []
            batch_loss_test = []
            count2 = 0
            
            prGreen('Client{} Test =>                   \tAcc: {:.3f} \tLoss: {:.4f}'.format(idx, acc_avg_test, loss_avg_test))
            
            # if a local epoch is completed   
            if l_epoch_check:
                l_epoch_check = False
                
                # Store the last accuracy and loss
                acc_avg_test_all = acc_avg_test
                loss_avg_test_all = loss_avg_test
                        
                loss_test_collect_user.append(loss_avg_test_all)
                acc_test_collect_user.append(acc_avg_test_all)
                
            # if federation is happened----------                    
            if fed_check:
                fed_check = False
                print("------------------------------------------------")
                print("------ Federation process at Server-Side ------- ")
                print("------------------------------------------------")
                
                acc_avg_all_user = sum(acc_test_collect_user)/len(acc_test_collect_user)
                loss_avg_all_user = sum(loss_test_collect_user)/len(loss_test_collect_user)
            
                loss_test_collect.append(loss_avg_all_user)
                acc_test_collect.append(acc_avg_all_user)
                acc_test_collect_user = []
                loss_test_collect_user= []
                              
                print("====================== SERVER==========================")
                print(' Train: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_avg_all_user_train, loss_avg_all_user_train))
                print(' Test: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_avg_all_user, loss_avg_all_user))
                print("==========================================================")
         
    return 

#==============================================================================================================
#                                       Clients-side Program
#==============================================================================================================
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class Client(object):
    def __init__(self, net_client_model, idx, lr, device, dataset_train = None, dataset_test = None, 
                 idxs = None, idxs_test = None, encoder = None,decoder = None):
        self.idx = idx
        self.device = device
        self.lr = lr
        self.local_ep = 1
        self.ldr_train = DataLoader(DatasetSplit(dataset_train, idxs), batch_size=256, shuffle=True)
        self.ldr_test = DataLoader(DatasetSplit(dataset_test, idxs_test), batch_size=256, shuffle=True)
        self.encoder = encoder
        self.decoder = decoder

    def train(self, net):
        encoder, decoder = pretrain_autoencoder(copy.deepcopy(net).to(device), self.encoder, self.decoder, self.ldr_train, epochs=20)

        net.train()
        optimizer_client = torch.optim.Adam(net.parameters(), lr = self.lr) 
        
        for iter in range(self.local_ep):
            len_batch = len(self.ldr_train)
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer_client.zero_grad()
                fx = net(images)
                z  = self.encoder(fx).detach().requires_grad_(True)
                # client_fx = fx.clone().detach().requires_grad_(True)
                dfx = train_server(z, labels, iter, self.local_ep, self.idx, len_batch, decoder)
                fx.backward(dfx)
                optimizer_client.step()
        return net.state_dict()

    def evaluate(self, net, ell):
        net.eval()
        with torch.no_grad():
            len_batch = len(self.ldr_test)
            for batch_idx, (images, labels) in enumerate(self.ldr_test):
                images, labels = images.to(self.device), labels.to(self.device)
                fx = net(images)
                evaluate_server(fx, labels, self.idx, len_batch, ell)
        return

#=====================================================================================================
# CIFAR-10 Dataset and DataLoader
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load CIFAR-10 data
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)



# 训练集 Dirichlet 划分
dict_users_train_non_idd = dirichlet_partition(train_dataset, num_users, alpha=1)

# 测试集对应划分
dict_users_test_non_idd = split_test_by_train(dict_users_train_non_idd, train_dataset, test_dataset)

# Define dataset splitting function (IID)
def dataset_iid(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

# Split the CIFAR-10 dataset
dict_users_train = dataset_iid(train_dataset, num_users)
dict_users_test = dataset_iid(test_dataset, num_users)


#------------ Training And Testing  -----------------
net_glob_client.train()
#copy weights
w_glob_client = net_glob_client.state_dict()
# Federation takes place after certain local epochs in train() client-side
# this epoch is global epoch, also known as rounds
# Training and testing loop
user_list = []
for idx in range(num_users):
    local = Client(net_glob_client, idx, lr, device, dataset_train=train_dataset, dataset_test=test_dataset, 
                       idxs=dict_users_train_non_idd[idx], idxs_test=dict_users_test_non_idd[idx],
                       encoder=copy.deepcopy(encoder_client).to(device), decoder=copy.deepcopy(decoder_client).to(device))
    user_list.append(local)
for iter in range(epochs):
    m = max(int(frac * num_users), 1)
    idxs_users = np.random.choice(range(num_users), m, replace=False)
    w_locals_client = []
    for idx in idxs_users:
        local = user_list[idx]
        w_client = local.train(net=copy.deepcopy(net_glob_client).to(device))
        w_locals_client.append(copy.deepcopy(w_client))
        local.evaluate(net=copy.deepcopy(net_glob_client).to(device), ell=iter)

    w_glob_client = FedAvg(w_locals_client)
    net_glob_client.load_state_dict(w_glob_client)

print("Training and Evaluation completed!")
