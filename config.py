import time
import random
import math
import sys
import argparse
import torch.nn as nn
import torch.nn.functional as F
import json
import xlrd
from dataset import *
from torch.utils import *

parser = argparse.ArgumentParser(description = 'retrieval')
parser.add_argument('--dataset', type = str, default = 'voc2007', help = "dataset name")    #coco, flickr, voc2007, voc2012, nuswide
parser.add_argument('--hash_bit', type = int, default = 48, help = "number of hash code bits")      #12, 16, 24, 32, 36, 48, 64
parser.add_argument('--batch_size', type = int, default = 100, help = "batch size")
parser.add_argument('--epochs', type = int, default = 100, help = "epochs")
parser.add_argument('--cuda', type = int, default = 0, help = "cuda id")
parser.add_argument('--backbone', type = str, default = 'googlenet', help = "backbone")     #googlenet, resnet, alexnet
parser.add_argument('--beta', type = float, default = 0.5, help = "hyper-parameter for regularization")
parser.add_argument('--retrieve', type = int, default = 0, help = "retrieval number")
parser.add_argument('--no_save', action = 'store_true', default = False, help = "No save")
parser.add_argument('--seed', type = int, default = 0, help = "random seed")
parser.add_argument('--rate', type = float, default = 0.02, help = "rate")
parser.add_argument('--test', action = 'store_true', default = False, help = "testing") # for testing
args = parser.parse_args()

# Hyper-parameters
train_flag = bool(1 - args.test)
backbone = args.backbone
retrieve = args.retrieve
save_flag = bool(1 - args.no_save)

dataset = args.dataset
num_epochs = args.epochs

batch_size = args.batch_size
if backbone == 'googlenet':
    feature_rate = 0.02
elif backbone == 'alexnet':
    feature_rate = 0.01
criterion_rate = args.rate
num_bits = args.hash_bit

# hyper-parameters
beta = args.beta
seed =args.seed


# path for loading and saving models
path = './result/' + dataset + '_' + backbone + '_' + str(num_bits)
model_path = path + '.ckpt'

if train_flag and save_flag:
    file_path = path + '.txt'
    f = open(file_path, 'w')

# Device configuration
device = torch.device('cuda:'+str(args.cuda) if torch.cuda.is_available() else 'cpu')

#  data pre-treatment
if backbone == 'googlenet':
    data_transform = {
        "train": transforms.Compose([transforms.Resize((448, 448)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize((448, 448)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
elif backbone in ['resnet', 'alexnet']:
    data_transform = {
        "train": transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

# load train data

if dataset == 'flickr':
    num_classes = 38
    if retrieve == 0:
        retrieve = 1000
    Flickr25k.init('./data/flickr25k/', 1000, 4000)
    trainset = Flickr25k('./data/flickr25k/', 'train', transform = data_transform['train'])
    testset = Flickr25k('./data/flickr25k/', 'query', transform = data_transform['val'])
    database = Flickr25k('./data/flickr25k/', 'retrieval', transform = data_transform['val'])

elif dataset == 'voc2007':
    if retrieve == 0:
        retrieve = 5011
    num_classes = 20
    trainset = VOCBase(root = './data', year = '2007', image_set = 'trainval', download = True, transform = data_transform['train'])
    testset = VOCBase(root = './data', year = '2007', image_set = 'test', download = True, transform = data_transform['val']) 
    database = VOCBase(root = './data', year = '2007', image_set = 'trainval', download = True, transform = data_transform['val'])

elif dataset == 'voc2012':
    num_classes = 20
    if retrieve == 0:
        retrieve = 5717 
    trainset = VOCBase(root = './data', year = '2012', image_set = 'train', download = True, transform = data_transform['train'])
    testset = VOCBase(root = './data', year = '2012', image_set = 'val', download = True, transform = data_transform['val']) 
    database = VOCBase(root = './data', year = '2012', image_set = 'train', download = True, transform = data_transform['val'])

elif dataset == 'nuswide':
    if retrieve == 0:
        retrieve = 5000
    num_classes = 21
    trainset = ImageList(open('./data/nus_wide/train.txt', 'r').readlines(), transform = data_transform['train'])
    testset = ImageList(open('./data/nus_wide/test.txt', 'r').readlines(), transform = data_transform['val'])
    database = ImageList(open('./data/nus_wide/database.txt', 'r').readlines(), transform = data_transform['val'])

    
train_num = len(trainset)
test_num = len(testset)
database_num = len(database)

trainloader = data.DataLoader(dataset = trainset,
                            batch_size = batch_size,
                            shuffle = True,
                            num_workers = 8)

testloader = data.DataLoader(dataset = testset,
                            batch_size = batch_size,
                            shuffle = False,
                            num_workers = 8)

databaseloader = data.DataLoader(dataset = database,
                            batch_size = batch_size,
                            shuffle = False,
                            num_workers = 8)

# find the value of ζ

sheet = xlrd.open_workbook('codetable.xlsx').sheet_by_index(0)
threshold = sheet.row(num_bits)[math.ceil(math.log(num_classes, 2))].value
print(threshold)

print('------------- data prepared -------------')
