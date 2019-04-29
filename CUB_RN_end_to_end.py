import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import numpy as np
import scipy.io as sio
import math
import argparse
import random
import os
from model_dataset import AttributeNetwork, AttributeNetwork_End_to_End, RelationNetwork, IntegratedDataset
from utils import evaluate_attribute_network, evaluate_relation_network, compute_accuracy_relation_network, compute_accuracy_whole_network


def main():
    parser = argparse.ArgumentParser(description="Zero Shot Learning")
    parser.add_argument("-s", "--seed", type = int, default = 1234)
    parser.add_argument("-b", "--batch_size", type = int, default = 50)
    parser.add_argument("-e", "--epochs", type = int, default = 1000)
    parser.add_argument("-t", "--test_episode", type = int, default = 1000)
    parser.add_argument("-l", "--learning_rate", type = float, default = 1e-4)
    parser.add_argument("-g", "--gpu", type=int, default=0)
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    TEST_EPISODE = args.test_episode
    LEARNING_RATE = args.learning_rate
    GPU = args.gpu

    np.set_printoptions(threshold=np.inf)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # step 1: init dataset
    print("init dataset")
    
    dataroot = './data'
    dataset = 'CUB1_data'
    image_embedding = 'res101'
    class_embedding = 'original_att_splits'

    attribute_values = np.load("CUB_200_2011/attribute_values.npy")

    matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + image_embedding + ".mat")
    feature = matcontent['features'].T

    label = matcontent['labels'].astype(int).squeeze() - 1

    matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + class_embedding + ".mat")
    # numpy array index starts from 0, matlab starts from 1
    trainval_loc = matcontent['trainval_loc'].squeeze() - 1
    test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
    test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
  
    embedding = matcontent['att'].T
    all_embeddings = np.array(embedding) # (200, 312)

    train_features = feature[trainval_loc] # train_features
    train_attributes = attribute_values[trainval_loc] # train_attribute_values
    train_label = label[trainval_loc].astype(int)  # train_label

    test_unseen_features = feature[test_unseen_loc]  # test_unseen_features
    test_unseen_attributes = attribute_values[test_unseen_loc] # test_unseen_attributes_values
    test_unseen_label = label[test_unseen_loc].astype(int) # test_unseen_label

    test_seen_features = feature[test_seen_loc]  #test_seen_features
    test_seen_attributes = attribute_values[test_seen_loc] # test_seen_attributes_values
    test_seen_label = label[test_seen_loc].astype(int) # test_seen_label
    
    test_features = np.concatenate((test_unseen_features, test_seen_features), 0)
    test_attributes = np.concatenate((test_unseen_attributes, test_seen_attributes), 0)
    test_label = np.concatenate((test_unseen_label, test_seen_label), 0)

    train_label_set = np.unique(train_label)
    test_unseen_label_set = np.unique(test_unseen_label)
    test_seen_label_set = np.unique(test_seen_label)
    test_label_set = np.unique(test_label)

    train_features = torch.from_numpy(train_features) # [5646, 2048]
    train_attributes = torch.from_numpy(train_attributes) # [5646, 312]
    train_label = torch.from_numpy(train_label).unsqueeze(1) # [5646, 1]

    test_unseen_features = torch.from_numpy(test_unseen_features) # [2967, 2048]
    test_unseen_attributes = torch.from_numpy(test_unseen_attributes) # [2967, 312]
    test_unseen_label = torch.from_numpy(test_unseen_label).unsqueeze(1) # [2967, 1]

    test_seen_features = torch.from_numpy(test_seen_features) # [1764, 2048]
    test_seen_attributes = torch.from_numpy(test_seen_attributes) # [1764, 312]
    test_seen_label = torch.from_numpy(test_seen_label).unsqueeze(1) # [1764, 1]

    test_features = torch.from_numpy(test_features) # [4731, 2048]
    test_attributes = torch.from_numpy(test_attributes) # [4731, 312]
    test_label = torch.from_numpy(test_label).unsqueeze(1) # [4731, 1]

    # init network
    print("init networks")
    attribute_network = AttributeNetwork_End_to_End(2048, 1200, 312).cuda()
    relation_network = RelationNetwork(624, 300, 100).cuda()
    
    train_data = IntegratedDataset(train_features, train_label, train_attributes)
    test_data = IntegratedDataset(test_features, test_label, test_attributes)
    test_unseen_data = IntegratedDataset(test_unseen_features, test_unseen_label, test_unseen_attributes)
    test_seen_data = IntegratedDataset(test_seen_features, test_seen_label, test_seen_attributes)

    mse = nn.MSELoss().cuda()
    ce = nn.CrossEntropyLoss().cuda()
    nll = torch.nn.NLLLoss(weight=torch.FloatTensor([0.1, 1.])).cuda()
    #mse = nn.BCELoss().cuda()

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
    test_unseen_loader = DataLoader(test_unseen_data, batch_size=BATCH_SIZE, shuffle=True)
    test_seen_loader = DataLoader(test_seen_data, batch_size=BATCH_SIZE, shuffle=True)


    attribute_network_optim = torch.optim.Adam(attribute_network.parameters(), lr=LEARNING_RATE)
    attribute_network_scheduler = StepLR(attribute_network_optim, step_size=30000, gamma=0.5)

    relation_network_optim = torch.optim.Adam(relation_network.parameters(), lr=LEARNING_RATE)
    relation_network_scheduler = StepLR(relation_network_optim, step_size=30000, gamma=0.5)
    print("training networks...")

    total_steps = 0
    best_loss_att = 10000

    for epoch in range(EPOCHS):
        attribute_network.train()
        relation_network.train()

        for i, (batch_features, batch_labels, batch_att) in enumerate(train_loader):
            attribute_network_scheduler.step(total_steps)
            relation_network_scheduler.step(total_steps)

            batch_features = batch_features.float().cuda()        
            pred_embeddings = attribute_network(batch_features)

            sample_labels = np.unique(batch_labels.squeeze().numpy())
            sample_embeddings = all_embeddings[sample_labels]
            sample_embeddings = torch.from_numpy(sample_embeddings).float().cuda()
            class_num = sample_embeddings.shape[0]

            embeddings_bunch = sample_embeddings.unsqueeze(0).repeat(len(batch_features), 1, 1)
            attributes_bunch = pred_embeddings.unsqueeze(0).repeat(class_num, 1, 1)
            attributes_bunch = torch.transpose(attributes_bunch, 0, 1)
            relation_pairs = torch.cat((embeddings_bunch, attributes_bunch), 2).view(-1, 624)
            relations = relation_network(relation_pairs).view(-1, class_num)

            re_batch_labels = []
            for label in batch_labels.numpy():
                index = np.argwhere(sample_labels == label)
                re_batch_labels.append(index[0][0])
            re_batch_labels = torch.LongTensor(re_batch_labels).cuda()

            loss_rel = ce(relations, re_batch_labels)

            attribute_network_optim.zero_grad()
            relation_network_optim.zero_grad()

            loss_rel.backward()

            attribute_network_optim.step()
            relation_network_optim.step()

            total_steps += 1

        zsl = compute_accuracy_whole_network(attribute_network, relation_network, test_unseen_loader, test_unseen_label_set, all_embeddings)
        gzsl_u = compute_accuracy_whole_network(attribute_network, relation_network, test_unseen_loader, test_label_set, all_embeddings)
        gzsl_s = compute_accuracy_whole_network(attribute_network, relation_network, test_seen_loader, test_label_set, all_embeddings)

        H = 2 * gzsl_s * gzsl_u / (gzsl_u + gzsl_s)
        print("Epoch: {:>3} zsl: {:.5f} gzsl_u: {:.5f} gzsl_s: {:.5f} H: {:.5f}".format(epoch, zsl, gzsl_u, gzsl_s, H))


        

if __name__ == '__main__':
    main()
