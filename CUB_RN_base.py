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
from sklearn.metrics import accuracy_score
from model_dataset import AttributeNetwork, RelationNetwork, IntegratedDataset
from utils import evaluate_attribute_network, evaluate_relation_network


def main():
    parser = argparse.ArgumentParser(description="Zero Shot Learning")
    parser.add_argument("-s", "--seed", type = int, default = 1234)
    parser.add_argument("-b", "--batch_size", type = int, default = 32)
    parser.add_argument("-e", "--epochs", type = int, default = 1000)
    parser.add_argument("-m", "--model", type = int, default = 1)
    parser.add_argument("-t", "--test_episode", type = int, default = 1000)
    parser.add_argument("-l", "--learning_rate", type = float, default = 1e-5)
    parser.add_argument("-g", "--gpu", type=int, default=0)
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    MODEL = args.model
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
    
    num_valid = int(len(train_label) * 0.2)
    
    valid_features = train_features[-num_valid:]
    valid_attributes = train_attributes[-num_valid:]
    valid_label = train_label[-num_valid:]

    train_features = train_features[:-num_valid]
    train_attributes = train_attributes[:-num_valid]
    train_label = train_label[:-num_valid]
    
    train_label_set = np.unique(train_label)
    valid_label_set = np.unique(valid_label)
    test_unseen_label_set = np.unique(test_unseen_label)
    test_seen_label_set = np.unique(test_seen_label)

    train_features = torch.from_numpy(train_features) # [5646, 2048]
    train_attributes = torch.from_numpy(train_attributes) # [5646, 312]
    train_label = torch.from_numpy(train_label).unsqueeze(1) # [5646, 1]

    valid_features = torch.from_numpy(valid_features) # [1411, 2048]
    valid_attributes = torch.from_numpy(valid_attributes) # [1411, 312]
    valid_label = torch.from_numpy(valid_label).unsqueeze(1) # [1411, 1]

    test_unseen_features = torch.from_numpy(test_unseen_features) # [2967, 2048]
    test_unseen_attributes = torch.from_numpy(test_unseen_attributes) # [2967, 312]
    test_unseen_label = torch.from_numpy(test_unseen_label).unsqueeze(1) # [2967, 1]

    test_seen_features = torch.from_numpy(test_seen_features) # [1764, 2048]
    test_seen_attributes = torch.from_numpy(test_seen_attributes) # [1764, 312]
    test_seen_label = torch.from_numpy(test_seen_label).unsqueeze(1) # [1764, 1]
    # init network
    print("init networks")
    attribute_network = AttributeNetwork(2048, 1200, 312).cuda()
    relation_network = RelationNetwork(624, 300).cuda()
    
    train_data = IntegratedDataset(train_features, train_label, train_attributes)
    valid_data = IntegratedDataset(valid_features, valid_label, valid_attributes)

    mse = nn.MSELoss().cuda()
    #ce = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1, 1.])).cuda()
    nll = torch.nn.NLLLoss(weight=torch.FloatTensor([0.1, 1.])).cuda()
    #mse = nn.BCELoss().cuda()

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)

    if MODEL == 1:
        attribute_network_optim = torch.optim.Adam(attribute_network.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        #attribute_network_optim = torch.optim.SGD(attribute_network.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4, nesterov=True)
        attribute_network_scheduler = StepLR(attribute_network_optim, step_size=30000, gamma=0.5)

        print("training attribute network...")

        total_steps = 0
        best_loss_att = 10000

        for epoch in range(EPOCHS):
            attribute_network.train()
            for i, (batch_features, batch_labels, batch_att) in enumerate(train_loader):

                batch_features, batch_att = batch_features.float().cuda(), batch_att.float().cuda()
                
                attribute_network_scheduler.step(total_steps)
                
                pred_embeddings = attribute_network(batch_features)

                cat_batch_att = batch_att.long().view(-1)
                cat_pred_att = torch.cat((pred_embeddings.unsqueeze(2), 1 - pred_embeddings.unsqueeze(2)), 2).view(-1, 2)
                
                #loss_att = ce(cat_pred_att, cat_batch_att)
                loss_att = nll(torch.log(cat_pred_att + 1e-10), cat_batch_att)

                attribute_network_optim.zero_grad()
                loss_att.backward()
                attribute_network_optim.step()

                total_steps += 1

            #if epoch % 50 == 0:
            loss_att_mean, acc_att = evaluate_attribute_network(attribute_network, nll, valid_loader)
            print("Epoch: {:>3} loss_att: {:.5f}".format(epoch, loss_att_mean))
            print("Epoch: {:>3} acc_att: {:.5f}".format(epoch, acc_att))

            if best_loss_att > loss_att_mean:
                torch.save(attribute_network.state_dict(), 'models/attribute_network.pt')
                best_loss_att = loss_att_mean

    else:
        relation_network_optim = torch.optim.Adam(relation_network.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        relation_network_scheduler = StepLR(relation_network_optim, step_size=30000, gamma=0.5)

        print("training relation network...")

        total_steps = 0
        best_loss_rel = 10000

        for epoch in range(EPOCHS):
            relation_network.train()
            for i, (batch_features, batch_labels, batch_att) in enumerate(train_loader):
                batch_att = batch_att.float().cuda()

                relation_network_scheduler.step(total_steps)

                sample_labels = np.unique(batch_labels.squeeze().numpy())
                        
                sample_embeddings = all_embeddings[sample_labels]
                sample_embeddings = torch.from_numpy(sample_embeddings).float().cuda()

                class_num = sample_embeddings.shape[0]

                embeddings_bunch = sample_embeddings.unsqueeze(0).repeat(len(batch_att), 1, 1)
                attributes_bunch = batch_att.unsqueeze(0).repeat(class_num, 1, 1)
                attributes_bunch = torch.transpose(attributes_bunch, 0, 1)
                
                relation_pairs = torch.cat((embeddings_bunch, attributes_bunch), 2).view(-1, 624)
                relations = relation_network(relation_pairs).view(-1, class_num)
                
                # re-build batch_labels according to sample_labels
                re_batch_labels = []
                for label in batch_labels.numpy():
                    index = np.argwhere(sample_labels == label)
                    re_batch_labels.append(index[0][0])
                re_batch_labels = torch.LongTensor(re_batch_labels)
                one_hot_labels = torch.zeros(len(batch_att), class_num).scatter_(1, re_batch_labels.view(-1, 1), 1).cuda()
                
                loss_rel = mse(relations, one_hot_labels)

                # update
                relation_network.zero_grad()
                loss_rel.backward()
                relation_network_optim.step()

            #if epoch % 50 == 0:
            loss_rel_mean, acc_rel = evaluate_relation_network(relation_network, mse, valid_loader, all_embeddings)
            print("Epoch: {:>3} loss_rel: {:.5f}".format(epoch, loss_rel_mean))
            print("Epoch: {:>3} acc_rel: {:.5f}".format(epoch, acc_rel))

            if best_loss_rel > loss_rel_mean:
                torch.save(relation_network.state_dict(), 'models/relation_network.pt')
                best_loss_rel = loss_rel_mean

        assert 1 == 0
        # if (episode+1)%100 == 0:
        #         print("episode:",episode+1,"loss",loss.data[0])
            # test
        print("Testing...")

        
        
        zsl_accuracy = compute_accuracy(test_features,test_label,test_id,test_attributes)
        gzsl_unseen_accuracy = compute_accuracy(test_features,test_label,np.arange(200),attributes)
        gzsl_seen_accuracy = compute_accuracy(test_seen_features,test_seen_label,np.arange(200),attributes)
        
        H = 2 * gzsl_seen_accuracy * gzsl_unseen_accuracy / (gzsl_unseen_accuracy + gzsl_seen_accuracy)
        
        print('zsl:', zsl_accuracy)
        print('gzsl: seen=%.4f, unseen=%.4f, h=%.4f' % (gzsl_seen_accuracy, gzsl_unseen_accuracy, H))
        
        if zsl_accuracy > last_accuracy:
            # save networks
            torch.save(attribute_network.state_dict(),"./models/zsl_cub_attribute_network_v35.pkl")
            torch.save(relation_network.state_dict(),"./models/zsl_cub_relation_network_v35.pkl")

            print("save networks for episode:",episode)
            
            last_accuracy = zsl_accuracy
        

if __name__ == '__main__':
    main()
