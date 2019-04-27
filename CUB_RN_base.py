import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
import scipy.io as sio
import math
import argparse
import random
import os
from sklearn.metrics import accuracy_score


parser = argparse.ArgumentParser(description="Zero Shot Learning")
parser.add_argument("-b", "--batch_size", type = int, default = 32)
parser.add_argument("-e", "--epochs", type = int, default = 1000)
parser.add_argument("-m", "--model", type = int, default = 1)
parser.add_argument("-t", "--test_episode", type = int, default = 1000)
parser.add_argument("-l", "--learning_rate", type = float, default = 1e-5)
parser.add_argument("-g", "--gpu", type=int, default=0)
args = parser.parse_args()

# Hyper Parameters

BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
MODEL = args.model
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu

class AttributeNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size,output_size):
        super(AttributeNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x

class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size,):
        super(RelationNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)

    def forward(self,x):

        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x
    
class IntegratedDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels, att_values, cer_values):
        super(IntegratedDataset, self).__init__()
        self.features = features
        self.labels = labels
        self.att_values = att_values
        self.cer_values = cer_values

    def __getitem__(self, index):
        single_feature, single_label = self.features[index], self.labels[index]
        single_att_value, single_cer_value = self.att_values[index], self.cer_values[index]
        
        single_cer_value = single_cer_value / 4
        return single_feature, single_label, single_att_value, single_cer_value

    def __len__(self):
        return len(self.features)

def evaluate_attribute_network(model, criteria_1, criteria_2, eval_loader):
    model.eval()
    total_loss_att = 0.
    total_loss_cer = 0.
    total_correct_num_att = 0.
    total_correct_num_cer = 0.
    sample_num = 0
    with torch.no_grad():
        for i, (batch_features, batch_labels, batch_att, batch_cer) in enumerate(eval_loader):
            batch_features, batch_att, batch_cer = batch_features.float().cuda(), batch_att.float().cuda(), batch_cer.float().cuda()
            pred_embeddings = model(batch_features)

            cat_batch_att = batch_att.long().view(-1)
            cat_pred_att = torch.cat((pred_embeddings[:,0:312].unsqueeze(2), 1 - pred_embeddings[:,0:312].unsqueeze(2)), 2).view(-1, 2)
                
            #weights_att = torch.where(batch_att == torch.Tensor([1]).cuda(), torch.Tensor([10]).cuda(), torch.Tensor([1]).cuda())
            #bce_att = nn.BCELoss(weight=weights_att).cuda()
            #loss_att = bce_att(pred_embeddings[:,0:312], batch_att)
            loss_att = criteria_1(torch.log(cat_pred_att + 1e-10), cat_batch_att)
            #loss_att = criteria(pred_embeddings[:,0:312], batch_att)
            batch_correct_att = ((pred_embeddings[:,0:312] < 0.5).int() == batch_att.int()).float().sum()

            # weights_cer = torch.where(batch_cer == torch.Tensor([1]).cuda(), torch.Tensor([10]).cuda(), torch.Tensor([1]).cuda())
            # bce_cer = nn.BCELoss(weight=weights_cer).cuda()
            loss_cer = criteria_2(pred_embeddings[:,312:624], batch_cer)
            batch_correct_cer = ((pred_embeddings[:,312:624] * 4 + 0.5).int() == (batch_cer * 4).int()).float().sum()

            if i == 5:
                print(pred_embeddings[14,0:312])
                print(batch_att[14])
                print(pred_embeddings[14,312:624])
                print(batch_cer[14])

            total_loss_att += loss_att.item() * 312 * len(batch_att)
            total_loss_cer += loss_cer.item() * 312 * len(batch_cer)
            total_correct_num_att += batch_correct_att
            total_correct_num_cer += batch_correct_cer
            sample_num += len(batch_att)

    loss_att_mean = total_loss_att / (sample_num * 312)
    loss_cer_mean = total_loss_cer / (sample_num * 312)
    acc_att = total_correct_num_att / (sample_num * 312)
    acc_cer = total_correct_num_cer / (sample_num * 312)
    return loss_att_mean, loss_cer_mean, acc_att, acc_cer

def main():
    # step 1: init dataset
    print("init dataset")
    
    dataroot = './data'
    dataset = 'CUB1_data'
    image_embedding = 'res101'
    class_embedding = 'original_att_splits'

    attribute_values = np.load("CUB_200_2011/attribute_values.npy")
    certainty_values = np.load("CUB_200_2011/certainty_values.npy")

    matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + image_embedding + ".mat")
    feature = matcontent['features'].T

    label = matcontent['labels'].astype(int).squeeze() - 1

    matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + class_embedding + ".mat")
    # numpy array index starts from 0, matlab starts from 1
    trainval_loc = matcontent['trainval_loc'].squeeze() - 1
    test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
    test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
  
    attribute = matcontent['att'].T

    x = feature[trainval_loc] # train_features

    train_attributes = attribute_values[trainval_loc] # train_attribute_values
    train_certainty = certainty_values[trainval_loc] # train_certainty_values

    train_label = label[trainval_loc].astype(int)  # train_label

    att = attribute[train_label] # train attributes

    x_test = feature[test_unseen_loc]  # test_feature
    test_label = label[test_unseen_loc].astype(int) # test_label

    x_test_seen = feature[test_seen_loc]  #test_seen_feature
    test_label_seen = label[test_seen_loc].astype(int) # test_seen_label

    test_id = np.unique(test_label)   # test_id
    att_pro = attribute[test_id]      # test_attribute
    
    
    # train set
    train_features = torch.from_numpy(x) # [7057, 2048]

    train_label = torch.from_numpy(train_label).unsqueeze(1) # [7057, 1]

    train_attributes = torch.from_numpy(train_attributes) # train_attribute_values

    train_certainty = torch.from_numpy(train_certainty) # train_certainty_values

    # attributes
    all_attributes = np.array(attribute) # (200, 312)
    
    attributes = torch.from_numpy(all_attributes)
    # test set
    test_features = torch.from_numpy(x_test) # [2967, 2048]

    test_label = torch.from_numpy(test_label).unsqueeze(1) # [2967, 1]

    testclasses_id = np.array(test_id) # (50, )

    test_attributes = torch.from_numpy(att_pro).float() # (50, 312)

    
    test_seen_features = torch.from_numpy(x_test_seen) # [1764, 2048]
    
    test_seen_label = torch.from_numpy(test_label_seen) # [1764]
    
    #train_data = TensorDataset(train_features,train_label)
    train_data = IntegratedDataset(train_features, train_label, train_attributes, train_certainty)    

    # init network
    print("init networks")
    attribute_network = AttributeNetwork(2048,1200,624).cuda()
    relation_network = RelationNetwork(936, 400).cuda()

    mse = nn.MSELoss().cuda()
    ce = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1, 1.])).cuda()
    nll = torch.nn.NLLLoss(weight=torch.FloatTensor([0.1, 1.])).cuda()
    #mse = nn.BCELoss().cuda()

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    if MODEL == 1:
        attribute_network_optim = torch.optim.Adam(attribute_network.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        #attribute_network_optim = torch.optim.SGD(attribute_network.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4, nesterov=True)
        attribute_network_scheduler = StepLR(attribute_network_optim, step_size=30000, gamma=0.5)

        # if os.path.exists("./models/zsl_cub2_attribute_network_v35.pkl"):
        #     attribute_network.load_state_dict(torch.load("./models/zsl_cub_attribute_network_v35.pkl"))
        #     print("load attribute network success")
        # if os.path.exists("./models/zsl_cub2_relation_network_v35.pkl"):
        #     relation_network.load_state_dict(torch.load("./models/zsl_cub_relation_network_v35.pkl"))
        #     print("load relation network success")

        print("training...")
        #last_accuracy = 0.0
        total_steps = 0
        best_loss_att = 10000
        best_loss_cer = 10000

        for epoch in range(EPOCHS):
            attribute_network.train()
            for i, (batch_features, batch_labels, batch_att, batch_cer) in enumerate(train_loader):

                batch_features, batch_att, batch_cer = batch_features.float().cuda(), batch_att.float().cuda(), batch_cer.float().cuda()
                
                attribute_network_scheduler.step(total_steps)
                
                #relation_network_scheduler.step(episode)

                #sample_labels = set(batch_labels.squeeze().numpy().tolist())
                
                #sample_attributes = torch.Tensor([all_attributes[i] for i in sample_labels])
                
                #class_num = sample_attributes.shape[0]
                pred_embeddings = attribute_network(batch_features)

                cat_batch_att = batch_att.long().view(-1)
                cat_pred_att = torch.cat((pred_embeddings[:,0:312].unsqueeze(2), 1 - pred_embeddings[:,0:312].unsqueeze(2)), 2).view(-1, 2)
                
                #loss_att = ce(cat_pred_att, cat_batch_att)
                loss_att = nll(torch.log(cat_pred_att + 1e-10), cat_batch_att)
                loss_cer = mse(pred_embeddings[:,312:624], batch_cer)
                loss_net1 = loss_att + loss_cer

                attribute_network_optim.zero_grad()
                loss_net1.backward()
                attribute_network_optim.step()

                total_steps += 1

            #if epoch % 50 == 0:
            loss_att_mean, loss_cer_mean, acc_att, acc_cer = evaluate_attribute_network(attribute_network, nll, mse, train_loader)
            print("Epoch: {:>3} loss_att: {:.5f} loss_cer: {:.5f}".format(epoch, loss_att_mean, loss_cer_mean))
            print("Epoch: {:>3} acc_att: {:.5f} acc_cer: {:.5f}".format(epoch, acc_att, acc_cer))

            if best_loss_att > loss_att_mean and best_loss_cer > loss_cer_mean:
                torch.save(attribute_network.state_dict(), 'models/attribute_network.pt')
            if best_loss_att > loss_att_mean:
                best_loss_att = loss_att_mean
            if best_loss_cer > loss_cer_mean:
                best_loss_cer = loss_cer_mean

    else:
        attribute_network.load_state_dict(torch.load("./models/attribute_network.pt"))
        print("load attribute network success")

        loss_att_mean, loss_cer_mean = evaluate_attribute_network(attribute_network, ce, train_loader)
        print("loss_att: {:.5f} loss_cer: {:.5f}".format(loss_att_mean, loss_cer_mean))

        assert 1 == 0
        relation_network_optim = torch.optim.Adam(relation_network.parameters(),lr=LEARNING_RATE)
        relation_network_scheduler = StepLR(relation_network_optim, step_size=30000, gamma=0.5)



        
        batch_features = Variable(batch_features).cuda(GPU).float()  # 32*2048
        sample_features = attribute_network(Variable(sample_attributes).cuda(GPU)) #c*2048

        sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_SIZE,1,1)
        batch_features_ext = batch_features.unsqueeze(0).repeat(class_num,1,1)
        batch_features_ext = torch.transpose(batch_features_ext,0,1)
        
        #print(sample_features_ext)
        #print(batch_features_ext)
        relation_pairs = torch.cat((sample_features_ext,batch_features_ext),2).view(-1,4096)
        relations = relation_network(relation_pairs).view(-1,class_num)
        #print(relations)

        # re-build batch_labels according to sample_labels
        sample_labels = np.array(sample_labels)
        re_batch_labels = []
        for label in batch_labels.numpy():
            index = np.argwhere(sample_labels==label)
            re_batch_labels.append(index[0][0])
        re_batch_labels = torch.LongTensor(re_batch_labels)

        # loss
        # relations = nn.functional.softmax(relations, 1)
        mse = nn.MSELoss().cuda(GPU)
        one_hot_labels = Variable(torch.zeros(BATCH_SIZE, class_num).scatter_(1, re_batch_labels.view(-1,1), 1)).cuda(GPU)
        loss = mse(relations,one_hot_labels)

        # update
        attribute_network.zero_grad()
        relation_network.zero_grad()

        loss.backward()

        attribute_network_optim.step()
        relation_network_optim.step()

        # if (episode+1)%100 == 0:
        #         print("episode:",episode+1,"loss",loss.data[0])

        if (episode + 1) % 2000 == 0:
            # test
            print("Testing...")

            def compute_accuracy(test_features,test_label,test_id,test_attributes):
                
                test_data = TensorDataset(test_features,test_label)
                test_batch = 32
                test_loader = DataLoader(test_data,batch_size=test_batch,shuffle=False)
                total_rewards = 0
                # fetch attributes
                sample_labels = test_id
                sample_attributes = test_attributes
                class_num = sample_attributes.shape[0]
                test_size = test_features.shape[0]
                
                print("class num:",class_num)
                predict_labels_total = []
                re_batch_labels_total = []
                
                for batch_features,batch_labels in test_loader:

                    batch_size = batch_labels.shape[0]

                    batch_features = Variable(batch_features).cuda(GPU).float()  # 32*1024
                    sample_features = attribute_network(Variable(sample_attributes).cuda(GPU).float())

                    sample_features_ext = sample_features.unsqueeze(0).repeat(batch_size,1,1)
                    batch_features_ext = batch_features.unsqueeze(0).repeat(class_num,1,1)
                    batch_features_ext = torch.transpose(batch_features_ext,0,1)

                    relation_pairs = torch.cat((sample_features_ext,batch_features_ext),2).view(-1,4096)
                    relations = relation_network(relation_pairs).view(-1,class_num)

                    # re-build batch_labels according to sample_labels

                    re_batch_labels = []
                    for label in batch_labels.numpy():
                        index = np.argwhere(sample_labels==label)
                        re_batch_labels.append(index[0][0])
                    re_batch_labels = torch.LongTensor(re_batch_labels)

                    _,predict_labels = torch.max(relations.data,1)
                    predict_labels = predict_labels.cpu().numpy()
                    re_batch_labels = re_batch_labels.cpu().numpy()
                    
                    predict_labels_total = np.append(predict_labels_total, predict_labels)
                    re_batch_labels_total = np.append(re_batch_labels_total, re_batch_labels)

                # compute averaged per class accuracy    
                predict_labels_total = np.array(predict_labels_total, dtype='int')
                re_batch_labels_total = np.array(re_batch_labels_total, dtype='int')
                unique_labels = np.unique(re_batch_labels_total)
                acc = 0
                for l in unique_labels:
                    idx = np.nonzero(re_batch_labels_total == l)[0]
                    acc += accuracy_score(re_batch_labels_total[idx], predict_labels_total[idx])
                acc = acc / unique_labels.shape[0]
                return acc
            
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
