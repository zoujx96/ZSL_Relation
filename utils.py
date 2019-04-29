import torch
import numpy as np
from sklearn.metrics import accuracy_score

def evaluate_attribute_network(model, criteria, eval_loader):
    model.eval()
    total_loss_att = 0.
    total_correct_num_att = 0.
    sample_num = 0
    with torch.no_grad():
        for i, (batch_features, batch_labels, batch_att) in enumerate(eval_loader):
            batch_features, batch_att = batch_features.float().cuda(), batch_att.float().cuda()
            pred_embeddings = model(batch_features)

            cat_batch_att = batch_att.long().view(-1)
            cat_pred_att = torch.cat((pred_embeddings.unsqueeze(2), 1 - pred_embeddings.unsqueeze(2)), 2).view(-1, 2)

            loss_att = criteria(torch.log(cat_pred_att + 1e-10), cat_batch_att)

            batch_correct_att = ((pred_embeddings < 0.5).int() == batch_att.int()).float().sum()
            '''
            if i == 5:
                print(pred_embeddings[14])
                print(batch_att[14])
            '''
            total_loss_att += loss_att.item() * 312 * len(batch_att)
            total_correct_num_att += batch_correct_att
            sample_num += len(batch_att)

    loss_att_mean = total_loss_att / (sample_num * 312)
    acc_att = total_correct_num_att / (sample_num * 312)
    return loss_att_mean, acc_att

def evaluate_relation_network(model, criteria, eval_loader, all_embeddings):
    model.eval()
    total_loss_rel = 0.
    total_correct_num_rel = 0
    sample_num = 0
    total_class_num = 0
    with torch.no_grad():
        for i, (batch_features, batch_labels, batch_att) in enumerate(eval_loader):
            batch_att = batch_att.float().cuda()

            sample_labels = np.unique(batch_labels.squeeze().numpy())
                    
            #sample_embeddings = torch.Tensor([all_embeddings[i] for i in sample_labels]).cuda()
            sample_embeddings = all_embeddings[sample_labels]
            sample_embeddings = torch.from_numpy(sample_embeddings).float().cuda()

            class_num = sample_embeddings.shape[0]

            embeddings_bunch = sample_embeddings.unsqueeze(0).repeat(len(batch_att), 1, 1)
            attributes_bunch = batch_att.unsqueeze(0).repeat(class_num, 1, 1)
            attributes_bunch = torch.transpose(attributes_bunch, 0, 1)
            
            relation_pairs = torch.cat((embeddings_bunch, attributes_bunch), 2).view(-1, 624)
            relations = model(relation_pairs).view(-1, class_num)

            # re-build batch_labels according to sample_labels
            re_batch_labels = []
            for label in batch_labels.numpy():
                index = np.argwhere(sample_labels == label)
                re_batch_labels.append(index[0][0])
            re_batch_labels = torch.LongTensor(re_batch_labels).cuda()
            #one_hot_labels = torch.zeros(len(batch_att), class_num).scatter_(1, re_batch_labels.view(-1, 1), 1).cuda()
            
            #loss_rel = criteria(relations, one_hot_labels)
            loss_rel = criteria(relations, re_batch_labels)
            batch_correct_rel = (torch.argmax(relations, 1) == re_batch_labels.cuda()).sum()
            '''
            if i == 5:
                print(pred_embeddings[14])
                print(batch_att[14])
            '''
            #total_loss_rel += loss_rel.item() * class_num * len(batch_att)
            total_loss_rel += loss_rel.item() * len(batch_att)
            total_correct_num_rel += batch_correct_rel.item()
            sample_num += len(batch_att)
            #total_class_num += class_num * len(batch_att)

    #loss_rel_mean = total_loss_rel / total_class_num
    loss_rel_mean = total_loss_rel / sample_num
    acc_rel = total_correct_num_rel / sample_num
    return loss_rel_mean, acc_rel

def compute_accuracy_relation_network(model, loader, label_set, all_embeddings):
    model.eval()
    sample_embeddings = all_embeddings[label_set]
    sample_embeddings = torch.from_numpy(sample_embeddings).float().cuda()
    class_num = sample_embeddings.shape[0]
    predict_labels_total = []
    true_labels_total = []
    
    for i, (batch_features, batch_labels, batch_att) in enumerate(loader):
        batch_att = batch_att.float().cuda()

        embeddings_bunch = sample_embeddings.unsqueeze(0).repeat(len(batch_att), 1, 1)
        attributes_bunch = batch_att.unsqueeze(0).repeat(class_num, 1, 1)
        attributes_bunch = torch.transpose(attributes_bunch, 0, 1)

        relation_pairs = torch.cat((embeddings_bunch, attributes_bunch), 2).view(-1, 624)
        relations = model(relation_pairs).view(-1, class_num)
        # re-build batch_labels according to sample_labels
        re_batch_labels = []
        for label in batch_labels.numpy():
            index = np.argwhere(label_set == label)
            re_batch_labels.append(index[0][0])

        predict_labels = torch.argmax(relations, 1).cpu().numpy()
        true_labels = np.array(re_batch_labels)
        
        predict_labels_total.extend(predict_labels)
        true_labels_total.extend(true_labels)

    # compute averaged per class accuracy    
    predict_labels_total = np.array(predict_labels_total)
    true_labels_total = np.array(true_labels_total)
    unique_labels = np.unique(true_labels_total)
    
    acc = 0
    for l in unique_labels:
        idx = np.nonzero(true_labels_total == l)[0]
        acc += accuracy_score(true_labels_total[idx], predict_labels_total[idx])
    acc = acc / unique_labels.shape[0]
    return acc

def compute_accuracy_whole_network(att_net, rel_net, loader, label_set, all_embeddings):
    att_net.eval()
    rel_net.eval()

    sample_embeddings = all_embeddings[label_set]
    sample_embeddings = torch.from_numpy(sample_embeddings).float().cuda()
    class_num = sample_embeddings.shape[0]
    predict_labels_total = []
    true_labels_total = []
    
    for i, (batch_features, batch_labels, batch_att) in enumerate(loader):
        batch_features = batch_features.float().cuda()
        pred_embeddings = att_net(batch_features)

        embeddings_bunch = sample_embeddings.unsqueeze(0).repeat(len(batch_features), 1, 1)
        attributes_bunch = pred_embeddings.unsqueeze(0).repeat(class_num, 1, 1)
        attributes_bunch = torch.transpose(attributes_bunch, 0, 1)

        relation_pairs = torch.cat((embeddings_bunch, attributes_bunch), 2).view(-1, 624)
        relations = rel_net(relation_pairs).view(-1, class_num)
        # re-build batch_labels according to sample_labels
        re_batch_labels = []
        for label in batch_labels.numpy():
            index = np.argwhere(label_set == label)
            re_batch_labels.append(index[0][0])

        predict_labels = torch.argmax(relations, 1).cpu().numpy()
        true_labels = np.array(re_batch_labels)
        
        predict_labels_total.extend(predict_labels)
        true_labels_total.extend(true_labels)

    # compute averaged per class accuracy    
    predict_labels_total = np.array(predict_labels_total)
    true_labels_total = np.array(true_labels_total)
    unique_labels = np.unique(true_labels_total)
    
    acc = 0
    for l in unique_labels:
        idx = np.nonzero(true_labels_total == l)[0]
        acc += accuracy_score(true_labels_total[idx], predict_labels_total[idx])
    acc = acc / unique_labels.shape[0]
    return acc