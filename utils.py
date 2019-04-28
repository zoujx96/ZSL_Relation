import torch
import numpy as np

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
            re_batch_labels = torch.LongTensor(re_batch_labels)
            one_hot_labels = torch.zeros(len(batch_att), class_num).scatter_(1, re_batch_labels.view(-1, 1), 1).cuda()
            
            loss_rel = criteria(relations, one_hot_labels)
            batch_correct_rel = (torch.argmax(relations, 1) == re_batch_labels.cuda()).sum()
            '''
            if i == 5:
                print(pred_embeddings[14])
                print(batch_att[14])
            '''
            total_loss_rel += loss_rel.item() * class_num * len(batch_att)
            total_correct_num_rel += batch_correct_rel.item()
            sample_num += len(batch_att)
            total_class_num += class_num * len(batch_att)

    loss_rel_mean = total_loss_rel / total_class_num
    acc_rel = total_correct_num_rel / sample_num
    return loss_rel_mean, acc_rel

def compute_accuracy_whole_network(test_loader, test_id, test_attributes):
    train_data = IntegratedDataset(test_features, test_label, train_attributes)
    test_data = IntegratedDataset(test_features,test_label)
    test_batch = 32
    test_loader = DataLoader(test_data,batch_size=test_batch,shuffle=False)
    total_rewards = 0
    # fetch attributes
    sample_labels = test_id
    sample_attributes = test_attributes
    class_num = sample_attributes.shape[0]
    
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