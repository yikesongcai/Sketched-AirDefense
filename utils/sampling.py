import numpy as np
from torchvision import datasets, transforms

def iid(dataset, num_users):
    """
    Sample I.I.D. client data from dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    alph = np.ones(num_users)
    alph = alph/num_users
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]

    for i in range(num_users):
        # 动态计算每个用户的样本数，避免样本不足
        sample_size = min(num_items, len(all_idxs))
        dict_users[i] = set(np.random.choice(all_idxs, sample_size, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])

    return alph, dict_users

def noniid(dataset, args, non_iid_p):
    """
    Sample non-I.I.D client data from dataset
    -> Different clients can hold vastly different amounts of data
    :param dataset:
    :param num_users:
    :return:
    """
    dict_users = {i: list() for i in range(args.num_users)}
    
    min_num = 800
    max_num = 800
    random_num_size = np.random.randint(min_num, max_num+1, size=args.num_users)
    print("Total number of datasets owned by clients : {sum(random_num_size)}")

    alpha = np.zeros(args.num_users)
    alpha = random_num_size / sum(random_num_size)
    datasize = sum(random_num_size)

    #divide the dataset according to labels
    idxs = np.arange(len(dataset))

    if args.dataset == "mnist":
        labels = dataset.targets.numpy()
    elif args.dataset == "cifar":
        labels = np.array(dataset.targets)
    else:
        exit('Error: unrecognized dataset')
    
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    
    idxs = idxs_labels[0]
    labels = idxs_labels[1]
    
    data_with_label = [[] for i in range(args.num_classes)]

    for i in range(args.num_classes):
        specific_class = np.extract(labels == i, idxs)
        data_with_label[i].extend(specific_class)


    # total dataset should be larger or equal to sum of splitted dataset.
    assert len(dataset) >= sum(random_num_size)

    # divide and assign
    for i, rand_num in enumerate(random_num_size):
        
        for j in range(args.num_classes):
            rand_num_i = int(rand_num * non_iid_p[i][j])
            rand_set = set(np.random.choice(data_with_label[j], rand_num_i, replace=False))
            data_with_label[j] = list(set(data_with_label[j]) - set(rand_set))
            dict_users[i].extend(rand_set)

    return alpha, dict_users, datasize

