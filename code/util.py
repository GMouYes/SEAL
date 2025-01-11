from sklearn.metrics import balanced_accuracy_score as BA
from sklearn.metrics import matthews_corrcoef as MCC
from sklearn.metrics import f1_score as F1
import numpy as np
import random
import time
import torch
from yaml import safe_load, safe_dump

labelText = [
    "the user has phone in pocket",
    "the user has phone in hand",
    "the user has phone in bag",
    "the user has phone on the table",
    "we do not know where the user put their phone",
    # check the above sentence
    "the user is lying down",
    "the user is sitting",
    "the user is walking",
    "the user is sleeping",
    "the user is talking",
    "the user is in bathroom taking a shower",
    "the user is in toilet",
    "the user is standing",
    "the user is running",
    "the user is going downstairs",
    "the user is going upstairs",
    "the user is doing exercise",
]


def seed_all(seed):
    # seed everything
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    return True


def timing(func):
    def wrapper(args):
        print("Running function {}".format(func.__name__))
        t1 = time.time()
        res = func(args)
        t2 = time.time()
        period = t2 - t1
        print("{} took {} hour {} min {} sec".format(func.__name__, period // 3600, (period % 3600) // 60,
                                                     int(period % 60)))
        return res

    return wrapper

def read_config(path):
    with open(path, 'r') as f:
        return safe_load(f)

def save_config(config, path):
    with open(path, "w") as f:
        safe_dump(config, f)
    return True

# must rewrite this one
def metric(label, pred, args, mask=None): 
    rows, columns = label.shape
    users, phonePlacements, activity = args["users"], args["phonePlacements"], args["activities"]

    if args["predict_user"]:
        # user label 
        label_user = label[:, :users]
        label_user = torch.argmax(label_user, dim=-1)
        # user pred
        pred_user = pred[:, :users]
        pred_user = torch.argmax(pred_user, dim=-1)
        # user metric
        user_BA = BA(label_user, pred_user)
        user_MCC = MCC(label_user, pred_user)
        user_F1 = F1(label_user, pred_user, average='macro')

        # pp label
        label_pp = label[:, users:users+phonePlacements]
        # pp pred
        pred_pp = pred[:, users:users+phonePlacements]
        pred_pp = torch.sigmoid(pred_pp)
        pred_pp = (pred_pp>0.5).long()
        # pp mask
        mask_pp = mask[:, users:users+phonePlacements]
        # pp metric
        pp_BA, pp_MCC, pp_F1 = [], [], []
        for i in range(phonePlacements):
            act_label = label_pp[:, i]
            act_pred = pred_pp[:, i]
            act_mask_tmp = mask_pp[:, i] > 0

            act_target = [(item1, item2) for item1, item2, item3 in zip(act_label, act_pred, act_mask_tmp) if item3]
            act_label, act_pred = list(zip(*act_target))

            pp_BA.append(BA(act_label, act_pred))
            pp_MCC.append(MCC(act_label, act_pred))
            pp_F1.append(F1(act_label, act_pred, average='macro'))

        # act label
        label_act = label[:, -activity:]
        # act pred
        pred_act = pred[:, -activity:]
        pred_act = torch.sigmoid(pred_act)
        pred_act = (pred_act>0.5).long()
        # act mask
        mask_act = mask[:, -activity:]
        # act metric
        act_BA, act_MCC, act_F1 = [], [], []
        for i in range(activity):
            act_label = label_act[:, i]
            act_pred = pred_act[:, i]
            act_mask_tmp = mask_act[:, i] > 0

            act_target = [(item1, item2) for item1, item2, item3 in zip(act_label, act_pred, act_mask_tmp) if item3]
            act_label, act_pred = list(zip(*act_target))

            act_BA.append(BA(act_label, act_pred))
            act_MCC.append(MCC(act_label, act_pred))
            act_F1.append(F1(act_label, act_pred, average='macro'))

        targetBA = [user_BA] + pp_BA + act_BA
        targetMCC = [user_MCC] + pp_MCC + act_MCC
        targetF1 = [user_F1] + pp_F1 + act_F1

    else:
        label = label[:, users:]
        mask = mask[:, users:]
        users = 0
    
        target_pred = pred[:, users:]
        target_label = label[:, users:]
        target_pred = torch.sigmoid(target_pred)
        target_pred = (target_pred > 0.5).long()

        if mask is None:
            targetBA = [BA(target_label[:, i], target_pred[:, i]) for i in range(args["activities"]+args["phonePlacements"])]
            targetMCC = [MCC(target_label[:, i], target_pred[:, i]) for i in range(args["activities"]+args["phonePlacements"])]
            targetF1 = [F1(target_label[:, i], target_pred[:, i], average='macro') for i in range(args["activities"]+args["phonePlacements"])]
        else:
            act_mask = mask[:, users:]
            targetBA, targetMCC, targetF1 = [], [], []
            for i in range(args["activities"]+args["phonePlacements"]):
                act_label = target_label[:, i]
                act_pred = target_pred[:, i]
                act_mask_tmp = act_mask[:, i] > 0

                act_target = [(item1, item2) for item1, item2, item3 in zip(act_label, act_pred, act_mask_tmp) if item3]
                act_label, act_pred = list(zip(*act_target))
                # act_label = [item1 for item1, item2 in zip(act_label, act_mask_tmp) if item2]
                # act_pred = [item1 for item1, item2 in zip(act_pred, act_mask_tmp) if item2]
                targetBA.append(BA(act_label, act_pred))
                targetMCC.append(MCC(act_label, act_pred))
                targetF1.append(F1(act_label, act_pred, average='macro'))


    return np.array(targetBA), np.array(targetMCC), np.array(targetF1)