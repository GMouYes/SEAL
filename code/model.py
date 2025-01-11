import torch
import torch.nn as nn
from transformers import BertModel

def save_model(model, args):
    """Save model."""
    torch.save(model.state_dict(), args["outputPath"] + '/' + args["modelPath"])
    print("Saved better model selected by validation.")
    return True


class DecisionMaking(nn.Module):
    """docstring for DecisionMaking"""
    def __init__(self, args):
        super(DecisionMaking, self).__init__()
        self.args = args
        self.TextEncoder = TextEncoder(args)
        self.InstanceEncoder = InstanceEncoderMLP(args)

    def loss_bce(self, pred, label, weight):
        # loss_fn1 = nn.BCELoss()
        # loss_fn2 = nn.MSELoss()
        rows, columns = label.shape
        users, phonePlacements, activity = self.args["users"], self.args["phonePlacements"], self.args["activities"]
        

        if self.args["predict_user"]:
            label_user = label[:, :users]
            label_pp = label[:, users:users+phonePlacements]
            label_act = label[:, -activity:]

            pred_user = pred[:, :users]
            pred_pp = pred[:, users:users+phonePlacements]
            pred_act = pred[:, -activity:]

            weight_pp = weight[:, users:users+phonePlacements]
            weight_act = weight[:, -activity:]

            # each user has equal weight(mask), weight in ce is class weight
            loss_user_fn = nn.CrossEntropyLoss(reduction=self.args["reduction"])
            loss_user = loss_user_fn(pred_user, torch.argmax(label_user,dim=-1))

            # pp has different weights(mask) to represent missing labels
            loss_pp_fn = nn.BCEWithLogitsLoss(reduction=self.args["reduction"], weight=weight_pp)
            loss_pp = loss_pp_fn(pred_pp, label_pp)

            # act also has missing labels
            loss_act_fn = nn.BCEWithLogitsLoss(reduction=self.args["reduction"], weight=weight_act)
            loss_act = loss_act_fn(pred_act, label_act)

            # act is the main target, the other two are contexts
            cls_loss = self.args["lambda_user"] * loss_user + self.args["lambda_pp"] * loss_pp + loss_act
            return cls_loss

        else:
            label_no_user = label[:, users:]
            # print("shape================",label_no_user.shape, pred.shape)
            loss_fn = nn.BCEWithLogitsLoss(reduction=self.args["reduction"], weight=weight[:, users:])
            return loss_fn(pred, label_no_user)


    def loss(self, pred, label, weight):
        return self.loss_bce(pred, label, weight)
        
    def forward(self, x, text):
        t = self.TextEncoder(text)
        x = self.InstanceEncoder(x)

        # matrix multiplication for final prediction
        result = torch.mm(x, t.T)
        # if self.args["predict_user"]:
        #     result = torch.cat(result, axis=1)
        # else:
        #     # we didn't include userID in our text encoder
        #     result = torch.cat(result, axis=1)
        return result

class InstanceEncoderLSTM(nn.Module):
    """docstring for InstanceEncoder"""
    def __init__(self, args):
        super(InstanceEncoderLSTM, self).__init__()
        self.args = args

        self.lstm = nn.LSTM(
            input_size=args["raw_dim"],
            hidden_size=args["hidden_dim"],
            num_layers=args["lstm_num_layers"],
            batch_first=True,
            dropout=args["lstm_dropout"],
            bidirectional=args["lstm_bidirectional"],
            )
        # map x_dim to commonD
        if args["lstm_bidirectional"]:
            self.x_dim = 2 * args["hidden_dim"]
        else:
            self.x_dim = args["hidden_dim"]
        self.x2c = nn.ModuleList([nn.Linear(self.x_dim, int(args["model_commonDim"])) for _ in range(3)])
        self.x2c_act = nn.LeakyReLU(args["model_leakySlope_x"])

    def forward(self, raw_feature):
        N, L, raw_dim = raw_feature.shape
        raw_out, _ = self.lstm(raw_feature)
        x_ = raw_out[:, -1, :].reshape(N, -1)

        x = [self.x2c_act(layer(x_)) for i, layer in enumerate(self.x2c)]

        return x

class InstanceEncoderMLP(nn.Module):
    """ """
    def __init__(self, args):
        super(InstanceEncoderMLP, self).__init__()
        self.args = args
        hidden_dim_1 = int(args["hidden_dim_1"])
        hidden_dim_2 = int(args["hidden_dim_2"])
        self.linear1 = nn.Linear(args["input_dim"], hidden_dim_1)
        self.act1 = nn.LeakyReLU(args["leakySlope"])
        self.dropout1 = nn.Dropout(args["dropout"])

        self.linear2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.act2 = nn.LeakyReLU(args["leakySlope"])
        self.dropout2 = nn.Dropout(args["dropout"])

        self.linear3 = nn.Linear(hidden_dim_2, int(args["model_commonDim"]))
        self.act3 = nn.LeakyReLU(args["leakySlope"])
        self.dropout3 = nn.Dropout(args["dropout"])

        self.mlp = nn.Sequential(
            self.linear1,
            self.dropout1,
            self.act1,
            self.linear2,
            self.dropout2,
            self.act2,
            self.linear3,
            self.dropout3,
            self.act3,
        )

    def forward(self, x):
        return self.mlp(x)

class TextEncoder(nn.Module):
    """ """
    def __init__(self, args):
        super(TextEncoder, self).__init__()
        self.args = args
        self.model = BertModel.from_pretrained("google-bert/bert-base-uncased")

        self.linear1 = nn.Linear(args["bert_dim"], int(args["model_commonDim"]))
        self.dropout = nn.Dropout(args["dropout"])
        self.act = nn.LeakyReLU(args["leakySlope"])
        self.mlp = nn.Sequential(
            self.linear1,
            self.dropout,
            self.act,
        )


    def forward(self, x):
        words = self.model(**x) # words[0]: (pp_a, 15, 768), words[1]:(pp_a,768)
        return self.mlp(words[1]) # (pp_a,768) --> (pp_a, commonD)




