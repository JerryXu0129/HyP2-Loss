from config import *

class HyP(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        torch.manual_seed(seed)
        # Initialization
        self.proxies = torch.nn.Parameter(torch.randn(num_classes, num_bits).to(device))
        nn.init.kaiming_normal_(self.proxies, mode = 'fan_out')

    def forward(self, x = None, batch_y = None):
        P_one_hot = batch_y

        cos = F.normalize(x, p = 2, dim = 1).mm(F.normalize(self.proxies, p = 2, dim = 1).T)
        pos = 1 - cos
        neg = F.relu(cos - threshold)

        P_num = len(P_one_hot.nonzero())
        N_num = len((P_one_hot == 0).nonzero())
        pos_term = torch.where(P_one_hot  ==  1, pos.to(torch.float32), torch.zeros_like(cos).to(torch.float32)).sum() / P_num
        neg_term = torch.where(P_one_hot  ==  0, neg.to(torch.float32), torch.zeros_like(cos).to(torch.float32)).sum() / N_num
        if beta > 0:
            index = batch_y.sum(dim = 1) > 1
            y_ = batch_y[index].float()
            x_ = x[index]
            cos_sim = y_.mm(y_.T)
            if len((cos_sim == 0).nonzero()) == 0:
                reg_term = 0
            else:
                x_sim = F.normalize(x_, p = 2, dim = 1).mm(F.normalize(x_, p = 2, dim = 1).T)
                neg = beta * F.relu(x_sim - threshold)
                reg_term = torch.where(cos_sim == 0, neg, torch.zeros_like(x_sim)).sum() / len((cos_sim == 0).nonzero())
        else:
            reg_term = 0

        return pos_term + neg_term + reg_term


class AlexNet(nn.Module):
    def __init__(self, hash_bit, pretrained=True):
        super(AlexNet, self).__init__()

        model_alexnet = models.alexnet(pretrained=pretrained)
        self.features = model_alexnet.features
        cl1 = nn.Linear(256 * 6 * 6, 4096)
        cl1.weight = model_alexnet.classifier[1].weight
        cl1.bias = model_alexnet.classifier[1].bias

        cl2 = nn.Linear(4096, 4096)
        cl2.weight = model_alexnet.classifier[4].weight
        cl2.bias = model_alexnet.classifier[4].bias

        self.hash_layer = nn.Sequential(
            nn.Dropout(),
            cl1,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            cl2,
            nn.ReLU(inplace=True),
            nn.Linear(4096, hash_bit),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.hash_layer(x)
        return x