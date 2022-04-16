import torch
import torch.nn as nn
import torch.nn.functional as F

# ------ TO DO ------
class cls_model(nn.Module):
    def __init__(self, num_classes=3):
        super(cls_model, self).__init__()
        # Feature encoding
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        # Feature decoding
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.3)
        self.cls_bn1 = nn.BatchNorm1d(512)
        self.cls_bn2 = nn.BatchNorm1d(256)

        self.acti = nn.LeakyReLU(0.1)

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''
        feature = self.feature_encode(points)
        prob = self.classification(feature)
        return prob

    def feature_encode(self, points):
        x = points.transpose(1,2)
        x = self.acti(self.bn1(self.conv1(x)))
        x = self.acti(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        x = torch.max(x, 2, keepdim=True)[0]
        feature = x.view(-1, 1024)
        return feature 

    def classification(self, feature):
        x = self.acti(self.cls_bn1(self.fc1(feature)))
        x = self.acti(self.cls_bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        outputs = F.log_softmax(x, dim=1)
        return outputs



# ------ TO DO ------
class seg_model(nn.Module):
    def __init__(self, num_seg_classes = 6):
        super(seg_model, self).__init__()
        # Feature encoding
        self.acti = nn.LeakyReLU(0.1)

        self.encode_raw = nn.Sequential(
            torch.nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), self.acti,
            )
        self.encode_feature = nn.Sequential(
            torch.nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), self.acti,
            torch.nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), self.acti,
            torch.nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), 
        )
        self.segment = nn.Sequential(
            torch.nn.Conv1d(1088, 512, 1), nn.BatchNorm1d(512), self.acti,
            torch.nn.Conv1d(512, 256, 1), nn.BatchNorm1d(256), self.acti,
            torch.nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), self.acti,
            torch.nn.Conv1d(128, num_seg_classes, 1),
        )


    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, N, num_seg_classes)
        '''
        num_points = points.shape[1]
        x = points.transpose(1,2)
        raw_feat = self.encode_raw(x)
        glob_feat = self.glob_feature(raw_feat, num_points)
        point_feat = torch.cat([raw_feat, glob_feat], dim = 1)
        return self.segment(point_feat)

    def glob_feature(self, feat, num_points):
        feat = self.encode_feature(feat)
        feat = torch.max(feat, 2, keepdim=True)[0]# (B,1024)
        feat = feat.view(-1, 1024)
        feat = feat.repeat(num_points, 1, 1)# (N, B,1024)
        feat = feat.permute(1,2,0)# (B, N,1024)
        return feat# (B, N,1024)