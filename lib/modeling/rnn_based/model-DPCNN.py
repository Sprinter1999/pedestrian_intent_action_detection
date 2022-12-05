'''
main function of our action-intention detection model
Action head
Intention head
'''
import torch
import torch.nn as nn
from .action_net import ActionNet
from .intent_net import IntentNet
from .action_intent_net import ActionIntentNet
from lib.modeling.layers.attention import AdditiveAttention2D
from lib.modeling.relation import RelationNet
import torch.nn.functional as F


class ActionIntentionDetection(nn.Module):
    def __init__(self, cfg, parameter_scheduler=None):
        super().__init__()
        self.cfg = cfg
        self.parameter_scheduler = parameter_scheduler
        self.hidden_size = self.cfg.MODEL.HIDDEN_SIZE
        self.pred_len = self.cfg.MODEL.PRED_LEN
        self.num_classes = self.cfg.DATASET.NUM_INTENT
        if self.num_classes == 2 and self.cfg.MODEL.INTENT_LOSS == 'bce':
            self.num_classes = 1

        self.bbox_embedding = nn.Sequential(nn.Linear(4, 272),
                                            nn.ReLU())
        # The classifier layer
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        self.intent_model = IntentNet(cfg, x_visual_extractor=None)
        self.word_embedding_dimension = 272
        self.sentence_size = 30

        self.channel_size = 250
        self.conv_region_embedding = nn.Conv2d(1, self.channel_size, (3, self.word_embedding_dimension), stride=1)
        self.conv3 = nn.Conv2d(self.channel_size, self.channel_size, (3, 1), stride=1)
        self.pooling = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding_conv = nn.ZeroPad2d((0, 0, 1, 1))
        self.padding_pool = nn.ZeroPad2d((0, 0, 0, 1))
        self.act_fun = nn.ReLU()
        self.linear_out = nn.Linear(2*self.channel_size, self.num_classes*self.sentence_size)


        
        # self.linear1 = nn.Linear(4, self.num_classes*self.sentence_size)


    def _init_visual_extractor(self):
        if self.cfg.MODEL.INPUT_LAYER == 'avg_pool':
            self.x_visual_extractor = nn.Sequential(nn.Dropout2d(0.4),
                                                    nn.AvgPool2d(kernel_size=[7,7], stride=(1,1)),
                                                    nn.Flatten(start_dim=1, end_dim=-1),
                                                    nn.Linear(512, 128),
                                                    nn.ReLU())
        elif self.cfg.MODEL.INPUT_LAYER == 'conv2d':
            self.x_visual_extractor = nn.Sequential(nn.Dropout2d(0.4),
                                                    nn.Conv2d(in_channels=512, out_channels=64, kernel_size=[2,2]),
                                                    nn.Flatten(start_dim=1, end_dim=-1),
                                                    nn.ReLU())
        elif self.cfg.MODEL.INPUT_LAYER == 'attention':
            self.x_visual_extractor = AdditiveAttention2D(self.cfg)
        else:
            raise NameError(self.cfg.MODEL.INPUT_LAYER)

    def _init_hidden_states(self, x, net_type='gru', task_exists=True):
        batch_size = x.shape[0]
        if not task_exists:
            return None
        elif 'gru' in net_type:
            return torch.zeros(batch_size, self.hidden_size)
        else:
            raise ValueError(net_type)

    def forward(self,x_bbox=None,masks=None):

        return self.forward_two_stream(x_bbox=x_bbox,masks=masks)

    def forward_two_stream(self, x_bbox=None, masks=None):
        '''
        NOTE: Action and Intent net use separate encoder networks for training only !
        x_bbox: bounding boxes(batch_size, SEG_LEN, 4)
        '''

        x_bbox = self.bbox_embedding(x_bbox)
        x_bbox = x_bbox.unsqueeze(1)


        # x_bbox: [30, 30, 272]
        batch = x_bbox.shape[0]


        # Convolution
        x = self.conv_region_embedding(x_bbox)        # [batch_size, channel_size, length, 1]

        x = self.padding_conv(x)                      # pad保证等长卷积，先通过激活函数再卷积
        x = self.act_fun(x)
        x = self.conv3(x)
        x = self.padding_conv(x)
        x = self.act_fun(x)
        x = self.conv3(x)

        while x.size()[-2] > 2:
            x = self._block(x)

        x = x.view(batch, 2*self.channel_size)
        x = self.linear_out(x)

        return x,None

        # print(f"x size {x.size}")
        return x,None

        # return intent_detection_scores, None

    def _block(self, x):
        # Pooling
        x = self.padding_pool(x)
        px = self.pooling(x)

        # Convolution
        x = self.padding_conv(px)
        x = F.relu(x)
        x = self.conv3(x)

        x = self.padding_conv(x)
        x = F.relu(x)
        x = self.conv3(x)

        # Short Cut
        x = x + px

        return x

    def predict(self, x):
        self.eval()
        out = self.forward(x)
        predict_labels = torch.max(out, 1)[1]
        self.train(mode=True)
        return predict_labels
