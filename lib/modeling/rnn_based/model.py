'''
main function of our action-intention detection model
Action head
Intention head
'''
import torch
import torch.nn as nn
# from .action_net import ActionNet
# from .intent_net import IntentNet
# from .action_intent_net import ActionIntentNet
# from lib.modeling.layers.attention import AdditiveAttention2D
# from lib.modeling.relation import RelationNet
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

        # self.bbox_embedding = nn.Sequential(nn.Linear(4, 256),
        #                                     nn.ReLU())
        # The classifier layer
        # self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        # self.intent_model = IntentNet(cfg, x_visual_extractor=None)
        self.word_embedding_dimension = 256
        self.sentence_size = 30

        self.conv1x1 = nn.Conv2d(in_channels=4, out_channels=256, kernel_size=[1,1])

        # self.out_channel = config.out_channel
        self.conv3 = nn.Conv2d(256, 1, (3, 1))
        self.conv4 = nn.Conv2d(256, 1, (4, 1))
        self.conv5 = nn.Conv2d(256, 1, (5, 1))
        self.conv6 = nn.Conv2d(256, 1, (6, 1))
        self.conv8 = nn.Conv2d(256, 1, (8, 1))
        self.conv10 = nn.Conv2d(256, 1, (10, 1))
        # self.conv9 = nn.Conv2d(1, 1, (13, self.word_embedding_dimension))

        # relu不带参数，所以不需要定义多个
        self.relu_ = nn.ReLU()

        self.Avg3_pool = nn.MaxPool2d((self.sentence_size-3+1, 1))
        self.Avg4_pool = nn.AvgPool2d((self.sentence_size-4+1, 1))
        self.Avg5_pool = nn.AvgPool2d((self.sentence_size-5+1, 1))
        self.Avg6_pool = nn.AvgPool2d((self.sentence_size-6+1, 1))
        self.Avg8_pool = nn.AvgPool2d((self.sentence_size-8+1, 1))
        self.Avg10_pool = nn.AvgPool2d((self.sentence_size-10+1, 1))
        # self.Max14_pool = nn.AvgPool2d((self.sentence_size-13+1, 1))
        
        # self.linear1 = nn.Linear(6, self.num_classes*self.sentence_size)
        
        self.lastconv1x1 = nn.Conv2d(in_channels=6, out_channels=30, kernel_size=[1,1])


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
        # elif self.cfg.MODEL.INPUT_LAYER == 'attention':
        #     self.x_visual_extractor = AdditiveAttention2D(self.cfg)
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
        '''
        NOTE: Action and Intent net use separate encoder networks for training only !
              x_bbox: bounding boxes(batch_size, SEG_LEN, 4)
        '''
        # batch = x_bbox.shape[0]
        # # print(f"========/nbefore embedding {x_bbox.shape}")
        # x_bbox = self.bbox_embedding(x_bbox)
        # # print(f"after embedding {x_bbox.shape}")
        # x_bbox = x_bbox.unsqueeze(1)
        # # print(f"after unsqueeze {x_bbox.shape}")

        # # x_bbox: [30, 1, 30, 256]
        
        # x_bbox = torch.reshape(x_bbox, (batch, 16, self.sentence_size,1))
        # # print(f"after reshape {x_bbox.shape}")

        x_bbox = self.conv1x1(x_bbox)
        # print(f"after conv1x1 {x_bbox.shape}")
        


        # Convolution
        x1 = self.relu_(self.conv3(x_bbox))
        # print(f"========\nafter conv3 {x1.shape}")
        x2 = self.relu_(self.conv4(x_bbox))
        # print(f"after conv4 {x2.shape}")
        x3 = self.relu_(self.conv5(x_bbox))
        # print(f"after conv5 {x3.shape}")
        x4 = self.relu_(self.conv6(x_bbox))
        # print(f"after conv6 {x4.shape}")
        x5 = self.relu_(self.conv8(x_bbox))
        # print(f"after conv8 {x5.shape}")
        x6 = self.relu_(self.conv10(x_bbox))
        # print(f"after conv10 {x6.shape}")


        # Pooling
        x1 = self.Avg3_pool(x1)
        # print(f"after x1 avgpool {x1.shape}")
        x2 = self.Avg4_pool(x2)
        # print(f"after x2 avgpool {x2.shape}")
        x3 = self.Avg5_pool(x3)
        # print(f"after x3 avgpool {x3.shape}")
        x4 = self.Avg6_pool(x4)
        # print(f"after x4 avgpool {x4.shape}")
        x5 = self.Avg8_pool(x5)
        # print(f"after x5 avgpool {x5.shape}")
        x6 = self.Avg10_pool(x6)
        # print(f"after x6 avgpool {x6.shape}")
        # x7 = self.Max14_pool(x7)


        # capture and concatenate the features
        x = torch.cat((x1, x2, x3,x4,x5,x6), 1)
        # print(f"========\nafter concat {x.shape} at dim-1")

        x = self.lastconv1x1(x)
        # print(f"after lastconv1x1 {x.shape}")

        # x = x.view(batch, 1, -1)
        # print(f"after view {x.shape}")

        # # project the features to the labels
        # x = self.linear1(x)
        # print(f"after linear {x.shape}")
        # # print(f"before permute x size {x.shape}")
        # # x = x.permute(0, 2, 1)
        # # print(f"after permute x size {x.shape}")
        x = torch.reshape(x,(x.shape[0],x.shape[1],x.shape[2]))
        # # x = x.view(-1, self.num_classes)

        # print(f"after reshape x size {x.shape}")

        return x,None


        # return intent_detection_scores, None

    def step_two_stream(self, int_hx, x_bbox=None, future_inputs=None):
        '''
        Directly call step when run inferencing.
        Params:
            x_visual:
            act_hx:
            int_hx:
            x_bbox:
            future_inputs:
        Return
        '''
        if x_bbox is not None:
            x_bbox = self.bbox_embedding(x_bbox)
        enc_int_score = None

        if 'intent' in self.cfg.MODEL.TASK:
            # print(f"shape {int_hx.size()} : {x_bbox.size()}")
            int_hx, enc_int_score = self.intent_model.step(
                enc_hx=int_hx, x_bbox=x_bbox)

        return int_hx, enc_int_score, future_inputs



if __name__ == '__main__':
    from configs import cfg
    test_model = ActionIntentionDetection()
    print(test_model)
    test_input = torch.randn(30,4,30,1)
    test_output = test_model(x_bbox=test_input)
