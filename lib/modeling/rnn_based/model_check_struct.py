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
    def __init__(self, parameter_scheduler=None):
        super().__init__()
        # self.cfg = cfg
        self.parameter_scheduler = parameter_scheduler
        self.hidden_size = 256 #self.cfg.MODEL.HIDDEN_SIZE
        self.pred_len = 5 #self.cfg.MODEL.PRED_LEN
        self.num_classes = 2
        # if self.num_classes == 2 and self.cfg.MODEL.INTENT_LOSS == 'bce':
        #     self.num_classes = 1

        self.bbox_embedding = nn.Sequential(nn.Linear(4, 256),
                                            nn.ReLU())
        # The classifier layer
        # self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        # self.intent_model = IntentNet(cfg, x_visual_extractor=None)
        self.word_embedding_dimension = 256
        self.sentence_size = 30

        # self.out_channel = config.out_channel
        self.conv3 = nn.Conv2d(16, 1, (3, self.word_embedding_dimension//16))
        self.conv4 = nn.Conv2d(16, 1, (4, self.word_embedding_dimension//16))
        self.conv5 = nn.Conv2d(16, 1, (5, self.word_embedding_dimension//16))
        self.conv6 = nn.Conv2d(16, 1, (6, self.word_embedding_dimension//16))
        self.conv7 = nn.Conv2d(16, 1, (8, self.word_embedding_dimension//16))
        self.conv8 = nn.Conv2d(16, 1, (10, self.word_embedding_dimension//16))
        # self.conv9 = nn.Conv2d(1, 1, (13, self.word_embedding_dimension))

        self.Max3_pool = nn.AvgPool2d((self.sentence_size-3+1, 1))
        self.Max4_pool = nn.AvgPool2d((self.sentence_size-4+1, 1))
        self.Max5_pool = nn.AvgPool2d((self.sentence_size-5+1, 1))
        self.Max6_pool = nn.AvgPool2d((self.sentence_size-6+1, 1))
        self.Max8_pool = nn.AvgPool2d((self.sentence_size-8+1, 1))
        self.Max10_pool = nn.AvgPool2d((self.sentence_size-10+1, 1))
        # self.Max14_pool = nn.AvgPool2d((self.sentence_size-13+1, 1))
        
        self.linear1 = nn.Linear(6, self.num_classes*self.sentence_size)


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

        return self.forward_two_stream(x_bbox=x_bbox,masks=masks)

    def forward_two_stream(self, x_bbox=None, masks=None):
        '''
        NOTE: Action and Intent net use separate encoder networks for training only !
        x_bbox: bounding boxes(batch_size, SEG_LEN, 4)
        '''
        # seg_len = x_visual.shape[1]
        # int_hx = self._init_hidden_states(x_visual, net_type=self.cfg.MODEL.INTENT_NET,  task_exists='intent' in self.cfg.MODEL.TASK)
        # future_inputs = None
        # int_hx = self._init_hidden_states(
        #     self.cfg.MODEL.SEG_LEN, net_type=self.cfg.MODEL.ACTION_NET,  task_exists=True)
        # int_hx = int_hx.cuda()
        # intent_detection_scores = []


        # for t in range(self.cfg.MODEL.SEG_LEN):
        #     # dec_input = dec_inputs[:, t] if dec_inputs else None
        #     ret = self.step_two_stream(int_hx, x_bbox[:, t], future_inputs)
        #     int_hx, enc_int_score, future_inputs = ret

        #     if enc_int_score is not None:
        #         intent_detection_scores.append(enc_int_score)

        # intent_detection_scores = torch.stack(
        #     intent_detection_scores, dim=1) if intent_detection_scores else None
        print(f"========\nbefore embedding {x_bbox.shape}")
        x_bbox = self.bbox_embedding(x_bbox)
        print(f"after embedding {x_bbox.shape}")
        x_bbox = x_bbox.unsqueeze(1)
        print(f"after unsqueeze {x_bbox.shape}")

        # x_bbox: [30, 1, 30, 256] [batch_size, 1, frame_length ,embedding_length]
        batch = x_bbox.shape[0]
        x_bbox = torch.reshape(x_bbox, (batch, 16, self.sentence_size,self.word_embedding_dimension//16))
        print(f"after reshape {x_bbox.shape}")
        


        # Convolution
        x1 = F.relu(self.conv3(x_bbox))
        print(f"========\nafter conv3 {x1.shape}")
        x2 = F.relu(self.conv4(x_bbox))
        print(f"after conv4 {x2.shape}")
        x3 = F.relu(self.conv5(x_bbox))
        print(f"after conv5 {x3.shape}")
        x4 = F.relu(self.conv6(x_bbox))
        print(f"after conv6 {x4.shape}")
        x5 = F.relu(self.conv7(x_bbox))
        print(f"after conv7 {x5.shape}")
        x6 = F.relu(self.conv8(x_bbox))
        print(f"after conv8 {x6.shape}")
        # x7 = F.relu(self.conv9(x_bbox))

        # Pooling
        x1 = self.Max3_pool(x1)
        print(f"after x1 maxpool {x1.shape}")
        x2 = self.Max4_pool(x2)
        print(f"after x2 avgpool {x2.shape}")
        x3 = self.Max5_pool(x3)
        print(f"after x3 avgpool {x3.shape}")
        x4 = self.Max6_pool(x4)
        print(f"after x4 avgpool {x4.shape}")
        x5 = self.Max8_pool(x5)
        print(f"after x5 avgpool {x5.shape}")
        x6 = self.Max10_pool(x6)
        print(f"after x6 avgpool {x6.shape}")
        # x7 = self.Max14_pool(x7)


        # capture and concatenate the features
        x = torch.cat((x1, x2, x3,x4,x5,x6), -1)
        print(f"========\nafter concat {x.shape}")

        x = x.view(batch, 1, -1)
        print(f"after view {x.shape}")

        # project the features to the labels
        x = self.linear1(x)
        print(f"after linear {x.shape}")
        # print(f"before x size {x.shape}")
        # x = x.permute(0, 2, 1)
        # x = x.view(-1, self.num_classes)
        # intent_detection_scores = x
        # intent_detection_scores = torch.stack(
        #     intent_detection_scores, dim=1) if intent_detection_scores else None
        # print(f"x size {x.shape}")

        # intent_detection_scores = []
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
    # from configs import cfg
    test_model = ActionIntentionDetection()
    print(test_model)