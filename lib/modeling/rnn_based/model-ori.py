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


class ActionIntentionDetection(nn.Module):
    def __init__(self, cfg, parameter_scheduler=None):
        super().__init__()
        self.cfg = cfg
        self.parameter_scheduler = parameter_scheduler
        self.hidden_size = self.cfg.MODEL.HIDDEN_SIZE

        self.bbox_embedding = nn.Sequential(nn.Linear(4, 256),
                                            nn.ReLU())
        # if self.cfg.MODEL.WITH_TRAFFIC:
        #     self.relation_model = RelationNet(cfg)

        self.x_visual_extractor = None
        if 'action' in self.cfg.MODEL.TASK and 'intent' in self.cfg.MODEL.TASK and 'single' in self.cfg.MODEL.TASK:
            if 'convlstm' not in self.cfg.MODEL.INTENT_NET:
                self._init_visual_extractor()
            self.action_intent_model = ActionIntentNet(
                cfg, x_visual_extractor=None)
        else:
            if 'action' in self.cfg.MODEL.TASK:
                if 'convlstm' not in self.cfg.MODEL.ACTION_NET:
                    self._init_visual_extractor()
                self.action_model = ActionNet(cfg, x_visual_extractor=None)

            # TODO: only intent should be considered
            # if 'intent' in self.cfg.MODEL.TASK:
            #     if 'convlstm' not in self.cfg.MODEL.INTENT_NET:
        self._init_visual_extractor()
        self.intent_model = IntentNet(cfg, x_visual_extractor=None)

    def _init_hidden_states(self, n, net_type='gru', task_exists=True):
        batch_size = n
        if not task_exists:
            return None
        elif 'gru' in net_type:
            return torch.zeros(batch_size, self.hidden_size)
        else:
            raise ValueError(net_type)

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

    def forward(self,
                x_bbox=None,
                masks=None):
        print(f"HIDDEN size {self.cfg.MODEL.HIDDEN_SIZE}")

        return self.forward_two_stream(x_bbox=x_bbox, masks=masks)

    def forward_two_stream(self, x_bbox=None,  masks=None):
        '''
        NOTE: Action and Intent net use separate encoder networks
        for training only !
        x_visual: extracted features, (batch_size, SEG_LEN, 512, 7, 7)
        x_bbox: bounding boxes(batch_size, SEG_LEN, 4)
        '''
        # seg_len = x_visual.shape[1]
        # int_hx = self._init_hidden_states(x_visual, net_type=self.cfg.MODEL.INTENT_NET,  task_exists='intent' in self.cfg.MODEL.TASK)
        future_inputs = None
        int_hx = self._init_hidden_states(
            self.cfg.MODEL.SEG_LEN, net_type=self.cfg.MODEL.ACTION_NET,  task_exists=True)
        int_hx = int_hx.cuda()
        intent_detection_scores = []
        for t in range(self.cfg.MODEL.SEG_LEN):
            # dec_input = dec_inputs[:, t] if dec_inputs else None
            ret = self.step_two_stream(int_hx, x_bbox[:, t], future_inputs)
            int_hx, enc_int_score, future_inputs = ret

            if enc_int_score is not None:
                intent_detection_scores.append(enc_int_score)

        intent_detection_scores = torch.stack(
            intent_detection_scores, dim=1) if intent_detection_scores else None
        return intent_detection_scores, None

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
