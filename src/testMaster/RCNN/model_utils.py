import torch
import torchvision as tv
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_model(device):

    model=tv.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT', min_size=1024, max_size=2048) #CHANGE

    # =============================================================================
    # If ResNet18, use model below
    # =============================================================================

    # backbone = tv.models.resnet18(weights = 'DEFAULT')
    # backbone = tv.models.detection.backbone_utils._resnet_fpn_extractor(backbone, 1)
    # model = tv.models.detection.MaskRCNN(backbone, num_classes=2, min_size=1024, max_size=2048)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.detections_per_img = 300
    model.roi_heads.box_predictor=FastRCNNPredictor(in_features,num_classes=2)         
    model.to(device)
    
    
    return model

def initialize_optimizer(model, lr):
    return torch.optim.AdamW(params=model.parameters(), lr=lr)  