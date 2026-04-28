import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
 
 
class CTModel(nn.Module):
 
    def __init__(self, num_classes=4):
        super().__init__()
 
        assert num_classes in [2, 4]
        self.num_classes = num_classes
 
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
 
        # Feature extraction (all layers except final FC)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
 
        # GradCAM target layer
        self.target_layer = backbone.layer4
 
        # ✅ Keep BatchNorm1d — matches your saved checkpoint keys
        #    (running_mean, running_var, num_batches_tracked)
        #    It works fine at inference as long as model is in eval() mode
        self.feature_layer = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
 
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
 
        self.gradients   = None
        self.activations = None

        self._fh = None   # forward hook handle
        self._bh = None   # backward hook handle

        self._register_hooks()
 
    def _register_hooks(self):
 
        def forward_hook(module, input, output):
            self.activations = output
 
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
 
        self._fh = self.target_layer.register_forward_hook(forward_hook)
        self._bh = self.target_layer.register_full_backward_hook(backward_hook)
 
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
 
        features = self.feature_layer(x)
 
        if self.num_classes == 4:
            features = F.normalize(features, dim=1, eps=1e-8)
 
        out = self.classifier(features)
 
        return out, features
 
    def get_activations(self):
        return self.activations
 
    def get_gradients(self):
        return self.gradients