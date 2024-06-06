from typing import Optional, Iterable
import torch
import torchvision
from torch import nn

try:
    from torchvision.models import ResNet18_Weights, ResNet50_Weights, VGG16_Weights, ViT_B_16_Weights
except:
    pass

from .blueprints import Blueprint

class _Module(nn.Module):
    def __init__(self, feature_list, state_dict_path=None, load_to_device=None):
        super(_Module, self).__init__()
        self.features = nn.ModuleList(feature_list)

        if state_dict_path is not None:
            state_dict = torch.load(state_dict_path, map_location=load_to_device)
            self.load_state_dict(state_dict)
            print("Loaded model state dict from %s." % state_dict_path)
        else:
            self._initialize_weights()

    def forward(self, x):
        y = x
        for l in self.features:
            y = l(y)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                break  # Using default init for now


def _convert_str(s):
    if type(s) == str:
        if s in ["True", "False"]:
            s = bool(s)
        elif s.startswith("(") or s.startswith("["):
            s = tuple([_convert_str(e) for e in s[1:-1].split(",")])
        else:
            try:
                s = int(s)
            except:
                try:
                    s = float(s)
                except:
                    pass
    return s


class CustomModel(nn.Module):
    name2cls = {
        "resnet18": torchvision.models.resnet18,
        "resnet50": torchvision.models.resnet50,
        "vgg16": torchvision.models.vgg16,
        "vit_b_16": torchvision.models.vit_b_16,
    }
    name2weights = {
        "resnet18": ResNet18_Weights.IMAGENET1K_V1,
        "resnet50": ResNet50_Weights.IMAGENET1K_V1,
        "vgg16": VGG16_Weights.IMAGENET1K_V1,
        "vit_b_16": ViT_B_16_Weights.IMAGENET1K_V1,
    }

    def __init__(
        self,
        blueprint: Blueprint,
        state_dict_path: Optional[str] = None,
        n_outputs: Optional[int] = None,
        out_classes: Optional[Iterable[int]] = None,
        load_to_device: Optional[str] = None,
    ):
        super(CustomModel, self).__init__()

        # Load base model from given path or from pytorch weights
        for name, model_cls in self.name2cls.items():
            if name in blueprint.name:
                """load_pretrained = False
                if state_dict_path is None:
                    print("Loading pretrained %s." % blueprint.name)
                    load_pretrained = True
                else:
                    try:
                        self.model = model_cls()
                        self.model.load_state_dict(
                                torch.load(state_dict_path, map_location=load_to_device)
                        )
                    except Exception as e:
                        print(
                                "Could not load %s. Loading pretrained model." % blueprint.name,
                                repr(e),
                        )
                        load_pretrained = True
                if load_pretrained:
                """
                try:
                    self.model = model_cls(weights=self.name2weights[name])
                except:
                    # Backwards compatibility
                    self.model = model_cls(pretrained=True)

        # Put layers into a feature list and modify architecture if necessary
        if "resnet" in blueprint.name:
            layers = [v for k, v in self.model._modules.items()]
            fc = nn.Linear(layers[-1].in_features, n_outputs)
            fc.weight.data = layers[-1].weight.data[:n_outputs, :]
            fc.bias.data = layers[-1].bias.data[:n_outputs]
            layers = layers[:-1] + [nn.Flatten(), fc]
            self.model = _Module(layers, state_dict_path)
        elif "vit" in blueprint.name:
            layers = [self.model]
            fc = nn.Linear(self.model.heads[-1].in_features, n_outputs)
            fc.weight.data = self.model.heads[-1].weight.data[:n_outputs, :]
            fc.bias.data = self.model.heads[-1].bias.data[:n_outputs]
            self.model.heads[-1] = nn.Identity()
            layers.append(fc)
            self.model = _Module(layers, state_dict_path)
        elif "vgg16" in blueprint.name:
            layers = list(self.model._modules["features"])

            if "vgg16_celeba" == blueprint.name:
                n_vgg_layers = 17
                layers = layers[:n_vgg_layers]
                pooling_layer1 = torch.nn.AdaptiveMaxPool2d(1)
                conv_layer2 = nn.Conv2d(256, 128, (3, 3), (1, 1), 1)
                out_layer1 = nn.Linear(128, 512)
                out_layer2 = nn.Linear(512, 2)
                fc_stack = [nn.Flatten(), out_layer1, nn.ReLU(), out_layer2]
                layers += [conv_layer2, pooling_layer1, nn.ReLU()] + fc_stack
            elif blueprint.name == "vgg16" or blueprint.name == "vgg16_isic":
                layers += [self.model._modules["avgpool"]]
                layers += [nn.Flatten()]
                layers += list(self.model._modules["classifier"][:-1])
                layers += [nn.Linear(4096, n_outputs)]
                # Retain last layer for desired n_outputs
                layers[-1].weight.data = self.model._modules["classifier"][
                    -1
                ].weight.data[:n_outputs, :]
                layers[-1].bias.data = self.model._modules["classifier"][-1].bias.data[
                    :n_outputs
                ]
            self.model = _Module(layers, state_dict_path, load_to_device=load_to_device)
        else:
            layers, _ = self.layers_from_blueprint(blueprint, n_outputs)
            self.model = _Module(layers, state_dict_path, load_to_device=load_to_device)

        # Add a linear layer to map to out_classes
        #if out_classes is None:
        #    out_classes = range(n_outputs)

        # Get device
        """for f in self.model.features:
            try:
                device = f.weight.device
                break
            except AttributeError:
                pass
         """
        if out_classes is not None and len(out_classes) < n_outputs:
            # reduce the last layer to only include the out classes
            last_linear = self.model.features[-1]
            device = last_linear.weight.device
            new_outlayer = nn.Linear(last_linear.in_features, len(out_classes), bias=True, device=device)
            for i, c in enumerate(out_classes):
                new_outlayer.weight.data[i, :] = self.model.features[-1].weight.data[c, :]
                new_outlayer.bias.data[i] = self.model.features[-1].bias.data[c]
            self.model.features[-1] = new_outlayer
        #    print(self.model)
        #    print(self.model.features)

        #out_proj = nn.Linear(n_outputs, len(out_classes), bias=False, device=device)
        #out_proj.weight.data = torch.zeros([len(out_classes), n_outputs], device=device)
        #for i, c in enumerate(out_classes):
        #    out_proj.weight.data[i, c] = 1
        #self.model.features.append(out_proj)
        # TODO: remove commented code

    def layers_from_blueprint(self, blueprint, n_outputs):
        layer_options = {
            "F": nn.Flatten,
            "L": nn.Linear,
            "C": nn.Conv2d,
            "R": nn.ReLU,
            "S": nn.Sigmoid,
            "AP": nn.AvgPool2d,
            "AAP": nn.AdaptiveAvgPool2d,
            "MP": nn.MaxPool2d,
            "AMP": nn.AdaptiveMaxPool2d,
            "BN2": nn.BatchNorm2d,
            "DO2": nn.Dropout2d,
        }

        layers = []
        channels = 1  # TODO adapt this to include RGB as well
        config = blueprint.config
        last_linear = None
        if n_outputs is not None:
            # Else: assume that it is already defined in the blueprint
            try:
                config[-1] = config[-1] % n_outputs
            except:
                print("Assuming fixed output dimension defined in blueprint.")
                pass
        for e_i, entry in enumerate(config):
            parts = entry.split(",")
            if len(parts) == 0:
                raise ValueError("Empty model config.")
            try:
                obj = layer_options[parts[0]]
            except:
                raise ValueError("Unknown module config.")
            if len(parts) == 1:
                if parts[0] == "BN2":
                    parts += [channels]
                else:
                    obj = obj()
            if len(parts) > 1:
                args = [_convert_str(arg) for arg in parts[1:]]
                if parts[0] in ["L", "C"]:
                    obj = obj(*args)
                    if parts[0] == "C":
                        channels = args[1]
                    if parts[0] == "L":
                        last_linear = e_i
                else:
                    obj = obj(*args)
            layers.append(obj)
        return layers, last_linear

    def forward(self, x):
        return self.model(x)

    def save_state_dict(self, path):
        torch.save(self.model.state_dict(), path)
