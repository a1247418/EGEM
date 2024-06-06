from typing import Optional, Iterable
from models.blueprints import get_blueprint
from models.custom_model import CustomModel


def load_custom_model(model_name: str, file_path: str, n_outputs: int, load_to_device: Optional[str] = None, out_classes:Optional[Iterable[int]]=None):
    model_blueprint = get_blueprint(model_name)
    model = CustomModel(blueprint=model_blueprint, state_dict_path=file_path, n_outputs=n_outputs, out_classes=out_classes, load_to_device=load_to_device)
    return model


def load_model(model_name: str, file_path: Optional[str] = None, out_classes:Optional[Iterable[int]]=None, original_n_out=1000, device: Optional[str] = "cuda"):
    if model_name in ["resnet18", "resnet50", "vgg16", "vit_b_16"]:
        model = load_custom_model(model_name, file_path, original_n_out, out_classes=out_classes)
    elif model_name == "mnistnet":
        model = load_custom_model("mnistnet", file_path, 10, out_classes=out_classes)
    elif model_name == "mnistnet_small":
        model = load_custom_model("mnistnet_small", file_path, 10, out_classes=out_classes)
    elif model_name == "mnistnetRGB":
        model = load_custom_model("mnistnetRGB", file_path, 10, out_classes=out_classes)
    elif model_name == "vgg16_isic":
        model = load_custom_model("vgg16_isic", file_path, 8, out_classes=out_classes, load_to_device="cpu")
    elif model_name == "vgg16_celeba":
        model = load_custom_model("vgg16_celeba", file_path, 2, out_classes=out_classes)
    else:
        raise ValueError("Model name not recognized.")
    model = model.eval()
    model = model.to(device)
    return model


#TODO: remove?
def get_default_layer_names(model_name: str):
    if model_name == "resnet50":
        layer_names = ["layer1.2.conv3", "layer2.3.conv3", "layer3.5.conv3", "layer4.2.conv3", "fc"]
    elif model_name == "vgg16":
        layer_names = ["layer1.0.conv1", "layer2.0.conv1", "layer3.0.conv1", "layer4.0.conv1", "avgpool", "fc"]
    elif model_name == "mnistnet":
        pass
    elif model_name == "mnistnetRGB":
        pass
    elif model_name == "vgg16_isic":
        pass
    return layer_names