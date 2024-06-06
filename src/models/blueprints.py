import copy


class Blueprint:
    def __init__(self, name: str, config):
        self.name = name
        self.config = config

    def set_out(self, output_size: int):
        self.config[-1] = self.config[-1] % str(output_size)
        return self

    def replace(self, rem: str, add: str):
        for l in range(len(self.config)):
            self.config[l] = self.config[l].replace(rem, add)
        self.name += ":r%s_%s" % (rem, add)
        return self

    def append(self, layer: str):
        self.config.append(layer)
        self.name += ":a%s"%layer
        return self


def get_list_of_blueprints():
    list_of_blueprints = [
        Blueprint("mnistnet", ["C,1,8,3,1,1", "R", "MP,2", "C,8,16,3,1,1", "R", "MP,2",
                             "F", "L,784,500", "R", "L,500,%d"]),
        Blueprint("mnistnet_small", ["C,1,8,3,1,1", "R", "MP,2", "C,8,16,3,1,1", "R", "MP,2",
                             "F", "L,784,50", "R", "L,50,%d"]),
        Blueprint("mnistnetRGB", ["C,3,8,3,1,1", "R", "MP,2", "C,8,16,3,1,1", "R", "MP,2",
                             "F", "L,784,500", "R", "L,500,%d"]),
        Blueprint("resnet18", []),
        Blueprint("resnet50", []),
        Blueprint("vgg16", []),
        Blueprint("vit_b_16", []),
        Blueprint("vgg16_celeba", []),
        Blueprint("vgg16_isic", []),
    ]
    list_of_blueprints = copy.deepcopy({bp.name : bp for bp in list_of_blueprints})
    return list_of_blueprints


def get_blueprint(name: str):
    parts = name.split(":")
    try:
        bp = get_list_of_blueprints()[str(parts[0])]
    except:
        raise ValueError("Unknown blueprint name: %s" % name)
    for p in parts[1:]:
        if p.startswith("r"):
            rem, add = p[1:].split("_")
            bp.replace(rem, add)
    return bp


def get_list_of_bp_names():
    return [str(k) for k in get_list_of_blueprints().keys()]


def check_bp_dimensions(name: str, input_size: int, channels: int = 1):
    """Asserts that the blueprint works with the given input size."""
    input_size = input_size[0]
    ok = True
    bp = get_blueprint(name)
    if not name.startswith("ensemble"):
        bp = [bp]
    for b in bp:
        dim = input_size
        ch = channels
        for layer in b.config:
            # print(dim, ch, layer)
            parts = layer.split(",")
            if parts[0] in ["MP", "AP"]:
                dim = dim//int(layer.split(",")[1])
            elif parts[0] == "L":
                if ch > 1:
                    print("Problem: Multi-channel input to Linear.")
                    ok = False
                if dim != int(parts[1]):
                    print("Problem: Wrong input dim to Linear: %d instead of %s" % (dim, parts[1]))
                    ok = False
                if parts[2] != "%d":
                    dim = int(parts[2])
            elif parts[0] == "C":
                dim = (dim - int(parts[3])) // int(parts[4]) + 1
                if ch != int(parts[1]):
                    print("Problem: Wrong input channels to Conv2D: %d instead of %s" % (ch, parts[1]))
                    ok = False
                if parts[2] != "%d":
                    ch = int(parts[2])
            elif parts[0] == "F":
                # assuming 2D input
                dim = dim**2 * ch
                ch = 1
            if dim <= 0:
                print("Problem: Dim is 0 after layer.")
                ok = False
            if not ok:
                print("Problem in:", name, layer)
                break
        if not ok: break
    assert ok
