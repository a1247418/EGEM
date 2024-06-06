import numpy as np
from typing import List
from torchvision.datasets import ImageNet


class SingletonIndexStorage(object):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            imagenet_train_offsets = {478: 614142,
                                      549: 704875,
                                      692: 887289,
                                      519: 666488,
                                      444: 570066,
                                      671: 860892
                                      }
            imagenet_test_offsets = {478: 23900,
                                     549: 27450,
                                     692: 34600,
                                     519: 25950,
                                     444: 22200,
                                     671: 33550
                                     }
            imagenet_train_dirty = {
                478: np.array(
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28,
                         29, 30, 32,
                         33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 51,
                         52, 54, 55, 56, 57, 58, 59, 60, 63, 64, 65, 66, 68, 70, 71, 73, 74, 75, 76, 79, 81, 82, 83, 84,
                         86, 87, 88,
                         91, 92, 93, 94, 113, 114, 118,
                         123, 131, 148, 173, 176, 183, 188, 191, 192, 194, 201, 202, 209, 211, 212, 217, 223, 225, 229,
                         232, 234,
                         239, 241, 256, 274, 278, 281, 301,
                         314, 317, 319, 325, 351, 356, 358, 376, 381, 394, 440, 452, 462, 468, 470, 472, 475, 492, 495,
                         497, 499, 506,
                         512, 513, 514, 521, 522, 525, 531,
                         534, 553, 559, 570, 571, 574, 576, 580, 585, 590, 595, 603, 621, 635, 650, 661, 672, 673, 675,
                         681, 694,
                         695, 696, 697, 701, 704, 705, 708,
                         709, 710, 712, 717, 719, 721, 723, 724, 725, 729, 730, 732, 733, 734, 735, 736, 737, 739, 741,
                         742, 745,
                         746, 751, 752, 753, 756, 758, 760,
                         774, 761, 763, 765, 767, 768, 769, 770, 771, 773, 774, 775, 778, 779, 782, 784, 785, 786, 788,
                         789, 790,
                         791, 792, 793, 795, 797, 799, 801, 803,
                         805, 806, 808, 809, 810, 811, 813, 814, 815, 816, 818, 819, 821, 822, 824, 825, 828, 829, 831,
                         834, 835, 836,
                         838, 839, 840, 841, 842, 843, 844, 848, 850,
                         852, 853, 854, 855, 859, 860, 863, 866, 870, 871, 872, 873, 875, 877, 882, 884, 885, 886, 888,
                         891, 892,
                         894, 895, 897, 898, 900, 901, 904, 905,
                         906, 910, 911, 914, 915, 917, 919, 920, 922, 926, 928, 932, 938, 941, 942, 944, 947, 948, 949,
                         952, 953,
                         954, 956, 957, 959, 960, 961, 962, 963,
                         964, 965, 967, 969, 971, 972, 973, 974, 976, 977, 978, 979, 980, 982, 983, 984, 985, 986, 987,
                         988, 990,
                         991, 992, 993, 995, 999, 1002, 1003,
                         1004, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1016, 1018, 1019, 1020, 1021, 1022,
                         1023, 1024,
                         1025, 1029, 1030, 1031,
                         1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042, 1045, 1046, 1047, 1048, 1049,
                         1050, 1051,
                         1053, 1054,
                         1055, 1056, 1057, 1058, 1059, 1060, 1061, 1062, 1064, 1065, 1066, 1067, 1068, 1069, 1070, 1071,
                         1072, 1073,
                         1074, 1076, 1079, 1080,
                         1081, 1082, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1092, 1093, 1094, 1095, 1096,
                         1097, 1098,
                         1099, 1100, 1101, 1103,
                         1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1119, 1120,
                         1121, 1123,
                         1124, 1125, 1126, 1127,
                         1129, 1130, 1133, 1134, 1135, 1137, 1138, 1139, 1140, 1141, 1142, 1143, 1144, 1145, 1146, 1147,
                         1148, 1150,
                         1152, 1153,
                         1157, 1158, 1159, 1160, 1162, 1163, 1164, 1166, 1169, 1171, 1172, 1173, 1174, 1175, 1176, 1177,
                         1178, 1180,
                         1183, 1184, 1186, 1187,
                         1189, 1190, 1191, 1192, 1193, 1195, 1196, 1197, 1198, 1199, 1200, 1201, 1202, 1205, 1206, 1209,
                         1210, 1211,
                         1214, 1215, 1217, 1218,
                         1219, 1220, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1229, 1231, 1232, 1233, 1234, 1235, 1236,
                         1237, 1238,
                         1240, 1241, 1243, 1246,
                         1247, 1248, 1249, 1252, 1254, 1255, 1257, 1258, 1259, 1260, 1261, 1262, 1266, 1267, 1268, 1269,
                         1271, 1272,
                         1273, 1277, 1278, 1280,
                         1281, 1282, 1283, 1284, 1285, 1286, 1287, 1288, 1289, 1290, 1292, 1293, 1296, 1297, 1298,
                         1299], dtype=int),
                692: np.array([241, 751, 1069, 1110, 1223], dtype=int),
                519: np.array(
                        [16, 133, 137, 277, 454, 469, 471, 486, 533, 648, 748, 910, 912, 931, 965, 974, 998, 1288],
                        dtype=int),
                549: np.array(
                        [134, 172, 311, 339, 408, 419, 467, 508, 589, 595, 540, 640, 660, 684, 697, 734, 775, 800, 804,
                         809, 812,
                         826, 834, 845, 849, 886, 887, 895, 912,
                         940, 953, 954, 957, 962, 966, 967, 970, 972, 981, 983, 985, 986, 988, 993, 1002, 1003,
                         1005, 1006, 1007, 1011, 1012, 1023, 1031, 1052, 1053, 1056, 1060, 1076, 1087, 1094, 1103, 1120,
                         1129, 1139,
                         1142, 1149, 1178, 1190, 1199, 1235,
                         1243, 1251], dtype=int),
                671: np.array(
                        [8, 14, 27, 34, 40, 41, 52, 61, 65, 73, 105, 119, 153, 204, 207, 209, 219, 268, 333, 336, 344,
                         357,
                         369, 386, 421, 433, 435, 450, 472, 644, 688, 781, 1120, 1160, 1166, 1256], dtype=int),
                444: np.array([], dtype=int)
            }
            imagenet_test_dirty = {
                478: np.array([], dtype=int),
                692: np.array([], dtype=int),
                519: np.array([], dtype=int),
                549: np.array([], dtype=int),
                671: np.array([13], dtype=int),
                444: np.array([], dtype=int)
            }
            imagenet_train_dirty = {k: v + imagenet_train_offsets[k] for k, v in imagenet_train_dirty.items()}
            imagenet_test_dirty = {k: v + imagenet_test_offsets[k] for k, v in imagenet_test_dirty.items()}
            imagenet_train_clean = {k: np.array(
                    [i + imagenet_train_offsets[k] for i in range(1300) if (i + imagenet_train_offsets[k]) not in v],
                    dtype=int) for k, v in imagenet_train_dirty.items()}
            imagenet_test_clean = {k: np.array(
                    [i + imagenet_test_offsets[k] for i in range(50) if (i + imagenet_test_offsets[k]) not in v],
                    dtype=int) for k, v in imagenet_test_dirty.items()}

            isic_train_samples_per_class = [4078, 11574, 2979, 786, 2358, 217, 236, 569]
            isic_test_samples_per_class = [444, 1301, 344, 81, 266, 22, 17, 59]
            isic_train_dirty = [i for i in range(432, 610)]
            isic_test_dirty = [i for i in range(50, 68)]
            isic_train_clean = [i for i in range(np.sum(isic_train_samples_per_class)) if i not in isic_train_dirty]
            isic_test_clean = [i for i in range(np.sum(isic_test_samples_per_class)) if i not in isic_test_dirty]

            sample_indicators = {
                "imagenet": {
                    "train": {
                        "clean": imagenet_train_clean,
                        "dirty": imagenet_train_dirty
                    },
                    "test": {
                        "clean": imagenet_test_clean,
                        "dirty": imagenet_train_dirty
                    }
                },
                "isic": {
                    "train": {
                        "clean": isic_train_clean,
                        "dirty": isic_train_dirty
                    },
                    "test": {
                        "clean": isic_test_clean,
                        "dirty": isic_train_dirty
                    }
                }
            }
            for dataset in sample_indicators.keys():
                for split in sample_indicators[dataset].keys():
                    if type(sample_indicators[dataset][split]["clean"]) == dict:
                        sample_indicators[dataset][split]["all"] = {
                            k: np.concatenate([sample_indicators[dataset][split]["clean"][k],
                                               sample_indicators[dataset][split]["dirty"][k]]) for k in
                            sample_indicators[dataset][split]["clean"].keys()}
                    else:
                        sample_indicators[dataset][split]["all"] = np.concatenate(
                                [sample_indicators[dataset][split]["clean"],
                                 sample_indicators[dataset][split]["dirty"]])

            cls.instance = super(SingletonIndexStorage, SingletonIndexStorage).__new__(cls)
            cls.instance.__setattr__("sample_indicators", sample_indicators)

        return cls.instance

    def fill_imagenet_classes(self, dataset_dir: str, classes: List[int]):
        # Fill in all other neccessary sample indices that are not hard-coded and consider them as clean (not
        # necessarily true!) This is done here, because we need to load the dataset to get the indices
        for split in ["train", "test"]:
            if any([k not in self.sample_indicators["imagenet"][split]["all"].keys() for k in classes]):
                imagenet_dataset = ImageNet(dataset_dir, split=split.replace("test", "val"))

                class_indices_dict = {}
                for k in classes:
                    if k not in self.sample_indicators["imagenet"][split]["all"]:
                        class_indices = np.where(np.array(imagenet_dataset.targets) == k)[0]
                        class_indices_dict[k] = class_indices

                self.sample_indicators["imagenet"][split]["all"].update(class_indices_dict)
                self.sample_indicators["imagenet"][split]["clean"].update(class_indices_dict)
                self.sample_indicators["imagenet"][split]["dirty"].update({k: np.array([], dtype=int) for k in classes if
                                                                           k not in
                                                                           self.sample_indicators["imagenet"][split][
                                                                               "dirty"]})

    def get_sample_indicators(self, dataset: str):
        return self.sample_indicators[dataset]
