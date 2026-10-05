import os.path


def get_experiment_config(experiment_name: str, refinement: str):
    basic_config = {
        "scenario_name": experiment_name,
        "refinement": refinement,
        "fraction_val": 0.2,
        "model_file_path": None,
        "poisoning_strategy": "none",
        "n_reps": 5,
        "explanation_type": "epsilon_alpha2_beta1_flat",#"epsilon_plus_flat" #"epsilon_gamma_box"#"gradient" #
        "decomposition_type": "none",
    }
    if any([s in experiment_name for s in ("carton", "mountain-bike", "mtb")]):
        if "carton-crate" in experiment_name:
            model_name = "resnet50"
            target_class = 478
            background_classes = [519,]
        elif "carton-envelope" in experiment_name:
            model_name = "resnet50"
            target_class = 478
            background_classes = [549,]
        elif "carton-packet" in experiment_name:
            model_name = "resnet50"
            target_class = 478
            background_classes = [692,]
        else:
            model_name = "vgg16"
            target_class = 671
            background_classes = [444,]

        if model_name == "resnet50":
            layer_names = ['features.6', 'features.10'] if refinement in ("pegem", "pep") else [
                'features.4.0.conv1',
                'features.5.0.conv1',
                'features.6.0.conv1',
                'features.7.0.conv1',
                'features.10'
            ]
        elif model_name == "vgg16":
            layer_names = ['features.36', 'features.39'] if refinement in ("pegem", "pep") else [
                'features.5',
                 'features.10',
                 'features.17',
                 'features.24',
                 'features.31',
                 'features.36',
                 'features.39'
            ]
        elif model_name == "vit_b_16":
            layer_names = ["features.0.encoder.layers.encoder_layer_11", "features.1"] if refinement in ("pegem", "pep") else [
                "features.0.encoder.layers.encoder_layer_5",
                "features.0.encoder.layers.encoder_layer_8",
                "features.0.encoder.layers.encoder_layer_11",
                 'features.1'
            ]
        basic_config.update({
            "model_name": model_name,
            "n_refine": 500,
            "n_test": None,
            "layer_names": layer_names,
            "target_class": target_class,
            "background_classes": background_classes,
            "batch_size": 128 if "mnist" in experiment_name else 16,
        })
    elif "mnist-rgb" in experiment_name:
        variant = experiment_name.split("-")[-1]
        assert variant in ("artifact", "blur", "remove", "color"), experiment_name
        model_file_path = os.path.join("model_weights", f"mnist-rgb-{variant}.model")

        basic_config.update({
            "dataset": "mnist-rgb",
            "model_name": "mnistnetRGB",
            "n_refine": 20,
            "n_test": 1000,
            "layer_names": ['features.4', 'features.9'] if refinement in ("pegem", "pep") else ['features.3', 'features.7', 'features.9'],
            "target_class": 8,
            "background_classes": [0,1,2,3,4,5,6,7,9],
            "batch_size": 128,
            "model_file_path": model_file_path
        })
    elif "mnist" in experiment_name:
        basic_config.update({
            "dataset": "mnist",
            "model_name": "mnistnet",
            "n_refine": 20,
            "n_test": 1000,
            "layer_names": ['features.4', 'features.9'] if refinement in ("pegem", "pep") else ['features.3', 'features.7', 'features.9'],
            "target_class": 8,
            "background_classes": [0,1,2,3,4,5,6,7,9],
            "batch_size": 128,
            "model_file_path": os.path.join("model_weights", "mnist.model")
        })
    elif "celeba" in experiment_name:
        basic_config.update({
            "dataset": "celeba",
            "model_name": "vgg16_short10_2",
            "n_refine": 200,
            "n_test": 5000,
            "layer_names": ['features.5.0.conv1', 'features.10'] if refinement in ("pegem", "pep") else [
                'features.4.0.conv1',
                'features.5.0.conv1',
                'features.6.0.conv1',
                'features.7.0.conv1',
                'features.10'
            ],
            "target_class": 1,
            "background_classes": [0,],
            "batch_size": 128,
        })
    elif "isic" in experiment_name:
        basic_config.update({
            "dataset": "isic",
            "model_name": "vgg16_isic",
            "n_refine": 500,
            "n_test": None,
            "layer_names": ['features.10', 'features.39'] if refinement in ("pegem", "pep") else [
                'features.5',
                'features.10',
                'features.17',
                'features.24',
                'features.33',
                'features.36',
                'features.39'
            ],
            "target_class": 1,
            "background_classes": [0,2,3,4,5,6,7],
            "batch_size": 128,
            "model_file_path": os.path.join("model_weights", "isic_vgg16.model")
        })
    else:
        raise AttributeError("No config for experiment: %s" % experiment_name)

    to_return = basic_config.copy()
    return to_return


def get_refinement_hyperparams(refinement_name:str):
    if refinement_name == "egem":
        hyperparams = {
            "alpha": [0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99],
        }
    elif refinement_name == "pcaegem":
        hyperparams = {
            "alpha": [0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99],
        }
    elif refinement_name == "pep":
        hyperparams = {
            "percent_pruned": [99,97,95,90,80,50,20,10,5,1]#[99.9,99.7,99.5,99,97,95,90,85]+[i*10 for i in range(2,9)][::-1],#[98,96,94,92,90,88,86,84,82,80,75,70,65],#[99.99,99.95,99.9,99.5,99,98,97,96,95,94,93,90],#,98.5,98.,97.],#99.5, 99.3, 99, 98, 97, 95, 0],
        }
    elif refinement_name == "pegem":
        hyperparams = {
            "lmbda": [1./(10**i) for i in range(-6,6)] + [0.],
        }
    elif refinement_name == "retrain":
        hyperparams = {
            "n_steps": [1,5,10,20,30,50,100,200,300,400,500,700]
        }
    elif refinement_name == "ridge":
        hyperparams = {
            "lmbda": sorted([1./(10**i) for i in range(4)] + [5./(10**i) for i in range(1,4)] + [10**i for i in range(7)][1:])
        }
    elif refinement_name == "wegem":
        hyperparams = {
            "lmbda": [1./(10**i) for i in range(2,11)],
        }
    elif refinement_name == "pcatrunc":
        hyperparams = {
            "pca_dims": [10, 20, 50, 100, 200, 500, 1000, 2000],
        }
    elif refinement_name == "none":
        hyperparams = {}
    else:
        raise AttributeError("No hyperparams for refinement: %s" % refinement_name)
    return hyperparams


def get_decomposition_config(decomposition_name:str):
    if decomposition_name == "pca":
        decomposition_config = {
            "n_components": 50
        }
    elif decomposition_name == "prca":
        decomposition_config = {
            "n_components": 20
        }
    elif decomposition_name == "drsa":
        decomposition_config = {
            "n_components": 4
        }
    else:
        raise AttributeError("No decomposition config for decomposition: %s" % decomposition_name)
    return decomposition_config