import model as model


def res_net_pyramidal_model_d110_w350(
        features,
        targets,
        mode,
        optimizer_type='SGD',
        learning_rate=0.001):
    """ Deep Pyramidal Residual Networks
    From https://arxiv.org/abs/1610.02915
    """
    return model.res_net_pyramidal_model(
        features=features,
        targets=targets,
        mode=mode,
        num_classes=2,
        num_blocks=36,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        groups=[16, 150, 250, 350],
        scope="rnp_d110_w350")


def res_net_pyramidal_model_d80_w256_k3_dr05(
        features,
        targets,
        mode,
        optimizer_type='SGD',
        learning_rate=0.001):
    return model.res_net_pyramidal_model(
        features=features,
        targets=targets,
        mode=mode,
        num_classes=2,
        num_blocks=int(80/3),
        multi_k=3,
        keep_prob=0.5,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        groups=[16, 64, 128, 256],
        scope="rnp_d80_w256_k3_dr05")


def res_net_pyramidal_model_d12_w64_k2_dr05(
        features,
        targets,
        mode,
        optimizer_type='SGD',
        learning_rate=0.001):
    return model.res_net_pyramidal_model(
        features=features,
        targets=targets,
        mode=mode,
        num_classes=2,
        num_blocks=int(12/3),
        multi_k=2,
        keep_prob=0.5,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        groups=[16, 16, 32, 64],
        scope="rnp_d12_w64_k2_dr05")


def res_net_pyramidal_model_d6_w32_k1_dr05(
        features,
        targets,
        mode,
        optimizer_type='SGD',
        learning_rate=0.001):
    return model.res_net_pyramidal_model(
        features=features,
        targets=targets,
        mode=mode,
        num_classes=2,
        num_blocks=int(6/3),
        multi_k=1,
        keep_prob=0.5,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        groups=[16, 16, 32, 32],
        scope="rnp_d6_w32_k1_dr05")


def res_net_pyramidal_model_d6_w32_k2_dr05(
        features,
        targets,
        mode,
        optimizer_type='SGD',
        learning_rate=0.001):
    return model.res_net_pyramidal_model(
        features=features,
        targets=targets,
        mode=mode,
        num_classes=2,
        num_blocks=int(6/3),
        multi_k=2,
        keep_prob=0.5,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        groups=[16, 16, 32, 32],
        scope="rnp_d6_w32_k2_dr05")


def res_net_pyramidal_model_d21_w128_k2_dr05(
        features,
        targets,
        mode,
        optimizer_type='SGD',
        learning_rate=0.001):
    return model.res_net_pyramidal_model(
        features=features,
        targets=targets,
        mode=mode,
        num_classes=2,
        num_blocks=int(21/3),
        multi_k=2,
        keep_prob=0.5,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        groups=[16, 32, 64, 128],
        scope="rnp_d21_w128_k2_dr05")


def res_net_pyramidal_model_d21_w128_k4_dr05(
        features,
        targets,
        mode,
        optimizer_type='SGD',
        learning_rate=0.001):
    return model.res_net_pyramidal_model(
        features=features,
        targets=targets,
        mode=mode,
        num_classes=2,
        num_blocks=int(21/3),
        multi_k=4,
        keep_prob=0.5,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        groups=[16, 32, 64, 128],
        scope="rnp_d21_w128_k4_dr05")


def res_net_pyramidal_model_d21_w256_k2_dr05(
        features,
        targets,
        mode,
        optimizer_type='SGD',
        learning_rate=0.001):
    return model.res_net_pyramidal_model(
        features=features,
        targets=targets,
        mode=mode,
        num_classes=2,
        num_blocks=int(21/3),
        multi_k=2,
        keep_prob=0.5,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        groups=[16, 64, 128, 256],
        scope="rnp_d21_w256_k2_dr05")


def res_net_pyramidal_model_d12_w256_k2_dr05(
        features,
        targets,
        mode,
        optimizer_type='SGD',
        learning_rate=0.001):
    return model.res_net_pyramidal_model(
        features=features,
        targets=targets,
        mode=mode,
        num_classes=2,
        num_blocks=int(12/3),
        multi_k=2,
        keep_prob=0.5,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        groups=[16, 32, 128, 256],
        scope="rnp_d12_w256_k2_dr05")


def res_net_pyramidal_model_d12_w256_k2_dr05_ds(
        features,
        targets,
        mode,
        optimizer_type='SGD',
        learning_rate=0.001):
    return model.res_net_pyramidal_model(
        features=features,
        targets=targets,
        mode=mode,
        num_classes=2,
        num_blocks=int(12/3),
        multi_k=2,
        keep_prob=0.5,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        groups=[16, 32, 128, 256],
        is_double_size=True,
        scope="rnp_d12_w256_k2_dr05_ds")


def res_net_pyramidal_model_d12_w256_k4_dr05(
        features,
        targets,
        mode,
        optimizer_type='SGD',
        learning_rate=0.001):
    return model.res_net_pyramidal_model(
        features=features,
        targets=targets,
        mode=mode,
        num_classes=2,
        num_blocks=int(12/3),
        multi_k=4,
        keep_prob=0.5,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        groups=[16, 32, 128, 256],
        scope="rnp_d12_w256_k4_dr05")


def res_net_pyramidal_model_d12_w1024_k2_dr05(
        features,
        targets,
        mode,
        optimizer_type='SGD',
        learning_rate=0.001):
    return model.res_net_pyramidal_model(
        features=features,
        targets=targets,
        mode=mode,
        num_classes=2,
        num_blocks=int(12/3),
        multi_k=2,
        keep_prob=0.5,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        groups=[16, 128, 512, 1024],
        scope="rnp_d12_w1024_k2_dr05")


def res_net_pyramidal_model_d12_w1024_k4_dr05(
        features,
        targets,
        mode,
        optimizer_type='SGD',
        learning_rate=0.001):
    return model.res_net_pyramidal_model(
        features=features,
        targets=targets,
        mode=mode,
        num_classes=2,
        num_blocks=int(12/3),
        multi_k=4,
        keep_prob=0.5,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        groups=[16, 128, 512, 1024],
        scope="rnp_d12_w1024_k4_dr05")




def res_net_pyramidal_model_d12_w512_k2_dr05(
        features,
        targets,
        mode,
        optimizer_type='SGD',
        learning_rate=0.001):
    return model.res_net_pyramidal_model(
        features=features,
        targets=targets,
        mode=mode,
        num_classes=2,
        num_blocks=int(12/3),
        multi_k=2,
        keep_prob=0.5,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        groups=[16, 128, 256, 512],
        scope="rnp_d12_w512_k2_dr05")


def res_net_pyramidal_model_d12_w512_k4_dr05(
        features,
        targets,
        mode,
        optimizer_type='SGD',
        learning_rate=0.001):
    return model.res_net_pyramidal_model(
        features=features,
        targets=targets,
        mode=mode,
        num_classes=2,
        num_blocks=int(12/3),
        multi_k=4,
        keep_prob=0.5,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        groups=[16, 128, 256, 512],
        scope="rnp_d12_w512_k4_dr05")


def res_net_pyramidal_model_d12_w128_k2_dr05(
        features,
        targets,
        mode,
        optimizer_type='SGD',
        learning_rate=0.001):
    return model.res_net_pyramidal_model(
        features=features,
        targets=targets,
        mode=mode,
        num_classes=2,
        num_blocks=int(12/3),
        multi_k=2,
        keep_prob=0.5,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        groups=[16, 32, 64, 128],
        scope="rnp_d12_w128_k2_dr05")


def res_net_pyramidal_model_d6_w128_k2_dr05(
        features,
        targets,
        mode,
        optimizer_type='SGD',
        learning_rate=0.001):
    return model.res_net_pyramidal_model(
        features=features,
        targets=targets,
        mode=mode,
        num_classes=2,
        num_blocks=int(6/3),
        multi_k=2,
        keep_prob=0.5,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        groups=[16, 32, 64, 128],
        scope="rnp_d6_w128_k2_dr05")


def res_net_pyramidal_model_d21_w256_k4_dr05(
        features,
        targets,
        mode,
        optimizer_type='SGD',
        learning_rate=0.001):
    return model.res_net_pyramidal_model(
        features=features,
        targets=targets,
        mode=mode,
        num_classes=2,
        num_blocks=int(21/3),
        multi_k=4,
        keep_prob=0.5,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        groups=[16, 64, 128, 256],
        scope="rnp_d21_w128_k4_dr05")


def res_net_wide_model_d28_w10(
        features,
        targets,
        mode,
        optimizer_type='SGD',
        learning_rate=0.001):
    """ Wide Residual Networks 28-10
    from https://arxiv.org/pdf/1605.07146v1.pdf
    """
    wide = 10
    return model.res_net_wide_model(
        features=features,
        targets=targets,
        mode=mode,
        num_classes=2,
        num_blocks=10,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        groups=[16 * wide, 32 * wide, 64 * wide],
        scope="wrn_d28_w10")


def res_net_wide_model_d16_w8(
        features,
        targets,
        mode,
        optimizer_type='SGD',
        learning_rate=0.001):
    """ Wide Residual Networks 16-8
    from https://arxiv.org/pdf/1605.07146v1.pdf
    """
    wide = 8
    return model.res_net_wide_model(
        features=features,
        targets=targets,
        mode=mode,
        num_classes=2,
        num_blocks=6,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        groups=[16 * wide, 32 * wide, 64 * wide],
        scope="wrn_d16_w8")


def convert_model(
    model,
    optimizer_type='SGD',
    learning_rate=0.001):

    def result_model(features, targets, mode):
        return model(features, targets, mode, optimizer_type, learning_rate)

    return result_model



models = {"rnp_d110_w350" :         res_net_pyramidal_model_d110_w350,
          "wrn_d28_w10":            res_net_wide_model_d28_w10,

          "rnp_d80_w256_k3_dr05":   res_net_pyramidal_model_d80_w256_k3_dr05,

          "rnp_d6_w32_k1_dr05":     res_net_pyramidal_model_d6_w32_k1_dr05,
          "rnp_d6_w32_k2_dr05":     res_net_pyramidal_model_d6_w32_k2_dr05,

          "rnp_d12_w64_k2_dr05":        res_net_pyramidal_model_d12_w64_k2_dr05,
          "rnp_d12_w128_k2_dr05":       res_net_pyramidal_model_d12_w128_k2_dr05,
          "rnp_d12_w256_k2_dr05":       res_net_pyramidal_model_d12_w256_k2_dr05,
          "rnp_d12_w256_k2_dr05_ds":    res_net_pyramidal_model_d12_w256_k2_dr05_ds,
          "rnp_d12_w256_k4_dr05":       res_net_pyramidal_model_d12_w256_k4_dr05,
          "rnp_d12_w512_k2_dr05":       res_net_pyramidal_model_d12_w512_k2_dr05,
          "rnp_d12_w512_k4_dr05":       res_net_pyramidal_model_d12_w512_k4_dr05,
          "rnp_d12_w1024_k2_dr05":      res_net_pyramidal_model_d12_w1024_k2_dr05,
          "rnp_d12_w1024_k4_dr05":      res_net_pyramidal_model_d12_w1024_k4_dr05,

          "rnp_d21_w128_k2_dr05": res_net_pyramidal_model_d21_w128_k2_dr05,
          "rnp_d21_w128_k4_dr05": res_net_pyramidal_model_d21_w128_k4_dr05,
          "rnp_d21_w256_k2_dr05": res_net_pyramidal_model_d21_w256_k2_dr05,
          "rnp_d21_w256_k4_dr05": res_net_pyramidal_model_d21_w256_k4_dr05,
          }
