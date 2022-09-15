import numpy as np
from matplotlib import pyplot as plt
from dataclasses import dataclass
import json

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


class ModelCfg:
    def __init__(self, file):
        with open(file, "r") as fin:
            d = json.load(fin)
            self.num_params = d["num_params"]
            self.num_markers = d["num_markers"]
            self.layer_widths = d["layer_widths"]


@dataclass
class Trajectory:
    params: np.array
    grad: np.array
    num_grad: np.array
    hamiltonian: np.array
    model_cfg: ModelCfg

    def num_markers(self):
        return self.model_cfg.num_markers

    def num_params(self):
        return self.model_cfg.num_params

    def layer_width(self, ix):
        return self.model_cfg.layer_widths[ix]

    def depth(self):
        return len(self.model_cfg.layer_widths)

    def layer_weight_ixs(self, lix: int):
        pix = 0
        prev_width = self.num_markers()
        for i in range(lix):
            pix += prev_width * self.layer_width(i)
            prev_width = self.layer_width(i)
        return pix, pix + prev_width * self.layer_width(lix)

    def bias_start_pix(self):
        pix = 0
        prev_width = self.num_markers()
        for width in self.model_cfg.layer_widths:
            pix += prev_width * width
            prev_width = width
        return pix

    def layer_bias_ixs(self, lix: int):
        pix = self.bias_start_pix()
        for i in range(lix):
            pix += self.layer_width(i)
        return pix, pix + self.layer_width(lix)

    def layer_weights(self, lix: int):
        start, stop = self.layer_weight_ixs(lix)
        return self.params[:, start:stop]

    def layer_biases(self, lix: int):
        start, stop = self.layer_bias_ixs(lix)
        return self.params[:, start:stop]

    def layer_weight_grad(self, lix: int):
        start, stop = self.layer_weight_ixs(lix)
        return self.grad[:, start:stop]

    def layer_bias_grad(self, lix: int):
        start, stop = self.layer_bias_ixs(lix)
        return self.grad[:, start:stop]

    def layer_weight_grad_num(self, lix: int):
        start, stop = self.layer_weight_ixs(lix)
        return self.num_grad[:, start:stop]

    def layer_bias_grad_num(self, lix: int):
        start, stop = self.layer_bias_ixs(lix)
        return self.num_grad[:, start:stop]

    def plot_params(self):
        fig, axes = plt.subplots(2, self.depth(), sharex=True, figsize=(10, 6))

        # weights
        for lix in range(self.depth()):
            axes[0, lix].set_title(f"LAYER {lix + 1}")
            axes[0, lix].plot(self.layer_weights(lix))
        axes[0, 0].set_ylabel(r"$W$")

        # biases
        for lix in range(self.depth() - 1):
            axes[1, lix].plot(self.layer_biases(lix))
        axes[1, 0].set_ylabel(r"$b$")

        axes[1, self.depth() - 1].plot(self.hamiltonian)
        axes[1, self.depth() - 1].set_title(r"$-H$")

        plt.tight_layout()

    def plot_grad_diff(self):
        fig, axes = plt.subplots(2, self.depth(), sharex=True, figsize=(10, 6))

        # weights
        for lix in range(self.depth()):
            axes[0, lix].set_title(f"LAYER {lix + 1}")
            axes[0, lix].plot(
                self.layer_weight_grad(lix) - self.layer_weight_grad_num(lix)
            )
        axes[0, 0].set_ylabel(r"$\Delta \partial P \partial W$")

        # biases
        for lix in range(self.depth() - 1):
            axes[1, lix].plot(self.layer_bias_grad(
                lix) - self.layer_bias_grad_num(lix))
        axes[1, 0].set_ylabel(r"$\Delta \partial P \partial b$")

        axes[1, self.depth() - 1].plot(self.hamiltonian)
        axes[1, self.depth() - 1].set_title(r"$-H$")

        plt.tight_layout()

    def plot_grads(self, num_grads=False):
        fig, axes = plt.subplots(2, self.depth(), sharex=True, figsize=(10, 6))

        # weights
        for lix in range(self.depth()):
            axes[0, lix].set_title(f"LAYER {lix + 1}")
            axes[0, lix].plot(self.layer_weight_grad(lix), label="analytic")
            if num_grads:
                axes[0, lix].plot(
                    self.layer_weight_grad_num(lix), ":", label="numerical"
                )
        axes[0, 0].set_ylabel(r"$\partial P \partial W$")

        # biases
        for lix in range(self.depth() - 1):
            axes[1, lix].plot(self.layer_bias_grad(lix), label="analytic")
            if num_grads:
                axes[1, lix].plot(self.layer_bias_grad_num(
                    lix), ":", label="numerical")
        axes[1, 0].set_ylabel(r"$\partial P \partial b$")

        axes[1, self.depth() - 1].plot(self.hamiltonian)
        axes[1, self.depth() - 1].set_title(r"$-H$")

        plt.tight_layout()


@dataclass
class Trace:
    model_cfg: ModelCfg
    params: np.array
    weight_precisions: np.array
    bias_precisions: np.array
    error_precision: np.array

    def num_markers(self):
        return self.model_cfg.num_markers

    def num_params(self):
        return self.model_cfg.num_params

    def layer_width(self, ix):
        return self.model_cfg.layer_widths[ix]

    def depth(self):
        return len(self.model_cfg.layer_widths)

    def layer_weights(self, lix: int):
        pix = 0
        prev_width = self.num_markers()
        for i in range(lix):
            pix += prev_width * self.layer_width(i)
            prev_width = self.layer_width(i)

        return self.params[:, pix: pix + prev_width * self.layer_width(lix)]

    def bias_start_pix(self):
        pix = 0
        prev_width = self.num_markers()
        for width in self.model_cfg.layer_widths:
            pix += prev_width * width
            prev_width = width
        return pix

    def layer_biases(self, lix: int):
        pix = self.bias_start_pix()
        for i in range(lix):
            pix += self.layer_width(i)

        return self.params[:, pix: pix + self.layer_width(lix)]


def load_json_trace(wdir: str, branch_ix=0):
    params = []
    wp = []
    bp = []
    ep = []
    mcfg = ModelCfg(wdir + "/meta")
    with open(wdir + "/trace", "r") as fin:
        for line in fin:
            l = json.loads(line)[branch_ix]
            params.append(l["params"])
            for lix, p in enumerate(l["hyperparams"]["weight_precisions"]):
                if len(wp) <= lix:
                    wp.append([])
                wp[lix].append(p)
            bp.append(l["hyperparams"]["bias_precisions"])
            ep.append(l["hyperparams"]["error_precision"])
    for lix in range(len(wp)):
        wp[lix] = np.asarray(wp[lix])
    return Trace(mcfg, np.asarray(params), wp, np.asarray(bp), np.asarray(ep))


def load_json_traj(wdir: str):
    res = []
    mcfg = ModelCfg(wdir + "/meta")
    with open(wdir + "/traj", "r") as fin:
        ix = 0
        for line in fin:
            l = json.loads(line)
            res.append(
                Trajectory(
                    np.asarray(l["params"]),
                    np.asarray(l["ldg"]),
                    np.asarray(l["num_ldg"]),
                    np.asarray(l["hamiltonian"]),
                    mcfg,
                )
            )
            ix += 1
    return res


def load_json_training_stats(wdir: str):
    with open(wdir + "/training_stats", "r") as fin:
        return json.load(fin)


def plot_single_branch_trace(wdir: str, branch_ix=0):
    training_stats = load_json_training_stats(wdir)
    trace = load_json_trace(wdir, branch_ix)
    fig, axes = plt.subplots(5, trace.depth(), sharex=True, figsize=(15, 10))

    fig.suptitle(wdir)

    axes[0, 0].set_title("ERROR PRECISION")
    axes[0, 0].plot(trace.error_precision)

    axes[0, 1].set_title("MSE")
    axes[0, 1].plot(training_stats["mse_train"], label="train")
    axes[0, 1].plot(training_stats["mse_test"], label="test")
    axes[0, 1].legend()

    axes[0, trace.depth() - 1].set_axis_off()

    # biases
    for lix in range(trace.depth() - 1):
        axes[1, lix].set_title(f"LAYER {lix + 1}")
        axes[1, lix].plot(trace.layer_biases(lix))
    axes[1, 0].set_ylabel(r"$b$")
    axes[1, trace.depth() - 1].set_axis_off()

    # bias precisions
    for lix in range(trace.depth() - 1):
        axes[2, lix].plot(trace.bias_precisions[:, lix], label="b")
    axes[2, 0].set_ylabel(r"$\sigma^{-2}_{b}$")
    axes[2, trace.depth() - 1].set_axis_off()

    # weights
    for lix in range(trace.depth()):
        axes[3, lix].plot(trace.layer_weights(lix))
    axes[3, 0].set_ylabel(r"$W$")

    # weight precisions
    for lix in range(trace.depth()):
        axes[4, lix].plot(trace.weight_precisions[lix], label="w")
        if lix != (trace.depth() - 1):
            axes[4, lix].plot(trace.bias_precisions[:, lix], label="b")
    axes[4, 0].set_ylabel(r"$\sigma^{-2}_{w}$")

    plt.tight_layout()
