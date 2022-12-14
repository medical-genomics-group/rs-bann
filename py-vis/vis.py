import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from matplotlib import pyplot as plt
from dataclasses import dataclass
import json
from typing import List
from pathlib import Path
import pandas as pd

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
    def __init__(self, file, branch_ix=0):
        with open(file, "r") as fin:
            d = json.load(fin)["branch_hyperparams"][branch_ix]
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
class ModelBranchParams:
    num_params: int
    num_markers: int
    layer_widths: np.array
    params: np.array
    weight_precisions: List[List[float]]
    bias_precisions: List[float]

    def layer_width(self, lix: int):
        return self.layer_widths[lix]

    def layer_weight_ixs(self, lix: int):
        pix = 0
        prev_width = self.num_markers
        for i in range(lix):
            pix += prev_width * self.layer_width(i)
            prev_width = self.layer_width(i)
        return pix, pix + prev_width * self.layer_width(lix)

    def bias_start_pix(self):
        pix = 0
        prev_width = self.num_markers
        for width in self.layer_widths:
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
        return self.params[start:stop]

    def layer_biases(self, lix: int):
        start, stop = self.layer_bias_ixs(lix)
        return self.params[start:stop]

    def layer_weight_precisions(self, lix: int):
        return self.weight_precisions[lix]

    def layer_bias_precision(self, lix: int):
        return self.bias_precisions[lix]


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

    def layer_weight_precisions(self, lix: int):
        return self.weight_precisions[lix]

    def layer_bias_precision(self, lix: int):
        return self.bias_precisions[:, lix]


@dataclass
class Data:
    x: List[np.array]
    y: np.array
    x_means: np.array
    x_stds: np.array
    num_markers_per_branch: int
    num_individuals: int
    num_branches: int
    standardized: bool

    def load_train(wdir: str):
        with open(wdir + "/gen_train.json", "r") as fin:
            gen = json.load(fin)
        with open(wdir + "/phen_train.json", "r") as fin:
            phen = json.load(fin)
        return Data.__from_json(gen, phen)

    def load_test(wdir: str):
        with open(wdir + "/gen_test.json", "r") as fin:
            gen = json.load(fin)
        with open(wdir + "/phen_test.json", "r") as fin:
            phen = json.load(fin)
        return Data.__from_json(gen, phen)

    def __from_json(gen, phen):
        x = []
        for bix, branch_data in enumerate(gen["x"]):
            x.append(np.array(branch_data, order="F").reshape(
                (gen["num_individuals"], gen["num_markers_per_branch"][bix])))
        return Data(
            x,
            np.array(phen['y']),
            np.array(gen['means']),
            np.array(gen['stds']),
            gen['num_markers_per_branch'],
            gen['num_individuals'],
            gen['num_branches'],
            gen['standardized'])


def load_true_params(wdir: str):
    with open(wdir + "/model.params", "r") as fin:
        model_params = json.load(fin)
    res = []
    for branch in model_params:
        res.append(ModelBranchParams(
            branch["num_params"],
            branch["num_markers"],
            branch["layer_widths"],
            np.array(branch["params"]),
            [e['data'] for e in branch["precisions"]["weight_precisions"]],
            branch["precisions"]["bias_precisions"]))
    return res


def load_json_trace(wdir: str, branch_ix=0):
    params = []
    wp = []
    bp = []
    ep = []
    mcfg = ModelCfg(wdir + "/hyperparams", branch_ix)
    with open(wdir + "/trace", "r") as fin:
        for line in fin:
            l = json.loads(line)[branch_ix]
            params.append(l["params"])
            for lix, p in enumerate(l["precisions"]["weight_precisions"]):
                if len(wp) <= lix:
                    wp.append([])
                wp[lix].append(p["data"])
            bp.append(l["precisions"]["bias_precisions"])
            ep.append(l["precisions"]["error_precision"])
    for lix in range(len(wp)):
        wp[lix] = np.asarray(wp[lix])
    return Trace(mcfg, np.asarray(params), wp, np.asarray(bp), np.asarray(ep))


def load_json_traj(wdir: str, branch_ix=0):
    res = []
    mcfg = ModelCfg(wdir + "/hyperparams", branch_ix)
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


def load_train_phen_stats(wdir: str):
    with open(wdir + "/train_phen_stats.json", "r") as fin:
        return json.load(fin)


def load_test_phen_stats(wdir: str):
    with open(wdir + "/test_phen_stats.json", "r") as fin:
        return json.load(fin)


def data_dir(wdir: str):
    return str(Path(wdir).parent)


def parent_dir(d: str):
    return str(Path(d).parent)


def plot_single_branch_posterior_means(wdir: str, burn_in, branch_ix=0):
    ddir = data_dir(wdir)
    trace = load_json_trace(wdir, branch_ix)
    truth = load_true_params(ddir)[branch_ix]

    fig, axes = plt.subplots(4, trace.depth(), figsize=(15, 10))

    axes[0, 0].set_ylabel(r"$E(W | D)$")
    axes[1, 0].set_ylabel(r"$E(\lambda_W | D)$")
    axes[2, 0].set_ylabel(r"$E(b | D)$")
    axes[3, 0].set_ylabel(r"$E(\lambda_b | D)$")

    for lix in range(trace.depth()):
        w_pm = trace.layer_weights(lix)[burn_in:].mean(axis=0)
        w_t = truth.layer_weights(lix)
        try:
            w_pm.sort()
            w_t.sort()
        except:
            pass
        axes[0, lix].plot(w_t, w_pm, 'k.')
        axes[0, lix].plot(w_t, w_t, 'k:')
        axes[0, lix].set_xlabel(r"$W$")

        wp_pm = trace.layer_weight_precisions(lix)[burn_in:].mean(axis=0)
        wp_t = truth.layer_weight_precisions(lix)
        try:
            wp_pm.sort()
            wp_t.sort()
        except:
            pass
        axes[1, lix].plot(wp_t, wp_pm, 'k.')
        axes[1, lix].plot(wp_t, wp_t, 'k:')
        axes[1, lix].set_xlabel(r"$\lambda_W$")

        if lix < (trace.depth() - 1):
            b_pm = trace.layer_biases(lix)[burn_in:].mean(axis=0)
            b_t = truth.layer_biases(lix)
            try:
                b_pm.sort()
                b_t.sort()
            except:
                pass
            axes[2, lix].plot(b_t, b_pm, 'k.')
            axes[2, lix].plot(b_t, b_t, 'k:')
            axes[2, lix].set_xlabel(r"$b$")

            bp_pm = trace.layer_bias_precision(lix)[burn_in:].mean()
            bp_t = truth.layer_bias_precision(lix)
            axes[3, lix].plot(bp_t, bp_pm, 'k.')
            axes[3, lix].set_xlabel(r"$\lambda_b$")

    plt.tight_layout()


def plot_single_branch_state(wdir: str, state_ix: int, branch_ix=0):
    ddir = data_dir(wdir)
    trace = load_json_trace(wdir, branch_ix)
    truth = load_true_params(ddir)[branch_ix]

    fig, axes = plt.subplots(4, trace.depth(), figsize=(15, 10))

    axes[0, 0].set_ylabel(r"$W | D$")
    axes[1, 0].set_ylabel(r"$\lambda_W | D$")
    axes[2, 0].set_ylabel(r"$b | D$")
    axes[3, 0].set_ylabel(r"$\lambda_b | D$")

    for lix in range(trace.depth()):
        w_pm = trace.layer_weights(lix)[state_ix]
        w_t = truth.layer_weights(lix)
        try:
            w_pm.sort()
            w_t.sort()
        except:
            pass
        axes[0, lix].plot(w_t, w_pm, 'k.')
        axes[0, lix].plot(w_t, w_t, 'k:')
        axes[0, lix].set_xlabel(r"$W$")

        wp_pm = trace.layer_weight_precisions(lix)[state_ix]
        wp_t = truth.layer_weight_precisions(lix)
        try:
            wp_pm.sort()
            wp_t.sort()
        except:
            pass
        axes[1, lix].plot(wp_t, wp_pm, 'k.')
        axes[1, lix].plot(wp_t, wp_t, 'k:')
        axes[1, lix].set_xlabel(r"$\lambda_W$")

        if lix < (trace.depth() - 1):
            b_pm = trace.layer_biases(lix)[state_ix]
            b_t = truth.layer_biases(lix)
            try:
                b_pm.sort()
                b_t.sort()
            except:
                pass
            axes[2, lix].plot(b_t, b_pm, 'k.')
            axes[2, lix].plot(b_t, b_t, 'k:')
            axes[2, lix].set_xlabel(r"$b$")

            bp_pm = trace.layer_bias_precision(lix)[state_ix]
            bp_t = truth.layer_bias_precision(lix)
            axes[3, lix].plot(bp_t, bp_pm, 'k.')
            axes[3, lix].set_xlabel(r"$\lambda_b$")

    plt.tight_layout()


def load_genetic_values(wdir: str):
    with open(wdir + "/genetic_values_train.json", 'r') as fin:
        train = json.load(fin)
    with open(wdir + "/genetic_values_test.json", 'r') as fin:
        test = json.load(fin)
    return train['y'], test['y']


# def plot_perf_r2_genetic_value(wdir: str, burn_in):
#     ddir = parent_dir(wdir)

#     test_g, train_g = load_genetic_values(wdir)

#     fig = plt.figure(figsize=(6, 3))
#     fig.suptitle(wdir)

#     r2_test = [r2(df.iloc[0].values, test_g)]

#     ax = plt.gca()
#     ax.set_title(r"$R^2$")
#     ax.plot(r2_train, label="nn train")
#     ax.plot(r2_test, label="nn test")
#     ax.hlines(h2_train, 0, len(trace.error_precision),
#               linestyle="dashed", color="#35063e", label="h2 train")
#     ax.hlines(h2_test, 0, len(trace.error_precision),
#               linestyle="dashdot", color="#35063e", label="h2 test")
#     ax.hlines(
#         ridge_r2_train,
#         0,
#         len(trace.error_precision),
#         color="gray",
#         linestyle="dashed",
#         label="ridge train"
#     )
#     ax.hlines(
#         ridge_r2_test,
#         0,
#         len(trace.error_precision),
#         color="gray",
#         linestyle="dotted",
#         label="ridge test"
#     )
#     ax.legend()
#     ax.set_ylim(0.0, 1.0)

#     plt.tight_layout()


def plot_perf_r2(wdir: str, burn_in):
    ddir = data_dir(wdir)
    train_data = Data.load_train(ddir)
    test_data = Data.load_test(ddir)

    ridge_r2_train, ridge_r2_test = r2_ridge(train_data, test_data)

    train_phen_stats = load_train_phen_stats(ddir)
    test_phen_stats = load_test_phen_stats(ddir)

    training_stats = load_json_training_stats(wdir)
    trace = load_json_trace(wdir, 0)
    fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10, 3))

    fig.suptitle(wdir)

    axes[0].set_title("ERROR PRECISION")
    axes[0].plot(trace.error_precision)
    axes[0].hlines(
        np.mean(trace.error_precision[burn_in:]),
        0,
        len(trace.error_precision),
        color="r",
        linestyle="dashed",
        label="nn posterior mean"
    )
    if train_phen_stats["env_variance"] > 0:
        axes[0].hlines(
            1 / train_phen_stats["env_variance"],
            0,
            len(trace.error_precision),
            color="k",
            linestyle="dotted",
            label="true"
        )
    axes[0].legend()
    axes[0].set_yscale("log")

    r2_train = 1 - \
        (np.array(training_stats["mse_train"]) / train_phen_stats["variance"])
    r2_test = 1 - \
        (np.array(training_stats["mse_test"]) / test_phen_stats["variance"])

    h2_train = (train_phen_stats["variance"] -
                train_phen_stats["env_variance"]) / train_phen_stats["variance"]
    h2_test = (test_phen_stats["variance"] -
               test_phen_stats["env_variance"]) / test_phen_stats["variance"]

    axes[1].set_title(r"$R^2$")
    axes[1].plot(r2_train, label="nn train")
    axes[1].plot(r2_test, label="nn test")
    axes[1].hlines(h2_train, 0, len(trace.error_precision),
                   linestyle="dashed", color="#35063e", label="h2 train")
    axes[1].hlines(h2_test, 0, len(trace.error_precision),
                   linestyle="dashdot", color="#35063e", label="h2 test")
    axes[1].hlines(
        ridge_r2_train,
        0,
        len(trace.error_precision),
        color="gray",
        linestyle="dashed",
        label="ridge train"
    )
    axes[1].hlines(
        ridge_r2_test,
        0,
        len(trace.error_precision),
        color="gray",
        linestyle="dotted",
        label="ridge test"
    )
    axes[1].legend()
    axes[1].set_ylim(0.0, 1.0)
    # axes[1].set_yscale("log")

    plt.tight_layout()


def plot_perf(wdir: str, burn_in):
    ddir = data_dir(wdir)
    train_data = Data.load_train(ddir)
    test_data = Data.load_test(ddir)

    ridge_mse_train, ridge_mse_test = mse_ridge(train_data, test_data)

    phen_stats = load_train_phen_stats(ddir)
    training_stats = load_json_training_stats(wdir)
    trace = load_json_trace(wdir, 0)
    fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10, 3))

    fig.suptitle(wdir)

    axes[0].set_title("ERROR PRECISION")
    axes[0].plot(trace.error_precision)
    axes[0].hlines(
        np.mean(trace.error_precision[burn_in:]),
        0,
        len(trace.error_precision),
        color="r",
        linestyle="dashed",
        label="nn posterior mean"
    )
    if phen_stats["env_variance"] > 0:
        axes[0].hlines(
            1 / phen_stats["env_variance"],
            0,
            len(trace.error_precision),
            color="k",
            linestyle="dotted",
            label="true"
        )
    axes[0].legend()
    axes[0].set_yscale("log")

    axes[1].set_title("MSE")
    axes[1].plot(training_stats["mse_train"], label="nn train")
    axes[1].plot(training_stats["mse_test"], label="nn test")
    axes[1].hlines(
        ridge_mse_train,
        0,
        len(trace.error_precision),
        color="gray",
        linestyle="dashed",
        label="ridge train"
    )
    axes[1].hlines(
        ridge_mse_test,
        0,
        len(trace.error_precision),
        color="gray",
        linestyle="dotted",
        label="ridge test"
    )
    axes[1].legend()
    axes[1].set_yscale("log")

    plt.tight_layout()


def plot_single_branch_trace(wdir: str, branch_ix=0):
    trace = load_json_trace(wdir, branch_ix)
    fig, axes = plt.subplots(4, trace.depth(), sharex=True, figsize=(15, 10))

    fig.suptitle(wdir)

    if trace.depth() > 2:
        axes[0, trace.depth() - 1].set_axis_off()

    # biases
    for lix in range(trace.depth() - 1):
        axes[0, lix].set_title(f"LAYER {lix + 1}")
        axes[0, lix].plot(trace.layer_biases(lix))
    axes[0, 0].set_ylabel(r"$b$")
    axes[0, trace.depth() - 1].set_axis_off()

    # bias precisions
    for lix in range(trace.depth() - 1):
        axes[1, lix].plot(trace.bias_precisions[:, lix], label="b")
    axes[1, 0].set_ylabel(r"$\sigma^{-2}_{b}$")
    axes[1, trace.depth() - 1].set_axis_off()

    # weights
    for lix in range(trace.depth()):
        axes[2, lix].plot(trace.layer_weights(lix))
    axes[2, 0].set_ylabel(r"$W$")

    # weight precisions
    for lix in range(trace.depth()):
        axes[3, lix].plot(trace.weight_precisions[lix], label="w")
        # if lix != (trace.depth() - 1):
        #     axes[4, lix].plot(trace.bias_precisions[:, lix], label="b")
    axes[3, 0].set_ylabel(r"$\sigma^{-2}_{w}$")

    plt.tight_layout()


def mse_ridge(train_data, test_data, alpha=1.0):
    x_train = np.hstack(train_data.x)
    y_train = train_data.y
    x_test = np.hstack(test_data.x)
    y_test = test_data.y
    reg = Ridge(alpha).fit(x_train, y_train)
    mse_train = mse(reg.predict(x_train), y_train)
    mse_test = mse(reg.predict(x_test), y_test)
    return mse_train, mse_test


def mse_linreg(train_data, test_data):
    x_train = np.hstack(train_data.x)
    y_train = train_data.y
    x_test = np.hstack(test_data.x)
    y_test = test_data.y
    reg = LinearRegression().fit(x_train, y_train)
    mse_train = mse(reg.predict(x_train), y_train)
    mse_test = mse(reg.predict(x_test), y_test)
    return mse_train, mse_test


def mse(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()


def r2_ridge(train_data, test_data, alpha=1.0):
    x_train = np.hstack(train_data.x)
    y_train = train_data.y
    x_test = np.hstack(test_data.x)
    y_test = test_data.y
    reg = Ridge(alpha).fit(x_train, y_train)
    r2_train = r2(reg.predict(x_train), y_train)
    r2_test = r2(reg.predict(x_test), y_test)
    return r2_train, r2_test


def r2(y_pred, y_true):
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
