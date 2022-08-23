import numpy as np
from matplotlib import pyplot as plt
from dataclasses import dataclass
import json
import pandas as pd

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

class ModelCfg:
    def __init__(self, file):
        with open(file, 'r') as fin:
            d = json.load(fin)
            self.num_params = d["num_params"]
            self.num_markers = d["num_markers"]
            self.layer_widths = d["layer_widths"] 


@dataclass
class Trajectory:
    traj: np.array
    model_cfg: ModelCfg
    
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
        
        return self.traj[:, pix:pix + prev_width * self.layer_width(lix)]
    
    def bias_start_pix(self):
        pix = 0
        prev_width = self.num_markers()
        for width in self.model_cfg.layer_widths:
            pix += prev_width * width
            prev_width = width
        return pix
    
    def layer_biases(self, lix:  int):
        pix = self.bias_start_pix()
        for i in range(lix):
            pix += self.layer_width(i)
        
        return self.traj[:, pix:pix+self.layer_width(lix)]
    
    def plot(self):
        fig, axes = plt.subplots(2, self.depth(), sharex=True, figsize=(15, 10))
                
        # weights
        for lix in range(self.depth()):
            axes[0, lix].set_title(f"LAYER {lix + 1}")
            axes[0, lix].plot(self.layer_weights(lix))
        axes[0, 0].set_ylabel(r"$W$")

        # biases
        for lix in range(self.depth() - 1):
            axes[1, lix].plot(self.layer_biases(lix))
        axes[1, 0].set_ylabel(r"$b$")
                
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
        
        return self.params[:, pix:pix + prev_width * self.layer_width(lix)]
    
    def bias_start_pix(self):
        pix = 0
        prev_width = self.num_markers()
        for width in self.model_cfg.layer_widths:
            pix += prev_width * width
            prev_width = width
        return pix
    
    def layer_biases(self, lix:  int):
        pix = self.bias_start_pix()
        for i in range(lix):
            pix += self.layer_width(i)
        
        return self.params[:, pix:pix+self.layer_width(lix)]


def load_json_trace(wdir: str):
    params = []
    wp = []
    bp = []
    ep = []
    mcfg = ModelCfg(wdir + '/meta')
    with open(wdir + '/trace', "r") as fin:
        for line in fin:
            l = json.loads(line)[0]
            params.append(l["params"])
            wp.append(np.asarray(l["hyperparams"]["weight_precisions"]).flatten())
            bp.append(l["hyperparams"]["bias_precisions"])
            ep.append(l["hyperparams"]["error_precision"])
    return Trace(
        mcfg,
        np.asarray(params),
        np.asarray(wp),
        np.asarray(bp),
        np.asarray(ep))


def load_json_traj(wdir: str):
    res = []
    mcfg = ModelCfg(wdir + '/meta')
    with open(wdir + '/traj', 'r') as fin:
        for line in fin:
            l = json.loads(line)
            res.append(Trajectory(np.asarray(l["params"]), mcfg))
    return res
            

def plot_single_arm_trace(file: str):
    trace = load_json_trace(file)
    fig, axes = plt.subplots(4, trace.depth(), sharex=True, figsize=(15, 10))
    
    fig.suptitle(file)
    
    # weights
    for lix in range(trace.depth()):
        axes[0, lix].set_title(f"LAYER {lix + 1}")
        axes[0, lix].plot(trace.layer_weights(lix))
    axes[0, 0].set_ylabel(r"$W$")

    # biases
    for lix in range(trace.depth() - 1):
        axes[1, lix].plot(trace.layer_biases(lix))
    axes[1, 0].set_ylabel(r"$b$")
    
    # precisions
    for lix in range(trace.depth()):
        axes[2, lix].plot(trace.weight_precisions[:, lix], label="w")
        if lix != (trace.depth() - 1):
            axes[2, lix].plot(trace.bias_precisions[:, lix], label="b")
    axes[2, 0].set_ylabel(r"$\sigma^{-2}$")
    axes[2, 0].legend()
    
    axes[3, trace.depth() - 1].set_title("ERROR PRECISION")
    axes[3, trace.depth() - 1].plot(trace.error_precision)
    
    plt.tight_layout()