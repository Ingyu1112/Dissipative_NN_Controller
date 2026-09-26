import gymnasium as gym
from matplotlib import pyplot as plt
import numpy as np
import imageio.v2 as imageio

def save_mujoco_video(states, out_path="cartpole.mp4", fps=50):
    env = gym.make("CustomCartPole-v0",render_mode="rgb_array")
    frames = []

    env.reset(options={"state": states[0]})
    base_env = env.unwrapped

    for state in states:
        base_env.state = np.asarray(state, dtype=np.float32)
        frame = env.render()
        frames.append(frame)

    env.close()

    imageio.mimsave(out_path, frames, fps=fps)
    print(f"  saved: {out_path}")

def draw_response(dt : float, traj : np.ndarray, plot_title : str, l2_dist : str):
    t_step = np.arange(traj.shape[1])*dt

    fig,ax = plt.subplots(2,2, figsize=(10,8), sharex=True)
    fig.suptitle(plot_title, fontweight="bold")
    
    sub_titles = ['pos', 'ang', 'vel', 'ang vel']
    name_legend = [f"seed {i}" for i in range(traj.shape[0])]

    for i, sub_title in enumerate(sub_titles):
        ax[i//2, i%2].plot(t_step, traj[:,:,i].T)
        ax[i//2, i%2].grid(True)
        ax[i//2, i%2].set_title(sub_title)
    ax[0,0].legend(name_legend)
    ax[1,0].set_xlabel("time [s]")
    ax[1,1].set_xlabel("time [s]")

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fig.savefig(f"results/response/{l2_dist}_{'_'.join(plot_title.split())}.png", dpi=300, bbox_inches="tight")
    
def draw_response_comparison(dt : float, group_traj : dict, plot_title : str, l2_dist : str):
    first_value = next(iter(group_traj.values()))
    fig, ax = plt.subplots(4,len(group_traj), figsize=(7*len(group_traj),20), sharex=True, squeeze=False)
    fig.suptitle(plot_title, fontweight="bold")

    y_labels = ['Position [m]', 'Angle [rad]', 'Velocity [m/s]', 'Angular velocity [rad/s]']
    name_legend = [f"seed {i}" for i in range(first_value.shape[0])]
    
    for j, (key, traj) in enumerate(group_traj.items()):
        t_step = np.arange(traj.shape[1]) * dt
        for i, y_label in enumerate(y_labels):
            ax[i, j].plot(t_step, traj[:,:,i].T)
            ax[i, j].grid(True)
            if j == 0: ax[i, j].set_ylabel(y_label)
            if i == 0: ax[i, j].set_title(key)
            if i == len(y_labels)-1: ax[i, j].set_xlabel("Time [s]")
    ax[0,0].legend(name_legend)
            
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"results/response/{l2_dist}_{'_'.join(plot_title.split())}.png", dpi=300, bbox_inches="tight")
    
def draw_response_mean_comparison(dt : float, group_traj_mean_std : dict, plot_title : str, l2_dist : str):
    fig, ax = plt.subplots(2,2, figsize=(10,8),sharex=True)
    fig.suptitle(plot_title, fontweight="bold")
    
    y_labels = ['Position [m]', 'Angle [rad]', 'Velocity [m/s]', 'Angular velocity [rad/s]']
    lines = []

    for j, (key, traj_mean_std) in enumerate(group_traj_mean_std.items()):
        t_step = np.arange(traj_mean_std[0].shape[0]) * dt
        for i, y_label in enumerate(y_labels):
            line, = ax[i//2, i%2].plot(t_step,traj_mean_std[0][:,i], label=key)
            ax[i//2, i%2].fill_between(
                t_step, 
                traj_mean_std[0][:,i]-traj_mean_std[1][:,i], 
                traj_mean_std[0][:,i]+traj_mean_std[1][:,i],
                alpha=0.25
            )
            ax[i//2, i%2].grid(True)
            ax[i//2, i%2].set_ylabel(y_label)
            if i == 0: lines.append(line)
            if i//2 == 1: ax[i//2, i%2].set_xlabel('Time [s]')
    ax[0, 0].legend(handles=lines)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fig.savefig(f"results/response/{l2_dist}_{'_'.join(plot_title.split())}.png", dpi=300, bbox_inches="tight")

def draw_norm(cutoff : int, dt : float, norm : list[np.ndarray], plot_title : str, l2_dist : str):
    r = int(norm[0].shape[1] / cutoff)
    t_step = np.arange(r)*dt

    fig,ax = plt.subplots(1,2, figsize=(10,4), sharex=True)
    fig.suptitle(plot_title, fontweight="bold")
    
    sub_titles = ['x norm', 'u norm']
    name_legend = [f"seed {i}" for i in range(norm[0].shape[0])]

    for i, sub_title in enumerate(sub_titles):
        ax[i].plot(t_step[:r], norm[i][:,:r].T)
        ax[i].grid(True)
        ax[i].set_title(sub_title)
        ax[i].set_xlabel("time [s]")
    ax[0].legend(name_legend)
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fig.savefig(f"results/norm/{l2_dist}_{'_'.join(plot_title.split())}.png", dpi=300, bbox_inches="tight")

def draw_norm_comparison(cutoff : int, dt : float, group_norm : list[dict], plot_title : str, l2_dist : str):    
    first_value = next(iter(group_norm[0].values()))
    fig, ax = plt.subplots(2,len(group_norm[0]), figsize=(7*len(group_norm[0]),10), sharex=True, squeeze=False)
    fig.suptitle(plot_title, fontweight="bold")
    
    y_labels = ['State norm', 'Action norm']
    name_legend = [f"seed {i}" for i in range(first_value.shape[0])]

    for i in range(2):
        for j, (key, traj) in enumerate(group_norm[i].items()):
            r = int(traj.shape[1] / cutoff)
            t_step = np.arange(r)*dt

            ax[i, j].plot(t_step, traj[:,:r].T)
            ax[i, j].grid(True)
            ax[i, j].set_title(key)
            if i == 1: ax[i, j].set_xlabel("time [s]")
        ax[i, 0].set_ylabel(y_labels[i])
    ax[0, 0].legend(name_legend)
                
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"results/norm/{l2_dist}_{'_'.join(plot_title.split())}.png", dpi=300, bbox_inches="tight")

    
def draw_norm_mean_comparison(cutoff : int, dt : float, group_norm_mean_std : list[dict], plot_title : str, l2_dist : str):
    fig, ax = plt.subplots(1,2, figsize=(10,4),sharex=True)
    fig.suptitle(plot_title, fontweight="bold")
    
    y_labels = ['State norm', 'Action norm']
    lines = []
            
    for i in range(2):
        for j, (key, traj_mean_std) in enumerate(group_norm_mean_std[i].items()):
            r = int(traj_mean_std[0].shape[0] / cutoff)
            t_step = np.arange(r)*dt
            line, = ax[i].plot(t_step,traj_mean_std[0][:r], label=key)
            ax[i].fill_between(
                t_step, 
                traj_mean_std[0][:r]-traj_mean_std[1][:r], 
                traj_mean_std[0][:r]+traj_mean_std[1][:r],
                alpha=0.25
            )
            if i == 0: lines.append(line)
            ax[i].grid(True)
            ax[i].set_xlabel("Time [s]")
            ax[i].set_title(y_labels[i])
    ax[0].legend(handles=lines)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fig.savefig(f"results/norm/{l2_dist}_{'_'.join(plot_title.split())}.png", dpi=300, bbox_inches="tight")