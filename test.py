import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from numpngw import write_apng
from IPython.display import Image
from tqdm.notebook import tqdm
import argparse
import sys

from panda_pushing_env import PandaImageSpacePushingEnv
from visualizers import GIFVisualizer, NotebookVisualizer
from learning_latent_dynamics import *
from utils import *

def test(model, val_loader, loss_fn) -> float:
    """
    Perfoms an epoch of model performance validation
    :param model: Pytorch nn.Module
    :param train_loader: Pytorch DataLoader
    :param loss: Loss function
    :return: val_loss <float> representing the average loss among the different mini-batches
    """
    val_loss = 0. # TODO: Modify the value
    # Initialize the validation loop
    model.eval()
    for batch in val_loader:
        loss = None
        states = batch['states']
        actions = batch['actions']
        loss = loss_fn(model, states, actions)
        val_loss += loss.item()
    return val_loss/len(val_loader)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate SINDy models and run a PyBullet GUI demo.')
    parser.add_argument('--poly-order', type=int, choices=[1, 2, 3, 4], default=3,
                        help='Select the polynomial order for the loaded SINDy model.')
    parser.add_argument('--run-all', action='store_true', help='Evaluate all poly orders 1-4 without starting the GUI.')
    return parser.parse_args()


def load_sindy_model(poly_order, latent_dim, action_dim, num_channels):
    model = SINDyModel(latent_dim=latent_dim,
                       action_dim=action_dim,
                       poly_order=poly_order,
                       include_sine=True,
                       num_channels=num_channels)
    model_path = f'single_step_sindy_dynamics_model_polyorder{poly_order}.pt'
    model.load_state_dict(torch.load(model_path))
    return model


def evaluate_model(model, single_step_loader, multi_step_loader, loss_fn):
    single_loss = test(model, single_step_loader, loss_fn)
    multi_loss = test(model, multi_step_loader, loss_fn)
    return single_loss, multi_loss


if __name__ == "__main__":
    args = parse_args()

    LR = 0.001
    NUM_EPOCHS = 2000
    BETA = 0.001
    LATENT_DIM = 16
    ACTION_DIM = 3
    NUM_CHANNELS = 1
    NUM_STEPS = 1

    collected_data = np.load('pushing_image_data.npy', allow_pickle=True)
    train_loader, val_loader, norm_constants = process_data_multiple_step(collected_data, batch_size=500, num_steps=NUM_STEPS)
    norm_tr = NormalizationTransform(norm_constants)

    test_data = np.load('pushing_image_validation_data.npy', allow_pickle=True)
    single_step_dataset = MultiStepDynamicsDataset(test_data, num_steps=1, transform=norm_tr)
    single_step_loader = DataLoader(single_step_dataset, batch_size=len(single_step_dataset))

    multi_step_dataset = MultiStepDynamicsDataset(test_data, num_steps=4, transform=norm_tr)
    multi_step_loader = DataLoader(multi_step_dataset, batch_size=len(multi_step_dataset))

    state_loss_fn = nn.MSELoss()
    latent_loss_fn = nn.MSELoss()
    loss = MultiStepLoss(state_loss_fn, latent_loss_fn, lambda_state_loss=0.1, lambda_latent_loss=0.1, lambda_reg_loss=0.2)

    if args.run_all:
        for poly_order in [1, 2, 3, 4]:
            print(f'==== POLY_ORDER={poly_order} ====')
            model = load_sindy_model(poly_order, LATENT_DIM, ACTION_DIM, NUM_CHANNELS)
            single_loss, multi_loss = evaluate_model(model, single_step_loader, multi_step_loader, loss)
            print(f'Poly {poly_order}: single-step loss = {single_loss:.6f}, multi-step loss = {multi_loss:.6f}')
        sys.exit(0)

    poly_order = args.poly_order
    single_step_sindy_dynamics_model = load_sindy_model(poly_order, LATENT_DIM, ACTION_DIM, NUM_CHANNELS)
    single_loss, multi_loss = evaluate_model(single_step_sindy_dynamics_model, single_step_loader, multi_step_loader, loss)
    print(f'Single-step model evaluated on single-step loss: {single_loss}')
    print(f'Single-step model evaluated on multi-step loss: {multi_loss}')
    print('')

    print('Launching PyBullet GUI demo...')
    env = PandaImageSpacePushingEnv(debug=True, visualizer=None, render_non_push_motions=False, camera_heigh=800, camera_width=800, render_every_n_steps=5, grayscale=True)
    state = env.reset()
    env.object_target_pose = env._planar_pose_to_world_pose(np.array([0.7, 0., 0.]))
    controller = PushingLatentController(env, single_step_sindy_dynamics_model, latent_space_pushing_cost_function, norm_constants, num_samples=100, horizon=10)

    for i in range(20):
        action = controller.control(state)
        state, reward, done, _ = env.step(action)
        end_pose = env.get_object_pos_planar()
        goal_distance = np.linalg.norm(end_pose[:2]-np.array([0.7,0.])[:2])
        if done or goal_distance < BOX_SIZE:
            print('Total steps to reach goal:', i+1)
            break

    input('PyBullet demo finished. Press Enter to close the window...')

    # multi_step_sindy_dynamics_model = SINDyModel(latent_dim=LATENT_DIM, action_dim=ACTION_DIM, poly_order=POLY_ORDER, include_sine=True, num_channels=NUM_CHANNELS)
    # model_path = 'multi_step_sindy_dynamics_model.pt'
    # multi_step_sindy_dynamics_model.load_state_dict(torch.load(model_path))
    # print(f'Multi-step model evaluated on single-step loss: {test(multi_step_sindy_dynamics_model, single_step_loader, loss)}')
    # print(f'Multi-step model evaluated on multi-step loss: {test(multi_step_sindy_dynamics_model, multi_step_loader, loss)}')
