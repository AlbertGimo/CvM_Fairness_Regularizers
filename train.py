# .py file containing methods for training models with mitigation (regularization) techniques
import torch
import torch.nn as nn
import datasets
from typing import Callable
import numpy as np
import random


from wandb_utils import log_metrics, log_model_artifact

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# define MLP model to train
class MLP(nn.Module):
    def __init__(self,input_dim,hiden_layers,output_dim=1,dropout=0.0,device='cpu'):
        super(MLP, self).__init__()
        self.model_type = 'MLP'
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hiden_layers[0]))
        self.layers.append(nn.ReLU())
        for i in range(len(hiden_layers) - 1):
            self.layers.append(nn.Linear(hiden_layers[i], hiden_layers[i + 1]))
            self.layers.append(nn.ReLU())
            if dropout > 0:
                self.layers.append(nn.Dropout(dropout))
        self.layers.append(nn.Linear(hiden_layers[-1], output_dim))
        self.device = device
        self.to(device)
        

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x

from regularizers.CvM import compute_cvm_chatterjee_autodiff
def get_cvm_mitigator(output, target, sensitive, args, regularization_strength=1e-8):
    if args.fairness_metric_goal == "DP":
        return compute_cvm_chatterjee_autodiff(sensitive, output, regularization_strength=args.epsilon)
    elif args.fairness_metric_goal == "EO":
        # compute separate CvM regularization terms for the positive and negative classes and take the average
        assert args.dataset_info["task"] == "classification" and args.dataset_info["num_classes"] == 2, "EO regularization is only applicable for binary classification tasks."
        positive_mask = target.squeeze() == 1
        negative_mask = target.squeeze() == 0
        cvm_positive = compute_cvm_chatterjee_autodiff(sensitive[positive_mask], output[positive_mask], regularization_strength=regularization_strength)
        cvm_negative = compute_cvm_chatterjee_autodiff(sensitive[negative_mask], output[negative_mask], regularization_strength=regularization_strength)
        return (cvm_positive + cvm_negative) / 2
    else:
        raise ValueError("Invalid fairness goal. Please choose from 'DP' or 'EO'.")

from regularizers.DISCO import sDISCO_regularizer
from sklearn.gaussian_process.kernels import RBF
def get_disco_mitigator(output, target, sensitive, args):
    RBF_kernel = RBF()
    return sDISCO_regularizer(pred=output, ground_true=target, sensitive_attr=sensitive, kernel=lambda x: RBF_kernel.__call__(X=x, eval_gradient=False))

from regularizers.DC_CDC import ComputeDCov_sqr
def get_dc_mitigator(output, target, sensitive, args):
    return ComputeDCov_sqr(output, sensitive.reshape(-1,1).float())

from regularizers.DC_CDC import ComputeCDC
def get_cdc_mitigator(output, target, sensitive, args):
    return ComputeCDC(output, sensitive.reshape(-1,1).float(), target.reshape(-1,1).float())

def get_mitigator(args):
    # code for initializing the mitigator based on the specified method
    if args.method.lower() == 'cvm':
        return get_cvm_mitigator
    elif args.method.lower() == "dc":
        return get_dc_mitigator
    elif args.method.lower() == "cdc":
        return get_cdc_mitigator
    elif args.method.lower() == "disco":
        return get_disco_mitigator
    
    else:
        raise NotImplementedError("Regylarizer method not implemented. Please choose from 'CvM', 'DC', 'CDC', 'DISCO', or None.")

def train_model(args, train_dataset, val_dataset, lam, trial=None):
    # code for training the model with the specified lambda and method
    model = MLP(input_dim=args.dataset_info["num_features"], hiden_layers=args.MLP_hidden_layers, output_dim=1, dropout=args.dropout, device=args.device)

    if args.finetuning: 
        base_model, _ = train_model_erm(model,train_dataset, args)
        mitigator = get_mitigator(args)
        model, avg_losses, avg_fair_losses = train_model_with_regularizer(base_model, train_dataset, val_dataset, args, lam, mitigator)
    else:
        if lam == 0.0:
            model, avg_losses = train_model_erm(model,train_dataset, args)
        else:
            mitigator = get_mitigator(args)
            model, avg_losses, avg_fair_losses = train_model_with_regularizer(model, train_dataset, val_dataset, args, lam, mitigator)
    return model, avg_losses, avg_fair_losses if lam != 0.0 else None

def train_model_erm(model, train_dataset, args):
    # code for training the model with ERM (no regularization)
    train_loader = datasets.getDataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.train_num_workers, pin_memory=True, shuffle=True)
    criterion = nn.BCEWithLogitsLoss() if args.dataset_info["task"] == "classification" else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    device = model.device
    avg_losses = []
    # training loop for ERM
    for epoch in range(args.epochs):
        epoch_losses = []
        for batch_idx, (data, target, sensitive) in enumerate(train_loader):
            data, target, sensitive = data.to(device), target.to(device), sensitive.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            
            lr = optimizer.param_groups[0]['lr']
            if lr < 1e-6:
                print("Early stop training due to lr < 1e-6")
                break
        scheduler.step()
        avg_losses.append(sum(epoch_losses) / len(epoch_losses))
    return model, avg_losses
        
def train_model_with_regularizer(model, train_dataset, val_dataset, args, lam, mitigator, trial = None):
    # code for training the model with the specified regularization technique and lambda
    train_loader = datasets.getDataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.train_num_workers, pin_memory=True, shuffle=True)
    val_loader = datasets.getDataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.train_num_workers, pin_memory=True, shuffle=False)
    criterion = nn.BCEWithLogitsLoss() if args.dataset_info["task"] == "classification" else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    device = model.device
    obj, utility_obj, fairness_obj = get_obj_functions(args)
    avg_losses = []
    avg_fair_losses = []

    # training loop for regularized model
    for epoch in range(args.finetuning_epochs if args.finetuning else args.epochs):
        epoch_losses = []
        epoch_fair_losses = []
        for batch_idx, (data, target, sensitive) in enumerate(train_loader):
            data, target, sensitive = data.to(device), target.to(device).float(), sensitive.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            # compute the regularization term based on the specified method and lambda
            fair_loss = mitigator(output, target, sensitive, args)
            # if batch_idx == 0:
            #     print(f"Fair loss at epoch {epoch+1}, batch {batch_idx+1}: {fair_loss.item():.4f}")
            total_loss = loss + lam * fair_loss
            total_loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            epoch_fair_losses.append(fair_loss.item()) 
        
        avg_losses.append(sum(epoch_losses) / len(epoch_losses))
        avg_fair_losses.append(sum(epoch_fair_losses) / len(epoch_fair_losses))
        val_metric = evaluate(model, val_loader, device, args, utility_obj=utility_obj, fairness_obj=fairness_obj, obj=obj)
        if args.wandb_logging:
            log_metrics(
                {
                    "epoch": epoch,
                    "train_loss": avg_losses[-1],
                    "fair_loss": avg_fair_losses[-1],
                    "utility_loss": avg_losses[-1] - lam * avg_fair_losses[-1],
                    "lr": optimizer.param_groups[0]['lr'],
                },
                step=epoch,
            )

            # deal with val_metric being a tuple of (utility_metric, fairness_metric) in the case of multi-objective optimization
            if isinstance(val_metric, tuple):
                log_metrics(
                    {
                        "val_utility_metric": val_metric[0],
                        "val_fairness_metric": val_metric[1],
                    },
                    step=epoch,)
                if trial is not None:
                    trial.report(val_metric[0], epoch)
                    trial.report(val_metric[1], epoch)
            else:
                log_metrics(
                    {
                        "val_metric": val_metric,
                    },
                    step=epoch,)
                if trial is not None:
                    trial.report(val_metric, epoch)

        # if best_metric is None or val_metric < best_metric:
        #     best_metric = val_metric

        #     torch.save(model.state_dict(), "best_model.pt")
        #     log_model_artifact("best_model.pt", name="best-model")

    

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        if lr < 1e-6:
            print("Early stop training due to lr < 1e-6")
            break
    return model, avg_losses, avg_fair_losses

def evaluate(model, loader, device, args, utility_obj=None, fairness_obj=None, obj=None):

    model.eval()
    total = 0
    if args.pareto:
        with torch.no_grad():

            for batch in loader:

                x, y, sensitive = batch
                x = x.to(device)
                y = y.to(device)
                sensitive = sensitive.to(device)

                output = model(x)

                utility_score = utility_obj(y_true=y, y_pred=output)
                fairness_score = fairness_obj(y_true=y, y_pred=output, sensitive_features=sensitive)

                total_utility += utility_score.item()
                total_fairness += fairness_score.item()

        return total_utility / len(loader), total_fairness / len(loader)

    else:
        with torch.no_grad():

            for batch in loader:

                x, y, sensitive = batch
                x = x.to(device)
                y = y.to(device)
                sensitive = sensitive.to(device)

                output = model(x)

                score = obj(output, y, sensitive, args)

                total += score.item()

        return total / len(loader)
    
def get_obj_functions(args, incline = 1.0):

    from sklearn.metrics import accuracy_score
    from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
    if args.pareto:
        # accuracy as utility measure
        utility_obj = accuracy_score if args.dataset_info["task"] == "classification" else nn.MSELoss()
        fairness_obj = demographic_parity_difference if args.fairness_metric_goal == "DP" else equalized_odds_difference
        return None, utility_obj, fairness_obj
    else:
        if args.fairness_metric_goal == "DP":
            def obj(output, target, sensitive, args):
                utility_loss = nn.BCEWithLogitsLoss()(output, target) if args.dataset_info["task"] == "classification" else nn.MSELoss()(output, target)
                fair_loss = demographic_parity_difference(target.cpu().numpy(), (output.cpu().numpy() >= 0.5).astype(int), sensitive_features=sensitive.cpu().numpy())
                return utility_loss + incline * fair_loss
            return obj, None, None
        elif args.fairness_metric_goal == "EO":
            def obj(output, target, sensitive, args):
                utility_loss = nn.BCEWithLogitsLoss()(output, target) if args.dataset_info["task"] == "classification" else nn.MSELoss()(output, target)
                fair_loss = equalized_odds_difference(target.cpu().numpy(), (output.cpu().numpy() >= 0.5).astype(int), sensitive_features=sensitive.cpu().numpy())
                return utility_loss + incline *fair_loss
            return obj, None, None