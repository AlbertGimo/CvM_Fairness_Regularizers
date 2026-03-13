import argparse 
from tqdm.auto import tqdm
import train
import datasets
import model_evaluation
import model_evaluation_copy
import warnings
import utils
import fairlearn
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
import regularizers.CvM as cvm

if __name__ == "__main__":

    warnings.filterwarnings("ignore", message=".*pin_memory.*MPS.*")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="adult", choices=["adult","acs"],
                        help="Choose a tarbular dataset.")
    # parser.add_argument("--split_list", type=str, default="0.7,0.5")
    # parser.add_argument("--train_num_workers", type=int, default=8)
    # parser.add_argument("--test_num_workers", type=int, default=4)
    parser.add_argument("--train_num_workers", type=int, default=0)
    parser.add_argument("--test_num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    # parser.add_argument("--device", type=str, default="cuda", choices=["cuda","cpu"], help="Choose a device.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cuda","cpu"],
                        help="Choose a device.")
    # parser.add_argument("--verbose", action="store_true")


    parser.add_argument('--lam', type=float, default=.0, help="balanced hyperparameter: lambda")
    parser.add_argument('--method', type=str, default='CvM', choices=['ERM', 'CvM', 'DC', 'CDC', 'DISCO', None], help="Choose an unfairness mitigation method.")
    parser.add_argument('--beta', type=float, default=0, 
                        help="balanced hyperparameter: ascent rate of lambda")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial Learning rate")
    parser.add_argument("--epochs", type=int, default=40, help="Epochs")

    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="fairness_regularizers", 
                        help="Weights & Biases project name")
    parser.add_argument("--goal", type=str, default="DP", choices=["DP", "EO"], help="Fairness goal")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate for MLP model")
    parser.add_argument("--MLP_hidden_layers", type=list, default=[200, 200, 20], help="Hidden layer sizes for MLP model")
    parser.add_argument("--split_list", type=list, default=[0.7, 0.5], help="Train/Val/Test split ratios in the form [train_size, val_size] where val_size is the proportion of the testvalid set to be used as validation set. Default is [0.7, 0.5], which means 70% train, 15% val, 15% test.")
    parser.add_argument("--fairness_metric_goal", type=str, default="DP", choices=["DP", "EO"], help="Fairness metric to optimize for when using a regularization method. Default is 'DP' (Demographic Parity). If the goal is 'EO' (Equalized Odds), the regularizer will optimize for equalized odds instead.")
    parser.add_argument("--epsilon", type=float, default=1e-8, help="Small constant for numerical stability in regularization terms (if necessary).")
    parser.add_argument("--finetuning", type=bool, default=False, help="whether the mitigation method should be used fro finetuning")
    parser.add_argument("--finetuning_epochs", type=int, default=20, help="number of finetuning epochs for finetuning")
    parser.add_argument("--pareto", action="store_true", help="Whether to perform Pareto optimization instead of single-objective optimization. If set, the study will optimize for both accuracy and fairness metric simultaneously and return a Pareto front of optimal solutions instead of a single best trial.")
    parser.add_argument("--wandb_logging", action="store_true", help="Whether to log training metrics and artifacts to Weights & Biases. If set, each trial will be logged as a separate W&B run under the specified project and group.")
    args = parser.parse_args()

    utils.seed_everything(args.seed)

    lambdas = []
    if args.method == 'ERM' or args.method is None:
        print('Training with ERM...')
        lambdas = [0.0, 0.0, 0.0, 0.0]
    elif args.method == 'CvM':
        print(f'Training with CvM regularizer with {args.goal} goal...')
        if args.goal == 'DP':
            lambdas = [0.001, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
            # lambdas = [0.001, 0.01]
        else:
            lambdas = [0.0, 1.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 40.0]
    elif args.method.lower() == 'dc':
        lambdas = [0.0, 0.1, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 40]
    elif args.method.lower() == 'cdc':
        lambdas = [0.0, 0.1, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 40]
    elif args.method.lower() == "disco":
        if args.finetuning == False:
            lambdas = [0.0, 0.1, 0.5, 1, 2, 3, 5, 7, 10, 20]
        else:
            lambdas = [0.0, 1, 5, 7, 10, 15, 20, 30, 40]
    else:
        raise ValueError("Invalid method choice. Please choose from 'ERM', 'CvM', 'DC', 'CDC', 'DISCO', or None.")
    
    print(args.split_list)
    # get the dataset once and perform the same partitioning for all methods and lambdas to ensure a fair comparison
    # X_train, X_val, X_test, y_train, y_val, y_test, s_train, s_val, s_test, dataset_info = datasets.datasetPreprocessing("../dataset/", dataset_name=args.dataset, split_list=args.split_list, seed=args.seed, sensitive_attribute="sex", to_tensors=True, verbose=False)
    train_dataset, val_dataset, test_dataset, dataset_info = datasets.datasetPreprocessing("../dataset/", dataset_name=args.dataset, split_list=args.split_list, seed=args.seed, sensitive_attribute="sex", to_tensors=True, verbose=False)
    # print(dataset_info)
    args.dataset_info = dataset_info
    print(args.dataset_info)

    # train_dataset = (X_train, y_train, s_train)
    # for x in train_dataset:
    #     print(x.shape)

    accuracies = []
    fairness_metrics = []
    CvMs = []
    for lam in tqdm(lambdas):
        # train the model with the specified lambda and method
        model, avg_losses, avg_fair_losses = train.train_model(args, train_dataset, val_dataset, lam)
        accuracy = model_evaluation_copy.evaluate_model(model, val_dataset[:][0], val_dataset[:][1])
        accuracies.append(accuracy)
        probabilities = model(val_dataset[:][0].to(model.device)).squeeze().cpu().detach().numpy()
        binary_predictions = (probabilities >= 0.5).astype(int)   
        if args.goal == "DP":
            fairness_metric = demographic_parity_difference(val_dataset[:][1].squeeze().cpu().numpy(), binary_predictions, sensitive_features=val_dataset[:][2].squeeze().cpu().numpy())
        elif args.goal == "EO":
            fairness_metric = equalized_odds_difference(val_dataset[:][1].squeeze().cpu().numpy(), binary_predictions, sensitive_features=val_dataset[:][2].squeeze().cpu().numpy())
        fairness_metrics.append(fairness_metric)
        CvMs.append(cvm.compute_cvm_classic(val_dataset[:][2].cpu().detach(), model(val_dataset[:][0].to(model.device)).cpu().detach()))
    # print(type(test_dataset[:][2]), test_dataset[:][2].shape)

    print("epsilon for numerical stability in regularization: ", args.epsilon)
    for lam, acc, fm, cvm_value in zip(lambdas, accuracies, fairness_metrics, CvMs):
        print(f"Lambda: {lam:.4f}, Accuracy: {acc:.4f}, Fairness Metric ({args.goal}): {fm:.4f}, CvM: {cvm_value:.4f}")
    print('All Done!')

