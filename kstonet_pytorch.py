import argparse
import os
import errno
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.utils.data
import time
from process_data import preprocess_data

# Basic Setting
parser = argparse.ArgumentParser(description='Running K-StoNet with PyTorch')
parser.add_argument('--seed', default=1, type=int, help='set seed')
parser.add_argument('--data_name', default='Boston', type=str, help='specify the name of the data')
parser.add_argument('--base_path', default='./result/', type=str, help='base path for saving result')
parser.add_argument('--model_path', default='torch_run/', type=str, help='folder name for saving model')
parser.add_argument('--cross_validate', default=0, type=int, help='specify which fold of 5 fold cross validation')
parser.add_argument('--regression_flag', default=True, type=int, help='true for regression and false for classification')
parser.add_argument('--normalize_y_flag', default=False, type=int, help='whether to normalize target value or not')
parser.add_argument('--confidence_interval_flag', default=False, type=int, help='whether to store result to compute confidence interval')

# model
parser.add_argument('--layer', default=1, type=int, help='number of hidden layer')
parser.add_argument('--unit', default=[10], type=int, nargs='+', help='number of hidden unit in each layer')
parser.add_argument('--sigma', default=[0.001], type=float, nargs='+', help='variance of each layer for the hidden variable model')
parser.add_argument('--C', default=5.0, type=float, help='C parameter (regularization)')
parser.add_argument('--epsilon', default=0.01, type=float, help='epsilon parameter')

# Training Setting
parser.add_argument('--nepoch', default=50, type=int, help='total number of training epochs')
parser.add_argument('--MH_step', default=25, type=int, help='SGLD step for imputation')
parser.add_argument('--lr', default=[0.00005], type=float, nargs='+', help='step size in imputation')
parser.add_argument('--alpha', default=0.1, type=float, help='momentum parameter for HMC')
parser.add_argument('--temperature', default=[1], type=float, nargs='+', help='temperature parameter for HMC')
parser.add_argument('--lasso', default=[0.0001], type=float, nargs='+', help='lambda parameter for L1 regularization')
parser.add_argument('--batch_size', default=100, type=int, help='batch size for SVR training')
parser.add_argument('--svr_epochs', default=50, type=int, help='number of epochs for SVR training')
parser.add_argument('--svr_lr', default=0.001, type=float, help='learning rate for SVR training')
parser.add_argument('--p_gamma', default=None, type=float, help='gamma parameter for RBF kernel')

args = parser.parse_args()

class RBFLayer(nn.Module):
    """Custom RBF kernel layer replacement for SVR"""
    def __init__(self, in_features, out_features, gamma=0.1):
        super(RBFLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centers = nn.Parameter(torch.Tensor(out_features, in_features))
        self.gamma = gamma
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.normal_(self.centers, 0, 1)
        
    def forward(self, input):
        size = (input.size(0), self.out_features, self.in_features)
        x = input.unsqueeze(1).expand(size)
        c = self.centers.unsqueeze(0).expand(size)
        distances = (x - c).pow(2).sum(-1).neg().mul(self.gamma)
        return torch.exp(distances)

class PyTorchSVR(nn.Module):
    """PyTorch implementation of SVR"""
    def __init__(self, input_dim, hidden_dim=50, gamma=0.1, epsilon=0.1):
        super(PyTorchSVR, self).__init__()
        self.rbf = RBFLayer(input_dim, hidden_dim, gamma)
        self.linear = nn.Linear(hidden_dim, 1)
        self.epsilon = epsilon
        
    def forward(self, x):
        x = self.rbf(x)
        x = self.linear(x)
        return x
    
    def epsilon_insensitive_loss(self, y_pred, y_true):
        """Epsilon-insensitive loss function for SVR"""
        return torch.mean(torch.max(torch.abs(y_pred - y_true) - self.epsilon, torch.zeros_like(y_pred)))
    
    def loss_function(self, y_pred, y_true, l1_weight=0.01):
        """Combined loss function with L1 regularization"""
        l1_loss = sum(p.abs().sum() for p in self.parameters())
        return self.epsilon_insensitive_loss(y_pred, y_true) + l1_weight * l1_loss

class KStoNet(nn.Module):
    """Complete K-StoNet model in PyTorch"""
    def __init__(self, input_dim, hidden_dims, output_dim, gamma=0.1, epsilon=0.1):
        super(KStoNet, self).__init__()
        self.num_hidden = len(hidden_dims)
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Create SVR layers for first layer
        self.svr_layers = nn.ModuleList([
            PyTorchSVR(input_dim, hidden_dim=50, gamma=gamma, epsilon=epsilon) 
            for _ in range(hidden_dims[0])
        ])
        
        # Create FC layers for subsequent layers
        self.fc_layers = nn.ModuleList()
        for i in range(self.num_hidden - 1):
            self.fc_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        
        # Final output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
    
    def forward(self, x):
        # Get outputs from SVR layers
        svr_outputs = []
        for svr in self.svr_layers:
            svr_outputs.append(svr(x).squeeze(-1))
        
        # Combine SVR outputs
        hidden = torch.stack(svr_outputs, dim=1)
        
        # Apply tanh activation
        hidden = torch.tanh(hidden)
        
        # Apply subsequent FC layers
        for fc in self.fc_layers:
            hidden = torch.tanh(fc(hidden))
        
        # Apply output layer
        output = self.output_layer(hidden)
        
        return output
    
    def get_first_layer_output(self, x):
        """Get the output of the first layer (SVR layers)"""
        svr_outputs = []
        for svr in self.svr_layers:
            svr_outputs.append(svr(x).squeeze(-1))
        
        return torch.stack(svr_outputs, dim=1)

def train_svr_layers(model, x_train, hidden_targets, lr=0.001, epochs=50, batch_size=100, l1_weight=0.01):
    """Train the SVR layers of the model"""
    device = x_train.device
    svr_optimizers = [optim.Adam(model.svr_layers[i].parameters(), lr=lr) for i in range(len(model.svr_layers))]
    
    # Create DataLoader
    dataset = torch.utils.data.TensorDataset(x_train, hidden_targets)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Train each SVR layer
    for i, svr in enumerate(model.svr_layers):
        print(f"Training SVR layer {i+1}/{len(model.svr_layers)}")
        optimizer = svr_optimizers[i]
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_hidden in dataloader:
                optimizer.zero_grad()
                outputs = svr(batch_x).squeeze(-1)
                loss = svr.epsilon_insensitive_loss(outputs, batch_hidden[:, i])
                l1_loss = sum(p.abs().sum() for p in svr.parameters())
                total_loss = loss + l1_weight * l1_loss
                total_loss.backward()
                optimizer.step()
                epoch_loss += total_loss.item()
            
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.6f}")

def train_fc_layers(model, x_train, y_train, hidden_targets, lr=0.001, epochs=50, l1_weight=0.01):
    """Train the FC layers of the model"""
    device = x_train.device
    
    # Freeze SVR layers
    for svr in model.svr_layers:
        for param in svr.parameters():
            param.requires_grad = False
    
    # Create optimizer for FC layers
    fc_params = list(model.fc_layers.parameters()) + list(model.output_layer.parameters())
    optimizer = optim.Adam(fc_params, lr=lr, weight_decay=l1_weight)
    
    # Loss function
    if model.output_dim == 1:  # Regression
        criterion = nn.MSELoss()
    else:  # Classification
        criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass with frozen SVR layers
        first_layer_output = model.get_first_layer_output(x_train)
        
        # Apply tanh activation
        hidden = torch.tanh(first_layer_output)
        
        # Apply subsequent FC layers
        for fc in model.fc_layers:
            hidden = torch.tanh(fc(hidden))
        
        # Apply output layer
        output = model.output_layer(hidden)
        
        # Compute loss
        if model.output_dim == 1:  # Regression
            output = output.squeeze(-1)
            loss = criterion(output, y_train)
        else:  # Classification
            loss = criterion(output, y_train)
        
        # Add L1 regularization
        l1_loss = sum(p.abs().sum() for p in fc_params)
        total_loss = loss + l1_weight * l1_loss
        
        total_loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f"FC Layers - Epoch {epoch+1}/{epochs}, Loss: {total_loss.item():.6f}")

def main():
    import pickle
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data_name = args.data_name
    cross_validate_index = args.cross_validate

    num_hidden = args.layer
    hidden_dim = args.unit

    regression_flag = args.regression_flag
    normalize_y_flag = args.normalize_y_flag

    num_epochs = args.nepoch

    # Load data
    x_train, y_train, x_test, y_test = preprocess_data(data_name, cross_validate_index, seed=args.seed)

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Move data to device
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    ntrain = x_train.shape[0]
    ntest = x_test.shape[0]
    dim = x_train.shape[1]

    # Set up loss functions
    if regression_flag:
        output_dim = 1
        loss_func = nn.MSELoss()
        train_loss_path = np.zeros(num_epochs)
        test_loss_path = np.zeros(num_epochs)
        if normalize_y_flag:
            y_train_mean = y_train.mean()
            y_train_std = y_train.std()
            y_train = (y_train - y_train_mean) / y_train_std
    else:
        output_dim = int((y_test.max() + 1).item())
        loss_func = nn.CrossEntropyLoss()
        train_loss_path = np.zeros(num_epochs)
        test_loss_path = np.zeros(num_epochs)
        train_accuracy_path = np.zeros(num_epochs)
        test_accuracy_path = np.zeros(num_epochs)
    time_used_path = np.zeros(num_epochs)

    # Gamma parameter for RBF kernel
    if args.p_gamma is None:
        gamma = 1.0 / (dim * x_train.var().item())
    else:
        gamma = 1.0 / args.p_gamma

    # Create model
    model = KStoNet(dim, hidden_dim, output_dim, gamma=gamma, epsilon=args.epsilon).to(device)

    # Create directory for saving results
    PATH = args.base_path + data_name + '/' + str(cross_validate_index) + '/' + args.model_path
    if not os.path.isdir(PATH):
        try:
            os.makedirs(PATH)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(PATH):
                pass
            else:
                raise

    # Initialize hidden state
    temp_init = nn.Linear(dim, hidden_dim[0]).to(device)
    hidden_init = temp_init(x_train)

    # Training loop
    for epoch in range(num_epochs):
        start_time = time.process_time()
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # MH sampling for hidden variables
        hidden_list = [hidden_init]
        momentum_list = [torch.zeros_like(hidden_init)]
        
        # Create subsequent hidden layers
        with torch.no_grad():
            for i in range(num_hidden - 1):
                if i < len(model.fc_layers):
                    hidden_list.append(model.fc_layers[i](torch.tanh(hidden_list[-1])))
                    momentum_list.append(torch.zeros_like(hidden_list[-1]))
        
        # Enable gradients for MH sampling
        for i in range(len(hidden_list)):
            hidden_list[i].requires_grad = True
        
        # Get parameters from args
        MH_step = args.MH_step
        alpha = args.alpha
        sigma_list = args.sigma
        if len(sigma_list) == 1 and num_hidden > 1:
            sigma_list = [sigma_list[0]] * num_hidden
        
        proposal_lr = args.lr
        if len(proposal_lr) == 1 and num_hidden > 1:
            proposal_lr = [proposal_lr[0]] * num_hidden
        
        temperature = args.temperature
        if len(temperature) == 1 and num_hidden > 1:
            temperature = [temperature[0]] * num_hidden
        
        # Get forward hidden state from model
        with torch.no_grad():
            forward_hidden = model.get_first_layer_output(x_train)
        
        # MH sampling
        for repeat in range(MH_step):
            for layer_index in reversed(range(num_hidden)):
                if hidden_list[layer_index].grad is not None:
                    hidden_list[layer_index].grad.zero_()
                
                # Compute hidden likelihood
                if layer_index == num_hidden - 1:
                    if layer_index < len(model.fc_layers):
                        output = model.output_layer(torch.tanh(model.fc_layers[layer_index](torch.tanh(hidden_list[layer_index]))))
                    else:
                        output = model.output_layer(torch.tanh(hidden_list[layer_index]))
                    
                    if regression_flag:
                        output = output.squeeze(-1)
                        hidden_likelihood = -((output - y_train).pow(2).sum()) / sigma_list[layer_index]
                    else:
                        hidden_likelihood = -loss_func(output, y_train) * ntrain / sigma_list[layer_index]
                else:
                    next_hidden = model.fc_layers[layer_index](torch.tanh(hidden_list[layer_index]))
                    hidden_likelihood = -((next_hidden - hidden_list[layer_index + 1]).pow(2).sum()) / sigma_list[layer_index]
                
                if layer_index == 0:
                    # Add SVR-like constraint for first layer
                    epsilon = args.epsilon
                    C = args.C
                    hidden_likelihood -= C * torch.where(
                        (hidden_list[layer_index] - forward_hidden).abs() - epsilon > 0,
                        (hidden_list[layer_index] - forward_hidden).abs() - epsilon,
                        torch.zeros_like(hidden_list[0])
                    ).sum()
                else:
                    # Constraint from previous layer
                    prev_hidden = model.fc_layers[layer_index - 1](torch.tanh(hidden_list[layer_index - 1]))
                    hidden_likelihood -= ((prev_hidden - hidden_list[layer_index]).pow(2).sum()) / sigma_list[layer_index - 1]
                
                # Compute gradient
                hidden_likelihood.backward()
                
                # Update with momentum
                step_proposal_lr = proposal_lr[layer_index]
                with torch.no_grad():
                    momentum_list[layer_index] = (
                        (1 - alpha) * momentum_list[layer_index] + 
                        step_proposal_lr / 2 * hidden_list[layer_index].grad + 
                        torch.FloatTensor(hidden_list[layer_index].shape).to(device).normal_().mul(
                            np.sqrt(alpha * step_proposal_lr * temperature[layer_index])
                        )
                    )
                    hidden_list[layer_index].data += momentum_list[layer_index]
        
        # Train SVR layers
        train_svr_layers(
            model, 
            x_train, 
            hidden_list[0].detach(), 
            lr=args.svr_lr, 
            epochs=args.svr_epochs, 
            batch_size=args.batch_size, 
            l1_weight=args.lasso[0] if len(args.lasso) > 0 else 0.0001
        )
        
        # Train FC layers
        train_fc_layers(
            model, 
            x_train, 
            y_train, 
            hidden_list[0].detach(), 
            lr=args.svr_lr, 
            epochs=args.svr_epochs, 
            l1_weight=args.lasso[0] if len(args.lasso) > 0 else 0.0001
        )
        
        # Evaluate model
        model.eval()
        with torch.no_grad():
            # Train evaluation
            train_output = model(x_train)
            
            if regression_flag:
                train_loss = loss_func(train_output.squeeze(-1), y_train)
                if normalize_y_flag:
                    train_loss = train_loss * y_train_std * y_train_std
                train_loss_path[epoch] = train_loss.item()
                print(f"Train loss: {train_loss.item():.6f}")
                
                # Test evaluation
                test_output = model(x_test)
                if normalize_y_flag:
                    test_output = test_output * y_train_std + y_train_mean
                test_loss = loss_func(test_output.squeeze(-1), y_test)
                test_loss_path[epoch] = test_loss.item()
                print(f"Test loss: {test_loss.item():.6f}")
            else:
                train_loss = loss_func(train_output, y_train)
                train_loss_path[epoch] = train_loss.item()
                prediction = train_output.data.max(1)[1]
                train_accuracy = prediction.eq(y_train.data).sum().item() / ntrain
                train_accuracy_path[epoch] = train_accuracy
                print(f"Train loss: {train_loss.item():.6f}, Train accuracy: {train_accuracy:.4f}")
                
                # Test evaluation
                test_output = model(x_test)
                test_loss = loss_func(test_output, y_test)
                test_loss_path[epoch] = test_loss.item()
                prediction = test_output.data.max(1)[1]
                test_accuracy = prediction.eq(y_test.data).sum().item() / ntest
                test_accuracy_path[epoch] = test_accuracy
                print(f"Test loss: {test_loss.item():.6f}, Test accuracy: {test_accuracy:.4f}")
        
        # Save model
        torch.save(model.state_dict(), PATH + f'model{epoch}.pt')
        
        # Save hidden state if confidence interval flag is set
        if args.confidence_interval_flag:
            filename = PATH + f'hidden_state{epoch}.pt'
            f = open(filename, 'wb')
            pickle.dump(hidden_list, f, protocol=4)
            f.close()
        
        # Record time used
        end_time = time.process_time()
        time_used_path[epoch] = end_time - start_time
        print(f"Time used: {end_time - start_time:.2f}s")
        
        # Save results
        if regression_flag:
            filename = PATH + 'result.txt'
            f = open(filename, 'wb')
            pickle.dump([train_loss_path, test_loss_path, time_used_path], f)
            f.close()
        else:
            filename = PATH + 'result.txt'
            f = open(filename, 'wb')
            pickle.dump([train_loss_path, test_loss_path, train_accuracy_path, test_accuracy_path, time_used_path], f)
            f.close()
        
        # Save data for confidence interval if required
        if args.confidence_interval_flag:
            filename = PATH + 'data.txt'
            f = open(filename, 'wb')
            pickle.dump([x_train, x_test, y_train, y_test], f)
            f.close()

if __name__ == '__main__':
    main()