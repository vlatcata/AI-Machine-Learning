import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Two hidden layers for better pattern recognition
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, state):
        # ReLU(x) = max(0, x) -> activation function
        state = F.relu(self.linear1(state))
        state = F.relu(self.linear2(state))
        state = self.linear3(state)
        return state
    
    def save(self, file_name='model.pth'):
        model_folder_path = '.\model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='model.pth'):
        model_folder_path = '.\model'
        file_name = os.path.join(model_folder_path, file_name)

        if os.path.exists(file_name):
            self.load_state_dict(torch.load(file_name))
            self.eval()  # Set the model to evaluation mode
            print(f'Model loaded from {file_name}')
            return True
        else:
            print(f'No saved model found at {file_name}')
            return False


class QTrainer:
    def __init__(self, model, learning_rate, gamma):
        self.learning_rate = learning_rate
        self.model = model
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)

        # Checks shape of the state, It's 1 if it's a single state and 2 if it's a batch of states
        # If it's a single state, we need to add a batch dimension at the beginning
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            next_state = torch.unsqueeze(next_state, 0)
            done = (done, )

        # Early on, the predictions are random values, the more we correct the model, the more accurate the predictions become
        # predicted Q values with old state
        pred = self.model(state)

        target = pred.clone()
        for index in range(len(done)): # Iterating over all games (all inputs have the same size/length)
            Q_new = reward[index]
            if not done[index]:
                # Q_new = r + y * max(next_predicted Q value) -> only do this if not done
                Q_new = reward[index] + self.gamma * torch.max(self.model(next_state[index]))

            # We update the value of the action taken in the old state
            # target[batch_index][max Q value index] = Q_new | argmax returns the index of the max Q value
            target[index][torch.argmax(action).item()] = Q_new 

        # We calculate the loss based on the model’s prediction and the actual value.
        # Then we use backpropagation to compute the gradient (steepest increase) of the loss function with respect to each weight.
        # Then we use gradient descent to update the weights/parameters in the opposite direction, to reduce the loss. weight := weight - lr * grad(loss)
        # Weight: parameters of the model (float values)
        self.optimizer.zero_grad() # clear previous gradients
        loss = self.criterion(target, pred)
        loss.backward() # backpropagation - compute gradients using greadient descent

        self.optimizer.step() # update weights