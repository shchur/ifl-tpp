import dpp
import numpy as np
import torch
from copy import deepcopy
torch.set_default_tensor_type(torch.cuda.FloatTensor)


# Config
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
dataset_name = 'synth/hawkes1'  # run dpp.data.list_datasets() to see the list of available datasets

# Model config
context_size = 64                 # Size of the RNN hidden vector
mark_embedding_size = 32          # Size of the mark embedding (used as RNN input)
num_mix_components = 64           # Number of components for a mixture model
rnn_type = "GRU"                  # What RNN to use as an encoder {"RNN", "GRU", "LSTM"}

# Training config
batch_size = 64        # Number of sequences in a batch
regularization = 1e-5  # L2 regularization parameter
learning_rate = 1e-3   # Learning rate for Adam optimizer
max_epochs = 1000      # For how many epochs to train
display_step = 50      # Display training statistics after every display_step
patience = 50          # After how many consecutive epochs without improvement of val loss to stop training


# Load the data
dataset = dpp.data.load_dataset(dataset_name)
d_train, d_val, d_test = dataset.train_val_test_split(seed=seed)

dl_train = d_train.get_dataloader(batch_size=batch_size, shuffle=True)
dl_val = d_val.get_dataloader(batch_size=batch_size, shuffle=False)
dl_test = d_test.get_dataloader(batch_size=batch_size, shuffle=False)


# Define the model
print('Building model...')
mean_log_inter_time, std_log_inter_time = d_train.get_inter_time_statistics()

model = dpp.models.LogNormMix(
    num_marks=d_train.num_marks,
    mean_log_inter_time=mean_log_inter_time,
    std_log_inter_time=std_log_inter_time,
    context_size=context_size,
    mark_embedding_size=mark_embedding_size,
    rnn_type=rnn_type,
    num_mix_components=num_mix_components,
)
opt = torch.optim.Adam(model.parameters(), weight_decay=regularization, lr=learning_rate)


# Traning
print('Starting training...')

def aggregate_loss_over_dataloader(dl):
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in dl:
            total_loss += -model.log_prob(batch).sum()
            total_count += batch.size
    return total_loss / total_count


impatient = 0
best_loss = np.inf
best_model = deepcopy(model.state_dict())
training_val_losses = []

for epoch in range(max_epochs):
    model.train()
    for batch in dl_train:
        opt.zero_grad()
        loss = -model.log_prob(batch).mean()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        loss_val = aggregate_loss_over_dataloader(dl_val)
        training_val_losses.append(loss_val)

    if (best_loss - loss_val) < 1e-4:
        impatient += 1
        if loss_val < best_loss:
            best_loss = loss_val
            best_model = deepcopy(model.state_dict())
    else:
        best_loss = loss_val
        best_model = deepcopy(model.state_dict())
        impatient = 0

    if impatient >= patience:
        print(f'Breaking due to early stopping at epoch {epoch}')
        break

    if epoch % display_step == 0:
        print(f"Epoch {epoch:4d}: loss_train_last_batch = {loss.item():.1f}, loss_val = {loss_val:.1f}")


# Evaluation
model.load_state_dict(best_model)
model.eval()

# All training & testing sequences stacked into a single batch
with torch.no_grad():
    final_loss_train = aggregate_loss_over_dataloader(dl_train)
    final_loss_val = aggregate_loss_over_dataloader(dl_val)
    final_loss_test = aggregate_loss_over_dataloader(dl_test)

print(f'Negative log-likelihood:\n'
      f' - Train: {final_loss_train:.1f}\n'
      f' - Val:   {final_loss_val:.1f}\n'
      f' - Test:  {final_loss_test:.1f}')
