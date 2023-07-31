import torch
from network import *
from s2s_dataset import *
from losses import ContrastiveLoss
import time
import math
from torch.utils.data import random_split, ConcatDataset

class MyDataset(Dataset):
    def __init__(self, activity, label, ratio):
        self.data = activity
        self.label = label
        self.ratio = ratio

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx], self.ratio[idx]

    def __len__(self):
        return len(self.label)



#device = torch.device('cpu')#('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cuda = torch.cuda.is_available()
train_dataset = S2sDataset(usage="train", path="/home/ubuntu/wcmc_attack_activity_detect_code/test/miss_s2s_msg64_more")
miss_test_dataset = S2sDataset(usage="test", path="/home/ubuntu/wcmc_attack_activity_detect_code/test/miss_s2s_msg64_more")
false_test_dataset = S2sDataset(usage="test", path="/home/ubuntu/wcmc_attack_activity_detect_code/test/false_s2s_msg64_more")

BATCH_SIZE = 128
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
miss_test_loader = torch.utils.data.DataLoader(miss_test_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)
false_test_loader = torch.utils.data.DataLoader(false_test_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)

INPUT_DIM = 64
OUTPUT_DIM = 64
HID_DIM = 128

#INPUT_DIM = 512
#OUTPUT_DIM = 512
#HID_DIM = 1024
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, INPUT_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, device).to(device)


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


model.apply(init_weights)

#def count_parameters(model):
#    return sum(p.numel() for p in model.parameters() if p.requires_grad)
#print(f'The model has {count_parameters(model):,} trainable parameters')

optimizer = optim.Adam(model.parameters(), lr=0.0005)

criterion = ContrastiveLoss()


def train(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0

    for batch_idx, (src, trg, label, ratio) in enumerate(train_loader):
        #print("batch_ids is : ", batch_idx)
        src = src.cuda() #128,50,512
        trg = trg.cuda() #128,14,512
        optimizer.zero_grad()
        output = model(src, trg)

        #print(src)
        output_dim1 = output.shape[-1]
        output_dim2 = output.shape[-2]
        trg_dim1 = output.shape[-1]
        trg_dim2 = output.shape[-2]

        output = output.view(-1, output_dim1 * output_dim2)
        trg = trg.view(-1, trg_dim1 * trg_dim2)
        loss = criterion(output, trg)

        loss.requires_grad_(True)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, train_path, test_path, train_phase_path, test_phase_path):
    model.eval()
    epoch_loss = 0
    train_datasets = []
    test_datasets = []
    train_phase_datasets = []
    test_phase_datasets = []


    start = time.time()
    with torch.no_grad():
        for batch_idx, (src, trg, label, ratio) in enumerate(train_loader):
            #print("dataloader src", src.shape)
            #print(trg.shape)
            src = src.cuda() #128,50,512
            trg = trg.cuda() #128,14,512
            label = label.cuda()
            output = model(src, trg, 0)  # turn off teacher forcing
    #end = time.time()
    #print("diff time is {}".format(end-start))

    
            if batch_idx == 0:
                outputs = output.cpu()
                labels = label.cpu()
                targets = trg.cpu()
                ratios = ratio
            else:
                outputs = np.vstack((outputs, output.cpu()))
                targets = np.vstack((targets, trg.cpu()))
                labels = np.hstack((labels, label.cpu()))
                ratios = np.hstack((ratios, ratio))

            # for evaluation
            output_dim1 = output.shape[-1]
            output_dim2 = output.shape[-2]
            trg_dim1 = output.shape[-1]
            trg_dim2 = output.shape[-2]

            output = output.view(-1, output_dim1 * output_dim2)
            trg = trg.view(-1, trg_dim1 * trg_dim2)

            loss = criterion(output, trg)

            epoch_loss += loss.item()
    print("outputs.shape : ", outputs.shape)
    dataset = MyDataset(outputs, labels, ratios)
    #dataset_phase = MyDataset(targets, labels, ratios)
    num_rows = labels.shape[0]
    print("count of dataset is : ", num_rows)

    test_split_num = int (num_rows * 0.2)   # choose 80% as train data
    train_split_num = num_rows - test_split_num
    train_set, test_set = random_split(dataset, [train_split_num, test_split_num])
    #train_phase_set, test_phase_set = random_split(dataset_phase, [train_split_num, test_split_num])
    train_datasets.append(train_set)
    test_datasets.append(test_set)
    #train_phase_datasets.append(train_phase_set)
    #test_phase_datasets.append(test_phase_set)
    print(len(train_datasets[0]))
    print(len(test_datasets[0]))
    Train = ConcatDataset(train_datasets)
    #Train_phase = ConcatDataset(train_phase_datasets)
    Test = ConcatDataset(test_datasets)
    #Test_phase = ConcatDataset(test_phase_datasets)
    torch.save(Train, train_path) #'/home/ubuntu/wcmc_attack_activity_detect_code/dataset/train_miss_msg64_more.t')
    torch.save(Test, test_path) #'/home/ubuntu/wcmc_attack_activity_detect_code/dataset/test_miss_msg64_more.t')
    #torch.save(Train_phase, train_phase_path)
    #torch.save(Test_phase, test_phase_path)
    return epoch_loss / len(iterator)
    
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

if __name__ == "__main__":
    N_EPOCHS = 20
    CLIP = 1
    test_losses = []
    best_train_loss = float('inf')
    miss_train_path = '/home/ubuntu/wcmc_attack_activity_detect_code/dataset/train_miss_msg64_more.t'
    miss_test_path = '/home/ubuntu/wcmc_attack_activity_detect_code/dataset/test_miss_msg64_more.t'
    false_train_path = '/home/ubuntu/wcmc_attack_activity_detect_code/dataset/train_false_msg64_more.t'
    false_test_path = '/home/ubuntu/wcmc_attack_activity_detect_code/dataset/test_false_msg64_more.t'
    # for phase evaluate
    miss_train_phase_path = '/home/ubuntu/wcmc_attack_activity_detect_code/dataset/train_miss_phase.t'
    miss_test_phase_path = '/home/ubuntu/wcmc_attack_activity_detect_code/dataset/test_miss_phase.t'
    false_train_phase_path = '/home/ubuntu/wcmc_attack_activity_detect_code/dataset/train_false_phase.t'
    false_test_phase_path = '/home/ubuntu/wcmc_attack_activity_detect_code/dataset/test_false_phase.t'



    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss = train(model, train_loader, optimizer, criterion, CLIP)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(model.state_dict(), 'tut1-model.pt')

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        #print(f'\tTrain Loss: {train_loss:.3f}')
        #print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        start = time.time()
        test_loss = evaluate(model, miss_test_loader, criterion, miss_train_path, miss_test_path, miss_train_phase_path, miss_test_phase_path)
        test_losses.append(round(test_loss,4))
        test_loss = evaluate(model, false_test_loader, criterion, false_train_path, false_test_path, false_train_phase_path, false_test_phase_path)
        test_losses.append(round(test_loss,4))
        end = time.time()
        print("diff time is {}".format(end-start))
        print(f'| Test Loss: {test_loss:.3f} | ')
    #model.load_state_dict(torch.load('tut1-model.pt'))

    with open('result', 'w+') as f:
        f.write(str(test_losses))
   
