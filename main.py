import torch
import dataset
import architecture
import utils
import tqdm
import numpy as np


@torch.enable_grad()
def update(model, dataloader, criterion, opt, batch_size=128):
    device = next(model.parameters()).device
    model.train()
    err = []
    model.batch_size = batch_size

    # init states
    h, c = model.init_state()
    h = h.to(device)
    c = c.to(device)

    # total is the total nr. of samples of the  train-set
    with tqdm.tqdm(total=1272851) as pbar:
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            y_hat, (h, c) = model(x, (h, c))
            y_hat = y_hat.permute(0, 2, 1)
            loss = criterion(y_hat, y)
            opt.zero_grad()

            h = h.detach()
            c = c.detach()

            loss.backward()
            opt.step()
            err.append(loss.item())
            pbar.update(len(x))

    return err


@torch.no_grad()
def sample(model, train_set, n_samples=100, start_index=39, t=1.):
    # set seed to start token
    seed = start_index
    nr = []
    model.to("cpu")
    model.batch_size = 1
    model.eval()
    h, c = model.init_state()
    cnt = 0

    while True:
        x = train_set.one_hot[seed]
        x = x.view(1, 1, -1)
        out, (h, c) = model(x, (h, c))
        probs = torch.nn.functional.softmax(out.squeeze()/t, dim=0).numpy()
        idx = np.random.choice(41, p=probs)
        nr.append(idx)
        if idx == 0:
            cnt += 1
            seed = start_index
            if cnt == n_samples:
                print(cnt)
                txt = ''.join([train_set.int2char[x] for x in nr])
                return txt
        else:
            seed = idx


if __name__ == "__main__":

    # flags for program control flow
    train = True
    generate = True

    # define hyper-parameters
    batch_size = 128
    lr = 0.001
    n_epochs = 7
    n_hidden = 396
    n_layers = 2
    max_seq_len = 100 + 1
    n_samples = 10000
    temp = 0.9

    # read in smiles as string
    path_train = "smiles_train.txt"
    with open(path_train, "r") as f:
        text = f.read()

    # check device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # create train-set and train-loader
    train_set = dataset.ChemDataset(text, max_seq_len)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               drop_last=True, num_workers=4)

    # define model
    model = architecture.Network(n_hidden, n_layers, batch_size).to(device)
    path_model = "model.pt"
    path_val = "val.txt"

    if train:
        # define loss, index 40 is the padding index
        criterion = torch.nn.CrossEntropyLoss(ignore_index=40)

        # define optimizer
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        train_err = []
        for e in range(n_epochs):
            train_err = update(model, train_loader, criterion, opt)
            # print avg. train err of an epoch
            print(f"Train Error: {sum(train_err) / len(train_err)}")

            # generate and validate during training w.r.t. validity, uniqueness, novelty
            txt = sample(model, train_set, n_samples=1000, start_index=39, t=1.)
            with open(path_val, "w") as f:
                f.write(txt)

            validity, uniqueness, novelty = utils.evaluate(path_val, path_train)
            print(f"validity: {validity}, uniqueness: {uniqueness}, novelty: {novelty}")

            # back from cpu to device, because generating samples is done on cpu
            model.to(device)

        torch.save(model, path_model)

    if generate:
        model = torch.load(path_model)
        txt = sample(model, train_set, n_samples=n_samples, start_index=39, t=temp)
        path_sub = "submission.txt"

        with open(path_sub, "w") as f:
            f.write(txt)

        validity, uniqueness, novelty = utils.evaluate(path_sub, path_train)
        print(f"validity: {validity}, uniqueness: {uniqueness}, novelty: {novelty}")