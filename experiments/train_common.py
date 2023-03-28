import os
import warnings

from typing import Optional, Callable

import torch
import torch.nn
import torch.utils.data
import torchinfo

from experiments.common import get_device, get_cifar10_dataset
from misc import get_file_name_net, get_file_name_opt, get_file_name_stat

from nns_aaf import KerasNetAaf, AfDefinition
from misc import RunningStat, ProgressRecorder, create_net


def train_variant(
        net_type: str, af_name: str, end_epoch: int = 100, *,
        start_epoch: int = 0, patched: bool = False,
        af_name_cnn: Optional[str] = None,
        param_freezer: Optional[Callable[[KerasNetAaf], None]] = None,
        save_as_fine_tuned: bool = False
):
    batch_size = 64
    rand_seed = 42

    dev = get_device()
    torch.manual_seed(rand_seed)
    torch.use_deterministic_algorithms(mode=True)

    train_set, test_set = get_cifar10_dataset(augment=True)
    input_size = (batch_size, 3, 32, 32)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=1000, num_workers=4
    )

    net = create_net(net_type, af_name, af_name_cnn=af_name_cnn)

    error_fn = torch.nn.CrossEntropyLoss()
    net.to(device=dev)
    torchinfo.summary(net, input_size=input_size, device=dev)

    opt = torch.optim.RMSprop(
        params=net.parameters(),
        lr=1e-4,
        alpha=0.9,  # default Keras
        momentum=0.0,  # default Keras
        eps=1e-7,  # default Keras
        centered=False  # default Keras
    )

    if start_epoch > 0:
        path_net = get_file_name_net(
            net_type, af_name, start_epoch, patched, af_name_cnn=af_name_cnn
        )
        path_opt = get_file_name_opt(
            net_type, af_name, start_epoch, patched, af_name_cnn=af_name_cnn
        )
        net.load_state_dict(torch.load(path_net))

        if os.path.isfile(path_opt):
            opt.load_state_dict(torch.load(path_opt))
        else:
            warnings.warn(
                "The old optimizer state is not available{}. Initialized the "
                "optimizer from scratch.".format(
                    " after patching" if patched else ""
                )
            )

    print(
        "Training the {} KerasNet network with {} in CNN and {} in FFN "
        "for {} epochs total.".format(
            net_type, af_name if af_name_cnn is None else af_name_cnn, af_name, end_epoch
        )
    )

    # Freeze the parameters if such hook is defined.
    if param_freezer is not None:
        param_freezer(net)

    progress = ProgressRecorder()

    for epoch in range(start_epoch, end_epoch):
        net.train()
        loss_stat = RunningStat()
        progress.start_ep()

        for mb in train_loader:
            x, y = mb[0].to(dev), mb[1].to(dev)

            y_hat = net.forward(x)
            loss = error_fn(y_hat, target=y)
            loss_stat.push(loss.item())

            # Update parameters
            opt.zero_grad()
            loss.backward()
            opt.step()

        progress.end_ep()
        net.eval()

        with torch.no_grad():
            test_total = 0
            test_correct = 0

            for batch in test_loader:
                x = batch[0].to(dev)
                y = batch[1].to(dev)
                y_hat = net(x)
                _, pred = torch.max(y_hat.data, 1)
                test_total += y.size(0)
                test_correct += torch.eq(pred, y).sum().item()

            test_acc = test_correct / test_total

            print("Train set loss stat: m={}, var={}".format(
                loss_stat.mean, loss_stat.variance
            ))
            print("Epoch: {}. Test set accuracy: {:.2%}".format(
                epoch, test_acc
            ))
            progress.push_ep(
                epoch, loss_stat.mean, loss_stat.variance, test_acc,
                lr=opt.param_groups[-1]["lr"]
            )

    progress.save_as_csv(
        get_file_name_stat(net_type, af_name, end_epoch, patched,
                           af_name_cnn=af_name_cnn,
                           fine_tuned=save_as_fine_tuned)
    )

    torch.save(
        net.state_dict(),
        get_file_name_net(net_type, af_name, end_epoch, patched,
                          af_name_cnn=af_name_cnn,
                          fine_tuned=save_as_fine_tuned)
    )
    torch.save(
        opt.state_dict(),
        get_file_name_opt(net_type, af_name, end_epoch, patched,
                          af_name_cnn=af_name_cnn,
                          fine_tuned=save_as_fine_tuned)
    )

