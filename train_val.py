from Utils import *
from model import *
from dataset import *
from torch_geometric.loader import DataLoader
import time
import datetime
from log.train_logger import TrainLogger

def train_eval(
    model, 
    optimizer,
    scheduler, 
    train_loader, 
    valid_loader,
    test_loader, 
    epochs=2, 
    dataset='davis',
    gpu=0,
    fold=None,
    test_cutoff=None,
):
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print('-----Training-----')
    starttime = datetime.datetime.now()
    last_epoch_time = starttime

    params = dict(
        data_root="data",
        save_dir="save",
        dataset=dataset,
        save_model=True,
        fold=fold,
        cutoff=test_cutoff,
    )

    logger = TrainLogger(params)
    logger.info(__file__)

    best_mse = 1000
    best_test_mse = 1000
    best_epoch = -1
    best_test_epoch = -1

    # best_test_ci = 0
    best_test_rm2 = 0
    best_test_pr = 0
    best_test_sp = 0

    # best_test_ci_epoch = -1
    best_test_rm2_epoch = -1
    best_test_pr_epoch = -1
    best_test_sp_epoch = -1
    for epoch in range(epochs):
        endtime = datetime.datetime.now()
        print('total run time: ', endtime - starttime)
        print('last epoch run time: ', endtime - last_epoch_time)
        last_epoch_time = endtime
        print('Epoch', epoch)
        train(model, device, train_loader, optimizer, epoch + 1)
        print('predicting for valid data')
        G, P = predicting(model, device, valid_loader)
        val1 = get_mse(G, P)
        if test_loader is not None:
            print('predicting for test data')
            G, P = predicting(model, device, test_loader)

            # st_time = datetime.datetime.now()
            val2 = get_mse(G, P)
            # ed_time = datetime.datetime.now()
            # print('Calculate test mse run time: ', ed_time - st_time)
            if val2 < best_test_mse:
                best_test_mse = val2
                best_test_epoch = epoch + 1
                msg = f"test mse has improved at epoch {best_test_epoch}, test mse: {best_test_mse}"
                logger.info(msg)
            
            # st_time = datetime.datetime.now()
            # ci = get_ci(G, P)
            # ed_time = datetime.datetime.now()
            # print('Calculate test ci run time: ', ed_time - st_time)
            # if ci < best_test_ci:
            #     best_test_ci = ci
            #     best_test_ci_epoch = epoch + 1
            #     msg = f"test ci has improved at epoch {best_test_ci_epoch}, test ci: {best_test_ci}"
            #     logger.info(msg)

            # st_time = datetime.datetime.now()
            rm2 = get_rm2(G, P)
            # ed_time = datetime.datetime.now()
            # print('Calculate test rm2 run time: ', ed_time - st_time)
            if rm2 < best_test_rm2:
                best_test_rm2 = rm2
                best_test_rm2_epoch = epoch + 1
                msg = f"test rm2 has improved at epoch {best_test_rm2_epoch}, test rm2: {best_test_rm2}"
                logger.info(msg)

            # st_time = datetime.datetime.now()
            pr = get_pearson(G, P)
            # ed_time = datetime.datetime.now()
            # print('Calculate test pr run time: ', ed_time - st_time)
            if pr < best_test_pr:
                best_test_pr = pr
                best_test_pr_epoch = epoch + 1
                msg = f"test pr has improved at epoch {best_test_pr_epoch}, test pr: {best_test_pr}"
                logger.info(msg)

            # st_time = datetime.datetime.now()
            sp = get_spearman(G, P)
            # ed_time = datetime.datetime.now()
            # print('Calculate test sp run time: ', ed_time - st_time)
            if sp < best_test_sp:
                best_test_sp = sp
                best_test_sp_epoch = epoch + 1
                msg = f"test sp has improved at epoch {best_test_sp_epoch}, test sp: {best_test_sp}"
                logger.info(msg)

        if val1 < best_mse:
            best_mse = val1
            best_epoch = epoch + 1
            if test_loader is not None:
                print('mse improved at epoch ', best_epoch, '; best_mse', best_mse, "test mse:", val2)
                msg = "epoch-%d, loss-%.4f, test_loss-%.4f, test_rm2-%.4f, test_pr-%.4f, test_sp-%.4f" % (epoch, val1, val2, rm2, pr, sp)
            else:
                print('mse improved at epoch ', best_epoch, '; best_mse', best_mse)
                msg = "epoch-%d, loss-%.4f" % (epoch, val1)
            model_path = os.path.join(logger.get_model_dir(), msg + '.pt')
            torch.save(model.state_dict(), model_path)
            print("model has been saved to %s." % (model_path))
        else:
            if test_loader is not None:
                print('######BestMetrics######',
                    'current mse: ', val1, ' No improvement since epoch ', best_epoch, '; best_mse', best_mse,
                      "Best test loss at:", best_test_epoch, '; best_test_mse', best_test_mse,
                      "Best test rm2 at:", best_test_rm2_epoch, '; best_test_rm2', best_test_rm2,
                      "Best test pr at:", best_test_pr_epoch, '; best_test_pr', best_test_pr,
                      "Best test sp at:", best_test_sp_epoch, '; best_test_sp', best_test_sp,
                      )
            else:
                print('######BestMetrics######',
                    'current mse: ', val1, ' No improvement since epoch ', best_epoch, '; best_mse', best_mse)
        scheduler.step()
        print('######LearningRate######',
            optimizer.state_dict()['param_groups'][0]['lr'])