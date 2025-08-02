from data.data_loader import Dataset_BJ13, Dataset_Bike, Dataset_TaxiNYC, Dataset_Bike2
from exp.exp_basic import Exp_Basic
from models.model import Pgot

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import MAE, MSE, RMSE
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time

import warnings

warnings.filterwarnings('ignore')
from utils.logger import Logger

import logging

from torch.utils.tensorboard import SummaryWriter


class Exp_Pgot(Exp_Basic):
    def __init__(self, args):
        super(Exp_Pgot, self).__init__(args)

    def _build_model(self):
        model = Pgot(
            self.args.enc_in,
            self.args.c_out,
            self.args.seq_len,
            self.args.label_len,
            self.args.pred_len,
            self.args.d_model,
            self.args.n_heads,
            self.args.e_layers,
            self.args.d_ff,
            self.args.l,
            self.args.w,
            self.args.dropout,
            self.args.freq,
            self.args.activation,
            self.args.output_attention,
            self.device
        ).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'TaxiBJ': Dataset_BJ13,
            'Bike1NYC': Dataset_Bike,
            'Bike2NYC': Dataset_Bike2,
            'TaxiNYC': Dataset_TaxiNYC,
            
        }
        Data = data_dict[self.args.data]
        timeenc = 1

        if flag == 'test':
            shuffle_flag = False;
            drop_last = True;
            batch_size = args.batch_size;
            freq = args.freq
        elif flag == 'pred':
            shuffle_flag = False;
            drop_last = False;
            batch_size = 1;
            freq = args.detail_freq
        else:
            shuffle_flag = True;
            drop_last = True;
            batch_size = args.batch_size;
            freq = args.freq
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []

        for i, (batch_x, batch_y, batch_x_mark, batch_space_mark_x) in enumerate(
                vali_loader):
            pred, true = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_space_mark_x)

            # 转到 CPU 并 numpy 化
            pred = pred.detach().cpu().numpy()
            true = true.detach().cpu().numpy()

            mask = (~np.isnan(pred)) & (~np.isnan(true)) & \
                   (~np.isinf(pred)) & (~np.isinf(true)) & \
                   (true < 1e3) & (true > -1)  # 可自定义上下限

            pred_masked = pred[mask]
            true_masked = true[mask]

            if pred_masked.size == 0:
                continue

            loss = criterion(torch.from_numpy(pred_masked), torch.from_numpy(true_masked))
            total_loss.append(loss.item())

        total_loss = np.average(total_loss) if total_loss else float('inf')
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # 首先，它检查是否存在用于保存结果的文件夹。如果不存在，则创建该文件夹。
        if not os.path.exists('./results/' + setting):
            os.makedirs('./results/' + setting)
        # 创建了一个日志记录器（logger）和一个 TensorBoard 的 SummaryWriter（writer）对象
        # 日志记录器用于记录训练过程中的信息，例如损失值、训练状态等 ，保存在日志文件中
        # SummaryWriter 用于将训练过程中的指标、损失等信息写入 TensorBoard 事件文件，以便后续可视化和分析
        logger = Logger(log_file_name='./results/' + setting + '/log.txt', log_level=logging.DEBUG,
                        logger_name="Transformer").get_log()
        writer = SummaryWriter('./results/' + setting + '/event')

        logger.info(f"Data : {self.args.data}")
        logger.info(f"batch_size : {self.args.batch_size}")
        logger.info(f"learning_rate : {self.args.learning_rate}")
        logger.info(f"d_model : {self.args.d_model}")
        logger.info(f"d_ff : {self.args.d_ff}")
        logger.info(f"e_layers : {self.args.e_layers}")
        logger.info(f"seq_len : {self.args.seq_len}")
        logger.info(f"pred_len : {self.args.pred_len}")
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_space_mark_x) in enumerate(
                    train_loader):
                iter_count += 1

                model_optim.zero_grad()
                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_space_mark_x)
                loss = criterion(pred, true)
                train_loss.append(loss.item())
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            logger.info("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            writer.add_scalar('Loss/train', train_loss, epoch + 1)
            writer.add_scalar('Loss/vali', vali_loss, epoch + 1)
            writer.add_scalar('Loss/test', test_loss, epoch + 1)
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        writer.close()
        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')

        logger = Logger(
            log_file_name='./results/' + setting + '/log.txt',
            log_level=logging.DEBUG,
            logger_name="P-GOT"
        ).get_log()

        self.model.eval()
        preds, trues, t1 = [], [], []

        for i, (batch_x, batch_y, batch_x_mark, batch_space_mark_x) in enumerate(
                test_loader):
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_space_mark_x)
            preds.append(pred.detach())
            trues.append(true.detach())
        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0)


        print('test shape:', preds.shape, trues.shape)

        preds_reshaped = preds.reshape(-1, preds.shape[-1]).cpu().numpy()
        trues_reshaped = trues.reshape(-1, trues.shape[-1]).cpu().numpy()
    
        preds_inverse = test_data.scaler.inverse_transform(preds_reshaped)  # 如果不支持 torch，可临时 .cpu().numpy()
        trues_inverse = test_data.scaler.inverse_transform(trues_reshaped)

        preds = preds_inverse.reshape(preds.shape)
        trues = trues_inverse.reshape(trues.shape)
        print('test shape (after inverse):', preds.shape, trues.shape)

        folder_path = './results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)

        mae = MAE(preds, trues)
        mse = MSE(preds, trues)
        rmse = RMSE(preds, trues)
        print(f'Overall => mse:{mse}, mae:{mae}, rmse:{rmse}')
        logger.info(f'Overall => mse:{mse}, mae:{mae}, rmse:{rmse}')

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()

        preds = []

        for i, (batch_x, batch_y, batch_x_mark, batch_space_mark_x) in enumerate(
                pred_loader):
            pred, true = self._process_one_batch(
                pred_data, batch_x, batch_y, batch_x_mark, batch_space_mark_x)
            preds.append(pred.detach().cpu().numpy())
        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_space_mark_x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)

        batch_x_mark = batch_x_mark.float().to(device)
        batch_space_mark_x = batch_space_mark_x.float().to(device)
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, batch_space_mark_x)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, batch_space_mark_x)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, batch_space_mark_x)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, batch_space_mark_x)
        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)

        batch_y = batch_y[:, -self.args.pred_len:, :].to(device)
        outputs = outputs.squeeze(-1)
        outputs = outputs.to(device)
        return outputs, batch_y
