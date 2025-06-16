# from Causal_Ours.Powerformer.models import Autotime
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
import os
import GPUtil
from models import (
    Informer,
    Autoformer,
    iTransformer,
    DLinear,
    Linear,
    NLinear,
    PatchTST,
    Powerformer,
    Autotime,
    TimeMixer
)
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

import os
import time
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        # Autoregressive prediction parameters
        self.pred_len = getattr(args, 'pred_len', 192)
        # self.total_pred_len = getattr(args, 'total_pred_len', 720)  # Default 720 if not specified
        self.chunk_size = getattr(args, 'chunk_size', 96)  # Default 96 if not specified

    def _build_model(self):
        model_dict = {
            "Autoformer": Autoformer,
            "iTransformer": iTransformer,
            "Informer": Informer,
            "DLinear": DLinear,
            "NLinear": NLinear,
            "Linear": Linear,
            "PatchTST": PatchTST,
            "Powerformer": Powerformer,
            "Autotime": Autotime,
            "TimeMixer": TimeMixer
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _prepare_predict_inputs(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float().to(self.device) if batch_x_mark is not None else None
        batch_y_mark = batch_y_mark.float().to(self.device) if batch_y_mark is not None else None
        
        # Prepare decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        
        return batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp

    def _run_model(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if any(x in self.args.model for x in ["Linear", "TST", "ower"]):
                    outputs = self.model(batch_x)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if any(x in self.args.model for x in ["Linear", "TST", "ower"]):
                outputs = self.model(batch_x)
            else:
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        return outputs

    def _autoregressive_predict(self, batch_x, batch_x_mark=None):
        """Core autoregressive prediction function"""
        all_preds = []
        current_input = batch_x.clone()
        current_mark = batch_x_mark.clone() if batch_x_mark is not None else None
        
        steps = (self.pred_len + self.chunk_size - 1) // self.chunk_size
        
        for step in range(steps):
            # Prepare decoder input (zeros for autoregressive)
            dec_inp = torch.zeros(current_input.size(0), self.chunk_size, current_input.size(2)).float().to(self.device)
            
            outputs = self._run_model(
                current_input, 
                current_mark,
                dec_inp,
                None
            )
            
            # Store predictions
            all_preds.append(outputs.detach())
            
            # Update input window
            if step < steps - 1: 
                current_input = torch.cat([
                    current_input[:, self.chunk_size:, :], 
                    outputs[:, -self.chunk_size:, :]    
                ], dim=1)
                
                if current_mark is not None:
                    last_time_features = current_mark[:, -1:, :].expand(-1, self.chunk_size, -1)
                    current_mark = torch.cat([
                        current_mark[:, self.chunk_size:, :],
                        last_time_features
                    ], dim=1)
        
        # Combine all predictions
        final_preds = torch.cat(all_preds, dim=1)[:, :self.pred_len, :]
        return final_preds.cpu().numpy()

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp = self._prepare_predict_inputs(
                    batch_x, batch_y, batch_x_mark, batch_y_mark
                )
                
                outputs = self._run_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                # Calculate loss
                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())
        
        self.model.train()
        return np.average(total_loss)

    def print_gpu_usage(self):
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print(f"GPU {gpu.id}: {gpu.memoryUsed} MB used / {gpu.memoryTotal} MB total, {gpu.memoryFree} MB free")

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=train_steps,
            pct_start=self.args.pct_start,
            epochs=self.args.train_epochs,
            max_lr=self.args.learning_rate,
        )

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            epoch_time = time.time()
            
            self.model.train()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device) if batch_x_mark is not None else None
                batch_y_mark = batch_y_mark.float().to(self.device) if batch_y_mark is not None else None

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._run_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        f_dim = -1 if self.args.features == "MS" else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                else:
                    outputs = self._run_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    f_dim = -1 if self.args.features == "MS" else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print("\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                if self.args.lradj == "TST":
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            self.print_gpu_usage()
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != "TST":
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print("Updating learning rate to {}".format(scheduler.get_last_lr()[0]))

        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting, test=0, autoregressive=False):
        test_data, test_loader = self._get_data(flag="test")
        if test:
            self.model.load_state_dict(
                torch.load(os.path.join("./checkpoints/" + setting, "checkpoint.pth"))
            )

        preds = []
        trues = []
        inputx = []
        
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                if autoregressive:
                    # Autoregressive prediction
                    batch_x = batch_x.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device) if batch_x_mark is not None else None
                    outputs = self._autoregressive_predict(batch_x, batch_x_mark)
                else:
                    # Standard prediction
                    batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp = self._prepare_predict_inputs(
                        batch_x, batch_y, batch_x_mark, batch_y_mark
                    )
                    outputs = self._run_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    outputs = outputs.detach().cpu().numpy()

                # Prepare for metrics calculation
                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].numpy()

                preds.append(outputs)
                trues.append(batch_y)
                inputx.append(batch_x.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, rse]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        np.save(folder_path + 'inputx.npy', inputx)

        return mae, mse, rmse, mape, mspe, rse, corr

    def predict(self, setting, load=False, autoregressive=False):
        pred_data, pred_loader = self._get_data(flag="pred")

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + "/" + "checkpoint.pth"
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []
        
        self.model.eval()
        with torch.no_grad():
            autoregressive = False
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                if autoregressive:
                    # Autoregressive prediction
                    batch_x = batch_x.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device) if batch_x_mark is not None else None
                    outputs = self._autoregressive_predict(batch_x, batch_x_mark)
                else:
                    # Standard prediction
                    batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp = self._prepare_predict_inputs(
                        batch_x, batch_y, batch_x_mark, batch_y_mark
                    )
                    outputs = self._run_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    outputs = outputs.detach().cpu().numpy()
                
                preds.append(outputs)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return 

    def _set_record_score(self, setting):
        if self.args.model.lower() == "transformer":
            for enc in self.model.encoder.attn_layers:
                enc.attention.inner_attention.record_scores = setting
            for dec in self.model.decoder.layers:
                dec.self_attention.inner_attention.record_scores = setting
            for dec in self.model.decoder.layers:
                dec.cross_attention.inner_attention.record_scores = setting
        else:
            if self.model.decomposition:
                for enc in self.model.model_trend.backbone.encoder.layers:
                    enc.self_attn.sdp_attn.record_scores = setting
                for enc in self.model.model_res.backbone.encoder.layers:
                    enc.self_attn.sdp_attn.record_scores = setting
            else:
                for enc in self.model.model.backbone.encoder.layers:
                    enc.self_attn.sdp_attn.record_scores = setting

    def _gather_transformer_attn(self, score_bins, weight_bins):
        raw_scores = []
        powerlaw_scores = []
        raw_weights = []
        powerlaw_weights = []
        attn_layers = [
            self.model.encoder.attn_layers,
            self.model.decoder.layers,
            self.model.decoder.layers,
        ]
        for idx, layers in enumerate(attn_layers):
            raw_scores.append([])
            powerlaw_scores.append([])
            raw_weights.append([])
            powerlaw_weights.append([])
            for layer in layers:
                if idx == 0:
                    attn = layer.attention
                elif idx == 1:
                    attn = layer.self_attention
                else:
                    attn = layer.cross_attention
                print(
                    "Raw weights",
                    attn.inner_attention.raw_weights.detach().cpu().numpy().shape,
                )
                raw_scores[-1].append(
                    np.histogram(
                        attn.inner_attention.raw_scores.detach()
                        .cpu()
                        .numpy()
                        .flatten(),
                        score_bins,
                    )[0]
                )
                powerlaw_scores[-1].append(
                    np.histogram(
                        attn.inner_attention.masked_scores.detach()
                        .cpu()
                        .numpy()
                        .flatten(),
                        score_bins,
                    )[0]
                )
                raw_weights[-1].append(
                    np.histogram(
                        attn.inner_attention.raw_weights.detach()
                        .cpu()
                        .numpy()
                        .flatten(),
                        weight_bins,
                    )[0]
                )
                powerlaw_weights[-1].append(
                    np.histogram(
                        attn.inner_attention.attn_weights.detach()
                        .cpu()
                        .numpy()
                        .flatten(),
                        weight_bins,
                    )[0]
                )
        return raw_scores, powerlaw_scores, raw_weights, powerlaw_weights

    def _gather_powerformer_attn(self, score_bins, weight_bins):
        raw_scores = [
            [
                np.histogram(
                    enc.self_attn.sdp_attn.raw_scores.detach().cpu().numpy().flatten(),
                    score_bins,
                )[0]
                for enc in self.model.model.backbone.encoder.layers
            ]
        ]
        powerlaw_scores = [
            [
                np.histogram(
                    enc.self_attn.sdp_attn.masked_scores.detach()
                    .cpu()
                    .numpy()
                    .flatten(),
                    score_bins,
                )[0]
                for enc in self.model.model.backbone.encoder.layers
            ]
        ]
        raw_weights = [
            [
                np.histogram(
                    enc.self_attn.sdp_attn.raw_weights.detach().cpu().numpy().flatten(),
                    weight_bins,
                )[0]
                for enc in self.model.model.backbone.encoder.layers
            ]
        ]
        powerlaw_weights = [
            [
                np.histogram(
                    enc.self_attn.sdp_attn.attn_weights.detach()
                    .cpu()
                    .numpy()
                    .flatten(),
                    weight_bins,
                )[0]
                for enc in self.model.model.backbone.encoder.layers
            ]
        ]
        return raw_scores, powerlaw_scores, raw_weights, powerlaw_weights

    def _record_attn_distributions(self, score_bins, weight_bins):
        if self.args.model.lower() == "transformer":
            return self._gather_transformer_attn(score_bins, weight_bins)
        else:
            return self._gather_powerformer_attn(score_bins, weight_bins)

    def _save_attn_results(
        self,
        attn_raw_scores,
        attn_powerlaw_scores,
        attn_raw_weights,
        attn_powerlaw_weights,
        score_bins,
        weight_bins,
        folder_path,
    ):
        labels = ["encoder_SA_", "decoder_SA_", "decoder_CA_"]
        np.save(os.path.join(folder_path, "score_bins.npy"), score_bins)
        np.save(os.path.join(folder_path, "weight_bins.npy"), weight_bins)
        for idx in range(len(attn_raw_scores)):
            label = labels[idx]
            print("LABEL", label)
            comb_attn_raw_scores = np.sum(np.array(attn_raw_scores[idx]), 0)
            comb_attn_powerlaw_scores = np.sum(np.array(attn_powerlaw_scores[idx]), 0)
            comb_attn_raw_weights = np.sum(np.array(attn_raw_weights[idx]), 0)
            comb_attn_powerlaw_weights = np.sum(np.array(attn_powerlaw_weights[idx]), 0)
            print(
                "SIZES",
                np.array(attn_raw_scores[idx]).shape,
                len(attn_raw_scores),
                comb_attn_raw_scores.shape,
            )
            np.save(
                os.path.join(folder_path, label + "attn_raw_scores.npy"),
                comb_attn_raw_scores,
            )
            np.save(
                os.path.join(folder_path, label + "attn_powerlaw_scores.npy"),
                comb_attn_powerlaw_scores,
            )
            np.save(
                os.path.join(folder_path, label + "attn_raw_weights.npy"),
                comb_attn_raw_weights,
            )
            np.save(
                os.path.join(folder_path, label + "attn_powerlaw_weights.npy"),
                comb_attn_powerlaw_weights,
            )

            if self.args.model.lower() == "transformer":
                if idx == 0:
                    decay_mask = (
                        self.model.encoder.attn_layers[0]
                        .attention.inner_attention.powerlaw_mask.detach()
                        .cpu()
                        .numpy()
                    )
                if idx == 1:
                    decay_mask = (
                        self.model.decoder.layers[0]
                        .self_attention.inner_attention.powerlaw_mask.detach()
                        .cpu()
                        .numpy()
                    )
                if idx == 2:
                    decay_mask = (
                        self.model.decoder.layers[0]
                        .cross_attention.inner_attention.powerlaw_mask.detach()
                        .cpu()
                        .numpy()
                    )
            else:
                decay_mask = (
                    self.model.model.backbone.encoder.layers[0]
                    .self_attn.sdp_attn.powerlaw_mask.detach()
                    .cpu()
                    .numpy()
                )
            np.save(os.path.join(folder_path, label + "powerlaw_mask.npy"), decay_mask)

    def _save_attn_matrices(self, save_attn_matrices, test_data, folder_path):
        data_idxs = np.arange(len(test_data))
        np.random.shuffle(data_idxs)
        data_idxs = data_idxs[:save_attn_matrices]
        batch_x, batch_y, batch_x_mark, batch_y_mark = [], [], [], []
        data = [test_data.__getitem__(idx) for idx in data_idxs]
        batch_x = torch.tensor([d[0] for d in data]).float().to(self.device)
        batch_y = torch.tensor([d[1] for d in data]).float().to(self.device)
        batch_x_mark = torch.tensor([d[0] for d in data]).float().to(self.device)
        batch_y_mark = torch.tensor([d[0] for d in data]).float().to(self.device)

        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
        dec_inp = (
            torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
            .float()
            .to(self.device)
        )
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if (
                    "Linear" in self.args.model
                    or "TST" in self.args.model
                    or "ower" in self.args.model
                ):
                    outputs = self.model(batch_x)
                else:
                    if self.args.output_attention:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )[0]
                    else:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )
        else:
            if (
                "Linear" in self.args.model
                or "TST" in self.args.model
                or "ower" in self.args.model
            ):
                outputs = self.model(batch_x)
                # outputs = self.model.evaluate(batch_x, self.args.pred_len)
            else:
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[
                        0
                    ]

                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        raw_weights, raw_scores = [], []
        weights, scores = [], []
        decay_mask = None
        if (
            self.args.model.lower() == "powerformer"
            or self.args.model.lower() == "patchtst"
        ):
            if self.model.decomposition:
                for ilr, enc in enumerate(
                    self.model.model_trend.backbone.encoder.layers
                ):
                    raw_scores.append((enc.self_attn.sdp_attn.raw_scores, "trend", ilr))
                    raw_weights.append(
                        (enc.self_attn.sdp_attn.raw_weights, "trend", ilr)
                    )
                    scores.append((enc.self_attn.sdp_attn.masked_scores, "trend", ilr))
                    weights.append((enc.self_attn.sdp_attn.attn_weights, "trend", ilr))
                for ilr, enc in enumerate(self.model.model_res.backbone.encoder.layers):
                    raw_scores.append(
                        (enc.self_attn.sdp_attn.raw_scores, "residual", ilr)
                    )
                    raw_weights.append(
                        (enc.self_attn.sdp_attn.raw_weights, "residual", ilr)
                    )
                    scores.append(
                        (enc.self_attn.sdp_attn.masked_scores, "residual", ilr)
                    )
                    weights.append(
                        (enc.self_attn.sdp_attn.attn_weights, "residual", ilr)
                    )
                decay_mask = self.model.model_res.backbone.encoder.layers[
                    -1
                ].self_attn.sdp_attn.powerlaw_mask
            else:
                decay_mask = self.model.model.backbone.encoder.layers[
                    -1
                ].self_attn.sdp_attn.powerlaw_mask
                for ilr, enc in enumerate(self.model.model.backbone.encoder.layers):
                    raw_scores.append((enc.self_attn.sdp_attn.raw_scores, "total", ilr))
                    raw_weights.append(
                        (enc.self_attn.sdp_attn.raw_weights, "total", ilr)
                    )
                    scores.append((enc.self_attn.sdp_attn.masked_scores, "total", ilr))
                    weights.append((enc.self_attn.sdp_attn.attn_weights, "total", ilr))
        elif self.args.model.lower() == "transformer":
            decay_mask = self.model.encoder.attn_layers[
                -1
            ].attention.inner_attention.powerlaw_mask
            for ilr, enc in enumerate(self.model.encoder.attn_layers):
                raw_scores.append(
                    (enc.attention.inner_attention.raw_scores, "encoder_SA", ilr)
                )
                raw_weights.append(
                    (enc.attention.inner_attention.raw_weights, "encoder_SA", ilr)
                )
                scores.append(
                    (enc.attention.inner_attention.masked_scores, "encoder_SA", ilr)
                )
                weights.append(
                    (enc.attention.inner_attention.attn_weights, "encoder_SA", ilr)
                )
            for ilr, dec in enumerate(self.model.decoder.layers):
                raw_scores.append(
                    (
                        dec.self_attention.inner_attention.raw_scores,
                        "decoder_SA",
                        ilr,
                    )
                )
                raw_weights.append(
                    (
                        dec.self_attention.inner_attention.raw_weights,
                        "decoder_SA",
                        ilr,
                    )
                )
                scores.append(
                    (
                        dec.self_attention.inner_attention.masked_scores,
                        "decoder_SA",
                        ilr,
                    )
                )
                weights.append(
                    (
                        dec.self_attention.inner_attention.attn_weights,
                        "decoder_SA",
                        ilr,
                    )
                )
            for ilr, dec in enumerate(self.model.decoder.layers):
                raw_scores.append(
                    (
                        dec.cross_attention.inner_attention.raw_scores,
                        "decoder_CA",
                        ilr,
                    )
                )
                raw_weights.append(
                    (
                        dec.cross_attention.inner_attention.raw_weights,
                        "decoder_CA",
                        ilr,
                    )
                )
                scores.append(
                    (
                        dec.cross_attention.inner_attention.masked_scores,
                        "decoder_CA",
                        ilr,
                    )
                )
                weights.append(
                    (
                        dec.cross_attention.inner_attention.attn_weights,
                        "decoder_CA",
                        ilr,
                    )
                )
        else:
            raise NotImplementedError(f"Cannot handle model type {self.args.model}")

        np.save(os.path.join(folder_path, f"attn_matrices_indices.npy"), data_idxs)
        np.save(
            os.path.join(folder_path, f"decay_mask.npy"),
            decay_mask.detach().cpu().numpy(),
        )
        loop = [
            (raw_scores, "raw_scores"),
            (scores, "scores"),
            (raw_weights, "raw_weights"),
            (weights, "weights"),
        ]
        for data, label in loop:
            for vals, layer_type, layer_num in data:
                np.save(
                    os.path.join(folder_path, f"{layer_type}_{label}_{layer_num}.npy"),
                    vals.detach().cpu().numpy(),
                )
        print("Saved attention matrices, now exiting")
        sys.exit()