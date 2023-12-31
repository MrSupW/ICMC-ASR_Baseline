# Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from contextlib import nullcontext

# if your python version < 3.7 use the below one
# from contextlib import suppress as nullcontext
import torch
from torch.nn.utils import clip_grad_norm_


class Executor:

    def __init__(self):
        self.step = 0

    def train(self, model, optimizer, scheduler, data_loader, device, writer,
              args, scaler):
        ''' Train one epoch
        '''
        model.train()
        clip = args.get('grad_clip', 50.0)
        log_interval = args.get('log_interval', 10)
        rank = args.get('rank', 0)
        epoch = args.get('epoch', 0)
        accum_grad = args.get('accum_grad', 1)
        is_distributed = args.get('is_distributed', True)
        is_deepspeed = args.get('is_deepspeed', False)
        use_amp = args.get('use_amp', False)
        ds_dtype = args.get('ds_dtype', "fp32")
        if ds_dtype == "fp16":
            ds_dtype = torch.float16
        elif ds_dtype == "bf16":
            ds_dtype = torch.bfloat16
        else:
            ds_dtype = None
        logging.info('using accumulate grad, new batch size is {} times'
                     ' larger than before'.format(accum_grad))
        if use_amp:
            assert scaler is not None
        # A context manager to be used in conjunction with an instance of
        # torch.nn.parallel.DistributedDataParallel to be able to train
        # with uneven inputs across participating processes.
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_context = model.join
        else:
            model_context = nullcontext
        num_seen_utts = 0
        with model_context():
            for batch_idx, batch in enumerate(data_loader):
                key, feats, target, feats_lengths, target_lengths = batch
                feats = feats.to(device)
                target = target.to(device)
                feats_lengths = feats_lengths.to(device)
                target_lengths = target_lengths.to(device)
                num_utts = target_lengths.size(0)
                if num_utts == 0:
                    continue
                context = None
                # Disable gradient synchronizations across DDP processes.
                # Within this context, gradients will be accumulated on module
                # variables, which will later be synchronized.
                if is_distributed and batch_idx % accum_grad != 0:
                    context = model.no_sync
                # Used for single gpu training and DDP gradient synchronization
                # processes.
                else:
                    context = nullcontext
                with context():
                    if is_deepspeed:  # deepspeed
                        with torch.cuda.amp.autocast(
                            enabled=ds_dtype is not None,
                            dtype=ds_dtype, cache_enabled=False
                        ):
                            loss_dict = model(feats, feats_lengths, target,
                                              target_lengths)
                        loss = loss_dict['loss']
                        loss_att = loss_dict['loss_att']
                        loss_ctc = loss_dict['loss_ctc']
                        # NOTE(xcsong): Zeroing the gradients is handled automatically by DeepSpeed after the weights # noqa
                        #   have been updated using a mini-batch. DeepSpeed also performs gradient averaging automatically # noqa
                        #   at the gradient accumulation boundaries and addresses clip_grad_norm internally. In other words # noqa
                        #   `model.backward(loss)` is equivalent to `loss.backward() + clip_grad_norm_() + optimizer.zero_grad() + accum_grad` # noqa
                        #   ref: https://www.deepspeed.ai/tutorials/megatron/#using-the-training-api  # noqa
                        model.backward(loss)
                    else:             # pytorch native ddp
                        # autocast context
                        # The more details about amp can be found in
                        # https://pytorch.org/docs/stable/notes/amp_examples.html
                        with torch.cuda.amp.autocast(scaler is not None):
                            loss_dict = model(feats, feats_lengths, target,
                                              target_lengths)
                            loss = loss_dict['loss'] / accum_grad
                            loss_att = loss_dict['loss_att'] / accum_grad
                            loss_ctc = loss_dict['loss_ctc'] / accum_grad
                            if "loss_rnnt" in loss_dict:
                                loss_rnnt = loss_dict['loss_rnnt'] / accum_grad
                        if use_amp:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()

                num_seen_utts += num_utts
                if is_deepspeed:
                    if rank == 0 and writer is not None \
                            and model.is_gradient_accumulation_boundary():
                        writer.add_scalar('train_loss', loss.item(), self.step)
                        writer.add_scalar('train_ctc_loss', loss_ctc.item(), self.step)
                        writer.add_scalar('train_att_loss', loss_att.item(), self.step)
                    # NOTE(xcsong): The step() function in DeepSpeed engine updates the model parameters as well as the learning rate. There is # noqa
                    #   no need to manually perform scheduler.step(). In other words: `ds_model.step() = optimizer.step() + scheduler.step()` # noqa
                    #   ref: https://www.deepspeed.ai/tutorials/megatron/#using-the-training-api  # noqa
                    model.step()
                    self.step += 1
                elif not is_deepspeed and batch_idx % accum_grad == 0:
                    if rank == 0 and writer is not None:
                        writer.add_scalar('train_loss', loss, self.step)
                        writer.add_scalar('train_ctc_loss', loss_ctc, self.step)
                        writer.add_scalar('train_att_loss', loss_att, self.step)
                        if "loss_rnnt" in loss_dict:
                            writer.add_scalar('train_rnnt_loss', loss_rnnt, self.step)
                    # Use mixed precision training
                    if use_amp:
                        scaler.unscale_(optimizer)
                        grad_norm = clip_grad_norm_(model.parameters(), clip)
                        # Must invoke scaler.update() if unscale_() is used in
                        # the iteration to avoid the following error:
                        #   RuntimeError: unscale_() has already been called
                        #   on this optimizer since the last update().
                        # We don't check grad here since that if the gradient
                        # has inf/nan values, scaler.step will skip
                        # optimizer.step().
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        grad_norm = clip_grad_norm_(model.parameters(), clip)
                        if torch.isfinite(grad_norm):
                            optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    self.step += 1
                if batch_idx % log_interval == 0:
                    lr = optimizer.param_groups[0]['lr']
                    log_str = 'TRAIN Batch {}/{} loss {:.6f} '.format(
                        epoch, batch_idx,
                        loss.item() * accum_grad)
                    for name, value in loss_dict.items():
                        if name != 'loss' and value is not None:
                            log_str += '{} {:.6f} '.format(name, value.item())
                    log_str += 'lr {:.8f} rank {}'.format(lr, rank)
                    logging.debug(log_str)

    def cv(self, model, data_loader, device, args):
        ''' Cross validation on
        '''
        model.eval()
        rank = args.get('rank', 0)
        epoch = args.get('epoch', 0)
        log_interval = args.get('log_interval', 10)
        is_deepspeed = args.get('is_deepspeed', False)
        ds_dtype = args.get('ds_dtype', "fp32")
        if ds_dtype == "fp16":
            ds_dtype = torch.float16
        elif ds_dtype == "bf16":
            ds_dtype = torch.bfloat16
        else:  # fp32
            ds_dtype = None
        # in order to avoid division by 0
        num_seen_utts = 1
        total_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                key, feats, target, feats_lengths, target_lengths = batch
                feats = feats.to(device)
                target = target.to(device)
                feats_lengths = feats_lengths.to(device)
                target_lengths = target_lengths.to(device)
                num_utts = target_lengths.size(0)
                if num_utts == 0:
                    continue
                if is_deepspeed:
                    with torch.cuda.amp.autocast(
                        enabled=ds_dtype is not None,
                        dtype=ds_dtype, cache_enabled=False
                    ):
                        loss_dict = model(feats, feats_lengths,
                                          target, target_lengths)
                else:
                    loss_dict = model(feats, feats_lengths, target, target_lengths)
                loss = loss_dict['loss']
                if torch.isfinite(loss):
                    num_seen_utts += num_utts
                    total_loss += loss.item() * num_utts
                if batch_idx % log_interval == 0:
                    log_str = 'CV Batch {}/{} loss {:.6f} '.format(
                        epoch, batch_idx, loss.item())
                    for name, value in loss_dict.items():
                        if name != 'loss' and value is not None:
                            log_str += '{} {:.6f} '.format(name, value.item())
                    log_str += 'history loss {:.6f}'.format(total_loss /
                                                            num_seen_utts)
                    log_str += ' rank {}'.format(rank)
                    logging.debug(log_str)
        return total_loss, num_seen_utts
