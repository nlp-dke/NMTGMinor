from __future__ import division


import datetime
import gc
import inspect
import math
import os
import re
import time
import torch
from apex import amp

from onmt.train_utils.trainer import XETrainer
import onmt
import onmt.markdown
import onmt.modules
from onmt.data.data_iterator import DataIterator
from onmt.data.multidata_iterator import MultiDataIterator
from onmt.data.dataset import rewrap
from onmt.model_factory import build_model, build_language_model, optimize_model
from onmt.model_factory import init_model_parameters
from onmt.train_utils.stats import Logger
from onmt.utils import checkpoint_paths, normalize_gradients


def generate_data_iterator(dataset, seed, num_workers=1, epoch=1., buffer_size=0):

    # check if dataset is a list:
    if isinstance(dataset, list):
        # this is a multidataset
        data_iterator = MultiDataIterator(dataset, seed=seed, num_workers=num_workers,
                                          epoch=epoch, buffer_size=buffer_size)
    else:

        data_iterator = DataIterator(dataset, dataset.collater, dataset.batches, seed=seed,
                                     num_workers=num_workers, epoch=epoch, buffer_size=buffer_size)

    return data_iterator


class XEAdversarialTrainer(XETrainer):
    def __init__(self, model, loss_function, train_data, valid_data, dicts, opt, setup_optimizer=True):
        super().__init__(model, loss_function, train_data, valid_data, dicts, opt, setup_optimizer)

        if setup_optimizer:

            self.optim2 = onmt.Optim(opt)
            self.optim2.lr = 0.001
            self.optim2.update_method = 'regular'
            self.optim2.set_parameters(self.model.parameters())

            if not self.opt.fp16:
                opt_level = "O0"
                keep_batchnorm_fp32 = False
            elif self.opt.fp16_mixed:
                opt_level = "O1"
                keep_batchnorm_fp32 = None
            else:
                opt_level = "O2"
                keep_batchnorm_fp32 = False

            if self.cuda:
                self.model, self.optim2.optimizer = amp.initialize(self.model,
                                                                  self.optim2.optimizer,
                                                                  opt_level=opt_level,
                                                                  keep_batchnorm_fp32=keep_batchnorm_fp32,
                                                                  loss_scale="dynamic",
                                                                  verbosity=1)

    def train_epoch(self, epoch, resume=False, itr_progress=None):
        global rec_ppl
        opt = self.opt
        train_data = self.train_data
        streaming = opt.streaming

        # Clear the gradients of the model
        self.model.zero_grad()
        self.model.reset_states()

        dataset = train_data

        data_iterator = generate_data_iterator(dataset, seed=self.opt.seed, num_workers=opt.num_workers,
                                               epoch=epoch, buffer_size=opt.buffer_size)

        if resume:
            data_iterator.load_state_dict(itr_progress)

        epoch_iterator = data_iterator.next_epoch_itr(not streaming, pin_memory=opt.pin_memory)

        total_tokens, total_loss, total_words = 0, 0, 0
        total_non_pads = 0
        report_loss, report_tgt_words = 0, 0
        report_classifier_loss, report_classifier_loss_rev = 0.0, 0.0
        report_src_words = 0
        start = time.time()
        n_samples = len(train_data)

        counter = 0
        update_counter = 0
        num_accumulated_words = 0
        num_accumulated_sents = 0
        denom = 3584
        nan = False
        optimize_classifier = False

        if opt.streaming:
            streaming_state = self.model.init_stream()
        else:
            streaming_state = None

        i = data_iterator.iterations_in_epoch if not isinstance(train_data, list) else epoch_iterator.n_yielded

        while not data_iterator.end_of_epoch():
            curriculum = (epoch < opt.curriculum)

            # this batch generator is not very clean atm
            batch = next(epoch_iterator)
            if isinstance(batch, list) and self.n_gpus == 1:
                batch = batch[0]
            batch = rewrap(batch)

            if self.cuda:
                batch.cuda(fp16=self.opt.fp16 and not self.opt.fp16_mixed)

            oom = False
            try:
                # Fetch ground truths for MT
                targets = batch.get('target_output')

                tgt_mask = targets.data.ne(onmt.constants.PAD)

                outputs = self.model(batch, streaming=opt.streaming, target_mask=tgt_mask,
                                     zero_encoder=opt.zero_encoder,
                                     mirror=opt.mirror_loss, streaming_state=streaming_state)

                batch_size = batch.size

                outputs['tgt_mask'] = tgt_mask

                loss_dict = self.loss_function(outputs, targets, model=self.model)
                loss_data = loss_dict['data']
                loss = loss_dict['loss'].div(denom)  # a little trick to avoid gradient overflow with fp16

                optimizer = self.optim.optimizer
                optimizer_classifier = self.optim2.optimizer
                # use 2nd loss
                alternate = epoch >= opt.adversarial_classifier_start_from

                if not optimize_classifier: #epoch % 2 == 0:
                    # freeze classifier
                    self.model.generator[1].requires_grad_(False)
                    # unfreeze enc & dec
                    self.model.encoder.requires_grad_(True)
                    self.model.decoder.requires_grad_(True)

                    # calculate gradient for enc & dec
                    if self.cuda:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward(retain_graph=alternate)
                    else:
                        loss.backward(retain_graph=alternate)

                    # gradient from classifier
                    if alternate:
                        if self.opt.token_classifier == 0:  # language ID
                            targets_classifier = batch.get('targets_source_lang')
                        elif self.opt.token_classifier == 1:  # predict source token ID
                            targets_classifier = batch.get('source')
                        elif self.opt.token_classifier == 2:  # predict positional ID
                            targets_classifier = batch.get('source_pos')
                            targets_classifier[targets_classifier != 0] += 1  # start from 0
                            targets_classifier[0, :] += 1
                        elif self.opt.token_classifier == 3:  # predict POS tag
                            raise NotImplementedError

                        classifier_loss_dict = self.loss_function(outputs, targets=targets_classifier,
                                                                  model=self.model,
                                                                  lan_classifier=True,
                                                                  reverse_landscape=True)

                        classifier_loss = classifier_loss_dict['loss'].div(
                            denom)  # a little trick to avoid gradient overflow with fp16
                        classifier_loss_data = 0
                        classifier_loss_data_rev = classifier_loss_dict['data'] if classifier_loss_dict[
                                                                                   'data'] is not None else 0

                        # calculate gradient for classifier
                        if self.cuda:
                            with amp.scale_loss(classifier_loss, optimizer) as scaled_loss:
                                scaled_loss.backward(retain_graph=True)
                        else:
                            classifier_loss.backward(retain_graph=True)
                    else:
                        classifier_loss_data, classifier_loss_data_rev = 0, 0

                elif alternate:
                    # freeze enc & dec, only train classifier
                    self.model.generator[1].requires_grad_(True)
                    self.model.encoder.requires_grad_(False)
                    self.model.decoder.requires_grad_(False)

                    if self.opt.token_classifier == 0:  # language ID
                        targets_classifier = batch.get('targets_source_lang')
                    elif self.opt.token_classifier == 1:  # predict source token ID
                        targets_classifier = batch.get('source')
                    elif self.opt.token_classifier == 2:  # predict positional ID
                        targets_classifier = batch.get('source_pos')
                        targets_classifier[targets_classifier != 0] += 1  # start from 0
                        targets_classifier[0, :] += 1
                    elif self.opt.token_classifier == 3:  # predict POS tag
                        raise NotImplementedError

                    classifier_loss_dict = self.loss_function(outputs, targets=targets_classifier,
                                                              model=self.model,
                                                              lan_classifier=True,
                                                              reverse_landscape=False)
                    classifier_loss = classifier_loss_dict['loss'].div(
                        denom)  # a little trick to avoid gradient overflow with fp16
                    classifier_loss_data = classifier_loss_dict['data'] if classifier_loss_dict[
                                                                           'data'] is not None else 0
                    classifier_loss_data_rev = 0
                    # calc gradient for lan classifier
                    if self.cuda:
                        with amp.scale_loss(classifier_loss, optimizer_classifier) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        classifier_loss.backward()

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory on GPU , skipping batch')
                    oom = True
                    torch.cuda.empty_cache()
                    loss = 0
                    if opt.streaming:  # reset stream in this case ...
                        streaming_state = self.model.init_stream()
                else:
                    raise e

            if loss != loss:
                # catching NAN problem
                oom = True
                self.model.zero_grad()
                self.optim.zero_grad()
                num_accumulated_words = 0
                num_accumulated_sents = 0

            if not oom:
                src_size = batch.src_size
                tgt_size = batch.tgt_size

                counter = counter + 1
                num_accumulated_words += tgt_size
                num_accumulated_sents += batch_size

                #   We only update the parameters after getting gradients from n mini-batches
                update_flag = False
                if counter >= opt.update_frequency > 0:     # if update_frequency==1, always update here
                    update_flag = True
                elif 0 < opt.batch_size_update <= num_accumulated_words:    # accmulate x words
                    update_flag = True
                elif i == n_samples:  # update for the last minibatch
                    update_flag = True
                # it was this:
                # if 0 < opt.batch_size_update <= num_accumulated_words:
                #     update_flag = True
                # elif counter >= opt.update_frequency and 0 >= opt.batch_size_update:
                #     update_flag = True
                # elif i == n_samples - 1:  # update for the last minibatch
                #     update_flag = True

                if update_flag:
                    grad_denom = 1 / denom
                    if self.opt.normalize_gradient:
                        grad_denom = num_accumulated_words / denom
                    normalize_gradients(amp.master_params(optimizer), grad_denom)
                    normalize_gradients(amp.master_params(optimizer_classifier), grad_denom)
                    # Update the parameters.
                    if self.opt.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.opt.max_grad_norm)
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer_classifier), self.opt.max_grad_norm)
                    if optimize_classifier:
                        self.optim2.step(grad_denom=grad_denom)
                        self.optim2.zero_grad()
                    else:
                        self.optim.step(grad_denom=grad_denom)
                        self.optim.zero_grad()
                    self.model.zero_grad()
                    counter = 0
                    num_accumulated_words = 0
                    num_accumulated_sents = 0
                    num_updates = self.optim._step

                    if opt.save_every > 0 and num_updates % opt.save_every == -1 % opt.save_every:
                        valid_loss, _ = self.eval(self.valid_data, report_classifier=True, report_cm=True)
                        valid_ppl = math.exp(min(valid_loss, 100))
                        print('Validation perplexity: %g' % valid_ppl)

                        ep = float(epoch) - 1. + ((float(i) + 1.) / n_samples)

                        self.save(ep, valid_ppl, itr=data_iterator)

                    update_counter += 1

                # TODO: how to make this less hand crafty?!
                    if alternate and ((not optimize_classifier and update_counter >= 1) or \
                        (optimize_classifier and update_counter >= 5)):
                        print('============optimize_classifier <--', optimize_classifier)
                        optimize_classifier = not optimize_classifier
                        update_counter = 0
                        valid_loss, valid_adv_loss = self.eval(self.valid_data, report_classifier=True, report_cm=False)
                        valid_ppl = math.exp(min(valid_loss, 100))
                        print('Validation perplexity: %g, adv loss: %6.6f' % (valid_ppl, valid_adv_loss))
                        print('============optimize_classifier -->', optimize_classifier)

                num_words = tgt_size
                report_loss += loss_data
                report_classifier_loss += classifier_loss_data if alternate else 0
                report_classifier_loss_rev += classifier_loss_data_rev if alternate else 0
                report_tgt_words += num_words
                report_src_words += src_size
                total_loss += loss_data
                total_words += num_words
                total_tokens += batch.get('target_output').nelement()
                total_non_pads += batch.get('target_output').ne(onmt.constants.PAD).sum().item()
                optim = self.optim
                batch_efficiency = total_non_pads / total_tokens

                if i == 0 or (i % opt.log_interval == -1 % opt.log_interval):
                    if epoch % 2 == 0:
                        valid_loss, valid_adv_loss = self.eval(self.valid_data, report_classifier=True, report_cm=True)
                        valid_ppl = math.exp(min(valid_loss, 100))
                        print('Validation perplexity: %g, adv loss: %6.6f' % (valid_ppl, valid_adv_loss))

                    print(("Epoch %2d, %5d/%5d; ; ppl: %6.2f ; classifier loss: %6.2f ; classifier rev loss: %6.2f ; lr: %.7f ; num updates: %7d " +
                           "%5.0f src tok/s; %5.0f tgt tok/s; %s elapsed") %
                          (epoch, i + 1, len(data_iterator),
                           math.exp(report_loss / report_tgt_words),
                           report_classifier_loss / float(report_src_words),
                           report_classifier_loss_rev / float(report_src_words),
                           optim.getLearningRate(),
                           optim._step,
                           report_src_words / (time.time() - start),
                           report_tgt_words / (time.time() - start),
                           str(datetime.timedelta(seconds=int(time.time() - self.start_time)))))

                    report_loss, report_tgt_words = 0, 0
                    report_classifier_loss, report_classifier_loss_rev = 0.0, 0.0
                    report_src_words = 0
                    start = time.time()

                i = i + 1

        return total_loss / total_words

#
#     def train_epoch(self, epoch, resume=False, itr_progress=None):
#
#         opt = self.opt
#         train_data = self.train_data
#         streaming = opt.streaming
#
#         self.model.train()
#         self.loss_function.train()
#         # Clear the gradients of the model
#         # self.runner.zero_grad()
#         self.model.zero_grad()
#         self.model.reset_states()
#
#         dataset = train_data
#
#         # data iterator: object that controls the
#         # data_iterator = DataIterator(dataset, dataset.collater, dataset.batches, seed=self.opt.seed,
#         #                              num_workers=opt.num_workers, epoch=epoch, buffer_size=opt.buffer_size)
#         data_iterator = generate_data_iterator(dataset, seed=self.opt.seed, num_workers=opt.num_workers,
#                                                epoch=epoch, buffer_size=opt.buffer_size)
#
#         if resume:
#             data_iterator.load_state_dict(itr_progress)
#
#         epoch_iterator = data_iterator.next_epoch_itr(not streaming, pin_memory=opt.pin_memory)
#
#         total_tokens, total_loss, total_words = 0, 0, 0
#         total_non_pads = 0
#         report_loss, report_tgt_words = 0, 0
#         report_classifier_loss, report_classifier_loss_rev = 0.0, 0.0
#         optimize_classifier = True
#         report_src_words = 0
#         start = time.time()
#         n_samples = len(train_data)
#
#         counter = 0
#         update_counter = 0
#         num_accumulated_words = 0
#         num_accumulated_sents = 0
#         denom = 3584
#         nan = False
#
#         if opt.streaming:
#             streaming_state = self.model.init_stream()
#         else:
#             streaming_state = None
#
#         for i in range(iteration, n_samples):
#
#             curriculum = (epoch < opt.curriculum)
#
#             batches = [train_data.next(curriculum=curriculum)[0]]
#
#             if (len(self.additional_data) > 0 and
#                     i % self.additional_data_ratio[0] == 0):
#                 for j in range(len(self.additional_data)):
#                     for k in range(self.additional_data_ratio[j + 1]):
#                         if self.additional_data_iteration[j] == len(self.additional_data[j]):
#                             self.additional_data_iteration[j] = 0
#                             self.additional_data[j].shuffle()
#                             self.additional_batch_order[j] = self.additional_data[j].create_order()
#
#                         batches.append(self.additional_data[j].next()[0])
#                         self.additional_data_iteration[j] += 1
#
#             for b in range(len(batches)):
#                 batch = batches[b]
#                 if self.cuda:
#                     batch.cuda(fp16=self.opt.fp16 and not self.opt.fp16_mixed)
#
#                 # if opt.streaming:
#                 #     if train_data.is_new_stream():
#                 #         streaming_state = self.model.init_stream()
#                 # else:
#                 #     streaming_state = None
#
#                 oom = False
#                 try:
#                     # outputs is a dictionary containing keys/values necessary for loss function
#                     # can be flexibly controlled within models for easier extensibility
#                     targets = batch.get('target_output')
#                     tgt_mask = targets.data.ne(onmt.constants.PAD)
#
#                     outputs = self.model(batch, streaming=opt.streaming, target_mask=tgt_mask,
#                                          zero_encoder=opt.zero_encoder,
#                                          mirror=opt.mirror_loss, streaming_state=streaming_state)
#
#                     batch_size = batch.size
#                     #
#                     outputs['tgt_mask'] = tgt_mask
#                     #
#                     loss_dict = self.loss_function(outputs, targets, model=self.model)
#                     loss_data = loss_dict['data']
#                     loss = loss_dict['loss'].div(denom)  # a little trick to avoid gradient overflow with fp16
#
#                     optimizer = self.optim.optimizer
#
#                     has_classifier_loss = self.opt.language_classifier and (not self.opt.freeze_language_classifier)
#
#                     if not has_classifier_loss:
#                         # calculate gradient for enc / dec
#                         if self.cuda:
#                             with amp.scale_loss(loss, optimizer) as scaled_loss:
#                                 scaled_loss.backward()
#                         else:
#                             loss.backward()
#
#                     else:
#                         if (not optimize_classifier and not (self.opt.freeze_encoder and self.opt.freeze_decoder)): #or epoch<=8:
#                             # calculate gradient for enc / dec
#                             if self.cuda:
#                                 with amp.scale_loss(loss, optimizer) as scaled_loss:
#                                     scaled_loss.backward(retain_graph=True)
#                             else:
#                                 loss.backward(retain_graph=True)
#
#                             # gradient from classifier
#                             if epoch > 8:
#                                 # freeze classifier
#                                 self.model.generator[1].requires_grad_(False)
#                                 # unfreeze enc & dec
#                                 self.model.encoder.requires_grad_(True)
#                                 self.model.decoder.requires_grad_(True)
#                                 # calc gradient for lan classifier
#                                 targets_src_lang = batch.get('targets_source_lang')
#                                 classifier_loss_dict = self.loss_function(outputs, targets=targets_src_lang,
#                                                                           model=self.model,
#                                                                           lan_classifier=True,
#                                                                           reverse_landscape=True)
#                                 classifier_loss = classifier_loss_dict['loss'].div(
#                                     denom)  # a little trick to avoid gradient overflow with fp16
#                                 classifier_loss_data_rev = classifier_loss_dict['data'] if classifier_loss_dict[
#                                                                                            'data'] is not None else 0
#
#                                 # calc gradient for lan classifier
#                                 if self.cuda:
#                                     with amp.scale_loss(classifier_loss, optimizer) as scaled_loss:
#                                         scaled_loss.backward(retain_graph=True)
#                                 else:
#                                     classifier_loss.backward(retain_graph=True)
#                             else:
#                                 classifier_loss_data, classifier_loss_data_rev = 0, 0
#
#                         else:  # odd number, freeze enc/dec, train other
#                             # unfreeze classifier
#                             self.model.generator[1].requires_grad_(True)
#                             # freeze enc & dec
#                             self.model.encoder.requires_grad_(False)
#                             self.model.decoder.requires_grad_(False)
#
#                             targets_src_lang = batch.get('targets_source_lang')
#                             classifier_loss_dict = self.loss_function(outputs, targets=targets_src_lang,
#                                                                       model=self.model,
#                                                                       lan_classifier=True,
#                                                                       reverse_landscape=False)
#                             classifier_loss = classifier_loss_dict['loss'].div(
#                                 denom)  # a little trick to avoid gradient overflow with fp16
#                             classifier_loss_data = classifier_loss_dict['data'] if classifier_loss_dict[
#                                                                                    'data'] is not None else 0
#                             classifier_loss_data_rev = 0
#                             # calc gradient for lan classifier
#                             if self.cuda:
#                                 with amp.scale_loss(classifier_loss, optimizer) as scaled_loss:
#                                     scaled_loss.backward()
#                             else:
#                                 classifier_loss.backward()
#
#                 except RuntimeError as e:
#                     if 'out of memory' in str(e):
#                         print('| WARNING: ran out of memory on GPU , skipping batch')
#                         oom = True
#                         torch.cuda.empty_cache()
#                         loss = 0
#                         if opt.streaming:  # reset stream in this case ...
#                             streaming_state = self.model.init_stream()
#                     else:
#                         raise e
#
#                 if loss != loss:
#                     # catching NAN problem
#                     oom = True
#                     self.model.zero_grad()
#                     self.optim.zero_grad()
#                     num_accumulated_words = 0
#                     num_accumulated_sents = 0
#
#                 if not oom:
#                     src_size = batch.src_size
#                     tgt_size = batch.tgt_size
#
#                     counter = counter + 1
#                     num_accumulated_words += tgt_size
#                     num_accumulated_sents += batch_size
#
#                     #   We only update the parameters after getting gradients from n mini-batches
#                     update_flag = False
#                     if 0 < opt.batch_size_update <= num_accumulated_words:
#                         update_flag = True
#                     elif counter >= opt.update_frequency and 0 >= opt.batch_size_update:
#                         update_flag = True
#                     elif i == n_samples - 1:  # update for the last minibatch
#                         update_flag = True
#
#                     if update_flag:
#                         grad_denom = 1 / denom
#                         if self.opt.normalize_gradient:
#                             grad_denom = num_accumulated_words / denom
#                         normalize_gradients(amp.master_params(optimizer), grad_denom)
#                         # Update the parameters.
#                         if self.opt.max_grad_norm > 0:
#                             torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.opt.max_grad_norm)
#                         self.optim.step(grad_denom=grad_denom)
#                         self.optim.zero_grad()
#                         self.model.zero_grad()
#                         counter = 0
#                         num_accumulated_words = 0
#                         num_accumulated_sents = 0
#                         num_updates = self.optim._step
#
#                         if opt.save_every > 0 and num_updates % opt.save_every == -1 % opt.save_every:
#                             valid_loss, _ = self.eval(self.valid_data)
#                             valid_ppl = math.exp(min(valid_loss, 100))
#                             print('Validation perplexity: %g' % valid_ppl)
#
#                             ep = float(epoch) - 1. + ((float(i) + 1.) / n_samples)
#
#                             self.save(ep, valid_ppl, itr=data_iterator)
#
#                         update_counter += 1
#
#                     # if (optimize_classifier and update_counter >= 200) or (not optimize_classifier and update_counter >=10):
#                     #     print('============optimize_classifier <--', optimize_classifier)
#                     #     optimize_classifier = not optimize_classifier
#                     #     update_counter = 0
#                     #     valid_loss, valid_adv_loss = self.eval(self.valid_data, report_classifier=True)
#                     #     valid_ppl = math.exp(min(valid_loss, 100))
#                     #     print('Validation perplexity: %g, adv loss: %6.6f' % (valid_ppl, valid_adv_loss))
#                     #     print('============optimize_classifier -->', optimize_classifier)
#
#                     num_words = tgt_size
#                     report_loss += loss_data
#                     report_classifier_loss += classifier_loss_data if self.opt.language_classifier else 0
#                     report_classifier_loss_rev += classifier_loss_data_rev if self.opt.language_classifier else 0
#                     report_tgt_words += num_words
#                     report_src_words += src_size
#                     total_loss += loss_data
#                     total_words += num_words
#                     total_tokens += batch.get('target_output').nelement()
#                     total_non_pads += batch.get('target_output').ne(onmt.constants.PAD).sum().item()
#                     optim = self.optim
#                     batch_efficiency = total_non_pads / total_tokens
#
#                     if b == 0 and (i == 0 or (i % opt.log_interval == -1 % opt.log_interval)):
#                         print(("Epoch %2d, %5d/%5d; ; ppl: %6.2f ; adv loss: %6.2f ; adv rev loss: %6.2f ; lr: %.7f ; num updates: %7d " +
#                                "%5.0f src tok/s; %5.0f tgt tok/s; %s elapsed") %
#                               (epoch, i + 1, len(train_data),
#                                math.exp(report_loss / report_tgt_words),
#                                report_classifier_loss / float(report_src_words),
#                                report_classifier_loss_rev / float(report_src_words), #math.log(max(0.000001, report_classifier_loss)) - math.log(report_src_words)),
#                                optim.getLearningRate(),
#                                optim._step,
#                                report_src_words / (time.time() - start),
#                                report_tgt_words / (time.time() - start),
#                                str(datetime.timedelta(seconds=int(time.time() - self.start_time)))))
#
#                         report_loss, report_tgt_words = 0, 0
#                         report_classifier_loss = 0.0
#                         report_src_words = 0
#                         start = time.time()
#
#         return total_loss / total_words