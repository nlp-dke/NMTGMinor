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

import sys
from onmt.train_utils.trainer import generate_data_iterator


class XEAdversarialTrainer(XETrainer):
    def __init__(self, model, loss_function, train_data, valid_data, dicts, opt, setup_optimizer=True):
        # Don't set up optimizer in superclass yet
        super().__init__(model, loss_function, train_data, valid_data, dicts, opt, setup_optimizer=False)

        if setup_optimizer:     # Set up optimizer for the classifier!
            # Normal optimzer for MT
            self.optim = onmt.Optim(opt)

            # main params exclude the classifier
            main_params = []
            for n, W in self.model.named_parameters():
                if "generator.1" not in n:
                    main_params.append(W)

            self.optim.set_parameters(main_params)
            for p in self.optim.params:
                print('*** Optim1 shape:', p.shape)

            # Optimizer for classifier
            self.optim2 = onmt.Optim(opt)
            self.optim2.lr = 0.01
            self.optim2.update_method = 'regular'
            self.optim2.set_parameters(self.model.generator[1].parameters())
            for p in self.optim2.params:
                print('*** Optim2 shape:', p.shape)

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
                # Documentation for num_losses comes from https://github.com/NVIDIA/apex/tree/master/examples/dcgan
                [self.model, self.model.generator[1]], [self.optim.optimizer, self.optim2.optimizer] = amp.initialize([self.model, self.model.generator[1]],
                                                                                                [self.optim.optimizer, self.optim2.optimizer],
                                                                                                opt_level=opt_level,
                                                                                                keep_batchnorm_fp32=keep_batchnorm_fp32,
                                                                                                loss_scale="dynamic",
                                                                                                verbosity=1,
                                                                                                num_losses=3)

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
        report_fwd_il_loss, report_bwd_il_loss, report_nmt_loss, report_dis_il_loss = 0, 0, 0, 0
        report_vq_loss_codebook, report_vq_loss_commitment, report_diversity_loss = 0, 0, 0
        report_norm_il_emb, report_norm_il_emb_var, report_norm_enc_out, report_minibatch_cnt = 0, 0, 0, 0

        start = time.time()
        n_samples = len(train_data)

        counter = 0
        update_counter = 0
        num_accumulated_words = 0
        num_accumulated_sents = 0
        denom = 3584
        nan = False
        optimize_classifier = True

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
                if self.opt.il_warmup:
                    if self.optim._step < self.optim.warmup_steps:
                        self.loss_function.il_weight = (self.optim._step + 1) / self.optim.warmup_steps * self.opt.il_loss
                    else:
                        self.loss_function.il_weight = self.opt.il_loss

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

                if opt.il_sim_loss:
                    fwd_il_loss_data = loss_dict["il_fwd_loss_data"]
                    bwd_il_loss_data = loss_dict["il_bwd_loss_data"]
                    dis_il_loss_data = loss_dict["il_dis_loss_data"]
                    nmt_loss_data = loss_dict["nmt_loss_data"] if "nmt_loss_data" in loss_dict else loss_data

                    vq_loss_codebook_data = loss_dict["vq_loss_codebook_data"]
                    vq_loss_commitment_data = loss_dict["vq_loss_commitment_data"]

                # Record norm of word LUT
                with torch.no_grad():
                    if hasattr(self.model, 'il_word_lut'):
                        norm_il_emb_data = self.model.il_word_lut.weight.norm(p=2, dim=1).mean().data #loss_dict["norm_enc_out"]
                        norm_il_emb_var_data = self.model.il_word_lut.weight.norm(p=2, dim=1).var().data
                    else:
                        norm_il_emb_data = self.model.encoder.word_lut.weight.norm(p=2, dim=1).mean().data
                        norm_il_emb_var_data = self.model.encoder.word_lut.weight.norm(p=2, dim=1).var().data

                    # Pre quantization enc out
                    enc_out = outputs["context"]
                    norm_enc_out_data = enc_out.view(-1, enc_out.shape[-1]).norm(p=2, dim=1).mean().data

                optimizer = self.optim.optimizer
                optimizer_classifier = self.optim2.optimizer
                # use 2nd loss
                alternate_training = (epoch >= opt.adversarial_classifier_start_from)

                # Normal MT loss
                if not alternate_training or not optimize_classifier:
                    # freeze classifier
                    self.model.generator[1].requires_grad_(False)
                    # unfreeze enc & dec
                    self.model.encoder.requires_grad_(True)
                    self.model.decoder.requires_grad_(True)

                    # calculate gradient for enc & dec
                    if self.cuda:
                        with amp.scale_loss(loss, optimizer, loss_id=0) as scaled_loss:
                            scaled_loss.backward(retain_graph=alternate_training)
                    else:
                        loss.backward(retain_graph=alternate_training)

                    # gradient from classifier
                    if alternate_training:
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
                            with amp.scale_loss(classifier_loss, optimizer, loss_id=1) as scaled_loss:
                                scaled_loss.backward() #retain_graph=True)
                        else:
                            classifier_loss.backward() #retain_graph=True)
                    else:
                        classifier_loss_data, classifier_loss_data_rev = 0, 0

                elif alternate_training:
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
                        with amp.scale_loss(classifier_loss, optimizer_classifier, loss_id=2) as scaled_loss:
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
                self.optim2.zero_grad()
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

                if update_flag:     # Update params
                    grad_denom = 1 / denom
                    if self.opt.normalize_gradient:
                        grad_denom = num_accumulated_words / denom
                    normalize_gradients(amp.master_params(optimizer), grad_denom)
                    normalize_gradients(amp.master_params(optimizer_classifier), grad_denom)
                    # Update the parameters.
                    if self.opt.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.opt.max_grad_norm)
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer_classifier), self.opt.max_grad_norm)
                    if not alternate_training or not optimize_classifier:
                        self.optim.step(grad_denom=grad_denom)
                    else:
                        self.optim2.step(grad_denom=grad_denom)

                    self.model.zero_grad()
                    self.optim.zero_grad()
                    self.optim2.zero_grad()

                    counter = 0
                    num_accumulated_words = 0
                    num_accumulated_sents = 0
                    num_updates = self.optim._step

                    if opt.save_every > 0 and num_updates % opt.save_every == -1 % opt.save_every:
                        valid_loss, _ = self.eval(self.valid_data,
                                                  report_classifier=True,
                                                  report_cm=True)
                        valid_ppl = math.exp(min(valid_loss, 100))
                        print('Validation perplexity: %g' % valid_ppl)

                        ep = float(epoch) - 1. + ((float(i) + 1.) / n_samples)

                        self.save(ep, valid_ppl, itr=data_iterator)

                    update_counter += 1

                    # TODO: how to make this less hand crafty?!
                    if alternate_training and \
                            ((not optimize_classifier and update_counter >= 10) or (optimize_classifier and update_counter >= 50)):
                        update_counter = 0
                        # valid_loss, valid_adv_loss = self.eval(self.valid_data, report_classifier=False, report_cm=False)
                        # valid_ppl = math.exp(min(valid_loss, 100))
                        # print('============optimize_classifier <--', optimize_classifier)
                        # print('Validation perplexity: %g, adv loss: %6.6f\n' % (valid_ppl, valid_adv_loss))
                        # print('============optimize_classifier -->', optimize_classifier)
                        # print("LR1", self.optim.lr, "LR2", self.optim2.lr)
                        optimize_classifier = not optimize_classifier

                num_words = tgt_size
                report_loss += loss_dict["nmt_loss_data"] if "nmt_loss_data" in loss_dict else loss_data
                report_classifier_loss += classifier_loss_data if alternate_training else 0
                report_classifier_loss_rev += classifier_loss_data_rev if alternate_training else 0
                report_tgt_words += num_words
                report_src_words += src_size
                total_loss += loss_dict["nmt_loss_data"] if "nmt_loss_data" in loss_dict else loss_data
                total_words += num_words
                total_tokens += batch.get('target_output').nelement()
                total_non_pads += batch.get('target_output').ne(onmt.constants.PAD).sum().item()
                optim = self.optim
                batch_efficiency = total_non_pads / total_tokens

                if opt.il_sim_loss:
                    report_fwd_il_loss += fwd_il_loss_data
                    report_bwd_il_loss += bwd_il_loss_data
                    report_dis_il_loss += dis_il_loss_data
                    report_nmt_loss += nmt_loss_data
                    report_vq_loss_codebook += vq_loss_codebook_data
                    report_vq_loss_commitment += vq_loss_commitment_data

                report_norm_il_emb += norm_il_emb_data
                report_norm_il_emb_var += norm_il_emb_var_data
                report_norm_enc_out += norm_enc_out_data
                report_minibatch_cnt += 1

                if i == 0 or (i % opt.log_interval == -1 % opt.log_interval):
                    valid_loss, valid_adv_loss = self.eval(self.valid_data, report_classifier=False, report_cm=False)
                    valid_ppl = math.exp(min(valid_loss, 100))
                    print('Validation perplexity: %g, adv loss: %6.6f' % (valid_ppl, valid_adv_loss))

                    log_string = ("Epoch %2d, %5d/%5d; ; ppl: %6.2f ; " %
                                  (epoch, i + 1, len(data_iterator),
                                   math.exp(report_loss / report_tgt_words)))

                    if opt.il_sim_loss:
                        fwd_il_ppl = math.exp(report_fwd_il_loss / (1.0 * report_src_words))
                        bwd_il_ppl = math.exp(report_bwd_il_loss / (1.0 * report_src_words))
                        dis_il_ppl = math.exp(report_dis_il_loss / (1.0 * report_src_words))
                        nmt_ppl = math.exp(report_nmt_loss / report_tgt_words)
                        log_string += (" il_fwd_ppl: %6.2f ; " % fwd_il_ppl)
                        log_string += (" il_bwd_ppl: %6.2f ; " % bwd_il_ppl)
                        log_string += (" il_dis_pll: %6.2f %6.2f ; " % (dis_il_ppl, report_dis_il_loss))
                        log_string += (" il_fwd_loss: %6.2f %6.2f %6.2f; " % (
                            report_fwd_il_loss,
                            report_src_words,
                            (report_fwd_il_loss / (1.0 * report_src_words.item()))
                        ))
                        log_string += (" ce_loss (raw): %6.2f ; " % report_nmt_loss)
                        log_string += (" vq_mse_codebook (raw): %6.2f ; " % report_vq_loss_codebook)
                        log_string += (" vq_mse_codebook: %6.2f ; " % (report_vq_loss_codebook / report_src_words))
                        log_string += (" vq_mse_commitment: %6.2f ; " % (report_vq_loss_commitment / report_src_words))
                        log_string += (" diversity_loss: %6.2f ; " % (report_diversity_loss / report_minibatch_cnt))

                    log_string += (" avg_norm_il_emb: %6.2f ; " % (report_norm_il_emb / report_minibatch_cnt))
                    log_string += (" var_norm_il_emb: %6.2f ; " % (report_norm_il_emb_var / report_minibatch_cnt))
                    log_string += (" avg_norm_enc_out: %6.2f ; " % (report_norm_enc_out / report_minibatch_cnt))

                    log_string += ("lr: %.7f ; updates: %7d; " %
                                   (self.optim.getLearningRate(),
                                    self.optim._step))

                    log_string += ("%5.0f src tok/s; %5.0f tgt tok/s; " %
                                   (report_src_words / (time.time() - start),
                                    report_tgt_words / (time.time() - start)))

                    log_string += ("%s elapsed" %
                                   str(datetime.timedelta(seconds=int(time.time() - self.start_time))))

                    print(log_string)

                    report_loss, report_tgt_words = 0, 0
                    report_classifier_loss, report_classifier_loss_rev = 0.0, 0.0
                    report_src_words = 0
                    report_aux_loss = 0.0
                    report_fwd_il_loss, report_bwd_il_loss, report_nmt_loss, report_dis_il_loss = 0, 0, 0, 0
                    report_vq_loss_codebook, report_vq_loss_commitment, report_diversity_loss = 0, 0, 0
                    report_norm_il_emb, report_norm_il_emb_var, report_norm_enc_out, report_minibatch_cnt = 0, 0, 0, 0
                    start = time.time()

                i = i + 1

        return total_loss / total_words

    def save(self, epoch, valid_ppl, itr=None):
        opt = self.opt
        model = self.model
        dicts = self.dicts

        model_state_dict = self.model.state_dict()
        optim_state_dict = self.optim.state_dict()

        if itr:
            itr_state_dict = itr.state_dict()
        else:
            itr_state_dict = None

        #  drop a checkpoint
        checkpoint = {
            'model': model_state_dict,
            'dicts': dicts,
            'opt': opt,
            'epoch': epoch,
            'itr': itr_state_dict,
            'optim': optim_state_dict,
            'optim2': self.optim2.state_dict(),
            'amp': amp.state_dict()
        }

        file_name = '%s_ppl_%.6f_e%.2f.pt' % (opt.save_model, valid_ppl, epoch)
        print('Writing to %s' % file_name)
        torch.save(checkpoint, file_name)

        # check the save directory here
        checkpoint_dir = os.path.dirname(opt.save_model)
        existed_save_files = checkpoint_paths(checkpoint_dir)
        for save_file in existed_save_files[opt.keep_save_files:]:
            print(" * Deleting old save file %s ...." % save_file)
            os.remove(save_file)

        best_epoch = float(re.search("_e(.*)\.pt", existed_save_files[0]).group(1))

        if epoch - best_epoch >= opt.early_stop_if_no_change:
            print(" * Early stopping at epoch %s as best epoch was %s ." % (epoch, best_epoch))
            sys.exit(0)


    def run(self, checkpoint=None):
        # TODO: This is a big bunch of copied code from the superclass

        opt = self.opt
        model = self.model
        # optim = self.optim

        if checkpoint is not None:
            self.model.load_state_dict(checkpoint['model'])
            prec_opt = checkpoint['opt'] if 'opt' in checkpoint else None

            if not opt.reset_optim:
                print("* Loading optimizer states ... ")
                self.optim.load_state_dict(checkpoint['optim'])
                if prec_opt is not None and hasattr(prec_opt, "fp16_mixed"):
                    # Only load amp information if the mode is the same
                    # Maybe its better to change between optimization mode?
                    if opt.fp16_mixed == prec_opt.fp16_mixed and opt.fp16 == prec_opt.fp16:
                        if 'amp' in checkpoint:
                            amp.load_state_dict(checkpoint['amp'])

                # Only load the progress when we use the same optimizer
                if 'itr' in checkpoint:
                    itr_progress = checkpoint['itr']
                else:
                    itr_progress = None

                resume = True
                start_epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 1
                if start_epoch is None:
                    start_epoch = 1

            else:
                itr_progress = None
                resume = False
                start_epoch = 1

            del checkpoint['model']
            del checkpoint['optim']
            del checkpoint
        else:
            itr_progress = None
            print('Initializing model parameters')
            init_model_parameters(model, opt)
            resume = False
            start_epoch = 1

        if opt.load_encoder_from:
            self.load_encoder_weight(opt.load_encoder_from)

        # if opt.load_decoder_from:
        #     self.load_decoder_weight(opt.load_decoder_from)

        report_classifier = opt.token_classifier is not None
        report_confusion_matrix = opt.token_classifier == 0
        # if we are on a GPU: warm up the memory allocator
        if self.cuda:
            self.warm_up()

            valid_loss, valid_adv_loss = self.eval(self.valid_data,
                                                   bidirectional_translation=self.opt.bidirectional_translation,
                                                   report_classifier=report_classifier,
                                                   report_cm=report_confusion_matrix)
            valid_ppl = math.exp(min(valid_loss, 100))
            print('Validation perplexity: %g, classifier loss: %6.6f' % (valid_ppl, valid_adv_loss))

        self.start_time = time.time()

        for epoch in range(opt.start_epoch, opt.start_epoch + opt.epochs):
            print('')

            #  (1) train for one epoch on the training set
            train_loss = self.train_epoch(epoch, resume=resume, itr_progress=itr_progress)
            train_ppl = math.exp(min(train_loss, 100))
            print('Train perplexity: %g' % train_ppl)

            #  (2) evaluate on the validation set
            valid_loss, valid_adv_loss = self.eval(self.valid_data,
                                                   bidirectional_translation=self.opt.bidirectional_translation,
                                                   report_classifier=report_classifier,
                                                   report_cm=report_confusion_matrix)
            valid_ppl = math.exp(min(valid_loss, 100))
            print('Validation perplexity: %g, adv loss: %6.6f' % (valid_ppl, valid_adv_loss))

            self.save(epoch, valid_ppl)
            itr_progress = None
            resume = False