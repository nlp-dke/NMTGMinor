import onmt
import onmt.modules
import torch.nn as nn
import torch, math
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from onmt.utils import flip
from collections import defaultdict

import numpy

def tiny_value_of_dtype(dtype: torch.dtype):
    """
    Returns a moderately tiny value for a given PyTorch data type that is used to avoid numerical
    issues such as division by zero.
    This is different from `info_value_of_dtype(dtype).tiny` because it causes some NaN bugs.
    Only supports floating point dtypes.
    """
    if not dtype.is_floating_point:
        raise TypeError("Only supports floating point dtypes.")
    if dtype == torch.float or dtype == torch.double:
        return 1e-13
    elif dtype == torch.half:
        return 1e-4
    else:
        raise TypeError("Does not support dtype " + str(dtype))


class CrossEntropyLossBase(_Loss):

    """
    Class for managing efficient loss computation.
    loss computations
    Users can implement their own loss computation strategy by making
    subclass of this one.
    Args:
        output_size: number of words in vocabulary()
    """

    def __init__(self, output_size, label_smoothing):
        super(CrossEntropyLossBase, self).__init__()
        self.output_size = output_size
        self.padding_idx = onmt.constants.PAD
        self.smoothing_value = label_smoothing / (output_size - 2)
        self.confidence = 1.0 - label_smoothing
        self.label_smoothing = label_smoothing

    def _compute_loss(self, scores, targets):

        gtruth = targets.view(-1)  # batch * time
        scores = scores.view(-1, scores.size(-1))  # batch * time X vocab_size

        lprobs = scores
        non_pad_mask = gtruth.ne(self.padding_idx)
        nll_loss = -lprobs.gather(1, gtruth.unsqueeze(1))[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()

        eps_i = self.smoothing_value
        loss = (1. - self.label_smoothing) * nll_loss + eps_i * smooth_loss
        loss_data = nll_loss.data.item()

        return loss, loss_data

    def _compute_adv_loss(self, scores, targets, reverse_landscape=False, multiclass=False):
        # no label smoothing
        # Note: language ID starts from 1
        if not reverse_landscape:
            try:
                gtruth = targets.view(-1)  # 1D, (batch X time).
            except RuntimeError:
                gtruth = targets.contiguous().view(-1)
            scores = scores.view(-1, scores.size(-1))  # 2D, batch * (time X vocab_size)
            lprobs = scores
            non_pad_mask = gtruth.ne(self.padding_idx)
            nll_loss = -lprobs.gather(1, torch.clamp(gtruth.unsqueeze(1)-1, min=0))[non_pad_mask]
            nll_loss = nll_loss.sum()
            loss = nll_loss
        else:
            gtruth = targets.view(-1)  # 1D, (batch X time).
            scores = scores.view(-1, scores.size(-1))  # 2D, batch * (time X vocab_size / # labels)
            lprobs = scores
            non_pad_mask = gtruth.ne(self.padding_idx)
            # # gtruth_complement = torch.ones((gtruth.shape[0], gtruth.max()+1), dtype=torch.long).to(device='cuda')
            # # gtruth_complement[torch.arange(gtruth.shape[0]), gtruth] = 0
            # # 2D, time * vocab_size
            total_tok = gtruth.shape[0]
            num_classes = scores.nelement() / total_tok
            base = torch.arange(num_classes, dtype=torch.long).to(device='cuda').unsqueeze(0).repeat(total_tok, 1)
            chosen_bol = torch.arange(num_classes).to(device='cuda').unsqueeze(1) != torch.clamp(gtruth-1, min=0)  # other than label index
            remaining = base[chosen_bol.T].view(total_tok, -1)
            #
            nll_loss = -lprobs.gather(1, remaining)[non_pad_mask]
            nll_loss = nll_loss.sum()
            loss = -nll_loss  # reverse

        loss_data = loss.data.item()

        return loss, loss_data

    def forward(self, model_outputs, targets, hiddens, **kwargs):

        return NotImplementedError


class NMTLossFunc(CrossEntropyLossBase):
    """
    Standard NMT Loss Computation.
    """
    def __init__(self, hidden_size, output_size, label_smoothing, mirror=False, aux_loss_code=None, aux_loss_weight=0.0):
        super(NMTLossFunc, self).__init__(output_size, label_smoothing)
        self.output_size = output_size
        self.padding_idx = onmt.constants.PAD
        self.smoothing_value = label_smoothing / (output_size - 2)
        self.confidence = 1.0 - label_smoothing
        self.label_smoothing = label_smoothing
        self.mirror = mirror

        # self.aux_loss_code = aux_loss_code
        # if self.aux_loss_code is not None:
        #     self.sim_loss_base = self.aux_loss_code // 10
        #     self.sim_loss_type = self.aux_loss_code % 10
        #     self.aux_loss_weight = aux_loss_weight

    def forward(self, model_outputs, targets, model=None, backward=False, normalizer=1, lan_classifier=False,
                reverse_landscape=False, **kwargs):
        """
        Compute the loss. Subclass must define this method.
        Args:

            model_outputs: a dictionary containing the predictive output from the model.
                                                      time x batch x vocab_size
                                                   or time x batch x hidden_size
            targets: the validate target to compare output with. time x batch
            model: passing the model in for necessary components
            backward: to control if we should perform backward pass (to free the graph) or not
            normalizer: the denominator of the loss before backward
            **kwargs(optional): additional info for computing loss.
        """

        outputs = model_outputs['hidden']
        logprobs = model_outputs['logprobs'] if not lan_classifier else model_outputs['logprobs_lan']

        mirror = self.mirror

        if mirror:
            reverse_outputs = model_outputs['reverse_hidden']
            reverse_logprobs = model_outputs['reverse_logprobs']
            reverse_targets = torch.flip(targets, (0, ))

            alpha = 1.0

        loss, loss_data = self._compute_loss(logprobs, targets) if not lan_classifier \
            else self._compute_adv_loss(logprobs, targets, reverse_landscape=reverse_landscape)  # no label smoothing

        total_loss = loss

        if mirror:
            reverse_loss, _ = self._compute_loss(reverse_logprobs, reverse_targets)

            mask = model_outputs['tgt_mask']
            flattened_mask = mask.view(-1)
            non_pad_indices = torch.nonzero(flattened_mask).squeeze(1)
            # remove the pads
            outputs = outputs.contiguous().view(-1, outputs.size(-1))
            outputs = outputs.index_select(0, non_pad_indices)

            # reverse_mask = torch.flip(mask.long(), (0, )).bool
            # Here we have to flip the reverse outputs again so the states have the same order with the original seq
            reverse_outputs = torch.flip(reverse_outputs, (0, ))
            reverse_mask = mask
            flattened_reverse_mask = reverse_mask.view(-1)
            reverse_non_pad_indices = torch.nonzero(flattened_reverse_mask).squeeze(1)
            reverse_outputs = reverse_outputs.contiguous().view(-1, reverse_outputs.size(-1))
            reverse_outputs = reverse_outputs.index_select(0, reverse_non_pad_indices)

            # mirror_loss = (outputs - reverse_outputs).float() ** 2
            mirror_outputs = reverse_outputs.detach()
            mirror_loss = F.mse_loss(outputs, mirror_outputs, reduction='sum')
            total_loss = total_loss + reverse_loss + alpha * mirror_loss

        if backward:
            total_loss.div(normalizer).backward()

        # if self.aux_loss_code is not None and 'context_main' in model_outputs:  # use aux loss & is training time
        #     # context is T x B x H, e.g. 10, 128, 512
        #     b_size = model_outputs['context_main'].shape[1]
        #     if self.sim_loss_base == 1:  # position by position difference, shorter one
        #         diff = model_outputs['context_main'] - model_outputs['context_aux']
        #         src_mask = model_outputs['src_mask'].permute(2, 0, 1)  # B, H, T
        #         diff = diff.float().masked_fill_(src_mask, 0).type_as(diff)  #inplace
        #     elif self.sim_loss_base == 2:  # max pool over time
        #         # print(torch.max(model_outputs['context_main'], 0)[0].shape, model_outputs['context_aux'].shape)
        #         src_mask_main = model_outputs['src_mask_main'].permute(2, 0, 1)
        #         src_mask_aux = model_outputs['src_mask_aux'].permute(2, 0, 1)
        #         main_ = model_outputs['context_main']
        #         aux_ = model_outputs['context_aux']
        #         masked_main = main_.float().masked_fill(src_mask_main, -float('inf')).type_as(main_)
        #         masked_aux = aux_.float().masked_fill(src_mask_aux, -float('inf')).type_as(aux_)
        #         repr_main = torch.max(masked_main, 0)[0]
        #         repr_aux = torch.max(masked_aux, 0)[0]
        #         diff = repr_main - repr_aux
        #     elif self.sim_loss_base == 3:  # mean pool over time
        #         src_mask_main = model_outputs['src_mask_main'].permute(2, 0, 1)
        #         src_mask_aux = model_outputs['src_mask_aux'].permute(2, 0, 1)
        #         main_ = model_outputs['context_main']
        #         aux_ = model_outputs['context_aux']
        #         masked_main = main_.float().masked_fill(src_mask_main, 0).type_as(main_)
        #         masked_aux = aux_.float().masked_fill(src_mask_aux, 0).type_as(aux_)
        #         repr_main = torch.sum(masked_main, dim=0, keepdim=True) / (1 - src_mask_main.float()).sum(dim=0)
        #         repr_aux = torch.sum(masked_aux, dim=0, keepdim=True) / (1 - src_mask_aux.float()).sum(dim=0)
        #         diff = repr_main - repr_aux
        #     else:
        #         raise NotImplementedError

        #     if self.sim_loss_type == 1:  # MSE
        #         sim_loss = (diff ** 2).sum()
        #     else:
        #         raise NotImplementedError
        #
        #     sim_loss_data = (sim_loss / b_size).data.item()  # before applying weights
        #     # apply weights
        #     if self.aux_loss_weight >= 0:
        #         sim_loss = sim_loss * self.aux_loss_weight
        #     else:
        #         raise NotImplementedError
        #
        #     # divide by # of sequences in batch
        #     sim_loss = sim_loss / b_size
        #
        output_dict = {"loss": loss, "data": loss_data}
        #
        # if self.aux_loss_code is not None:
        #     output_dict["sim_loss"] = sim_loss
        #     output_dict["sim_loss_data"] = sim_loss_data

        # return loss, loss_data, None
        return output_dict

        #

        # targets = targets.view(-1)
        # if mask is not None and logprobs.dim() < 3:
        #     """ We remove all positions with PAD """
        #     flattened_mask = mask.view(-1)
        #
        #     non_pad_indices = torch.nonzero(flattened_mask).squeeze(1)
        #
        #     clean_targets = targets.index_select(0, non_pad_indices)
        #
        # else:
        #     clean_targets = targets


class CTCLossFunc(_Loss):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, output_size, label_smoothing=0.0):
        super(CTCLossFunc, self).__init__(output_size)
        self.ctc = nn.CTCLoss(output_size-1, reduction='sum')

    def forward(self, model_outputs, targets, model=None, backward=False, normalizer=1, **kwargs):
        """
        Args:

            model_outputs: a dictionary containing the predictive output from the model.
                                                      time x batch x vocab_size
                                                   or time x batch x hidden_size
            targets: the validate target to compare output with. time x batch
            model: passing the model in for necessary components
            backward: to control if we should perform backward pass (to free the graph) or not
            normalizer: the denominator of the loss before backward
            **kwargs(optional): additional info for computing loss.
        """
        raise NotImplementedError

        # outputs = model_outputs['encoder']
        # original_outputs = outputs
        # batch_size = outputs.size(1)
        # h_size = outputs.size(-1)
        #
        # source_mask = model_outputs['src_mask']
        # target_mask = model_outputs['tgt_mask']
        #
        # target_length = target_mask.sum(0)
        # if source_mask.dim() == 3:
        #     input_length = (1-source_mask).squeeze(1).sum(1)
        # else:
        #     input_length = (1-source_mask).sum(1)
        #
        # # remove elements with more targets than input
        # comp = torch.lt(target_length,input_length)
        # target_length = target_length.index_select(0,comp.nonzero().squeeze())
        # input_length = input_length.index_select(0,comp.nonzero().squeeze())
        # outputs = outputs.index_select(1,comp.nonzero().squeeze())
        # targets = targets.index_select(1,comp.nonzero().squeeze())
        #
        # # flatten the output
        # size = outputs.size()
        # outputs = outputs.contiguous().view(-1, outputs.size(-1))
        #
        # clean_input = outputs
        #
        # # dists = generator(outputs)
        # if model is not None:
        #     # the 'second' generator is the encoder softmax one
        #     dists = model.generator[1](clean_input)
        # else:
        #     dists = clean_input
        #
        # # reshape back to 3D for CTC
        # dists = dists.view(size[0], size[1], -1)
        #
        # loss = self.ctc(dists,targets.transpose(0,1), input_length, target_length)
        #
        # loss_data = loss.data.item()
        #
        # # if not numpy.isfinite(loss_data):
        # #     print("Input:", input_length)
        # #     print("Target:", target_length)
        # #     print("Compare:", comp)
        # #     print("Selected:", comp.nonzero().squeeze().size())
        # #     loss = torch.zeros_like(loss)
        # #     loss_data = loss.data.item()
        #
        # if backward:
        #     loss.div(normalizer).backward()
        #
        # output_dict = {"loss": loss, "data": loss_data}
        # return output_dict
        # return loss,loss_data, None


class NMTAndCTCLossFunc(_Loss):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, output_size, label_smoothing=0.0, ctc_weight=0.0):
        super(NMTAndCTCLossFunc, self).__init__(output_size)
        self.ctc_weight = ctc_weight
        self.ce_loss = NMTLossFunc(output_size, label_smoothing)
        self.ctc_loss = CTCLossFunc(output_size + 1, label_smoothing)

    def forward(self, model_outputs, targets, model=None, backward=False, normalizer=1, **kwargs):
        """
        Args:
            model_outputs: a dictionary containing the predictive output from the model.
                                                      time x batch x vocab_size
                                                   or time x batch x hidden_size
            targets: the validate target to compare output with. time x batch
            model: passing the model in for necessary components
            backward: to control if we should perform backward pass (to free the graph) or not
            normalizer: the denominator of the loss before backward
            **kwargs(optional): additional info for computing loss.
        """
        ce_loss = self.ce_loss(model_outputs, targets, model, False, normalizer)
        ctc_loss = self.ctc_loss(model_outputs, targets, model, False, normalizer)

        loss = self.ctc_weight * ctc_loss['loss'] + (1 - self.ctc_weight) * ce_loss['loss']
        loss_data = self.ctc_weight * ctc_loss['data'] + (1 - self.ctc_weight) * ce_loss['data']

        if not numpy.isfinite(ctc_loss['data']):
            print("CTC_Loss:", ctc_loss['data'])
            print("NMT_Loss:", ce_loss['data'])
            print("Loss:", loss_data)
            exit()

        if backward:
            loss.div(normalizer).backward()

        output_dict = {"loss": loss, "data": loss_data}

        return output_dict

    def cuda(self):
        self.ce_loss = self.ce_loss.cuda()
        self.ctc_loss = self.ctc_loss.cuda()
        return self


class FusionLoss(CrossEntropyLossBase):

    def forward(self, model_outputs, targets, model=None, backward=False, normalizer=1, **kwargs):
        """
        Args:

            model_outputs: a dictionary containing the predictive output from the model.
                                                      time x batch x vocab_size
                                                   or time x batch x hidden_size
            targets: the validate target to compare output with. time x batch
            model: passing the model in for necessary components
            backward: to control if we should perform backward pass (to free the graph) or not
            normalizer: the denominator of the loss before backward
            **kwargs(optional): additional info for computing loss.
        """

        # in this implementation, the PRENORM algorithm is used

        tm_outputs = model_outputs['tm']['hidden']

        lm_outputs = model_outputs['lm']['hidden']

        mask = model_outputs['tgt_mask']

        # flatten the output
        tm_outputs = tm_outputs.contiguous().view(-1, tm_outputs.size(-1))
        lm_outputs = lm_outputs.contiguous().view(-1, lm_outputs.size(-1))
        targets = targets.view(-1)

        if mask is not None:
            """ We remove all positions with PAD """
            flattened_mask = mask.view(-1)

            non_pad_indices = torch.nonzero(flattened_mask).squeeze(1)

            clean_tm_input = tm_outputs.index_select(0, non_pad_indices)
            clean_lm_input = lm_outputs.index_select(0, non_pad_indices)

            clean_targets = targets.index_select(0, non_pad_indices)

        else:
            clean_tm_input = tm_outputs
            clean_lm_input = lm_outputs
            clean_targets = targets

        if model is not None:
            # the 'first' generator is the decoder softmax one

            # PRENORM algorithm from
            # https://arxiv.org/pdf/1809.00125.pdf
            # Simple Fusion: Return of the Language Model
            tm_logits = model.tm_model.generator[0](clean_tm_input, log_softmax=False)

            with torch.no_grad():
                log_lm = model.lm_model.generator[0](clean_lm_input, log_softmax=True)

            dists = F.log_softmax(tm_logits + log_lm, dim=-1)

            # # POSTNORM algorithm
            # tm_logits =  model.tm_model.generator[0](clean_tm_input, log_softmax=False)
            #
            # with torch.no_grad():
            #     lm_logits = model.lm_model.generator[0](clean_lm_input, log_softmax=False)
            #
            # dists = F.log_softmax(F.softmax(tm_logits, dim=-1) * F.softmax(lm_logits, dim=-1), dim=-1)

        else:
            raise NotImplementedError

        loss, loss_data = self._compute_loss(dists, clean_targets)

        if backward:
            loss.div(normalizer).backward()

        output_dict = {"loss": loss, "data": loss_data}

        return output_dict


class MSEEncoderLoss(_Loss):
    def __init__(self, input_type, weight=0.0):
        super(MSEEncoderLoss, self).__init__()
        self.input_type = input_type
        self.weight = weight

    def forward(self, context1, context2, mask1, mask2):
        if self.input_type == 1:    # meanpool
            mask1_ = mask1.permute(2, 0, 1)  # B, H, T --> T, B, H
            mask2_ = mask2.permute(2, 0, 1)

            masked_context1 = context1.masked_fill(mask1_, 0).type_as(context1)
            masked_context2 = context2.masked_fill(mask2_, 0).type_as(context2)

            # (T, B, H) / (T, B, H)
            input1 = torch.sum(masked_context1, dim=0, keepdim=True) / (1 - mask1_.float()).sum(dim=0)
            input2 = torch.sum(masked_context2, dim=0, keepdim=True) / (1 - mask2_.float()).sum(dim=0)

            l2_loss = (input1 - input2) ** 2
            # multiply by seq length to make aux. loss weight comparable
            l2_loss = l2_loss.sum() * min(context1.shape[0], context1.shape[1])

        elif self.input_type == 2:  # by position
            # (T1, B, D) --> (min(T1, T2), B, D)
            # (T2, B, D) --> (min(T1, T2), B, D)
            max_len1 = context1.shape[0]
            max_len2 = context2.shape[0]

            if max_len1 > max_len2:
                input1 = context1[:max_len2, :, :]
                input2 = context2
            else:
                input1 = context1
                input2 = context2[:max_len1, :, :]

            l2_loss = (input1 - input2) ** 2
            l2_loss = l2_loss.sum()

        elif self.input_type == 3:
            mask1_ = mask1.permute(2, 0, 1)  # B, H, T --> T, B, H
            mask2_ = mask2.permute(2, 0, 1)

            masked_context1 = context1.masked_fill(mask1_, float("-Inf")).type_as(context1)
            masked_context2 = context2.masked_fill(mask2_, float("-Inf")).type_as(context2)

            # (T, B, H)
            input1, _ = torch.max(masked_context1, dim=0)   # (T1, B, D) --> (B, D)
            input2, _ = torch.max(masked_context2, dim=0)   # (T2, B, D) --> (B, D)

            l2_loss = (input1 - input2) ** 2
            # multiply by seq length to make aux. loss weight comparable
            l2_loss = l2_loss.sum() * min(context1.shape[0], context1.shape[1])

        elif self.input_type == 4:
            # (T1, B, D) --> (T1, B, D'). Start with D'= D/2
            # (T2, B, D) --> (T2, B, D')
            # (T1, B, D) --> (min(T1, T2), B, D)
            # (T2, B, D) --> (min(T1, T2), B, D)
            max_len1 = context1.shape[0]
            max_len2 = context2.shape[0]
            lan_inv_emb_dim = context1.shape[2] // 2

            if max_len1 > max_len2:
                input1 = context1[:max_len2, :, :lan_inv_emb_dim]
                input2 = context2[:, :, :lan_inv_emb_dim]
            else:
                input1 = context1[:, :, :lan_inv_emb_dim]
                input2 = context2[:max_len1, :, :lan_inv_emb_dim]

            l2_loss = (input1 - input2) ** 2
            # multiply by seq length to make aux. loss weight comparable
            l2_loss = l2_loss.sum() * 2

        elif self.input_type == 5:      # meanpool + position by position
            mask1_ = mask1.permute(2, 0, 1)  # B, H, T --> T, B, H
            mask2_ = mask2.permute(2, 0, 1)

            masked_context1 = context1.masked_fill(mask1_, 0).type_as(context1)
            masked_context2 = context2.masked_fill(mask2_, 0).type_as(context2)

            # (T, B, H) / (T, B, H)
            input1 = torch.sum(masked_context1, dim=0, keepdim=True) / (1 - mask1_.float()).sum(dim=0)
            input2 = torch.sum(masked_context2, dim=0, keepdim=True) / (1 - mask2_.float()).sum(dim=0)

            l2_loss = (input1 - input2) ** 2
            l2_loss = l2_loss.sum() * min(context1.shape[0], context1.shape[1])

            # (T1, B, D) --> (min(T1, T2), B, D)
            # (T2, B, D) --> (min(T1, T2), B, D)
            max_len1 = context1.shape[0]
            max_len2 = context2.shape[0]

            if max_len1 > max_len2:
                input1 = context1[:max_len2, :, :]
                input2 = context2
            else:
                input1 = context1
                input2 = context2[:max_len1, :, :]

            l2_loss += ((input1 - input2) ** 2).sum()
            l2_loss /= 2.0

        elif self.input_type == 6:  # meanpool and maxpool
            mask1_ = mask1.permute(2, 0, 1)  # B, H, T --> T, B, H
            mask2_ = mask2.permute(2, 0, 1)

            masked_context1 = context1.masked_fill(mask1_, 0).type_as(context1)
            masked_context2 = context2.masked_fill(mask2_, 0).type_as(context2)

            # (T, B, H) / (T, B, H)
            input1_meanpool = (torch.sum(masked_context1, dim=0, keepdim=True) / (1 - mask1_.float()).sum(dim=0)).squeeze()
            input2_meanpool = (torch.sum(masked_context2, dim=0, keepdim=True) / (1 - mask2_.float()).sum(dim=0)).squeeze()

            masked_context1 = context1.masked_fill(mask1_, float("-Inf")).type_as(context1)
            masked_context2 = context2.masked_fill(mask2_, float("-Inf")).type_as(context2)

            # (T, B, H)
            input1_maxpool, _ = torch.max(masked_context1, dim=0)   # (T1, B, D) --> (B, D)
            input2_maxpool, _ = torch.max(masked_context2, dim=0)   # (T2, B, D) --> (B, D)

            l2_loss = (torch.cat((input1_meanpool, input1_maxpool)) - torch.cat((input2_meanpool, input2_maxpool))) ** 2
            l2_loss = l2_loss.sum() * min(context1.shape[0], context1.shape[1])
            l2_loss /= 2.0

        elif self.input_type == 7:  # minpool and maxpool
            mask1_ = mask1.permute(2, 0, 1)  # B, H, T --> T, B, H
            mask2_ = mask2.permute(2, 0, 1)

            masked_context1 = context1.masked_fill(mask1_, float("Inf")).type_as(context1)
            masked_context2 = context2.masked_fill(mask2_, float("Inf")).type_as(context2)

            # (T, B, H) / (T, B, H)
            input1_minpool, _ = torch.min(masked_context1, dim=0)  # (T1, B, D) --> (B, D)
            input2_minpool, _ = torch.min(masked_context2, dim=0)  # (T2, B, D) --> (B, D)

            masked_context1 = context1.masked_fill(mask1_, float("-Inf")).type_as(context1)
            masked_context2 = context2.masked_fill(mask2_, float("-Inf")).type_as(context2)

            # (T, B, H)
            input1_maxpool, _ = torch.max(masked_context1, dim=0)  # (T1, B, D) --> (B, D)
            input2_maxpool, _ = torch.max(masked_context2, dim=0)  # (T2, B, D) --> (B, D)

            l2_loss = (torch.cat((input1_minpool, input1_maxpool)) - torch.cat((input2_minpool, input2_maxpool))) ** 2
            l2_loss = l2_loss.sum() * min(context1.shape[0], context1.shape[1])
            l2_loss /= 2.0

        else:
            raise NotImplementedError

        l2_loss = l2_loss * self.weight

        output = defaultdict(lambda: None)
        output['loss'] = l2_loss
        output['data'] = l2_loss.item()

        return output


class CosineEncoderLoss(_Loss):
    def __init__(self, input_type, weight=0.0):
        super(CosineEncoderLoss, self).__init__()
        self.input_type = input_type
        self.weight = weight
        self.cos_sim = nn.CosineSimilarity(dim=-1, eps=1e-6)

    def forward(self, context1, context2, mask1, mask2):
        if self.input_type == 1:    # meanpool
            mask1_ = mask1.permute(2, 0, 1)  # B, H, T --> T, B, H
            mask2_ = mask2.permute(2, 0, 1)

            masked_context1 = context1.masked_fill(mask1_, 0).type_as(context1)
            masked_context2 = context2.masked_fill(mask2_, 0).type_as(context2)

            # (T, B, H) / (T, B, H) --> (B, H)
            input1 = (torch.sum(masked_context1, dim=0, keepdim=True) / (1 - mask1_.float()).sum(dim=0)).squeeze(0)
            input2 = (torch.sum(masked_context2, dim=0, keepdim=True) / (1 - mask2_.float()).sum(dim=0)).squeeze(0)

            # (B, H) --> (B)
            cos_dist = 1.0 - self.cos_sim(input1, input2)
            # multiply by seq length to make aux. loss weight comparable
            cos_loss = cos_dist.sum() * min(context1.shape[0], context1.shape[1])

        elif self.input_type == 2:  # by position
            # (T1, B, D) --> (min(T1, T2), B, D)
            # (T2, B, D) --> (min(T1, T2), B, D)
            max_len1 = context1.shape[0]
            max_len2 = context2.shape[0]

            if max_len1 > max_len2:
                input1 = context1[:max_len2, :, :]
                input2 = context2
            else:
                input1 = context1
                input2 = context2[:max_len1, :, :]

            cos_dist = 1.0 - self.cos_sim(input1, input2)
            cos_loss = cos_dist.sum()

        elif self.input_type == 3:
            mask1_ = mask1.permute(2, 0, 1)  # B, H, T --> T, B, H
            mask2_ = mask2.permute(2, 0, 1)

            masked_context1 = context1.masked_fill(mask1_, float("-Inf")).type_as(context1)
            masked_context2 = context2.masked_fill(mask2_, float("-Inf")).type_as(context2)

            # (T, B, H)
            input1, _ = torch.max(masked_context1, dim=0)   # (T1, B, D) --> (B, D)
            input2, _ = torch.max(masked_context2, dim=0)   # (T2, B, D) --> (B, D)

            # (B, H) --> (B)
            cos_dist = 1.0 - self.cos_sim(input1, input2)
            # multiply by seq length to make aux. loss weight comparable
            cos_loss = cos_dist.sum() * min(context1.shape[0], context1.shape[1])

        else:
            raise NotImplementedError

        cos_loss = cos_loss * self.weight

        output = defaultdict(lambda: None)
        output['loss'] = cos_loss
        output['data'] = cos_loss.item()

        return output