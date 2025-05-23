""" Onmt NMT Model base class definition """
import torch.nn as nn


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """

    def __init__(self, encoder, decoder, refer=None):
        super(NMTModel, self).__init__()
        self.encoder = encoder.cuda()
        self.refer = refer
        self.decoder = decoder.cuda()

    def forward(self, src, tgt, lengths, ref=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.

        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
        """
        tgt = tgt[:-1]  # exclude last target from inputs

        if ref:
            ref_lengths, ind = ref[1].sort(descending=True)
            sorted_ref = ref[0][:, ind, :]
            ref_state, ref_bank, ref_lengths = self.encoder(sorted_ref, ref_lengths)
            _, recover = ind.sort(descending=False)
            # ref_state = ref_state[:, recover, :]
            ref_bank = ref_bank[:, recover, :]
            ref_lengths = ref_lengths[recover]
            ref_tuple = (ref_bank, ref_lengths)
        else:
            ref_tuple = None
        src = src.cuda()
        lengths = lengths.cuda()
        enc_state, memory_bank, lengths = self.encoder(src, lengths)
        self.decoder.init_state(src, memory_bank, enc_state)
        dec_out, attns = self.decoder(tgt, memory_bank,
                                      memory_lengths=lengths, ref=ref_tuple)

        return dec_out, attns
