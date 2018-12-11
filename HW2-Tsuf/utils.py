import torch


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, batch_size, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    batch_amount = data.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, batch_amount * batch_size)
    # Evenly divide the data across the batch_size batches.
    # Calling contiguous() is necessary because narrow() and view() doesn't change the memory itself, but
    #   the indices. Some functions later in code require the memory to be reorganized as continuous, that's
    #   why we call this function.
    data = data.view(batch_size, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data


def get_batch(source, i, args, seq_len=None):
    seq_len = min(seq_len if seq_len else args.seq_len, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target