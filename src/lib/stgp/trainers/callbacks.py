def progress_bar_callback(num_epochs):
    """
        Simple progressbar - does not display learning objective value
    """
    from tqdm import tqdm


    bar = tqdm(total=num_epochs)

    def inner(epoch, grad, val):
        bar.update(1)

    return inner


def progress_bar_callback_notebook(num_epochs):
    """
        Simple progressbar - does not display learning objective value
    """
    from tqdm.notebook import trange, tqdm


    bar = tqdm(total=num_epochs)

    def inner(epoch, grad, val):
        bar.update(1)

    return inner

_lowest_val = None # assuming there is only every ONE checkpoint callback wrapper, otherwise will have sync issues
_lowest_epoch = None
def checkpoint_callback_wrapper(inner_callback, model=None, checkpoint_every=100, checkpoint_name_callback=lambda epoch:'{epoch}', checkpoint_lowest_val = False):
    global _lowest_val
    global _lowest_epoch

    if model is None:
        raise RuntimeError('Model must be passed!')

    # reset lowest val
    _lowest_val = None
    _lowest_epoch = None

    def inner(epoch, grad, val):
        global _lowest_val
        global _lowest_epoch

        if epoch % checkpoint_every == 0:
            model.checkpoint(checkpoint_name_callback(epoch=epoch))

        else:
            # dont need to checkpoint multiple times
            if checkpoint_lowest_val:
                if (_lowest_val == None) or (val < _lowest_val)  :
                    _lowest_val = val
                    _lowest_epoch = epoch
                    model.checkpoint(checkpoint_name_callback(epoch=epoch))

        inner_callback(epoch, grad, val)

    return inner
    



