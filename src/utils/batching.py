from itertools import islice


def batched(iterable, batch_size: int):
    iterator = iter(iterable)

    while batch := list(islice(iterator, batch_size)):
        yield batch
