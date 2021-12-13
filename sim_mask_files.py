import torch
import matplotlib.pyplot as plt


def chart_GR(batch_size):
    # GR
    pos_sample_indicators = torch.roll(torch.eye(2 * batch_size), batch_size, 1)
    neg_sample_indicators = (torch.ones(2 * batch_size) - torch.eye(2 * batch_size))
    plot_matrix(pos_sample_indicators, neg_sample_indicators, 'Positive and negative sample matrices GR')
    similarity_matrix = torch.ones(2 * batch_size, 2 * batch_size)

    numerator_pos = torch.exp(similarity_matrix)[pos_sample_indicators.bool()]  # shape: [3*batch-size]
    # calculate the denominator by summing over each pair except for the diagonal elements
    denominator = torch.sum((torch.exp(similarity_matrix) * neg_sample_indicators), dim=1)

    loss = torch.mean(-torch.log(numerator_pos / denominator))

    print(loss, numerator_pos.shape, denominator.shape)


def chart_GDMinus(batch_size, partition_size):
    # GD-
    N = 3 * batch_size
    pos_sample_indicators_1 = torch.roll(torch.eye(N), batch_size, 1)
    pos_sample_indicators_2 = torch.roll(torch.eye(N), 2 * batch_size, 1)

    neg_sample_indicators = torch.ones(N, N)
    for i in range(batch_size // partition_size):
        neg_sample_indicators = neg_sample_indicators \
                                - torch.roll(torch.eye(N), i * partition_size, 1) \
                                - torch.roll(torch.eye(N), i * partition_size + batch_size, 1) \
                                - torch.roll(torch.eye(N), i * partition_size + 2 * batch_size, 1)

    plot_matrix((pos_sample_indicators_1 + pos_sample_indicators_2), neg_sample_indicators,
                'Positive and negative sample matrices GD-')

    similarity_matrix = torch.ones(N, N)
    numerator_pos_1 = torch.exp(similarity_matrix)[
        pos_sample_indicators_1.bool()]  # shape: [3*batch-size]
    numerator_pos_2 = torch.exp(similarity_matrix)[
        pos_sample_indicators_2.bool()]  # shape: [3*batch-size]
    # calculate the denominator by summing over each pair except for the diagonal elements
    denominator = torch.sum((torch.exp(similarity_matrix) * neg_sample_indicators), dim=1)

    loss = torch.mean(-torch.log((numerator_pos_1 + numerator_pos_2) / denominator))

    print(loss, numerator_pos_1.shape, numerator_pos_2.shape, denominator.shape)


def chart_GD_1(batch_size, partition_size):
    # GD
    N = 3 * batch_size
    similarity_matrix = torch.ones(N, N)

    pos_all = []

    for i in range(batch_size // partition_size):
        if i != 0:
            pos_sample_indicator = torch.zeros(N, N)
            pos_sample_indicator = pos_sample_indicator + torch.roll(torch.eye(N), i * partition_size, 1)
            pos_all.append(pos_sample_indicator)

        pos_sample_indicator = torch.zeros(N, N)
        pos_sample_indicator = (
                 pos_sample_indicator + torch.roll(torch.eye(N), i * partition_size + batch_size, 1))
        pos_all.append(pos_sample_indicator)
        pos_sample_indicator = torch.zeros(N, N)
        pos_sample_indicator = (
                pos_sample_indicator + torch.roll(torch.eye(N), i * partition_size + 2 * batch_size, 1))
        pos_all.append(pos_sample_indicator)

        # pos_sample_indicators_1 = pos_sample_indicators_1 - torch.eye(N)

    # the negative sample indices remain unchanged from GD-'s negative sample indices
    neg_sample_indicators = torch.ones(N, N)
    for i in range(batch_size // partition_size):
        neg_sample_indicators = neg_sample_indicators \
                                - torch.roll(torch.eye(N), i * partition_size, 1) \
                                - torch.roll(torch.eye(N), i * partition_size + batch_size, 1) \
                                - torch.roll(torch.eye(N), i * partition_size + 2 * batch_size, 1)
    # print(len(pos_all))
    pos_all_mask = torch.zeros(N, N)
    for i in range(len(pos_all)):
        plot_matrix(pos_all[i], neg_sample_indicators, f'Positive and negative sample matrices GD {i}')
        pos_all_mask = pos_all_mask + pos_all[i]

    plot_matrix(pos_all_mask, neg_sample_indicators, f'Positive and negative sample matrices GD overlay')


    # calculate the denominator by summing over each pair except for the diagonal elements
    denominator = torch.sum((torch.exp(similarity_matrix) * neg_sample_indicators), dim=1)

    loss = 0

    # calculate the numerator by selecting the appropriate indices of the positive samples using the
    # pos_sample_indicators matrix
    for i in range(len(pos_all)):
        numerator_pos = torch.exp(similarity_matrix)[pos_all[i].bool()]
        loss += torch.mean(-torch.log(numerator_pos/denominator))
        print(loss, numerator_pos.shape, denominator.shape)

    loss /= len(pos_all)

def chart_GD_2(batch_size, partition_size):
    # GD
    N = 3 * batch_size
    similarity_matrix = torch.ones(N, N)

    pos_all = []

    for i in range(batch_size // partition_size):
        if i != 0:
            pos_sample_indicator = torch.zeros(N, N)
            pos = torch.zeros(batch_size, batch_size)
            pos = pos + torch.roll(torch.eye(batch_size), i * partition_size, 1)
            pos_sample_indicator[0:batch_size, 0:batch_size] = pos
            pos_sample_indicator[batch_size:2*batch_size, batch_size:2*batch_size] = pos
            pos_sample_indicator[2*batch_size:3*batch_size, 2*batch_size:3*batch_size] = pos
            pos_all.append(pos_sample_indicator)

    pos_sample_indicator = torch.zeros(N, N)
    pos_sample_indicator = (
                 pos_sample_indicator + torch.roll(torch.eye(N), batch_size, 1))
    pos_all.append(pos_sample_indicator)
    pos_sample_indicator = torch.zeros(N, N)
    pos_sample_indicator = (
            pos_sample_indicator + torch.roll(torch.eye(N), 2 * batch_size, 1))
    pos_all.append(pos_sample_indicator)

        # pos_sample_indicators_1 = pos_sample_indicators_1 - torch.eye(N)

    # the negative sample indices remain unchanged from GD-'s negative sample indices
    neg_sample_indicators = torch.ones(N, N)
    for i in range(batch_size // partition_size):
        neg_sample_indicators = neg_sample_indicators \
                                - torch.roll(torch.eye(N), i * partition_size, 1) \
                                - torch.roll(torch.eye(N), i * partition_size + batch_size, 1) \
                                - torch.roll(torch.eye(N), i * partition_size + 2 * batch_size, 1)
    # print(len(pos_all))
    pos_all_mask = torch.zeros(N, N)
    for i in range(len(pos_all)):
        plot_matrix(pos_all[i], neg_sample_indicators, f'Positive and negative sample matrices GD {i}')
        pos_all_mask = pos_all_mask + pos_all[i]

    plot_matrix(pos_all_mask, neg_sample_indicators, f'Positive and negative sample matrices GD overlay')


    # calculate the denominator by summing over each pair except for the diagonal elements
    denominator = torch.sum((torch.exp(similarity_matrix) * neg_sample_indicators), dim=1)

    loss = 0

    # calculate the numerator by selecting the appropriate indices of the positive samples using the
    # pos_sample_indicators matrix
    for i in range(len(pos_all)):
        numerator_pos = torch.exp(similarity_matrix)[pos_all[i].bool()]
        loss += torch.mean(-torch.log(numerator_pos/denominator))
        print(loss, numerator_pos.shape, denominator.shape)

    loss /= len(pos_all)

def chart_GDMinus_alt(batch_size, partition_size):
    # GD- alt
    N = 2 * batch_size
    similarity_matrix = torch.ones(N, N)
    # for GD-, the positive sample indices are similar to GR (simclr)
    # pos_sample_indicators = torch.roll(torch.eye(N), batch_size, 1) +

    # for the neg sample indices, we also need to consider the partition size here because we do not want to contrast against them
    # so for each row in the neg indicator matrix, we have the diagonal zeros (by default) and now we also have 0s spaced out
    # according to the partition size as well (within each batch). This is replicated across the 3 batches
    neg_sample_indicators = torch.ones((N, N))
    for i in range(batch_size // partition_size):
        neg_sample_indicators = neg_sample_indicators \
                                - torch.roll(torch.eye(N), i * partition_size, 1) \
                                - torch.roll(torch.eye(N), i * partition_size + batch_size, 1)

    plot_matrix(pos_sample_indicators, neg_sample_indicators, f'Positive and negative sample matrices GD Minus alternative')

    # calculate the numerator by selecting the appropriate indices of the positive samples using the
    # pos_sample_indicators matrix
    numerator_pos = torch.exp(similarity_matrix)[
        pos_sample_indicators.bool()]  # shape: [3*batch-size]
    # calculate the denominator by summing over each pair except for the diagonal elements
    denominator = torch.sum((torch.exp(similarity_matrix) * neg_sample_indicators), dim=1)

    loss = torch.mean(-torch.log(numerator_pos/ denominator))

    print(loss)

def chart_GD_alt(batch_size, partition_size):
    # GD
    N = 2 * batch_size
    similarity_matrix = torch.ones(N, N)

    pos_all = []

    for i in range(batch_size // partition_size):
        if i != 0:
            pos_sample_indicator = torch.zeros(N, N)
            pos_sample_indicator = pos_sample_indicator + torch.roll(torch.eye(N), i * partition_size, 1)
            pos_all.append(pos_sample_indicator)

        pos_sample_indicator = torch.zeros(N, N)
        pos_sample_indicator = (
                pos_sample_indicator + torch.roll(torch.eye(N), i * partition_size + batch_size, 1))
        pos_all.append(pos_sample_indicator)

        # pos_sample_indicators_1 = pos_sample_indicators_1 - torch.eye(N)

    # the negative sample indices remain unchanged from GD-'s negative sample indices
    neg_sample_indicators = torch.ones(N, N)
    for i in range(batch_size // partition_size):
        neg_sample_indicators = neg_sample_indicators \
                                - torch.roll(torch.eye(N), i * partition_size, 1) \
                                - torch.roll(torch.eye(N), i * partition_size + batch_size, 1) \
    # print(len(pos_all))
    pos_all_mask = torch.zeros(N, N)
    for i in range(len(pos_all)):
        plot_matrix(pos_all[i], neg_sample_indicators, f'Positive and negative sample matrices GD alt {i}')
        pos_all_mask = pos_all_mask + pos_all[i]

    plot_matrix(pos_all_mask, neg_sample_indicators, f'Positive and negative sample matrices GD alt overlay')


    # calculate the denominator by summing over each pair except for the diagonal elements
    denominator = torch.sum((torch.exp(similarity_matrix) * neg_sample_indicators), dim=1)

    loss = 0

    # calculate the numerator by selecting the appropriate indices of the positive samples using the
    # pos_sample_indicators matrix
    for i in range(len(pos_all)):
        numerator_pos = torch.exp(similarity_matrix)[pos_all[i].bool()]
        loss += torch.mean(-torch.log(numerator_pos/denominator))
        print(loss, numerator_pos.shape, denominator.shape)

    loss /= len(pos_all)

def chart_GD_alt_2(batch_size, partition_size):
    # GD
    N = 2 * batch_size
    similarity_matrix = torch.ones(N, N)

    pos_all = []

    for i in range(batch_size // partition_size):
        if i != 0:
            pos_sample_indicator = torch.zeros(N, N)
            pos_sample_indicator = pos_sample_indicator + torch.roll(torch.eye(N), i * partition_size, 1)
            pos_all.append(pos_sample_indicator)

    pos_sample_indicator = torch.zeros(N, N)
    pos_sample_indicator = (
            pos_sample_indicator + torch.roll(torch.eye(N), i * batch_size, 1))
    pos_all.append(pos_sample_indicator)

        # pos_sample_indicators_1 = pos_sample_indicators_1 - torch.eye(N)

    # the negative sample indices remain unchanged from GD-'s negative sample indices
    neg_sample_indicators = torch.ones(N, N)
    for i in range(batch_size // partition_size):
        neg_sample_indicators = neg_sample_indicators \
                                - torch.roll(torch.eye(N), i * partition_size, 1) \
                                - torch.roll(torch.eye(N), i * partition_size + batch_size, 1) \
    # print(len(pos_all))
    pos_all_mask = torch.zeros(N, N)
    for i in range(len(pos_all)):
        plot_matrix(pos_all[i], neg_sample_indicators, f'Positive and negative sample matrices GD alt {i}')
        pos_all_mask = pos_all_mask + pos_all[i]

    plot_matrix(pos_all_mask, neg_sample_indicators, f'Positive and negative sample matrices GD alt overlay')


    # calculate the denominator by summing over each pair except for the diagonal elements
    denominator = torch.sum((torch.exp(similarity_matrix) * neg_sample_indicators), dim=1)

    loss = 0

    # calculate the numerator by selecting the appropriate indices of the positive samples using the
    # pos_sample_indicators matrix
    for i in range(len(pos_all)):
        numerator_pos = torch.exp(similarity_matrix)[pos_all[i].bool()]
        loss += torch.mean(-torch.log(numerator_pos/denominator))
        print(loss, numerator_pos.shape, denominator.shape)

    loss /= len(pos_all)


def plot_matrix(pos_sample_indicators, neg_sample_indicators, title):
    fig, axs = plt.subplots(1, 2)
    fig.suptitle(title)
    axs[0].imshow(pos_sample_indicators.numpy(), cmap='gray')
    axs[1].imshow(neg_sample_indicators.numpy(), cmap='gray')
    fig.show()


if __name__ == "__main__":
    batch_size = 16  # slices
    partition = 8

    # chart_GR(batch_size)
    # chart_GDMinus(batch_size, partition)
    # chart_GDMinus_alt(batch_size, partition)
    # chart_GD_1(batch_size, partition)
    # chart_GD_2(batch_size, partition)
    chart_GD_alt(batch_size, partition)
