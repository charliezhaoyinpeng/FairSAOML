from texttable import Texttable


def hyper_params(d_feature, lamb, dataset,
                 K, val_batch_size, num_neighbors,
                 eta_1, delta, eps):
    table = Texttable()
    table.set_deco(Texttable.HEADER)
    table.set_cols_align(["l", "l"])
    table.add_rows([
        ["Parameters", "Values"],
        ["Data", dataset],
        ["d_feature", d_feature],
        ["lambda", lamb],
        ["K", K],
        ["Validation shots", val_batch_size],
        ["Number of neighbors", num_neighbors],
        ["eta_1", eta_1],
        ["delta", delta],
        ["eps", eps]
    ])
    return table.draw().encode()
