from texttable import Texttable


def hyper_params(d_feature, dataset,
                 K, Kq, val_batch_size, num_neighbors,
                 num_iterations, inner_steps, meta_batch,
                 eta_1, eta_3):
    table = Texttable()
    table.set_deco(Texttable.HEADER)
    table.set_cols_align(["l", "l"])
    table.add_rows([
        ["Parameters", "Values"],
        ["Data", dataset],
        ["d_feature", d_feature],
        ["Support shots", K],
        ["Query shots", Kq],
        ["Validation shots", val_batch_size],
        ["Number of neighbors", num_neighbors],
        ["Inner gradient steps", inner_steps],
        ["Outer loops", num_iterations],
        ["Meta batches", meta_batch],
        ["eta_1", eta_1],
        ["eta_3", eta_3],
    ])
    return table.draw().encode()
