from texttable import Texttable


def hyper_params(d_feature, lamb, dataset,
                 K, Kq, val_batch_size, num_neighbors,
                 num_iterations, inner_steps, pd_updates, meta_batch,
                 eta_1, eta_2, eta_3, eta_4, delta, eps, xi):
    table = Texttable()
    table.set_deco(Texttable.HEADER)
    table.set_cols_align(["l", "l"])
    table.add_rows([
        ["Parameters", "Values"],
        ["Data", dataset],
        ["d_feature", d_feature],
        ["Lambda", lamb],
        ["Support shots", K],
        ["Query shots", Kq],
        ["Validation shots", val_batch_size],
        ["Number of neighbors", num_neighbors],
        ["Inner gradient steps", inner_steps],
        ["Primal-Dual Iterates", pd_updates],
        ["Outer loops", num_iterations],
        ["Meta batches", meta_batch],
        ["eta_1", eta_1],
        ["eta_2", eta_2],
        ["eta_3", eta_3],
        ["eta_4", eta_4],
        ["delta", delta],
        ["eps", eps],
        ["xi", xi]
    ])
    return table.draw().encode()
