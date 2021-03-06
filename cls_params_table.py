from texttable import Texttable


def hyper_params(d_feature, lamb, tasks, data_path, dataset,
            K, Kq, val_batch_size, num_neighbors,
            num_iterations, inner_steps, pd_updates,
            eta_1, eta_2, eps,radius,meta_eta_1,meta_eta_2,delta,net_dim):
    table = Texttable(120)
    table.set_deco(Texttable.HEADER)
    table.set_cols_width([30, 80])
    table.set_precision(6)
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
        ["num_iterations", num_iterations],
        ["eta_1", eta_1],
        ["eta_2", eta_2],
        ["meta_eta_1", meta_eta_1],
        ["meta_eta_2", meta_eta_2],
        ["delta", delta],
        ["eps", eps],
        ["radius", radius],
        ["net_dim",net_dim],

    ])
    return table.draw().encode()
