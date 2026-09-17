import os

import pandas as pd


def fmt(mean, std, prec=4):
    return f"${mean:.{prec}f} \\pm {std:.{prec}f}$"


def to_latex(df, caption, label, column_format=None):
    cf = column_format or ("l" * df.index.nlevels + "c" * len(df.columns))
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{" + cf + r"}",
        r"\toprule",
    ]
    headers = list(df.index.names) + list(df.columns)
    lines.append(" & ".join(str(h) for h in headers) + r" \\")
    lines.append(r"\midrule")

    if df.index.nlevels > 1:
        # pre-compute consecutive group sizes for \multirow
        idx0_seq = [idx[0] for idx in df.index]
        group_sizes = {}
        i = 0
        while i < len(idx0_seq):
            key = idx0_seq[i]
            j = i
            while j < len(idx0_seq) and idx0_seq[j] == key:
                j += 1
            group_sizes[i] = (key, j - i)
            i = j
        group_start = set(group_sizes.keys())

        prev_idx0 = None
        for row_i, (idx, row) in enumerate(df.iterrows()):
            idx_parts = list(idx)
            if idx_parts[0] != prev_idx0 and prev_idx0 is not None:
                lines.append(r"\midrule")
            if row_i in group_start:
                _, n = group_sizes[row_i]
                idx_parts[0] = rf"\multirow{{{n}}}{{*}}{{{idx_parts[0]}}}"
            else:
                idx_parts[0] = ""
            prev_idx0 = list(idx)[0]
            cells = [str(v) for v in idx_parts] + [str(v) for v in row.values]
            lines.append(" & ".join(cells) + r" \\")
    else:
        for idx, row in df.iterrows():
            idx_parts = [idx]
            cells = [str(v) for v in idx_parts] + [str(v) for v in row.values]
            lines.append(" & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def save_table(df, name, dir_tables, caption, label, column_format=None):
    from IPython.display import display as _display

    os.makedirs(dir_tables, exist_ok=True)
    df.to_csv(f"{dir_tables}/{name}.csv")
    latex_str = to_latex(df, caption, label, column_format)
    with open(f"{dir_tables}/{name}.tex", "w") as f:
        f.write(latex_str)

    print(f"  Saved: {name}.csv  |  {name}.tex")
    print()
    _display(df)
    print("\n[LaTeX source]")
    print(latex_str)
    print("=" * 70)
