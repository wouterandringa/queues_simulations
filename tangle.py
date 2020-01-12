#!/usr/bin/python3


def tangle(fname):
    source = []
    with open(fname, "r") as fp:
        do_print = False
        for line in fp:  # info.split("\n"):
            if line and line[0] == "%":
                continue
            if r"\begin{pyblock}" in line:
                source.append("\n# %%\n")
                do_print = True
                continue
            if r"\end{pyblock}" in line:
                do_print = False
                continue
            if r"\begin{pycode}" in line:
                source.append("\n# %%\n")
                do_print = True
                continue
            if r"\end{pycode}" in line:
                do_print = False
                continue
            if r"\begin{pyverbatim}" in line:
                source.append("\n# %%\n")
                do_print = True
                continue
            if r"\end{pyverbatim}" in line:
                do_print = False
                continue
            if do_print:
                source.append(line)
    return source


files = [
    "tutorial_1_contents.tex",
    "tutorial_2_contents.tex",
    "tutorial_3_contents.tex",
    "tutorial_4_contents.tex",
    "tutorial_5_contents.tex",
]


title = """# %%
'''
# Tutorial {}, solutions


This solution is a jupyter notebook which allows you to directly interact with the code so that
you can see the effect of any changes you may like to make.

Author: Nicky van Foreest
'''
"""


for i, fname in enumerate(files):
    source = tangle(fname)
    f_to = f"tutorial_{i+1}.py"
    source.insert(0, title.format(i + 1))
    with open(f_to, "w") as fp:
        fp.write("".join(source))
