#!/usr/bin/python3
import enum
import glob


class PrintMachine(enum.Enum):
    PRINT = 0
    DONT_PRINT = 1

    @property
    def print_triggers(self):
        return {r"\begin{pyblock}", r"\begin{pycode}", r"\begin{pyverbatim}"}

    @property
    def dont_print_triggers(self):
        return {r"\end{pyblock}", r"\end{pycode}", r"\end{pyverbatim}"}

    def should_print_line(self):
        return self == PrintMachine.PRINT

    def determine_next_state(self, line):
        """
        Determines the next state based on the content of the passed-in line.
        Returns a tuple of (next state, current state).
        """
        if self == self.PRINT and self._should_go_to_dont_print(line):
            return self.DONT_PRINT, self.PRINT

        if self == self.DONT_PRINT and self._should_go_to_print(line):
            return self.PRINT, self.DONT_PRINT

        return self, self.PRINT

    def _should_go_to_dont_print(self, line):
        return any(trigger in line for trigger in self.dont_print_triggers)

    def _should_go_to_print(self, line):
        return any(trigger in line for trigger in self.print_triggers)


def tangle(fname):
    state = PrintMachine.DONT_PRINT
    source = []

    with open(fname, "r") as fp:
        for line in fp:  # info.split("\n"):
            if line and line[0] == "%":
                continue

            state, state_from = state.determine_next_state(line)

            if state == PrintMachine.PRINT != state_from:
                source.append("\n# %%\n")
                continue

            if state.should_print_line():
                source.append(line)

    return source


title = """# %%
'''
# Tutorial {}, solutions


This solution is a jupyter notebook which allows you to directly interact with the code so that
you can see the effect of any changes you may like to make.

Author: Nicky van Foreest
'''
"""

for i, fname in enumerate(glob.glob("tex_files/tutorial_*_contents.tex")):
    source = tangle(fname)

    if source == "":
        continue

    f_to = f"python_files/tutorial_{i + 1}.py"

    source.insert(0, title.format(i + 1))

    with open(f_to, "w") as fp:
        fp.write("".join(source))
