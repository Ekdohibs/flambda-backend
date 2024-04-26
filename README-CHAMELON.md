
 The source code of the chamelon delta-debugger is available in the `chamelon` directory.

Demonstration
=============

The demonstration can be run with `./demo.sh`, which builds the tool then minimizes 
the given input, `ocaml/testsuite/tests/backtrace/inline_traversal_test.ml`,which fails
 when compiled with `_build/_bootinstall/bin/ocamlopt.opt -c -Oclassic`. The output
is `minimized.ml`.

In the standard output, you should see:
+ First, logs from building the tool
+ Second, the steps performed by the tool:
 - `Starting to minimize minimized.ml` before a new minimization attempt.
 - Then for each atomic heuristics: `Trying heuristic-name: pos=i, len=j...` before trying
   to perform `heuristic-name` at `j` program points starting from position `i`, and:
   * `Reduced.` when the transformation is applied.
   * `Removes error.` when the transformation removes the error.
   * `No more changes.` when the end of the program is reached.
