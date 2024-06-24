
 The source code of the chamelon delta-debugger is available in the `chamelon` directory.
 This is a version of the ocaml-flambda compiler, on top of which we reverted a certain
 number of bugfixes, in order to demonstrate real, in the wild minimisation problems.

Demonstration
=============

The demonstration can be run with `./demo.sh`, which builds the tool then minimizes 
sevaral inputs. If you use the provided docker image, you can simply do:
```bash
docker image import chamelon-docker.tar.bz2 chamelon:chamelon
docker run -u user -w /home/user -ti chamelon:chamelon /bin/bash
```
This will get you a shell inside the docker image, from which you can launch `./demo.sh`.

Otherwise, you can install `ocaml-4.14.1`, `menhir.20210419` and `dune.3.8.1` (for
instance using opam), before running `./demo.sh`.

You should see in the standard output:
+ First, logs from building the tool
+ Second, for each input, the steps performed by the tool:
 - `Starting to minimize [minimized_name].ml` before a new minimization attempt.
 - Then for each atomic heuristics: `Trying heuristic-name: pos=i, len=j...` before trying
   to perform `heuristic-name` at `j` program points starting from position `i`, and:
   * `Reduced.` when the transformation is applied.
   * `Removes error.` when the transformation removes the error.
   * `No more changes.` when the end of the program is reached.

After this, the minimized files appear directly in this folder. They have the same name as
the original files, with the suffix `_min`: this way, `env.ml` is minimized in `env_min.ml`.
If you wish, you can then compare original files to minimized files.

We describe each of the test cases below:

- `ocaml/testsuite/tests/backtrace/inline_traversal_test.ml`:
  This file failed when compiled with `_build/_bootinstall/bin/ocamlopt.opt -Oclassic`.
  This was due to a bug in the then in-progress global pass, when inlining across
  different optimisation modes.

- `binary_packing.ml`:
  This was the original motivation for writing chamelon. This file failed to compile due
  to a bug in the optimisation of pattern-matching, as we can see in the output where
  pattern matching is one of the few constructions remaining.

- `letrec.ml`:
  This failed due to a bug in the compilation of recursive values. While this example
  was written by hand after suspecting a bug from the code, chamelon is still able to
  minimize it to provide an easier-to-debug example.

- `seqtest.ml`:
  This example failed due to a bug where non-simplified versions of functions ended up
  in the output, which the code didn't expect.

- `ocaml/typing/env.ml`:
  This example is the largest we show here, and as such, takes longer than the others
  (about 20 minutes on a laptop).
  It was due to a bug in the computation of the possible shapes of the values for some
  types.

  It is notable in several aspects besides its size:

  - The file itself does not compile without its associated `.mli` file. Consequently,
    this prevents some minimisations; as such, we perform two successive minimisations,
    one with the `.mli` file, and a second one by minimizing the result of the first,
    not need the `.mli` file to compile. This allows us to successively reduce a ~4000
    lines file, to a ~450 lines file, to a file with 34 lines.

  - It does not crash the compiler like the others, instead producing an error at runtime.
    Fortunately, there is a pattern in the produced code which, here, only appears when
    the bug is present. Thus, we minimize the example by ensuring that pattern still
    appears in the minimized file.


Using chamelon
==============

If you want to run chamelon outside the demonstration setting, see chamelon's own README
(`chamelon/README.md`) for more information about its usage. Chamelon can also be directly
downloaded from github, either in the `chamelon` subfolder of
https://github.com/ocaml-flambda/flambda-backend for a flambda-enabled version, or at
the stand-alone chamelon repository https://github.com/Ekdohibs/chamelon .



