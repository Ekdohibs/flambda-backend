#!/bin/sh

# Runs the test tool, the first argument, on the arguments that follow, and
# prints: what the tool printed on its standard output; the first line of
# what it printed on its standard error, if it printed anything there; and
# the status with which it exited. For the controls of this directory, on
# which the tool does not exit with the status 0 (see dune).

set -u

tool=$1
shift

errors=$(mktemp)
trap 'rm -f "$errors"' EXIT

"$tool" "$@" 2>"$errors"
status=$?

if [ -s "$errors" ]; then
  printf 'the standard error, its first line: %s\n' "$(head -n 1 "$errors")"
else
  echo "the standard error: nothing"
fi
echo "exit status: $status"
