#!/bin/sh

# Runs the test tool, the first argument, on each control X.fl that follows,
# and prints what the tool says of it. Each run is in a directory of its own
# that holds the texts given to the tool and nothing else.
#
#   alone   X.fl is the term before; if the guard lets it through, the tool's
#           Simplify makes the term after
#   both    X.fl is the term before and, by the option -after, the term
#           after; Simplify is not run
#
# And for a control whose comment begins "MUST BE REFUSED", beside a plain
# unit of the same name, which binds one block and nothing else:
#
#   as the term after    the plain unit is the term before
#   as the term before   the plain unit is the term after
#
# (These two runs are made of the controls that must be refused and of no
# other: they show that the tool refuses the text as either term, with no
# run and no verdict.) After what the tool printed on its standard output
# come the first line of its standard error, if it printed anything there,
# and the status with which it exited.
#
# The comment of a control begins "MUST BE REFUSED" or "MUST PASS". This
# script holds the control to it: of one that must be refused, every run has
# to say that a name is not in scope; of one that must pass, no run may say
# so. Where a run does otherwise the script prints a line "NOT AS ITS COMMENT
# SAYS" and exits with the status 1, whatever the expected file has. See dune
# and README.md.

set -u

# The tool is run from other directories than this one.
case "$1" in
/*) tool=$1 ;;
*) tool=$(pwd)/$1 ;;
esac
shift

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
failed=0

# one NAME BEFORE-TEXT [AFTER-TEXT]: a run of the tool in a new directory;
# its standard output is left in $work/output
one() {
  name=$1
  rm -rf "$work/run"
  mkdir "$work/run"
  cp "$2" "$work/run/$name.fl"
  if [ $# -eq 3 ]; then
    cp "$3" "$work/run/after.fl"
    (cd "$work/run" && "$tool" -after after.fl "$name.fl") \
      >"$work/output" 2>"$work/errors"
  else
    (cd "$work/run" && "$tool" "$name.fl") >"$work/output" 2>"$work/errors"
  fi
  status=$?
  cat "$work/output"
  if [ -s "$work/errors" ]; then
    printf 'the standard error, its first line: %s\n' \
      "$(head -n 1 "$work/errors")"
  else
    echo "the standard error: nothing"
  fi
  echo "exit status: $status"
}

# held NAME RUN MUST: compares what the last run said with the comment
held() {
  if grep -q -e 'not in scope' -e 'NOT IN SCOPE' "$work/output"; then
    said=refused
  else
    said=passed
  fi
  if [ "$said" != "$3" ]; then
    echo "$1.fl, $2: NOT AS ITS COMMENT SAYS"
    failed=1
  fi
}

for text in "$@"; do
  name=$(basename "$text" .fl)
  case "$(head -n 1 "$text")" in
  "(* MUST BE REFUSED"*) must=refused ;;
  "(* MUST PASS"*) must=passed ;;
  *)
    echo "== $name.fl: its comment begins with neither" \
      "MUST BE REFUSED nor MUST PASS"
    failed=1
    continue
    ;;
  esac
  if [ "$must" = refused ]; then
    echo "== $name.fl (must be refused)"
  else
    echo "== $name.fl (must pass)"
  fi
  echo "-- alone"
  one "$name" "$text"
  held "$name" alone "$must"
  echo "-- both"
  one "$name" "$text" "$text"
  held "$name" both "$must"
  if [ "$must" = refused ]; then
    first=$(printf '%s' "$name" | cut -c 1 | tr a-z A-Z)
    unit=$first$(printf '%s' "$name" | cut -c 2-)
    printf 'let $caml%s = Block 0 (0) in\ncont done ($caml%s)\n' \
      "$unit" "$unit" >"$work/plain.fl"
    echo "-- as the term after"
    one "$name" "$work/plain.fl" "$text"
    held "$name" "as the term after" "$must"
    echo "-- as the term before"
    one "$name" "$text" "$work/plain.fl"
    held "$name" "as the term before" "$must"
  fi
done

exit $failed
