#!/bin/sh

# Makes random units (gen.py), has the test tool check each, and counts what
# it says:
#
#   run.sh <ocamlopt> <flsemantics.exe> <first seed> <count> <directory> \
#     [profile ...]
#
# from the directory above this one. The units, their terms and what the tool
# printed are left in the directory given. See README.md.

set -eu

ocamlopt=$1
tool=$2
first=$3
count=$4
out=$5
shift 5
profiles=${*:-known unknown functions known_scalar unknown_scalar}

here=$(dirname "$0")
mkdir -p "$out"
: >"$out/verdicts.txt"
: >"$out/related.txt"
seed=$first
while [ "$seed" -lt $((first + count)) ]; do
  for profile in $profiles; do
    python3 "$here/gen.py" "$seed" "$profile" "$out"
    unit=$out/c_${profile}_$(printf %04d "$seed")
    if sh "$here/../from_ocaml.sh" "$ocamlopt" "$unit.ml" "$unit.fl" \
      >"$unit.log" 2>&1; then
      # The limit is on the processor time of the tool, not on the time
      # that passes, which depends on what else the machine is running. At
      # the soft limit the system sends the tool SIGXCPU, which ends it with
      # the status 128 + 24; the hard limit is one second more, so that this
      # signal is the one that ends it. The soft limit is set first, so that
      # it does not exceed the hard limit that is asked next. Where the soft
      # limit that is inherited is unlimited, a finite hard limit set first
      # is refused, and the message of the shell would be the first line of
      # what is kept as the output of the tool. Any other signal is reported
      # with its status and no cause.
      code=0
      (
        ulimit -S -t 60
        ulimit -H -t 61
        exec "$tool" "$unit.fl"
      ) >"$unit.out" 2>&1 || code=$?
      if [ "$code" = 152 ]; then
        verdict="no answer in 60 seconds of processor time"
      elif [ "$code" -ge 128 ]; then
        verdict="the tool was ended by a signal (status $code)"
      elif grep -q "DIFFERENT" "$unit.out"; then
        verdict="THE RUNS DIFFER"
      elif grep -q "NOT WELL-FORMED" "$unit.out"; then
        verdict="THE TERM AFTER SIMPLIFY IS NOT WELL-FORMED"
      elif [ "$code" != 0 ]; then
        # The tool exits with 1 in the two cases above. Any other status
        # that is not 0, and is not counted above as a limit or a signal,
        # is counted as a failure of the tool, with the status: a text
        # that it cannot read, for one. Such a unit has no verdict to
        # count.
        verdict="the tool failed (status $code)"
      elif grep -q "not checked" "$unit.out"; then
        verdict="not checked: $(sed -n \
          's/.*not checked: the model has not got \([a-z ]*[A-Za-z_]*\).*/\1/p' \
          "$unit.out" | head -1)"
      elif grep -q "undefined" "$unit.out"; then
        verdict="undefined before Simplify"
      elif grep -q "not decided" "$unit.out"; then
        verdict="not decided"
      elif grep -q "^  related once lifted" "$unit.out"; then
        # The line of the second check is under "not validated", and the
        # unit is counted so; what the check answered of it is counted apart.
        verdict="not validated"
        echo "$profile | $(sed -n \
          's/^  \(related once lifted[^:]*\):.*/\1/p' "$unit.out")" \
          >>"$out/related.txt"
      else
        # The verdict is read by its text and not by its place: the tool
        # prints lines after it, those that are under it and the lines of
        # the regions. It is the last line that begins as a verdict does.
        verdict=$(grep -E '^  (not )?validated' "$unit.out" | tail -1 |
          sed 's/^ *//')
        if [ -z "$verdict" ]; then
          verdict="no line of a verdict"
        fi
      fi
    else
      verdict="does not compile"
    fi
    echo "$profile | $verdict" >>"$out/verdicts.txt"
  done
  seed=$((seed + 1))
done
sort "$out/verdicts.txt" | uniq -c | sort -k2,2 -k1,1nr
if [ -s "$out/related.txt" ]; then
  echo "of the units not validated, the second check answers of:"
  sort "$out/related.txt" | uniq -c
fi
