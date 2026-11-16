#!/bin/sh

# Runs the test that what Simplify makes of a unit does not depend on the
# order of the names of the unit (README.md; the program is
# ../../tools/nameorder/flnameorder.ml):
#
#   run.sh <flnameorder.exe> K1 K2 OPTIONS LABELS COUNTS UNIT.fl ...
#
# K1 and K2 are two numbers, not 0 and not the same; OPTIONS is one argument,
# the options of the compiler given to each run of Simplify, which may be
# empty; LABELS is a file of lines "UNIT LABEL" (no unit has a name that
# begins with "#", so such a line is a comment); COUNTS is a file that the
# script writes.
#
# First, two lines of control, one for K1 and one for K2: whether eight
# variables, and eight continuations, made after K variables and K
# continuations have been made and thrown away ("K = ...") are held in
# another order than with none thrown away.
#
# Then, for each unit, in the order of the names: Simplify is run three
# times, each in a process of its own, with K = 0, K1 and K2: so many
# variables and continuations are made and thrown away before the text is
# converted. The result of the first run is compared with itself, read
# twice, and then with each of the two others, each comparison in a process
# of its own. The unit has a line, its name and then:
#
# - "the same": the three runs gave a result, and the three comparisons said
#   so;
# - "differs: WHAT": the two comparisons with the other results said so, and
#   named the same thing as the first that they found to differ;
# - "with K = K1, ...; with K = K2, ...": those two comparisons did not
#   print the same line;
# - "not compared: the result with K = 0, read twice, differs: WHAT": the
#   comparison found the first result to differ from itself, so it is not
#   asked about the others;
# - "not compared: WHY": the reader refused the text of a result, or the
#   comparison stopped, or its exit status is not that of its line;
# - "not read": the reader refused the text of the unit, in the three runs;
# - "not simplified": Simplify stopped, in the three runs;
# - "not written": the printer of texts stopped on the result, in the three
#   runs;
# - "not run: the program exits with S": the three runs ended with the same
#   status S, which is none of those (2 if it refuses an option);
# - "differs: the runs of Simplify exit with S0 S1 S2": the three runs did
#   not end with the same status (0: a result; 3: not read; 4: Simplify
#   stopped; 6: not written).
#
# A status of a run of Simplify is written with "?" after it if the last line
# that the run printed is not the line that the program prints with that
# status; such a run is not taken for one that ended with the status.
#
# If LABELS has a line for the unit, its label is put after the line, in
# brackets, whatever the line says.
#
# The line of a unit is printed unless it is "the same" and no more. After
# the lines of the units comes one line, "summary: ...", with the number of
# the lines printed of each kind: "differ" (the lines "differs: ..." and
# "with K = ..."), "not compared", "not simplified", and the others together
# ("not read", "not written", "not run: ...", and "the same" with a label);
# and it says that each of the other units that the script was given was run
# and says "the same", or, if the script did not run as many units as it was
# given, the two numbers. So a new unit whose line is "the same" changes
# nothing of what is printed. The number of the units given and of those
# that say "the same" are written to COUNTS, one line, which is compared
# with nothing.
#
# The exit status is 0 whatever the lines say: they are compared with the
# expected file.

set -u

tool=$1
first=$2
second=$3
options=$4
labels=$5
counts=$6
shift 6

case "$tool" in
/*) ;;
*) tool="$PWD/$tool" ;;
esac

if [ "$first" -eq 0 ] || [ "$second" -eq 0 ] || [ "$first" -eq "$second" ]
then
  echo "run.sh: the two numbers are to be other than 0 and not the same" >&2
  exit 2
fi

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
mkdir "$work/0" "$work/$first" "$work/$second"

# The line "$2: ..." of the file $1, which a run of the program with -order
# wrote, if the run exited with 0 (if not, the file is empty), the file is
# two lines and no more (a last line that does not end is a line), one of
# them is that line, and the line has the numbers 0 to 7, each once; nothing
# if not.
order_of() {
  [ "$(grep -c '' "$1")" -eq 2 ] || return 0
  [ "$(grep -c "^$2: " "$1")" -eq 1 ] || return 0
  order=$(grep "^$2: " "$1")
  sorted=$(printf '%s' "${order#"$2: "}" | tr ' ' '\n' | sort -n | tr '\n' ' ')
  [ "$sorted" = "0 1 2 3 4 5 6 7 " ] || return 0
  echo "$order"
}

# The control: the order of names made after some are made and thrown away.
"$tool" -order 0 >"$work/order" 2>/dev/null || : >"$work/order"
for burnt in "$first" "$second"; do
  "$tool" -order "$burnt" >"$work/order.$burnt" 2>/dev/null ||
    : >"$work/order.$burnt"
  line="control, K = $burnt:"
  for what in variables continuations; do
    one=$(order_of "$work/order" "$what")
    two=$(order_of "$work/order.$burnt" "$what")
    if [ -z "$one" ] || [ -z "$two" ]; then
      line="$line the order of the $what is not known;"
    elif [ "$one" = "$two" ]; then
      line="$line $what in the same order;"
    else
      line="$line $what in another order;"
    fi
  done
  echo "${line%;}"
done

# The line that the comparison prints of the result with K = 0 and the
# result with K = $2, of the unit $1, if it is one line and the exit status
# is the one that the program gives with that line (0 with "the same", 1 with
# "differs: ...", 2 with "not compared: ..."); a line that says so if not.
newline='
'
compare() {
  line=$("$tool" -compare "$work/0/$1.fl" "$work/$2/$1.fl" 2>/dev/null)
  status=$?
  case "$status:$line" in
  *"$newline"*) line= ;;
  "0:the same" | "1:differs: "* | "2:not compared: "*) ;;
  *) line= ;;
  esac
  if [ -n "$line" ]; then
    echo "$line"
  else
    echo "not compared: the comparison exits with $status," \
      "and its line is not the line of that status"
  fi
}

given=$#
run=0
same=0
differ=0
not_compared=0
not_simplified=0
others=0

for file in $(printf '%s\n' "$@" | LC_ALL=C sort); do
  run=$((run + 1))
  name=$(basename "$file" .fl)
  statuses=
  for burnt in 0 "$first" "$second"; do
    # $options is split into words on purpose.
    # shellcheck disable=SC2086
    "$tool" -simplify-to "$work/$burnt/$name.fl" -burn "$burnt" $options \
      "$file" >"$work/line" 2>/dev/null
    status=$?
    # A status that the program gives with a line is held to that line.
    case "$status:$(tail -n 1 "$work/line")" in
    "0:simplified" | "3:not read: "* | "4:not simplified: "*) ;;
    "6:not written: "*) ;;
    [0346]:*) status="$status?" ;;
    esac
    statuses="$statuses $status"
  done
  case "$statuses" in
  " 0 0 0")
    itself=$(compare "$name" 0)
    case "$itself" in
    "the same")
      one=$(compare "$name" "$first")
      two=$(compare "$name" "$second")
      if [ "$one" = "$two" ]; then
        line=$one
      else
        line="with K = $first, $one; with K = $second, $two"
      fi
      ;;
    "differs: "*)
      line="not compared: the result with K = 0, read twice, $itself"
      ;;
    *) line=$itself ;;
    esac
    ;;
  " 3 3 3") line="not read" ;;
  " 4 4 4") line="not simplified" ;;
  " 6 6 6") line="not written" ;;
  *)
    # shellcheck disable=SC2086
    set -- $statuses
    if [ "$1" = "$2" ] && [ "$1" = "$3" ]; then
      line="not run: the program exits with $1"
    else
      line="differs: the runs of Simplify exit with$statuses"
    fi
    ;;
  esac
  label=$(sed -n "s/^$name //p" "$labels")
  if [ -n "$label" ]; then
    line="$line [$label]"
  fi
  # Every line that is printed is counted in one kind, and no other is.
  case "$line" in
  "the same") same=$((same + 1)) ;;
  "differs: "* | "with K = "*) differ=$((differ + 1)) ;;
  "not compared: "*) not_compared=$((not_compared + 1)) ;;
  "not simplified" | "not simplified ["*)
    not_simplified=$((not_simplified + 1))
    ;;
  *) others=$((others + 1)) ;;
  esac
  if [ "$line" != "the same" ]; then
    echo "$name: $line"
  fi
done

kinds="differ $differ; not compared $not_compared;"
kinds="$kinds not simplified $not_simplified; others $others"
if [ "$given" -eq "$run" ]; then
  echo "summary: $kinds; each other unit given was run and says the same"
else
  echo "summary: $kinds; BUT $given units were given and $run were run"
fi
echo "$given units given, $run run: $same the same; $kinds" >"$counts"
