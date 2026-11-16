#!/bin/sh

# Compares what the driver of the compiler's own run prints of a unit
# (own_run.sh) with the expected file of the unit, which has what the tool
# prints of the text of the unit. See README.md, "The compiler's own run".
#
#   own_same.sh <X.expected> <X.own_output>
#
# The two files must be the same bytes.
#
#   own_same.sh numbers <X.expected> <X.own_output>
#
# For a unit that own_run_numbers.txt names. Each of the two files must be
# one line, of the form
#
#   X.fl: not checked: the model has not got primitive (...)
#
# with no byte after the newline of that line, no null byte, and no "/#",
# which is what is put below in the place of a number.
#
# The tool and the driver print a variable as its name, a slash, the number
# that their process gave it, and UV if the variable is user-visible or N if
# it is not. In each of the two lines, a word that stands between a space or
# a parenthesis and a space or a parenthesis, and that is a name (letters,
# digits, underscores and quotes, of which the first is a letter or an
# underscore), a slash, digits, and UV or N, has "/#" put in the place of
# its slash, its digits and those letters: prim/37UV and prim/29N both
# become prim/#. Nothing else of a line is changed, and the two lines must
# then be the same bytes. This is a test on the text of the line and no
# parser: it takes such a word for a variable with its number, and leaves a
# variable alone whose name has any other character. The two files must not
# be the same bytes as they are: the list is for the units that need it.
# The locale is set to C, so that a letter is one of the 52 of ASCII and
# every byte is a character.
#
# Where the files are not as they must be, it says so, prints the lines
# that differ where that is what is wrong (those of X.expected after <,
# those of the own run after >), and exits with 1.

set -eu

LC_ALL=C
export LC_ALL

numbers=
if [ "${1-}" = numbers ]; then
  numbers=yes
  shift
fi

if [ $# -ne 2 ]; then
  echo "usage: $0 [numbers] <X.expected> <X.own_output>" >&2
  exit 2
fi

expected=$1
output=$2
name=$(basename "$expected" .expected)

if [ -z "$numbers" ]; then
  if cmp -s "$expected" "$output"; then
    exit 0
  fi
  echo "$name: the compiler's own run does not print what $name.expected"
  echo "has, which is what the tool prints of the text of the unit:"
  diff "$expected" "$output" || true
  echo "README.md, \"The compiler's own run\", says what this may mean."
  exit 1
fi

if cmp -s "$expected" "$output"; then
  echo "$name: own_run_numbers.txt names the unit, and the compiler's own"
  echo "run prints what $name.expected has, byte for byte: the unit is to"
  echo "be taken out of the list."
  exit 1
fi

# The name of the unit goes into a pattern below: it must have no
# character that means something there.
case "$name" in
"" | *[!A-Za-z0-9_\']*)
  echo "$name: own_run_numbers.txt names the unit, and its name has a"
  echo "character that is not a letter, a digit, an underscore or a quote."
  exit 1
  ;;
esac

# The one line that each file must be. A file is one line if it has no null
# byte, has one newline, and is its own first line: so nothing comes after
# that newline.
form="^$name\\.fl: not checked: the model has not got primitive (.*)\$"
one_line() {
  tr -d '\000' <"$1" | cmp -s - "$1" &&
    [ "$(wc -l <"$1")" -eq 1 ] &&
    head -n 1 "$1" | cmp -s - "$1"
}
for file in "$expected" "$output"; do
  if ! one_line "$file" || ! grep -q "$form" "$file" ||
    grep -q '/#' "$file"; then
    echo "$name: own_run_numbers.txt names the unit, and one of"
    echo "$name.expected and what the compiler's own run prints is not one"
    echo "line of the form"
    echo "  $name.fl: not checked: the model has not got primitive (...)"
    echo "without \"/#\" in it."
    echo "$name.expected:"
    cat "$expected"
    echo "the compiler's own run:"
    cat "$output"
    exit 1
  fi
done

# The words of a line that are a name, a slash, digits and UV or N, between
# a space or a parenthesis and a space or a parenthesis: "/#" for the
# slash, the digits and the letters. One word at a time, as the space
# after one word is the space before the next.
word="\\([ (]\\)\\([A-Za-z_][A-Za-z0-9_']*\\)/[0-9][0-9]*"
without_numbers() {
  sed -e ':uv' -e "s|${word}UV\\([ )]\\)|\\1\\2/#\\3|" -e 't uv' \
    -e ':n' -e "s|${word}N\\([ )]\\)|\\1\\2/#\\3|" -e 't n' "$1"
}

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
without_numbers "$expected" >"$work/expected"
without_numbers "$output" >"$work/output"
if cmp -s "$work/expected" "$work/output"; then
  exit 0
fi
echo "$name: the compiler's own run does not print what $name.expected"
echo "has, which is what the tool prints of the text of the unit, even"
echo "with the numbers of the variables of the line left out of both:"
diff "$work/expected" "$work/output" || true
echo "README.md, \"The compiler's own run\", says what this may mean."
exit 1
