(* The characters of a string that are a given one, counted by a [for] loop
   whose bound is the length of the string. The read [s.[i]] in the loop checks
   its index, and the check takes the length of the string again. Simplify has
   it once: the handler of the loop is simplified with what is known where the
   loop is defined, the length taken for the bound among it. The sharing of
   primitives of the validation carries what it knows into the handler of a
   recursive continuation in the same way, and the unit is validated with its
   imports. The unit is T1's, written for that. *)

external ( + ) : int -> int -> int = "%addint"

external ( - ) : int -> int -> int = "%subint"

external ( = ) : 'a -> 'a -> bool = "%equal"

type 'a ref = { mutable contents : 'a }

external ref : 'a -> 'a ref = "%makemutable"

external ( ! ) : 'a ref -> 'a = "%field0"

external incr : int ref -> unit = "%incr"

external length : string -> int = "%string_length"

external get : string -> int -> char = "%string_safe_get"

let count (c : char) (s : string) =
  let n = ref 0 in
  for i = 0 to length s - 1 do
    if get s i = c then incr n
  done;
  !n

let result = count 'a' "banana"
