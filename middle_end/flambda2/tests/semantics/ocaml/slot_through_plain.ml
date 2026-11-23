(* [u] is a parameter of [make], so it stays a variable. [t] captures [u]; [w]
   captures [t] and [u], calls [t] and calls [u] itself. Where Simplify inlines
   [t] into the code of [w], the read of [u] in the copy is a projection from
   the closure that [w] holds in a value slot; Simplify makes it the read of the
   value slot of [w] that holds [u]. The rounds that use what [my_closure] is
   known to be from its code make the same read
   (FlambdaSlotsThroughSpec.read_slots_through): every set of the unit with the
   code of [w] gave the two value slots the one variable. It is the shape of
   [union], [try_concat] and [filter_map] in stdlib/set.ml. [make] is called
   through [opaque], so that the unit's own run enters the code of [w].

   [make], [t] and [w] are a source that V3 wrote for the measure of the rewrite
   named above, of the shape of that rewrite. By V3's record, what it read for
   the source is a file of the compiler,
   middle_end/flambda2/from_lambda/closure_conversion.ml ([close_one_function]),
   and the record names no function of the library. C3 wrote the rewrite for
   what the validation met in the term of [filter_map] of stdlib/set.ml of this
   repository (lines 546 to 559, with [try_concat], lines 540 to 544, which it
   calls; the file has both a second time further down). V3 found that on the
   term that the test tool prints, and had the text of the function before it
   that evening: that is why this comment names those functions, and it is no
   record of how the source was written. What the unit shares with those
   functions is one thing of their shape: a function calls another function and
   uses a value that the other uses too. C3 added the declaration of [opaque],
   [next] and [result] for this unit. Of the names of the code, [t] alone is a
   name in those functions, of a tree and of no function; the types and the
   bodies are not the library's (the library's functions rebuild a tree; these
   add integers). No line of those functions is here. The operator [+] and
   [opaque] are declared as many units of the tests declare them; this comment
   names no file of the library for them. The line numbers are those of the file
   when this comment was written. *)

external ( + ) : int -> int -> int = "%addint"

external opaque : 'a -> 'a = "%opaque"

let make (u : int -> int) =
  let[@inline] t x = u x + 1 in
  let w x = t x + u x in
  w

let next x = x + 1

let result = (opaque make) next 3
