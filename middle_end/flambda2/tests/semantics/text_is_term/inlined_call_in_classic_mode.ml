(* With -Oclassic the calls of [f] are inlined before the term is given to
   Simplify, and the body that stands in the place of each call begins with a
   mark, which has the inlining attribute of the call and debugging information:
   a stack, which is not empty. The text writes the attribute
   ([%inlined_apply.always] for the call marked [@inlined]) and not what the
   stack holds, which the comparison does not compare; the reader gives the mark
   a stack that is not empty, and the comparison compares whether it is. The
   text wrote no attribute, the reader gave the empty stack, and the comparison
   compared the attribute and all of the stack. So the text is the term. *)

external ( + ) : int -> int -> int = "%addint"

let f (x : int) = x + 1

let g (n : int) = f n

let h (n : int) = (f [@inlined]) n
