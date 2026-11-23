(* With -Oclassic a direct call of a function that does not use its closure has
   no callee. The text writes the arities of such a call with a blank in the
   place of the callee ([(_ : imm tagged -> imm tagged)]); it wrote none, and
   the call was read with one value for each argument and one for the result. So
   the text is the term. *)

external ( + ) : int -> int -> int = "%addint"

let[@inline never] f (x : int) = x + 1

let g (n : int) = f n
