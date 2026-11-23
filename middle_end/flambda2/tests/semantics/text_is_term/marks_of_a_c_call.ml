(* Four calls of C functions: of one that the compiler may take for a builtin,
   of one that has no effects, of one that has no coeffects, and of one whose
   only effects are those of an allocation. The text says each of these of the
   call ([ccall] with [builtin], [no_effects], [no_coeffects] and
   [only_generative_effects mutable]); it said none, and every C call was read
   as one of a function that is no builtin and has effects and coeffects. So the
   text is the term. *)

external builtin : int -> int = "c_builtin" [@@builtin]

external without_effects : int -> int = "c_without_effects" [@@no_effects]

external without_coeffects : int -> int = "c_without_coeffects" [@@no_coeffects]

external only_allocates : int -> int = "c_only_allocates"
[@@only_generative_effects]

let f (x : int) =
  builtin (without_effects (without_coeffects (only_allocates x)))
