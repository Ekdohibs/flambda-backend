(* A control of what is left out: one of the seven comparisons of which Simplify
   makes a primitive where it knows both arguments to be immediates
   (simplify/simplify_extcall.ml). [generic] calls the C function caml_compare;
   inlined in [ints], whose arguments are integers, the call becomes a primitive
   in the term after. Nothing in the model relates the call to the primitive,
   and the unit is not validated. *)

external compare : 'a -> 'a -> int = "%compare"

let[@inline] generic a b = compare a b

let ints (x : int) (y : int) = generic x y
