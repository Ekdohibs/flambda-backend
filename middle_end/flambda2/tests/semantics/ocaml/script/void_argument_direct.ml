(* A function with a parameter of no kind, called directly: from_ocaml.sh
   makes its text (void_argument_direct.says). *)

external opaque : 'a -> 'a = "%opaque"

let[@inline never] h (c : int) #() = c

let result = h (opaque 5) #()
