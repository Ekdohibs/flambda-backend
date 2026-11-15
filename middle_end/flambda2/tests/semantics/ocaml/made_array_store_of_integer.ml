(* A mutable array of integers is made, an element is stored without a check of
   the index and two are read. The front end writes the store as one into an
   array of values that the collector may ignore; Simplify, which knows from the
   allocation that the elements are integers, writes it as one into an array of
   immediates. What is stored, and how, is the same. *)

external opaque : 'a -> 'a = "%opaque"

external ( + ) : int -> int -> int = "%addint"

external unsafe_get : 'a array -> int -> 'a = "%array_unsafe_get"

external unsafe_set : 'a array -> int -> 'a -> unit = "%array_unsafe_set"

let result =
  let a = [| opaque 1; opaque 2 |] in
  unsafe_set a 0 (opaque 7);
  unsafe_get a 0 + unsafe_get a 1
