open Datatypes
open Nat

val hd : 'a1 -> 'a1 list -> 'a1

val tl : 'a1 list -> 'a1 list

val nth_error : 'a1 list -> nat -> 'a1 option

val rev : 'a1 list -> 'a1 list

val concat : 'a1 list list -> 'a1 list

val list_eq_dec : ('a1 -> 'a1 -> bool) -> 'a1 list -> 'a1 list -> bool

val flat_map : ('a1 -> 'a2 list) -> 'a1 list -> 'a2 list

val fold_left : ('a1 -> 'a2 -> 'a1) -> 'a2 list -> 'a1 -> 'a1

val fold_right : ('a2 -> 'a1 -> 'a1) -> 'a1 -> 'a2 list -> 'a1

val existsb : ('a1 -> bool) -> 'a1 list -> bool

val forallb : ('a1 -> bool) -> 'a1 list -> bool

val filter : ('a1 -> bool) -> 'a1 list -> 'a1 list

val find : ('a1 -> bool) -> 'a1 list -> 'a1 option

val combine : 'a1 list -> 'a2 list -> ('a1 * 'a2) list

val list_sum : nat list -> nat
