open Datatypes

val map : ('a1 -> 'a2) -> 'a1 list -> 'a2 list

val seq : nat -> nat -> nat list

val repeat : 'a1 -> nat -> 'a1 list

val firstn : nat -> 'a1 list -> 'a1 list

val skipn : nat -> 'a1 list -> 'a1 list
