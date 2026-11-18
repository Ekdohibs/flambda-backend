open Datatypes

(** val map : ('a1 -> 'a2) -> 'a1 list -> 'a2 list **)

let rec map f = function [] -> [] | a :: l0 -> f a :: map f l0

(** val seq : nat -> nat -> nat list **)

let rec seq start = function O -> [] | S len0 -> start :: seq (S start) len0

(** val repeat : 'a1 -> nat -> 'a1 list **)

let rec repeat x = function O -> [] | S k -> x :: repeat x k

(** val firstn : nat -> 'a1 list -> 'a1 list **)

let rec firstn n l =
  match n with
  | O -> []
  | S n0 -> ( match l with [] -> [] | a :: l0 -> a :: firstn n0 l0)

(** val skipn : nat -> 'a1 list -> 'a1 list **)

let rec skipn n l =
  match n with
  | O -> l
  | S n0 -> ( match l with [] -> [] | _ :: l0 -> skipn n0 l0)
