open Datatypes
open Nat

(** val hd : 'a1 -> 'a1 list -> 'a1 **)

let hd default = function [] -> default | x :: _ -> x

(** val tl : 'a1 list -> 'a1 list **)

let tl = function [] -> [] | _ :: l' -> l'

(** val nth_error : 'a1 list -> nat -> 'a1 option **)

let rec nth_error l = function
  | O -> ( match l with [] -> None | x :: _ -> Some x)
  | S n0 -> ( match l with [] -> None | _ :: l' -> nth_error l' n0)

(** val rev : 'a1 list -> 'a1 list **)

let rec rev = function [] -> [] | x :: l' -> app (rev l') (x :: [])

(** val concat : 'a1 list list -> 'a1 list **)

let rec concat = function [] -> [] | x :: l0 -> app x (concat l0)

(** val list_eq_dec : ('a1 -> 'a1 -> bool) -> 'a1 list -> 'a1 list -> bool **)

let rec list_eq_dec eq_dec l l' =
  match l with
  | [] -> ( match l' with [] -> true | _ :: _ -> false)
  | a :: l0 -> (
    match l' with
    | [] -> false
    | a0 :: l1 -> if eq_dec a a0 then list_eq_dec eq_dec l0 l1 else false)

(** val flat_map : ('a1 -> 'a2 list) -> 'a1 list -> 'a2 list **)

let rec flat_map f = function [] -> [] | x :: l0 -> app (f x) (flat_map f l0)

(** val fold_left : ('a1 -> 'a2 -> 'a1) -> 'a2 list -> 'a1 -> 'a1 **)

let rec fold_left f l a0 =
  match l with [] -> a0 | b :: l0 -> fold_left f l0 (f a0 b)

(** val fold_right : ('a2 -> 'a1 -> 'a1) -> 'a1 -> 'a2 list -> 'a1 **)

let rec fold_right f a0 = function
  | [] -> a0
  | b :: l0 -> f b (fold_right f a0 l0)

(** val existsb : ('a1 -> bool) -> 'a1 list -> bool **)

let rec existsb f = function [] -> false | a :: l0 -> f a || existsb f l0

(** val forallb : ('a1 -> bool) -> 'a1 list -> bool **)

let rec forallb f = function [] -> true | a :: l0 -> f a && forallb f l0

(** val filter : ('a1 -> bool) -> 'a1 list -> 'a1 list **)

let rec filter f = function
  | [] -> []
  | x :: l0 -> if f x then x :: filter f l0 else filter f l0

(** val find : ('a1 -> bool) -> 'a1 list -> 'a1 option **)

let rec find f = function
  | [] -> None
  | x :: tl0 -> if f x then Some x else find f tl0

(** val combine : 'a1 list -> 'a2 list -> ('a1 * 'a2) list **)

let rec combine l l' =
  match l with
  | [] -> []
  | x :: tl0 -> (
    match l' with [] -> [] | y :: tl' -> (x, y) :: combine tl0 tl')

(** val list_sum : nat list -> nat **)

let list_sum l = fold_right add O l
