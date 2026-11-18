(* functor_member_kept_before_copy.ml without the function [or_zero] that takes
   the result: the control of that unit. The validation is told the recursive
   code alone, inlines [put] on both sides, and the two terms are the same. It
   was validated so before the test tool gave the validation the older versions
   of the code that the term after calls, and it is validated with them. *)

external ( = ) : int -> int -> bool = "%equal"

module type KEY = sig
  type t

  val same : t -> t -> bool
end

module Table (K : KEY) = struct
  let rec lookup k t =
    match t with
    | [] -> None
    | (k', v) :: rest -> if K.same k k' then Some v else lookup k rest

  let rec without k t =
    match t with
    | [] -> []
    | ((k', _) as entry) :: rest ->
      if K.same k k' then rest else entry :: without k rest

  let put k v t = (k, v) :: without k t
end

module T = Table (struct
  type t = int

  let same a b = a = b
end)

let result = T.lookup 1 (T.put 1 10 [])
