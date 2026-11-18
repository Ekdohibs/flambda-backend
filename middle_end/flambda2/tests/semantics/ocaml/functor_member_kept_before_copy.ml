(* A functor that is inlined at its application, and a member of it, [put], that
   Simplify does not inline where it is called. Simplify gives the closures of
   the inlined body new code, and the term after calls [put] by its new code.
   The term before has those closures with the old code until a lifting of the
   validation copies the code under the new code ids, and the rounds that come
   before that lifting make the call of [put] direct, by the old code. The test
   tool tells the validation to leave the calls of the code that the term after
   calls and of the code that it is a newer version of, so the call stays, as it
   does in the term after. Told the newer code alone, the validation inlined the
   call; told the recursive code alone, it inlines [or_zero] on both sides, and
   the two terms then differ. functor_member_kept_before_copy_control.ml is this
   unit without [or_zero]. *)

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

let or_zero o = match o with Some v -> v | None -> 0

let result = or_zero (T.lookup 1 (T.put 1 10 []))
