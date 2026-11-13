open BinNums
open Datatypes

module Pos = struct
  (** val succ : positive -> positive **)

  let rec succ = function
    | Coq_xI p -> Coq_xO (succ p)
    | Coq_xO p -> Coq_xI p
    | Coq_xH -> Coq_xO Coq_xH

  (** val compare_cont : comparison -> positive -> positive -> comparison **)

  let rec compare_cont r x y =
    match x with
    | Coq_xI p -> (
      match y with
      | Coq_xI q -> compare_cont r p q
      | Coq_xO q -> compare_cont Gt p q
      | Coq_xH -> Gt)
    | Coq_xO p -> (
      match y with
      | Coq_xI q -> compare_cont Lt p q
      | Coq_xO q -> compare_cont r p q
      | Coq_xH -> Gt)
    | Coq_xH -> ( match y with Coq_xH -> r | _ -> Lt)

  (** val compare : positive -> positive -> comparison **)

  let compare = compare_cont Eq

  (** val eqb : positive -> positive -> bool **)

  let rec eqb p q =
    match p with
    | Coq_xI p0 -> ( match q with Coq_xI q0 -> eqb p0 q0 | _ -> false)
    | Coq_xO p0 -> ( match q with Coq_xO q0 -> eqb p0 q0 | _ -> false)
    | Coq_xH -> ( match q with Coq_xH -> true | _ -> false)

  (** val leb : positive -> positive -> bool **)

  let leb x y = match compare x y with Gt -> false | _ -> true

  (** val max : positive -> positive -> positive **)

  let max p p' = match compare p p' with Gt -> p | _ -> p'

  (** val ltb : positive -> positive -> bool **)

  let ltb x y = match compare x y with Lt -> true | _ -> false

  (** val eq_dec : positive -> positive -> bool **)

  let rec eq_dec p x0 =
    match p with
    | Coq_xI p0 -> ( match x0 with Coq_xI p1 -> eq_dec p0 p1 | _ -> false)
    | Coq_xO p0 -> ( match x0 with Coq_xO p1 -> eq_dec p0 p1 | _ -> false)
    | Coq_xH -> ( match x0 with Coq_xH -> true | _ -> false)
end
