(**************************************************************************)
(*                                                                        *)
(*                                 OCaml                                  *)
(*                                                                        *)
(*                  Liam Stevenson, Jane Street, New York                 *)
(*                                                                        *)
(*   Copyright 2024 Jane Street Group LLC                                 *)
(*                                                                        *)
(*   All rights reserved.  This file is distributed under the terms of    *)
(*   the GNU Lesser General Public License version 2.1, with the          *)
(*   special exception on linking described in the file LICENSE.          *)
(*                                                                        *)
(**************************************************************************)

module Sort = struct
  type base =
    | Void
    | Value
    | Float64
    | Float32
    | Word
    | Bits32
    | Bits64

  type const =
    | Const_base of base
    | Const_product of const list

  type t =
    | Var of var
    | Base of base
    | Product of t list

  and var =
    { mutable contents : t option;
      uid : int
    }

  let equal_base b1 b2 =
    match b1, b2 with
    | Void, Void
    | Value, Value
    | Float64, Float64
    | Float32, Float32
    | Word, Word
    | Bits32, Bits32
    | Bits64, Bits64 ->
      true
    | (Void | Value | Float64 | Float32 | Word | Bits32 | Bits64), _ -> false

  let to_string_base = function
    | Value -> "value"
    | Void -> "void"
    | Float64 -> "float64"
    | Float32 -> "float32"
    | Word -> "word"
    | Bits32 -> "bits32"
    | Bits64 -> "bits64"

  module Const = struct
    type t = const =
      | Const_base of base
      | Const_product of const list

    let rec equal c1 c2 =
      match c1, c2 with
      | Const_base b1, Const_base b2 -> equal_base b1 b2
      | Const_product cs1, Const_product cs2 -> List.equal equal cs1 cs2
      | (Const_base _ | Const_product _), _ -> false

    (* let rec to_string = function
     *   | Const_base b -> to_string_base b
     *   | Const_product cs -> String.concat " * " (List.map to_string cs) *)

    let format ppf c =
      let rec format nested ppf = function
        | Const_base b -> Format.fprintf ppf "%s" (to_string_base b)
        | Const_product [] ->
          Misc.fatal_error "Jkind_types.Sort.Const.format: empty product"
        | Const_product cs ->
          Format.fprintf ppf "@[%a@]"
            (Misc.pp_parens_if nested
               (Format.pp_print_list
                  ~pp_sep:(fun ppf () -> Format.fprintf ppf "@ & ")
                  (format true)))
            cs
      in
      format false ppf c
  end

  module Var = struct
    type t = var

    let name { uid; _ } = "'_representable_layout_" ^ Int.to_string uid
  end

  module Debug_printers = struct
    open Format

    let rec t ppf = function
      | Var v -> fprintf ppf "Var %a" var v
      | Base b ->
        fprintf ppf
          (match b with
          | Void -> "Void"
          | Value -> "Value"
          | Float64 -> "Float64"
          | Float32 -> "Float32"
          | Word -> "Word"
          | Bits32 -> "Bits32"
          | Bits64 -> "Bits64")
      | Product ts ->
        fprintf ppf "Product [ %a ]"
          (pp_print_list ~pp_sep:(fun ppf () -> pp_print_text ppf "; ") t)
          ts

    and opt_t ppf = function
      | Some s -> fprintf ppf "Some %a" t s
      | None -> fprintf ppf "None"

    and var ppf v =
      fprintf ppf "{@[@ contents = %a;@ uid = %d@ @]}" opt_t v.contents v.uid
  end

  (* To record changes to sorts, for use with `Types.{snapshot, backtrack}` *)
  type change = var * t option

  let change_log : (change -> unit) ref = ref (fun _ -> ())

  let set_change_log cl = change_log := cl

  let log_change change = !change_log change

  let undo_change (v, t_op) = v.contents <- t_op

  let set : var -> t option -> unit =
   fun v t_op ->
    log_change (v, v.contents);
    v.contents <- t_op

  module Static = struct
    (* Statically allocated values of various consts and sorts to save
       allocations in in the core hot path functions. [T] is also included in
       the outer module to provide the core sorts. *)

    module T = struct
      let void = Base Void

      let value = Base Value

      let float64 = Base Float64

      let float32 = Base Float32

      let word = Base Word

      let bits32 = Base Bits32

      let bits64 = Base Bits64

      let of_base = function
        | Void -> void
        | Value -> value
        | Float64 -> float64
        | Float32 -> float32
        | Word -> word
        | Bits32 -> bits32
        | Bits64 -> bits64

      let rec of_const = function
        | Const_base b -> of_base b
        | Const_product cs -> Product (List.map of_const cs)
    end

    module Const = struct
      let value = Const_base Value

      let void = Const_base Void

      let float64 = Const_base Float64

      let float32 = Const_base Float32

      let word = Const_base Word

      let bits32 = Const_base Bits32

      let bits64 = Const_base Bits64

      let of_base : base -> const = function
        | Value -> value
        | Void -> void
        | Float64 -> float64
        | Float32 -> float32
        | Word -> word
        | Bits32 -> bits32
        | Bits64 -> bits64
    end

    module T_option = struct
      let value = Some T.value

      let void = Some T.void

      let float64 = Some T.float64

      let float32 = Some T.float32

      let word = Some T.word

      let bits32 = Some T.bits32

      let bits64 = Some T.bits64

      let of_base = function
        | Void -> void
        | Value -> value
        | Float64 -> float64
        | Float32 -> float32
        | Word -> word
        | Bits32 -> bits32
        | Bits64 -> bits64

      let rec of_const = function
        | Const_base b -> of_base b
        | Const_product cs ->
          Option.map
            (fun x -> Product x)
            (Misc.Stdlib.List.map_option of_const cs)
    end
  end

  let of_var v = Var v

  let last_var_uid = ref 0

  let new_var () =
    incr last_var_uid;
    Var { contents = None; uid = !last_var_uid }

  (* Post-condition: If the result is a [Var v], then [!v] is [None]. *)
  let rec get : t -> t = function
    | Base _ as t -> t
    | Product ts as t ->
      let ts' = List.map get ts in
      if List.for_all2 ( == ) ts ts' then t else Product ts'
    | Var r as t -> (
      match r.contents with
      | None -> t
      | Some s ->
        let result = get s in
        if result != s then set r (Some result);
        (* path compression *)
        result)

  let rec default_to_value_and_get : t -> const = function
    | Base b -> Static.Const.of_base b
    | Product ts -> Const_product (List.map default_to_value_and_get ts)
    | Var r -> (
      match r.contents with
      | None ->
        set r Static.T_option.value;
        Static.Const.value
      | Some s ->
        let result = default_to_value_and_get s in
        set r (Static.T_option.of_const result);
        (* path compression *)
        result)

  (* XXX unused?  jkind.ml has one *)
  let default_to_value t = ignore (default_to_value_and_get t)

  (* let get_opt t =
   *   match get t with
   *   | Base b -> Some (Static.Const.of_base b)
   *   | Product ts ->
   *     Option.map
   *       (fun x -> Const_product x)
   *       (Misc.Stdlib.List.map_option get_opt ts)
   *   | Var _ -> None *)

  (***********************)
  (* equality *)

  type equate_result =
    | Unequal
    | Equal_mutated_first
    | Equal_mutated_second
    | Equal_mutated_both
    | Equal_no_mutation

  let swap_equate_result = function
    | Equal_mutated_first -> Equal_mutated_second
    | Equal_mutated_second -> Equal_mutated_first
    | (Unequal | Equal_no_mutation | Equal_mutated_both) as r -> r

  let[@inline] sorts_of_product s =
    (* In the equate functions, it's useful to pass around lists of sorts inside
       the product constructor they came from to avoid re-allocating it if we
       end up wanting to store it in a variable. We could probably eliminate the
       use of this by collapsing a bunch of the functions below into each other,
       but that would be much less readable. *)
    match s with
    | Product sorts -> sorts
    | Var _ | Base _ -> Misc.fatal_error "Jkind_types.sorts_of_product"

  let rec equate_sort_sort s1 s2 =
    match s1 with
    | Base b1 -> swap_equate_result (equate_sort_base s2 b1)
    | Var v1 -> equate_var_sort v1 s2
    | Product _ -> swap_equate_result (equate_sort_product s2 s1)

  and equate_sort_base s1 b2 =
    match s1 with
    | Base b1 -> if equal_base b1 b2 then Equal_no_mutation else Unequal
    | Var v1 -> equate_var_base v1 b2
    | Product _ -> Unequal

  and equate_var_base v1 b2 =
    match v1.contents with
    | Some s1 -> equate_sort_base s1 b2
    | None ->
      set v1 (Static.T_option.of_base b2);
      Equal_mutated_first

  and equate_var_sort v1 s2 =
    match s2 with
    | Base b2 -> equate_var_base v1 b2
    | Var v2 -> equate_var_var v1 v2
    | Product _ -> equate_var_product v1 s2

  and equate_var_var v1 v2 =
    if v1 == v2
    then Equal_no_mutation
    else
      match v1.contents, v2.contents with
      | Some s1, _ -> swap_equate_result (equate_var_sort v2 s1)
      | _, Some s2 -> equate_var_sort v1 s2
      | None, None ->
        set v1 (Some (of_var v2));
        Equal_mutated_first

  and equate_var_product v1 s2 =
    match v1.contents with
    | Some s1 -> equate_sort_product s1 s2
    | None ->
      set v1 (Some s2);
      Equal_mutated_first

  and equate_sort_product s1 s2 =
    match s1 with
    | Base _ -> Unequal
    | Product sorts1 ->
      let sorts2 = sorts_of_product s2 in
      equate_sorts sorts1 sorts2
    | Var v1 -> equate_var_product v1 s2

  and equate_sorts sorts1 sorts2 =
    let rec go sorts1 sorts2 acc =
      match sorts1, sorts2 with
      | [], [] -> acc
      | sort1 :: sorts1, sort2 :: sorts2 -> (
        match equate_sort_sort sort1 sort2, acc with
        | Unequal, _ -> Unequal
        | _, Unequal -> assert false
        | Equal_no_mutation, acc | acc, Equal_no_mutation ->
          go sorts1 sorts2 acc
        | Equal_mutated_both, _ | _, Equal_mutated_both ->
          go sorts1 sorts2 Equal_mutated_both
        | Equal_mutated_first, Equal_mutated_first ->
          go sorts1 sorts2 Equal_mutated_first
        | Equal_mutated_second, Equal_mutated_second ->
          go sorts1 sorts2 Equal_mutated_second
        | Equal_mutated_first, Equal_mutated_second
        | Equal_mutated_second, Equal_mutated_first ->
          go sorts1 sorts2 Equal_mutated_both)
      | _, _ -> assert false
    in
    if List.compare_lengths sorts1 sorts2 = 0
    then go sorts1 sorts2 Equal_no_mutation
    else Unequal

  let equate_tracking_mutation = equate_sort_sort

  (* Don't expose whether or not mutation happened; we just need that for
     [Jkind] *)
  let equate s1 s2 =
    match equate_tracking_mutation s1 s2 with
    | Unequal -> false
    | Equal_mutated_first | Equal_mutated_second | Equal_no_mutation
    | Equal_mutated_both ->
      true

  let rec is_void_defaulting = function
    | Base Void -> true
    | Var v -> (
      match v.contents with
      (* CR layouts v5: this should probably default to void now *)
      | None ->
        set v Static.T_option.value;
        false
      | Some s -> is_void_defaulting s)
    | Base (Value | Float64 | Float32 | Word | Bits32 | Bits64) -> false
    | Product _ -> false
  (* XXX does this need to default deeply? update comment one way or another. *)

  (*** pretty printing ***)

  let rec to_string s =
    match get s with
    | Base b -> to_string_base b
    | Var v -> Var.name v
    | Product ts -> String.concat " * " (List.map to_string ts)

  let format ppf t =
    let rec format nested ppf t =
      match get t with
      | Base b -> Format.fprintf ppf "%s" (to_string_base b)
      | Var v -> Format.fprintf ppf "%s" (Var.name v)
      | Product ts ->
        Format.fprintf ppf "@[%a@]"
          (Misc.pp_parens_if nested
             (Format.pp_print_list
                ~pp_sep:(fun ppf () -> Format.fprintf ppf " & ")
                (format true)))
          ts
    in
    format false ppf t

  (*** debug printing **)

  include Static.T

  let for_function = value

  let for_predef_value = value

  let for_block_element = value

  let for_probe_body = value

  let for_poly_variant = value

  let for_record = value

  let for_object = value

  let for_lazy_body = value

  let for_tuple_element = value

  let for_variant_arg = value

  let for_instance_var = value

  let for_class_arg = value

  let for_method = value

  let for_initializer = value

  let for_module = value

  let for_tuple = value

  let for_array_get_result = value

  let for_array_comprehension_element = value

  let for_list_element = value
end

module Layout = struct
  type 'sort layout =
    | Sort of 'sort
    | Product of 'sort layout list
    | Any

  module Const = struct
    type t =
      | Any
      | Base of Sort.base
      | Product of t list

    module Legacy = struct
      type t =
        | Any
        | Value
        | Void
        | Immediate64
        | Immediate
        | Float64
        | Float32
        | Word
        | Bits32
        | Bits64
        | Product of t list
    end
  end

  type t = Sort.t layout
end

module Externality = struct
  type t =
    | External
    | External64
    | Internal
end

module Modes = Mode.Alloc.Const

module Jkind_desc = struct
  type 'type_expr t =
    { layout : Layout.t;
      modes_upper_bounds : Modes.t;
      externality_upper_bound : Externality.t
    }
end

(* A history of conditions placed on a jkind.

   INVARIANT: at most one sort variable appears in this history.
   This is a natural consequence of producing this history by comparing
   jkinds.
*)
type 'type_expr history =
  | Interact of
      { reason : Jkind_intf.History.interact_reason;
        lhs_jkind : 'type_expr Jkind_desc.t;
        lhs_history : 'type_expr history;
        rhs_jkind : 'type_expr Jkind_desc.t;
        rhs_history : 'type_expr history
      }
  | Creation of Jkind_intf.History.creation_reason

type 'type_expr t =
  { jkind : 'type_expr Jkind_desc.t;
    history : 'type_expr history;
    has_warned : bool
  }

module Const = struct
  type 'type_expr t =
    { layout : Layout.Const.t;
      modes_upper_bounds : Modes.t;
      externality_upper_bound : Externality.t
    }
end

type 'type_expr const = 'type_expr Const.t

type 'type_expr annotation = 'type_expr const * Jane_syntax.Jkind.annotation
