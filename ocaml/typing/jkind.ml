(**************************************************************************)
(*                                                                        *)
(*                                 OCaml                                  *)
(*                                                                        *)
(*                Chris Casinghino, Jane Street, New York                 *)
(*                                                                        *)
(*   Copyright 2021 Jane Street Group LLC                                 *)
(*                                                                        *)
(*   All rights reserved.  This file is distributed under the terms of    *)
(*   the GNU Lesser General Public License version 2.1, with the          *)
(*   special exception on linking described in the file LICENSE.          *)
(*                                                                        *)
(**************************************************************************)

open Mode
open Jkind_types

[@@@warning "+9"]

(* A *sort* is the information the middle/back ends need to be able to
   compile a manipulation (storing, passing, etc) of a runtime value. *)
module Sort = Jkind_types.Sort

type sort = Sort.t

type type_expr = Types.type_expr

(* A *layout* of a type describes the way values of that type are stored at
   runtime, including details like width, register convention, calling
   convention, etc. A layout may be *representable* or *unrepresentable*.  The
   middle/back ends are unable to cope with values of types with an
   unrepresentable layout. The only unrepresentable layout is `any`, which is
   the top of the layout lattice. *)
module Layout = struct
  open Jkind_types.Layout

  type nonrec 'sort layout = 'sort layout

  module Debug_printers = struct
    open Format

    let rec t ppf = function
      | Any -> fprintf ppf "Any"
      | Sort s -> fprintf ppf "Sort %a" Sort.Debug_printers.t s
      | Product ts ->
        fprintf ppf "Product [ %a ]"
          (pp_print_list ~pp_sep:(fun ppf () -> Format.fprintf ppf ";@ ") t)
          ts
  end

  module Const = struct
    type t = Const.t =
      | Any
      | Base of Sort.base
      | Product of t list

    let max = Any

    let rec equal c1 c2 =
      match c1, c2 with
      | Base b1, Base b2 -> Sort.equal_base b1 b2
      | Any, Any -> true
      | Product cs1, Product cs2 -> List.equal equal cs1 cs2
      | (Base _ | Any | Product _), _ -> false

    let rec sub (c1 : t) (c2 : t) : Misc.Le_result.t =
      match c1, c2 with
      | _ when equal c1 c2 -> Equal
      | _, Any -> Less
      | Product consts1, Product consts2 ->
        if List.compare_lengths consts1 consts2 = 0
        then Misc.Le_result.combine_list (List.map2 sub consts1 consts2)
        else Not_le
      | (Any | Base _ | Product _), _ -> Not_le

    module Static = struct
      let value = Base Sort.Value

      let void = Base Sort.Void

      let float64 = Base Sort.Float64

      let float32 = Base Sort.Float32

      let word = Base Sort.Word

      let bits32 = Base Sort.Bits32

      let bits64 = Base Sort.Bits64

      let of_base : Sort.base -> t = function
        | Value -> value
        | Void -> void
        | Float64 -> float64
        | Float32 -> float32
        | Word -> word
        | Bits32 -> bits32
        | Bits64 -> bits64
    end

    include Static

    let rec get_sort : t -> Sort.Const.t option = function
      | Any -> None
      | Base b -> Some (Const_base b)
      | Product ts ->
        Option.map
          (fun x -> Sort.Const_product x)
          (Misc.Stdlib.List.map_option get_sort ts)

    let rec of_sort s =
      match Sort.get s with
      | Var _ -> None
      | Base b -> Some (Static.of_base b)
      | Product sorts ->
        Option.map
          (fun x -> Product x)
          (Misc.Stdlib.List.map_option of_sort sorts)

    let to_string t =
      let rec to_string nested (t : t) =
        match t with
        | Any -> "any"
        | Base b -> Sort.to_string_base b
        | Product ts ->
          String.concat ""
            [ (if nested then "(" else "");
              String.concat " & " (List.map (to_string true) ts);
              (if nested then ")" else "") ]
      in
      to_string false t

    module Legacy = struct
      (* CR layouts v2.8: get rid of this *)
      type t = Jkind_types.Layout.Const.Legacy.t =
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

      let to_string t =
        let rec to_string nested = function
          | Any -> "any"
          | Value -> "value"
          | Void -> "void"
          | Immediate64 -> "immediate64"
          | Immediate -> "immediate"
          | Float64 -> "float64"
          | Float32 -> "float32"
          | Word -> "word"
          | Bits32 -> "bits32"
          | Bits64 -> "bits64"
          | Product ts ->
            String.concat ""
              [ (if nested then "(" else "");
                String.concat " & " (List.map (to_string true) ts);
                (if nested then ")" else "") ]
        in
        to_string false t
    end

    (* XXX is this used? *)
    let rec of_sort_const : Sort.const -> t = function
      | Const_base b -> Base b
      | Const_product consts -> Product (List.map of_sort_const consts)
  end

  type t = Sort.t layout

  let rec of_const (const : Const.t) : t =
    match const with
    | Any -> Any
    | Base b -> Sort (Sort.of_base b)
    | Product cs -> Product (List.map of_const cs)

  let rec to_sort = function
    | Any -> None
    | Sort s -> Some s
    | Product ts -> to_product_sort ts

  and to_product_sort ts =
    let components = List.map to_sort ts in
    Option.map
      (fun x -> Sort.Product x)
      (Misc.Stdlib.List.some_if_all_elements_are_some components)

  (* XXX get rid of this *)
  let rec to_sort_lub = function
    | Any -> None
    | Sort s -> Some (s, `Exact)
    | Product ts -> to_product_sort_lub ts

  and to_product_sort_lub ts =
    let components = List.map to_sort_lub ts in
    match Misc.Stdlib.List.some_if_all_elements_are_some components with
    | None -> None
    | Some components ->
      let components, lub_or_exact = List.split components in
      let lub_or_exact =
        if List.for_all (fun x -> x = `Exact) lub_or_exact then `Exact else `Lub
      in
      Some (Sort.Product components, lub_or_exact)

  let sort_equal_result ~allow_mutation result =
    match (result : Sort.equate_result) with
    | (Equal_mutated_first | Equal_mutated_second | Equal_mutated_both)
      when not allow_mutation ->
      Misc.fatal_errorf "Jkind.equal: Performed unexpected mutation"
    | Unequal -> false
    | Equal_no_mutation | Equal_mutated_first | Equal_mutated_second
    | Equal_mutated_both ->
      true

  let rec equate_or_equal ~allow_mutation t1 t2 =
    match t1, t2 with
    | Sort s1, Sort s2 ->
      sort_equal_result ~allow_mutation (Sort.equate_tracking_mutation s1 s2)
    | Product ts, Sort sort | Sort sort, Product ts -> (
      match to_product_sort ts with
      | None -> false
      | Some sort' ->
        sort_equal_result ~allow_mutation
          (Sort.equate_tracking_mutation sort sort'))
    | Product ts1, Product ts2 ->
      List.equal (equate_or_equal ~allow_mutation) ts1 ts2
    | Any, Any -> true
    | (Any | Sort _ | Product _), _ -> false

  let rec sub t1 t2 : Misc.Le_result.t =
    match t1, t2 with
    | Any, Any -> Equal
    | _, Any -> Less
    | Any, _ -> Not_le
    | Sort s1, Sort s2 -> if Sort.equate s1 s2 then Equal else Not_le
    | Product ts1, Product ts2 ->
      if List.compare_lengths ts1 ts2 = 0
      then Misc.Le_result.combine_list (List.map2 sub ts1 ts2)
      else Not_le
    | Product ts1, Sort s2 -> (
      match to_product_sort_lub ts1 with
      | None -> Not_le
      | Some (s1, lub_or_exact) ->
        if Sort.equate s1 s2
        then match lub_or_exact with `Lub -> Less | `Exact -> Equal
        else Not_le)
    | Sort s1, Product ts2 -> (
      match to_product_sort ts2 with
      | None -> Not_le
      | Some s2 -> if Sort.equate s1 s2 then Equal else Not_le)

  let rec intersection t1 t2 =
    match t1, t2 with
    | _, Any -> Some t1
    | Any, _ -> Some t2
    | Sort s1, Sort s2 -> if Sort.equate s1 s2 then Some t1 else None
    | Product ts1, Product ts2 ->
      if List.compare_lengths ts1 ts2 = 0
      then
        let components = List.map2 intersection ts1 ts2 in
        Option.map
          (fun x -> Product x)
          (Misc.Stdlib.List.some_if_all_elements_are_some components)
      else None
    | (Product ts as t), Sort sort | Sort sort, (Product ts as t) -> (
      match to_product_sort_lub ts with
      | None -> None
      | Some (sort', _) -> if Sort.equate sort sort' then Some t else None)

  let of_new_sort_var () =
    let sort = Sort.new_var () in
    Sort sort, sort

  let rec default_to_value_and_get : Layout.t -> Const.t = function
    | Any -> Any
    | Sort s -> Const.of_sort_const (Sort.default_to_value_and_get s)
    | Product p -> Product (List.map default_to_value_and_get p)

  (* needed? *)
  (* let value = Sort Sort.value
   *
   * let void = Sort Sort.void
   *
   * let float64 = Sort Sort.float64
   *
   * let word = Sort Sort.word
   *
   * let bits32 = Sort Sort.bits32
   *
   * let bits64 = Sort Sort.bits64
   *
   * let rec get_default_value : t -> Const.t = function
   *   | Any -> Any
   *   | Non_null_value -> Non_null_value
   *   | Sort s -> Const.of_sort_const (Sort.get_default_value s)
   *   | Product layouts -> Product (List.map get_default_value layouts)
   *
   * let rec format ppf =
   *   let open Format in
   *   function
   *   | Sort s -> fprintf ppf "%s" (Sort.to_string s)
   *   | Any -> fprintf ppf "any"
   *   | Non_null_value -> fprintf ppf "non_null_value"
   *   | Product ts ->
   *     Format.pp_print_list
   *       ~pp_sep:(fun ppf () -> pp_print_text ppf " * ")
   *       format ppf ts
   *
   * (* These assume they are run on the result of get - so any Var contains None
   *    *)
   * let rec sort_is_constant : Sort.t -> Const.t option = function
   *   | Base b -> Some (Base b)
   *   | Var _ -> None
   *   | Product sorts ->
   *     Option.map (fun x -> Const.Product x) (sorts_are_constant sorts)
   *
   * and sorts_are_constant : Sort.t list -> Const.t list option = function
   *   | [] -> Some []
   *   | sort :: sorts ->
   *     Option.bind (sort_is_constant sort) (fun const ->
   *         Option.bind (sorts_are_constant sorts) (fun consts ->
   *             Some (const :: consts)))
   *
   * (* CR ccasinghino: surely this can be abstracted out with the sort one *)
   * let rec layout_is_constant : t -> Const.t option = function
   *   | Sort s -> sort_is_constant (Sort.get s)
   *   | Any -> Some Any
   *   | Non_null_value -> Some Non_null_value
   *   | Product layouts ->
   *     Option.map (fun x -> Const.Product x) (layouts_are_constant layouts)
   *
   * and layouts_are_constant : t list -> Const.t list option = function
   *   | [] -> Some []
   *   | layout :: layouts ->
   *     Option.bind (layout_is_constant layout) (fun const ->
   *         Option.bind (layouts_are_constant layouts) (fun consts ->
   *             Some (const :: consts)))
   *
   * (* Post-condition: For any [Var v] in the result, [!v] is [None]. *) *)
end

module Externality = struct
  type t = Jkind_types.Externality.t =
    | External
    | External64
    | Internal

  let max = Internal

  let min = External

  let equal e1 e2 =
    match e1, e2 with
    | External, External -> true
    | External64, External64 -> true
    | Internal, Internal -> true
    | (External | External64 | Internal), _ -> false

  let less_or_equal t1 t2 : Misc.Le_result.t =
    match t1, t2 with
    | External, External -> Equal
    | External, (External64 | Internal) -> Less
    | External64, External -> Not_le
    | External64, External64 -> Equal
    | External64, Internal -> Less
    | Internal, (External | External64) -> Not_le
    | Internal, Internal -> Equal

  let le t1 t2 = Misc.Le_result.is_le (less_or_equal t1 t2)

  let meet t1 t2 =
    match t1, t2 with
    | External, (External | External64 | Internal)
    | (External64 | Internal), External ->
      External
    | External64, (External64 | Internal) | Internal, External64 -> External64
    | Internal, Internal -> Internal

  let join t1 t2 =
    match t1, t2 with
    | Internal, (External | External64 | Internal)
    | (External | External64), Internal ->
      Internal
    | External64, (External | External64) | External, External64 -> External64
    | External, External -> External

  let print ppf = function
    | External -> Format.fprintf ppf "external_"
    | External64 -> Format.fprintf ppf "external64"
    | Internal -> Format.fprintf ppf "internal"
end

module Modes = struct
  include Alloc.Const

  let less_or_equal a b : Misc.Le_result.t =
    match le a b, le b a with
    | true, true -> Equal
    | true, false -> Less
    | false, _ -> Not_le

  let equal a b = Misc.Le_result.is_equal (less_or_equal a b)
end

module History = struct
  include Jkind_intf.History

  let has_imported_history t =
    match t.history with Creation Imported -> true | _ -> false

  let update_reason t reason = { t with history = Creation reason }

  let with_warning t = { t with has_warned = true }

  let has_warned t = t.has_warned
end

(* forward declare [Const.t] so we can use it for [Error.t] *)
type const = type_expr Jkind_types.Const.t

(******************************)
(*** user errors ***)

module Error = struct
  type t =
    | Insufficient_level of
        { jkind : const;
          required_layouts_level : Language_extension.maturity
        }
    | Unknown_jkind of Jane_syntax.Jkind.t
    | Unknown_mode of Jane_syntax.Mode_expr.Const.t
    | Multiple_jkinds of
        { from_annotation : const;
          from_attribute : const
        }

  exception User_error of Location.t * t
end

let raise ~loc err = raise (Error.User_error (loc, err))

module Const = struct
  open Jkind_types.Const

  type t = const

  let max =
    { layout = Layout.Const.max;
      modes_upper_bounds = Modes.max;
      externality_upper_bound = Externality.max
    }

  let get_layout const = const.layout

  let get_modal_upper_bounds const = const.modes_upper_bounds

  let get_externality_upper_bound const = const.externality_upper_bound

  let rec get_legacy_layout
      ({ layout; modes_upper_bounds = _; externality_upper_bound } as k) :
      Layout.Const.Legacy.t =
    match layout, externality_upper_bound with
    | Any, _ -> Any
    | Base Value, Internal -> Value
    | Base Value, External64 -> Immediate64
    | Base Value, External -> Immediate
    | Base Void, _ -> Void
    | Base Float64, _ -> Float64
    | Base Float32, _ -> Float32
    | Base Word, _ -> Word
    | Base Bits32, _ -> Bits32
    | Base Bits64, _ -> Bits64
    | Product layouts, _ ->
      Product
        (List.map (fun layout -> get_legacy_layout { k with layout }) layouts)

  let equal
      { layout = lay1;
        modes_upper_bounds = modes1;
        externality_upper_bound = ext1
      }
      { layout = lay2;
        modes_upper_bounds = modes2;
        externality_upper_bound = ext2
      } =
    Layout.Const.equal lay1 lay2
    && Modes.equal modes1 modes2
    && Externality.equal ext1 ext2

  let sub
      { layout = lay1;
        modes_upper_bounds = modes1;
        externality_upper_bound = ext1
      }
      { layout = lay2;
        modes_upper_bounds = modes2;
        externality_upper_bound = ext2
      } =
    Misc.Le_result.combine_list
      [ Layout.Const.sub lay1 lay2;
        Modes.less_or_equal modes1 modes2;
        Externality.less_or_equal ext1 ext2 ]

  let not_mode_crossing layout =
    { layout;
      modes_upper_bounds = Modes.max;
      externality_upper_bound = Externality.max
    }

  let mode_crossing layout =
    { layout;
      modes_upper_bounds = Modes.min;
      externality_upper_bound = Externality.min
    }

  module Primitive = struct
    type nonrec t =
      { jkind : t;
        name : string
      }

    let any = { jkind = not_mode_crossing Any; name = "any" }

    let value = { jkind = not_mode_crossing Layout.Const.value; name = "value" }

    let void = { jkind = not_mode_crossing Layout.Const.void; name = "void" }

    let immediate =
      { jkind = mode_crossing Layout.Const.value; name = "immediate" }

    (* [immediate64] describes types that are stored directly (no indirection)
       on 64-bit platforms but indirectly on 32-bit platforms. The key question:
       along which modes should a [immediate64] cross? As of today, all of them,
       but the reasoning for each is independent and somewhat subtle:

       * Locality: This is fine, because we do not have stack-allocation on
       32-bit platforms. Thus mode-crossing is sound at any type on 32-bit,
       including immediate64 types.

       * Linearity: This is fine, because linearity matters only for function
       types, and an immediate64 cannot be a function type and cannot store
       one either.

       * Uniqueness: This is fine, because uniqueness matters only for
       in-place update, and no record supporting in-place update is an
       immediate64. ([@@unboxed] records do not support in-place update.)

       * Syncness: This is fine, because syncness matters only for function
       types, and an immediate64 cannot be a function type and cannot store
       one either.

       * Contention: This is fine, because contention matters only for
       types with mutable fields, and an immediate64 does not have immutable
       fields.

       In practice, the functor that creates immediate64s,
       [Stdlib.Sys.Immediate64.Make], will require these conditions on its
       argument. But the arguments that we expect here will have no trouble
       meeting the conditions.
    *)
    let immediate64 =
      { jkind = { immediate.jkind with externality_upper_bound = External64 };
        name = "immediate64"
      }

    (* CR layouts v2.8: This should not mode cross, but we need syntax for mode
       crossing first *)
    let float64 =
      { jkind = mode_crossing Layout.Const.float64; name = "float64" }

    (* CR layouts v2.8: This should not mode cross, but we need syntax for mode
       crossing first *)
    let float32 =
      { jkind = mode_crossing Layout.Const.float32; name = "float32" }

    let word = { jkind = not_mode_crossing Layout.Const.word; name = "word" }

    let bits32 =
      { jkind = not_mode_crossing Layout.Const.bits32; name = "bits32" }

    let bits64 =
      { jkind = not_mode_crossing Layout.Const.bits64; name = "bits64" }

    let all =
      [ any;
        value;
        void;
        immediate;
        immediate64;
        float64;
        float32;
        word;
        bits32;
        bits64 ]
  end

  module To_out_jkind_const = struct
    open Outcometree

    module Bounds = struct
      type t =
        { alloc_bounds : Alloc.Const.t;
          externality_bound : Externality.t
        }

      let of_jkind jkind =
        { alloc_bounds = jkind.modes_upper_bounds;
          externality_bound = jkind.externality_upper_bound
        }
    end

    let get_modal_bound ~le ~print ~base actual =
      match le actual base with
      | true -> (
        match le base actual with
        | true -> `Valid None
        | false -> `Valid (Some (Format.asprintf "%a" print actual)))
      | false -> `Invalid

    let get_modal_bounds ~(base : Bounds.t) (actual : Bounds.t) =
      [ get_modal_bound ~le:Locality.Const.le ~print:Locality.Const.print
          ~base:base.alloc_bounds.areality actual.alloc_bounds.areality;
        get_modal_bound ~le:Uniqueness.Const.le ~print:Uniqueness.Const.print
          ~base:base.alloc_bounds.uniqueness actual.alloc_bounds.uniqueness;
        get_modal_bound ~le:Linearity.Const.le ~print:Linearity.Const.print
          ~base:base.alloc_bounds.linearity actual.alloc_bounds.linearity;
        get_modal_bound ~le:Contention.Const.le ~print:Contention.Const.print
          ~base:base.alloc_bounds.contention actual.alloc_bounds.contention;
        get_modal_bound ~le:Portability.Const.le ~print:Portability.Const.print
          ~base:base.alloc_bounds.portability actual.alloc_bounds.portability;
        get_modal_bound ~le:Externality.le ~print:Externality.print
          ~base:base.externality_bound actual.externality_bound ]
      |> List.rev
      |> List.fold_left
           (fun acc mode ->
             match acc, mode with
             | _, `Invalid | None, _ -> None
             | acc, `Valid None -> acc
             | Some acc, `Valid (Some mode) -> Some (mode :: acc))
           (Some [])

    (** Write [actual] in terms of [base] *)
    let convert_with_base ~(base : Primitive.t) actual =
      let matching_layouts =
        Layout.Const.equal base.jkind.layout actual.layout
      in
      let modal_bounds =
        get_modal_bounds
          ~base:(Bounds.of_jkind base.jkind)
          (Bounds.of_jkind actual)
      in
      match matching_layouts, modal_bounds with
      | true, Some modal_bounds -> Some { base = base.name; modal_bounds }
      | false, _ | _, None -> None

    (** Select the out_jkind_const with the least number of modal bounds to print *)
    let rec select_simplest = function
      | a :: b :: tl ->
        let simpler =
          if List.length a.modal_bounds < List.length b.modal_bounds
          then a
          else b
        in
        select_simplest (simpler :: tl)
      | [out] -> Some out
      | [] -> None

    let convert jkind =
      (* For each primitive jkind, we try to print the jkind in terms of it (this is
         possible if the primitive is a subjkind of it). We then choose the "simplest". The
           "simplest" is taken to mean the one with the least number of modes that need to
         follow the [mod]. *)
      let simplest =
        Primitive.all
        |> List.filter_map (fun base -> convert_with_base ~base jkind)
        |> select_simplest
      in
      match simplest with
      | Some simplest -> simplest
      | None ->
        (* CR layouts v2.8: sometimes there is no valid way to build a jkind from a
           built-in abbreviation. For now, we just pretend that the layout name is a valid
           jkind abbreviation whose modal bounds are all max, even though this is a
           lie. *)
        let out_jkind_verbose =
          convert_with_base
            ~base:
              { jkind =
                  { layout = jkind.layout;
                    modes_upper_bounds = Modes.max;
                    externality_upper_bound = Externality.max
                  };
                name = Layout.Const.to_string jkind.layout
              }
            jkind
        in
        (* convert_with_base is guaranteed to succeed since the layout matches and the
           modal bounds are all max *)
        Option.get out_jkind_verbose
  end

  let to_out_jkind_const = To_out_jkind_const.convert

  let format ppf jkind =
    let legacy_layout = get_legacy_layout jkind in
    let layout_str = Layout.Const.Legacy.to_string legacy_layout in
    Format.fprintf ppf "%s" layout_str

  let of_attribute : Builtin_attributes.jkind_attribute -> t = function
    | Immediate -> Primitive.immediate.jkind
    | Immediate64 -> Primitive.immediate64.jkind

  module ModeParser = struct
    type mode =
      | Areality of Locality.Const.t
      | Linearity of Linearity.Const.t
      | Uniqueness of Uniqueness.Const.t
      | Contention of Contention.Const.t
      | Portability of Portability.Const.t
      | Externality of Externality.t

    let parse_mode unparsed_mode =
      let { txt = name; loc } =
        (unparsed_mode : Jane_syntax.Mode_expr.Const.t :> _ Location.loc)
      in
      match name with
      | "global" -> Areality Global
      | "local" -> Areality Local
      | "many" -> Linearity Many
      | "once" -> Linearity Once
      | "unique" -> Uniqueness Unique
      | "shared" -> Uniqueness Shared
      | "internal" -> Externality Internal
      | "external64" -> Externality External64
      | "external_" -> Externality External
      | "contended" -> Contention Contended
      | "uncontended" -> Contention Uncontended
      | "portable" -> Portability Portable
      | "nonportable" -> Portability Nonportable
      | _ -> raise ~loc (Unknown_mode unparsed_mode)

    let parse_modes
        (Location.{ txt = modes; loc = _ } : Jane_syntax.Mode_expr.t) =
      List.map parse_mode modes
  end

  (* XXX replace/unify this with the similar product function below *)
  let jkind_of_product_annotations jkinds =
    let folder (layouts, mode_ub, ext_ub)
        { layout; modes_upper_bounds; externality_upper_bound } =
      ( layout :: layouts,
        Modes.join mode_ub modes_upper_bounds,
        Externality.join ext_ub externality_upper_bound )
    in
    let layouts, mode_ub, ext_ub =
      List.fold_left folder ([], Modes.min, Externality.min) jkinds
    in
    { layout = Product (List.rev layouts);
      modes_upper_bounds = mode_ub;
      externality_upper_bound = ext_ub
    }

  let rec of_user_written_annotation_unchecked_level
      (jkind : Jane_syntax.Jkind.t) : t =
    match jkind with
    | Abbreviation const -> (
      let { txt = name; loc } =
        (const : Jane_syntax.Jkind.Const.t :> _ Location.loc)
      in
      (* CR layouts 2.8: move this to predef *)
      match name with
      | "any" -> Primitive.any.jkind
      | "value" -> Primitive.value.jkind
      | "void" -> Primitive.void.jkind
      | "immediate64" -> Primitive.immediate64.jkind
      | "immediate" -> Primitive.immediate.jkind
      | "float64" -> Primitive.float64.jkind
      | "float32" -> Primitive.float32.jkind
      | "word" -> Primitive.word.jkind
      | "bits32" -> Primitive.bits32.jkind
      | "bits64" -> Primitive.bits64.jkind
      | _ -> raise ~loc (Unknown_jkind jkind))
    | Mod (jkind, modes) ->
      let base = of_user_written_annotation_unchecked_level jkind in
      (* for each mode, lower the corresponding modal bound to be that mode *)
      let parsed_modes = ModeParser.parse_modes modes in
      let meet_mode jkind (mode : ModeParser.mode) =
        match mode with
        | Areality areality ->
          { jkind with
            modes_upper_bounds =
              { jkind.modes_upper_bounds with
                areality =
                  Locality.Const.meet jkind.modes_upper_bounds.areality areality
              }
          }
        | Linearity linearity ->
          { jkind with
            modes_upper_bounds =
              Modes.meet jkind.modes_upper_bounds
                { jkind.modes_upper_bounds with
                  linearity =
                    Linearity.Const.meet jkind.modes_upper_bounds.linearity
                      linearity
                }
          }
        | Uniqueness uniqueness ->
          { jkind with
            modes_upper_bounds =
              Modes.meet jkind.modes_upper_bounds
                { jkind.modes_upper_bounds with
                  uniqueness =
                    Uniqueness.Const.meet jkind.modes_upper_bounds.uniqueness
                      uniqueness
                }
          }
        | Contention contention ->
          { jkind with
            modes_upper_bounds =
              Modes.meet jkind.modes_upper_bounds
                { jkind.modes_upper_bounds with
                  contention =
                    Contention.Const.meet jkind.modes_upper_bounds.contention
                      contention
                }
          }
        | Portability portability ->
          { jkind with
            modes_upper_bounds =
              Modes.meet jkind.modes_upper_bounds
                { jkind.modes_upper_bounds with
                  portability =
                    Portability.Const.meet jkind.modes_upper_bounds.portability
                      portability
                }
          }
        | Externality externality ->
          { jkind with
            externality_upper_bound =
              Externality.meet jkind.externality_upper_bound externality
          }
      in
      List.fold_left meet_mode base parsed_modes
    | Product ts ->
      let jkinds = List.map of_user_written_annotation_unchecked_level ts in
      jkind_of_product_annotations jkinds
    | Default | With _ | Kind_of _ -> Misc.fatal_error "XXX unimplemented"

  module Sort = Sort.Const
  module Layout = Layout.Const
end

module Desc = struct
  (* This type is used only for printing.  We use the mode upper bounds in the
     const case to print things like "immediate" nicely.  But we don't need
     modes in the var case, because they can only be (or contain) base
     layouts, not more interesting kinds. *)
  type t =
    | Const of Const.t
    | Var of Sort.var
    | Product of t list

  let format ppf =
    let rec format nested ppf =
      let open Format in
      function
      | Const c -> fprintf ppf "%a" Const.format c
      | Var v -> fprintf ppf "%s" (Sort.Var.name v)
      | Product ts ->
        fprintf ppf "@[%a@]"
          (Misc.pp_parens_if nested
             (pp_print_list
                ~pp_sep:(fun ppf () -> fprintf ppf "@ & ")
                (format true)))
          ts
    in
    format false ppf

  (* considers sort variables < Any. Two sort variables are in a [sub]
     relationship only when they are equal.
     Never does mutation.
     Pre-condition: no filled-in sort variables. product must contain a var. *)
  let rec sub d1 d2 : Misc.Le_result.t =
    match d1, d2 with
    | Const c1, Const c2 -> Const.sub c1 c2
    | Var _, Const c when Const.equal Const.max c -> Less
    | Var v1, Var v2 -> if v1 == v2 then Equal else Not_le
    | Product ds1, Product ds2 ->
      if List.compare_lengths ds1 ds2 = 0
      then Misc.Le_result.combine_list (List.map2 sub ds1 ds2)
      else Not_le
    | Const _, Product _ | Product _, Const _ | Const _, Var _ | Var _, Const _
      ->
      Not_le
    | Var _, Product _ | Product _, Var _ -> Not_le
end

module Jkind_desc = struct
  open Jkind_types.Jkind_desc

  let of_const
      ({ layout; modes_upper_bounds; externality_upper_bound } : Const.t) =
    { layout = Layout.of_const layout;
      modes_upper_bounds;
      externality_upper_bound
    }

  let not_mode_crossing layout =
    { layout;
      modes_upper_bounds = Modes.max;
      externality_upper_bound = Externality.max
    }

  let add_mode_crossing t =
    { t with
      modes_upper_bounds = Modes.min;
      externality_upper_bound = Externality.min
    }

  let max = of_const Const.max

  let equate_or_equal ~allow_mutation
      { layout = lay1;
        modes_upper_bounds = modes1;
        externality_upper_bound = ext1
      }
      { layout = lay2;
        modes_upper_bounds = modes2;
        externality_upper_bound = ext2
      } =
    Layout.equate_or_equal ~allow_mutation lay1 lay2
    && Modes.equal modes1 modes2
    && Externality.equal ext1 ext2

  let sub
      { layout = lay1;
        modes_upper_bounds = modes1;
        externality_upper_bound = ext1
      }
      { layout = lay2;
        modes_upper_bounds = modes2;
        externality_upper_bound = ext2
      } =
    Misc.Le_result.combine_list
      [ Layout.sub lay1 lay2;
        Modes.less_or_equal modes1 modes2;
        Externality.less_or_equal ext1 ext2 ]

  let intersection
      { layout = lay1;
        modes_upper_bounds = modes1;
        externality_upper_bound = ext1
      }
      { layout = lay2;
        modes_upper_bounds = modes2;
        externality_upper_bound = ext2
      } =
    Option.bind (Layout.intersection lay1 lay2) (fun layout ->
        Some
          { layout;
            modes_upper_bounds = Modes.meet modes1 modes2;
            externality_upper_bound = Externality.meet ext1 ext2
          })

  let of_new_sort_var () =
    let layout, sort = Layout.of_new_sort_var () in
    not_mode_crossing layout, sort

  module Primitive = struct
    let any = max

    let value = of_const Const.Primitive.value.jkind

    let void = of_const Const.Primitive.void.jkind

    (* [immediate64] describes types that are stored directly (no indirection)
       on 64-bit platforms but indirectly on 32-bit platforms. The key question:
       along which modes should a [immediate64] cross? As of today, all of them,
       but the reasoning for each is independent and somewhat subtle:

       * Locality: This is fine, because we do not have stack-allocation on
       32-bit platforms. Thus mode-crossing is sound at any type on 32-bit,
       including immediate64 types.

       * Linearity: This is fine, because linearity matters only for function
       types, and an immediate64 cannot be a function type and cannot store
       one either.

       * Uniqueness: This is fine, because uniqueness matters only for
       in-place update, and no record supporting in-place update is an
       immediate64. ([@@unboxed] records do not support in-place update.)

       * Portability: This is fine, because portability matters only for function
       types, and an immediate64 cannot be a function type and cannot store
       one either.

       * Contention: This is fine, because contention matters only for
       types with mutable fields, and an immediate64 does not have immutable
       fields.

       In practice, the functor that creates immediate64s,
       [Stdlib.Sys.Immediate64.Make], will require these conditions on its
       argument. But the arguments that we expect here will have no trouble
       meeting the conditions.
    *)
    let immediate64 = of_const Const.Primitive.immediate64.jkind

    let immediate = of_const Const.Primitive.immediate.jkind

    let float64 = of_const Const.Primitive.float64.jkind

    let float32 = of_const Const.Primitive.float32.jkind

    let word = of_const Const.Primitive.word.jkind

    let bits32 = of_const Const.Primitive.bits32.jkind

    let bits64 = of_const Const.Primitive.bits64.jkind
  end

  let product jkinds =
    (* Here we throw away the history of the component jkinds. This is not
       great. We should, as part of a broader pass on error messages around
       product kinds, zip them up into some kind of product history. *)
    let folder (layouts, mode_ub, ext_ub)
        { jkind = { layout; modes_upper_bounds; externality_upper_bound };
          history = _;
          has_warned = _
        } =
      ( layout :: layouts,
        Modes.join mode_ub modes_upper_bounds,
        Externality.join ext_ub externality_upper_bound )
    in
    let layouts, mode_ub, ext_ub =
      List.fold_left folder ([], Modes.min, Externality.min) jkinds
    in
    { layout = Product (List.rev layouts);
      modes_upper_bounds = mode_ub;
      externality_upper_bound = ext_ub
    }

  let rec get_sort modes_upper_bounds externality_upper_bound s : Desc.t =
    match Sort.get s with
    | Base b ->
      Const { layout = Base b; modes_upper_bounds; externality_upper_bound }
    | Var v -> Var v
    | Product sorts ->
      Desc.Product
        (List.map
           (fun x -> get_sort modes_upper_bounds externality_upper_bound x)
           sorts)

  let rec get ({ layout; modes_upper_bounds; externality_upper_bound } as k) :
      Desc.t =
    match layout with
    | Any -> Const { layout = Any; modes_upper_bounds; externality_upper_bound }
    | Sort s -> get_sort modes_upper_bounds externality_upper_bound s
    | Product layouts ->
      Product (List.map (fun layout -> get { k with layout }) layouts)

  module Debug_printers = struct
    open Format

    let t ppf { layout; modes_upper_bounds; externality_upper_bound } =
      fprintf ppf
        "{ layout = %a;@ modes_upper_bounds = %a;@ externality_upper_bound = \
         %a }"
        Layout.Debug_printers.t layout Modes.print modes_upper_bounds
        Externality.print externality_upper_bound
  end
end

type t = type_expr Jkind_types.t

let fresh_jkind jkind ~why =
  { jkind; history = Creation why; has_warned = false }

(******************************)
(* constants *)

module Primitive = struct
  let any_dummy_jkind =
    { jkind = Jkind_desc.max;
      history = Creation (Any_creation Dummy_jkind);
      has_warned = false
    }

  (* CR layouts: Should we be doing more memoization here? *)
  let any ~(why : History.any_creation_reason) =
    match why with
    | Dummy_jkind -> any_dummy_jkind (* share this one common case *)
    | _ -> fresh_jkind Jkind_desc.Primitive.any ~why:(Any_creation why)

  let value_v1_safety_check =
    { jkind = Jkind_desc.Primitive.value;
      history = Creation (Value_creation V1_safety_check);
      has_warned = false
    }

  let void ~why = fresh_jkind Jkind_desc.Primitive.void ~why:(Void_creation why)

  let value ~(why : History.value_creation_reason) =
    match why with
    | V1_safety_check -> value_v1_safety_check
    | _ -> fresh_jkind Jkind_desc.Primitive.value ~why:(Value_creation why)

  let immediate64 ~why =
    fresh_jkind Jkind_desc.Primitive.immediate64 ~why:(Immediate64_creation why)

  let immediate ~why =
    fresh_jkind Jkind_desc.Primitive.immediate ~why:(Immediate_creation why)

  let float64 ~why =
    fresh_jkind Jkind_desc.Primitive.float64 ~why:(Float64_creation why)

  let float32 ~why =
    fresh_jkind Jkind_desc.Primitive.float32 ~why:(Float32_creation why)

  let word ~why = fresh_jkind Jkind_desc.Primitive.word ~why:(Word_creation why)

  let bits32 ~why =
    fresh_jkind Jkind_desc.Primitive.bits32 ~why:(Bits32_creation why)

  let bits64 ~why =
    fresh_jkind Jkind_desc.Primitive.bits64 ~why:(Bits64_creation why)

  let product ~why ts =
    fresh_jkind (Jkind_desc.product ts) ~why:(Product_creation why)
end

let add_mode_crossing t =
  { t with jkind = Jkind_desc.add_mode_crossing t.jkind }

(*** extension requirements ***)
(* The [annotation_context] parameter can be used to allow annotations / kinds
   in different contexts to be enabled with different extension settings.
   At some points in time, we will not care about the context, and so this
   parameter might effectively be unused.
*)
(* CR layouts: When everything is stable, remove this function. *)
let get_required_layouts_level (context : History.annotation_context)
    (jkind : Const.t) : Language_extension.maturity =
  let legacy_layout = Const.get_legacy_layout jkind in
  match context, legacy_layout with
  | ( _,
      ( Value | Immediate | Immediate64 | Any | Float64 | Float32 | Word
      | Bits32 | Bits64 ) ) ->
    Stable
  | _, Void -> Alpha
  | _, Product _ -> Beta

(******************************)
(* construction *)

let of_new_sort_var ~why =
  let jkind, sort = Jkind_desc.of_new_sort_var () in
  fresh_jkind jkind ~why:(Concrete_creation why), sort

let of_new_sort ~why = fst (of_new_sort_var ~why)

(* CR layouts v2.8: remove this function *)
let of_const ~why
    ({ layout; modes_upper_bounds; externality_upper_bound } : Const.t) =
  { jkind =
      { layout = Layout.of_const layout;
        modes_upper_bounds;
        externality_upper_bound
      };
    history = Creation why;
    has_warned = false
  }

let const_of_user_written_annotation ~context Location.{ loc; txt = annot } =
  let const = Const.of_user_written_annotation_unchecked_level annot in
  let required_layouts_level = get_required_layouts_level context const in
  if not (Language_extension.is_at_least Layouts required_layouts_level)
  then raise ~loc (Insufficient_level { jkind = const; required_layouts_level });
  const

let of_annotated_const ~context ~const ~const_loc =
  of_const ~why:(Annotated (context, const_loc)) const

let of_annotation ~context (annot : _ Location.loc) =
  let const = const_of_user_written_annotation ~context annot in
  let jkind = of_annotated_const ~const ~const_loc:annot.loc ~context in
  jkind, (const, annot)

let of_annotation_option_default ~default ~context =
  Option.fold ~none:(default, None) ~some:(fun annot ->
      let t, annot = of_annotation ~context annot in
      t, Some annot)

let of_attribute ~context
    (attribute : Builtin_attributes.jkind_attribute Location.loc) =
  let const = Const.of_attribute attribute.txt in
  of_annotated_const ~context ~const ~const_loc:attribute.loc, const

let of_type_decl ~context (decl : Parsetree.type_declaration) =
  let jkind_of_annotation =
    Jane_syntax.Layouts.of_type_declaration decl
    |> Option.map (fun (annot, attrs) ->
           let t, const = of_annotation ~context annot in
           t, const, attrs)
  in
  let jkind_of_attribute =
    Builtin_attributes.jkind decl.ptype_attributes
    |> Option.map (fun attr ->
           let t, const = of_attribute ~context attr in
           (* This is a bit of a lie: the "annotation" here is being
              forged based on the jkind attribute. But: the jkind
              annotation is just used in printing/untypeast, and the
              all strings valid to use as a jkind attribute are
              valid (and equivalent) to write as an annotation, so
              this lie is harmless.
           *)
           let annot =
             Location.map
               (fun attr ->
                 let name = Builtin_attributes.jkind_attribute_to_string attr in
                 Jane_syntax.Jkind.(Abbreviation (Const.mk name Location.none)))
               attr
           in
           t, (const, annot), decl.ptype_attributes)
  in
  match jkind_of_annotation, jkind_of_attribute with
  | None, None -> None
  | (Some _ as x), None | None, (Some _ as x) -> x
  | Some (_, (from_annotation, _), _), Some (_, (from_attribute, _), _) ->
    raise ~loc:decl.ptype_loc
      (Multiple_jkinds { from_annotation; from_attribute })

let of_type_decl_default ~context ~default (decl : Parsetree.type_declaration) =
  match of_type_decl ~context decl with
  | Some (t, const, attrs) -> t, Some const, attrs
  | None -> default, None, decl.ptype_attributes

let for_boxed_record ~all_void =
  if all_void
  then Primitive.immediate ~why:Empty_record
  else Primitive.value ~why:Boxed_record

let for_boxed_variant ~all_voids =
  if all_voids
  then Primitive.immediate ~why:Enumeration
  else Primitive.value ~why:Boxed_variant

(******************************)
(* elimination and defaulting *)

let default_to_value_and_get
    { jkind = { layout; modes_upper_bounds; externality_upper_bound }; _ } :
    Const.t =
  { layout = Layout.default_to_value_and_get layout;
    modes_upper_bounds;
    externality_upper_bound
  }

let default_to_value t = ignore (default_to_value_and_get t)

let get t = Jkind_desc.get t.jkind

(* CR layouts: this function is suspect; it seems likely to reisenberg
   that refactoring could get rid of it *)
let sort_of_jkind (t : t) : sort =
  let rec sort_of_layout (t : Layout.t) =
    match t with
    | Any -> Misc.fatal_error "Jkind.sort_of_jkind"
    | Sort s -> s
    | Product ls -> Sort.Product (List.map sort_of_layout ls)
  in
  sort_of_layout t.jkind.layout

let get_layout jk : Layout.Const.t option =
  let rec aux : Layout.t -> Layout.Const.t option = function
    | Any -> Some Any
    | Sort s -> Layout.Const.of_sort s
    | Product layouts ->
      Option.map
        (fun x -> Layout.Const.Product x)
        (Misc.Stdlib.List.map_option aux layouts)
  in
  aux jk.jkind.layout

let get_modal_upper_bounds jk = jk.jkind.modes_upper_bounds

let get_externality_upper_bound jk = jk.jkind.externality_upper_bound

let set_externality_upper_bound jk externality_upper_bound =
  { jk with jkind = { jk.jkind with externality_upper_bound } }

(*********************************)
(* pretty printing *)

let format ppf jkind =
  let rec format_desc nested ppf (d : Desc.t) =
    match d with
    | Const c -> Format.fprintf ppf "%a" Const.format c
    | Var v -> Format.fprintf ppf "%s" (Sort.Var.name v)
    | Product p ->
      Format.fprintf ppf "@[%a@]"
        (Misc.pp_parens_if nested
           (Format.pp_print_list
              ~pp_sep:(fun ppf () -> Format.fprintf ppf "@ & ")
              (format_desc true)))
        p
  in
  format_desc false ppf (get jkind)

let printtyp_path = ref (fun _ _ -> assert false)

let set_printtyp_path f = printtyp_path := f

module Report_missing_cmi : sig
  (* used both in format_history and in Violation.report_general *)
  val report_missing_cmi : Format.formatter -> Path.t option -> unit
end = struct
  open Format

  (* CR layouts: Remove this horrible (but useful) heuristic once we have
     transitive dependencies in jenga. *)
  let missing_cmi_hint ppf type_path =
    let root_module_name p = p |> Path.head |> Ident.name in
    let delete_trailing_double_underscore s =
      if Misc.Stdlib.String.ends_with ~suffix:"__" s
      then String.sub s 0 (String.length s - 2)
      else s
    in
    (* A heuristic for guessing at a plausible library name for an identifier
       with a missing .cmi file; definitely less likely to be right outside of
       Jane Street. *)
    let guess_library_name : Path.t -> string option = function
      | Pdot _ as p ->
        Some
          (match root_module_name p with
          | "Location" | "Longident" -> "ocamlcommon"
          | mn ->
            mn |> String.lowercase_ascii |> delete_trailing_double_underscore)
      | Pident _ | Papply _ | Pextra_ty _ -> None
    in
    Option.iter
      (fprintf ppf "@,Hint: Adding \"%s\" to your dependencies might help.")
      (guess_library_name type_path)

  let report_missing_cmi ppf = function
    | Some p ->
      fprintf ppf "@,@[No .cmi file found containing %a.%a@]" !printtyp_path p
        missing_cmi_hint p
    | None -> ()
end

include Report_missing_cmi

(* CR layouts: should this be configurable? In the meantime, you
   may want to change these to experiment / debug. *)

(* should we print histories at all? *)
let display_histories = true

(* should we print histories in a way users can understand?
   The alternative is to print out all the data, which may be useful
   during debugging. *)
let flattened_histories = true

(* This module is just to keep all the helper functions more locally
   scoped. *)
module Format_history : sig
  val format_history :
    intro:(Format.formatter -> unit) -> Format.formatter -> t -> unit
end = struct
  (* CR layouts: all the output in this section is subject to change;
     actually look closely at error messages once this is activated *)

  open Format

  let format_with_notify_js ppf str =
    fprintf ppf
      "@[%s.@ Please notify the Jane Street compilers group if you see this \
       output@]"
      str

  let format_position ~arity position =
    let to_ordinal num = Int.to_string num ^ Misc.ordinal_suffix num in
    match arity with 1 -> "" | _ -> to_ordinal position ^ " "

  let format_concrete_jkind_reason ppf : History.concrete_jkind_reason -> unit =
    function
    | Match -> fprintf ppf "a value of this type is matched against a pattern"
    | Constructor_declaration _ ->
      fprintf ppf "it's the type of a constructor field"
    | Label_declaration lbl ->
      fprintf ppf "it is the type of record field %s" (Ident.name lbl)
    | Unannotated_type_parameter path ->
      fprintf ppf "it instantiates an unannotated type parameter of %a"
        !printtyp_path path
    | Record_projection ->
      fprintf ppf "it's the record type used in a projection"
    | Record_assignment ->
      fprintf ppf "it's the record type used in an assignment"
    | Let_binding -> fprintf ppf "it's the type of a variable bound by a `let`"
    | Function_argument ->
      fprintf ppf "we must know concretely how to pass a function argument"
    | Function_result ->
      fprintf ppf "we must know concretely how to return a function result"
    | Structure_item_expression ->
      fprintf ppf "it's the type of an expression in a structure"
    | External_argument ->
      fprintf ppf "it's the type of an argument in an external declaration"
    | External_result ->
      fprintf ppf "it's the type of the result of an external declaration"
    | Statement -> fprintf ppf "it's the type of a statement"
    | Wildcard -> fprintf ppf "it's a _ in the type"
    | Unification_var -> fprintf ppf "it's a fresh unification variable"
    | Optional_arg_default ->
      fprintf ppf "it's the type of an optional argument default"
    | Layout_poly_in_external ->
      fprintf ppf
        "it's the layout polymorphic type in an external declaration@ \
         ([@@layout_poly] forces all variables of layout 'any' to be@ \
         representable at call sites)"
    | Array_element -> fprintf ppf "it's the type of an array element"
    | Unboxed_tuple_element ->
      fprintf ppf "it's the type of unboxed tuple element"

  let rec format_annotation_context ppf : History.annotation_context -> unit =
    function
    | Type_declaration p ->
      fprintf ppf "the declaration of the type %a" !printtyp_path p
    | Type_parameter (path, var) ->
      let var_string = match var with None -> "_" | Some v -> "'" ^ v in
      fprintf ppf "@[%s@ in the declaration of the type@ %a@]" var_string
        !printtyp_path path
    | Newtype_declaration name ->
      fprintf ppf "the abstract type declaration for %s" name
    | Constructor_type_parameter (cstr, name) ->
      fprintf ppf "@[%s@ in the declaration of constructor@ %a@]" name
        !printtyp_path cstr
    | Univar name -> fprintf ppf "the universal variable %s" name
    | Type_variable name -> fprintf ppf "the type variable %s" name
    | Type_wildcard loc ->
      fprintf ppf "the wildcard _ at %a" Location.print_loc_in_lowercase loc
    | With_error_message (_message, context) ->
      (* message gets printed in [format_flattened_history] so we ignore it here *)
      format_annotation_context ppf context

  let format_any_creation_reason ppf : History.any_creation_reason -> unit =
    function
    | Missing_cmi p ->
      fprintf ppf "the .cmi file for %a is missing" !printtyp_path p
    | Wildcard -> format_with_notify_js ppf "there's a _ in the type"
    | Unification_var ->
      format_with_notify_js ppf "it's a fresh unification variable"
    | Initial_typedecl_env ->
      format_with_notify_js ppf
        "a dummy layout of any is used to check mutually recursive datatypes"
    | Dummy_jkind ->
      format_with_notify_js ppf
        "it's assigned a dummy layout that should have been overwritten"
    (* CR layouts: Improve output or remove this constructor ^^ *)
    | Type_expression_call ->
      format_with_notify_js ppf
        "there's a call to [type_expression] via the ocaml API"
    | Inside_of_Tarrow -> fprintf ppf "argument or result of a function type"
    | Array_type_argument ->
      fprintf ppf "it's the type argument to the array type"

  let format_immediate_creation_reason ppf :
      History.immediate_creation_reason -> _ = function
    | Empty_record ->
      fprintf ppf "it's a record type containing all void elements"
    | Enumeration ->
      fprintf ppf
        "it's an enumeration variant type (all constructors are constant)"
    | Primitive id ->
      fprintf ppf "it is the primitive immediate type %s" (Ident.name id)
    | Immediate_polymorphic_variant ->
      fprintf ppf
        "it's an enumeration variant type (all constructors are constant)"

  let format_immediate64_creation_reason ppf :
      History.immediate64_creation_reason -> _ = function
    | Separability_check ->
      fprintf ppf "the check that a type is definitely not `float`"

  let format_value_creation_reason ppf : History.value_creation_reason -> _ =
    function
    | Class_let_binding ->
      fprintf ppf "it's the type of a let-bound variable in a class expression"
    | Tuple_element -> fprintf ppf "it's the type of a tuple element"
    | Probe -> format_with_notify_js ppf "it's a probe"
    | Object -> fprintf ppf "it's the type of an object"
    | Instance_variable -> fprintf ppf "it's the type of an instance variable"
    | Object_field -> fprintf ppf "it's the type of an object field"
    | Class_field -> fprintf ppf "it's the type of a class field"
    | Boxed_record -> fprintf ppf "it's a boxed record type"
    | Boxed_variant -> fprintf ppf "it's a boxed variant type"
    | Extensible_variant -> fprintf ppf "it's an extensible variant type"
    | Primitive id ->
      fprintf ppf "it is the primitive value type %s" (Ident.name id)
    | Type_argument { parent_path; position; arity } ->
      fprintf ppf "the %stype argument of %a has layout value"
        (format_position ~arity position)
        !printtyp_path parent_path
    | Tuple -> fprintf ppf "it's a tuple type"
    | Row_variable -> format_with_notify_js ppf "it's a row variable"
    | Polymorphic_variant -> fprintf ppf "it's a polymorphic variant type"
    | Arrow -> fprintf ppf "it's a function type"
    | Tfield ->
      format_with_notify_js ppf
        "it's an internal Tfield type (you shouldn't see this)"
    | Tnil ->
      format_with_notify_js ppf
        "it's an internal Tnil type (you shouldn't see this)"
    | First_class_module -> fprintf ppf "it's a first-class module type"
    | Separability_check ->
      fprintf ppf "the check that a type is definitely not `float`"
    | Univar ->
      fprintf ppf "it is or unifies with an unannotated universal variable"
    | Polymorphic_variant_field ->
      fprintf ppf "it's the type of the field of a polymorphic variant"
    | Default_type_jkind ->
      fprintf ppf "an abstract type has the value layout by default"
    | Existential_type_variable ->
      fprintf ppf "it's an unannotated existential type variable"
    | Array_comprehension_element ->
      fprintf ppf "it's the element type of array comprehension"
    | Lazy_expression -> fprintf ppf "it's the type of a lazy expression"
    | Class_type_argument ->
      fprintf ppf "it's a type argument to a class constructor"
    | Class_term_argument ->
      fprintf ppf
        "it's the type of a term-level argument to a class constructor"
    | Structure_element ->
      fprintf ppf "it's the type of something stored in a module structure"
    | Debug_printer_argument ->
      format_with_notify_js ppf
        "it's the type of an argument to a debugger printer function"
    | V1_safety_check ->
      fprintf ppf "it has to be value for the V1 safety check"
    | Captured_in_object ->
      fprintf ppf "it's the type of a variable captured in an object"
    | Recmod_fun_arg ->
      fprintf ppf
        "it's the type of the first argument to a function in a recursive \
         module"
    | Unknown s ->
      fprintf ppf
        "unknown @[(please alert the Jane Street@;\
         compilers team with this message: %s)@]" s

  let format_float64_creation_reason ppf : History.float64_creation_reason -> _
      = function
    | Primitive id ->
      fprintf ppf "it is the primitive float64 type %s" (Ident.name id)

  let format_float32_creation_reason ppf : History.float32_creation_reason -> _
      = function
    | Primitive id ->
      fprintf ppf "it is the primitive float32 type %s" (Ident.name id)

  let format_word_creation_reason ppf : History.word_creation_reason -> _ =
    function
    | Primitive id ->
      fprintf ppf "it is the primitive word type %s" (Ident.name id)

  let format_bits32_creation_reason ppf : History.bits32_creation_reason -> _ =
    function
    | Primitive id ->
      fprintf ppf "it is the primitive bits32 type %s" (Ident.name id)

  let format_bits64_creation_reason ppf : History.bits64_creation_reason -> _ =
    function
    | Primitive id ->
      fprintf ppf "it is the primitive bits64 type %s" (Ident.name id)

  let format_product_creation_reason ppf : History.product_creation_reason -> _
      = function
    | Unboxed_tuple -> fprintf ppf "it is an unboxed tuple"

  let format_creation_reason ppf : History.creation_reason -> unit = function
    | Annotated (ctx, _) ->
      fprintf ppf "of the annotation on %a" format_annotation_context ctx
    | Missing_cmi p ->
      fprintf ppf "the .cmi file for %a is missing" !printtyp_path p
    | Any_creation any -> format_any_creation_reason ppf any
    | Immediate_creation immediate ->
      format_immediate_creation_reason ppf immediate
    | Immediate64_creation immediate64 ->
      format_immediate64_creation_reason ppf immediate64
    | Void_creation _ -> .
    | Value_creation value -> format_value_creation_reason ppf value
    | Float64_creation float -> format_float64_creation_reason ppf float
    | Float32_creation float -> format_float32_creation_reason ppf float
    | Word_creation word -> format_word_creation_reason ppf word
    | Bits32_creation bits32 -> format_bits32_creation_reason ppf bits32
    | Bits64_creation bits64 -> format_bits64_creation_reason ppf bits64
    | Product_creation product -> format_product_creation_reason ppf product
    | Concrete_creation concrete -> format_concrete_jkind_reason ppf concrete
    | Imported ->
      fprintf ppf "of layout requirements from an imported definition"
    | Imported_type_argument { parent_path; position; arity } ->
      fprintf ppf "the %stype argument of %a has this layout"
        (format_position ~arity position)
        !printtyp_path parent_path
    | Generalized (id, loc) ->
      let format_id ppf = function
        | Some id -> fprintf ppf " of %s" (Ident.name id)
        | None -> ()
      in
      fprintf ppf "of the definition%a at %a" format_id id
        Location.print_loc_in_lowercase loc

  let format_interact_reason ppf : History.interact_reason -> _ = function
    | Gadt_equation name ->
      fprintf ppf "a GADT match refining the type %a" !printtyp_path name
    | Tyvar_refinement_intersection -> fprintf ppf "updating a type variable"
    | Subjkind -> fprintf ppf "sublayout check"

  (* CR layouts: An older implementation of format_flattened_history existed
      which displays more information not limited to one layout and one creation_reason
      around commit 66a832d70bf61d9af3b0ec6f781dcf0a188b324d in main.

      Consider revisiting that if the current implementation becomes insufficient. *)

  let format_flattened_history ~intro ppf t =
    let jkind_desc = Jkind_desc.get t.jkind in
    fprintf ppf "@[<v 2>%t" intro;
    (match t.history with
    | Creation reason -> (
      fprintf ppf ", because@ %a" format_creation_reason reason;
      match reason, jkind_desc with
      | Concrete_creation _, Const _ ->
        fprintf ppf ", defaulted to layout %a" Desc.format jkind_desc
      | _ -> ())
    | _ -> assert false);
    fprintf ppf ".";
    (match t.history with
    | Creation (Annotated (With_error_message (message, _), _)) ->
      fprintf ppf "@ @[%s@]" message
    | _ -> ());
    fprintf ppf "@]"

  (* this isn't really formatted for user consumption *)
  let format_history_tree ~intro ppf t =
    let rec in_order ppf = function
      | Interact
          { reason; lhs_history; rhs_history; lhs_jkind = _; rhs_jkind = _ } ->
        fprintf ppf "@[<v 2>  %a@]@;%a@ @[<v 2>  %a@]" in_order lhs_history
          format_interact_reason reason in_order rhs_history
      | Creation c -> format_creation_reason ppf c
    in
    fprintf ppf "@;%t has this layout history:@;@[<v 2>  %a@]" intro in_order
      t.history

  let format_history ~intro ppf t =
    if display_histories
    then
      if flattened_histories
      then format_flattened_history ~intro ppf t
      else format_history_tree ~intro ppf t
end

include Format_history

(******************************)
(* errors *)

module Violation = struct
  open Format

  type violation =
    | Not_a_subjkind of t * t
    | No_intersection of t * t

  type nonrec t =
    { violation : violation;
      missing_cmi : Path.t option
    }
  (* [missing_cmi]: is this error a result of a missing cmi file?
     This is stored separately from the [violation] because it's
     used to change the behavior of [value_kind], and we don't
     want that function to inspect something that is purely about
     the choice of error message. (Though the [Path.t] payload *is*
     indeed just about the payload.) *)

  let of_ ?missing_cmi violation = { violation; missing_cmi }

  let is_missing_cmi viol = Option.is_some viol.missing_cmi

  let report_general preamble pp_former former ppf t =
    let subjkind_format verb l2 =
      match get l2 with
      | Var _ -> dprintf "%s representable" verb
      | Const _ -> dprintf "%s a sublayout of %a" verb format l2
      | Product _ -> dprintf "idk some error man"
      (* CR ccasinghino *)
    in
    let l1, l2, fmt_l1, fmt_l2, missing_cmi_option =
      match t with
      | { violation = Not_a_subjkind (l1, l2); missing_cmi } -> (
        let missing_cmi =
          match missing_cmi with
          | None -> (
            match l1.history with
            | Creation (Missing_cmi p) -> Some p
            | Creation (Any_creation (Missing_cmi p)) -> Some p
            | _ -> None)
          | Some _ -> missing_cmi
        in
        match missing_cmi with
        | None ->
          ( l1,
            l2,
            dprintf "layout %a" format l1,
            subjkind_format "is not" l2,
            None )
        | Some p ->
          ( l1,
            l2,
            dprintf "an unknown layout",
            subjkind_format "might not be" l2,
            Some p ))
      | { violation = No_intersection (l1, l2); missing_cmi } ->
        assert (Option.is_none missing_cmi);
        ( l1,
          l2,
          dprintf "layout %a" format l1,
          dprintf "does not overlap with %a" format l2,
          None )
    in
    if display_histories
    then
      let connective =
        match t.violation, get l2 with
        | Not_a_subjkind _, (Const _ | Product _) ->
          dprintf "be a sublayout of %a" format l2
        | No_intersection _, (Const _ | Product _) ->
          dprintf "overlap with %a" format l2
        | _, Var _ -> dprintf "be representable"
      in
      fprintf ppf "@[<v>%a@;%a@]"
        (format_history
           ~intro:(dprintf "The layout of %a is %a" pp_former former format l1))
        l1
        (format_history
           ~intro:
             (dprintf "But the layout of %a must %t" pp_former former connective))
        l2
    else
      fprintf ppf "@[<hov 2>%s%a has %t,@ which %t.@]" preamble pp_former former
        fmt_l1 fmt_l2;
    report_missing_cmi ppf missing_cmi_option

  let pp_t ppf x = fprintf ppf "%t" x

  let report_with_offender ~offender = report_general "" pp_t offender

  let report_with_offender_sort ~offender =
    report_general "A representable layout was expected, but " pp_t offender

  let report_with_name ~name = report_general "" pp_print_string name
end

(******************************)
(* relations *)

let equate_or_equal ~allow_mutation
    { jkind = jkind1; history = _; has_warned = _ }
    { jkind = jkind2; history = _; has_warned = _ } =
  Jkind_desc.equate_or_equal ~allow_mutation jkind1 jkind2

(* CR layouts v2.8: Switch this back to ~allow_mutation:false *)
let equal = equate_or_equal ~allow_mutation:true

let () = Types.set_jkind_equal equal

let equate = equate_or_equal ~allow_mutation:true

(* Not all jkind history reasons are created equal. Some are more helpful than others.
    This function encodes that information.

    The reason with higher score should get preserved when combined with one of lower
    score. *)
let score_reason = function
  (* error_message annotated by the user should always take priority *)
  | Creation (Annotated (With_error_message _, _)) -> 1
  (* Concrete creation is quite vague, prefer more specific reasons *)
  | Creation (Concrete_creation _) -> -1
  | _ -> 0

let combine_histories reason lhs rhs =
  if flattened_histories
  then
    match Desc.sub (Jkind_desc.get lhs.jkind) (Jkind_desc.get rhs.jkind) with
    | Less -> lhs.history
    | Not_le ->
      rhs.history
      (* CR layouts: this will be wrong if we ever have a non-trivial meet in the layout lattice *)
    | Equal ->
      if score_reason lhs.history >= score_reason rhs.history
      then lhs.history
      else rhs.history
  else
    Interact
      { reason;
        lhs_jkind = lhs.jkind;
        lhs_history = lhs.history;
        rhs_jkind = rhs.jkind;
        rhs_history = rhs.history
      }

let has_intersection t1 t2 =
  Option.is_some (Jkind_desc.intersection t1.jkind t2.jkind)

let intersection_or_error ~reason t1 t2 =
  match Jkind_desc.intersection t1.jkind t2.jkind with
  | None -> Error (Violation.of_ (No_intersection (t1, t2)))
  | Some jkind ->
    Ok
      { jkind;
        history = combine_histories reason t1 t2;
        has_warned = t1.has_warned || t2.has_warned
      }

(* this is hammered on; it must be fast! *)
let check_sub sub super = Jkind_desc.sub sub.jkind super.jkind

let sub sub super = Misc.Le_result.is_le (check_sub sub super)

let sub_or_error t1 t2 =
  if sub t1 t2 then Ok () else Error (Violation.of_ (Not_a_subjkind (t1, t2)))

let sub_with_history sub super =
  match check_sub sub super with
  | Less | Equal ->
    Ok { sub with history = combine_histories Subjkind sub super }
  | Not_le -> Error (Violation.of_ (Not_a_subjkind (sub, super)))

let is_void_defaulting = function
  | { jkind = { layout = Sort s; _ }; _ } -> Sort.is_void_defaulting s
  | _ -> false

(* This doesn't do any mutation because mutating a sort variable can't make it
   any, and modal upper bounds are constant. *)
let is_max jkind = sub Primitive.any_dummy_jkind jkind

let has_layout_any jkind =
  match jkind.jkind.layout with Any -> true | _ -> false

(*********************************)
(* debugging *)

module Debug_printers = struct
  open Format

  let concrete_jkind_reason ppf : History.concrete_jkind_reason -> unit =
    function
    | Match -> fprintf ppf "Match"
    | Constructor_declaration idx ->
      fprintf ppf "Constructor_declaration %d" idx
    | Label_declaration lbl ->
      fprintf ppf "Label_declaration %a" Ident.print lbl
    | Unannotated_type_parameter path ->
      fprintf ppf "Unannotated_type_parameter %a" !printtyp_path path
    | Record_projection -> fprintf ppf "Record_projection"
    | Record_assignment -> fprintf ppf "Record_assignment"
    | Let_binding -> fprintf ppf "Let_binding"
    | Function_argument -> fprintf ppf "Function_argument"
    | Function_result -> fprintf ppf "Function_result"
    | Structure_item_expression -> fprintf ppf "Structure_item_expression"
    | External_argument -> fprintf ppf "External_argument"
    | External_result -> fprintf ppf "External_result"
    | Statement -> fprintf ppf "Statement"
    | Wildcard -> fprintf ppf "Wildcard"
    | Unification_var -> fprintf ppf "Unification_var"
    | Optional_arg_default -> fprintf ppf "Optional_arg_default"
    | Layout_poly_in_external -> fprintf ppf "Layout_poly_in_external"
    | Array_element -> fprintf ppf "Array_element"
    | Unboxed_tuple_element -> fprintf ppf "Unboxed_tuple_element"

  let rec annotation_context ppf : History.annotation_context -> unit = function
    | Type_declaration p -> fprintf ppf "Type_declaration %a" Path.print p
    | Type_parameter (p, var) ->
      fprintf ppf "Type_parameter (%a, %a)" Path.print p
        (Misc.Stdlib.Option.print Misc.Stdlib.String.print)
        var
    | Newtype_declaration name -> fprintf ppf "Newtype_declaration %s" name
    | Constructor_type_parameter (cstr, name) ->
      fprintf ppf "Constructor_type_parameter (%a, %S)" Path.print cstr name
    | Univar name -> fprintf ppf "Univar %S" name
    | Type_variable name -> fprintf ppf "Type_variable %S" name
    | Type_wildcard loc ->
      fprintf ppf "Type_wildcard (%a)" Location.print_loc loc
    | With_error_message (message, context) ->
      fprintf ppf "With_error_message (%s, %a)" message annotation_context
        context

  let any_creation_reason ppf : History.any_creation_reason -> unit = function
    | Missing_cmi p -> fprintf ppf "Missing_cmi %a" Path.print p
    | Initial_typedecl_env -> fprintf ppf "Initial_typedecl_env"
    | Dummy_jkind -> fprintf ppf "Dummy_jkind"
    | Type_expression_call -> fprintf ppf "Type_expression_call"
    | Inside_of_Tarrow -> fprintf ppf "Inside_of_Tarrow"
    | Wildcard -> fprintf ppf "Wildcard"
    | Unification_var -> fprintf ppf "Unification_var"
    | Array_type_argument -> fprintf ppf "Array_type_argument"

  let immediate_creation_reason ppf : History.immediate_creation_reason -> _ =
    function
    | Empty_record -> fprintf ppf "Empty_record"
    | Enumeration -> fprintf ppf "Enumeration"
    | Primitive id -> fprintf ppf "Primitive %s" (Ident.unique_name id)
    | Immediate_polymorphic_variant ->
      fprintf ppf "Immediate_polymorphic_variant"

  let immediate64_creation_reason ppf : History.immediate64_creation_reason -> _
      = function
    | Separability_check -> fprintf ppf "Separability_check"

  let value_creation_reason ppf : History.value_creation_reason -> _ = function
    | Class_let_binding -> fprintf ppf "Class_let_binding"
    | Tuple_element -> fprintf ppf "Tuple_element"
    | Probe -> fprintf ppf "Probe"
    | Object -> fprintf ppf "Object"
    | Instance_variable -> fprintf ppf "Instance_variable"
    | Object_field -> fprintf ppf "Object_field"
    | Class_field -> fprintf ppf "Class_field"
    | Boxed_record -> fprintf ppf "Boxed_record"
    | Boxed_variant -> fprintf ppf "Boxed_variant"
    | Extensible_variant -> fprintf ppf "Extensible_variant"
    | Primitive id -> fprintf ppf "Primitive %s" (Ident.unique_name id)
    | Type_argument { parent_path; position; arity } ->
      fprintf ppf "Type_argument (pos %d, arity %d) of %a" position arity
        !printtyp_path parent_path
    | Tuple -> fprintf ppf "Tuple"
    | Row_variable -> fprintf ppf "Row_variable"
    | Polymorphic_variant -> fprintf ppf "Polymorphic_variant"
    | Arrow -> fprintf ppf "Arrow"
    | Tfield -> fprintf ppf "Tfield"
    | Tnil -> fprintf ppf "Tnil"
    | First_class_module -> fprintf ppf "First_class_module"
    | Separability_check -> fprintf ppf "Separability_check"
    | Univar -> fprintf ppf "Univar"
    | Polymorphic_variant_field -> fprintf ppf "Polymorphic_variant_field"
    | Default_type_jkind -> fprintf ppf "Default_type_jkind"
    | Existential_type_variable -> fprintf ppf "Existential_type_variable"
    | Array_comprehension_element -> fprintf ppf "Array_comprehension_element"
    | Lazy_expression -> fprintf ppf "Lazy_expression"
    | Class_type_argument -> fprintf ppf "Class_type_argument"
    | Class_term_argument -> fprintf ppf "Class_term_argument"
    | Structure_element -> fprintf ppf "Structure_element"
    | Debug_printer_argument -> fprintf ppf "Debug_printer_argument"
    | V1_safety_check -> fprintf ppf "V1_safety_check"
    | Captured_in_object -> fprintf ppf "Captured_in_object"
    | Recmod_fun_arg -> fprintf ppf "Recmod_fun_arg"
    | Unknown s -> fprintf ppf "Unknown %s" s

  let float64_creation_reason ppf : History.float64_creation_reason -> _ =
    function
    | Primitive id -> fprintf ppf "Primitive %s" (Ident.unique_name id)

  let float32_creation_reason ppf : History.float32_creation_reason -> _ =
    function
    | Primitive id -> fprintf ppf "Primitive %s" (Ident.unique_name id)

  let word_creation_reason ppf : History.word_creation_reason -> _ = function
    | Primitive id -> fprintf ppf "Primitive %s" (Ident.unique_name id)

  let bits32_creation_reason ppf : History.bits32_creation_reason -> _ =
    function
    | Primitive id -> fprintf ppf "Primitive %s" (Ident.unique_name id)

  let bits64_creation_reason ppf : History.bits64_creation_reason -> _ =
    function
    | Primitive id -> fprintf ppf "Primitive %s" (Ident.unique_name id)

  let product_creation_reason ppf : History.product_creation_reason -> _ =
    function
    | Unboxed_tuple -> fprintf ppf "Unboxed_tuple"

  let creation_reason ppf : History.creation_reason -> unit = function
    | Annotated (ctx, loc) ->
      fprintf ppf "Annotated (%a,%a)" annotation_context ctx Location.print_loc
        loc
    | Missing_cmi p -> fprintf ppf "Missing_cmi %a" !printtyp_path p
    | Any_creation any -> fprintf ppf "Any_creation %a" any_creation_reason any
    | Immediate_creation immediate ->
      fprintf ppf "Immediate_creation %a" immediate_creation_reason immediate
    | Immediate64_creation immediate64 ->
      fprintf ppf "Immediate64_creation %a" immediate64_creation_reason
        immediate64
    | Value_creation value ->
      fprintf ppf "Value_creation %a" value_creation_reason value
    | Void_creation _ -> .
    | Float64_creation float ->
      fprintf ppf "Float64_creation %a" float64_creation_reason float
    | Float32_creation float ->
      fprintf ppf "Float32_creation %a" float32_creation_reason float
    | Word_creation word ->
      fprintf ppf "Word_creation %a" word_creation_reason word
    | Bits32_creation bits32 ->
      fprintf ppf "Bits32_creation %a" bits32_creation_reason bits32
    | Bits64_creation bits64 ->
      fprintf ppf "Bits64_creation %a" bits64_creation_reason bits64
    | Product_creation product ->
      fprintf ppf "Product_creation %a" product_creation_reason product
    | Concrete_creation concrete ->
      fprintf ppf "Concrete_creation %a" concrete_jkind_reason concrete
    | Imported -> fprintf ppf "Imported"
    | Imported_type_argument { parent_path; position; arity } ->
      fprintf ppf "Imported_type_argument (pos %d, arity %d) of %a" position
        arity !printtyp_path parent_path
    | Generalized (id, loc) ->
      fprintf ppf "Generalized (%s, %a)"
        (match id with Some id -> Ident.unique_name id | None -> "")
        Location.print_loc loc

  let interact_reason ppf : History.interact_reason -> _ = function
    | Gadt_equation p -> fprintf ppf "Gadt_equation %a" Path.print p
    | Tyvar_refinement_intersection ->
      fprintf ppf "Tyvar_refinement_intersection"
    | Subjkind -> fprintf ppf "Subjkind"

  let rec history ppf = function
    | Interact { reason; lhs_jkind; lhs_history; rhs_jkind; rhs_history } ->
      fprintf ppf
        "Interact {@[reason = %a;@ lhs_jkind = %a;@ lhs_history = %a;@ \
         rhs_jkind = %a;@ rhs_history = %a}@]"
        interact_reason reason Jkind_desc.Debug_printers.t lhs_jkind history
        lhs_history Jkind_desc.Debug_printers.t rhs_jkind history rhs_history
    | Creation c -> fprintf ppf "Creation (%a)" creation_reason c

  let t ppf ({ jkind; history = h; has_warned = _ } : t) : unit =
    fprintf ppf "@[<v 2>{ jkind = %a@,; history = %a }@]"
      Jkind_desc.Debug_printers.t jkind history h
end

(*** formatting user errors ***)
let report_error ~loc : Error.t -> _ = function
  | Unknown_jkind jkind ->
    Location.errorf ~loc
      (* CR layouts v2.9: use the context to produce a better error message.
         When RAE tried this, some types got printed like [t/2], but the
         [/2] shouldn't be there. Investigate and fix. *)
      "@[<v>Unknown layout %a@]" Pprintast.jkind jkind
  | Unknown_mode mode ->
    Location.errorf ~loc "@[<v>Unknown mode %a@]" Pprintast.mode mode
  | Multiple_jkinds { from_annotation; from_attribute } ->
    Location.errorf ~loc
      "@[<v>A type declaration's layout can be given at most once.@;\
       This declaration has an layout annotation (%a) and a layout attribute \
       ([@@@@%a]).@]"
      Const.format from_annotation Const.format from_attribute
  | Insufficient_level { jkind; required_layouts_level } -> (
    let hint ppf =
      Format.fprintf ppf "You must enable -extension %s to use this feature."
        (Language_extension.to_command_line_string Layouts
           required_layouts_level)
    in
    match Language_extension.is_enabled Layouts with
    | false ->
      Location.errorf ~loc
        "@[<v>The appropriate layouts extension is not enabled.@;%t@]" hint
    | true ->
      Location.errorf ~loc
        (* CR layouts errors: use the context to produce a better error message.
           When RAE tried this, some types got printed like [t/2], but the
           [/2] shouldn't be there. Investigate and fix. *)
        "@[<v>Layout %a is more experimental than allowed by the enabled \
         layouts extension.@;\
         %t@]"
        Const.format jkind hint)

let () =
  Location.register_error_of_exn (function
    | Error.User_error (loc, err) -> Some (report_error ~loc err)
    | _ -> None)

(* CR layouts v2.8: Remove the definitions below by propagating changes
   outside of this file. *)

type annotation = Const.t * Jane_syntax.Jkind.annotation

let default_to_value_and_get t = default_to_value_and_get t

(* CR ccasinghino: move these to the appropriate places in the unlikely event we
   keep them.  Also is this just layout.sub on some new sort variables now? Or
   at least the sort one is sort.equate? *)
let constrain_sort_to_nary_product n (s : Sort.t) =
  match Sort.get s with
  | Base _ -> None
  | Var v ->
    let vars = List.init n (fun _ -> Sort.new_var ()) in
    Sort.set v (Some (Sort.Product vars));
    Some vars
  | Product sorts ->
    if List.compare_length_with sorts n = 0 then Some sorts else None

let constrain_layout_to_nary_product n (layout : Layout.t) =
  match layout with
  | Any -> None
  | Sort s ->
    Option.map
      (List.map (fun x -> Jkind_types.Layout.Sort x))
      (constrain_sort_to_nary_product n s)
  | Product ts -> if List.compare_length_with ts n = 0 then Some ts else None

let constrain_jkind_to_nary_product n t =
  match constrain_layout_to_nary_product n t.jkind.layout with
  | None -> None
  | Some layouts ->
    Some
      (List.map
         (fun layout -> { t with jkind = { t.jkind with layout } })
         layouts)
