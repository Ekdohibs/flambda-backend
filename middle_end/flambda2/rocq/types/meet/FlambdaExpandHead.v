(******************************************************************************
 *                                  OxCaml                                    *
 * -------------------------------------------------------------------------- *
 *                               MIT License                                  *
 *                                                                            *
 * Copyright (c) 2026 Jane Street Group LLC                                   *
 * opensource-contacts@janestreet.com                                         *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included    *
 * in all copies or substantial portions of the Software.                     *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER *
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    *
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        *
 * DEALINGS IN THE SOFTWARE.                                                  *
 ******************************************************************************)



(* Expanded types, as the model sees them: a type that is not an alias,
   given by its kind and its head.

   This file is to be reviewed, line by line against the module
   [Expanded_type] of middle_end/flambda2/types/expand_head.ml, as
   FlambdaTypeGrammar.v is against type_grammar.ml. It holds definitions
   only and it extracts. Meet works on expanded types
   ([Meet_env.meet_expanded_head], FlambdaMeetEnv.v), and the join that
   meet is given calls it on them.

   What the model has of [Expanded_type]:
   - [descr]: the fifteen constructors, one for each kind, are five
     here, as those of [Type_grammar.t] are in FlambdaTypeGrammar.v: the
     eleven of the naked numbers other than the immediates are the one
     [Head_naked_number], which carries the kind. The type is named
     [head] here, [descr] being the name of the field.
   - [t]: in full, a kind and [Unknown], [Bottom] or a head.
   - [to_type], [of_non_alias_type]: in full, but that a [Simple] has no
     coercion (FlambdaModel.v), so [of_non_alias_type] has no
     [?coercion] and [Type_grammar.apply_coercion_head_of_kind_value]
     is not there. [None] is the fatal error of the compiler for a type
     that is an alias.
   - The creation functions: [create_value], [create_naked_immediate],
     [create_rec_info], [create_region], [create_bottom],
     [create_unknown] in full; the eleven for the other naked numbers
     are the one [create_naked_number]. [create_const]: in full; the
     type of a poison constant is the unknown one, as in the compiler.
   - [bottom_like], [unknown_like], [is_bottom], [is_unknown]: in full.
     [is_unknown_maybe_null] and [descr_oub] are left out: meet does
     not use them.
   After the module: [expand_head_of_alias_type], [expand_head0] and
   [expand_head], of the same file. [None] is a fatal error of the
   compiler: a name that is not bound, or a canonical name whose type
   is an alias ("Canonical alias ... should never have [Equals]
   type"). The rest of the file ([make_suitable_for_environment],
   [is_useful], ...) is not here.

   Nothing asks that the head of an expanded type be of its kind: the
   compiler's creation functions give that, and those here. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar FlambdaBindingTime
  FlambdaEnvAliases FlambdaTypingEnv FlambdaMeetGrammar.
Import ListNotations.

(* [Expand_head.Expanded_type]. *)
Module Expanded_type.

  (* [Expanded_type.descr]. *)
  Inductive head : Type :=
    | Head_value (head : head_of_kind_value)
    | Head_naked_immediate (head : head_of_kind_naked_immediate)
    | Head_naked_number (k : naked_number_kind) (head : list Z)
    | Head_rec_info (head : unit)
    | Head_region (head : unit).

  (* [Expanded_type.t]. *)
  Record t : Type := create {
    kind : flambda_kind;
    descr : or_unknown_or_bottom head
  }.

  Definition create_bottom (kind : flambda_kind) : t := create kind Bottom.

  Definition create_unknown (kind : flambda_kind) : t := create kind Unknown.

  Definition create_value (head : head_of_kind_value) : t :=
    create Value (Ok (Head_value head)).

  Definition create_naked_immediate (head : head_of_kind_naked_immediate)
      : t :=
    create (Naked_number Kind_naked_immediate) (Ok (Head_naked_immediate head)).

  (* [create_naked_float32], [create_naked_float], ..., [create_naked_mask]. *)
  Definition create_naked_number (k : naked_number_kind) (head : list Z) : t :=
    create (Naked_number k) (Ok (Head_naked_number k head)).

  Definition create_rec_info (head : unit) : t :=
    create Rec_info (Ok (Head_rec_info head)).

  Definition create_region (head : unit) : t :=
    create Region (Ok (Head_region head)).

  (* [Expanded_type.create_const]. *)
  Definition create_const (const : const) : t :=
    match const with
    | Naked_immediate i =>
      create_naked_immediate
        (Head_of_kind_naked_immediate (Or_unknown.Known [i]) [])
    | Tagged_immediate i =>
      create_value (head_of_kind_value_tagged_immediate i)
    | Naked_float32 f => create_naked_number Kind_naked_float32 [f]
    | Naked_float f => create_naked_number Kind_naked_float [f]
    | Naked_int8 i => create_naked_number Kind_naked_int8 [i]
    | Naked_int16 i => create_naked_number Kind_naked_int16 [i]
    | Naked_int32 i => create_naked_number Kind_naked_int32 [i]
    | Naked_int64 i => create_naked_number Kind_naked_int64 [i]
    | Naked_nativeint i => create_naked_number Kind_naked_nativeint [i]
    | Naked_vec128 i => create_naked_number Kind_naked_vec128 [i]
    | Naked_vec256 i => create_naked_number Kind_naked_vec256 [i]
    | Naked_vec512 i => create_naked_number Kind_naked_vec512 [i]
    | Naked_mask i => create_naked_number Kind_naked_mask [i]
    | Null => create_value head_of_kind_value_null
    | Poison kind => create_unknown kind
    end.

  Definition bottom_like (expanded : t) : t := create_bottom (kind expanded).

  Definition unknown_like (expanded : t) : t :=
    create_unknown (kind expanded).

  Definition is_bottom (expanded : t) : bool :=
    match descr expanded with
    | Bottom => true
    | Unknown | Ok _ => false
    end.

  Definition is_unknown (expanded : t) : bool :=
    match descr expanded with
    | Unknown => true
    | Bottom | Ok _ => false
    end.

  (* [Expanded_type.to_type]. *)
  Definition to_type (expanded : t) : flambda_type :=
    match descr expanded with
    | Unknown => unknown (kind expanded)
    | Bottom => bottom (kind expanded)
    | Ok (Head_value head) => Type_value (Ok (No_alias head))
    | Ok (Head_naked_immediate head) =>
      Type_naked_immediate (Ok (No_alias head))
    | Ok (Head_naked_number k head) => Type_naked_number k (Ok (No_alias head))
    | Ok (Head_rec_info head) => Type_rec_info (Ok (No_alias head))
    | Ok (Head_region head) => Type_region (Ok (No_alias head))
    end.

  (* [Expanded_type.of_non_alias_type]. [None] is a fatal error of the
     compiler: the type is an alias. *)
  Definition of_non_alias_type (ty : flambda_type) : option t :=
    let of_descr {A : Type} (k : flambda_kind) (make : A -> head)
        (ty : type_descr A) : option t :=
      match ty with
      | Unknown => Some (create_unknown k)
      | Bottom => Some (create_bottom k)
      | Ok (No_alias head) => Some (create k (Ok (make head)))
      | Ok (Equals _) => None
      end
    in
    match ty with
    | Type_value ty => of_descr Value Head_value ty
    | Type_naked_immediate ty =>
      of_descr (Naked_number Kind_naked_immediate) Head_naked_immediate ty
    | Type_naked_number k ty =>
      of_descr (Naked_number k) (Head_naked_number k) ty
    | Type_rec_info ty => of_descr Rec_info Head_rec_info ty
    | Type_region ty => of_descr Region Head_region ty
    end.

End Expanded_type.

(* [Expand_head.expand_head_of_alias_type]: the type that the
   environment has for a canonical element, which is not an alias. *)
Definition expand_head_of_alias_type (tenv : Typing_env.t)
    (kind : flambda_kind)
    (known_canonical_simple_at_in_types_mode : simple)
    : option Expanded_type.t :=
  let of_name (name : name) : option Expanded_type.t :=
    let* ty := Typing_env.find tenv name (Some kind) in
    Expanded_type.of_non_alias_type ty
  in
  match known_canonical_simple_at_in_types_mode with
  | Const const => Some (Expanded_type.create_const const)
  | Var x => of_name (Name_var x)
  | Symbol s => of_name (Name_symbol s)
  end.

(* [Expand_head.expand_head0]: the head of a type, given the canonical
   element of the name it is an alias to if it is one. With none given,
   an alias expands to the unknown type. *)
Definition expand_head0 (tenv : Typing_env.t) (ty : flambda_type)
    (known_canonical_simple_at_in_types_mode : option simple)
    : option Expanded_type.t :=
  match get_alias_opt ty with
  | None => Expanded_type.of_non_alias_type ty
  | Some _ =>
    match known_canonical_simple_at_in_types_mode with
    | Some simple => expand_head_of_alias_type tenv (kind ty) simple
    | None => Expanded_type.of_non_alias_type (unknown (kind ty))
    end
  end.

(* [Expand_head.expand_head]. *)
Definition expand_head (tenv : Typing_env.t) (ty : flambda_type)
    : option Expanded_type.t :=
  match get_alias_opt ty with
  | None => Expanded_type.of_non_alias_type ty
  | Some simple =>
    let kind := kind ty in
    let* canonical :=
      Typing_env.get_canonical_simple_exn tenv (Some Name_mode.in_types) None
        simple
    in
    match canonical with
    | None => Expanded_type.of_non_alias_type (unknown kind)
    | Some simple => expand_head_of_alias_type tenv kind simple
    end
  end.
