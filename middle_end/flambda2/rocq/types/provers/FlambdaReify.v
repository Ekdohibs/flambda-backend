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



(* Reification, as the model sees it: the structure that a type is
   turned into, from which Simplify builds a term.

   This file is to be reviewed, line by line against
   middle_end/flambda2/types/reify.ml. It holds definitions only, in a
   module [Reify], so that [Reify.reify] is that function; and it
   extracts. What an answer means is FlambdaReifySpec.v.

   What is here, of reify.ml (the lines are of that file):
   - [to_lift] and [reification_result], each in a module of its name
     ([To_lift.t], [Reification_result.t]), so that their constructors
     are told apart from those of the grammar of types and of the
     answers of the provers (25 to 64);
   - [try_to_reify_fields] (66 to 91);
   - [reify] (222 to 793), without the cases of arrays.

   How the definitions differ from the compiler's.
   - A stop of the compiler is [None], as in FlambdaProvers.v: where
     [Expand_head.expand_head], [Typing_env.find] or a prover stops,
     and where reify.ml itself has an assertion or a fatal error (the
     size of a block against the number of the types of its fields;
     the kinds of a mixed shape against those types, which
     [List.combine] takes to be as many; a tag that is not that of a
     scannable block).
   - A head has no [Array] (FlambdaTypeGrammar.v), so the two cases of
     arrays, the functor [Make_lift_array_of_naked_numbers] and its
     twelve instances are not here, and [to_lift] has not the
     fourteen constructors of arrays.
   - The nine [Boxed_float32], ..., [Boxed_mask] of [to_lift] are the
     one [To_lift.Boxed_number], which carries the number boxed
     ([boxable_number]) with the number; the nine cases of [reify] for
     a boxed number are one, which asks [Provers.meet_naked_number] at
     the kind of the contents. The eleven cases for a type of naked
     numbers are one, and the constant it answers is
     [const_of_naked_number] (FlambdaTypeGrammar.v).
   - A [Simple] of the model has no coercion, so the case of
     [try_to_reify_fields] for a [Simple] with a coercion is not here.
   - [try_to_reify_fields] filters the fields with a function that
     may stop: [reify_field] is that function, whose inner [None] is a
     field that is not reified, and [reify_fields] the filter. The
     local function [var_allowed] of [reify] is [var_allowed]; it
     answers an [option], as [Provers.never_holds_locally_allocated_values]
     may stop.
   - The types of the fields with the kinds expected of them are made
     by [field_kinds] (FlambdaTypeGrammar.v) for the three shapes.
   - The [match] of [reify] on the expanded type is a function,
     [reify_expanded], which takes what the local function
     [try_canonical_simple] answers. Its cases are in the compiler's
     order but for those that answer the same: every case that only
     tries the canonical [Simple] is the last.
   - [unboxed_number_kind] is [Flambda_kind.Boxable_number.unboxed_kind]
     as a kind of naked numbers, where FlambdaKindModel.v has it as a
     kind.
   - The arguments are in the compiler's order; a labelled argument is
     an ordinary one. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeAccessSpec FlambdaBindingTime FlambdaEnvAliases
  FlambdaTypingEnv FlambdaExpandHead FlambdaProvers.
Import ListNotations.

(* [Reify.to_lift]. *)
Module To_lift.
  Inductive t : Type :=
    | Immutable_block (tag : Z) (is_unique : bool)
        (shape : scannable_block_shape) (fields : list simple)
    | Boxed_number (k : boxable_number) (n : Z).
End To_lift.

(* [Reify.reification_result]. *)
Module Reification_result.
  Inductive t : Type :=
    | Lift (to_lift : To_lift.t)
    | Simple (simple : simple)
    | Cannot_reify
    | Invalid.
End Reification_result.

Module Reify.

  (* The function that [Reify.try_to_reify_fields] filters the fields
     with: the [Simple] that a field is, if it is one that may be
     lifted. The inner [None] is a field that is not reified. *)
  Definition reify_field (tenv : Typing_env.t)
      (var_allowed : Alloc_mode.t -> variable -> option bool)
      (alloc_mode : Alloc_mode.t) (field : flambda_type * flambda_kind)
      : option (option simple) :=
    let* proof :=
      Provers.prove_equals_to_simple_of_kind tenv (fst field) (snd field)
    in
    match proof with
    | Proof_of_property.Proved simple =>
      match simple with
      | Var var =>
        let* allowed := var_allowed alloc_mode var in
        if allowed then Some (Some simple) else Some None
      | Symbol _ | Const _ => Some (Some simple)
      end
    | Proof_of_property.Unknown => Some None
    end.

  (* [List.filter_map] with [reify_field]. *)
  Fixpoint reify_fields (tenv : Typing_env.t)
      (var_allowed : Alloc_mode.t -> variable -> option bool)
      (alloc_mode : Alloc_mode.t)
      (fields : list (flambda_type * flambda_kind)) : option (list simple) :=
    match fields with
    | [] => Some []
    | field :: fields =>
      let* simple := reify_field tenv var_allowed alloc_mode field in
      let* simples := reify_fields tenv var_allowed alloc_mode fields in
      match simple with
      | Some simple => Some (simple :: simples)
      | None => Some simples
      end
    end.

  (* [Reify.try_to_reify_fields]: the [Simple]s of the fields, if every
     field is reified. *)
  Definition try_to_reify_fields (tenv : Typing_env.t)
      (var_allowed : Alloc_mode.t -> variable -> option bool)
      (alloc_mode : Alloc_mode.t)
      (field_types_and_expected_kinds : list (flambda_type * flambda_kind))
      : option (option (list simple)) :=
    let* field_simples :=
      reify_fields tenv var_allowed alloc_mode field_types_and_expected_kinds
    in
    if Nat.eqb (length field_types_and_expected_kinds) (length field_simples)
    then Some (Some field_simples)
    else Some None.

  (* The local function [var_allowed] of [Reify.reify]. *)
  Definition var_allowed (allowed_if_free_vars_defined_in : Typing_env.t)
      (var_is_defined_at_toplevel var_is_symbol_projection : variable -> bool)
      (tenv : Typing_env.t) (min_name_mode : Name_mode.t)
      (alloc_mode : Alloc_mode.t) (var : variable) : option bool :=
    if Typing_env.mem (Some min_name_mode) allowed_if_free_vars_defined_in
         (Name_var var)
    then
      if var_is_symbol_projection var
      then Some true
      else if var_is_defined_at_toplevel var
      then
        match alloc_mode with
        | Alloc_mode.Heap => Some true
        | Alloc_mode.Heap_or_local | Alloc_mode.Local =>
          let* proof :=
            Provers.never_holds_locally_allocated_values tenv var
          in
          match proof with
          | Proof_of_property.Proved _ => Some true
          | Proof_of_property.Unknown => Some false
          end
        end
      else Some false
    else Some false.

  (* [Flambda_kind.Boxable_number.unboxed_kind], as a kind of naked
     numbers. *)
  Definition unboxed_number_kind (k : boxable_number) : naked_number_kind :=
    match k with
    | Boxable_naked_float32 => Kind_naked_float32
    | Boxable_naked_float => Kind_naked_float
    | Boxable_naked_int32 => Kind_naked_int32
    | Boxable_naked_int64 => Kind_naked_int64
    | Boxable_naked_nativeint => Kind_naked_nativeint
    | Boxable_naked_vec128 => Kind_naked_vec128
    | Boxable_naked_vec256 => Kind_naked_vec256
    | Boxable_naked_vec512 => Kind_naked_vec512
    | Boxable_naked_mask => Kind_naked_mask
    end.

  (* The [match] of [Reify.reify] on the expanded type.
     [try_canonical_simple] is what the local function of that name
     answers. *)
  Definition reify_expanded
      (var_allowed : Alloc_mode.t -> variable -> option bool)
      (tenv : Typing_env.t) (try_canonical_simple : Reification_result.t)
      (t : flambda_type) : option Reification_result.t :=
    let* expanded := Provers.expand_head tenv t in
    match expanded with
    | Oub_value
        (Ok
           (Head_of_kind_value
              (Ok (Variant _ immediates _ blocks _ is_unique)) Not_null)) =>
      match blocks, immediates with
      | Or_unknown.Known blocks, Or_unknown.Known imms =>
        let* imms_bottom := Provers.is_bottom tenv imms in
        if imms_bottom
        then
          match row_like_for_blocks_get_singleton blocks with
          | None => Some try_canonical_simple
          | Some (tag, shape, size, field_types, alloc_mode) =>
            if negb (Nat.eqb size (length field_types))
            then None
            else
              let kinds := field_kinds shape (length field_types) in
              if negb (Nat.eqb (length kinds) (length field_types))
              then None
              else
                let field_types_and_expected_kinds :=
                  combine field_types kinds
                in
                match shape with
                | Float_record => Some try_canonical_simple
                | Scannable shape =>
                  if negb (Provers.tag_is_scannable tag)
                  then None
                  else
                    let* fields :=
                      try_to_reify_fields tenv var_allowed alloc_mode
                        field_types_and_expected_kinds
                    in
                    match fields with
                    | Some fields =>
                      Some
                        (Reification_result.Lift
                           (To_lift.Immutable_block tag is_unique shape
                              fields))
                    | None => Some try_canonical_simple
                    end
                end
          end
        else if row_like_for_blocks_is_bottom blocks
        then
          let* proof := Provers.meet_naked_immediates tenv imms in
          match proof with
          | Meet_shortcut.Known_result imms =>
            match Keyed.get_singleton Z.eqb imms with
            | None => Some try_canonical_simple
            | Some imm =>
              Some (Reification_result.Simple (Const (Tagged_immediate imm)))
            end
          | Meet_shortcut.Need_meet => Some try_canonical_simple
          | Meet_shortcut.Invalid => Some Reification_result.Invalid
          end
        else Some try_canonical_simple
      | _, _ => Some try_canonical_simple
      end
    | Oub_naked_immediate (Ok head) =>
      let* proof :=
        Provers.meet_naked_immediates tenv
          (Type_naked_immediate (Ok (No_alias head)))
      in
      match proof with
      | Meet_shortcut.Known_result imms =>
        match Keyed.get_singleton Z.eqb imms with
        | None => Some try_canonical_simple
        | Some i =>
          Some (Reification_result.Simple (Const (Naked_immediate i)))
        end
      | Meet_shortcut.Need_meet => Some try_canonical_simple
      | Meet_shortcut.Invalid => Some Reification_result.Invalid
      end
    | Oub_naked_number k (Ok ns) =>
      match Keyed.get_singleton Z.eqb ns with
      | None => Some try_canonical_simple
      | Some n =>
        Some (Reification_result.Simple (Const (const_of_naked_number k n)))
      end
    | Oub_value
        (Ok (Head_of_kind_value (Ok (Boxed_number k contents _)) Not_null)) =>
      let* proof :=
        Provers.meet_naked_number (unboxed_number_kind k) tenv contents
      in
      match proof with
      | Meet_shortcut.Need_meet => Some try_canonical_simple
      | Meet_shortcut.Invalid => Some Reification_result.Invalid
      | Meet_shortcut.Known_result ns =>
        match Keyed.get_singleton Z.eqb ns with
        | None => Some try_canonical_simple
        | Some n => Some (Reification_result.Lift (To_lift.Boxed_number k n))
        end
      end
    | Oub_value Bottom | Oub_naked_immediate Bottom
    | Oub_naked_number _ Bottom | Oub_rec_info Bottom | Oub_region Bottom =>
      Some Reification_result.Invalid
    | Oub_value (Ok (Head_of_kind_value _ (Maybe_null _)))
    | Oub_value (Ok (Head_of_kind_value (Bottom | Unknown) _))
    | Oub_value
        (Ok
           (Head_of_kind_value
              (Ok (Mutable_block _ | Closures _ _ | String _)) Not_null))
    | Oub_value Unknown | Oub_naked_immediate Unknown
    | Oub_naked_number _ Unknown | Oub_rec_info (Unknown | Ok _)
    | Oub_region (Unknown | Ok _) =>
      Some try_canonical_simple
    end.

  (* [Reify.reify]. *)
  Definition reify (allowed_if_free_vars_defined_in : Typing_env.t)
      (var_is_defined_at_toplevel var_is_symbol_projection : variable -> bool)
      (tenv : Typing_env.t) (t : flambda_type)
      : option Reification_result.t :=
    let min_name_mode := Name_mode.normal in
    let var_allowed :=
      var_allowed allowed_if_free_vars_defined_in var_is_defined_at_toplevel
        var_is_symbol_projection tenv min_name_mode
    in
    let* canonical_simple :=
      Typing_env.get_alias_then_canonical_simple_exn tenv (Some min_name_mode)
        None t
    in
    match canonical_simple with
    | Some (Symbol sym) => Some (Reification_result.Simple (Symbol sym))
    | Some (Const c) => Some (Reification_result.Simple (Const c))
    | Some (Var _) | None =>
      let try_canonical_simple : Reification_result.t :=
        match canonical_simple with
        | None => Reification_result.Cannot_reify
        | Some canonical_simple => Reification_result.Simple canonical_simple
        end
      in
      reify_expanded var_allowed tenv try_canonical_simple t
    end.

End Reify.
