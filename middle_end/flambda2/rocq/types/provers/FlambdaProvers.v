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



(* The provers, as the model sees them: the questions that Simplify asks
   of a type in a typing environment.

   This file is to be reviewed, line by line against
   middle_end/flambda2/types/provers.ml, as FlambdaTypeGrammar.v is
   against type_grammar.ml. It holds definitions only, in a module
   [Provers], so that [Provers.prove_is_int] is that function; and it
   extracts. What an answer means is FlambdaProversSpec.v. The plan of
   the workstream, with the table of the provers, is
   middle_end/flambda2/rocq/requests/T6-provers.md.

   What is here, of provers.ml (the lines are of that file):
   - the three types of answer, [meet_shortcut], [proof_of_property]
     and [generic_proof], each in a module of its name so that their
     constructors are told apart ([Proof_of_property.Proved],
     [Generic_proof.Proved], [Meet_shortcut.Invalid]), and from
     [Unknown] of FlambdaTypeGrammar.v; [as_meet_shortcut],
     [as_property], [gen_value_to_gen], [gen_value_to_proof],
     [gen_value_to_meet] (lines 41 to 103);
   - [prove_is_int_generic_value], [prove_is_int_generic],
     [prove_is_int], [meet_is_int_variant_only],
     [prove_is_not_a_pointer_generic_value], [prove_is_not_a_pointer]
     (131 to 179);
   - [prove_get_tag_generic_value], [prove_get_tag_generic],
     [prove_get_tag] (186 to 217);
   - [prove_is_null_generic], [meet_is_null] (219 to 233);
   - [prove_naked_immediates_generic], [meet_naked_immediates] (235 to
     303);
   - [prove_equals_tagged_immediates_value],
     [prove_equals_tagged_immediates],
     [meet_equals_tagged_immediates_value],
     [meet_equals_tagged_immediates],
     [meet_equals_single_tagged_immediate] (308 to 369);
   - [meet_naked_number] and the eleven functions defined with it,
     [meet_naked_float32s] to [meet_naked_masks] (371 to 520);
   - [prove_equals_to_simple_of_kind] (105 to 127);
   - [boxed_or_tagged_number], in a module of its name,
     [prove_is_a_boxed_or_tagged_number_value], [prove_nothing],
     [prove_is_a_boxed_or_tagged_number], [prove_is_a_tagged_immediate],
     the nine [prove_is_a_boxed_float32] to [prove_is_a_boxed_mask],
     [prove_is_or_is_not_a_boxed_float_value] and
     [prove_is_or_is_not_a_boxed_float] (582 to 695);
   - [prove_strings_value], [meet_strings], [prove_strings] (922 to
     933);
   - [tagging_proof_kind], [inspect_tagging_of_simple_value],
     [prove_tagging_of_simple], [meet_tagging_of_simple] (935 to 991);
   - [meet_boxed_number_containing_simple] and the nine functions
     defined with it, [meet_boxed_float32_containing_simple] to
     [meet_boxed_mask_containing_simple] (992 to 1110).
   - [variant_like_proof], [prove_variant_like_generic_value],
     [meet_variant_like], [prove_variant_like] (522 to 580);
   - [prove_unique_tag_and_size_value], [prove_unique_tag_and_size],
     [prove_unique_fully_constructed_immutable_heap_block] (697 to
     752);
   - [meet_block_field_simple_value], [meet_block_field_simple] (1112
     to 1157);
   - [prove_single_closures_entry_generic_value],
     [meet_single_closures_entry], [prove_single_closures_entry],
     [prove_code_ids_generic_value], [meet_code_ids], [prove_code_ids]
     (835 to 901);
   - [meet_project_function_slot_simple_value],
     [meet_project_function_slot_simple],
     [meet_project_value_slot_simple_value],
     [meet_project_value_slot_simple] (1159 to 1212);
   - [prove_physical_equality] (1297 to 1624);
   - [prove_alloc_mode_of_boxed_number_value],
     [prove_alloc_mode_of_boxed_number],
     [never_holds_locally_allocated_values] (1239 to 1295), of which
     nothing is stated: an allocation mode has no meaning in the
     model.
   What is not here: the provers of arrays ([meet_is_flat_float_array],
   [meet_is_non_empty_naked_number_array], [prove_is_immediates_array],
   [meet_is_immutable_array], [prove_is_immutable_array]: 754 to 833
   and 903 to 920), as the grammar has no [Array]; and [meet_rec_info]
   (1213 to 1237), which answers the expression that a head of kind
   [Rec_info] is: the grammar has the types of that kind
   ([Type_rec_info]), but their head is [unit], the expression being
   left out (FlambdaTypeGrammar.v), so there is nothing to answer.

   How the definitions differ from the compiler's.
   - A stop of the compiler is [None]. A prover expands the head of a
     type ([Expand_head.expand_head]), which is a fatal error on a name
     that is not bound and on a canonical name whose type is an alias.
     So every prover answers an [option], [None] where the compiler
     stops, and a function on heads that a prover is made with does
     too. The compiler calls [Expand_head.is_bottom] and the provers
     where the model does, in the same order, so the model stops where
     the compiler does.
   - [Expand_head.Expanded_type.descr_oub], the view of an expanded
     type that the provers match on, is here ([descr_oub], the type and
     the function): the file of [expand_head] left it out, the meet
     having no use for it. Its fifteen constructors, one for each
     kind, are five, as those of [Type_grammar.t] are in
     FlambdaTypeGrammar.v: [Naked_float32] to [Naked_mask] are the one
     [Oub_naked_number], which carries the kind. The prefix is there
     because [Value], [Naked_immediate] and the others are the kinds
     and the constants of the model.
   - A head has no [Array] (FlambdaTypeGrammar.v), so no prover has
     that case; and the nine [Boxed_float], ..., [Boxed_mask] are the
     one [Boxed_number].
   - [wrong_kind] answers its default. Its two other arguments, which
     are for the message, are left out.
   - A set of numbers ([Target_ocaml_int.Set.t]) and a set of tags
     ([Tag.Set.t]) are lists, as in the grammar: [Set.is_empty],
     [Set.inter] and [Set.get_singleton] are [Keyed.is_empty],
     [Keyed.inter Z.eqb] and [Keyed.get_singleton Z.eqb]
     (FlambdaMapSpec.v). An answer that is a set is to be compared as
     a set: the list may have a number twice, and its order is not the
     compiler's.
   - The model has no machine width. [Target_ocaml_int.bool] is 1 or 0
     ([Z.b2z]), and [Tag.to_targetint_31_63] is the number of the tag:
     the set of numbers that [prove_naked_immediates_generic] makes of
     a set of tags, by a fold, is the list of the tags.
   - [inverse_relations] is a list of pairs of a relation and a name
     (FlambdaTypeGrammar.v) where the compiler has a map from relations
     to sets of names. [prove_naked_immediates_generic] folds over the
     list from its head; the compiler folds over the relations in
     their order and, for each, over its names in theirs. The three
     local functions of [prove_naked_immediates_generic] are
     definitions before it, and the function it folds is
     [prove_naked_immediates_relation].
   - [meet_naked_number] takes the kind where the compiler takes a
     witness of it ([meet_naked_number_kind]), which has no case for
     the immediates. At [Kind_naked_immediate] it is no function of the
     compiler.
   - The provers that answer a [Simple] call
     [Typing_env.get_canonical_simple_exn] on the alias that a type is
     ([Type_grammar.get_alias_exn]) and catch [Not_found] from either:
     that is [canonical_simple_of_alias], whose inner [None] is that
     exception. The local function [inspect_immediates] of
     [inspect_tagging_of_simple_value] is [inspect_tagging_immediates].
   - The nine [prove_is_a_boxed_*] are [prove_is_a_boxed_number], and
     the nine [meet_boxed_*_containing_simple] are
     [meet_boxed_number_containing_simple], each of which takes the
     number boxed ([boxable_number]); the second takes it where the
     compiler takes the function that reads the contents of a head of
     that number.
   - A size and the index of a field are natural numbers, as in
     FlambdaTypeAccessSpec.v; the compiler has target integers.
   - [prove_variant_like_generic_value] keeps the sizes of the tags if
     every tag is scannable, by a fold that makes a map keyed by
     scannable tags; here it is a test of the tags
     ([tag_is_scannable], which is [Tag.Scannable.of_tag]) and the map
     is the same map. The function that
     [prove_unique_fully_constructed_immutable_heap_block] folds over
     the types of the fields is [field_aliases_rev].
   - A string of a [String] head is the list of its bytes
     (FlambdaTypeGrammar.v), where the compiler has a [String_info.t].
   - The local function [of_singleton_type] of
     [prove_single_closures_entry_generic_value] is
     [single_closures_entry_of_singleton], which takes the allocation
     mode.
   - [prove_code_ids_generic_value] folds over the cases of the
     closures and leaves the fold by an exception at a case that has
     no function type for its own slot. Here the function folded is
     [add_code_id_of_case], which answers [Or_unknown.Unknown] from
     there on; the fold is over the bindings of the map that count
     ([compact] of FlambdaMapSpec.v), in their order in that list; and
     a set of code ids is a list, to be compared as a set.
   - [prove_physical_equality] is in parts. Its local functions
     [incompatible_naked_numbers] and [check_heads] are
     [incompatible_naked_numbers] and [physical_equality_check_heads];
     the case of two heads that are not null is
     [physical_equality_of_heads], and that of two variants
     [physical_equality_of_variants], whose local definitions
     [immediates_equality] and [blocks_equality] are
     [physical_equality_of_immediates] and
     [physical_equality_of_blocks]; the two guards of its first two
     cases are one test, [no_immediate_and_no_block], and its last
     [match] is [physical_equality_of_parts]. The eleven cases of
     [incompatible_naked_numbers] are one, at two kinds that are the
     same. An assertion that fails is [None], as a stop.
   - The cases of two heads in [physical_equality_of_heads] are not in
     the compiler's order, and each pair of constructors is written
     once. The compiler has, before the case of two variants, a case
     for a variant whose blocks are bottom against a head that is not
     a variant; it answers as the later cases do for that pair, but
     against a mutable block, where the later case answers [Unknown]:
     that is the case of a variant and a mutable block here. Two boxed
     numbers of two kinds are the nine constructors taken two by two.
     The tests of whether two heads may be null and of whether their
     parts that are not null are known are nested here, in the
     compiler's order.
   - [Reg_width_const.equal] is [const_eqb] (FlambdaCompareSpec.v). A
     poison constant of the model has a kind and no name, so two of
     one kind are equal here, where the compiler's are equal only if
     they have one name too: of two aliases of poison constants of one
     kind and two names the compiler answers [Proved false], and this
     [Proved true]. The test that two sets of strings have none in
     common is on lists of bytes ([lists_eqb Z.eqb]).
   - A value slot of the model has no kind. The kind that
     [meet_project_value_slot_simple_value] reads off the value slot
     ([Value_slot.kind]) is an argument, the last of
     [meet_project_value_slot_simple].
   - The arguments are in the compiler's order, the environment
     first; a labelled argument is an ordinary one. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaCompareSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningBoolSpec FlambdaTypeEqualSpec FlambdaTypeAccessSpec
  FlambdaBindingTime
  FlambdaEnvAliases FlambdaTypingEnv FlambdaExpandHead.
Import ListNotations.

(* ======================================================================== *)
(* The view of an expanded type                                             *)
(* ======================================================================== *)

(* [Expand_head.Expanded_type.descr_oub]. *)
Inductive descr_oub : Type :=
  | Oub_value (head : or_unknown_or_bottom head_of_kind_value)
  | Oub_naked_immediate
      (head : or_unknown_or_bottom head_of_kind_naked_immediate)
  | Oub_naked_number (k : naked_number_kind)
      (head : or_unknown_or_bottom (list Z))
  | Oub_rec_info (head : or_unknown_or_bottom unit)
  | Oub_region (head : or_unknown_or_bottom unit).

(* [Expand_head.Expanded_type.descr_oub], the function: an expanded type
   that is unknown or bottom is so at its kind. *)
Definition descr_oub_of_expanded (expanded : Expanded_type.t) : descr_oub :=
  match Expanded_type.descr expanded with
  | Unknown =>
    match Expanded_type.kind expanded with
    | Value => Oub_value Unknown
    | Naked_number Kind_naked_immediate => Oub_naked_immediate Unknown
    | Naked_number k => Oub_naked_number k Unknown
    | Rec_info => Oub_rec_info Unknown
    | Region => Oub_region Unknown
    end
  | Bottom =>
    match Expanded_type.kind expanded with
    | Value => Oub_value Bottom
    | Naked_number Kind_naked_immediate => Oub_naked_immediate Bottom
    | Naked_number k => Oub_naked_number k Bottom
    | Rec_info => Oub_rec_info Bottom
    | Region => Oub_region Bottom
    end
  | Ok (Expanded_type.Head_value head) => Oub_value (Ok head)
  | Ok (Expanded_type.Head_naked_immediate head) =>
    Oub_naked_immediate (Ok head)
  | Ok (Expanded_type.Head_naked_number k head) => Oub_naked_number k (Ok head)
  | Ok (Expanded_type.Head_rec_info head) => Oub_rec_info (Ok head)
  | Ok (Expanded_type.Head_region head) => Oub_region (Ok head)
  end.

(* ======================================================================== *)
(* The answers                                                              *)
(* ======================================================================== *)

(* [Provers.meet_shortcut]. *)
Module Meet_shortcut.
  Inductive t (A : Type) : Type :=
    | Known_result (a : A)
    | Need_meet
    | Invalid.
  Arguments Known_result {A} a.
  Arguments Need_meet {A}.
  Arguments Invalid {A}.
End Meet_shortcut.

(* [Provers.proof_of_property]. *)
Module Proof_of_property.
  Inductive t (A : Type) : Type :=
    | Proved (a : A)
    | Unknown.
  Arguments Proved {A} a.
  Arguments Unknown {A}.
End Proof_of_property.

(* [Provers.generic_proof]. *)
Module Generic_proof.
  Inductive t (A : Type) : Type :=
    | Proved (a : A)
    | Unknown
    | Invalid.
  Arguments Proved {A} a.
  Arguments Unknown {A}.
  Arguments Invalid {A}.
End Generic_proof.

(* [Provers.boxed_or_tagged_number]. *)
Module Boxed_or_tagged_number.
  Inductive t : Type :=
    | Boxed (alloc_mode : Alloc_mode.t) (k : boxable_number)
        (contents : flambda_type)
    | Tagged_immediate.
End Boxed_or_tagged_number.

(* [Provers.variant_like_proof]. *)
Record variant_like_proof : Type := Variant_like_proof {
  const_ctors : Or_unknown.t (list Z);
  non_const_ctors_with_sizes : Keyed.finite_map Z (nat * block_shape)
}.

(* ======================================================================== *)
(* The provers                                                              *)
(* ======================================================================== *)

Module Provers.

  (* [Provers.is_bottom], which is [Expand_head.is_bottom]. *)
  Definition is_bottom (tenv : Typing_env.t) (t : flambda_type)
      : option bool :=
    let* expanded := expand_head tenv t in
    Some (Expanded_type.is_bottom expanded).

  (* [Provers.expand_head]. *)
  Definition expand_head (tenv : Typing_env.t) (t : flambda_type)
      : option descr_oub :=
    let* expanded := expand_head tenv t in
    Some (descr_oub_of_expanded expanded).

  (* [Provers.wrong_kind]: it answers its default. *)
  Definition wrong_kind {A : Type} (default : A) : A := default.

  (* [Provers.as_meet_shortcut]. *)
  Definition as_meet_shortcut {A : Type} (p : Generic_proof.t A)
      : Meet_shortcut.t A :=
    match p with
    | Generic_proof.Proved x => Meet_shortcut.Known_result x
    | Generic_proof.Unknown => Meet_shortcut.Need_meet
    | Generic_proof.Invalid => Meet_shortcut.Invalid
    end.

  (* [Provers.as_property]. *)
  Definition as_property {A : Type} (p : Generic_proof.t A)
      : Proof_of_property.t A :=
    match p with
    | Generic_proof.Proved x => Proof_of_property.Proved x
    | Generic_proof.Unknown | Generic_proof.Invalid =>
      Proof_of_property.Unknown
    end.

  (* [Provers.gen_value_to_gen]. *)
  Definition gen_value_to_gen {A : Type}
      (prove_gen :
        Typing_env.t -> head_of_kind_value_non_null
        -> option (Generic_proof.t A))
      (tenv : Typing_env.t) (t : flambda_type) : option (Generic_proof.t A) :=
    let* expanded := expand_head tenv t in
    match expanded with
    | Oub_value Unknown => Some Generic_proof.Unknown
    | Oub_value Bottom => Some Generic_proof.Invalid
    | Oub_value (Ok (Head_of_kind_value _ (Maybe_null _)))
    | Oub_value (Ok (Head_of_kind_value Unknown Not_null)) =>
      Some Generic_proof.Unknown
    | Oub_value (Ok (Head_of_kind_value Bottom Not_null)) =>
      Some Generic_proof.Invalid
    | Oub_value (Ok (Head_of_kind_value (Ok head) Not_null)) =>
      prove_gen tenv head
    | Oub_naked_immediate _ | Oub_naked_number _ _ | Oub_rec_info _
    | Oub_region _ =>
      Some (wrong_kind Generic_proof.Invalid)
    end.

  (* [Provers.gen_value_to_proof]. *)
  Definition gen_value_to_proof {A : Type}
      (prove_gen :
        Typing_env.t -> head_of_kind_value_non_null
        -> option (Generic_proof.t A))
      (tenv : Typing_env.t) (t : flambda_type)
      : option (Proof_of_property.t A) :=
    let* expanded := expand_head tenv t in
    match expanded with
    | Oub_value (Unknown | Bottom)
    | Oub_value (Ok (Head_of_kind_value _ (Maybe_null _)))
    | Oub_value (Ok (Head_of_kind_value (Unknown | Bottom) Not_null)) =>
      Some Proof_of_property.Unknown
    | Oub_value (Ok (Head_of_kind_value (Ok head) Not_null)) =>
      let* proof := prove_gen tenv head in
      Some (as_property proof)
    | Oub_naked_immediate _ | Oub_naked_number _ _ | Oub_rec_info _
    | Oub_region _ =>
      Some (wrong_kind Proof_of_property.Unknown)
    end.

  (* [Provers.gen_value_to_meet]. *)
  Definition gen_value_to_meet {A : Type}
      (prove_gen :
        Typing_env.t -> head_of_kind_value_non_null
        -> option (Generic_proof.t A))
      (tenv : Typing_env.t) (t : flambda_type) : option (Meet_shortcut.t A) :=
    let* expanded := expand_head tenv t in
    match expanded with
    | Oub_value Unknown | Oub_value (Ok (Head_of_kind_value Unknown _)) =>
      Some Meet_shortcut.Need_meet
    | Oub_value Bottom | Oub_value (Ok (Head_of_kind_value Bottom _)) =>
      Some Meet_shortcut.Invalid
    | Oub_value (Ok (Head_of_kind_value (Ok head) _)) =>
      let* proof := prove_gen tenv head in
      Some (as_meet_shortcut proof)
    | Oub_naked_immediate _ | Oub_naked_number _ _ | Oub_rec_info _
    | Oub_region _ =>
      Some (wrong_kind Meet_shortcut.Invalid)
    end.

  (* [Provers.prove_is_int_generic_value]. *)
  Definition prove_is_int_generic_value (variant_only : bool)
      (tenv : Typing_env.t) (value_head : head_of_kind_value_non_null)
      : option (Generic_proof.t bool) :=
    match value_head with
    | Variant _ immediates _ blocks _ _ =>
      match blocks, immediates with
      | Or_unknown.Unknown, Or_unknown.Unknown => Some Generic_proof.Unknown
      | Or_unknown.Unknown, Or_unknown.Known imms =>
        let* imms_bottom := is_bottom tenv imms in
        if imms_bottom
        then Some (Generic_proof.Proved false)
        else Some Generic_proof.Unknown
      | Or_unknown.Known blocks, Or_unknown.Unknown =>
        if row_like_for_blocks_is_bottom blocks
        then Some (Generic_proof.Proved true)
        else Some Generic_proof.Unknown
      | Or_unknown.Known blocks, Or_unknown.Known imms =>
        if row_like_for_blocks_is_bottom blocks
        then
          let* imms_bottom := is_bottom tenv imms in
          if imms_bottom
          then Some Generic_proof.Invalid
          else Some (Generic_proof.Proved true)
        else
          let* imms_bottom := is_bottom tenv imms in
          if imms_bottom
          then Some (Generic_proof.Proved false)
          else Some Generic_proof.Unknown
      end
    | Mutable_block _ => Some (Generic_proof.Proved false)
    | Boxed_number _ _ _ | Closures _ _ | String _ =>
      if variant_only
      then Some Generic_proof.Invalid
      else Some (Generic_proof.Proved false)
    end.

  (* [Provers.prove_is_int_generic]. *)
  Definition prove_is_int_generic (variant_only : bool) (tenv : Typing_env.t)
      (t : flambda_type) : option (Generic_proof.t bool) :=
    gen_value_to_gen (prove_is_int_generic_value variant_only) tenv t.

  (* [Provers.prove_is_int]. *)
  Definition prove_is_int (tenv : Typing_env.t) (t : flambda_type)
      : option (Proof_of_property.t bool) :=
    gen_value_to_proof (prove_is_int_generic_value false) tenv t.

  (* [Provers.meet_is_int_variant_only]. *)
  Definition meet_is_int_variant_only (tenv : Typing_env.t)
      (t : flambda_type) : option (Meet_shortcut.t bool) :=
    gen_value_to_meet (prove_is_int_generic_value true) tenv t.

  (* [Provers.prove_is_not_a_pointer_generic_value]. *)
  Definition prove_is_not_a_pointer_generic_value (tenv : Typing_env.t)
      (t : flambda_type) : option (Generic_proof.t bool) :=
    let* expanded := expand_head tenv t in
    match expanded with
    | Oub_value Unknown => Some Generic_proof.Unknown
    | Oub_value Bottom => Some Generic_proof.Invalid
    | Oub_value (Ok (Head_of_kind_value Bottom (Maybe_null _))) =>
      Some (Generic_proof.Proved true)
    | Oub_value (Ok (Head_of_kind_value Bottom Not_null)) =>
      Some Generic_proof.Invalid
    | Oub_value (Ok (Head_of_kind_value Unknown (Maybe_null _ | Not_null))) =>
      Some Generic_proof.Unknown
    | Oub_value (Ok (Head_of_kind_value (Ok head) is_null)) =>
      let* proof := prove_is_int_generic_value false tenv head in
      match proof, is_null with
      | Generic_proof.Proved true, (Maybe_null _ | Not_null) =>
        Some (Generic_proof.Proved true)
      | (Generic_proof.Proved false | Generic_proof.Unknown), Maybe_null _
      | Generic_proof.Unknown, Not_null =>
        Some Generic_proof.Unknown
      | Generic_proof.Proved false, Not_null =>
        Some (Generic_proof.Proved false)
      | Generic_proof.Invalid, _ => Some Generic_proof.Invalid
      end
    | Oub_naked_immediate _ | Oub_naked_number _ _ | Oub_rec_info _
    | Oub_region _ =>
      Some (wrong_kind Generic_proof.Invalid)
    end.

  (* [Provers.prove_is_not_a_pointer]. *)
  Definition prove_is_not_a_pointer (tenv : Typing_env.t) (t : flambda_type)
      : option (Proof_of_property.t bool) :=
    let* proof := prove_is_not_a_pointer_generic_value tenv t in
    Some (as_property proof).

  (* [Provers.prove_get_tag_generic_value]. *)
  Definition prove_get_tag_generic_value (tenv : Typing_env.t)
      (value_head : head_of_kind_value_non_null)
      : option (Generic_proof.t (list Z)) :=
    match value_head with
    | Variant _ immediates _ blocks _ _ =>
      match immediates with
      | Or_unknown.Unknown => Some Generic_proof.Unknown
      | Or_unknown.Known imms =>
        let* imms_bottom := is_bottom tenv imms in
        if negb imms_bottom
        then Some Generic_proof.Unknown
        else
          match blocks with
          | Or_unknown.Unknown => Some Generic_proof.Unknown
          | Or_unknown.Known blocks =>
            match row_like_for_blocks_all_tags blocks with
            | Or_unknown.Unknown => Some Generic_proof.Unknown
            | Or_unknown.Known tags =>
              if Keyed.is_empty tags
              then Some Generic_proof.Invalid
              else Some (Generic_proof.Proved tags)
            end
          end
      end
    | Boxed_number _ _ _ => Some Generic_proof.Unknown
    | Mutable_block _ => Some Generic_proof.Unknown
    | Closures _ _ => Some Generic_proof.Unknown
    | String _ => Some Generic_proof.Unknown
    end.

  (* [Provers.prove_get_tag_generic]. *)
  Definition prove_get_tag_generic (tenv : Typing_env.t) (t : flambda_type)
      : option (Generic_proof.t (list Z)) :=
    gen_value_to_gen prove_get_tag_generic_value tenv t.

  (* [Provers.prove_get_tag]. *)
  Definition prove_get_tag (tenv : Typing_env.t) (t : flambda_type)
      : option (Proof_of_property.t (list Z)) :=
    gen_value_to_proof prove_get_tag_generic_value tenv t.

  (* [Provers.prove_is_null_generic]. *)
  Definition prove_is_null_generic (tenv : Typing_env.t) (t : flambda_type)
      : option (Generic_proof.t bool) :=
    let* expanded := expand_head tenv t in
    match expanded with
    | Oub_value Unknown => Some Generic_proof.Unknown
    | Oub_value Bottom => Some Generic_proof.Invalid
    | Oub_value (Ok (Head_of_kind_value Bottom Not_null)) =>
      Some Generic_proof.Invalid
    | Oub_value (Ok (Head_of_kind_value _ Not_null)) =>
      Some (Generic_proof.Proved false)
    | Oub_value (Ok (Head_of_kind_value Bottom _)) =>
      Some (Generic_proof.Proved true)
    | Oub_value (Ok (Head_of_kind_value (Unknown | Ok _) (Maybe_null _))) =>
      Some Generic_proof.Unknown
    | Oub_naked_immediate _ | Oub_naked_number _ _ | Oub_rec_info _
    | Oub_region _ =>
      Some (wrong_kind Generic_proof.Invalid)
    end.

  (* [Provers.meet_is_null]. *)
  Definition meet_is_null (tenv : Typing_env.t) (t : flambda_type)
      : option (Meet_shortcut.t bool) :=
    let* proof := prove_is_null_generic tenv t in
    Some (as_meet_shortcut proof).

  (* The local functions [these_immediates], [meet_these_immediates] and
     [meet_this_immediate] of [Provers.prove_naked_immediates_generic]. *)
  Definition these_immediates (imms : list Z) : Generic_proof.t (list Z) :=
    if Keyed.is_empty imms
    then Generic_proof.Invalid
    else Generic_proof.Proved imms.

  Definition meet_these_immediates (imms : list Z)
      (proof : Generic_proof.t (list Z)) : Generic_proof.t (list Z) :=
    match proof with
    | Generic_proof.Proved imms' =>
      these_immediates (Keyed.inter Z.eqb imms imms')
    | Generic_proof.Unknown => these_immediates imms
    | Generic_proof.Invalid => Generic_proof.Invalid
    end.

  Definition meet_this_immediate (imm : Z) (proof : Generic_proof.t (list Z))
      : Generic_proof.t (list Z) :=
    meet_these_immediates [imm] proof.

  (* The function that [Provers.prove_naked_immediates_generic] folds
     over the inverse relations: what the relation with one name adds to
     what is known of the number. *)
  Definition prove_naked_immediates_relation (tenv : Typing_env.t)
      (relation : Relation.t) (name : name) (imms : Generic_proof.t (list Z))
      : option (Generic_proof.t (list Z)) :=
    let ty := alias_type_of Value (simple_of_name name) in
    match relation with
    | Relation.Is_null =>
      let* proof := prove_is_null_generic tenv ty in
      match proof with
      | Generic_proof.Proved b => Some (meet_this_immediate (Z.b2z b) imms)
      | Generic_proof.Unknown => Some imms
      | Generic_proof.Invalid => Some Generic_proof.Invalid
      end
    | Relation.Is_int =>
      let* proof := prove_is_int_generic true tenv ty in
      match proof with
      | Generic_proof.Proved b => Some (meet_this_immediate (Z.b2z b) imms)
      | Generic_proof.Unknown => Some imms
      | Generic_proof.Invalid => Some Generic_proof.Invalid
      end
    | Relation.Get_tag =>
      let* proof := prove_get_tag_generic tenv ty in
      match proof with
      | Generic_proof.Proved tags => Some (meet_these_immediates tags imms)
      | Generic_proof.Unknown => Some imms
      | Generic_proof.Invalid => Some Generic_proof.Invalid
      end
    end.

  (* [Provers.prove_naked_immediates_generic]. *)
  Definition prove_naked_immediates_generic (tenv : Typing_env.t)
      (t : flambda_type) : option (Generic_proof.t (list Z)) :=
    let* expanded := expand_head tenv t in
    match expanded with
    | Oub_naked_immediate (Ok head) =>
      let proof : Generic_proof.t (list Z) :=
        match naked_immediates head with
        | Or_unknown.Known imms => these_immediates imms
        | Or_unknown.Unknown => Generic_proof.Unknown
        end
      in
      List.fold_left
        (fun (imms : option (Generic_proof.t (list Z)))
             (relation_and_name : Relation.t * name) =>
           let* imms := imms in
           prove_naked_immediates_relation tenv (fst relation_and_name)
             (snd relation_and_name) imms)
        (inverse_relations head) (Some proof)
    | Oub_naked_immediate Unknown => Some Generic_proof.Unknown
    | Oub_naked_immediate Bottom => Some Generic_proof.Invalid
    | Oub_value _ | Oub_naked_number _ _ | Oub_rec_info _ | Oub_region _ =>
      Some (wrong_kind Generic_proof.Invalid)
    end.

  (* [Provers.meet_naked_immediates]. *)
  Definition meet_naked_immediates (tenv : Typing_env.t) (t : flambda_type)
      : option (Meet_shortcut.t (list Z)) :=
    let* proof := prove_naked_immediates_generic tenv t in
    Some (as_meet_shortcut proof).

  (* [Provers.prove_equals_tagged_immediates_value]. *)
  Definition prove_equals_tagged_immediates_value (tenv : Typing_env.t)
      (value_head : head_of_kind_value_non_null)
      : option (Generic_proof.t (list Z)) :=
    match value_head with
    | Variant _ immediates _ blocks _ _ =>
      match blocks with
      | Or_unknown.Unknown => Some Generic_proof.Unknown
      | Or_unknown.Known blocks =>
        if row_like_for_blocks_is_bottom blocks
        then
          match immediates with
          | Or_unknown.Unknown => Some Generic_proof.Unknown
          | Or_unknown.Known imms =>
            let* proof := prove_naked_immediates_generic tenv imms in
            match proof with
            | Generic_proof.Proved imms => Some (Generic_proof.Proved imms)
            | Generic_proof.Invalid => Some (Generic_proof.Proved [])
            | Generic_proof.Unknown => Some Generic_proof.Unknown
            end
          end
        else Some Generic_proof.Unknown
      end
    | Mutable_block _ | Boxed_number _ _ _ | Closures _ _ | String _ =>
      Some Generic_proof.Unknown
    end.

  (* [Provers.prove_equals_tagged_immediates]. *)
  Definition prove_equals_tagged_immediates (tenv : Typing_env.t)
      (t : flambda_type) : option (Proof_of_property.t (list Z)) :=
    gen_value_to_proof prove_equals_tagged_immediates_value tenv t.

  (* [Provers.meet_equals_tagged_immediates_value]. *)
  Definition meet_equals_tagged_immediates_value (tenv : Typing_env.t)
      (value_head : head_of_kind_value_non_null)
      : option (Generic_proof.t (list Z)) :=
    match value_head with
    | Variant _ immediates _ _ _ _ =>
      match immediates with
      | Or_unknown.Unknown => Some Generic_proof.Unknown
      | Or_unknown.Known imms => prove_naked_immediates_generic tenv imms
      end
    | Mutable_block _ | Boxed_number _ _ _ | Closures _ _ | String _ =>
      Some Generic_proof.Invalid
    end.

  (* [Provers.meet_equals_tagged_immediates]. *)
  Definition meet_equals_tagged_immediates (tenv : Typing_env.t)
      (t : flambda_type) : option (Meet_shortcut.t (list Z)) :=
    gen_value_to_meet meet_equals_tagged_immediates_value tenv t.

  (* [Provers.meet_equals_single_tagged_immediate]. *)
  Definition meet_equals_single_tagged_immediate (tenv : Typing_env.t)
      (t : flambda_type) : option (Meet_shortcut.t Z) :=
    let* shortcut := meet_equals_tagged_immediates tenv t in
    match shortcut with
    | Meet_shortcut.Known_result imms =>
      match Keyed.get_singleton Z.eqb imms with
      | Some imm => Some (Meet_shortcut.Known_result imm)
      | None => Some Meet_shortcut.Need_meet
      end
    | Meet_shortcut.Need_meet => Some Meet_shortcut.Need_meet
    | Meet_shortcut.Invalid => Some Meet_shortcut.Invalid
    end.

  (* [Provers.meet_naked_number]: the numbers that a type of naked
     numbers of a kind may be. *)
  Definition meet_naked_number (kind : naked_number_kind)
      (tenv : Typing_env.t) (t : flambda_type)
      : option (Meet_shortcut.t (list Z)) :=
    let head_to_proof (head : or_unknown_or_bottom (list Z))
        : Meet_shortcut.t (list Z) :=
      match head with
      | Ok set =>
        if Keyed.is_empty set
        then Meet_shortcut.Invalid
        else Meet_shortcut.Known_result set
      | Unknown => Meet_shortcut.Need_meet
      | Bottom => Meet_shortcut.Invalid
      end
    in
    let* expanded := expand_head tenv t in
    match expanded with
    | Oub_value _ => Some (wrong_kind Meet_shortcut.Invalid)
    | Oub_naked_immediate _ => Some (wrong_kind Meet_shortcut.Invalid)
    | Oub_rec_info _ => Some (wrong_kind Meet_shortcut.Invalid)
    | Oub_region _ => Some (wrong_kind Meet_shortcut.Invalid)
    | Oub_naked_number k numbers =>
      if kind_eqb (Naked_number k) (Naked_number kind)
      then Some (head_to_proof numbers)
      else Some (wrong_kind Meet_shortcut.Invalid)
    end.

  (* [Provers.meet_naked_float32s], ..., [meet_naked_masks]. *)
  Definition meet_naked_float32s := meet_naked_number Kind_naked_float32.
  Definition meet_naked_floats := meet_naked_number Kind_naked_float.
  Definition meet_naked_int8s := meet_naked_number Kind_naked_int8.
  Definition meet_naked_int16s := meet_naked_number Kind_naked_int16.
  Definition meet_naked_int32s := meet_naked_number Kind_naked_int32.
  Definition meet_naked_int64s := meet_naked_number Kind_naked_int64.
  Definition meet_naked_nativeints := meet_naked_number Kind_naked_nativeint.
  Definition meet_naked_vec128s := meet_naked_number Kind_naked_vec128.
  Definition meet_naked_vec256s := meet_naked_number Kind_naked_vec256.
  Definition meet_naked_vec512s := meet_naked_number Kind_naked_vec512.
  Definition meet_naked_masks := meet_naked_number Kind_naked_mask.

  (* [Typing_env.get_canonical_simple_exn env ~min_name_mode
     (Type_grammar.get_alias_exn t)], as the provers call it: the
     canonical element of the [Simple] that a type is an alias of. The
     inner [None] is [Not_found], of either function, which the provers
     catch with one handler. *)
  Definition canonical_simple_of_alias (tenv : Typing_env.t)
      (min_name_mode : Name_mode.t) (t : flambda_type)
      : option (option simple) :=
    match get_alias_opt t with
    | None => Some None
    | Some simple =>
      Typing_env.get_canonical_simple_exn tenv (Some min_name_mode) None
        simple
    end.

  (* [Provers.prove_equals_to_simple_of_kind]. *)
  Definition prove_equals_to_simple_of_kind (tenv : Typing_env.t)
      (t : flambda_type) (kind : flambda_kind)
      : option (Proof_of_property.t simple) :=
    let original_kind := FlambdaTypeGrammar.kind t in
    if negb (kind_eqb original_kind kind)
    then Some (wrong_kind Proof_of_property.Unknown)
    else
      let* canonical := canonical_simple_of_alias tenv Name_mode.normal t in
      match canonical with
      | None => Some Proof_of_property.Unknown
      | Some simple => Some (Proof_of_property.Proved simple)
      end.

  (* [Provers.prove_is_a_boxed_or_tagged_number_value]. *)
  Definition prove_is_a_boxed_or_tagged_number_value (tenv : Typing_env.t)
      (value_head : head_of_kind_value_non_null)
      : option (Generic_proof.t Boxed_or_tagged_number.t) :=
    match value_head with
    | Variant _ _ _ blocks _ _ =>
      match blocks with
      | Or_unknown.Unknown => Some Generic_proof.Unknown
      | Or_unknown.Known blocks =>
        if row_like_for_blocks_is_bottom blocks
        then
          Some (Generic_proof.Proved Boxed_or_tagged_number.Tagged_immediate)
        else Some Generic_proof.Unknown
      end
    | Boxed_number k contents_ty alloc_mode =>
      Some
        (Generic_proof.Proved
           (Boxed_or_tagged_number.Boxed alloc_mode k contents_ty))
    | Mutable_block _ | Closures _ _ | String _ => Some Generic_proof.Unknown
    end.

  (* [Provers.prove_nothing]. *)
  Definition prove_nothing {A : Type} (tenv : Typing_env.t)
      (t : flambda_type) : option (Proof_of_property.t A) :=
    Some Proof_of_property.Unknown.

  (* [Provers.prove_is_a_boxed_or_tagged_number]. *)
  Definition prove_is_a_boxed_or_tagged_number (tenv : Typing_env.t)
      (t : flambda_type)
      : option (Proof_of_property.t Boxed_or_tagged_number.t) :=
    gen_value_to_proof prove_is_a_boxed_or_tagged_number_value tenv t.

  (* [Provers.prove_is_a_tagged_immediate]. *)
  Definition prove_is_a_tagged_immediate (tenv : Typing_env.t)
      (t : flambda_type) : option (Proof_of_property.t unit) :=
    let* proof := prove_is_a_boxed_or_tagged_number tenv t in
    match proof with
    | Proof_of_property.Proved Boxed_or_tagged_number.Tagged_immediate =>
      Some (Proof_of_property.Proved tt)
    | Proof_of_property.Proved _ => Some Proof_of_property.Unknown
    | Proof_of_property.Unknown => Some Proof_of_property.Unknown
    end.

  (* [Provers.prove_is_a_boxed_float32], ..., [prove_is_a_boxed_mask]:
     the nine are this one, which takes the number boxed. *)
  Definition prove_is_a_boxed_number (kind : boxable_number)
      (tenv : Typing_env.t) (t : flambda_type)
      : option (Proof_of_property.t unit) :=
    let* proof := prove_is_a_boxed_or_tagged_number tenv t in
    match proof with
    | Proof_of_property.Proved (Boxed_or_tagged_number.Boxed _ k _) =>
      if boxable_number_eqb k kind
      then Some (Proof_of_property.Proved tt)
      else Some Proof_of_property.Unknown
    | Proof_of_property.Proved _ => Some Proof_of_property.Unknown
    | Proof_of_property.Unknown => Some Proof_of_property.Unknown
    end.

  Definition prove_is_a_boxed_float32 :=
    prove_is_a_boxed_number Boxable_naked_float32.
  Definition prove_is_a_boxed_float :=
    prove_is_a_boxed_number Boxable_naked_float.
  Definition prove_is_a_boxed_int32 :=
    prove_is_a_boxed_number Boxable_naked_int32.
  Definition prove_is_a_boxed_int64 :=
    prove_is_a_boxed_number Boxable_naked_int64.
  Definition prove_is_a_boxed_nativeint :=
    prove_is_a_boxed_number Boxable_naked_nativeint.
  Definition prove_is_a_boxed_vec128 :=
    prove_is_a_boxed_number Boxable_naked_vec128.
  Definition prove_is_a_boxed_vec256 :=
    prove_is_a_boxed_number Boxable_naked_vec256.
  Definition prove_is_a_boxed_vec512 :=
    prove_is_a_boxed_number Boxable_naked_vec512.
  Definition prove_is_a_boxed_mask :=
    prove_is_a_boxed_number Boxable_naked_mask.

  (* [Provers.prove_is_or_is_not_a_boxed_float_value]. *)
  Definition prove_is_or_is_not_a_boxed_float_value (tenv : Typing_env.t)
      (value_head : head_of_kind_value_non_null)
      : option (Generic_proof.t bool) :=
    match value_head with
    | Boxed_number Boxable_naked_float _ _ =>
      Some (Generic_proof.Proved true)
    | _ => Some (Generic_proof.Proved false)
    end.

  (* [Provers.prove_is_or_is_not_a_boxed_float]. *)
  Definition prove_is_or_is_not_a_boxed_float (tenv : Typing_env.t)
      (t : flambda_type) : option (Proof_of_property.t bool) :=
    gen_value_to_proof prove_is_or_is_not_a_boxed_float_value tenv t.

  (* [Provers.prove_strings_value]. *)
  Definition prove_strings_value (tenv : Typing_env.t)
      (value_head : head_of_kind_value_non_null)
      : option (Generic_proof.t (list (list Z))) :=
    match value_head with
    | String strs => Some (Generic_proof.Proved strs)
    | Variant _ _ _ _ _ _ | Mutable_block _ | Boxed_number _ _ _
    | Closures _ _ =>
      Some Generic_proof.Invalid
    end.

  (* [Provers.meet_strings]. *)
  Definition meet_strings (tenv : Typing_env.t) (t : flambda_type)
      : option (Meet_shortcut.t (list (list Z))) :=
    gen_value_to_meet prove_strings_value tenv t.

  (* [Provers.prove_strings]. *)
  Definition prove_strings (tenv : Typing_env.t) (t : flambda_type)
      : option (Proof_of_property.t (list (list Z))) :=
    gen_value_to_proof prove_strings_value tenv t.

  (* [Provers.tagging_proof_kind]. *)
  Inductive tagging_proof_kind : Type :=
    | Prove
    | Meet.

  (* The local function [inspect_immediates] of
     [Provers.inspect_tagging_of_simple_value]: a [Simple] whose value
     is the number of the immediates of a variant. *)
  Definition inspect_tagging_immediates (min_name_mode : Name_mode.t)
      (tenv : Typing_env.t) (immediates : Or_unknown.t flambda_type)
      : option (Generic_proof.t simple) :=
    match immediates with
    | Or_unknown.Unknown => Some Generic_proof.Unknown
    | Or_unknown.Known t =>
      let* from_alias := canonical_simple_of_alias tenv min_name_mode t in
      match from_alias with
      | Some simple => Some (Generic_proof.Proved simple)
      | None =>
        let* shortcut := meet_naked_immediates tenv t in
        match shortcut with
        | Meet_shortcut.Need_meet => Some Generic_proof.Unknown
        | Meet_shortcut.Invalid => Some Generic_proof.Invalid
        | Meet_shortcut.Known_result imms =>
          match Keyed.get_singleton Z.eqb imms with
          | Some imm =>
            Some (Generic_proof.Proved (Const (Naked_immediate imm)))
          | None => Some Generic_proof.Unknown
          end
        end
      end
    end.

  (* [Provers.inspect_tagging_of_simple_value]. *)
  Definition inspect_tagging_of_simple_value (proof_kind : tagging_proof_kind)
      (min_name_mode : Name_mode.t) (tenv : Typing_env.t)
      (value_head : head_of_kind_value_non_null)
      : option (Generic_proof.t simple) :=
    match value_head with
    | Variant _ immediates _ blocks _ _ =>
      match proof_kind, blocks with
      | Prove, Or_unknown.Unknown => Some Generic_proof.Unknown
      | Prove, Or_unknown.Known blocks =>
        if row_like_for_blocks_is_bottom blocks
        then inspect_tagging_immediates min_name_mode tenv immediates
        else Some Generic_proof.Unknown
      | Meet, _ => inspect_tagging_immediates min_name_mode tenv immediates
      end
    | Mutable_block _ | Boxed_number _ _ _ | String _ | Closures _ _ =>
      Some Generic_proof.Unknown
    end.

  (* [Provers.prove_tagging_of_simple]. *)
  Definition prove_tagging_of_simple (tenv : Typing_env.t)
      (min_name_mode : Name_mode.t) (t : flambda_type)
      : option (Proof_of_property.t simple) :=
    gen_value_to_proof (inspect_tagging_of_simple_value Prove min_name_mode)
      tenv t.

  (* [Provers.meet_tagging_of_simple]. *)
  Definition meet_tagging_of_simple (tenv : Typing_env.t)
      (min_name_mode : Name_mode.t) (t : flambda_type)
      : option (Meet_shortcut.t simple) :=
    gen_value_to_meet (inspect_tagging_of_simple_value Meet min_name_mode)
      tenv t.

  (* [Provers.meet_boxed_number_containing_simple], which takes the
     number boxed where the compiler takes the function that reads the
     contents of a head of that number. *)
  Definition meet_boxed_number_containing_simple (kind : boxable_number)
      (tenv : Typing_env.t) (min_name_mode : Name_mode.t) (t : flambda_type)
      : option (Meet_shortcut.t simple) :=
    let contents_of_boxed_number (ty_value : head_of_kind_value_non_null)
        : option flambda_type :=
      match ty_value with
      | Boxed_number k ty _ =>
        if boxable_number_eqb k kind then Some ty else None
      | Variant _ _ _ _ _ _ | Mutable_block _ | Closures _ _ | String _ =>
        None
      end
    in
    let* expanded := expand_head tenv t in
    match expanded with
    | Oub_value (Ok (Head_of_kind_value (Ok ty_value) _)) =>
      match contents_of_boxed_number ty_value with
      | None => Some Meet_shortcut.Invalid
      | Some ty =>
        let* canonical := canonical_simple_of_alias tenv min_name_mode ty in
        match canonical with
        | Some simple => Some (Meet_shortcut.Known_result simple)
        | None => Some Meet_shortcut.Need_meet
        end
      end
    | Oub_value (Ok (Head_of_kind_value Unknown _)) | Oub_value Unknown =>
      Some Meet_shortcut.Need_meet
    | Oub_value (Ok (Head_of_kind_value Bottom _)) | Oub_value Bottom =>
      Some Meet_shortcut.Invalid
    | Oub_naked_immediate _ | Oub_naked_number _ _ | Oub_rec_info _
    | Oub_region _ =>
      Some (wrong_kind Meet_shortcut.Invalid)
    end.

  (* [Provers.meet_boxed_float32_containing_simple], ...,
     [meet_boxed_mask_containing_simple]. *)
  Definition meet_boxed_float32_containing_simple :=
    meet_boxed_number_containing_simple Boxable_naked_float32.
  Definition meet_boxed_float_containing_simple :=
    meet_boxed_number_containing_simple Boxable_naked_float.
  Definition meet_boxed_int32_containing_simple :=
    meet_boxed_number_containing_simple Boxable_naked_int32.
  Definition meet_boxed_int64_containing_simple :=
    meet_boxed_number_containing_simple Boxable_naked_int64.
  Definition meet_boxed_nativeint_containing_simple :=
    meet_boxed_number_containing_simple Boxable_naked_nativeint.
  Definition meet_boxed_vec128_containing_simple :=
    meet_boxed_number_containing_simple Boxable_naked_vec128.
  Definition meet_boxed_vec256_containing_simple :=
    meet_boxed_number_containing_simple Boxable_naked_vec256.
  Definition meet_boxed_vec512_containing_simple :=
    meet_boxed_number_containing_simple Boxable_naked_vec512.
  Definition meet_boxed_mask_containing_simple :=
    meet_boxed_number_containing_simple Boxable_naked_mask.

  (* [Tag.Scannable.of_tag], as a test: a tag is that of a block that
     the garbage collector scans. *)
  Definition tag_is_scannable (tag : Z) : bool :=
    negb (Z.ltb tag 0 || Z.leb no_scan_tag tag).

  (* [Provers.prove_variant_like_generic_value]. *)
  Definition prove_variant_like_generic_value (tenv : Typing_env.t)
      (value_head : head_of_kind_value_non_null)
      : option (Generic_proof.t variant_like_proof) :=
    match value_head with
    | Variant _ immediates _ blocks _ _ =>
      match blocks with
      | Or_unknown.Unknown => Some Generic_proof.Unknown
      | Or_unknown.Known blocks =>
        match row_like_for_blocks_all_tags_and_sizes blocks with
        | Or_unknown.Unknown => Some Generic_proof.Unknown
        | Or_unknown.Known non_const_ctors_with_sizes =>
          let non_const_ctors_with_sizes
              : Or_unknown.t (Keyed.finite_map Z (nat * block_shape)) :=
            if
              Keyed.for_all_bindings_b Z.eqb
                (fun tag _ => tag_is_scannable tag)
                non_const_ctors_with_sizes
            then Or_unknown.Known non_const_ctors_with_sizes
            else Or_unknown.Unknown
          in
          match non_const_ctors_with_sizes with
          | Or_unknown.Unknown => Some Generic_proof.Unknown
          | Or_unknown.Known non_const_ctors_with_sizes =>
            let* const_ctors :=
              match immediates with
              | Or_unknown.Unknown => Some Or_unknown.Unknown
              | Or_unknown.Known imms =>
                let* proof := prove_naked_immediates_generic tenv imms in
                match proof with
                | Generic_proof.Unknown => Some Or_unknown.Unknown
                | Generic_proof.Invalid => Some (Or_unknown.Known [])
                | Generic_proof.Proved const_ctors =>
                  Some (Or_unknown.Known const_ctors)
                end
              end
            in
            Some
              (Generic_proof.Proved
                 (Variant_like_proof const_ctors non_const_ctors_with_sizes))
          end
        end
      end
    | Mutable_block _ => Some Generic_proof.Unknown
    | Closures _ _ | Boxed_number _ _ _ | String _ =>
      Some Generic_proof.Invalid
    end.

  (* [Provers.meet_variant_like]. *)
  Definition meet_variant_like (tenv : Typing_env.t) (t : flambda_type)
      : option (Meet_shortcut.t variant_like_proof) :=
    gen_value_to_meet prove_variant_like_generic_value tenv t.

  (* [Provers.prove_variant_like]. *)
  Definition prove_variant_like (tenv : Typing_env.t) (t : flambda_type)
      : option (Proof_of_property.t variant_like_proof) :=
    gen_value_to_proof prove_variant_like_generic_value tenv t.

  (* [Provers.prove_unique_tag_and_size_value]. *)
  Definition prove_unique_tag_and_size_value (tenv : Typing_env.t)
      (value_head : head_of_kind_value_non_null)
      : option
          (Generic_proof.t
             (Z * block_shape * nat * list flambda_type * Alloc_mode.t)) :=
    match value_head with
    | Variant _ immediates _ blocks _ _ =>
      match immediates with
      | Or_unknown.Unknown => Some Generic_proof.Unknown
      | Or_unknown.Known immediates =>
        let* immediates_bottom := is_bottom tenv immediates in
        if immediates_bottom
        then
          match blocks with
          | Or_unknown.Unknown => Some Generic_proof.Unknown
          | Or_unknown.Known blocks =>
            match row_like_for_blocks_get_singleton blocks with
            | None => Some Generic_proof.Unknown
            | Some (tag, shape, size, product, alloc_mode) =>
              Some
                (Generic_proof.Proved (tag, shape, size, product, alloc_mode))
            end
          end
        else Some Generic_proof.Unknown
      end
    | Mutable_block _ | Closures _ _ | Boxed_number _ _ _ | String _ =>
      Some Generic_proof.Unknown
    end.

  (* [Provers.prove_unique_tag_and_size]. *)
  Definition prove_unique_tag_and_size (tenv : Typing_env.t)
      (t : flambda_type)
      : option (Proof_of_property.t (Z * block_shape * nat)) :=
    let* proof := gen_value_to_proof prove_unique_tag_and_size_value tenv t in
    match proof with
    | Proof_of_property.Proved (tag, shape, size, _, _) =>
      Some (Proof_of_property.Proved (tag, shape, size))
    | Proof_of_property.Unknown => Some Proof_of_property.Unknown
    end.

  (* The function that
     [Provers.prove_unique_fully_constructed_immutable_heap_block]
     folds over the types of the fields: the [Simple]s that they are
     aliases of, last first, if all are aliases. *)
  Definition field_aliases_rev (result : Proof_of_property.t (list simple))
      (field_ty : flambda_type) : Proof_of_property.t (list simple) :=
    match result with
    | Proof_of_property.Unknown => result
    | Proof_of_property.Proved simples_rev =>
      match get_alias_opt field_ty with
      | None => Proof_of_property.Unknown
      | Some simple => Proof_of_property.Proved (simple :: simples_rev)
      end
    end.

  (* [Provers.prove_unique_fully_constructed_immutable_heap_block]. *)
  Definition prove_unique_fully_constructed_immutable_heap_block
      (tenv : Typing_env.t) (t : flambda_type)
      : option (Proof_of_property.t (Z * block_shape * nat * list simple)) :=
    let* proof := gen_value_to_proof prove_unique_tag_and_size_value tenv t in
    match proof with
    | Proof_of_property.Unknown
    | Proof_of_property.Proved
        (_, _, _, _, (Alloc_mode.Heap_or_local | Alloc_mode.Local)) =>
      Some Proof_of_property.Unknown
    | Proof_of_property.Proved (tag, shape, size, product, Alloc_mode.Heap) =>
      let result :=
        List.fold_left field_aliases_rev product (Proof_of_property.Proved [])
      in
      match result with
      | Proof_of_property.Unknown => Some Proof_of_property.Unknown
      | Proof_of_property.Proved simples =>
        Some (Proof_of_property.Proved (tag, shape, size, rev simples))
      end
    end.

  (* [Provers.meet_block_field_simple_value]. *)
  Definition meet_block_field_simple_value (min_name_mode : Name_mode.t)
      (field_kind : flambda_kind) (field_index : nat) (tenv : Typing_env.t)
      (value_head : head_of_kind_value_non_null)
      : option (Generic_proof.t simple) :=
    match value_head with
    | Variant _ _ _ blocks _ _ =>
      match blocks with
      | Or_unknown.Unknown => Some Generic_proof.Unknown
      | Or_unknown.Known blocks =>
        if row_like_for_blocks_is_bottom blocks
        then Some Generic_proof.Invalid
        else
          match row_like_for_blocks_get_field blocks field_index with
          | Bottom => Some Generic_proof.Invalid
          | Unknown => Some Generic_proof.Unknown
          | Ok ty =>
            if negb (kind_eqb (kind ty) field_kind)
            then Some Generic_proof.Invalid
            else
              let* canonical :=
                canonical_simple_of_alias tenv min_name_mode ty
              in
              match canonical with
              | Some simple => Some (Generic_proof.Proved simple)
              | None => Some Generic_proof.Unknown
              end
          end
      end
    | Mutable_block _ => Some Generic_proof.Unknown
    | Boxed_number _ _ _ | Closures _ _ | String _ =>
      Some Generic_proof.Invalid
    end.

  (* [Provers.meet_block_field_simple]. *)
  Definition meet_block_field_simple (tenv : Typing_env.t)
      (min_name_mode : Name_mode.t) (field_kind : flambda_kind)
      (t : flambda_type) (field_index : nat)
      : option (Meet_shortcut.t simple) :=
    gen_value_to_meet
      (meet_block_field_simple_value min_name_mode field_kind field_index)
      tenv t.

  (* The local function [of_singleton_type] of
     [Provers.prove_single_closures_entry_generic_value]. *)
  Definition single_closures_entry_of_singleton (alloc_mode : Alloc_mode.t)
      (exact : bool) (slot : function_slot) (entry : closures_entry)
      : Generic_proof.t
          (function_slot * Alloc_mode.t * closures_entry * function_type) :=
    match closures_entry_find_function_type entry exact slot with
    | Bottom => Generic_proof.Invalid
    | Unknown => Generic_proof.Unknown
    | Ok function_type =>
      Generic_proof.Proved (slot, alloc_mode, entry, function_type)
    end.

  (* [Provers.prove_single_closures_entry_generic_value]. *)
  Definition prove_single_closures_entry_generic_value (tenv : Typing_env.t)
      (value_head : head_of_kind_value_non_null)
      : option
          (Generic_proof.t
             (function_slot * Alloc_mode.t * closures_entry * function_type)) :=
    match value_head with
    | Closures by_function_slot alloc_mode =>
      let* single := row_like_for_closures_get_single_tag by_function_slot in
      match single with
      | No_singleton => Some Generic_proof.Unknown
      | Exact_closure slot entry =>
        Some (single_closures_entry_of_singleton alloc_mode true slot entry)
      | Incomplete_closure slot entry =>
        Some (single_closures_entry_of_singleton alloc_mode false slot entry)
      end
    | Variant _ _ _ _ _ _ | Mutable_block _ | Boxed_number _ _ _ | String _ =>
      Some Generic_proof.Invalid
    end.

  (* [Provers.meet_single_closures_entry]. *)
  Definition meet_single_closures_entry (tenv : Typing_env.t)
      (t : flambda_type)
      : option
          (Meet_shortcut.t
             (function_slot * Alloc_mode.t * closures_entry * function_type)) :=
    gen_value_to_meet prove_single_closures_entry_generic_value tenv t.

  (* [Provers.prove_single_closures_entry]. *)
  Definition prove_single_closures_entry (tenv : Typing_env.t)
      (t : flambda_type)
      : option
          (Proof_of_property.t
             (function_slot * Alloc_mode.t * closures_entry * function_type)) :=
    gen_value_to_proof prove_single_closures_entry_generic_value tenv t.

  (* The function that [Provers.prove_code_ids_generic_value] folds over
     the cases of the closures: the code id of the function type that
     the case has for its own function slot, added to those found.
     [Or_unknown.Unknown] is the exception [Unknown_code_id]: a case
     that has no such function type, or an unknown one. *)
  Definition add_code_id_of_case (acc : Or_unknown.t (list code_id))
      (binding : function_slot * row_like_closures_case)
      : Or_unknown.t (list code_id) :=
    match acc with
    | Or_unknown.Unknown => Or_unknown.Unknown
    | Or_unknown.Known code_ids =>
      match case_maps_to (snd binding) with
      | Closures_entry function_types _ _ =>
        match lookup (fst binding) function_types with
        | Some Or_unknown.Unknown | None => Or_unknown.Unknown
        | Some (Or_unknown.Known function_type) =>
          Or_unknown.Known (function_type_code_id function_type :: code_ids)
        end
      end
    end.

  (* [Provers.prove_code_ids_generic_value]. *)
  Definition prove_code_ids_generic_value (tenv : Typing_env.t)
      (value_head : head_of_kind_value_non_null)
      : option (Generic_proof.t (list code_id)) :=
    match value_head with
    | Closures (Row_like_for_closures known_closures) _ =>
      match
        List.fold_left add_code_id_of_case (compact known_closures)
          (Or_unknown.Known [])
      with
      | Or_unknown.Unknown => Some Generic_proof.Unknown
      | Or_unknown.Known code_ids =>
        if Keyed.is_empty code_ids
        then Some Generic_proof.Invalid
        else Some (Generic_proof.Proved code_ids)
      end
    | Variant _ _ _ _ _ _ | Mutable_block _ | Boxed_number _ _ _ | String _ =>
      Some Generic_proof.Invalid
    end.

  (* [Provers.meet_code_ids]. *)
  Definition meet_code_ids (tenv : Typing_env.t) (t : flambda_type)
      : option (Meet_shortcut.t (list code_id)) :=
    gen_value_to_meet prove_code_ids_generic_value tenv t.

  (* [Provers.prove_code_ids]. *)
  Definition prove_code_ids (tenv : Typing_env.t) (t : flambda_type)
      : option (Proof_of_property.t (list code_id)) :=
    gen_value_to_proof prove_code_ids_generic_value tenv t.

  (* [Provers.meet_project_function_slot_simple_value]. *)
  Definition meet_project_function_slot_simple_value
      (min_name_mode : Name_mode.t) (slot : function_slot)
      (tenv : Typing_env.t) (value_head : head_of_kind_value_non_null)
      : option (Generic_proof.t simple) :=
    match value_head with
    | Closures by_function_slot _ =>
      let* closure :=
        row_like_for_closures_get_closure by_function_slot slot
      in
      match closure with
      | Or_unknown.Unknown => Some Generic_proof.Unknown
      | Or_unknown.Known ty =>
        let* canonical := canonical_simple_of_alias tenv min_name_mode ty in
        match canonical with
        | Some simple => Some (Generic_proof.Proved simple)
        | None => Some Generic_proof.Unknown
        end
      end
    | Variant _ _ _ _ _ _ | Mutable_block _ | Boxed_number _ _ _ | String _ =>
      Some Generic_proof.Invalid
    end.

  (* [Provers.meet_project_function_slot_simple]. *)
  Definition meet_project_function_slot_simple (tenv : Typing_env.t)
      (min_name_mode : Name_mode.t) (t : flambda_type) (slot : function_slot)
      : option (Meet_shortcut.t simple) :=
    gen_value_to_meet
      (meet_project_function_slot_simple_value min_name_mode slot) tenv t.

  (* [Provers.meet_project_value_slot_simple_value]. The kind of the
     value slot ([Value_slot.kind]) is an argument: a value slot of the
     model has no kind. *)
  Definition meet_project_value_slot_simple_value
      (min_name_mode : Name_mode.t) (slot : value_slot)
      (value_slot_kind : flambda_kind) (tenv : Typing_env.t)
      (value_head : head_of_kind_value_non_null)
      : option (Generic_proof.t simple) :=
    match value_head with
    | Closures by_function_slot _ =>
      let* env_var :=
        row_like_for_closures_get_env_var by_function_slot slot
      in
      match env_var with
      | Or_unknown.Unknown => Some Generic_proof.Unknown
      | Or_unknown.Known ty =>
        if negb (kind_eqb (kind ty) value_slot_kind)
        then Some Generic_proof.Invalid
        else
          let* canonical :=
            canonical_simple_of_alias tenv min_name_mode ty
          in
          match canonical with
          | Some simple => Some (Generic_proof.Proved simple)
          | None => Some Generic_proof.Unknown
          end
      end
    | Variant _ _ _ _ _ _ | Mutable_block _ | Boxed_number _ _ _ | String _ =>
      Some Generic_proof.Invalid
    end.

  (* [Provers.meet_project_value_slot_simple]. *)
  Definition meet_project_value_slot_simple (tenv : Typing_env.t)
      (min_name_mode : Name_mode.t) (t : flambda_type) (slot : value_slot)
      (value_slot_kind : flambda_kind) : option (Meet_shortcut.t simple) :=
    gen_value_to_meet
      (meet_project_value_slot_simple_value min_name_mode slot
         value_slot_kind)
      tenv t.

  (* The local function [incompatible_naked_numbers] of
     [Provers.prove_physical_equality]: two types of naked numbers of
     one kind, whose sets of numbers have none in common. *)
  Definition incompatible_naked_numbers (tenv : Typing_env.t)
      (t1 t2 : flambda_type) : option bool :=
    let* expanded1 := expand_head tenv t1 in
    let* expanded2 := expand_head tenv t2 in
    match expanded1, expanded2 with
    | Oub_naked_number k1 (Ok s1), Oub_naked_number k2 (Ok s2) =>
      Some
        (naked_number_kind_eqb k1 k2
         && Keyed.is_empty (Keyed.inter Z.eqb s1 s2))
    | _, _ => Some false
    end.

  (* The local definition [immediates_equality] of
     [Provers.prove_physical_equality], for the immediates of two
     variants. [Invalid]: neither can be an immediate. [Proved false]:
     they are not both immediates and equal. [Proved true]: if both
     are immediates, they are equal. The inner [None] is the
     assertion that two singletons that meet are the same number. *)
  Definition physical_equality_of_immediates (tenv : Typing_env.t)
      (imms1 imms2 : flambda_type) : option (Generic_proof.t bool) :=
    let* proof1 := prove_naked_immediates_generic tenv imms1 in
    let* proof2 := prove_naked_immediates_generic tenv imms2 in
    match proof1, proof2 with
    | Generic_proof.Invalid, Generic_proof.Invalid =>
      Some Generic_proof.Invalid
    | Generic_proof.Invalid, _ | _, Generic_proof.Invalid =>
      Some (Generic_proof.Proved false)
    | Generic_proof.Unknown, _ | _, Generic_proof.Unknown =>
      Some Generic_proof.Unknown
    | Generic_proof.Proved imms1, Generic_proof.Proved imms2 =>
      if Keyed.is_empty (Keyed.inter Z.eqb imms1 imms2)
      then Some (Generic_proof.Proved false)
      else
        match
          Keyed.get_singleton Z.eqb imms1, Keyed.get_singleton Z.eqb imms2
        with
        | None, _ | _, None => Some Generic_proof.Unknown
        | Some imm1, Some imm2 =>
          if Z.eqb imm1 imm2 then Some (Generic_proof.Proved true) else None
        end
    end.

  (* The local definition [blocks_equality] of
     [Provers.prove_physical_equality], for the blocks of two
     variants, with the same reading of the answers. *)
  Definition physical_equality_of_blocks (blocks1 blocks2 : row_like_for_blocks)
      : Generic_proof.t bool :=
    let is_bottom1 := row_like_for_blocks_is_bottom blocks1 in
    let is_bottom2 := row_like_for_blocks_is_bottom blocks2 in
    if is_bottom1 && is_bottom2
    then Generic_proof.Invalid
    else if is_bottom1 || is_bottom2
    then Generic_proof.Proved false
    else
      match
        row_like_for_blocks_get_singleton blocks1,
        row_like_for_blocks_get_singleton blocks2
      with
      | None, _ | _, None => Generic_proof.Unknown
      | Some (tag1, shape1, size1, _, _), Some (tag2, shape2, size2, _, _) =>
        if Z.eqb tag1 tag2 && Nat.eqb size1 size2
           && block_shape_eqb shape1 shape2
        then Generic_proof.Unknown
        else Generic_proof.Proved false
      end.

  (* The test of the first two cases of two variants in
     [Provers.prove_physical_equality]: the immediates of one variant
     are obviously bottom and the blocks of the other are bottom. *)
  Definition no_immediate_and_no_block
      (immediates : Or_unknown.t flambda_type)
      (blocks : Or_unknown.t row_like_for_blocks) : bool :=
    match immediates, blocks with
    | Or_unknown.Known imms, Or_unknown.Known blocks =>
      is_obviously_bottom imms && row_like_for_blocks_is_bottom blocks
    | _, _ => false
    end.

  (* The last [match] of the case of two variants in
     [Provers.prove_physical_equality]: the proof that the answers for
     the immediates and for the blocks make. *)
  Definition physical_equality_of_parts
      (immediates_equality blocks_equality : Generic_proof.t bool)
      : Proof_of_property.t bool :=
    match immediates_equality, blocks_equality with
    | Generic_proof.Proved b, Generic_proof.Invalid
    | Generic_proof.Invalid, Generic_proof.Proved b =>
      Proof_of_property.Proved b
    | Generic_proof.Proved false, Generic_proof.Proved false =>
      Proof_of_property.Proved false
    | _, _ => Proof_of_property.Unknown
    end.

  (* The case of two variants of [Provers.prove_physical_equality]. *)
  Definition physical_equality_of_variants (tenv : Typing_env.t)
      (immediates1 immediates2 : Or_unknown.t flambda_type)
      (blocks1 blocks2 : Or_unknown.t row_like_for_blocks)
      : option (Proof_of_property.t bool) :=
    if no_immediate_and_no_block immediates1 blocks2
    then Some (Proof_of_property.Proved false)
    else if no_immediate_and_no_block immediates2 blocks1
    then Some (Proof_of_property.Proved false)
    else
      match immediates1, immediates2, blocks1, blocks2 with
      | Or_unknown.Known imms1, Or_unknown.Known imms2,
        Or_unknown.Known blocks1, Or_unknown.Known blocks2 =>
        let* immediates_equality :=
          physical_equality_of_immediates tenv imms1 imms2
        in
        let blocks_equality := physical_equality_of_blocks blocks1 blocks2 in
        Some (physical_equality_of_parts immediates_equality blocks_equality)
      | _, _, _, _ => Some Proof_of_property.Unknown
      end.

  (* The cases of two heads that are not null of
     [Provers.prove_physical_equality]. *)
  Definition physical_equality_of_heads (tenv : Typing_env.t)
      (head1 head2 : head_of_kind_value_non_null)
      : option (Proof_of_property.t bool) :=
    match head1, head2 with
    | Boxed_number k1 contents1 _, Boxed_number k2 contents2 _ =>
      if boxable_number_eqb k1 k2
      then
        let* incompatible :=
          incompatible_naked_numbers tenv contents1 contents2
        in
        if incompatible
        then Some (Proof_of_property.Proved false)
        else Some Proof_of_property.Unknown
      else Some (Proof_of_property.Proved false)
    | Closures _ _, Closures _ _ => Some Proof_of_property.Unknown
    | String strs1, String strs2 =>
      if Keyed.is_empty (Keyed.inter (lists_eqb Z.eqb) strs1 strs2)
      then Some (Proof_of_property.Proved false)
      else Some Proof_of_property.Unknown
    | Variant _ immediates1 _ blocks1 _ _,
      Variant _ immediates2 _ blocks2 _ _ =>
      physical_equality_of_variants tenv immediates1 immediates2 blocks1
        blocks2
    | Variant _ _ _ blocks _ _, Mutable_block _
    | Mutable_block _, Variant _ _ _ blocks _ _ =>
      match blocks with
      | Or_unknown.Known blocks =>
        if row_like_for_blocks_is_bottom blocks
        then Some (Proof_of_property.Proved false)
        else Some Proof_of_property.Unknown
      | Or_unknown.Unknown => Some Proof_of_property.Unknown
      end
    | Mutable_block _, Mutable_block _ => Some Proof_of_property.Unknown
    | Boxed_number _ _ _,
      (Variant _ _ _ _ _ _ | Mutable_block _ | Closures _ _ | String _)
    | (Variant _ _ _ _ _ _ | Mutable_block _ | Closures _ _ | String _),
      Boxed_number _ _ _ =>
      Some (Proof_of_property.Proved false)
    | Closures _ _, (Variant _ _ _ _ _ _ | Mutable_block _ | String _)
    | (Variant _ _ _ _ _ _ | Mutable_block _ | String _), Closures _ _ =>
      Some (Proof_of_property.Proved false)
    | String _, (Variant _ _ _ _ _ _ | Mutable_block _)
    | (Variant _ _ _ _ _ _ | Mutable_block _), String _ =>
      Some (Proof_of_property.Proved false)
    end.

  (* The local function [check_heads] of
     [Provers.prove_physical_equality]. *)
  Definition physical_equality_check_heads (tenv : Typing_env.t)
      (t1 t2 : flambda_type) : option (Proof_of_property.t bool) :=
    let* expanded1 := expand_head tenv t1 in
    let* expanded2 := expand_head tenv t2 in
    match expanded1, expanded2 with
    | (Oub_naked_immediate _ | Oub_naked_number _ _ | Oub_rec_info _
      | Oub_region _), _
    | _, (Oub_naked_immediate _ | Oub_naked_number _ _ | Oub_rec_info _
         | Oub_region _) =>
      Some (wrong_kind Proof_of_property.Unknown)
    | Oub_value (Unknown | Bottom), _ | _, Oub_value (Unknown | Bottom) =>
      Some Proof_of_property.Unknown
    | Oub_value (Ok (Head_of_kind_value non_null1 is_null1)),
      Oub_value (Ok (Head_of_kind_value non_null2 is_null2)) =>
      match is_null1, is_null2 with
      | Maybe_null _, Maybe_null _ =>
        match non_null1, non_null2 with
        | Bottom, Bottom => Some (Proof_of_property.Proved true)
        | _, _ => Some Proof_of_property.Unknown
        end
      | Maybe_null _, Not_null | Not_null, Maybe_null _ =>
        Some Proof_of_property.Unknown
      | Not_null, Not_null =>
        match non_null1, non_null2 with
        | Ok head1, Ok head2 => physical_equality_of_heads tenv head1 head2
        | _, _ => Some Proof_of_property.Unknown
        end
      end
    end.

  (* [Provers.prove_physical_equality]. *)
  Definition prove_physical_equality (tenv : Typing_env.t)
      (t1 t2 : flambda_type) : option (Proof_of_property.t bool) :=
    match get_alias_opt t1, get_alias_opt t2 with
    | Some s1, Some s2 =>
      match s1, s2 with
      | Const c1, Const c2 =>
        if const_eqb c1 c2
        then Some (Proof_of_property.Proved true)
        else Some (Proof_of_property.Proved false)
      | Const _, Symbol _ | Symbol _, Const _ =>
        Some (Proof_of_property.Proved false)
      | Symbol sym1, Symbol sym2 =>
        if Pos.eqb sym1 sym2
        then Some (Proof_of_property.Proved true)
        else Some (Proof_of_property.Proved false)
      | Var var1, Var var2 =>
        if Pos.eqb var1 var2
        then Some (Proof_of_property.Proved true)
        else physical_equality_check_heads tenv t1 t2
      | Const _, Var _ | Symbol _, Var _ | Var _, Const _ | Var _, Symbol _ =>
        physical_equality_check_heads tenv t1 t2
      end
    | None, Some _ | Some _, None | None, None =>
      physical_equality_check_heads tenv t1 t2
    end.

  (* [Provers.prove_alloc_mode_of_boxed_number_value]. *)
  Definition prove_alloc_mode_of_boxed_number_value (tenv : Typing_env.t)
      (value_head : head_of_kind_value_non_null)
      : option (Generic_proof.t Alloc_mode.t) :=
    match value_head with
    | Boxed_number _ _ alloc_mode => Some (Generic_proof.Proved alloc_mode)
    | Variant _ _ _ _ _ _ | Mutable_block _ | String _ | Closures _ _ =>
      Some Generic_proof.Unknown
    end.

  (* [Provers.prove_alloc_mode_of_boxed_number]. *)
  Definition prove_alloc_mode_of_boxed_number (tenv : Typing_env.t)
      (t : flambda_type) : option (Proof_of_property.t Alloc_mode.t) :=
    gen_value_to_proof prove_alloc_mode_of_boxed_number_value tenv t.

  (* [Provers.never_holds_locally_allocated_values]. *)
  Definition never_holds_locally_allocated_values (tenv : Typing_env.t)
      (var : variable) : option (Proof_of_property.t unit) :=
    let heap_only (alloc_mode : Alloc_mode.t) : Proof_of_property.t unit :=
      match alloc_mode with
      | Alloc_mode.Heap => Proof_of_property.Proved tt
      | Alloc_mode.Local | Alloc_mode.Heap_or_local =>
        Proof_of_property.Unknown
      end
    in
    let* ty := Typing_env.find tenv (Name_var var) None in
    let* expanded := expand_head tenv ty in
    match expanded with
    | Oub_value (Ok (Head_of_kind_value (Unknown | Bottom) _))
    | Oub_value (Unknown | Bottom) =>
      Some Proof_of_property.Unknown
    | Oub_value (Ok (Head_of_kind_value (Ok value_head) _)) =>
      match value_head with
      | Variant _ _ _ blocks _ _ =>
        match blocks with
        | Or_unknown.Unknown => Some Proof_of_property.Unknown
        | Or_unknown.Known blocks =>
          if row_like_for_blocks_is_bottom blocks
          then Some (Proof_of_property.Proved tt)
          else
            match blocks with
            | Row_like_for_blocks _ _ alloc_mode =>
              Some (heap_only alloc_mode)
            end
        end
      | Boxed_number _ _ alloc_mode | Mutable_block alloc_mode
      | Closures _ alloc_mode =>
        Some (heap_only alloc_mode)
      | String _ => Some (Proof_of_property.Proved tt)
      end
    | Oub_naked_immediate _ | Oub_naked_number _ _ | Oub_rec_info _
    | Oub_region _ =>
      Some (Proof_of_property.Proved tt)
    end.

End Provers.
