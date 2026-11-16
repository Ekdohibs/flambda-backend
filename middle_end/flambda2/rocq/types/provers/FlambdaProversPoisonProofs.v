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


(* Proofs of the statements of FlambdaProversPoisonSpec.v. Nothing here
   needs review.

   Each is the lemma of FlambdaProversProofs.v that is made for the two
   notions of satisfaction (its name ends in [_in]), at the notion that
   exempts the names that hold poison. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaValueSpec FlambdaValueProofs FlambdaPrimitiveModel
  FlambdaPrimitiveSpec FlambdaPrimitiveProofs FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypeWellFormedSpec FlambdaBindingTime
  FlambdaEnvAliases FlambdaEnvAliasesSpec FlambdaTypingEnv
  FlambdaTypingEnvSpec FlambdaTypingEnvInvariantSpec FlambdaExpandHead
  FlambdaProvers FlambdaProversSpec FlambdaProversProofs
  FlambdaProversPoisonSpec.
Import ListNotations.

Theorem prove_is_int_sound_exempt : prove_is_int_sound_exempt_statement.
Proof. exact (prove_is_int_sound_proof_in (m := Exempt)). Qed.

Theorem meet_is_int_variant_only_sound_exempt :
  meet_is_int_variant_only_sound_exempt_statement.
Proof. exact (meet_is_int_variant_only_sound_proof_in (m := Exempt)). Qed.

Theorem prove_is_not_a_pointer_sound_exempt :
  prove_is_not_a_pointer_sound_exempt_statement.
Proof. exact (prove_is_not_a_pointer_sound_proof_in (m := Exempt)). Qed.

Theorem prove_get_tag_sound_exempt : prove_get_tag_sound_exempt_statement.
Proof. exact (prove_get_tag_sound_proof_in (m := Exempt)). Qed.

Theorem meet_is_null_sound_exempt : meet_is_null_sound_exempt_statement.
Proof. exact (meet_is_null_sound_proof_in (m := Exempt)). Qed.

Theorem meet_naked_number_sound_exempt :
  meet_naked_number_sound_exempt_statement.
Proof. exact (meet_naked_number_sound_proof_in (m := Exempt)). Qed.

Theorem prove_equals_to_simple_of_kind_sound_exempt :
  prove_equals_to_simple_of_kind_sound_exempt_statement.
Proof. exact (prove_equals_to_simple_of_kind_sound_proof_in (m := Exempt)). Qed.

Theorem prove_is_a_boxed_or_tagged_number_sound_exempt :
  prove_is_a_boxed_or_tagged_number_sound_exempt_statement.
Proof.
  exact (prove_is_a_boxed_or_tagged_number_sound_proof_in (m := Exempt)).
Qed.

Theorem prove_is_a_tagged_immediate_sound_exempt :
  prove_is_a_tagged_immediate_sound_exempt_statement.
Proof. exact (prove_is_a_tagged_immediate_sound_proof_in (m := Exempt)). Qed.

Theorem prove_is_a_boxed_number_sound_exempt :
  prove_is_a_boxed_number_sound_exempt_statement.
Proof. exact (prove_is_a_boxed_number_sound_proof_in (m := Exempt)). Qed.

Theorem prove_is_or_is_not_a_boxed_float_sound_exempt :
  prove_is_or_is_not_a_boxed_float_sound_exempt_statement.
Proof.
  exact (prove_is_or_is_not_a_boxed_float_sound_proof_in (m := Exempt)).
Qed.

Theorem meet_boxed_number_containing_simple_sound_exempt :
  meet_boxed_number_containing_simple_sound_exempt_statement.
Proof.
  exact (meet_boxed_number_containing_simple_sound_proof_in (m := Exempt)).
Qed.

Theorem prove_strings_sound_exempt : prove_strings_sound_exempt_statement.
Proof. exact (prove_strings_sound_proof_in (m := Exempt)). Qed.

Theorem meet_strings_sound_exempt : meet_strings_sound_exempt_statement.
Proof. exact (meet_strings_sound_proof_in (m := Exempt)). Qed.

Theorem prove_unique_tag_and_size_sound_exempt :
  prove_unique_tag_and_size_sound_exempt_statement.
Proof. exact (prove_unique_tag_and_size_sound_proof_in (m := Exempt)). Qed.

Theorem prove_unique_immutable_heap_block_sound_exempt :
  prove_unique_immutable_heap_block_sound_exempt_statement.
Proof.
  exact (prove_unique_immutable_heap_block_sound_proof_in (m := Exempt)).
Qed.

Theorem prove_unique_immutable_heap_block_fields_exempt :
  prove_unique_immutable_heap_block_fields_exempt_statement.
Proof.
  exact (prove_unique_immutable_heap_block_fields_proof_in (m := Exempt)).
Qed.

Theorem meet_block_field_simple_sound_exempt :
  meet_block_field_simple_sound_exempt_statement.
Proof. exact (meet_block_field_simple_sound_proof_in (m := Exempt)). Qed.

Theorem prove_single_closures_entry_sound_exempt :
  prove_single_closures_entry_sound_exempt_statement.
Proof. exact (prove_single_closures_entry_sound_proof_in (m := Exempt)). Qed.

Theorem meet_single_closures_entry_sound_exempt :
  meet_single_closures_entry_sound_exempt_statement.
Proof. exact (meet_single_closures_entry_sound_proof_in (m := Exempt)). Qed.

Theorem prove_code_ids_sound_exempt : prove_code_ids_sound_exempt_statement.
Proof. exact (prove_code_ids_sound_proof_in (m := Exempt)). Qed.

Theorem meet_code_ids_sound_exempt : meet_code_ids_sound_exempt_statement.
Proof. exact (meet_code_ids_sound_proof_in (m := Exempt)). Qed.

Theorem meet_project_function_slot_simple_sound_exempt :
  meet_project_function_slot_simple_sound_exempt_statement.
Proof.
  exact (meet_project_function_slot_simple_sound_proof_in (m := Exempt)).
Qed.

Theorem meet_project_value_slot_simple_sound_exempt :
  meet_project_value_slot_simple_sound_exempt_statement.
Proof. exact (meet_project_value_slot_simple_sound_proof_in (m := Exempt)). Qed.

Theorem meet_project_value_slot_simple_known_exempt :
  meet_project_value_slot_simple_known_exempt_statement.
Proof. exact (meet_project_value_slot_simple_known_proof_in (m := Exempt)). Qed.

(* ======================================================================== *)
(* Tagging, where the answer is not a constant                              *)
(* ======================================================================== *)

Lemma inspect_tagging_immediates_name_sound (statics : static_env)
    (valuation : env) (tenv : Typing_env.t) (min_name_mode : Name_mode.t)
    (immediates : Or_unknown.t flambda_type) (i : Z) (s : simple) :
  typing_env_satisfied statics valuation tenv ->
  match immediates with
  | Or_unknown.Unknown => True
  | Or_unknown.Known ty =>
    type_satisfied statics valuation (Value_const (Naked_immediate i)) ty
  end ->
  Provers.inspect_tagging_immediates min_name_mode tenv immediates
  = Some (Generic_proof.Proved s) ->
  (∀ c : const, s <> Const c) ->
  eval_simple valuation s = Some (Value_const (Naked_immediate i)).
Proof.
  intros Hsat Himms. unfold Provers.inspect_tagging_immediates.
  destruct immediates as [ty | ]; [ | discriminate ].
  destruct (Provers.canonical_simple_of_alias tenv min_name_mode ty)
    as [[canonical | ] | ] eqn:Hcanonical;
    [ | | discriminate ].
  - intros [= <-] _.
    exact
      (canonical_simple_of_alias_sound_in (m := Exempt) _ _ _ _ _ _ _ Hsat
         Himms Hcanonical).
  - destruct (Provers.meet_naked_immediates tenv ty) as [[imms | | ] | ];
      try discriminate.
    destruct (Keyed.get_singleton Z.eqb imms) as [imm | ];
      [ | discriminate ].
    intros [= <-] Hname. destruct (Hname _ eq_refl).
Qed.

Lemma inspect_tagging_of_simple_value_name_sound
    (proof_kind : Provers.tagging_proof_kind) (min_name_mode : Name_mode.t)
    (statics : static_env) (valuation : env) (tenv : Typing_env.t)
    (head : head_of_kind_value_non_null) (v : value) (s : simple) :
  typing_env_satisfied statics valuation tenv ->
  head_of_kind_value_non_null_satisfied statics valuation v head ->
  match proof_kind with
  | Provers.Prove => True
  | Provers.Meet => is_immediate v
  end ->
  Provers.inspect_tagging_of_simple_value proof_kind min_name_mode tenv head
  = Some (Generic_proof.Proved s) ->
  (∀ c : const, s <> Const c) ->
  tagging_of valuation s v.
Proof.
  intros Hsat Hhead Hdomain.
  destruct head as [is_int immediates get_tag blocks extensions is_unique
                   | alloc_mode | k contents alloc_mode
                   | by_function_slot alloc_mode | strs];
    simpl; try (destruct proof_kind; discriminate).
  assert (Hinspect :
    (∃ i : Z,
      v = Value_const (Tagged_immediate i)
      ∧ match immediates with
        | Or_unknown.Unknown => True
        | Or_unknown.Known ty =>
          type_satisfied statics valuation (Value_const (Naked_immediate i))
            ty
        end) ->
    Provers.inspect_tagging_immediates min_name_mode tenv immediates
    = Some (Generic_proof.Proved s) ->
    (∀ c : const, s <> Const c) ->
    tagging_of valuation s v).
  { intros [i [-> Himms]] Hproved Hname.
    exists i. split; [ reflexivity | ].
    exact
      (inspect_tagging_immediates_name_sound _ _ _ _ _ _ _ Hsat Himms Hproved
         Hname). }
  destruct proof_kind.
  - destruct blocks as [blocks | ]; [ | discriminate ].
    destruct (row_like_for_blocks_is_bottom blocks) eqn:Hblocks;
      [ | discriminate ].
    apply Hinspect.
    exact (variant_no_blocks _ _ _ _ _ _ _ _ _ Hhead Hblocks).
  - assert (Himmediate :
      ∃ i : Z,
        v = Value_const (Tagged_immediate i)
        ∧ match immediates with
          | Or_unknown.Unknown => True
          | Or_unknown.Known ty =>
            type_satisfied statics valuation
              (Value_const (Naked_immediate i)) ty
          end).
    { destruct (variant_cases _ _ _ _ _ _ _ _ _ Hhead)
        as [Himmediate | [tag [fields [Hblock _]]]];
        [ exact Himmediate | ].
      rewrite (immediate_not_block statics v Hdomain) in Hblock.
      discriminate Hblock. }
    destruct blocks; exact (Hinspect Himmediate).
Qed.

Theorem prove_tagging_of_simple_name_exempt :
  prove_tagging_of_simple_name_exempt_statement.
Proof.
  intros min_name_mode.
  apply
    (gen_value_to_proof_proved_sound_in (m := Exempt) _ any_input any_head).
  - exact any_input_to_head.
  - intros statics valuation tenv head v s Hsat _ Hhead Hproved.
    exact
      (inspect_tagging_of_simple_value_name_sound Provers.Prove min_name_mode
         statics valuation tenv head v s Hsat Hhead I Hproved).
Qed.

Theorem meet_tagging_of_simple_name_exempt :
  meet_tagging_of_simple_name_exempt_statement.
Proof.
  intros statics valuation tenv min_name_mode t s v Hsat Hv Himmediate Hknown.
  assert (Hok : ~ is_poison v).
  { destruct Himmediate as [i ->]. intros [k Hk]. discriminate Hk. }
  assert (Hnot_null : v <> Value_const Null).
  { destruct Himmediate as [i ->]. discriminate. }
  exact
    (gen_value_to_meet_known_sound_in (m := Exempt) _
       (fun _ valuation s v =>
          (∀ c : const, s <> Const c) -> tagging_of valuation s v)
       statics valuation tenv t s v
       (fun head Hhead Hproved =>
          inspect_tagging_of_simple_value_name_sound Provers.Meet
            min_name_mode statics valuation tenv head v s Hsat Hhead
            Himmediate Hproved)
       Hsat Hok Hv Hnot_null Hknown).
Qed.

(* ======================================================================== *)
(* What Simplify does with [Provers.prove_is_int]                           *)
(* ======================================================================== *)

(* [Is_int] has no value on poison. *)
Lemma is_int_not_poison (statics : static_env) (variant_only : bool)
    (v r : value) :
  eval_prim statics (Unary (Is_int variant_only)) [v] = Some r ->
  ~ is_poison v.
Proof.
  intros Heval [k ->]. rewrite eval_prim_unary in Heval.
  destruct variant_only; discriminate Heval.
Qed.

Theorem prove_is_int_is_int_exempt : prove_is_int_is_int_exempt_statement.
Proof.
  intros statics valuation tenv t b v r Hsat Hv Hproved Heval.
  pose proof
    (prove_is_int_sound_proof_in (m := Exempt) statics valuation tenv t b v
       Hsat (is_int_not_poison _ _ _ _ Heval) I Hv Hproved)
    as Hb.
  destruct (is_int_defined _ _ _ _ Heval) as [_ [b' [-> Hb']]].
  rewrite (is_int_is_unique _ _ _ Hb Hb'). reflexivity.
Qed.

Theorem prove_is_int_get_tag_exempt : prove_is_int_get_tag_exempt_statement.
Proof.
  intros statics valuation tenv t v Hsat Hv Hproved.
  rewrite eval_prim_unary. simpl.
  destruct (block_of statics v) as [[tag fields] | ] eqn:Hblock;
    [ | reflexivity ].
  exfalso.
  assert (Hnot : ~ is_poison v).
  { intros [k ->]. discriminate Hblock. }
  pose proof
    (prove_is_int_sound_proof_in (m := Exempt) statics valuation tenv t true
       v Hsat Hnot I Hv Hproved)
    as Himmediate.
  simpl in Himmediate.
  rewrite (immediate_not_block statics v Himmediate) in Hblock.
  discriminate Hblock.
Qed.
