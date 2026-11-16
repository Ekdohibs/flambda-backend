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


(* Proofs of the statements of FlambdaProversRenameSpec.v. Nothing here
   needs review.

   The expansion of a head first; then each prover on heads, once, and
   three lemmas that take it through [Provers.gen_value_to_gen],
   [Provers.gen_value_to_proof] and [Provers.gen_value_to_meet]. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaMapProofs FlambdaCompareSpec FlambdaNominalSpec
  FlambdaNominalProofs.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeEqualSpec FlambdaTypeAccessSpec FlambdaTypeAccessProofs
  FlambdaBindingTime FlambdaEnvAliases FlambdaTypingEnv FlambdaTypeRenameSpec
  FlambdaTypeRenameProofs FlambdaEnvRenameSpec FlambdaEnvRenameProofs
  FlambdaExpandHead FlambdaProvers FlambdaReify FlambdaProversRenameSpec.
Import ListNotations.

(* ======================================================================== *)
(* The expansion of a head                                                  *)
(* ======================================================================== *)

Lemma of_non_alias_type_renamed (r : renaming) (t : flambda_type) :
  Expanded_type.of_non_alias_type (rename_type r t)
  = option_map (rename_expanded r) (Expanded_type.of_non_alias_type t).
Proof.
  destruct t as [ty | ty | k ty | ty | ty];
    destruct ty as [ | [head | s] | ]; reflexivity.
Qed.

Lemma of_non_alias_type_unknown (r : renaming) (k : flambda_kind) :
  Expanded_type.of_non_alias_type (unknown k)
  = option_map (rename_expanded r)
      (Expanded_type.of_non_alias_type (unknown k)).
Proof. destruct k as [ | [] | | ]; reflexivity. Qed.

Lemma create_const_renamed (r : renaming) (c : const) :
  rename_expanded r (Expanded_type.create_const c)
  = Expanded_type.create_const c.
Proof. destruct c; reflexivity. Qed.

Lemma expand_head_of_alias_type_renamed (r : renaming) (tenv : Typing_env.t)
    (k : flambda_kind) (s : simple) :
  expand_head_of_alias_type (rename_typing_env r tenv) k (rename_simple r s)
  = option_map (rename_expanded r) (expand_head_of_alias_type tenv k s).
Proof.
  assert
    (Hname : ∀ n : name,
       match Typing_env.find (rename_typing_env r tenv) (rename_name r n)
               (Some k) with
       | Some ty => Expanded_type.of_non_alias_type ty
       | None => None
       end
       = option_map (rename_expanded r)
           match Typing_env.find tenv n (Some k) with
           | Some ty => Expanded_type.of_non_alias_type ty
           | None => None
           end).
  { intros n. rewrite find_equivariant.
    destruct (Typing_env.find tenv n (Some k)) as [ty | ]; simpl;
      [ apply of_non_alias_type_renamed | reflexivity ]. }
  destruct s as [x | sym | c]; simpl.
  - apply (Hname (Name_var x)).
  - apply (Hname (Name_symbol sym)).
  - rewrite create_const_renamed. reflexivity.
Qed.

Theorem expand_head_rename : expand_head_rename_statement.
Proof.
  intros r tenv t. unfold expand_head.
  rewrite get_alias_opt_renamed, kind_rename_type.
  destruct (get_alias_opt t) as [s | ]; simpl;
    [ | apply of_non_alias_type_renamed ].
  rewrite get_canonical_simple_exn_equivariant.
  destruct
    (Typing_env.get_canonical_simple_exn tenv (Some Name_mode.in_types) None
       s)
    as [[canonical | ] | ]; simpl.
  - apply expand_head_of_alias_type_renamed.
  - apply of_non_alias_type_unknown.
  - reflexivity.
Qed.

(* ======================================================================== *)
(* The view of an expanded type                                             *)
(* ======================================================================== *)

Lemma descr_oub_of_expanded_renamed (r : renaming)
    (expanded : Expanded_type.t) :
  descr_oub_of_expanded (rename_expanded r expanded)
  = rename_descr_oub r (descr_oub_of_expanded expanded).
Proof.
  destruct expanded as [k [ | [head | head | k' head | head | head] | ]];
    try reflexivity; destruct k as [ | [] | | ]; reflexivity.
Qed.

Theorem is_bottom_rename : is_bottom_rename_statement.
Proof.
  intros r tenv t. unfold Provers.is_bottom. rewrite expand_head_rename.
  destruct (expand_head tenv t) as [[k [ | head | ]] | ]; reflexivity.
Qed.

Theorem provers_expand_head_rename : provers_expand_head_rename_statement.
Proof.
  intros r tenv t. unfold Provers.expand_head. rewrite expand_head_rename.
  destruct (expand_head tenv t) as [expanded | ]; simpl;
    [ rewrite descr_oub_of_expanded_renamed | ]; reflexivity.
Qed.

(* ======================================================================== *)
(* The answers and the three ways from a prover on heads                    *)
(* ======================================================================== *)

Lemma generic_proof_map_id {A : Type} (proof : Generic_proof.t A) :
  generic_proof_map (fun a => a) proof = proof.
Proof. destruct proof; reflexivity. Qed.

Lemma option_generic_proof_map_id {A : Type}
    (proof : option (Generic_proof.t A)) :
  option_map (generic_proof_map (fun a => a)) proof = proof.
Proof. destruct proof as [[a | | ] | ]; reflexivity. Qed.

Lemma option_proof_of_property_map_id {A : Type}
    (proof : option (Proof_of_property.t A)) :
  option_map (proof_of_property_map (fun a => a)) proof = proof.
Proof. destruct proof as [[a | ] | ]; reflexivity. Qed.

Lemma option_meet_shortcut_map_id {A : Type}
    (shortcut : option (Meet_shortcut.t A)) :
  option_map (meet_shortcut_map (fun a => a)) shortcut = shortcut.
Proof. destruct shortcut as [[a | | ] | ]; reflexivity. Qed.

(* A prover on heads that commutes with the renaming, its answer
   renamed by [f]. *)
Definition value_prover_renamed {A : Type} (r : renaming) (f : A -> A)
    (prove_gen :
      Typing_env.t -> head_of_kind_value_non_null
      -> option (Generic_proof.t A)) : Prop :=
  ∀ (tenv : Typing_env.t) (head : head_of_kind_value_non_null),
    prove_gen (rename_typing_env r tenv)
      (rename_head_of_kind_value_non_null r head)
    = option_map (generic_proof_map f) (prove_gen tenv head).

(* The same of a prover on heads whose answer holds no variable. *)
Lemma value_prover_unchanged {A : Type} (r : renaming)
    (prove_gen :
      Typing_env.t -> head_of_kind_value_non_null
      -> option (Generic_proof.t A)) :
  (∀ (tenv : Typing_env.t) (head : head_of_kind_value_non_null),
    prove_gen (rename_typing_env r tenv)
      (rename_head_of_kind_value_non_null r head)
    = prove_gen tenv head) ->
  value_prover_renamed r (fun a => a) prove_gen.
Proof.
  intros H tenv head. rewrite H. symmetry. apply option_generic_proof_map_id.
Qed.

Lemma gen_value_to_gen_renamed {A : Type} (r : renaming) (f : A -> A)
    (prove_gen :
      Typing_env.t -> head_of_kind_value_non_null
      -> option (Generic_proof.t A))
    (tenv : Typing_env.t) (t : flambda_type) :
  value_prover_renamed r f prove_gen ->
  Provers.gen_value_to_gen prove_gen (rename_typing_env r tenv)
    (rename_type r t)
  = option_map (generic_proof_map f)
      (Provers.gen_value_to_gen prove_gen tenv t).
Proof.
  intros Hgen. unfold Provers.gen_value_to_gen.
  rewrite provers_expand_head_rename.
  destruct (Provers.expand_head tenv t)
    as [[[ | [[ | head | ] [ | x]] | ] | head | k head | head | head] | ];
    try reflexivity.
  apply Hgen.
Qed.

Lemma gen_value_to_proof_renamed {A : Type} (r : renaming) (f : A -> A)
    (prove_gen :
      Typing_env.t -> head_of_kind_value_non_null
      -> option (Generic_proof.t A))
    (tenv : Typing_env.t) (t : flambda_type) :
  value_prover_renamed r f prove_gen ->
  Provers.gen_value_to_proof prove_gen (rename_typing_env r tenv)
    (rename_type r t)
  = option_map (proof_of_property_map f)
      (Provers.gen_value_to_proof prove_gen tenv t).
Proof.
  intros Hgen. unfold Provers.gen_value_to_proof.
  rewrite provers_expand_head_rename.
  destruct (Provers.expand_head tenv t)
    as [[[ | [[ | head | ] [ | x]] | ] | head | k head | head | head] | ];
    try reflexivity.
  simpl. rewrite Hgen.
  destruct (prove_gen tenv head) as [[a | | ] | ]; reflexivity.
Qed.

Lemma gen_value_to_meet_renamed {A : Type} (r : renaming) (f : A -> A)
    (prove_gen :
      Typing_env.t -> head_of_kind_value_non_null
      -> option (Generic_proof.t A))
    (tenv : Typing_env.t) (t : flambda_type) :
  value_prover_renamed r f prove_gen ->
  Provers.gen_value_to_meet prove_gen (rename_typing_env r tenv)
    (rename_type r t)
  = option_map (meet_shortcut_map f)
      (Provers.gen_value_to_meet prove_gen tenv t).
Proof.
  intros Hgen. unfold Provers.gen_value_to_meet.
  rewrite provers_expand_head_rename.
  destruct (Provers.expand_head tenv t)
    as [[[ | [[ | head | ] null] | ] | head | k head | head | head] | ];
    try reflexivity.
  simpl. rewrite Hgen.
  destruct (prove_gen tenv head) as [[a | | ] | ]; reflexivity.
Qed.

(* The three, for a prover on heads whose answer holds no variable. *)
Lemma gen_value_to_gen_unchanged {A : Type} (r : renaming)
    (prove_gen :
      Typing_env.t -> head_of_kind_value_non_null
      -> option (Generic_proof.t A))
    (tenv : Typing_env.t) (t : flambda_type) :
  value_prover_renamed r (fun a => a) prove_gen ->
  Provers.gen_value_to_gen prove_gen (rename_typing_env r tenv)
    (rename_type r t)
  = Provers.gen_value_to_gen prove_gen tenv t.
Proof.
  intros Hgen. rewrite (gen_value_to_gen_renamed r (fun a => a)) by exact Hgen.
  apply option_generic_proof_map_id.
Qed.

Lemma gen_value_to_proof_unchanged {A : Type} (r : renaming)
    (prove_gen :
      Typing_env.t -> head_of_kind_value_non_null
      -> option (Generic_proof.t A))
    (tenv : Typing_env.t) (t : flambda_type) :
  value_prover_renamed r (fun a => a) prove_gen ->
  Provers.gen_value_to_proof prove_gen (rename_typing_env r tenv)
    (rename_type r t)
  = Provers.gen_value_to_proof prove_gen tenv t.
Proof.
  intros Hgen.
  rewrite (gen_value_to_proof_renamed r (fun a => a)) by exact Hgen.
  apply option_proof_of_property_map_id.
Qed.

Lemma gen_value_to_meet_unchanged {A : Type} (r : renaming)
    (prove_gen :
      Typing_env.t -> head_of_kind_value_non_null
      -> option (Generic_proof.t A))
    (tenv : Typing_env.t) (t : flambda_type) :
  value_prover_renamed r (fun a => a) prove_gen ->
  Provers.gen_value_to_meet prove_gen (rename_typing_env r tenv)
    (rename_type r t)
  = Provers.gen_value_to_meet prove_gen tenv t.
Proof.
  intros Hgen.
  rewrite (gen_value_to_meet_renamed r (fun a => a)) by exact Hgen.
  apply option_meet_shortcut_map_id.
Qed.

(* ======================================================================== *)
(* The scrutinee questions and the numbers                                  *)
(* ======================================================================== *)

Ltac blocks_bottom_renamed := rewrite ?row_like_for_blocks_is_bottom_renamed.

Lemma prove_is_int_generic_value_renamed (r : renaming)
    (variant_only : bool) :
  value_prover_renamed r (fun a => a)
    (Provers.prove_is_int_generic_value variant_only).
Proof.
  apply value_prover_unchanged. intros tenv head.
  destruct head
    as [is_int [imms | ] get_tag [blocks | ] extensions is_unique
       | alloc_mode | k contents alloc_mode | closures alloc_mode | strs];
    simpl; try reflexivity; blocks_bottom_renamed;
    rewrite ?is_bottom_rename; reflexivity.
Qed.

Theorem prove_is_int_rename : prove_is_int_rename_statement.
Proof.
  intros r tenv t.
  apply gen_value_to_proof_unchanged, prove_is_int_generic_value_renamed.
Qed.

Theorem meet_is_int_variant_only_rename :
  meet_is_int_variant_only_rename_statement.
Proof.
  intros r tenv t.
  apply gen_value_to_meet_unchanged, prove_is_int_generic_value_renamed.
Qed.

Lemma prove_is_int_generic_renamed (r : renaming) (variant_only : bool)
    (tenv : Typing_env.t) (t : flambda_type) :
  Provers.prove_is_int_generic variant_only (rename_typing_env r tenv)
    (rename_type r t)
  = Provers.prove_is_int_generic variant_only tenv t.
Proof.
  apply gen_value_to_gen_unchanged, prove_is_int_generic_value_renamed.
Qed.

Lemma prove_is_not_a_pointer_generic_value_renamed (r : renaming)
    (tenv : Typing_env.t) (t : flambda_type) :
  Provers.prove_is_not_a_pointer_generic_value (rename_typing_env r tenv)
    (rename_type r t)
  = Provers.prove_is_not_a_pointer_generic_value tenv t.
Proof.
  unfold Provers.prove_is_not_a_pointer_generic_value.
  rewrite provers_expand_head_rename.
  destruct (Provers.expand_head tenv t)
    as [[[ | [[ | head | ] [ | x]] | ] | head | k head | head | head] | ];
    try reflexivity; simpl;
    rewrite (prove_is_int_generic_value_renamed r false);
    destruct (Provers.prove_is_int_generic_value false tenv head)
      as [[[ | ] | | ] | ];
    reflexivity.
Qed.

Theorem prove_is_not_a_pointer_rename :
  prove_is_not_a_pointer_rename_statement.
Proof.
  intros r tenv t. unfold Provers.prove_is_not_a_pointer.
  rewrite prove_is_not_a_pointer_generic_value_renamed. reflexivity.
Qed.

Lemma prove_get_tag_generic_value_renamed (r : renaming) :
  value_prover_renamed r (fun a => a) Provers.prove_get_tag_generic_value.
Proof.
  apply value_prover_unchanged. intros tenv head.
  destruct head
    as [is_int [imms | ] get_tag [blocks | ] extensions is_unique
       | alloc_mode | k contents alloc_mode | closures alloc_mode | strs];
    simpl; try reflexivity; rewrite ?is_bottom_rename;
    rewrite ?(proj1 (all_tags_rename r blocks)); reflexivity.
Qed.

Theorem prove_get_tag_rename : prove_get_tag_rename_statement.
Proof.
  intros r tenv t.
  apply gen_value_to_proof_unchanged, prove_get_tag_generic_value_renamed.
Qed.

Lemma prove_get_tag_generic_renamed (r : renaming) (tenv : Typing_env.t)
    (t : flambda_type) :
  Provers.prove_get_tag_generic (rename_typing_env r tenv) (rename_type r t)
  = Provers.prove_get_tag_generic tenv t.
Proof.
  apply gen_value_to_gen_unchanged, prove_get_tag_generic_value_renamed.
Qed.

Lemma prove_is_null_generic_renamed (r : renaming) (tenv : Typing_env.t)
    (t : flambda_type) :
  Provers.prove_is_null_generic (rename_typing_env r tenv) (rename_type r t)
  = Provers.prove_is_null_generic tenv t.
Proof.
  unfold Provers.prove_is_null_generic. rewrite provers_expand_head_rename.
  destruct (Provers.expand_head tenv t)
    as [[[ | [[ | head | ] [ | x]] | ] | head | k head | head | head] | ];
    reflexivity.
Qed.

Theorem meet_is_null_rename : meet_is_null_rename_statement.
Proof.
  intros r tenv t. unfold Provers.meet_is_null.
  rewrite prove_is_null_generic_renamed. reflexivity.
Qed.

(* The relation of a number with a renamed name adds what the relation
   with the name adds. *)
Lemma prove_naked_immediates_relation_renamed (r : renaming)
    (tenv : Typing_env.t) (relation : Relation.t) (n : name)
    (imms : Generic_proof.t (list Z)) :
  Provers.prove_naked_immediates_relation (rename_typing_env r tenv) relation
    (rename_name r n) imms
  = Provers.prove_naked_immediates_relation tenv relation n imms.
Proof.
  unfold Provers.prove_naked_immediates_relation.
  replace (alias_type_of Value (simple_of_name (rename_name r n)))
    with (rename_type r (alias_type_of Value (simple_of_name n)))
    by (destruct n; reflexivity).
  destruct relation.
  - rewrite prove_is_null_generic_renamed. reflexivity.
  - rewrite prove_is_int_generic_renamed. reflexivity.
  - rewrite prove_get_tag_generic_renamed. reflexivity.
Qed.

Lemma fold_relations_renamed (r : renaming) (tenv : Typing_env.t)
    (relations : list (Relation.t * name))
    (proof : option (Generic_proof.t (list Z))) :
  List.fold_left
    (fun (imms : option (Generic_proof.t (list Z)))
         (relation_and_name : Relation.t * name) =>
       let* imms := imms in
       Provers.prove_naked_immediates_relation (rename_typing_env r tenv)
         (fst relation_and_name) (snd relation_and_name) imms)
    (map (fun relation => (fst relation, rename_name r (snd relation)))
       relations)
    proof
  = List.fold_left
      (fun (imms : option (Generic_proof.t (list Z)))
           (relation_and_name : Relation.t * name) =>
         let* imms := imms in
         Provers.prove_naked_immediates_relation tenv (fst relation_and_name)
           (snd relation_and_name) imms)
      relations proof.
Proof.
  revert proof.
  induction relations as [ | [relation n] relations IH]; intros proof;
    [ reflexivity | ].
  simpl. rewrite IH. f_equal.
  destruct proof as [imms | ]; [ | reflexivity ].
  apply prove_naked_immediates_relation_renamed.
Qed.

Lemma prove_naked_immediates_generic_renamed (r : renaming)
    (tenv : Typing_env.t) (t : flambda_type) :
  Provers.prove_naked_immediates_generic (rename_typing_env r tenv)
    (rename_type r t)
  = Provers.prove_naked_immediates_generic tenv t.
Proof.
  unfold Provers.prove_naked_immediates_generic.
  rewrite provers_expand_head_rename.
  destruct (Provers.expand_head tenv t)
    as [[head | [ | [imms relations] | ] | k head | head | head] | ];
    try reflexivity.
  simpl. apply fold_relations_renamed.
Qed.

Theorem meet_naked_immediates_rename : meet_naked_immediates_rename_statement.
Proof.
  intros r tenv t. unfold Provers.meet_naked_immediates.
  rewrite prove_naked_immediates_generic_renamed. reflexivity.
Qed.

Lemma prove_equals_tagged_immediates_value_renamed (r : renaming) :
  value_prover_renamed r (fun a => a)
    Provers.prove_equals_tagged_immediates_value.
Proof.
  apply value_prover_unchanged. intros tenv head.
  destruct head
    as [is_int [imms | ] get_tag [blocks | ] extensions is_unique
       | alloc_mode | k contents alloc_mode | closures alloc_mode | strs];
    simpl; try reflexivity; blocks_bottom_renamed;
    rewrite ?prove_naked_immediates_generic_renamed; reflexivity.
Qed.

Theorem prove_equals_tagged_immediates_rename :
  prove_equals_tagged_immediates_rename_statement.
Proof.
  intros r tenv t.
  apply gen_value_to_proof_unchanged,
    prove_equals_tagged_immediates_value_renamed.
Qed.

Lemma meet_equals_tagged_immediates_value_renamed (r : renaming) :
  value_prover_renamed r (fun a => a)
    Provers.meet_equals_tagged_immediates_value.
Proof.
  apply value_prover_unchanged. intros tenv head.
  destruct head
    as [is_int [imms | ] get_tag blocks extensions is_unique
       | alloc_mode | k contents alloc_mode | closures alloc_mode | strs];
    simpl; try reflexivity.
  apply prove_naked_immediates_generic_renamed.
Qed.

Theorem meet_equals_tagged_immediates_rename :
  meet_equals_tagged_immediates_rename_statement.
Proof.
  intros r tenv t.
  apply gen_value_to_meet_unchanged,
    meet_equals_tagged_immediates_value_renamed.
Qed.

Theorem meet_equals_single_tagged_immediate_rename :
  meet_equals_single_tagged_immediate_rename_statement.
Proof.
  intros r tenv t. unfold Provers.meet_equals_single_tagged_immediate.
  rewrite meet_equals_tagged_immediates_rename. reflexivity.
Qed.

Theorem meet_naked_number_rename : meet_naked_number_rename_statement.
Proof.
  intros kind r tenv t. unfold Provers.meet_naked_number.
  rewrite provers_expand_head_rename.
  destruct (Provers.expand_head tenv t)
    as [[head | head | k head | head | head] | ]; reflexivity.
Qed.

(* ======================================================================== *)
(* Aliases, boxed numbers and strings                                       *)
(* ======================================================================== *)

Lemma canonical_simple_of_alias_renamed (r : renaming) (tenv : Typing_env.t)
    (min_name_mode : Name_mode.t) (t : flambda_type) :
  Provers.canonical_simple_of_alias (rename_typing_env r tenv) min_name_mode
    (rename_type r t)
  = option_map (option_map (rename_simple r))
      (Provers.canonical_simple_of_alias tenv min_name_mode t).
Proof.
  unfold Provers.canonical_simple_of_alias. rewrite get_alias_opt_renamed.
  destruct (get_alias_opt t) as [s | ]; simpl; [ | reflexivity ].
  apply get_canonical_simple_exn_equivariant.
Qed.

Theorem prove_equals_to_simple_of_kind_rename :
  prove_equals_to_simple_of_kind_rename_statement.
Proof.
  intros kind r tenv t. unfold Provers.prove_equals_to_simple_of_kind.
  rewrite kind_rename_type.
  destruct (negb (kind_eqb (FlambdaTypeGrammar.kind t) kind));
    [ reflexivity | ].
  rewrite canonical_simple_of_alias_renamed.
  destruct (Provers.canonical_simple_of_alias tenv Name_mode.normal t)
    as [[s | ] | ]; reflexivity.
Qed.

Lemma inspect_tagging_immediates_renamed (r : renaming)
    (min_name_mode : Name_mode.t) (tenv : Typing_env.t)
    (immediates : Or_unknown.t flambda_type) :
  Provers.inspect_tagging_immediates min_name_mode (rename_typing_env r tenv)
    (or_unknown_map (rename_type r) immediates)
  = option_map (generic_proof_map (rename_simple r))
      (Provers.inspect_tagging_immediates min_name_mode tenv immediates).
Proof.
  destruct immediates as [t | ]; simpl; [ | reflexivity ].
  rewrite canonical_simple_of_alias_renamed.
  destruct (Provers.canonical_simple_of_alias tenv min_name_mode t)
    as [[s | ] | ]; simpl; try reflexivity.
  rewrite meet_naked_immediates_rename.
  destruct (Provers.meet_naked_immediates tenv t) as [[imms | | ] | ];
    simpl; try reflexivity.
  destruct (Keyed.get_singleton Z.eqb imms) as [imm | ]; reflexivity.
Qed.

Lemma inspect_tagging_of_simple_value_renamed (r : renaming)
    (proof_kind : Provers.tagging_proof_kind) (min_name_mode : Name_mode.t) :
  value_prover_renamed r (rename_simple r)
    (Provers.inspect_tagging_of_simple_value proof_kind min_name_mode).
Proof.
  intros tenv head.
  destruct head
    as [is_int immediates get_tag [blocks | ] extensions is_unique
       | alloc_mode | k contents alloc_mode | closures alloc_mode | strs];
    destruct proof_kind; simpl; try reflexivity; blocks_bottom_renamed;
    try apply inspect_tagging_immediates_renamed.
  destruct (row_like_for_blocks_is_bottom blocks);
    [ apply inspect_tagging_immediates_renamed | reflexivity ].
Qed.

Theorem prove_tagging_of_simple_rename :
  prove_tagging_of_simple_rename_statement.
Proof.
  intros min_name_mode r tenv t.
  apply gen_value_to_proof_renamed, inspect_tagging_of_simple_value_renamed.
Qed.

Theorem meet_tagging_of_simple_rename :
  meet_tagging_of_simple_rename_statement.
Proof.
  intros min_name_mode r tenv t.
  apply gen_value_to_meet_renamed, inspect_tagging_of_simple_value_renamed.
Qed.

Lemma prove_is_a_boxed_or_tagged_number_value_renamed (r : renaming) :
  value_prover_renamed r (rename_boxed_or_tagged_number r)
    Provers.prove_is_a_boxed_or_tagged_number_value.
Proof.
  intros tenv head.
  destruct head
    as [is_int immediates get_tag [blocks | ] extensions is_unique
       | alloc_mode | k contents alloc_mode | closures alloc_mode | strs];
    simpl; try reflexivity; blocks_bottom_renamed.
  destruct (row_like_for_blocks_is_bottom blocks); reflexivity.
Qed.

Theorem prove_is_a_boxed_or_tagged_number_rename :
  prove_is_a_boxed_or_tagged_number_rename_statement.
Proof.
  intros r tenv t.
  apply gen_value_to_proof_renamed,
    prove_is_a_boxed_or_tagged_number_value_renamed.
Qed.

Theorem prove_is_a_tagged_immediate_rename :
  prove_is_a_tagged_immediate_rename_statement.
Proof.
  intros r tenv t. unfold Provers.prove_is_a_tagged_immediate.
  rewrite prove_is_a_boxed_or_tagged_number_rename.
  destruct (Provers.prove_is_a_boxed_or_tagged_number tenv t)
    as [[[alloc_mode k contents | ] | ] | ]; reflexivity.
Qed.

Theorem prove_is_a_boxed_number_rename :
  prove_is_a_boxed_number_rename_statement.
Proof.
  intros kind r tenv t. unfold Provers.prove_is_a_boxed_number.
  rewrite prove_is_a_boxed_or_tagged_number_rename.
  destruct (Provers.prove_is_a_boxed_or_tagged_number tenv t)
    as [[[alloc_mode k contents | ] | ] | ]; reflexivity.
Qed.

Lemma prove_is_or_is_not_a_boxed_float_value_renamed (r : renaming) :
  value_prover_renamed r (fun a => a)
    Provers.prove_is_or_is_not_a_boxed_float_value.
Proof.
  apply value_prover_unchanged. intros tenv head.
  destruct head
    as [is_int immediates get_tag blocks extensions is_unique
       | alloc_mode | k contents alloc_mode | closures alloc_mode | strs];
    reflexivity.
Qed.

Theorem prove_is_or_is_not_a_boxed_float_rename :
  prove_is_or_is_not_a_boxed_float_rename_statement.
Proof.
  intros r tenv t.
  apply gen_value_to_proof_unchanged,
    prove_is_or_is_not_a_boxed_float_value_renamed.
Qed.

Theorem meet_boxed_number_containing_simple_rename :
  meet_boxed_number_containing_simple_rename_statement.
Proof.
  intros kind min_name_mode r tenv t.
  unfold Provers.meet_boxed_number_containing_simple.
  rewrite provers_expand_head_rename.
  destruct (Provers.expand_head tenv t)
    as [[[ | [[ | head | ] null] | ] | head | k head | head | head] | ];
    try reflexivity.
  destruct head
    as [is_int immediates get_tag blocks extensions is_unique
       | alloc_mode | k contents alloc_mode | closures alloc_mode | strs];
    simpl; try reflexivity.
  destruct (boxable_number_eqb k kind); [ | reflexivity ].
  rewrite canonical_simple_of_alias_renamed.
  destruct (Provers.canonical_simple_of_alias tenv min_name_mode contents)
    as [[s | ] | ]; reflexivity.
Qed.

Theorem prove_nothing_rename : prove_nothing_rename_statement.
Proof. intros A r tenv t. reflexivity. Qed.

Lemma prove_strings_value_renamed (r : renaming) :
  value_prover_renamed r (fun a => a) Provers.prove_strings_value.
Proof.
  apply value_prover_unchanged. intros tenv head.
  destruct head
    as [is_int immediates get_tag blocks extensions is_unique
       | alloc_mode | k contents alloc_mode | closures alloc_mode | strs];
    reflexivity.
Qed.

Theorem prove_strings_rename : prove_strings_rename_statement.
Proof.
  intros r tenv t.
  apply gen_value_to_proof_unchanged, prove_strings_value_renamed.
Qed.

Theorem meet_strings_rename : meet_strings_rename_statement.
Proof.
  intros r tenv t.
  apply gen_value_to_meet_unchanged, prove_strings_value_renamed.
Qed.

(* ======================================================================== *)
(* Blocks                                                                   *)
(* ======================================================================== *)

Lemma prove_variant_like_generic_value_renamed (r : renaming) :
  value_prover_renamed r (fun a => a)
    Provers.prove_variant_like_generic_value.
Proof.
  apply value_prover_unchanged. intros tenv head.
  destruct head
    as [is_int [imms | ] get_tag [blocks | ] extensions is_unique
       | alloc_mode | k contents alloc_mode | closures alloc_mode | strs];
    simpl; try reflexivity;
    rewrite (proj2 (all_tags_rename r blocks));
    rewrite ?prove_naked_immediates_generic_renamed; reflexivity.
Qed.

Theorem prove_variant_like_rename : prove_variant_like_rename_statement.
Proof.
  intros r tenv t.
  apply gen_value_to_proof_unchanged,
    prove_variant_like_generic_value_renamed.
Qed.

Theorem meet_variant_like_rename : meet_variant_like_rename_statement.
Proof.
  intros r tenv t.
  apply gen_value_to_meet_unchanged, prove_variant_like_generic_value_renamed.
Qed.

(* The one block that a type has, with the types of its fields
   renamed. *)
Definition rename_singleton (r : renaming)
    (singleton : Z * block_shape * nat * list flambda_type * Alloc_mode.t)
    : Z * block_shape * nat * list flambda_type * Alloc_mode.t :=
  match singleton with
  | (tag, shape, size, tys, alloc_mode) =>
    (tag, shape, size, map (rename_type r) tys, alloc_mode)
  end.

Lemma get_singleton_renamed (r : renaming) (blocks : row_like_for_blocks) :
  row_like_for_blocks_get_singleton (rename_row_like_for_blocks r blocks)
  = option_map (rename_singleton r)
      (row_like_for_blocks_get_singleton blocks).
Proof. apply get_singleton_rename. Qed.

Lemma prove_unique_tag_and_size_value_renamed (r : renaming) :
  value_prover_renamed r (rename_singleton r)
    Provers.prove_unique_tag_and_size_value.
Proof.
  intros tenv head.
  destruct head
    as [is_int [imms | ] get_tag [blocks | ] extensions is_unique
       | alloc_mode | k contents alloc_mode | closures alloc_mode | strs];
    simpl; try reflexivity; rewrite is_bottom_rename;
    destruct (Provers.is_bottom tenv imms) as [[ | ] | ]; try reflexivity.
  rewrite get_singleton_renamed.
  destruct (row_like_for_blocks_get_singleton blocks)
    as [[[[[tag shape] size] product] alloc_mode] | ]; reflexivity.
Qed.

Theorem prove_unique_tag_and_size_rename :
  prove_unique_tag_and_size_rename_statement.
Proof.
  intros r tenv t. unfold Provers.prove_unique_tag_and_size.
  rewrite (gen_value_to_proof_renamed r (rename_singleton r))
    by apply prove_unique_tag_and_size_value_renamed.
  destruct
    (Provers.gen_value_to_proof Provers.prove_unique_tag_and_size_value tenv
       t)
    as [[[[[[tag shape] size] product] alloc_mode] | ] | ]; reflexivity.
Qed.

Lemma field_aliases_rev_renamed (r : renaming) (product : list flambda_type)
    (result : Proof_of_property.t (list simple)) :
  List.fold_left Provers.field_aliases_rev (map (rename_type r) product)
    (proof_of_property_map (map (rename_simple r)) result)
  = proof_of_property_map (map (rename_simple r))
      (List.fold_left Provers.field_aliases_rev product result).
Proof.
  revert result.
  induction product as [ | ty product IH]; intros result; [ reflexivity | ].
  simpl. rewrite <- IH. f_equal.
  destruct result as [simples | ]; simpl; [ | reflexivity ].
  rewrite get_alias_opt_renamed.
  destruct (get_alias_opt ty) as [s | ]; reflexivity.
Qed.

Theorem prove_unique_immutable_heap_block_rename :
  prove_unique_immutable_heap_block_rename_statement.
Proof.
  intros r tenv t.
  unfold Provers.prove_unique_fully_constructed_immutable_heap_block.
  rewrite (gen_value_to_proof_renamed r (rename_singleton r))
    by apply prove_unique_tag_and_size_value_renamed.
  destruct
    (Provers.gen_value_to_proof Provers.prove_unique_tag_and_size_value tenv
       t)
    as [[[[[[tag shape] size] product] [ | | ]] | ] | ]; try reflexivity.
  simpl.
  pose proof
    (field_aliases_rev_renamed r product (Proof_of_property.Proved []))
    as Hfold.
  simpl in Hfold. rewrite Hfold.
  destruct
    (List.fold_left Provers.field_aliases_rev product
       (Proof_of_property.Proved []))
    as [simples | ]; simpl; [ | reflexivity ].
  rewrite map_rev. reflexivity.
Qed.

Lemma meet_block_field_simple_value_renamed (r : renaming)
    (min_name_mode : Name_mode.t) (field_kind : flambda_kind)
    (field_index : nat) :
  value_prover_renamed r (rename_simple r)
    (Provers.meet_block_field_simple_value min_name_mode field_kind
       field_index).
Proof.
  intros tenv head.
  destruct head
    as [is_int immediates get_tag [blocks | ] extensions is_unique
       | alloc_mode | k contents alloc_mode | closures alloc_mode | strs];
    simpl; try reflexivity; blocks_bottom_renamed.
  destruct (row_like_for_blocks_is_bottom blocks); [ reflexivity | ].
  rewrite get_field_rename.
  destruct (row_like_for_blocks_get_field blocks field_index) as [ | ty | ];
    simpl; try reflexivity.
  rewrite kind_rename_type.
  destruct (negb (kind_eqb (kind ty) field_kind)); [ reflexivity | ].
  rewrite canonical_simple_of_alias_renamed.
  destruct (Provers.canonical_simple_of_alias tenv min_name_mode ty)
    as [[s | ] | ]; reflexivity.
Qed.

Theorem meet_block_field_simple_rename :
  meet_block_field_simple_rename_statement.
Proof.
  intros min_name_mode field_kind field_index r tenv t.
  apply gen_value_to_meet_renamed, meet_block_field_simple_value_renamed.
Qed.

(* ======================================================================== *)
(* Closures                                                                 *)
(* ======================================================================== *)

Lemma single_closures_entry_of_singleton_renamed (r : renaming)
    (alloc_mode : Alloc_mode.t) (exact : bool) (slot : function_slot)
    (entry : closures_entry) :
  Provers.single_closures_entry_of_singleton alloc_mode exact slot
    (rename_closures_entry r entry)
  = generic_proof_map (rename_single_closures_entry r)
      (Provers.single_closures_entry_of_singleton alloc_mode exact slot
         entry).
Proof.
  unfold Provers.single_closures_entry_of_singleton.
  rewrite find_function_type_rename.
  destruct (closures_entry_find_function_type entry exact slot);
    reflexivity.
Qed.

Lemma prove_single_closures_entry_generic_value_renamed (r : renaming) :
  value_prover_renamed r (rename_single_closures_entry r)
    Provers.prove_single_closures_entry_generic_value.
Proof.
  intros tenv head.
  destruct head
    as [is_int immediates get_tag blocks extensions is_unique
       | alloc_mode | k contents alloc_mode | closures alloc_mode | strs];
    simpl; try reflexivity.
  rewrite (proj1 (get_single_tag_rename r closures)).
  destruct (row_like_for_closures_get_single_tag closures)
    as [[ | slot entry | slot entry] | ]; simpl; try reflexivity;
    rewrite single_closures_entry_of_singleton_renamed; reflexivity.
Qed.

Theorem prove_single_closures_entry_rename :
  prove_single_closures_entry_rename_statement.
Proof.
  intros r tenv t.
  apply gen_value_to_proof_renamed,
    prove_single_closures_entry_generic_value_renamed.
Qed.

Theorem meet_single_closures_entry_rename :
  meet_single_closures_entry_rename_statement.
Proof.
  intros r tenv t.
  apply gen_value_to_meet_renamed,
    prove_single_closures_entry_generic_value_renamed.
Qed.

(* The bindings that count of a map whose values are changed. *)
Lemma tabulate_map_values {A B : Type} (f : A -> B)
    (g : positive -> option A) (names : list positive) :
  tabulate (fun name => option_map f (g name)) names
  = map_values f (tabulate g names).
Proof.
  unfold tabulate, map_values.
  induction names as [ | name names IH]; [ reflexivity | ].
  simpl. rewrite map_app, IH. f_equal.
  destruct (g name) as [a | ]; reflexivity.
Qed.

Lemma compact_map_values {A B : Type} (f : A -> B) (m : finite_map A) :
  compact (map_values f m) = map_values f (compact m).
Proof.
  unfold compact. rewrite domain_map_values, <- tabulate_map_values.
  unfold tabulate. apply flat_map_ext. intros name.
  rewrite lookup_map_values. reflexivity.
Qed.

Lemma add_code_ids_renamed (r : renaming)
    (cases : finite_map row_like_closures_case)
    (acc : Or_unknown.t (list code_id)) :
  List.fold_left Provers.add_code_id_of_case
    (map_values
       (rename_closures_case (rename_closures_entry r)
          (rename_env_extension r))
       cases)
    acc
  = List.fold_left Provers.add_code_id_of_case cases acc.
Proof.
  revert acc.
  induction cases as [ | [slot case] cases IH]; intros acc;
    [ reflexivity | ].
  simpl. rewrite IH. f_equal.
  destruct acc as [code_ids | ]; [ | reflexivity ].
  destruct case as [[function_types closure_types value_slot_types] index
                      extension].
  reflexivity.
Qed.

Lemma prove_code_ids_generic_value_renamed (r : renaming) :
  value_prover_renamed r (fun a => a) Provers.prove_code_ids_generic_value.
Proof.
  apply value_prover_unchanged. intros tenv head.
  destruct head
    as [is_int immediates get_tag blocks extensions is_unique
       | alloc_mode | k contents alloc_mode | [known_closures] alloc_mode
       | strs];
    simpl; try reflexivity.
  rewrite compact_map_values, add_code_ids_renamed. reflexivity.
Qed.

Theorem prove_code_ids_rename : prove_code_ids_rename_statement.
Proof.
  intros r tenv t.
  apply gen_value_to_proof_unchanged, prove_code_ids_generic_value_renamed.
Qed.

Theorem meet_code_ids_rename : meet_code_ids_rename_statement.
Proof.
  intros r tenv t.
  apply gen_value_to_meet_unchanged, prove_code_ids_generic_value_renamed.
Qed.

Lemma meet_project_function_slot_simple_value_renamed (r : renaming)
    (min_name_mode : Name_mode.t) (slot : function_slot) :
  value_prover_renamed r (rename_simple r)
    (Provers.meet_project_function_slot_simple_value min_name_mode slot).
Proof.
  intros tenv head.
  destruct head
    as [is_int immediates get_tag blocks extensions is_unique
       | alloc_mode | k contents alloc_mode | closures alloc_mode | strs];
    simpl; try reflexivity.
  rewrite (proj1 (get_closure_rename r closures)).
  destruct (row_like_for_closures_get_closure closures slot) as [[ty | ] | ];
    simpl; try reflexivity.
  rewrite canonical_simple_of_alias_renamed.
  destruct (Provers.canonical_simple_of_alias tenv min_name_mode ty)
    as [[s | ] | ]; reflexivity.
Qed.

Theorem meet_project_function_slot_simple_rename :
  meet_project_function_slot_simple_rename_statement.
Proof.
  intros min_name_mode slot r tenv t.
  apply gen_value_to_meet_renamed,
    meet_project_function_slot_simple_value_renamed.
Qed.

Lemma meet_project_value_slot_simple_value_renamed (r : renaming)
    (min_name_mode : Name_mode.t) (slot : value_slot)
    (value_slot_kind : flambda_kind) :
  value_prover_renamed r (rename_simple r)
    (Provers.meet_project_value_slot_simple_value min_name_mode slot
       value_slot_kind).
Proof.
  intros tenv head.
  destruct head
    as [is_int immediates get_tag blocks extensions is_unique
       | alloc_mode | k contents alloc_mode | closures alloc_mode | strs];
    simpl; try reflexivity.
  rewrite (proj2 (get_closure_rename r closures)).
  destruct (row_like_for_closures_get_env_var closures slot) as [[ty | ] | ];
    simpl; try reflexivity.
  rewrite kind_rename_type.
  destruct (negb (kind_eqb (kind ty) value_slot_kind)); [ reflexivity | ].
  rewrite canonical_simple_of_alias_renamed.
  destruct (Provers.canonical_simple_of_alias tenv min_name_mode ty)
    as [[s | ] | ]; reflexivity.
Qed.

Theorem meet_project_value_slot_simple_rename :
  meet_project_value_slot_simple_rename_statement.
Proof.
  intros min_name_mode slot value_slot_kind r tenv t.
  apply gen_value_to_meet_renamed,
    meet_project_value_slot_simple_value_renamed.
Qed.

(* ======================================================================== *)
(* Physical equality and allocation modes                                   *)
(* ======================================================================== *)

Lemma incompatible_naked_numbers_renamed (r : renaming) (tenv : Typing_env.t)
    (t1 t2 : flambda_type) :
  Provers.incompatible_naked_numbers (rename_typing_env r tenv)
    (rename_type r t1) (rename_type r t2)
  = Provers.incompatible_naked_numbers tenv t1 t2.
Proof.
  unfold Provers.incompatible_naked_numbers.
  rewrite !provers_expand_head_rename.
  destruct (Provers.expand_head tenv t1)
    as [[head1 | head1 | k1 head1 | head1 | head1] | ];
    destruct (Provers.expand_head tenv t2)
      as [[head2 | head2 | k2 head2 | head2 | head2] | ];
    reflexivity.
Qed.

Lemma physical_equality_of_immediates_renamed (r : renaming)
    (tenv : Typing_env.t) (imms1 imms2 : flambda_type) :
  Provers.physical_equality_of_immediates (rename_typing_env r tenv)
    (rename_type r imms1) (rename_type r imms2)
  = Provers.physical_equality_of_immediates tenv imms1 imms2.
Proof.
  unfold Provers.physical_equality_of_immediates.
  rewrite !prove_naked_immediates_generic_renamed. reflexivity.
Qed.

Lemma physical_equality_of_blocks_renamed (r : renaming)
    (blocks1 blocks2 : row_like_for_blocks) :
  Provers.physical_equality_of_blocks (rename_row_like_for_blocks r blocks1)
    (rename_row_like_for_blocks r blocks2)
  = Provers.physical_equality_of_blocks blocks1 blocks2.
Proof.
  unfold Provers.physical_equality_of_blocks.
  rewrite !row_like_for_blocks_is_bottom_renamed, !get_singleton_renamed.
  destruct (row_like_for_blocks_get_singleton blocks1)
    as [[[[[tag1 shape1] size1] product1] alloc_mode1] | ];
    destruct (row_like_for_blocks_get_singleton blocks2)
      as [[[[[tag2 shape2] size2] product2] alloc_mode2] | ];
    reflexivity.
Qed.

Lemma no_immediate_and_no_block_renamed (r : renaming)
    (immediates : Or_unknown.t flambda_type)
    (blocks : Or_unknown.t row_like_for_blocks) :
  Provers.no_immediate_and_no_block
    (or_unknown_map (rename_type r) immediates)
    (or_unknown_map (rename_row_like_for_blocks r) blocks)
  = Provers.no_immediate_and_no_block immediates blocks.
Proof.
  destruct immediates as [imms | ], blocks as [blocks | ]; simpl;
    try reflexivity.
  rewrite is_obviously_bottom_renamed, row_like_for_blocks_is_bottom_renamed.
  reflexivity.
Qed.

Lemma physical_equality_of_variants_renamed (r : renaming)
    (tenv : Typing_env.t) (immediates1 immediates2 : Or_unknown.t flambda_type)
    (blocks1 blocks2 : Or_unknown.t row_like_for_blocks) :
  Provers.physical_equality_of_variants (rename_typing_env r tenv)
    (or_unknown_map (rename_type r) immediates1)
    (or_unknown_map (rename_type r) immediates2)
    (or_unknown_map (rename_row_like_for_blocks r) blocks1)
    (or_unknown_map (rename_row_like_for_blocks r) blocks2)
  = Provers.physical_equality_of_variants tenv immediates1 immediates2 blocks1
      blocks2.
Proof.
  unfold Provers.physical_equality_of_variants.
  rewrite !no_immediate_and_no_block_renamed.
  destruct immediates1 as [imms1 | ], immediates2 as [imms2 | ],
    blocks1 as [blocks1 | ], blocks2 as [blocks2 | ];
    simpl; try reflexivity.
  rewrite physical_equality_of_immediates_renamed,
    physical_equality_of_blocks_renamed.
  reflexivity.
Qed.

Lemma physical_equality_of_heads_renamed (r : renaming) (tenv : Typing_env.t)
    (head1 head2 : head_of_kind_value_non_null) :
  Provers.physical_equality_of_heads (rename_typing_env r tenv)
    (rename_head_of_kind_value_non_null r head1)
    (rename_head_of_kind_value_non_null r head2)
  = Provers.physical_equality_of_heads tenv head1 head2.
Proof.
  destruct head1
    as [is_int1 immediates1 get_tag1 blocks1 extensions1 is_unique1
       | alloc_mode1 | k1 contents1 alloc_mode1 | closures1 alloc_mode1
       | strs1];
    destruct head2
      as [is_int2 immediates2 get_tag2 blocks2 extensions2 is_unique2
         | alloc_mode2 | k2 contents2 alloc_mode2 | closures2 alloc_mode2
         | strs2];
    simpl; try reflexivity.
  - apply physical_equality_of_variants_renamed.
  - destruct blocks1 as [blocks1 | ]; simpl; [ | reflexivity ].
    rewrite row_like_for_blocks_is_bottom_renamed. reflexivity.
  - destruct blocks2 as [blocks2 | ]; simpl; [ | reflexivity ].
    rewrite row_like_for_blocks_is_bottom_renamed. reflexivity.
  - rewrite incompatible_naked_numbers_renamed. reflexivity.
Qed.

Lemma physical_equality_check_heads_renamed (r : renaming)
    (tenv : Typing_env.t) (t1 t2 : flambda_type) :
  Provers.physical_equality_check_heads (rename_typing_env r tenv)
    (rename_type r t1) (rename_type r t2)
  = Provers.physical_equality_check_heads tenv t1 t2.
Proof.
  unfold Provers.physical_equality_check_heads.
  rewrite !provers_expand_head_rename.
  destruct (Provers.expand_head tenv t1)
    as [[[ | [[ | head1 | ] [ | x1]] | ] | head1 | k1 head1 | head1 | head1]
       | ];
    destruct (Provers.expand_head tenv t2)
      as [[[ | [[ | head2 | ] [ | x2]] | ] | head2 | k2 head2 | head2
          | head2]
         | ];
    try reflexivity.
  apply physical_equality_of_heads_renamed.
Qed.

Theorem prove_physical_equality_rename :
  prove_physical_equality_rename_statement.
Proof.
  intros r tenv t1 t2. unfold Provers.prove_physical_equality.
  rewrite !get_alias_opt_renamed.
  destruct (get_alias_opt t1) as [[x1 | sym1 | c1] | ];
    destruct (get_alias_opt t2) as [[x2 | sym2 | c2] | ];
    simpl; try reflexivity;
    try apply physical_equality_check_heads_renamed.
  rewrite rename_variable_eqb.
  destruct (Pos.eqb x1 x2);
    [ reflexivity | apply physical_equality_check_heads_renamed ].
Qed.

Lemma prove_alloc_mode_of_boxed_number_value_renamed (r : renaming) :
  value_prover_renamed r (fun a => a)
    Provers.prove_alloc_mode_of_boxed_number_value.
Proof.
  apply value_prover_unchanged. intros tenv head.
  destruct head
    as [is_int immediates get_tag blocks extensions is_unique
       | alloc_mode | k contents alloc_mode | closures alloc_mode | strs];
    reflexivity.
Qed.

Theorem prove_alloc_mode_of_boxed_number_rename :
  prove_alloc_mode_of_boxed_number_rename_statement.
Proof.
  intros r tenv t.
  apply gen_value_to_proof_unchanged,
    prove_alloc_mode_of_boxed_number_value_renamed.
Qed.

Theorem never_holds_locally_allocated_values_rename :
  never_holds_locally_allocated_values_rename_statement.
Proof.
  intros r tenv var. unfold Provers.never_holds_locally_allocated_values.
  change (Name_var (rename_variable r var))
    with (rename_name r (Name_var var)).
  rewrite find_equivariant.
  destruct (Typing_env.find tenv (Name_var var) None) as [ty | ]; simpl;
    [ | reflexivity ].
  rewrite provers_expand_head_rename.
  destruct (Provers.expand_head tenv ty)
    as [[[ | [[ | head | ] null] | ] | head | k head | head | head] | ];
    try reflexivity.
  destruct head
    as [is_int immediates get_tag [blocks | ] extensions is_unique
       | alloc_mode | k contents alloc_mode | closures alloc_mode | strs];
    simpl; try reflexivity.
  blocks_bottom_renamed. destruct blocks as [known_tags other_tags alloc_mode].
  reflexivity.
Qed.

(* ======================================================================== *)
(* Reification                                                              *)
(* ======================================================================== *)

Lemma combine_map_fst {A B C : Type} (f : A -> C) (l : list A)
    (l' : list B) :
  combine (map f l) l' = map (fun p => (f (fst p), snd p)) (combine l l').
Proof.
  revert l'.
  induction l as [ | a l IH]; intros [ | b l']; try reflexivity.
  simpl. rewrite IH. reflexivity.
Qed.

Section Reify_renamed.

Variable r : renaming.

(* The test of the variables that may be answered, on the two sides. *)
Variables var_allowed renamed_var_allowed
  : Alloc_mode.t -> variable -> option bool.

Hypothesis Hallowed :
  ∀ (alloc_mode : Alloc_mode.t) (var : variable),
    renamed_var_allowed alloc_mode (rename_variable r var)
    = var_allowed alloc_mode var.

Definition rename_field (field : flambda_type * flambda_kind)
    : flambda_type * flambda_kind :=
  (rename_type r (fst field), snd field).

Lemma reify_field_renamed (tenv : Typing_env.t) (alloc_mode : Alloc_mode.t)
    (field : flambda_type * flambda_kind) :
  Reify.reify_field (rename_typing_env r tenv) renamed_var_allowed alloc_mode
    (rename_field field)
  = option_map (option_map (rename_simple r))
      (Reify.reify_field tenv var_allowed alloc_mode field).
Proof.
  unfold Reify.reify_field, rename_field. simpl.
  rewrite prove_equals_to_simple_of_kind_rename.
  destruct
    (Provers.prove_equals_to_simple_of_kind tenv (fst field) (snd field))
    as [[[x | sym | c] | ] | ]; simpl; try reflexivity.
  rewrite Hallowed.
  destruct (var_allowed alloc_mode x) as [[ | ] | ]; reflexivity.
Qed.

Lemma reify_fields_renamed (tenv : Typing_env.t) (alloc_mode : Alloc_mode.t)
    (fields : list (flambda_type * flambda_kind)) :
  Reify.reify_fields (rename_typing_env r tenv) renamed_var_allowed alloc_mode
    (map rename_field fields)
  = option_map (map (rename_simple r))
      (Reify.reify_fields tenv var_allowed alloc_mode fields).
Proof.
  induction fields as [ | field fields IH]; [ reflexivity | ].
  simpl. rewrite reify_field_renamed, IH.
  destruct (Reify.reify_field tenv var_allowed alloc_mode field)
    as [[s | ] | ];
    destruct (Reify.reify_fields tenv var_allowed alloc_mode fields)
      as [simples | ];
    reflexivity.
Qed.

Lemma try_to_reify_fields_renamed (tenv : Typing_env.t)
    (alloc_mode : Alloc_mode.t)
    (fields : list (flambda_type * flambda_kind)) :
  Reify.try_to_reify_fields (rename_typing_env r tenv) renamed_var_allowed
    alloc_mode (map rename_field fields)
  = option_map (option_map (map (rename_simple r)))
      (Reify.try_to_reify_fields tenv var_allowed alloc_mode fields).
Proof.
  unfold Reify.try_to_reify_fields. rewrite reify_fields_renamed, length_map.
  destruct (Reify.reify_fields tenv var_allowed alloc_mode fields)
    as [simples | ]; simpl; [ | reflexivity ].
  rewrite length_map.
  destruct (Nat.eqb (length fields) (length simples)); reflexivity.
Qed.

Lemma reify_expanded_renamed (tenv : Typing_env.t)
    (try_canonical_simple : Reification_result.t) (t : flambda_type) :
  Reify.reify_expanded renamed_var_allowed (rename_typing_env r tenv)
    (rename_reification_result r try_canonical_simple) (rename_type r t)
  = option_map (rename_reification_result r)
      (Reify.reify_expanded var_allowed tenv try_canonical_simple t).
Proof.
  unfold Reify.reify_expanded. rewrite provers_expand_head_rename.
  destruct (Provers.expand_head tenv t)
    as [[[ | [[ | head | ] [ | x]] | ] | [ | head | ] | k [ | ns | ]
        | [ | head | ] | [ | head | ]]
       | ];
    try reflexivity.
  - destruct head
      as [is_int [imms | ] get_tag [blocks | ] extensions is_unique
         | alloc_mode | k contents alloc_mode | closures alloc_mode | strs];
      simpl; try reflexivity.
    + rewrite is_bottom_rename.
      destruct (Provers.is_bottom tenv imms) as [[ | ] | ];
        [ | | reflexivity ].
      * rewrite get_singleton_renamed.
        destruct (row_like_for_blocks_get_singleton blocks)
          as [[[[[tag shape] size] field_types] alloc_mode] | ]; simpl;
          [ | reflexivity ].
        rewrite !length_map.
        destruct (negb (Nat.eqb size (length field_types)));
          [ reflexivity | ].
        destruct
          (negb
             (Nat.eqb (length (field_kinds shape (length field_types)))
                (length field_types)));
          [ reflexivity | ].
        destruct shape as [shape | ]; [ | reflexivity ].
        destruct (negb (Provers.tag_is_scannable tag)); [ reflexivity | ].
        rewrite combine_map_fst.
        change (fun p : flambda_type * flambda_kind =>
                  (rename_type r (fst p), snd p))
          with rename_field.
        rewrite try_to_reify_fields_renamed.
        destruct
          (Reify.try_to_reify_fields tenv var_allowed alloc_mode
             (combine field_types
                (field_kinds (Scannable shape) (length field_types))))
          as [[fields | ] | ];
          reflexivity.
      * blocks_bottom_renamed.
        destruct (row_like_for_blocks_is_bottom blocks); [ | reflexivity ].
        rewrite meet_naked_immediates_rename.
        destruct (Provers.meet_naked_immediates tenv imms)
          as [[numbers | | ] | ]; try reflexivity.
        destruct (Keyed.get_singleton Z.eqb numbers); reflexivity.
    + rewrite meet_naked_number_rename.
      destruct
        (Provers.meet_naked_number (Reify.unboxed_number_kind k) tenv
           contents)
        as [[numbers | | ] | ]; try reflexivity.
      destruct (Keyed.get_singleton Z.eqb numbers); reflexivity.
  - destruct head
      as [is_int immediates get_tag blocks extensions is_unique
         | alloc_mode | k contents alloc_mode | closures alloc_mode | strs];
      reflexivity.
  - simpl.
    change
      (Type_naked_immediate
         (Ok (No_alias (rename_head_of_kind_naked_immediate r head))))
      with (rename_type r (Type_naked_immediate (Ok (No_alias head)))).
    rewrite meet_naked_immediates_rename.
    destruct
      (Provers.meet_naked_immediates tenv
         (Type_naked_immediate (Ok (No_alias head))))
      as [[numbers | | ] | ]; try reflexivity.
    destruct (Keyed.get_singleton Z.eqb numbers); reflexivity.
  - simpl. destruct (Keyed.get_singleton Z.eqb ns); reflexivity.
Qed.

End Reify_renamed.

Lemma var_allowed_renamed (r : renaming)
    (allowed_if_free_vars_defined_in : Typing_env.t)
    (var_is_defined_at_toplevel var_is_symbol_projection
     renamed_var_is_defined_at_toplevel renamed_var_is_symbol_projection
     : variable -> bool)
    (tenv : Typing_env.t) (min_name_mode : Name_mode.t)
    (alloc_mode : Alloc_mode.t) (var : variable) :
  (∀ var : variable,
    renamed_var_is_defined_at_toplevel (rename_variable r var)
    = var_is_defined_at_toplevel var) ->
  (∀ var : variable,
    renamed_var_is_symbol_projection (rename_variable r var)
    = var_is_symbol_projection var) ->
  Reify.var_allowed (rename_typing_env r allowed_if_free_vars_defined_in)
    renamed_var_is_defined_at_toplevel renamed_var_is_symbol_projection
    (rename_typing_env r tenv) min_name_mode alloc_mode
    (rename_variable r var)
  = Reify.var_allowed allowed_if_free_vars_defined_in
      var_is_defined_at_toplevel var_is_symbol_projection tenv min_name_mode
      alloc_mode var.
Proof.
  intros Htoplevel Hprojection. unfold Reify.var_allowed.
  change (Name_var (rename_variable r var))
    with (rename_name r (Name_var var)).
  rewrite mem_equivariant, Htoplevel, Hprojection,
    never_holds_locally_allocated_values_rename.
  reflexivity.
Qed.

Theorem reify_rename : reify_rename_statement.
Proof.
  intros r allowed_if_free_vars_defined_in var_is_defined_at_toplevel
    var_is_symbol_projection renamed_var_is_defined_at_toplevel
    renamed_var_is_symbol_projection tenv t Htoplevel Hprojection.
  unfold Reify.reify.
  rewrite get_alias_then_canonical_simple_exn_equivariant.
  pose proof
    (fun alloc_mode var =>
       var_allowed_renamed r allowed_if_free_vars_defined_in
         var_is_defined_at_toplevel var_is_symbol_projection
         renamed_var_is_defined_at_toplevel renamed_var_is_symbol_projection
         tenv Name_mode.normal alloc_mode var Htoplevel Hprojection)
    as Hallowed.
  destruct
    (Typing_env.get_alias_then_canonical_simple_exn tenv
       (Some Name_mode.normal) None t)
    as [[[x | sym | c] | ] | ]; simpl; try reflexivity.
  - exact
      (reify_expanded_renamed r _ _ Hallowed tenv
         (Reification_result.Simple (Var x)) t).
  - exact
      (reify_expanded_renamed r _ _ Hallowed tenv
         Reification_result.Cannot_reify t).
Qed.
