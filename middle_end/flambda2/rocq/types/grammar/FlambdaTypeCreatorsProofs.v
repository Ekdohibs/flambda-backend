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


(* Proofs of the statements of FlambdaTypeCreatorsSpec.v. Nothing here
   needs review. *)

From Stdlib Require Import ZArith List Bool Lia Setoid.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec FlambdaSemanticsSpec
  FlambdaNominalSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeMeaningSpec FlambdaTypeMeaningProofs
  FlambdaTypeWellFormedSpec FlambdaTypeWellFormedProofs
  FlambdaTypeAliasKindsSpec FlambdaTypeAliasKindsProofs FlambdaTypeRenameSpec
  FlambdaTypeRenameProofs FlambdaTypeCreatorsSpec.
Import ListNotations.

(* ======================================================================== *)
(* Unknown types, kinds and aliases                                         *)
(* ======================================================================== *)

Theorem unknown_like_satisfied : unknown_like_satisfied_statement.
Proof.
  intros statics vars v t. split.
  - apply unknown_satisfied.
  - apply bottom_satisfied.
Qed.

Theorem unknown_from_shape_satisfied : unknown_from_shape_satisfied_statement.
Proof.
  intros statics vars shape tag fields index field t Hshape Hfield.
  unfold unknown_from_shape.
  destruct (block_shape_element_kind shape index) as [k | ] eqn:Hkind;
    [ | discriminate ].
  intros [= <-]. apply unknown_satisfied.
  exact (block_shape_element_kind_field shape tag fields index field k Hshape
           Hfield Hkind).
Qed.

Theorem unknown_with_subkind_satisfied
  : unknown_with_subkind_satisfied_statement.
Proof.
  intros statics vars v k.
  destruct k as [ | k | | ]; try apply unknown_satisfied.
  unfold unknown_with_subkind. rewrite type_satisfied_value. simpl.
  rewrite head_of_kind_value_satisfied_eq. simpl. split.
  - intros [[-> _] | [_ [Hkind _]]]; [ reflexivity | exact Hkind ].
  - intros Hkind.
    assert (Hnull : v = Value_const Null ∨ v <> Value_const Null).
    { destruct v; try (right; discriminate).
      match goal with
      | c : const |- _ =>
        destruct c; first [ left; reflexivity | right; discriminate ]
      end. }
    destruct Hnull as [-> | Hnot].
    + left. split; [ reflexivity | exact I ].
    + right. split; [ exact Hnot | ]. split; [ exact Hkind | exact I ].
Qed.

Theorem unknown_types_from_arity_satisfied
  : unknown_types_from_arity_satisfied_statement.
Proof.
  intros statics vars vs a. unfold unknown_types_from_arity.
  generalize (unarize a) as kinds. intros kinds. revert vs.
  induction kinds as [ | k kinds IH]; intros [ | v vs];
    cbn [map kind_of_value].
  - split; [ reflexivity | constructor ].
  - split; [ intros Hall; inversion Hall | discriminate ].
  - split; [ intros Hall; inversion Hall | discriminate ].
  - split.
    + intros Hall. inversion Hall as [ | ? ? ? ? Hv Hvs]; subst.
      apply unknown_with_subkind_satisfied in Hv. apply IH in Hvs.
      congruence.
    + intros [= Hv Hvs]. constructor.
      * apply unknown_with_subkind_satisfied. exact Hv.
      * apply IH. exact Hvs.
Qed.

Lemma unknown_with_subkind_well_formed (k : flambda_kind) :
  type_well_formed (unknown_with_subkind k) = true.
Proof. destruct k as [ | [] | | ]; reflexivity. Qed.

Lemma rename_type_unknown_with_subkind (r : renaming) (k : flambda_kind) :
  rename_type r (unknown_with_subkind k) = unknown_with_subkind k.
Proof. destruct k as [ | [] | | ]; reflexivity. Qed.

Lemma kind_alias_type_of (k : flambda_kind) (s : simple) :
  kind (alias_type_of k s) = k.
Proof. destruct k as [ | [] | | ]; reflexivity. Qed.

Lemma kind_unknown (k : flambda_kind) : kind (unknown k) = k.
Proof. destruct k as [ | [] | | ]; reflexivity. Qed.

Theorem kind_for_const_of_const : kind_for_const_of_const_statement.
Proof.
  intros c. unfold kind_for_const.
  destruct c; try apply kind_alias_type_of; [ reflexivity | ].
  apply kind_unknown.
Qed.

Theorem is_alias_of_name_iff : is_alias_of_name_iff_statement.
Proof.
  intros t n. unfold is_alias_of_name.
  destruct (get_alias_opt t) as [[x | s | c] | ];
    try (split; discriminate).
  - rewrite (name_eqb_eq _ _). split.
    + intros <-. reflexivity.
    + destruct n as [x' | s']; simpl; intros [= <-]; reflexivity.
  - rewrite (name_eqb_eq _ _). split.
    + intros <-. reflexivity.
    + destruct n as [x' | s']; simpl; intros [= <-]; reflexivity.
  - split; [ discriminate | ]. destruct n; discriminate.
Qed.

(* ======================================================================== *)
(* Numbers                                                                  *)
(* ======================================================================== *)

Lemma kind_of_const_of_naked_number (k : naked_number_kind) (i : Z) :
  kind_of_const (const_of_naked_number k i) = Naked_number k.
Proof. destruct k; reflexivity. Qed.

Lemma number_of_const_of_naked_number (k : naked_number_kind) (i : Z) :
  number_of_const (const_of_naked_number k i) = Some i.
Proof. destruct k; reflexivity. Qed.

(* The numbers of a kind that are in a set, whatever way the set is
   written. *)
Lemma naked_numbers_of_list (statics : static_env) (vars : env) (v : value)
    (k : naked_number_kind) (numbers : list Z) :
  type_satisfied statics vars v
    match numbers with
    | [] => bottom (Naked_number k)
    | [i] => this_const (const_of_naked_number k i)
    | _ => Type_naked_number k (Ok (No_alias numbers))
    end
  <-> ∃ (c : const) (n : Z),
        v = Value_const c
        ∧ kind_of_const c = Naked_number k
        ∧ number_of_const c = Some n
        ∧ n ∈ numbers.
Proof.
  destruct numbers as [ | i [ | i' numbers]].
  - split; [ intros Hbottom | intros [c [n [_ [_ [_ []]]]]] ].
    destruct (bottom_satisfied _ _ _ _ Hbottom).
  - rewrite (this_const_satisfied statics vars v). split.
    + intros ->. exists (const_of_naked_number k i), i.
      split; [ reflexivity | ].
      split; [ apply kind_of_const_of_naked_number | ].
      split; [ apply number_of_const_of_naked_number | left; reflexivity ].
    + intros [c [n [-> [Hkind [Hnumber [<- | []]]]]]]. f_equal.
      exact (const_of_its_number k c i Hkind Hnumber).
  - apply type_naked_number_satisfied.
Qed.

Theorem these_naked_numbers_satisfied
  : these_naked_numbers_satisfied_statement.
Proof.
  intros statics vars v k numbers.
  destruct k; try apply naked_numbers_of_list.
  unfold these_naked_numbers.
  rewrite (these_naked_immediates_satisfied statics vars v). split.
  - intros [i [-> Hin]]. exists (Naked_immediate i), i. repeat split.
    exact Hin.
  - intros [c [n [-> [Hkind [Hnumber Hin]]]]]. exists n.
    split; [ | exact Hin ]. f_equal.
    exact (const_of_its_number Kind_naked_immediate c n Hkind Hnumber).
Qed.

Theorem any_bool_satisfied : any_bool_satisfied_statement.
Proof.
  intros statics vars v. unfold any_tagged_bool, any_naked_bool, all_bools.
  rewrite (these_tagged_immediates_satisfied statics vars v),
    (these_naked_immediates_satisfied statics vars v).
  split; split.
  - intros [i [-> [<- | [<- | []]]]]; [ left | right ]; reflexivity.
  - intros [-> | ->]; eexists; (split; [ reflexivity | simpl; auto ]).
  - intros [i [-> [<- | [<- | []]]]]; [ left | right ]; reflexivity.
  - intros [-> | ->]; eexists; (split; [ reflexivity | simpl; auto ]).
Qed.

Theorem any_tagged_immediate_or_null_satisfied
  : any_tagged_immediate_or_null_satisfied_statement.
Proof.
  intros statics vars v. unfold any_tagged_immediate_or_null.
  rewrite type_satisfied_value. simpl.
  rewrite head_of_kind_value_satisfied_eq, variant_satisfied. simpl. split.
  - intros [[-> _] | [_ [Hvariant _]]]; [ right; reflexivity | ].
    destruct Hvariant as [[i [-> _]] | [tag [fields [_ [_ [_ [Hrow _]]]]]]].
    + left. exists i. reflexivity.
    + destruct (row_like_for_blocks_bottom_satisfied _ _ _ _ Hrow).
  - intros [[i ->] | ->]; [ right | left; split; [ reflexivity | exact I ] ].
    split; [ discriminate | ]. split; [ | exact I ]. left. exists i.
    repeat split.
Qed.

Theorem tagged_immediate_alias_to_satisfied
  : tagged_immediate_alias_to_satisfied_statement.
Proof.
  intros statics vars v x. unfold tagged_immediate_alias_to.
  rewrite (tag_immediate_satisfied statics vars v). split.
  - intros [i [-> Halias]]. exists i. split; [ reflexivity | ].
    apply alias_satisfied in Halias as [Heval _]. exact Heval.
  - intros [i [-> Hx]]. exists i. split; [ reflexivity | ].
    apply alias_satisfied. split; [ exact Hx | reflexivity ].
Qed.

(* ======================================================================== *)
(* Boxed numbers                                                            *)
(* ======================================================================== *)

Theorem unboxed_kind_naked_number : unboxed_kind_naked_number_statement.
Proof. intros k. destruct k; reflexivity. Qed.

Theorem boxed_number_alias_to_satisfied
  : boxed_number_alias_to_satisfied_statement.
Proof.
  intros statics vars v k x alloc_mode. unfold boxed_number_alias_to.
  rewrite (box_number_satisfied statics vars v). split.
  - intros [c [Hboxed [Hkind Halias]]]. exists c.
    split; [ exact Hboxed | ]. split; [ exact Hkind | ].
    apply alias_satisfied in Halias as [Heval _]. exact Heval.
  - intros [c [Hboxed [Hkind Hx]]]. exists c. split; [ exact Hboxed | ].
    split; [ exact Hkind | ]. apply alias_satisfied.
    split; [ exact Hx | exact Hkind ].
Qed.

Theorem this_boxed_number_satisfied : this_boxed_number_satisfied_statement.
Proof.
  intros statics vars v k i alloc_mode. unfold this_boxed_number.
  rewrite (box_number_satisfied statics vars v). split.
  - intros [c [Hboxed [_ Hconst]]].
    apply this_const_satisfied in Hconst as [= ->]. exact Hboxed.
  - intros Hboxed. eexists. split; [ exact Hboxed | ].
    split;
      [ rewrite kind_of_const_of_naked_number, unboxed_kind_naked_number;
        reflexivity
      | ].
    apply this_const_satisfied. reflexivity.
Qed.

Theorem these_boxed_numbers_satisfied
  : these_boxed_numbers_satisfied_statement.
Proof.
  intros statics vars v k numbers alloc_mode. unfold these_boxed_numbers.
  rewrite (box_number_satisfied statics vars v). split.
  - intros [c [Hboxed [Hkind Hnumbers]]].
    apply these_naked_numbers_satisfied in Hnumbers
      as [c' [n [[= <-] [_ [Hnumber Hin]]]]].
    exists c, n. repeat split; assumption.
  - intros [c [n [Hboxed [Hkind [Hnumber Hin]]]]]. exists c.
    split; [ exact Hboxed | ]. split; [ exact Hkind | ].
    apply these_naked_numbers_satisfied. exists c, n.
    split; [ reflexivity | ].
    split; [ rewrite Hkind; apply unboxed_kind_naked_number | ].
    split; assumption.
Qed.

Theorem any_boxed_number_satisfied : any_boxed_number_satisfied_statement.
Proof.
  intros statics vars v k. unfold any_boxed_number.
  rewrite (box_number_satisfied statics vars v). split.
  - intros [c [Hboxed [Hkind _]]]. exists c. split; assumption.
  - intros [c [Hboxed Hkind]]. exists c. split; [ exact Hboxed | ].
    split; [ exact Hkind | ]. apply unknown_satisfied. exact Hkind.
Qed.

(* ======================================================================== *)
(* Blocks                                                                   *)
(* ======================================================================== *)

Theorem row_like_for_blocks_create_defined
  : row_like_for_blocks_create_defined_statement.
Proof.
  intros shape field_tys open_or_closed alloc_mode.
  unfold row_like_for_blocks_create.
  destruct open_or_closed as [[tag | ] | tag], shape as [scannable | ];
    try destruct (Z.ltb_spec tag no_scan_tag) as [Hlt | Hge];
    try destruct (Z.eqb_spec tag double_array_tag) as [Heq | Hneq];
    (split;
     [ try discriminate; intros _; exists tag; split; auto
     | try (intros _; reflexivity);
       intros [tag' [[[= <-] | [= <-]] Htag]]; first [ lia | contradiction ]
     ]).
Qed.

(* A case with no extension admits a block when its index does and the
   fields have the types. *)
Lemma plain_case_satisfied (statics : static_env) (vars : env) (tag : Z)
    (fields : list value) (maps_to : list flambda_type)
    (index : row_like_index nat block_shape) :
  row_like_block_case_satisfied statics vars tag fields
    (Row_like_case maps_to index empty_env_extension)
  <-> block_index_satisfied tag fields index
      ∧ fields_satisfied (type_satisfied statics vars) maps_to fields.
Proof.
  split.
  - intros [Hindex [Hfields _]]. split; assumption.
  - intros [Hindex Hfields]. split; [ exact Hindex | ].
    split; [ exact Hfields | apply empty_env_extension_satisfied ].
Qed.

Lemma create_exactly_satisfied (statics : static_env) (vars : env) (tag : Z)
    (fields : list value) (tag' : Z) (size : nat) (shape : block_shape)
    (maps_to : list flambda_type) (alloc_mode : Alloc_mode.t) :
  row_like_for_blocks_satisfied statics vars tag fields
    (row_like_for_blocks_create_exactly tag' size shape maps_to alloc_mode)
  <-> tag = tag'
      ∧ length fields = size
      ∧ block_has_shape shape tag fields
      ∧ fields_satisfied (type_satisfied statics vars) maps_to fields.
Proof.
  unfold row_like_for_blocks_create_exactly.
  rewrite row_like_for_blocks_find_iff. simpl Keyed.lookup.
  destruct (Z.eqb_spec tag tag') as [-> | Hneq]; cbv iota.
  - rewrite plain_case_satisfied. unfold block_index_satisfied. simpl.
    tauto.
  - split; [ intros [] | intros [Heq _]; contradiction ].
Qed.

Lemma create_at_least_satisfied (statics : static_env) (vars : env) (tag : Z)
    (fields : list value) (tag' : Z) (size : nat) (shape : block_shape)
    (maps_to : list flambda_type) (alloc_mode : Alloc_mode.t) :
  row_like_for_blocks_satisfied statics vars tag fields
    (row_like_for_blocks_create_at_least tag' size shape maps_to alloc_mode)
  <-> tag = tag'
      ∧ size <= length fields
      ∧ block_has_shape shape tag fields
      ∧ fields_satisfied (type_satisfied statics vars) maps_to fields.
Proof.
  unfold row_like_for_blocks_create_at_least.
  rewrite row_like_for_blocks_find_iff. simpl Keyed.lookup.
  destruct (Z.eqb_spec tag tag') as [-> | Hneq]; cbv iota.
  - rewrite plain_case_satisfied. unfold block_index_satisfied. simpl.
    tauto.
  - split; [ intros [] | intros [Heq _]; contradiction ].
Qed.

Lemma create_at_least_unknown_tag_satisfied (statics : static_env)
    (vars : env) (tag : Z) (fields : list value) (size : nat)
    (shape : block_shape) (maps_to : list flambda_type)
    (alloc_mode : Alloc_mode.t) :
  row_like_for_blocks_satisfied statics vars tag fields
    (row_like_for_blocks_create_at_least_unknown_tag size shape maps_to
       alloc_mode)
  <-> size <= length fields
      ∧ block_has_shape shape tag fields
      ∧ fields_satisfied (type_satisfied statics vars) maps_to fields.
Proof.
  unfold row_like_for_blocks_create_at_least_unknown_tag.
  rewrite row_like_for_blocks_find_iff. simpl Keyed.lookup. cbv iota.
  rewrite plain_case_satisfied. unfold block_index_satisfied. simpl. tauto.
Qed.

Theorem row_like_for_blocks_create_satisfied
  : row_like_for_blocks_create_satisfied_statement.
Proof.
  intros statics vars tag fields shape field_tys open_or_closed alloc_mode
    blocks.
  unfold row_like_for_blocks_create.
  destruct open_or_closed as [[tag' | ] | tag'], shape as [scannable | ];
    try destruct (Z.ltb_spec tag' no_scan_tag) as [Hlt | Hge];
    try destruct (Z.eqb_spec tag' double_array_tag) as [Heq | Hneq];
    try discriminate; intros [= <-];
    rewrite ?create_exactly_satisfied, ?create_at_least_satisfied,
      ?create_at_least_unknown_tag_satisfied;
    try tauto.
  (* A record of floats of which no tag is given. *)
  unfold block_has_shape. tauto.
Qed.

Theorem immutable_block_create : immutable_block_create_statement.
Proof.
  intros is_unique tag shape alloc_mode fields blocks.
  unfold row_like_for_blocks_create, immutable_block.
  destruct shape as [scannable | ];
    [ destruct (Z.ltb_spec tag no_scan_tag)
    | destruct (Z.eqb_spec tag double_array_tag) ];
    try discriminate; intros [= <-]; reflexivity.
Qed.

(* The unknown types of the first fields of a shape: as many as asked
   for, each what [unknown_from_shape] gives its index. *)
Lemma unknown_types_from_shape_nth (shape : block_shape) (n : nat)
    (tys : list flambda_type) :
  unknown_types_from_shape shape n = Some tys ->
  length tys = n
  ∧ ∀ (i : nat) (ty : flambda_type),
      nth_error tys i = Some ty -> unknown_from_shape shape i = Some ty.
Proof.
  revert tys. induction n as [ | n IH]; intros tys; simpl.
  - intros [= <-]. split; [ reflexivity | ]. intros [ | i] ty; discriminate.
  - destruct (unknown_types_from_shape shape n) as [before | ];
      [ | discriminate ].
    destruct (unknown_from_shape shape n) as [last | ] eqn:Hlast;
      [ | discriminate ].
    intros [= <-]. destruct (IH before eq_refl) as [Hlength Hnth].
    split; [ rewrite length_app; simpl; lia | ].
    intros i ty Hty.
    destruct (Nat.lt_ge_cases i (length before)) as [Hlt | Hge].
    + rewrite nth_error_app1 in Hty by exact Hlt. exact (Hnth i ty Hty).
    + rewrite nth_error_app2 in Hty by exact Hge.
      destruct (i - length before) as [ | j] eqn:Hi; simpl in Hty.
      * injection Hty as <-. replace i with n by lia. exact Hlast.
      * destruct j; discriminate Hty.
Qed.

(* What the types of [immutable_block_with_size_at_least] say of the
   fields of a block of the shape that has as many. *)
Lemma size_at_least_field_tys_satisfied (statics : static_env) (vars : env)
    (shape : block_shape) (x : variable) (n : nat)
    (field_tys : list flambda_type) (tag : Z) (fields : list value) :
  size_at_least_field_tys shape x n = Some field_tys ->
  length field_tys = n
  ∧ (block_has_shape shape tag fields ->
     n <= length fields ->
     (fields_satisfied (type_satisfied statics vars) field_tys fields
      <-> match n with
          | O => True
          | S index =>
            ∃ field : value,
              nth_error fields index = Some field
              ∧ lookup x vars = Some field
          end)).
Proof.
  unfold size_at_least_field_tys. destruct n as [ | index].
  - intros [= <-]. split; [ reflexivity | ]. intros _ _. simpl. tauto.
  - destruct (unknown_types_from_shape shape index) as [before | ] eqn:Hbefore;
      [ | discriminate ].
    destruct (block_shape_element_kind shape index) as [k | ] eqn:Hkind;
      [ | discriminate ].
    intros [= <-].
    destruct (unknown_types_from_shape_nth shape index before Hbefore)
      as [Hlength Hnth].
    split; [ rewrite length_app; simpl; lia | ].
    intros Hshape Hsize. rewrite fields_satisfied_nth_iff.
    destruct (nth_error fields index) as [field | ] eqn:Hfield;
      [ | apply nth_error_None in Hfield; lia ].
    split.
    + intros Hall. exists field. split; [ reflexivity | ].
      assert (Halias :
                type_satisfied statics vars field
                  (alias_type_of k (Var x))).
      { apply (Hall index); [ | exact Hfield ].
        rewrite nth_error_app2 by lia.
        replace (index - length before) with 0 by lia. reflexivity. }
      apply alias_satisfied in Halias as [Heval _]. exact Heval.
    + intros [field0 [[= <-] Hx]] i ty field' Hty Hfield'.
      destruct (Nat.lt_ge_cases i (length before)) as [Hlt | Hge].
      * rewrite nth_error_app1 in Hty by exact Hlt.
        exact (unknown_from_shape_satisfied statics vars shape tag fields i
                 field' ty Hshape Hfield' (Hnth i ty Hty)).
      * rewrite nth_error_app2 in Hty by exact Hge.
        destruct (i - length before) as [ | j] eqn:Hi; simpl in Hty;
          [ | destruct j; discriminate Hty ].
        injection Hty as <-. replace i with index in Hfield' by lia.
        rewrite Hfield in Hfield'. injection Hfield' as <-.
        apply alias_satisfied. split; [ exact Hx | ].
        exact (block_shape_element_kind_field shape tag fields index field k
                 Hshape Hfield Hkind).
Qed.

Theorem immutable_block_with_size_at_least_satisfied
  : immutable_block_with_size_at_least_satisfied_statement.
Proof.
  intros statics vars v tag n shape x t.
  unfold immutable_block_with_size_at_least.
  destruct (size_at_least_field_tys shape x n) as [field_tys | ] eqn:Htys;
    [ | discriminate ].
  destruct
    (row_like_for_blocks_create shape field_tys (Open_or_closed.Open tag)
       Alloc_mode.Heap_or_local)
    as [blocks | ] eqn:Hblocks;
    [ | discriminate ].
  intros [= <-]. rewrite (create_variant_satisfied statics vars v). split.
  - intros [[i [_ [Hbottom _]]] | [tag' [fields [Hblock [Hrow _]]]]];
      [ destruct
          (bottom_satisfied statics vars _
             (Naked_number Kind_naked_immediate) Hbottom)
      | ].
    apply (row_like_for_blocks_create_satisfied statics vars tag' fields
             shape field_tys _ _ blocks Hblocks)
      in Hrow as [Htag [Hshape Hfields]].
    destruct (size_at_least_field_tys_satisfied statics vars shape x n
                field_tys tag' fields Htys)
      as [Hlength Hiff].
    rewrite Hlength in Htag.
    assert (Hsize : n <= length fields) by (destruct tag; tauto).
    exists tag', fields. split; [ exact Hblock | ].
    split; [ destruct tag; tauto | ]. split; [ exact Hsize | ].
    split; [ exact Hshape | ]. exact (proj1 (Hiff Hshape Hsize) Hfields).
  - intros [tag' [fields [Hblock [Htag [Hsize [Hshape Hfield]]]]]]. right.
    exists tag', fields. split; [ exact Hblock | ]. split; [ | exact I ].
    destruct (size_at_least_field_tys_satisfied statics vars shape x n
                field_tys tag' fields Htys)
      as [Hlength Hiff].
    apply (row_like_for_blocks_create_satisfied statics vars tag' fields
             shape field_tys _ _ blocks Hblocks).
    rewrite Hlength.
    split; [ destruct tag; tauto | ]. split; [ exact Hshape | ].
    exact (proj2 (Hiff Hshape Hsize) Hfield).
Qed.

(* ======================================================================== *)
(* Closures                                                                 *)
(* ======================================================================== *)

(* What is projected from a closure is of kind value. *)
Lemma project_function_slot_kind (statics : static_env)
    (move_from move_to : function_slot) (v closure : value) :
  project_function_slot statics move_from move_to v = Some closure ->
  kind_of_value closure = Value.
Proof.
  unfold project_function_slot.
  repeat
    match goal with
    | |- match ?x with _ => _ end = Some _ -> _ => destruct x
    end;
    try discriminate; intros [= <-]; reflexivity.
Qed.

Theorem closure_with_these_function_slots_satisfied
  : closure_with_these_function_slots_satisfied_statement.
Proof.
  intros statics vars v slot bindings.
  unfold closure_with_at_least_these_function_slots.
  rewrite (at_least_this_closure_satisfied statics vars v). split.
  - intros [functions [value_slots [Hclosure [Hfunctions [_ Hentry]]]]].
    exists functions, value_slots. split; [ exact Hclosure | ].
    rewrite domain_map_values in Hfunctions. split; [ exact Hfunctions | ].
    apply closures_entry_lookup_iff in Hentry as [_ [Hclosures _]].
    intros g s closure Hs Hproject.
    specialize (Hclosures g (alias_type_of Value s) closure).
    rewrite lookup_map_values, Hs in Hclosures.
    specialize (Hclosures eq_refl Hproject).
    apply alias_satisfied in Hclosures as [Heval _]. exact Heval.
  - intros [functions [value_slots [Hclosure [Hall Hbindings]]]].
    exists functions, value_slots. split; [ exact Hclosure | ].
    rewrite domain_map_values. split; [ exact Hall | ].
    split; [ reflexivity | ]. apply closures_entry_lookup_iff.
    split; [ | split ].
    + intros g function_type code Hfound.
      rewrite lookup_map_values in Hfound.
      destruct (lookup g bindings); discriminate Hfound.
    + intros g ty closure Hty Hproject. rewrite lookup_map_values in Hty.
      destruct (lookup g bindings) as [s | ] eqn:Hs; [ | discriminate Hty ].
      injection Hty as <-. apply (alias_satisfied statics vars closure Value s).
      split; [ exact (Hbindings g s closure Hs Hproject) | ].
      exact (project_function_slot_kind statics slot g v closure Hproject).
    + intros w ty contents Hty. discriminate Hty.
Qed.

Theorem static_closure_with_this_code_satisfied
  : static_closure_with_this_code_satisfied_statement.
Proof.
  intros statics vars v slot closure_symbol code.
  unfold static_closure_with_this_code.
  rewrite (at_least_this_closure_satisfied statics vars v). split.
  - intros [functions [value_slots [Hclosure [Hfunctions [_ Hentry]]]]].
    exists functions, value_slots. split; [ exact Hclosure | ].
    apply closures_entry_lookup_iff in Hentry as [Hcode [Hclosures _]].
    assert (Hbound : is_bound slot functions = true).
    { rewrite <- among_domain.
      apply (proj1 (all_among _ _) Hfunctions). simpl.
      rewrite Pos.eqb_refl. reflexivity. }
    apply is_bound_lookup in Hbound as [code' Hfound].
    specialize (Hcode slot (Function_type code) code'). simpl in Hcode.
    rewrite Pos.eqb_refl in Hcode. specialize (Hcode eq_refl Hfound).
    simpl in Hcode. subst code'. split; [ exact Hfound | ].
    destruct closure_symbol as [closure_symbol | ]; [ | exact I ].
    intros closure Hproject.
    specialize (Hclosures slot (alias_type_of Value (Symbol closure_symbol))
                  closure).
    simpl in Hclosures. rewrite Pos.eqb_refl in Hclosures.
    specialize (Hclosures eq_refl Hproject).
    apply (alias_satisfied statics vars closure Value) in Hclosures
      as [[= <-] _].
    reflexivity.
  - intros [functions [value_slots [Hclosure [Hfound Hsymbol]]]].
    exists functions, value_slots. split; [ exact Hclosure | ]. split.
    { apply all_among. intros name. simpl. rewrite orb_false_r.
      intros Hname. apply Pos.eqb_eq in Hname. subst name.
      rewrite among_domain. apply is_bound_lookup. exists code. exact Hfound. }
    split; [ reflexivity | ]. apply closures_entry_lookup_iff.
    split; [ | split ].
    + intros g function_type code'. simpl.
      destruct (Pos.eqb_spec g slot) as [-> | _]; [ | discriminate ].
      intros [= <-]. rewrite Hfound. intros [= <-]. reflexivity.
    + intros g ty closure. simpl.
      destruct (Pos.eqb_spec g slot) as [-> | _]; [ | discriminate ].
      intros [= <-] Hproject.
      pose proof (project_function_slot_kind statics slot slot v closure
                    Hproject)
        as Hkind.
      destruct closure_symbol as [closure_symbol | ].
      * apply (alias_satisfied statics vars closure Value).
        split; [ | exact Hkind ]. simpl.
        rewrite (Hsymbol closure Hproject). reflexivity.
      * apply (unknown_satisfied statics vars closure Value). exact Hkind.
    + intros w ty contents Hty. discriminate Hty.
Qed.

Theorem closure_with_these_value_slots_satisfied
  : closure_with_these_value_slots_satisfied_statement.
Proof.
  intros statics vars v slot slots.
  unfold closure_with_at_least_these_value_slots.
  rewrite (at_least_this_closure_satisfied statics vars v). split.
  - intros [functions [value_slots [Hclosure [_ [Hslots Hentry]]]]].
    exists functions, value_slots. split; [ exact Hclosure | ].
    rewrite domain_map_values in Hslots. split; [ exact Hslots | ].
    apply closures_entry_lookup_iff in Hentry as [_ [_ Hcontents]].
    intros w x k contents Hw Hfound.
    specialize (Hcontents w (alias_type_of k (Var x)) contents).
    rewrite lookup_map_values, Hw in Hcontents.
    specialize (Hcontents eq_refl Hfound).
    apply alias_satisfied in Hcontents. exact Hcontents.
  - intros [functions [value_slots [Hclosure [Hall Hcontents]]]].
    exists functions, value_slots. split; [ exact Hclosure | ].
    split; [ reflexivity | ]. rewrite domain_map_values.
    split; [ exact Hall | ]. apply closures_entry_lookup_iff.
    split; [ | split ].
    + intros g function_type code Hfound. discriminate Hfound.
    + intros g ty closure Hty. discriminate Hty.
    + intros w ty contents Hty Hfound. rewrite lookup_map_values in Hty.
      destruct (lookup w slots) as [[x k] | ] eqn:Hw; [ | discriminate Hty ].
      injection Hty as <-.
      apply (alias_satisfied statics vars contents k (Var x)).
      exact (Hcontents w x k contents Hw Hfound).
Qed.

Theorem closure_with_this_value_slot_satisfied
  : closure_with_this_value_slot_satisfied_statement.
Proof.
  intros statics vars v slot w x k.
  unfold closure_with_at_least_this_value_slot.
  rewrite (closure_with_these_value_slots_satisfied statics vars v).
  split.
  - intros [functions [value_slots [Hclosure [Hall Hcontents]]]].
    assert (Hbound : is_bound w value_slots = true).
    { rewrite <- among_domain. apply (proj1 (all_among _ _) Hall). simpl.
      rewrite Pos.eqb_refl. reflexivity. }
    apply is_bound_lookup in Hbound as [contents Hfound].
    exists functions, value_slots, contents. split; [ exact Hclosure | ].
    split; [ exact Hfound | ].
    apply (Hcontents w x k contents); [ | exact Hfound ].
    simpl. rewrite Pos.eqb_refl. reflexivity.
  - intros [functions [value_slots [contents [Hclosure [Hfound [Hx Hkind]]]]]].
    exists functions, value_slots. split; [ exact Hclosure | ]. split.
    { apply all_among. intros name. simpl. rewrite orb_false_r.
      intros Hname. apply Pos.eqb_eq in Hname. subst name.
      rewrite among_domain. apply is_bound_lookup. exists contents.
      exact Hfound. }
    intros w' x' k' contents'. simpl.
    destruct (Pos.eqb_spec w' w) as [-> | _]; [ | discriminate ].
    intros [= <- <-]. rewrite Hfound. intros [= <-]. split; assumption.
Qed.

(* ======================================================================== *)
(* They are well formed                                                     *)
(* ======================================================================== *)

Lemma kind_these_naked_numbers (k : naked_number_kind) (numbers : list Z) :
  kind (these_naked_numbers k numbers) = Naked_number k.
Proof. destruct k, numbers as [ | i [ | i' numbers]]; reflexivity. Qed.

Lemma these_naked_numbers_well_formed (k : naked_number_kind)
    (numbers : list Z) :
  type_well_formed (these_naked_numbers k numbers) = true.
Proof.
  destruct k; try apply these_naked_immediates_well_formed;
    destruct numbers as [ | i [ | i' numbers]]; reflexivity.
Qed.

(* The types that [unknown_types_from_shape] gives are unknown types,
   and those of [size_at_least_field_tys] are unknown types and an
   alias to the variable. *)
Lemma unknown_types_from_shape_unknown (shape : block_shape) (n : nat)
    (tys : list flambda_type) :
  unknown_types_from_shape shape n = Some tys ->
  ∀ ty : flambda_type, ty ∈ tys -> ∃ k : flambda_kind, ty = unknown k.
Proof.
  revert tys. induction n as [ | n IH]; intros tys; simpl.
  - intros [= <-] ty [].
  - destruct (unknown_types_from_shape shape n) as [before | ];
      [ | discriminate ].
    unfold unknown_from_shape.
    destruct (block_shape_element_kind shape n) as [k | ]; [ | discriminate ].
    simpl. intros [= <-] ty Hin.
    apply in_app_or in Hin as [Hin | [<- | []]];
      [ exact (IH before eq_refl ty Hin) | exists k; reflexivity ].
Qed.

Lemma size_at_least_field_tys_parts (shape : block_shape) (x : variable)
    (n : nat) (tys : list flambda_type) :
  size_at_least_field_tys shape x n = Some tys ->
  ∀ ty : flambda_type,
    ty ∈ tys ->
    ∃ k : flambda_kind, ty = unknown k ∨ ty = alias_type_of k (Var x).
Proof.
  unfold size_at_least_field_tys. destruct n as [ | index].
  - intros [= <-] ty [].
  - destruct (unknown_types_from_shape shape index) as [before | ] eqn:Hbefore;
      [ | discriminate ].
    destruct (block_shape_element_kind shape index) as [k | ];
      [ | discriminate ].
    intros [= <-] ty Hin. apply in_app_or in Hin as [Hin | [<- | []]].
    + destruct (unknown_types_from_shape_unknown shape index before Hbefore
                  ty Hin)
        as [k' ->].
      exists k'. left. reflexivity.
    + exists k. right. reflexivity.
Qed.

Theorem row_like_for_blocks_create_checked
  : row_like_for_blocks_create_checked_statement.
Proof.
  intros shape field_tys open_or_closed alloc_mode blocks.
  unfold row_like_for_blocks_create.
  pose proof (fun tag =>
                row_like_for_blocks_create_well_formed tag (length field_tys)
                  shape field_tys alloc_mode)
    as Hcreate.
  cbv zeta in Hcreate.
  destruct open_or_closed as [[tag | ] | tag], shape as [scannable | ];
    try destruct (Z.ltb_spec tag no_scan_tag) as [Hlt | Hge];
    try destruct (Z.eqb_spec tag double_array_tag) as [Heq | Hneq];
    try discriminate; intros [= <-];
    first
      [ destruct (Hcreate tag) as [Hexactly [Hat_least Hunknown_tag]]
      | destruct (Hcreate 0%Z) as [Hexactly [Hat_least Hunknown_tag]]
      | destruct (Hcreate double_array_tag)
          as [Hexactly [Hat_least Hunknown_tag]] ];
    rewrite ?Hexactly, ?Hat_least, ?Hunknown_tag; intuition lia.
Qed.

Theorem immutable_block_with_size_at_least_well_formed
  : immutable_block_with_size_at_least_well_formed_statement.
Proof.
  intros tag n shape x t. unfold immutable_block_with_size_at_least.
  destruct (size_at_least_field_tys shape x n) as [field_tys | ] eqn:Htys;
    [ | discriminate ].
  destruct
    (row_like_for_blocks_create shape field_tys (Open_or_closed.Open tag)
       Alloc_mode.Heap_or_local)
    as [blocks | ] eqn:Hblocks;
    [ | discriminate ].
  intros [= <-].
  rewrite
    (create_variant_well_formed false
       (Or_unknown.Known (bottom (Naked_number Kind_naked_immediate)))
       (Or_unknown.Known blocks) No_extensions),
    (row_like_for_blocks_create_checked shape field_tys _ _ blocks Hblocks).
  assert (Hfields :
            ∀ ty : flambda_type,
              ty ∈ field_tys -> type_well_formed ty = true).
  { intros ty Hin.
    destruct (size_at_least_field_tys_parts shape x n field_tys Htys ty Hin)
      as [k [-> | ->]];
      [ apply unknown_well_formed | apply alias_type_of_well_formed ]. }
  split.
  - intros [_ [[Hshape _] _]]. exact Hshape.
  - intros Hshape. split; [ split; reflexivity | ].
    split; [ split; assumption | exact I ].
Qed.

Lemma keys_of_map_values {A B : Type} (f : A -> B) (m : finite_map A) :
  keys (map_values f m) = keys m.
Proof. unfold keys, map_values. rewrite map_map. reflexivity. Qed.

Theorem closure_with_at_least_well_formed
  : closure_with_at_least_well_formed_statement.
Proof.
  split.
  - intros slot bindings.
    unfold closure_with_at_least_these_function_slots.
    rewrite (at_least_this_closure_well_formed _ _ _ _ _),
      !domain_map_values, !keys_of_map_values, all_among_refl.
    split.
    + intros [_ [[Hdistinct _] _]]. exact Hdistinct.
    + intros Hdistinct. split; [ reflexivity | ].
      split; [ split; [ exact Hdistinct | split; [ exact Hdistinct | ] ] | ];
        [ reflexivity | ].
      split.
      * intros g ty Hty. rewrite lookup_map_values in Hty.
        destruct (lookup g bindings) as [s | ]; [ | discriminate Hty ].
        injection Hty as <-.
        split; [ reflexivity | exact (alias_type_of_well_formed Value s) ].
      * intros w ty Hty. discriminate Hty.
  - intros slot slots. unfold closure_with_at_least_these_value_slots.
    rewrite (at_least_this_closure_well_formed _ _ _ _ _),
      keys_of_map_values.
    split.
    + intros [_ [[_ [_ Hdistinct]] _]]. exact Hdistinct.
    + intros Hdistinct. split; [ reflexivity | ].
      split; [ split; [ reflexivity | split; [ reflexivity | ] ] | ];
        [ exact Hdistinct | ].
      split.
      * intros g ty Hty. discriminate Hty.
      * intros w ty Hty. rewrite lookup_map_values in Hty.
        destruct (lookup w slots) as [[x k] | ]; [ | discriminate Hty ].
        injection Hty as <-. apply alias_type_of_well_formed.
Qed.

Theorem more_creators_well_formed : more_creators_well_formed_statement.
Proof.
  split.
  { intros t. split; [ apply unknown_well_formed | apply bottom_well_formed ]. }
  split.
  { intros shape index t. unfold unknown_from_shape.
    destruct (block_shape_element_kind shape index) as [k | ];
      [ | discriminate ].
    intros [= <-]. apply unknown_well_formed. }
  split; [ exact unknown_with_subkind_well_formed | ].
  split.
  { intros a t Hin. unfold unknown_types_from_arity in Hin.
    apply in_map_iff in Hin as [k [<- _]].
    apply unknown_with_subkind_well_formed. }
  split; [ exact these_naked_numbers_well_formed | ].
  split; [ apply these_tagged_immediates_well_formed | ].
  split; [ apply these_naked_immediates_well_formed | ].
  split; [ reflexivity | ].
  split.
  { intros x. apply tag_immediate_well_formed.
    split; [ reflexivity | apply alias_type_of_well_formed ]. }
  split.
  { intros k x alloc_mode. apply box_number_well_formed.
    split; [ apply kind_alias_type_of | apply alias_type_of_well_formed ]. }
  split.
  { intros k i alloc_mode. apply box_number_well_formed.
    split; [ | apply alias_type_of_well_formed ].
    unfold this_const.
    rewrite kind_alias_type_of, kind_of_const_of_naked_number,
      unboxed_kind_naked_number.
    reflexivity. }
  split.
  { intros k numbers alloc_mode. apply box_number_well_formed.
    split; [ | apply these_naked_numbers_well_formed ].
    rewrite kind_these_naked_numbers, unboxed_kind_naked_number.
    reflexivity. }
  split.
  { intros k. apply box_number_well_formed.
    split; [ apply kind_unknown | apply unknown_well_formed ]. }
  split.
  { intros slot closure_symbol code. unfold static_closure_with_this_code.
    apply at_least_this_closure_well_formed.
    split; [ apply all_among_refl | ].
    split; [ repeat split | ].
    split.
    - intros g ty. simpl.
      destruct (Pos.eqb g slot); [ | discriminate ]. intros [= <-].
      destruct closure_symbol; split; reflexivity.
    - intros w ty Hty. discriminate Hty. }
  intros slot w x k. unfold closure_with_at_least_this_value_slot.
  apply (proj2 closure_with_at_least_well_formed). reflexivity.
Qed.

(* ======================================================================== *)
(* Under a renaming                                                         *)
(* ======================================================================== *)

Lemma rename_type_these_naked_numbers (r : renaming) (k : naked_number_kind)
    (numbers : list Z) :
  rename_type r (these_naked_numbers k numbers) = these_naked_numbers k numbers.
Proof. destruct k, numbers as [ | i [ | i' numbers]]; reflexivity. Qed.

Theorem more_creators_closed : more_creators_closed_statement.
Proof.
  intros r.
  split.
  { intros t. unfold unknown_like, bottom_like.
    rewrite kind_rename_type, rename_type_unknown, rename_type_bottom.
    repeat split. }
  split.
  { intros shape index t. unfold unknown_from_shape.
    destruct (block_shape_element_kind shape index) as [k | ];
      [ | discriminate ].
    intros [= <-]. apply rename_type_unknown. }
  split; [ exact (rename_type_unknown_with_subkind r) | ].
  split.
  { intros a. unfold unknown_types_from_arity. rewrite map_map.
    apply map_ext. intros k. apply rename_type_unknown_with_subkind. }
  split; [ exact (rename_type_these_naked_numbers r) | ].
  split; [ reflexivity | ]. split; [ reflexivity | ].
  split; [ reflexivity | ].
  split; [ intros k i alloc_mode; destruct k; reflexivity | ].
  split.
  { intros k numbers alloc_mode. unfold these_boxed_numbers.
    rewrite rename_type_box_number, rename_type_these_naked_numbers.
    reflexivity. }
  split; [ intros k; destruct k; reflexivity | ].
  intros slot closure_symbol code. destruct closure_symbol; reflexivity.
Qed.

Lemma size_at_least_field_tys_renamed (r : renaming) (shape : block_shape)
    (x : variable) (n : nat) :
  size_at_least_field_tys shape (rename_variable r x) n
  = option_map (map (rename_type r)) (size_at_least_field_tys shape x n).
Proof.
  unfold size_at_least_field_tys. destruct n as [ | index]; [ reflexivity | ].
  destruct (unknown_types_from_shape shape index) as [before | ] eqn:Hbefore;
    [ | reflexivity ].
  destruct (block_shape_element_kind shape index) as [k | ];
    [ | reflexivity ].
  simpl. rewrite map_app. simpl. rewrite rename_type_alias_type_of. simpl.
  rewrite map_fixes; [ reflexivity | ]. intros ty Hin.
  destruct (unknown_types_from_shape_unknown shape index before Hbefore ty
              Hin)
    as [k' ->].
  apply rename_type_unknown.
Qed.

Lemma row_like_for_blocks_create_renamed (r : renaming) (shape : block_shape)
    (field_tys : list flambda_type) (open_or_closed : Open_or_closed.t)
    (alloc_mode : Alloc_mode.t) :
  option_map (rename_row_like_for_blocks r)
    (row_like_for_blocks_create shape field_tys open_or_closed alloc_mode)
  = row_like_for_blocks_create shape (map (rename_type r) field_tys)
      open_or_closed alloc_mode.
Proof.
  unfold row_like_for_blocks_create. rewrite length_map.
  destruct open_or_closed as [[tag | ] | tag], shape as [scannable | ];
    try destruct (Z.ltb tag no_scan_tag);
    try destruct (Z.eqb tag double_array_tag);
    reflexivity.
Qed.

Lemma renamed_value_slot_aliases (r : renaming)
    (slots : finite_map (variable * flambda_kind)) :
  map_values (rename_type r)
    (map_values
       (fun value_slot => alias_type_of (snd value_slot) (Var (fst value_slot)))
       slots)
  = map_values
      (fun value_slot => alias_type_of (snd value_slot) (Var (fst value_slot)))
      (map_values
         (fun value_slot =>
            (rename_variable r (fst value_slot), snd value_slot))
         slots).
Proof.
  unfold map_values. rewrite !map_map. apply map_ext. intros [w [x k]].
  simpl. rewrite rename_type_alias_type_of. reflexivity.
Qed.

Theorem more_creators_rename : more_creators_rename_statement.
Proof.
  intros r.
  split.
  { intros t n. unfold is_alias_of_name. rewrite get_alias_opt_renamed.
    destruct (get_alias_opt t) as [[x | s | c] | ]; simpl; try reflexivity.
    - exact (name_eqb_renamed r (Name_var x) n).
    - exact (name_eqb_renamed r (Name_symbol s) n). }
  split; [ reflexivity | ].
  split; [ intros k x alloc_mode; destruct k; reflexivity | ].
  split; [ exact (row_like_for_blocks_create_renamed r) | ].
  split.
  { intros tag n shape x. unfold immutable_block_with_size_at_least.
    rewrite size_at_least_field_tys_renamed.
    destruct (size_at_least_field_tys shape x n) as [field_tys | ];
      [ | reflexivity ].
    simpl. rewrite <- row_like_for_blocks_create_renamed.
    destruct
      (row_like_for_blocks_create shape field_tys (Open_or_closed.Open tag)
         Alloc_mode.Heap_or_local);
      reflexivity. }
  split.
  { intros slot bindings.
    unfold closure_with_at_least_these_function_slots.
    rewrite rename_type_at_least_this_closure. unfold map_values.
    rewrite !map_map. reflexivity. }
  split.
  { intros slot slots. unfold closure_with_at_least_these_value_slots.
    rewrite rename_type_at_least_this_closure, renamed_value_slot_aliases.
    reflexivity. }
  intros slot w x k. unfold closure_with_at_least_this_value_slot,
    closure_with_at_least_these_value_slots.
  rewrite rename_type_at_least_this_closure. simpl.
  rewrite rename_type_alias_type_of. reflexivity.
Qed.

(* ======================================================================== *)
(* Their aliases, with their kinds                                          *)
(* ======================================================================== *)

Lemma aliases_with_kinds_alias_type_of (k : flambda_kind) (s : simple) :
  aliases_with_kinds (alias_type_of k s) = [(s, k)].
Proof. destruct k as [ | [] | | ]; reflexivity. Qed.

Lemma aliases_with_kinds_of_unknown (k : flambda_kind) :
  aliases_with_kinds (unknown k) = [].
Proof. destruct k as [ | [] | | ]; reflexivity. Qed.

(* The aliases of the blocks that [Row_like_for_blocks.create]
   answers are those of the types of the fields. *)
Lemma aliases_of_created_blocks (shape : block_shape)
    (field_tys : list flambda_type) (open_or_closed : Open_or_closed.t)
    (alloc_mode : Alloc_mode.t) (blocks : row_like_for_blocks) :
  row_like_for_blocks_create shape field_tys open_or_closed alloc_mode
  = Some blocks ->
  aliases_of_row_like_for_blocks blocks = flat_map aliases_with_kinds field_tys.
Proof.
  unfold row_like_for_blocks_create.
  destruct open_or_closed as [[tag | ] | tag], shape as [scannable | ];
    try destruct (Z.ltb tag no_scan_tag);
    try destruct (Z.eqb tag double_array_tag);
    try discriminate; intros [= <-]; simpl; unfold aliases_of_block_case;
    simpl; rewrite ?app_nil_r; reflexivity.
Qed.

Lemma flat_map_none {A B : Type} (f : A -> list B) (l : list A) :
  (∀ a : A, a ∈ l -> f a = []) -> flat_map f l = [].
Proof.
  induction l as [ | a l IH]; simpl; intros Hnone; [ reflexivity | ].
  rewrite (Hnone a (or_introl eq_refl)), IH; [ reflexivity | ].
  intros b Hin. exact (Hnone b (or_intror Hin)).
Qed.

Theorem more_creators_aliases : more_creators_aliases_statement.
Proof.
  split; [ reflexivity | ].
  split; [ intros k x alloc_mode; destruct k; reflexivity | ].
  split.
  { intros tag index shape x t. unfold immutable_block_with_size_at_least.
    destruct (size_at_least_field_tys shape x (S index)) as [field_tys | ]
      eqn:Htys;
      [ | discriminate ].
    destruct
      (row_like_for_blocks_create shape field_tys (Open_or_closed.Open tag)
         Alloc_mode.Heap_or_local)
      as [blocks | ] eqn:Hblocks;
      [ | discriminate ].
    intros [= <-]. unfold size_at_least_field_tys in Htys.
    destruct (unknown_types_from_shape shape index) as [before | ] eqn:Hbefore;
      [ | discriminate Htys ].
    destruct (block_shape_element_kind shape index) as [k | ];
      [ | discriminate Htys ].
    injection Htys as <-. exists k. split; [ reflexivity | ].
    rewrite
      (aliases_with_kinds_create_variant false
         (Or_unknown.Known (Type_naked_immediate Bottom))
         (Or_unknown.Known blocks) No_extensions),
      (aliases_of_created_blocks _ _ _ _ blocks Hblocks), flat_map_app.
    simpl. rewrite aliases_with_kinds_alias_type_of, !app_nil_r.
    rewrite (flat_map_none aliases_with_kinds before); [ reflexivity | ].
    intros ty Hin.
    destruct (unknown_types_from_shape_unknown shape index before Hbefore ty
                Hin)
      as [k' ->].
    apply aliases_with_kinds_of_unknown. }
  split.
  { intros slot bindings.
    unfold closure_with_at_least_these_function_slots.
    rewrite (proj2 (aliases_with_kinds_this_closure _ _ _ _ _)). simpl.
    rewrite app_nil_r.
    induction bindings as [ | [g s] bindings IH]; simpl;
      [ reflexivity | rewrite IH; reflexivity ]. }
  intros slot slots. unfold closure_with_at_least_these_value_slots.
  rewrite (proj2 (aliases_with_kinds_this_closure _ _ _ _ _)). simpl.
  induction slots as [ | [w [x k]] slots IH]; simpl; [ reflexivity | ].
  rewrite aliases_with_kinds_alias_type_of, IH. reflexivity.
Qed.
