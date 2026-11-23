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

(* Proofs of the statements of FlambdaSubkindBinderSpec.v. There is no
   statement here, and nothing here needs review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaWellFormedSpec FlambdaSubkindProofs
  FlambdaSubkindBinderSpec.
From OxCamlFlambda2 Require FlambdaSemanticsProofs.
Import ListNotations.

Theorem handler_binds_at_subkinds : handler_binds_at_subkinds_statement.
Proof.
  intros params metadata handler def_vars def_conts recursive vs statics hp
    rd s.
  simpl.
  destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
    [ | discriminate ].
  destruct (values_of_subkinds statics vs (hm_params_subkinds metadata));
    [ reflexivity | discriminate ].
Qed.

Theorem call_binds_at_subkinds : call_binds_at_subkinds_statement.
Proof.
  intros closure c vs args_arity return_arity return_to exn_to statics hp rd
    s.
  apply FlambdaSemanticsProofs.call_code_step_subkinds.
Qed.

Theorem values_of_subkinds_extend : values_of_subkinds_extend_statement.
Proof.
  intros statics statics' vs sks [_ Hvalues].
  exact (FlambdaSubkindProofs.values_of_subkinds_extend statics statics' vs
           sks Hvalues).
Qed.

Theorem value_of_subkind_extend : value_of_subkind_extend_statement.
Proof.
  intros statics statics' sk v Hextend Hv.
  destruct v as [c | | | | | | | | ].
  1: { destruct sk, c; exact Hv. }
  all:
    match goal with
    | |- value_of_subkind _ _ ?w = true =>
      pose proof
        (values_of_subkinds_extend statics statics' [w] [(true, sk)] Hextend)
        as H
    end;
    cbn [values_of_subkinds value_at fst snd] in H;
    rewrite !andb_true_r in H; exact (H Hv).
Qed.

Theorem values_of_subkinds_related : values_of_subkinds_related_statement.
Proof. exact FlambdaSubkindProofs.values_of_subkinds_related. Qed.

Lemma fields_of_subkinds_length (statics : static_env) (sks : field_subkinds) :
  ∀ fields : list value,
    fields_of_subkinds statics sks fields = true ->
    length fields = number_of_fields sks.
Proof.
  induction sks as [ | nullable sk rest IH]; intros [ | v fields]; simpl;
    try discriminate; [ reflexivity | ].
  intros H. apply andb_true_iff in H as [_ Hrest]. f_equal. apply IH, Hrest.
Qed.

Lemma fields_of_subkinds_values (statics : static_env)
    (sks : field_subkinds) :
  ∀ fields : list value,
    fields_of_subkinds statics sks fields = true ->
    Forall (fun field => kind_of_value field = Value) fields.
Proof.
  induction sks as [ | nullable sk rest IH]; intros [ | v fields];
    try discriminate; [ constructor | ].
  rewrite fields_of_subkinds_cons. intros H.
  apply andb_true_iff in H as [Hhead Hrest].
  apply andb_true_iff in Hhead as [Hvalue _].
  constructor; [ | apply IH, Hrest ].
  unfold kind_eqb in Hvalue.
  destruct (kind_eq_dec (kind_of_value v) Value);
    [ assumption | discriminate Hvalue ].
Qed.

Theorem value_of_block_subkind : value_of_block_subkind_statement.
Proof.
  intros statics tag sks v. unfold block_subkind. simpl.
  assert (Hblock :
    match block_of statics v with
    | Some (tag0, fields) =>
      if Z.eqb tag0 tag then fields_of_subkinds statics sks fields
      else false
    | None => false
    end = true ->
    ∃ fields : list value,
      block_of statics v = Some (tag, fields)
      ∧ length fields = number_of_fields sks
      ∧ fields_of_subkinds statics sks fields = true
      ∧ Forall (fun field => kind_of_value field = Value) fields).
  { destruct (block_of statics v) as [[tag0 fields] | ]; [ | discriminate ].
    destruct (Z.eqb_spec tag0 tag) as [-> | _]; [ | discriminate ].
    intros Hfields. exists fields. split; [ reflexivity | ].
    split; [ apply (fields_of_subkinds_length statics), Hfields | ].
    split; [ exact Hfields | ].
    apply (fields_of_subkinds_values statics sks), Hfields. }
  destruct v as [c | | | | | | | | ]; try exact Hblock.
  destruct c; try exact Hblock.
  rewrite andb_false_r. discriminate.
Qed.

Theorem load_at_block_subkind : load_at_block_subkind_statement.
Proof.
  intros statics tag sks v field_kind field Hv.
  destruct (value_of_block_subkind statics tag sks v Hv)
    as (fields & Hblock & Hlength & _).
  unfold eval_block_load. rewrite Hblock. simpl.
  rewrite Z.eqb_refl, Hlength, Z.eqb_refl. reflexivity.
Qed.

(* The field at a position that the subkinds give. *)
Lemma fields_of_subkinds_nth (statics : static_env) (sks : field_subkinds) :
  ∀ (fields : list value) (field : nat) (ps : param_subkind),
    fields_of_subkinds statics sks fields = true ->
    field_subkind_at sks field = Some ps ->
    ∃ loaded : value,
      nth_error fields field = Some loaded
      ∧ kind_eqb (kind_of_value loaded) Value = true
      ∧ value_at statics ps loaded = true.
Proof.
  induction sks as [ | nullable sk rest IH]; intros [ | v fields] field ps;
    try discriminate.
  rewrite fields_of_subkinds_cons. intros H.
  apply andb_true_iff in H as [Hhead Hrest].
  apply andb_true_iff in Hhead as [Hvalue Hhead].
  destruct field as [ | field]; simpl.
  - intros [= <-]. exists v. auto.
  - apply IH, Hrest.
Qed.

Theorem load_of_block_subkind : load_of_block_subkind_statement.
Proof.
  intros statics tag sks v field ps Hv Hfield.
  destruct (value_of_block_subkind statics tag sks v Hv)
    as (fields & Hblock & _ & Hfields & _).
  destruct (fields_of_subkinds_nth statics sks fields field ps Hfields Hfield)
    as (loaded & Hnth & Hvalue & Hat).
  exists loaded. unfold eval_block_load. rewrite Hblock. simpl.
  assert (Hpos : Z.leb 0 (Z.of_nat field) = true)
    by (apply Z.leb_le, Nat2Z.is_nonneg).
  rewrite Hpos, Nat2Z.id, Hnth. simpl. rewrite Hvalue.
  split; [ reflexivity | ]. split; [ | exact Hat ].
  unfold kind_eqb in Hvalue.
  destruct (kind_eq_dec (kind_of_value loaded) Value);
    [ assumption | discriminate Hvalue ].
Qed.
