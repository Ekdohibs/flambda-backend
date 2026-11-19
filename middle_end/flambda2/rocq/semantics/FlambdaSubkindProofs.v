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

(* Facts about the definitions of FlambdaSubkindSpec.v, for the proofs
   that step through a binder. There is no statement here, and nothing
   here needs review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaValueSpec FlambdaPrimitiveSpec FlambdaSubkindSpec.
From OxCamlFlambda2 Require FlambdaPrimitiveProofs.
Import ListNotations.

(* ======================================================================== *)
(* The shape of the tests                                                   *)
(* ======================================================================== *)

(* What is tested of a value against a subkind that may be nullable:
   the test of a field of a block, and of a parameter. *)
Definition subkind_at (statics : static_env) (nullable : bool) (sk : subkind)
    (v : value) : bool :=
  match v with
  | Value_const (Poison _) => says_nothing nullable sk
  | Value_const Null => nullable
  | _ => value_of_subkind statics sk v
  end.

Lemma value_at_eq (statics : static_env) (ps : param_subkind) (v : value) :
  value_at statics ps v = subkind_at statics (fst ps) (snd ps) v.
Proof. reflexivity. Qed.

Lemma fields_of_subkinds_cons (statics : static_env) (nullable : bool)
    (sk : subkind) (sks : field_subkinds) (v : value) (fields : list value) :
  fields_of_subkinds statics (Fields_cons nullable sk sks) (v :: fields)
  = kind_eqb (kind_of_value v) Value && subkind_at statics nullable sk v
    && fields_of_subkinds statics sks fields.
Proof. reflexivity. Qed.

Lemma values_of_subkinds_cons (statics : static_env) (v : value)
    (vs : list value) (ps : param_subkind) (sks : list param_subkind) :
  values_of_subkinds statics (v :: vs) (ps :: sks)
  = value_at statics ps v && values_of_subkinds statics vs sks.
Proof. reflexivity. Qed.

Lemma values_of_subkinds_nil (statics : static_env) (vs : list value) :
  values_of_subkinds statics vs [] = true.
Proof. reflexivity. Qed.

Lemma values_of_subkinds_no_values (statics : static_env)
    (sks : list param_subkind) :
  values_of_subkinds statics [] sks = true.
Proof. destruct sks; reflexivity. Qed.

(* A subkind that says nothing is one that every value is of. *)
Lemma says_nothing_any (statics : static_env) (nullable : bool) (sk : subkind)
    (v : value) :
  says_nothing nullable sk = true -> subkind_at statics nullable sk v = true.
Proof.
  unfold says_nothing. intros H. apply andb_true_iff in H as [-> Hsk].
  destruct sk; try discriminate.
  destruct v as [c | | | | | | | | ]; try reflexivity.
  destruct c; reflexivity.
Qed.

Lemma value_at_any_value (statics : static_env) (v : value) :
  value_at statics any_value v = true.
Proof. rewrite value_at_eq. apply says_nothing_any. reflexivity. Qed.

(* Subkinds that say nothing ask nothing. *)
Lemma values_of_subkinds_nothing (statics : static_env) :
  ∀ (sks : list param_subkind) (vs : list value),
    subkinds_say_nothing sks = true ->
    values_of_subkinds statics vs sks = true.
Proof.
  unfold subkinds_say_nothing.
  induction sks as [ | ps sks IH]; intros [ | v vs]; simpl; auto.
  intros H. apply andb_true_iff in H as [Hhead Hrest].
  rewrite value_at_eq, (says_nothing_any statics _ _ v Hhead). simpl.
  apply IH, Hrest.
Qed.

(* The test of a constant does not read the static environment. *)
Lemma subkind_at_const (statics statics' : static_env) (nullable : bool)
    (sk : subkind) (c : const) :
  subkind_at statics nullable sk (Value_const c)
  = subkind_at statics' nullable sk (Value_const c).
Proof. destruct sk, c; reflexivity. Qed.

Lemma subkind_at_not_const (statics : static_env) (nullable : bool)
    (sk : subkind) (v : value) :
  (∀ c : const, v <> Value_const c) ->
  subkind_at statics nullable sk v = value_of_subkind statics sk v.
Proof.
  intros Hv. destruct v as [c | | | | | | | | ]; try reflexivity.
  destruct (Hv c eq_refl).
Qed.

(* The test of the values that two lists begin with, and of the rest. *)
Lemma values_of_subkinds_app (statics : static_env) :
  ∀ (sks : list param_subkind) (vs vs' : list value),
    values_of_subkinds statics (vs ++ vs') sks
    = values_of_subkinds statics vs (firstn (length vs) sks)
      && values_of_subkinds statics vs' (skipn (length vs) sks).
Proof.
  induction sks as [ | ps sks IH]; intros [ | v vs] vs'; simpl;
    try reflexivity.
  rewrite IH, andb_assoc. reflexivity.
Qed.

(* ======================================================================== *)
(* Related values                                                           *)
(* ======================================================================== *)

(* Two runs whose values are related: the same constants, but that the
   first may have poison where the second has any value; and what the
   first is observed to be, the second is, with related parts. *)
Section Related.

Variable R : value -> value -> Prop.
Variables statics statics' : static_env.

Hypothesis Hleft :
  ∀ (c : const) (v' : value),
    R (Value_const c) v' ->
    v' = Value_const c ∨ ∃ kind : flambda_kind, c = Poison kind.
Hypothesis Hright :
  ∀ (v : value) (c : const),
    R v (Value_const c) ->
    v = Value_const c ∨ ∃ kind : flambda_kind, v = Value_const (Poison kind).
Hypothesis Hboxed :
  ∀ (v v' : value) (c : const),
    R v v' ->
    boxed_number_of statics v = Some c ->
    boxed_number_of statics' v' = Some c.
Hypothesis Hblock :
  ∀ (v v' : value) (tag : Z) (fields : list value),
    R v v' ->
    block_of statics v = Some (tag, fields) ->
    ∃ fields' : list value,
      block_of statics' v' = Some (tag, fields') ∧ Forall2 R fields fields'.
Hypothesis Hkind :
  ∀ v v' : value, R v v' -> kind_of_value v = kind_of_value v'.

Lemma related_not_const (v v' : value) :
  R v v' ->
  (∀ c : const, v <> Value_const c) ->
  ∀ c : const, v' <> Value_const c.
Proof.
  intros HR Hv c ->. destruct (Hright v c HR) as [-> | [kind ->]];
    [ exact (Hv c eq_refl) | exact (Hv _ eq_refl) ].
Qed.

(* The test at a subkind that may be nullable, given the test of the
   subkind on values that are not constants. *)
Lemma subkind_at_related (nullable : bool) (sk : subkind) :
  (∀ v v' : value,
    R v v' ->
    (∀ c : const, v <> Value_const c) ->
    value_of_subkind statics sk v = true ->
    value_of_subkind statics' sk v' = true) ->
  ∀ v v' : value,
    R v v' ->
    subkind_at statics nullable sk v = true ->
    subkind_at statics' nullable sk v' = true.
Proof.
  intros IH v v' HR Hv.
  destruct v as [c | | | | | | | | ].
  1: { destruct (Hleft c v' HR) as [-> | [kind ->]];
         [ rewrite <- (subkind_at_const statics statics'); exact Hv
         | apply says_nothing_any; exact Hv ]. }
  all:
    (rewrite subkind_at_not_const
       by exact (related_not_const _ _ HR ltac:(discriminate));
     apply (IH _ _ HR); [ discriminate | exact Hv ]).
Qed.

Fixpoint value_of_subkind_related_aux (sk : subkind) {struct sk} :
  ∀ v v' : value,
    R v v' ->
    (∀ c : const, v <> Value_const c) ->
    value_of_subkind statics sk v = true ->
    value_of_subkind statics' sk v' = true
with block_of_shapes_related_aux (shapes : constructor_shapes)
    {struct shapes} :
  ∀ (tag : Z) (fields fields' : list value),
    Forall2 R fields fields' ->
    block_of_shapes statics shapes tag fields = true ->
    block_of_shapes statics' shapes tag fields' = true
with fields_of_subkinds_related_aux (sks : field_subkinds) {struct sks} :
  ∀ fields fields' : list value,
    Forall2 R fields fields' ->
    fields_of_subkinds statics sks fields = true ->
    fields_of_subkinds statics' sks fields' = true.
Proof.
  - destruct sk as [ | | n | consts non_consts]; intros v v' HR Hv; simpl.
    + reflexivity.
    + destruct v as [c | | | | | | | | ]; try discriminate.
      destruct (Hv c eq_refl).
    + destruct (boxed_number_of statics v) as [c | ] eqn:Hc;
        [ | discriminate ].
      rewrite (Hboxed v v' c HR Hc). exact (fun H => H).
    + assert (Hof :
        match block_of statics v with
        | Some (tag, fields) =>
          block_of_shapes statics non_consts tag fields
        | None => false
        end = true ->
        match block_of statics' v' with
        | Some (tag, fields) =>
          block_of_shapes statics' non_consts tag fields
        | None => false
        end = true).
      { destruct (block_of statics v) as [[tag fields] | ] eqn:Hb;
          [ | discriminate ].
        destruct (Hblock v v' tag fields HR Hb) as (fields' & -> & Hfields).
        exact (block_of_shapes_related_aux non_consts tag fields fields'
          Hfields). }
      pose proof (related_not_const v v' HR Hv) as Hv'.
      destruct v as [c | | | | | | | | ]; [ destruct (Hv c eq_refl) | .. ];
        (destruct v' as [c' | | | | | | | | ];
         [ destruct (Hv' c' eq_refl) | .. ]; exact Hof).
  - destruct shapes as [ | tag' rest | tag' sks rest];
      intros tag fields fields' Hfields; simpl.
    + discriminate.
    + destruct (Z.eqb tag tag');
        [ reflexivity
        | exact (block_of_shapes_related_aux rest tag fields fields'
            Hfields) ].
    + destruct (Z.eqb tag tag');
        [ exact (fields_of_subkinds_related_aux sks fields fields' Hfields)
        | exact (block_of_shapes_related_aux rest tag fields fields'
            Hfields) ].
  - destruct sks as [ | nullable sk rest]; intros fields fields' Hfields.
    + destruct Hfields; [ reflexivity | discriminate ].
    + destruct Hfields as [ | v v' fields fields' Hv Hfields];
        [ discriminate | ].
      rewrite !fields_of_subkinds_cons. intros H.
      apply andb_true_iff in H as [Hhead Hrest].
      apply andb_true_iff in Hhead as [Hvalue Hhead].
      rewrite <- (Hkind v v' Hv), Hvalue. simpl.
      apply andb_true_iff. split.
      * exact (subkind_at_related nullable sk
          (value_of_subkind_related_aux sk) v v' Hv Hhead).
      * exact (fields_of_subkinds_related_aux rest fields fields' Hfields
          Hrest).
Qed.

Lemma value_at_related (ps : param_subkind) (v v' : value) :
  R v v' -> value_at statics ps v = true -> value_at statics' ps v' = true.
Proof.
  rewrite !value_at_eq.
  apply subkind_at_related, value_of_subkind_related_aux.
Qed.

Lemma values_of_subkinds_related_aux (vs vs' : list value) :
  Forall2 R vs vs' ->
  ∀ sks : list param_subkind,
    values_of_subkinds statics vs sks = true ->
    values_of_subkinds statics' vs' sks = true.
Proof.
  induction 1 as [ | v v' vs vs' Hv _ IH]; intros [ | ps sks]; simpl; auto.
  intros H. apply andb_true_iff in H as [Hhead Hrest].
  apply andb_true_iff. split; [ exact (value_at_related ps v v' Hv Hhead) | ].
  apply IH, Hrest.
Qed.

End Related.

(* Values that the observations read alike are of the same subkinds:
   where the first are, the second are. *)
Lemma values_of_subkinds_related (R : value -> value -> Prop)
    (statics statics' : static_env) (vs vs' : list value)
    (sks : list param_subkind) :
  relates_constants R ->
  relates_observations R statics statics' ->
  Forall2 R vs vs' ->
  values_of_subkinds statics vs sks = true ->
  values_of_subkinds statics' vs' sks = true.
Proof.
  intros [Hleft Hright] Hobserved Hvs.
  apply (values_of_subkinds_related_aux R statics statics'); try exact Hvs.
  - intros c v' HR. left. exact (Hleft c v' HR).
  - intros v c HR. left. exact (Hright v c HR).
  - intros v v' c HR. exact (proj1 (proj2 (Hobserved v v' HR)) c).
  - intros v v' tag fields HR. exact (proj1 (Hobserved v v' HR) tag fields).
  - intros v v' HR.
    destruct v as [c | | | | | | | | ];
      [ rewrite (Hleft c v' HR); reflexivity | .. ];
      (destruct v' as [c' | | | | | | | | ];
       [ discriminate (Hright _ c' HR) | .. ]; reflexivity).
Qed.

Lemma Forall2_eq_same {A : Type} (l : list A) : Forall2 eq l l.
Proof. induction l; constructor; auto. Qed.

(* The test under two static environments that bind the same symbols to
   the same things. *)
Lemma values_of_subkinds_same_values (statics statics' : static_env)
    (vs : list value) (sks : list param_subkind) :
  static_values statics = static_values statics' ->
  values_of_subkinds statics vs sks = values_of_subkinds statics' vs sks.
Proof.
  intros Hvalues.
  assert (Hone :
    ∀ st st' : static_env,
      static_values st = static_values st' ->
      values_of_subkinds st vs sks = true ->
      values_of_subkinds st' vs sks = true).
  { intros st st' Hsame.
    apply (values_of_subkinds_related_aux eq st st').
    - intros c v' <-. left. reflexivity.
    - intros v c ->. left. reflexivity.
    - intros v v' c <-. unfold boxed_number_of. rewrite Hsame.
      exact (fun H => H).
    - intros v v' tag fields <- Hb. exists fields.
      split; [ | apply Forall2_eq_same ].
      unfold block_of in Hb |- *. rewrite <- Hsame. exact Hb.
    - intros v v' <-. reflexivity.
    - apply Forall2_eq_same. }
  destruct (values_of_subkinds statics vs sks) eqn:H.
  - symmetry. exact (Hone _ _ Hvalues H).
  - destruct (values_of_subkinds statics' vs sks) eqn:H'; [ | reflexivity ].
    rewrite (Hone _ _ (eq_sym Hvalues) H') in H. discriminate H.
Qed.

(* The test under two static environments that the observations of a
   block and of a boxed number read alike. *)
Lemma values_of_subkinds_same_observations (statics statics' : static_env)
    (vs : list value) (sks : list param_subkind) :
  (∀ v : value, block_of statics' v = block_of statics v) ->
  (∀ v : value, boxed_number_of statics' v = boxed_number_of statics v) ->
  values_of_subkinds statics' vs sks = values_of_subkinds statics vs sks.
Proof.
  intros Hblock Hboxed.
  assert (Hone :
    ∀ st st' : static_env,
      (∀ v : value, block_of st' v = block_of st v) ->
      (∀ v : value, boxed_number_of st' v = boxed_number_of st v) ->
      values_of_subkinds st vs sks = true ->
      values_of_subkinds st' vs sks = true).
  { intros st st' Hb Hn.
    apply (values_of_subkinds_related_aux eq st st').
    - intros c v' <-. left. reflexivity.
    - intros v c ->. left. reflexivity.
    - intros v v' c <-. rewrite Hn. exact (fun H => H).
    - intros v v' tag fields <- Hv. exists fields.
      split; [ rewrite Hb; exact Hv | apply Forall2_eq_same ].
    - intros v v' <-. reflexivity.
    - apply Forall2_eq_same. }
  destruct (values_of_subkinds statics vs sks) eqn:H.
  - exact (Hone _ _ Hblock Hboxed H).
  - destruct (values_of_subkinds statics' vs sks) eqn:H'; [ | reflexivity ].
    rewrite
      (Hone _ _ (fun v => eq_sym (Hblock v)) (fun v => eq_sym (Hboxed v)) H')
      in H.
    discriminate H.
Qed.

(* ======================================================================== *)
(* The static environment grows                                             *)
(* ======================================================================== *)

(* What a subkind says of a value stays true where every symbol that
   was bound stays bound to the same thing. *)
Lemma values_of_subkinds_extend (statics statics' : static_env)
    (vs : list value) (sks : list param_subkind) :
  (∀ (s : symbol) (allocated : statically_allocated_value),
    lookup s (static_values statics) = Some allocated ->
    lookup s (static_values statics') = Some allocated) ->
  values_of_subkinds statics vs sks = true ->
  values_of_subkinds statics' vs sks = true.
Proof.
  intros Hextend.
  apply (values_of_subkinds_related_aux eq statics statics').
  - intros c v' <-. left. reflexivity.
  - intros v c ->. left. reflexivity.
  - intros v v' c <-.
    apply FlambdaPrimitiveProofs.boxed_number_of_statics_extend.
    intros s allocated _. apply Hextend.
  - intros v v' tag fields <- Hb. exists fields.
    split; [ | apply Forall2_eq_same ].
    apply (FlambdaPrimitiveProofs.block_of_statics_extend statics statics' v);
      [ | exact Hb ].
    intros s allocated _. apply Hextend.
  - intros v v' <-. reflexivity.
  - apply Forall2_eq_same.
Qed.
