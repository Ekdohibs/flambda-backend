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

(* Proofs of the statements of FlambdaPrimitiveSpec.v, with what the
   other proofs use of the primitives: the equations of [eval_prim] by
   arity, that the equality tests decide equality, and that the integers
   of the primitives are those of rocq/. Nothing here needs review.

   [eval_prim] is opaque after this file. A proof that is not about one
   primitive goes through the laws ([eval_prim_static_values],
   [eval_prim_statics_extend], [eval_prim_kinds],
   [eval_prim_preserves], [eval_prim_in_range], [eval_prim_related]);
   one that is rewrites with [eval_prim_unary], [eval_prim_binary] or
   [eval_prim_variadic].

   To add a primitive, add its case where a proof here goes by cases on
   the primitives: each such proof says so. No proof here names the
   constructors of [value] by their position, so one more does not break
   them. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList
  OxCamlInt OxCamlIntProofs OxCamlInt64.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaValueSpec FlambdaPrimitiveSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* The equality tests                                                       *)
(* ======================================================================== *)

Lemma kind_eqb_eq (k k' : flambda_kind) : kind_eqb k k' = true <-> k = k'.
Proof.
  unfold kind_eqb. destruct (kind_eq_dec k k'); split; congruence.
Qed.

Lemma kind_eqb_refl (k : flambda_kind) : kind_eqb k k = true.
Proof. apply kind_eqb_eq. reflexivity. Qed.

Lemma prim_eqb_sound (p p' : prim) : prim_eqb p p' = true -> p = p'.
Proof. unfold prim_eqb. destruct (prim_eq_dec p p'); congruence. Qed.

Lemma prim_eqb_refl (p : prim) : prim_eqb p p = true.
Proof. unfold prim_eqb. destruct (prim_eq_dec p p); congruence. Qed.

(* ======================================================================== *)
(* The equations of [eval_prim]                                             *)
(* ======================================================================== *)

Lemma eval_prim_unary (statics : static_env) (p : unary_primitive)
    (v : value) :
  eval_prim statics (Unary p) [v] = eval_unary_primitive statics p v.
Proof. reflexivity. Qed.

Lemma eval_prim_binary (statics : static_env) (p : binary_primitive)
    (v1 v2 : value) :
  eval_prim statics (Binary p) [v1; v2]
  = eval_binary_primitive statics p v1 v2.
Proof. reflexivity. Qed.

Lemma eval_prim_variadic (statics : static_env) (p : variadic_primitive)
    (vs : list value) :
  eval_prim statics (Variadic p) vs = eval_variadic_primitive statics p vs.
Proof. reflexivity. Qed.

(* A primitive that has a value was applied to as many arguments as it
   takes. (By cases on the arities.) *)
Lemma eval_prim_arity (statics : static_env) (p : prim) (vs : list value)
    (v : value) :
  eval_prim statics p vs = Some v ->
  (∃ (p' : unary_primitive) (v1 : value),
    p = Unary p' ∧ vs = [v1] ∧ eval_unary_primitive statics p' v1 = Some v)
  ∨ (∃ (p' : binary_primitive) (v1 v2 : value),
      p = Binary p' ∧ vs = [v1; v2]
      ∧ eval_binary_primitive statics p' v1 v2 = Some v)
  ∨ (∃ p' : variadic_primitive,
      p = Variadic p' ∧ eval_variadic_primitive statics p' vs = Some v).
Proof.
  destruct p as [p | p | p | p | rp].
  - destruct vs as [ | v1 [ | v2 vs]]; simpl; try discriminate.
    intros Heval. left. exists p, v1. auto.
  - destruct vs as [ | v1 [ | v2 [ | v3 vs]]]; simpl; try discriminate.
    intros Heval. right. left. exists p, v1, v2. auto.
  - rewrite eval_prim_variadic. intros Heval. right. right. exists p. auto.
  - destruct vs; discriminate.
  - destruct vs; discriminate.
Qed.

(* ======================================================================== *)
(* Integers and their kinds                                                 *)
(* ======================================================================== *)

(* A value that is an integer of a kind is the constant of that integer,
   in range. *)
Lemma int_of_value_const (kind : standard_int) (v : value) (i : Z) :
  int_of_value kind v = Some i ->
  v = Value_const (const_of_int kind i)
  ∧ in_range (int_width kind) i = true.
Proof.
  destruct kind, v; simpl; try discriminate;
    (match goal with c : const |- _ => destruct c end);
    simpl; try discriminate;
    match goal with
    | |- context [if ?test then _ else _] =>
      destruct test eqn:Hrange; [ | discriminate ]
    end;
    intros [= <-]; auto.
Qed.

Lemma kind_of_const_of_int (kind : standard_int) (i : Z) :
  kind_of_const (const_of_int kind i) = kind_of_standard_int kind.
Proof. destruct kind; reflexivity. Qed.

Lemma int_of_value_kind (kind : standard_int) (v : value) (i : Z) :
  int_of_value kind v = Some i ->
  kind_of_value v = kind_of_standard_int kind.
Proof.
  intros Hint. apply int_of_value_const in Hint as [-> _]. simpl.
  apply kind_of_const_of_int.
Qed.

Lemma is_tagged_immediate_kind (v : value) :
  is_tagged_immediate v = true -> kind_of_value v = Value.
Proof.
  unfold is_tagged_immediate.
  destruct (int_of_value Standard_tagged_immediate v) as [i | ] eqn:Hint;
    [ | discriminate ].
  intros _. exact (int_of_value_kind _ _ _ Hint).
Qed.

(* ======================================================================== *)
(* The observations                                                         *)
(* ======================================================================== *)

(* An allocated value is of kind [Value]: a constant is none of a closure,
   a block, a boxed number and a string. *)
Lemma block_of_kind (statics : static_env) (v : value) :
  is_some (block_of statics v) = true -> kind_of_value v = Value.
Proof. destruct v; simpl; try discriminate; reflexivity. Qed.

Lemma is_allocated_kind (statics : static_env) (v : value) :
  is_allocated statics v = true -> kind_of_value v = Value.
Proof. destruct v; simpl; try discriminate; reflexivity. Qed.

Lemma boxed_number_of_kind (statics : static_env) (v : value) (c : const) :
  boxed_number_of statics v = Some c -> kind_of_value v = Value.
Proof. destruct v; simpl; try discriminate; reflexivity. Qed.

Lemma string_of_kind (statics : static_env) (v : value) (bytes : list Z) :
  string_of statics v = Some bytes -> kind_of_value v = Value.
Proof. destruct v; simpl; try discriminate; reflexivity. Qed.

Lemma array_of_kind (statics : static_env) (v : value)
    (elements : list value) :
  array_of statics v = Some elements -> kind_of_value v = Value.
Proof. destruct v; simpl; try discriminate; reflexivity. Qed.

(* An immutable array is one made by a primitive, or what a symbol is
   bound to. *)
Lemma array_of_cases (statics : static_env) (v : value)
    (elements : list value) :
  array_of statics v = Some elements ->
  v = Value_immutable_array elements
  ∨ ∃ s : symbol,
      v = Value_symbol s
      ∧ lookup s (static_values statics)
        = Some (Static_immutable_array elements).
Proof.
  destruct v; simpl; try discriminate;
    try (intros [= <-]; left; reflexivity).
  match goal with
  | |- context [lookup ?s ?values] =>
    destruct (lookup s values) as [allocated | ] eqn:Hlookup;
      [ destruct allocated | ]
  end;
    try discriminate.
  intros [= <-]. right. eexists. split; [ reflexivity | exact Hlookup ].
Qed.

(* A block is one made by a primitive, or what a symbol is bound to. *)
Lemma block_of_cases (statics : static_env) (v : value) (tag : Z)
    (fields : list value) :
  block_of statics v = Some (tag, fields) ->
  v = Value_block tag fields
  ∨ ∃ s : symbol,
      v = Value_symbol s
      ∧ lookup s (static_values statics) = Some (Static_block tag fields).
Proof.
  destruct v; simpl; try discriminate.
  - match goal with
    | |- context [lookup ?s ?values] =>
      destruct (lookup s values) as [allocated | ] eqn:Hlookup;
        [ destruct allocated | ]
    end;
      try discriminate.
    intros [= <- <-]. right. eexists. split; [ reflexivity | exact Hlookup ].
  - intros [= <- <-]. left. reflexivity.
Qed.

Lemma is_number_of_kind_kind (kind : boxable_number) (c : const) :
  is_number_of_kind kind c = true -> kind_of_const c = unboxed_kind kind.
Proof. destruct kind, c; simpl; try discriminate; reflexivity. Qed.

(* An observation depends on the static environment only through what
   the value, if it is a symbol, is bound to. *)
Section Static_values.

Variables statics statics' : static_env.
Variable v : value.
Hypothesis Hsame :
  ∀ s : symbol,
    v = Value_symbol s ->
    lookup s (static_values statics) = lookup s (static_values statics').

Lemma closure_of_static_values : closure_of statics v = closure_of statics' v.
Proof.
  destruct v; simpl; try reflexivity. rewrite (Hsame _ eq_refl).
  reflexivity.
Qed.

Lemma block_of_static_values : block_of statics v = block_of statics' v.
Proof.
  destruct v; simpl; try reflexivity. rewrite (Hsame _ eq_refl).
  reflexivity.
Qed.

Lemma boxed_number_of_static_values :
  boxed_number_of statics v = boxed_number_of statics' v.
Proof.
  destruct v; simpl; try reflexivity. rewrite (Hsame _ eq_refl).
  reflexivity.
Qed.

Lemma string_of_static_values : string_of statics v = string_of statics' v.
Proof.
  destruct v; simpl; try reflexivity. rewrite (Hsame _ eq_refl).
  reflexivity.
Qed.

Lemma array_of_static_values : array_of statics v = array_of statics' v.
Proof.
  destruct v; simpl; try reflexivity. rewrite (Hsame _ eq_refl).
  reflexivity.
Qed.

Lemma is_allocated_static_values :
  is_allocated statics v = is_allocated statics' v.
Proof.
  unfold is_allocated.
  rewrite closure_of_static_values, block_of_static_values,
    boxed_number_of_static_values, string_of_static_values,
    array_of_static_values.
  reflexivity.
Qed.

Lemma bound_if_symbol_static_values :
  bound_if_symbol statics v = bound_if_symbol statics' v.
Proof.
  destruct v; simpl; try reflexivity. rewrite (Hsame _ eq_refl).
  reflexivity.
Qed.

Lemma imported_of_static_values :
  imported_of statics v = imported_of statics' v.
Proof.
  destruct v; simpl; try reflexivity. rewrite (Hsame _ eq_refl).
  reflexivity.
Qed.

(* By cases on the unary primitives: one that reads an allocated value
   does so through an observation. *)
Lemma eval_unary_primitive_static_values (p : unary_primitive) :
  eval_unary_primitive statics p v = eval_unary_primitive statics' p v.
Proof.
  destruct p as [[tag size field_kind] mut field | | | | | | | | | | | |
                | | ];
    simpl;
    unfold eval_is_int, eval_is_null, eval_block_load;
    rewrite ?block_of_static_values, ?is_allocated_static_values,
      ?boxed_number_of_static_values, ?string_of_static_values,
      ?array_of_static_values, ?bound_if_symbol_static_values;
    reflexivity.
Qed.

End Static_values.

(* An imported symbol is a symbol, bound to what is imported; and an
   allocated value is not one. *)
Lemma imported_of_symbol (statics : static_env) (v : value) (s : symbol)
    (c : import_class) :
  imported_of statics v = Some (s, c) ->
  v = Value_symbol s
  ∧ lookup s (static_values statics) = Some (Static_imported c).
Proof.
  destruct v as [ | | | s0 | | | | | ]; simpl; try discriminate.
  destruct (lookup s0 (static_values statics)) as [[ | | | | | | c0] | ]
    eqn:Hlookup; try discriminate.
  intros [= <- <-]. split; [ reflexivity | exact Hlookup ].
Qed.

Lemma is_allocated_not_imported (statics : static_env) (v : value) :
  is_allocated statics v = true -> imported_of statics v = None.
Proof.
  destruct v as [ | | | s | | | | | ]; try reflexivity.
  unfold is_allocated. simpl.
  destruct (lookup s (static_values statics)) as [[ | | | | | | c] | ];
    try (intros _; reflexivity).
  discriminate.
Qed.

(* An identity is an imported symbol of that class. *)
Lemma identity_of_imported (statics : static_env) (v : value) (s : symbol) :
  identity_of statics v = Some s
  <-> imported_of statics v = Some (s, Import_identity).
Proof.
  unfold identity_of.
  destruct (imported_of statics v) as [[s0 [ | | ]] | ]; split; intros H;
    try discriminate H; injection H as ->; reflexivity.
Qed.

(* What [Phys_equal] reads of a value that is not a tagged immediate
   goes from one value to another, when the second is allocated where
   the first is and is the same imported symbol where the first is
   one. *)
Section Phys_equal_transfer.

Context (statics statics' : static_env) (v v' : value).
Context
  (Hallocated :
    is_allocated statics v = true -> is_allocated statics' v' = true)
  (Himported :
    ∀ i : symbol * import_class,
      imported_of statics v = Some i -> imported_of statics' v' = Some i).

Lemma identity_of_transfer (s : symbol) :
  identity_of statics v = Some s -> identity_of statics' v' = Some s.
Proof.
  intros Hs. apply identity_of_imported. apply Himported.
  apply identity_of_imported. exact Hs.
Qed.

Lemma allocated_or_imported_transfer :
  is_allocated statics v || is_imported_pointer statics v = true ->
  is_allocated statics' v' || is_imported_pointer statics' v' = true.
Proof.
  intros H. apply orb_true_iff in H as [H | H].
  - rewrite (Hallocated H). reflexivity.
  - revert H. unfold is_imported_pointer.
    destruct (imported_of statics v) as [i | ] eqn:Hi; [ | discriminate ].
    rewrite (Himported i eq_refl). intros ->. apply orb_true_r.
Qed.

Lemma not_identity_transfer :
  identity_of statics v = None ->
  is_allocated statics v || is_imported_pointer statics v = true ->
  identity_of statics' v' = None
  ∧ is_allocated statics' v' || is_imported_pointer statics' v' = true.
Proof.
  intros Hnone H.
  split; [ | exact (allocated_or_imported_transfer H) ].
  apply orb_true_iff in H as [H | H].
  - unfold identity_of.
    rewrite (is_allocated_not_imported statics' v' (Hallocated H)).
    reflexivity.
  - revert H Hnone. unfold is_imported_pointer, identity_of.
    destruct (imported_of statics v) as [[s [ | | ]] | ] eqn:Hi;
      intros H Hnone; try discriminate.
    rewrite (Himported _ eq_refl). reflexivity.
Qed.

End Phys_equal_transfer.

(* [Phys_equal] gives of two values what it gives of two others, when
   these are the same tagged immediates, are allocated where those are,
   and are the same imported symbols where those are imported. *)
Lemma eval_phys_equal_transfer (statics statics' : static_env)
    (v1 v2 v1' v2' : value) (same : bool) :
  int_of_value Standard_tagged_immediate v1'
  = int_of_value Standard_tagged_immediate v1 ->
  int_of_value Standard_tagged_immediate v2'
  = int_of_value Standard_tagged_immediate v2 ->
  (is_allocated statics v1 = true -> is_allocated statics' v1' = true) ->
  (is_allocated statics v2 = true -> is_allocated statics' v2' = true) ->
  (∀ i : symbol * import_class,
    imported_of statics v1 = Some i -> imported_of statics' v1' = Some i) ->
  (∀ i : symbol * import_class,
    imported_of statics v2 = Some i -> imported_of statics' v2' = Some i) ->
  eval_phys_equal statics v1 v2 = Some same ->
  eval_phys_equal statics' v1' v2' = Some same.
Proof.
  intros Hint1 Hint2 Ha1 Ha2 Hi1 Hi2. unfold eval_phys_equal.
  rewrite Hint1, Hint2.
  destruct (int_of_value Standard_tagged_immediate v1),
    (int_of_value Standard_tagged_immediate v2).
  - exact (fun H => H).
  - destruct (is_allocated statics v2 || is_imported_pointer statics v2)
      eqn:H2; [ | discriminate ].
    rewrite (allocated_or_imported_transfer statics statics' v2 v2' Ha2 Hi2
               H2).
    exact (fun H => H).
  - destruct (is_allocated statics v1 || is_imported_pointer statics v1)
      eqn:H1; [ | discriminate ].
    rewrite (allocated_or_imported_transfer statics statics' v1 v1' Ha1 Hi1
               H1).
    exact (fun H => H).
  - destruct (identity_of statics v1) as [s1 | ] eqn:Hid1,
      (identity_of statics v2) as [s2 | ] eqn:Hid2.
    + rewrite (identity_of_transfer statics statics' v1 v1' Hi1 s1 Hid1),
        (identity_of_transfer statics statics' v2 v2' Hi2 s2 Hid2).
      exact (fun H => H).
    + rewrite (identity_of_transfer statics statics' v1 v1' Hi1 s1 Hid1).
      destruct (is_allocated statics v2 || is_imported_pointer statics v2)
        eqn:H2; [ | discriminate ].
      destruct
        (not_identity_transfer statics statics' v2 v2' Ha2 Hi2 Hid2 H2)
        as [-> ->].
      exact (fun H => H).
    + rewrite (identity_of_transfer statics statics' v2 v2' Hi2 s2 Hid2).
      destruct (is_allocated statics v1 || is_imported_pointer statics v1)
        eqn:H1; [ | discriminate ].
      destruct
        (not_identity_transfer statics statics' v1 v1' Ha1 Hi1 Hid1 H1)
        as [-> ->].
      exact (fun H => H).
    + discriminate.
Qed.

Lemma forallb_bound_if_symbol_static_values (statics statics' : static_env)
    (vs : list value) :
  (∀ s : symbol,
    Value_symbol s ∈ vs ->
    lookup s (static_values statics) = lookup s (static_values statics')) ->
  forallb (bound_if_symbol statics) vs
  = forallb (bound_if_symbol statics') vs.
Proof.
  induction vs as [ | v vs IH]; simpl; intros Hsame; [ reflexivity | ].
  rewrite IH by (intros s Hin; apply Hsame; right; exact Hin).
  rewrite (bound_if_symbol_static_values statics statics' v)
    by (intros s ->; apply Hsame; left; reflexivity).
  reflexivity.
Qed.

(* By cases on the binary primitives: [Array_load],
   [String_or_bigstring_load] and [Phys_equal] read an allocated value,
   and through an observation. *)
Lemma eval_binary_primitive_static_values (statics statics' : static_env)
    (p : binary_primitive) (v1 v2 : value) :
  (∀ s : symbol,
    Value_symbol s ∈ [v1; v2] ->
    lookup s (static_values statics) = lookup s (static_values statics')) ->
  eval_binary_primitive statics p v1 v2
  = eval_binary_primitive statics' p v1 v2.
Proof.
  intros Hsame.
  destruct p as [array_kind load_kind mut | width | op | kind op | kind op
                | kind behaviour];
    simpl; try reflexivity.
  - rewrite (array_of_static_values statics statics' v1)
      by (intros s ->; apply Hsame; left; reflexivity).
    reflexivity.
  - unfold eval_string_load.
    rewrite (string_of_static_values statics statics' v1)
      by (intros s ->; apply Hsame; left; reflexivity).
    reflexivity.
  - unfold eval_phys_equal, identity_of, is_imported_pointer.
    rewrite (is_allocated_static_values statics statics' v1)
      by (intros s ->; apply Hsame; left; reflexivity).
    rewrite (is_allocated_static_values statics statics' v2)
      by (intros s ->; apply Hsame; right; left; reflexivity).
    rewrite (imported_of_static_values statics statics' v1)
      by (intros s ->; apply Hsame; left; reflexivity).
    rewrite (imported_of_static_values statics statics' v2)
      by (intros s ->; apply Hsame; right; left; reflexivity).
    reflexivity.
Qed.

(* By cases on the variadic primitives. *)
Lemma eval_variadic_primitive_static_values (statics statics' : static_env)
    (p : variadic_primitive) (vs : list value) :
  (∀ s : symbol,
    Value_symbol s ∈ vs ->
    lookup s (static_values statics) = lookup s (static_values statics')) ->
  eval_variadic_primitive statics p vs
  = eval_variadic_primitive statics' p vs.
Proof.
  intros Hsame. destruct p as [[tag] | kind]; simpl.
  - rewrite (forallb_bound_if_symbol_static_values statics statics' vs Hsame).
    reflexivity.
  - rewrite (forallb_bound_if_symbol_static_values statics statics' vs Hsame).
    reflexivity.
Qed.

Theorem eval_prim_static_values : eval_prim_static_values_statement.
Proof.
  intros statics statics' p vs Hsame.
  destruct p as [p | p | p | p | rp];
    [ | | exact (eval_variadic_primitive_static_values _ _ p vs Hsame) | | ];
    destruct vs as [ | v1 [ | v2 [ | v3 vs]]]; simpl; try reflexivity.
  - apply eval_unary_primitive_static_values. intros s ->.
    apply Hsame. left. reflexivity.
  - exact (eval_binary_primitive_static_values _ _ p v1 v2 Hsame).
Qed.

(* A primitive does not look at the code. *)
Lemma eval_prim_same_static_values (statics statics' : static_env)
    (p : prim) (vs : list value) :
  static_values statics = static_values statics' ->
  eval_prim statics p vs = eval_prim statics' p vs.
Proof.
  intros Hvalues. apply eval_prim_static_values. intros s _.
  rewrite Hvalues. reflexivity.
Qed.

(* On constants, a primitive does not look at the static environment. *)
Lemma eval_prim_consts (statics statics' : static_env) (p : prim)
    (cs : list const) :
  eval_prim statics p (map Value_const cs)
  = eval_prim statics' p (map Value_const cs).
Proof.
  apply eval_prim_static_values. intros s Hin.
  apply in_map_iff in Hin as (c & Hc & _). discriminate.
Qed.

(* ======================================================================== *)
(* Values that are constants, and lists                                     *)
(* ======================================================================== *)

(* A value is a constant, or it is none and is of kind [Value]. *)
Lemma const_or_not (v : value) :
  (∃ c : const, v = Value_const c)
  ∨ ((∀ c : const, v <> Value_const c) ∧ kind_of_value v = Value).
Proof.
  destruct v; simpl;
    try (right; split; [ intros c'; discriminate | reflexivity ]).
  left. eauto.
Qed.

Lemma is_some_not_none {A : Type} (o : option A) :
  is_some o = true <-> o <> None.
Proof. destruct o; simpl; split; congruence. Qed.

Lemma forall2_nth_error {A B : Type} (R : A -> B -> Prop) (l : list A)
    (l' : list B) :
  Forall2 R l l' ->
  ∀ (n : nat) (a : A),
    nth_error l n = Some a -> ∃ b : B, nth_error l' n = Some b ∧ R a b.
Proof.
  induction 1 as [ | a0 b0 l l' Hhead _ IH]; intros [ | n] a; simpl;
    try discriminate.
  - intros [= <-]. exists b0. auto.
  - apply IH.
Qed.

Lemma forall2_forallb {A B : Type} (R : A -> B -> Prop) (f : A -> bool)
    (g : B -> bool) (l : list A) (l' : list B) :
  (∀ (a : A) (b : B), R a b -> f a = true -> g b = true) ->
  Forall2 R l l' ->
  forallb f l = true ->
  forallb g l' = true.
Proof.
  intros Hfg. induction 1 as [ | a b l l' Hhead _ IH]; simpl;
    [ reflexivity | ].
  intros Hall. apply andb_true_iff in Hall as [Ha Hl].
  rewrite (Hfg a b Hhead Ha), (IH Hl). reflexivity.
Qed.

(* ======================================================================== *)
(* Arrays                                                                   *)
(* ======================================================================== *)

Lemma replace_nth_length {A : Type} (n : nat) (x : A) (l l' : list A) :
  replace_nth n x l = Some l' -> length l' = length l.
Proof.
  revert l l'. induction n as [ | n IH]; intros [ | y l] l'; simpl;
    try discriminate.
  - intros [= <-]. reflexivity.
  - destruct (replace_nth n x l) as [l'' | ] eqn:Hreplace; simpl;
      [ | discriminate ].
    intros [= <-]. simpl. rewrite (IH _ _ Hreplace). reflexivity.
Qed.

Lemma replace_nth_in {A : Type} (n : nat) (x : A) (l l1 : list A) (y : A) :
  replace_nth n x l = Some l1 -> y ∈ l1 -> y = x ∨ y ∈ l.
Proof.
  revert n l1. induction l as [ | a l IH]; intros [ | n] l1; simpl;
    try discriminate.
  - intros [= <-] [<- | Hin];
      [ left; reflexivity | right; right; exact Hin ].
  - destruct (replace_nth n x l) as [l2 | ] eqn:Hreplace; simpl;
      [ | discriminate ].
    intros [= <-] [<- | Hin]; [ right; left; reflexivity | ].
    destruct (IH _ _ Hreplace Hin) as [-> | Hin'];
      [ left; reflexivity | right; right; exact Hin' ].
Qed.

Lemma Forall2_replace_nth {A B : Type} (R : A -> B -> Prop) (n : nat) (x : A)
    (y : B) (l l1 : list A) (l' : list B) :
  R x y ->
  Forall2 R l l' ->
  replace_nth n x l = Some l1 ->
  ∃ l1' : list B, replace_nth n y l' = Some l1' ∧ Forall2 R l1 l1'.
Proof.
  intros Hxy Hall. revert n l1.
  induction Hall as [ | a b l l' Hab Hall IH]; intros [ | n] l1; simpl;
    try discriminate.
  - intros [= <-]. eexists. split; [ reflexivity | constructor; assumption ].
  - destruct (replace_nth n x l) as [l2 | ] eqn:Hreplace; simpl;
      [ | discriminate ].
    intros [= <-]. destruct (IH _ _ Hreplace) as (l2' & -> & Hl2). simpl.
    eexists. split; [ reflexivity | constructor; assumption ].
Qed.

(* The element at the position replaced is the new one, and the others
   are as they were. *)
Lemma replace_nth_same {A : Type} (n : nat) (x : A) (l l' : list A) :
  replace_nth n x l = Some l' -> nth_error l' n = Some x.
Proof.
  revert l l'. induction n as [ | n IH]; intros [ | y l] l'; simpl;
    try discriminate.
  - intros [= <-]. reflexivity.
  - destruct (replace_nth n x l) as [l'' | ] eqn:Hreplace; simpl;
      [ | discriminate ].
    intros [= <-]. simpl. exact (IH _ _ Hreplace).
Qed.

Lemma replace_nth_other {A : Type} (n m : nat) (x : A) (l l' : list A) :
  replace_nth n x l = Some l' -> n <> m -> nth_error l' m = nth_error l m.
Proof.
  revert m l l'.
  induction n as [ | n IH]; intros [ | m] [ | y l] l'; simpl;
    try discriminate.
  - intros _ Hneq. contradiction Hneq. reflexivity.
  - intros [= <-] _. reflexivity.
  - destruct (replace_nth n x l) as [l'' | ] eqn:Hreplace; simpl;
      [ | discriminate ].
    intros [= <-] _. reflexivity.
  - destruct (replace_nth n x l) as [l'' | ] eqn:Hreplace; simpl;
      [ | discriminate ].
    intros [= <-] Hneq. simpl. apply (IH _ _ _ Hreplace). congruence.
Qed.

(* An index that is a position is the tagged immediate of that position,
   which the array has. *)
Lemma array_index_position (elements : list value) (index : value)
    (i : nat) :
  array_index elements index = Some i ->
  (i < length elements)%nat
  ∧ int_of_value Standard_tagged_immediate index = Some (Z.of_nat i).
Proof.
  unfold array_index.
  destruct (int_of_value Standard_tagged_immediate index) as [z | ];
    [ | discriminate ].
  cbv beta iota.
  destruct ((0 <=? z) && (z <? Z.of_nat (length elements))) eqn:Hbounds;
    [ | discriminate ].
  intros [= <-]. apply andb_true_iff in Hbounds as [Hlow Hhigh].
  apply Z.leb_le in Hlow. apply Z.ltb_lt in Hhigh.
  split; [ lia | ]. rewrite Z2Nat.id by exact Hlow. reflexivity.
Qed.

(* The position of an index depends on the number of elements only. *)
Lemma array_index_same_length (elements elements' : list value)
    (index : value) :
  length elements' = length elements ->
  array_index elements' index = array_index elements index.
Proof. intros Hlength. unfold array_index. rewrite Hlength. reflexivity. Qed.

(* The index of a load that has a value is a tagged immediate. *)
Lemma array_loaded_index_kind (load_kind : array_load_kind)
    (elements : list value) (index v : value) :
  array_loaded load_kind elements index = Some v ->
  kind_of_value index = Value.
Proof.
  unfold array_loaded.
  destruct (array_index elements index) as [i | ] eqn:Hindex;
    [ | discriminate ].
  intros _. apply array_index_position in Hindex as [_ Hindex].
  exact (int_of_value_kind _ _ _ Hindex).
Qed.

Theorem array_element_kind : array_element_kind_statement.
Proof.
  intros kind v. destruct kind; simpl;
    [ apply is_tagged_immediate_kind | apply kind_eqb_eq | apply kind_eqb_eq ].
Qed.

Theorem array_made_of_arguments : array_made_of_arguments_statement.
Proof.
  intros kind vs elements. unfold array_made.
  destruct (forallb (array_element_ok kind) vs) eqn:Hall; [ | discriminate ].
  intros [= <-]. split; [ reflexivity | ].
  intros v Hin. rewrite forallb_forall in Hall. exact (Hall v Hin).
Qed.

Theorem array_length_of_same_length : array_length_of_same_length_statement.
Proof.
  intros elements elements' Hlength. unfold array_length_of.
  rewrite Hlength. reflexivity.
Qed.

Theorem array_loaded_element : array_loaded_element_statement.
Proof.
  intros load_kind elements index v. unfold array_loaded.
  destruct (array_index elements index) as [i | ]; [ | discriminate ].
  cbv beta iota.
  destruct (nth_error elements i) as [w | ] eqn:Hnth; [ | discriminate ].
  cbv beta iota.
  destruct (array_element_ok (array_kind_of_load_kind load_kind) w) eqn:Hok;
    [ | discriminate ].
  intros [= <-]. split; [ exact (nth_error_In _ _ Hnth) | exact Hok ].
Qed.

(* A store that has a value: the two tests of the new value passed, the
   index is a position, and the elements are replaced there. *)
Lemma array_stored_cases (set_kind : array_set_kind) (kind : array_kind)
    (elements : list value) (index new : value) (stored : list value) :
  array_stored set_kind kind elements index new = Some stored ->
  array_element_ok (array_kind_of_set_kind set_kind) new = true
  ∧ array_element_ok kind new = true
  ∧ ∃ i : nat,
      array_index elements index = Some i
      ∧ replace_nth i new elements = Some stored.
Proof.
  unfold array_stored.
  destruct (array_element_ok (array_kind_of_set_kind set_kind) new);
    [ | discriminate ].
  destruct (array_element_ok kind new); [ | discriminate ].
  simpl.
  destruct (array_index elements index) as [i | ]; [ | discriminate ].
  cbv beta iota. intros Hstored. eauto.
Qed.

Theorem array_stored_elements : array_stored_elements_statement.
Proof.
  intros set_kind kind elements index new stored Hstored.
  apply array_stored_cases in Hstored as (_ & Hok & i & _ & Hreplace).
  split; [ exact (replace_nth_length _ _ _ _ Hreplace) | ].
  split; [ exact Hok | ].
  intros v Hin. exact (replace_nth_in _ _ _ _ _ Hreplace Hin).
Qed.

Theorem array_load_after_store : array_load_after_store_statement.
Proof.
  intros set_kind kind load_kind elements index new stored Hstored.
  apply array_stored_cases in Hstored as (_ & _ & i & Hindex & Hreplace).
  unfold array_loaded.
  rewrite (array_index_same_length elements stored index)
    by exact (replace_nth_length _ _ _ _ Hreplace).
  rewrite Hindex. cbv beta iota.
  rewrite (replace_nth_same _ _ _ _ Hreplace). reflexivity.
Qed.

Theorem array_load_after_other_store :
  array_load_after_other_store_statement.
Proof.
  intros set_kind kind load_kind elements index new other stored i j
    Hstored Hi Hj Hneq.
  apply array_stored_cases in Hstored as (_ & _ & n & Hindex & Hreplace).
  unfold array_loaded.
  rewrite (array_index_same_length elements stored other)
    by exact (replace_nth_length _ _ _ _ Hreplace).
  destruct (array_index elements other) as [m | ] eqn:Hother;
    [ | reflexivity ].
  cbv beta iota.
  rewrite (replace_nth_other n m new elements stored Hreplace);
    [ reflexivity | ].
  intros ->.
  apply array_index_position in Hindex as [_ Hindex].
  apply array_index_position in Hother as [_ Hother].
  congruence.
Qed.

(* A relation that relates a constant to itself only keeps the kind and
   the integers. *)
Lemma relates_constants_numbers (R : value -> value -> Prop) :
  relates_constants R -> relates_numbers R.
Proof.
  intros [Hconst_l Hconst_r] v v' Hr. split.
  - destruct (const_or_not v) as [[c ->] | [Hv Hkind]];
      [ rewrite (Hconst_l c v' Hr); reflexivity | ].
    destruct (const_or_not v') as [[c ->] | [_ Hkind']];
      [ rewrite (Hconst_r v c Hr); reflexivity | congruence ].
  - intros kind i Hint.
    destruct (int_of_value_const _ _ _ Hint) as [-> _].
    rewrite (Hconst_l _ v' Hr). exact Hint.
Qed.

Section Arrays_related.

Variable R : value -> value -> Prop.
Hypothesis Hnumbers : relates_numbers R.

(* A value that an array holds is related to values that it holds. *)
Lemma array_element_ok_related (kind : array_kind) (v v' : value) :
  R v v' -> array_element_ok kind v = true -> array_element_ok kind v' = true.
Proof.
  intros Hr. destruct (Hnumbers v v' Hr) as [Hkind Hint].
  destruct kind; simpl; try (rewrite Hkind; exact (fun H => H)).
  unfold is_tagged_immediate.
  destruct (int_of_value Standard_tagged_immediate v) as [i | ] eqn:Hi;
    [ | discriminate ].
  rewrite (Hint _ i Hi). exact (fun H => H).
Qed.

Lemma array_index_related (elements elements' : list value)
    (index index' : value) (i : nat) :
  Forall2 R elements elements' ->
  R index index' ->
  array_index elements index = Some i ->
  array_index elements' index' = Some i.
Proof.
  intros Helements Hr. unfold array_index.
  destruct (int_of_value Standard_tagged_immediate index) as [z | ] eqn:Hz;
    [ | discriminate ].
  rewrite (proj2 (Hnumbers index index' Hr) _ z Hz).
  rewrite (Forall2_length Helements). exact (fun H => H).
Qed.

Lemma array_made_related_in (kind : array_kind)
    (vs vs' elements : list value) :
  Forall2 R vs vs' ->
  array_made kind vs = Some elements ->
  ∃ elements' : list value,
    array_made kind vs' = Some elements' ∧ Forall2 R elements elements'.
Proof.
  intros Hvs. unfold array_made.
  destruct (forallb (array_element_ok kind) vs) eqn:Hall; [ | discriminate ].
  intros [= <-].
  rewrite
    (forall2_forallb R (array_element_ok kind) (array_element_ok kind) vs vs'
       (array_element_ok_related kind) Hvs Hall).
  exists vs'. split; [ reflexivity | exact Hvs ].
Qed.

Lemma array_loaded_related_in (load_kind : array_load_kind)
    (elements elements' : list value) (index index' v : value) :
  Forall2 R elements elements' ->
  R index index' ->
  array_loaded load_kind elements index = Some v ->
  ∃ v' : value, array_loaded load_kind elements' index' = Some v' ∧ R v v'.
Proof.
  intros Helements Hr. unfold array_loaded.
  destruct (array_index elements index) as [i | ] eqn:Hindex;
    [ | discriminate ].
  rewrite (array_index_related _ _ _ _ _ Helements Hr Hindex).
  cbv beta iota.
  destruct (nth_error elements i) as [w | ] eqn:Hnth; [ | discriminate ].
  destruct (forall2_nth_error R elements elements' Helements i w Hnth)
    as (w' & -> & Hw).
  cbv beta iota.
  destruct (array_element_ok (array_kind_of_load_kind load_kind) w) eqn:Hok;
    [ | discriminate ].
  rewrite (array_element_ok_related _ w w' Hw Hok).
  intros [= <-]. exists w'. auto.
Qed.

Lemma array_stored_related_in (set_kind : array_set_kind) (kind : array_kind)
    (elements elements' : list value) (index index' new new' : value)
    (stored : list value) :
  Forall2 R elements elements' ->
  R index index' ->
  R new new' ->
  array_stored set_kind kind elements index new = Some stored ->
  ∃ stored' : list value,
    array_stored set_kind kind elements' index' new' = Some stored'
    ∧ Forall2 R stored stored'.
Proof.
  intros Helements Hindex Hnew Hstored.
  apply array_stored_cases in Hstored as (Hset & Hok & i & Hi & Hreplace).
  unfold array_stored.
  rewrite (array_element_ok_related _ new new' Hnew Hset),
    (array_element_ok_related _ new new' Hnew Hok).
  simpl. rewrite (array_index_related _ _ _ _ _ Helements Hindex Hi).
  cbv beta iota.
  exact
    (Forall2_replace_nth R i new new' elements stored elements' Hnew
       Helements Hreplace).
Qed.

End Arrays_related.

Theorem array_made_related : array_made_related_statement.
Proof.
  intros R kind vs vs' elements Hnumbers.
  exact (array_made_related_in R Hnumbers kind vs vs' elements).
Qed.

Theorem array_loaded_related : array_loaded_related_statement.
Proof.
  intros R load_kind elements elements' index index' v Hnumbers.
  exact
    (array_loaded_related_in R Hnumbers load_kind elements elements' index
       index' v).
Qed.

Theorem array_stored_related : array_stored_related_statement.
Proof.
  intros R set_kind kind elements elements' index index' new new' stored
    Hnumbers.
  exact
    (array_stored_related_in R Hnumbers set_kind kind elements elements'
       index index' new new' stored).
Qed.

(* ======================================================================== *)
(* When the static environment grows                                        *)
(* ======================================================================== *)

(* An observation that has an answer keeps it when the symbol that the
   value is, if it is one, stays bound to the same thing. *)
Section Statics_extend.

Variables statics statics' : static_env.
Variable v : value.
Hypothesis Hextend :
  ∀ (s : symbol) (allocated : statically_allocated_value),
    v = Value_symbol s ->
    lookup s (static_values statics) = Some allocated ->
    lookup s (static_values statics') = Some allocated.

Lemma closure_of_statics_extend
    (closure : code_id * list (value_slot * value)) :
  closure_of statics v = Some closure -> closure_of statics' v = Some closure.
Proof.
  destruct v; simpl; try exact (fun H => H).
  match goal with
  | |- context [lookup ?s (static_values statics)] =>
    destruct (lookup s (static_values statics)) eqn:Hlookup;
      [ rewrite (Hextend _ _ eq_refl Hlookup); exact (fun H => H)
      | discriminate ]
  end.
Qed.

Lemma block_of_statics_extend (block : Z * list value) :
  block_of statics v = Some block -> block_of statics' v = Some block.
Proof.
  destruct v; simpl; try exact (fun H => H).
  match goal with
  | |- context [lookup ?s (static_values statics)] =>
    destruct (lookup s (static_values statics)) eqn:Hlookup;
      [ rewrite (Hextend _ _ eq_refl Hlookup); exact (fun H => H)
      | discriminate ]
  end.
Qed.

Lemma boxed_number_of_statics_extend (c : const) :
  boxed_number_of statics v = Some c -> boxed_number_of statics' v = Some c.
Proof.
  destruct v; simpl; try exact (fun H => H).
  match goal with
  | |- context [lookup ?s (static_values statics)] =>
    destruct (lookup s (static_values statics)) eqn:Hlookup;
      [ rewrite (Hextend _ _ eq_refl Hlookup); exact (fun H => H)
      | discriminate ]
  end.
Qed.

Lemma string_of_statics_extend (bytes : list Z) :
  string_of statics v = Some bytes -> string_of statics' v = Some bytes.
Proof.
  destruct v; simpl; try exact (fun H => H).
  match goal with
  | |- context [lookup ?s (static_values statics)] =>
    destruct (lookup s (static_values statics)) eqn:Hlookup;
      [ rewrite (Hextend _ _ eq_refl Hlookup); exact (fun H => H)
      | discriminate ]
  end.
Qed.

Lemma array_of_statics_extend (elements : list value) :
  array_of statics v = Some elements -> array_of statics' v = Some elements.
Proof.
  destruct v; simpl; try exact (fun H => H).
  match goal with
  | |- context [lookup ?s (static_values statics)] =>
    destruct (lookup s (static_values statics)) eqn:Hlookup;
      [ rewrite (Hextend _ _ eq_refl Hlookup); exact (fun H => H)
      | discriminate ]
  end.
Qed.

Lemma is_some_statics_extend {A : Type} (o o' : option A) :
  (∀ x : A, o = Some x -> o' = Some x) ->
  is_some o = true -> is_some o' = true.
Proof.
  intros Hkept. destruct o as [x | ]; [ | discriminate ].
  rewrite (Hkept x eq_refl). exact (fun H => H).
Qed.

Lemma is_allocated_statics_extend :
  is_allocated statics v = true -> is_allocated statics' v = true.
Proof.
  unfold is_allocated. rewrite !orb_true_iff.
  intros [[[[[Hsome | Hsome] | Hsome] | Hsome] | Hsome] | Hsome].
  - do 5 left. revert Hsome. apply is_some_statics_extend.
    exact closure_of_statics_extend.
  - do 4 left. right. revert Hsome. apply is_some_statics_extend.
    exact block_of_statics_extend.
  - do 3 left. right. revert Hsome. apply is_some_statics_extend.
    exact boxed_number_of_statics_extend.
  - do 2 left. right. revert Hsome. apply is_some_statics_extend.
    exact string_of_statics_extend.
  - left. right. revert Hsome. apply is_some_statics_extend.
    exact array_of_statics_extend.
  - right. exact Hsome.
Qed.

Lemma eval_is_int_statics_extend (variant_only b : bool) :
  eval_is_int statics variant_only v = Some b ->
  eval_is_int statics' variant_only v = Some b.
Proof.
  unfold eval_is_int.
  destruct (is_tagged_immediate v); [ exact (fun H => H) | ].
  destruct variant_only.
  - destruct (is_some (block_of statics v)) eqn:Hblock; [ | discriminate ].
    rewrite (is_some_statics_extend _ _ block_of_statics_extend Hblock).
    exact (fun H => H).
  - destruct (is_allocated statics v) eqn:Hallocated; [ | discriminate ].
    rewrite (is_allocated_statics_extend Hallocated). exact (fun H => H).
Qed.

Lemma eval_is_null_statics_extend (b : bool) :
  eval_is_null statics v = Some b -> eval_is_null statics' v = Some b.
Proof.
  unfold eval_is_null.
  destruct (is_tagged_immediate v); simpl.
  - exact (fun H => H).
  - destruct (is_allocated statics v) eqn:Hallocated.
    + rewrite (is_allocated_statics_extend Hallocated). exact (fun H => H).
    + destruct v as [[ ] | | | | | | | | ]; simpl; try discriminate;
        exact (fun H => H).
Qed.

Lemma eval_block_load_statics_extend (kind : block_access_kind) (field : Z)
    (loaded : value) :
  eval_block_load statics kind field v = Some loaded ->
  eval_block_load statics' kind field v = Some loaded.
Proof.
  unfold eval_block_load. destruct kind as [tag size field_kind].
  destruct (block_of statics v) as [block | ] eqn:Hblock; [ | discriminate ].
  rewrite (block_of_statics_extend _ Hblock). exact (fun H => H).
Qed.

Lemma bound_if_symbol_statics_extend :
  bound_if_symbol statics v = true -> bound_if_symbol statics' v = true.
Proof.
  destruct v; simpl; try exact (fun H => H).
  match goal with
  | |- context [lookup ?s (static_values statics)] =>
    destruct (lookup s (static_values statics)) eqn:Hlookup;
      [ rewrite (Hextend _ _ eq_refl Hlookup); exact (fun H => H)
      | discriminate ]
  end.
Qed.

Lemma imported_of_statics_extend (i : symbol * import_class) :
  imported_of statics v = Some i -> imported_of statics' v = Some i.
Proof.
  destruct v; simpl; try exact (fun H => H).
  match goal with
  | |- context [lookup ?s (static_values statics)] =>
    destruct (lookup s (static_values statics)) eqn:Hlookup;
      [ rewrite (Hextend _ _ eq_refl Hlookup); exact (fun H => H)
      | discriminate ]
  end.
Qed.

(* By cases on the unary primitives: one that reads an allocated value
   does so through an observation, and has no value where the
   observation has no answer. *)
Lemma eval_unary_primitive_statics_extend (p : unary_primitive)
    (result : value) :
  eval_unary_primitive statics p v = Some result ->
  eval_unary_primitive statics' p v = Some result.
Proof.
  destruct p; simpl; try exact (fun H => H);
    try apply eval_block_load_statics_extend;
    match goal with
    | |- context [eval_is_int statics ?variant_only v] =>
      destruct (eval_is_int statics variant_only v) eqn:Hvalue;
        [ rewrite (eval_is_int_statics_extend _ _ Hvalue) | discriminate ]
    | |- context [eval_is_null statics v] =>
      destruct (eval_is_null statics v) eqn:Hvalue;
        [ rewrite (eval_is_null_statics_extend _ Hvalue) | discriminate ]
    | |- context [block_of statics v] =>
      destruct (block_of statics v) eqn:Hvalue;
        [ rewrite (block_of_statics_extend _ Hvalue) | discriminate ]
    | |- context [boxed_number_of statics v] =>
      destruct (boxed_number_of statics v) eqn:Hvalue;
        [ rewrite (boxed_number_of_statics_extend _ Hvalue) | discriminate ]
    | |- context [string_of statics v] =>
      destruct (string_of statics v) eqn:Hvalue;
        [ rewrite (string_of_statics_extend _ Hvalue) | discriminate ]
    | |- context [array_of statics v] =>
      destruct (array_of statics v) eqn:Hvalue;
        [ rewrite (array_of_statics_extend _ Hvalue) | discriminate ]
    | |- context [bound_if_symbol statics v] =>
      destruct (bound_if_symbol statics v) eqn:Hvalue;
        [ rewrite (bound_if_symbol_statics_extend Hvalue)
        | rewrite andb_false_r; discriminate ]
    end;
    exact (fun H => H).
Qed.

End Statics_extend.

Lemma forallb_bound_if_symbol_statics_extend (statics statics' : static_env)
    (vs : list value) :
  (∀ (s : symbol) (allocated : statically_allocated_value),
    Value_symbol s ∈ vs ->
    lookup s (static_values statics) = Some allocated ->
    lookup s (static_values statics') = Some allocated) ->
  forallb (bound_if_symbol statics) vs = true ->
  forallb (bound_if_symbol statics') vs = true.
Proof.
  induction vs as [ | v vs IH]; simpl; intros Hextend;
    [ exact (fun H => H) | ].
  rewrite !andb_true_iff. intros [Hv Hvs]. split.
  - revert Hv. apply bound_if_symbol_statics_extend.
    intros s allocated ->. apply Hextend. left. reflexivity.
  - apply IH; [ | exact Hvs ].
    intros s allocated Hin. apply Hextend. right. exact Hin.
Qed.

(* By cases on the binary primitives: [Array_load] has a value on an
   immutable array, [String_or_bigstring_load] on a string, and
   [Phys_equal] on an allocated value by its being allocated, which
   are kept. *)
Lemma eval_binary_primitive_statics_extend (statics statics' : static_env)
    (p : binary_primitive) (v1 v2 result : value) :
  (∀ (s : symbol) (allocated : statically_allocated_value),
    Value_symbol s ∈ [v1; v2] ->
    lookup s (static_values statics) = Some allocated ->
    lookup s (static_values statics') = Some allocated) ->
  eval_binary_primitive statics p v1 v2 = Some result ->
  eval_binary_primitive statics' p v1 v2 = Some result.
Proof.
  intros Hextend.
  destruct p as [array_kind load_kind mut | width | op | kind op | kind op
                | kind behaviour];
    simpl; try exact (fun H => H).
  - (* [Array_load] *)
    destruct (array_of statics v1) as [elements | ] eqn:Harray;
      [ | discriminate ].
    rewrite (array_of_statics_extend statics statics' v1)
      with (elements := elements);
      [ exact (fun H => H) | | exact Harray ].
    intros s allocated ->. apply Hextend. left. reflexivity.
  - (* [String_or_bigstring_load] *)
    unfold eval_string_load.
    destruct (string_of statics v1) as [bytes | ] eqn:Hstring;
      [ | discriminate ].
    rewrite (string_of_statics_extend statics statics' v1)
      with (bytes := bytes);
      [ exact (fun H => H) | | exact Hstring ].
    intros s allocated ->. apply Hextend. left. reflexivity.
  - (* [Phys_equal] *)
    destruct (eval_phys_equal statics v1 v2) as [same | ] eqn:Hsame;
      [ | discriminate ].
    rewrite (eval_phys_equal_transfer statics statics' v1 v2 v1 v2 same)
      ; [ exact (fun H => H) | reflexivity | reflexivity | | | |
        | exact Hsame ].
    + apply is_allocated_statics_extend. intros s allocated ->.
      apply Hextend. left. reflexivity.
    + apply is_allocated_statics_extend. intros s allocated ->.
      apply Hextend. right. left. reflexivity.
    + intros i. apply imported_of_statics_extend. intros s0 allocated ->.
      apply Hextend. left. reflexivity.
    + intros i. apply imported_of_statics_extend. intros s0 allocated ->.
      apply Hextend. right. left. reflexivity.
Qed.

(* By cases on the variadic primitives. *)
Lemma eval_variadic_primitive_statics_extend
    (statics statics' : static_env) (p : variadic_primitive)
    (vs : list value) (result : value) :
  (∀ (s : symbol) (allocated : statically_allocated_value),
    Value_symbol s ∈ vs ->
    lookup s (static_values statics) = Some allocated ->
    lookup s (static_values statics') = Some allocated) ->
  eval_variadic_primitive statics p vs = Some result ->
  eval_variadic_primitive statics' p vs = Some result.
Proof.
  intros Hextend. destruct p as [[tag] | kind]; simpl.
  - destruct (forallb (fun v => kind_eqb (kind_of_value v) Value) vs);
      [ | discriminate ].
    destruct (forallb (bound_if_symbol statics) vs) eqn:Hbound;
      [ | discriminate ].
    rewrite
      (forallb_bound_if_symbol_statics_extend statics statics' vs Hextend
         Hbound).
    exact (fun H => H).
  - destruct (array_made kind vs) as [elements | ]; simpl;
      [ | discriminate ].
    destruct (forallb (bound_if_symbol statics) vs) eqn:Hbound;
      [ | discriminate ].
    rewrite
      (forallb_bound_if_symbol_statics_extend statics statics' vs Hextend
         Hbound).
    exact (fun H => H).
Qed.

Theorem eval_prim_statics_extend : eval_prim_statics_extend_statement.
Proof.
  intros statics statics' p vs v Hextend.
  destruct p as [p | p | p | p | rp];
    [ | | exact (eval_variadic_primitive_statics_extend _ _ p vs v Hextend)
    | | ];
    destruct vs as [ | v1 [ | v2 [ | v3 vs]]]; simpl;
    try exact (fun H => H).
  - apply eval_unary_primitive_statics_extend. intros s allocated ->.
    apply Hextend. left. reflexivity.
  - exact (eval_binary_primitive_statics_extend _ _ p v1 v2 v Hextend).
Qed.

(* ======================================================================== *)
(* No value on a symbol that is bound to nothing                            *)
(* ======================================================================== *)

Lemma lookup_bound {A : Type} (s : symbol) (statics : list (symbol * A))
    (x : A) :
  lookup s statics = Some x -> lookup s statics <> None.
Proof. intros ->. discriminate. Qed.

(* An observation has no answer on a symbol that is bound to nothing. *)
Lemma is_allocated_symbol_bound (statics : static_env) (s : symbol) :
  is_allocated statics (Value_symbol s) = true ->
  lookup s (static_values statics) <> None.
Proof.
  unfold is_allocated. simpl.
  destruct (lookup s (static_values statics)); discriminate.
Qed.

(* A symbol is not an integer. *)
Lemma int_of_value_symbol (kind : standard_int) (s : symbol) :
  int_of_value kind (Value_symbol s) = None.
Proof. destruct kind; reflexivity. Qed.

(* By cases on the unary primitives. *)
Lemma eval_unary_primitive_symbol_bound (statics : static_env)
    (p : unary_primitive) (s : symbol) (v : value) :
  eval_unary_primitive statics p (Value_symbol s) = Some v ->
  lookup s (static_values statics) <> None.
Proof.
  destruct p as [[tag size field_kind] mut field | [ | ] | | | | | kind
                | kind op | src dst | | kind | kind | | | [ | ]];
    simpl; unfold eval_is_int, eval_is_null, eval_block_load; simpl;
    rewrite ?int_of_value_symbol;
    try discriminate;
    try (destruct (lookup s (static_values statics)); discriminate).
  - destruct (is_allocated statics (Value_symbol s)) eqn:Hallocated;
      [ | discriminate ].
    intros _. exact (is_allocated_symbol_bound statics s Hallocated).
  - destruct (is_allocated statics (Value_symbol s)) eqn:Hallocated;
      [ | discriminate ].
    intros _. exact (is_allocated_symbol_bound statics s Hallocated).
  - destruct (lookup s (static_values statics)); [ discriminate | ].
    rewrite andb_false_r. discriminate.
Qed.

(* [Phys_equal] has a value on a symbol only if the symbol is
   allocated or imported, and so bound. *)
Lemma eval_phys_equal_symbol_bound (statics : static_env) (s : symbol)
    (v1 v2 : value) (same : bool) :
  eval_phys_equal statics v1 v2 = Some same ->
  v1 = Value_symbol s ∨ v2 = Value_symbol s ->
  lookup s (static_values statics) <> None.
Proof.
  intros Heval Hs.
  destruct (lookup s (static_values statics)) eqn:Hlookup; [ discriminate | ].
  exfalso.
  assert (Hallocated : is_allocated statics (Value_symbol s) = false).
  { unfold is_allocated. simpl. rewrite Hlookup. reflexivity. }
  assert (Himported : imported_of statics (Value_symbol s) = None).
  { simpl. rewrite Hlookup. reflexivity. }
  assert (Hidentity : identity_of statics (Value_symbol s) = None).
  { unfold identity_of. rewrite Himported. reflexivity. }
  revert Heval. unfold eval_phys_equal, is_imported_pointer.
  destruct Hs as [-> | ->];
    rewrite int_of_value_symbol, Hidentity, Hallocated, Himported.
  - destruct (int_of_value Standard_tagged_immediate v2); [ discriminate | ].
    destruct (identity_of statics v2); discriminate.
  - destruct (int_of_value Standard_tagged_immediate v1); [ discriminate | ].
    destruct (identity_of statics v1); discriminate.
Qed.

(* By cases on the binary primitives: a symbol is not an integer. *)
Lemma eval_binary_primitive_symbol_bound (statics : static_env)
    (p : binary_primitive) (s : symbol) (v1 v2 v : value) :
  eval_binary_primitive statics p v1 v2 = Some v ->
  v1 = Value_symbol s ∨ v2 = Value_symbol s ->
  lookup s (static_values statics) <> None.
Proof.
  destruct p as [array_kind load_kind mut | width | op | kind op | kind op
                | kind behaviour];
    cbn [eval_binary_primitive].
  - (* [Array_load]: the array, if a symbol, is bound to one, and the
       index is not a symbol. *)
    destruct (array_of statics v1) as [elements | ] eqn:Harray;
      [ | discriminate ].
    intros Hloaded [-> | ->].
    + simpl in Harray.
      destruct (lookup s (static_values statics)); discriminate.
    + exfalso. revert Hloaded. unfold array_loaded, array_index.
      rewrite int_of_value_symbol. simpl. discriminate.
  - (* [String_or_bigstring_load]: the string, if a symbol, is bound to
       one, and the index is not a symbol. *)
    unfold eval_string_load.
    destruct (string_of statics v1) as [bytes | ] eqn:Hstring;
      [ | discriminate ].
    destruct (int_of_value Standard_naked_nativeint v2) eqn:Hindex;
      [ | discriminate ].
    intros _ [-> | ->].
    + simpl in Hstring.
      destruct (lookup s (static_values statics)); discriminate.
    + rewrite int_of_value_symbol in Hindex. discriminate.
  - (* [Phys_equal] *)
    destruct (eval_phys_equal statics v1 v2) as [same | ] eqn:Hsame;
      [ | discriminate ].
    intros _. exact (eval_phys_equal_symbol_bound statics s v1 v2 same Hsame).
  - (* [Int_arith] *)
    destruct (int_of_value kind v1) eqn:Hv1; [ | discriminate ].
    destruct (int_of_value kind v2) eqn:Hv2; [ | discriminate ].
    intros _ [-> | ->]; rewrite int_of_value_symbol in *; discriminate.
  - (* [Int_shift] *)
    destruct (int_of_value kind v1) eqn:Hv1; [ | discriminate ].
    destruct (int_of_value Standard_naked_immediate v2) eqn:Hv2;
      [ | discriminate ].
    intros _ [-> | ->]; rewrite int_of_value_symbol in *; discriminate.
  - (* [Int_comp] *)
    destruct (int_of_value kind v1) eqn:Hv1; [ | discriminate ].
    destruct (int_of_value kind v2) eqn:Hv2; [ | discriminate ].
    intros _ [-> | ->]; rewrite int_of_value_symbol in *; discriminate.
Qed.

Theorem eval_prim_symbols_bound : eval_prim_symbols_bound_statement.
Proof.
  intros statics p vs v s Heval Hin.
  destruct (eval_prim_arity statics p vs v Heval)
    as [(p' & v1 & -> & -> & Hp)
       | [(p' & v1 & v2 & -> & -> & Hp) | (p' & -> & Hp)]].
  - destruct Hin as [-> | [ ]].
    exact (eval_unary_primitive_symbol_bound statics p' s v Hp).
  - apply (eval_binary_primitive_symbol_bound statics p' s v1 v2 v Hp).
    destruct Hin as [<- | [<- | [ ]]]; auto.
  - revert Hp. destruct p' as [[tag] | kind]; simpl.
    + destruct (forallb (fun v => kind_eqb (kind_of_value v) Value) vs);
        [ | discriminate ].
      destruct (forallb (bound_if_symbol statics) vs) eqn:Hbound;
        [ | discriminate ].
      intros _. rewrite forallb_forall in Hbound.
      specialize (Hbound _ Hin). simpl in Hbound.
      destruct (lookup s (static_values statics)); discriminate.
    + destruct (array_made kind vs) as [elements | ]; simpl;
        [ | discriminate ].
      destruct (forallb (bound_if_symbol statics) vs) eqn:Hbound;
        [ | discriminate ].
      intros _. rewrite forallb_forall in Hbound.
      specialize (Hbound _ Hin). simpl in Hbound.
      destruct (lookup s (static_values statics)); discriminate.
Qed.

(* ======================================================================== *)
(* Kinds                                                                    *)
(* ======================================================================== *)

(* By cases on the unary primitives. *)
Lemma eval_unary_primitive_kinds (statics : static_env)
    (p : unary_primitive) (arg v : value) :
  eval_unary_primitive statics p arg = Some v ->
  kind_of_value arg = arg_kind_of_unary_primitive p
  ∧ kind_of_value v = result_kind_of_unary_primitive p.
Proof.
  destruct p as [access mut field | variant_only | | | length_kind | | kind
                | kind op | src dst | | kind | kind | | | reinterpret];
    cbn [eval_unary_primitive arg_kind_of_unary_primitive
         result_kind_of_unary_primitive].
  - (* [Block_load] *)
    unfold eval_block_load. destruct access as [tag size field_kind].
    destruct (block_of statics arg) as [[block_tag fields] | ] eqn:Hblock;
      [ | discriminate ].
    assert (Harg : kind_of_value arg = Value)
      by (apply (block_of_kind statics); rewrite Hblock; reflexivity).
    destruct
      (known_is tag block_tag && known_is size (Z.of_nat (length fields))
       && (0 <=? field));
      [ | discriminate ].
    destruct (nth_error fields (Z.to_nat field)) as [loaded | ];
      [ | discriminate ].
    destruct field_kind.
    + destruct (kind_eqb (kind_of_value loaded) Value) eqn:Hkind;
        [ | discriminate ].
      intros [= <-]. split; [ exact Harg | apply kind_eqb_eq, Hkind ].
    + destruct (is_tagged_immediate loaded) eqn:Htagged; [ | discriminate ].
      intros [= <-]. split; [ exact Harg | ].
      exact (is_tagged_immediate_kind loaded Htagged).
  - (* [Is_int] *)
    unfold eval_is_int.
    destruct (is_tagged_immediate arg) eqn:Htagged.
    + intros [= <-]. split; [ | reflexivity ].
      exact (is_tagged_immediate_kind arg Htagged).
    + destruct variant_only.
      * destruct (is_some (block_of statics arg)) eqn:Hblock;
          [ | discriminate ].
        intros [= <-]. split; [ | reflexivity ].
        exact (block_of_kind statics arg Hblock).
      * destruct (is_allocated statics arg) eqn:Hallocated;
          [ | discriminate ].
        intros [= <-]. split; [ | reflexivity ].
        exact (is_allocated_kind statics arg Hallocated).
  - (* [Is_null] *)
    unfold eval_is_null.
    assert (Hother :
      option_map naked_bool
        (if is_tagged_immediate arg || is_allocated statics arg
         then Some false
         else None)
      = Some v ->
      kind_of_value arg = Value ∧ kind_of_value v = naked_immediate_kind).
    { destruct (is_tagged_immediate arg) eqn:Htagged; simpl.
      - intros [= <-]. split; [ | reflexivity ].
        exact (is_tagged_immediate_kind arg Htagged).
      - destruct (is_allocated statics arg) eqn:Hallocated;
          [ | discriminate ].
        intros [= <-]. split; [ | reflexivity ].
        exact (is_allocated_kind statics arg Hallocated). }
    destruct arg; try exact Hother.
    match goal with c : const |- _ => destruct c end; try exact Hother.
    intros [= <-]. split; reflexivity.
  - (* [Get_tag] *)
    destruct (block_of statics arg) as [[tag fields] | ] eqn:Hblock;
      [ | discriminate ].
    intros [= <-]. split; [ | reflexivity ].
    apply (block_of_kind statics). rewrite Hblock. reflexivity.
  - (* [Array_length] *)
    destruct (array_of statics arg) as [elements | ] eqn:Harray;
      [ | discriminate ].
    unfold array_length_of. destruct (in_range _ _); [ | discriminate ].
    intros [= <-]. split; [ | reflexivity ].
    exact (array_of_kind statics arg elements Harray).
  - (* [String_length] *)
    destruct (string_of statics arg) as [bytes | ] eqn:Hstring;
      [ | discriminate ].
    destruct (in_range _ _); [ | discriminate ].
    intros [= <-]. split; [ | reflexivity ].
    exact (string_of_kind statics arg bytes Hstring).
  - (* [Opaque_identity] *)
    destruct (kind_eqb (kind_of_value arg) kind) eqn:Hkind;
      [ | discriminate ].
    destruct (bound_if_symbol statics arg); [ | discriminate ].
    apply kind_eqb_eq in Hkind. intros [= <-]. auto.
  - (* [Unary_int_arith] *)
    destruct (int_of_value kind arg) as [a | ] eqn:Hint; [ | discriminate ].
    intros [= <-]. split.
    + exact (int_of_value_kind _ _ _ Hint).
    + apply kind_of_const_of_int.
  - (* [Num_conv] *)
    destruct (int_of_value src arg) as [i | ] eqn:Hint; [ | discriminate ].
    intros [= <-]. split.
    + exact (int_of_value_kind _ _ _ Hint).
    + apply kind_of_const_of_int.
  - (* [Boolean_not] *)
    destruct (int_of_value Standard_tagged_immediate arg) as [i | ] eqn:Hint;
      [ | discriminate ].
    pose proof (int_of_value_kind _ _ _ Hint) as Harg.
    destruct (i =? 0).
    + intros [= <-]. split; [ exact Harg | reflexivity ].
    + destruct (i =? 1); [ | discriminate ].
      intros [= <-]. split; [ exact Harg | reflexivity ].
  - (* [Unbox_number] *)
    destruct (boxed_number_of statics arg) as [c | ] eqn:Hboxed;
      [ | discriminate ].
    destruct (is_number_of_kind kind c) eqn:Hkind; [ | discriminate ].
    intros [= <-]. split.
    + exact (boxed_number_of_kind statics arg c Hboxed).
    + exact (is_number_of_kind_kind kind c Hkind).
  - (* [Box_number] *)
    destruct arg; try discriminate.
    match goal with
    | |- context [is_number_of_kind kind ?c] =>
      destruct (is_number_of_kind kind c) eqn:Hkind
    end;
      [ | discriminate ].
    intros [= <-]. split; [ | reflexivity ].
    exact (is_number_of_kind_kind kind _ Hkind).
  - (* [Untag_immediate] *)
    destruct (int_of_value Standard_tagged_immediate arg) as [i | ] eqn:Hint;
      [ | discriminate ].
    intros [= <-]. split; [ | reflexivity ].
    exact (int_of_value_kind _ _ _ Hint).
  - (* [Tag_immediate] *)
    destruct (int_of_value Standard_naked_immediate arg) as [i | ] eqn:Hint;
      [ | discriminate ].
    intros [= <-]. split; [ | reflexivity ].
    exact (int_of_value_kind _ _ _ Hint).
  - (* [Reinterpret_64_bit_word] *)
    destruct reinterpret; cbn [eval_reinterpret_64_bit_word];
      (destruct (int_of_value _ arg) as [i | ] eqn:Hint;
        [ | discriminate ]);
      intros [= <-]; (split; [ | reflexivity ]);
      exact (int_of_value_kind _ _ _ Hint).
Qed.

(* The arguments on which [Phys_equal] has a value are of kind
   [Value]. *)
Lemma imported_of_kind (statics : static_env) (v : value)
    (i : symbol * import_class) :
  imported_of statics v = Some i -> kind_of_value v = Value.
Proof. destruct v; simpl; try discriminate; reflexivity. Qed.

Lemma identity_of_kind (statics : static_env) (v : value) (s : symbol) :
  identity_of statics v = Some s -> kind_of_value v = Value.
Proof.
  intros Hs. apply identity_of_imported in Hs.
  exact (imported_of_kind statics v _ Hs).
Qed.

Lemma allocated_or_imported_kind (statics : static_env) (v : value) :
  is_allocated statics v || is_imported_pointer statics v = true ->
  kind_of_value v = Value.
Proof.
  intros H.
  apply orb_true_iff in H as [H | H];
    [ exact (is_allocated_kind statics v H) | ].
  unfold is_imported_pointer in H.
  destruct (imported_of statics v) as [i | ] eqn:Hi; [ | discriminate ].
  exact (imported_of_kind statics v i Hi).
Qed.

Lemma eval_phys_equal_kinds (statics : static_env) (arg1 arg2 : value)
    (same : bool) :
  eval_phys_equal statics arg1 arg2 = Some same ->
  kind_of_value arg1 = Value ∧ kind_of_value arg2 = Value.
Proof.
  unfold eval_phys_equal.
  destruct (int_of_value Standard_tagged_immediate arg1) as [a | ] eqn:Ha,
    (int_of_value Standard_tagged_immediate arg2) as [b | ] eqn:Hb.
  - intros _. split.
    + exact (int_of_value_kind _ _ _ Ha).
    + exact (int_of_value_kind _ _ _ Hb).
  - destruct (is_allocated statics arg2 || is_imported_pointer statics arg2)
      eqn:H2; [ | discriminate ].
    intros _. split.
    + exact (int_of_value_kind _ _ _ Ha).
    + exact (allocated_or_imported_kind statics arg2 H2).
  - destruct (is_allocated statics arg1 || is_imported_pointer statics arg1)
      eqn:H1; [ | discriminate ].
    intros _. split.
    + exact (allocated_or_imported_kind statics arg1 H1).
    + exact (int_of_value_kind _ _ _ Hb).
  - destruct (identity_of statics arg1) as [s1 | ] eqn:Hi1,
      (identity_of statics arg2) as [s2 | ] eqn:Hi2.
    + intros _. split.
      * exact (identity_of_kind _ _ _ Hi1).
      * exact (identity_of_kind _ _ _ Hi2).
    + destruct
        (is_allocated statics arg2 || is_imported_pointer statics arg2)
        eqn:H2; [ | discriminate ].
      intros _. split.
      * exact (identity_of_kind _ _ _ Hi1).
      * exact (allocated_or_imported_kind statics arg2 H2).
    + destruct
        (is_allocated statics arg1 || is_imported_pointer statics arg1)
        eqn:H1; [ | discriminate ].
      intros _. split.
      * exact (allocated_or_imported_kind statics arg1 H1).
      * exact (identity_of_kind _ _ _ Hi2).
    + discriminate.
Qed.

(* By cases on the binary primitives. *)
Lemma eval_binary_primitive_kinds (statics : static_env)
    (p : binary_primitive) (arg1 arg2 v : value) :
  eval_binary_primitive statics p arg1 arg2 = Some v ->
  kind_of_value arg1 = fst (args_kind_of_binary_primitive p)
  ∧ kind_of_value arg2 = snd (args_kind_of_binary_primitive p)
  ∧ kind_of_value v = result_kind_of_binary_primitive p.
Proof.
  destruct p as [array_kind load_kind mut | width | op | kind op | kind op
                | kind behaviour];
    cbn [eval_binary_primitive args_kind_of_binary_primitive
         result_kind_of_binary_primitive fst snd].
  - (* [Array_load]: the array is of kind [Value], the index a tagged
       immediate, and the element one that the load kind takes. *)
    destruct (array_of statics arg1) as [elements | ] eqn:Harray;
      [ | discriminate ].
    intros Hloaded.
    split; [ exact (array_of_kind statics arg1 elements Harray) | ].
    split; [ exact (array_loaded_index_kind _ _ _ _ Hloaded) | ].
    exact
      (array_element_kind _ v
         (proj2 (array_loaded_element _ _ _ _ Hloaded))).
  - (* [String_or_bigstring_load] *)
    unfold eval_string_load.
    destruct (string_of statics arg1) as [bytes | ] eqn:Hstring;
      [ | discriminate ].
    destruct (int_of_value Standard_naked_nativeint arg2) as [index | ]
      eqn:Hindex;
      [ | discriminate ].
    rewrite (string_of_kind statics arg1 bytes Hstring),
      (int_of_value_kind _ _ _ Hindex).
    cbv zeta. destruct (_ && _); [ | discriminate ].
    destruct width; intros [= <-]; auto.
  - (* [Phys_equal] *)
    destruct (eval_phys_equal statics arg1 arg2) as [same | ] eqn:Hsame;
      [ | discriminate ].
    destruct (eval_phys_equal_kinds statics arg1 arg2 same Hsame)
      as [-> ->].
    intros [= <-]. auto.
  - (* [Int_arith] *)
    destruct (int_of_value kind arg1) as [a | ] eqn:Ha; [ | discriminate ].
    destruct (int_of_value kind arg2) as [b | ] eqn:Hb; [ | discriminate ].
    rewrite (int_of_value_kind _ _ _ Ha), (int_of_value_kind _ _ _ Hb).
    destruct (eval_int_arith (int_width kind) op a b) as [result | ];
      [ | discriminate ].
    intros [= <-]. simpl. rewrite kind_of_const_of_int. auto.
  - (* [Int_shift] *)
    destruct (int_of_value kind arg1) as [a | ] eqn:Ha; [ | discriminate ].
    destruct (int_of_value Standard_naked_immediate arg2) as [amount | ]
      eqn:Hamount;
      [ | discriminate ].
    rewrite (int_of_value_kind _ _ _ Ha), (int_of_value_kind _ _ _ Hamount).
    destruct ((0 <=? amount) && (amount <? shift_bound kind));
      [ | discriminate ].
    intros [= <-]. simpl. rewrite kind_of_const_of_int. auto.
  - (* [Int_comp] *)
    destruct (int_of_value kind arg1) as [a | ] eqn:Ha; [ | discriminate ].
    destruct (int_of_value kind arg2) as [b | ] eqn:Hb; [ | discriminate ].
    rewrite (int_of_value_kind _ _ _ Ha), (int_of_value_kind _ _ _ Hb).
    intros [= <-]. auto.
Qed.

(* By cases on the variadic primitives. *)
Lemma eval_variadic_primitive_kinds (statics : static_env)
    (p : variadic_primitive) (args : list value) (v : value) :
  eval_variadic_primitive statics p args = Some v ->
  forallb (fun kind => kind_eqb kind (args_kind_of_variadic_primitive p))
    (map kind_of_value args)
  = true
  ∧ kind_of_value v = result_kind_of_variadic_primitive p.
Proof.
  destruct p as [[tag] | kind]; simpl.
  - destruct (forallb (fun v => kind_eqb (kind_of_value v) Value) args)
      eqn:Hargs;
      [ | discriminate ].
    destruct (forallb (bound_if_symbol statics) args); [ | discriminate ].
    intros [= <-]. split; [ | reflexivity ]. clear tag.
    induction args as [ | arg args IH]; simpl in *; [ reflexivity | ].
    apply andb_true_iff in Hargs as [Harg Hargs]. rewrite Harg.
    exact (IH Hargs).
  - destruct (array_made kind args) as [elements | ] eqn:Hmade; simpl;
      [ | discriminate ].
    destruct (forallb (bound_if_symbol statics) args); [ | discriminate ].
    intros [= <-]. split; [ | reflexivity ].
    destruct (array_made_of_arguments kind args elements Hmade) as [_ Hok].
    clear Hmade.
    induction args as [ | arg args IH]; simpl; [ reflexivity | ].
    rewrite (array_element_kind kind arg (Hok arg (or_introl eq_refl))).
    exact (IH (fun v Hv => Hok v (or_intror Hv))).
Qed.

Theorem eval_prim_kinds : eval_prim_kinds_statement.
Proof.
  intros statics p vs v Heval.
  destruct (eval_prim_arity statics p vs v Heval)
    as [(p' & v1 & -> & -> & Hp)
       | [(p' & v1 & v2 & -> & -> & Hp) | (p' & -> & Hp)]];
    simpl.
  - apply eval_unary_primitive_kinds in Hp as [-> ->].
    rewrite kind_eqb_refl. auto.
  - apply (eval_binary_primitive_kinds statics) in Hp as (-> & -> & ->).
    rewrite !kind_eqb_refl. auto.
  - exact (eval_variadic_primitive_kinds statics p' vs v Hp).
Qed.

(* ======================================================================== *)
(* A primitive makes its value out of its arguments                         *)
(* ======================================================================== *)

Section Preserves.

Variable P : value -> Prop.
Variable statics : static_env.
Hypothesis Hnumbers : holds_of_numbers P.
Hypothesis Hblocks : holds_of_blocks P.
Hypothesis Helements :
  ∀ elements : list value,
    P (Value_immutable_array elements) -> ∀ v : value, v ∈ elements -> P v.
Hypothesis Hstatic : holds_in_static_blocks statics P.
Hypothesis Harrays : holds_in_static_arrays statics P.

Lemma naked_bool_preserved (b : bool) : P (naked_bool b).
Proof. apply Hnumbers. Qed.

(* The elements of an immutable array with the property have it. *)
Lemma array_of_elements_preserved (arg : value) (elements : list value) :
  P arg ->
  array_of statics arg = Some elements ->
  ∀ v : value, v ∈ elements -> P v.
Proof.
  intros Harg Harray.
  destruct (array_of_cases statics arg elements Harray)
    as [-> | (s & -> & Hlookup)].
  - exact (Helements elements Harg).
  - exact (Harrays s elements Harg Hlookup).
Qed.

(* The fields of a block with the property have it, whether the block
   was made by a primitive or is what a symbol is bound to. *)
Lemma block_of_fields_preserved (arg : value) (tag : Z)
    (fields : list value) :
  P arg ->
  block_of statics arg = Some (tag, fields) ->
  ∀ v : value, v ∈ fields -> P v.
Proof.
  intros Harg Hblock.
  destruct (block_of_cases statics arg tag fields Hblock)
    as [-> | (s & -> & Hlookup)].
  - exact (proj1 (Hblocks tag fields) Harg).
  - exact (Hstatic s tag fields Harg Hlookup).
Qed.

(* By cases on the unary primitives. *)
Lemma eval_unary_primitive_preserves (p : unary_primitive) (arg v : value) :
  P arg -> eval_unary_primitive statics p arg = Some v -> P v.
Proof.
  intros Harg.
  destruct p as [access mut field | variant_only | | | length_kind | | kind
                | kind op | src dst | | kind | kind | | | reinterpret];
    cbn [eval_unary_primitive].
  - unfold eval_block_load. destruct access as [tag size field_kind].
    destruct (block_of statics arg) as [[block_tag fields] | ] eqn:Hblock;
      [ | discriminate ].
    destruct
      (known_is tag block_tag && known_is size (Z.of_nat (length fields))
       && (0 <=? field));
      [ | discriminate ].
    destruct (nth_error fields (Z.to_nat field)) as [loaded | ] eqn:Hnth;
      [ | discriminate ].
    pose proof
      (block_of_fields_preserved arg block_tag fields Harg Hblock loaded
         (nth_error_In _ _ Hnth))
      as Hloaded.
    destruct
      (match field_kind with
       | Any_value => kind_eqb (kind_of_value loaded) Value
       | Immediate => is_tagged_immediate loaded
       end);
      [ | discriminate ].
    intros [= <-]. exact Hloaded.
  - destruct (eval_is_int statics variant_only arg); [ | discriminate ].
    intros [= <-]. apply naked_bool_preserved.
  - destruct (eval_is_null statics arg); [ | discriminate ].
    intros [= <-]. apply naked_bool_preserved.
  - destruct (block_of statics arg) as [[tag fields] | ]; [ | discriminate ].
    intros [= <-]. apply Hnumbers.
  - (* [Array_length] *)
    destruct (array_of statics arg) as [elements | ]; [ | discriminate ].
    unfold array_length_of. destruct (in_range _ _); [ | discriminate ].
    intros [= <-]. apply Hnumbers.
  - destruct (string_of statics arg) as [bytes | ]; [ | discriminate ].
    destruct (in_range _ _); [ | discriminate ].
    intros [= <-]. apply Hnumbers.
  - destruct (kind_eqb (kind_of_value arg) kind); [ | discriminate ].
    destruct (bound_if_symbol statics arg); [ | discriminate ].
    intros [= <-]. exact Harg.
  - destruct (int_of_value kind arg); [ | discriminate ].
    intros [= <-]. apply Hnumbers.
  - destruct (int_of_value src arg); [ | discriminate ].
    intros [= <-]. apply Hnumbers.
  - destruct (int_of_value Standard_tagged_immediate arg) as [i | ];
      [ | discriminate ].
    destruct (i =? 0); [ intros [= <-]; apply Hnumbers | ].
    destruct (i =? 1); [ | discriminate ].
    intros [= <-]. apply Hnumbers.
  - destruct (boxed_number_of statics arg) as [c | ]; [ | discriminate ].
    destruct (is_number_of_kind kind c); [ | discriminate ].
    intros [= <-]. apply Hnumbers.
  - destruct arg; try discriminate.
    match goal with
    | |- context [is_number_of_kind kind ?c] =>
      destruct (is_number_of_kind kind c)
    end;
      [ | discriminate ].
    intros [= <-]. apply Hnumbers.
  - destruct (int_of_value Standard_tagged_immediate arg); [ | discriminate ].
    intros [= <-]. apply Hnumbers.
  - destruct (int_of_value Standard_naked_immediate arg); [ | discriminate ].
    intros [= <-]. apply Hnumbers.
  - destruct reinterpret; cbn [eval_reinterpret_64_bit_word];
      (destruct (int_of_value _ arg); [ | discriminate ]);
      intros [= <-]; apply Hnumbers.
Qed.

(* By cases on the binary primitives. *)
Lemma eval_binary_primitive_preserves (p : binary_primitive)
    (arg1 arg2 v : value) :
  P arg1 -> eval_binary_primitive statics p arg1 arg2 = Some v -> P v.
Proof.
  intros Harg1.
  destruct p as [array_kind load_kind mut | width | op | kind op | kind op
                | kind behaviour];
    cbn [eval_binary_primitive].
  - (* [Array_load]: an element of the array. *)
    destruct (array_of statics arg1) as [elements | ] eqn:Harray;
      [ | discriminate ].
    intros Hloaded.
    exact
      (array_of_elements_preserved arg1 elements Harg1 Harray v
         (proj1 (array_loaded_element _ _ _ _ Hloaded))).
  - (* [String_or_bigstring_load] *)
    unfold eval_string_load.
    destruct (string_of statics arg1) as [bytes | ]; [ | discriminate ].
    destruct (int_of_value Standard_naked_nativeint arg2) as [index | ];
      [ | discriminate ].
    cbv zeta. destruct (_ && _); [ | discriminate ].
    destruct width; intros [= <-]; apply Hnumbers.
  - (* [Phys_equal] *)
    destruct (eval_phys_equal statics arg1 arg2); [ | discriminate ].
    intros [= <-]. apply naked_bool_preserved.
  - (* [Int_arith] *)
    destruct (int_of_value kind arg1) as [a | ]; [ | discriminate ].
    destruct (int_of_value kind arg2) as [b | ]; [ | discriminate ].
    destruct (eval_int_arith (int_width kind) op a b); [ | discriminate ].
    intros [= <-]. apply Hnumbers.
  - (* [Int_shift] *)
    destruct (int_of_value kind arg1) as [a | ]; [ | discriminate ].
    destruct (int_of_value Standard_naked_immediate arg2) as [amount | ];
      [ | discriminate ].
    destruct ((0 <=? amount) && (amount <? shift_bound kind));
      [ | discriminate ].
    intros [= <-]. apply Hnumbers.
  - (* [Int_comp] *)
    destruct (int_of_value kind arg1) as [a | ]; [ | discriminate ].
    destruct (int_of_value kind arg2) as [b | ]; [ | discriminate ].
    intros [= <-]. apply Hnumbers.
Qed.

(* By cases on the variadic primitives. That the property holds of an
   array of which it holds of the elements is asked of the primitive
   that makes one only. *)
Lemma eval_variadic_primitive_preserves (p : variadic_primitive)
    (args : list value) (v : value) :
  (∀ kind : array_kind,
    p = Make_immutable_array kind ->
    (∀ arg : value, arg ∈ args -> P arg) ->
    P (Value_immutable_array args)) ->
  (∀ arg : value, arg ∈ args -> P arg) ->
  eval_variadic_primitive statics p args = Some v ->
  P v.
Proof.
  intros Hmade_array Hargs. destruct p as [[tag] | kind]; simpl.
  - destruct (forallb (fun v => kind_eqb (kind_of_value v) Value) args);
      [ | discriminate ].
    destruct (forallb (bound_if_symbol statics) args); [ | discriminate ].
    intros [= <-]. exact (proj2 (Hblocks tag args) Hargs).
  - destruct (array_made kind args) as [elements | ] eqn:Hmade; simpl;
      [ | discriminate ].
    destruct (forallb (bound_if_symbol statics) args); [ | discriminate ].
    intros [= <-].
    destruct (array_made_of_arguments kind args elements Hmade) as [-> _].
    exact (Hmade_array kind eq_refl Hargs).
Qed.

End Preserves.

(* The law with what it asks of an array value in its two directions:
   from the array to its elements, of every primitive; from the
   elements to the array, of the primitive that makes one. *)
Lemma eval_prim_preserves_directed (P : value -> Prop)
    (statics : static_env) (p : prim) (vs : list value) (v : value) :
  holds_of_numbers P ->
  holds_of_blocks P ->
  (∀ elements : list value,
    P (Value_immutable_array elements) ->
    ∀ w : value, w ∈ elements -> P w) ->
  (∀ kind : array_kind,
    p = Variadic (Make_immutable_array kind) ->
    (∀ arg : value, arg ∈ vs -> P arg) ->
    P (Value_immutable_array vs)) ->
  holds_in_static_blocks statics P ->
  holds_in_static_arrays statics P ->
  (∀ arg : value, arg ∈ vs -> P arg) ->
  eval_prim statics p vs = Some v ->
  P v.
Proof.
  intros Hnumbers Hblocks Helements Hmade Hstatic Harrays Hargs Heval.
  destruct (eval_prim_arity statics p vs v Heval)
    as [(p' & v1 & -> & -> & Hp)
       | [(p' & v1 & v2 & -> & -> & Hp) | (p' & -> & Hp)]].
  - apply
      (eval_unary_primitive_preserves P statics Hnumbers Hblocks Hstatic p'
         v1 v);
      [ | exact Hp ].
    apply Hargs. left. reflexivity.
  - apply
      (eval_binary_primitive_preserves P statics Hnumbers Helements Harrays
         p' v1 v2 v);
      [ | exact Hp ].
    apply Hargs. left. reflexivity.
  - apply (eval_variadic_primitive_preserves P statics Hblocks p' vs v);
      [ | exact Hargs | exact Hp ].
    intros kind ->. exact (Hmade kind eq_refl).
Qed.

Theorem eval_prim_preserves : eval_prim_preserves_statement.
Proof.
  intros P statics p vs v Hnumbers Hblocks Hmade.
  apply (eval_prim_preserves_directed P statics p vs v Hnumbers Hblocks).
  - intros elements. exact (proj1 (Hmade elements)).
  - intros kind _. exact (proj2 (Hmade vs)).
Qed.

(* Of a primitive that makes no array, the law asks one direction only:
   that the property holds of the elements of an array of which it
   holds. So a property that holds of no array value at all, such as
   to have none inside (FlambdaNoArrayValueSpec.no_array_value), is
   kept by every other primitive, though it is not one that holds of
   an array exactly when it holds of its elements. *)
Lemma eval_prim_preserves_making_no_array (P : value -> Prop)
    (statics : static_env) (p : prim) (vs : list value) (v : value) :
  (∀ kind : array_kind, p <> Variadic (Make_immutable_array kind)) ->
  holds_of_numbers P ->
  holds_of_blocks P ->
  (∀ elements : list value,
    P (Value_immutable_array elements) ->
    ∀ w : value, w ∈ elements -> P w) ->
  holds_in_static_blocks statics P ->
  holds_in_static_arrays statics P ->
  (∀ arg : value, arg ∈ vs -> P arg) ->
  eval_prim statics p vs = Some v ->
  P v.
Proof.
  intros Hp Hnumbers Hblocks Helements.
  apply
    (eval_prim_preserves_directed P statics p vs v Hnumbers Hblocks
       Helements).
  intros kind Heq. destruct (Hp kind Heq).
Qed.

(* ======================================================================== *)
(* The integers are those of rocq/                                          *)
(* ======================================================================== *)

(* What wraps is in range, so the results of arithmetic are. *)
Lemma wrap_in_range (width z : Z) :
  1 <= width -> in_range width (wrap width z) = true.
Proof.
  intros Hwidth. unfold in_range, wrap.
  assert (Hhalf : 0 < 2 ^ (width - 1)) by (apply Z.pow_pos_nonneg; lia).
  assert (Hdouble : 2 ^ width = 2 * 2 ^ (width - 1)).
  { rewrite <- Z.pow_succ_r by lia. f_equal. lia. }
  pose proof (Z.mod_pos_bound (z + 2 ^ (width - 1)) (2 ^ width)) as Hmod.
  apply andb_true_iff. split; [ apply Z.leb_le | apply Z.ltb_lt ]; lia.
Qed.

Lemma int_width_positive (kind : standard_int) : 1 <= int_width kind.
Proof. destruct kind; simpl; lia. Qed.

(* An integer that is in range is not changed by wrapping, so [Num_conv]
   to a kind that holds the integer is the identity on it. *)
Lemma wrap_of_in_range (width z : Z) :
  1 <= width -> in_range width z = true -> wrap width z = z.
Proof.
  intros Hwidth Hrange. unfold in_range in Hrange. unfold wrap.
  apply andb_true_iff in Hrange as [Hlow Hhigh].
  apply Z.leb_le in Hlow. apply Z.ltb_lt in Hhigh.
  assert (Hdouble : 2 ^ width = 2 * 2 ^ (width - 1)).
  { rewrite <- Z.pow_succ_r by lia. f_equal. lia. }
  rewrite Z.mod_small by lia. lia.
Qed.

(* The integers of a width are among those of a greater one. *)
Lemma in_range_wider (width width' z : Z) :
  1 <= width <= width' ->
  in_range width z = true -> in_range width' z = true.
Proof.
  intros Hwidths Hrange. unfold in_range in *.
  apply andb_true_iff in Hrange as [Hlow Hhigh].
  apply Z.leb_le in Hlow. apply Z.ltb_lt in Hhigh.
  assert (Hpow : 2 ^ (width - 1) <= 2 ^ (width' - 1))
    by (apply Z.pow_le_mono_r; lia).
  apply andb_true_iff. split; [ apply Z.leb_le | apply Z.ltb_lt ]; lia.
Qed.

(* [Num_conv] wraps, and its result is in the range of [dst]. *)
Lemma eval_num_conv (statics : static_env) (src dst : standard_int)
    (v : value) (i : Z) :
  int_of_value src v = Some i ->
  eval_prim statics (Unary (Num_conv src dst)) [v]
  = Some (Value_const (const_of_int dst (wrap (int_width dst) i)))
  ∧ in_range (int_width dst) (wrap (int_width dst) i) = true.
Proof.
  intros Hint. rewrite eval_prim_unary. simpl. rewrite Hint.
  split; [ reflexivity | apply wrap_in_range, int_width_positive ].
Qed.

(* To a kind that is at least as wide, [Num_conv] keeps the integer. *)
Lemma eval_num_conv_wider (statics : static_env) (src dst : standard_int)
    (v : value) (i : Z) :
  int_width src <= int_width dst ->
  int_of_value src v = Some i ->
  eval_prim statics (Unary (Num_conv src dst)) [v]
  = Some (Value_const (const_of_int dst i)).
Proof.
  intros Hwidths Hint.
  destruct (eval_num_conv statics src dst v i Hint) as [-> _].
  apply int_of_value_const in Hint as [_ Hrange].
  rewrite wrap_of_in_range; [ reflexivity | apply int_width_positive | ].
  apply (in_range_wider (int_width src)); [ | exact Hrange ].
  split; [ apply int_width_positive | exact Hwidths ].
Qed.

Lemma eval_int_arith_in_range (width : Z) (op : binary_int_arith_op)
    (a b result : Z) :
  1 <= width ->
  eval_int_arith width op a b = Some result ->
  in_range width result = true.
Proof.
  intros Hwidth.
  destruct op; simpl; try destruct (b =? 0); try discriminate;
    intros [= <-]; apply wrap_in_range, Hwidth.
Qed.

(* The integer that some bytes make is at least 0, and below the first
   that needs one more byte. *)
Lemma little_endian_bound (bytes : list Z) :
  0 <= little_endian bytes < 256 ^ Z.of_nat (length bytes).
Proof.
  induction bytes as [ | byte rest IH]; [ simpl; lia | ].
  cbn [little_endian length].
  rewrite Nat2Z.inj_succ, Z.pow_succ_r by lia.
  pose proof (Z.mod_pos_bound byte 256 eq_refl) as Hbyte. lia.
Qed.

Lemma length_bytes_of (count : nat) (z : Z) :
  length (bytes_of count z) = count.
Proof.
  revert z. induction count as [ | count IH]; intros z; simpl;
    [ reflexivity | rewrite IH; reflexivity ].
Qed.

(* The swap of [count] bytes is an integer of [count] bytes without a
   sign. *)
Lemma swap_bytes_bound (count : nat) (z : Z) :
  0 <= swap_bytes count z < 256 ^ Z.of_nat count.
Proof.
  unfold swap_bytes.
  pose proof (little_endian_bound (rev (bytes_of count z))) as Hbound.
  rewrite length_rev, length_bytes_of in Hbound. exact Hbound.
Qed.

(* The result of the unary arithmetic is in range. *)
Lemma eval_unary_int_arith_in_range (kind : standard_int)
    (op : unary_int_arith_op) (a : Z) :
  in_range (int_width kind) (eval_unary_int_arith kind op a) = true.
Proof.
  destruct op; cbn [eval_unary_int_arith].
  - apply wrap_in_range, int_width_positive.
  - destruct kind; cbn [int_width]; try (apply wrap_in_range; lia);
      pose proof (swap_bytes_bound 2 a) as Hswap;
      change (256 ^ Z.of_nat 2) with 65536 in Hswap;
      unfold in_range; change (2 ^ (63 - 1)) with 4611686018427387904;
      apply andb_true_iff; (split; [ apply Z.leb_le | apply Z.ltb_lt ]);
      lia.
Qed.

(* The complement of an integer in range is in range, so the wrap of
   [Not] changes nothing. *)
Lemma eval_unary_int_arith_not (kind : standard_int) (a : Z) :
  in_range (int_width kind) a = true ->
  eval_unary_int_arith kind Not a = Z.lnot a.
Proof.
  intros Hrange. cbn [eval_unary_int_arith].
  pose proof (int_width_positive kind) as Hwidth.
  apply wrap_of_in_range; [ exact Hwidth | ].
  unfold in_range in *. unfold Z.lnot.
  apply andb_true_iff in Hrange as [Hlow Hhigh].
  apply Z.leb_le in Hlow. apply Z.ltb_lt in Hhigh.
  apply andb_true_iff. split; [ apply Z.leb_le | apply Z.ltb_lt ]; lia.
Qed.

(* An integer of one byte is its own swap: the wrap changes nothing. *)
Lemma eval_unary_int_arith_swap_int8 (a : Z) :
  in_range 8 a = true ->
  eval_unary_int_arith Standard_naked_int8 Swap_byte_endianness a = a.
Proof. intros Hrange. apply wrap_of_in_range; [ lia | exact Hrange ]. Qed.

(* A swap done twice. The bytes of an integer make its low bytes
   again. *)
Lemma little_endian_bytes_of (count : nat) (z : Z) :
  little_endian (bytes_of count z) = z mod 256 ^ Z.of_nat count.
Proof.
  revert z. induction count as [ | count IH]; intros z.
  - simpl. symmetry. apply Z.mod_1_r.
  - cbn [bytes_of little_endian].
    rewrite IH, Z.mod_mod by lia.
    rewrite Nat2Z.inj_succ, Z.pow_succ_r by lia.
    symmetry. apply Z.rem_mul_r; [ lia | ]. apply Z.pow_pos_nonneg; lia.
Qed.

(* The bytes of the integer that some bytes make are those bytes. *)
Lemma bytes_of_little_endian (bytes : list Z) :
  bytes_of (length bytes) (little_endian bytes)
  = map (fun byte => byte mod 256) bytes.
Proof.
  induction bytes as [ | byte rest IH]; [ reflexivity | ].
  cbn [length bytes_of little_endian map].
  pose proof (Z.mod_pos_bound byte 256 eq_refl) as Hbyte.
  rewrite (Z.mul_comm 256 (little_endian rest)).
  rewrite Z_mod_plus_full, Z.div_add by lia.
  rewrite Z.mod_mod, Z.div_small, Z.add_0_l by lia.
  rewrite IH. reflexivity.
Qed.

(* Each of the bytes of an integer is a byte. *)
Lemma map_mod_bytes_of (count : nat) (z : Z) :
  map (fun byte => byte mod 256) (bytes_of count z) = bytes_of count z.
Proof.
  revert z. induction count as [ | count IH]; intros z; [ reflexivity | ].
  cbn [bytes_of map]. rewrite Z.mod_mod, IH by lia. reflexivity.
Qed.

(* The low bytes of an integer are not changed by what is added above
   them. *)
Lemma bytes_of_add_above (count : nat) (z k : Z) :
  bytes_of count (z + k * 256 ^ Z.of_nat count) = bytes_of count z.
Proof.
  revert z k. induction count as [ | count IH]; intros z k;
    [ reflexivity | ].
  cbn [bytes_of].
  rewrite Nat2Z.inj_succ, Z.pow_succ_r by lia.
  replace (k * (256 * 256 ^ Z.of_nat count))
    with (k * 256 ^ Z.of_nat count * 256) by lia.
  rewrite Z_mod_plus_full, Z.div_add by lia.
  rewrite IH. reflexivity.
Qed.

(* Two swaps of [count] bytes give the low [count] bytes back. *)
Lemma swap_bytes_twice (count : nat) (z : Z) :
  swap_bytes count (swap_bytes count z) = z mod 256 ^ Z.of_nat count.
Proof.
  unfold swap_bytes.
  pose proof (bytes_of_little_endian (rev (bytes_of count z))) as Hbytes.
  rewrite length_rev, length_bytes_of in Hbytes. rewrite Hbytes.
  rewrite <- map_rev, rev_involutive, map_mod_bytes_of.
  apply little_endian_bytes_of.
Qed.

(* Wrapping adds a multiple of two to the width. *)
Lemma wrap_adds_multiple (width z : Z) :
  0 <= width -> ∃ k : Z, wrap width z = z + k * 2 ^ width.
Proof.
  intros Hwidth. unfold wrap.
  exists (- ((z + 2 ^ (width - 1)) / 2 ^ width)).
  assert (Hpow : 2 ^ width <> 0) by (apply Z.pow_nonzero; lia).
  pose proof (Z.div_mod (z + 2 ^ (width - 1)) (2 ^ width) Hpow) as Hdiv.
  lia.
Qed.

(* Wrapping looks at the low bits only. *)
Lemma wrap_mod (width z : Z) :
  wrap width (z mod 2 ^ width) = wrap width z.
Proof. unfold wrap. rewrite Zplus_mod_idemp_l. reflexivity. Qed.

(* So two swaps of an integer of [count] bytes with a sign, each
   wrapped to the width, give the integer back. *)
Lemma swap_bytes_wrap_twice (count : nat) (a : Z) :
  (1 <= count)%nat ->
  in_range (8 * Z.of_nat count) a = true ->
  wrap (8 * Z.of_nat count)
    (swap_bytes count (wrap (8 * Z.of_nat count) (swap_bytes count a)))
  = a.
Proof.
  intros Hcount Hrange.
  assert (Hpow : 2 ^ (8 * Z.of_nat count) = 256 ^ Z.of_nat count).
  { rewrite Z.pow_mul_r by lia. reflexivity. }
  destruct (wrap_adds_multiple (8 * Z.of_nat count) (swap_bytes count a))
    as [k Hk]; [ lia | ].
  rewrite Hk, Hpow.
  unfold swap_bytes at 1. rewrite bytes_of_add_above.
  fold (swap_bytes count (swap_bytes count a)).
  rewrite swap_bytes_twice, <- Hpow, wrap_mod.
  apply wrap_of_in_range; [ lia | exact Hrange ].
Qed.

(* At the kinds other than the immediates, the swap done twice is the
   identity. *)
Lemma eval_unary_int_arith_swap_twice (kind : standard_int) (a : Z) :
  kind <> Standard_tagged_immediate ->
  kind <> Standard_naked_immediate ->
  in_range (int_width kind) a = true ->
  eval_unary_int_arith kind Swap_byte_endianness
    (eval_unary_int_arith kind Swap_byte_endianness a)
  = a.
Proof.
  intros Htagged Hnaked Hrange.
  destruct kind; try contradiction;
    cbn [eval_unary_int_arith int_width] in *.
  - rewrite !(wrap_of_in_range 8 a) by (lia || exact Hrange). reflexivity.
  - exact (swap_bytes_wrap_twice 2 a ltac:(lia) Hrange).
  - exact (swap_bytes_wrap_twice 4 a ltac:(lia) Hrange).
  - exact (swap_bytes_wrap_twice 8 a ltac:(lia) Hrange).
  - exact (swap_bytes_wrap_twice 8 a ltac:(lia) Hrange).
Qed.

(* At the immediates it gives the two low bytes of the integer. *)
Lemma eval_unary_int_arith_swap_twice_immediate (kind : standard_int)
    (a : Z) :
  kind = Standard_tagged_immediate ∨ kind = Standard_naked_immediate ->
  eval_unary_int_arith kind Swap_byte_endianness
    (eval_unary_int_arith kind Swap_byte_endianness a)
  = a mod 65536.
Proof.
  intros [-> | ->]; cbn [eval_unary_int_arith];
    exact (swap_bytes_twice 2 a).
Qed.

(* Simplify rewrites the exclusive or of an integer with -1, on either
   side, into [Not] of the integer ([Simplify_binary_primitive], where
   an operation answers [Bitwise_not_of_the_other_side]). The two are
   the same integer. *)
Lemma eval_int_arith_xor_minus_one_r (kind : standard_int) (a : Z) :
  eval_int_arith (int_width kind) Xor a (-1)
  = Some (eval_unary_int_arith kind Not a).
Proof.
  unfold eval_int_arith, eval_unary_int_arith.
  rewrite Z.lxor_m1_r. reflexivity.
Qed.

Lemma eval_int_arith_xor_minus_one_l (kind : standard_int) (a : Z) :
  eval_int_arith (int_width kind) Xor (-1) a
  = Some (eval_unary_int_arith kind Not a).
Proof.
  unfold eval_int_arith, eval_unary_int_arith.
  rewrite Z.lxor_m1_l. reflexivity.
Qed.

(* The result of a shift is in range. *)
Lemma eval_int_shift_in_range (width : Z) (op : int_shift_op)
    (a amount : Z) :
  1 <= width -> in_range width (eval_int_shift width op a amount) = true.
Proof. intros Hwidth. destruct op; apply wrap_in_range, Hwidth. Qed.

(* An arithmetic shift right keeps an integer in range, so the wrap of
   [Asr] changes nothing. *)
Lemma shiftr_in_range (width a amount : Z) :
  1 <= width ->
  0 <= amount ->
  in_range width a = true ->
  in_range width (Z.shiftr a amount) = true.
Proof.
  intros Hwidth Hamount Hrange. unfold in_range in *.
  apply andb_true_iff in Hrange as [Hlow Hhigh].
  apply Z.leb_le in Hlow. apply Z.ltb_lt in Hhigh.
  assert (Hhalf : 0 < 2 ^ (width - 1)) by (apply Z.pow_pos_nonneg; lia).
  assert (Hpow : 0 < 2 ^ amount) by (apply Z.pow_pos_nonneg; lia).
  rewrite Z.shiftr_div_pow2 by exact Hamount.
  apply andb_true_iff. split; [ apply Z.leb_le | apply Z.ltb_lt ].
  - apply Z.div_le_lower_bound; [ exact Hpow | nia ].
  - apply Z.div_lt_upper_bound; [ exact Hpow | nia ].
Qed.

Lemma eval_int_shift_asr (width a amount : Z) :
  1 <= width ->
  0 <= amount ->
  in_range width a = true ->
  eval_int_shift width Asr a amount = Z.shiftr a amount.
Proof.
  intros Hwidth Hamount Hrange. simpl.
  apply wrap_of_in_range; [ exact Hwidth | ].
  exact (shiftr_in_range width a amount Hwidth Hamount Hrange).
Qed.

(* The unsigned reading of an integer in range is below two to the
   width, and wraps back to the integer. *)
Lemma unsigned_bounds (width a : Z) :
  1 <= width -> in_range width a = true -> 0 <= unsigned width a < 2 ^ width.
Proof.
  intros Hwidth Hrange. unfold in_range in Hrange. unfold unsigned.
  apply andb_true_iff in Hrange as [Hlow Hhigh].
  apply Z.leb_le in Hlow. apply Z.ltb_lt in Hhigh.
  assert (Hdouble : 2 ^ width = 2 * 2 ^ (width - 1)).
  { rewrite <- Z.pow_succ_r by lia. f_equal. lia. }
  destruct (Z.ltb_spec a 0); lia.
Qed.

Lemma wrap_unsigned (width a : Z) :
  1 <= width -> in_range width a = true -> wrap width (unsigned width a) = a.
Proof.
  intros Hwidth Hrange. unfold in_range in Hrange. unfold wrap, unsigned.
  apply andb_true_iff in Hrange as [Hlow Hhigh].
  apply Z.leb_le in Hlow. apply Z.ltb_lt in Hhigh.
  assert (Hdouble : 2 ^ width = 2 * 2 ^ (width - 1)).
  { rewrite <- Z.pow_succ_r by lia. f_equal. lia. }
  destruct (Z.ltb_spec a 0).
  - replace (a + 2 ^ width + 2 ^ (width - 1))
      with (a + 2 ^ (width - 1) + 1 * 2 ^ width) by lia.
    rewrite Z_mod_plus_full, Z.mod_small by lia. lia.
  - rewrite Z.mod_small by lia. lia.
Qed.

(* A logical shift right by an amount that is not 0 gives an integer in
   range, so the wrap of [Lsr] changes nothing there; by 0, it gives
   back the integer shifted. *)
Lemma eval_int_shift_lsr (width a amount : Z) :
  1 <= width ->
  0 <= amount ->
  in_range width a = true ->
  eval_int_shift width Lsr a amount
  = if amount =? 0 then a else Z.shiftr (unsigned width a) amount.
Proof.
  intros Hwidth Hamount Hrange. simpl.
  destruct (Z.eqb_spec amount 0) as [-> | Hpositive].
  - rewrite Z.shiftr_0_r. exact (wrap_unsigned width a Hwidth Hrange).
  - apply wrap_of_in_range; [ exact Hwidth | ].
    pose proof (unsigned_bounds width a Hwidth Hrange) as Hunsigned.
    assert (Hdouble : 2 ^ width = 2 * 2 ^ (width - 1)).
    { rewrite <- Z.pow_succ_r by lia. f_equal. lia. }
    assert (Hhalf : 0 < 2 ^ (width - 1)) by (apply Z.pow_pos_nonneg; lia).
    assert (Hpow : 2 ^ 1 <= 2 ^ amount) by (apply Z.pow_le_mono_r; lia).
    change (2 ^ 1) with 2 in Hpow.
    rewrite Z.shiftr_div_pow2 by exact Hamount.
    unfold in_range.
    apply andb_true_iff. split; [ apply Z.leb_le | apply Z.ltb_lt ].
    + apply Z.div_le_lower_bound; [ lia | nia ].
    + apply Z.div_lt_upper_bound; [ lia | nia ].
Qed.

(* At 64 bits, the arithmetic of rocq/OxCamlInt64.v. *)
Lemma wrap_64 (z : Z) : wrap 64 z = wrap64 z.
Proof. reflexivity. Qed.

Lemma unsigned_64 (z : Z) : unsigned 64 z = unsigned64 z.
Proof. reflexivity. Qed.

Lemma in_range_64 (z : Z) : in_range 64 z = fits_int64 z.
Proof. reflexivity. Qed.

(* At 63 bits, OCaml's [int] as rocq/OxCamlInt.v has it: Rocq's [Sint63].
   Every [int] is in range, and its sums, differences and products are
   those of the primitives. *)
Lemma in_range_63 (z : Z) : in_range 63 z = fits_int z.
Proof.
  unfold in_range, fits_int. change (2 ^ (63 - 1)) with two62.
  destruct (- two62 <=? z); [ | reflexivity ]. simpl.
  destruct (Z.ltb_spec z two62), (Z.leb_spec z (two62 - 1)); try reflexivity;
    unfold two62 in *; lia.
Qed.

Lemma wrap_63 (z : Z) : wrap 63 z = Sint63.cmod z Uint63.wB.
Proof. reflexivity. Qed.

Lemma int_in_range (i : int) : in_range 63 (Sint63.to_Z i) = true.
Proof.
  rewrite in_range_63. unfold fits_int. pose proof (int_bounds i).
  apply andb_true_iff. split; apply Z.leb_le; lia.
Qed.

Lemma int_add (a b : int) :
  Sint63.to_Z (a + b)%sint63 = wrap 63 (Sint63.to_Z a + Sint63.to_Z b).
Proof. rewrite wrap_63. apply Sint63.add_spec. Qed.

Lemma int_sub (a b : int) :
  Sint63.to_Z (a - b)%sint63 = wrap 63 (Sint63.to_Z a - Sint63.to_Z b).
Proof. rewrite wrap_63. apply Sint63.sub_spec. Qed.

Lemma int_mul (a b : int) :
  Sint63.to_Z (a * b)%sint63 = wrap 63 (Sint63.to_Z a * Sint63.to_Z b).
Proof. rewrite wrap_63. apply Sint63.mul_spec. Qed.

(* ======================================================================== *)
(* A primitive brings no number out of range                                *)
(* ======================================================================== *)

Lemma const_in_range_const_of_int (kind : standard_int) (i : Z) :
  const_in_range (const_of_int kind i) = in_range (int_width kind) i.
Proof. destruct kind; reflexivity. Qed.

Theorem int_of_value_in_range : int_of_value_in_range_statement.
Proof.
  intros kind i Hrange. rewrite const_in_range_const_of_int in Hrange.
  destruct kind; cbn [const_of_int int_of_value]; rewrite Hrange;
    reflexivity.
Qed.

Theorem phys_equal_same : phys_equal_same_statement.
Proof.
  intros statics op v result. cbn [eval_binary_primitive].
  unfold eval_phys_equal.
  destruct (int_of_value Standard_tagged_immediate v) as [a | ].
  - rewrite Z.eqb_refl. cbn. intros H. injection H as <-.
    destruct op; reflexivity.
  - destruct (identity_of statics v) as [s | ]; [ | discriminate ].
    rewrite Pos.eqb_refl. cbn. intros H. injection H as <-.
    destruct op; reflexivity.
Qed.

(* The identity of a symbol is that symbol. *)
Lemma identity_of_symbol (statics : static_env) (s s' : symbol) :
  identity_of statics (Value_symbol s) = Some s' -> s' = s.
Proof.
  unfold identity_of, imported_of.
  destruct (lookup s (static_values statics)) as [allocated | ];
    [ | discriminate ].
  destruct allocated; try discriminate.
  match goal with
  | c : import_class |- _ => destruct c
  end;
    try discriminate.
  intros H. injection H as <-. reflexivity.
Qed.

Theorem phys_equal_symbols : phys_equal_symbols_statement.
Proof.
  intros statics op s1 s2 result Hne. cbn [eval_binary_primitive].
  unfold eval_phys_equal. cbn [int_of_value].
  destruct (identity_of statics (Value_symbol s1)) as [s1' | ] eqn:H1;
    destruct (identity_of statics (Value_symbol s2)) as [s2' | ] eqn:H2.
  - apply identity_of_symbol in H1, H2. subst s1' s2'.
    destruct (Pos.eqb_spec s1 s2) as [Heq | _]; [ contradiction | ].
    cbn. intros H. injection H as <-. destruct op; reflexivity.
  - destruct (is_allocated statics (Value_symbol s2)
              || is_imported_pointer statics (Value_symbol s2));
      [ | discriminate ].
    cbn. intros H. injection H as <-. destruct op; reflexivity.
  - destruct (is_allocated statics (Value_symbol s1)
              || is_imported_pointer statics (Value_symbol s1));
      [ | discriminate ].
    cbn. intros H. injection H as <-. destruct op; reflexivity.
  - discriminate.
Qed.

(* A tag is an immediate, so [Get_tag] returns one. *)
Lemma tag_in_range_ocaml_int (tag : Z) :
  tag_in_range tag = true -> ocaml_int_in_range tag = true.
Proof.
  unfold tag_in_range, ocaml_int_in_range, in_range. cbn [int_width].
  intros Htag. apply andb_true_iff in Htag as [Hlow Hhigh].
  apply Z.leb_le in Hlow. apply Z.ltb_lt in Hhigh.
  change (2 ^ (63 - 1)) with 4611686018427387904.
  apply andb_true_iff. split; [ apply Z.leb_le | apply Z.ltb_lt ]; lia.
Qed.

Lemma naked_bool_in_range (b : bool) : value_in_range (naked_bool b) = true.
Proof. destruct b; reflexivity. Qed.

Lemma eval_int_comp_in_range (width : Z)
    (behaviour : int_comparison_behaviour) (a b : Z) :
  ocaml_int_in_range (eval_int_comp width behaviour a b) = true.
Proof.
  destruct behaviour as [c | s]; cbn [eval_int_comp].
  - destruct (eval_int_comparison width c a b); reflexivity.
  - destruct (reading width s a ?= reading width s b); reflexivity.
Qed.

(* The machine word of a tagged immediate is an int64, and half an
   int64 is an immediate. *)
Lemma tagged_word_in_range (i : Z) :
  in_range (int_width Standard_tagged_immediate) i = true ->
  in_range 64 (2 * i + 1) = true.
Proof.
  unfold in_range. cbn [int_width].
  change (2 ^ (63 - 1)) with 4611686018427387904.
  change (2 ^ (64 - 1)) with 9223372036854775808.
  intros Hrange. apply andb_true_iff in Hrange as [Hlow Hhigh].
  apply Z.leb_le in Hlow. apply Z.ltb_lt in Hhigh.
  apply andb_true_iff. split; [ apply Z.leb_le | apply Z.ltb_lt ]; lia.
Qed.

Lemma half_word_in_range (word : Z) :
  in_range (int_width Standard_naked_int64) word = true ->
  ocaml_int_in_range (word / 2) = true.
Proof.
  unfold ocaml_int_in_range, in_range. cbn [int_width].
  change (2 ^ (63 - 1)) with 4611686018427387904.
  change (2 ^ (64 - 1)) with 9223372036854775808.
  intros Hrange. apply andb_true_iff in Hrange as [Hlow Hhigh].
  apply Z.leb_le in Hlow. apply Z.ltb_lt in Hhigh.
  pose proof (Z.div_mod word 2 ltac:(lia)) as Hdiv.
  pose proof (Z.mod_pos_bound word 2 eq_refl) as Hmod.
  apply andb_true_iff. split; [ apply Z.leb_le | apply Z.ltb_lt ]; lia.
Qed.

(* An integer of two bytes without a sign is an immediate. *)
Lemma two_bytes_in_range (z : Z) :
  0 <= z < 65536 -> ocaml_int_in_range z = true.
Proof.
  intros Hz. unfold ocaml_int_in_range, in_range.
  change (2 ^ (int_width Standard_tagged_immediate - 1))
    with 4611686018427387904.
  apply andb_true_iff. split; [ apply Z.leb_le | apply Z.ltb_lt ]; lia.
Qed.

(* What a load reads is at least 0, and below the first integer that
   needs more bytes than its width has. *)
Lemma little_endian_read_bound (bytes : list Z) (index count : nat) :
  0 <= little_endian (firstn count (skipn index bytes))
  < 256 ^ Z.of_nat count.
Proof.
  pose proof (little_endian_bound (firstn count (skipn index bytes)))
    as Hbound.
  pose proof (firstn_le_length count (skipn index bytes)) as Hlength.
  assert (Hpow :
    256 ^ Z.of_nat (length (firstn count (skipn index bytes)))
    <= 256 ^ Z.of_nat count)
    by (apply Z.pow_le_mono_r; lia).
  lia.
Qed.

Section In_range.

Variable statics : static_env.

(* The tag and the fields of a block in range are, whether the block was
   made by a primitive or is what a symbol is bound to. *)
Lemma block_of_in_range (arg : value) (tag : Z) (fields : list value) :
  (∀ (s : symbol) (allocated : statically_allocated_value),
    Value_symbol s ∈ [arg] ->
    lookup s (static_values statics) = Some allocated ->
    allocated_in_range allocated = true) ->
  value_in_range arg = true ->
  block_of statics arg = Some (tag, fields) ->
  tag_in_range tag = true ∧ forallb value_in_range fields = true.
Proof.
  intros Hsymbols Harg Hblock.
  destruct (block_of_cases statics arg tag fields Hblock)
    as [-> | (s & -> & Hlookup)];
    apply andb_true_iff.
  - exact Harg.
  - exact (Hsymbols s _ (or_introl eq_refl) Hlookup).
Qed.

(* So is the number inside a boxed number. *)
Lemma boxed_number_of_in_range (arg : value) (c : const) :
  (∀ (s : symbol) (allocated : statically_allocated_value),
    Value_symbol s ∈ [arg] ->
    lookup s (static_values statics) = Some allocated ->
    allocated_in_range allocated = true) ->
  value_in_range arg = true ->
  boxed_number_of statics arg = Some c ->
  const_in_range c = true.
Proof.
  intros Hsymbols Harg. destruct arg; simpl; try discriminate.
  - match goal with
    | |- context [lookup ?s ?values] =>
      destruct (lookup s values) as [allocated | ] eqn:Hlookup;
        [ | discriminate ];
        pose proof (Hsymbols s allocated (or_introl eq_refl) Hlookup)
          as Hallocated
    end.
    destruct allocated; try discriminate.
    match goal with
    | contents : value |- _ => destruct contents
    end;
      try discriminate.
    unfold boxable_contents.
    match goal with
    | |- context [boxable_number_of_const ?c'] =>
      destruct (boxable_number_of_const c')
    end;
      [ | discriminate ].
    intros [= <-]. exact Hallocated.
  - unfold boxable_contents.
    match goal with
    | |- context [boxable_number_of_const ?c'] =>
      destruct (boxable_number_of_const c')
    end;
      [ | discriminate ].
    intros [= <-]. exact Harg.
Qed.

(* By cases on the unary primitives. *)
Lemma eval_unary_primitive_in_range (p : unary_primitive) (arg v : value) :
  (∀ (s : symbol) (allocated : statically_allocated_value),
    Value_symbol s ∈ [arg] ->
    lookup s (static_values statics) = Some allocated ->
    allocated_in_range allocated = true) ->
  value_in_range arg = true ->
  eval_unary_primitive statics p arg = Some v ->
  value_in_range v = true.
Proof.
  intros Hsymbols Harg.
  destruct p as [access mut field | variant_only | | | length_kind | | kind
                | kind op | src dst | | kind | kind | | | reinterpret];
    cbn [eval_unary_primitive].
  - unfold eval_block_load. destruct access as [tag size field_kind].
    destruct (block_of statics arg) as [[block_tag fields] | ] eqn:Hblock;
      [ | discriminate ].
    destruct
      (known_is tag block_tag && known_is size (Z.of_nat (length fields))
       && (0 <=? field));
      [ | discriminate ].
    destruct (nth_error fields (Z.to_nat field)) as [loaded | ] eqn:Hnth;
      [ | discriminate ].
    destruct (block_of_in_range arg block_tag fields Hsymbols Harg Hblock)
      as [_ Hfields].
    pose proof
      (proj1 (forallb_forall _ _) Hfields loaded (nth_error_In _ _ Hnth))
      as Hloaded.
    destruct
      (match field_kind with
       | Any_value => kind_eqb (kind_of_value loaded) Value
       | Immediate => is_tagged_immediate loaded
       end);
      [ | discriminate ].
    intros [= <-]. exact Hloaded.
  - destruct (eval_is_int statics variant_only arg); [ | discriminate ].
    intros [= <-]. apply naked_bool_in_range.
  - destruct (eval_is_null statics arg); [ | discriminate ].
    intros [= <-]. apply naked_bool_in_range.
  - destruct (block_of statics arg) as [[tag fields] | ] eqn:Hblock;
      [ | discriminate ].
    intros [= <-]. cbn [value_in_range const_in_range].
    apply tag_in_range_ocaml_int.
    exact (proj1 (block_of_in_range arg tag fields Hsymbols Harg Hblock)).
  - (* [Array_length] *)
    destruct (array_of statics arg) as [elements | ]; [ | discriminate ].
    unfold array_length_of.
    destruct (in_range _ _) eqn:Hlength; [ | discriminate ].
    intros [= <-]. exact Hlength.
  - (* [String_length] *)
    destruct (string_of statics arg) as [bytes | ]; [ | discriminate ].
    destruct (in_range _ _) eqn:Hlength; [ | discriminate ].
    intros [= <-]. exact Hlength.
  - destruct (kind_eqb (kind_of_value arg) kind); [ | discriminate ].
    destruct (bound_if_symbol statics arg); [ | discriminate ].
    intros [= <-]. exact Harg.
  - (* [Unary_int_arith] *)
    destruct (int_of_value kind arg) as [a | ]; [ | discriminate ].
    intros [= <-]. cbn [value_in_range].
    rewrite const_in_range_const_of_int.
    apply eval_unary_int_arith_in_range.
  - (* [Num_conv] *)
    destruct (int_of_value src arg) as [i | ]; [ | discriminate ].
    intros [= <-]. cbn [value_in_range].
    rewrite const_in_range_const_of_int.
    apply wrap_in_range, int_width_positive.
  - (* [Boolean_not] *)
    destruct (int_of_value Standard_tagged_immediate arg) as [i | ];
      [ | discriminate ].
    destruct (i =? 0); [ intros [= <-]; reflexivity | ].
    destruct (i =? 1); [ intros [= <-]; reflexivity | discriminate ].
  - destruct (boxed_number_of statics arg) as [c | ] eqn:Hboxed;
      [ | discriminate ].
    destruct (is_number_of_kind kind c); [ | discriminate ].
    intros [= <-].
    exact (boxed_number_of_in_range arg c Hsymbols Harg Hboxed).
  - destruct arg; try discriminate.
    match goal with
    | |- context [is_number_of_kind kind ?c] =>
      destruct (is_number_of_kind kind c)
    end;
      [ | discriminate ].
    intros [= <-]. exact Harg.
  - destruct (int_of_value Standard_tagged_immediate arg) as [i | ] eqn:Hint;
      [ | discriminate ].
    intros [= <-]. exact (proj2 (int_of_value_const _ _ _ Hint)).
  - destruct (int_of_value Standard_naked_immediate arg) as [i | ] eqn:Hint;
      [ | discriminate ].
    intros [= <-]. exact (proj2 (int_of_value_const _ _ _ Hint)).
  - (* [Reinterpret_64_bit_word] *)
    destruct reinterpret; cbn [eval_reinterpret_64_bit_word];
      (destruct (int_of_value _ arg) as [i | ] eqn:Hint;
        [ | discriminate ]);
      intros [= <-]; cbn [value_in_range const_in_range].
    + exact (tagged_word_in_range i (proj2 (int_of_value_const _ _ _ Hint))).
    + exact (half_word_in_range i (proj2 (int_of_value_const _ _ _ Hint))).
Qed.

(* By cases on the binary primitives. *)
Lemma eval_binary_primitive_in_range (p : binary_primitive)
    (arg1 arg2 v : value) :
  (∀ (s : symbol) (allocated : statically_allocated_value),
    Value_symbol s ∈ [arg1; arg2] ->
    lookup s (static_values statics) = Some allocated ->
    allocated_in_range allocated = true) ->
  value_in_range arg1 = true ->
  eval_binary_primitive statics p arg1 arg2 = Some v ->
  value_in_range v = true.
Proof.
  intros Hsymbols Harg1.
  destruct p as [array_kind load_kind mut | width | op | kind op | kind op
                | kind behaviour];
    cbn [eval_binary_primitive].
  - (* [Array_load]: an element of an array that is in range. *)
    destruct (array_of statics arg1) as [elements | ] eqn:Harray;
      [ | discriminate ].
    intros Hloaded.
    destruct (array_of_cases statics arg1 elements Harray)
      as [-> | (s & -> & Hlookup)].
    + cbn [value_in_range] in Harg1.
      exact
        (proj1 (forallb_forall _ _) Harg1 v
           (proj1 (array_loaded_element _ _ _ _ Hloaded))).
    + pose proof (Hsymbols s _ (or_introl eq_refl) Hlookup) as Helements.
      cbn [allocated_in_range] in Helements.
      exact
        (proj1 (forallb_forall _ _) Helements v
           (proj1 (array_loaded_element _ _ _ _ Hloaded))).
  - (* [String_or_bigstring_load] *)
    unfold eval_string_load.
    destruct (string_of statics arg1) as [bytes | ]; [ | discriminate ].
    destruct (int_of_value Standard_naked_nativeint arg2) as [index | ];
      [ | discriminate ].
    cbv zeta. destruct (_ && _); [ | discriminate ].
    pose proof
      (little_endian_read_bound bytes (Z.to_nat index)
         (byte_width_of_string_accessor_width width))
      as Hread.
    destruct width; intros [= <-]; cbn [value_in_range const_in_range];
      try (apply wrap_in_range; lia);
      apply two_bytes_in_range; cbn in Hread; lia.
  - (* [Phys_equal] *)
    destruct (eval_phys_equal statics arg1 arg2); [ | discriminate ].
    intros [= <-]. apply naked_bool_in_range.
  - (* [Int_arith] *)
    destruct (int_of_value kind arg1) as [a | ]; [ | discriminate ].
    destruct (int_of_value kind arg2) as [b | ]; [ | discriminate ].
    destruct (eval_int_arith (int_width kind) op a b) as [result | ]
      eqn:Harith;
      [ | discriminate ].
    intros [= <-]. cbn [value_in_range].
    rewrite const_in_range_const_of_int.
    exact
      (eval_int_arith_in_range _ op a b result (int_width_positive kind)
         Harith).
  - (* [Int_shift] *)
    destruct (int_of_value kind arg1) as [a | ]; [ | discriminate ].
    destruct (int_of_value Standard_naked_immediate arg2) as [amount | ];
      [ | discriminate ].
    destruct ((0 <=? amount) && (amount <? shift_bound kind));
      [ | discriminate ].
    intros [= <-]. cbn [value_in_range].
    rewrite const_in_range_const_of_int.
    apply eval_int_shift_in_range, int_width_positive.
  - (* [Int_comp] *)
    destruct (int_of_value kind arg1) as [a | ]; [ | discriminate ].
    destruct (int_of_value kind arg2) as [b | ]; [ | discriminate ].
    intros [= <-]. cbn [value_in_range const_in_range].
    apply eval_int_comp_in_range.
Qed.

(* By cases on the variadic primitives. *)
Lemma eval_variadic_primitive_in_range (p : variadic_primitive)
    (args : list value) (v : value) :
  variadic_primitive_in_range p = true ->
  (∀ arg : value, arg ∈ args -> value_in_range arg = true) ->
  eval_variadic_primitive statics p args = Some v ->
  value_in_range v = true.
Proof.
  intros Hp Hargs. destruct p as [[tag] | kind].
  - cbn [variadic_primitive_in_range] in Hp. cbn [eval_variadic_primitive].
    destruct (forallb (fun v => kind_eqb (kind_of_value v) Value) args);
      [ | discriminate ].
    destruct (forallb (bound_if_symbol statics) args); [ | discriminate ].
    intros [= <-]. cbn [value_in_range]. rewrite Hp.
    apply forallb_forall. exact Hargs.
  - cbn [eval_variadic_primitive].
    destruct (array_made kind args) as [elements | ] eqn:Hmade; simpl;
      [ | discriminate ].
    destruct (forallb (bound_if_symbol statics) args); [ | discriminate ].
    intros [= <-].
    destruct (array_made_of_arguments kind args elements Hmade) as [-> _].
    cbn [value_in_range]. apply forallb_forall. exact Hargs.
Qed.

End In_range.

Theorem eval_prim_in_range : eval_prim_in_range_statement.
Proof.
  intros statics p vs v Hp Hsymbols Hargs Heval.
  destruct (eval_prim_arity statics p vs v Heval)
    as [(p' & v1 & -> & -> & Hp')
       | [(p' & v1 & v2 & -> & -> & Hp') | (p' & -> & Hp')]].
  - apply (eval_unary_primitive_in_range statics p' v1 v);
      [ exact Hsymbols | | exact Hp' ].
    apply Hargs. left. reflexivity.
  - apply (eval_binary_primitive_in_range statics p' v1 v2 v Hsymbols);
      [ | exact Hp' ].
    apply Hargs. left. reflexivity.
  - exact
      (eval_variadic_primitive_in_range statics p' vs v Hp Hargs Hp').
Qed.

(* ======================================================================== *)
(* A primitive cannot tell related values apart                             *)
(* ======================================================================== *)

Section Related.

Variable R : value -> value -> Prop.
Variables statics statics' : static_env.
Hypothesis Hrespects : respects_observations R statics statics'.

Let Hconst_l : ∀ (c : const) (v' : value),
    R (Value_const c) v' -> v' = Value_const c :=
  proj1 (proj1 Hrespects).
Let Hconst_r : ∀ (v : value) (c : const),
    R v (Value_const c) -> v = Value_const c :=
  proj2 (proj1 Hrespects).
Let Hnumber : ∀ c : const,
    R (Value_const c) (Value_const c)
    ∧ R (Value_boxed_number c) (Value_boxed_number c) :=
  proj1 (proj1 (proj2 Hrespects)).
Let Hblock : ∀ (tag : Z) (fields fields' : list value),
    Forall2 R fields fields' ->
    R (Value_block tag fields) (Value_block tag fields') :=
  proj1 (proj2 (proj1 (proj2 Hrespects))).
Let Harray : ∀ elements elements' : list value,
    Forall2 R elements elements' ->
    R (Value_immutable_array elements) (Value_immutable_array elements') :=
  proj2 (proj2 (proj1 (proj2 Hrespects))).
Let Hobserved : relates_observations R statics statics' :=
  proj2 (proj2 Hrespects).

(* Related values are of the same kind: the same constant, or two values
   that are not constants. *)
Lemma related_not_const (v v' : value) :
  R v v' ->
  (∀ c : const, v <> Value_const c) ->
  ∀ c : const, v' <> Value_const c.
Proof. intros Hr Hv c ->. exact (Hv c (Hconst_r v c Hr)). Qed.

Lemma related_same_kind (v v' : value) :
  R v v' -> kind_of_value v' = kind_of_value v.
Proof.
  intros Hr.
  destruct (const_or_not v) as [[c ->] | [Hv Hkind]];
    [ rewrite (Hconst_l c v' Hr); reflexivity | ].
  destruct (const_or_not v') as [[c ->] | [_ Hkind']];
    [ rewrite (Hconst_r v c Hr); reflexivity | congruence ].
Qed.

(* A value that is an integer of a kind is a constant, so what is
   related to it is the value itself. *)
Lemma related_int (kind : standard_int) (v v' : value) (i : Z) :
  R v v' -> int_of_value kind v = Some i -> v' = v.
Proof.
  intros Hr Hint. apply int_of_value_const in Hint as [-> _].
  exact (Hconst_l _ v' Hr).
Qed.

Lemma related_int_r (kind : standard_int) (v v' : value) (i : Z) :
  R v v' -> int_of_value kind v' = Some i -> v' = v.
Proof.
  intros Hr Hint. apply int_of_value_const in Hint as [-> _].
  symmetry. exact (Hconst_r v _ Hr).
Qed.

(* So related values are the same integer of a kind, or neither is
   one. *)
Lemma related_int_of_value (kind : standard_int) (v v' : value) :
  R v v' -> int_of_value kind v' = int_of_value kind v.
Proof.
  intros Hr.
  destruct (int_of_value kind v) as [i | ] eqn:Hint.
  - rewrite (related_int _ v v' i Hr Hint). exact Hint.
  - destruct (int_of_value kind v') as [i | ] eqn:Hint'; [ | reflexivity ].
    rewrite (related_int_r _ v v' i Hr Hint') in Hint'. congruence.
Qed.

Lemma related_is_tagged_immediate (v v' : value) :
  R v v' -> is_tagged_immediate v' = is_tagged_immediate v.
Proof.
  intros Hr. unfold is_tagged_immediate.
  destruct (int_of_value Standard_tagged_immediate v) as [i | ] eqn:Hint.
  - rewrite (related_int _ v v' i Hr Hint), Hint. reflexivity.
  - destruct (int_of_value Standard_tagged_immediate v') as [i | ] eqn:Hint';
      [ | reflexivity ].
    rewrite (related_int_r _ v v' i Hr Hint') in Hint'. congruence.
Qed.

Lemma related_is_block (v v' : value) :
  R v v' ->
  is_some (block_of statics v) = true ->
  is_some (block_of statics' v') = true.
Proof.
  intros Hr.
  destruct (block_of statics v) as [[tag fields] | ] eqn:Hv;
    [ | discriminate ].
  destruct (proj1 (Hobserved v v' Hr) tag fields Hv) as (fields' & -> & _).
  reflexivity.
Qed.

Lemma related_is_allocated (v v' : value) :
  R v v' ->
  is_allocated statics v = true ->
  is_allocated statics' v' = true.
Proof.
  intros Hr. unfold is_allocated. rewrite !orb_true_iff.
  destruct (Hobserved v v' Hr)
    as (_ & Hboxed & Hstring & Hclosure & Hpartial & _ & _ & Harrays).
  intros [[[[[Hsome | Hsome] | Hsome] | Hsome] | Hsome] | Hsome].
  - do 5 left.
    apply is_some_not_none, Hclosure, is_some_not_none. exact Hsome.
  - do 4 left. right. exact (related_is_block v v' Hr Hsome).
  - do 3 left. right.
    destruct (boxed_number_of statics v) as [c | ] eqn:Hv; [ | discriminate ].
    rewrite (Hboxed c eq_refl). reflexivity.
  - do 2 left. right.
    destruct (string_of statics v) as [bytes | ] eqn:Hv; [ | discriminate ].
    rewrite (Hstring bytes eq_refl). reflexivity.
  - left. right.
    destruct (array_of statics v) as [elements | ] eqn:Hv; [ | discriminate ].
    destruct (Harrays elements eq_refl) as (elements' & -> & _).
    reflexivity.
  - right. apply is_some_not_none, Hpartial, is_some_not_none. exact Hsome.
Qed.

Lemma related_eval_is_int (variant_only b : bool) (v v' : value) :
  R v v' ->
  eval_is_int statics variant_only v = Some b ->
  eval_is_int statics' variant_only v' = Some b.
Proof.
  intros Hr. unfold eval_is_int.
  rewrite (related_is_tagged_immediate v v' Hr).
  destruct (is_tagged_immediate v); [ exact (fun H => H) | ].
  destruct variant_only.
  - destruct (is_some (block_of statics v)) eqn:Hv; [ | discriminate ].
    rewrite (related_is_block v v' Hr Hv). exact (fun H => H).
  - destruct (is_allocated statics v) eqn:Hv; [ | discriminate ].
    rewrite (related_is_allocated v v' Hr Hv). exact (fun H => H).
Qed.

Lemma related_imported_of (v v' : value) (i : symbol * import_class) :
  R v v' ->
  imported_of statics v = Some i ->
  imported_of statics' v' = Some i.
Proof.
  intros Hr.
  exact
    (proj1
       (proj2 (proj2 (proj2 (proj2 (proj2 (proj2 (Hobserved v v' Hr)))))))
       i).
Qed.

(* [Phys_equal] reads its arguments as tagged immediates, which are
   constants, by whether they are allocated, which the observations
   say, and as imported symbols, which related values are alike. *)
Lemma related_eval_phys_equal (same : bool) (v1 v2 v1' v2' : value) :
  R v1 v1' ->
  R v2 v2' ->
  eval_phys_equal statics v1 v2 = Some same ->
  eval_phys_equal statics' v1' v2' = Some same.
Proof.
  intros Hr1 Hr2. apply eval_phys_equal_transfer.
  - exact (related_int_of_value _ v1 v1' Hr1).
  - exact (related_int_of_value _ v2 v2' Hr2).
  - exact (related_is_allocated v1 v1' Hr1).
  - exact (related_is_allocated v2 v2' Hr2).
  - intros i. exact (related_imported_of v1 v1' i Hr1).
  - intros i. exact (related_imported_of v2 v2' i Hr2).
Qed.

(* On a value that is not a constant, [Is_null] answers no if it
   answers. *)
Lemma eval_is_null_not_const (st : static_env) (w : value) :
  (∀ c : const, w <> Value_const c) ->
  eval_is_null st w
  = if is_tagged_immediate w || is_allocated st w then Some false else None.
Proof.
  intros Hw. destruct w; try reflexivity. exfalso. exact (Hw _ eq_refl).
Qed.

Lemma related_eval_is_null (b : bool) (v v' : value) :
  R v v' ->
  eval_is_null statics v = Some b ->
  eval_is_null statics' v' = Some b.
Proof.
  intros Hr.
  destruct (const_or_not v) as [[c ->] | [Hv _]];
    [ rewrite (Hconst_l c v' Hr); destruct c; exact (fun H => H) | ].
  rewrite (eval_is_null_not_const statics v Hv),
    (eval_is_null_not_const statics' v' (related_not_const v v' Hr Hv)),
    (related_is_tagged_immediate v v' Hr).
  destruct (is_tagged_immediate v); [ exact (fun H => H) | ]. simpl.
  destruct (is_allocated statics v) eqn:Hallocated; [ | discriminate ].
  rewrite (related_is_allocated v v' Hr Hallocated). exact (fun H => H).
Qed.

(* By cases on the unary primitives: each reads its argument as a
   constant, which is related to itself only, or through an observation,
   which answers alike of related values. *)
Lemma eval_unary_primitive_related (p : unary_primitive)
    (v v' result : value) :
  R v v' ->
  eval_unary_primitive statics p v = Some result ->
  ∃ result' : value,
    eval_unary_primitive statics' p v' = Some result' ∧ R result result'.
Proof.
  intros Hr.
  destruct (Hobserved v v' Hr)
    as (Hblocks & Hboxed & Hstring & _ & _ & Hbound & _ & Harrays).
  destruct p as [access mut field | variant_only | | | length_kind | | kind
                | kind op | src dst | | kind | kind | | | reinterpret];
    cbn [eval_unary_primitive].
  - unfold eval_block_load. destruct access as [tag size field_kind].
    destruct (block_of statics v) as [[block_tag fields] | ] eqn:Hv;
      [ | discriminate ].
    destruct (Hblocks block_tag fields eq_refl) as (fields' & -> & Hfields).
    rewrite <- (Forall2_length Hfields).
    destruct
      (known_is tag block_tag && known_is size (Z.of_nat (length fields))
       && (0 <=? field));
      [ | discriminate ].
    destruct (nth_error fields (Z.to_nat field)) as [loaded | ] eqn:Hnth;
      [ | discriminate ].
    destruct (forall2_nth_error R fields fields' Hfields _ loaded Hnth)
      as (loaded' & -> & Hloaded).
    rewrite (related_same_kind loaded loaded' Hloaded),
      (related_is_tagged_immediate loaded loaded' Hloaded).
    destruct
      (match field_kind with
       | Any_value => kind_eqb (kind_of_value loaded) Value
       | Immediate => is_tagged_immediate loaded
       end);
      [ | discriminate ].
    intros [= <-]. exists loaded'. auto.
  - destruct (eval_is_int statics variant_only v) as [b | ] eqn:Hv;
      [ | discriminate ].
    rewrite (related_eval_is_int variant_only b v v' Hr Hv).
    intros [= <-]. exists (naked_bool b). split; [ reflexivity | ].
    apply Hnumber.
  - destruct (eval_is_null statics v) as [b | ] eqn:Hv; [ | discriminate ].
    rewrite (related_eval_is_null b v v' Hr Hv).
    intros [= <-]. exists (naked_bool b). split; [ reflexivity | ].
    apply Hnumber.
  - destruct (block_of statics v) as [[tag fields] | ] eqn:Hv;
      [ | discriminate ].
    destruct (Hblocks tag fields eq_refl) as (fields' & -> & _).
    intros [= <-]. eexists. split; [ reflexivity | apply Hnumber ].
  - (* [Array_length]: related arrays have as many elements. *)
    destruct (array_of statics v) as [elements | ] eqn:Hv; [ | discriminate ].
    destruct (Harrays elements eq_refl) as (elements' & -> & Helements).
    unfold array_length_of. rewrite <- (Forall2_length Helements).
    destruct (in_range _ _); [ | discriminate ].
    intros [= <-]. eexists. split; [ reflexivity | apply Hnumber ].
  - (* [String_length] *)
    destruct (string_of statics v) as [bytes | ] eqn:Hv; [ | discriminate ].
    rewrite (Hstring bytes eq_refl).
    destruct (in_range _ _); [ | discriminate ].
    intros [= <-]. eexists. split; [ reflexivity | apply Hnumber ].
  - rewrite (related_same_kind v v' Hr).
    destruct (kind_eqb (kind_of_value v) kind); [ | discriminate ].
    destruct (bound_if_symbol statics v) eqn:Hv; [ | discriminate ].
    rewrite (Hbound eq_refl). intros [= <-]. exists v'. auto.
  - (* [Unary_int_arith] *)
    destruct (int_of_value kind v) as [a | ] eqn:Hv; [ | discriminate ].
    rewrite (related_int _ v v' a Hr Hv), Hv.
    intros [= <-]. eexists. split; [ reflexivity | apply Hnumber ].
  - (* [Num_conv] *)
    destruct (int_of_value src v) as [i | ] eqn:Hv; [ | discriminate ].
    rewrite (related_int _ v v' i Hr Hv), Hv.
    intros [= <-]. eexists. split; [ reflexivity | apply Hnumber ].
  - (* [Boolean_not] *)
    destruct (int_of_value Standard_tagged_immediate v) as [i | ] eqn:Hv;
      [ | discriminate ].
    rewrite (related_int _ v v' i Hr Hv), Hv.
    destruct (i =? 0);
      [ intros [= <-]; eexists; split; [ reflexivity | apply Hnumber ] | ].
    destruct (i =? 1); [ | discriminate ].
    intros [= <-]. eexists. split; [ reflexivity | apply Hnumber ].
  - destruct (boxed_number_of statics v) as [c | ] eqn:Hv;
      [ | discriminate ].
    rewrite (Hboxed c eq_refl).
    destruct (is_number_of_kind kind c); [ | discriminate ].
    intros [= <-]. eexists. split; [ reflexivity | apply Hnumber ].
  - destruct (const_or_not v) as [[c ->] | [Hv _]].
    + rewrite (Hconst_l c v' Hr).
      destruct (is_number_of_kind kind c); [ | discriminate ].
      intros [= <-]. eexists. split; [ reflexivity | apply Hnumber ].
    + destruct v; try discriminate. exfalso. exact (Hv _ eq_refl).
  - destruct (int_of_value Standard_tagged_immediate v) as [i | ] eqn:Hv;
      [ | discriminate ].
    rewrite (related_int _ v v' i Hr Hv), Hv.
    intros [= <-]. eexists. split; [ reflexivity | apply Hnumber ].
  - destruct (int_of_value Standard_naked_immediate v) as [i | ] eqn:Hv;
      [ | discriminate ].
    rewrite (related_int _ v v' i Hr Hv), Hv.
    intros [= <-]. eexists. split; [ reflexivity | apply Hnumber ].
  - (* [Reinterpret_64_bit_word] *)
    destruct reinterpret; cbn [eval_reinterpret_64_bit_word];
      (destruct (int_of_value _ v) as [i | ] eqn:Hv; [ | discriminate ]);
      rewrite (related_int _ v v' i Hr Hv), Hv;
      intros [= <-]; eexists; (split; [ reflexivity | apply Hnumber ]).
Qed.

(* By cases on the binary primitives: the arguments are integers, but
   for the array of [Array_load] and the string of
   [String_or_bigstring_load], which are read through an observation,
   and those of [Phys_equal]. *)
Lemma eval_binary_primitive_related (p : binary_primitive)
    (v1 v2 v1' v2' result : value) :
  R v1 v1' ->
  R v2 v2' ->
  eval_binary_primitive statics p v1 v2 = Some result ->
  ∃ result' : value,
    eval_binary_primitive statics' p v1' v2' = Some result'
    ∧ R result result'.
Proof.
  intros Hr1 Hr2.
  destruct p as [array_kind load_kind mut | width | op | kind op | kind op
                | kind behaviour];
    cbn [eval_binary_primitive].
  - (* [Array_load]: related arrays have related elements. *)
    destruct (array_of statics v1) as [elements | ] eqn:Hv1;
      [ | discriminate ].
    destruct
      (proj2
         (proj2 (proj2 (proj2 (proj2 (proj2 (proj2 (Hobserved v1 v1' Hr1)))))))
         elements Hv1)
      as (elements' & -> & Helements).
    exact
      (array_loaded_related_in R
         (relates_constants_numbers R (proj1 Hrespects)) load_kind elements
         elements' v2 v2' result Helements Hr2).
  - (* [String_or_bigstring_load] *)
    unfold eval_string_load.
    destruct (string_of statics v1) as [bytes | ] eqn:Hv1;
      [ | discriminate ].
    rewrite (proj1 (proj2 (proj2 (Hobserved v1 v1' Hr1))) bytes Hv1).
    destruct (int_of_value Standard_naked_nativeint v2) as [index | ]
      eqn:Hv2;
      [ | discriminate ].
    rewrite (related_int _ v2 v2' index Hr2 Hv2), Hv2.
    cbv zeta. destruct (_ && _); [ | discriminate ].
    destruct width; intros [= <-]; eexists;
      (split; [ reflexivity | apply Hnumber ]).
  - (* [Phys_equal] *)
    destruct (eval_phys_equal statics v1 v2) as [same | ] eqn:Hsame;
      [ | discriminate ].
    rewrite (related_eval_phys_equal same v1 v2 v1' v2' Hr1 Hr2 Hsame).
    intros [= <-]. eexists. split; [ reflexivity | apply Hnumber ].
  - (* [Int_arith] *)
    destruct (int_of_value kind v1) as [a | ] eqn:Hv1; [ | discriminate ].
    destruct (int_of_value kind v2) as [b | ] eqn:Hv2; [ | discriminate ].
    rewrite (related_int kind v1 v1' a Hr1 Hv1),
      (related_int kind v2 v2' b Hr2 Hv2), Hv1, Hv2.
    destruct (eval_int_arith (int_width kind) op a b); [ | discriminate ].
    intros [= <-]. eexists. split; [ reflexivity | apply Hnumber ].
  - (* [Int_shift] *)
    destruct (int_of_value kind v1) as [a | ] eqn:Hv1; [ | discriminate ].
    destruct (int_of_value Standard_naked_immediate v2) as [amount | ]
      eqn:Hv2;
      [ | discriminate ].
    rewrite (related_int kind v1 v1' a Hr1 Hv1),
      (related_int _ v2 v2' amount Hr2 Hv2), Hv1, Hv2.
    destruct ((0 <=? amount) && (amount <? shift_bound kind));
      [ | discriminate ].
    intros [= <-]. eexists. split; [ reflexivity | apply Hnumber ].
  - (* [Int_comp] *)
    destruct (int_of_value kind v1) as [a | ] eqn:Hv1; [ | discriminate ].
    destruct (int_of_value kind v2) as [b | ] eqn:Hv2; [ | discriminate ].
    rewrite (related_int kind v1 v1' a Hr1 Hv1),
      (related_int kind v2 v2' b Hr2 Hv2), Hv1, Hv2.
    intros [= <-]. eexists. split; [ reflexivity | apply Hnumber ].
Qed.

(* By cases on the variadic primitives. *)
Lemma eval_variadic_primitive_related (p : variadic_primitive)
    (vs vs' : list value) (result : value) :
  Forall2 R vs vs' ->
  eval_variadic_primitive statics p vs = Some result ->
  ∃ result' : value,
    eval_variadic_primitive statics' p vs' = Some result' ∧ R result result'.
Proof.
  intros Hargs.
  assert (Hbound' :
    forallb (bound_if_symbol statics) vs = true ->
    forallb (bound_if_symbol statics') vs' = true).
  { apply (forall2_forallb R); [ | exact Hargs ].
    intros v v' Hr.
    exact
      (proj1 (proj2 (proj2 (proj2 (proj2 (proj2 (Hobserved v v' Hr))))))). }
  destruct p as [[tag] | kind]; simpl.
  - destruct (forallb (fun v => kind_eqb (kind_of_value v) Value) vs)
      eqn:Hkinds;
      [ | discriminate ].
    destruct (forallb (bound_if_symbol statics) vs) eqn:Hbound;
      [ | discriminate ].
    intros [= <-]. exists (Value_block tag vs').
    split; [ | apply Hblock, Hargs ].
    assert (Hkinds' :
      forallb (fun v => kind_eqb (kind_of_value v) Value) vs' = true).
    { revert Hkinds. apply (forall2_forallb R); [ | exact Hargs ].
      intros v v' Hr. rewrite (related_same_kind v v' Hr).
      exact (fun H => H). }
    rewrite Hkinds', (Hbound' eq_refl). reflexivity.
  - destruct (array_made kind vs) as [elements | ] eqn:Hmade; simpl;
      [ | discriminate ].
    destruct (forallb (bound_if_symbol statics) vs) eqn:Hbound;
      [ | discriminate ].
    intros [= <-].
    destruct
      (array_made_related_in R
         (relates_constants_numbers R (proj1 Hrespects)) kind vs vs'
         elements Hargs Hmade)
      as (elements' & Hmade' & Helements).
    exists (Value_immutable_array elements').
    split; [ | apply Harray, Helements ].
    rewrite Hmade'. simpl. rewrite (Hbound' eq_refl). reflexivity.
Qed.

End Related.

Theorem eval_prim_related : eval_prim_related_statement.
Proof.
  intros R statics statics' p vs vs' v Hrespects Hargs Heval.
  destruct (eval_prim_arity statics p vs v Heval)
    as [(p' & v1 & -> & -> & Hp)
       | [(p' & v1 & v2 & -> & -> & Hp) | (p' & -> & Hp)]].
  - inversion Hargs as [ | ? v1' ? ? Hr1 Hrest]; subst.
    inversion Hrest; subst. rewrite eval_prim_unary.
    exact
      (eval_unary_primitive_related R statics statics' Hrespects p' v1 v1' v
         Hr1 Hp).
  - inversion Hargs as [ | ? v1' ? ? Hr1 Hrest]; subst.
    inversion Hrest as [ | ? v2' ? ? Hr2 Hrest']; subst.
    inversion Hrest'; subst. rewrite eval_prim_binary.
    exact
      (eval_binary_primitive_related R statics statics' Hrespects p' v1 v2
         v1' v2' v Hr1 Hr2 Hp).
  - rewrite eval_prim_variadic.
    exact
      (eval_variadic_primitive_related R statics statics' Hrespects p' vs
         vs' v Hargs Hp).
Qed.

(* ======================================================================== *)
(* Each primitive has a value somewhere                                     *)
(* ======================================================================== *)

(* So that no statement about a primitive holds for its having no value
   at all, and as a check of the definitions against known answers. *)

Module Examples.

Definition tagged (i : Z) : value := Value_const (Tagged_immediate i).
Definition naked (i : Z) : value := Value_const (Naked_immediate i).

Definition max_int63 : Z := 4611686018427387903.
Definition min_int63 : Z := -4611686018427387904.

Definition a_block : static_env :=
  Static_env [] [(1%positive, Static_block 3 [tagged 10; tagged 20])].

Definition int_arith (kind : standard_int) (op : binary_int_arith_op)
    (a b : value) : option value :=
  eval_prim empty_static_env (Binary (Int_arith kind op)) [a; b].

Definition int_comp (kind : standard_int)
    (behaviour : int_comparison_behaviour) (a b : value) : option value :=
  eval_prim empty_static_env (Binary (Int_comp kind behaviour)) [a; b].

Definition unary (statics : static_env) (p : unary_primitive) (v : value)
    : option value :=
  eval_prim statics (Unary p) [v].

Example add_tagged :
  int_arith Standard_tagged_immediate Add (tagged 3) (tagged 4)
  = Some (tagged 7).
Proof. reflexivity. Qed.

Example add_wraps :
  int_arith Standard_tagged_immediate Add (tagged max_int63) (tagged 1)
  = Some (tagged min_int63).
Proof. reflexivity. Qed.

Example mul_int8_wraps :
  int_arith Standard_naked_int8 Mul (Value_const (Naked_int8 100))
    (Value_const (Naked_int8 3))
  = Some (Value_const (Naked_int8 44)).
Proof. reflexivity. Qed.

Example div_truncates :
  int_arith Standard_tagged_immediate (Div Signed) (tagged (-7)) (tagged 2)
  = Some (tagged (-3)).
Proof. reflexivity. Qed.

Example mod_has_sign_of_dividend :
  int_arith Standard_tagged_immediate (Mod Signed) (tagged (-7)) (tagged 2)
  = Some (tagged (-1)).
Proof. reflexivity. Qed.

Example div_min_by_minus_one :
  int_arith Standard_tagged_immediate (Div Signed) (tagged min_int63)
    (tagged (-1))
  = Some (tagged min_int63).
Proof. reflexivity. Qed.

Example div_unsigned :
  int_arith Standard_naked_int8 (Div Unsigned)
    (Value_const (Naked_int8 (-2))) (Value_const (Naked_int8 2))
  = Some (Value_const (Naked_int8 127)).
Proof. reflexivity. Qed.

Example div_by_zero :
  int_arith Standard_tagged_immediate (Div Signed) (tagged 1) (tagged 0)
  = None.
Proof. reflexivity. Qed.

Example xor_tagged :
  int_arith Standard_tagged_immediate Xor (tagged 5) (tagged (-1))
  = Some (tagged (-6)).
Proof. reflexivity. Qed.

Example add_out_of_range :
  int_arith Standard_tagged_immediate Add (tagged (max_int63 + 1)) (tagged 0)
  = None.
Proof. reflexivity. Qed.

Example add_wrong_kind :
  int_arith Standard_tagged_immediate Add (naked 3) (tagged 4) = None.
Proof. reflexivity. Qed.

Definition int_shift (kind : standard_int) (op : int_shift_op)
    (a amount : value) : option value :=
  eval_prim empty_static_env (Binary (Int_shift kind op)) [a; amount].

Example lsl_tagged :
  int_shift Standard_tagged_immediate Lsl (tagged 12) (naked 2)
  = Some (tagged 48).
Proof. reflexivity. Qed.

Example lsl_wraps :
  int_shift Standard_tagged_immediate Lsl (tagged 1) (naked 62)
  = Some (tagged min_int63).
Proof. reflexivity. Qed.

Example lsr_of_negative :
  int_shift Standard_tagged_immediate Lsr (tagged (-1)) (naked 1)
  = Some (tagged max_int63).
Proof. reflexivity. Qed.

Example lsr_by_zero :
  int_shift Standard_tagged_immediate Lsr (tagged (-5)) (naked 0)
  = Some (tagged (-5)).
Proof. reflexivity. Qed.

Example asr_keeps_sign :
  int_shift Standard_tagged_immediate Asr (tagged (-7)) (naked 1)
  = Some (tagged (-4)).
Proof. reflexivity. Qed.

Example asr_tagged_by_63 :
  int_shift Standard_tagged_immediate Asr (tagged (-7)) (naked 63)
  = Some (tagged (-1)).
Proof. reflexivity. Qed.

Example asr_naked_immediate_by_63 :
  int_shift Standard_naked_immediate Asr (naked (-7)) (naked 63) = None.
Proof. reflexivity. Qed.

Example lsl_tagged_by_64 :
  int_shift Standard_tagged_immediate Lsl (tagged 1) (naked 64) = None.
Proof. reflexivity. Qed.

Example lsl_by_negative_amount :
  int_shift Standard_tagged_immediate Lsl (tagged 1) (naked (-1)) = None.
Proof. reflexivity. Qed.

Example lsl_int8_wraps :
  int_shift Standard_naked_int8 Lsl (Value_const (Naked_int8 1)) (naked 7)
  = Some (Value_const (Naked_int8 (-128))).
Proof. reflexivity. Qed.

Example lsr_int8 :
  int_shift Standard_naked_int8 Lsr (Value_const (Naked_int8 (-128)))
    (naked 7)
  = Some (Value_const (Naked_int8 1)).
Proof. reflexivity. Qed.

Example lsl_int8_by_8 :
  int_shift Standard_naked_int8 Lsl (Value_const (Naked_int8 1)) (naked 8)
  = None.
Proof. reflexivity. Qed.

Example shift_by_tagged_amount :
  int_shift Standard_tagged_immediate Lsl (tagged 1) (tagged 2) = None.
Proof. reflexivity. Qed.

Example lt_signed :
  int_comp Standard_tagged_immediate (Yielding_bool (Int_lt Signed))
    (tagged (-1)) (tagged 0)
  = Some (naked 1).
Proof. reflexivity. Qed.

Example lt_unsigned :
  int_comp Standard_tagged_immediate (Yielding_bool (Int_lt Unsigned))
    (tagged (-1)) (tagged 0)
  = Some (naked 0).
Proof. reflexivity. Qed.

Example compare_like_functions :
  int_comp Standard_naked_int64 (Yielding_int_like_compare_functions Signed)
    (Value_const (Naked_int64 5)) (Value_const (Naked_int64 9))
  = Some (naked (-1)).
Proof. reflexivity. Qed.

Definition phys_equal (op : equality_comparison) (a b : value)
    : option value :=
  eval_prim empty_static_env (Binary (Phys_equal op)) [a; b].

Example phys_eq_same_immediate :
  phys_equal Phys_eq (tagged 5) (tagged 5) = Some (naked 1).
Proof. reflexivity. Qed.

Example phys_eq_other_immediate :
  phys_equal Phys_eq (tagged 5) (tagged (-5)) = Some (naked 0).
Proof. reflexivity. Qed.

Example phys_neq_same_immediate :
  phys_equal Phys_neq (tagged 5) (tagged 5) = Some (naked 0).
Proof. reflexivity. Qed.

Example phys_neq_other_immediate :
  phys_equal Phys_neq (tagged min_int63) (tagged max_int63) = Some (naked 1).
Proof. reflexivity. Qed.

Example phys_eq_out_of_range :
  phys_equal Phys_eq (tagged (max_int63 + 1)) (tagged (max_int63 + 1)) = None.
Proof. reflexivity. Qed.

Example phys_eq_naked_immediates :
  phys_equal Phys_eq (naked 5) (naked 5) = None.
Proof. reflexivity. Qed.

Example phys_eq_null :
  phys_equal Phys_eq (Value_const Null) (Value_const Null) = None.
Proof. reflexivity. Qed.

Example phys_eq_poison :
  phys_equal Phys_eq (Value_const (Poison Value)) (tagged 5) = None.
Proof. reflexivity. Qed.

Example phys_eq_blocks :
  phys_equal Phys_eq (Value_block 0 [tagged 1]) (Value_block 0 [tagged 1])
  = None.
Proof. reflexivity. Qed.

Example phys_eq_unbound_symbol :
  phys_equal Phys_eq (Value_symbol 1%positive) (Value_symbol 1%positive)
  = None.
Proof. reflexivity. Qed.

Example phys_eq_immediate_and_block :
  phys_equal Phys_eq (tagged 0) (Value_block 0 []) = Some (naked 0).
Proof. reflexivity. Qed.

Example phys_neq_closure_and_immediate :
  phys_equal Phys_neq (Value_closure 1%positive []) (tagged 0)
  = Some (naked 1).
Proof. reflexivity. Qed.

Example phys_eq_immediate_and_static_block :
  eval_prim a_block (Binary (Phys_equal Phys_eq))
    [tagged 10; Value_symbol 1%positive]
  = Some (naked 0).
Proof. reflexivity. Qed.

Example phys_eq_immediate_and_unbound_symbol :
  phys_equal Phys_eq (tagged 10) (Value_symbol 1%positive) = None.
Proof. reflexivity. Qed.

Example phys_eq_immediate_and_null :
  phys_equal Phys_eq (tagged 0) (Value_const Null) = None.
Proof. reflexivity. Qed.

Example phys_eq_immediate_and_naked_immediate :
  phys_equal Phys_eq (tagged 0) (naked 0) = None.
Proof. reflexivity. Qed.

Example phys_eq_static_block_with_itself :
  eval_prim a_block (Binary (Phys_equal Phys_eq))
    [Value_symbol 1%positive; Value_symbol 1%positive]
  = None.
Proof. reflexivity. Qed.

Example tag_immediate :
  unary empty_static_env Tag_immediate (naked 5) = Some (tagged 5).
Proof. reflexivity. Qed.

Example untag_immediate :
  unary empty_static_env Untag_immediate (tagged 5) = Some (naked 5).
Proof. reflexivity. Qed.

(* The word of a tagged immediate, and the tagged immediate of a word. *)
Definition reinterpret (r : reinterpret_64_bit_word) (v : value)
    : option value :=
  unary empty_static_env (Reinterpret_64_bit_word r) v.

Example tagged_as_int64 :
  reinterpret Tagged_int63_as_unboxed_int64 (tagged 5)
  = Some (Value_const (Naked_int64 11)).
Proof. reflexivity. Qed.

Example tagged_as_int64_negative :
  reinterpret Tagged_int63_as_unboxed_int64 (tagged (-1))
  = Some (Value_const (Naked_int64 (-1))).
Proof. reflexivity. Qed.

Example tagged_as_int64_of_least :
  reinterpret Tagged_int63_as_unboxed_int64 (tagged min_int63)
  = Some (Value_const (Naked_int64 (-9223372036854775807))).
Proof. reflexivity. Qed.

Example tagged_as_int64_of_naked :
  reinterpret Tagged_int63_as_unboxed_int64 (naked 5) = None.
Proof. reflexivity. Qed.

Example int64_as_tagged_odd :
  reinterpret Unboxed_int64_as_tagged_int63 (Value_const (Naked_int64 11))
  = Some (tagged 5).
Proof. reflexivity. Qed.

Example int64_as_tagged_even :
  reinterpret Unboxed_int64_as_tagged_int63 (Value_const (Naked_int64 10))
  = Some (tagged 5).
Proof. reflexivity. Qed.

Example int64_as_tagged_negative :
  reinterpret Unboxed_int64_as_tagged_int63 (Value_const (Naked_int64 (-3)))
  = Some (tagged (-2)).
Proof. reflexivity. Qed.

Example int64_as_tagged_of_least :
  reinterpret Unboxed_int64_as_tagged_int63
    (Value_const (Naked_int64 (-9223372036854775808)))
  = Some (tagged min_int63).
Proof. reflexivity. Qed.

Example int64_as_tagged_of_greatest :
  reinterpret Unboxed_int64_as_tagged_int63
    (Value_const (Naked_int64 9223372036854775807))
  = Some (tagged max_int63).
Proof. reflexivity. Qed.

Example int64_as_tagged_of_tagged :
  reinterpret Unboxed_int64_as_tagged_int63 (tagged 11) = None.
Proof. reflexivity. Qed.

Example not_tagged :
  unary empty_static_env (Unary_int_arith Standard_tagged_immediate Not)
    (tagged 5)
  = Some (tagged (-6)).
Proof. reflexivity. Qed.

Example not_of_least :
  unary empty_static_env (Unary_int_arith Standard_tagged_immediate Not)
    (tagged min_int63)
  = Some (tagged max_int63).
Proof. reflexivity. Qed.

Example not_int8 :
  unary empty_static_env (Unary_int_arith Standard_naked_int8 Not)
    (Value_const (Naked_int8 (-1)))
  = Some (Value_const (Naked_int8 0)).
Proof. reflexivity. Qed.

Example not_wrong_kind :
  unary empty_static_env (Unary_int_arith Standard_tagged_immediate Not)
    (naked 5)
  = None.
Proof. reflexivity. Qed.

(* The swap of bytes. The values are those of [bswap16] and the like on
   the same integers. *)
Definition swap (kind : standard_int) (v : value) : option value :=
  unary empty_static_env (Unary_int_arith kind Swap_byte_endianness) v.

Example swap_tagged :
  swap Standard_tagged_immediate (tagged 4660) = Some (tagged 13330).
Proof. reflexivity. Qed.

Example swap_tagged_drops_the_high_bytes :
  swap Standard_tagged_immediate (tagged 1193046) = Some (tagged 22068).
Proof. reflexivity. Qed.

Example swap_tagged_negative :
  swap Standard_tagged_immediate (tagged (-1)) = Some (tagged 65535).
Proof. reflexivity. Qed.

Example swap_naked_immediate :
  swap Standard_naked_immediate (naked (-256)) = Some (naked 255).
Proof. reflexivity. Qed.

Example swap_int8 :
  swap Standard_naked_int8 (Value_const (Naked_int8 (-5)))
  = Some (Value_const (Naked_int8 (-5))).
Proof. reflexivity. Qed.

Example swap_int16_changes_the_sign :
  swap Standard_naked_int16 (Value_const (Naked_int16 128))
  = Some (Value_const (Naked_int16 (-32768))).
Proof. reflexivity. Qed.

Example swap_int16_negative :
  swap Standard_naked_int16 (Value_const (Naked_int16 (-2)))
  = Some (Value_const (Naked_int16 (-257))).
Proof. reflexivity. Qed.

Example swap_int32 :
  swap Standard_naked_int32 (Value_const (Naked_int32 305419896))
  = Some (Value_const (Naked_int32 2018915346)).
Proof. reflexivity. Qed.

Example swap_int32_changes_the_sign :
  swap Standard_naked_int32 (Value_const (Naked_int32 128))
  = Some (Value_const (Naked_int32 (-2147483648))).
Proof. reflexivity. Qed.

Example swap_int32_negative :
  swap Standard_naked_int32 (Value_const (Naked_int32 (-2)))
  = Some (Value_const (Naked_int32 (-16777217))).
Proof. reflexivity. Qed.

Example swap_int64 :
  swap Standard_naked_int64 (Value_const (Naked_int64 72623859790382856))
  = Some (Value_const (Naked_int64 578437695752307201)).
Proof. reflexivity. Qed.

Example swap_int64_of_one :
  swap Standard_naked_int64 (Value_const (Naked_int64 1))
  = Some (Value_const (Naked_int64 72057594037927936)).
Proof. reflexivity. Qed.

Example swap_nativeint_negative :
  swap Standard_naked_nativeint (Value_const (Naked_nativeint (-2)))
  = Some (Value_const (Naked_nativeint (-72057594037927937))).
Proof. reflexivity. Qed.

Example swap_wrong_kind :
  swap Standard_naked_int32 (Value_const (Naked_int64 1)) = None.
Proof. reflexivity. Qed.

Example swap_out_of_range :
  swap Standard_naked_int16 (Value_const (Naked_int16 32768)) = None.
Proof. reflexivity. Qed.

Definition num_conv (src dst : standard_int) (v : value) : option value :=
  unary empty_static_env (Num_conv src dst) v.

Example num_conv_immediate_to_int64 :
  num_conv Standard_naked_immediate Standard_naked_int64 (naked min_int63)
  = Some (Value_const (Naked_int64 min_int63)).
Proof. reflexivity. Qed.

Example num_conv_int64_to_immediate_wraps :
  num_conv Standard_naked_int64 Standard_naked_immediate
    (Value_const (Naked_int64 (max_int63 + 1)))
  = Some (naked min_int63).
Proof. reflexivity. Qed.

Example num_conv_tagged_to_int8_wraps :
  num_conv Standard_tagged_immediate Standard_naked_int8 (tagged 200)
  = Some (Value_const (Naked_int8 (-56))).
Proof. reflexivity. Qed.

Example num_conv_int8_to_tagged :
  num_conv Standard_naked_int8 Standard_tagged_immediate
    (Value_const (Naked_int8 (-56)))
  = Some (tagged (-56)).
Proof. reflexivity. Qed.

Example num_conv_int64_to_int32_wraps :
  num_conv Standard_naked_int64 Standard_naked_int32
    (Value_const (Naked_int64 4294967295))
  = Some (Value_const (Naked_int32 (-1))).
Proof. reflexivity. Qed.

Example num_conv_nativeint_to_int64 :
  num_conv Standard_naked_nativeint Standard_naked_int64
    (Value_const (Naked_nativeint (-9223372036854775808)))
  = Some (Value_const (Naked_int64 (-9223372036854775808))).
Proof. reflexivity. Qed.

Example num_conv_same_kind :
  num_conv Standard_naked_int16 Standard_naked_int16
    (Value_const (Naked_int16 (-32768)))
  = Some (Value_const (Naked_int16 (-32768))).
Proof. reflexivity. Qed.

Example num_conv_wrong_kind :
  num_conv Standard_naked_immediate Standard_naked_int64 (tagged 5) = None.
Proof. reflexivity. Qed.

Example num_conv_out_of_range :
  num_conv Standard_naked_int8 Standard_naked_int64
    (Value_const (Naked_int8 200))
  = None.
Proof. reflexivity. Qed.

Example num_conv_of_block :
  num_conv Standard_tagged_immediate Standard_naked_int64 (Value_block 0 [])
  = None.
Proof. reflexivity. Qed.

Example boolean_not_of_false :
  unary empty_static_env Boolean_not (tagged 0) = Some (tagged 1).
Proof. reflexivity. Qed.

Example boolean_not_of_true :
  unary empty_static_env Boolean_not (tagged 1) = Some (tagged 0).
Proof. reflexivity. Qed.

Example boolean_not_of_other_integer :
  unary empty_static_env Boolean_not (tagged 2) = None.
Proof. reflexivity. Qed.

Example boolean_not_of_naked_immediate :
  unary empty_static_env Boolean_not (naked 0) = None.
Proof. reflexivity. Qed.

Example boolean_not_of_block :
  unary empty_static_env Boolean_not (Value_block 0 []) = None.
Proof. reflexivity. Qed.

Example is_int_of_immediate :
  unary empty_static_env (Is_int true) (tagged 5) = Some (naked 1).
Proof. reflexivity. Qed.

Example is_int_of_static_block :
  unary a_block (Is_int true) (Value_symbol 1%positive) = Some (naked 0).
Proof. reflexivity. Qed.

Example is_int_of_closure_of_variant :
  unary empty_static_env (Is_int true) (Value_closure 1%positive []) = None.
Proof. reflexivity. Qed.

Example is_int_of_closure :
  unary empty_static_env (Is_int false) (Value_closure 1%positive [])
  = Some (naked 0).
Proof. reflexivity. Qed.

Example is_int_of_unbound_symbol :
  unary empty_static_env (Is_int false) (Value_symbol 1%positive) = None.
Proof. reflexivity. Qed.

Example is_null_of_null :
  unary empty_static_env Is_null (Value_const Null) = Some (naked 1).
Proof. reflexivity. Qed.

Example is_null_of_immediate :
  unary empty_static_env Is_null (tagged 0) = Some (naked 0).
Proof. reflexivity. Qed.

Example get_tag_of_static_block :
  unary a_block Get_tag (Value_symbol 1%positive) = Some (naked 3).
Proof. reflexivity. Qed.

Example get_tag_of_immediate :
  unary a_block Get_tag (tagged 0) = None.
Proof. reflexivity. Qed.

Definition a_boxed_int64 : static_env :=
  Static_env []
    [(1%positive, Static_boxed_number (Value_const (Naked_int64 5)))].

(* A boxed poison is a boxed number, and allocated, but no primitive
   reads the number in it or makes one. *)
Definition a_boxed_poison : static_env :=
  Static_env []
    [(1%positive,
      Static_boxed_number
        (Value_const (Poison (Naked_number Kind_naked_float))))].

Example boxed_poison_is_a_boxed_number :
  boxed_number_of a_boxed_poison (Value_symbol 1%positive)
  = Some (Poison (Naked_number Kind_naked_float)).
Proof. reflexivity. Qed.

Example is_null_of_boxed_poison :
  unary a_boxed_poison Is_null (Value_symbol 1%positive) = Some (naked 0).
Proof. reflexivity. Qed.

Example unbox_boxed_poison :
  unary a_boxed_poison (Unbox_number Boxable_naked_float)
    (Value_symbol 1%positive)
  = None.
Proof. reflexivity. Qed.

Example box_poison :
  unary empty_static_env (Box_number Boxable_naked_float)
    (Value_const (Poison (Naked_number Kind_naked_float)))
  = None.
Proof. reflexivity. Qed.

Example box_number :
  unary empty_static_env (Box_number Boxable_naked_int64)
    (Value_const (Naked_int64 5))
  = Some (Value_boxed_number (Naked_int64 5)).
Proof. reflexivity. Qed.

Example box_number_wrong_kind :
  unary empty_static_env (Box_number Boxable_naked_int64)
    (Value_const (Naked_int32 5))
  = None.
Proof. reflexivity. Qed.

Example unbox_number :
  unary empty_static_env (Unbox_number Boxable_naked_int64)
    (Value_boxed_number (Naked_int64 5))
  = Some (Value_const (Naked_int64 5)).
Proof. reflexivity. Qed.

Example unbox_static_number :
  unary a_boxed_int64 (Unbox_number Boxable_naked_int64)
    (Value_symbol 1%positive)
  = Some (Value_const (Naked_int64 5)).
Proof. reflexivity. Qed.

Example unbox_number_of_other_kind :
  unary a_boxed_int64 (Unbox_number Boxable_naked_float)
    (Value_symbol 1%positive)
  = None.
Proof. reflexivity. Qed.

Example make_block :
  eval_prim empty_static_env (Variadic (Make_block (Block_kind_values 3)))
    [tagged 10; tagged 20]
  = Some (Value_block 3 [tagged 10; tagged 20]).
Proof. reflexivity. Qed.

Example make_block_of_naked_number :
  eval_prim empty_static_env (Variadic (Make_block (Block_kind_values 3)))
    [tagged 10; naked 20]
  = None.
Proof. reflexivity. Qed.

Example make_block_of_symbol :
  eval_prim a_block (Variadic (Make_block (Block_kind_values 3)))
    [tagged 10; Value_symbol 1%positive]
  = Some (Value_block 3 [tagged 10; Value_symbol 1%positive]).
Proof. reflexivity. Qed.

Example make_block_of_unbound_symbol :
  eval_prim empty_static_env (Variadic (Make_block (Block_kind_values 3)))
    [tagged 10; Value_symbol 1%positive]
  = None.
Proof. reflexivity. Qed.

Definition load (tag size : option Z) (field : Z) : unary_primitive :=
  Block_load (Block_access_values tag size Any_value) Immutable field.

Example block_load :
  unary empty_static_env (load (Some 3) (Some 2) 1)
    (Value_block 3 [tagged 10; tagged 20])
  = Some (tagged 20).
Proof. reflexivity. Qed.

Example block_load_from_static_block :
  unary a_block (load None None 0) (Value_symbol 1%positive)
  = Some (tagged 10).
Proof. reflexivity. Qed.

Example block_load_marked_mutable :
  unary empty_static_env
    (Block_load (Block_access_values None None Any_value) Mutable 0)
    (Value_block 0 [tagged 10; tagged 20])
  = Some (tagged 10).
Proof. reflexivity. Qed.

Example block_load_marked_mutable_from_constant :
  unary empty_static_env
    (Block_load (Block_access_values None None Any_value) Mutable 0)
    (tagged 10)
  = None.
Proof. reflexivity. Qed.

Example block_load_out_of_bounds :
  unary a_block (load None None 2) (Value_symbol 1%positive) = None.
Proof. reflexivity. Qed.

Example block_load_negative_field :
  unary a_block (load None None (-1)) (Value_symbol 1%positive) = None.
Proof. reflexivity. Qed.

Example block_load_wrong_tag :
  unary a_block (load (Some 0) None 0) (Value_symbol 1%positive) = None.
Proof. reflexivity. Qed.

Example block_load_wrong_size :
  unary a_block (load None (Some 3) 0) (Value_symbol 1%positive) = None.
Proof. reflexivity. Qed.

Example block_load_from_closure :
  unary empty_static_env (load None None 0) (Value_closure 1%positive [])
  = None.
Proof. reflexivity. Qed.

Definition a_string : static_env :=
  Static_env [] [(1%positive, Static_string [104; 101; 108; 108; 111])].

Example string_length :
  unary a_string String_length (Value_symbol 1%positive) = Some (naked 5).
Proof. reflexivity. Qed.

Example string_length_of_empty_string :
  unary (Static_env [] [(1%positive, Static_string [])]) String_length
    (Value_symbol 1%positive)
  = Some (naked 0).
Proof. reflexivity. Qed.

Definition string_load (statics : static_env) (width : string_accessor_width)
    (v : value) (index : Z) : option value :=
  eval_prim statics (Binary (String_or_bigstring_load width))
    [v; Value_const (Naked_nativeint index)].

Example string_load_byte :
  string_load a_string Eight (Value_symbol 1%positive) 1 = Some (naked 101).
Proof. reflexivity. Qed.

Example string_load_last_byte :
  string_load a_string Eight (Value_symbol 1%positive) 4 = Some (naked 111).
Proof. reflexivity. Qed.

Example string_load_signed_byte :
  string_load (Static_env [] [(1%positive, Static_string [200])])
    Eight_signed (Value_symbol 1%positive) 0
  = Some (Value_const (Naked_int8 (-56))).
Proof. reflexivity. Qed.

Example string_load_past_the_end :
  string_load a_string Eight (Value_symbol 1%positive) 5 = None.
Proof. reflexivity. Qed.

Example string_load_negative_index :
  string_load a_string Eight (Value_symbol 1%positive) (-1) = None.
Proof. reflexivity. Qed.

Example string_load_from_block :
  string_load a_block Eight (Value_symbol 1%positive) 0 = None.
Proof. reflexivity. Qed.

Example string_load_tagged_index :
  eval_prim a_string (Binary (String_or_bigstring_load Eight))
    [Value_symbol 1%positive; tagged 1]
  = None.
Proof. reflexivity. Qed.

(* The loads of more than one byte: the least significant byte is the
   first. The values are those that [Bytes.get_uint16_le] and the like
   give on the same bytes. *)
Example string_load_two_bytes :
  string_load a_string Sixteen (Value_symbol 1%positive) 1
  = Some (naked 27749).
Proof. reflexivity. Qed.

Example string_load_last_two_bytes :
  string_load a_string Sixteen (Value_symbol 1%positive) 3
  = Some (naked 28524).
Proof. reflexivity. Qed.

Example string_load_two_bytes_past_the_end :
  string_load a_string Sixteen (Value_symbol 1%positive) 4 = None.
Proof. reflexivity. Qed.

Example string_load_four_bytes :
  string_load a_string Thirty_two (Value_symbol 1%positive) 1
  = Some (Value_const (Naked_int32 1869376613)).
Proof. reflexivity. Qed.

Example string_load_eight_bytes_of_five :
  string_load a_string Sixty_four (Value_symbol 1%positive) 0 = None.
Proof. reflexivity. Qed.

Definition some_bytes : static_env :=
  Static_env []
    [(1%positive,
      Static_string
        [0; 128; 255; 255; 255; 255; 255; 255; 255; 127; 1; 2; 3; 4; 5; 6;
         7; 200])].

Example string_load_two_bytes_unsigned :
  string_load some_bytes Sixteen (Value_symbol 1%positive) 0
  = Some (naked 32768).
Proof. reflexivity. Qed.

Example string_load_two_bytes_signed :
  string_load some_bytes Sixteen_signed (Value_symbol 1%positive) 0
  = Some (Value_const (Naked_int16 (-32768))).
Proof. reflexivity. Qed.

Example string_load_four_bytes_negative :
  string_load some_bytes Thirty_two (Value_symbol 1%positive) 2
  = Some (Value_const (Naked_int32 (-1))).
Proof. reflexivity. Qed.

Example string_load_last_four_bytes :
  string_load some_bytes Thirty_two (Value_symbol 1%positive) 14
  = Some (Value_const (Naked_int32 (-939063803))).
Proof. reflexivity. Qed.

Example string_load_eight_bytes :
  string_load some_bytes Sixty_four (Value_symbol 1%positive) 2
  = Some (Value_const (Naked_int64 9223372036854775807)).
Proof. reflexivity. Qed.

Example string_load_eight_bytes_negative :
  string_load some_bytes Sixty_four (Value_symbol 1%positive) 1
  = Some (Value_const (Naked_int64 (-128))).
Proof. reflexivity. Qed.

Example string_load_last_eight_bytes :
  string_load some_bytes Sixty_four (Value_symbol 1%positive) 10
  = Some (Value_const (Naked_int64 (-4033248322675080703))).
Proof. reflexivity. Qed.

Example string_load_eight_bytes_past_the_end :
  string_load some_bytes Sixty_four (Value_symbol 1%positive) 11 = None.
Proof. reflexivity. Qed.

(* An index that no string has, however long: the test of the index is
   on integers, so that nothing is counted up to it. *)
Example string_load_far_past_the_end :
  string_load a_string Sixteen (Value_symbol 1%positive)
    4611686018427387904
  = None.
Proof. reflexivity. Qed.

Example string_length_of_block :
  unary a_block String_length (Value_symbol 1%positive) = None.
Proof. reflexivity. Qed.

Example string_length_of_immediate :
  unary a_string String_length (tagged 5) = None.
Proof. reflexivity. Qed.

Example string_length_of_unbound_symbol :
  unary empty_static_env String_length (Value_symbol 1%positive) = None.
Proof. reflexivity. Qed.

Example get_tag_of_made_block :
  unary empty_static_env Get_tag (Value_block 3 []) = Some (naked 3).
Proof. reflexivity. Qed.

Example opaque_identity :
  unary a_block (Opaque_identity Value) (Value_symbol 1%positive)
  = Some (Value_symbol 1%positive).
Proof. reflexivity. Qed.

Example opaque_identity_of_unbound_symbol :
  unary empty_static_env (Opaque_identity Value) (Value_symbol 1%positive)
  = None.
Proof. reflexivity. Qed.

Example opaque_identity_wrong_kind :
  unary empty_static_env (Opaque_identity Value) (naked 1) = None.
Proof. reflexivity. Qed.

(* The functions of arrays. *)
Definition a_block_value : value := Value_block 0 [].

Example array_made_of_immediates :
  array_made Array_immediates [tagged 1; tagged 2]
  = Some [tagged 1; tagged 2].
Proof. reflexivity. Qed.

Example array_made_of_immediates_with_a_block :
  array_made Array_immediates [tagged 1; a_block_value] = None.
Proof. reflexivity. Qed.

Example array_made_of_values_with_a_block :
  array_made Array_values [tagged 1; a_block_value]
  = Some [tagged 1; a_block_value].
Proof. reflexivity. Qed.

Example array_made_of_values_with_a_naked_number :
  array_made Array_gc_ignorable_values [tagged 1; naked 2] = None.
Proof. reflexivity. Qed.

Example array_length_of_three :
  array_length_of [tagged 5; tagged 6; tagged 7] = Some (tagged 3).
Proof. reflexivity. Qed.

Example array_loaded_second :
  array_loaded Load_immediates [tagged 5; tagged 6; tagged 7] (tagged 1)
  = Some (tagged 6).
Proof. reflexivity. Qed.

Example array_loaded_past_the_end :
  array_loaded Load_immediates [tagged 5; tagged 6; tagged 7] (tagged 3)
  = None.
Proof. reflexivity. Qed.

Example array_loaded_before_the_start :
  array_loaded Load_values [tagged 5] (tagged (-1)) = None.
Proof. reflexivity. Qed.

Example array_loaded_at_a_naked_index :
  array_loaded Load_values [tagged 5] (naked 0) = None.
Proof. reflexivity. Qed.

Example array_loaded_a_block_as_an_immediate :
  array_loaded Load_immediates [a_block_value] (tagged 0) = None.
Proof. reflexivity. Qed.

Example array_loaded_a_block_as_a_value :
  array_loaded Load_values [a_block_value] (tagged 0) = Some a_block_value.
Proof. reflexivity. Qed.

Example array_stored_second :
  array_stored Set_gc_ignorable_values Array_immediates [tagged 5; tagged 6]
    (tagged 1) (tagged 9)
  = Some [tagged 5; tagged 9].
Proof. reflexivity. Qed.

Example array_stored_a_block_among_immediates :
  array_stored (Set_values Assignment) Array_immediates [tagged 5] (tagged 0)
    a_block_value
  = None.
Proof. reflexivity. Qed.

Example array_stored_a_block_among_values :
  array_stored (Set_values Assignment) Array_values [tagged 5] (tagged 0)
    a_block_value
  = Some [a_block_value].
Proof. reflexivity. Qed.

Example array_stored_a_block_as_an_immediate :
  array_stored Set_immediates Array_values [tagged 5] (tagged 0)
    a_block_value
  = None.
Proof. reflexivity. Qed.

Example array_stored_past_the_end :
  array_stored Set_immediates Array_immediates [tagged 5] (tagged 1)
    (tagged 9)
  = None.
Proof. reflexivity. Qed.

(* A static immutable array, read through its symbol. *)
Definition a_table : static_env :=
  Static_env []
    [(1%positive,
      Static_immutable_array [tagged 8; tagged 6; Value_symbol 2%positive]);
     (2%positive, Static_block 0 [])].

Definition table_load (load_kind : array_load_kind) (v index : value)
    : option value :=
  eval_prim a_table (Binary (Array_load Array_values load_kind Immutable))
    [v; index].

Example static_array_length :
  unary a_table (Array_length (Array_kind Array_values))
    (Value_symbol 1%positive)
  = Some (tagged 3).
Proof. reflexivity. Qed.

Example static_array_length_of_block :
  unary a_table (Array_length (Array_kind Array_values))
    (Value_symbol 2%positive)
  = None.
Proof. reflexivity. Qed.

Example static_array_load :
  table_load Load_immediates (Value_symbol 1%positive) (tagged 1)
  = Some (tagged 6).
Proof. reflexivity. Qed.

(* The load kind of the access decides, not the one that the array was
   made for: the same element at [Load_values]. *)
Example static_array_load_an_immediate_as_a_value :
  table_load Load_values (Value_symbol 1%positive) (tagged 1)
  = Some (tagged 6).
Proof. reflexivity. Qed.

Example static_array_load_a_symbol :
  table_load Load_values (Value_symbol 1%positive) (tagged 2)
  = Some (Value_symbol 2%positive).
Proof. reflexivity. Qed.

Example static_array_load_a_symbol_as_an_immediate :
  table_load Load_immediates (Value_symbol 1%positive) (tagged 2) = None.
Proof. reflexivity. Qed.

Example static_array_load_past_the_end :
  table_load Load_values (Value_symbol 1%positive) (tagged 3) = None.
Proof. reflexivity. Qed.

Example static_array_load_at_a_naked_index :
  table_load Load_values (Value_symbol 1%positive) (naked 0) = None.
Proof. reflexivity. Qed.

Example static_array_load_from_block :
  table_load Load_values (Value_symbol 2%positive) (tagged 0) = None.
Proof. reflexivity. Qed.

Example block_load_from_static_array :
  unary a_table (load None None 0) (Value_symbol 1%positive) = None.
Proof. reflexivity. Qed.

Example get_tag_of_static_array :
  unary a_table Get_tag (Value_symbol 1%positive) = None.
Proof. reflexivity. Qed.

Example is_int_of_static_array :
  unary a_table (Is_int false) (Value_symbol 1%positive) = Some (naked 0).
Proof. reflexivity. Qed.

Example is_null_of_static_array :
  unary a_table Is_null (Value_symbol 1%positive) = Some (naked 0).
Proof. reflexivity. Qed.

(* Two allocated values: no value, as for two static blocks. *)
Example phys_equal_of_static_array_and_block :
  eval_prim a_table (Binary (Phys_equal Phys_eq))
    [Value_symbol 1%positive; Value_symbol 2%positive]
  = None.
Proof. reflexivity. Qed.

Example phys_equal_of_static_array_and_immediate :
  eval_prim a_table (Binary (Phys_equal Phys_eq))
    [Value_symbol 1%positive; tagged 8]
  = Some (naked 0).
Proof. reflexivity. Qed.

(* An immutable array made as the term runs: the array of the
   arguments, each a value that the array kind holds and none a symbol
   bound to nothing. It is read as a static one is. *)
Definition an_array_value : value :=
  Value_immutable_array [tagged 8; tagged 6; Value_symbol 2%positive].

Example make_immutable_array :
  eval_prim a_table (Variadic (Make_immutable_array Array_values))
    [tagged 8; tagged 6; Value_symbol 2%positive]
  = Some an_array_value.
Proof. reflexivity. Qed.

Example make_immutable_array_of_nothing :
  eval_prim empty_static_env (Variadic (Make_immutable_array Array_values))
    []
  = Some (Value_immutable_array []).
Proof. reflexivity. Qed.

Example make_immutable_array_of_immediates_with_a_symbol :
  eval_prim a_table (Variadic (Make_immutable_array Array_immediates))
    [tagged 8; Value_symbol 2%positive]
  = None.
Proof. reflexivity. Qed.

Example make_immutable_array_of_a_naked_number :
  eval_prim empty_static_env (Variadic (Make_immutable_array Array_values))
    [naked 2]
  = None.
Proof. reflexivity. Qed.

Example make_immutable_array_of_an_unbound_symbol :
  eval_prim empty_static_env (Variadic (Make_immutable_array Array_values))
    [Value_symbol 2%positive]
  = None.
Proof. reflexivity. Qed.

Example array_value_length :
  unary empty_static_env (Array_length (Array_kind Array_values))
    an_array_value
  = Some (tagged 3).
Proof. reflexivity. Qed.

Example array_value_load :
  table_load Load_immediates an_array_value (tagged 1) = Some (tagged 6).
Proof. reflexivity. Qed.

Example array_value_load_a_symbol :
  table_load Load_values an_array_value (tagged 2)
  = Some (Value_symbol 2%positive).
Proof. reflexivity. Qed.

Example array_value_load_a_symbol_as_an_immediate :
  table_load Load_immediates an_array_value (tagged 2) = None.
Proof. reflexivity. Qed.

Example array_value_load_past_the_end :
  table_load Load_values an_array_value (tagged 3) = None.
Proof. reflexivity. Qed.

Example block_load_from_array_value :
  unary a_table (load None None 0) an_array_value = None.
Proof. reflexivity. Qed.

Example get_tag_of_array_value :
  unary a_table Get_tag an_array_value = None.
Proof. reflexivity. Qed.

Example is_int_of_array_value :
  unary a_table (Is_int false) an_array_value = Some (naked 0).
Proof. reflexivity. Qed.

Example is_null_of_array_value :
  unary a_table Is_null an_array_value = Some (naked 0).
Proof. reflexivity. Qed.

(* Two allocated values: no value, as for two blocks, and as for the
   array against itself. *)
Example phys_equal_of_array_value_and_itself :
  eval_prim a_table (Binary (Phys_equal Phys_eq))
    [an_array_value; an_array_value]
  = None.
Proof. reflexivity. Qed.

Example phys_equal_of_array_value_and_immediate :
  eval_prim a_table (Binary (Phys_equal Phys_eq)) [an_array_value; tagged 8]
  = Some (naked 0).
Proof. reflexivity. Qed.

End Examples.

#[global] Opaque eval_prim.
