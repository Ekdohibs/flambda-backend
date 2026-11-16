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


(* Proofs of the statements of FlambdaTypeMeaningBoolSpec.v: each test
   answers [true] exactly when the meaning holds. Nothing here needs
   review. *)

From Stdlib Require Import ZArith List Bool Lia Setoid.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaPrimitiveProofs
  FlambdaCompareSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeMeaningSpec FlambdaTypeMeaningProofs
  FlambdaTypeMeaningBoolSpec.
Import ListNotations.

(* ======================================================================== *)
(* Equality tests of the machine                                            *)
(* ======================================================================== *)

Lemma const_eqb_eq : decides_equality const_eqb.
Proof.
  intros c c'. unfold const_eqb.
  destruct (const_eq_dec c c'); split; congruence.
Qed.

(* [lists_eqb] tells whether two lists are the same if the test tells
   whether an element of the first is the same as another. *)
Lemma lists_eqb_eq {A : Type} (eqb : A -> A -> bool) (l l' : list A) :
  (∀ a a' : A, a ∈ l -> (eqb a a' = true <-> a = a')) ->
  (lists_eqb eqb l l' = true <-> l = l').
Proof.
  revert l'. induction l as [ | a l IH]; intros [ | a' l'] Heqb; simpl;
    try (split; [ reflexivity | reflexivity ]);
    try (split; discriminate).
  rewrite andb_true_iff, (Heqb a a' (or_introl eq_refl)), IH.
  - split; [ intros [-> ->]; reflexivity | ].
    intros [= -> ->]. split; reflexivity.
  - intros b b' Hin. apply Heqb. right. exact Hin.
Qed.

(* Induction on values, which hold lists of values. *)
Section Value_induction.

Variable P : value -> Prop.

Hypothesis H_const : ∀ c : const, P (Value_const c).
Hypothesis H_closure :
  ∀ (code : code_id) (value_slots : list (value_slot * value)),
    (∀ (w : value_slot) (v : value), (w, v) ∈ value_slots -> P v) ->
    P (Value_closure code value_slots).
Hypothesis H_closure_of_set :
  ∀ (slot : function_slot) (function_decls : list (function_slot * code_id))
    (value_slots : list (value_slot * value)),
    (∀ (w : value_slot) (v : value), (w, v) ∈ value_slots -> P v) ->
    P (Value_closure_of_set slot function_decls value_slots).
Hypothesis H_symbol : ∀ s : symbol, P (Value_symbol s).
Hypothesis H_block :
  ∀ (tag : Z) (fields : list value),
    (∀ v : value, v ∈ fields -> P v) -> P (Value_block tag fields).
Hypothesis H_immutable_array :
  ∀ elements : list value,
    (∀ v : value, v ∈ elements -> P v) -> P (Value_immutable_array elements).
Hypothesis H_boxed_number : ∀ c : const, P (Value_boxed_number c).
Hypothesis H_location : ∀ l : location, P (Value_location l).
Hypothesis H_partial_application :
  ∀ (callee : value) (args : list value) (args_arity : arity),
    P callee -> (∀ v : value, v ∈ args -> P v) ->
    P (Value_partial_application callee args args_arity).

Fixpoint value_induction (v : value) : P v :=
  match v with
  | Value_const c => H_const c
  | Value_closure code value_slots =>
    H_closure code value_slots (bindings_all P value_induction value_slots)
  | Value_closure_of_set slot function_decls value_slots =>
    H_closure_of_set slot function_decls value_slots
      (bindings_all P value_induction value_slots)
  | Value_symbol s => H_symbol s
  | Value_block tag fields =>
    H_block tag fields (list_all P value_induction fields)
  | Value_immutable_array elements =>
    H_immutable_array elements (list_all P value_induction elements)
  | Value_boxed_number c => H_boxed_number c
  | Value_location l => H_location l
  | Value_partial_application callee args args_arity =>
    H_partial_application callee args args_arity (value_induction callee)
      (list_all P value_induction args)
  end.

End Value_induction.

Lemma value_eqb_eq : decides_equality value_eqb.
Proof.
  intros v.
  induction v
    as [c | code value_slots IH | slot decls value_slots IH | s
       | tag fields IH | elements IH | c | l
       | callee args args_arity IHcallee IH]
    using value_induction;
    intros [c' | code' value_slots' | slot' decls' value_slots' | s'
           | tag' fields' | elements' | c' | l'
           | callee' args' args_arity'];
    simpl; try (split; discriminate).
  - rewrite (const_eqb_eq c c'). split; [ intros ->; reflexivity | ].
    intros [= ->]. reflexivity.
  - rewrite andb_true_iff, Pos.eqb_eq, lists_eqb_eq.
    + split; [ intros [-> ->]; reflexivity | ].
      intros [= -> ->]. split; reflexivity.
    + intros [w v] [w' v'] Hin. simpl.
      rewrite andb_true_iff, Pos.eqb_eq, (IH w v Hin v').
      split; [ intros [-> ->]; reflexivity | ].
      intros [= -> ->]. split; reflexivity.
  - rewrite !andb_true_iff, Pos.eqb_eq, !lists_eqb_eq.
    + split; [ intros [[-> ->] ->]; reflexivity | ].
      intros [= -> -> ->]. repeat split.
    + intros [w v] [w' v'] Hin. simpl.
      rewrite andb_true_iff, Pos.eqb_eq, (IH w v Hin v').
      split; [ intros [-> ->]; reflexivity | ].
      intros [= -> ->]. split; reflexivity.
    + intros [g code] [g' code'] _. simpl.
      rewrite andb_true_iff, !Pos.eqb_eq.
      split; [ intros [-> ->]; reflexivity | ].
      intros [= -> ->]. split; reflexivity.
  - rewrite Pos.eqb_eq. split; [ intros ->; reflexivity | ].
    intros [= ->]. reflexivity.
  - rewrite andb_true_iff, Z.eqb_eq, lists_eqb_eq.
    + split; [ intros [-> ->]; reflexivity | ].
      intros [= -> ->]. split; reflexivity.
    + intros v v' Hin. exact (IH v Hin v').
  - rewrite lists_eqb_eq.
    + split; [ intros ->; reflexivity | ].
      intros [= ->]. reflexivity.
    + intros v v' Hin. exact (IH v Hin v').
  - rewrite (const_eqb_eq c c'). split; [ intros ->; reflexivity | ].
    intros [= ->]. reflexivity.
  - rewrite Pos.eqb_eq. split; [ intros ->; reflexivity | ].
    intros [= ->]. reflexivity.
  - assert (Harity :
      arity_eqb args_arity args_arity' = true <-> args_arity = args_arity').
    { unfold arity_eqb.
      destruct (list_eq_dec arity_component_eq_dec args_arity args_arity');
        split; congruence. }
    rewrite !andb_true_iff, (IHcallee callee'), Harity, lists_eqb_eq.
    + split; [ intros [[-> ->] ->]; reflexivity | ].
      intros [= -> -> ->]. repeat split.
    + intros v v' Hin. exact (IH v Hin v').
Qed.

Theorem value_eqb_decides_equality : value_eqb_decides_equality_statement.
Proof. exact value_eqb_eq. Qed.

(* ======================================================================== *)
(* The relations, the shapes and the indexes                                *)
(* ======================================================================== *)

Lemma is_poison_b_eq (v : value) : is_poison_b v = true <-> is_poison v.
Proof.
  unfold is_poison.
  destruct v as [c | code slots | slot decls slots | s | tag fields
    | elements | c' | l | pa_callee pa_args pa_arity];
    [ destruct c | | | | | | | | ]; simpl;
    first
      [ split; [ intros _; eexists; reflexivity | reflexivity ]
      | split; [ discriminate | intros [k Hv]; discriminate Hv ] ].
Qed.

Lemma is_tag_b_eq (n : Z) : is_tag_b n = true <-> is_tag n.
Proof.
  unfold is_tag_b, is_tag. rewrite andb_true_iff, !Z.leb_le. reflexivity.
Qed.

Lemma relation_holds_b_eq (statics : static_env) (relation : Relation.t)
    (v : value) (n : Z) :
  relation_holds_b statics relation v n = true
  <-> relation_holds statics relation v n.
Proof.
  destruct relation; unfold relation_holds_b, relation_holds,
    is_immediate, is_block.
  - destruct (null_or_not v (Z.eqb n 1)
                (kind_eqb (kind_of_value v) Value && Z.eqb n 0))
      as [[-> _] | [Hnot_null ->]].
    + simpl. rewrite Z.eqb_eq.
      split; [ intros ->; left; split; reflexivity | ].
      intros [[_ ->] | [_ [Hnot_null _]]];
        [ reflexivity | destruct (Hnot_null eq_refl) ].
    + rewrite andb_true_iff, (kind_eqb_eq _ _), Z.eqb_eq. split.
      * intros [Hkind ->]. right. auto.
      * intros [[Hnull _] | [Hkind [_ ->]]];
          [ destruct (Hnot_null Hnull) | split; [ exact Hkind | reflexivity ] ].
  - destruct (tagged_immediate_or_not v (fun _ => Z.eqb n 1)
                match block_of statics v with
                | Some _ => Z.eqb n 0
                | None => false
                end)
      as [[i [-> _]] | [Hnot_immediate ->]].
    + simpl. rewrite Z.eqb_eq. split.
      * intros ->. left. split; [ exists i; reflexivity | reflexivity ].
      * intros [[_ ->] | [[tag [fields Hblock]] _]];
          [ reflexivity | discriminate Hblock ].
    + destruct (block_of statics v) as [[tag fields] | ].
      * rewrite Z.eqb_eq. split.
        -- intros ->. right.
           split; [ exists tag, fields; reflexivity | reflexivity ].
        -- intros [[[i Hv] _] | [_ ->]];
             [ destruct (Hnot_immediate i Hv) | reflexivity ].
      * split; [ discriminate | ].
        intros [[[i Hv] _] | [[tag [fields Hblock]] _]];
          [ destruct (Hnot_immediate i Hv) | discriminate Hblock ].
  - destruct (tagged_immediate_or_not v (fun _ => true)
                match block_of statics v with
                | Some (tag, _) => Z.eqb tag n && is_tag_b n
                | None => false
                end)
      as [[i [-> _]] | [Hnot_immediate ->]].
    + simpl.
      split; [ intros _; left; exists i; reflexivity | reflexivity ].
    + destruct (block_of statics v) as [[tag fields] | ].
      * rewrite andb_true_iff, Z.eqb_eq, is_tag_b_eq. split.
        -- intros [-> Htag]. right.
           split; [ exists fields; reflexivity | exact Htag ].
        -- intros [[i Hv] | [[fields' [= -> _]] Htag]];
             [ destruct (Hnot_immediate i Hv)
             | split; [ reflexivity | exact Htag ] ].
      * split; [ discriminate | ].
        intros [[i Hv] | [[fields Hblock] _]];
          [ destruct (Hnot_immediate i Hv) | discriminate Hblock ].
Qed.

Theorem relation_holds_b_iff : relation_holds_b_iff_statement.
Proof. exact relation_holds_b_eq. Qed.

Lemma bytes_eqb_eq : decides_equality (lists_eqb Z.eqb).
Proof.
  intros l l'. apply lists_eqb_eq. intros a a' _. apply Z.eqb_eq.
Qed.

Lemma block_has_shape_b_eq (shape : block_shape) (tag : Z)
    (fields : list value) :
  block_has_shape_b shape tag fields = true
  <-> block_has_shape shape tag fields.
Proof.
  unfold block_has_shape_b, block_has_shape.
  rewrite andb_true_iff, lists_eqb_eq.
  - destruct shape as [shape | ]; [ rewrite Z.ltb_lt | rewrite Z.eqb_eq ];
      reflexivity.
  - intros k k' _. apply kind_eqb_eq.
Qed.

Lemma block_index_satisfied_b_eq (tag : Z) (fields : list value)
    (index : row_like_index nat block_shape) :
  block_index_satisfied_b tag fields index = true
  <-> block_index_satisfied tag fields index.
Proof.
  unfold block_index_satisfied_b, block_index_satisfied.
  rewrite andb_true_iff, block_has_shape_b_eq.
  destruct (index_domain index) as [size | size];
    [ rewrite Nat.eqb_eq | rewrite Nat.leb_le ]; reflexivity.
Qed.

Lemma closures_index_satisfied_b_eq
    (functions : list (function_slot * code_id))
    (value_slots : list (value_slot * value))
    (index : row_like_index set_of_closures_contents unit) :
  closures_index_satisfied_b functions value_slots index = true
  <-> closures_index_satisfied functions value_slots index.
Proof.
  unfold closures_index_satisfied_b, closures_index_satisfied.
  destruct (index_domain index) as [contents | contents];
    rewrite andb_true_iff; [ | reflexivity ].
  rewrite !(keyed_same_keys_b positive Pos.eqb Pos.eqb_eq). reflexivity.
Qed.

(* ======================================================================== *)
(* What the meaning is made of                                              *)
(* ======================================================================== *)

Lemma null_or_other (v : value) :
  v = Value_const Null ∨ v <> Value_const Null.
Proof.
  destruct (null_or_not v true false) as [[Hnull _] | [Hnot_null _]];
    [ left; exact Hnull | right; exact Hnot_null ].
Qed.

Lemma tagged_immediate_or_other (v : value) :
  (∃ i : Z, v = Value_const (Tagged_immediate i))
  ∨ (∀ i : Z, v <> Value_const (Tagged_immediate i)).
Proof.
  destruct (tagged_immediate_or_not v (fun _ => true) false)
    as [[i [Hv _]] | [Hnot _]];
    [ left; exists i; exact Hv | right; exact Hnot ].
Qed.

Lemma variable_is_b_eq (vars : env) (x : option variable) (n : Z) :
  variable_is_b vars x n = true <-> variable_is vars x n.
Proof.
  destruct x as [x | ]; simpl; [ | split; [ intros _; exact I | reflexivity ] ].
  destruct (lookup x vars) as [v | ]; [ | split; discriminate ].
  destruct (naked_immediate_or_not v (fun i => Z.eqb i n) false)
    as [[i [-> _]] | [Hnot ->]].
  - simpl. rewrite Z.eqb_eq. split; [ intros ->; reflexivity | ].
    intros [= ->]. reflexivity.
  - split; [ discriminate | ]. intros [= Hv]. destruct (Hnot n Hv).
Qed.

(* The test of a type that may be unknown, bottom or an alias is right
   if that of its head is. *)
Lemma descr_satisfied_b_reflects {head : Type} (vars : env)
    (k : flambda_kind) (head_b : head -> bool) (head_P : head -> Prop)
    (v : value) (ty : type_descr head) :
  match ty with
  | Ok (No_alias head) => head_b head = true <-> head_P head
  | _ => True
  end ->
  (descr_satisfied_b vars k head_b v ty = true
   <-> descr_satisfied vars k head_P v ty).
Proof.
  destruct ty as [ | [head' | s] | ]; simpl.
  - intros _. apply kind_eqb_eq.
  - auto.
  - intros _. destruct (eval_simple vars s) as [w | ].
    + rewrite andb_true_iff, (value_eqb_eq w v), (kind_eqb_eq _ k). split.
      * intros [-> Hkind]. auto.
      * intros [[= ->] Hkind]. auto.
    + split; [ discriminate | ]. intros [Hv _]. discriminate Hv.
  - intros _. split; [ discriminate | intros [] ].
Qed.

Lemma head_of_kind_naked_immediate_satisfied_b_eq (statics : static_env)
    (vars : env) (v : value) (head : head_of_kind_naked_immediate) :
  head_of_kind_naked_immediate_satisfied_b statics vars v head = true
  <-> head_of_kind_naked_immediate_satisfied statics vars v head.
Proof.
  unfold head_of_kind_naked_immediate_satisfied_b,
    head_of_kind_naked_immediate_satisfied.
  destruct v as [c | code slots | slot decls slots | s | tag fields
    | elements | c' | l | pa_callee pa_args pa_arity];
    [ destruct c | | | | | | | | ];
    try (split; [ discriminate | intros [? [Hv _]]; discriminate Hv ]).
  rewrite andb_true_iff, (every_b_every (Relation.t * name)%type),
    (every_in (Relation.t * name)%type).
  split.
  - intros [Hnumbers Hrelations]. eexists. split; [ reflexivity | ]. split.
    + destruct (naked_immediates head) as [numbers | ]; [ | exact I ].
      apply (keyed_among_in Z Z.eqb Z.eqb_eq). exact Hnumbers.
    + intros relation n Hin. specialize (Hrelations (relation, n) Hin).
      simpl in Hrelations.
      destruct (value_of_name vars n) as [w | ];
        [ | discriminate Hrelations ].
      exists w. split; [ reflexivity | ].
      apply relation_holds_b_eq. exact Hrelations.
  - intros [j [[= <-] [Hnumbers Hrelations]]]. split.
    + destruct (naked_immediates head) as [numbers | ]; [ | reflexivity ].
      apply (keyed_among_in Z Z.eqb Z.eqb_eq). exact Hnumbers.
    + intros [relation n] Hin. simpl.
      destruct (Hrelations relation n Hin) as [w [-> Hholds]].
      apply relation_holds_b_eq. exact Hholds.
Qed.

Lemma naked_numbers_satisfied_b_eq (k : naked_number_kind) (v : value)
    (numbers : list Z) :
  naked_numbers_satisfied_b k v numbers = true
  <-> naked_numbers_satisfied k v numbers.
Proof.
  unfold naked_numbers_satisfied_b, naked_numbers_satisfied.
  destruct v as [c | code slots | slot decls slots | s | tag fields
    | elements | c' | l | pa_callee pa_args pa_arity];
    try (split; [ discriminate | intros [? [? [Hv _]]]; discriminate Hv ]).
  rewrite andb_true_iff, (kind_eqb_eq _ _). split.
  - intros [Hkind Hnumber].
    destruct (number_of_const c) as [n | ] eqn:Hn; [ | discriminate Hnumber ].
    exists c, n. split; [ reflexivity | ]. split; [ exact Hkind | ].
    split; [ exact Hn | ].
    apply (keyed_among_in Z Z.eqb Z.eqb_eq). exact Hnumber.
  - intros [c' [n [[= <-] [Hkind [Hn Hin]]]]]. split; [ exact Hkind | ].
    rewrite Hn. apply (keyed_among_in Z Z.eqb Z.eqb_eq). exact Hin.
Qed.

Lemma fields_satisfied_b_reflects (type_b : value -> flambda_type -> bool)
    (type_P : value -> flambda_type -> Prop) (tys : list flambda_type)
    (fields : list value) :
  (∀ ty : flambda_type,
    ty ∈ tys -> ∀ field : value, type_b field ty = true <-> type_P field ty) ->
  (fields_satisfied_b type_b tys fields = true
   <-> fields_satisfied type_P tys fields).
Proof.
  revert fields. induction tys as [ | ty tys IH]; intros fields Hreflects;
    simpl.
  - split; [ intros _; exact I | reflexivity ].
  - destruct fields as [ | field fields];
      [ split; [ intros _; exact I | reflexivity ] | ].
    rewrite andb_true_iff, (Hreflects ty (or_introl eq_refl) field), IH;
      [ reflexivity | ].
    intros ty' Hin. apply Hreflects. right. exact Hin.
Qed.

Lemma block_case_satisfied_b_reflects
    (type_b : value -> flambda_type -> bool)
    (type_P : value -> flambda_type -> Prop)
    (extension_b : env_extension -> bool)
    (extension_P : env_extension -> Prop) (tag : Z) (fields : list value)
    (case : row_like_block_case) :
  (∀ ty : flambda_type,
    ty ∈ case_maps_to case ->
    ∀ field : value, type_b field ty = true <-> type_P field ty) ->
  (extension_b (case_env_extension case) = true
   <-> extension_P (case_env_extension case)) ->
  (block_case_satisfied_b type_b extension_b tag fields case = true
   <-> block_case_satisfied type_P extension_P tag fields case).
Proof.
  intros Htypes Hextension.
  unfold block_case_satisfied_b, block_case_satisfied.
  rewrite !andb_true_iff, block_index_satisfied_b_eq,
    (fields_satisfied_b_reflects _ _ _ _ Htypes), Hextension.
  tauto.
Qed.

Lemma closures_case_satisfied_b_reflects (entry_b : closures_entry -> bool)
    (entry_P : closures_entry -> Prop) (extension_b : env_extension -> bool)
    (extension_P : env_extension -> Prop)
    (functions : list (function_slot * code_id))
    (value_slots : list (value_slot * value))
    (case : row_like_closures_case) :
  (entry_b (case_maps_to case) = true <-> entry_P (case_maps_to case)) ->
  (extension_b (case_env_extension case) = true
   <-> extension_P (case_env_extension case)) ->
  (closures_case_satisfied_b entry_b extension_b functions value_slots case
   = true
   <-> closures_case_satisfied entry_P extension_P functions value_slots
         case).
Proof.
  intros Hentry Hextension.
  unfold closures_case_satisfied_b, closures_case_satisfied.
  rewrite !andb_true_iff, closures_index_satisfied_b_eq, Hentry, Hextension.
  tauto.
Qed.

(* ======================================================================== *)
(* One equation for each constructor                                        *)
(* ======================================================================== *)

(* What [type_satisfied_b] and the functions defined with it come to on
   each constructor, as for the meaning (FlambdaTypeMeaningProofs.v).
   They hold by definition. Where the test goes by cases on the value,
   there is one equation for each case. *)

Section Equations.

Variable statics : static_env.
Variable vars : env.

Lemma type_satisfied_b_value (v : value)
    (ty : type_descr head_of_kind_value) :
  type_satisfied_b statics vars v (Type_value ty)
  = descr_satisfied_b vars Value
      (head_of_kind_value_satisfied_b statics vars v) v ty.
Proof. reflexivity. Qed.

Lemma type_satisfied_b_naked_immediate (v : value)
    (ty : type_descr head_of_kind_naked_immediate) :
  type_satisfied_b statics vars v (Type_naked_immediate ty)
  = descr_satisfied_b vars (Naked_number Kind_naked_immediate)
      (head_of_kind_naked_immediate_satisfied_b statics vars v) v ty.
Proof. reflexivity. Qed.

Lemma type_satisfied_b_naked_number (v : value) (k : naked_number_kind)
    (ty : type_descr (list Z)) :
  type_satisfied_b statics vars v (Type_naked_number k ty)
  = descr_satisfied_b vars (Naked_number k) (naked_numbers_satisfied_b k v)
      v ty.
Proof. reflexivity. Qed.

Lemma type_satisfied_b_rec_info (v : value) (ty : type_descr unit) :
  type_satisfied_b statics vars v (Type_rec_info ty)
  = descr_satisfied_b vars Rec_info
      (fun _ => kind_eqb (kind_of_value v) Rec_info) v ty.
Proof. reflexivity. Qed.

Lemma type_satisfied_b_region (v : value) (ty : type_descr unit) :
  type_satisfied_b statics vars v (Type_region ty)
  = descr_satisfied_b vars Region
      (fun _ => kind_eqb (kind_of_value v) Region) v ty.
Proof. reflexivity. Qed.

Lemma head_of_kind_value_satisfied_b_null
    (non_null : or_unknown_or_bottom head_of_kind_value_non_null)
    (is_null : is_null) :
  head_of_kind_value_satisfied_b statics vars (Value_const Null)
    (Head_of_kind_value non_null is_null)
  = match is_null with
    | Not_null => false
    | Maybe_null is_null => variable_is_b vars is_null 1
    end.
Proof. reflexivity. Qed.

Lemma head_of_kind_value_satisfied_b_not_null (v : value)
    (non_null : or_unknown_or_bottom head_of_kind_value_non_null)
    (is_null : is_null) :
  v <> Value_const Null ->
  head_of_kind_value_satisfied_b statics vars v
    (Head_of_kind_value non_null is_null)
  = match non_null with
    | Unknown => kind_eqb (kind_of_value v) Value
    | Bottom => false
    | Ok head => head_of_kind_value_non_null_satisfied_b statics vars v head
    end
    && match is_null with
       | Not_null => true
       | Maybe_null is_null => variable_is_b vars is_null 0
       end.
Proof.
  intros Hnot_null.
  destruct
    (null_or_not v
       match is_null with
       | Not_null => false
       | Maybe_null is_null => variable_is_b vars is_null 1
       end
       (match non_null with
        | Unknown => kind_eqb (kind_of_value v) Value
        | Bottom => false
        | Ok head =>
          head_of_kind_value_non_null_satisfied_b statics vars v head
        end
        && match is_null with
           | Not_null => true
           | Maybe_null is_null => variable_is_b vars is_null 0
           end))
    as [[Hnull _] | [_ Heq]]; [ destruct (Hnot_null Hnull) | exact Heq ].
Qed.

Lemma variant_satisfied_b_immediate (i : Z) (is_int : option variable)
    (immediates : Or_unknown.t flambda_type) (get_tag : option variable)
    (blocks : Or_unknown.t row_like_for_blocks)
    (extensions : variant_extensions) (is_unique : bool) :
  head_of_kind_value_non_null_satisfied_b statics vars
    (Value_const (Tagged_immediate i))
    (Variant is_int immediates get_tag blocks extensions is_unique)
  = variable_is_b vars is_int 1
    && match immediates with
       | Or_unknown.Unknown => true
       | Or_unknown.Known ty =>
         type_satisfied_b statics vars (Value_const (Naked_immediate i)) ty
       end
    && match extensions with
       | No_extensions => true
       | Ext when_immediate _ =>
         env_extension_satisfied_b statics vars when_immediate
       end.
Proof. reflexivity. Qed.

Lemma variant_satisfied_b_not_immediate (v : value)
    (is_int : option variable) (immediates : Or_unknown.t flambda_type)
    (get_tag : option variable) (blocks : Or_unknown.t row_like_for_blocks)
    (extensions : variant_extensions) (is_unique : bool) :
  (∀ i : Z, v <> Value_const (Tagged_immediate i)) ->
  head_of_kind_value_non_null_satisfied_b statics vars v
    (Variant is_int immediates get_tag blocks extensions is_unique)
  = match block_of statics v with
    | Some (tag, fields) =>
      variable_is_b vars is_int 0
      && variable_is_b vars get_tag tag
      && match blocks with
         | Or_unknown.Unknown => true
         | Or_unknown.Known blocks =>
           row_like_for_blocks_satisfied_b statics vars tag fields blocks
         end
      && match extensions with
         | No_extensions => true
         | Ext _ when_block =>
           env_extension_satisfied_b statics vars when_block
         end
    | None => false
    end.
Proof.
  intros Hnot.
  destruct
    (tagged_immediate_or_not v
       (fun i =>
          variable_is_b vars is_int 1
          && match immediates with
             | Or_unknown.Unknown => true
             | Or_unknown.Known ty =>
               type_satisfied_b statics vars (Value_const (Naked_immediate i))
                 ty
             end
          && match extensions with
             | No_extensions => true
             | Ext when_immediate _ =>
               env_extension_satisfied_b statics vars when_immediate
             end)
       match block_of statics v with
       | Some (tag, fields) =>
         variable_is_b vars is_int 0
         && variable_is_b vars get_tag tag
         && match blocks with
            | Or_unknown.Unknown => true
            | Or_unknown.Known blocks =>
              row_like_for_blocks_satisfied_b statics vars tag fields blocks
            end
         && match extensions with
            | No_extensions => true
            | Ext _ when_block =>
              env_extension_satisfied_b statics vars when_block
            end
       | None => false
       end)
    as [[i [Hv _]] | [_ Heq]]; [ destruct (Hnot i Hv) | exact Heq ].
Qed.

Lemma mutable_block_satisfied_b (v : value) (alloc_mode : Alloc_mode.t) :
  head_of_kind_value_non_null_satisfied_b statics vars v
    (Mutable_block alloc_mode)
  = false.
Proof. reflexivity. Qed.

Lemma boxed_number_satisfied_b (v : value) (k : boxable_number)
    (contents : flambda_type) (alloc_mode : Alloc_mode.t) :
  head_of_kind_value_non_null_satisfied_b statics vars v
    (Boxed_number k contents alloc_mode)
  = match boxed_number_of statics v with
    | Some c =>
      kind_eqb (kind_of_const c) (unboxed_kind k)
      && type_satisfied_b statics vars (Value_const c) contents
    | None => false
    end.
Proof. reflexivity. Qed.

Lemma closures_satisfied_b (v : value)
    (by_function_slot : row_like_for_closures) (alloc_mode : Alloc_mode.t) :
  head_of_kind_value_non_null_satisfied_b statics vars v
    (Closures by_function_slot alloc_mode)
  = match closure_in_set statics v with
    | Some (slot, functions, value_slots) =>
      row_like_for_closures_satisfied_b statics vars v slot functions
        value_slots by_function_slot
    | None => false
    end.
Proof. reflexivity. Qed.

Lemma string_satisfied_b (v : value) (strs : list (list Z)) :
  head_of_kind_value_non_null_satisfied_b statics vars v (String strs)
  = match string_of statics v with
    | Some bytes => Keyed.among (lists_eqb Z.eqb) bytes strs
    | None => false
    end.
Proof. reflexivity. Qed.

Lemma row_like_for_blocks_satisfied_b_find (tag : Z) (fields : list value)
    (known_tags : Keyed.finite_map Z (Or_unknown.t row_like_block_case))
    (other_tags : Or_bottom.t row_like_block_case)
    (alloc_mode : Alloc_mode.t) :
  row_like_for_blocks_satisfied_b statics vars tag fields
    (Row_like_for_blocks known_tags other_tags alloc_mode)
  = match Keyed.lookup Z.eqb tag known_tags with
    | Some (Or_unknown.Known case) =>
      row_like_block_case_satisfied_b statics vars tag fields case
    | Some Or_unknown.Unknown => true
    | None =>
      match other_tags with
      | Or_bottom.Ok case =>
        row_like_block_case_satisfied_b statics vars tag fields case
      | Or_bottom.Bottom => false
      end
    end.
Proof.
  change
    (Keyed.at_binding_b Z.eqb
       (fun case =>
          match case with
          | Or_unknown.Unknown => true
          | Or_unknown.Known case =>
            row_like_block_case_satisfied_b statics vars tag fields case
          end)
       tag known_tags
     && match Keyed.lookup Z.eqb tag known_tags with
        | Some _ => true
        | None =>
          match other_tags with
          | Or_bottom.Bottom => false
          | Or_bottom.Ok case =>
            row_like_block_case_satisfied_b statics vars tag fields case
          end
        end
     = match Keyed.lookup Z.eqb tag known_tags with
       | Some (Or_unknown.Known case) =>
         row_like_block_case_satisfied_b statics vars tag fields case
       | Some Or_unknown.Unknown => true
       | None =>
         match other_tags with
         | Or_bottom.Ok case =>
           row_like_block_case_satisfied_b statics vars tag fields case
         | Or_bottom.Bottom => false
         end
       end).
  rewrite keyed_at_binding_b_lookup.
  destruct (Keyed.lookup Z.eqb tag known_tags) as [[case | ] | ];
    [ apply andb_true_r | reflexivity | reflexivity ].
Qed.

Lemma row_like_for_closures_satisfied_b_lookup (v : value)
    (slot : function_slot) (functions : list (function_slot * code_id))
    (value_slots : list (value_slot * value))
    (known_closures : finite_map row_like_closures_case) :
  row_like_for_closures_satisfied_b statics vars v slot functions
    value_slots (Row_like_for_closures known_closures)
  = match lookup slot known_closures with
    | Some case =>
      row_like_closures_case_satisfied_b statics vars v slot functions
        value_slots case
    | None => false
    end.
Proof.
  change
    (is_bound slot known_closures
     && at_binding_b
          (row_like_closures_case_satisfied_b statics vars v slot functions
             value_slots)
          slot known_closures
     = match lookup slot known_closures with
       | Some case =>
         row_like_closures_case_satisfied_b statics vars v slot functions
           value_slots case
       | None => false
       end).
  rewrite keyed_at_binding_b_lookup. unfold is_bound.
  change (Keyed.lookup Pos.eqb slot known_closures)
    with (lookup slot known_closures).
  destruct (lookup slot known_closures) as [case | ]; reflexivity.
Qed.

Lemma closures_entry_satisfied_b_eq (v : value) (slot : function_slot)
    (functions : list (function_slot * code_id))
    (value_slots : list (value_slot * value))
    (function_types : finite_map (Or_unknown.t function_type))
    (closure_types value_slot_types : finite_map flambda_type) :
  closures_entry_satisfied_b statics vars v slot functions value_slots
    (Closures_entry function_types closure_types value_slot_types)
  = for_all_bindings_b
      (fun g function_type =>
         match function_type, lookup g functions with
         | Or_unknown.Known function_type, Some code =>
           Pos.eqb code (function_type_code_id function_type)
         | Or_unknown.Unknown, _ | _, None => true
         end)
      function_types
    && for_all_bindings_b
         (fun g ty =>
            match project_function_slot statics slot g v with
            | Some closure => type_satisfied_b statics vars closure ty
            | None => true
            end)
         closure_types
    && for_all_bindings_b
         (fun w ty =>
            match lookup w value_slots with
            | Some contents => type_satisfied_b statics vars contents ty
            | None => true
            end)
         value_slot_types.
Proof. reflexivity. Qed.

Lemma env_extension_satisfied_b_eq
    (equations : Keyed.finite_map name flambda_type) :
  env_extension_satisfied_b statics vars (Env_extension equations)
  = Keyed.for_all_bindings_b name_eqb
      (fun n ty =>
         match value_of_name vars n with
         | Some v => type_satisfied_b statics vars v ty
         | None => false
         end)
      equations.
Proof. reflexivity. Qed.

End Equations.

(* ======================================================================== *)
(* Each test answers [true] exactly when the meaning holds                  *)
(* ======================================================================== *)

Lemma type_satisfied_b_all (statics : static_env) (vars : env) :
  (∀ (t : flambda_type) (v : value),
    type_satisfied_b statics vars v t = true
    <-> type_satisfied statics vars v t)
  ∧ (∀ (head : head_of_kind_value) (v : value),
      head_of_kind_value_satisfied_b statics vars v head = true
      <-> head_of_kind_value_satisfied statics vars v head)
  ∧ (∀ (head : head_of_kind_value_non_null) (v : value),
      head_of_kind_value_non_null_satisfied_b statics vars v head = true
      <-> head_of_kind_value_non_null_satisfied statics vars v head)
  ∧ (∀ (blocks : row_like_for_blocks) (tag : Z) (fields : list value),
      row_like_for_blocks_satisfied_b statics vars tag fields blocks = true
      <-> row_like_for_blocks_satisfied statics vars tag fields blocks)
  ∧ (∀ (closures : row_like_for_closures) (v : value) (slot : function_slot)
       (functions : list (function_slot * code_id))
       (value_slots : list (value_slot * value)),
      row_like_for_closures_satisfied_b statics vars v slot functions
        value_slots closures
      = true
      <-> row_like_for_closures_satisfied statics vars v slot functions
            value_slots closures)
  ∧ (∀ (entry : closures_entry) (v : value) (slot : function_slot)
       (functions : list (function_slot * code_id))
       (value_slots : list (value_slot * value)),
      closures_entry_satisfied_b statics vars v slot functions value_slots
        entry
      = true
      <-> closures_entry_satisfied statics vars v slot functions value_slots
            entry)
  ∧ (∀ extension : env_extension,
      env_extension_satisfied_b statics vars extension = true
      <-> env_extension_satisfied statics vars extension).
Proof.
  apply types_induction.
  - (* Type_value *)
    intros ty Hhead v.
    rewrite type_satisfied_b_value, type_satisfied_value.
    apply descr_satisfied_b_reflects.
    destruct ty as [ | [head | s] | ]; try exact I. exact (Hhead v).
  - (* Type_naked_immediate *)
    intros ty v.
    rewrite type_satisfied_b_naked_immediate, type_satisfied_naked_immediate.
    apply descr_satisfied_b_reflects.
    destruct ty as [ | [head | s] | ]; try exact I.
    apply head_of_kind_naked_immediate_satisfied_b_eq.
  - (* Type_naked_number *)
    intros k ty v.
    rewrite type_satisfied_b_naked_number, type_satisfied_naked_number.
    apply descr_satisfied_b_reflects.
    destruct ty as [ | [numbers | s] | ]; try exact I.
    apply naked_numbers_satisfied_b_eq.
  - (* Type_rec_info *)
    intros ty v. rewrite type_satisfied_b_rec_info, type_satisfied_rec_info.
    apply descr_satisfied_b_reflects.
    destruct ty as [ | [head | s] | ]; try exact I. apply kind_eqb_eq.
  - (* Type_region *)
    intros ty v. rewrite type_satisfied_b_region, type_satisfied_region.
    apply descr_satisfied_b_reflects.
    destruct ty as [ | [head | s] | ]; try exact I. apply kind_eqb_eq.
  - (* Head_of_kind_value *)
    intros non_null is_null Hnon_null v.
    rewrite head_of_kind_value_satisfied_eq.
    destruct (null_or_other v) as [-> | Hnot_null].
    + rewrite head_of_kind_value_satisfied_b_null.
      destruct is_null as [ | is_null].
      * split; [ discriminate | ].
        intros [[_ []] | [Hne _]]. destruct (Hne eq_refl).
      * rewrite variable_is_b_eq. split.
        -- intros His_null. left. split; [ reflexivity | exact His_null ].
        -- intros [[_ His_null] | [Hne _]];
             [ exact His_null | destruct (Hne eq_refl) ].
    + rewrite (head_of_kind_value_satisfied_b_not_null _ _ _ _ _ Hnot_null),
        andb_true_iff.
      assert (Hhead :
        match non_null with
        | Unknown => kind_eqb (kind_of_value v) Value
        | Bottom => false
        | Ok head =>
          head_of_kind_value_non_null_satisfied_b statics vars v head
        end
        = true
        <-> match non_null with
            | Unknown => kind_of_value v = Value
            | Bottom => False
            | Ok head =>
              head_of_kind_value_non_null_satisfied statics vars v head
            end).
      { destruct non_null as [ | head | ];
          [ apply kind_eqb_eq | exact (Hnon_null v)
          | split; [ discriminate | intros [] ] ]. }
      assert (His_null :
        match is_null with
        | Not_null => true
        | Maybe_null is_null => variable_is_b vars is_null 0
        end
        = true
        <-> match is_null with
            | Not_null => True
            | Maybe_null is_null => variable_is vars is_null 0
            end).
      { destruct is_null as [ | is_null];
          [ split; [ intros _; exact I | reflexivity ]
          | apply variable_is_b_eq ]. }
      rewrite Hhead, His_null. split.
      * intros H. right. split; [ exact Hnot_null | exact H ].
      * intros [[Hnull _] | [_ H]]; [ destruct (Hnot_null Hnull) | exact H ].
  - (* Variant *)
    intros is_int immediates get_tag blocks extensions is_unique Himmediates
      Hblocks Hextensions v.
    rewrite variant_satisfied.
    destruct (tagged_immediate_or_other v) as [[i ->] | Hnot].
    + rewrite variant_satisfied_b_immediate, !andb_true_iff, variable_is_b_eq.
      assert (Himmediate :
        match immediates with
        | Or_unknown.Unknown => true
        | Or_unknown.Known ty =>
          type_satisfied_b statics vars (Value_const (Naked_immediate i)) ty
        end
        = true
        <-> match immediates with
            | Or_unknown.Unknown => True
            | Or_unknown.Known ty =>
              type_satisfied statics vars (Value_const (Naked_immediate i))
                ty
            end).
      { destruct immediates as [ty | ];
          [ exact (Himmediates _)
          | split; [ intros _; exact I | reflexivity ] ]. }
      assert (Hextension :
        match extensions with
        | No_extensions => true
        | Ext when_immediate _ =>
          env_extension_satisfied_b statics vars when_immediate
        end
        = true
        <-> match extensions with
            | No_extensions => True
            | Ext when_immediate _ =>
              env_extension_satisfied statics vars when_immediate
            end).
      { destruct extensions as [ | when_immediate when_block];
          [ split; [ intros _; exact I | reflexivity ]
          | exact (proj1 Hextensions) ]. }
      rewrite Himmediate, Hextension. split.
      * intros [[His_int Himm] Hext]. left. exists i.
        split; [ reflexivity | ]. split; [ exact His_int | ].
        split; [ exact Himm | exact Hext ].
      * intros [[j [[= <-] [His_int [Himm Hext]]]]
               | [tag [fields [Hblock _]]]];
          [ | discriminate Hblock ].
        split; [ split; [ exact His_int | exact Himm ] | exact Hext ].
    + rewrite (variant_satisfied_b_not_immediate _ _ _ _ _ _ _ _ _ Hnot).
      destruct (block_of statics v) as [[tag fields] | ] eqn:Hblock.
      * rewrite !andb_true_iff, !variable_is_b_eq.
        assert (Hrow :
          match blocks with
          | Or_unknown.Unknown => true
          | Or_unknown.Known blocks =>
            row_like_for_blocks_satisfied_b statics vars tag fields blocks
          end
          = true
          <-> match blocks with
              | Or_unknown.Unknown => True
              | Or_unknown.Known blocks =>
                row_like_for_blocks_satisfied statics vars tag fields blocks
              end).
        { destruct blocks as [blocks | ];
            [ exact (Hblocks tag fields)
            | split; [ intros _; exact I | reflexivity ] ]. }
        assert (Hextension :
          match extensions with
          | No_extensions => true
          | Ext _ when_block =>
            env_extension_satisfied_b statics vars when_block
          end
          = true
          <-> match extensions with
              | No_extensions => True
              | Ext _ when_block =>
                env_extension_satisfied statics vars when_block
              end).
        { destruct extensions as [ | when_immediate when_block];
            [ split; [ intros _; exact I | reflexivity ]
            | exact (proj2 Hextensions) ]. }
        rewrite Hrow, Hextension. split.
        -- intros [[[His_int Hget_tag] Hrow'] Hext]. right.
           exists tag, fields. split; [ reflexivity | ].
           split; [ exact His_int | ]. split; [ exact Hget_tag | ].
           split; [ exact Hrow' | exact Hext ].
        -- intros [[i [Hv _]] | [tag' [fields' [[= <- <-] Hrest]]]];
             [ destruct (Hnot i Hv) | ].
           destruct Hrest as [His_int [Hget_tag [Hrow' Hext]]].
           split; [ | exact Hext ]. split; [ | exact Hrow' ].
           split; [ exact His_int | exact Hget_tag ].
      * split; [ discriminate | ].
        intros [[i [Hv _]] | [tag [fields [Hblock' _]]]];
          [ destruct (Hnot i Hv) | discriminate Hblock' ].
  - (* Mutable_block *)
    intros alloc_mode v.
    rewrite mutable_block_satisfied_b, mutable_block_head_satisfied.
    split; [ discriminate | intros [] ].
  - (* Boxed_number *)
    intros k contents alloc_mode Hcontents v.
    rewrite boxed_number_satisfied_b, boxed_number_satisfied.
    destruct (boxed_number_of statics v) as [c | ].
    + rewrite andb_true_iff, (kind_eqb_eq _ _), (Hcontents (Value_const c)).
      split.
      * intros H. exists c. split; [ reflexivity | exact H ].
      * intros [c' [[= <-] H]]. exact H.
    + split; [ discriminate | ]. intros [c [Hboxed _]]. discriminate Hboxed.
  - (* Closures *)
    intros by_function_slot alloc_mode Hclosures v.
    rewrite closures_satisfied_b, closures_satisfied.
    destruct (closure_in_set statics v)
      as [[[slot functions] value_slots] | ].
    + rewrite (Hclosures v slot functions value_slots). split.
      * intros H. exists slot, functions, value_slots.
        split; [ reflexivity | exact H ].
      * intros [slot' [functions' [value_slots' [[= <- <- <-] H]]]]. exact H.
    + split; [ discriminate | ].
      intros [slot [functions [value_slots [Hclosure _]]]].
      discriminate Hclosure.
  - (* String *)
    intros strs v. rewrite string_satisfied_b, string_satisfied.
    destruct (string_of statics v) as [bytes | ].
    + rewrite (keyed_among_in (list Z) (lists_eqb Z.eqb) bytes_eqb_eq). split.
      * intros H. exists bytes. split; [ reflexivity | exact H ].
      * intros [bytes' [[= <-] H]]. exact H.
    + split; [ discriminate | ]. intros [bytes [Hstring _]].
      discriminate Hstring.
  - (* Row_like_for_blocks *)
    intros known_tags other_tags alloc_mode Hknown Hother tag fields.
    rewrite row_like_for_blocks_satisfied_b_find,
      row_like_for_blocks_find_iff.
    assert (Hcase :
      ∀ case : row_like_block_case,
        P_block_case
          (fun t =>
             ∀ v : value,
               type_satisfied_b statics vars v t = true
               <-> type_satisfied statics vars v t)
          (fun extension =>
             env_extension_satisfied_b statics vars extension = true
             <-> env_extension_satisfied statics vars extension)
          case ->
        (row_like_block_case_satisfied_b statics vars tag fields case = true
         <-> row_like_block_case_satisfied statics vars tag fields case)).
    { intros case [Htypes Hextension].
      unfold row_like_block_case_satisfied_b, row_like_block_case_satisfied.
      apply block_case_satisfied_b_reflects; [ | exact Hextension ].
      intros ty Hin field. exact (Htypes ty Hin field). }
    destruct (Keyed.lookup Z.eqb tag known_tags) as [[case | ] | ] eqn:Hfind.
    + apply Hcase.
      exact (Hknown tag case (keyed_lookup_in Z.eqb Z.eqb_eq _ _ _ Hfind)).
    + split; [ intros _; exact I | reflexivity ].
    + destruct other_tags as [case | ];
        [ exact (Hcase case Hother) | split; [ discriminate | intros [] ] ].
  - (* Row_like_for_closures *)
    intros known_closures Hknown v slot functions value_slots.
    rewrite row_like_for_closures_satisfied_b_lookup,
      row_like_for_closures_lookup_iff.
    destruct (lookup slot known_closures) as [case | ] eqn:Hlookup.
    + destruct (Hknown slot case (lookup_in _ _ _ Hlookup))
        as [IHentry IHextension].
      assert (Hcase :
        row_like_closures_case_satisfied_b statics vars v slot functions
          value_slots case
        = true
        <-> row_like_closures_case_satisfied statics vars v slot functions
              value_slots case).
      { unfold row_like_closures_case_satisfied_b,
          row_like_closures_case_satisfied.
        apply closures_case_satisfied_b_reflects;
          [ exact (IHentry v slot functions value_slots)
          | exact IHextension ]. }
      rewrite Hcase. split.
      * intros H. exists case. split; [ reflexivity | exact H ].
      * intros [case' [[= <-] H]]. exact H.
    + split; [ discriminate | ]. intros [case [Hcase _]]. discriminate Hcase.
  - (* Closures_entry *)
    intros function_types closure_types value_slot_types Hclosures Hslots v
      slot functions value_slots.
    rewrite closures_entry_satisfied_b_eq, closures_entry_lookup_iff,
      !andb_true_iff.
    rewrite (keyed_for_all_bindings_b positive Pos.eqb Pos.eqb_eq
               (Or_unknown.t function_type)),
      !(keyed_for_all_bindings_b positive Pos.eqb Pos.eqb_eq flambda_type),
      (for_all_bindings_lookup (Or_unknown.t function_type)),
      !(for_all_bindings_lookup flambda_type).
    split.
    + intros [[Hfunctions Hclosure_types] Hvalue_slots]. split; [ | split ].
      * intros g function_type code Hlookup Hcode.
        specialize (Hfunctions g _ Hlookup). simpl in Hfunctions.
        rewrite Hcode in Hfunctions. apply Pos.eqb_eq. exact Hfunctions.
      * intros g ty closure Hlookup Hproject.
        specialize (Hclosure_types g ty Hlookup). simpl in Hclosure_types.
        rewrite Hproject in Hclosure_types.
        apply (Hclosures g ty (lookup_in _ _ _ Hlookup)).
        exact Hclosure_types.
      * intros w ty contents Hlookup Hcontents.
        specialize (Hvalue_slots w ty Hlookup). simpl in Hvalue_slots.
        rewrite Hcontents in Hvalue_slots.
        apply (Hslots w ty (lookup_in _ _ _ Hlookup)). exact Hvalue_slots.
    + intros [Hfunctions [Hclosure_types Hvalue_slots]]. split; [ split | ].
      * intros g [function_type | ] Hlookup; [ | reflexivity ].
        destruct (lookup g functions) as [code | ] eqn:Hcode;
          [ | reflexivity ].
        apply Pos.eqb_eq.
        exact (Hfunctions g function_type code Hlookup Hcode).
      * intros g ty Hlookup.
        destruct (project_function_slot statics slot g v) as [closure | ]
          eqn:Hproject; [ | reflexivity ].
        apply (Hclosures g ty (lookup_in _ _ _ Hlookup)).
        exact (Hclosure_types g ty closure Hlookup Hproject).
      * intros w ty Hlookup.
        destruct (lookup w value_slots) as [contents | ] eqn:Hcontents;
          [ | reflexivity ].
        apply (Hslots w ty (lookup_in _ _ _ Hlookup)).
        exact (Hvalue_slots w ty contents Hlookup Hcontents).
  - (* Env_extension *)
    intros equations Htypes.
    rewrite env_extension_satisfied_b_eq, env_extension_lookup_iff,
      (keyed_for_all_bindings_b name name_eqb name_eqb_eq),
      (keyed_for_all_bindings_lookup name name_eqb name_eqb_eq).
    split.
    + intros Hequations n ty Hlookup.
      specialize (Hequations n ty Hlookup). simpl in Hequations.
      destruct (value_of_name vars n) as [v | ]; [ | discriminate Hequations ].
      exists v. split; [ reflexivity | ].
      apply (Htypes n ty (keyed_lookup_in name_eqb name_eqb_eq _ _ _ Hlookup)).
      exact Hequations.
    + intros Hequations n ty Hlookup.
      destruct (Hequations n ty Hlookup) as [v [Hv Hty]].
      simpl. rewrite Hv.
      apply (Htypes n ty (keyed_lookup_in name_eqb name_eqb_eq _ _ _ Hlookup)).
      exact Hty.
Qed.

Theorem type_satisfied_b_iff : type_satisfied_b_iff_statement.
Proof.
  intros statics vars v t.
  exact (proj1 (type_satisfied_b_all statics vars) t v).
Qed.

Theorem head_of_kind_value_satisfied_b_iff
  : head_of_kind_value_satisfied_b_iff_statement.
Proof.
  intros statics vars v head.
  exact (proj1 (proj2 (type_satisfied_b_all statics vars)) head v).
Qed.

Theorem head_of_kind_value_non_null_satisfied_b_iff
  : head_of_kind_value_non_null_satisfied_b_iff_statement.
Proof.
  intros statics vars v head.
  exact (proj1 (proj2 (proj2 (type_satisfied_b_all statics vars))) head v).
Qed.

Theorem row_like_for_blocks_satisfied_b_iff
  : row_like_for_blocks_satisfied_b_iff_statement.
Proof.
  intros statics vars tag fields blocks.
  destruct (type_satisfied_b_all statics vars) as [_ [_ [_ [H _]]]].
  exact (H blocks tag fields).
Qed.

Theorem row_like_for_closures_satisfied_b_iff
  : row_like_for_closures_satisfied_b_iff_statement.
Proof.
  intros statics vars v slot functions value_slots closures.
  destruct (type_satisfied_b_all statics vars) as [_ [_ [_ [_ [H _]]]]].
  exact (H closures v slot functions value_slots).
Qed.

Theorem closures_entry_satisfied_b_iff
  : closures_entry_satisfied_b_iff_statement.
Proof.
  intros statics vars v slot functions value_slots entry.
  destruct (type_satisfied_b_all statics vars) as [_ [_ [_ [_ [_ [H _]]]]]].
  exact (H entry v slot functions value_slots).
Qed.

Theorem env_extension_satisfied_b_iff
  : env_extension_satisfied_b_iff_statement.
Proof.
  intros statics vars extension.
  destruct (type_satisfied_b_all statics vars) as [_ [_ [_ [_ [_ [_ H]]]]]].
  exact (H extension).
Qed.

Theorem row_like_block_case_satisfied_b_iff
  : row_like_block_case_satisfied_b_iff_statement.
Proof.
  intros statics vars tag fields case.
  unfold row_like_block_case_satisfied_b, row_like_block_case_satisfied.
  apply block_case_satisfied_b_reflects.
  - intros ty _ field. apply type_satisfied_b_iff.
  - apply env_extension_satisfied_b_iff.
Qed.

Theorem row_like_closures_case_satisfied_b_iff
  : row_like_closures_case_satisfied_b_iff_statement.
Proof.
  intros statics vars v slot functions value_slots case.
  unfold row_like_closures_case_satisfied_b,
    row_like_closures_case_satisfied.
  apply closures_case_satisfied_b_reflects.
  - apply closures_entry_satisfied_b_iff.
  - apply env_extension_satisfied_b_iff.
Qed.
