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

(* Proofs of the statements of FlambdaCopyLetsSpec.v. Nothing here
   needs review.

   The unit and the unit with one copy are run side by side, in step,
   through the unit with its static copies alone
   (FlambdaCopyCodeProofs.copy_state of the first state, which runs in
   step with it). The third state is that one with some of its [Let]s
   making their closures of the copies, here and in the code that its
   static environment binds. A value of the third run is the value of
   the second but that a closure that a [Let] made may have the copy
   of its code, here and inside closures, blocks, arrays and partial
   applications ([value_copied]); the heaps are alike up to that. The
   copy of a code id is bound, in the second run, to what the id is
   bound to ([follows]: the test FlambdaCopyLetsSpec.copies_apart is
   for this), so a call of a closure of the copy runs the same body,
   but for the [Let]s copied. A direct call of the second run whose
   callee is no symbol names code that has no copy
   (FlambdaCopyCodeSpec.copy_call), so the closure called has that
   code in the third run too. No primitive looks at the code of a
   closure (FlambdaPrimitiveProofs.eval_prim_related,
   FlambdaHeapProofs.exec_prim_related). The first run exits with
   constants and symbols, which are related to themselves only, and
   with the empty heap. *)

From Stdlib Require Import ZArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaPrimitiveModel FlambdaValueSpec
  FlambdaPrimitiveSpec FlambdaPrimitiveProofs FlambdaHeapSpec
  FlambdaHeapProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaSimulationProofs FlambdaNominalSpec FlambdaNominalProofs
  FlambdaClosuresProofs
  FlambdaUnusedStaticsSpec FlambdaImportsSpec FlambdaHeapFreeSpec
  FlambdaHeapFreeProofs FlambdaLiftSpec FlambdaLiftProofs
  FlambdaCopyCodeSpec FlambdaCopyCodeProofs FlambdaEquivarianceTactic
  FlambdaCopyLetsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The rewrite with nothing asked, and what it keeps                        *)
(* ======================================================================== *)

(* With no path given, no [Let] is changed. *)
Fixpoint copy_lets_expr_nil (ids : list (code_id * code_id)) (at_path : path)
    (e : expr) {struct e} :
  copy_lets_expr ids [] at_path e = e
with copy_lets_handler_nil (ids : list (code_id * code_id)) (at_path : path)
    (h : continuation_handler) {struct h} :
  copy_lets_handler ids [] at_path h = h.
Proof.
  - destruct e
      as [ x xkind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try reflexivity.
    + rewrite copy_lets_expr_nil. reflexivity.
    + rewrite copy_lets_expr_nil. reflexivity.
    + rewrite copy_lets_handler_nil, copy_lets_expr_nil. reflexivity.
    + rewrite copy_lets_expr_nil. f_equal.
      generalize (xI at_path) as q. revert handlers. fix IH 1.
      intros [ | [k h] handlers] q; simpl; [ reflexivity | ].
      rewrite copy_lets_handler_nil, IH. reflexivity.
  - destruct h as [params metadata handler]. simpl.
    rewrite copy_lets_expr_nil. reflexivity.
Qed.

(* The [Let]s copied bind what the term binds. *)
Fixpoint binds_statics_copy_lets (ids : list (code_id * code_id))
    (lets : list path) (at_path : path) (e : expr) {struct e} :
  binds_statics (copy_lets_expr ids lets at_path e) = binds_statics e
with handler_binds_statics_copy_lets (ids : list (code_id * code_id))
    (lets : list path) (at_path : path) (h : continuation_handler)
    {struct h} :
  handler_binds_statics (copy_lets_handler ids lets at_path h)
  = handler_binds_statics h.
Proof.
  - destruct e
      as [ x xkind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try reflexivity.
    + apply binds_statics_copy_lets.
    + rewrite binds_statics_copy_lets. reflexivity.
    + rewrite handler_binds_statics_copy_lets, binds_statics_copy_lets.
      reflexivity.
    + rewrite binds_statics_copy_lets. f_equal.
      generalize (xI at_path) as q. revert handlers. fix IH 1.
      intros [ | [k h] handlers] q; simpl; [ reflexivity | ].
      rewrite handler_binds_statics_copy_lets, IH. reflexivity.
  - destruct h as [params metadata handler]. simpl.
    apply binds_statics_copy_lets.
Qed.

Lemma binding_binds_copy_lets (ids : list (code_id * code_id))
    (in_code : list (code_id * list path)) (binding : static_binding) :
  binding_binds (copy_lets_binding ids in_code binding)
  = binding_binds binding.
Proof.
  destruct binding as [id [params my_closure ret exn metadata body] | id
    | s const];
    simpl; try reflexivity.
  destruct (lookup id in_code); reflexivity.
Qed.

Fixpoint binds_statics_copy_lets_in_code (ids : list (code_id * code_id))
    (in_code : list (code_id * list path)) (e : expr) {struct e} :
  binds_statics (copy_lets_in_code ids in_code e) = binds_statics e
with handler_binds_statics_copy_lets_in_code
    (ids : list (code_id * code_id)) (in_code : list (code_id * list path))
    (h : continuation_handler) {struct h} :
  handler_binds_statics (copy_lets_in_code_handler ids in_code h)
  = handler_binds_statics h.
Proof.
  - destruct e
      as [ x xkind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try reflexivity.
    + apply binds_statics_copy_lets_in_code.
    + rewrite binds_statics_copy_lets_in_code. f_equal.
      induction bindings as [ | binding bindings IH]; simpl;
        [ reflexivity | ].
      rewrite binding_binds_copy_lets, IH. reflexivity.
    + rewrite handler_binds_statics_copy_lets_in_code,
        binds_statics_copy_lets_in_code.
      reflexivity.
    + rewrite binds_statics_copy_lets_in_code. f_equal.
      revert handlers. fix IH 1.
      intros [ | [k h] handlers]; simpl; [ reflexivity | ].
      rewrite handler_binds_statics_copy_lets_in_code, IH. reflexivity.
  - destruct h as [params metadata handler]. simpl.
    apply binds_statics_copy_lets_in_code.
Qed.

(* A term that binds nothing static has no code to go into. *)
Fixpoint copy_lets_in_code_none (ids : list (code_id * code_id))
    (in_code : list (code_id * list path)) (e : expr) {struct e} :
  binds_statics e = false -> copy_lets_in_code ids in_code e = e
with copy_lets_in_code_handler_none (ids : list (code_id * code_id))
    (in_code : list (code_id * list path)) (h : continuation_handler)
    {struct h} :
  handler_binds_statics h = false ->
  copy_lets_in_code_handler ids in_code h = h.
Proof.
  - destruct e
      as [ x xkind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try reflexivity.
    + intros H. rewrite (copy_lets_in_code_none ids in_code body H).
      reflexivity.
    + intros H. apply orb_false_iff in H as [Hbindings Hbody].
      rewrite (copy_lets_in_code_none ids in_code body Hbody). f_equal.
      induction bindings as [ | binding bindings IH]; [ reflexivity | ].
      simpl in Hbindings. apply orb_false_iff in Hbindings as [Hb Hrest].
      simpl. rewrite (IH Hrest). f_equal.
      destruct binding; try reflexivity; discriminate Hb.
    + intros H. apply orb_false_iff in H as [Hh Hbody].
      rewrite (copy_lets_in_code_handler_none ids in_code h Hh),
        (copy_lets_in_code_none ids in_code body Hbody).
      reflexivity.
    + intros H. apply orb_false_iff in H as [Hhandlers Hbody].
      rewrite (copy_lets_in_code_none ids in_code body Hbody). f_equal.
      revert handlers Hhandlers. fix IH 1.
      intros [ | [k h] handlers]; simpl; [ reflexivity | ].
      intros H. apply orb_false_iff in H as [Hh Hrest].
      rewrite (copy_lets_in_code_handler_none ids in_code h Hh),
        (IH handlers Hrest).
      reflexivity.
  - destruct h as [params metadata handler]. simpl. intros H.
    rewrite (copy_lets_in_code_none ids in_code handler H). reflexivity.
Qed.

Section Lets.

Variable ids : list (code_id * code_id).
Variable in_code : list (code_id * list path).

(* ======================================================================== *)
(* Values                                                                   *)
(* ======================================================================== *)

(* A code id, and it or its copy; the functions of a set, and they or
   their copies, all of them. *)
Definition id_copied (id id' : code_id) : Prop :=
  id' = id ∨ id' = copy_of ids id.

Definition decls_copied (decls decls' : list (function_slot * code_id))
    : Prop :=
  decls' = decls ∨ decls' = copy_decls ids decls.

(* [value_copied v v']: [v'] is [v], but that a closure that a [Let]
   made, here or inside a closure, a block, an array or a partial
   application, may have the copy of its code. *)
#[local] Set Warnings "-register-all".
Inductive value_copied : value -> value -> Prop :=
  | Copied_const (c : const) : value_copied (Value_const c) (Value_const c)
  | Copied_symbol (s : symbol) :
    value_copied (Value_symbol s) (Value_symbol s)
  | Copied_boxed_number (c : const) :
    value_copied (Value_boxed_number c) (Value_boxed_number c)
  | Copied_location (l : location) :
    value_copied (Value_location l) (Value_location l)
  | Copied_block (tag : Z) (fields fields' : list value) :
    Forall2 value_copied fields fields' ->
    value_copied (Value_block tag fields) (Value_block tag fields')
  | Copied_array (elements elements' : list value) :
    Forall2 value_copied elements elements' ->
    value_copied (Value_immutable_array elements)
      (Value_immutable_array elements')
  | Copied_closure (code code' : code_id)
      (value_slots value_slots' : list (value_slot * value)) :
    id_copied code code' ->
    Forall2 (slot_lifted value_copied) value_slots value_slots' ->
    value_copied (Value_closure code value_slots)
      (Value_closure code' value_slots')
  | Copied_closure_of_set (slot : function_slot)
      (function_decls function_decls' : list (function_slot * code_id))
      (value_slots value_slots' : list (value_slot * value)) :
    decls_copied function_decls function_decls' ->
    Forall2 (slot_lifted value_copied) value_slots value_slots' ->
    value_copied (Value_closure_of_set slot function_decls value_slots)
      (Value_closure_of_set slot function_decls' value_slots')
  | Copied_partial_application (callee callee' : value)
      (args args' : list value) (args_arity : arity) :
    value_copied callee callee' ->
    Forall2 value_copied args args' ->
    value_copied (Value_partial_application callee args args_arity)
      (Value_partial_application callee' args' args_arity).
#[local] Set Warnings "register-all".

Ltac copied_cases H :=
  destruct H
    as [ c | s | c | loc | tag fields fields' Hfields
       | items items' Hitems
       | code code' slots slots' Hcode Hslots
       | fslot decls decls' slots slots' Hdecls Hslots
       | pcallee pcallee' pargs pargs' parity Hpcallee Hpargs ].

(* Every value is related to itself. *)
Fixpoint value_copied_refl (v : value) {struct v} : value_copied v v.
Proof.
  destruct v
    as [ c | code slots | fslot decls slots | s | tag fields | elements | c | l
       | callee args args_arity ].
  - apply Copied_const.
  - apply Copied_closure; [ left; reflexivity | ]. revert slots. fix IH 1.
    intros [ | [slot w] slots]; constructor;
      [ split; [ reflexivity | exact (value_copied_refl w) ]
      | exact (IH slots) ].
  - apply Copied_closure_of_set; [ left; reflexivity | ]. revert slots.
    fix IH 1.
    intros [ | [slot w] slots]; constructor;
      [ split; [ reflexivity | exact (value_copied_refl w) ]
      | exact (IH slots) ].
  - apply Copied_symbol.
  - apply Copied_block. revert fields. fix IH 1.
    intros [ | w fields]; constructor;
      [ exact (value_copied_refl w) | exact (IH fields) ].
  - apply Copied_array. revert elements. fix IH 1.
    intros [ | w elements]; constructor;
      [ exact (value_copied_refl w) | exact (IH elements) ].
  - apply Copied_boxed_number.
  - apply Copied_location.
  - apply Copied_partial_application; [ exact (value_copied_refl callee) | ].
    revert args. fix IH 1.
    intros [ | w args]; constructor;
      [ exact (value_copied_refl w) | exact (IH args) ].
Qed.

Lemma values_copied_refl (vs : list value) : Forall2 value_copied vs vs.
Proof.
  induction vs as [ | v vs IH]; constructor;
    [ apply value_copied_refl | exact IH ].
Qed.

Lemma slots_copied_refl (slots : list (value_slot * value)) :
  Forall2 (slot_lifted value_copied) slots slots.
Proof.
  induction slots as [ | [slot v] slots IH]; constructor;
    [ split; [ reflexivity | apply value_copied_refl ] | exact IH ].
Qed.

(* A constant or a symbol is related to itself only. *)
Lemma plain_value_copied (v v' : value) :
  plain_value v -> value_copied v v' -> v' = v.
Proof. intros Hplain H. copied_cases H; simpl in Hplain; tauto. Qed.

Lemma plain_values_copied (vs vs' : list value) :
  (∀ v : value, v ∈ vs -> plain_value v) ->
  Forall2 value_copied vs vs' ->
  vs' = vs.
Proof.
  intros Hplain. induction 1 as [ | v v' vs vs' Hv _ IH]; [ reflexivity | ].
  rewrite (plain_value_copied v v' (Hplain v (or_introl eq_refl)) Hv), IH;
    [ reflexivity | ].
  intros w Hw. apply Hplain. right. exact Hw.
Qed.

Lemma kind_of_value_copied (v v' : value) :
  value_copied v v' -> kind_of_value v' = kind_of_value v.
Proof. intros H. copied_cases H; reflexivity. Qed.

Lemma kinds_of_values_copied (vs vs' : list value) :
  Forall2 value_copied vs vs' ->
  map kind_of_value vs' = map kind_of_value vs.
Proof.
  induction 1 as [ | v v' vs vs' Hv _ IH]; simpl; [ reflexivity | ].
  rewrite (kind_of_value_copied v v' Hv), IH. reflexivity.
Qed.

Lemma partial_application_copied (v v' : value) :
  value_copied v v' ->
  match partial_application_of v with
  | Some (closure, applied, applied_arity) =>
    ∃ (closure' : value) (applied' : list value),
      v' = Value_partial_application closure' applied' applied_arity
      ∧ value_copied closure closure'
      ∧ Forall2 value_copied applied applied'
  | None => partial_application_of v' = None
  end.
Proof.
  intros H. copied_cases H; simpl; try reflexivity.
  exists pcallee', pargs'. auto.
Qed.

Lemma lookup_slots_copied (slot : value_slot)
    (slots slots' : list (value_slot * value)) (v : value) :
  Forall2 (slot_lifted value_copied) slots slots' ->
  lookup slot slots = Some v ->
  ∃ v' : value, lookup slot slots' = Some v' ∧ value_copied v v'.
Proof.
  induction 1 as [ | [k w] [k' w'] slots slots' [Hkey Hw] _ IH]; simpl;
    [ discriminate | ].
  simpl in Hkey, Hw. subst k'.
  destruct (Pos.eqb slot k); [ intros [= <-]; eauto | exact IH ].
Qed.

Lemma value_copied_locations : relates_locations value_copied.
Proof.
  split; [ intros l; apply Copied_location | split ].
  - intros l v' H. inversion H. reflexivity.
  - intros v l H. inversion H. reflexivity.
Qed.

Lemma naked_immediate_copied (v v' : value) (i : Z) :
  value_copied v v' ->
  naked_immediate_of_value v = Some i ->
  naked_immediate_of_value v' = Some i.
Proof.
  intros Hv. copied_cases Hv; simpl; try discriminate. exact (fun H => H).
Qed.

(* The copies of the functions of a set are at the slots of the
   functions. *)
Lemma lookup_decls_copied (slot : function_slot)
    (decls decls' : list (function_slot * code_id)) :
  decls_copied decls decls' ->
  match lookup slot decls with
  | Some code =>
    ∃ code' : code_id, lookup slot decls' = Some code' ∧ id_copied code code'
  | None => lookup slot decls' = None
  end.
Proof.
  intros [-> | ->].
  - destruct (lookup slot decls) as [code | ]; [ | reflexivity ].
    exists code. split; [ reflexivity | left; reflexivity ].
  - rewrite (lookup_copy_decls ids slot decls).
    destruct (lookup slot decls) as [code | ]; [ | reflexivity ].
    exists (copy_of ids code). split; [ reflexivity | right; reflexivity ].
Qed.

(* ======================================================================== *)
(* Static environments                                                      *)
(* ======================================================================== *)

(* What a symbol is bound to in the second run and in the third: the
   same, with related parts. The code of a static closure is the same:
   the [Let]s alone are given the copies here. *)
Inductive allocated_copied
    : statically_allocated_value -> statically_allocated_value -> Prop :=
  | Copied_static_closure (code : code_id)
      (slots slots' : list (value_slot * value)) :
    Forall2 (slot_lifted value_copied) slots slots' ->
    allocated_copied (Static_closure code slots) (Static_closure code slots')
  | Copied_static_closure_of_set (fslot : function_slot)
      (syms : list (function_slot * symbol))
      (decls : list (function_slot * code_id))
      (slots slots' : list (value_slot * value)) :
    Forall2 (slot_lifted value_copied) slots slots' ->
    allocated_copied (Static_closure_of_set fslot syms decls slots)
      (Static_closure_of_set fslot syms decls slots')
  | Copied_static_block (tag : Z) (fields fields' : list value) :
    Forall2 value_copied fields fields' ->
    allocated_copied (Static_block tag fields) (Static_block tag fields')
  | Copied_static_boxed_number (v v' : value) :
    value_copied v v' ->
    allocated_copied (Static_boxed_number v) (Static_boxed_number v')
  | Copied_static_string (bytes : list Z) :
    allocated_copied (Static_string bytes) (Static_string bytes)
  | Copied_static_imported (c : import_class) :
    allocated_copied (Static_imported c) (Static_imported c)
  | Copied_static_immutable_array (elements elements' : list value) :
    Forall2 value_copied elements elements' ->
    allocated_copied (Static_immutable_array elements)
      (Static_immutable_array elements').

Ltac allocated_copied_cases H :=
  destruct H
    as [ acode aslots aslots' Haslots
       | afslot asyms adecls aslots aslots' Haslots
       | atag afields afields' Hafields | av av' Hav | abytes |
       | aelements aelements' Haelements ].

Lemma allocated_copied_refl (allocated : statically_allocated_value) :
  allocated_copied allocated allocated.
Proof.
  destruct allocated; constructor;
    auto using value_copied_refl, values_copied_refl, slots_copied_refl.
Qed.

(* Code with some of the [Let]s of its body copied. *)
Definition code_with_lets (lets : list path) (c : code) : code :=
  match c with
  | Code params my_closure ret exn metadata body =>
    Code params my_closure ret exn metadata
      (copy_lets_expr ids lets root body)
  end.

Definition code_lets (c c' : code) : Prop :=
  ∃ lets : list path, c' = code_with_lets lets c.

Lemma code_lets_refl (c : code) : code_lets c c.
Proof.
  exists []. destruct c as [params my_closure ret exn metadata body]. simpl.
  rewrite copy_lets_expr_nil. reflexivity.
Qed.

(* The third static environment binds the code ids and the symbols
   that the second binds, and no other: a code id to the code with
   some [Let]s copied, a symbol to the same thing with related
   parts. *)
Definition symbols_copied (statics statics' : static_env) : Prop :=
  ∀ (s : symbol) (allocated : statically_allocated_value),
    lookup s (static_values statics) = Some allocated ->
    ∃ allocated' : statically_allocated_value,
      lookup s (static_values statics') = Some allocated'
      ∧ allocated_copied allocated allocated'.

Definition statics_copied (statics statics' : static_env) : Prop :=
  (∀ id : code_id,
    match lookup id (static_code statics) with
    | Some c =>
      ∃ c' : code, lookup id (static_code statics') = Some c' ∧ code_lets c c'
    | None => lookup id (static_code statics') = None
    end)
  ∧ symbols_copied statics statics'
  ∧ (∀ s : symbol,
      lookup s (static_values statics) = None ->
      lookup s (static_values statics') = None).

Lemma statics_copied_refl (statics : static_env) :
  statics_copied statics statics.
Proof.
  split; [ | split ].
  - intros id. destruct (lookup id (static_code statics)) as [c | ];
      [ | reflexivity ].
    exists c. split; [ reflexivity | apply code_lets_refl ].
  - intros s allocated Hs. exists allocated.
    split; [ exact Hs | apply allocated_copied_refl ].
  - intros s Hs. exact Hs.
Qed.

(* ======================================================================== *)
(* The observations answer alike of two related values                      *)
(* ======================================================================== *)

Section Observations.

Variables statics statics' : static_env.
Hypothesis Hsymbols : symbols_copied statics statics'.

Lemma block_of_copied (v v' : value) (block_tag : Z)
    (block_fields : list value) :
  value_copied v v' ->
  block_of statics v = Some (block_tag, block_fields) ->
  ∃ block_fields' : list value,
    block_of statics' v' = Some (block_tag, block_fields')
    ∧ Forall2 value_copied block_fields block_fields'.
Proof.
  intros Hv. copied_cases Hv; simpl; try discriminate.
  - destruct (lookup s (static_values statics)) as [allocated | ] eqn:Hs;
      [ | discriminate ].
    destruct (Hsymbols s allocated Hs) as (allocated' & -> & Hallocated).
    allocated_copied_cases Hallocated; try discriminate.
    intros [= <- <-]. eauto.
  - intros [= <- <-]. eauto.
Qed.

Lemma boxed_number_of_copied (v v' : value) (number : const) :
  value_copied v v' ->
  boxed_number_of statics v = Some number ->
  boxed_number_of statics' v' = Some number.
Proof.
  intros Hv. copied_cases Hv; simpl; try discriminate;
    try exact (fun H => H).
  destruct (lookup s (static_values statics)) as [allocated | ] eqn:Hs;
    [ | discriminate ].
  destruct (Hsymbols s allocated Hs) as (allocated' & -> & Hallocated).
  allocated_copied_cases Hallocated; try discriminate.
  destruct av; try discriminate. inversion Hav. exact (fun H => H).
Qed.

Lemma string_of_copied (v v' : value) (bytes : list Z) :
  value_copied v v' ->
  string_of statics v = Some bytes ->
  string_of statics' v' = Some bytes.
Proof.
  intros Hv. copied_cases Hv; simpl; try discriminate.
  destruct (lookup s (static_values statics)) as [allocated | ] eqn:Hs;
    [ | discriminate ].
  destruct (Hsymbols s allocated Hs) as (allocated' & -> & Hallocated).
  allocated_copied_cases Hallocated; try discriminate. exact (fun H => H).
Qed.

Lemma array_of_copied (v v' : value) (elements : list value) :
  value_copied v v' ->
  array_of statics v = Some elements ->
  ∃ elements' : list value,
    array_of statics' v' = Some elements'
    ∧ Forall2 value_copied elements elements'.
Proof.
  intros Hv. copied_cases Hv; simpl; try discriminate.
  - destruct (lookup s (static_values statics)) as [allocated | ] eqn:Hs;
      [ | discriminate ].
    destruct (Hsymbols s allocated Hs) as (allocated' & -> & Hallocated).
    allocated_copied_cases Hallocated; try discriminate.
    intros [= <-]. eauto.
  - intros [= <-]. eauto.
Qed.

Lemma closure_in_set_copied (v v' : value) (at_slot : function_slot)
    (function_decls : list (function_slot * code_id))
    (value_slots : list (value_slot * value)) :
  value_copied v v' ->
  closure_in_set statics v = Some (at_slot, function_decls, value_slots) ->
  ∃ (function_decls' : list (function_slot * code_id))
    (value_slots' : list (value_slot * value)),
    closure_in_set statics' v' = Some (at_slot, function_decls', value_slots')
    ∧ decls_copied function_decls function_decls'
    ∧ Forall2 (slot_lifted value_copied) value_slots value_slots'.
Proof.
  intros Hv. copied_cases Hv; simpl; try discriminate.
  - destruct (lookup s (static_values statics)) as [allocated | ] eqn:Hs;
      [ | discriminate ].
    destruct (Hsymbols s allocated Hs) as (allocated' & -> & Hallocated).
    allocated_copied_cases Hallocated; try discriminate.
    + intros [= <- <- <-]. eexists. eexists.
      split; [ reflexivity | split; [ left; reflexivity | exact Haslots ] ].
    + destruct (lookup afslot adecls); [ | discriminate ]. simpl.
      intros [= <- <- <-]. eexists. eexists.
      split; [ reflexivity | split; [ left; reflexivity | exact Haslots ] ].
  - intros [= <- <- <-]. eexists. eexists.
    split; [ reflexivity | split; [ | exact Hslots ] ].
    destruct Hcode as [-> | ->]; [ left | right ]; reflexivity.
  - pose proof (lookup_decls_copied fslot decls decls' Hdecls) as Hlookup.
    destruct (lookup fslot decls) as [code0 | ]; [ | discriminate ].
    destruct Hlookup as (code0' & -> & _). simpl.
    intros [= <- <- <-]. eexists. eexists.
    split; [ reflexivity | split; assumption ].
Qed.

(* The code of a closure is its code or the copy; that of a closure
   that is a symbol is its code. *)
Lemma closure_of_copied (v v' : value) (called : code_id)
    (value_slots : list (value_slot * value)) :
  value_copied v v' ->
  closure_of statics v = Some (called, value_slots) ->
  ∃ (called' : code_id) (value_slots' : list (value_slot * value)),
    closure_of statics' v' = Some (called', value_slots')
    ∧ id_copied called called'
    ∧ (∀ s : symbol, v = Value_symbol s -> called' = called)
    ∧ Forall2 (slot_lifted value_copied) value_slots value_slots'.
Proof.
  intros Hv. copied_cases Hv; simpl; try discriminate.
  - destruct (lookup s (static_values statics)) as [allocated | ] eqn:Hs;
      [ | discriminate ].
    destruct (Hsymbols s allocated Hs) as (allocated' & -> & Hallocated).
    allocated_copied_cases Hallocated; try discriminate.
    + intros [= <- <-]. eexists. eexists.
      split; [ reflexivity | ].
      split; [ left; reflexivity | split; [ reflexivity | exact Haslots ] ].
    + destruct (lookup afslot adecls); [ | discriminate ]. simpl.
      intros [= <- <-]. eexists. eexists.
      split; [ reflexivity | ].
      split; [ left; reflexivity | split; [ reflexivity | exact Haslots ] ].
  - intros [= <- <-]. eexists. eexists.
    split; [ reflexivity | ].
    split; [ exact Hcode | split; [ discriminate | exact Hslots ] ].
  - pose proof (lookup_decls_copied fslot decls decls' Hdecls) as Hlookup.
    destruct (lookup fslot decls) as [code0 | ]; [ | discriminate ].
    destruct Hlookup as (code0' & -> & Hcode0). simpl.
    intros [= <- <-]. eexists. eexists.
    split; [ reflexivity | ].
    split; [ exact Hcode0 | split; [ discriminate | exact Hslots ] ].
Qed.

Lemma project_value_slot_copied (project_from : function_slot)
    (slot : value_slot) (v v' w : value) :
  value_copied v v' ->
  project_value_slot statics project_from slot v = Some w ->
  ∃ w' : value,
    project_value_slot statics' project_from slot v' = Some w'
    ∧ value_copied w w'.
Proof.
  intros Hv. unfold project_value_slot.
  destruct (closure_in_set statics v) as [[[at_slot decls] slots] | ] eqn:Hc;
    [ | discriminate ].
  destruct (closure_in_set_copied v v' at_slot decls slots Hv Hc)
    as (decls' & slots' & -> & _ & Hslots).
  destruct (Pos.eqb at_slot project_from); [ | discriminate ].
  exact (lookup_slots_copied slot slots slots' w Hslots).
Qed.

Lemma project_function_slot_copied (move_from move_to : function_slot)
    (v v' w : value) :
  value_copied v v' ->
  project_function_slot statics move_from move_to v = Some w ->
  ∃ w' : value,
    project_function_slot statics' move_from move_to v' = Some w'
    ∧ value_copied w w'.
Proof.
  intros Hv. unfold project_function_slot.
  copied_cases Hv; try discriminate.
  - (* A symbol: bound alike in both runs. *)
    destruct (lookup s (static_values statics)) as [allocated | ] eqn:Hs;
      [ | discriminate ].
    destruct (Hsymbols s allocated Hs) as (allocated' & -> & Hallocated).
    allocated_copied_cases Hallocated; try discriminate.
    + destruct (Pos.eqb move_from unnamed_function_slot); [ | discriminate ].
      destruct (Pos.eqb move_to unnamed_function_slot); [ | discriminate ].
      intros [= <-]. eexists. split; [ reflexivity | apply Copied_symbol ].
    + destruct (Pos.eqb afslot move_from); [ | discriminate ].
      destruct (lookup afslot adecls); [ | discriminate ].
      destruct (lookup move_to adecls); [ | discriminate ]. simpl.
      destruct (lookup move_to asyms) as [s' | ]; [ | discriminate ]. simpl.
      intros [= <-]. eexists. split; [ reflexivity | apply Copied_symbol ].
  - (* A closure of a set of one function. *)
    destruct (Pos.eqb move_from unnamed_function_slot); [ | discriminate ].
    destruct (Pos.eqb move_to unnamed_function_slot); [ | discriminate ].
    intros [= <-]. eexists. split; [ reflexivity | ].
    apply Copied_closure; assumption.
  - (* A closure of a set. *)
    destruct (Pos.eqb fslot move_from); [ | discriminate ].
    pose proof (lookup_decls_copied fslot decls decls' Hdecls) as Hat.
    destruct (lookup fslot decls); [ | discriminate ].
    destruct Hat as (code_at & -> & _).
    pose proof (lookup_decls_copied move_to decls decls' Hdecls) as Hto.
    destruct (lookup move_to decls); [ | discriminate ].
    destruct Hto as (code_to & -> & _). simpl.
    intros [= <-]. eexists. split; [ reflexivity | ].
    apply Copied_closure_of_set; assumption.
Qed.

(* The relation is one that the observations cannot tell from equality,
   so no primitive can. *)
Lemma value_copied_respects :
  respects_observations value_copied statics statics'.
Proof.
  split; [ | split ].
  - split.
    + intros c v' Hv. inversion Hv. reflexivity.
    + intros v c Hv. inversion Hv. reflexivity.
  - split; [ | split ].
    + intros c. split; constructor.
    + intros tag fields fields' Hfields. constructor. exact Hfields.
    + intros elements elements' Helements. constructor. exact Helements.
  - intros v v' Hv.
    split;
      [ | split; [ | split; [ | split; [ | split; [ | split; [ | split ] ] ] ] ]
      ].
    + intros tag fields. exact (block_of_copied v v' tag fields Hv).
    + intros c. exact (boxed_number_of_copied v v' c Hv).
    + intros bytes. exact (string_of_copied v v' bytes Hv).
    + intros Hclosure.
      destruct (closure_of statics v) as [[called cslots] | ] eqn:Hc;
        [ | congruence ].
      destruct (closure_of_copied v v' called cslots Hv Hc)
        as (called' & cslots' & -> & _).
      discriminate.
    + pose proof (partial_application_copied v v' Hv) as Hpartial.
      destruct (partial_application_of v) as [[[closure applied] arity] | ];
        [ | congruence ].
      destruct Hpartial as (closure' & applied' & -> & _). discriminate.
    + copied_cases Hv; simpl; try reflexivity.
      destruct (lookup s (static_values statics)) as [allocated | ] eqn:Hs;
        [ | discriminate ].
      destruct (Hsymbols s allocated Hs) as (allocated' & -> & _).
      reflexivity.
    + intros [s0 c] Hs0.
      apply imported_of_symbol in Hs0 as [-> Hs0].
      inversion Hv; subst. simpl.
      destruct (Hsymbols s0 _ Hs0) as (allocated' & -> & Hallocated).
      inversion Hallocated. reflexivity.
    + intros elements. exact (array_of_copied v v' elements Hv).
Qed.

End Observations.

(* ======================================================================== *)
(* Evaluation in related environments                                       *)
(* ======================================================================== *)

(* Every variable of the second environment is bound in the third, to
   a related value. *)
Definition env_copied (vars vars' : env) : Prop :=
  ∀ (x : variable) (v : value),
    lookup x vars = Some v ->
    ∃ v' : value, lookup x vars' = Some v' ∧ value_copied v v'.

Lemma env_copied_nil : env_copied [] [].
Proof. intros x v Hx. discriminate Hx. Qed.

Lemma env_copied_cons (x : variable) (v v' : value) (vars vars' : env) :
  value_copied v v' ->
  env_copied vars vars' ->
  env_copied ((x, v) :: vars) ((x, v') :: vars').
Proof.
  intros Hv Hvars y w. simpl.
  destruct (Pos.eqb y x); [ intros [= <-]; eauto | apply Hvars ].
Qed.

Lemma bind_params_copied (params : list variable) :
  ∀ (vs vs' : list value) (vars vars' bound : env),
    Forall2 value_copied vs vs' ->
    env_copied vars vars' ->
    bind_params params vs vars = Some bound ->
    ∃ bound' : env,
      bind_params params vs' vars' = Some bound' ∧ env_copied bound bound'.
Proof.
  induction params as [ | p params IH]; intros vs vs' vars vars' bound Hvs;
    destruct Hvs as [ | v v' vs vs' Hv Hvs]; simpl; try discriminate.
  - intros Hvars [= <-]. eauto.
  - intros Hvars. apply (IH vs vs' _ _ bound Hvs).
    apply env_copied_cons; assumption.
Qed.

Lemma eval_simple_copied (vars vars' : env) (s : simple) (v : value) :
  env_copied vars vars' ->
  eval_simple vars s = Some v ->
  ∃ v' : value, eval_simple vars' s = Some v' ∧ value_copied v v'.
Proof.
  intros Hvars. destruct s as [x | sym | c]; simpl.
  - apply Hvars.
  - intros [= <-]. eexists. split; [ reflexivity | apply Copied_symbol ].
  - intros [= <-]. eexists. split; [ reflexivity | apply Copied_const ].
Qed.

Lemma eval_simples_copied (vars vars' : env) (args : list simple) :
  env_copied vars vars' ->
  ∀ vs : list value,
    eval_simples vars args = Some vs ->
    ∃ vs' : list value,
      eval_simples vars' args = Some vs' ∧ Forall2 value_copied vs vs'.
Proof.
  intros Hvars. induction args as [ | s args IH]; intros vs; simpl.
  - intros [= <-]. exists []. auto.
  - destruct (eval_simple vars s) as [v | ] eqn:Hs; [ | discriminate ].
    destruct (eval_simples vars args) as [rest | ]; [ | discriminate ].
    intros [= <-].
    destruct (eval_simple_copied vars vars' s v Hvars Hs) as (v' & -> & Hv).
    destruct (IH rest eq_refl) as (rest' & -> & Hrest).
    exists (v' :: rest'). auto.
Qed.

Lemma eval_value_slots_copied (vars vars' : env)
    (slots : list (value_slot * simple)) :
  env_copied vars vars' ->
  ∀ evaluated : list (value_slot * value),
    eval_value_slots vars slots = Some evaluated ->
    ∃ evaluated' : list (value_slot * value),
      eval_value_slots vars' slots = Some evaluated'
      ∧ Forall2 (slot_lifted value_copied) evaluated evaluated'.
Proof.
  intros Hvars. induction slots as [ | [slot s] slots IH]; intros evaluated;
    simpl.
  - intros [= <-]. exists []. auto.
  - destruct (eval_simple vars s) as [v | ] eqn:Hs; [ | discriminate ].
    destruct (eval_value_slots vars slots) as [rest | ]; [ | discriminate ].
    intros [= <-].
    destruct (eval_simple_copied vars vars' s v Hvars Hs) as (v' & -> & Hv).
    destruct (IH rest eq_refl) as (rest' & -> & Hrest).
    exists ((slot, v') :: rest'). split; [ reflexivity | ].
    constructor; [ split; [ reflexivity | exact Hv ] | exact Hrest ].
Qed.

(* A defining expression that has a value in the second run has one in
   the third, related. *)
Lemma eval_named_copied (statics statics' : static_env) (vars vars' : env)
    (d : named) (v : value) :
  symbols_copied statics statics' ->
  env_copied vars vars' ->
  eval_named statics vars d = Some v ->
  ∃ v' : value, eval_named statics' vars' d = Some v' ∧ value_copied v v'.
Proof.
  intros Hstatics Hvars.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure]; simpl.
  - exact (eval_simple_copied vars vars' s v Hvars).
  - destruct (eval_simples vars args) as [vs | ] eqn:Hargs;
      [ | discriminate ].
    destruct (eval_simples_copied vars vars' args Hvars vs Hargs)
      as (vs' & -> & Hvs).
    exact
      (eval_prim_related value_copied statics statics' p vs vs' v
         (value_copied_respects statics statics' Hstatics) Hvs).
  - destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots;
      [ | discriminate ].
    destruct (eval_value_slots_copied vars vars' slots Hvars vs Hslots)
      as (vs' & -> & Hvs).
    intros [= <-]. eexists. split; [ reflexivity | ].
    apply Copied_closure; [ left; reflexivity | exact Hvs ].
  - destruct (lookup fslot decls); [ | discriminate ]. simpl.
    destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots;
      [ | discriminate ].
    destruct (eval_value_slots_copied vars vars' slots Hvars vs Hslots)
      as (vs' & -> & Hvs).
    intros [= <-]. eexists. split; [ reflexivity | ].
    apply Copied_closure_of_set; [ left; reflexivity | exact Hvs ].
  - destruct (eval_simple vars closure) as [w | ] eqn:Hclosure;
      [ | discriminate ].
    destruct (eval_simple_copied vars vars' closure w Hvars Hclosure)
      as (w' & -> & Hw).
    simpl.
    destruct (closure_of statics w) as [[called cslots] | ] eqn:Hc;
      [ | discriminate ].
    destruct (closure_of_copied statics statics' Hstatics w w' called cslots
                Hw Hc)
      as (called' & cslots' & -> & _ & _ & Hcslots).
    simpl. exact (lookup_slots_copied slot cslots cslots' v Hcslots).
  - destruct (eval_simple vars closure) as [w | ] eqn:Hclosure;
      [ | discriminate ].
    destruct (eval_simple_copied vars vars' closure w Hvars Hclosure)
      as (w' & -> & Hw).
    exact (project_value_slot_copied statics statics' Hstatics pfrom slot w
             w' v Hw).
  - destruct (eval_simple vars closure) as [w | ] eqn:Hclosure;
      [ | discriminate ].
    destruct (eval_simple_copied vars vars' closure w Hvars Hclosure)
      as (w' & -> & Hw).
    exact (project_function_slot_copied statics statics' Hstatics move_from
             move_to w w' v Hw).
Qed.

(* The same where the third run has the defining expression with the
   copies of its code: a set of closures is made of the copies. *)
Lemma eval_named_copy_named (statics statics' : static_env)
    (vars vars' : env) (d : named) (v : value) :
  symbols_copied statics statics' ->
  env_copied vars vars' ->
  eval_named statics vars d = Some v ->
  ∃ v' : value,
    eval_named statics' vars' (copy_named ids d) = Some v'
    ∧ value_copied v v'.
Proof.
  intros Hstatics Hvars.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    try exact
          (eval_named_copied statics statics' vars vars' _ v Hstatics Hvars).
  - cbn [copy_named eval_named].
    destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots;
      [ | discriminate ].
    destruct (eval_value_slots_copied vars vars' slots Hvars vs Hslots)
      as (vs' & -> & Hvs).
    simpl. intros [= <-]. eexists. split; [ reflexivity | ].
    apply Copied_closure; [ right; reflexivity | exact Hvs ].
  - cbn [copy_named eval_named].
    rewrite (lookup_copy_decls ids fslot decls).
    destruct (lookup fslot decls); [ | discriminate ]. simpl.
    destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots;
      [ | discriminate ].
    destruct (eval_value_slots_copied vars vars' slots Hvars vs Hslots)
      as (vs' & -> & Hvs).
    simpl. intros [= <-]. eexists. split; [ reflexivity | ].
    apply Copied_closure_of_set; [ right; reflexivity | exact Hvs ].
Qed.

(* What the machine runs a defining expression to, on its heap, the
   third run runs it to, as it is or with the copies of its code, on a
   heap alike. *)
Lemma exec_named_copied (statics statics' : static_env) (hp hp' hp1 : heap)
    (rd rd1 rd' : read_record) (vars vars' : env) (d d' : named) (v : value) :
  symbols_copied statics statics' ->
  env_copied vars vars' ->
  heap_alike value_copied hp hp' ->
  d' = d ∨ d' = copy_named ids d ->
  exec_named statics hp rd vars d = Some (v, hp1, rd1) ->
  ∃ (v' : value) (hp1' : heap),
    exec_named statics' hp' rd' vars' d' = Some (v', hp1', rd')
    ∧ value_copied v v'
    ∧ heap_alike value_copied hp1 hp1'.
Proof.
  intros Hstatics Hvars Hheap Hd' Hexec.
  assert (Hcases :
            (∃ (p : prim) (args : list simple), d = Prim p args)
            ∨ (∀ (p : prim) (args : list simple), d <> Prim p args)).
  { destruct d; try (right; intros ? ?; discriminate).
    left. eexists. eexists. reflexivity. }
  destruct Hcases as [(p & args & ->) | Hnot].
  - assert (Hsame : d' = Prim p args)
      by (destruct Hd' as [-> | ->]; reflexivity).
    subst d'.
    apply exec_named_prim_some in Hexec as (vs & Hargs & Hprim & _).
    destruct (eval_simples_copied vars vars' args Hvars vs Hargs)
      as (vs' & Hargs' & Hvs).
    destruct
      (exec_prim_related value_copied statics statics' hp hp' hp1 p vs vs' v
         (value_copied_respects statics statics' Hstatics)
         value_copied_locations Hheap Hvs Hprim)
      as (v' & hp1' & Hprim' & Hv & Hheap1).
    exists v', hp1'. split; [ | split; assumption ].
    apply exec_named_prim_some. exists vs'. auto.
  - rewrite (exec_named_not_prim _ _ _ _ _ Hnot) in Hexec.
    destruct (eval_named statics vars d) as [w | ] eqn:Hw; [ | discriminate ].
    injection Hexec as <- <- <-.
    assert (Hnot' : ∀ (p : prim) (args : list simple), d' <> Prim p args).
    { destruct Hd' as [-> | ->]; [ exact Hnot | ].
      destruct d; simpl; try (intros ? ?; discriminate). exact Hnot. }
    assert (Heval :
      ∃ v' : value, eval_named statics' vars' d' = Some v' ∧ value_copied w v').
    { destruct Hd' as [-> | ->];
        [ exact (eval_named_copied statics statics' vars vars' d w Hstatics
                   Hvars Hw)
        | exact (eval_named_copy_named statics statics' vars vars' d w
                   Hstatics Hvars Hw) ]. }
    destruct Heval as (v' & Heval & Hv).
    exists v', hp'. split; [ | split; assumption ].
    rewrite (exec_named_not_prim _ _ _ _ _ Hnot'), Heval. reflexivity.
Qed.

Lemma eval_static_const_copied (vars vars' : env) (const : static_const)
    (allocated : statically_allocated_value) :
  env_copied vars vars' ->
  eval_static_const vars const = Some allocated ->
  ∃ allocated' : statically_allocated_value,
    eval_static_const vars' const = Some allocated'
    ∧ allocated_copied allocated allocated'.
Proof.
  intros Hvars.
  destruct const
    as [code slots | fslot syms decls slots | tag fields | contents | bytes
        | elements];
    simpl.
  - destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots;
      [ | discriminate ].
    destruct (eval_value_slots_copied vars vars' slots Hvars vs Hslots)
      as (vs' & -> & Hvs).
    intros [= <-]. eexists. split; [ reflexivity | ].
    apply Copied_static_closure, Hvs.
  - destruct (lookup fslot decls); [ | discriminate ]. simpl.
    destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots;
      [ | discriminate ].
    destruct (eval_value_slots_copied vars vars' slots Hvars vs Hslots)
      as (vs' & -> & Hvs).
    intros [= <-]. eexists. split; [ reflexivity | ].
    apply Copied_static_closure_of_set, Hvs.
  - destruct (eval_simples vars fields) as [vs | ] eqn:Hfields;
      [ | discriminate ].
    destruct (eval_simples_copied vars vars' fields Hvars vs Hfields)
      as (vs' & -> & Hvs).
    intros [= <-]. eexists. split; [ reflexivity | ].
    apply Copied_static_block, Hvs.
  - destruct (eval_simple vars contents) as [w | ] eqn:Hcontents;
      [ | discriminate ].
    destruct (eval_simple_copied vars vars' contents w Hvars Hcontents)
      as (w' & -> & Hw).
    intros [= <-]. eexists. split; [ reflexivity | ].
    apply Copied_static_boxed_number, Hw.
  - intros [= <-]. eexists. split; [ reflexivity | ].
    apply Copied_static_string.
  - destruct (eval_simples vars elements) as [vs | ] eqn:Helements;
      [ | discriminate ].
    destruct (eval_simples_copied vars vars' elements Hvars vs Helements)
      as (vs' & -> & Hvs).
    intros [= <-]. eexists. split; [ reflexivity | ].
    apply Copied_static_immutable_array, Hvs.
Qed.

(* ======================================================================== *)
(* Static bindings                                                          *)
(* ======================================================================== *)

(* A binding of code binds the code with some [Let]s copied, or the
   code as it is. *)
Lemma copy_lets_binding_code (id : code_id) (c : code) :
  ∃ c' : code,
    copy_lets_binding ids in_code (Bind_code id c) = Bind_code id c'
    ∧ code_lets c c'.
Proof.
  destruct c as [params my_closure ret exn metadata body]. simpl.
  destruct (lookup id in_code) as [lets | ].
  - eexists. split; [ reflexivity | ]. exists lets. reflexivity.
  - eexists. split; [ reflexivity | ].
    apply (code_lets_refl (Code params my_closure ret exn metadata body)).
Qed.

(* A group of bindings that the second run evaluates, the third does,
   with its code rewritten: to related static environments. *)
Lemma eval_static_bindings_copied (vars vars' : env) :
  ∀ (bindings : list static_binding) (statics statics' extended : static_env),
    env_copied vars vars' ->
    statics_copied statics statics' ->
    eval_static_bindings vars bindings statics = Some extended ->
    ∃ extended' : static_env,
      eval_static_bindings vars'
        (map (copy_lets_binding ids in_code) bindings) statics'
      = Some extended'
      ∧ statics_copied extended extended'.
Proof.
  induction bindings as [ | binding bindings IH];
    intros statics statics' extended Hvars Hstatics.
  - simpl. intros [= <-]. eauto.
  - destruct binding as [id c | id | sym const].
    + destruct (copy_lets_binding_code id c) as (c' & Hbinding & Hc).
      cbn [map]. rewrite Hbinding. cbn [eval_static_bindings].
      pose proof (proj1 Hstatics id) as Hid.
      destruct (lookup id (static_code statics)) eqn:Hsource;
        [ discriminate | ].
      rewrite Hid. apply IH; [ exact Hvars | ].
      split; [ | split ]; cbn [static_code static_values].
      * intros id0. rewrite !lookup_cons.
        destruct (Pos.eqb id0 id);
          [ exists c'; split; [ reflexivity | exact Hc ]
          | apply (proj1 Hstatics) ].
      * exact (proj1 (proj2 Hstatics)).
      * exact (proj2 (proj2 Hstatics)).
    + cbn [map copy_lets_binding eval_static_bindings]. apply IH; assumption.
    + cbn [map copy_lets_binding eval_static_bindings].
      destruct (lookup sym (static_values statics)) eqn:Hsym;
        [ discriminate | ].
      rewrite (proj2 (proj2 Hstatics) sym Hsym).
      destruct (eval_static_const vars const) as [allocated | ] eqn:Hconst;
        [ | discriminate ].
      destruct (eval_static_const_copied vars vars' const allocated Hvars
                  Hconst)
        as (allocated' & -> & Hallocated).
      simpl. apply IH; [ exact Hvars | ].
      split; [ | split ]; cbn [static_code static_values].
      * exact (proj1 Hstatics).
      * intros s a. cbn [static_values]. rewrite !lookup_cons.
        destruct (Pos.eqb s sym);
          [ intros [= <-]; exists allocated';
            split; [ reflexivity | exact Hallocated ]
          | apply (proj1 (proj2 Hstatics)) ].
      * intros s. rewrite !lookup_cons.
        destruct (Pos.eqb s sym);
          [ discriminate | apply (proj2 (proj2 Hstatics)) ].
Qed.

(* ======================================================================== *)
(* Terms, continuations and states                                          *)
(* ======================================================================== *)

(* A term of the third run: one of the second with some of its [Let]s
   copied, and with those of the code that it binds copied as
   [in_code] says. *)
Definition expr_lets (e e' : expr) : Prop :=
  ∃ (lets : list path) (at_path : path),
    e' = copy_lets_in_code ids in_code (copy_lets_expr ids lets at_path e).

Definition handler_lets (h h' : continuation_handler) : Prop :=
  ∃ (lets : list path) (at_path : path),
    h'
    = copy_lets_in_code_handler ids in_code
        (copy_lets_handler ids lets at_path h).

Definition handlers_lets
    (handlers handlers' : list (continuation * continuation_handler))
    : Prop :=
  Forall2
    (fun kh kh' => fst kh = fst kh' ∧ handler_lets (snd kh) (snd kh'))
    handlers handlers'.

(* The handlers of a recursive group, rewritten. *)
Lemma rec_handlers_lets (lets : list path)
    (handlers : list (continuation * continuation_handler)) :
  ∀ q : path,
    handlers_lets handlers
      (map (fun '(k, h) => (k, copy_lets_in_code_handler ids in_code h))
         (rec_handlers_map (copy_lets_handler ids lets) q handlers)).
Proof.
  induction handlers as [ | [k h] handlers IH]; intros q; simpl; constructor;
    [ | apply IH ].
  split; [ reflexivity | ]. exists lets, (xO q). reflexivity.
Qed.

Lemma rec_handlers_bind_statics (lets : list path)
    (handlers : list (continuation * continuation_handler)) :
  ∀ q : path,
    handlers_bind_statics
      (map (fun '(k, h) => (k, copy_lets_in_code_handler ids in_code h))
         (rec_handlers_map (copy_lets_handler ids lets) q handlers))
    = handlers_bind_statics handlers.
Proof.
  unfold handlers_bind_statics.
  induction handlers as [ | [k h] handlers IH]; intros q; simpl;
    [ reflexivity | ].
  rewrite handler_binds_statics_copy_lets_in_code,
    handler_binds_statics_copy_lets, IH.
  reflexivity.
Qed.

#[local] Set Warnings "-register-all".
Inductive cont_copied : cont_value -> cont_value -> Prop :=
  | Outer_copied (k : continuation) : cont_copied (Outer k) (Outer k)
  | Closure_copied (h h' : continuation_handler) (vars vars' : env)
      (conts conts' : cont_env)
      (recursive recursive' : list (continuation * continuation_handler)) :
    handler_lets h h' ->
    env_copied vars vars' ->
    Forall2
      (fun kc kc' => fst kc = fst kc' ∧ cont_copied (snd kc) (snd kc'))
      conts conts' ->
    handlers_lets recursive recursive' ->
    cont_copied (Closure h vars conts recursive)
      (Closure h' vars' conts' recursive')
  | Extra_args_copied (c c' : cont_value) (extra extra' : list value) :
    cont_copied c c' ->
    Forall2 value_copied extra extra' ->
    cont_copied (Extra_args c extra) (Extra_args c' extra')
  | Over_application_copied (remaining remaining' : list value)
      (remaining_arity return_arity : arity)
      (return_to return_to' exn_to exn_to' : cont_value) :
    Forall2 value_copied remaining remaining' ->
    cont_copied return_to return_to' ->
    cont_copied exn_to exn_to' ->
    cont_copied
      (Over_application remaining remaining_arity return_arity return_to
         exn_to)
      (Over_application remaining' remaining_arity return_arity return_to'
         exn_to').
#[local] Set Warnings "register-all".

Definition conts_copied (conts conts' : cont_env) : Prop :=
  Forall2
    (fun kc kc' => fst kc = fst kc' ∧ cont_copied (snd kc) (snd kc'))
    conts conts'.

Lemma with_extra_args_copied (c c' : cont_value) (extra extra' : list value) :
  cont_copied c c' ->
  Forall2 value_copied extra extra' ->
  cont_copied (with_extra_args c extra) (with_extra_args c' extra').
Proof.
  intros Hc Hextra.
  destruct Hextra as [ | v v' extra extra' Hv Hextra]; [ exact Hc | ].
  apply Extra_args_copied; [ exact Hc | constructor; assumption ].
Qed.

Lemma resolve_copied (k : continuation) (conts conts' : cont_env) :
  conts_copied conts conts' ->
  cont_copied (resolve k conts) (resolve k conts').
Proof.
  unfold resolve.
  induction 1 as [ | [k1 c] [k1' c'] conts conts' [Hk Hc] _ IH]; simpl;
    [ apply Outer_copied | ].
  simpl in Hk. subst k1'. destruct (Pos.eqb k k1); [ exact Hc | exact IH ].
Qed.

(* [all] is the whole group, kept apart from the handlers still to be gone
   through. *)
Lemma rec_closures_copied_aux (vars vars' : env) (conts conts' : cont_env)
    (all all' : list (continuation * continuation_handler)) :
  env_copied vars vars' ->
  conts_copied conts conts' ->
  handlers_lets all all' ->
  ∀ handlers handlers' : list (continuation * continuation_handler),
    handlers_lets handlers handlers' ->
    conts_copied
      (map (fun '(k, h) => (k, Closure h vars conts all)) handlers)
      (map (fun '(k, h) => (k, Closure h vars' conts' all')) handlers').
Proof.
  intros Hvars Hconts Hall.
  induction 1 as [ | [k h] [k' h'] handlers handlers' [Hk Hh] _ IH]; simpl;
    constructor; [ | exact IH ].
  simpl in Hk, Hh. subst k'. split; [ reflexivity | ]. simpl.
  apply Closure_copied; assumption.
Qed.

Lemma rec_closures_copied (vars vars' : env) (conts conts' : cont_env)
    (handlers handlers' : list (continuation * continuation_handler)) :
  env_copied vars vars' ->
  conts_copied conts conts' ->
  handlers_lets handlers handlers' ->
  conts_copied (rec_closures handlers vars conts ++ conts)
    (rec_closures handlers' vars' conts' ++ conts').
Proof.
  intros Hvars Hconts Hhandlers. apply Forall2_app; [ | exact Hconts ].
  unfold rec_closures. apply rec_closures_copied_aux; assumption.
Qed.

(* The second state and the third. The read records are not
   compared. *)
Inductive state_copied : state -> state -> Prop :=
  | State_copied (e e' : expr) (vars vars' : env) (conts conts' : cont_env)
      (statics statics' : static_env) (hp hp' : heap)
      (rd rd' : read_record) :
    expr_lets e e' ->
    env_copied vars vars' ->
    conts_copied conts conts' ->
    statics_copied statics statics' ->
    heap_alike value_copied hp hp' ->
    state_copied (State e vars conts statics hp rd)
      (State e' vars' conts' statics' hp' rd').

(* What the third run answers to a result of the second: nothing is
   asked if the second is undefined; an exit is answered by an exit
   through the same continuation, with related values and a heap
   alike; a step, by a step to a related state. *)
Definition result_lets (result result' : step_result) : Prop :=
  match result with
  | Undefined => True
  | Exit k exit_vs hp _ =>
    ∃ (exit_vs' : list value) (hp' : heap) (rd' : read_record),
      result' = Exit k exit_vs' hp' rd'
      ∧ Forall2 value_copied exit_vs exit_vs'
      ∧ heap_alike value_copied hp hp'
  | Step t => ∃ t' : state, result' = Step t' ∧ state_copied t t'
  end.

(* ======================================================================== *)
(* Calls                                                                    *)
(* ======================================================================== *)

(* The copy of a code id is bound to what the id is bound to. *)
Definition follows (statics : static_env) : Prop :=
  ∀ (id id' : code_id) (c : code),
    lookup id ids = Some id' ->
    lookup id (static_code statics) = Some c ->
    lookup id' (static_code statics) = Some c.

(* A direct call is of a symbol, or names code that has no copy. *)
Definition kind_ok (callee : value) (kind : call_kind) : Prop :=
  match kind with
  | Direct named =>
    (∃ s : symbol, callee = Value_symbol s) ∨ lookup named ids = None
  | Indirect_unknown_arity => True
  end.

(* Code with some [Let]s copied has the parameters, the arities and the
   continuations of the code, and binds what it binds; where it binds
   nothing static, its body is a term of the third run. *)
Lemma code_lets_parts (c c' : code) :
  code_lets c c' ->
  code_is_tupled c' = code_is_tupled c
  ∧ code_params_arity c' = code_params_arity c
  ∧ code_result_arity c' = code_result_arity c
  ∧ code_params c' = code_params c
  ∧ code_my_closure c' = code_my_closure c
  ∧ code_return_continuation c' = code_return_continuation c
  ∧ code_exn_continuation c' = code_exn_continuation c
  ∧ binds_statics (code_body c') = binds_statics (code_body c)
  ∧ (binds_statics (code_body c) = false ->
     expr_lets (code_body c) (code_body c')).
Proof.
  intros [lets ->]. destruct c as [params my_closure ret exn metadata body].
  cbn [code_with_lets code_body].
  split; [ reflexivity | ]. split; [ reflexivity | ].
  split; [ reflexivity | ]. split; [ reflexivity | ].
  split; [ reflexivity | ]. split; [ reflexivity | ].
  split; [ reflexivity | ].
  split; [ apply binds_statics_copy_lets | ].
  intros Hbinds. exists lets, root. symmetry.
  apply copy_lets_in_code_none. rewrite binds_statics_copy_lets.
  exact Hbinds.
Qed.

Inductive shape_copied : call_shape -> call_shape -> Prop :=
  | Full_call_copied (closure closure' : value) (c c' : code)
      (vs vs' : list value) (args_arity : arity) :
    value_copied closure closure' ->
    code_lets c c' ->
    Forall2 value_copied vs vs' ->
    shape_copied (Full_call closure c vs args_arity)
      (Full_call closure' c' vs' args_arity)
  | Over_call_copied (closure closure' : value) (c c' : code)
      (vs vs' remaining remaining' : list value)
      (args_arity remaining_arity : arity) :
    value_copied closure closure' ->
    code_lets c c' ->
    Forall2 value_copied vs vs' ->
    Forall2 value_copied remaining remaining' ->
    shape_copied
      (Over_call closure c vs remaining args_arity remaining_arity)
      (Over_call closure' c' vs' remaining' args_arity remaining_arity)
  | Partial_call_copied (partial partial' : value) :
    value_copied partial partial' ->
    shape_copied (Partial_call partial) (Partial_call partial').

(* What the call of a closure is in the second run, the call of the
   related closure is in the third: of the code that the closure has
   there, which is the code or its copy, bound to the code with some
   [Let]s copied. *)
Lemma shape_of_closure_call_copied (statics statics' : static_env)
    (closure closure' : value) (kind : call_kind) (vs vs' : list value)
    (args_arity return_arity : arity) (shape : call_shape) :
  statics_copied statics statics' ->
  follows statics ->
  value_copied closure closure' ->
  Forall2 value_copied vs vs' ->
  kind_ok closure kind ->
  shape_of_closure_call statics closure kind vs args_arity return_arity
  = Some shape ->
  ∃ shape' : call_shape,
    shape_of_closure_call statics' closure' kind vs' args_arity return_arity
    = Some shape'
    ∧ shape_copied shape shape'.
Proof.
  intros Hstatics Hfollows Hclosure Hvs Hkind. unfold shape_of_closure_call.
  destruct (closure_of statics closure) as [[called slots] | ] eqn:Hc;
    [ | discriminate ].
  destruct (closure_of_copied statics statics' (proj1 (proj2 Hstatics))
              closure closure' called slots Hclosure Hc)
    as (called' & slots' & Hc' & Hid & Hsym & _).
  rewrite Hc'. cbv zeta.
  destruct
    (match kind with
     | Direct named => Pos.eqb called named
     | Indirect_unknown_arity => true
     end) eqn:Hexpected;
    [ | discriminate ].
  assert (Hexpected' :
    match kind with
    | Direct named => Pos.eqb called' named
    | Indirect_unknown_arity => true
    end = true).
  { destruct kind as [named | ]; [ | reflexivity ].
    apply Pos.eqb_eq in Hexpected. subst named. apply Pos.eqb_eq.
    destruct Hkind as [[s ->] | Hnone]; [ exact (Hsym s eq_refl) | ].
    destruct Hid as [-> | ->]; [ reflexivity | ].
    unfold copy_of. rewrite Hnone. reflexivity. }
  rewrite Hexpected'.
  destruct (lookup called (static_code statics)) as [c | ] eqn:Hcode;
    [ | discriminate ].
  assert (Hcode' :
    ∃ c' : code,
      lookup called' (static_code statics') = Some c' ∧ code_lets c c').
  { assert (Hsource : lookup called' (static_code statics) = Some c).
    { destruct Hid as [-> | ->]; [ exact Hcode | ]. unfold copy_of.
      destruct (lookup called ids) as [id' | ] eqn:Hcopy;
        [ exact (Hfollows called id' c Hcopy Hcode) | exact Hcode ]. }
    pose proof (proj1 Hstatics called') as H. rewrite Hsource in H.
    exact H. }
  destruct Hcode' as (c' & -> & Hc_lets).
  cbv iota.
  destruct (code_lets_parts c c' Hc_lets)
    as (Htupled & Hparams & Hresult & _).
  destruct (arguments_for_code statics c vs args_arity)
    as [[ws args_arity'] | ] eqn:Hargs; [ | discriminate ].
  destruct
    (arguments_for_code_related value_copied statics statics' c c' vs vs'
       args_arity ws args_arity'
       (value_copied_respects statics statics' (proj1 (proj2 Hstatics)))
       Htupled Hparams Hvs (kinds_of_values_copied vs vs' Hvs) Hargs)
    as (ws' & -> & Hws).
  simpl. unfold shape_of_code_call. cbv zeta.
  rewrite Hparams, Hresult, (kinds_of_values_copied ws ws' Hws), Htupled.
  clear Hargs Hvs vs vs'. rename ws into vs, ws' into vs', Hws into Hvs.
  rename args_arity into call_arity, args_arity' into args_arity.
  destruct (Nat.eqb (length args_arity) (length (code_params_arity c))).
  - intros [= <-]. eexists. split; [ reflexivity | ].
    apply Full_call_copied; assumption.
  - destruct
      (negb
         (kinds_eqb (map kind_of_value vs) (unarize args_arity)
          && partial_or_over_application args_arity return_arity
               (code_params_arity c) (code_result_arity c)));
      [ discriminate | ].
    destruct
      (Forall2_firstn_skipn value_copied
         (cardinal_unarized (code_params_arity c)) vs vs' Hvs)
      as [Hfirst Hrest].
    destruct (Nat.ltb (length (code_params_arity c)) (length args_arity));
      [ intros [= <-]; eexists; (split; [ reflexivity | ]);
        apply Over_call_copied; assumption
      | ].
    destruct (code_is_tupled c); [ discriminate | ].
    intros [= <-]. eexists. split; [ reflexivity | ].
    apply Partial_call_copied, Copied_partial_application; assumption.
Qed.

(* The same of a call: the closure that a partial application keeps is
   related with it, and is called indirectly. *)
Lemma shape_of_call_copied (statics statics' : static_env)
    (callee callee' : value) (kind : call_kind) (vs vs' : list value)
    (args_arity return_arity : arity) (shape : call_shape) :
  statics_copied statics statics' ->
  follows statics ->
  value_copied callee callee' ->
  Forall2 value_copied vs vs' ->
  kind_ok callee kind ->
  shape_of_call statics callee kind vs args_arity return_arity = Some shape ->
  ∃ shape' : call_shape,
    shape_of_call statics' callee' kind vs' args_arity return_arity
    = Some shape'
    ∧ shape_copied shape shape'.
Proof.
  intros Hstatics Hfollows Hcallee Hvs Hkind. unfold shape_of_call.
  destruct kind as [named | ];
    [ apply shape_of_closure_call_copied; assumption | ].
  pose proof (partial_application_copied callee callee' Hcallee) as Hpartial.
  destruct (partial_application_of callee)
    as [[[closure applied] applied_arity] | ].
  - destruct Hpartial as (closure' & applied' & -> & Hclosure & Happlied).
    simpl.
    destruct args_arity as [ | component args_arity]; [ discriminate | ].
    rewrite (kinds_of_values_copied vs vs' Hvs).
    destruct (kinds_eqb _ _); [ | discriminate ].
    apply shape_of_closure_call_copied;
      [ exact Hstatics | exact Hfollows | exact Hclosure
      | apply Forall2_app; assumption | exact I ].
  - rewrite Hpartial. apply shape_of_closure_call_copied; assumption.
Qed.

(* A full call: the body of the code with some [Let]s copied, under
   related variables and continuations. *)
Lemma call_code_copied (closure closure' : value) (c c' : code)
    (vs vs' : list value) (args_arity return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics statics' : static_env) (hp hp' : heap) (rd rd' : read_record) :
  value_copied closure closure' ->
  code_lets c c' ->
  Forall2 value_copied vs vs' ->
  cont_copied return_to return_to' ->
  cont_copied exn_to exn_to' ->
  statics_copied statics statics' ->
  heap_alike value_copied hp hp' ->
  result_lets
    (call_code closure c vs args_arity return_arity return_to exn_to statics
       hp rd)
    (call_code closure' c' vs' args_arity return_arity return_to' exn_to'
       statics' hp' rd').
Proof.
  intros Hclosure Hc Hvs Hreturn Hexn Hstatics Hheap.
  destruct (code_lets_parts c c' Hc)
    as (_ & Hparams_arity & Hresult & Hparams & Hmy & Hret & Hexn_k & Hbinds
        & Hbody).
  unfold call_code.
  rewrite Hbinds, Hparams_arity, Hresult, Hparams, Hmy, Hret, Hexn_k,
    (kinds_of_values_copied vs vs' Hvs).
  destruct (binds_statics (code_body c)) eqn:Hb; [ exact I | ].
  destruct (negb _); [ exact I | ].
  destruct (bind_params (code_params c) vs [(code_my_closure c, closure)])
    as [bound | ] eqn:Hbind;
    [ | exact I ].
  destruct
    (bind_params_copied (code_params c) vs vs'
       [(code_my_closure c, closure)] [(code_my_closure c, closure')] bound
       Hvs
       (env_copied_cons (code_my_closure c) closure closure' [] [] Hclosure
          env_copied_nil)
       Hbind)
    as (bound' & -> & Hbound).
  cbn [result_lets]. eexists. split; [ reflexivity | ].
  apply State_copied;
    [ exact (Hbody eq_refl) | exact Hbound | | exact Hstatics | exact Hheap ].
  constructor; [ split; [ reflexivity | exact Hreturn ] | ].
  constructor; [ split; [ reflexivity | exact Hexn ] | constructor ].
Qed.

Lemma run_call_copied (return_partial return_partial' : value -> step_result)
    (shape shape' : call_shape) (return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics statics' : static_env) (hp hp' : heap) (rd rd' : read_record) :
  shape_copied shape shape' ->
  cont_copied return_to return_to' ->
  cont_copied exn_to exn_to' ->
  statics_copied statics statics' ->
  heap_alike value_copied hp hp' ->
  (∀ partial partial' : value,
    value_copied partial partial' ->
    result_lets (return_partial partial) (return_partial' partial')) ->
  result_lets
    (run_call return_partial (Some shape) return_arity return_to exn_to
       statics hp rd)
    (run_call return_partial' (Some shape') return_arity return_to' exn_to'
       statics' hp' rd').
Proof.
  intros Hshape Hreturn Hexn Hstatics Hheap Hpartial.
  destruct Hshape
    as [ closure closure' c c' vs vs' args_arity Hclosure Hc Hvs
       | closure closure' c c' vs vs' remaining remaining' args_arity
           remaining_arity Hclosure Hc Hvs Hremaining
       | partial partial' Hp ];
    cbn [run_call].
  - apply call_code_copied; assumption.
  - rewrite (proj1 (proj2 (proj2 (code_lets_parts c c' Hc)))).
    apply call_code_copied; try assumption.
    apply Over_application_copied; assumption.
  - exact (Hpartial partial partial' Hp).
Qed.

(* A call, given what the two runs do when the return continuation is
   applied to a partial application. *)
Lemma call_function_copied_from (callee callee' : value) (kind : call_kind)
    (vs vs' : list value) (args_arity return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics statics' : static_env) (hp hp' : heap) (rd rd' : read_record) :
  value_copied callee callee' ->
  Forall2 value_copied vs vs' ->
  cont_copied return_to return_to' ->
  cont_copied exn_to exn_to' ->
  statics_copied statics statics' ->
  follows statics ->
  heap_alike value_copied hp hp' ->
  kind_ok callee kind ->
  (∀ partial partial' : value,
    value_copied partial partial' ->
    result_lets (apply_cont_value return_to [partial] statics hp rd)
      (apply_cont_value return_to' [partial'] statics' hp' rd')) ->
  result_lets
    (call_function callee kind vs args_arity return_arity return_to exn_to
       statics hp rd)
    (call_function callee' kind vs' args_arity return_arity return_to'
       exn_to' statics' hp' rd').
Proof.
  intros Hcallee Hvs Hreturn Hexn Hstatics Hfollows Hheap Hkind Hpartial.
  unfold call_function.
  destruct (shape_of_call statics callee kind vs args_arity return_arity)
    as [shape | ] eqn:Hshape;
    [ | exact I ].
  destruct (shape_of_call_copied statics statics' callee callee' kind vs vs'
              args_arity return_arity shape Hstatics Hfollows Hcallee Hvs
              Hkind Hshape)
    as (shape' & -> & Hshapes).
  exact (run_call_copied _ _ shape shape' return_arity _ _ _ _ statics
           statics' _ _ _ _ Hshapes Hreturn Hexn Hstatics Hheap Hpartial).
Qed.

(* Applying a continuation. What is left of an over-application calls
   the value it is given, indirectly. *)
Lemma apply_cont_value_copied (c c' : cont_value) :
  cont_copied c c' ->
  ∀ (vs vs' : list value) (statics statics' : static_env) (hp hp' : heap)
    (rd rd' : read_record),
    Forall2 value_copied vs vs' ->
    statics_copied statics statics' ->
    follows statics ->
    heap_alike value_copied hp hp' ->
    result_lets (apply_cont_value c vs statics hp rd)
      (apply_cont_value c' vs' statics' hp' rd').
Proof.
  induction 1
    as [ k
       | h h' vars vars' conts conts' recursive recursive' Hh Hvars Hconts
           Hrecursive
       | c c' extra extra' Hc IH Hextra
       | remaining remaining' remaining_arity return_arity return_to
           return_to' exn_to exn_to' Hremaining Hreturn IHreturn Hexn _ ];
    intros vs vs' statics statics' hp hp' rd rd' Hvs Hstatics Hfollows Hheap.
  - simpl. exists vs', hp', rd'.
    split; [ reflexivity | split; [ exact Hvs | exact Hheap ] ].
  - destruct Hh as (lets & at_path & ->).
    destruct h as [params metadata handler]. simpl.
    rewrite (kinds_of_values_copied vs vs' Hvs).
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
      [ | exact I ].
    destruct (bind_params params vs vars) as [bound | ] eqn:Hbind;
      [ | exact I ].
    destruct (bind_params_copied params vs vs' vars vars' bound Hvs Hvars
                Hbind)
      as (bound' & -> & Hbound).
    eexists. split; [ reflexivity | ].
    apply State_copied;
      [ exists lets, at_path; reflexivity | exact Hbound
      | apply rec_closures_copied; assumption | exact Hstatics | exact Hheap ].
  - simpl.
    apply IH;
      [ apply Forall2_app; assumption | exact Hstatics | exact Hfollows
      | exact Hheap ].
  - rewrite !apply_cont_value_over_application.
    destruct Hvs as [ | v v' vs vs' Hv Hvs]; [ exact I | ].
    destruct Hvs as [ | w w' vs vs' Hw Hvs]; [ | exact I ].
    apply call_function_copied_from; try assumption.
    + exact I.
    + intros partial partial' Hpartial.
      apply IHreturn;
        [ constructor; [ exact Hpartial | constructor ] | exact Hstatics
        | exact Hfollows | exact Hheap ].
Qed.

(* A call. *)
Lemma call_function_copied (callee callee' : value) (kind : call_kind)
    (vs vs' : list value) (args_arity return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics statics' : static_env) (hp hp' : heap) (rd rd' : read_record) :
  value_copied callee callee' ->
  Forall2 value_copied vs vs' ->
  cont_copied return_to return_to' ->
  cont_copied exn_to exn_to' ->
  statics_copied statics statics' ->
  follows statics ->
  heap_alike value_copied hp hp' ->
  kind_ok callee kind ->
  result_lets
    (call_function callee kind vs args_arity return_arity return_to exn_to
       statics hp rd)
    (call_function callee' kind vs' args_arity return_arity return_to'
       exn_to' statics' hp' rd').
Proof.
  intros Hcallee Hvs Hreturn Hexn Hstatics Hfollows Hheap Hkind.
  apply call_function_copied_from; try assumption.
  intros partial partial' Hpartial.
  apply (apply_cont_value_copied return_to return_to' Hreturn);
    [ constructor; [ exact Hpartial | constructor ] | exact Hstatics
    | exact Hfollows | exact Hheap ].
Qed.

(* ======================================================================== *)
(* A step                                                                   *)
(* ======================================================================== *)

(* What is asked of the expression of the second state: if it is a
   direct call, the callee is a symbol or the code named has no
   copy. *)
Definition apply_ok (e : expr) : Prop :=
  match e with
  | Apply ap =>
    match ap_call_kind ap with
    | Direct named =>
      (∃ s : symbol, ap_callee ap = Symbol s) ∨ lookup named ids = None
    | Indirect_unknown_arity => True
    end
  | Let _ _ _ _ | Let_static _ _ | Let_cont_non_recursive _ _ _
  | Let_cont_recursive _ _ | Apply_cont _ | Switch _ _ | Invalid =>
    True
  end.

(* A step of the second state is answered by one of the third. *)
Lemma lets_step (s s' : state) :
  state_copied s s' ->
  follows (state_statics s) ->
  apply_ok (state_expr s) ->
  result_lets (step s) (step s').
Proof.
  intros [e e' vars vars' conts conts' statics statics' hp hp' rd rd' He
            Hvars Hconts Hstatics Hheap] Hfollows Hok.
  simpl_state. destruct He as (lets & at_path & ->).
  destruct e
    as [ x xkind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    cbn [copy_lets_expr copy_lets_in_code].
  - (* Let *)
    rewrite !step_let_exec.
    destruct (exec_named statics hp rd vars d) as [[[v hp1] rd1] | ] eqn:Hd;
      [ | exact I ].
    destruct (kind_eqb (kind_of_value v) xkind) eqn:Hkind; [ | exact I ].
    set (d' := if among at_path lets then copy_named ids d else d).
    assert (Hd' : d' = d ∨ d' = copy_named ids d)
      by (unfold d'; destruct (among at_path lets);
          [ right | left ]; reflexivity).
    destruct (exec_named_copied statics statics' hp hp' hp1 rd rd1 rd' vars
                vars' d d' v (proj1 (proj2 Hstatics)) Hvars Hheap Hd' Hd)
      as (v' & hp1' & -> & Hv & Hheap1).
    rewrite (kind_of_value_copied v v' Hv), Hkind.
    cbn [result_lets]. eexists. split; [ reflexivity | ].
    apply State_copied;
      [ exists lets, (xO at_path); reflexivity
      | apply env_copied_cons; assumption | exact Hconts | exact Hstatics
      | exact Hheap1 ].
  - (* Let_static *)
    rewrite !step_let_static.
    destruct (eval_static_bindings vars bindings statics) as [extended | ]
      eqn:Hbindings;
      [ | exact I ].
    destruct (eval_static_bindings_copied vars vars' bindings statics statics'
                extended Hvars Hstatics Hbindings)
      as (extended' & -> & Hextended).
    cbn [result_lets]. eexists. split; [ reflexivity | ].
    apply State_copied;
      [ exists lets, (xO at_path); reflexivity | exact Hvars | exact Hconts
      | exact Hextended | exact Hheap ].
  - (* Let_cont_non_recursive *)
    rewrite !step_let_cont_non_recursive.
    cbn [result_lets]. eexists. split; [ reflexivity | ].
    apply State_copied;
      [ exists lets, (xO at_path); reflexivity | exact Hvars |
      | exact Hstatics | exact Hheap ].
    constructor; [ | exact Hconts ].
    split; [ reflexivity | ]. cbn [snd].
    apply Closure_copied;
      [ exists lets, (xI at_path); reflexivity | exact Hvars | exact Hconts
      | constructor ].
  - (* Let_cont_recursive *)
    rewrite !step_let_cont_recursive, rec_handlers_bind_statics.
    destruct (handlers_bind_statics handlers) eqn:Hbinds; [ exact I | ].
    cbn [result_lets]. eexists. split; [ reflexivity | ].
    apply State_copied;
      [ exists lets, (xO at_path); reflexivity | exact Hvars |
      | exact Hstatics | exact Hheap ].
    apply rec_closures_copied;
      [ exact Hvars | exact Hconts | apply rec_handlers_lets ].
  - (* Apply *)
    rewrite !step_apply.
    destruct (eval_simple vars (ap_callee ap)) as [callee | ] eqn:Hcallee;
      [ | exact I ].
    destruct (eval_simples vars (ap_args ap)) as [vs | ] eqn:Hargs;
      [ | exact I ].
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ]
      eqn:Hextra;
      [ | exact I ].
    destruct (eval_simple_copied vars vars' _ callee Hvars Hcallee)
      as (callee' & -> & Hcopied).
    destruct (eval_simples_copied vars vars' _ Hvars vs Hargs)
      as (vs' & -> & Hvs).
    destruct (eval_simples_copied vars vars' _ Hvars extra Hextra)
      as (extra' & -> & Hextras).
    apply call_function_copied; try assumption.
    + exact (resolve_copied _ conts conts' Hconts).
    + exact (with_extra_args_copied _ _ extra extra'
               (resolve_copied _ conts conts' Hconts) Hextras).
    + cbn [apply_ok] in Hok. unfold kind_ok.
      destruct (ap_call_kind ap) as [named | ]; [ | exact I ].
      destruct Hok as [[sym Hsym] | Hnone]; [ | right; exact Hnone ].
      left. exists sym. rewrite Hsym in Hcallee. simpl in Hcallee.
      injection Hcallee as <-. reflexivity.
  - (* Apply_cont *)
    rewrite !step_apply_cont.
    destruct (eval_simples vars (ac_args ac)) as [vs | ] eqn:Hargs;
      [ | exact I ].
    destruct (eval_simples_copied vars vars' _ Hvars vs Hargs)
      as (vs' & -> & Hvs).
    unfold apply_continuation.
    apply (apply_cont_value_copied _ _ (resolve_copied _ conts conts' Hconts));
      assumption.
  - (* Switch *)
    rewrite !step_switch.
    destruct (eval_simple vars scrutinee) as [v | ] eqn:Hscrutinee;
      [ | exact I ].
    destruct (eval_simple_copied vars vars' _ v Hvars Hscrutinee)
      as (v' & -> & Hv).
    destruct (naked_immediate_of_value v) as [i | ] eqn:Hi; [ | exact I ].
    rewrite (naked_immediate_copied v v' i Hv Hi).
    destruct (find_arm i arms) as [ac | ] eqn:Harm; [ | exact I ].
    cbn [result_lets]. eexists. split; [ reflexivity | ].
    apply State_copied;
      [ exists lets, at_path; reflexivity | exact Hvars | exact Hconts
      | exact Hstatics | exact Hheap ].
  - (* Invalid *)
    rewrite !step_invalid. exact I.
Qed.

End Lets.

(* ======================================================================== *)
(* The run of the unit with one copy                                        *)
(* ======================================================================== *)

Section Unit.

Variable ids : list (code_id * code_id).
Variable symbols : list symbol.
Variable in_code : list (code_id * list path).

Hypothesis Hnew : distinct (map snd ids) = true.
Hypothesis Hapart : copies_apart ids = true.

(* In the unit with its static copies, the copy of a code id is bound
   to what the id is bound to: the id is none of the new ones, so the
   first run binds it, and the copy is bound beside it. *)
Lemma follows_copy_statics (statics : static_env) :
  no_new ids (static_code statics) ->
  follows ids (copy_statics ids symbols statics).
Proof.
  intros Hno id id' c Hcopy. unfold copy_statics. cbn [static_code].
  assert (Hold : among id (map snd ids) = false).
  { unfold copies_apart in Hapart. rewrite forallb_forall in Hapart.
    apply negb_true_iff. apply Hapart.
    exact (in_map fst ids (id, id') (lookup_in id id' ids Hcopy)). }
  rewrite (lookup_copy_code ids symbols id (static_code statics) Hno Hold),
    (lookup_copy_code_new ids symbols Hnew id id' (static_code statics) Hno
       Hcopy).
  exact (fun H => H).
Qed.

(* A direct call of the unit with its static copies is of a symbol, or
   names code that has no copy. *)
Lemma apply_ok_copy_expr (e : expr) :
  apply_ok ids (copy_expr ids symbols e).
Proof.
  destruct e
    as [ x xkind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    try exact I.
  destruct ap as [callee kind args k exn extra args_arity return_arity].
  cbn [copy_expr]. unfold copy_call. cbn [ap_call_kind ap_callee].
  destruct kind as [named | ]; [ | exact I ].
  destruct callee as [x | s | c].
  - destruct (lookup named ids) eqn:Hcopy; [ exact I | ].
    right. exact Hcopy.
  - destruct (among s symbols); left; exists s; reflexivity.
  - destruct (lookup named ids) eqn:Hcopy; [ exact I | ].
    right. exact Hcopy.
Qed.

(* What two exits are compared by: the same values, and heaps alike. *)
Definition exit_same (last : state) (vs : list value) (hp : heap)
    (vs' : list value) (hp' : heap) : Prop :=
  vs' = vs ∧ heap_alike (value_copied ids) hp hp'.

(* The first state, of the unit, and the third: the first binds no new
   id, exits with constants and symbols only, and the second state,
   which it determines, is related to the third. *)
Definition unit_related (s s' : state) : Prop :=
  copied_inv ids s
  ∧ plain_state s
  ∧ state_copied ids in_code (copy_state ids symbols s) s'.

Lemma unit_related_step (s s' : state) :
  unit_related s s' -> answered_heaps unit_related exit_same (step s) s'.
Proof.
  intros (Hinv & Hplain & Hrelated).
  pose proof (copy_step ids symbols Hnew s Hinv) as Hcopy.
  pose proof (plain_step s Hplain) as Hplain'.
  pose proof
    (lets_step ids in_code (copy_state ids symbols s) s' Hrelated
       (follows_copy_statics (state_statics s) (proj2 Hinv))
       (apply_ok_copy_expr (state_expr s)))
    as Hlets.
  destruct (step s) as [t | k vs hp rd | ]; unfold answered_heaps;
    [ | | exact I ].
  - destruct Hcopy as [Hstep Hinv']. rewrite Hstep in Hlets.
    destruct Hlets as (t' & Ht' & Hrelated').
    exists 0, t'. split; [ simpl; rewrite Ht'; reflexivity | ].
    split; [ exact Hinv' | split; [ exact Hplain' | exact Hrelated' ] ].
  - rewrite Hcopy in Hlets.
    destruct Hlets as (vs' & hp' & rd' & Hstep' & Hvs & Hheap).
    exists vs', hp', rd'. split; [ exact Hstep' | ].
    split; [ exact (plain_values_copied ids vs vs' Hplain' Hvs) | exact Hheap ].
Qed.

(* The unit with one copy refines the unit, the two run from a static
   environment that holds no code and that the copy of static code
   leaves as it is, and from the empty heap. *)
Lemma copy_lets_refines_from (lets : list path) (e : expr)
    (statics : static_env) :
  binds_no_code (map snd ids) e = true ->
  exits_plain_knowing [] [] e = true ->
  expr_free_of_heap e = true ->
  static_code statics = [] ->
  copy_statics ids symbols statics = statics ->
  refines (State e [] [] statics empty_heap no_reads)
    (State
       (copy_lets_in_code ids in_code
          (copy_lets_expr ids lets root (copy_expr ids symbols e)))
       [] [] statics empty_heap no_reads).
Proof.
  intros Hbinds Hplain Hheap Hno_code Hsame.
  apply (stuttering_heaps_refines unit_related exit_same unit_related_step).
  - intros k vs hp last vs' hp' Hexits [Hvs Halike]. split; [ exact Hvs | ].
    assert (Hempty : hp = empty_heap).
    { apply (free_of_heap_exits_empty e statics k vs hp Hheap);
        [ | exact Hexits ].
      unfold statics_free_of_heap. rewrite Hno_code. reflexivity. }
    subst hp. exact (heap_alike_empty _ hp' Halike).
  - split; [ | split ].
    + split.
      * unfold terms_inv. simpl_state. refine (conj Hbinds (conj I _)).
        intros id c Hlookup. rewrite Hno_code in Hlookup.
        discriminate Hlookup.
      * simpl_state. intros id _. rewrite Hno_code. reflexivity.
    + exact (plain_from e statics empty_heap no_reads Hplain Hno_code).
    + unfold copy_state. simpl_state. rewrite Hsame.
      apply State_copied;
        [ exists lets, root; reflexivity | apply env_copied_nil
        | constructor | apply statics_copied_refl | constructor ].
Qed.

End Unit.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Lemma copy_let_code_refines_from (copy : let_copy) (e : expr)
    (statics : static_env) :
  static_code statics = [] ->
  (∀ (ids : list (code_id * code_id)) (symbols : list symbol),
    copy_statics ids symbols statics = statics) ->
  refines (State e [] [] statics empty_heap no_reads)
    (State (copy_let_code copy e) [] [] statics empty_heap no_reads).
Proof.
  intros Hno_code Hsame. destruct copy as [[ids symbols] [lets in_code]].
  unfold copy_let_code.
  destruct (exits_plain_knowing [] [] e) eqn:Hplain; [ | apply refines_refl ].
  destruct (expr_free_of_heap e) eqn:Hheap; [ | apply refines_refl ].
  destruct (copies_apart ids) eqn:Hapart; [ | apply refines_refl ].
  cbn [andb]. unfold copy_static_code, copy_ok.
  destruct (distinct (map fst ids) && distinct (map snd ids)
            && binds_no_code (map snd ids) e) eqn:Hok;
    [ | apply refines_refl ].
  apply andb_true_iff in Hok as [Hok Hbinds].
  apply andb_true_iff in Hok as [_ Hnew].
  exact (copy_lets_refines_from ids symbols in_code Hnew Hapart lets e
           statics Hbinds Hplain Hheap Hno_code (Hsame ids symbols)).
Qed.

Theorem copy_let_code_refines_imports :
  copy_let_code_refines_imports_statement.
Proof.
  intros imports copy e. unfold unit_refines.
  apply copy_let_code_refines_from; [ reflexivity | ].
  intros ids symbols. apply copy_statics_imports.
Qed.

Theorem copy_let_code_program_refines :
  copy_let_code_program_refines_statement.
Proof. intros copy e. exact (copy_let_code_refines_imports [] copy e). Qed.

Theorem with_let_copies_none : with_let_copies_none_statement.
Proof. intros e. reflexivity. Qed.

Theorem with_let_copies_refines_imports :
  with_let_copies_refines_imports_statement.
Proof.
  intros imports copies. unfold unit_refines.
  induction copies as [ | copy copies IH]; intros e; cbn [with_let_copies];
    [ apply refines_refl | ].
  exact (refines_trans _ _ _ (copy_let_code_refines_imports imports copy e)
           (IH (copy_let_code copy e))).
Qed.

Theorem with_let_copies_program_refines :
  with_let_copies_program_refines_statement.
Proof.
  intros copies e. exact (with_let_copies_refines_imports [] copies e).
Qed.

(* ======================================================================== *)
(* The rewrite under renaming                                               *)
(* ======================================================================== *)

(* The two laws are proved from the definitions by the tactic of
   FlambdaEquivarianceTactic.v, which states and proves by itself the
   laws of the functions that the rewrite is made of. It is given
   three things.

   The handlers of a recursive [Let_cont] are rewritten by
   [FlambdaLiftSpec.rec_handlers_map], which takes a function of the
   path of a handler: its law is proved here by hand, with a step
   that uses it.

   The first test of the rewrite, [FlambdaLiftSpec.exits_plain_knowing]
   at nothing known, has its law stated at nothing known; that law,
   and those of the two other tests, are in other files and are given
   as they are. So is the law of the copy of static code, from right
   to left. *)
Lemma rec_handlers_map_renamed (r : renaming)
    (f : path -> continuation_handler -> continuation_handler)
    (handlers : list (continuation * continuation_handler)) :
  (∀ x : continuation * continuation_handler,
     In x handlers ->
     ∀ q : path,
       f q (rename_handler r (snd x)) = rename_handler r (f q (snd x))) ->
  ∀ q : path,
    rec_handlers_map f q
      (map (fun '(k, h) => (rename_continuation r k, rename_handler r h))
         handlers)
    = map (fun '(k, h) => (rename_continuation r k, rename_handler r h))
        (rec_handlers_map f q handlers).
Proof.
  induction handlers as [ | [k h] handlers IH]; intros Hf q;
    [ reflexivity | ].
  cbn [map rec_handlers_map].
  rewrite (Hf (k, h) (or_introl eq_refl)), IH; [ reflexivity | ].
  intros x Hx. apply Hf. right. exact Hx.
Qed.

#[local] Ltac eqv_extra ::=
  lazymatch goal with
  | |- context [rec_handlers_map ?f ?q (map _ ?l)] =>
    rewrite (rec_handlers_map_renamed _ f l);
    [ | eqv_enter l; intros ?; solve [ eqv_core ] ]
  end.

Lemma exits_plain_renamed (r : renaming) (e : expr) :
  exits_plain_knowing [] [] (rename_expr r e) = exits_plain_knowing [] [] e.
Proof. exact (exits_plain_knowing_equivariant r [] [] e). Qed.

#[local] Hint Rewrite exits_plain_renamed expr_free_of_heap_rename
  binds_no_code_equivariant : equivariance.
#[local] Hint Rewrite <- copy_expr_equivariant : equivariance.

Theorem copy_let_code_equivariant : copy_let_code_equivariant_statement.
Proof. equivariance. Qed.

#[local] Hint Rewrite <- copy_let_code_equivariant : equivariance.

Theorem with_let_copies_equivariant : with_let_copies_equivariant_statement.
Proof. equivariance. Qed.

(* ======================================================================== *)
(* One [Let] copied, and one unit left as it is                             *)
(* ======================================================================== *)

Theorem copy_let_code_one_let : copy_let_code_one_let_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem copy_let_code_exit_with_closure :
  copy_let_code_exit_with_closure_statement.
Proof. vm_compute. reflexivity. Qed.
