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

(* The erasure of regions (FlambdaRegionsErasureSpec.v), read from the
   state that runs the erased term to the state of the machine of
   regions. Nothing here needs review: no statement of a Spec file is
   proved here.

   FlambdaRegionsErasedProofs.v has that what the machine of regions
   evaluates or observes, the second state does, with the answers
   related. Here is the other way, for the values that reach only
   symbols that may be evaluated (FlambdaRegionsCheckedSpec.value_all
   of FlambdaRegionsInvariantSpec.live_symbol): where the second state
   has a value for a simple, for a defining expression or for an
   observation, the machine of regions has one. It is what a proof
   needs that the erased term goes wrong where the machine of regions
   has no step.

   It does not hold of every value. A partial application is erased
   with its parts related only where they may be evaluated or are
   locations (FlambdaRegionsErasureSpec.value_erased), so of one whose
   callee is a symbol that is gone the second state may have a closure
   to call where the first has none. A value that reaches only
   symbols that may be evaluated has no such part.

   - [erased_back]: the relation read the other way, of such a value.
   - The observations ([block_of_back] to [project_function_slot_back])
     and [erased_back_respects]: the relation is one that the
     observations cannot tell from equality, so a primitive that has a
     value on the second arguments has one on the first
     ([eval_prim_back]).
   - Simples and defining expressions ([eval_simple_back] to
     [eval_named_back]): the second environment binds no variable that
     the first does not (FlambdaRegionsErasedProofs.env_erased).
   - A group of static bindings ([eval_static_bindings_back]). *)

From Stdlib Require Import ZArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaValueSpec FlambdaPrimitiveModel FlambdaPrimitiveSpec
  FlambdaPrimitiveProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaHeapProofs FlambdaClosuresSpec FlambdaClosuresProofs
  FlambdaLiftProofs FlambdaRegionsSpec FlambdaRegionsProofs
  FlambdaRegionsInvariantSpec FlambdaRegionsErasureSpec
  FlambdaRegionsReadsProofs FlambdaRegionsErasedProofs
  FlambdaRegionsPlainStepProofs FlambdaRegionsErasureProofs
  FlambdaRegionsCheckedSpec.
Import ListNotations.

(* The static environment that [step] is given is not unfolded: the
   rules of FlambdaRegionsErasureSpec.value_erased name it. *)
#[local] Opaque live_statics.

(* The relation is asked of a value all of whose symbols may be
   evaluated (FlambdaRegionsErasedProofs.erased_asked): it may be
   evaluated, or it is a location, whose cell may be dead. *)
Lemma value_all_live_asked (rs : region_state) (v : value) :
  value_all (live_symbol rs) v = true -> erased_asked rs v.
Proof.
  destruct v as [ | | | s | | | | l | ];
    try (intros _; left; reflexivity).
  - cbn [value_all]. unfold live_symbol.
    exact (usable_asked rs (Value_symbol s)).
  - intros _. exact (location_asked rs l).
Qed.

(* The simples whose values reach only symbols that may be evaluated. *)
Definition simples_all_live (rs : region_state) (vars : env)
    (simples : list simple) : Prop :=
  ∀ (s : simple) (v : value),
    s ∈ simples ->
    eval_simple vars s = Some v ->
    value_all (live_symbol rs) v = true.

Lemma simples_all_live_values (rs : region_state) (vars : env)
    (args : list simple) (vs : list value) :
  simples_all_live rs vars args ->
  eval_simples vars args = Some vs ->
  forallb (value_all (live_symbol rs)) vs = true.
Proof.
  intros Hall Hvs. apply forallb_forall. apply Forall_forall.
  exact
    (eval_simples_forall (fun v => value_all (live_symbol rs) v = true) vars
       args vs Hall Hvs).
Qed.

(* ======================================================================== *)
(* The observations, from the second state to the first                     *)
(* ======================================================================== *)

Section Back.

Variable rs : region_state.
Variable statics' : static_env.
Hypothesis Hfresh : symbols_fresh rs = true.
Hypothesis Hstatics : statics_erased rs statics'.
Hypothesis Hlive :
  forallb (fun '(_, allocated) => allocated_all (live_symbol rs) allocated)
    (static_values (live_statics rs))
  = true.

(* The relation read the other way, of a value of the first state that
   reaches only symbols that may be evaluated. *)
Definition erased_back (v' v : value) : Prop :=
  value_erased rs v v' ∧ value_all (live_symbol rs) v = true.

(* What a symbol is bound to reaches only symbols that may be
   evaluated. *)
Lemma bound_live (s : symbol) (allocated : statically_allocated_value) :
  lookup s (static_values (live_statics rs)) = Some allocated ->
  allocated_all (live_symbol rs) allocated = true.
Proof.
  intros Hs.
  exact (forallb_in _ _ (s, allocated) Hlive (lookup_in _ _ _ Hs)).
Qed.

(* A symbol that the machine did not draw and that the second static
   environment binds is bound in the first state, to the same thing
   with its parts related. *)
Lemma undrawn_bound (s : symbol) (allocated' : statically_allocated_value) :
  drawn rs s = false ->
  lookup s (static_values statics') = Some allocated' ->
  ∃ allocated : statically_allocated_value,
    lookup s (static_values (live_statics rs)) = Some allocated
    ∧ allocated_erased rs allocated allocated'.
Proof.
  intros Hnot_drawn Hs'.
  pose proof Hstatics as (_ & _ & Hsymbols & Hbound).
  destruct (lookup s (static_values (state_statics (rs_state rs))))
    as [allocated | ] eqn:Hs;
    [ | contradiction (Hbound s allocated' Hs' Hs) ].
  assert (Hheap : heap_symbol rs s = true)
    by (unfold heap_symbol; rewrite Hnot_drawn; reflexivity).
  rewrite <- (heap_lookup rs s Hfresh Hheap) in Hs.
  destruct (Hsymbols s allocated Hnot_drawn Hs) as (allocated0 & Hs0 & Ha).
  rewrite Hs' in Hs0. injection Hs0 as <-.
  exists allocated. split; [ exact Hs | exact Ha ].
Qed.

Lemma values_back (vs vs' : list value) :
  Forall2 (value_erased rs) vs vs' ->
  forallb (value_all (live_symbol rs)) vs = true ->
  Forall2 erased_back vs' vs.
Proof.
  induction 1 as [ | v v' vs vs' Hv _ IH]; simpl; intros Hall;
    [ constructor | ].
  apply andb_true_iff in Hall as [Hhead Htail].
  constructor; [ exact (conj Hv Hhead) | exact (IH Htail) ].
Qed.

Lemma back_values (vs' vs : list value) :
  Forall2 erased_back vs' vs ->
  Forall2 (value_erased rs) vs vs'
  ∧ forallb (value_all (live_symbol rs)) vs = true.
Proof.
  induction 1 as [ | v' v vs' vs [Hv Hall] _ [IH IHall]]; simpl;
    [ split; [ constructor | reflexivity ] | ].
  rewrite Hall, IHall. split; [ constructor; assumption | reflexivity ].
Qed.

Lemma block_of_back (v' v : value) (block_tag : Z)
    (block_fields' : list value) :
  erased_back v' v ->
  block_of statics' v' = Some (block_tag, block_fields') ->
  ∃ block_fields : list value,
    block_of (live_statics rs) v = Some (block_tag, block_fields)
    ∧ Forall2 erased_back block_fields' block_fields.
Proof.
  intros [Hv Hall]. erased_cases Hv; simpl in Hall |- *; try discriminate.
  - destruct (lookup s (static_values statics')) as [allocated' | ] eqn:Hs';
      [ | discriminate ].
    destruct (undrawn_bound s allocated' Hnot_drawn Hs')
      as (allocated & Hs & Hallocated).
    rewrite Hs. pose proof (bound_live s allocated Hs) as Hbound_live.
    allocated_erased_cases Hallocated; try discriminate.
    intros [= <- <-]. exists afields. split; [ reflexivity | ].
    exact (values_back _ _ Hafields Hbound_live).
  - intros [= <- <-]. exists fields. split; [ reflexivity | ].
    exact (values_back _ _ Hfields Hall).
  - intros [= <- <-]. rewrite Hlookup. exists fields.
    split; [ reflexivity | ].
    exact (values_back _ _ Hfields (bound_live s _ Hlookup)).
Qed.

Lemma boxed_number_of_back (v' v : value) (number : const) :
  value_erased rs v v' ->
  boxed_number_of statics' v' = Some number ->
  boxed_number_of (live_statics rs) v = Some number.
Proof.
  intros Hv. erased_cases Hv; simpl; try discriminate;
    try exact (fun H => H).
  - destruct (lookup s (static_values statics')) as [allocated' | ] eqn:Hs';
      [ | discriminate ].
    destruct (undrawn_bound s allocated' Hnot_drawn Hs')
      as (allocated & Hs & Hallocated).
    rewrite Hs.
    allocated_erased_cases Hallocated; try discriminate.
    destruct av'; try discriminate. inversion Hav. exact (fun H => H).
  - rewrite Hlookup. exact (fun H => H).
Qed.

Lemma string_of_back (v' v : value) (bytes : list Z) :
  value_erased rs v v' ->
  string_of statics' v' = Some bytes ->
  string_of (live_statics rs) v = Some bytes.
Proof.
  intros Hv. erased_cases Hv; simpl; try discriminate.
  destruct (lookup s (static_values statics')) as [allocated' | ] eqn:Hs';
    [ | discriminate ].
  destruct (undrawn_bound s allocated' Hnot_drawn Hs')
    as (allocated & Hs & Hallocated).
  rewrite Hs.
  allocated_erased_cases Hallocated; try discriminate. exact (fun H => H).
Qed.

Lemma array_of_back (v' v : value) (elements' : list value) :
  erased_back v' v ->
  array_of statics' v' = Some elements' ->
  ∃ elements : list value,
    array_of (live_statics rs) v = Some elements
    ∧ Forall2 erased_back elements' elements.
Proof.
  intros [Hv Hall]. erased_cases Hv; simpl in Hall |- *; try discriminate.
  - destruct (lookup s (static_values statics')) as [allocated' | ] eqn:Hs';
      [ | discriminate ].
    destruct (undrawn_bound s allocated' Hnot_drawn Hs')
      as (allocated & Hs & Hallocated).
    rewrite Hs. pose proof (bound_live s allocated Hs) as Hbound_live.
    allocated_erased_cases Hallocated; try discriminate.
    intros [= <-]. exists aelements. split; [ reflexivity | ].
    exact (values_back _ _ Haelements Hbound_live).
  - intros [= <-]. exists items. split; [ reflexivity | ].
    exact (values_back _ _ Hitems Hall).
  - intros [= <-]. rewrite Hlookup. exists items. split; [ reflexivity | ].
    exact (values_back _ _ Hitems (bound_live s _ Hlookup)).
Qed.

Lemma closure_in_set_back (v' v : value) (at_slot : function_slot)
    (function_decls : list (function_slot * code_id))
    (value_slots' : list (value_slot * value)) :
  value_erased rs v v' ->
  closure_in_set statics' v' = Some (at_slot, function_decls, value_slots') ->
  ∃ value_slots : list (value_slot * value),
    closure_in_set (live_statics rs) v
    = Some (at_slot, function_decls, value_slots)
    ∧ Forall2 (slot_erased (value_erased rs)) value_slots value_slots'.
Proof.
  intros Hv. erased_cases Hv; simpl; try discriminate.
  - destruct (lookup s (static_values statics')) as [allocated' | ] eqn:Hs';
      [ | discriminate ].
    destruct (undrawn_bound s allocated' Hnot_drawn Hs')
      as (allocated & Hs & Hallocated).
    rewrite Hs.
    allocated_erased_cases Hallocated; try discriminate.
    + intros [= <- <- <-]. eauto.
    + destruct (lookup afslot adecls); [ | discriminate ]. simpl.
      intros [= <- <- <-]. eauto.
  - intros [= <- <- <-]. eauto.
  - destruct (lookup fslot decls); [ | discriminate ]. simpl.
    intros [= <- <- <-]. eauto.
  - rewrite Hlookup. intros [= <- <- <-]. eauto.
  - rewrite (Hbound fslot s (lookup_in _ _ _ Hsym)).
    destruct (lookup fslot decls); [ | discriminate ]. simpl.
    intros [= <- <- <-]. eauto.
Qed.

Lemma closure_of_back (v' v : value) (called : code_id)
    (value_slots' : list (value_slot * value)) :
  value_erased rs v v' ->
  closure_of statics' v' = Some (called, value_slots') ->
  ∃ value_slots : list (value_slot * value),
    closure_of (live_statics rs) v = Some (called, value_slots)
    ∧ Forall2 (slot_erased (value_erased rs)) value_slots value_slots'.
Proof.
  intros Hv. erased_cases Hv; simpl; try discriminate.
  - destruct (lookup s (static_values statics')) as [allocated' | ] eqn:Hs';
      [ | discriminate ].
    destruct (undrawn_bound s allocated' Hnot_drawn Hs')
      as (allocated & Hs & Hallocated).
    rewrite Hs.
    allocated_erased_cases Hallocated; try discriminate.
    + intros [= <- <-]. eauto.
    + destruct (lookup afslot adecls); [ | discriminate ]. simpl.
      intros [= <- <-]. eauto.
  - intros [= <- <-]. eauto.
  - destruct (lookup fslot decls); [ | discriminate ]. simpl.
    intros [= <- <-]. eauto.
  - rewrite Hlookup. intros [= <- <-]. eauto.
  - rewrite (Hbound fslot s (lookup_in _ _ _ Hsym)).
    destruct (lookup fslot decls); [ | discriminate ]. simpl.
    intros [= <- <-]. eauto.
Qed.

(* A slot that erased value slots have, the value slots have. *)
Lemma lookup_slots_back (slot : value_slot)
    (slots slots' : list (value_slot * value)) (v' : value) :
  Forall2 (slot_erased (value_erased rs)) slots slots' ->
  lookup slot slots' = Some v' ->
  ∃ v : value, lookup slot slots = Some v.
Proof.
  induction 1 as [ | [k w] [k' w'] slots slots' [Hkey Hw] _ IH]; simpl;
    [ discriminate | ].
  simpl in Hkey. subst k'.
  destruct (Pos.eqb slot k); [ intros _; eauto | exact IH ].
Qed.

Lemma project_value_slot_back (project_from : function_slot)
    (slot : value_slot) (v' v w' : value) :
  value_erased rs v v' ->
  project_value_slot statics' project_from slot v' = Some w' ->
  ∃ w : value,
    project_value_slot (live_statics rs) project_from slot v = Some w.
Proof.
  intros Hv. unfold project_value_slot.
  destruct (closure_in_set statics' v') as [[[at_slot decls] slots'] | ]
    eqn:Hc;
    [ | discriminate ].
  destruct (closure_in_set_back v' v at_slot decls slots' Hv Hc)
    as (slots & -> & Hslots).
  destruct (Pos.eqb at_slot project_from); [ | discriminate ].
  exact (lookup_slots_back slot slots slots' w' Hslots).
Qed.

Lemma project_function_slot_back (move_from move_to : function_slot)
    (v' v w' : value) :
  value_erased rs v v' ->
  project_function_slot statics' move_from move_to v' = Some w' ->
  ∃ w : value,
    project_function_slot (live_statics rs) move_from move_to v = Some w.
Proof.
  intros Hv. unfold project_function_slot.
  erased_cases Hv; try discriminate.
  - (* A symbol that the machine did not draw: bound alike in both. *)
    destruct (lookup s (static_values statics')) as [allocated' | ] eqn:Hs';
      [ | discriminate ].
    destruct (undrawn_bound s allocated' Hnot_drawn Hs')
      as (allocated & Hs & Hallocated).
    rewrite Hs.
    allocated_erased_cases Hallocated; try discriminate.
    + destruct (Pos.eqb move_from unnamed_function_slot); [ | discriminate ].
      destruct (Pos.eqb move_to unnamed_function_slot); [ | discriminate ].
      intros _. eauto.
    + destruct (Pos.eqb afslot move_from); [ | discriminate ].
      destruct (lookup afslot adecls); [ | discriminate ].
      destruct (lookup move_to adecls); [ | discriminate ]. simpl.
      destruct (lookup move_to asyms) as [s' | ]; [ | discriminate ].
      simpl. intros _. eauto.
  - (* A closure of a set of one function. *)
    destruct (Pos.eqb move_from unnamed_function_slot); [ | discriminate ].
    destruct (Pos.eqb move_to unnamed_function_slot); [ | discriminate ].
    intros _. eauto.
  - (* A closure of a set, on the heap in both. *)
    destruct (Pos.eqb fslot move_from); [ | discriminate ].
    destruct (lookup fslot decls); [ | discriminate ].
    destruct (lookup move_to decls); [ | discriminate ]. simpl.
    intros _. eauto.
  - (* A local closure of a set of one function. *)
    rewrite Hlookup.
    destruct (Pos.eqb move_from unnamed_function_slot); [ | discriminate ].
    destruct (Pos.eqb move_to unnamed_function_slot); [ | discriminate ].
    intros _. eauto.
  - (* A local closure of a set: the set gives a symbol to the slot
       asked for, as it gives it code. *)
    rewrite (Hbound fslot s (lookup_in _ _ _ Hsym)).
    destruct (Pos.eqb fslot move_from); [ | discriminate ].
    destruct (lookup fslot decls); [ | discriminate ].
    destruct (lookup move_to decls) as [code' | ] eqn:Hto; [ | discriminate ].
    simpl.
    destruct (lookup_same_keys move_to decls syms (eq_sym Hsyms) code' Hto)
      as [s' Hs'].
    rewrite Hs'. simpl. intros _. eauto.
Qed.

(* The relation read the other way is one that the observations cannot
   tell from equality, so no primitive can. *)
Lemma erased_back_respects :
  respects_observations erased_back statics' (live_statics rs).
Proof.
  split; [ | split ].
  - split.
    + intros c v [Hv _]. inversion Hv. reflexivity.
    + intros v' c [Hv _]. inversion Hv. reflexivity.
  - split; [ | split ].
    + intros c. split; (split; [ constructor | reflexivity ]).
    + intros tag fields' fields Hfields.
      destruct (back_values fields' fields Hfields) as [Hrelated Hall].
      split; [ constructor; exact Hrelated | exact Hall ].
    + intros elements' elements Helements.
      destruct (back_values elements' elements Helements)
        as [Hrelated Hall].
      split; [ constructor; exact Hrelated | exact Hall ].
  - intros v' v Hback. pose proof Hback as [Hv Hall].
    split;
      [ | split; [ | split; [ | split; [ | split; [ | split; [ | split ] ] ] ] ]
      ].
    + intros tag fields'. exact (block_of_back v' v tag fields' Hback).
    + intros c. exact (boxed_number_of_back v' v c Hv).
    + intros bytes. exact (string_of_back v' v bytes Hv).
    + intros Hclosure.
      destruct (closure_of statics' v') as [[called cslots'] | ] eqn:Hc;
        [ | congruence ].
      destruct (closure_of_back v' v called cslots' Hv Hc)
        as (cslots & -> & _).
      discriminate.
    + clear Hback Hall.
      erased_cases Hv; simpl;
        try (intros Hnone; contradiction (Hnone eq_refl)).
      intros _. discriminate.
    + clear Hback Hall. erased_cases Hv; simpl; try (intros _; reflexivity).
      * destruct (lookup s (static_values statics')) as [allocated' | ]
          eqn:Hs';
          [ | discriminate ].
        destruct (undrawn_bound s allocated' Hnot_drawn Hs')
          as (allocated & -> & _).
        reflexivity.
      * rewrite Hlookup. reflexivity.
      * rewrite Hlookup. reflexivity.
      * rewrite Hlookup. reflexivity.
      * rewrite Hlookup. reflexivity.
      * rewrite (Hbound fslot s (lookup_in _ _ _ Hsym)). reflexivity.
    + intros [s0 c] Hs0.
      apply imported_of_symbol in Hs0 as [-> Hs0].
      inversion Hv as [ | | | s1 Hnot_drawn | | | | | | | | | | ]; subst.
      destruct (undrawn_bound s0 _ Hnot_drawn Hs0)
        as (allocated & Hs & Hallocated).
      inversion Hallocated; subst. simpl. rewrite Hs. reflexivity.
    + intros elements'. exact (array_of_back v' v elements' Hback).
Qed.

Lemma eval_prim_back (p : prim) (vs' vs : list value) (v' : value) :
  Forall2 erased_back vs' vs ->
  eval_prim statics' p vs' = Some v' ->
  ∃ v : value, eval_prim (live_statics rs) p vs = Some v ∧ erased_back v' v.
Proof.
  exact
    (eval_prim_related erased_back statics' (live_statics rs) p vs' vs v'
       erased_back_respects).
Qed.

(* The relation read the other way, with nothing asked of what the
   value reaches: what a cell holds is related so, and of it nothing
   says that it reaches only symbols that may be evaluated. It is one
   that the observations cannot tell from equality either: where
   [erased_back] asks the liveness, it is to give it of the parts
   again. So a primitive that the second state runs on its heap, the
   machine of regions runs on its own ([exec_prim_to]), by the law of
   a primitive on two heaps
   (FlambdaHeapSpec.exec_prim_related_given_statement) read from the
   second state to the first. *)
Definition erased_to (v' v : value) : Prop := value_erased rs v v'.

Lemma values_to (vs vs' : list value) :
  Forall2 (value_erased rs) vs vs' -> Forall2 erased_to vs' vs.
Proof.
  induction 1 as [ | v v' vs vs' Hv _ IH]; constructor; assumption.
Qed.

Lemma to_values (vs' vs : list value) :
  Forall2 erased_to vs' vs -> Forall2 (value_erased rs) vs vs'.
Proof.
  induction 1 as [ | v' v vs' vs Hv _ IH]; constructor; assumption.
Qed.

Lemma block_of_to (v' v : value) (block_tag : Z)
    (block_fields' : list value) :
  erased_to v' v ->
  block_of statics' v' = Some (block_tag, block_fields') ->
  ∃ block_fields : list value,
    block_of (live_statics rs) v = Some (block_tag, block_fields)
    ∧ Forall2 erased_to block_fields' block_fields.
Proof.
  intros Hv. unfold erased_to in Hv.
  erased_cases Hv; simpl; try discriminate.
  - destruct (lookup s (static_values statics')) as [allocated' | ] eqn:Hs';
      [ | discriminate ].
    destruct (undrawn_bound s allocated' Hnot_drawn Hs')
      as (allocated & Hs & Hallocated).
    rewrite Hs.
    allocated_erased_cases Hallocated; try discriminate.
    intros [= <- <-]. exists afields. split; [ reflexivity | ].
    exact (values_to _ _ Hafields).
  - intros [= <- <-]. exists fields. split; [ reflexivity | ].
    exact (values_to _ _ Hfields).
  - intros [= <- <-]. rewrite Hlookup. exists fields.
    split; [ reflexivity | ].
    exact (values_to _ _ Hfields).
Qed.

Lemma array_of_to (v' v : value) (elements' : list value) :
  erased_to v' v ->
  array_of statics' v' = Some elements' ->
  ∃ elements : list value,
    array_of (live_statics rs) v = Some elements
    ∧ Forall2 erased_to elements' elements.
Proof.
  intros Hv. unfold erased_to in Hv.
  erased_cases Hv; simpl; try discriminate.
  - destruct (lookup s (static_values statics')) as [allocated' | ] eqn:Hs';
      [ | discriminate ].
    destruct (undrawn_bound s allocated' Hnot_drawn Hs')
      as (allocated & Hs & Hallocated).
    rewrite Hs.
    allocated_erased_cases Hallocated; try discriminate.
    intros [= <-]. exists aelements. split; [ reflexivity | ].
    exact (values_to _ _ Haelements).
  - intros [= <-]. exists items. split; [ reflexivity | ].
    exact (values_to _ _ Hitems).
  - intros [= <-]. rewrite Hlookup. exists items. split; [ reflexivity | ].
    exact (values_to _ _ Hitems).
Qed.

Lemma erased_to_respects :
  respects_observations erased_to statics' (live_statics rs).
Proof.
  split; [ | split ].
  - split.
    + intros c v Hv. inversion Hv. reflexivity.
    + intros v' c Hv. inversion Hv. reflexivity.
  - split; [ | split ].
    + intros c. split; constructor.
    + intros tag fields' fields Hfields. constructor.
      exact (to_values fields' fields Hfields).
    + intros elements' elements Helements. constructor.
      exact (to_values elements' elements Helements).
  - intros v' v Hv.
    split;
      [ | split; [ | split; [ | split; [ | split; [ | split; [ | split ] ] ] ] ]
      ].
    + intros tag fields'. exact (block_of_to v' v tag fields' Hv).
    + intros c. exact (boxed_number_of_back v' v c Hv).
    + intros bytes. exact (string_of_back v' v bytes Hv).
    + intros Hclosure.
      destruct (closure_of statics' v') as [[called cslots'] | ] eqn:Hc;
        [ | congruence ].
      destruct (closure_of_back v' v called cslots' Hv Hc)
        as (cslots & -> & _).
      discriminate.
    + unfold erased_to in Hv.
      erased_cases Hv; simpl;
        try (intros Hnone; contradiction (Hnone eq_refl)).
      intros _. discriminate.
    + unfold erased_to in Hv.
      erased_cases Hv; simpl; try (intros _; reflexivity).
      * destruct (lookup s (static_values statics')) as [allocated' | ]
          eqn:Hs';
          [ | discriminate ].
        destruct (undrawn_bound s allocated' Hnot_drawn Hs')
          as (allocated & -> & _).
        reflexivity.
      * rewrite Hlookup. reflexivity.
      * rewrite Hlookup. reflexivity.
      * rewrite Hlookup. reflexivity.
      * rewrite Hlookup. reflexivity.
      * rewrite (Hbound fslot s (lookup_in _ _ _ Hsym)). reflexivity.
    + intros [s0 c] Hs0.
      apply imported_of_symbol in Hs0 as [-> Hs0].
      unfold erased_to in Hv.
      inversion Hv as [ | | | s1 Hnot_drawn | | | | | | | | | | ]; subst.
      destruct (undrawn_bound s0 _ Hnot_drawn Hs0)
        as (allocated & Hs & Hallocated).
      inversion Hallocated; subst. simpl. rewrite Hs. reflexivity.
    + intros elements'. exact (array_of_to v' v elements' Hv).
Qed.

Lemma erased_to_relates_locations : relates_locations erased_to.
Proof.
  destruct (value_erased_relates_locations rs) as (Hsame & Hfirst & Hsecond).
  split; [ exact Hsame | ].
  split; [ intros l v; exact (Hsecond v l) | intros v' l; exact (Hfirst l v') ].
Qed.

Lemma cell_alike_to (c c' : cell) :
  cell_alike (value_erased rs) c c' -> cell_alike erased_to c' c.
Proof.
  destruct c as [mut tag fields | kind elements],
    c' as [mut' tag' fields' | kind' elements'];
    simpl; try exact (fun H => H).
  - intros (Hmut & Htag & Hfields). split; [ symmetry; exact Hmut | ].
    split; [ symmetry; exact Htag | exact (values_to _ _ Hfields) ].
  - intros (Hkind & Helements). split; [ symmetry; exact Hkind | ].
    exact (values_to _ _ Helements).
Qed.

(* Two heaps that bind the same locations, entry for entry: a location
   that one binds, the other binds. *)
Lemma lookup_same_locations (h h' : heap) (l : location) :
  map fst h = map fst h' -> lookup l h = None -> lookup l h' = None.
Proof.
  revert h'. induction h as [ | [l0 c] h IH]; intros [ | [l0' c'] h'];
    simpl; try discriminate; [ reflexivity | ].
  intros [= <- Hkeys]. destruct (Pos.eqb l l0); [ discriminate | ].
  exact (IH h' Hkeys).
Qed.

(* The premise of that law, from the second heap to the first: the
   cells that the arguments of the second state name are those that
   the arguments of the first name, which may be evaluated, so none is
   dead. *)
Lemma heaps_erased_given_to (h h' : heap) (vs' vs : list value) :
  heaps_erased rs h h' ->
  Forall2 erased_to vs' vs ->
  Forall (fun v => value_usable rs v = true) vs ->
  heap_alike_given erased_to h' h vs'.
Proof.
  intros [Hkeys Hcells] Hvs Husable. split; [ symmetry; exact Hkeys | ].
  intros l c' Hin Hl'.
  assert (Hdead : cell_dead rs l = false).
  { clear Hl'. induction Hvs as [ | v' v vs' vs Hv _ IH];
      [ contradiction Hin | ].
    inversion Husable as [ | v0 vs0 Hhead Htail]; subst.
    destruct Hin as [-> | Hin]; [ | exact (IH Htail Hin) ].
    unfold erased_to in Hv. inversion Hv; subst.
    simpl in Hhead. apply negb_true_iff in Hhead. exact Hhead. }
  destruct (lookup l h) as [c | ] eqn:Hl.
  - destruct (Hcells l c Hdead Hl) as (c0 & Hl0 & Halike).
    rewrite Hl' in Hl0. injection Hl0 as <-.
    exists c. split; [ reflexivity | exact (cell_alike_to c c' Halike) ].
  - rewrite (lookup_same_locations h h' l Hkeys Hl) in Hl'. discriminate.
Qed.

(* A primitive that the second state runs on its heap, the machine of
   regions runs on its own, on arguments that may be evaluated, to a
   value that the value of the second is the erasure of. *)
Lemma exec_prim_to (p : prim) (vs' vs : list value) (hp hp' hp1' : heap)
    (v' : value) :
  heaps_erased rs hp hp' ->
  Forall2 erased_to vs' vs ->
  Forall (fun v => value_usable rs v = true) vs ->
  exec_prim statics' hp' p vs' = Some (v', hp1') ->
  ∃ (v : value) (hp1 : heap),
    exec_prim (live_statics rs) hp p vs = Some (v, hp1)
    ∧ value_erased rs v v'.
Proof.
  intros Hheaps Hvs Husable Hprim'.
  destruct
    (exec_prim_related_given erased_to statics' (live_statics rs) hp' hp hp1'
       p vs' vs v' erased_to_respects erased_to_relates_locations
       (heaps_erased_given_to hp hp' vs' vs Hheaps Hvs Husable) Hvs Hprim')
    as (v & hp1 & Hprim & Hv & _).
  exists v, hp1. split; [ exact Hprim | exact Hv ].
Qed.

End Back.

(* ======================================================================== *)
(* Evaluation, from the second state to the first                           *)
(* ======================================================================== *)

(* A simple that has a value in the second environment has one in the
   first: the second binds no variable that the first does not. *)
Lemma eval_simple_back (rs : region_state) (vars vars' : env) (s : simple)
    (v' : value) :
  env_erased rs vars vars' ->
  eval_simple vars' s = Some v' ->
  ∃ v : value, eval_simple vars s = Some v.
Proof.
  intros [_ Hunbound].
  destruct s as [x | sym | c]; simpl; [ | intros _; eauto | intros _; eauto ].
  intros Hx'. destruct (lookup x vars) as [v | ] eqn:Hx; [ eauto | ].
  rewrite (Hunbound x Hx) in Hx'. discriminate.
Qed.

Lemma eval_simples_back (rs : region_state) (vars vars' : env)
    (args : list simple) :
  env_erased rs vars vars' ->
  ∀ vs' : list value,
    eval_simples vars' args = Some vs' ->
    ∃ vs : list value, eval_simples vars args = Some vs.
Proof.
  intros Hvars. induction args as [ | s args IH]; intros vs'; simpl;
    [ intros _; eauto | ].
  destruct (eval_simple vars' s) as [v' | ] eqn:Hs'; [ | discriminate ].
  destruct (eval_simples vars' args) as [rest' | ]; [ | discriminate ].
  intros _.
  destruct (eval_simple_back rs vars vars' s v' Hvars Hs') as [v ->].
  destruct (IH rest' eq_refl) as [rest ->]. eauto.
Qed.

Lemma eval_value_slots_back (rs : region_state) (vars vars' : env)
    (slots : list (value_slot * simple)) :
  env_erased rs vars vars' ->
  ∀ evaluated' : list (value_slot * value),
    eval_value_slots vars' slots = Some evaluated' ->
    ∃ evaluated : list (value_slot * value),
      eval_value_slots vars slots = Some evaluated.
Proof.
  intros Hvars. induction slots as [ | [slot s] slots IH];
    intros evaluated'; simpl; [ intros _; eauto | ].
  destruct (eval_simple vars' s) as [v' | ] eqn:Hs'; [ | discriminate ].
  destruct (eval_value_slots vars' slots) as [rest' | ]; [ | discriminate ].
  intros _.
  destruct (eval_simple_back rs vars vars' s v' Hvars Hs') as [v ->].
  destruct (IH rest' eq_refl) as [rest ->]. eauto.
Qed.

(* And the value in the first is erased to the value in the second, if
   it may be evaluated and reaches only symbols that may be: the first
   state has a value, which
   FlambdaRegionsErasedProofs.eval_simple_erased gives erased in the
   second. That it may be evaluated is asked apart: a value that
   reaches only symbols that may be evaluated may be the location of a
   dead cell. *)
Lemma eval_simple_related_back (rs : region_state) (vars vars' : env)
    (s : simple) (v' : value) :
  env_erased rs vars vars' ->
  simple_below (rs_base rs) s = true ->
  (∀ v : value, eval_simple vars s = Some v -> value_usable rs v = true) ->
  (∀ v : value,
    eval_simple vars s = Some v -> value_all (live_symbol rs) v = true) ->
  eval_simple vars' s = Some v' ->
  ∃ v : value, eval_simple vars s = Some v ∧ erased_back rs v' v.
Proof.
  intros Hvars Hbelow Husable Hall Hs'.
  destruct (eval_simple_back rs vars vars' s v' Hvars Hs') as [v Hs].
  pose proof (Hall v Hs) as Hv_live.
  destruct
    (eval_simple_erased rs vars vars' s v Hvars Hbelow Hs (Husable v Hs))
    as (v0 & Hs0 & Hv).
  rewrite Hs' in Hs0. injection Hs0 as <-.
  exists v. split; [ exact Hs | exact (conj Hv Hv_live) ].
Qed.

Lemma eval_simples_related_back (rs : region_state) (vars vars' : env)
    (args : list simple) (vs' : list value) :
  env_erased rs vars vars' ->
  forallb (simple_below (rs_base rs)) args = true ->
  simples_usable rs vars args ->
  simples_all_live rs vars args ->
  eval_simples vars' args = Some vs' ->
  ∃ vs : list value,
    eval_simples vars args = Some vs ∧ Forall2 (erased_back rs) vs' vs.
Proof.
  intros Hvars Hbelow Husable Hall Hargs'.
  destruct (eval_simples_back rs vars vars' args Hvars vs' Hargs')
    as [vs Hargs].
  destruct
    (eval_simples_erased rs vars vars' args Hvars Hbelow Husable vs Hargs)
    as (vs0 & Hargs0 & Hvs).
  rewrite Hargs' in Hargs0. injection Hargs0 as <-.
  exists vs. split; [ exact Hargs | ].
  exact
    (values_back rs vs vs' Hvs
       (simples_all_live_values rs vars args vs Hall Hargs)).
Qed.

(* A defining expression that has a value in the second state has one
   on the machine of regions, under the static environment that [step]
   is given there. *)
Lemma eval_named_defined (rs : region_state) (statics' : static_env)
    (vars vars' : env) (d : named) (v' : value) :
  symbols_fresh rs = true ->
  statics_erased rs statics' ->
  forallb (fun '(_, allocated) => allocated_all (live_symbol rs) allocated)
    (static_values (live_statics rs))
  = true ->
  env_erased rs vars vars' ->
  forallb (simple_below (rs_base rs)) (simples_of_named d) = true ->
  simples_usable rs vars (simples_of_named d) ->
  simples_all_live rs vars (simples_of_named d) ->
  eval_named statics' vars' d = Some v' ->
  ∃ v : value, eval_named (live_statics rs) vars d = Some v.
Proof.
  intros Hfresh Hstatics Hlive Hvars Hbelow Husable Hall.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    simpl in Hbelow, Husable, Hall |- *.
  - exact (eval_simple_back rs vars vars' s v' Hvars).
  - destruct (eval_simples vars' args) as [vs' | ] eqn:Hargs';
      [ | discriminate ].
    destruct
      (eval_simples_related_back rs vars vars' args vs' Hvars Hbelow Husable
         Hall Hargs')
      as (vs & -> & Hvs).
    intros Hp.
    destruct
      (eval_prim_back rs statics' Hfresh Hstatics Hlive p vs' vs v' Hvs Hp)
      as (v & Hv & _).
    eauto.
  - destruct (eval_value_slots vars' slots) as [vs' | ] eqn:Hslots';
      [ | discriminate ].
    destruct (eval_value_slots_back rs vars vars' slots Hvars vs' Hslots')
      as [vs ->].
    intros _. eauto.
  - destruct (lookup fslot decls); [ | discriminate ]. simpl.
    destruct (eval_value_slots vars' slots) as [vs' | ] eqn:Hslots';
      [ | discriminate ].
    destruct (eval_value_slots_back rs vars vars' slots Hvars vs' Hslots')
      as [vs ->].
    intros _. eauto.
  - apply andb_true_iff in Hbelow as [Hbelow _].
    destruct (eval_simple vars' closure) as [w' | ] eqn:Hclosure';
      [ | discriminate ].
    destruct
      (eval_simple_related_back rs vars vars' closure w' Hvars Hbelow
         (fun v => Husable closure v (or_introl eq_refl))
         (fun v => Hall closure v (or_introl eq_refl)) Hclosure')
      as (w & -> & [Hw _]).
    simpl.
    destruct (closure_of statics' w') as [[called cslots'] | ] eqn:Hc';
      [ | discriminate ].
    destruct
      (closure_of_back rs statics' Hfresh Hstatics w' w called cslots' Hw Hc')
      as (cslots & -> & Hcslots).
    simpl. exact (lookup_slots_back rs slot cslots cslots' v' Hcslots).
  - apply andb_true_iff in Hbelow as [Hbelow _].
    destruct (eval_simple vars' closure) as [w' | ] eqn:Hclosure';
      [ | discriminate ].
    destruct
      (eval_simple_related_back rs vars vars' closure w' Hvars Hbelow
         (fun v => Husable closure v (or_introl eq_refl))
         (fun v => Hall closure v (or_introl eq_refl)) Hclosure')
      as (w & -> & [Hw _]).
    exact
      (project_value_slot_back rs statics' Hfresh Hstatics pfrom slot w' w v'
         Hw).
  - apply andb_true_iff in Hbelow as [Hbelow _].
    destruct (eval_simple vars' closure) as [w' | ] eqn:Hclosure';
      [ | discriminate ].
    destruct
      (eval_simple_related_back rs vars vars' closure w' Hvars Hbelow
         (fun v => Husable closure v (or_introl eq_refl))
         (fun v => Hall closure v (or_introl eq_refl)) Hclosure')
      as (w & -> & [Hw _]).
    exact
      (project_function_slot_back rs statics' Hfresh Hstatics move_from
         move_to w' w v' Hw).
Qed.

(* And that value is erased to the value in the second state. *)
Lemma eval_named_back (rs : region_state) (statics' : static_env)
    (vars vars' : env) (d : named) (v' : value) :
  symbols_fresh rs = true ->
  statics_erased rs statics' ->
  forallb (fun '(_, allocated) => allocated_all (live_symbol rs) allocated)
    (static_values (live_statics rs))
  = true ->
  env_erased rs vars vars' ->
  forallb (simple_below (rs_base rs)) (simples_of_named d) = true ->
  simples_usable rs vars (simples_of_named d) ->
  simples_all_live rs vars (simples_of_named d) ->
  eval_named statics' vars' d = Some v' ->
  ∃ v : value,
    eval_named (live_statics rs) vars d = Some v ∧ value_erased rs v v'.
Proof.
  intros Hfresh Hstatics Hlive Hvars Hbelow Husable Hall Hd'.
  destruct
    (eval_named_defined rs statics' vars vars' d v' Hfresh Hstatics Hlive
       Hvars Hbelow Husable Hall Hd')
    as [v Hd].
  destruct
    (eval_named_erased rs statics' vars vars' d v
       (proj1 (proj2 (proj2 Hstatics))) Hvars Hbelow Husable Hd)
    as (v0 & Hd0 & Hv).
  rewrite Hd' in Hd0. injection Hd0 as <-.
  exists v. split; [ exact Hd | exact Hv ].
Qed.

(* A defining expression that the second state runs on its heap, the
   machine of regions runs on its own, under the static environment
   that [step] is given there: a primitive by
   [exec_prim_to], on arguments that may be evaluated; anything else
   has the value of [eval_named], and the heap is as it came. *)
Lemma exec_named_defined (rs : region_state) (statics' : static_env)
    (vars vars' : env) (d : named) (hp hp' hp1' : heap)
    (rd rd1' : read_record) (v' : value) :
  symbols_fresh rs = true ->
  statics_erased rs statics' ->
  forallb (fun '(_, allocated) => allocated_all (live_symbol rs) allocated)
    (static_values (live_statics rs))
  = true ->
  env_erased rs vars vars' ->
  forallb (simple_below (rs_base rs)) (simples_of_named d) = true ->
  simples_usable rs vars (simples_of_named d) ->
  simples_all_live rs vars (simples_of_named d) ->
  heaps_erased rs hp hp' ->
  exec_named statics' hp' rd vars' d = Some (v', hp1', rd1') ->
  ∃ (v : value) (hp1 : heap) (rd1 : read_record),
    exec_named (live_statics rs) hp rd vars d = Some (v, hp1, rd1).
Proof.
  intros Hfresh Hstatics Hlive Hvars Hbelow Husable Hall Hheaps Hexec'.
  assert (Hpure :
    (∀ (p : prim) (args : list simple), d <> Prim p args) ->
    ∃ (v : value) (hp1 : heap) (rd1 : read_record),
      exec_named (live_statics rs) hp rd vars d = Some (v, hp1, rd1)).
  { intros Hnot.
    rewrite (exec_named_not_prim statics' hp' rd vars' d Hnot) in Hexec'.
    destruct (eval_named statics' vars' d) as [w' | ] eqn:Hw';
      [ | discriminate Hexec' ].
    destruct
      (eval_named_defined rs statics' vars vars' d w' Hfresh Hstatics Hlive
         Hvars Hbelow Husable Hall Hw')
      as [w Hw].
    exists w, hp, rd.
    rewrite (exec_named_not_prim (live_statics rs) hp rd vars d Hnot), Hw.
    reflexivity. }
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    try (apply Hpure; intros p0 args0; discriminate).
  clear Hpure.
  apply exec_named_prim_some in Hexec' as (vs' & Hargs' & Hprim' & ->).
  simpl in Hbelow, Husable, Hall.
  destruct (eval_simples_back rs vars vars' args Hvars vs' Hargs')
    as [vs Hargs].
  destruct
    (eval_simples_erased rs vars vars' args Hvars Hbelow Husable vs Hargs)
    as (vs0 & Hargs0 & Hvs).
  rewrite Hargs' in Hargs0. injection Hargs0 as <-.
  destruct
    (exec_prim_to rs statics' Hfresh Hstatics p vs' vs hp hp' hp1' v' Hheaps
       (values_to rs vs vs' Hvs)
       (simples_usable_values rs vars args Husable vs Hargs) Hprim')
    as (v & hp1 & Hprim & _).
  exists v, hp1, rd. apply exec_named_prim_some. exists vs.
  split; [ exact Hargs | ]. split; [ exact Hprim | reflexivity ].
Qed.

(* And its value is erased to the value in the second state, with the
   heaps after related and the same reads. *)
Lemma exec_named_back (rs : region_state) (statics' : static_env)
    (vars vars' : env) (d : named) (hp hp' hp1' : heap)
    (rd rd1' : read_record) (v' : value) :
  symbols_fresh rs = true ->
  statics_erased rs statics' ->
  forallb (fun '(_, allocated) => allocated_all (live_symbol rs) allocated)
    (static_values (live_statics rs))
  = true ->
  env_erased rs vars vars' ->
  forallb (simple_below (rs_base rs)) (simples_of_named d) = true ->
  simples_usable rs vars (simples_of_named d) ->
  simples_all_live rs vars (simples_of_named d) ->
  heaps_erased rs hp hp' ->
  exec_named statics' hp' rd vars' d = Some (v', hp1', rd1') ->
  ∃ (v : value) (hp1 : heap),
    exec_named (live_statics rs) hp rd vars d = Some (v, hp1, rd1')
    ∧ value_erased rs v v'
    ∧ heaps_erased rs hp1 hp1'.
Proof.
  intros Hfresh Hstatics Hlive Hvars Hbelow Husable Hall Hheaps Hexec'.
  destruct
    (exec_named_defined rs statics' vars vars' d hp hp' hp1' rd rd1' v'
       Hfresh Hstatics Hlive Hvars Hbelow Husable Hall Hheaps Hexec')
    as (v & hp1 & rd1 & Hexec).
  destruct
    (exec_named_erased rs statics' vars vars' d hp hp' hp1 rd rd1 v
       (proj1 (proj2 (proj2 Hstatics))) Hvars Hbelow Husable Hheaps Hexec)
    as (v0 & hp0 & Hexec0 & Hv & Hheaps1).
  rewrite Hexec' in Hexec0. injection Hexec0 as <- <- <-.
  exists v, hp1. split; [ exact Hexec | ]. split; [ exact Hv | exact Hheaps1 ].
Qed.

(* ======================================================================== *)
(* Static bindings                                                          *)
(* ======================================================================== *)

Lemma eval_static_const_back (rs : region_state) (vars vars' : env)
    (const : static_const) (allocated' : statically_allocated_value) :
  env_erased rs vars vars' ->
  eval_static_const vars' const = Some allocated' ->
  ∃ allocated : statically_allocated_value,
    eval_static_const vars const = Some allocated.
Proof.
  intros Hvars.
  destruct const as [code slots | fslot syms decls slots | tag fields
    | contents | bytes | elements];
    simpl.
  - destruct (eval_value_slots vars' slots) as [vs' | ] eqn:Hslots';
      [ | discriminate ].
    destruct (eval_value_slots_back rs vars vars' slots Hvars vs' Hslots')
      as [vs ->].
    intros _. eauto.
  - destruct (lookup fslot decls); [ | discriminate ]. simpl.
    destruct (eval_value_slots vars' slots) as [vs' | ] eqn:Hslots';
      [ | discriminate ].
    destruct (eval_value_slots_back rs vars vars' slots Hvars vs' Hslots')
      as [vs ->].
    intros _. eauto.
  - destruct (eval_simples vars' fields) as [vs' | ] eqn:Hfields';
      [ | discriminate ].
    destruct (eval_simples_back rs vars vars' fields Hvars vs' Hfields')
      as [vs ->].
    intros _. eauto.
  - destruct (eval_simple vars' contents) as [v' | ] eqn:Hcontents';
      [ | discriminate ].
    destruct (eval_simple_back rs vars vars' contents v' Hvars Hcontents')
      as [v ->].
    intros _. eauto.
  - intros _. eauto.
  - destruct (eval_simples vars' elements) as [vs' | ] eqn:Helements';
      [ | discriminate ].
    destruct (eval_simples_back rs vars vars' elements Hvars vs' Helements')
      as [vs ->].
    intros _. eauto.
Qed.

(* A group of static bindings that the second state can add, erased,
   to its static environment, the first can add to one that binds no
   code that the second does not bind, and no symbol that the machine
   did not draw and that the second does not bind. *)
Lemma eval_static_bindings_back (rs : region_state) (vars vars' : env)
    (bindings : list static_binding) :
  env_erased rs vars vars' ->
  forallb (binding_below (rs_base rs)) bindings = true ->
  ∀ statics statics' statics1' : static_env,
    (∀ id : code_id,
      lookup id (static_code statics') = None ->
      lookup id (static_code statics) = None) ->
    (∀ s : symbol,
      symbol_below (rs_base rs) s = true ->
      lookup s (static_values statics') = None ->
      lookup s (static_values statics) = None) ->
    eval_static_bindings vars' (map erase_binding bindings) statics'
    = Some statics1' ->
    ∃ statics1 : static_env,
      eval_static_bindings vars bindings statics = Some statics1.
Proof.
  intros Hvars.
  induction bindings as [ | binding bindings IH];
    intros Hbelow statics statics' statics1' Hcode Hsymbols;
    [ simpl; intros _; eauto | ].
  simpl in Hbelow. apply andb_true_iff in Hbelow as [Hbinding Hbelow].
  destruct binding as [id c | id | s const]; simpl.
  - destruct (lookup id (static_code statics')) eqn:Hid'; [ discriminate | ].
    rewrite (Hcode id Hid').
    apply (IH Hbelow); [ | exact Hsymbols ].
    intros id0. simpl.
    destruct (Pos.eqb id0 id); [ discriminate | apply Hcode ].
  - exact (IH Hbelow statics statics' statics1' Hcode Hsymbols).
  - simpl in Hbinding. apply andb_true_iff in Hbinding as [Hs _].
    destruct (lookup s (static_values statics')) eqn:Hs'; [ discriminate | ].
    rewrite (Hsymbols s Hs Hs').
    destruct (eval_static_const vars' const) as [allocated' | ]
      eqn:Hallocated';
      [ | discriminate ].
    destruct
      (eval_static_const_back rs vars vars' const allocated' Hvars
         Hallocated')
      as [allocated ->].
    simpl. apply (IH Hbelow); [ exact Hcode | ].
    intros s0 Hs0. simpl.
    destruct (Pos.eqb s0 s); [ discriminate | exact (Hsymbols s0 Hs0) ].
Qed.
