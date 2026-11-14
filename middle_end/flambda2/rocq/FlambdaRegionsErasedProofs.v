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

(* The erasure of regions (FlambdaRegionsErasureSpec.v): the relation
   between a state of the machine of regions and a state of the machine
   of FlambdaSemanticsSpec.v that runs the term with its regions
   dropped, and what follows from it of the evaluation of simples, of
   defining expressions and of the observations of a value. Nothing
   here needs review: no statement of a Spec file is proved here. It
   is what the proofs of the two statements of
   FlambdaRegionsErasureSpec.v are to be built on: a step of either
   machine is to keep the relation.

   The relation is FlambdaLiftProofs.state_lifted read the other way:
   the first state, that of the machine of regions, is the one with
   symbols.
   - A value is related as FlambdaRegionsErasureSpec.value_erased says.
   - A variable is asked nothing where its value may not be evaluated
     and is no location: a mark, or the symbol of a local allocation
     that is gone ([env_erased]). The second environment may bind it
     to anything, or not at all. A location is asked, whether or not
     its cell is dead.
   - The second expression is [erase] of the first under a list of the
     continuations in scope that agrees with what the continuations
     stand for ([conts_agree]): one that a [Let_cont] of the body
     binds is listed with the places of the regions among the
     parameters of its handler, and any other stands for something
     that takes no region.
   - A continuation is related to itself with its handler erased, its
     environments related and the values that it keeps related where
     they may be evaluated or are locations ([cont_erased]).
   - The second static environment has the code erased and binds the
     symbols that the machine did not draw, to the same things with
     their parts related ([statics_erased]). The symbols of the local
     allocations are bound on the first side only.
   - The two heaps bind the same locations, entry for entry, and a
     cell of the first that is not dead (FlambdaRegionsSpec.cell_dead)
     is like the cell of the second at its location, with the values
     related ([heaps_erased]). Nothing is asked of what a dead cell
     holds. The reads are the same.
   - The frames, the counter, the counts and the list of the cells that
     local allocations made are of the first state only. *)

From Stdlib Require Import ZArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaPrimitiveModel FlambdaPrimitiveProofs
  FlambdaSemanticsSpec FlambdaSemanticsProofs FlambdaHeapProofs
  FlambdaClosuresSpec FlambdaClosuresProofs FlambdaLiftProofs
  FlambdaRegionsSpec FlambdaRegionsProofs FlambdaRegionsErasureSpec.
Import ListNotations.

(* The static environment that [step] is given is not unfolded: the
   rules of FlambdaRegionsErasureSpec.value_erased name it. *)
#[local] Opaque live_statics.

(* ======================================================================== *)
(* Values                                                                   *)
(* ======================================================================== *)

Ltac erased_cases H :=
  destruct H
    as [ c | c | l | s Hnot_drawn | tag fields fields' Hfields
       | items items' Hitems
       | code slots slots' Hslots | fslot decls slots slots' Hslots
       | pcallee pcallee' pargs pargs' parity Hpcallee Hpargs
       | s tag fields fields' Hdrawn Hlookup Hfields
       | s c Hdrawn Hlookup
       | s items items' Hdrawn Hlookup Hitems
       | s code slots slots' Hdrawn Hlookup Hslots
       | s fslot syms decls slots slots' Hdrawn Hsym Hsyms Hbound Hslots ].

(* A value erased has the kind of the value: a symbol stands for what
   is allocated, of kind [Value]. *)
Lemma kind_of_value_erased (rs : region_state) (v v' : value) :
  value_erased rs v v' -> kind_of_value v' = kind_of_value v.
Proof. intros H. erased_cases H; reflexivity. Qed.

Lemma kinds_of_values_erased (rs : region_state) (vs vs' : list value) :
  Forall2 (value_erased rs) vs vs' ->
  map kind_of_value vs' = map kind_of_value vs.
Proof.
  induction 1 as [ | v v' vs vs' Hv _ IH]; simpl; [ reflexivity | ].
  rewrite (kind_of_value_erased rs v v' Hv), IH. reflexivity.
Qed.

Lemma value_erased_if_usable_intro (rs : region_state) (v v' : value) :
  value_erased rs v v' -> value_erased_if_usable rs v v'.
Proof.
  intros H. split; [ exact (kind_of_value_erased rs v v' H) | ].
  intros _. exact H.
Qed.

Lemma values_erased_if_usable_intro (rs : region_state)
    (vs vs' : list value) :
  Forall2 (value_erased rs) vs vs' ->
  Forall2 (value_erased_if_usable rs) vs vs'.
Proof.
  induction 1 as [ | v v' vs vs' Hv _ IH]; constructor;
    [ exact (value_erased_if_usable_intro rs v v' Hv) | exact IH ].
Qed.

Lemma kinds_of_values_if_usable (rs : region_state) (vs vs' : list value) :
  Forall2 (value_erased_if_usable rs) vs vs' ->
  map kind_of_value vs' = map kind_of_value vs.
Proof.
  induction 1 as [ | v v' vs vs' [Hkind _] _ IH]; simpl; [ reflexivity | ].
  rewrite Hkind, IH. reflexivity.
Qed.

(* The values of which the relation is asked wherever a value that is
   gone may be (the parts of a partial application, a value that a
   continuation keeps, a value at an exit, the value of a variable):
   those that may be evaluated, and the locations. A location is
   erased to itself whether or not its cell is dead. It is the premise
   of the rule of a partial application and of
   FlambdaRegionsErasureSpec.value_erased_if_usable, named. *)
Definition erased_asked (rs : region_state) (v : value) : Prop :=
  value_usable rs v = true ∨ is_location v = true.

Lemma usable_asked (rs : region_state) (v : value) :
  value_usable rs v = true -> erased_asked rs v.
Proof. intros H. left. exact H. Qed.

Lemma location_asked (rs : region_state) (l : location) :
  erased_asked rs (Value_location l).
Proof. right. reflexivity. Qed.

(* Values that may all be evaluated and are related where they may be
   are related. *)
Lemma values_erased_of_usable (rs : region_state) (vs vs' : list value) :
  Forall2 (value_erased_if_usable rs) vs vs' ->
  Forall (fun v => value_usable rs v = true) vs ->
  Forall2 (value_erased rs) vs vs'.
Proof.
  induction 1 as [ | v v' vs vs' [_ Hv] _ IH]; intros Husable;
    [ constructor | ].
  inversion Husable as [ | v0 vs0 Hhead Htail]; subst.
  constructor; [ exact (Hv (usable_asked rs v Hhead)) | exact (IH Htail) ].
Qed.

(* A partial application is erased to a partial application, and
   nothing else is. *)
Lemma partial_application_erased (rs : region_state) (v v' : value) :
  value_erased rs v v' ->
  match partial_application_of v with
  | Some (closure, applied, applied_arity) =>
    ∃ (closure' : value) (applied' : list value),
      v' = Value_partial_application closure' applied' applied_arity
      ∧ (erased_asked rs closure -> value_erased rs closure closure')
      ∧ Forall2 (value_erased_if_usable rs) applied applied'
  | None => partial_application_of v' = None
  end.
Proof.
  intros H. erased_cases H; simpl; try reflexivity.
  exists pcallee', pargs'. auto.
Qed.

(* The value of a slot of erased value slots is erased. *)
Lemma lookup_slots_erased (rs : region_state) (slot : value_slot)
    (slots slots' : list (value_slot * value)) (v : value) :
  Forall2 (slot_erased (value_erased rs)) slots slots' ->
  lookup slot slots = Some v ->
  ∃ v' : value, lookup slot slots' = Some v' ∧ value_erased rs v v'.
Proof.
  induction 1 as [ | [k w] [k' w'] slots slots' [Hkey Hw] _ IH]; simpl;
    [ discriminate | ].
  simpl in Hkey, Hw. subst k'.
  destruct (Pos.eqb slot k); [ intros [= <-]; eauto | exact IH ].
Qed.

Lemma naked_immediate_erased (rs : region_state) (v v' : value) (i : Z) :
  value_erased rs v v' ->
  naked_immediate_of_value v = Some i ->
  naked_immediate_of_value v' = Some i.
Proof.
  intros Hv. erased_cases Hv; simpl; try discriminate. exact (fun H => H).
Qed.

(* ======================================================================== *)
(* From one state of the machine of regions to another                      *)
(* ======================================================================== *)

(* The second state has the base of the first, binds every symbol that
   the first binds, to the same thing, binds no symbol that the
   machine drew and that the first does not bind, and has the dead
   cells of the first and no other. So a value may be evaluated in the
   second exactly if it may in the first, and what is related in the
   first is related in the second. A step that pops no frame and makes
   no local allocation is such. *)
Definition region_extends (rs rs1 : region_state) : Prop :=
  rs_base rs1 = rs_base rs
  ∧ (∀ (s : symbol) (allocated : statically_allocated_value),
      lookup s (static_values (live_statics rs)) = Some allocated ->
      lookup s (static_values (live_statics rs1)) = Some allocated)
  ∧ (∀ s : symbol,
      drawn rs s = true ->
      lookup s (static_values (live_statics rs)) = None ->
      lookup s (static_values (live_statics rs1)) = None)
  ∧ (∀ l : location, cell_dead rs1 l = cell_dead rs l).

Lemma region_extends_refl (rs : region_state) : region_extends rs rs.
Proof. split; [ reflexivity | ]. split; [ auto | ]. split; auto. Qed.

Lemma region_extends_trans (rs rs1 rs2 : region_state) :
  region_extends rs rs1 -> region_extends rs1 rs2 -> region_extends rs rs2.
Proof.
  intros (Hbase & Hkept & Hgone & Hdead) (Hbase1 & Hkept1 & Hgone1 & Hdead1).
  split; [ congruence | ]. split; [ | split ].
  - intros s allocated Hs. exact (Hkept1 s _ (Hkept s _ Hs)).
  - intros s Hdrawn Hs. apply Hgone1; [ | exact (Hgone s Hdrawn Hs) ].
    unfold drawn in *. rewrite Hbase. exact Hdrawn.
  - intros l. rewrite Hdead1. apply Hdead.
Qed.

(* The same base, the same symbols bound and the same dead cells. *)
Lemma region_extends_same (rs rs1 : region_state) :
  rs_base rs1 = rs_base rs ->
  static_values (live_statics rs1) = static_values (live_statics rs) ->
  (∀ l : location, cell_dead rs1 l = cell_dead rs l) ->
  region_extends rs rs1.
Proof.
  intros Hbase Hvalues Hdead. split; [ exact Hbase | ].
  rewrite Hvalues. split; [ auto | ]. split; [ auto | exact Hdead ].
Qed.

Lemma cell_dead_extends (rs rs1 : region_state) (l : location) :
  region_extends rs rs1 -> cell_dead rs1 l = cell_dead rs l.
Proof. intros (_ & _ & _ & Hdead). apply Hdead. Qed.

Lemma drawn_extends (rs rs1 : region_state) (s : symbol) :
  region_extends rs rs1 -> drawn rs1 s = drawn rs s.
Proof. intros (Hbase & _). unfold drawn. rewrite Hbase. reflexivity. Qed.

Lemma value_usable_extends (rs rs1 : region_state) (v : value) :
  region_extends rs rs1 -> value_usable rs1 v = value_usable rs v.
Proof.
  intros Hextends. destruct v as [ | | | s | | | | l | ]; try reflexivity.
  - unfold value_usable. rewrite (drawn_extends rs rs1 s Hextends).
    destruct Hextends as (_ & Hkept & Hgone & _).
    destruct (drawn rs s) eqn:Hdrawn; [ | reflexivity ].
    destruct (lookup s (static_values (live_statics rs))) as [allocated | ]
      eqn:Hs.
    + rewrite (Hkept s allocated Hs). reflexivity.
    + rewrite (Hgone s Hdrawn Hs). reflexivity.
  - unfold value_usable. rewrite (cell_dead_extends rs rs1 l Hextends).
    reflexivity.
Qed.

Lemma erased_asked_extends (rs rs1 : region_state) (v : value) :
  region_extends rs rs1 -> erased_asked rs1 v -> erased_asked rs v.
Proof.
  intros Hextends [Husable | Hlocation]; [ left | right; exact Hlocation ].
  rewrite <- (value_usable_extends rs rs1 v Hextends). exact Husable.
Qed.

Fixpoint value_erased_extend (rs rs1 : region_state)
    (Hextends : region_extends rs rs1) (v v' : value)
    (H : value_erased rs v v') {struct H} :
  value_erased rs1 v v'.
Proof.
  erased_cases H.
  - apply Erased_const.
  - apply Erased_boxed_number.
  - apply Erased_location.
  - apply Erased_symbol.
    rewrite (drawn_extends rs rs1 s Hextends). exact Hnot_drawn.
  - apply Erased_block.
    revert fields fields' Hfields. fix IH 3.
    intros fields fields' [ | w w' rest rest' Hhead Htail]; constructor;
      [ exact (value_erased_extend _ _ Hextends _ _ Hhead)
      | exact (IH _ _ Htail) ].
  - apply Erased_array.
    revert items items' Hitems. fix IH 3.
    intros items items' [ | w w' rest rest' Hhead Htail]; constructor;
      [ exact (value_erased_extend _ _ Hextends _ _ Hhead)
      | exact (IH _ _ Htail) ].
  - apply Erased_closure.
    revert slots slots' Hslots. fix IH 3.
    intros slots slots' [ | w w' rest rest' [Hkey Hhead] Htail]; constructor;
      [ split;
        [ exact Hkey | exact (value_erased_extend _ _ Hextends _ _ Hhead) ]
      | exact (IH _ _ Htail) ].
  - apply Erased_closure_of_set.
    revert slots slots' Hslots. fix IH 3.
    intros slots slots' [ | w w' rest rest' [Hkey Hhead] Htail]; constructor;
      [ split;
        [ exact Hkey | exact (value_erased_extend _ _ Hextends _ _ Hhead) ]
      | exact (IH _ _ Htail) ].
  - apply Erased_partial_application.
    + intros Hasked.
      exact
        (value_erased_extend _ _ Hextends _ _
           (Hpcallee (erased_asked_extends rs rs1 pcallee Hextends Hasked))).
    + clear Hpcallee. revert pargs pargs' Hpargs. fix IH 3.
      intros args args' [ | w w' rest rest' [Hkind Hhead] Htail];
        constructor; [ | exact (IH _ _ Htail) ].
      split; [ exact Hkind | ].
      intros Hasked.
      exact
        (value_erased_extend _ _ Hextends _ _
           (Hhead (erased_asked_extends rs rs1 w Hextends Hasked))).
  - apply (Erased_local_block rs1 s tag fields fields');
      [ rewrite (drawn_extends rs rs1 s Hextends); exact Hdrawn
      | exact (proj1 (proj2 Hextends) s _ Hlookup) | ].
    clear Hlookup. revert fields fields' Hfields. fix IH 3.
    intros fields fields' [ | w w' rest rest' Hhead Htail]; constructor;
      [ exact (value_erased_extend _ _ Hextends _ _ Hhead)
      | exact (IH _ _ Htail) ].
  - apply Erased_local_boxed_number;
      [ rewrite (drawn_extends rs rs1 s Hextends); exact Hdrawn
      | exact (proj1 (proj2 Hextends) s _ Hlookup) ].
  - apply (Erased_local_array rs1 s items items');
      [ rewrite (drawn_extends rs rs1 s Hextends); exact Hdrawn
      | exact (proj1 (proj2 Hextends) s _ Hlookup) | ].
    clear Hlookup. revert items items' Hitems. fix IH 3.
    intros items items' [ | w w' rest rest' Hhead Htail]; constructor;
      [ exact (value_erased_extend _ _ Hextends _ _ Hhead)
      | exact (IH _ _ Htail) ].
  - apply (Erased_local_closure rs1 s code slots slots');
      [ rewrite (drawn_extends rs rs1 s Hextends); exact Hdrawn
      | exact (proj1 (proj2 Hextends) s _ Hlookup) | ].
    clear Hlookup. revert slots slots' Hslots. fix IH 3.
    intros slots slots' [ | w w' rest rest' [Hkey Hhead] Htail]; constructor;
      [ split;
        [ exact Hkey | exact (value_erased_extend _ _ Hextends _ _ Hhead) ]
      | exact (IH _ _ Htail) ].
  - apply (Erased_local_closure_of_set rs1 s fslot syms decls slots slots');
      [ rewrite (drawn_extends rs rs1 s Hextends); exact Hdrawn
      | exact Hsym | exact Hsyms
      | intros slot0 s0 Hin;
        exact (proj1 (proj2 Hextends) s0 _ (Hbound slot0 s0 Hin)) | ].
    clear Hbound. revert slots slots' Hslots. fix IH 3.
    intros slots slots' [ | w w' rest rest' [Hkey Hhead] Htail]; constructor;
      [ split;
        [ exact Hkey | exact (value_erased_extend _ _ Hextends _ _ Hhead) ]
      | exact (IH _ _ Htail) ].
Qed.

Lemma values_erased_extend (rs rs1 : region_state) (vs vs' : list value) :
  region_extends rs rs1 ->
  Forall2 (value_erased rs) vs vs' ->
  Forall2 (value_erased rs1) vs vs'.
Proof.
  intros Hextends. induction 1 as [ | v v' vs vs' Hv _ IH]; constructor;
    [ exact (value_erased_extend _ _ Hextends _ _ Hv) | exact IH ].
Qed.

Lemma slots_erased_extend (rs rs1 : region_state)
    (slots slots' : list (value_slot * value)) :
  region_extends rs rs1 ->
  Forall2 (slot_erased (value_erased rs)) slots slots' ->
  Forall2 (slot_erased (value_erased rs1)) slots slots'.
Proof.
  intros Hextends.
  induction 1 as [ | slot slot' slots slots' [Hkey Hv] _ IH]; constructor;
    [ split; [ exact Hkey | exact (value_erased_extend _ _ Hextends _ _ Hv) ]
    | exact IH ].
Qed.

Lemma value_erased_if_usable_extend (rs rs1 : region_state) (v v' : value) :
  region_extends rs rs1 ->
  value_erased_if_usable rs v v' ->
  value_erased_if_usable rs1 v v'.
Proof.
  intros Hextends [Hkind Hv]. split; [ exact Hkind | ].
  intros Hasked.
  exact
    (value_erased_extend _ _ Hextends _ _
       (Hv (erased_asked_extends rs rs1 v Hextends Hasked))).
Qed.

Lemma values_erased_if_usable_extend (rs rs1 : region_state)
    (vs vs' : list value) :
  region_extends rs rs1 ->
  Forall2 (value_erased_if_usable rs) vs vs' ->
  Forall2 (value_erased_if_usable rs1) vs vs'.
Proof.
  intros Hextends. induction 1 as [ | v v' vs vs' Hv _ IH]; constructor;
    [ exact (value_erased_if_usable_extend _ _ _ _ Hextends Hv) | exact IH ].
Qed.

(* ======================================================================== *)
(* Static environments                                                      *)
(* ======================================================================== *)

(* What a symbol that the machine did not draw is bound to in the first
   state and in the second: the same, with erased parts. The symbols
   of a set are none that the machine drew. *)
Inductive allocated_erased (rs : region_state)
    : statically_allocated_value -> statically_allocated_value -> Prop :=
  | Erased_static_closure (code : code_id)
      (slots slots' : list (value_slot * value)) :
    Forall2 (slot_erased (value_erased rs)) slots slots' ->
    allocated_erased rs (Static_closure code slots)
      (Static_closure code slots')
  | Erased_static_closure_of_set (fslot : function_slot)
      (syms : list (function_slot * symbol))
      (decls : list (function_slot * code_id))
      (slots slots' : list (value_slot * value)) :
    (∀ (slot : function_slot) (s : symbol),
      (slot, s) ∈ syms -> drawn rs s = false) ->
    Forall2 (slot_erased (value_erased rs)) slots slots' ->
    allocated_erased rs (Static_closure_of_set fslot syms decls slots)
      (Static_closure_of_set fslot syms decls slots')
  | Erased_static_block (tag : Z) (fields fields' : list value) :
    Forall2 (value_erased rs) fields fields' ->
    allocated_erased rs (Static_block tag fields) (Static_block tag fields')
  | Erased_static_boxed_number (v v' : value) :
    value_erased rs v v' ->
    allocated_erased rs (Static_boxed_number v) (Static_boxed_number v')
  | Erased_static_string (bytes : list Z) :
    allocated_erased rs (Static_string bytes) (Static_string bytes)
  | Erased_static_imported (c : import_class) :
    allocated_erased rs (Static_imported c) (Static_imported c)
  | Erased_static_immutable_array (elements elements' : list value) :
    Forall2 (value_erased rs) elements elements' ->
    allocated_erased rs (Static_immutable_array elements)
      (Static_immutable_array elements').

Ltac allocated_erased_cases H :=
  destruct H
    as [ acode aslots aslots' Haslots
       | afslot asyms adecls aslots aslots' Hasyms Haslots
       | atag afields afields' Hafields | av av' Hav | abytes |
       | aelements aelements' Haelements ].

Lemma allocated_erased_extend (rs rs1 : region_state)
    (allocated allocated' : statically_allocated_value) :
  region_extends rs rs1 ->
  allocated_erased rs allocated allocated' ->
  allocated_erased rs1 allocated allocated'.
Proof.
  intros Hextends H. allocated_erased_cases H; constructor;
    eauto using slots_erased_extend, values_erased_extend,
      value_erased_extend.
  intros slot s Hin. rewrite (drawn_extends rs rs1 s Hextends).
  exact (Hasyms slot s Hin).
Qed.

(* Every symbol that the machine did not draw and that the first state
   binds, in a frame or in its static environment, is bound in the
   second static environment to the same thing, erased. *)
Definition symbols_erased (rs : region_state) (statics' : static_env)
    : Prop :=
  ∀ (s : symbol) (allocated : statically_allocated_value),
    drawn rs s = false ->
    lookup s (static_values (live_statics rs)) = Some allocated ->
    ∃ allocated' : statically_allocated_value,
      lookup s (static_values statics') = Some allocated'
      ∧ allocated_erased rs allocated allocated'.

(* The second static environment:
   - has the code of the first state, erased;
   - that code has no call that returns or raises to a handler with a
     region parameter;
   - binds the symbols that the machine did not draw as the first
     state does ([symbols_erased]);
   - binds no symbol that the static environment of the first state
     does not bind. So a group of static bindings that the first can
     add, the second can. *)
Definition statics_erased (rs : region_state) (statics' : static_env)
    : Prop :=
  (∀ id : code_id,
    lookup id (static_code statics')
    = option_map erase_code
        (lookup id (static_code (state_statics (rs_state rs)))))
  ∧ (∀ (id : code_id) (c : code),
      lookup id (static_code (state_statics (rs_state rs))) = Some c ->
      code_returns_no_region c = true)
  ∧ symbols_erased rs statics'
  ∧ (∀ (s : symbol) (allocated' : statically_allocated_value),
      lookup s (static_values statics') = Some allocated' ->
      lookup s (static_values (state_statics (rs_state rs))) <> None).

(* The symbols of a set bound at a symbol that the machine drew are all
   drawn: one that is not would be bound alike in the second static
   environment, to a set that names no drawn symbol. *)
Lemma set_symbols_drawn (rs : region_state) (statics' : static_env)
    (s : symbol) (fslot : function_slot)
    (syms : list (function_slot * symbol))
    (decls : list (function_slot * code_id))
    (slots : list (value_slot * value)) :
  symbols_erased rs statics' ->
  drawn rs s = true ->
  lookup fslot syms = Some s ->
  static_set_bound (live_statics rs) syms decls slots ->
  ∀ (slot' : function_slot) (s' : symbol),
    lookup slot' syms = Some s' -> drawn rs s' = true.
Proof.
  intros Hsymbols Hdrawn Hsym Hbound slot' s' Hs'.
  destruct (drawn rs s') eqn:Hdrawn'; [ reflexivity | ].
  destruct
    (Hsymbols s' _ Hdrawn' (Hbound slot' s' (lookup_in _ _ _ Hs')))
    as (allocated' & _ & Hallocated).
  inversion Hallocated as [ | ? ? ? ? ? Hno | | | | | ]; subst.
  rewrite (Hno fslot s (lookup_in _ _ _ Hsym)) in Hdrawn. discriminate.
Qed.

(* ======================================================================== *)
(* The observations answer alike of a value and of the value erased         *)
(* ======================================================================== *)

Section Observations.

Variable rs : region_state.
Variable statics' : static_env.
Hypothesis Hsymbols : symbols_erased rs statics'.

Lemma block_of_erased (v v' : value) (block_tag : Z)
    (block_fields : list value) :
  value_erased rs v v' ->
  block_of (live_statics rs) v = Some (block_tag, block_fields) ->
  ∃ block_fields' : list value,
    block_of statics' v' = Some (block_tag, block_fields')
    ∧ Forall2 (value_erased rs) block_fields block_fields'.
Proof.
  intros Hv. erased_cases Hv; simpl; try discriminate;
    try (rewrite Hlookup; try discriminate).
  - destruct (lookup s (static_values (live_statics rs)))
      as [allocated | ] eqn:Hs;
      [ | discriminate ].
    destruct (Hsymbols s allocated Hnot_drawn Hs)
      as (allocated' & -> & Hallocated).
    allocated_erased_cases Hallocated; try discriminate.
    intros [= <- <-]. eauto.
  - intros [= <- <-]. eauto.
  - intros [= <- <-]. eauto.
  - rewrite (Hbound fslot s (lookup_in _ _ _ Hsym)). discriminate.
Qed.

Lemma boxed_number_of_erased (v v' : value) (number : const) :
  value_erased rs v v' ->
  boxed_number_of (live_statics rs) v = Some number ->
  boxed_number_of statics' v' = Some number.
Proof.
  intros Hv. erased_cases Hv; simpl; try discriminate;
    try exact (fun H => H); try (rewrite Hlookup; try discriminate).
  - destruct (lookup s (static_values (live_statics rs)))
      as [allocated | ] eqn:Hs;
      [ | discriminate ].
    destruct (Hsymbols s allocated Hnot_drawn Hs)
      as (allocated' & -> & Hallocated).
    allocated_erased_cases Hallocated; try discriminate.
    destruct av; try discriminate. inversion Hav. exact (fun H => H).
  - exact (fun H => H).
  - rewrite (Hbound fslot s (lookup_in _ _ _ Hsym)). discriminate.
Qed.

Lemma string_of_erased (v v' : value) (bytes : list Z) :
  value_erased rs v v' ->
  string_of (live_statics rs) v = Some bytes ->
  string_of statics' v' = Some bytes.
Proof.
  intros Hv. erased_cases Hv; simpl; try discriminate;
    try (rewrite Hlookup; try discriminate).
  - destruct (lookup s (static_values (live_statics rs)))
      as [allocated | ] eqn:Hs;
      [ | discriminate ].
    destruct (Hsymbols s allocated Hnot_drawn Hs)
      as (allocated' & -> & Hallocated).
    allocated_erased_cases Hallocated; try discriminate. exact (fun H => H).
  - rewrite (Hbound fslot s (lookup_in _ _ _ Hsym)). discriminate.
Qed.

Lemma array_of_erased (v v' : value) (elements : list value) :
  value_erased rs v v' ->
  array_of (live_statics rs) v = Some elements ->
  ∃ elements' : list value,
    array_of statics' v' = Some elements'
    ∧ Forall2 (value_erased rs) elements elements'.
Proof.
  intros Hv. erased_cases Hv; simpl; try discriminate;
    try (rewrite Hlookup; try discriminate).
  - destruct (lookup s (static_values (live_statics rs)))
      as [allocated | ] eqn:Hs;
      [ | discriminate ].
    destruct (Hsymbols s allocated Hnot_drawn Hs)
      as (allocated' & -> & Hallocated).
    allocated_erased_cases Hallocated; try discriminate.
    intros [= <-]. eauto.
  - intros [= <-]. eauto.
  - intros [= <-]. eauto.
  - rewrite (Hbound fslot s (lookup_in _ _ _ Hsym)). discriminate.
Qed.

Lemma closure_in_set_erased (v v' : value) (at_slot : function_slot)
    (function_decls : list (function_slot * code_id))
    (value_slots : list (value_slot * value)) :
  value_erased rs v v' ->
  closure_in_set (live_statics rs) v
  = Some (at_slot, function_decls, value_slots) ->
  ∃ value_slots' : list (value_slot * value),
    closure_in_set statics' v' = Some (at_slot, function_decls, value_slots')
    ∧ Forall2 (slot_erased (value_erased rs)) value_slots value_slots'.
Proof.
  intros Hv. erased_cases Hv; simpl; try discriminate;
    try (rewrite Hlookup; try discriminate).
  - destruct (lookup s (static_values (live_statics rs)))
      as [allocated | ] eqn:Hs;
      [ | discriminate ].
    destruct (Hsymbols s allocated Hnot_drawn Hs)
      as (allocated' & -> & Hallocated).
    allocated_erased_cases Hallocated; try discriminate.
    + intros [= <- <- <-]. eauto.
    + destruct (lookup afslot adecls); [ | discriminate ]. simpl.
      intros [= <- <- <-]. eauto.
  - intros [= <- <- <-]. eauto.
  - destruct (lookup fslot decls); [ | discriminate ]. simpl.
    intros [= <- <- <-]. eauto.
  - intros [= <- <- <-]. eauto.
  - rewrite (Hbound fslot s (lookup_in _ _ _ Hsym)).
    destruct (lookup fslot decls); [ | discriminate ]. simpl.
    intros [= <- <- <-]. eauto.
Qed.

Lemma closure_of_erased (v v' : value) (called : code_id)
    (value_slots : list (value_slot * value)) :
  value_erased rs v v' ->
  closure_of (live_statics rs) v = Some (called, value_slots) ->
  ∃ value_slots' : list (value_slot * value),
    closure_of statics' v' = Some (called, value_slots')
    ∧ Forall2 (slot_erased (value_erased rs)) value_slots value_slots'.
Proof.
  intros Hv. erased_cases Hv; simpl; try discriminate;
    try (rewrite Hlookup; try discriminate).
  - destruct (lookup s (static_values (live_statics rs)))
      as [allocated | ] eqn:Hs;
      [ | discriminate ].
    destruct (Hsymbols s allocated Hnot_drawn Hs)
      as (allocated' & -> & Hallocated).
    allocated_erased_cases Hallocated; try discriminate.
    + intros [= <- <-]. eauto.
    + destruct (lookup afslot adecls); [ | discriminate ]. simpl.
      intros [= <- <-]. eauto.
  - intros [= <- <-]. eauto.
  - destruct (lookup fslot decls); [ | discriminate ]. simpl.
    intros [= <- <-]. eauto.
  - intros [= <- <-]. eauto.
  - rewrite (Hbound fslot s (lookup_in _ _ _ Hsym)).
    destruct (lookup fslot decls); [ | discriminate ]. simpl.
    intros [= <- <-]. eauto.
Qed.

Lemma project_value_slot_erased (project_from : function_slot)
    (slot : value_slot) (v v' w : value) :
  value_erased rs v v' ->
  project_value_slot (live_statics rs) project_from slot v = Some w ->
  ∃ w' : value,
    project_value_slot statics' project_from slot v' = Some w'
    ∧ value_erased rs w w'.
Proof.
  intros Hv. unfold project_value_slot.
  destruct (closure_in_set (live_statics rs) v)
    as [[[at_slot decls] slots] | ] eqn:Hc;
    [ | discriminate ].
  destruct (closure_in_set_erased v v' at_slot decls slots Hv Hc)
    as (slots' & -> & Hslots).
  destruct (Pos.eqb at_slot project_from); [ | discriminate ].
  exact (lookup_slots_erased rs slot slots slots' w Hslots).
Qed.

Lemma project_function_slot_erased (move_from move_to : function_slot)
    (v v' w : value) :
  value_erased rs v v' ->
  project_function_slot (live_statics rs) move_from move_to v = Some w ->
  ∃ w' : value,
    project_function_slot statics' move_from move_to v' = Some w'
    ∧ value_erased rs w w'.
Proof.
  intros Hv. unfold project_function_slot.
  erased_cases Hv; try discriminate;
    try (rewrite Hlookup; try discriminate).
  - (* A symbol that the machine did not draw: bound alike in both. *)
    destruct (lookup s (static_values (live_statics rs)))
      as [allocated | ] eqn:Hs;
      [ | discriminate ].
    destruct (Hsymbols s allocated Hnot_drawn Hs)
      as (allocated' & -> & Hallocated).
    allocated_erased_cases Hallocated; try discriminate.
    + destruct (Pos.eqb move_from unnamed_function_slot); [ | discriminate ].
      destruct (Pos.eqb move_to unnamed_function_slot); [ | discriminate ].
      intros [= <-]. eexists. split; [ reflexivity | ].
      apply Erased_symbol, Hnot_drawn.
    + destruct (Pos.eqb afslot move_from); [ | discriminate ].
      destruct (lookup afslot adecls); [ | discriminate ].
      destruct (lookup move_to adecls); [ | discriminate ]. simpl.
      destruct (lookup move_to asyms) as [s' | ] eqn:Hs'; [ | discriminate ].
      simpl. intros [= <-]. eexists. split; [ reflexivity | ].
      apply Erased_symbol. exact (Hasyms move_to s' (lookup_in _ _ _ Hs')).
  - (* A closure of a set of one function. *)
    destruct (Pos.eqb move_from unnamed_function_slot); [ | discriminate ].
    destruct (Pos.eqb move_to unnamed_function_slot); [ | discriminate ].
    intros [= <-]. eexists. split; [ reflexivity | ].
    apply Erased_closure, Hslots.
  - (* A closure of a set, on the heap in both. *)
    destruct (Pos.eqb fslot move_from); [ | discriminate ].
    destruct (lookup fslot decls); [ | discriminate ].
    destruct (lookup move_to decls); [ | discriminate ]. simpl.
    intros [= <-]. eexists. split; [ reflexivity | ].
    apply Erased_closure_of_set, Hslots.
  - (* A local closure of a set of one function. *)
    destruct (Pos.eqb move_from unnamed_function_slot); [ | discriminate ].
    destruct (Pos.eqb move_to unnamed_function_slot); [ | discriminate ].
    intros [= <-]. eexists. split; [ reflexivity | ].
    exact (Erased_local_closure rs s code slots slots' Hdrawn Hlookup Hslots).
  - (* A local closure of a set: the first state has the symbol of the
       slot asked for, which the machine drew with the others. *)
    rewrite (Hbound fslot s (lookup_in _ _ _ Hsym)).
    destruct (Pos.eqb fslot move_from); [ | discriminate ].
    destruct (lookup fslot decls); [ | discriminate ].
    destruct (lookup move_to decls) as [code' | ] eqn:Hto; [ | discriminate ].
    simpl.
    destruct (lookup_same_keys move_to decls syms (eq_sym Hsyms) code' Hto)
      as [s' Hs'].
    rewrite Hs'. simpl. intros [= <-]. eexists. split; [ reflexivity | ].
    apply (Erased_local_closure_of_set rs s' move_to syms decls slots slots');
      try assumption.
    exact (set_symbols_drawn rs statics' s fslot syms decls slots Hsymbols
             Hdrawn Hsym Hbound move_to s' Hs').
Qed.

(* The relation is one that the observations cannot tell from equality,
   so no primitive can. *)
Lemma value_erased_respects :
  respects_observations (value_erased rs) (live_statics rs) statics'.
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
    + intros tag fields. exact (block_of_erased v v' tag fields Hv).
    + intros c. exact (boxed_number_of_erased v v' c Hv).
    + intros bytes. exact (string_of_erased v v' bytes Hv).
    + intros Hclosure.
      destruct (closure_of (live_statics rs) v) as [[called cslots] | ]
        eqn:Hc;
        [ | congruence ].
      destruct (closure_of_erased v v' called cslots Hv Hc)
        as (cslots' & -> & _).
      discriminate.
    + pose proof (partial_application_erased rs v v' Hv) as Hpartial.
      destruct (partial_application_of v) as [[[closure applied] arity] | ];
        [ | congruence ].
      destruct Hpartial as (closure' & applied' & -> & _). discriminate.
    + erased_cases Hv; simpl; try reflexivity.
      destruct (lookup s (static_values (live_statics rs)))
        as [allocated | ] eqn:Hs;
        [ | discriminate ].
      destruct (Hsymbols s allocated Hnot_drawn Hs) as (allocated' & -> & _).
      reflexivity.
    + intros [s0 c] Hs0.
      apply imported_of_symbol in Hs0 as [-> Hs0].
      inversion Hv; subst; simpl; try congruence.
      * destruct (Hsymbols s0 _ H0 Hs0) as (allocated' & -> & Hallocated).
        inversion Hallocated. reflexivity.
      * match goal with
        | Hb : static_set_bound _ ?syms _ _, Hl : lookup ?fs ?syms = Some s0
          |- _ =>
          rewrite (Hb fs s0 (lookup_in _ _ _ Hl)) in Hs0; discriminate
        end.
    + intros elements. exact (array_of_erased v v' elements Hv).
Qed.

Lemma eval_prim_erased (p : prim) (vs vs' : list value) (v : value) :
  Forall2 (value_erased rs) vs vs' ->
  eval_prim (live_statics rs) p vs = Some v ->
  ∃ v' : value, eval_prim statics' p vs' = Some v' ∧ value_erased rs v v'.
Proof.
  exact
    (eval_prim_related (value_erased rs) (live_statics rs) statics' p vs vs' v
       value_erased_respects).
Qed.

End Observations.

(* ======================================================================== *)
(* Evaluation in erased environments                                        *)
(* ======================================================================== *)

(* Every variable of the first environment whose value may be
   evaluated, or is a location, is bound in the second, to that value
   erased ([erased_asked]). Nothing is asked of the value of a variable
   that holds a mark or the symbol of a local allocation that is gone.
   And the second environment binds no variable that the first does
   not: the term with its regions dropped binds a variable only where
   the term binds it. *)
Definition env_erased (rs : region_state) (vars vars' : env) : Prop :=
  (∀ (x : variable) (v : value),
    lookup x vars = Some v ->
    erased_asked rs v ->
    ∃ v' : value, lookup x vars' = Some v' ∧ value_erased rs v v')
  ∧ ∀ x : variable, lookup x vars = None -> lookup x vars' = None.

Lemma env_erased_nil (rs : region_state) : env_erased rs [] [].
Proof.
  split; [ intros x v Hx; discriminate Hx | intros x _; reflexivity ].
Qed.

Lemma env_erased_extend (rs rs1 : region_state) (vars vars' : env) :
  region_extends rs rs1 ->
  env_erased rs vars vars' ->
  env_erased rs1 vars vars'.
Proof.
  intros Hextends [Hvars Hunbound]. split; [ | exact Hunbound ].
  intros x v Hx Hasked.
  destruct (Hvars x v Hx (erased_asked_extends rs rs1 v Hextends Hasked))
    as (v' & Hx' & Hv).
  exists v'. split; [ exact Hx' | ].
  exact (value_erased_extend _ _ Hextends _ _ Hv).
Qed.

Lemma env_erased_cons (rs : region_state) (x : variable) (v v' : value)
    (vars vars' : env) :
  (erased_asked rs v -> value_erased rs v v') ->
  env_erased rs vars vars' ->
  env_erased rs ((x, v) :: vars) ((x, v') :: vars').
Proof.
  intros Hv [Hvars Hunbound]. split.
  - intros y w. simpl.
    destruct (Pos.eqb y x); [ intros [= <-] Husable; eauto | apply Hvars ].
  - intros y. simpl.
    destruct (Pos.eqb y x); [ discriminate | apply Hunbound ].
Qed.

(* A variable bound in the first environment alone, to a value that may
   not be evaluated and is no location: the variable of a region, bound
   to its mark. *)
Lemma env_erased_cons_dead (rs : region_state) (x : variable) (v : value)
    (vars vars' : env) :
  value_usable rs v = false ->
  is_location v = false ->
  env_erased rs vars vars' ->
  env_erased rs ((x, v) :: vars) vars'.
Proof.
  intros Hdead Hnot_location [Hvars Hunbound]. split.
  - intros y w. simpl.
    destruct (Pos.eqb y x);
      [ intros [= <-] [Husable | Hlocation]; congruence | apply Hvars ].
  - intros y. simpl.
    destruct (Pos.eqb y x); [ discriminate | apply Hunbound ].
Qed.

Lemma bind_params_erased (rs : region_state) (params : list variable) :
  ∀ (vs vs' : list value) (vars vars' bound : env),
    Forall2 (value_erased_if_usable rs) vs vs' ->
    env_erased rs vars vars' ->
    bind_params params vs vars = Some bound ->
    ∃ bound' : env,
      bind_params params vs' vars' = Some bound'
      ∧ env_erased rs bound bound'.
Proof.
  induction params as [ | p params IH]; intros vs vs' vars vars' bound Hvs;
    destruct Hvs as [ | v v' vs vs' [_ Hv] Hvs]; simpl; try discriminate.
  - intros Hvars [= <-]. eauto.
  - intros Hvars. apply (IH vs vs' _ _ bound Hvs).
    apply env_erased_cons; assumption.
Qed.

(* The simples whose values may be evaluated. The test that the machine
   of regions makes before a step gives it of the simples that the
   step evaluates, but for an argument at the place of a region. *)
Definition simples_usable (rs : region_state) (vars : env)
    (simples : list simple) : Prop :=
  ∀ (s : simple) (v : value),
    s ∈ simples -> eval_simple vars s = Some v -> value_usable rs v = true.

Lemma simples_usable_of_test (rs : region_state) (simples : list simple) :
  forallb (simple_usable rs) simples = true ->
  simples_usable rs (state_vars (rs_state rs)) simples.
Proof.
  intros Hall s v Hin Hs. rewrite forallb_forall in Hall.
  specialize (Hall s Hin). unfold simple_usable in Hall.
  rewrite Hs in Hall. exact Hall.
Qed.

Lemma simples_usable_values (rs : region_state) (vars : env)
    (args : list simple) :
  simples_usable rs vars args ->
  ∀ vs : list value,
    eval_simples vars args = Some vs ->
    Forall (fun v => value_usable rs v = true) vs.
Proof.
  induction args as [ | s args IH]; intros Husable vs; simpl.
  - intros [= <-]. constructor.
  - destruct (eval_simple vars s) as [v | ] eqn:Hs; [ | discriminate ].
    destruct (eval_simples vars args) as [rest | ]; [ | discriminate ].
    intros [= <-]. constructor.
    + exact (Husable s v (or_introl eq_refl) Hs).
    + exact (IH (fun s0 v0 Hin => Husable s0 v0 (or_intror Hin)) rest eq_refl).
Qed.

(* A symbol below the base is none that the machine drew. *)
Lemma drawn_below (rs : region_state) (s : symbol) :
  symbol_below (rs_base rs) s = true -> drawn rs s = false.
Proof.
  unfold symbol_below, drawn. intros H. apply Pos.ltb_lt in H.
  apply Pos.leb_gt. exact H.
Qed.

Lemma eval_simple_erased (rs : region_state) (vars vars' : env) (s : simple)
    (v : value) :
  env_erased rs vars vars' ->
  simple_below (rs_base rs) s = true ->
  eval_simple vars s = Some v ->
  value_usable rs v = true ->
  ∃ v' : value, eval_simple vars' s = Some v' ∧ value_erased rs v v'.
Proof.
  intros Hvars Hbelow. destruct s as [x | sym | c]; simpl.
  - intros Hx Husable.
    exact (proj1 Hvars x v Hx (usable_asked rs v Husable)).
  - intros [= <-] _. eexists. split; [ reflexivity | ].
    apply Erased_symbol, drawn_below, Hbelow.
  - intros [= <-] _. eexists. split; [ reflexivity | apply Erased_const ].
Qed.

Lemma eval_simples_erased (rs : region_state) (vars vars' : env)
    (args : list simple) :
  env_erased rs vars vars' ->
  forallb (simple_below (rs_base rs)) args = true ->
  simples_usable rs vars args ->
  ∀ vs : list value,
    eval_simples vars args = Some vs ->
    ∃ vs' : list value,
      eval_simples vars' args = Some vs' ∧ Forall2 (value_erased rs) vs vs'.
Proof.
  intros Hvars. induction args as [ | s args IH]; intros Hbelow Husable vs;
    simpl.
  - intros [= <-]. exists []. auto.
  - simpl in Hbelow. apply andb_true_iff in Hbelow as [Hs_below Hbelow].
    destruct (eval_simple vars s) as [v | ] eqn:Hs; [ | discriminate ].
    destruct (eval_simples vars args) as [rest | ]; [ | discriminate ].
    intros [= <-].
    destruct
      (eval_simple_erased rs vars vars' s v Hvars Hs_below Hs
         (Husable s v (or_introl eq_refl) Hs))
      as (v' & -> & Hv).
    destruct
      (IH Hbelow (fun s0 v0 Hin => Husable s0 v0 (or_intror Hin)) rest
         eq_refl)
      as (rest' & -> & Hrest).
    exists (v' :: rest'). auto.
Qed.

Lemma eval_value_slots_erased (rs : region_state) (vars vars' : env)
    (slots : list (value_slot * simple)) :
  env_erased rs vars vars' ->
  forallb (simple_below (rs_base rs)) (map snd slots) = true ->
  simples_usable rs vars (map snd slots) ->
  ∀ evaluated : list (value_slot * value),
    eval_value_slots vars slots = Some evaluated ->
    ∃ evaluated' : list (value_slot * value),
      eval_value_slots vars' slots = Some evaluated'
      ∧ Forall2 (slot_erased (value_erased rs)) evaluated evaluated'.
Proof.
  intros Hvars.
  induction slots as [ | [slot s] slots IH]; intros Hbelow Husable evaluated;
    simpl.
  - intros [= <-]. exists []. auto.
  - simpl in Hbelow. apply andb_true_iff in Hbelow as [Hs_below Hbelow].
    destruct (eval_simple vars s) as [v | ] eqn:Hs; [ | discriminate ].
    destruct (eval_value_slots vars slots) as [rest | ]; [ | discriminate ].
    intros [= <-].
    destruct
      (eval_simple_erased rs vars vars' s v Hvars Hs_below Hs
         (Husable s v (or_introl eq_refl) Hs))
      as (v' & -> & Hv).
    destruct
      (IH Hbelow (fun s0 v0 Hin => Husable s0 v0 (or_intror Hin)) rest
         eq_refl)
      as (rest' & -> & Hrest).
    exists ((slot, v') :: rest'). split; [ reflexivity | ].
    constructor; [ split; [ reflexivity | exact Hv ] | exact Hrest ].
Qed.

(* A defining expression that has a value on the machine of regions,
   under the static environment that [step] is given there, has one in
   the second state: that value erased. *)
Lemma eval_named_erased (rs : region_state) (statics' : static_env)
    (vars vars' : env) (d : named) (v : value) :
  symbols_erased rs statics' ->
  env_erased rs vars vars' ->
  forallb (simple_below (rs_base rs)) (simples_of_named d) = true ->
  simples_usable rs vars (simples_of_named d) ->
  eval_named (live_statics rs) vars d = Some v ->
  ∃ v' : value, eval_named statics' vars' d = Some v' ∧ value_erased rs v v'.
Proof.
  intros Hstatics Hvars Hbelow Husable.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    simpl in Hbelow, Husable |- *.
  - apply andb_true_iff in Hbelow as [Hbelow _]. intros Hs.
    exact (eval_simple_erased rs vars vars' s v Hvars Hbelow Hs
             (Husable s v (or_introl eq_refl) Hs)).
  - destruct (eval_simples vars args) as [vs | ] eqn:Hargs;
      [ | discriminate ].
    destruct
      (eval_simples_erased rs vars vars' args Hvars Hbelow Husable vs Hargs)
      as (vs' & -> & Hvs).
    exact (eval_prim_erased rs statics' Hstatics p vs vs' v Hvs).
  - destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots;
      [ | discriminate ].
    destruct
      (eval_value_slots_erased rs vars vars' slots Hvars Hbelow Husable vs
         Hslots)
      as (vs' & -> & Hvs).
    intros [= <-]. eexists. split; [ reflexivity | ].
    apply Erased_closure, Hvs.
  - destruct (lookup fslot decls); [ | discriminate ]. simpl.
    destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots;
      [ | discriminate ].
    destruct
      (eval_value_slots_erased rs vars vars' slots Hvars Hbelow Husable vs
         Hslots)
      as (vs' & -> & Hvs).
    intros [= <-]. eexists. split; [ reflexivity | ].
    apply Erased_closure_of_set, Hvs.
  - apply andb_true_iff in Hbelow as [Hbelow _].
    destruct (eval_simple vars closure) as [w | ] eqn:Hclosure;
      [ | discriminate ].
    destruct
      (eval_simple_erased rs vars vars' closure w Hvars Hbelow Hclosure
         (Husable closure w (or_introl eq_refl) Hclosure))
      as (w' & -> & Hw).
    simpl.
    destruct (closure_of (live_statics rs) w) as [[called cslots] | ] eqn:Hc;
      [ | discriminate ].
    destruct (closure_of_erased rs statics' Hstatics w w' called cslots Hw Hc)
      as (cslots' & -> & Hcslots).
    simpl. exact (lookup_slots_erased rs slot cslots cslots' v Hcslots).
  - apply andb_true_iff in Hbelow as [Hbelow _].
    destruct (eval_simple vars closure) as [w | ] eqn:Hclosure;
      [ | discriminate ].
    destruct
      (eval_simple_erased rs vars vars' closure w Hvars Hbelow Hclosure
         (Husable closure w (or_introl eq_refl) Hclosure))
      as (w' & -> & Hw).
    exact (project_value_slot_erased rs statics' Hstatics pfrom slot w w' v
             Hw).
  - apply andb_true_iff in Hbelow as [Hbelow _].
    destruct (eval_simple vars closure) as [w | ] eqn:Hclosure;
      [ | discriminate ].
    destruct
      (eval_simple_erased rs vars vars' closure w Hvars Hbelow Hclosure
         (Husable closure w (or_introl eq_refl) Hclosure))
      as (w' & -> & Hw).
    exact (project_function_slot_erased rs statics' Hstatics move_from
             move_to w w' v Hw).
Qed.

(* ======================================================================== *)
(* The heaps                                                                *)
(* ======================================================================== *)

(* The heap of the machine of regions and the heap of the state that
   runs the erased term. They bind the same locations, entry for
   entry; and the cell of a location that is not dead
   (FlambdaRegionsSpec.cell_dead) is like the cell of the second heap
   at that location, with the values related. Of such a location only
   the entry that [lookup] finds is compared. Nothing is asked of what
   a dead cell holds, nor of an entry that a later write hides: a dead
   cell may hold the symbol of a local allocation that is gone, which
   is related to nothing.

   It is FlambdaHeapSpec.heap_alike_given of the locations that are
   not dead, and that is what the law of a primitive on two heaps
   (FlambdaHeapSpec.exec_prim_related_given_statement) asks of the
   cells that its arguments name: an argument that may be evaluated is
   no dead cell ([heaps_erased_given]). *)
Definition heaps_erased (rs : region_state) (h h' : heap) : Prop :=
  map fst h = map fst h'
  ∧ (∀ (l : location) (c : cell),
      cell_dead rs l = false ->
      lookup l h = Some c ->
      ∃ c' : cell, lookup l h' = Some c' ∧ cell_alike (value_erased rs) c c').

Lemma heaps_erased_empty (rs : region_state) :
  heaps_erased rs empty_heap empty_heap.
Proof. split; [ reflexivity | ]. intros l c _ Hl. discriminate Hl. Qed.

Lemma cell_alike_erased_extend (rs rs1 : region_state) (c c' : cell) :
  region_extends rs rs1 ->
  cell_alike (value_erased rs) c c' ->
  cell_alike (value_erased rs1) c c'.
Proof.
  intros Hextends.
  destruct c as [mut tag fields | kind elements],
    c' as [mut' tag' fields' | kind' elements'];
    simpl; try exact (fun H => H).
  - intros (Hmut & Htag & Hfields). split; [ exact Hmut | ].
    split; [ exact Htag | ].
    exact (values_erased_extend _ _ _ _ Hextends Hfields).
  - intros (Hkind & Helements). split; [ exact Hkind | ].
    exact (values_erased_extend _ _ _ _ Hextends Helements).
Qed.

Lemma heaps_erased_extend (rs rs1 : region_state) (h h' : heap) :
  region_extends rs rs1 -> heaps_erased rs h h' -> heaps_erased rs1 h h'.
Proof.
  intros Hextends [Hkeys Hcells]. split; [ exact Hkeys | ].
  intros l c Hdead Hl.
  rewrite (cell_dead_extends rs rs1 l Hextends) in Hdead.
  destruct (Hcells l c Hdead Hl) as (c' & Hl' & Halike).
  exists c'. split; [ exact Hl' | ].
  exact (cell_alike_erased_extend rs rs1 c c' Hextends Halike).
Qed.

(* The cells that values which may be evaluated name are alike. *)
Lemma heaps_erased_given (rs : region_state) (h h' : heap)
    (vs : list value) :
  heaps_erased rs h h' ->
  Forall (fun v => value_usable rs v = true) vs ->
  heap_alike_given (value_erased rs) h h' vs.
Proof.
  intros [Hkeys Hcells] Husable. split; [ exact Hkeys | ].
  intros l c Hin Hl. apply (Hcells l c); [ | exact Hl ].
  rewrite Forall_forall in Husable. specialize (Husable _ Hin).
  simpl in Husable. apply negb_true_iff in Husable. exact Husable.
Qed.

(* What one primitive has done to the two heaps
   (FlambdaHeapSpec.heap_alike_added) leaves them related: an entry
   that it adds is found first at its location, in both, and the two
   cells are alike. This does not ask which location that is. *)
Lemma heaps_erased_added (rs : region_state) (h h1 h' h1' : heap) :
  heaps_erased rs h h' ->
  heap_alike_added (value_erased rs) h h1 h' h1' ->
  heaps_erased rs h1 h1'.
Proof.
  intros [Hkeys Hcells] [[-> ->] | (l & c & c' & -> & -> & Halike)];
    [ split; assumption | ].
  split; [ simpl; rewrite Hkeys; reflexivity | ].
  intros l0 c0 Hdead. simpl.
  destruct (Pos.eqb l0 l).
  - intros [= <-]. exists c'. split; [ reflexivity | exact Halike ].
  - exact (Hcells l0 c0 Hdead).
Qed.

(* A location is related to itself, and to nothing else. *)
Lemma value_erased_relates_locations (rs : region_state) :
  relates_locations (value_erased rs).
Proof.
  split; [ intros l; apply Erased_location | ].
  split.
  - intros l v' H. inversion H. reflexivity.
  - intros v l H. inversion H. reflexivity.
Qed.

(* A defining expression that the machine of regions runs on its heap,
   under the static environment that [step] is given there, is run in
   the second state on a heap related, with the same reads: its value
   is the value erased, and it adds to each heap nothing, or an entry
   at one location with cells alike. Which location that is,
   FlambdaHeapSpec.exec_prim_adds_statement says of each heap. *)
Lemma exec_named_erased_added (rs : region_state) (statics' : static_env)
    (vars vars' : env) (d : named) (hp hp' hp1 : heap) (rd rd1 : read_record)
    (v : value) :
  symbols_erased rs statics' ->
  env_erased rs vars vars' ->
  forallb (simple_below (rs_base rs)) (simples_of_named d) = true ->
  simples_usable rs vars (simples_of_named d) ->
  heaps_erased rs hp hp' ->
  exec_named (live_statics rs) hp rd vars d = Some (v, hp1, rd1) ->
  ∃ (v' : value) (hp1' : heap),
    exec_named statics' hp' rd vars' d = Some (v', hp1', rd1)
    ∧ value_erased rs v v'
    ∧ heap_alike_added (value_erased rs) hp hp1 hp' hp1'.
Proof.
  intros Hstatics Hvars Hbelow Husable Hheaps Hexec.
  assert (Hpure :
    (∀ (p : prim) (args : list simple), d <> Prim p args) ->
    ∃ (v' : value) (hp1' : heap),
      exec_named statics' hp' rd vars' d = Some (v', hp1', rd1)
      ∧ value_erased rs v v'
      ∧ heap_alike_added (value_erased rs) hp hp1 hp' hp1').
  { intros Hnot.
    pose proof (exec_named_not_prim (live_statics rs) hp rd vars d Hnot)
      as Hrun.
    rewrite Hrun in Hexec.
    destruct (eval_named (live_statics rs) vars d) as [w | ] eqn:Hw;
      [ | discriminate Hexec ].
    injection Hexec as <- <- <-.
    destruct
      (eval_named_erased rs statics' vars vars' d w Hstatics Hvars Hbelow
         Husable Hw)
      as (w' & Hw' & Hrelated).
    exists w', hp'.
    rewrite (exec_named_not_prim statics' hp' rd vars' d Hnot), Hw'.
    split; [ reflexivity | ].
    split; [ exact Hrelated | left; split; reflexivity ]. }
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    try (apply Hpure; intros p0 args0; discriminate).
  clear Hpure.
  apply exec_named_prim_some in Hexec as (vs & Hargs & Hprim & ->).
  simpl in Hbelow, Husable.
  destruct
    (eval_simples_erased rs vars vars' args Hvars Hbelow Husable vs Hargs)
    as (vs' & Hargs' & Hvs).
  destruct
    (exec_prim_related_given (value_erased rs) (live_statics rs) statics' hp
       hp' hp1 p vs vs' v (value_erased_respects rs statics' Hstatics)
       (value_erased_relates_locations rs)
       (heaps_erased_given rs hp hp' vs Hheaps
          (simples_usable_values rs vars args Husable vs Hargs))
       Hvs Hprim)
    as (v' & hp1' & Hprim' & Hrelated & Hadded).
  exists v', hp1'. split; [ | split; [ exact Hrelated | exact Hadded ] ].
  apply exec_named_prim_some. exists vs'.
  split; [ exact Hargs' | ]. split; [ exact Hprim' | reflexivity ].
Qed.

(* The same, with the heaps after related. *)
Lemma exec_named_erased (rs : region_state) (statics' : static_env)
    (vars vars' : env) (d : named) (hp hp' hp1 : heap) (rd rd1 : read_record)
    (v : value) :
  symbols_erased rs statics' ->
  env_erased rs vars vars' ->
  forallb (simple_below (rs_base rs)) (simples_of_named d) = true ->
  simples_usable rs vars (simples_of_named d) ->
  heaps_erased rs hp hp' ->
  exec_named (live_statics rs) hp rd vars d = Some (v, hp1, rd1) ->
  ∃ (v' : value) (hp1' : heap),
    exec_named statics' hp' rd vars' d = Some (v', hp1', rd1)
    ∧ value_erased rs v v'
    ∧ heaps_erased rs hp1 hp1'.
Proof.
  intros Hstatics Hvars Hbelow Husable Hheaps Hexec.
  destruct
    (exec_named_erased_added rs statics' vars vars' d hp hp' hp1 rd rd1 v
       Hstatics Hvars Hbelow Husable Hheaps Hexec)
    as (v' & hp1' & Hexec' & Hrelated & Hadded).
  exists v', hp1'. split; [ exact Hexec' | ]. split; [ exact Hrelated | ].
  exact (heaps_erased_added rs hp hp1 hp' hp1' Hheaps Hadded).
Qed.

(* ======================================================================== *)
(* Continuations                                                            *)
(* ======================================================================== *)

(* A handler none of whose parameters is a region. *)
Definition handler_takes_no_region (h : continuation_handler) : bool :=
  negb (existsb (fun place => place) (region_params h)).

(* What a continuation stands for takes no region: it is not the
   closure of a handler that has a region parameter. A way out, a
   continuation with extra arguments and what is left of an
   over-application are never given a region: the machine of regions
   has no step for a mark given to one
   (FlambdaRegionsSpec.region_places). *)
Definition cont_takes_no_region (c : cont_value) : Prop :=
  match c with
  | Closure h _ _ _ => handler_takes_no_region h = true
  | Outer _ | Extra_args _ _ | Over_application _ _ _ _ _ => True
  end.

(* The list of continuations that [erase] is given agrees with what the
   continuations of the state stand for: a continuation that it lists
   is bound to the closure of a handler, and is listed with the places
   of the regions among the parameters of that handler; one that it
   does not list (the return or the exception continuation of code, a
   way out of the term) stands for something that takes no region. *)
Definition conts_agree (rconts : region_conts) (conts : cont_env) : Prop :=
  ∀ k : continuation,
    match lookup k rconts with
    | Some places =>
      ∃ (h : continuation_handler) (vars : env) (conts0 : cont_env)
        (recursive : list (continuation * continuation_handler)),
        lookup k conts = Some (Closure h vars conts0 recursive)
        ∧ places = region_params h
    | None => cont_takes_no_region (resolve k conts)
    end.

(* The continuations that a recursive [Let_cont] puts in scope, as
   [erase] lists them, and its handlers erased. *)
Definition region_conts_of
    (handlers : list (continuation * continuation_handler)) : region_conts :=
  map (fun '(k, h) => (k, region_params h)) handlers.

Definition erase_handlers (rconts : region_conts)
    (handlers : list (continuation * continuation_handler))
    : list (continuation * continuation_handler) :=
  map (fun '(k, h) => (k, erase_handler rconts h)) handlers.

(* A continuation of the first state and one of the second: the same
   way out; closures of a handler and of that handler erased, under the
   list of continuations of its definition with those of its recursive
   group in front, with related environments; the same continuation
   with extra arguments, or what is left of the same over-application,
   with the values kept related where they may be evaluated or are
   locations.

   The closure of a handler carries what a state carries for its
   expression: the handler, and those of its group, have no call that
   returns or raises to a region parameter, and the list agrees with
   the continuations of the definition. A continuation that is kept
   inside another, to be applied by the machine and not by an
   [Apply_cont] of a term, takes no region. *)
#[local] Set Warnings "-register-all".
Inductive cont_erased (rs : region_state) : cont_value -> cont_value -> Prop :=
  | Erased_outer (k : continuation) : cont_erased rs (Outer k) (Outer k)
  | Erased_handler (rconts : region_conts) (h : continuation_handler)
      (vars vars' : env) (conts conts' : cont_env)
      (recursive : list (continuation * continuation_handler)) :
    handler_returns_no_region (region_conts_of recursive ++ rconts) h = true ->
    forallb
      (fun '(_, h0) =>
         handler_returns_no_region (region_conts_of recursive ++ rconts) h0)
      recursive
    = true ->
    conts_agree rconts conts ->
    env_erased rs vars vars' ->
    Forall2
      (fun kc kc' => fst kc = fst kc' ∧ cont_erased rs (snd kc) (snd kc'))
      conts conts' ->
    cont_erased rs (Closure h vars conts recursive)
      (Closure (erase_handler (region_conts_of recursive ++ rconts) h) vars'
         conts'
         (erase_handlers (region_conts_of recursive ++ rconts) recursive))
  | Erased_extra_args (c c' : cont_value) (extra extra' : list value) :
    cont_takes_no_region c ->
    cont_erased rs c c' ->
    Forall2 (value_erased_if_usable rs) extra extra' ->
    cont_erased rs (Extra_args c extra) (Extra_args c' extra')
  | Erased_over_application (remaining remaining' : list value)
      (remaining_arity return_arity : arity)
      (return_to return_to' exn_to exn_to' : cont_value) :
    Forall2 (value_erased_if_usable rs) remaining remaining' ->
    cont_takes_no_region return_to ->
    cont_takes_no_region exn_to ->
    cont_erased rs return_to return_to' ->
    cont_erased rs exn_to exn_to' ->
    cont_erased rs
      (Over_application remaining remaining_arity return_arity return_to
         exn_to)
      (Over_application remaining' remaining_arity return_arity return_to'
         exn_to').

Definition conts_erased (rs : region_state) (conts conts' : cont_env) : Prop :=
  Forall2
    (fun kc kc' => fst kc = fst kc' ∧ cont_erased rs (snd kc) (snd kc'))
    conts conts'.

Fixpoint cont_erased_extend (rs rs1 : region_state)
    (Hextends : region_extends rs rs1) (c c' : cont_value)
    (H : cont_erased rs c c') {struct H} :
  cont_erased rs1 c c'.
Proof.
  destruct H
    as [ k
       | rconts h vars vars' conts conts' recursive Hh Hrecursive Hagree
           Hvars Hconts
       | c0 c0' extra extra' Htakes Hc0 Hextra
       | remaining remaining' remaining_arity return_arity return_to
           return_to' exn_to exn_to' Hremaining Hreturn_takes Hexn_takes
           Hreturn Hexn ].
  - apply Erased_outer.
  - apply Erased_handler;
      [ exact Hh | exact Hrecursive | exact Hagree
      | exact (env_erased_extend _ _ _ _ Hextends Hvars) | ].
    clear Hvars Hagree. revert conts conts' Hconts. fix IH 3.
    intros conts conts' [ | kc kc' rest rest' [Hkey Hhead] Htail];
      constructor;
      [ split;
        [ exact Hkey | exact (cont_erased_extend _ _ Hextends _ _ Hhead) ]
      | exact (IH _ _ Htail) ].
  - apply Erased_extra_args;
      [ exact Htakes | exact (cont_erased_extend _ _ Hextends _ _ Hc0)
      | exact (values_erased_if_usable_extend _ _ _ _ Hextends Hextra) ].
  - apply Erased_over_application;
      [ exact (values_erased_if_usable_extend _ _ _ _ Hextends Hremaining)
      | exact Hreturn_takes | exact Hexn_takes
      | exact (cont_erased_extend _ _ Hextends _ _ Hreturn)
      | exact (cont_erased_extend _ _ Hextends _ _ Hexn) ].
Qed.

Lemma conts_erased_extend (rs rs1 : region_state) (conts conts' : cont_env) :
  region_extends rs rs1 ->
  conts_erased rs conts conts' ->
  conts_erased rs1 conts conts'.
Proof.
  intros Hextends.
  induction 1 as [ | kc kc' conts conts' [Hkey Hc] _ IH]; constructor;
    [ split; [ exact Hkey | exact (cont_erased_extend _ _ Hextends _ _ Hc) ]
    | exact IH ].
Qed.

Lemma resolve_erased (rs : region_state) (k : continuation)
    (conts conts' : cont_env) :
  conts_erased rs conts conts' ->
  cont_erased rs (resolve k conts) (resolve k conts').
Proof.
  unfold resolve.
  induction 1 as [ | [k1 c] [k1' c'] conts conts' [Hk Hc] _ IH]; simpl;
    [ apply Erased_outer | ].
  simpl in Hk. subst k1'. destruct (Pos.eqb k k1); [ exact Hc | exact IH ].
Qed.

Lemma with_extra_args_erased (rs : region_state) (c c' : cont_value)
    (extra extra' : list value) :
  cont_takes_no_region c ->
  cont_erased rs c c' ->
  Forall2 (value_erased_if_usable rs) extra extra' ->
  cont_erased rs (with_extra_args c extra) (with_extra_args c' extra').
Proof.
  intros Htakes Hc Hextra.
  destruct Hextra as [ | v v' extra extra' Hv Hextra]; [ exact Hc | ].
  apply Erased_extra_args; [ exact Htakes | exact Hc | ].
  constructor; assumption.
Qed.

Lemma with_extra_args_takes_no_region (c : cont_value) (extra : list value) :
  cont_takes_no_region c -> cont_takes_no_region (with_extra_args c extra).
Proof. intros Hc. destruct extra; [ exact Hc | exact I ]. Qed.

(* The closures of a recursive group. [all] is the whole group, kept
   apart from the handlers still to be gone through. *)
Lemma rec_closures_erased_aux (rs : region_state) (rconts : region_conts)
    (vars vars' : env) (conts conts' : cont_env)
    (all handlers : list (continuation * continuation_handler)) :
  forallb
    (fun '(_, h) =>
       handler_returns_no_region (region_conts_of all ++ rconts) h)
    all
  = true ->
  (∀ kh : continuation * continuation_handler, kh ∈ handlers -> kh ∈ all) ->
  conts_agree rconts conts ->
  env_erased rs vars vars' ->
  conts_erased rs conts conts' ->
  conts_erased rs (map (fun '(k, h) => (k, Closure h vars conts all)) handlers)
    (map
       (fun '(k, h) =>
          (k,
           Closure h vars' conts'
             (erase_handlers (region_conts_of all ++ rconts) all)))
       (erase_handlers (region_conts_of all ++ rconts) handlers)).
Proof.
  intros Hall Hsub Hagree Hvars Hconts.
  induction handlers as [ | [k h] handlers IH]; simpl; constructor.
  - split; [ reflexivity | ]. simpl.
    apply Erased_handler; try assumption.
    rewrite forallb_forall in Hall.
    exact (Hall (k, h) (Hsub (k, h) (or_introl eq_refl))).
  - apply IH. intros kh Hin. apply Hsub. right. exact Hin.
Qed.

Lemma rec_closures_erased (rs : region_state) (rconts : region_conts)
    (vars vars' : env) (conts conts' : cont_env)
    (handlers : list (continuation * continuation_handler)) :
  forallb
    (fun '(_, h) =>
       handler_returns_no_region (region_conts_of handlers ++ rconts) h)
    handlers
  = true ->
  conts_agree rconts conts ->
  env_erased rs vars vars' ->
  conts_erased rs conts conts' ->
  conts_erased rs (rec_closures handlers vars conts ++ conts)
    (rec_closures (erase_handlers (region_conts_of handlers ++ rconts) handlers)
       vars' conts'
     ++ conts').
Proof.
  intros Hall Hagree Hvars Hconts. apply Forall2_app; [ | exact Hconts ].
  apply rec_closures_erased_aux; auto.
Qed.

(* A continuation that [erase] takes to have no region parameter stands
   for something that takes no region. *)
Lemma takes_no_region_resolve (rconts : region_conts) (conts : cont_env)
    (k : continuation) :
  conts_agree rconts conts ->
  takes_no_region rconts k = true ->
  cont_takes_no_region (resolve k conts).
Proof.
  intros Hagree. specialize (Hagree k). unfold takes_no_region.
  destruct (lookup k rconts) as [places | ]; [ | intros _; exact Hagree ].
  destruct Hagree as (h & vars & conts0 & recursive & Hk & ->).
  unfold resolve. rewrite Hk. exact (fun H => H).
Qed.

(* The list agrees with the continuations after a [Let_cont]. *)
Lemma conts_agree_cons (rconts : region_conts) (conts : cont_env)
    (k : continuation) (h : continuation_handler) (vars : env)
    (conts0 : cont_env)
    (recursive : list (continuation * continuation_handler)) :
  conts_agree rconts conts ->
  conts_agree ((k, region_params h) :: rconts)
    ((k, Closure h vars conts0 recursive) :: conts).
Proof.
  intros Hagree k0. specialize (Hagree k0). unfold resolve in *. simpl.
  destruct (Pos.eqb k0 k); [ | exact Hagree ].
  exists h, vars, conts0, recursive. split; reflexivity.
Qed.

Lemma conts_agree_rec (rconts : region_conts) (conts : cont_env)
    (vars : env) (conts0 : cont_env)
    (all handlers : list (continuation * continuation_handler)) :
  conts_agree rconts conts ->
  conts_agree (region_conts_of handlers ++ rconts)
    (map (fun '(k, h) => (k, Closure h vars conts0 all)) handlers ++ conts).
Proof.
  intros Hagree.
  induction handlers as [ | [k h] handlers IH]; [ exact Hagree | ].
  exact (conts_agree_cons _ _ k h vars conts0 all IH).
Qed.

(* And in the body of code: nothing is listed, and the return and
   exception continuations stand for what the caller's do. *)
Lemma conts_agree_call (ret exn : continuation)
    (return_to exn_to : cont_value) :
  cont_takes_no_region return_to ->
  cont_takes_no_region exn_to ->
  conts_agree [] [(ret, return_to); (exn, exn_to)].
Proof.
  intros Hreturn Hexn k. unfold resolve. simpl.
  destruct (Pos.eqb k ret); [ exact Hreturn | ].
  destruct (Pos.eqb k exn); [ exact Hexn | exact I ].
Qed.

(* A list with no region drops nothing. *)
Lemma drop_regions_none {A : Type} (places : list bool) :
  existsb (fun place => place) places = false ->
  ∀ l : list A, drop_regions places l = l.
Proof.
  induction places as [ | place places IH]; intros Hnone l;
    [ reflexivity | ].
  simpl in Hnone. apply orb_false_iff in Hnone as [-> Hnone].
  destruct l as [ | a l]; [ reflexivity | ]. simpl.
  rewrite (IH Hnone l). reflexivity.
Qed.

(* A handler that has no region parameter is erased to itself with its
   body erased. *)
Lemma erase_handler_no_region (rconts : region_conts)
    (params : list variable) (metadata : handler_metadata) (handler : expr) :
  handler_takes_no_region (Handler params metadata handler) = true ->
  erase_handler rconts (Handler params metadata handler)
  = Handler params metadata (erase rconts handler).
Proof.
  unfold handler_takes_no_region, region_params. simpl. intros Hnone.
  apply negb_true_iff in Hnone.
  rewrite !(drop_regions_none _ Hnone). destruct metadata. reflexivity.
Qed.

(* An [Apply_cont] of a continuation that takes no region is erased to
   itself. *)
Lemma erase_apply_cont_no_region (rconts : region_conts) (conts : cont_env)
    (ac : apply_cont) :
  conts_agree rconts conts ->
  cont_takes_no_region (resolve (ac_continuation ac) conts) ->
  erase_apply_cont rconts ac = ac.
Proof.
  intros Hagree Htakes. unfold erase_apply_cont.
  specialize (Hagree (ac_continuation ac)).
  destruct (lookup (ac_continuation ac) rconts) as [places | ];
    [ | reflexivity ].
  destruct Hagree as (h & vars & conts0 & recursive & Hk & ->).
  unfold resolve in Htakes. rewrite Hk in Htakes. simpl in Htakes.
  unfold handler_takes_no_region in Htakes. apply negb_true_iff in Htakes.
  rewrite (drop_regions_none _ Htakes). destruct ac. reflexivity.
Qed.

(* ======================================================================== *)
(* States                                                                   *)
(* ======================================================================== *)

(* A state of the machine of regions and a state of the machine of
   FlambdaSemanticsSpec.v: the expression of the second is [erase] of
   that of the first, under a list of continuations that agrees with
   those of the first state; the first expression has no call that
   returns or raises to a region parameter; the environments, the
   continuations, the static environments and the heaps are related;
   the reads are the same. *)
Definition state_erased (rs : region_state) (s' : state) : Prop :=
  ∃ rconts : region_conts,
    state_expr s' = erase rconts (state_expr (rs_state rs))
    ∧ calls_return_no_region rconts (state_expr (rs_state rs)) = true
    ∧ conts_agree rconts (state_conts (rs_state rs))
    ∧ env_erased rs (state_vars (rs_state rs)) (state_vars s')
    ∧ conts_erased rs (state_conts (rs_state rs)) (state_conts s')
    ∧ statics_erased rs (state_statics s')
    ∧ heaps_erased rs (state_heap (rs_state rs)) (state_heap s')
    ∧ state_reads s' = state_reads (rs_state rs).

(* Two states of the machine of regions with the same base, the same
   static environment, the same symbols bound and the same dead cells.
   They relate the same values, the same static environments and the
   same heaps. *)
Definition same_statics (rs rs1 : region_state) : Prop :=
  rs_base rs1 = rs_base rs
  ∧ state_statics (rs_state rs1) = state_statics (rs_state rs)
  ∧ static_values (live_statics rs1) = static_values (live_statics rs)
  ∧ (∀ l : location, cell_dead rs1 l = cell_dead rs l).

Lemma same_statics_extends (rs rs1 : region_state) :
  same_statics rs rs1 -> region_extends rs rs1.
Proof.
  intros (Hbase & _ & Hvalues & Hdead).
  exact (region_extends_same rs rs1 Hbase Hvalues Hdead).
Qed.

Lemma statics_erased_same (rs rs1 : region_state) (statics' : static_env) :
  same_statics rs rs1 ->
  statics_erased rs statics' ->
  statics_erased rs1 statics'.
Proof.
  intros Hsame (Hcode & Hreturns & Hsymbols & Hbound).
  pose proof (same_statics_extends rs rs1 Hsame) as Hextends.
  destruct Hsame as (Hbase & Hstatics & Hvalues & _).
  unfold statics_erased. rewrite Hstatics.
  split; [ exact Hcode | ]. split; [ exact Hreturns | ].
  split; [ | exact Hbound ].
  intros s allocated Hnot_drawn Hs.
  rewrite (drawn_extends rs rs1 s Hextends) in Hnot_drawn.
  rewrite Hvalues in Hs.
  destruct (Hsymbols s allocated Hnot_drawn Hs)
    as (allocated' & Hs' & Hallocated).
  exists allocated'. split; [ exact Hs' | ].
  exact (allocated_erased_extend _ _ _ _ Hextends Hallocated).
Qed.

(* What [live_statics] is made of. *)
Lemma live_statics_code (rs : region_state) :
  static_code (live_statics rs) = static_code (state_statics (rs_state rs)).
Proof. Transparent live_statics. reflexivity. Opaque live_statics. Qed.

Lemma live_statics_values (rs : region_state) :
  static_values (live_statics rs)
  = live_locals (rs_frames rs)
    ++ static_values (state_statics (rs_state rs)).
Proof. Transparent live_statics. reflexivity. Opaque live_statics. Qed.

(* The same base, the same frames, the same list of cells and the same
   static environment. *)
Lemma same_statics_intro (rs rs1 : region_state) :
  rs_base rs1 = rs_base rs ->
  rs_frames rs1 = rs_frames rs ->
  rs_cells rs1 = rs_cells rs ->
  state_statics (rs_state rs1) = state_statics (rs_state rs) ->
  same_statics rs rs1.
Proof.
  intros Hbase Hframes Hcells Hstatics. split; [ exact Hbase | ].
  split; [ exact Hstatics | ].
  split; [ rewrite !live_statics_values, Hframes, Hstatics; reflexivity | ].
  intros l. unfold cell_dead. rewrite Hcells, Hframes. reflexivity.
Qed.

(* The state after a step that [step] makes under the static
   environment of the frames, which is not a [Let_static]
   (FlambdaRegionsSpec.plain_step); and a state with other counts. *)
Lemma same_statics_plain (rs : region_state) (s1 : state) (next : symbol)
    (counts : region_counts) :
  same_statics rs
    (Region_state (with_statics s1 (state_statics (rs_state rs)))
       (rs_frames rs) (rs_base rs) next counts (rs_cells rs)).
Proof. apply same_statics_intro; reflexivity. Qed.

Lemma same_statics_with_counts (rs : region_state) (counts : region_counts) :
  same_statics rs (with_counts rs counts).
Proof. apply same_statics_intro; reflexivity. Qed.

(* The relation does not look at the frames, the counter, the counts or
   the list of cells but through the symbols that the frames bind and
   the cells that are dead. *)
Lemma state_erased_same (rs rs1 : region_state) (s' : state) :
  same_statics rs rs1 ->
  rs_state rs1 = rs_state rs ->
  state_erased rs s' ->
  state_erased rs1 s'.
Proof.
  intros Hsame Hstate
    (rconts & He & Hreturns & Hagree & Hvars & Hconts & Hstatics & Hheap
     & Hreads).
  pose proof (same_statics_extends rs rs1 Hsame) as Hextends.
  exists rconts. rewrite Hstate.
  split; [ exact He | ]. split; [ exact Hreturns | ].
  split; [ exact Hagree | ].
  split; [ exact (env_erased_extend _ _ _ _ Hextends Hvars) | ].
  split; [ exact (conts_erased_extend _ _ _ _ Hextends Hconts) | ].
  split; [ exact (statics_erased_same _ _ _ Hsame Hstatics) | ].
  split; [ exact (heaps_erased_extend _ _ _ _ Hextends Hheap) | exact Hreads ].
Qed.

(* ======================================================================== *)
(* The state that a run starts in                                           *)
(* ======================================================================== *)

(* A value none of whose symbols the machine drew is erased to itself. *)
Fixpoint value_erased_refl (rs : region_state) (v : value) {struct v} :
  value_below (rs_base rs) v = true -> value_erased rs v v.
Proof.
  destruct v
    as [ c | code slots | fslot decls slots | s | tag fields | elements | c | l
       | callee args args_arity ];
    simpl; intros Hbelow.
  - apply Erased_const.
  - apply Erased_closure. revert slots Hbelow. fix IH 1.
    intros [ | [slot w] slots] Hbelow; constructor; simpl in Hbelow;
      apply andb_true_iff in Hbelow as [Hw Hrest];
      [ split; [ reflexivity | exact (value_erased_refl rs w Hw) ]
      | exact (IH slots Hrest) ].
  - apply Erased_closure_of_set. revert slots Hbelow. fix IH 1.
    intros [ | [slot w] slots] Hbelow; constructor; simpl in Hbelow;
      apply andb_true_iff in Hbelow as [Hw Hrest];
      [ split; [ reflexivity | exact (value_erased_refl rs w Hw) ]
      | exact (IH slots Hrest) ].
  - apply Erased_symbol, drawn_below, Hbelow.
  - apply Erased_block. revert fields Hbelow. fix IH 1.
    intros [ | w fields] Hbelow; constructor; simpl in Hbelow;
      apply andb_true_iff in Hbelow as [Hw Hrest];
      [ exact (value_erased_refl rs w Hw) | exact (IH fields Hrest) ].
  - apply Erased_array. revert elements Hbelow. fix IH 1.
    intros [ | w elements] Hbelow; constructor; simpl in Hbelow;
      apply andb_true_iff in Hbelow as [Hw Hrest];
      [ exact (value_erased_refl rs w Hw) | exact (IH elements Hrest) ].
  - apply Erased_boxed_number.
  - apply Erased_location.
  - apply andb_true_iff in Hbelow as [Hcallee Hargs].
    apply Erased_partial_application;
      [ intros _; exact (value_erased_refl rs callee Hcallee) | ].
    revert args Hargs. fix IH 1.
    intros [ | w args] Hargs; constructor; simpl in Hargs;
      apply andb_true_iff in Hargs as [Hw Hrest];
      [ split; [ reflexivity | intros _; exact (value_erased_refl rs w Hw) ]
      | exact (IH args Hrest) ].
Qed.

Lemma values_erased_refl (rs : region_state) (vs : list value) :
  forallb (value_below (rs_base rs)) vs = true ->
  Forall2 (value_erased rs) vs vs.
Proof.
  induction vs as [ | v vs IH]; simpl; intros Hbelow; constructor;
    apply andb_true_iff in Hbelow as [Hv Hrest];
    [ exact (value_erased_refl rs v Hv) | exact (IH Hrest) ].
Qed.

Lemma slots_erased_refl (rs : region_state)
    (slots : list (value_slot * value)) :
  forallb (fun '(_, v) => value_below (rs_base rs) v) slots = true ->
  Forall2 (slot_erased (value_erased rs)) slots slots.
Proof.
  induction slots as [ | [slot v] slots IH]; simpl; intros Hbelow;
    constructor; apply andb_true_iff in Hbelow as [Hv Hrest];
    [ split; [ reflexivity | exact (value_erased_refl rs v Hv) ]
    | exact (IH Hrest) ].
Qed.

Lemma allocated_erased_refl (rs : region_state)
    (allocated : statically_allocated_value) :
  allocated_below (rs_base rs) allocated = true ->
  allocated_erased rs allocated allocated.
Proof.
  destruct allocated
    as [ code slots | fslot syms decls slots | tag fields | contents | bytes
       | elements | c ];
    simpl; intros Hbelow; constructor;
    auto using slots_erased_refl, values_erased_refl, value_erased_refl.
  - apply andb_true_iff in Hbelow as [Hsyms _].
    intros slot s Hin. rewrite forallb_forall in Hsyms.
    exact (drawn_below rs s (Hsyms (slot, s) Hin)).
  - apply andb_true_iff in Hbelow as [_ Hslots].
    exact (slots_erased_refl rs slots Hslots).
Qed.

Lemma lookup_erased_code (id : code_id) (code : list (code_id * code)) :
  lookup id (map (fun '(id0, c) => (id0, erase_code c)) code)
  = option_map erase_code (lookup id code).
Proof.
  induction code as [ | [id0 c] code IH]; simpl; [ reflexivity | ].
  destruct (Pos.eqb id id0); [ reflexivity | exact IH ].
Qed.

(* The state that the test tool starts the machine of regions in, and
   the state in which the erased term is run. *)
Lemma state_erased_initial (e : expr) (statics : static_env) (base : symbol) :
  region_names_fresh base e statics = true ->
  calls_return_no_region [] e = true ->
  statics_return_no_region statics = true ->
  state_erased (initial_region_state e statics base)
    (erased_initial_state e statics).
Proof.
  intros Hfresh Hreturns Hstatics.
  apply andb_true_iff in Hfresh as [Hfresh _].
  apply andb_true_iff in Hfresh as [_ Hbelow].
  apply andb_true_iff in Hbelow as [_ Hvalues].
  exists []. split; [ reflexivity | ]. split; [ exact Hreturns | ].
  split; [ intros k; exact I | ].
  split; [ apply env_erased_nil | ].
  split; [ constructor | ].
  split; [ | split; [ apply heaps_erased_empty | reflexivity ] ].
  split; [ intros id; apply lookup_erased_code | ].
  split.
  { intros id c Hid. unfold statics_return_no_region in Hstatics.
    rewrite forallb_forall in Hstatics.
    exact (Hstatics (id, c) (lookup_in _ _ _ Hid)). }
  split.
  - intros s allocated _ Hs.
    rewrite
      (live_statics_no_frames (initial_region_state e statics base) eq_refl)
      in Hs.
    simpl in Hs.
    exists allocated. split; [ exact Hs | ].
    rewrite forallb_forall in Hvalues.
    pose proof (Hvalues (s, allocated) (lookup_in _ _ _ Hs)) as Hallocated.
    apply andb_true_iff in Hallocated as [_ Hallocated].
    exact
      (allocated_erased_refl (initial_region_state e statics base) allocated
         Hallocated).
  - intros s allocated' Hs. simpl in Hs |- *. rewrite Hs. discriminate.
Qed.
