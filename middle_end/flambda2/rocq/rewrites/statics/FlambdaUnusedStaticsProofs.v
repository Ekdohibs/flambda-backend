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

(* Proofs of the statements of FlambdaUnusedStaticsSpec.v. Nothing here
   needs review.

   A state and the same state with the bindings erased step alike, as long
   as the first is defined (FlambdaSemanticsProofs.simulation_refines).
   The relation ([state_erased]) says that the expression, the values, the
   continuation closures and the static environment of the first refer to
   none of the names removed; so the values of the two are the same, and
   nothing the first looks up has gone from the second. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaNormaliseSpec FlambdaUnusedStaticsSpec.
Import ListNotations.

Section Erasure.

Variable D : dropped.

(* ======================================================================== *)
(* Values that refer to no name removed                                     *)
(* ======================================================================== *)

Definition env_avoids (vars : env) : Prop :=
  ∀ (x : variable) (v : value),
    lookup x vars = Some v -> value_avoids D v = true.

Definition slot_values_avoid (slots : list (value_slot * value)) : bool :=
  forallb (fun '(_, v) => value_avoids D v) slots.

Lemma env_avoids_cons (vars : env) (x : variable) (v : value) :
  value_avoids D v = true -> env_avoids vars -> env_avoids ((x, v) :: vars).
Proof.
  intros Hv Hvars y w. simpl.
  destruct (Pos.eqb y x); [ intros [= <-]; exact Hv | apply Hvars ].
Qed.

Lemma bind_params_avoids (params : list variable) :
  ∀ (vs : list value) (vars bound : env),
    forallb (value_avoids D) vs = true ->
    env_avoids vars ->
    bind_params params vs vars = Some bound ->
    env_avoids bound.
Proof.
  induction params as [ | p params IH]; intros [ | v vs] vars bound; simpl;
    try discriminate.
  - intros _ Hvars [= <-]. exact Hvars.
  - intros Hvs Hvars. apply andb_true_iff in Hvs as [Hv Hvs].
    apply (IH vs _ bound Hvs), env_avoids_cons; assumption.
Qed.

Lemma eval_simple_avoids (vars : env) (s : simple) (v : value) :
  env_avoids vars ->
  simple_avoids D s = true ->
  eval_simple vars s = Some v ->
  value_avoids D v = true.
Proof.
  intros Hvars. destruct s as [x | sym | c]; simpl.
  - intros _. apply Hvars.
  - intros Hkept [= <-]. exact Hkept.
  - intros _ [= <-]. reflexivity.
Qed.

Lemma eval_simples_avoid (vars : env) (args : list simple) :
  env_avoids vars ->
  ∀ vs : list value,
    simples_avoid D args = true ->
    eval_simples vars args = Some vs ->
    forallb (value_avoids D) vs = true.
Proof.
  intros Hvars. unfold simples_avoid.
  induction args as [ | s args IH]; intros vs; simpl.
  - intros _ [= <-]. reflexivity.
  - intros Hav. apply andb_true_iff in Hav as [Hs Hargs].
    destruct (eval_simple vars s) as [v | ] eqn:Heval; [ | discriminate ].
    destruct (eval_simples vars args) as [vs' | ]; [ | discriminate ].
    simpl. intros [= <-]. simpl.
    rewrite (eval_simple_avoids vars s v Hvars Hs Heval), (IH vs' Hargs);
      reflexivity.
Qed.

Lemma eval_value_slots_avoid (vars : env)
    (slots : list (value_slot * simple)) :
  env_avoids vars ->
  ∀ evaluated : list (value_slot * value),
    value_slots_avoid D slots = true ->
    eval_value_slots vars slots = Some evaluated ->
    slot_values_avoid evaluated = true.
Proof.
  intros Hvars. unfold value_slots_avoid, slot_values_avoid.
  induction slots as [ | [slot s] slots IH]; intros evaluated; simpl.
  - intros _ [= <-]. reflexivity.
  - intros Hav. apply andb_true_iff in Hav as [Hs Hslots].
    destruct (eval_simple vars s) as [v | ] eqn:Heval; [ | discriminate ].
    destruct (eval_value_slots vars slots) as [rest | ]; [ | discriminate ].
    simpl. intros [= <-]. simpl.
    rewrite (eval_simple_avoids vars s v Hvars Hs Heval), (IH rest Hslots);
      reflexivity.
Qed.

Lemma lookup_slot_avoids (slot : value_slot)
    (slots : list (value_slot * value)) (v : value) :
  slot_values_avoid slots = true ->
  lookup slot slots = Some v ->
  value_avoids D v = true.
Proof.
  unfold slot_values_avoid.
  induction slots as [ | [slot' w] slots IH]; simpl; [ discriminate | ].
  intros Hav. apply andb_true_iff in Hav as [Hw Hslots].
  destruct (Pos.eqb slot slot');
    [ intros [= <-]; exact Hw | exact (IH Hslots) ].
Qed.

(* ======================================================================== *)
(* Static environments                                                      *)
(* ======================================================================== *)

Definition allocated_avoids (allocated : statically_allocated_value) : bool :=
  match allocated with
  | Static_closure code slots => code_kept D code && slot_values_avoid slots
  | Static_closure_of_set _ symbols decls slots =>
    closure_symbols_avoid D symbols && function_decls_avoid D decls
    && slot_values_avoid slots
  | Static_block _ fields => forallb (value_avoids D) fields
  | Static_immutable_array elements => forallb (value_avoids D) elements
  | Static_boxed_number contents => value_avoids D contents
  | Static_string _ | Static_imported _ => true
  end.

(* The second static environment is the first without the names removed
   and with its code erased; what the first keeps refers to none of the
   names removed. *)
Definition statics_erased (statics statics' : static_env) : Prop :=
  (∀ id : code_id,
    code_kept D id = true ->
    lookup id (static_code statics')
    = option_map (erase_code D) (lookup id (static_code statics)))
  ∧ (∀ (id : code_id) (c : code),
      code_kept D id = true ->
      lookup id (static_code statics) = Some c ->
      code_avoids D c = true)
  ∧ (∀ s : symbol,
      symbol_kept D s = true ->
      lookup s (static_values statics') = lookup s (static_values statics))
  ∧ (∀ (s : symbol) (allocated : statically_allocated_value),
      symbol_kept D s = true ->
      lookup s (static_values statics) = Some allocated ->
      allocated_avoids allocated = true).

Lemma statics_erased_add_code (statics statics' : static_env) (id : code_id)
    (c : code) :
  statics_erased statics statics' ->
  code_avoids D c = true ->
  statics_erased
    (Static_env ((id, c) :: static_code statics) (static_values statics))
    (Static_env ((id, erase_code D c) :: static_code statics')
       (static_values statics')).
Proof.
  intros (Hcode & Hcode_avoids & Hvalues & Hvalues_avoid) Hc.
  split; [ | split; [ | split ] ]; simpl.
  - intros id' Hkept. destruct (Pos.eqb id' id); [ reflexivity | ].
    apply Hcode, Hkept.
  - intros id' c' Hkept. destruct (Pos.eqb id' id);
      [ intros [= <-]; exact Hc | apply Hcode_avoids, Hkept ].
  - exact Hvalues.
  - exact Hvalues_avoid.
Qed.

Lemma statics_erased_drop_code (statics statics' : static_env) (id : code_id)
    (c : code) :
  statics_erased statics statics' ->
  code_kept D id = false ->
  statics_erased
    (Static_env ((id, c) :: static_code statics) (static_values statics))
    statics'.
Proof.
  intros (Hcode & Hcode_avoids & Hvalues & Hvalues_avoid) Hdropped.
  split; [ | split; [ | split ] ]; simpl.
  - intros id' Hkept.
    destruct (Pos.eqb_spec id' id) as [-> | _]; [ congruence | ].
    apply Hcode, Hkept.
  - intros id' c' Hkept.
    destruct (Pos.eqb_spec id' id) as [-> | _]; [ congruence | ].
    apply Hcode_avoids, Hkept.
  - exact Hvalues.
  - exact Hvalues_avoid.
Qed.

Lemma statics_erased_add_value (statics statics' : static_env) (s : symbol)
    (allocated : statically_allocated_value) :
  statics_erased statics statics' ->
  allocated_avoids allocated = true ->
  statics_erased
    (Static_env (static_code statics)
       ((s, allocated) :: static_values statics))
    (Static_env (static_code statics')
       ((s, allocated) :: static_values statics')).
Proof.
  intros (Hcode & Hcode_avoids & Hvalues & Hvalues_avoid) Hallocated.
  split; [ | split; [ | split ] ]; simpl.
  - exact Hcode.
  - exact Hcode_avoids.
  - intros s' Hkept. destruct (Pos.eqb s' s); [ reflexivity | ].
    apply Hvalues, Hkept.
  - intros s' allocated' Hkept. destruct (Pos.eqb s' s);
      [ intros [= <-]; exact Hallocated | apply Hvalues_avoid, Hkept ].
Qed.

Lemma statics_erased_drop_value (statics statics' : static_env) (s : symbol)
    (allocated : statically_allocated_value) :
  statics_erased statics statics' ->
  symbol_kept D s = false ->
  statics_erased
    (Static_env (static_code statics)
       ((s, allocated) :: static_values statics))
    statics'.
Proof.
  intros (Hcode & Hcode_avoids & Hvalues & Hvalues_avoid) Hdropped.
  split; [ | split; [ | split ] ]; simpl.
  - exact Hcode.
  - exact Hcode_avoids.
  - intros s' Hkept.
    destruct (Pos.eqb_spec s' s) as [-> | _]; [ congruence | ].
    apply Hvalues, Hkept.
  - intros s' allocated' Hkept.
    destruct (Pos.eqb_spec s' s) as [-> | _]; [ congruence | ].
    apply Hvalues_avoid, Hkept.
Qed.

(* The code of a function of a set none of whose code is removed is
   kept, and so is the symbol of a closure of a static set none of whose
   symbols is removed. *)
Lemma lookup_function_decl_kept (slot : function_slot)
    (decls : list (function_slot * code_id)) (code : code_id) :
  function_decls_avoid D decls = true ->
  lookup slot decls = Some code ->
  code_kept D code = true.
Proof.
  unfold function_decls_avoid.
  induction decls as [ | [slot' code'] decls IH]; simpl; [ discriminate | ].
  intros Hav. apply andb_true_iff in Hav as [Hcode Hdecls].
  destruct (Pos.eqb slot slot');
    [ intros [= <-]; exact Hcode | exact (IH Hdecls) ].
Qed.

Lemma lookup_closure_symbol_kept (slot : function_slot)
    (symbols : list (function_slot * symbol)) (s : symbol) :
  closure_symbols_avoid D symbols = true ->
  lookup slot symbols = Some s ->
  symbol_kept D s = true.
Proof.
  unfold closure_symbols_avoid.
  induction symbols as [ | [slot' s'] symbols IH]; simpl; [ discriminate | ].
  intros Hav. apply andb_true_iff in Hav as [Hs Hsymbols].
  destruct (Pos.eqb slot slot');
    [ intros [= <-]; exact Hs | exact (IH Hsymbols) ].
Qed.

(* A value that refers to no name removed is the same closure under both
   static environments, of code that is kept. *)
Lemma closure_of_erased (statics statics' : static_env) (v : value) :
  statics_erased statics statics' ->
  value_avoids D v = true ->
  closure_of statics' v = closure_of statics v
  ∧ (∀ (called : code_id) (slots : list (value_slot * value)),
      closure_of statics v = Some (called, slots) ->
      code_kept D called = true ∧ slot_values_avoid slots = true).
Proof.
  intros (_ & _ & Hvalues & Hvalues_avoid).
  destruct v as [c | code slots | fslot decls slots | s | tag fields
    | elements | contents | l | pa_callee pa_args pa_arity]; simpl.
  - intros _. split; [ reflexivity | discriminate ].
  - intros Hv. apply andb_true_iff in Hv as [Hcode Hslots].
    split; [ reflexivity | ]. intros called slots' [= <- <-]. auto.
  - intros Hv. apply andb_true_iff in Hv as [Hdecls Hslots].
    split; [ reflexivity | ]. intros called slots'.
    destruct (lookup fslot decls) as [code | ] eqn:Hcode; [ | discriminate ].
    simpl. intros [= <- <-].
    split;
      [ exact (lookup_function_decl_kept fslot decls code Hdecls Hcode)
      | exact Hslots ].
  - intros Hkept. rewrite (Hvalues s Hkept). split; [ reflexivity | ].
    intros called slots.
    destruct (lookup s (static_values statics))
      as [[code slots' | fslot syms decls slots' | tag fields | contents
          | bytes | elements | ] | ] eqn:Hlookup;
      try discriminate.
    + intros [= <- <-].
      pose proof (Hvalues_avoid s _ Hkept Hlookup) as Hallocated.
      simpl in Hallocated. apply andb_true_iff in Hallocated.
      exact Hallocated.
    + pose proof (Hvalues_avoid s _ Hkept Hlookup) as Hallocated.
      simpl in Hallocated.
      apply andb_true_iff in Hallocated as [Hallocated Hslots].
      apply andb_true_iff in Hallocated as [_ Hdecls].
      destruct (lookup fslot decls) as [code | ] eqn:Hcode;
        [ | discriminate ].
      simpl. intros [= <- <-].
      split;
        [ exact (lookup_function_decl_kept fslot decls code Hdecls Hcode)
        | exact Hslots ].
  - intros _. split; [ reflexivity | discriminate ].
  - intros _. split; [ reflexivity | discriminate ].
  - intros _. split; [ reflexivity | discriminate ].
  - intros _. split; [ reflexivity | discriminate ].
  - intros _. split; [ reflexivity | discriminate ].
Qed.

(* A value that refers to no name removed is the same closure of a set
   under both static environments, and its value slots refer to none. *)
Lemma closure_in_set_erased (statics statics' : static_env) (v : value) :
  statics_erased statics statics' ->
  value_avoids D v = true ->
  closure_in_set statics' v = closure_in_set statics v
  ∧ (∀ (at_slot : function_slot)
       (decls : list (function_slot * code_id))
       (slots : list (value_slot * value)),
      closure_in_set statics v = Some (at_slot, decls, slots) ->
      slot_values_avoid slots = true).
Proof.
  intros (_ & _ & Hvalues & Hvalues_avoid).
  destruct v as [c | code slots | fslot decls slots | s | tag fields
    | elements | contents | l | pa_callee pa_args pa_arity]; simpl.
  - intros _. split; [ reflexivity | discriminate ].
  - intros Hv. apply andb_true_iff in Hv as [_ Hslots].
    split; [ reflexivity | ]. intros at_slot decls slots' [= _ _ <-].
    exact Hslots.
  - intros Hv. apply andb_true_iff in Hv as [_ Hslots].
    split; [ reflexivity | ]. intros at_slot decls' slots'.
    destruct (lookup fslot decls); [ | discriminate ].
    simpl. intros [= _ _ <-]. exact Hslots.
  - intros Hkept. rewrite (Hvalues s Hkept). split; [ reflexivity | ].
    intros at_slot decls slots.
    destruct (lookup s (static_values statics))
      as [[code slots' | fslot syms decls' slots' | tag fields | contents
          | bytes | elements | ] | ] eqn:Hlookup;
      try discriminate;
      pose proof (Hvalues_avoid s _ Hkept Hlookup) as Hallocated;
      simpl in Hallocated; apply andb_true_iff in Hallocated as [_ Hslots].
    + intros [= _ _ <-]. exact Hslots.
    + destruct (lookup fslot decls'); [ | discriminate ].
      simpl. intros [= _ _ <-]. exact Hslots.
  - intros _. split; [ reflexivity | discriminate ].
  - intros _. split; [ reflexivity | discriminate ].
  - intros _. split; [ reflexivity | discriminate ].
  - intros _. split; [ reflexivity | discriminate ].
  - intros _. split; [ reflexivity | discriminate ].
Qed.

(* So a value slot projected from it is the same, and refers to none. *)
Lemma project_value_slot_erased (statics statics' : static_env)
    (project_from : function_slot) (slot : value_slot) (v : value) :
  statics_erased statics statics' ->
  value_avoids D v = true ->
  project_value_slot statics' project_from slot v
  = project_value_slot statics project_from slot v
  ∧ (∀ w : value,
      project_value_slot statics project_from slot v = Some w ->
      value_avoids D w = true).
Proof.
  intros Hstatics Hv. unfold project_value_slot.
  destruct (closure_in_set_erased statics statics' v Hstatics Hv)
    as [Hsame Hslots].
  rewrite Hsame. split; [ reflexivity | ].
  destruct (closure_in_set statics v) as [[[at_slot decls] slots] | ];
    [ | discriminate ].
  destruct (Pos.eqb at_slot project_from); [ | discriminate ].
  intros w.
  exact (lookup_slot_avoids slot slots w (Hslots at_slot decls slots eq_refl)).
Qed.

(* The closure at another slot of the set of a value that refers to no
   name removed is the same under both static environments, and refers
   to none either: for a symbol, it is one of the symbols of its set,
   which are kept with it. *)
Lemma project_function_slot_erased (statics statics' : static_env)
    (move_from move_to : function_slot) (v : value) :
  statics_erased statics statics' ->
  value_avoids D v = true ->
  project_function_slot statics' move_from move_to v
  = project_function_slot statics move_from move_to v
  ∧ (∀ w : value,
      project_function_slot statics move_from move_to v = Some w ->
      value_avoids D w = true).
Proof.
  intros (_ & _ & Hvalues & Hvalues_avoid) Hv.
  assert (Hunnamed :
    ∀ w : value,
      (if Pos.eqb move_from unnamed_function_slot
       then if Pos.eqb move_to unnamed_function_slot then Some v else None
       else None)
      = Some w ->
      value_avoids D w = true).
  { intros w. destruct (Pos.eqb move_from unnamed_function_slot);
      [ | discriminate ].
    destruct (Pos.eqb move_to unnamed_function_slot); [ | discriminate ].
    intros [= <-]. exact Hv. }
  destruct v as [c | code slots | fslot decls slots | s | tag fields
    | elements | contents | l | pa_callee pa_args pa_arity]; simpl.
  - split; [ reflexivity | discriminate ].
  - split; [ reflexivity | exact Hunnamed ].
  - split; [ reflexivity | ]. intros w.
    destruct (Pos.eqb fslot move_from); [ | discriminate ].
    destruct (lookup fslot decls); [ | discriminate ].
    destruct (lookup move_to decls); [ | discriminate ].
    simpl. intros [= <-]. exact Hv.
  - simpl in Hv. rewrite (Hvalues s Hv). split; [ reflexivity | ].
    destruct (lookup s (static_values statics))
      as [[code slots' | fslot syms decls slots' | tag fields | contents
          | bytes | elements | ] | ] eqn:Hlookup;
      try discriminate.
    + exact Hunnamed.
    + pose proof (Hvalues_avoid s _ Hv Hlookup) as Hallocated.
      simpl in Hallocated.
      apply andb_true_iff in Hallocated as [Hallocated _].
      apply andb_true_iff in Hallocated as [Hsyms _].
      intros w. destruct (Pos.eqb fslot move_from); [ | discriminate ].
      destruct (lookup fslot decls); [ | discriminate ].
      destruct (lookup move_to decls); [ | discriminate ].
      simpl.
      destruct (lookup move_to syms) as [s' | ] eqn:Hsym; [ | discriminate ].
      simpl. intros [= <-]. simpl.
      exact (lookup_closure_symbol_kept move_to syms s' Hsyms Hsym).
  - split; [ reflexivity | discriminate ].
  - split; [ reflexivity | discriminate ].
  - split; [ reflexivity | discriminate ].
  - split; [ reflexivity | discriminate ].
  - split; [ reflexivity | discriminate ].
Qed.

(* The primitives introduce no symbols and no code ids: referring to no
   name removed is a property that every primitive preserves
   (FlambdaPrimitiveSpec.eval_prim_preserves_statement). *)
Lemma eval_prim_avoids (statics statics' : static_env) (p : prim)
    (vs : list value) (v : value) :
  statics_erased statics statics' ->
  forallb (value_avoids D) vs = true ->
  eval_prim statics p vs = Some v ->
  value_avoids D v = true.
Proof.
  intros (_ & _ & _ & Hvalues_avoid) Hvs.
  apply (eval_prim_preserves (fun w => value_avoids D w = true)).
  - intros c. split; reflexivity.
  - intros tag fields. simpl. apply forallb_forall.
  - intros elements. simpl. apply forallb_forall.
  - intros s tag fields Hkept Hlookup.
    pose proof (Hvalues_avoid s _ Hkept Hlookup) as Hfields.
    simpl in Hfields. apply forallb_forall. exact Hfields.
  - intros s elements Hkept Hlookup.
    pose proof (Hvalues_avoid s _ Hkept Hlookup) as Helements.
    simpl in Helements. apply forallb_forall. exact Helements.
  - apply forallb_forall. exact Hvs.
Qed.

(* The value of a primitive is the same under both static environments,
   as its arguments refer to no name removed. *)
Lemma eval_prim_erased (statics statics' : static_env) (p : prim)
    (vs : list value) :
  statics_erased statics statics' ->
  forallb (value_avoids D) vs = true ->
  eval_prim statics' p vs = eval_prim statics p vs.
Proof.
  intros (_ & _ & Hvalues & _) Hvs. apply eval_prim_static_values.
  intros s Hin. apply Hvalues.
  exact (proj1 (forallb_forall _ _) Hvs _ Hin).
Qed.

(* The same on a heap: what a primitive does is the same under both
   static environments; and where no value of the heap refers to a name
   removed, none of the heap after does, nor the value. *)
Lemma heap_free_of_values_all (hp : heap) :
  heap_free_of D hp = true
  <-> heap_values_all (fun w => value_avoids D w = true) hp.
Proof.
  unfold heap_free_of, heap_values_all. rewrite forallb_forall. split.
  - intros Hfree l c w Hin Hw.
    exact (proj1 (forallb_forall _ _) (Hfree (l, c) Hin) w Hw).
  - intros Hall [l c] Hin. apply forallb_forall. intros w Hw.
    exact (Hall l c w Hin Hw).
Qed.

Lemma exec_prim_erased (statics statics' : static_env) (hp : heap)
    (p : prim) (vs : list value) :
  statics_erased statics statics' ->
  forallb (value_avoids D) vs = true ->
  exec_prim statics' hp p vs = exec_prim statics hp p vs.
Proof.
  intros (_ & _ & Hvalues & _) Hvs.
  apply FlambdaHeapProofs.exec_prim_static_values.
  intros s Hin. apply Hvalues.
  exact (proj1 (forallb_forall _ _) Hvs _ Hin).
Qed.

Lemma exec_prim_avoids (statics statics' : static_env) (hp hp' : heap)
    (p : prim) (vs : list value) (v : value) :
  statics_erased statics statics' ->
  forallb (value_avoids D) vs = true ->
  heap_free_of D hp = true ->
  exec_prim statics hp p vs = Some (v, hp') ->
  value_avoids D v = true ∧ heap_free_of D hp' = true.
Proof.
  intros (_ & _ & _ & Hvalues_avoid) Hvs Hfree Hexec.
  rewrite heap_free_of_values_all in Hfree |- *.
  apply
    (FlambdaHeapProofs.exec_prim_preserves (fun w => value_avoids D w = true)
       statics hp hp' p vs v);
    [ | | | | | | | exact Hfree | exact Hexec ].
  - intros c. split; reflexivity.
  - intros tag fields. simpl. apply forallb_forall.
  - intros elements. simpl. apply forallb_forall.
  - intros s tag fields Hkept Hlookup.
    pose proof (Hvalues_avoid s _ Hkept Hlookup) as Hfields.
    simpl in Hfields. apply forallb_forall. exact Hfields.
  - intros s elements Hkept Hlookup.
    pose proof (Hvalues_avoid s _ Hkept Hlookup) as Hfields.
    simpl in Hfields. apply forallb_forall. exact Hfields.
  - intros l. reflexivity.
  - apply forallb_forall. exact Hvs.
Qed.

Lemma eval_named_erased (statics statics' : static_env) (vars : env)
    (d : named) :
  statics_erased statics statics' ->
  env_avoids vars ->
  named_avoids D d = true ->
  eval_named statics' vars d = eval_named statics vars d
  ∧ (∀ v : value,
      eval_named statics vars d = Some v -> value_avoids D v = true).
Proof.
  intros Hstatics Hvars.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure]; simpl; intros Hd.
  - split; [ reflexivity | ]. intros v.
    apply eval_simple_avoids; assumption.
  - destruct (eval_simples vars args) as [vs | ] eqn:Hargs; simpl;
      [ | split; [ reflexivity | discriminate ] ].
    pose proof (eval_simples_avoid vars args Hvars vs Hd Hargs) as Hvs.
    split; [ exact (eval_prim_erased statics statics' p vs Hstatics Hvs) | ].
    intros v. exact (eval_prim_avoids statics statics' p vs v Hstatics Hvs).
  - apply andb_true_iff in Hd as [Hcode Hslots].
    split; [ reflexivity | ]. intros v.
    destruct (eval_value_slots vars slots) as [evaluated | ] eqn:Heval;
      [ | discriminate ].
    simpl. intros [= <-]. simpl. rewrite Hcode. simpl.
    exact (eval_value_slots_avoid vars slots Hvars evaluated Hslots Heval).
  - apply andb_true_iff in Hd as [Hdecls Hslots].
    split; [ reflexivity | ]. intros v.
    destruct (lookup fslot decls) as [code | ]; [ | discriminate ]. simpl.
    destruct (eval_value_slots vars slots) as [evaluated | ] eqn:Heval;
      [ | discriminate ].
    simpl. intros [= <-]. simpl. rewrite Hdecls. simpl.
    exact (eval_value_slots_avoid vars slots Hvars evaluated Hslots Heval).
  - destruct (eval_simple vars closure) as [v | ] eqn:Hclosure; simpl;
      [ | split; [ reflexivity | discriminate ] ].
    destruct (closure_of_erased statics statics' v Hstatics
                (eval_simple_avoids vars closure v Hvars Hd Hclosure))
      as [Hsame Hslots].
    rewrite Hsame. split; [ reflexivity | ].
    destruct (closure_of statics v) as [[called slots] | ]; simpl;
      [ | discriminate ].
    intros w.
    exact (lookup_slot_avoids slot slots w
      (proj2 (Hslots called slots eq_refl))).
  - destruct (eval_simple vars closure) as [v | ] eqn:Hclosure; simpl;
      [ | split; [ reflexivity | discriminate ] ].
    exact (project_value_slot_erased statics statics' pfrom slot v Hstatics
             (eval_simple_avoids vars closure v Hvars Hd Hclosure)).
  - destruct (eval_simple vars closure) as [v | ] eqn:Hclosure; simpl;
      [ | split; [ reflexivity | discriminate ] ].
    exact (project_function_slot_erased statics statics' move_from move_to v
             Hstatics (eval_simple_avoids vars closure v Hvars Hd Hclosure)).
Qed.

Lemma eval_static_const_avoids (vars : env) (const : static_const)
    (allocated : statically_allocated_value) :
  env_avoids vars ->
  static_const_avoids D const = true ->
  eval_static_const vars const = Some allocated ->
  allocated_avoids allocated = true.
Proof.
  intros Hvars.
  destruct const
    as [code slots | fslot syms decls slots | tag fields | contents | bytes
        | elements];
    simpl; intros Hconst.
  - apply andb_true_iff in Hconst as [Hcode Hslots].
    destruct (eval_value_slots vars slots) as [evaluated | ] eqn:Heval;
      [ | discriminate ].
    simpl. intros [= <-]. simpl. rewrite Hcode. simpl.
    exact (eval_value_slots_avoid vars slots Hvars evaluated Hslots Heval).
  - apply andb_true_iff in Hconst as [Hconst Hslots].
    destruct (lookup fslot decls) as [code | ]; [ | discriminate ]. simpl.
    destruct (eval_value_slots vars slots) as [evaluated | ] eqn:Heval;
      [ | discriminate ].
    simpl. intros [= <-]. simpl. rewrite Hconst. simpl.
    exact (eval_value_slots_avoid vars slots Hvars evaluated Hslots Heval).
  - destruct (eval_simples vars fields) as [vs | ] eqn:Heval;
      [ | discriminate ].
    simpl. intros [= <-]. simpl.
    exact (eval_simples_avoid vars fields Hvars vs Hconst Heval).
  - destruct (eval_simple vars contents) as [v | ] eqn:Heval;
      [ | discriminate ].
    simpl. intros [= <-]. simpl.
    exact (eval_simple_avoids vars contents v Hvars Hconst Heval).
  - intros [= <-]. reflexivity.
  - destruct (eval_simples vars elements) as [vs | ] eqn:Heval;
      [ | discriminate ].
    simpl. intros [= <-]. simpl.
    exact (eval_simples_avoid vars elements Hvars vs Hconst Heval).
Qed.

(* A group of bindings and the group erased, added to static environments
   one of which is the other erased: if the first has a result, the second
   has, and it is the first's erased. *)
Lemma eval_static_bindings_erased (vars : env) :
  env_avoids vars ->
  ∀ (bindings : list static_binding) (statics statics' : static_env),
    forallb (binding_avoids D) bindings = true ->
    statics_erased statics statics' ->
    match eval_static_bindings vars bindings statics with
    | Some extended =>
      ∃ extended' : static_env,
        eval_static_bindings vars
          (keep_map (binding_kept D) (erase_binding D) bindings) statics'
        = Some extended'
        ∧ statics_erased extended extended'
    | None => True
    end.
Proof.
  intros Hvars.
  induction bindings as [ | binding bindings IH]; intros statics statics';
    simpl.
  - intros _ Hstatics. exists statics'. auto.
  - intros Hav Hstatics. apply andb_true_iff in Hav as [Hbinding Hrest].
    destruct binding as [id c | id | sym const]; simpl in Hbinding |- *.
    + destruct (lookup id (static_code statics)) eqn:Hbound; [ exact I | ].
      destruct (code_kept D id) eqn:Hkept; simpl.
      * destruct Hstatics as [Hcode Hothers].
        rewrite (Hcode id Hkept), Hbound. simpl.
        apply (IH _ _ Hrest), statics_erased_add_code;
          [ exact (conj Hcode Hothers) | assumption .. ].
      * apply (IH _ _ Hrest), statics_erased_drop_code; assumption.
    + exact (IH _ _ Hrest Hstatics).
    + destruct (lookup sym (static_values statics)) eqn:Hbound;
        [ exact I | ].
      destruct (eval_static_const vars const) as [allocated | ] eqn:Hconst;
        simpl; [ | exact I ].
      destruct (symbol_kept D sym) eqn:Hkept; simpl.
      * destruct Hstatics as (Hcode & Hcode_avoids & Hvalues & Hothers).
        rewrite (Hvalues sym Hkept), Hbound, Hconst. simpl.
        apply (IH _ _ Hrest), statics_erased_add_value;
          [ exact (conj Hcode (conj Hcode_avoids (conj Hvalues Hothers))) | ].
        exact (eval_static_const_avoids vars const allocated Hvars Hbinding
          Hconst).
      * apply (IH _ _ Hrest), statics_erased_drop_value; assumption.
Qed.

(* ======================================================================== *)
(* Continuations and states                                                 *)
(* ======================================================================== *)

Definition handlers_avoid
    (handlers : list (continuation * continuation_handler)) : bool :=
  forallb (fun '(_, h) => handler_avoids D h) handlers.

Definition erase_handlers
    (handlers : list (continuation * continuation_handler))
    : list (continuation * continuation_handler) :=
  map (fun '(k, h) => (k, erase_handler D h)) handlers.

#[local] Set Warnings "-register-all".
Inductive cont_erased : cont_value -> cont_value -> Prop :=
  | Outer_erased (k : continuation) : cont_erased (Outer k) (Outer k)
  | Closure_erased (h : continuation_handler) (vars : env)
      (conts conts' : cont_env)
      (recursive : list (continuation * continuation_handler)) :
    handler_avoids D h = true ->
    env_avoids vars ->
    Forall2
      (fun kc kc' => fst kc = fst kc' ∧ cont_erased (snd kc) (snd kc'))
      conts conts' ->
    handlers_avoid recursive = true ->
    cont_erased (Closure h vars conts recursive)
      (Closure (erase_handler D h) vars conts' (erase_handlers recursive))
  | Extra_args_erased (c c' : cont_value) (extra : list value) :
    cont_erased c c' ->
    forallb (value_avoids D) extra = true ->
    cont_erased (Extra_args c extra) (Extra_args c' extra)
  | Over_application_erased (remaining : list value)
      (remaining_arity return_arity : arity)
      (return_to return_to' exn_to exn_to' : cont_value) :
    forallb (value_avoids D) remaining = true ->
    cont_erased return_to return_to' ->
    cont_erased exn_to exn_to' ->
    cont_erased
      (Over_application remaining remaining_arity return_arity return_to
         exn_to)
      (Over_application remaining remaining_arity return_arity return_to'
         exn_to').

Lemma with_extra_args_erased (c c' : cont_value) (extra : list value) :
  cont_erased c c' ->
  forallb (value_avoids D) extra = true ->
  cont_erased (with_extra_args c extra) (with_extra_args c' extra).
Proof.
  intros Hc Hextra. destruct extra; simpl in *;
    [ exact Hc | apply Extra_args_erased; assumption ].
Qed.

Definition conts_erased (conts conts' : cont_env) : Prop :=
  Forall2 (fun kc kc' => fst kc = fst kc' ∧ cont_erased (snd kc) (snd kc'))
    conts conts'.

Lemma resolve_erased (k : continuation) (conts conts' : cont_env) :
  conts_erased conts conts' ->
  cont_erased (resolve k conts) (resolve k conts').
Proof.
  unfold resolve.
  induction 1 as [ | [k1 c] [k1' c'] conts conts' [Hk Hc] _ IH]; simpl;
    [ apply Outer_erased | ].
  simpl in Hk. subst k1'. destruct (Pos.eqb k k1); [ exact Hc | exact IH ].
Qed.

(* Erasing binds no more than was bound. *)
Fixpoint erase_binds_statics (e : expr) :
  binds_statics e = false -> binds_statics (erase_expr D e) = false.
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    simpl; try (intros _; reflexivity).
  - apply erase_binds_statics.
  - intros Hbinds. apply orb_false_iff in Hbinds as [Hbindings Hbody].
    rewrite (erase_binds_statics body Hbody), orb_false_r.
    induction bindings as [ | binding bindings IH]; simpl; [ reflexivity | ].
    simpl in Hbindings. apply orb_false_iff in Hbindings as [Hbinding Hrest].
    destruct (binding_kept D binding); simpl; [ | exact (IH Hrest) ].
    rewrite (IH Hrest), orb_false_r.
    destruct binding; simpl in Hbinding |- *; congruence.
  - destruct h as [params metadata handler]. simpl. intros Hbinds.
    apply orb_false_iff in Hbinds as [Hhandler Hbody].
    rewrite (erase_binds_statics handler Hhandler),
      (erase_binds_statics body Hbody).
    reflexivity.
  - intros Hbinds. apply orb_false_iff in Hbinds as [Hhandlers Hbody].
    rewrite (erase_binds_statics body Hbody), orb_false_r.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      simpl; [ reflexivity | ].
    simpl in Hhandlers.
    apply orb_false_iff in Hhandlers as [Hhandler Hrest].
    rewrite (erase_binds_statics handler Hhandler), (IH Hrest).
    reflexivity.
Qed.

Lemma erase_handlers_bind_statics
    (handlers : list (continuation * continuation_handler)) :
  handlers_bind_statics handlers = false ->
  handlers_bind_statics (map (fun '(k, h) => (k, erase_handler D h)) handlers)
  = false.
Proof.
  unfold handlers_bind_statics.
  induction handlers as [ | [k [params metadata handler]] handlers IH]; simpl;
    [ reflexivity | ].
  intros Hbinds. apply orb_false_iff in Hbinds as [Hhandler Hrest].
  rewrite (erase_binds_statics handler Hhandler), (IH Hrest). reflexivity.
Qed.

(* [all] is the whole group, kept apart from the handlers still to be gone
   through. *)
Lemma rec_closures_erased_aux (vars : env) (conts conts' : cont_env)
    (all handlers : list (continuation * continuation_handler)) :
  env_avoids vars ->
  conts_erased conts conts' ->
  handlers_avoid all = true ->
  handlers_avoid handlers = true ->
  conts_erased (map (fun '(k, h) => (k, Closure h vars conts all)) handlers)
    (map (fun '(k, h) => (k, Closure h vars conts' (erase_handlers all)))
       (erase_handlers handlers)).
Proof.
  intros Hvars Hconts Hall. unfold handlers_avoid.
  induction handlers as [ | [k h] handlers IH]; simpl;
    [ intros _; apply Forall2_nil | ].
  intros Hav. apply andb_true_iff in Hav as [Hh Hrest].
  apply Forall2_cons; [ | exact (IH Hrest) ].
  split; [ reflexivity | ]. apply Closure_erased; assumption.
Qed.

Lemma rec_closures_erased (vars : env) (conts conts' : cont_env)
    (handlers : list (continuation * continuation_handler)) :
  env_avoids vars ->
  conts_erased conts conts' ->
  handlers_avoid handlers = true ->
  conts_erased (rec_closures handlers vars conts ++ conts)
    (rec_closures (erase_handlers handlers) vars conts' ++ conts').
Proof.
  intros Hvars Hconts Hhandlers. apply Forall2_app; [ | exact Hconts ].
  apply rec_closures_erased_aux; assumption.
Qed.

(* What a named expression does is the same under both static
   environments, from a heap none of whose values refers to a name
   removed; and then no value of the heap after does, nor the value. *)
Lemma exec_named_erased (statics statics' : static_env) (hp : heap)
    (rd : read_record)
    (vars : env) (d : named) :
  statics_erased statics statics' ->
  env_avoids vars ->
  named_avoids D d = true ->
  heap_free_of D hp = true ->
  exec_named statics' hp rd vars d = exec_named statics hp rd vars d
  ∧ (∀ (v : value) (hp' : heap) (rd' : read_record),
      exec_named statics hp rd vars d = Some (v, hp', rd') ->
      value_avoids D v = true ∧ heap_free_of D hp' = true).
Proof.
  intros Hstatics Hvars Hd Hfree.
  destruct (eval_named_erased statics statics' vars d Hstatics Hvars Hd)
    as [Hsame Hvalue].
  assert (Hpure :
    (let* v := eval_named statics' vars d in Some (v, hp, rd))
    = (let* v := eval_named statics vars d in Some (v, hp, rd))
    ∧ (∀ (v : value) (hp' : heap) (rd' : read_record),
        (let* v := eval_named statics vars d in Some (v, hp, rd))
        = Some (v, hp', rd') ->
        value_avoids D v = true ∧ heap_free_of D hp' = true)).
  { rewrite Hsame. split; [ reflexivity | ]. intros v hp' rd'.
    destruct (eval_named statics vars d) as [w | ] eqn:Hw; [ | discriminate ].
    intros [= <- <- <-]. split; [ exact (Hvalue w eq_refl) | exact Hfree ]. }
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    try exact Hpure.
  clear Hpure Hsame Hvalue. rewrite !exec_named_prim. simpl in Hd.
  destruct (eval_simples vars args) as [vs | ] eqn:Hargs; simpl;
    [ | split; [ reflexivity | discriminate ] ].
  pose proof (eval_simples_avoid vars args Hvars vs Hd Hargs) as Hvs.
  rewrite (exec_prim_erased statics statics' hp p vs Hstatics Hvs).
  split; [ reflexivity | ].
  intros v hp' rd'.
  destruct (exec_prim statics hp p vs) as [[w h] | ] eqn:Hprim; simpl;
    [ | discriminate ].
  intros [= <- <- <-].
  exact (exec_prim_avoids statics statics' hp h p vs w Hstatics Hvs Hfree
           Hprim).
Qed.

Inductive state_erased : state -> state -> Prop :=
  | State_erased (e : expr) (vars : env) (conts conts' : cont_env)
      (statics statics' : static_env) (hp : heap) (rd : read_record) :
    expr_avoids D e = true ->
    env_avoids vars ->
    conts_erased conts conts' ->
    statics_erased statics statics' ->
    state_erased (State e vars conts statics hp rd)
      (State (erase_expr D e) vars conts' statics' hp rd).

(* A value that avoids the names removed, as a property: that of a
   partial application is that of the value it calls and of its
   arguments. *)
Lemma value_avoids_partial (callee : value) (args : list value)
    (args_arity : arity) :
  value_avoids D (Value_partial_application callee args args_arity) = true
  <-> value_avoids D callee = true
      ∧ Forall (fun v => value_avoids D v = true) args.
Proof. simpl. rewrite andb_true_iff, forallb_Forall. reflexivity. Qed.

(* The arguments that the closure of a call is called with avoid the
   names removed, where the callee and the arguments of the call do. *)
Lemma called_arguments_avoid (callee : value) (vs : list value) :
  value_avoids D callee = true ->
  forallb (value_avoids D) vs = true ->
  forallb (value_avoids D) (called_arguments callee vs) = true.
Proof.
  unfold called_arguments.
  destruct callee; simpl; try (intros _ H; exact H).
  intros Hcallee Hvs. apply andb_true_iff in Hcallee as [_ Hargs].
  rewrite forallb_app, Hargs, Hvs. reflexivity.
Qed.

(* Under the two static environments the primitives read alike the
   first of those arguments, which a call of tupled code takes apart. *)
Lemma called_tuple_erased (statics statics' : static_env) (callee : value)
    (vs : list value) (tuple : value) (rest : list value) (p : prim) :
  statics_erased statics statics' ->
  value_avoids D callee = true ->
  forallb (value_avoids D) vs = true ->
  called_arguments callee vs = tuple :: rest ->
  eval_prim statics' p [tuple] = eval_prim statics p [tuple].
Proof.
  intros Hstatics Hcallee Hvs Hcalled.
  pose proof (called_arguments_avoid callee vs Hcallee Hvs) as Havoid.
  rewrite Hcalled in Havoid. simpl in Havoid.
  apply andb_true_iff in Havoid as [Htuple _].
  apply (eval_prim_erased statics statics' p [tuple] Hstatics).
  simpl. rewrite Htuple. reflexivity.
Qed.

(* The fields of a tuple that avoids the names removed avoid them. *)
Lemma tuple_fields_avoid (statics statics' : static_env) :
  statics_erased statics statics' ->
  P_fields (fun v => value_avoids D v = true) statics.
Proof.
  intros Hstatics size tuple fields Htuple Hfields.
  apply (tuple_fields_forall _ statics size tuple fields);
    [ | exact Hfields ].
  intros field v Hv.
  apply (eval_prim_avoids statics statics'
           (Unary (Block_load (tuple_access size) Immutable field)) [tuple] v
           Hstatics);
    [ simpl; rewrite Htuple; reflexivity | exact Hv ].
Qed.

Lemma called_closure_avoids (callee : value) :
  value_avoids D callee = true ->
  value_avoids D (called_closure callee) = true.
Proof.
  destruct callee; simpl; try (intros H; exact H).
  intros H. apply andb_true_iff in H. exact (proj1 H).
Qed.

(* The code of a closure that avoids the names removed is bound alike,
   erased, by the two static environments. *)
Lemma codes_agree_erased (statics statics' : static_env) (closure : value)
    (called : code_id) (slots : list (value_slot * value)) :
  statics_erased statics statics' ->
  value_avoids D closure = true ->
  closure_of statics closure = Some (called, slots) ->
  codes_agree (fun c c' => c' = erase_code D c ∧ code_avoids D c = true)
    statics statics' called.
Proof.
  intros Hstatics Hclosure Hcalled.
  destruct (closure_of_erased statics statics' closure Hstatics Hclosure)
    as [_ Hslots].
  destruct (Hslots called slots Hcalled) as [Hkept _].
  destruct Hstatics as (Hcode & Hcode_avoids & _).
  unfold codes_agree. rewrite (Hcode called Hkept).
  destruct (lookup called (static_code statics)) as [c | ] eqn:Hlookup;
    simpl; [ | exact I ].
  split; [ split; [ reflexivity | exact (Hcode_avoids called c Hkept Hlookup) ]
         | ].
  destruct c. repeat split.
Qed.

Lemma call_code_erased (closure : value) (c : code) (vs : list value)
    (args_arity return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics statics' : static_env) (hp : heap) (rd : read_record) :
  code_avoids D c = true ->
  value_avoids D closure = true ->
  forallb (value_avoids D) vs = true ->
  cont_erased return_to return_to' ->
  cont_erased exn_to exn_to' ->
  statics_erased statics statics' ->
  result_simulated state_erased
    (call_code closure c vs args_arity return_arity return_to exn_to statics hp
       rd)
    (call_code closure (erase_code D c) vs args_arity return_arity return_to'
       exn_to' statics' hp rd).
Proof.
  intros Hc Hclosure Hvs Hreturn Hexn Hstatics.
  destruct c as [params my_closure ret exn metadata body].
  unfold call_code. simpl.
  destruct (binds_statics body) eqn:Hbinds; [ exact I | ].
  rewrite (erase_binds_statics body Hbinds).
  destruct
    (negb
       (kinds_eqb (map kind_of_value vs) (unarize args_arity)
        && kinds_per_parameter_eqb (unarize_per_parameter args_arity)
             (unarize_per_parameter (cm_params_arity metadata))
        && arity_eqb return_arity (cm_result_arity metadata)));
    [ exact I | ].
  destruct (bind_params params vs [(my_closure, closure)]) as [bound | ]
    eqn:Hbind; simpl; [ | exact I ].
  apply State_erased.
  - exact Hc.
  - apply (bind_params_avoids params vs [(my_closure, closure)] bound Hvs);
      [ | exact Hbind ].
    apply env_avoids_cons; [ exact Hclosure | ]. intros x v [=].
  - apply Forall2_cons; [ split; [ reflexivity | exact Hreturn ] | ].
    apply Forall2_cons; [ split; [ reflexivity | exact Hexn ] | ].
    apply Forall2_nil.
  - exact Hstatics.
Qed.

Lemma run_call_erased (return_partial return_partial' : value -> step_result)
    (shape shape' : option call_shape) (return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics statics' : static_env) (hp : heap) (rd : read_record) :
  shapes_related (fun c c' => c' = erase_code D c ∧ code_avoids D c = true)
    shape shape' ->
  shape_values_all (fun v => value_avoids D v = true) shape ->
  cont_erased return_to return_to' ->
  cont_erased exn_to exn_to' ->
  statics_erased statics statics' ->
  (∀ partial : value,
    value_avoids D partial = true ->
    result_simulated state_erased (return_partial partial)
      (return_partial' partial)) ->
  result_simulated state_erased
    (run_call return_partial shape return_arity return_to exn_to statics hp rd)
    (run_call return_partial' shape' return_arity return_to' exn_to'
       statics' hp rd).
Proof.
  intros Hshapes Hvalues Hreturn Hexn Hstatics Hpartial.
  destruct Hshapes
    as [ | closure c c' vs args_arity [-> Hc]
       | closure c c' vs remaining args_arity remaining_arity [-> Hc]
       | partial];
    simpl in Hvalues |- *.
  - exact I.
  - destruct Hvalues as [Hclosure Hvs]. apply forallb_Forall in Hvs.
    apply call_code_erased; assumption.
  - destruct Hvalues as (Hclosure & Hvs & Hremaining).
    apply forallb_Forall in Hvs, Hremaining.
    replace (code_result_arity (erase_code D c)) with (code_result_arity c)
      by (destruct c; reflexivity).
    apply call_code_erased; try assumption.
    apply Over_application_erased; assumption.
  - exact (Hpartial partial Hvalues).
Qed.

Lemma apply_cont_value_erased (c c' : cont_value) (vs : list value)
    (statics statics' : static_env) (hp : heap) (rd : read_record) :
  cont_erased c c' ->
  forallb (value_avoids D) vs = true ->
  statics_erased statics statics' ->
  result_simulated state_erased (apply_cont_value c vs statics hp rd)
    (apply_cont_value c' vs statics' hp rd).
Proof.
  intros Hc Hvs Hstatics. revert vs Hvs.
  induction Hc
    as [k
       | [params metadata handler] vars conts conts' recursive Hh Hvars
           Hconts Hrec
       | c c' extra Hc IH Hextra
       | remaining remaining_arity return_arity return_to return_to'
           exn_to exn_to' Hremaining Hreturn IHreturn Hexn _];
    intros vs Hvs.
  - simpl. auto.
  - simpl.
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
      [ | exact I ].
    destruct (bind_params params vs vars) as [bound | ] eqn:Hbind; simpl;
      [ | exact I ].
    apply State_erased;
      [ exact Hh
      | exact (bind_params_avoids params vs vars bound Hvs Hvars Hbind)
      | apply rec_closures_erased; assumption | exact Hstatics ].
  - simpl. apply IH. rewrite forallb_app, Hvs, Hextra. reflexivity.
  - rewrite !apply_cont_value_over_application.
    destruct vs as [ | v [ | v' vs]]; [ exact I | | exact I ].
    simpl in Hvs. rewrite andb_true_r in Hvs. unfold call_function.
    apply run_call_erased; try assumption.
    + apply shape_of_call_related.
      * exact (proj1 (closure_of_erased statics statics' _ Hstatics
                        (called_closure_avoids v Hvs))).
      * intros called slots.
        exact (codes_agree_erased statics statics' _ called slots Hstatics
                 (called_closure_avoids v Hvs)).
      * intros tuple rest p.
        exact (called_tuple_erased statics statics' v remaining tuple rest p
                 Hstatics Hvs Hremaining).
    + apply (shape_of_call_values_all _ value_avoids_partial);
        [ exact (tuple_fields_avoid statics statics' Hstatics) | exact Hvs
        | apply forallb_Forall; exact Hremaining ].
    + intros partial Hpartial. apply IHreturn. simpl.
      rewrite Hpartial. reflexivity.
Qed.

Lemma call_function_erased (callee : value) (kind : call_kind)
    (vs : list value) (args_arity return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics statics' : static_env) (hp : heap) (rd : read_record) :
  value_avoids D callee = true ->
  forallb (value_avoids D) vs = true ->
  cont_erased return_to return_to' ->
  cont_erased exn_to exn_to' ->
  statics_erased statics statics' ->
  result_simulated state_erased
    (call_function callee kind vs args_arity return_arity return_to exn_to
       statics hp rd)
    (call_function callee kind vs args_arity return_arity return_to' exn_to'
       statics' hp rd).
Proof.
  intros Hcallee Hvs Hreturn Hexn Hstatics. unfold call_function.
  apply run_call_erased; try assumption.
  - apply shape_of_call_related.
    + exact (proj1 (closure_of_erased statics statics' _ Hstatics
                      (called_closure_avoids callee Hcallee))).
    + intros called slots.
      exact (codes_agree_erased statics statics' _ called slots Hstatics
               (called_closure_avoids callee Hcallee)).
    + intros tuple rest p.
      exact (called_tuple_erased statics statics' callee vs tuple rest p
               Hstatics Hcallee Hvs).
  - apply (shape_of_call_values_all _ value_avoids_partial);
      [ exact (tuple_fields_avoid statics statics' Hstatics) | exact Hcallee
      | apply forallb_Forall; exact Hvs ].
  - intros partial Hpartial.
    apply apply_cont_value_erased; [ exact Hreturn | | exact Hstatics ].
    simpl. rewrite Hpartial. reflexivity.
Qed.

(* A state and the state erased step alike, where the first is defined
   and no value of its heap refers to a name removed. *)
Lemma erased_step (s s' : state) :
  state_erased s s' ->
  heap_free_of D (state_heap s) = true ->
  result_simulated state_erased (step s) (step s').
Proof.
  intros [e vars conts conts' statics statics' hp rd He Hvars Hconts Hstatics]
    Hfree.
  simpl in Hfree.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    simpl in He |- *; autorewrite with step; simpl.
  - (* Let *)
    apply andb_true_iff in He as [Hd Hbody].
    destruct
      (exec_named_erased statics statics' hp rd vars d Hstatics Hvars Hd Hfree)
      as [Hsame Hvalue].
    rewrite Hsame.
    destruct (exec_named statics hp rd vars d) as [[[v hp'] rd'] | ]; simpl;
      [ | exact I ].
    destruct (kind_eqb (kind_of_value v) kind); simpl; [ | exact I ].
    apply State_erased;
      [ exact Hbody
      | apply env_avoids_cons;
        [ exact (proj1 (Hvalue v hp' rd_ eq_refl)) | auto ]
      | exact Hconts | exact Hstatics ].
  - (* Let_static *)
    apply andb_true_iff in He as [Hbindings Hbody].
    pose proof (eval_static_bindings_erased vars Hvars bindings statics
      statics' Hbindings Hstatics) as Hextended.
    destruct (eval_static_bindings vars bindings statics) as [extended | ];
      [ | exact I ].
    destruct Hextended as (extended' & -> & Hextended).
    simpl. apply State_erased; assumption.
  - (* Let_cont_non_recursive *)
    apply andb_true_iff in He as [Hh Hbody].
    apply State_erased; [ exact Hbody | exact Hvars | | exact Hstatics ].
    apply Forall2_cons; [ | exact Hconts ].
    split; [ reflexivity | ].
    exact (Closure_erased h vars conts conts' [] Hh Hvars Hconts eq_refl).
  - (* Let_cont_recursive *)
    apply andb_true_iff in He as [Hhandlers Hbody].
    destruct (handlers_bind_statics handlers) eqn:Hbinds; [ exact I | ].
    rewrite (erase_handlers_bind_statics handlers Hbinds).
    apply State_erased; [ exact Hbody | exact Hvars | | exact Hstatics ].
    apply rec_closures_erased; assumption.
  - (* Apply *)
    unfold apply_avoids in He.
    apply andb_true_iff in He as [He Hextra].
    apply andb_true_iff in He as [He Hargs].
    apply andb_true_iff in He as [Hcallee _].
    destruct (eval_simple vars (ap_callee ap)) as [callee | ] eqn:Heval;
      [ | exact I ].
    destruct (eval_simples vars (ap_args ap)) as [vs | ] eqn:Hevals;
      [ | exact I ].
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ]
      eqn:Hevals_extra;
      [ | exact I ].
    apply call_function_erased;
      [ exact (eval_simple_avoids vars _ callee Hvars Hcallee Heval)
      | exact (eval_simples_avoid vars _ Hvars vs Hargs Hevals)
      | apply resolve_erased, Hconts
      | apply with_extra_args_erased;
        [ apply resolve_erased, Hconts
        | exact (eval_simples_avoid vars _ Hvars extra Hextra Hevals_extra) ]
      | exact Hstatics ].
  - (* Apply_cont *)
    destruct (eval_simples vars (ac_args ac)) as [vs | ] eqn:Hevals;
      [ | exact I ].
    apply apply_cont_value_erased;
      [ apply resolve_erased, Hconts
      | exact (eval_simples_avoid vars _ Hvars vs He Hevals) | exact Hstatics ].
  - (* Switch *)
    apply andb_true_iff in He as [_ Harms].
    destruct (eval_simple vars scrutinee) as [v | ]; [ | exact I ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | exact I ].
    destruct (find_arm i arms) as [ac | ] eqn:Hfind; simpl; [ | exact I ].
    apply (State_erased (Apply_cont ac));
      [ | exact Hvars | exact Hconts | exact Hstatics ].
    rewrite forallb_forall in Harms.
    exact (Harms (i, ac) (find_arm_in i arms ac Hfind)).
  - (* Invalid *)
    exact I.
Qed.

End Erasure.

(* ======================================================================== *)
(* Environments in which the names removed do not occur                     *)
(* ======================================================================== *)

(* Erasing changes nothing in what binds none of the names removed. *)
Fixpoint erase_binds_only_kept (D : dropped) (e : expr) :
  binds_only_kept D e = true -> erase_expr D e = e
with erase_handler_binds_only_kept (D : dropped) (h : continuation_handler) :
  handler_binds_only_kept D h = true -> erase_handler D h = h
with erase_binding_binds_only_kept (D : dropped) (binding : static_binding) :
  binding_binds_only_kept D binding = true ->
  binding_kept D binding = true ∧ erase_binding D binding = binding
with erase_code_binds_only_kept (D : dropped) (c : code) :
  code_binds_only_kept D c = true -> erase_code D c = c.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try reflexivity; intros Hkept.
    + rewrite (erase_binds_only_kept D body Hkept). reflexivity.
    + apply andb_true_iff in Hkept as [Hbindings Hbody].
      rewrite (erase_binds_only_kept D body Hbody). f_equal.
      induction bindings as [ | binding bindings IH]; simpl;
        [ reflexivity | ].
      simpl in Hbindings. apply andb_true_iff in Hbindings as [Hbinding Hrest].
      destruct (erase_binding_binds_only_kept D binding Hbinding)
        as [Hkept Hsame].
      rewrite Hkept, Hsame, (IH Hrest). reflexivity.
    + apply andb_true_iff in Hkept as [Hh Hbody].
      rewrite (erase_handler_binds_only_kept D h Hh),
        (erase_binds_only_kept D body Hbody).
      reflexivity.
    + apply andb_true_iff in Hkept as [Hhandlers Hbody].
      rewrite (erase_binds_only_kept D body Hbody). f_equal.
      induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
      simpl in Hhandlers. apply andb_true_iff in Hhandlers as [Hh Hrest].
      rewrite (erase_handler_binds_only_kept D h Hh), (IH Hrest).
      reflexivity.
  - destruct h as [params metadata handler]. simpl. intros Hkept.
    rewrite (erase_binds_only_kept D handler Hkept). reflexivity.
  - destruct binding as [id c | id | sym const]; simpl; intros Hkept.
    + apply andb_true_iff in Hkept as [Hid Hc].
      rewrite (erase_code_binds_only_kept D c Hc). auto.
    + auto.
    + auto.
  - destruct c as [params my_closure ret exn metadata body]. simpl.
    intros Hkept.
    rewrite (erase_binds_only_kept D body Hkept). reflexivity.
Qed.

Lemma erase_handlers_free_of (D : dropped)
    (handlers : list (continuation * continuation_handler)) :
  forallb (fun '(_, h) => handler_free_of D h) handlers = true ->
  handlers_avoid D handlers = true ∧ erase_handlers D handlers = handlers.
Proof.
  unfold handlers_avoid, erase_handlers.
  induction handlers as [ | [k h] handlers IH]; simpl; [ auto | ].
  intros Hfree. apply andb_true_iff in Hfree as [Hh Hrest].
  apply andb_true_iff in Hh as [Havoids Hkept].
  destruct (IH Hrest) as [Hav Hsame].
  rewrite Havoids, Hav, (erase_handler_binds_only_kept D h Hkept), Hsame.
  auto.
Qed.

Lemma env_free_of_avoids (D : dropped) (vars : env) :
  env_free_of D vars = true -> env_avoids D vars.
Proof.
  intros Hfree x v Hlookup.
  exact (lookup_forallb (value_avoids D) x v vars Hfree Hlookup).
Qed.

(* A continuation in which the names removed do not occur is its own
   erasure. *)
Fixpoint cont_free_of_erased (D : dropped) (c : cont_value) :
  cont_free_of D c = true -> cont_erased D c c.
Proof.
  destruct c
    as [h vars conts recursive | k | c extra
       | remaining remaining_arity return_arity return_to exn_to];
    simpl;
    [ | intros _; apply Outer_erased
    | intros Hfree; apply andb_true_iff in Hfree as [Hc Hextra];
      apply Extra_args_erased;
      [ exact (cont_free_of_erased D c Hc) | exact Hextra ]
    | intros Hfree; apply andb_true_iff in Hfree as [Hfree Hexn];
      apply andb_true_iff in Hfree as [Hremaining Hreturn];
      exact (Over_application_erased D remaining remaining_arity
               return_arity return_to return_to exn_to exn_to Hremaining
               (cont_free_of_erased D return_to Hreturn)
               (cont_free_of_erased D exn_to Hexn)) ].
  intros Hfree.
  apply andb_true_iff in Hfree as [Hfree Hrecursive].
  apply andb_true_iff in Hfree as [Hfree Hconts].
  apply andb_true_iff in Hfree as [Hh Hvars].
  apply andb_true_iff in Hh as [Havoids Hkept].
  destruct (erase_handlers_free_of D recursive Hrecursive) as [Hav Hsame].
  pose proof (Closure_erased D h vars conts conts recursive Havoids
    (env_free_of_avoids D vars Hvars)) as Hclosure.
  rewrite (erase_handler_binds_only_kept D h Hkept), Hsame in Hclosure.
  apply Hclosure; [ | exact Hav ].
  clear Hclosure.
  induction conts as [ | [k c] conts IH]; simpl; [ apply Forall2_nil | ].
  simpl in Hconts. apply andb_true_iff in Hconts as [Hc Hrest].
  apply Forall2_cons; [ | exact (IH Hrest) ].
  split; [ reflexivity | exact (cont_free_of_erased D c Hc) ].
Qed.

Lemma statics_free_of_erased (D : dropped) (statics : static_env) :
  statics_free_of D statics = true -> statics_erased D statics statics.
Proof.
  unfold statics_free_of. intros Hfree.
  apply andb_true_iff in Hfree as [Hcode Hvalues].
  assert (Hc :
    ∀ (id : code_id) (c : code),
      lookup id (static_code statics) = Some c -> code_free_of D c = true)
    by (intros id c; apply (lookup_forallb (code_free_of D)), Hcode).
  split; [ | split; [ | split ] ].
  - intros id _. destruct (lookup id (static_code statics)) as [c | ] eqn:Hid;
      [ | reflexivity ].
    simpl. specialize (Hc id c Hid).
    apply andb_true_iff in Hc as [_ Hkept].
    rewrite (erase_code_binds_only_kept D c Hkept). reflexivity.
  - intros id c _ Hid. specialize (Hc id c Hid).
    apply andb_true_iff in Hc as [Havoids _]. exact Havoids.
  - reflexivity.
  - intros s allocated _ Hs.
    pose proof (lookup_forallb (allocated_free_of D) s allocated _ Hvalues Hs)
      as Hallocated.
    destruct allocated; exact Hallocated.
Qed.

(* The relation of the simulation: a state and the state erased, where
   no value of the heap refers to a name removed. A step keeps the
   second part: it changes the heap only at a [Let]
   (FlambdaSemanticsSpec.step_heap_statement), and there to a heap of
   which the same holds. *)
Definition state_erased_free (D : dropped) (s s' : state) : Prop :=
  state_erased D s s' ∧ heap_free_of D (state_heap s) = true.

Lemma erased_free_step (D : dropped) (s s' : state) :
  state_erased_free D s s' ->
  result_simulated (state_erased_free D) (step s) (step s').
Proof.
  intros [Herased Hfree].
  pose proof (erased_step D s s' Herased Hfree) as Hsim.
  destruct (step s) as [t | k vs hp rd | ] eqn:Hstep;
    [ | exact Hsim | exact I ].
  destruct (step s') as [t' | k' vs' hp' rd' | ];
    [ | contradiction | contradiction ].
  split; [ exact Hsim | ].
  destruct (step_heap s t Hstep)
    as [-> | (x & kind & d & body & v & He & Hexec)];
    [ exact Hfree | ].
  destruct Herased
    as [e vars conts conts' statics statics' hp rd He' Hvars Hconts Hstatics].
  simpl in He, Hexec, Hfree. subst e. simpl in He'.
  apply andb_true_iff in He' as [Hd _].
  exact
    (proj2
       (proj2
          (exec_named_erased D statics statics' hp rd_ vars d Hstatics Hvars Hd
             Hfree)
          v (state_heap t) rd_ Hexec)).
Qed.

Theorem drop_statics_refines : drop_statics_refines_statement.
Proof.
  intros D e e'. unfold drop_statics.
  destruct (expr_avoids D e) eqn:He; [ | discriminate ].
  intros [= <-] vars conts statics hp rd Hfree Hheap.
  unfold envs_free_of in Hfree.
  apply andb_true_iff in Hfree as [Hfree Hstatics].
  apply andb_true_iff in Hfree as [Hvars Hconts].
  apply (simulation_refines (state_erased_free D) (erased_free_step D)).
  split; [ | exact Hheap ].
  apply State_erased;
    [ exact He | exact (env_free_of_avoids D vars Hvars) |
    | exact (statics_free_of_erased D statics Hstatics) ].
  induction conts as [ | [k c] conts IH]; simpl; [ apply Forall2_nil | ].
  simpl in Hconts. apply andb_true_iff in Hconts as [Hc Hrest].
  apply Forall2_cons; [ | exact (IH Hrest) ].
  split; [ reflexivity | exact (cont_free_of_erased D c Hc) ].
Qed.

Theorem drop_statics_program_refines :
  drop_statics_program_refines_statement.
Proof.
  intros D e e' Hdrop.
  exact
    (drop_statics_refines D e e' Hdrop [] [] empty_static_env empty_heap
       rd_ eq_refl eq_refl).
Qed.
