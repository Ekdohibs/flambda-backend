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

(* Proofs of the statements of FlambdaMyClosureSpec.v, with what
   FlambdaLiftProofs.v uses of the rewrite. Nothing here needs review.

   That a unit which passes the check calls the code through the symbol
   only is an invariant of its run ([state_ok]): no value holds a
   closure made by a [Let] of a set that has the code ([value_free]),
   every symbol bound to a closure of the code is the symbol
   ([allocated_ok]), and what is still to be run passes the check. The
   primitives keep the first, as they make their values out of their
   arguments and of the cells they read
   (FlambdaSemanticsProofs.exec_named_values_all): so the invariant
   says the first of the values in the cells of the heap too
   ([heap_free]). *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaNominalSpec FlambdaNominalProofs FlambdaAlphaProofs
  FlambdaWellFormedProofs FlambdaMyClosureSpec FlambdaEquivarianceTactic.
Import ListNotations.

(* ======================================================================== *)
(* The code is called through the symbol only                               *)
(* ======================================================================== *)

Section Only_closure.

Variable id : code_id.
Variable s : symbol.

(* The value holds no closure, made by a [Let], of a set that has the
   code. *)
Fixpoint value_free (v : value) : bool :=
  match v with
  | Value_const _ | Value_symbol _ | Value_boxed_number _
  | Value_location _ => true
  | Value_closure code slots =>
    negb (Pos.eqb code id) && forallb (fun '(_, v) => value_free v) slots
  | Value_closure_of_set _ decls slots =>
    negb (among id (map snd decls))
    && forallb (fun '(_, v) => value_free v) slots
  | Value_block _ fields | Value_immutable_array fields =>
    forallb value_free fields
  | Value_partial_application callee args _ =>
    value_free callee && forallb value_free args
  end.

Definition slots_free (slots : list (value_slot * value)) : bool :=
  forallb (fun '(_, v) => value_free v) slots.

Definition env_free (vars : env) : Prop :=
  ∀ (x : variable) (v : value), lookup x vars = Some v -> value_free v = true.

(* What a symbol is bound to: its parts hold no such closure, and if it
   is a closure of the code, the symbol is [s]. *)
Definition allocated_ok (sym : symbol)
    (allocated : statically_allocated_value) : bool :=
  match allocated with
  | Static_closure code slots =>
    (negb (Pos.eqb code id) || Pos.eqb sym s) && slots_free slots
  | Static_closure_of_set slot _ decls slots =>
    match lookup slot decls with
    | Some code => negb (Pos.eqb code id) || Pos.eqb sym s
    | None => true
    end
    && slots_free slots
  | Static_block _ fields => forallb value_free fields
  | Static_immutable_array elements => forallb value_free elements
  | Static_boxed_number v => value_free v
  | Static_string _ | Static_imported _ => true
  end.

Definition statics_ok (statics : static_env) : Prop :=
  (∀ (id' : code_id) (c : code),
    lookup id' (static_code statics) = Some c ->
    code_only_closure_of id s c = true)
  ∧ (∀ (sym : symbol) (allocated : statically_allocated_value),
      lookup sym (static_values statics) = Some allocated ->
      allocated_ok sym allocated = true).

#[local] Set Warnings "-register-all".
Inductive cont_ok : cont_value -> Prop :=
  | Outer_ok (k : continuation) : cont_ok (Outer k)
  | Closure_ok (h : continuation_handler) (vars : env) (conts : cont_env)
      (recursive : list (continuation * continuation_handler)) :
    handler_only_closure_of id s h = true ->
    env_free vars ->
    Forall (fun kc => cont_ok (snd kc)) conts ->
    forallb (fun '(_, h') => handler_only_closure_of id s h') recursive
    = true ->
    cont_ok (Closure h vars conts recursive)
  | Extra_args_ok (c : cont_value) (extra : list value) :
    cont_ok c ->
    forallb value_free extra = true ->
    cont_ok (Extra_args c extra)
  | Over_application_ok (remaining : list value)
      (remaining_arity return_arity : arity)
      (return_to exn_to : cont_value) :
    forallb value_free remaining = true ->
    cont_ok return_to ->
    cont_ok exn_to ->
    cont_ok
      (Over_application remaining remaining_arity return_arity return_to
         exn_to).

Definition conts_ok (conts : cont_env) : Prop :=
  Forall (fun kc => cont_ok (snd kc)) conts.

Definition state_ok (st : state) : Prop :=
  only_closure_of id s (state_expr st) = true
  ∧ env_free (state_vars st)
  ∧ conts_ok (state_conts st)
  ∧ statics_ok (state_statics st).

(* No value in a cell of the heap holds such a closure. A primitive that
   reads a cell gives a value out of it, so this is part of the
   invariant; it is kept apart from [state_ok] since only a [Let]
   changes the heap. *)
Definition heap_free (hp : heap) : Prop :=
  heap_values_all (fun v => value_free v = true) hp.

Lemma env_free_cons (x : variable) (v : value) (vars : env) :
  value_free v = true -> env_free vars -> env_free ((x, v) :: vars).
Proof.
  intros Hv Hvars y w. simpl.
  destruct (Pos.eqb y x); [ intros [= <-]; exact Hv | apply Hvars ].
Qed.

Lemma bind_params_free (params : list variable) :
  ∀ (vs : list value) (vars bound : env),
    forallb value_free vs = true ->
    env_free vars ->
    bind_params params vs vars = Some bound ->
    env_free bound.
Proof.
  induction params as [ | p params IH]; intros [ | v vs] vars bound; simpl;
    try discriminate.
  - intros _ Hvars [= <-]. exact Hvars.
  - intros Hvs Hvars. apply andb_true_iff in Hvs as [Hv Hvs].
    apply (IH vs _ bound Hvs), env_free_cons; assumption.
Qed.

Lemma eval_simple_free (vars : env) (a : simple) (v : value) :
  env_free vars -> eval_simple vars a = Some v -> value_free v = true.
Proof.
  intros Hvars. destruct a as [x | sym | c]; simpl;
    [ apply Hvars | intros [= <-]; reflexivity .. ].
Qed.

Lemma eval_simples_free (vars : env) (args : list simple) :
  env_free vars ->
  ∀ vs : list value,
    eval_simples vars args = Some vs -> forallb value_free vs = true.
Proof.
  intros Hvars. induction args as [ | a args IH]; intros vs; simpl.
  - intros [= <-]. reflexivity.
  - destruct (eval_simple vars a) as [v | ] eqn:Ha; [ | discriminate ].
    destruct (eval_simples vars args) as [rest | ]; [ | discriminate ].
    intros [= <-]. simpl.
    rewrite (eval_simple_free vars a v Hvars Ha), (IH rest eq_refl).
    reflexivity.
Qed.

Lemma eval_value_slots_free (vars : env)
    (slots : list (value_slot * simple)) :
  env_free vars ->
  ∀ evaluated : list (value_slot * value),
    eval_value_slots vars slots = Some evaluated ->
    slots_free evaluated = true.
Proof.
  intros Hvars. unfold slots_free.
  induction slots as [ | [slot a] slots IH]; intros evaluated; simpl.
  - intros [= <-]. reflexivity.
  - destruct (eval_simple vars a) as [v | ] eqn:Ha; [ | discriminate ].
    destruct (eval_value_slots vars slots) as [rest | ]; [ | discriminate ].
    intros [= <-]. simpl.
    rewrite (eval_simple_free vars a v Hvars Ha), (IH rest eq_refl).
    reflexivity.
Qed.

Lemma lookup_slot_free (slot : value_slot)
    (slots : list (value_slot * value)) (v : value) :
  slots_free slots = true -> lookup slot slots = Some v -> value_free v = true.
Proof.
  unfold slots_free.
  induction slots as [ | [slot' w] slots IH]; simpl; [ discriminate | ].
  intros Hfree. apply andb_true_iff in Hfree as [Hw Hslots].
  destruct (Pos.eqb slot slot');
    [ intros [= <-]; exact Hw | exact (IH Hslots) ].
Qed.

(* The code of a slot of a set is one of the codes of the set. *)
Lemma lookup_among_codes (slot : function_slot)
    (decls : list (function_slot * code_id)) (code : code_id) :
  lookup slot decls = Some code -> among code (map snd decls) = true.
Proof.
  intros Hlookup. apply among_In.
  exact (in_map snd decls (slot, code) (lookup_in slot code decls Hlookup)).
Qed.

(* A value that holds no such closure and is a closure has value slots
   that hold none; if it is one of the code, it is the symbol. *)
Lemma closure_of_free (statics : static_env) (v : value) (code : code_id)
    (slots : list (value_slot * value)) :
  statics_ok statics ->
  value_free v = true ->
  closure_of statics v = Some (code, slots) ->
  slots_free slots = true ∧ (code = id -> v = Value_symbol s).
Proof.
  intros [_ Hvalues].
  destruct v as [c | code' slots' | fslot decls slots' | sym | tag fields
    | elements | contents | loc | callee args args_arity];
    simpl; try discriminate.
  - intros Hv [= <- <-]. apply andb_true_iff in Hv as [Hcode Hslots].
    split; [ exact Hslots | ]. intros ->.
    rewrite Pos.eqb_refl in Hcode. discriminate.
  - intros Hv. apply andb_true_iff in Hv as [Hdecls Hslots].
    destruct (lookup fslot decls) as [code' | ] eqn:Hcode; [ | discriminate ].
    simpl. intros [= <- <-]. split; [ exact Hslots | ]. intros ->.
    rewrite (lookup_among_codes fslot decls id Hcode) in Hdecls. discriminate.
  - intros _.
    destruct (lookup sym (static_values statics))
      as [[code' slots' | fslot syms decls slots' | tag fields | contents
          | bytes | elements | ] | ] eqn:Hsym;
      try discriminate;
      pose proof (Hvalues sym _ Hsym) as Hok; simpl in Hok;
      apply andb_true_iff in Hok as [Hcode Hslots].
    + intros [= <- <-]. split; [ exact Hslots | ]. intros ->.
      rewrite Pos.eqb_refl in Hcode. simpl in Hcode.
      apply Pos.eqb_eq in Hcode. congruence.
    + destruct (lookup fslot decls) as [code' | ]; [ | discriminate ].
      simpl. intros [= <- <-]. split; [ exact Hslots | ]. intros ->.
      rewrite Pos.eqb_refl in Hcode. simpl in Hcode.
      apply Pos.eqb_eq in Hcode. congruence.
Qed.

Lemma closure_in_set_free (statics : static_env) (v : value)
    (at_slot : function_slot) (decls : list (function_slot * code_id))
    (slots : list (value_slot * value)) :
  statics_ok statics ->
  value_free v = true ->
  closure_in_set statics v = Some (at_slot, decls, slots) ->
  slots_free slots = true.
Proof.
  intros [_ Hvalues].
  destruct v as [c | code' slots' | fslot decls' slots' | sym | tag fields
    | elements | contents | loc | callee args args_arity];
    simpl; try discriminate.
  - intros Hv [= _ _ <-]. apply andb_true_iff in Hv as [_ Hslots].
    exact Hslots.
  - intros Hv. apply andb_true_iff in Hv as [_ Hslots].
    destruct (lookup fslot decls'); [ | discriminate ].
    simpl. intros [= _ _ <-]. exact Hslots.
  - intros _.
    destruct (lookup sym (static_values statics))
      as [[code' slots' | fslot syms decls' slots' | tag fields | contents
          | bytes | elements | ] | ] eqn:Hsym;
      try discriminate;
      pose proof (Hvalues sym _ Hsym) as Hok; simpl in Hok;
      apply andb_true_iff in Hok as [_ Hslots].
    + intros [= _ _ <-]. exact Hslots.
    + destruct (lookup fslot decls'); [ | discriminate ].
      simpl. intros [= _ _ <-]. exact Hslots.
Qed.

Lemma project_function_slot_free (statics : static_env)
    (move_from move_to : function_slot) (v w : value) :
  value_free v = true ->
  project_function_slot statics move_from move_to v = Some w ->
  value_free w = true.
Proof.
  intros Hv. unfold project_function_slot.
  assert (Hunnamed :
    (if Pos.eqb move_from unnamed_function_slot
     then if Pos.eqb move_to unnamed_function_slot then Some v else None
     else None)
    = Some w ->
    value_free w = true).
  { destruct (Pos.eqb move_from unnamed_function_slot); [ | discriminate ].
    destruct (Pos.eqb move_to unnamed_function_slot); [ | discriminate ].
    intros [= <-]. exact Hv. }
  destruct v as [c | code' slots' | fslot decls slots' | sym | tag fields
    | elements | contents | loc | callee args args_arity]; try discriminate.
  - exact Hunnamed.
  - destruct (Pos.eqb fslot move_from); [ | discriminate ].
    destruct (lookup fslot decls); [ | discriminate ].
    destruct (lookup move_to decls); [ | discriminate ].
    simpl. intros [= <-]. exact Hv.
  - destruct (lookup sym (static_values statics))
      as [[code' slots' | fslot syms decls slots' | tag fields | contents
          | bytes | elements | ] | ];
      try discriminate.
    + exact Hunnamed.
    + destruct (Pos.eqb fslot move_from); [ | discriminate ].
      destruct (lookup fslot decls); [ | discriminate ].
      destruct (lookup move_to decls); [ | discriminate ].
      simpl. destruct (lookup move_to syms); [ | discriminate ].
      simpl. intros [= <-]. reflexivity.
Qed.

(* The primitives make no closure. *)
Lemma eval_prim_free (statics : static_env) (p : prim) (vs : list value)
    (v : value) :
  statics_ok statics ->
  forallb value_free vs = true ->
  eval_prim statics p vs = Some v ->
  value_free v = true.
Proof.
  intros [_ Hvalues] Hvs.
  apply (eval_prim_preserves (fun w => value_free w = true)).
  - intros c. split; reflexivity.
  - intros tag fields. simpl. apply forallb_forall.
  - intros elements. simpl. apply forallb_forall.
  - intros sym tag fields _ Hsym.
    pose proof (Hvalues sym _ Hsym) as Hok. simpl in Hok.
    apply forallb_forall. exact Hok.
  - intros sym elements _ Hsym.
    pose proof (Hvalues sym _ Hsym) as Hok. simpl in Hok.
    apply forallb_forall. exact Hok.
  - apply forallb_forall. exact Hvs.
Qed.

Lemma eval_named_free (statics : static_env) (vars : env) (d : named)
    (v : value) :
  statics_ok statics ->
  env_free vars ->
  named_makes_no_closure_of id d = true ->
  eval_named statics vars d = Some v ->
  value_free v = true.
Proof.
  intros Hstatics Hvars.
  destruct d as [a | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure]; simpl; intros Hd.
  - exact (eval_simple_free vars a v Hvars).
  - destruct (eval_simples vars args) as [vs | ] eqn:Hargs;
      [ | discriminate ].
    exact (eval_prim_free statics p vs v Hstatics
             (eval_simples_free vars args Hvars vs Hargs)).
  - destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots;
      [ | discriminate ].
    intros [= <-]. simpl. rewrite Hd. simpl.
    exact (eval_value_slots_free vars slots Hvars vs Hslots).
  - destruct (lookup fslot decls); [ | discriminate ]. simpl.
    destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots;
      [ | discriminate ].
    intros [= <-]. simpl. rewrite Hd. simpl.
    exact (eval_value_slots_free vars slots Hvars vs Hslots).
  - destruct (eval_simple vars closure) as [w | ] eqn:Hclosure;
      [ | discriminate ].
    simpl.
    destruct (closure_of statics w) as [[called cslots] | ] eqn:Hc;
      [ | discriminate ].
    simpl.
    exact (lookup_slot_free slot cslots v
             (proj1 (closure_of_free statics w called cslots Hstatics
                       (eval_simple_free vars closure w Hvars Hclosure) Hc))).
  - destruct (eval_simple vars closure) as [w | ] eqn:Hclosure;
      [ | discriminate ].
    unfold project_value_slot.
    destruct (closure_in_set statics w) as [[[at_slot decls] cslots] | ]
      eqn:Hc;
      [ | discriminate ].
    destruct (Pos.eqb at_slot pfrom); [ | discriminate ].
    exact (lookup_slot_free slot cslots v
             (closure_in_set_free statics w at_slot decls cslots Hstatics
                (eval_simple_free vars closure w Hvars Hclosure) Hc)).
  - destruct (eval_simple vars closure) as [w | ] eqn:Hclosure;
      [ | discriminate ].
    exact (project_function_slot_free statics move_from move_to w v
             (eval_simple_free vars closure w Hvars Hclosure)).
Qed.

(* What a named expression that makes no closure of the code runs to
   holds no such closure, and the heap it leaves holds none. *)
Lemma exec_named_free (statics : static_env) (hp hp' : heap)
    (rd rd' : read_record) (vars : env) (d : named) (v : value) :
  statics_ok statics ->
  env_free vars ->
  heap_free hp ->
  named_makes_no_closure_of id d = true ->
  exec_named statics hp rd vars d = Some (v, hp', rd') ->
  value_free v = true ∧ heap_free hp'.
Proof.
  intros Hstatics Hvars Hheap Hd.
  apply (exec_named_values_all (fun w => value_free w = true));
    [ | | | | | | exact Hheap | | ].
  - intros c. split; reflexivity.
  - intros tag fields. simpl. apply forallb_forall.
  - intros elements. simpl. apply forallb_forall.
  - intros sym tag fields _ Hsym.
    pose proof (proj2 Hstatics sym _ Hsym) as Hok. simpl in Hok.
    apply forallb_forall. exact Hok.
  - intros sym elements _ Hsym.
    pose proof (proj2 Hstatics sym _ Hsym) as Hok. simpl in Hok.
    apply forallb_forall. exact Hok.
  - intros l. reflexivity.
  - intros p args vs _ Hvs. apply forallb_forall.
    exact (eval_simples_free vars args Hvars vs Hvs).
  - intros _ w. exact (eval_named_free statics vars d w Hstatics Hvars Hd).
Qed.

Lemma eval_static_const_ok (vars : env) (sym : symbol) (const : static_const)
    (allocated : statically_allocated_value) :
  env_free vars ->
  static_closure_of_is id s sym const = true ->
  eval_static_const vars const = Some allocated ->
  allocated_ok sym allocated = true.
Proof.
  intros Hvars.
  destruct const
    as [code slots | fslot syms decls slots | tag fields | contents | bytes
        | elements];
    simpl; intros Hconst.
  - destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots;
      [ | discriminate ].
    intros [= <-]. simpl. rewrite Hconst. simpl.
    exact (eval_value_slots_free vars slots Hvars vs Hslots).
  - destruct (lookup fslot decls) as [code | ] eqn:Hcode; [ | discriminate ].
    simpl.
    destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots;
      [ | discriminate ].
    intros [= <-]. simpl. rewrite Hcode, Hconst. simpl.
    exact (eval_value_slots_free vars slots Hvars vs Hslots).
  - destruct (eval_simples vars fields) as [vs | ] eqn:Hfields;
      [ | discriminate ].
    intros [= <-]. simpl.
    exact (eval_simples_free vars fields Hvars vs Hfields).
  - destruct (eval_simple vars contents) as [w | ] eqn:Hcontents;
      [ | discriminate ].
    intros [= <-]. simpl.
    exact (eval_simple_free vars contents w Hvars Hcontents).
  - intros [= <-]. reflexivity.
  - destruct (eval_simples vars elements) as [vs | ] eqn:Helements;
      [ | discriminate ].
    intros [= <-]. simpl.
    exact (eval_simples_free vars elements Hvars vs Helements).
Qed.

Lemma eval_static_bindings_ok (vars : env) :
  env_free vars ->
  ∀ (bindings : list static_binding) (statics extended : static_env),
    forallb (binding_only_closure_of id s) bindings = true ->
    statics_ok statics ->
    eval_static_bindings vars bindings statics = Some extended ->
    statics_ok extended.
Proof.
  intros Hvars.
  induction bindings as [ | binding bindings IH]; intros statics extended;
    simpl.
  - intros _ Hstatics [= <-]. exact Hstatics.
  - intros Hbindings [Hcode Hvalues].
    apply andb_true_iff in Hbindings as [Hbinding Hbindings].
    destruct binding as [id' c | id' | sym const]; simpl in Hbinding.
    + destruct (lookup id' (static_code statics)); [ discriminate | ].
      apply (IH _ _ Hbindings). split; [ | exact Hvalues ].
      intros id'' c'. simpl.
      destruct (Pos.eqb id'' id');
        [ intros [= <-]; exact Hbinding | apply Hcode ].
    + exact (IH _ _ Hbindings (conj Hcode Hvalues)).
    + destruct (lookup sym (static_values statics)); [ discriminate | ].
      destruct (eval_static_const vars const) as [allocated | ] eqn:Hconst;
        [ | discriminate ].
      simpl. apply (IH _ _ Hbindings). split; [ exact Hcode | ].
      intros sym' allocated'. simpl.
      destruct (Pos.eqb_spec sym' sym) as [-> | _];
        [ | apply Hvalues ].
      intros [= <-].
      exact (eval_static_const_ok vars sym const allocated Hvars Hbinding
               Hconst).
Qed.

Lemma lookup_ok (k : continuation) (conts : cont_env) (c : cont_value) :
  conts_ok conts -> lookup k conts = Some c -> cont_ok c.
Proof.
  induction 1 as [ | [k' c'] conts Hc _ IH]; simpl; [ discriminate | ].
  destruct (Pos.eqb k k'); [ intros [= <-]; exact Hc | exact IH ].
Qed.

Lemma resolve_ok (k : continuation) (conts : cont_env) :
  conts_ok conts -> cont_ok (resolve k conts).
Proof.
  intros Hconts. unfold resolve.
  destruct (lookup k conts) as [c | ] eqn:Hk;
    [ exact (lookup_ok k conts c Hconts Hk) | apply Outer_ok ].
Qed.

Lemma rec_closures_ok (vars : env) (conts : cont_env)
    (recursive : list (continuation * continuation_handler)) :
  forallb (fun '(_, h) => handler_only_closure_of id s h) recursive = true ->
  env_free vars ->
  conts_ok conts ->
  conts_ok (rec_closures recursive vars conts ++ conts).
Proof.
  intros Hrecursive Hvars Hconts. apply Forall_app. split; [ | exact Hconts ].
  unfold rec_closures. apply Forall_forall. intros [k c] Hin.
  apply in_map_iff in Hin as ([k' h] & [= <- <-] & Hin). simpl.
  apply Closure_ok; [ | exact Hvars | exact Hconts | exact Hrecursive ].
  rewrite forallb_forall in Hrecursive. exact (Hrecursive (k', h) Hin).
Qed.

Lemma with_extra_args_ok (c : cont_value) (extra : list value) :
  cont_ok c -> forallb value_free extra = true ->
  cont_ok (with_extra_args c extra).
Proof.
  intros Hc Hextra. destruct extra as [ | v extra]; [ exact Hc | ].
  apply Extra_args_ok; assumption.
Qed.

(* A partial application holds no such closure exactly when the value
   it calls and its arguments hold none. *)
Lemma value_free_partial (callee : value) (args : list value)
    (args_arity : arity) :
  value_free (Value_partial_application callee args args_arity) = true
  <-> value_free callee = true ∧ Forall (fun v => value_free v = true) args.
Proof. simpl. rewrite andb_true_iff, forallb_Forall. reflexivity. Qed.

(* The fields of a tuple that holds no such closure hold none. *)
Lemma tuple_fields_free (statics : static_env) :
  statics_ok statics ->
  P_fields (fun v => value_free v = true) statics.
Proof.
  intros Hstatics size tuple fields Htuple Hfields.
  apply (tuple_fields_forall _ statics size tuple fields);
    [ | exact Hfields ].
  intros field v Hv.
  apply (eval_prim_free statics
           (Unary (Block_load (tuple_access size) Immutable field)) [tuple] v
           Hstatics);
    [ simpl; rewrite Htuple; reflexivity | exact Hv ].
Qed.

(* A full call keeps the invariant. *)
Lemma call_code_ok (closure : value) (called : code_id) (c : code)
    (vs : list value) (args_arity return_arity : arity)
    (return_to exn_to : cont_value) (statics : static_env) (hp : heap)
      (rd : read_record)
      (st : state) :
  statics_ok statics ->
  lookup called (static_code statics) = Some c ->
  value_free closure = true ->
  forallb value_free vs = true ->
  cont_ok return_to ->
  cont_ok exn_to ->
  call_code closure c vs args_arity return_arity return_to exn_to statics hp rd
  = Step st ->
  state_ok st.
Proof.
  intros Hstatics Hcode Hclosure Hvs Hreturn Hexn Hstep.
  apply call_code_step in Hstep as (bound & _ & _ & _ & _ & Hbind & ->).
  split; [ | split; [ | split; [ | exact Hstatics ] ] ]; simpl.
  - pose proof (proj1 Hstatics called c Hcode) as Hc. destruct c. exact Hc.
  - apply (bind_params_free (code_params c) vs
             [(code_my_closure c, closure)] bound);
      [ exact Hvs | | exact Hbind ].
    apply env_free_cons; [ exact Hclosure | intros y w [=] ].
  - constructor; [ exact Hreturn | ].
    constructor; [ exact Hexn | constructor ].
Qed.

(* A call of a given shape keeps the invariant, given that what is done
   with a partial application keeps it. *)
Lemma run_call_ok (return_partial : value -> step_result) (kind : call_kind)
    (shape : option call_shape) (return_arity : arity)
    (return_to exn_to : cont_value) (statics : static_env) (hp : heap)
      (rd : read_record)
      (st : state) :
  statics_ok statics ->
  shape_calls statics kind shape ->
  shape_values_all (fun v => value_free v = true) shape ->
  cont_ok return_to ->
  cont_ok exn_to ->
  (∀ partial : value,
    value_free partial = true -> return_partial partial = Step st ->
    state_ok st) ->
  run_call return_partial shape return_arity return_to exn_to statics hp rd
  = Step st ->
  state_ok st.
Proof.
  intros Hstatics Hcalls Hvalues Hreturn Hexn Hpartial.
  destruct shape
    as [[closure c vs args_arity
        | closure c vs remaining args_arity remaining_arity | partial] | ];
    simpl in Hcalls, Hvalues |- *.
  - destruct Hcalls as (called & slots & _ & Hcode & _).
    destruct Hvalues as [Hclosure Hvs]. apply forallb_Forall in Hvs.
    exact (call_code_ok closure called c vs _ _ _ _ statics _ rd_ st Hstatics
             Hcode Hclosure Hvs Hreturn Hexn).
  - destruct Hcalls as (called & slots & _ & Hcode & _).
    destruct Hvalues as (Hclosure & Hvs & Hremaining).
    apply forallb_Forall in Hvs, Hremaining.
    apply (call_code_ok closure called c vs _ _ _ _ statics _ rd_ st Hstatics
             Hcode Hclosure Hvs);
      [ apply Over_application_ok; assumption | exact Hexn ].
  - exact (Hpartial partial Hvalues).
  - discriminate.
Qed.

(* Applying a continuation keeps the invariant. *)
Lemma apply_cont_value_ok (c : cont_value) :
  cont_ok c ->
  ∀ (vs : list value) (statics : static_env) (hp : heap) (rd : read_record)
    (st : state),
    forallb value_free vs = true ->
    statics_ok statics ->
    apply_cont_value c vs statics hp rd = Step st ->
    state_ok st.
Proof.
  induction 1
    as [ k | [params metadata handler] dvars dconts recursive Hh Hdvars
             Hdconts Hrecursive
       | c extra Hc IH Hextra
       | remaining remaining_arity return_arity return_to exn_to Hremaining
           Hreturn IHreturn Hexn _ ];
    intros vs statics hp rd st Hvs Hstatics.
  - discriminate.
  - simpl.
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
      [ | discriminate ].
    destruct (bind_params params vs dvars) as [bound | ] eqn:Hbind;
      [ | discriminate ].
    intros [= <-].
    split; [ exact Hh | ]. split; [ | split; [ | exact Hstatics ] ].
    + exact (bind_params_free params vs dvars bound Hvs Hdvars Hbind).
    + apply rec_closures_ok; assumption.
  - simpl. apply IH; [ | exact Hstatics ].
    rewrite forallb_app, Hvs, Hextra. reflexivity.
  - rewrite apply_cont_value_over_application.
    destruct vs as [ | v [ | v' vs]]; [ discriminate | | discriminate ].
    simpl in Hvs. rewrite andb_true_r in Hvs. unfold call_function.
    apply (run_call_ok _ Indirect_unknown_arity _ _ _ _ statics _ rd_ st
             Hstatics (shape_of_call_calls _ _ _ _ _ _));
      [ | exact Hreturn | exact Hexn | ].
    + apply (shape_of_call_values_all _ value_free_partial);
        [ exact (tuple_fields_free statics Hstatics) | exact Hvs
        | apply forallb_Forall; exact Hremaining ].
    + intros partial Hpartial. apply IHreturn; [ | exact Hstatics ].
      simpl. rewrite Hpartial. reflexivity.
Qed.

(* A step keeps the invariant. *)
Lemma ok_step_components (e : expr) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) (st : state) :
  only_closure_of id s e = true ->
  env_free vars ->
  conts_ok conts ->
  statics_ok statics ->
  heap_free hp ->
  step (State e vars conts statics hp rd) = Step st ->
  state_ok st ∧ heap_free (state_heap st).
Proof.
  intros He Hvars Hconts Hstatics Hheap Hstep. split.
  2: { destruct (step_heap _ _ Hstep)
         as [-> | (x & xkind & d & body & v & Hexpr & Hexec)];
         [ exact Hheap | ].
       simpl in Hexpr, Hexec. subst e. simpl in He.
       apply andb_true_iff in He as [Hd _].
       exact (proj2 (exec_named_free statics hp _ rd _ vars d v Hstatics
                       Hvars Hheap Hd Hexec)). }
  revert Hstep.
  destruct e
    as [ x xkind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    simpl in He.
  - (* Let *)
    apply andb_true_iff in He as [Hd Hbody]. rewrite step_let_exec.
    destruct (exec_named statics hp rd vars d) as [[[v hp'] rd'] | ] eqn:Hv;
      [ | discriminate ].
    destruct (kind_eqb (kind_of_value v) xkind); [ | discriminate ].
    intros [= <-].
    split; [ exact Hbody | ]. split; [ | split; assumption ].
    apply env_free_cons; [ | exact Hvars ].
    exact (proj1 (exec_named_free statics hp hp' rd rd' vars d v Hstatics
                    Hvars Hheap Hd Hv)).
  - (* Let_static *)
    apply andb_true_iff in He as [Hbindings Hbody]. rewrite step_let_static.
    destruct (eval_static_bindings vars bindings statics) as [extended | ]
      eqn:Hextended;
      [ | discriminate ].
    intros [= <-].
    split; [ exact Hbody | ]. split; [ exact Hvars | ].
    split; [ exact Hconts | ].
    exact (eval_static_bindings_ok vars Hvars bindings statics extended
             Hbindings Hstatics Hextended).
  - (* Let_cont_non_recursive *)
    apply andb_true_iff in He as [Hh Hbody].
    rewrite step_let_cont_non_recursive. intros [= <-].
    split; [ exact Hbody | ]. split; [ exact Hvars | ].
    split; [ | exact Hstatics ].
    constructor; [ | exact Hconts ]. simpl.
    apply Closure_ok; [ exact Hh | exact Hvars | exact Hconts | reflexivity ].
  - (* Let_cont_recursive *)
    apply andb_true_iff in He as [Hhandlers Hbody].
    rewrite step_let_cont_recursive.
    destruct (handlers_bind_statics handlers); [ discriminate | ].
    intros [= <-].
    split; [ exact Hbody | ]. split; [ exact Hvars | ].
    split; [ | exact Hstatics ].
    apply rec_closures_ok; assumption.
  - (* Apply *)
    rewrite step_apply.
    destruct (eval_simple vars (ap_callee ap)) as [callee | ] eqn:Hcallee;
      [ | discriminate ].
    destruct (eval_simples vars (ap_args ap)) as [vs | ] eqn:Hargs;
      [ | discriminate ].
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ]
      eqn:Hextra;
      [ | discriminate ].
    unfold call_function.
    assert (Hreturn : cont_ok (resolve (ap_continuation ap) conts))
      by exact (resolve_ok _ conts Hconts).
    assert (Hexn :
      cont_ok
        (with_extra_args (resolve (ap_exn_continuation ap) conts) extra)).
    { apply with_extra_args_ok; [ exact (resolve_ok _ conts Hconts) | ].
      exact (eval_simples_free vars _ Hvars extra Hextra). }
    apply (run_call_ok _ (ap_call_kind ap) _ _ _ _ statics _ _ st Hstatics
             (shape_of_call_calls _ _ _ _ _ _));
      [ | exact Hreturn | exact Hexn | ].
    + apply (shape_of_call_values_all _ value_free_partial);
        [ exact (tuple_fields_free statics Hstatics)
        | exact (eval_simple_free vars _ callee Hvars Hcallee) | ].
      apply forallb_Forall. exact (eval_simples_free vars _ Hvars vs Hargs).
    + intros partial Hpartial.
      apply (apply_cont_value_ok _ Hreturn); [ | exact Hstatics ].
      simpl. rewrite Hpartial. reflexivity.
  - (* Apply_cont *)
    rewrite step_apply_cont.
    destruct (eval_simples vars (ac_args ac)) as [vs | ] eqn:Hargs;
      [ | discriminate ].
    unfold apply_continuation.
    apply (apply_cont_value_ok _
      (resolve_ok (ac_continuation ac) conts Hconts));
      [ | exact Hstatics ].
    exact (eval_simples_free vars _ Hvars vs Hargs).
  - (* Switch *)
    rewrite step_switch.
    destruct (eval_simple vars scrutinee) as [v | ]; [ | discriminate ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | discriminate ].
    destruct (find_arm i arms) as [ac | ]; [ | discriminate ].
    intros [= <-]. split; [ reflexivity | ]. split; [ exact Hvars | ].
    split; assumption.
  - (* Invalid *)
    rewrite step_invalid. discriminate.
Qed.

Lemma ok_step (st st' : state) :
  state_ok st ->
  heap_free (state_heap st) ->
  step st = Step st' ->
  state_ok st' ∧ heap_free (state_heap st').
Proof.
  intros (He & Hvars & Hconts & Hstatics) Hheap. rewrite (state_eta st).
  exact (ok_step_components _ _ _ _ _ rd_ st' He Hvars Hconts Hstatics
           Hheap).
Qed.

Lemma ok_run (n : nat) :
  ∀ st st' : state,
    state_ok st ->
    heap_free (state_heap st) ->
    run n st = Step st' ->
    state_ok st'.
Proof.
  induction n as [ | n IH]; intros st st' Hok Hheap; simpl;
    [ intros [= <-]; exact Hok | ].
  destruct (step st) as [t | k vs | ] eqn:Hstep; try discriminate.
  destruct (ok_step st t Hok Hheap Hstep) as [Hok' Hheap'].
  exact (IH t st' Hok' Hheap').
Qed.

Lemma ok_initial (e : expr) :
  only_closure_of id s e = true ->
  state_ok (State e [] [] empty_static_env empty_heap no_reads).
Proof.
  intros He. split; [ exact He | ]. split; [ intros x v [=] | ].
  split; [ constructor | ]. split; [ intros id' c [=] | intros sym a [=] ].
Qed.

(* Where the invariant holds, a callee that is a closure of the code is
   the symbol. *)
Lemma ok_callee (st : state) (ap : apply_expr) (callee : value)
    (value_slots : list (value_slot * value)) :
  state_ok st ->
  eval_simple (state_vars st) (ap_callee ap) = Some callee ->
  closure_of (state_statics st) callee = Some (id, value_slots) ->
  callee = Value_symbol s.
Proof.
  intros (_ & Hvars & _ & Hstatics) Hcallee Hclosure.
  exact (proj2
           (closure_of_free (state_statics st) callee id value_slots Hstatics
              (eval_simple_free _ _ callee Hvars Hcallee) Hclosure)
           eq_refl).
Qed.

End Only_closure.

Theorem only_closure_sound : only_closure_sound_statement.
Proof.
  intros id s e n st ap callee value_slots He Hrun _.
  exact (ok_callee id s st ap callee value_slots
           (ok_run id s n _ st (ok_initial id s e He)
              (heap_values_all_empty _) Hrun)).
Qed.

(* ======================================================================== *)
(* The rewrite                                                              *)
(* ======================================================================== *)

(* It makes no closure, so the check is the same before it and after. *)
Fixpoint bind_my_closures_only_closure_expr
    (known : list (code_id * symbol)) (id : code_id) (s : symbol) (e : expr)
    {struct e} :
  only_closure_of id s (bind_my_closures known e) = only_closure_of id s e
with bind_my_closures_only_closure_handler
    (known : list (code_id * symbol)) (id : code_id) (s : symbol)
    (h : continuation_handler) {struct h} :
  handler_only_closure_of id s (bind_my_closures_in_handler known h)
  = handler_only_closure_of id s h
with bind_my_closures_only_closure_binding
    (known : list (code_id * symbol)) (id : code_id) (s : symbol)
    (binding : static_binding) {struct binding} :
  binding_only_closure_of id s (bind_my_closures_in_binding known binding)
  = binding_only_closure_of id s binding
with bind_my_closures_only_closure_code
    (known : list (code_id * symbol)) (id : code_id) (s : symbol)
    (id' : code_id) (c : code) {struct c} :
  code_only_closure_of id s (bind_my_closure known id' c)
  = code_only_closure_of id s c.
Proof.
  - destruct e
      as [ x xkind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try reflexivity.
    + rewrite bind_my_closures_only_closure_expr. reflexivity.
    + rewrite bind_my_closures_only_closure_expr. f_equal.
      induction bindings as [ | binding bindings IH]; simpl; [ reflexivity | ].
      rewrite bind_my_closures_only_closure_binding, IH. reflexivity.
    + rewrite bind_my_closures_only_closure_handler,
        bind_my_closures_only_closure_expr.
      reflexivity.
    + rewrite bind_my_closures_only_closure_expr. f_equal.
      induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
      rewrite bind_my_closures_only_closure_handler, IH. reflexivity.
  - destruct h as [params metadata handler]. simpl.
    apply bind_my_closures_only_closure_expr.
  - destruct binding as [id' c | id' | sym const]; simpl; try reflexivity.
    apply bind_my_closures_only_closure_code.
  - destruct c as [params my_closure ret exn metadata body]. simpl.
    destruct (lookup id' known) as [sym | ];
      [ destruct (among my_closure params) | ];
      simpl; apply bind_my_closures_only_closure_expr.
Qed.

Theorem bind_my_closures_only_closure :
  bind_my_closures_only_closure_statement.
Proof. exact bind_my_closures_only_closure_expr. Qed.

(* It keeps whether a term binds something static. *)
Fixpoint bind_my_closures_binds_statics (known : list (code_id * symbol))
    (e : expr) {struct e} :
  binds_statics (bind_my_closures known e) = binds_statics e
with bind_my_closures_handler_binds_statics
    (known : list (code_id * symbol)) (h : continuation_handler) {struct h} :
  handler_binds_statics (bind_my_closures_in_handler known h)
  = handler_binds_statics h.
Proof.
  - destruct e
      as [ x xkind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try reflexivity.
    + apply bind_my_closures_binds_statics.
    + rewrite bind_my_closures_binds_statics. f_equal.
      induction bindings as [ | binding bindings IH]; simpl; [ reflexivity | ].
      rewrite IH. destruct binding; reflexivity.
    + rewrite bind_my_closures_handler_binds_statics,
        bind_my_closures_binds_statics.
      reflexivity.
    + rewrite bind_my_closures_binds_statics. f_equal.
      induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
      rewrite bind_my_closures_handler_binds_statics, IH. reflexivity.
  - destruct h as [params metadata handler]. simpl.
    apply bind_my_closures_binds_statics.
Qed.

Lemma bind_my_closures_handlers_bind_statics
    (known : list (code_id * symbol))
    (handlers : list (continuation * continuation_handler)) :
  handlers_bind_statics
    (map (fun '(k, h) => (k, bind_my_closures_in_handler known h)) handlers)
  = handlers_bind_statics handlers.
Proof.
  unfold handlers_bind_statics.
  induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
  rewrite bind_my_closures_handler_binds_statics, IH. reflexivity.
Qed.

(* The static names that a group binds are those it bound. *)
Lemma bind_my_closures_symbols (known : list (code_id * symbol))
    (bindings : list static_binding) :
  flat_map FlambdaWellFormedSpec.symbols_of_binding
    (map (bind_my_closures_in_binding known) bindings)
  = flat_map FlambdaWellFormedSpec.symbols_of_binding bindings.
Proof.
  induction bindings as [ | binding bindings IH]; simpl; [ reflexivity | ].
  rewrite IH. destruct binding; reflexivity.
Qed.

(* What the rewrite does to code: nothing but to its body, which is
   rewritten and, where the code is given a symbol, begins by binding
   [my_closure] to it. *)
Lemma bind_my_closure_code (known : list (code_id * symbol)) (id : code_id)
    (c : code) :
  code_params (bind_my_closure known id c) = code_params c
  ∧ code_my_closure (bind_my_closure known id c) = code_my_closure c
  ∧ code_return_continuation (bind_my_closure known id c)
    = code_return_continuation c
  ∧ code_exn_continuation (bind_my_closure known id c)
    = code_exn_continuation c
  ∧ (code_body (bind_my_closure known id c)
     = bind_my_closures known (code_body c)
     ∨ ∃ s : symbol,
         lookup id known = Some s
         ∧ among (code_my_closure c) (code_params c) = false
         ∧ code_body (bind_my_closure known id c)
           = Let (code_my_closure c) Value (Simple (Symbol s))
               (bind_my_closures known (code_body c))).
Proof.
  destruct c as [params my_closure ret exn metadata body]. simpl.
  destruct (lookup id known) as [s | ]; [ | simpl; auto 6 ].
  destruct (among my_closure params) eqn:Hparam; simpl; [ auto 6 | ].
  repeat (split; [ reflexivity | ]). right. exists s. auto.
Qed.

(* It commutes with renaming, and so does the check: proved from the
   definitions by [equivariance] (FlambdaEquivarianceTactic.v). In code
   the test that [my_closure] is not a parameter has the same answer of
   the names renamed: a renaming keeps names apart. *)
Theorem bind_my_closures_equivariant :
  bind_my_closures_equivariant_statement.
Proof. equivariance. Qed.

Theorem only_closure_of_equivariant : only_closure_of_equivariant_statement.
Proof. equivariance. Qed.
