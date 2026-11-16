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


(* Proofs of FlambdaWellKindedSpec.v. Nothing here needs review. *)

From Stdlib Require Import ZArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaHeapSpec FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaWellKindedSpec.
From OxCamlFlambda2 Require FlambdaHeapProofs.
Import ListNotations.

(* ======================================================================== *)
(* Kinds and arities found                                                  *)
(* ======================================================================== *)

Lemma has_kind_eq (found : option flambda_kind) (kind : flambda_kind) :
  has_kind found kind = true <-> found = Some kind.
Proof.
  destruct found as [kind' | ]; simpl; [ | split; discriminate ].
  rewrite kind_eqb_eq. split; [ intros ->; reflexivity | congruence ].
Qed.

Lemma has_arity_eq (found : option (list flambda_kind))
    (expected : list flambda_kind) :
  has_arity found expected = true <-> found = Some expected.
Proof.
  destruct found as [found | ]; simpl; [ | split; discriminate ].
  rewrite kinds_eqb_eq. split; [ intros ->; reflexivity | congruence ].
Qed.

(* ======================================================================== *)
(* Parameters                                                               *)
(* ======================================================================== *)

(* Binding the kinds of the values is binding the values and taking the
   kinds. *)
Lemma bind_params_kinds (params : list variable) :
  ∀ (vs : list value) (vars vars' : env),
    bind_params params vs vars = Some vars' ->
    bind_param_kinds params (map kind_of_value vs) (kinds_of_env vars)
    = Some (kinds_of_env vars').
Proof.
  induction params as [ | x params IH]; intros [ | v vs] vars vars'; simpl;
    try discriminate.
  - intros [= <-]. reflexivity.
  - intros Hbind. exact (IH vs _ _ Hbind).
Qed.

Lemma bind_param_kinds_length (params : list variable) :
  ∀ (params_arity : list flambda_kind) (kinds kinds' : kind_env),
    bind_param_kinds params params_arity kinds = Some kinds' ->
    length params = length params_arity.
Proof.
  induction params as [ | x params IH]; intros [ | k params_arity] kinds
    kinds'; simpl; try discriminate; [ reflexivity | ].
  intros Hbind. f_equal. exact (IH _ _ _ Hbind).
Qed.

Lemma bind_params_defined (params : list variable) :
  ∀ (vs : list value) (vars : env),
    length params = length vs ->
    ∃ vars' : env, bind_params params vs vars = Some vars'.
Proof.
  induction params as [ | x params IH]; intros [ | v vs] vars; simpl;
    try discriminate.
  - intros _. exists vars. reflexivity.
  - intros [= Hlength]. exact (IH vs _ Hlength).
Qed.

Section Kinds.

Variable slots : slot_kinds.

#[local] Abbreviation value_wk := (value_well_kinded slots).
#[local] Abbreviation env_wk := (env_well_kinded slots).
#[local] Abbreviation heap_wk := (heap_well_kinded slots).

(* ======================================================================== *)
(* Values and environments                                                  *)
(* ======================================================================== *)

Lemma env_well_kinded_cons (x : variable) (v : value) (vars : env) :
  value_wk v = true -> env_wk vars = true -> env_wk ((x, v) :: vars) = true.
Proof. intros Hv Hvars. simpl. rewrite Hv. exact Hvars. Qed.

(* The test of a heap is FlambdaHeapSpec.heap_values_all of the test of a
   value. *)
Lemma heap_well_kinded_values_all (hp : heap) :
  heap_wk hp = true <-> heap_values_all (fun v => value_wk v = true) hp.
Proof.
  unfold heap_well_kinded, heap_values_all. rewrite forallb_forall. split.
  - intros Hheap l c v Hin Hv.
    exact (proj1 (forallb_forall _ _) (Hheap (l, c) Hin) v Hv).
  - intros Hall [l c] Hin. apply forallb_forall. intros v Hv.
    exact (Hall l c v Hin Hv).
Qed.

Lemma bind_params_well_kinded (params : list variable) :
  ∀ (vs : list value) (vars vars' : env),
    forallb value_wk vs = true ->
    env_wk vars = true ->
    bind_params params vs vars = Some vars' ->
    env_wk vars' = true.
Proof.
  induction params as [ | x params IH]; intros [ | v vs] vars vars'; simpl;
    try discriminate.
  - intros _ Hvars [= <-]. exact Hvars.
  - intros Hvs Hvars. apply andb_true_iff in Hvs as [Hv Hvs].
    apply (IH vs _ _ Hvs). exact (env_well_kinded_cons x v vars Hv Hvars).
Qed.

(* A [Simple] has the kind of its value, and a kind only if it has a
   value. *)
Lemma eval_simple_kind (vars : env) (s : simple) (v : value) :
  eval_simple vars s = Some v ->
  kind_of_simple (kinds_of_env vars) s = Some (kind_of_value v).
Proof.
  destruct s as [x | sym | c]; simpl.
  - intros Hx. unfold kinds_of_env. rewrite lookup_map_values, Hx.
    reflexivity.
  - intros [= <-]. reflexivity.
  - intros [= <-]. reflexivity.
Qed.

Lemma kind_of_simple_eval (vars : env) (s : simple) (k : flambda_kind) :
  kind_of_simple (kinds_of_env vars) s = Some k ->
  ∃ v : value, eval_simple vars s = Some v ∧ kind_of_value v = k.
Proof.
  destruct s as [x | sym | c]; simpl.
  - unfold kinds_of_env. rewrite lookup_map_values.
    destruct (lookup x vars) as [v | ]; simpl; [ | discriminate ].
    intros [= <-]. exists v. auto.
  - intros [= <-]. eexists. split; reflexivity.
  - intros [= <-]. eexists. split; reflexivity.
Qed.

Lemma kinds_of_simples_eval (vars : env) (args : list simple) :
  ∀ ks : list flambda_kind,
    kinds_of_simples (kinds_of_env vars) args = Some ks ->
    ∃ vs : list value,
      eval_simples vars args = Some vs ∧ map kind_of_value vs = ks.
Proof.
  induction args as [ | s args IH]; simpl; intros ks.
  - intros [= <-]. exists []. auto.
  - destruct (kind_of_simple (kinds_of_env vars) s) as [k | ] eqn:Hk;
      [ | discriminate ].
    destruct (kinds_of_simples (kinds_of_env vars) args) as [ks' | ];
      [ | discriminate ].
    intros [= <-].
    destruct (kind_of_simple_eval vars s k Hk) as (v & Hv & Hkind).
    destruct (IH ks' eq_refl) as (vs & Hvs & Hkinds).
    exists (v :: vs). rewrite Hv, Hvs. simpl. rewrite Hkind, Hkinds. auto.
Qed.

Lemma eval_simple_well_kinded (vars : env) (s : simple) (v : value) :
  env_wk vars = true -> eval_simple vars s = Some v -> value_wk v = true.
Proof.
  intros Hvars. destruct s as [x | sym | c]; simpl.
  - exact (lookup_forallb value_wk x v vars Hvars).
  - intros [= <-]. reflexivity.
  - intros [= <-]. reflexivity.
Qed.

Lemma eval_simples_well_kinded (vars : env) (args : list simple) :
  ∀ vs : list value,
    env_wk vars = true ->
    eval_simples vars args = Some vs ->
    forallb value_wk vs = true.
Proof.
  induction args as [ | s args IH]; simpl; intros vs Hvars.
  - intros [= <-]. reflexivity.
  - destruct (eval_simple vars s) as [v | ] eqn:Hv; [ | discriminate ].
    destruct (eval_simples vars args) as [vs' | ]; [ | discriminate ].
    intros [= <-]. simpl.
    rewrite (eval_simple_well_kinded vars s v Hvars Hv).
    exact (IH vs' Hvars eq_refl).
Qed.

(* The value slots of a closure made of well-kinded [Simple]s. *)
Lemma eval_value_slots_well_kinded (vars : env)
    (value_slots : list (value_slot * simple)) :
  ∀ (code : code_id) (vs : list (value_slot * value)),
    env_wk vars = true ->
    value_slots_well_kinded slots (kinds_of_env vars) value_slots = true ->
    eval_value_slots vars value_slots = Some vs ->
    value_wk (Value_closure code vs) = true.
Proof.
  induction value_slots as [ | [slot s] value_slots IH]; simpl;
    intros code vs Hvars.
  - intros _ [= <-]. reflexivity.
  - intros Hslots. apply andb_true_iff in Hslots as [Hslot Hslots].
    destruct (eval_simple vars s) as [v | ] eqn:Hv; [ | discriminate ].
    destruct (eval_value_slots vars value_slots) as [vs' | ];
      [ | discriminate ].
    intros [= <-]. simpl.
    destruct (lookup slot slots) as [kind | ]; [ | discriminate ].
    apply has_kind_eq in Hslot.
    rewrite (eval_simple_kind vars s v Hv) in Hslot. injection Hslot as <-.
    simpl. rewrite (proj2 (kind_eqb_eq _ _) eq_refl).
    rewrite (eval_simple_well_kinded vars s v Hvars Hv).
    exact (IH code vs' Hvars Hslots eq_refl).
Qed.

(* What a value slot of a well-kinded closure holds. *)
Lemma slot_well_kinded (code : code_id) (vs : list (value_slot * value))
    (slot : value_slot) (v : value) :
  value_wk (Value_closure code vs) = true ->
  lookup slot vs = Some v ->
  lookup slot slots = Some (kind_of_value v) ∧ value_wk v = true.
Proof.
  simpl. intros Hvs Hslot. apply lookup_in in Hslot.
  rewrite forallb_forall in Hvs. specialize (Hvs _ Hslot). simpl in Hvs.
  apply andb_true_iff in Hvs as [Hkind Hv]. apply has_kind_eq in Hkind.
  auto.
Qed.

(* ======================================================================== *)
(* Defining expressions and static constants                                *)
(* ======================================================================== *)

#[local] Abbreviation statics_wk := (statics_well_kinded slots).

(* A value that is a closure, or a symbol bound to one, has well-kinded
   value slots. *)
Lemma closure_of_well_kinded (statics : static_env) (v : value)
    (code : code_id) (vs : list (value_slot * value)) :
  value_wk v = true ->
  statics_wk statics = true ->
  closure_of statics v = Some (code, vs) ->
  value_wk (Value_closure code vs) = true.
Proof.
  intros Hv Hstatics.
  destruct v as [c | code' vs' | slot decls vs' | sym | tag fields | | c
    | l | pa_callee pa_args pa_arity];
    simpl; try discriminate.
  - intros [= <- <-]. exact Hv.
  - destruct (lookup slot decls); [ | discriminate ].
    intros [= <- <-]. exact Hv.
  - apply andb_true_iff in Hstatics as [_ Hvalues].
    destruct (lookup sym (static_values statics))
      as [[code' vs' | slot syms decls vs' | tag fields | contents | bytes
           | elements | ] | ] eqn:Hsym;
      try discriminate.
    + intros [= <- <-].
      exact
        (lookup_forallb (allocated_well_kinded slots) sym _ _ Hvalues Hsym).
    + destruct (lookup slot decls); [ | discriminate ].
      intros [= <- <-].
      exact
        (lookup_forallb (allocated_well_kinded slots) sym _ _ Hvalues Hsym).
Qed.

(* The same of a value read as a closure of a set. *)
Lemma closure_in_set_well_kinded (statics : static_env) (v : value)
    (slot : function_slot) (decls : list (function_slot * code_id))
    (vs : list (value_slot * value)) :
  value_wk v = true ->
  statics_wk statics = true ->
  closure_in_set statics v = Some (slot, decls, vs) ->
  value_wk (Value_closure_of_set slot decls vs) = true.
Proof.
  intros Hv Hstatics.
  destruct v as [c | code' vs' | slot' decls' vs' | sym | tag fields | | c
    | l | pa_callee pa_args pa_arity];
    simpl; try discriminate.
  - intros [= <- <- <-]. exact Hv.
  - destruct (lookup slot' decls'); [ | discriminate ].
    intros [= <- <- <-]. exact Hv.
  - apply andb_true_iff in Hstatics as [_ Hvalues].
    destruct (lookup sym (static_values statics))
      as [[code' vs' | slot' syms decls' vs' | tag fields | contents | bytes
           | elements | ] | ] eqn:Hsym;
      try discriminate.
    + intros [= <- <- <-].
      exact
        (lookup_forallb (allocated_well_kinded slots) sym _ _ Hvalues Hsym).
    + destruct (lookup slot' decls'); [ | discriminate ].
      intros [= <- <- <-].
      exact
        (lookup_forallb (allocated_well_kinded slots) sym _ _ Hvalues Hsym).
Qed.

(* The projection of a function slot is a closure of the same set, or a
   symbol: it has kind [Value], and it is well-kinded if what it is
   projected from is. *)
Lemma project_function_slot_well_kinded (statics : static_env)
    (move_from move_to : function_slot) (v w : value) :
  value_wk v = true ->
  project_function_slot statics move_from move_to v = Some w ->
  kind_of_value w = Value ∧ value_wk w = true.
Proof.
  intros Hv. unfold project_function_slot.
  destruct v as [c | code vs | slot decls vs | sym | tag fields | | c
    | l | pa_callee pa_args pa_arity];
    try discriminate.
  - destruct (Pos.eqb move_from unnamed_function_slot); [ | discriminate ].
    destruct (Pos.eqb move_to unnamed_function_slot); [ | discriminate ].
    intros [= <-]. split; [ reflexivity | exact Hv ].
  - destruct (Pos.eqb slot move_from); [ | discriminate ].
    destruct (lookup slot decls); [ | discriminate ].
    destruct (lookup move_to decls); [ | discriminate ].
    intros [= <-]. split; [ reflexivity | exact Hv ].
  - destruct (lookup sym (static_values statics))
      as [[code vs | slot syms decls vs | tag fields | contents | bytes
           | elements | ] | ];
      try discriminate.
    + destruct (Pos.eqb move_from unnamed_function_slot); [ | discriminate ].
      destruct (Pos.eqb move_to unnamed_function_slot); [ | discriminate ].
      intros [= <-]. split; reflexivity.
    + destruct (Pos.eqb slot move_from); [ | discriminate ].
      destruct (lookup slot decls); [ | discriminate ].
      destruct (lookup move_to decls); [ | discriminate ].
      destruct (lookup move_to syms); [ | discriminate ].
      intros [= <-]. split; reflexivity.
Qed.

(* A primitive makes its value out of its arguments
   (FlambdaPrimitiveSpec.eval_prim_preserves_statement). *)
Lemma eval_prim_well_kinded (statics : static_env) (p : prim)
    (vs : list value) (v : value) :
  statics_wk statics = true ->
  forallb value_wk vs = true ->
  eval_prim statics p vs = Some v ->
  value_wk v = true.
Proof.
  intros Hstatics Hvs.
  apply (eval_prim_preserves (fun v => value_wk v = true) statics p vs v).
  - intros c. split; reflexivity.
  - intros tag fields. simpl. apply forallb_forall.
  - intros elements. simpl. apply forallb_forall.
  - intros s tag fields _ Hlookup field Hin.
    apply andb_true_iff in Hstatics as [_ Hvalues].
    pose proof
      (lookup_forallb (allocated_well_kinded slots) s _ _ Hvalues Hlookup)
      as Hfields.
    simpl in Hfields. rewrite forallb_forall in Hfields.
    specialize (Hfields field Hin).
    apply andb_true_iff in Hfields as [_ Hfield]. exact Hfield.
  - intros s elements _ Hlookup element Hin.
    apply andb_true_iff in Hstatics as [_ Hvalues].
    pose proof
      (lookup_forallb (allocated_well_kinded slots) s _ _ Hvalues Hlookup)
      as Helements.
    simpl in Helements. rewrite forallb_forall in Helements.
    specialize (Helements element Hin).
    apply andb_true_iff in Helements as [_ Helement]. exact Helement.
  - rewrite forallb_forall in Hvs. exact Hvs.
Qed.

(* The value of a defining expression has the kind the check finds for
   it, and is well-kinded. *)
Lemma eval_named_kind (statics : static_env) (vars : env) (d : named)
    (k : flambda_kind) (v : value) :
  env_wk vars = true ->
  statics_wk statics = true ->
  kind_of_named slots (kinds_of_env vars) d = Some k ->
  eval_named statics vars d = Some v ->
  kind_of_value v = k ∧ value_wk v = true.
Proof.
  intros Hvars Hstatics.
  destruct d
    as [s | p args | code value_slots | fslot decls value_slots
       | slot closure | project_from slot closure
       | move_from move_to closure];
    simpl.
  - (* Simple *)
    intros Hk Hv. rewrite (eval_simple_kind vars s v Hv) in Hk.
    injection Hk as <-.
    split; [ reflexivity | exact (eval_simple_well_kinded vars s v Hvars Hv) ].
  - (* Prim *)
    destruct (kinds_of_simples (kinds_of_env vars) args) as [ks | ] eqn:Hks;
      [ | discriminate ].
    intros [= <-].
    destruct (kinds_of_simples_eval vars args ks Hks) as (vs & Hvs & _).
    rewrite Hvs. simpl. intros Hprim.
    split; [ exact (proj2 (eval_prim_kinds statics p vs v Hprim)) | ].
    exact (eval_prim_well_kinded statics p vs v Hstatics
      (eval_simples_well_kinded vars args vs Hvars Hvs) Hprim).
  - (* Set_of_closures *)
    destruct (value_slots_well_kinded slots (kinds_of_env vars) value_slots)
      eqn:Hslots; [ | discriminate ].
    intros [= <-].
    destruct (eval_value_slots vars value_slots) as [vs | ] eqn:Hvs;
      [ | discriminate ].
    intros [= <-]. split; [ reflexivity | ].
    exact (eval_value_slots_well_kinded vars value_slots code vs Hvars
      Hslots Hvs).
  - (* Closure_of_set *)
    destruct (value_slots_well_kinded slots (kinds_of_env vars) value_slots)
      eqn:Hslots; [ | discriminate ].
    intros [= <-].
    destruct (lookup fslot decls) as [code | ]; [ | discriminate ].
    destruct (eval_value_slots vars value_slots) as [vs | ] eqn:Hvs;
      [ | discriminate ].
    intros [= <-]. split; [ reflexivity | ].
    exact (eval_value_slots_well_kinded vars value_slots code vs Hvars
      Hslots Hvs).
  - (* Project_value_slot *)
    destruct (kind_of_simple (kinds_of_env vars) closure);
      [ | discriminate ].
    simpl. intros Hslot.
    destruct (eval_simple vars closure) as [vc | ] eqn:Hvc;
      [ | discriminate ].
    destruct (closure_of statics vc) as [[code vs] | ] eqn:Hclosure;
      [ | discriminate ].
    simpl. intros Hv.
    pose proof (closure_of_well_kinded statics vc code vs
      (eval_simple_well_kinded vars closure vc Hvars Hvc) Hstatics Hclosure)
      as Hwk.
    destruct (slot_well_kinded code vs slot v Hwk Hv) as [Hkind Hv'].
    rewrite Hkind in Hslot. injection Hslot as <-. auto.
  - (* Project_value_slot_from *)
    destruct (kind_of_simple (kinds_of_env vars) closure);
      [ | discriminate ].
    simpl. intros Hslot.
    destruct (eval_simple vars closure) as [vc | ] eqn:Hvc;
      [ | discriminate ].
    unfold project_value_slot.
    destruct (closure_in_set statics vc) as [[[at_slot decls] vs] | ]
      eqn:Hclosure; [ | discriminate ].
    destruct (Pos.eqb at_slot project_from); [ | discriminate ].
    intros Hv.
    pose proof (closure_in_set_well_kinded statics vc at_slot decls vs
      (eval_simple_well_kinded vars closure vc Hvars Hvc) Hstatics Hclosure)
      as Hwk.
    destruct (slot_well_kinded at_slot vs slot v Hwk Hv) as [Hkind Hv'].
    rewrite Hkind in Hslot. injection Hslot as <-. auto.
  - (* Project_function_slot *)
    destruct (kind_of_simple (kinds_of_env vars) closure);
      [ | discriminate ].
    simpl. intros [= <-].
    destruct (eval_simple vars closure) as [vc | ] eqn:Hvc;
      [ | discriminate ].
    exact (project_function_slot_well_kinded statics move_from move_to vc v
      (eval_simple_well_kinded vars closure vc Hvars Hvc)).
Qed.

(* The fields of a static block. *)
Lemma eval_fields_well_kinded (vars : env) (fields : list simple) :
  ∀ vs : list value,
    env_wk vars = true ->
    forallb (fun s => has_kind (kind_of_simple (kinds_of_env vars) s) Value)
      fields
    = true ->
    eval_simples vars fields = Some vs ->
    forallb (fun v => kind_eqb (kind_of_value v) Value && value_wk v) vs
    = true.
Proof.
  induction fields as [ | s fields IH]; simpl; intros vs Hvars.
  - intros _ [= <-]. reflexivity.
  - intros Hfields. apply andb_true_iff in Hfields as [Hfield Hfields].
    destruct (eval_simple vars s) as [v | ] eqn:Hv; [ | discriminate ].
    destruct (eval_simples vars fields) as [vs' | ]; [ | discriminate ].
    intros [= <-]. simpl.
    rewrite (eval_simple_kind vars s v Hv) in Hfield. simpl in Hfield.
    rewrite Hfield, (eval_simple_well_kinded vars s v Hvars Hv).
    exact (IH vs' Hvars Hfields eq_refl).
Qed.

Lemma eval_static_const_well_kinded (vars : env) (const : static_const)
    (allocated : statically_allocated_value) :
  env_wk vars = true ->
  static_const_well_kinded slots (kinds_of_env vars) const = true ->
  eval_static_const vars const = Some allocated ->
  allocated_well_kinded slots allocated = true.
Proof.
  intros Hvars.
  destruct const
    as [code value_slots | fslot syms decls value_slots | tag fields
       | contents | bytes | elements];
    simpl.
  - intros Hslots.
    destruct (eval_value_slots vars value_slots) as [vs | ] eqn:Hvs;
      [ | discriminate ].
    intros [= <-].
    exact (eval_value_slots_well_kinded vars value_slots code vs Hvars
      Hslots Hvs).
  - intros Hslots.
    destruct (lookup fslot decls) as [code | ]; [ | discriminate ].
    destruct (eval_value_slots vars value_slots) as [vs | ] eqn:Hvs;
      [ | discriminate ].
    intros [= <-].
    exact (eval_value_slots_well_kinded vars value_slots code vs Hvars
      Hslots Hvs).
  - intros Hfields.
    destruct (eval_simples vars fields) as [vs | ] eqn:Hvs;
      [ | discriminate ].
    intros [= <-].
    exact (eval_fields_well_kinded vars fields vs Hvars Hfields Hvs).
  - destruct (eval_simple vars contents) as [v | ] eqn:Hv;
      [ | discriminate ].
    rewrite (eval_simple_kind vars contents v Hv).
    intros Hkind [= <-]. exact Hkind.
  - intros _ [= <-]. reflexivity.
  - intros Helements.
    destruct (eval_simples vars elements) as [vs | ] eqn:Hvs;
      [ | discriminate ].
    intros [= <-].
    exact (eval_fields_well_kinded vars elements vs Hvars Helements Hvs).
Qed.

Lemma eval_static_bindings_well_kinded (vars : env)
    (bindings : list static_binding) :
  ∀ statics statics' : static_env,
    env_wk vars = true ->
    forallb (binding_well_kinded slots (kinds_of_env vars)) bindings
    = true ->
    statics_wk statics = true ->
    eval_static_bindings vars bindings statics = Some statics' ->
    statics_wk statics' = true.
Proof.
  induction bindings as [ | [id c | id | sym const] bindings IH];
    intros statics statics' Hvars; simpl.
  - intros _ Hstatics [= <-]. exact Hstatics.
  - intros Hbindings Hstatics.
    apply andb_true_iff in Hbindings as [Hc Hbindings].
    destruct (lookup id (static_code statics)); [ discriminate | ].
    apply (IH _ _ Hvars Hbindings).
    apply andb_true_iff in Hstatics as [Hcode Hvalues].
    unfold statics_well_kinded. simpl. rewrite Hc, Hcode, Hvalues.
    reflexivity.
  - intros Hbindings Hstatics. exact (IH _ _ Hvars Hbindings Hstatics).
  - intros Hbindings Hstatics.
    apply andb_true_iff in Hbindings as [Hconst Hbindings].
    destruct (lookup sym (static_values statics)); [ discriminate | ].
    destruct (eval_static_const vars const) as [allocated | ] eqn:Hallocated;
      simpl; [ | discriminate ].
    apply (IH _ _ Hvars Hbindings).
    apply andb_true_iff in Hstatics as [Hcode Hvalues].
    unfold statics_well_kinded. simpl.
    rewrite (eval_static_const_well_kinded vars const allocated Hvars Hconst
      Hallocated), Hcode, Hvalues.
    reflexivity.
Qed.

(* What a [Let] runs gives a well-kinded value and leaves a well-kinded
   heap. A primitive makes its value, and what it puts in the heap, out
   of its arguments and of what the heap holds
   (FlambdaSemanticsSpec.exec_named_preserves_statement); the other
   defining expressions do not look at the heap. *)
Lemma exec_named_well_kinded (statics : static_env) (hp hp' : heap)
    (rd rd' : read_record) (vars : env) (d : named) (k : flambda_kind)
    (v : value) :
  env_wk vars = true ->
  statics_wk statics = true ->
  heap_wk hp = true ->
  kind_of_named slots (kinds_of_env vars) d = Some k ->
  exec_named statics hp rd vars d = Some (v, hp', rd') ->
  value_wk v = true ∧ heap_wk hp' = true.
Proof.
  intros Hvars Hstatics Hheap Hk Hexec.
  apply heap_well_kinded_values_all in Hheap.
  assert (Hboth :
            value_wk v = true
            ∧ heap_values_all (fun w => value_wk w = true) hp').
  { apply (exec_named_values_all (fun w => value_wk w = true) statics hp hp'
             rd rd' vars d v);
      [ | | | | | | exact Hheap | | | exact Hexec ].
    - intros c. split; reflexivity.
    - intros tag fields. simpl. apply forallb_forall.
    - intros elements. simpl. apply forallb_forall.
    - intros s tag fields _ Hlookup field Hin.
      apply andb_true_iff in Hstatics as [_ Hvalues].
      pose proof
        (lookup_forallb (allocated_well_kinded slots) s _ _ Hvalues Hlookup)
        as Hfields.
      simpl in Hfields. rewrite forallb_forall in Hfields.
      specialize (Hfields field Hin).
      apply andb_true_iff in Hfields as [_ Hfield]. exact Hfield.
    - intros s elements _ Hlookup element Hin.
      apply andb_true_iff in Hstatics as [_ Hvalues].
      pose proof
        (lookup_forallb (allocated_well_kinded slots) s _ _ Hvalues Hlookup)
        as Helements.
      simpl in Helements. rewrite forallb_forall in Helements.
      specialize (Helements element Hin).
      apply andb_true_iff in Helements as [_ Helement]. exact Helement.
    - intros l. reflexivity.
    - intros p args vs _ Hvs arg Hin.
      pose proof (eval_simples_well_kinded vars args vs Hvars Hvs) as Hwk.
      rewrite forallb_forall in Hwk. exact (Hwk arg Hin).
    - intros _ w Hw.
      exact (proj2 (eval_named_kind statics vars d k w Hvars Hstatics Hk Hw)).
  }
  destruct Hboth as [Hv Hheap']. split; [ exact Hv | ].
  apply heap_well_kinded_values_all. exact Hheap'.
Qed.

(* And the value is of the kind that the check finds for the defining
   expression. A primitive is run on the heap, and what it gives is of
   the kind of its result whatever the heap holds
   (FlambdaHeapSpec.exec_prim_result_kind_statement); the other
   defining expressions have the value that [eval_named] gives them. *)
Lemma exec_named_kind (statics : static_env) (hp hp' : heap)
    (rd rd' : read_record) (vars : env) (d : named) (k : flambda_kind)
    (v : value) :
  env_wk vars = true ->
  statics_wk statics = true ->
  kind_of_named slots (kinds_of_env vars) d = Some k ->
  exec_named statics hp rd vars d = Some (v, hp', rd') ->
  kind_of_value v = k.
Proof.
  intros Hvars Hstatics Hk Hexec.
  destruct d as [s | p args | | | | | ];
    try (rewrite exec_named_not_prim in Hexec by (intros ? ?; discriminate);
         destruct (eval_named statics vars _) as [w | ] eqn:Hw;
           [ | discriminate ];
         injection Hexec as <- _ _;
         exact
           (proj1 (eval_named_kind statics vars _ k w Hvars Hstatics Hk Hw))).
  apply exec_named_prim_some in Hexec as (vs & _ & Hexec & _).
  simpl in Hk.
  destruct (kinds_of_simples (kinds_of_env vars) args) as [ks | ];
    [ | discriminate ].
  injection Hk as <-.
  exact (FlambdaHeapProofs.exec_prim_result_kind statics hp hp' p vs v Hexec).
Qed.

(* ======================================================================== *)
(* Continuations                                                            *)
(* ======================================================================== *)

Variable outer : arity_env.

#[local] Abbreviation cont_wk := (cont_well_kinded slots outer).
#[local] Abbreviation conts_wk := (conts_well_kinded slots outer).
#[local] Abbreviation handler_wk := (handler_well_kinded slots).

Lemma lookup_arities_of_conts (conts : cont_env) (k : continuation) :
  lookup k (arities_of_conts outer conts)
  = match lookup k conts with
    | Some c => Some (arity_of_cont outer c)
    | None => lookup k outer
    end.
Proof.
  unfold arities_of_conts. rewrite lookup_app, lookup_map_values.
  destruct (lookup k conts); reflexivity.
Qed.

(* What a continuation in scope stands for is well-kinded and has the
   arity it has in scope. *)
Lemma resolve_well_kinded (conts : cont_env) (k : continuation)
    (params_arity : list flambda_kind) :
  conts_wk conts = true ->
  lookup k (arities_of_conts outer conts) = Some params_arity ->
  cont_wk (resolve k conts) = true
  ∧ arity_of_cont outer (resolve k conts) = params_arity.
Proof.
  intros Hconts. rewrite lookup_arities_of_conts. unfold resolve.
  destruct (lookup k conts) as [c | ] eqn:Hk.
  - intros [= <-].
    split; [ | reflexivity ].
    exact (lookup_forallb cont_wk k c conts Hconts Hk).
  - intros Houter. simpl. unfold is_bound. rewrite Houter. auto.
Qed.

Lemma cont_well_kinded_closure (h : continuation_handler) (vars : env)
    (conts : cont_env)
    (recursive : list (continuation * continuation_handler)) :
  cont_wk (Closure h vars conts recursive)
  = env_wk vars && conts_wk conts
    && handler_wk (kinds_of_env vars)
         (handler_arities recursive ++ arities_of_conts outer conts) h
    && forallb
         (fun '(_, h) =>
            handler_wk (kinds_of_env vars)
              (handler_arities recursive ++ arities_of_conts outer conts) h)
         recursive.
Proof. reflexivity. Qed.

(* The arities of the closures of recursive handlers are those of the
   handlers. *)
Lemma arities_of_closures
    (handlers all : list (continuation * continuation_handler)) (vars : env)
    (conts rest : cont_env) :
  arities_of_conts outer
    (map (fun '(k, h) => (k, Closure h vars conts all)) handlers ++ rest)
  = handler_arities handlers ++ arities_of_conts outer rest.
Proof.
  unfold arities_of_conts, handler_arities, map_values.
  rewrite map_app, <- app_assoc. f_equal.
  induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
  rewrite IH. reflexivity.
Qed.

Lemma arities_of_rec_closures
    (handlers : list (continuation * continuation_handler)) (vars : env)
    (conts rest : cont_env) :
  arities_of_conts outer (rec_closures handlers vars conts ++ rest)
  = handler_arities handlers ++ arities_of_conts outer rest.
Proof. apply arities_of_closures. Qed.

Lemma conts_well_kinded_app (conts conts' : cont_env) :
  conts_wk (conts ++ conts') = conts_wk conts && conts_wk conts'.
Proof. apply forallb_app. Qed.

(* The closures of well-kinded recursive handlers are well-kinded. [all]
   is the whole list of handlers, kept apart from those gone through. *)
Lemma closures_well_kinded (vars : env) (conts : cont_env)
    (all handlers : list (continuation * continuation_handler)) :
  env_wk vars = true ->
  conts_wk conts = true ->
  forallb
    (fun '(_, h) =>
       handler_wk (kinds_of_env vars)
         (handler_arities all ++ arities_of_conts outer conts) h)
    all
  = true ->
  (∀ kh : continuation * continuation_handler, kh ∈ handlers -> kh ∈ all) ->
  conts_wk (map (fun '(k, h) => (k, Closure h vars conts all)) handlers)
  = true.
Proof.
  intros Hvars Hconts Hall.
  induction handlers as [ | [k h] handlers IH]; intros Hin;
    [ reflexivity | ].
  change
    (cont_wk (Closure h vars conts all)
     && conts_wk (map (fun '(k, h) => (k, Closure h vars conts all)) handlers)
     = true).
  rewrite cont_well_kinded_closure, Hvars, Hconts, Hall.
  pose proof Hall as Hh. rewrite forallb_forall in Hh.
  specialize (Hh (k, h) (Hin (k, h) (or_introl eq_refl))). simpl in Hh.
  rewrite Hh. simpl. apply IH. intros kh Hkh. apply Hin. right. exact Hkh.
Qed.

Lemma rec_closures_well_kinded (vars : env) (conts : cont_env)
    (handlers : list (continuation * continuation_handler)) :
  env_wk vars = true ->
  conts_wk conts = true ->
  forallb
    (fun '(_, h) =>
       handler_wk (kinds_of_env vars)
         (handler_arities handlers ++ arities_of_conts outer conts) h)
    handlers
  = true ->
  conts_wk (rec_closures handlers vars conts ++ conts) = true.
Proof.
  intros Hvars Hconts Hall. rewrite conts_well_kinded_app. unfold rec_closures.
  rewrite
    (closures_well_kinded vars conts handlers handlers Hvars Hconts Hall
       (fun kh Hkh => Hkh)).
  exact Hconts.
Qed.

(* ======================================================================== *)
(* More continuations in scope                                              *)
(* ======================================================================== *)

Lemma map_included_cons {A : Type} (name : positive) (a : A)
    (bindings bindings' : finite_map A) :
  map_included bindings bindings' ->
  map_included ((name, a) :: bindings) ((name, a) :: bindings').
Proof.
  intros Hincluded name' a'. simpl.
  destruct (Pos.eqb name' name); [ auto | apply Hincluded ].
Qed.

Lemma map_included_app {A : Type} (first bindings bindings' : finite_map A) :
  map_included bindings bindings' ->
  map_included (first ++ bindings) (first ++ bindings').
Proof.
  intros Hincluded name a. rewrite !lookup_app.
  destruct (lookup name first); [ auto | apply Hincluded ].
Qed.

Lemma map_included_app_r {A : Type} (bindings rest : finite_map A) :
  map_included bindings (bindings ++ rest).
Proof.
  intros name a Hlookup. rewrite lookup_app, Hlookup. reflexivity.
Qed.

Lemma apply_cont_well_kinded_weaken (kinds : kind_env)
    (conts conts' : arity_env) (ac : apply_cont) :
  map_included conts conts' ->
  apply_cont_well_kinded kinds conts ac = true ->
  apply_cont_well_kinded kinds conts' ac = true.
Proof.
  intros Hincluded. unfold apply_cont_well_kinded. intros Hac.
  apply andb_true_iff in Hac as [Htrap Hac]. apply andb_true_iff. split.
  - unfold trap_action_in_scope in *.
    destruct (ac_trap_action ac) as [action | ]; [ | reflexivity ].
    unfold is_bound in *.
    destruct (lookup (trap_action_handler action) conts) as [found | ]
      eqn:Hk;
      [ | discriminate ].
    rewrite (Hincluded _ _ Hk). reflexivity.
  - destruct (lookup (ac_continuation ac) conts) as [params_arity | ]
      eqn:Hk;
      [ | discriminate ].
    rewrite (Hincluded _ _ Hk). exact Hac.
Qed.

Lemma has_arity_weaken (conts conts' : arity_env) (k : continuation)
    (expected : list flambda_kind) :
  map_included conts conts' ->
  has_arity (lookup k conts) expected = true ->
  has_arity (lookup k conts') expected = true.
Proof.
  intros Hincluded Hk. apply has_arity_eq in Hk.
  rewrite (Hincluded _ _ Hk). apply has_arity_eq. reflexivity.
Qed.

(* A well-kinded expression stays so with more continuations in scope. *)
Fixpoint well_kinded_weaken (e : expr) :
  ∀ (kinds : kind_env) (conts conts' : arity_env),
    map_included conts conts' ->
    well_kinded slots kinds conts e = true ->
    well_kinded slots kinds conts' e = true.
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    intros kinds conts conts' Hincluded; simpl.
  - intros H. apply andb_true_iff in H as [Hd Hbody]. rewrite Hd.
    exact (well_kinded_weaken body _ _ _ Hincluded Hbody).
  - intros H. apply andb_true_iff in H as [Hbindings Hbody].
    rewrite Hbindings. exact (well_kinded_weaken body _ _ _ Hincluded Hbody).
  - destruct h as [params metadata handler]. simpl. intros H.
    apply andb_true_iff in H as [Hh Hbody].
    apply andb_true_iff in Hh as [Hdistinct Hhandler]. rewrite Hdistinct.
    destruct (bind_param_kinds params (hm_params_kinds metadata) kinds);
      [ | discriminate ].
    rewrite (well_kinded_weaken handler _ _ _ Hincluded Hhandler).
    exact
      (well_kinded_weaken body _ _ _
         (map_included_cons k (hm_params_kinds metadata) _ _ Hincluded)
         Hbody).
  - intros H. apply andb_true_iff in H as [Hhandlers Hbody].
    pose proof
      (map_included_app (handler_arities handlers) _ _ Hincluded) as Hrec.
    rewrite (well_kinded_weaken body _ _ _ Hrec Hbody).
    rewrite andb_true_r. clear Hbody. revert Hhandlers Hrec.
    generalize (handler_arities handlers ++ conts).
    generalize (handler_arities handlers ++ conts').
    intros rec' rec Hhandlers Hrec.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      simpl in *; [ reflexivity | ].
    apply andb_true_iff in Hhandlers as [Hh Hhandlers].
    apply andb_true_iff in Hh as [Hdistinct Hhandler]. rewrite Hdistinct.
    destruct (bind_param_kinds params (hm_params_kinds metadata) kinds);
      [ | discriminate ].
    rewrite (well_kinded_weaken handler _ _ _ Hrec Hhandler).
    exact (IH Hhandlers).
  - unfold apply_well_kinded. intros H.
    apply andb_true_iff in H as [H Hexn]. apply andb_true_iff in H as [H Hk].
    rewrite H, (has_arity_weaken _ _ _ _ Hincluded Hk),
      (has_arity_weaken _ _ _ _ Hincluded Hexn).
    reflexivity.
  - exact (apply_cont_well_kinded_weaken kinds conts conts' ac Hincluded).
  - intros H. apply andb_true_iff in H as [Hscrutinee Harms].
    rewrite Hscrutinee. simpl. rewrite forallb_forall in *. intros arm Harm.
    exact (apply_cont_well_kinded_weaken kinds conts conts' (snd arm)
      Hincluded (Harms arm Harm)).
  - reflexivity.
Qed.

(* ======================================================================== *)
(* A step keeps a state well-kinded                                         *)
(* ======================================================================== *)

(* The body of well-kinded code is well-kinded under its parameters. *)
Lemma code_body_well_kinded (c : code) (kinds : kind_env) :
  code_well_kinded slots c = true ->
  bind_param_kinds (code_params c) (code_params_kinds c)
    [(code_my_closure c, Value)]
  = Some kinds ->
  well_kinded slots kinds
    [(code_return_continuation c, code_result_kinds c);
     (code_exn_continuation c, [Value])]
    (code_body c)
  = true.
Proof.
  destruct c as [params my_closure ret exn metadata body].
  unfold code_params_kinds, code_result_kinds. simpl. intros Hc Hkinds.
  apply andb_true_iff in Hc as [_ Hc]. rewrite Hkinds in Hc. exact Hc.
Qed.

#[local] Abbreviation state_wk := (state_well_kinded slots outer).

Lemma state_well_kinded_intro (e : expr) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) :
  well_kinded slots (kinds_of_env vars) (arities_of_conts outer conts) e
  = true ->
  env_wk vars = true ->
  conts_wk conts = true ->
  statics_wk statics = true ->
  heap_wk hp = true ->
  state_wk (State e vars conts statics hp rd) = true.
Proof.
  intros He Hvars Hconts Hstatics Hheap. unfold state_well_kinded.
  simpl_state. rewrite He, Hvars, Hconts, Hstatics, Hheap. reflexivity.
Qed.

Lemma state_well_kinded_elim (s : state) :
  state_wk s = true ->
  well_kinded slots (kinds_of_env (state_vars s))
    (arities_of_conts outer (state_conts s)) (state_expr s)
  = true
  ∧ env_wk (state_vars s) = true
  ∧ conts_wk (state_conts s) = true
  ∧ statics_wk (state_statics s) = true
  ∧ heap_wk (state_heap s) = true.
Proof.
  unfold state_well_kinded. intros Hs.
  apply andb_true_iff in Hs as [Hs Hheap].
  apply andb_true_iff in Hs as [Hs Hstatics].
  apply andb_true_iff in Hs as [Hs Hconts].
  apply andb_true_iff in Hs as [He Hvars]. auto.
Qed.

(* The arguments of a well-kinded [Apply_cont] or [Apply] have values, of
   the kinds expected and well-kinded. *)
Lemma eval_args (vars : env) (args : list simple)
    (expected : list flambda_kind) :
  env_wk vars = true ->
  has_arity (kinds_of_simples (kinds_of_env vars) args) expected = true ->
  ∃ vs : list value,
    eval_simples vars args = Some vs
    ∧ map kind_of_value vs = expected
    ∧ forallb value_wk vs = true.
Proof.
  intros Hvars Hargs. apply has_arity_eq in Hargs.
  destruct (kinds_of_simples_eval vars args expected Hargs)
    as (vs & Hvs & Hkinds).
  exists vs. split; [ exact Hvs | ]. split; [ exact Hkinds | ].
  exact (eval_simples_well_kinded vars args vs Hvars Hvs).
Qed.

(* The exception continuation that a call gives its callee: the caller's
   takes the exception followed by the extra arguments, and with those
   kept it takes the exception alone. *)
Lemma with_extra_args_well_kinded (c : cont_value) (extra : list value)
    (kinds : list flambda_kind) :
  cont_wk c = true ->
  arity_of_cont outer c = Value :: kinds ->
  map kind_of_value extra = kinds ->
  forallb value_wk extra = true ->
  cont_wk (with_extra_args c extra) = true
  ∧ arity_of_cont outer (with_extra_args c extra) = [Value].
Proof.
  intros Hc Harity Hkinds Hextra.
  destruct extra as [ | v extra]; [ simpl in *; subst kinds; auto | ].
  remember (v :: extra) as given eqn:Hgiven.
  assert (Hlength : (length (Value :: kinds) - length given = 1)%nat).
  { rewrite <- Hkinds. cbn [length]. rewrite length_map. lia. }
  assert (Hwith : with_extra_args c given = Extra_args c given)
    by (rewrite Hgiven; reflexivity).
  rewrite Hwith. cbn [cont_well_kinded arity_of_cont].
  rewrite Harity, Hlength, Hc, Hextra. cbn [skipn firstn].
  rewrite Hkinds, kinds_eqb_refl. auto.
Qed.

(* A well-kinded value, as a property: that of a partial application is
   that of the value it calls and of its arguments. *)
Lemma value_well_kinded_partial (callee : value) (args : list value)
    (args_arity : arity) :
  value_wk (Value_partial_application callee args args_arity) = true
  <-> value_wk callee = true ∧ Forall (fun v => value_wk v = true) args.
Proof. simpl. rewrite andb_true_iff, forallb_Forall. reflexivity. Qed.

(* The fields of a well-kinded tuple are well-kinded, where what the
   symbols are bound to is. *)
Lemma tuple_fields_well_kinded (statics : static_env) :
  statics_wk statics = true ->
  P_fields (fun v => value_wk v = true) statics.
Proof.
  intros Hstatics size tuple fields Htuple Hfields.
  apply (tuple_fields_of_blocks _ statics size tuple fields);
    [ | exact Hfields ].
  intros tag block Hblock. apply forallb_Forall.
  destruct (block_of_cases _ _ _ _ Hblock) as [-> | (s & -> & Hs)].
  - exact Htuple.
  - apply andb_true_iff in Hstatics as [_ Hvalues].
    pose proof
      (lookup_forallb (allocated_well_kinded slots) s _ _ Hvalues Hs)
      as Hallocated.
    simpl in Hallocated. rewrite forallb_forall in Hallocated |- *.
    intros v Hin. specialize (Hallocated v Hin).
    apply andb_true_iff in Hallocated as [_ Hv]. exact Hv.
Qed.

(* What a call, or the application of a continuation, comes to in a
   well-kinded state: a well-kinded state, or an exit by a continuation
   of [outer] with well-kinded values of the kinds of its parameters. *)
Definition result_wk (result : step_result) : Prop :=
  match result with
  | Step s' => state_wk s' = true
  | Exit k vs _ _ =>
    lookup k outer = Some (map kind_of_value vs)
    ∧ forallb value_wk vs = true
  | Undefined => True
  end.

(* A full call of well-kinded code with well-kinded values, returning
   to well-kinded continuations of the arities of the call. *)
Lemma call_code_well_kinded (closure : value) (called : code_id)
    (value_slots : list (value_slot * value)) (c : code) (vs : list value)
    (args_arity return_arity : arity) (return_to exn_to : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record) :
  closure_of statics closure = Some (called, value_slots) ->
  lookup called (static_code statics) = Some c ->
  value_wk closure = true ->
  forallb value_wk vs = true ->
  cont_wk return_to = true ->
  arity_of_cont outer return_to = unarize return_arity ->
  cont_wk exn_to = true ->
  arity_of_cont outer exn_to = [Value] ->
  statics_wk statics = true ->
  heap_wk hp = true ->
  result_wk
    (call_code closure c vs args_arity return_arity return_to exn_to statics
       hp rd).
Proof.
  intros Hclosure Hcode Hclosure_wk Hvs_wk Hreturn_wk Hreturn_to Hexn_wk
    Hexn_to Hstatics Hheap.
  destruct
    (call_code closure c vs args_arity return_arity return_to exn_to statics hp
       rd)
    as [s' | k values | ] eqn:Hstep; [ | | exact I ].
  2:
    { destruct
     (call_code_cases closure c vs args_arity return_arity statics hp rd)
         as [Hundefined | [callee_vars Hsteps]];
         [ rewrite Hundefined in Hstep | rewrite Hsteps in Hstep ];
         discriminate. }
  (* The call tests that the arguments have the kinds of its arity. *)
  apply call_code_step in Hstep
    as (callee_vars & _ & Hkinds & Hargs_arity & Hreturn_arity & Hbind & ->).
  pose proof (closure_of_kind _ closure called value_slots Hclosure)
    as Hclosure_kind.
  (* The code called is well-kinded. *)
  pose proof Hstatics as Hc. apply andb_true_iff in Hc as [Hc _].
  apply (lookup_forallb (code_well_kinded slots) called c _) in Hc;
    [ | exact Hcode ].
  pose proof (bind_params_kinds _ vs _ callee_vars Hbind) as Hkinds_bound.
  rewrite Hkinds, (unarize_per_parameter_eq _ _ Hargs_arity)
    in Hkinds_bound.
  change (kinds_of_env [(code_my_closure c, closure)])
    with [(code_my_closure c, kind_of_value closure)] in Hkinds_bound.
  rewrite Hclosure_kind in Hkinds_bound.
  pose proof (code_body_well_kinded c _ Hc Hkinds_bound) as Hbody.
  apply state_well_kinded_intro; [ | | | exact Hstatics | exact Hheap ].
  - refine (well_kinded_weaken (code_body c) _ _ _ _ Hbody).
    unfold arities_of_conts. simpl.
    rewrite Hreturn_to, Hexn_to, Hreturn_arity.
    exact
      (map_included_app_r
         [(code_return_continuation c, code_result_kinds c);
          (code_exn_continuation c, [Value])]
         outer).
  - refine
      (bind_params_well_kinded (code_params c) vs _ callee_vars Hvs_wk _
         Hbind).
    simpl. rewrite Hclosure_wk. reflexivity.
  - unfold conts_well_kinded. simpl. rewrite Hreturn_wk, Hexn_wk.
    reflexivity.
Qed.

(* A call of a shape whose values are well-kinded, given what is done
   with a partial application: a well-kinded value of kind [Value], and
   the call then expects one [Value]. *)
Lemma run_call_well_kinded (return_partial : value -> step_result)
    (kind : call_kind) (shape : option call_shape) (return_arity : arity)
    (return_to exn_to : cont_value) (statics : static_env) (hp : heap)
      (rd : read_record) :
  shape_calls statics kind shape ->
  shape_values_all (fun v => value_wk v = true) shape ->
  shape_arities return_arity shape ->
  cont_wk return_to = true ->
  arity_of_cont outer return_to = unarize return_arity ->
  cont_wk exn_to = true ->
  arity_of_cont outer exn_to = [Value] ->
  statics_wk statics = true ->
  heap_wk hp = true ->
  (∀ partial : value,
    value_wk partial = true ->
    kind_of_value partial = Value ->
    return_arity = [Singleton Value] ->
    result_wk (return_partial partial)) ->
  result_wk
    (run_call return_partial shape return_arity return_to exn_to statics hp rd).
Proof.
  intros Hcalls Hvalues Harities Hreturn_wk Hreturn_to Hexn_wk Hexn_to
    Hstatics Hheap Hpartial.
  destruct shape
    as [[closure c vs args_arity
        | closure c vs remaining args_arity remaining_arity | partial] | ];
    simpl in Hcalls, Hvalues, Harities |- *.
  - destruct Hcalls as (called & value_slots & Hclosure & Hcode & _).
    destruct Hvalues as [Hclosure_wk Hvs]. apply forallb_Forall in Hvs.
    exact (call_code_well_kinded closure called value_slots c vs _ _ _ _
             statics _ rd_ Hclosure Hcode Hclosure_wk Hvs Hreturn_wk Hreturn_to
             Hexn_wk Hexn_to Hstatics Hheap).
  - destruct Hcalls as (called & value_slots & Hclosure & Hcode & _).
    destruct Hvalues as (Hclosure_wk & Hvs & Hremaining).
    apply forallb_Forall in Hvs, Hremaining.
    apply (call_code_well_kinded closure called value_slots c vs _ _ _ _
             statics _ rd_ Hclosure Hcode Hclosure_wk Hvs);
      try assumption.
    + simpl. rewrite Hremaining, Hreturn_wk, Hexn_wk, Hreturn_to, Hexn_to.
      rewrite ?kinds_eqb_refl. reflexivity.
    + rewrite Harities. reflexivity.
  - destruct Harities as [Hreturn_arity Hkind].
    exact (Hpartial partial Hvalues Hkind Hreturn_arity).
  - exact I.
Qed.

(* Applying a well-kinded continuation to well-kinded values of the
   kinds of its parameters. *)
Lemma apply_cont_value_well_kinded (c : cont_value) :
  ∀ (vs : list value) (statics : static_env) (hp : heap) (rd : read_record),
    cont_wk c = true ->
    map kind_of_value vs = arity_of_cont outer c ->
    forallb value_wk vs = true ->
    statics_wk statics = true ->
    heap_wk hp = true ->
    result_wk (apply_cont_value c vs statics hp rd).
Proof.
  induction c
    as [[params metadata handler] def_vars def_conts recursive | k
       | c IH extra
       | remaining remaining_arity return_arity return_to exn_to IHreturn _]
    using cont_value_over_ind;
    intros vs statics hp rd Hc Hkinds Hvs_wk Hstatics Hheap.
  - cbn [apply_cont_value].
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
      [ | exact I ].
    destruct (bind_params params vs def_vars) as [handler_vars | ] eqn:Hbind;
      [ | exact I ].
    simpl in Hkinds.
    rewrite cont_well_kinded_closure in Hc.
    apply andb_true_iff in Hc as [Hc Hrecursive].
    apply andb_true_iff in Hc as [Hc Hh].
    apply andb_true_iff in Hc as [Hdef_vars Hdef_conts].
    simpl in Hh. apply andb_true_iff in Hh as [_ Hh].
    pose proof (bind_params_kinds params vs def_vars handler_vars Hbind)
      as Hkinds_bound.
    rewrite Hkinds in Hkinds_bound. rewrite Hkinds_bound in Hh.
    apply state_well_kinded_intro; [ | | | exact Hstatics | exact Hheap ].
    + rewrite arities_of_rec_closures. exact Hh.
    + exact (bind_params_well_kinded params vs def_vars handler_vars Hvs_wk
        Hdef_vars Hbind).
    + exact (rec_closures_well_kinded def_vars def_conts recursive Hdef_vars
        Hdef_conts Hrecursive).
  - simpl in Hc, Hkinds |- *. unfold is_bound in Hc.
    destruct (lookup k outer) as [found | ]; [ | discriminate ].
    rewrite Hkinds. auto.
  - cbn [apply_cont_value]. cbn [cont_well_kinded] in Hc.
    apply andb_true_iff in Hc as [Hc Hextra_kinds].
    apply andb_true_iff in Hc as [Hc Hextra].
    apply kinds_eqb_eq in Hextra_kinds. cbn [arity_of_cont] in Hkinds.
    apply (IH (vs ++ extra) statics hp rd_ Hc);
      [ | | exact Hstatics | exact Hheap ].
    + rewrite map_app, Hkinds, Hextra_kinds. apply firstn_skipn.
    + rewrite forallb_app, Hvs_wk, Hextra. reflexivity.
  - rewrite apply_cont_value_over_application.
    destruct vs as [ | v [ | v' vs]]; [ exact I | | exact I ].
    simpl in Hc, Hvs_wk. rewrite andb_true_r in Hvs_wk.
    apply andb_true_iff in Hc as [Hc Hexn_to].
    apply andb_true_iff in Hc as [Hc Hreturn_to].
    apply andb_true_iff in Hc as [Hc Hexn_wk].
    apply andb_true_iff in Hc as [Hremaining Hreturn_wk].
    apply kinds_eqb_eq in Hreturn_to, Hexn_to.
    unfold call_function.
    apply (run_call_well_kinded _ Indirect_unknown_arity); try assumption.
    + apply shape_of_call_calls.
    + apply (shape_of_call_values_all _ value_well_kinded_partial);
        [ exact (tuple_fields_well_kinded statics Hstatics) | exact Hvs_wk
        | apply forallb_Forall; exact Hremaining ].
    + apply shape_of_call_arities.
    + intros partial Hpartial Hkind Hreturn_arity.
      apply IHreturn;
        [ exact Hreturn_wk | | | exact Hstatics | exact Hheap ].
      * simpl. rewrite Hkind, Hreturn_to, Hreturn_arity. reflexivity.
      * simpl. rewrite Hpartial. reflexivity.
Qed.

(* A call of a well-kinded value with well-kinded arguments, returning
   to well-kinded continuations of the arities of the call. *)
Lemma call_function_well_kinded (callee : value) (kind : call_kind)
    (vs : list value) (args_arity return_arity : arity)
    (return_to exn_to : cont_value) (statics : static_env) (hp : heap)
      (rd : read_record) :
  value_wk callee = true ->
  forallb value_wk vs = true ->
  cont_wk return_to = true ->
  arity_of_cont outer return_to = unarize return_arity ->
  cont_wk exn_to = true ->
  arity_of_cont outer exn_to = [Value] ->
  statics_wk statics = true ->
  heap_wk hp = true ->
  result_wk
    (call_function callee kind vs args_arity return_arity return_to exn_to
       statics hp rd).
Proof.
  intros Hcallee_wk Hvs_wk Hreturn_wk Hreturn_to Hexn_wk Hexn_to Hstatics
    Hheap.
  unfold call_function.
  apply (run_call_well_kinded _ kind); try assumption.
  - apply shape_of_call_calls.
  - apply (shape_of_call_values_all _ value_well_kinded_partial);
      [ exact (tuple_fields_well_kinded statics Hstatics) | exact Hcallee_wk
      | apply forallb_Forall; exact Hvs_wk ].
  - apply shape_of_call_arities.
  - intros partial Hpartial Hkind Hreturn_arity.
    apply apply_cont_value_well_kinded;
      [ exact Hreturn_wk | | | exact Hstatics | exact Hheap ].
    + simpl. rewrite Hkind, Hreturn_to, Hreturn_arity. reflexivity.
    + simpl. rewrite Hpartial. reflexivity.
Qed.

(* A well-kinded continuation given as many values as it has parameters
   is not given the wrong number. *)
Lemma right_number_given (c : cont_value) :
  ∀ given : nat,
    cont_wk c = true ->
    given = length (arity_of_cont outer c) ->
    wrong_number_given c given = false.
Proof.
  induction c
    as [[params metadata handler] def_vars def_conts recursive | k
       | c IH extra
       | remaining remaining_arity return_arity return_to _ exn_to _];
    intros given Hc Hgiven.
  - simpl in Hgiven |- *. rewrite cont_well_kinded_closure in Hc.
    apply andb_true_iff in Hc as [Hc _]. apply andb_true_iff in Hc as [_ Hh].
    simpl in Hh. apply andb_true_iff in Hh as [_ Hh].
    destruct
      (bind_param_kinds params (hm_params_kinds metadata)
         (kinds_of_env def_vars))
      as [handler_kinds | ] eqn:Hbind; [ | discriminate ].
    rewrite (bind_param_kinds_length _ _ _ _ Hbind), Hgiven, Nat.eqb_refl.
    reflexivity.
  - reflexivity.
  - cbn [wrong_number_given]. cbn [cont_well_kinded] in Hc.
    apply andb_true_iff in Hc as [Hc Hextra_kinds].
    apply andb_true_iff in Hc as [Hc _].
    apply kinds_eqb_eq in Hextra_kinds. cbn [arity_of_cont] in Hgiven.
    apply (IH _ Hc). rewrite Hgiven.
    apply (f_equal (@length _)) in Hextra_kinds.
    rewrite length_map, length_skipn in Hextra_kinds.
    rewrite length_firstn. lia.
  - simpl in Hgiven |- *. rewrite Hgiven. reflexivity.
Qed.

(* And the kinds of what a well-kinded continuation is given are right,
   if they are those of its arity. *)
Lemma right_kinds_given (c : cont_value) :
  ∀ given : list flambda_kind,
    cont_wk c = true ->
    given = arity_of_cont outer c ->
    wrong_kinds_given c given = false.
Proof.
  induction c
    as [[params metadata handler] def_vars def_conts recursive | k
       | c IH extra
       | remaining remaining_arity return_arity return_to exn_to _ _]
    using cont_value_over_ind;
    intros given Hc Hgiven.
  - simpl in Hgiven |- *. rewrite Hgiven, kinds_eqb_refl. reflexivity.
  - reflexivity.
  - cbn [wrong_kinds_given]. cbn [cont_well_kinded] in Hc.
    apply andb_true_iff in Hc as [Hc Hextra_kinds].
    apply andb_true_iff in Hc as [Hc _].
    apply kinds_eqb_eq in Hextra_kinds. cbn [arity_of_cont] in Hgiven.
    apply (IH _ Hc). rewrite Hgiven, Hextra_kinds. apply firstn_skipn.
  - simpl in Hgiven |- *. rewrite Hgiven, kinds_eqb_refl. reflexivity.
Qed.

Lemma step_well_kinded_here (s s' : state) :
  state_wk s = true -> step s = Step s' -> state_wk s' = true.
Proof.
  intros Hs Hstep.
  destruct (state_well_kinded_elim s Hs)
    as (He & Hvars & Hconts & Hstatics & Hheap).
  rewrite (state_eta s) in Hstep.
  set (vars := state_vars s) in *. set (conts := state_conts s) in *.
  set (statics := state_statics s) in *.
  destruct (state_expr s)
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    autorewrite with step in Hstep.
  - (* Let *)
    simpl in He. apply andb_true_iff in He as [Hd Hbody].
    apply has_kind_eq in Hd.
    destruct (exec_named _ _ _ _ d) as [[[v hp'] rd'] | ] eqn:Hv;
      [ | discriminate ].
    destruct (kind_eqb (kind_of_value v) kind) eqn:Hkind; [ | discriminate ].
    injection Hstep as <-. apply kind_eqb_eq in Hkind.
    destruct
      (exec_named_well_kinded statics _ hp' _ rd' vars d kind v Hvars
         Hstatics Hheap Hd Hv)
      as [Hwk Hheap'].
    apply state_well_kinded_intro;
      [ | exact (env_well_kinded_cons x v vars Hwk Hvars) | exact Hconts
        | exact Hstatics | exact Hheap' ].
    change (kinds_of_env ((x, v) :: vars))
      with ((x, kind_of_value v) :: kinds_of_env vars).
    rewrite Hkind. exact Hbody.
  - (* Let_static *)
    simpl in He. apply andb_true_iff in He as [Hbindings Hbody].
    destruct (eval_static_bindings vars bindings statics) as [statics' | ]
      eqn:Heval; [ | discriminate ].
    injection Hstep as <-.
    apply state_well_kinded_intro;
      [ exact Hbody | exact Hvars | exact Hconts | | exact Hheap ].
    exact (eval_static_bindings_well_kinded vars bindings statics statics'
      Hvars Hbindings Hstatics Heval).
  - (* Let_cont_non_recursive *)
    injection Hstep as <-. simpl in He.
    apply andb_true_iff in He as [Hh Hbody].
    apply state_well_kinded_intro;
      [ exact Hbody | exact Hvars | | exact Hstatics | exact Hheap ].
    change
      (cont_wk (Closure h vars conts []) && conts_wk conts = true).
    rewrite cont_well_kinded_closure, Hvars, Hconts. simpl.
    rewrite Hh. reflexivity.
  - (* Let_cont_recursive *)
    destruct (handlers_bind_statics handlers); [ discriminate | ].
    injection Hstep as <-. simpl in He.
    apply andb_true_iff in He as [Hhandlers Hbody].
    apply state_well_kinded_intro;
      [ | exact Hvars | | exact Hstatics | exact Hheap ].
    + rewrite arities_of_rec_closures. exact Hbody.
    + exact (rec_closures_well_kinded vars conts handlers Hvars Hconts
        Hhandlers).
  - (* Apply *)
    simpl in He. unfold apply_well_kinded in He.
    apply andb_true_iff in He as [He Hexn].
    apply andb_true_iff in He as [He Hreturn].
    apply andb_true_iff in He as [He Hextra].
    clear He. apply has_arity_eq in Hreturn, Hexn.
    destruct (eval_simple vars (ap_callee ap)) as [callee | ] eqn:Hcallee;
      [ | discriminate ].
    destruct (eval_simples vars (ap_args ap)) as [vs | ] eqn:Hvs;
      [ | discriminate ].
    destruct (eval_args vars (exn_extra_simples ap) _ Hvars Hextra)
      as (extra & Hextra_values & Hextra_kinds & Hextra_wk).
    rewrite Hextra_values in Hstep.
    pose proof (eval_simples_well_kinded vars _ vs Hvars Hvs) as Hvs_wk.
    pose proof (eval_simple_well_kinded vars _ callee Hvars Hcallee)
      as Hcallee_wk.
    destruct (resolve_well_kinded conts _ _ Hconts Hreturn)
      as [Hreturn_wk Hreturn_to].
    destruct (resolve_well_kinded conts _ _ Hconts Hexn)
      as [Hexn_wk Hexn_to].
    destruct
      (with_extra_args_well_kinded _ extra _ Hexn_wk Hexn_to Hextra_kinds
         Hextra_wk)
      as [Hraise_wk Hraise_to].
    pose proof
      (call_function_well_kinded callee (ap_call_kind ap) vs
         (ap_args_arity ap) (ap_return_arity ap) _ _ statics (state_heap s)
         (state_reads s) Hcallee_wk Hvs_wk Hreturn_wk Hreturn_to Hraise_wk
         Hraise_to Hstatics Hheap)
      as Hresult.
    rewrite Hstep in Hresult. exact Hresult.
  - (* Apply_cont *)
    simpl in He. unfold apply_cont_well_kinded in He.
    apply andb_true_iff in He as [_ He].
    destruct (lookup (ac_continuation ac) (arities_of_conts outer conts))
      as [params_arity | ] eqn:Hk; [ | discriminate ].
    destruct (eval_args vars (ac_args ac) params_arity Hvars He)
      as (vs & Hvs & Hkinds & Hvs_wk).
    rewrite Hvs in Hstep. unfold apply_continuation in Hstep.
    destruct (resolve_well_kinded conts _ _ Hconts Hk) as [Hc Harity].
    rewrite <- Harity in Hkinds.
    pose proof
      (apply_cont_value_well_kinded _ vs statics (state_heap s)
         (state_reads s) Hc Hkinds Hvs_wk Hstatics Hheap)
      as Hresult.
    rewrite Hstep in Hresult. exact Hresult.
  - (* Switch *)
    destruct (eval_simple vars scrutinee) as [v | ]; [ | discriminate ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | discriminate ].
    destruct (find_arm i arms) as [ac | ] eqn:Harm; [ | discriminate ].
    injection Hstep as <-. simpl in He.
    apply andb_true_iff in He as [_ Harms]. rewrite forallb_forall in Harms.
    apply state_well_kinded_intro;
      [ | exact Hvars | exact Hconts | exact Hstatics | exact Hheap ].
    exact (Harms (i, ac) (find_arm_in i arms ac Harm)).
  - (* Invalid *)
    discriminate.
Qed.

(* ======================================================================== *)
(* What a well-kinded state cannot do                                       *)
(* ======================================================================== *)

(* [Simple]s that have kinds under the kinds of an environment are bound
   in it. *)
Lemma none_unbound (vars : env) (simples : list simple) :
  (∀ s : simple,
    s ∈ simples ->
    ∃ k : flambda_kind, kind_of_simple (kinds_of_env vars) s = Some k) ->
  existsb (unbound vars) simples = false.
Proof.
  intros Hall. destruct (existsb (unbound vars) simples) eqn:H;
    [ | reflexivity ].
  apply existsb_exists in H as (s & Hin & Hs). destruct (Hall s Hin) as [k Hk].
  destruct (kind_of_simple_eval vars s k Hk) as (v & Hv & _).
  unfold unbound in Hs. rewrite Hv in Hs. discriminate.
Qed.

Lemma bound_of_kind (vars : env) (s : simple) (k : flambda_kind) :
  kind_of_simple (kinds_of_env vars) s = Some k -> unbound vars s = false.
Proof.
  intros Hk. destruct (kind_of_simple_eval vars s k Hk) as (v & Hv & _).
  unfold unbound. rewrite Hv. reflexivity.
Qed.

Lemma kinds_of_simples_each (kinds : kind_env) (args : list simple) :
  ∀ ks : list flambda_kind,
    kinds_of_simples kinds args = Some ks ->
    length ks = length args
    ∧ ∀ s : simple,
        s ∈ args -> ∃ k : flambda_kind, kind_of_simple kinds s = Some k.
Proof.
  induction args as [ | s args IH]; simpl; intros ks.
  - intros [= <-]. split; [ reflexivity | intros s [] ].
  - destruct (kind_of_simple kinds s) as [k | ] eqn:Hk; [ | discriminate ].
    destruct (kinds_of_simples kinds args) as [ks' | ]; [ | discriminate ].
    intros [= <-]. destruct (IH ks' eq_refl) as [Hlength Heach].
    split; [ simpl; rewrite Hlength; reflexivity | ].
    intros s' [<- | Hin]; [ exists k; exact Hk | exact (Heach s' Hin) ].
Qed.

Lemma value_slots_each (kinds : kind_env)
    (value_slots : list (value_slot * simple)) :
  value_slots_well_kinded slots kinds value_slots = true ->
  ∀ s : simple,
    s ∈ map snd value_slots ->
    ∃ k : flambda_kind, kind_of_simple kinds s = Some k.
Proof.
  unfold value_slots_well_kinded. rewrite forallb_forall.
  intros Hslots s Hin. apply in_map_iff in Hin as ([slot s'] & <- & Hin).
  specialize (Hslots _ Hin). simpl in Hslots.
  destruct (lookup slot slots) as [kind | ]; [ | discriminate ].
  apply has_kind_eq in Hslots. exists kind. exact Hslots.
Qed.

Lemma static_const_each (kinds : kind_env) (const : static_const) :
  static_const_well_kinded slots kinds const = true ->
  ∀ s : simple,
    s ∈ simples_of_static_const const ->
    ∃ k : flambda_kind, kind_of_simple kinds s = Some k.
Proof.
  destruct const
    as [code value_slots | fslot syms decls value_slots | tag fields
       | contents | bytes | elements];
    simpl.
  - apply value_slots_each.
  - apply value_slots_each.
  - rewrite forallb_forall. intros Hfields s Hin.
    exists Value. apply has_kind_eq. exact (Hfields s Hin).
  - intros Hcontents s [<- | []].
    destruct (kind_of_simple kinds contents) as [k | ]; [ | discriminate ].
    exists k. reflexivity.
  - intros _ s [].
  - rewrite forallb_forall. intros Helements s Hin.
    exists Value. apply has_kind_eq. exact (Helements s Hin).
Qed.

Lemma bindings_each (kinds : kind_env) (bindings : list static_binding) :
  forallb (binding_well_kinded slots kinds) bindings = true ->
  ∀ s : simple,
    s ∈ flat_map simples_of_binding bindings ->
    ∃ k : flambda_kind, kind_of_simple kinds s = Some k.
Proof.
  rewrite forallb_forall. intros Hbindings s Hin.
  apply in_flat_map in Hin as (binding & Hbinding & Hin).
  specialize (Hbindings binding Hbinding).
  destruct binding as [id c | id | sym const]; simpl in Hin;
    [ destruct Hin | destruct Hin | ].
  exact (static_const_each kinds const Hbindings s Hin).
Qed.

Lemma no_kind_fault_here (s : state) :
  state_wk s = true -> kind_fault s = false.
Proof.
  intros Hs.
  destruct (state_well_kinded_elim s Hs)
    as (He & Hvars & Hconts & Hstatics & _).
  unfold kind_fault.
  set (vars := state_vars s) in *. set (conts := state_conts s) in *.
  set (statics := state_statics s) in *.
  destruct (state_expr s)
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    simpl in He; try reflexivity.
  - (* Let *)
    apply andb_true_iff in He as [Hd _]. apply has_kind_eq in Hd.
    apply orb_false_iff. split;
      [ | unfold wrong_kind_bound;
          destruct (exec_named statics (state_heap s) (state_reads s) vars d)
            as [[[v hp'] rd'] | ] eqn:Hv;
          [ | reflexivity ];
          rewrite
            (exec_named_kind statics _ hp' _ rd' vars d kind v Hvars Hstatics
               Hd Hv);
          rewrite kind_eqb_refl; reflexivity ].
    destruct d
      as [s' | p args | code value_slots | fslot decls value_slots
         | slot closure | project_from slot closure
         | move_from move_to closure];
      simpl in Hd |- *.
    + rewrite (bound_of_kind vars s' kind Hd). reflexivity.
    + destruct (kinds_of_simples (kinds_of_env vars) args) as [ks | ] eqn:Hks;
        [ | discriminate ].
      apply none_unbound.
      exact (proj2 (kinds_of_simples_each _ _ _ Hks)).
    + destruct (value_slots_well_kinded slots (kinds_of_env vars) value_slots)
        eqn:Hslots; [ | discriminate ].
      apply none_unbound.
      exact (value_slots_each _ _ Hslots).
    + destruct (value_slots_well_kinded slots (kinds_of_env vars) value_slots)
        eqn:Hslots; [ | discriminate ].
      apply none_unbound.
      exact (value_slots_each _ _ Hslots).
    + destruct (kind_of_simple (kinds_of_env vars) closure) as [k | ] eqn:Hk;
        [ | discriminate ].
      rewrite (bound_of_kind vars closure k Hk). reflexivity.
    + destruct (kind_of_simple (kinds_of_env vars) closure) as [k | ] eqn:Hk;
        [ | discriminate ].
      rewrite (bound_of_kind vars closure k Hk). reflexivity.
    + destruct (kind_of_simple (kinds_of_env vars) closure) as [k | ] eqn:Hk;
        [ | discriminate ].
      rewrite (bound_of_kind vars closure k Hk). reflexivity.
  - (* Let_static *)
    apply andb_true_iff in He as [Hbindings _]. rewrite orb_false_r.
    apply none_unbound. exact (bindings_each _ _ Hbindings).
  - (* Apply *)
    unfold apply_well_kinded in He.
    apply andb_true_iff in He as [He _]. apply andb_true_iff in He as [He _].
    apply andb_true_iff in He as [He Hextra].
    apply andb_true_iff in He as [Hcallee Hargs].
    apply has_arity_eq in Hextra.
    destruct (kinds_of_simples (kinds_of_env vars) (ap_args ap))
      as [ks | ] eqn:Hks;
      [ clear Hargs; rename Hks into Hargs | discriminate ].
    destruct (kind_of_simple (kinds_of_env vars) (ap_callee ap))
      as [k | ] eqn:Hk; [ | discriminate ].
    simpl. rewrite (bound_of_kind vars _ k Hk), existsb_app.
    rewrite
      (none_unbound vars _ (proj2 (kinds_of_simples_each _ _ _ Hargs))),
      (none_unbound vars _ (proj2 (kinds_of_simples_each _ _ _ Hextra))).
    reflexivity.
  - (* Apply_cont *)
    unfold apply_cont_well_kinded in He.
    apply andb_true_iff in He as [_ He].
    destruct (lookup (ac_continuation ac) (arities_of_conts outer conts))
      as [params_arity | ] eqn:Hk; [ | discriminate ].
    apply has_arity_eq in He.
    destruct (kinds_of_simples_each _ _ _ He) as [Hlength Heach].
    simpl. rewrite (none_unbound vars (ac_args ac) Heach). simpl.
    unfold wrong_number_of_arguments, wrong_kinds_of_arguments.
    destruct (resolve_well_kinded conts _ _ Hconts Hk) as [Hc Harity].
    rewrite (right_number_given _ _ Hc)
      by (rewrite Harity; symmetry; exact Hlength).
    destruct (kinds_of_simples_eval vars (ac_args ac) _ He)
      as (vs & Hvs & Hkinds).
    rewrite Hvs. simpl. apply (right_kinds_given _ _ Hc).
    rewrite Harity. exact Hkinds.
  - (* Switch *)
    apply andb_true_iff in He as [Hscrutinee _].
    apply has_kind_eq in Hscrutinee.
    destruct (kind_of_simple_eval vars scrutinee _ Hscrutinee)
      as (v & Hv & Hkind).
    simpl. unfold unbound. rewrite Hv, Hkind. reflexivity.
Qed.

(* A state exits only at an [Apply_cont], or at an [Apply] that gives
   a partial application to its return continuation. *)
Lemma exit_well_kinded_here (s : state) (k : continuation)
    (vs : list value) (hp : heap) (rd : read_record) :
  state_wk s = true ->
  step s = Exit k vs hp rd ->
  lookup k outer = Some (map kind_of_value vs)
  ∧ forallb value_wk vs = true.
Proof.
  intros Hs Hstep.
  destruct (state_well_kinded_elim s Hs)
    as (He & Hvars & Hconts & Hstatics & Hheap).
  rewrite (state_eta s) in Hstep.
  set (vars := state_vars s) in *. set (conts := state_conts s) in *.
  set (statics := state_statics s) in *.
  destruct (state_expr s)
    as [ x kind d body | bindings body | k' h body | handlers body | ap | ac
       | scrutinee arms | ];
    autorewrite with step in Hstep.
  - destruct (exec_named _ _ rd_ _ d) as [[[v hp'] rd'] | ]; [ | discriminate ].
    destruct (kind_eqb (kind_of_value v) kind); discriminate.
  - destruct (eval_static_bindings vars bindings statics); discriminate.
  - discriminate.
  - destruct (handlers_bind_statics handlers); discriminate.
  - simpl in He. unfold apply_well_kinded in He.
    apply andb_true_iff in He as [He Hexn].
    apply andb_true_iff in He as [He Hreturn].
    apply andb_true_iff in He as [He Hextra].
    clear He. apply has_arity_eq in Hreturn, Hexn.
    destruct (eval_simple vars (ap_callee ap)) as [callee | ] eqn:Hcallee;
      [ | discriminate ].
    destruct (eval_simples vars (ap_args ap)) as [args | ] eqn:Hargs;
      [ | discriminate ].
    destruct (eval_args vars (exn_extra_simples ap) _ Hvars Hextra)
      as (extra & Hextra_values & Hextra_kinds & Hextra_wk).
    rewrite Hextra_values in Hstep.
    destruct (resolve_well_kinded conts _ _ Hconts Hreturn)
      as [Hreturn_wk Hreturn_to].
    destruct (resolve_well_kinded conts _ _ Hconts Hexn)
      as [Hexn_wk Hexn_to].
    destruct
      (with_extra_args_well_kinded _ extra _ Hexn_wk Hexn_to Hextra_kinds
         Hextra_wk)
      as [Hraise_wk Hraise_to].
    pose proof
      (call_function_well_kinded callee (ap_call_kind ap) args
         (ap_args_arity ap) (ap_return_arity ap) _ _ statics (state_heap s)
           (state_reads s)
         (eval_simple_well_kinded vars _ callee Hvars Hcallee)
         (eval_simples_well_kinded vars _ args Hvars Hargs) Hreturn_wk
         Hreturn_to Hraise_wk Hraise_to Hstatics Hheap)
      as Hresult.
    rewrite Hstep in Hresult. exact Hresult.
  - simpl in He. unfold apply_cont_well_kinded in He.
    apply andb_true_iff in He as [_ He].
    destruct (lookup (ac_continuation ac) (arities_of_conts outer conts))
      as [params_arity | ] eqn:Hk; [ | discriminate ].
    destruct (eval_args vars (ac_args ac) params_arity Hvars He)
      as (args & Hargs & Hkinds & Hargs_wk).
    rewrite Hargs in Hstep. unfold apply_continuation in Hstep.
    destruct (resolve_well_kinded conts _ _ Hconts Hk) as [Hc Harity].
    rewrite <- Harity in Hkinds.
    pose proof
      (apply_cont_value_well_kinded _ args statics (state_heap s)
         (state_reads s) Hc Hkinds Hargs_wk Hstatics Hheap)
      as Hresult.
    rewrite Hstep in Hresult. exact Hresult.
  - destruct (eval_simple vars scrutinee) as [v | ]; [ | discriminate ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | discriminate ].
    destruct (find_arm i arms); discriminate.
  - discriminate.
Qed.

Lemma run_well_kinded_here (n : nat) :
  ∀ s s' : state,
    state_wk s = true -> run n s = Step s' -> state_wk s' = true.
Proof.
  induction n as [ | n IH]; intros s s' Hs.
  - intros [= <-]. exact Hs.
  - rewrite run_succ. destruct (step s) as [s'' | k vs | ] eqn:Hstep;
      [ | discriminate | discriminate ].
    exact (IH s'' s' (step_well_kinded_here s s'' Hs Hstep)).
Qed.

End Kinds.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Theorem call_at_wrong_kinds_undefined :
  call_at_wrong_kinds_undefined_statement.
Proof.
  intros ap vars conts statics hp rd callee vs called value_slots c Hcallee Hvs
    Hclosure Hcode Hmismatch.
  rewrite step_apply, Hcallee, Hvs.
  destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
    [ | reflexivity ].
  unfold call_function.
  rewrite (shape_of_call_closure statics callee _ _ _ _
             (closure_not_partial statics callee _ Hclosure)).
  unfold shape_of_closure_call. rewrite Hclosure.
  destruct
    (match ap_call_kind ap with
     | Direct named => Pos.eqb called named
     | Indirect_unknown_arity => true
     end);
    [ | reflexivity ].
  rewrite Hcode.
  destruct (code_is_tupled c) eqn:Htupled.
  { (* Tupled code: the arguments are not of the kinds of the arity. *)
    unfold arguments_for_code. rewrite Htupled. unfold detuple.
    destruct vs as [ | tuple rest]; [ reflexivity | ].
    destruct (ap_args_arity ap) as [ | first rest_arity] eqn:Harity;
      [ reflexivity | ].
    destruct
      (kinds_eqb (map kind_of_value (tuple :: rest))
         (unarize (first :: rest_arity)))
      eqn:Hkinds; [ | reflexivity ].
    exfalso. apply kinds_eqb_eq in Hkinds.
    destruct Hmismatch as [H | [Hfalse _]]; [ contradiction | discriminate ]. }
  rewrite (arguments_for_code_not_tupled _ _ _ _ Htupled). cbv beta iota.
  unfold shape_of_code_call. cbv zeta.
  destruct
    (Nat.eqb (length (ap_args_arity ap)) (length (code_params_arity c)))
    eqn:Hcount.
  - simpl. unfold call_code.
    destruct (binds_statics (code_body c)); [ reflexivity | ].
    destruct
      (kinds_eqb (map kind_of_value vs) (unarize (ap_args_arity ap))
       && kinds_per_parameter_eqb (unarize_per_parameter (ap_args_arity ap))
            (unarize_per_parameter (code_params_arity c))
       && arity_eqb (ap_return_arity ap) (code_result_arity c))
      eqn:Hagree; simpl; [ | reflexivity ].
    exfalso.
    apply andb_true_iff in Hagree as [Hagree Hreturn].
    apply andb_true_iff in Hagree as [Hkinds Hargs].
    apply kinds_eqb_eq in Hkinds. apply kinds_per_parameter_eqb_eq in Hargs.
    apply arity_eqb_eq in Hreturn. apply Nat.eqb_eq in Hcount.
    destruct Hmismatch as [H | [_ [[_ [H | H]] | [H _]]]]; contradiction.
  - destruct
      (kinds_eqb (map kind_of_value vs) (unarize (ap_args_arity ap))
       && partial_or_over_application (ap_args_arity ap) (ap_return_arity ap)
            (code_params_arity c) (code_result_arity c))
      eqn:Htests; simpl; [ | reflexivity ].
    exfalso.
    apply andb_true_iff in Htests as [Hkinds Htest].
    apply kinds_eqb_eq in Hkinds. apply Nat.eqb_neq in Hcount.
    destruct Hmismatch as [H | [_ [[H _] | [_ H]]]];
      [ contradiction | contradiction | congruence ].
Qed.

Theorem let_binds_at_kind : let_binds_at_kind_statement.
Proof.
  intros x kind d body vars conts statics hp rd s. rewrite step_let_exec.
  destruct (exec_named statics hp rd vars d) as [[[v hp'] rd'] | ];
    [ | discriminate ].
  destruct (kind_eqb (kind_of_value v) kind) eqn:Hkind; [ | discriminate ].
  intros [= <-]. exists v, hp', rd'. apply kind_eqb_eq in Hkind. auto.
Qed.

Theorem handler_binds_at_kinds : handler_binds_at_kinds_statement.
Proof.
  intros params metadata handler def_vars def_conts recursive vs statics hp rd
    s.
  simpl.
  destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata))
    eqn:Hkinds; [ | discriminate ].
  destruct (bind_params params vs def_vars) as [handler_vars | ];
    [ | discriminate ].
  intros [= <-]. apply kinds_eqb_eq in Hkinds. eauto.
Qed.

Theorem call_binds_at_kinds : call_binds_at_kinds_statement.
Proof.
  intros closure c vs args_arity return_arity return_to exn_to statics hp rd s
    Hcall.
  apply call_code_step in Hcall
    as (callee_vars & _ & Hkinds & Hargs & _ & Hbind & ->).
  repeat split.
  - unfold code_params_kinds.
    rewrite <- (unarize_per_parameter_eq _ _ Hargs). exact Hkinds.
  - exact Hbind.
Qed.

Theorem called_closure_at_kind : called_closure_at_kind_statement.
Proof.
  intros statics callee kind vs args_arity return_arity.
  pose proof (shape_of_call_calls statics callee kind vs args_arity
                return_arity) as Hcalls.
  destruct (shape_of_call statics callee kind vs args_arity return_arity)
    as [[closure c vs' args_arity'
        | closure c vs' remaining args_arity' remaining_arity | partial] | ];
    simpl in Hcalls |- *; try exact I;
    destruct Hcalls as (called & slots & Hclosure & _);
    exact (closure_of_kind _ _ _ _ Hclosure).
Qed.

Theorem well_kinded_unit_state : well_kinded_unit_state_statement.
Proof.
  intros slots unit_return unit_exn e He.
  unfold state_well_kinded. simpl_state.
  exact (proj2 (andb_true_iff _ _) (conj
    (proj2 (andb_true_iff _ _) (conj
      (proj2 (andb_true_iff _ _) (conj
        (proj2 (andb_true_iff _ _) (conj He eq_refl)) eq_refl)) eq_refl))
    eq_refl)).
Qed.

Theorem step_well_kinded : step_well_kinded_statement.
Proof.
  intros slots outer s s'. exact (step_well_kinded_here slots outer s s').
Qed.

Theorem run_well_kinded : run_well_kinded_statement.
Proof.
  intros slots outer n s s'. exact (run_well_kinded_here slots outer n s s').
Qed.

Theorem well_kinded_no_kind_fault : well_kinded_no_kind_fault_statement.
Proof.
  intros slots outer s. exact (no_kind_fault_here slots outer s).
Qed.

Theorem well_kinded_exit : well_kinded_exit_statement.
Proof.
  intros slots outer s k vs hp rd.
  exact (exit_well_kinded_here slots outer s k vs hp rd_).
Qed.

Theorem well_kinded_unit_never_faults :
  well_kinded_unit_never_faults_statement.
Proof.
  intros slots unit_return unit_exn e n s He Hrun.
  apply (well_kinded_no_kind_fault slots
    (unit_arities unit_return unit_exn)).
  exact (run_well_kinded slots _ n _ s
    (well_kinded_unit_state slots unit_return unit_exn e He) Hrun).
Qed.

(* A run that exits has a last state, which is well-kinded. *)
Lemma run_exit (n : nat) :
  ∀ (s : state) (k : continuation) (vs : list value) (hp : heap)
    (rd : read_record),
    run n s = Exit k vs hp rd ->
    ∃ (m : nat) (s' : state), run m s = Step s' ∧ step s' = Exit k vs hp rd.
Proof.
  induction n as [ | n IH]; intros s k vs hp rd; [ discriminate | ].
  rewrite run_succ. destruct (step s) as [s'' | k' vs' hp' rd' | ] eqn:Hstep;
    [ | | discriminate ].
  - intros Hrun. destruct (IH s'' k vs hp rd Hrun) as (m & s' & Hm & Hexit).
    exists (S m), s'. rewrite run_succ, Hstep. auto.
  - intros [= -> -> -> ->]. exists 0, s. auto.
Qed.

Theorem well_kinded_unit_exits : well_kinded_unit_exits_statement.
Proof.
  intros slots unit_return unit_exn e n k vs hp rd He Hrun.
  destruct (run_exit n _ k vs _ rd_ Hrun) as (m & s & Hm & Hexit).
  pose proof (run_well_kinded slots _ m _ s
    (well_kinded_unit_state slots unit_return unit_exn e He) Hm) as Hs.
  destruct (well_kinded_exit slots _ s k vs _ rd_ Hs Hexit) as [Hk _].
  unfold unit_arities in Hk. simpl in Hk.
  assert (Hkinds : (k = unit_return ∨ k = unit_exn)
                   ∧ map kind_of_value vs = [Value]).
  { destruct (Pos.eqb_spec k unit_return) as [-> | _];
      [ injection Hk as Hk; auto | ].
    destruct (Pos.eqb_spec k unit_exn) as [-> | _];
      [ injection Hk as Hk; auto | discriminate ]. }
  destruct Hkinds as [Hwhich Hkinds]. split; [ exact Hwhich | ].
  destruct vs as [ | v [ | v' vs]]; try discriminate.
  injection Hkinds as Hkind. exists v. auto.
Qed.

(* ======================================================================== *)
(* Examples                                                                 *)
(* ======================================================================== *)

(* That the statements are not vacuous: a unit that passes the check and
   whose run is defined. It binds code with a parameter that is a naked
   immediate, and a closure of it with a value slot; calls the closure
   directly; the code projects the value slot, switches on its parameter
   and returns what the slot holds, which the unit returns in turn.
   Names: the unit returns through continuation 1 and raises through 2. *)

#[local] Abbreviation naked_immediate :=
  (Naked_number Kind_naked_immediate).

Definition example_code : code :=
  Code [1%positive] 2%positive 3%positive 4%positive
    (make_code_metadata [Singleton naked_immediate] [Singleton Value])
    (Let 3%positive Value (Project_value_slot 1%positive (Var 2%positive))
       (Switch (Var 1%positive)
          [(0%Z, Apply_cont_expr 3%positive [Var 3%positive] None);
           (1%Z,
            Apply_cont_expr 3%positive [Const (Tagged_immediate 7)] None)])).

Definition example_call (args_arity : arity) (arg : simple) : expr :=
  Let_cont_non_recursive 5%positive
    (Handler [5%positive] (make_handler_metadata [Value])
       (Apply_cont (Apply_cont_expr 1%positive [Var 5%positive] None)))
    (Apply
       (Apply_expr (Symbol 1%positive) (Direct 1%positive) [arg] 5%positive
          2%positive [] args_arity [Singleton Value])).

Definition example_unit (call : expr) : expr :=
  Let_static
    [Bind_code 1%positive example_code;
     Bind_symbol 1%positive
       (Static_const_set_of_closures 1%positive
          [(1%positive, Const (Tagged_immediate 5))])]
    (Let 4%positive naked_immediate (Simple (Const (Naked_immediate 0)))
       call).

Definition example_slots : slot_kinds := [(1%positive, Value)].

Definition example_well_kinded : expr :=
  example_unit (example_call [Singleton naked_immediate] (Var 4%positive)).

Example example_passes :
  well_kinded_unit example_slots 1%positive 2%positive example_well_kinded
  = true.
Proof. reflexivity. Qed.

Example example_runs :
  run 8 (State example_well_kinded [] [] empty_static_env empty_heap no_reads)
  = Exit 1%positive [Value_const (Tagged_immediate 5)] empty_heap no_reads.
Proof. vm_compute. reflexivity. Qed.

(* What the check does not look at: a call at an arity that is not that
   of the code. The same unit, calling at the arity of one [Value] with
   an argument of kind [Value], passes the check; the machine makes the
   call undefined. *)
Definition example_wrong_arity : expr :=
  example_unit
    (example_call [Singleton Value] (Const (Tagged_immediate 0))).

Example example_wrong_arity_passes :
  well_kinded_unit example_slots 1%positive 2%positive example_wrong_arity
  = true.
Proof. reflexivity. Qed.

Example example_wrong_arity_undefined :
  run 4 (State example_wrong_arity [] [] empty_static_env empty_heap no_reads)
    = Undefined.
Proof. vm_compute. reflexivity. Qed.

(* Nor does it look at whether an argument of a call is of the kind of
   the arity the call is made at: the same unit, calling at the arity of
   the code with an argument of kind [Value], passes the check, and the
   machine makes the call undefined. Were the call made, the code would
   [Switch] on a tagged immediate. *)
Definition example_wrong_argument : expr :=
  example_unit
    (example_call [Singleton naked_immediate] (Const (Tagged_immediate 0))).

Example example_wrong_argument_passes :
  well_kinded_unit example_slots 1%positive 2%positive
    example_wrong_argument
  = true.
Proof. reflexivity. Qed.

Example example_wrong_argument_undefined :
  run 4
    (State example_wrong_argument [] [] empty_static_env empty_heap no_reads)
    = Undefined.
Proof. vm_compute. reflexivity. Qed.

(* Where the check answers no: a slot given another kind than that of
   what it holds, and a continuation applied to an argument that is not
   of the kind of its parameter (here the return continuation of the
   unit, which takes a [Value]). *)
Example example_slot_kind_refused :
  well_kinded_unit [(1%positive, naked_immediate)] 1%positive 2%positive
    example_well_kinded
  = false.
Proof. reflexivity. Qed.

Example example_apply_cont_kind_refused :
  well_kinded_unit [] 1%positive 2%positive
    (Apply_cont (Apply_cont_expr 1%positive [Const (Naked_immediate 0)] None))
  = false.
Proof. reflexivity. Qed.

(* The check is not a condition that every term the compiler accepts
   meets. This unit has a [Switch] on a constant, whose arm that is not
   taken passes a naked immediate to the return continuation. Simplify
   removes that arm before it looks at it and compiles the unit; the
   check, which looks at every branch, answers no; and the run is
   defined. *)
Definition example_dead_arm : expr :=
  Switch (Const (Naked_immediate 0))
    [(0%Z, Apply_cont_expr 1%positive [Const (Tagged_immediate 5)] None);
     (1%Z, Apply_cont_expr 1%positive [Const (Naked_immediate 1)] None)].

Example example_dead_arm_refused :
  well_kinded_unit [] 1%positive 2%positive example_dead_arm = false.
Proof. reflexivity. Qed.

Example example_dead_arm_runs :
  run 2 (State example_dead_arm [] [] empty_static_env empty_heap no_reads)
  = Exit 1%positive [Value_const (Tagged_immediate 5)] empty_heap no_reads.
Proof. vm_compute. reflexivity. Qed.

(* What the check does not look at: a call of what is not of kind
   [Value]. The unit passes the check, and the machine makes the call
   undefined. *)
Definition example_call_of_number : expr :=
  Let 4%positive naked_immediate (Simple (Const (Naked_immediate 0)))
    (Apply
       (Apply_expr (Var 4%positive) Indirect_unknown_arity [] 1%positive
          2%positive [] [] [Singleton Value])).

Example example_call_of_number_passes :
  well_kinded_unit [] 1%positive 2%positive example_call_of_number = true.
Proof. reflexivity. Qed.

Example example_call_of_number_undefined :
  run 2
    (State example_call_of_number [] [] empty_static_env empty_heap no_reads)
    = Undefined.
Proof. vm_compute. reflexivity. Qed.

(* Nor at the arguments of a primitive. [Opaque_identity] at kind [Value]
   applied to a naked immediate, and [Is_int] applied to one, pass the
   check and have no value: the unit is undefined. *)
Definition example_prim (p : unary_primitive) (kind : flambda_kind) : expr :=
  Let 4%positive kind (Prim (Unary p) [Const (Naked_immediate 1)])
    (Apply_cont (Apply_cont_expr 1%positive [Const (Tagged_immediate 5)] None)).

Example example_opaque_passes :
  well_kinded_unit [] 1%positive 2%positive
    (example_prim (Opaque_identity Value) Value)
  = true.
Proof. reflexivity. Qed.

Example example_opaque_undefined :
  run 1
    (State (example_prim (Opaque_identity Value) Value) [] []
       empty_static_env empty_heap no_reads)
  = Undefined.
Proof. vm_compute. reflexivity. Qed.

Example example_is_int_passes :
  well_kinded_unit [] 1%positive 2%positive
    (example_prim (Is_int false) naked_immediate)
  = true.
Proof. reflexivity. Qed.

Example example_is_int_undefined :
  run 1
    (State (example_prim (Is_int false) naked_immediate) [] []
       empty_static_env empty_heap no_reads)
  = Undefined.
Proof. vm_compute. reflexivity. Qed.

(* Unboxed products. The code takes one parameter that is an unboxed pair
   of values, so two variables, and returns an unboxed triple, nested as
   a value and a pair. The call passes two values and binds the three
   results. *)

#[local] Abbreviation pair :=
  (Unboxed_product [Singleton Value; Singleton Value]).

Definition product_code : code :=
  Code [1%positive; 2%positive] 3%positive 4%positive 5%positive
    (make_code_metadata [pair] [Unboxed_product [Singleton Value; pair]])
    (Apply_cont
       (Apply_cont_expr 4%positive
          [Var 1%positive; Var 2%positive; Var 1%positive] None)).

Definition product_unit (args_arity return_arity : arity) : expr :=
  Let_static
    [Bind_code 1%positive product_code;
     Bind_symbol 1%positive (Static_const_set_of_closures 1%positive [])]
    (Let_cont_non_recursive 5%positive
       (Handler [5%positive; 6%positive; 7%positive]
          (make_handler_metadata [Value; Value; Value])
          (Apply_cont (Apply_cont_expr 1%positive [Var 6%positive] None)))
       (Apply
          (Apply_expr (Symbol 1%positive) (Direct 1%positive)
             [Const (Tagged_immediate 1); Const (Tagged_immediate 2)]
             5%positive 2%positive [] args_arity return_arity))).

(* At the arities of the code, the unit passes the check and runs. *)
Example product_passes :
  well_kinded_unit [] 1%positive 2%positive
    (product_unit [pair] [Unboxed_product [Singleton Value; pair]])
  = true.
Proof. reflexivity. Qed.

Example product_runs :
  run 5
    (State (product_unit [pair] [Unboxed_product [Singleton Value; pair]])
       [] [] empty_static_env empty_heap no_reads)
  = Exit 1%positive [Value_const (Tagged_immediate 2)] empty_heap no_reads.
Proof. vm_compute. reflexivity. Qed.

(* Two parameters, each a value, are not one parameter that is a pair of
   values: the call is undefined, though the unarized kinds are the
   same. *)
Example product_two_parameters_undefined :
  run 3
    (State
       (product_unit [Singleton Value; Singleton Value]
          [Unboxed_product [Singleton Value; pair]])
       [] [] empty_static_env empty_heap no_reads)
  = Undefined.
Proof. vm_compute. reflexivity. Qed.

(* The results of a call are compared with those of the code as they are
   nested: a pair and a value are not a value and a pair. The check does
   not see it, as the three kinds are the same. *)
Example product_result_nesting_passes :
  well_kinded_unit [] 1%positive 2%positive
    (product_unit [pair] [Unboxed_product [pair; Singleton Value]])
  = true.
Proof. reflexivity. Qed.

Example product_result_nesting_undefined :
  run 3
    (State (product_unit [pair] [Unboxed_product [pair; Singleton Value]])
       [] [] empty_static_env empty_heap no_reads)
  = Undefined.
Proof. vm_compute. reflexivity. Qed.

(* The arguments are compared parameter by parameter, each unarized: how
   products are nested inside one parameter does not count. *)
Example product_argument_nesting_runs :
  run 5
    (State
       (product_unit [Unboxed_product [Unboxed_product [Singleton Value];
                                       Singleton Value]]
          [Unboxed_product [Singleton Value; pair]])
       [] [] empty_static_env empty_heap no_reads)
  = Exit 1%positive [Value_const (Tagged_immediate 2)] empty_heap no_reads.
Proof. vm_compute. reflexivity. Qed.

(* A partial application: one argument, of the kind of the first of two
   parameters, to code that returns a value, by a call that expects one.
   The partial application is given to the return continuation of the
   call, here a way out. *)
Definition two_parameter_code : code :=
  Code [1%positive; 2%positive] 3%positive 4%positive 5%positive
    (make_code_metadata [Singleton Value; Singleton Value] [Singleton Value])
    (Apply_cont (Apply_cont_expr 4%positive [Var 1%positive] None)).

Definition partial_application_of (arg : simple) : state :=
  State
    (Apply
       (Apply_expr (Symbol 1%positive) (Direct 1%positive) [arg] 1%positive
          2%positive [] [Singleton Value] [Singleton Value]))
    [] []
    (Static_env [(1%positive, two_parameter_code)]
       [(1%positive, Static_closure 1%positive [])]) empty_heap no_reads.

Definition partial_application_state : state :=
  partial_application_of (Const (Tagged_immediate 1)).

Example partial_application_returns :
  step partial_application_state
  = Exit 1%positive
      [Value_partial_application (Value_symbol 1%positive)
         [Value_const (Tagged_immediate 1)] [Singleton Value]] empty_heap
           no_reads.
Proof. vm_compute. reflexivity. Qed.

(* The same call with an argument that is not of the kind of the arity
   the call is made at is not one that the compiler takes for a partial
   application: Simplify replaces it by [Invalid] before it looks at the
   code. The machine has no step. *)
Example partial_application_at_wrong_kind_undefined :
  step (partial_application_of (Const (Naked_immediate 1))) = Undefined.
Proof. vm_compute. reflexivity. Qed.

(* A call with no argument of code that has parameters is not a partial
   application: the compiler stops on it. *)
Example no_argument_not_partial :
  partial_or_over_application [] [Singleton Value]
    [Singleton Value; Singleton Value] [Singleton Value]
  = false.
Proof. reflexivity. Qed.
