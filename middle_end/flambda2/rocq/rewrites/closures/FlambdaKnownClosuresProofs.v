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

(* Proofs of the statements of FlambdaKnownClosuresSpec.v. Nothing here
   needs review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaValueSpec FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaContextProofs FlambdaNominalSpec
  FlambdaNominalProofs FlambdaWellFormedSpec FlambdaWellFormedProofs
  FlambdaClosuresProofs FlambdaNormaliseSpec FlambdaKnownClosuresSpec
  FlambdaEquivarianceTactic.
Import ListNotations.

(* ======================================================================== *)
(* What it means for what is known to hold                                  *)
(* ======================================================================== *)

(* A symbol known to be a closure is bound to what the constant makes in
   the environment at hand. *)
Definition closure_holds (vars : env) (statics : static_env)
    (entry : symbol * static_const) : Prop :=
  ∃ allocated : statically_allocated_value,
    eval_static_const vars (snd entry) = Some allocated
    ∧ lookup (fst entry) (static_values statics) = Some allocated.

Definition closures_hold (known : known_closures) (vars : env)
    (statics : static_env) : Prop :=
  ∀ entry : symbol * static_const,
    entry ∈ known -> closure_holds vars statics entry.

(* ======================================================================== *)
(* Binding a variable that a constant does not mention                      *)
(* ======================================================================== *)

Lemma eval_simple_fresh (x : variable) (v : value) (vars : env) (s : simple) :
  var_in_simple x s = false ->
  eval_simple ((x, v) :: vars) s = eval_simple vars s.
Proof.
  destruct s as [y | sym | c]; simpl; try reflexivity.
  intros Hfresh. rewrite Pos.eqb_sym, Hfresh. reflexivity.
Qed.

Lemma eval_simples_fresh (x : variable) (v : value) (vars : env)
    (args : list simple) :
  var_in_simples x args = false ->
  eval_simples ((x, v) :: vars) args = eval_simples vars args.
Proof.
  unfold var_in_simples.
  induction args as [ | s args IH]; [ reflexivity | ].
  cbn [existsb eval_simples]. intros Hfresh.
  apply orb_false_iff in Hfresh as [Hs Hargs].
  rewrite (eval_simple_fresh x v vars s Hs), (IH Hargs). reflexivity.
Qed.

Lemma eval_value_slots_fresh (x : variable) (v : value) (vars : env)
    (slots : list (value_slot * simple)) :
  var_in_value_slots x slots = false ->
  eval_value_slots ((x, v) :: vars) slots = eval_value_slots vars slots.
Proof.
  unfold var_in_value_slots.
  induction slots as [ | [slot s] slots IH]; [ reflexivity | ].
  cbn [existsb eval_value_slots]. intros Hfresh.
  apply orb_false_iff in Hfresh as [Hs Hslots].
  rewrite (eval_simple_fresh x v vars s Hs), (IH Hslots). reflexivity.
Qed.

Lemma eval_static_const_fresh (x : variable) (v : value) (vars : env)
    (s : symbol) (const : static_const) :
  var_in_static_binding x (Bind_symbol s const) = false ->
  eval_static_const ((x, v) :: vars) const = eval_static_const vars const.
Proof.
  destruct const as [code slots | fslot syms decls slots | tag fields
    | contents | bytes | elements];
    simpl; intros Hfresh.
  - rewrite (eval_value_slots_fresh x v vars slots Hfresh). reflexivity.
  - rewrite (eval_value_slots_fresh x v vars slots Hfresh). reflexivity.
  - rewrite (eval_simples_fresh x v vars fields Hfresh). reflexivity.
  - rewrite (eval_simple_fresh x v vars contents Hfresh). reflexivity.
  - reflexivity.
  - rewrite (eval_simples_fresh x v vars elements Hfresh). reflexivity.
Qed.

(* ======================================================================== *)
(* What is known stays so                                                   *)
(* ======================================================================== *)

Lemma closures_hold_forget_slot_var (known : known_closures) (vars : env)
    (statics : static_env) (x : variable) (v : value) :
  closures_hold known vars statics ->
  closures_hold (forget_slot_var x known) ((x, v) :: vars) statics.
Proof.
  intros Hhold entry Hin. apply filter_In in Hin as [Hin Hfresh].
  apply negb_true_iff in Hfresh.
  destruct (Hhold entry Hin) as (allocated & Hconst & Hlookup).
  exists allocated. split; [ | exact Hlookup ].
  rewrite (eval_static_const_fresh x v vars (fst entry) (snd entry) Hfresh).
  exact Hconst.
Qed.

Lemma closures_hold_bind_params (statics : static_env)
    (params : list variable) :
  ∀ (known : known_closures) (vs : list value) (vars handler_vars : env),
    closures_hold known vars statics ->
    bind_params params vs vars = Some handler_vars ->
    closures_hold (forget_slot_vars params known) handler_vars statics.
Proof.
  induction params as [ | param params IH];
    intros known [ | v vs] vars handler_vars Hhold; simpl; try discriminate.
  - intros [= <-]. exact Hhold.
  - apply IH, closures_hold_forget_slot_var, Hhold.
Qed.

(* A static environment that binds more. *)
Lemma closures_hold_extend (known : known_closures) (vars : env)
    (statics statics' : static_env) :
  closures_hold known vars statics -> statics_extend statics statics' ->
  closures_hold known vars statics'.
Proof.
  intros Hhold [_ Hvalues] entry Hin.
  destruct (Hhold entry Hin) as (allocated & Hconst & Hlookup).
  exists allocated. split; [ exact Hconst | apply Hvalues, Hlookup ].
Qed.

(* Each closure of a group that is bound is bound to what its constant
   makes. *)
Lemma static_closures_bound (vars : env) (bindings : list static_binding) :
  ∀ statics extended : static_env,
    eval_static_bindings vars bindings statics = Some extended ->
    ∀ entry : symbol * static_const,
      entry ∈ flat_map static_closure_made bindings ->
      closure_holds vars extended entry.
Proof.
  induction bindings as [ | binding bindings IH]; intros statics extended;
    simpl; [ intros _ entry [] | ].
  destruct binding as [id c | id | sym const].
  - destruct (lookup id (static_code statics)); [ discriminate | ].
    simpl. apply IH.
  - simpl. apply IH.
  - destruct (lookup sym (static_values statics)); [ discriminate | ].
    destruct (eval_static_const vars const) as [allocated | ] eqn:Hconst;
      simpl; [ | discriminate ].
    intros Heval entry Hin. apply in_app_or in Hin.
    destruct Hin as [Hin | Hin]; [ | exact (IH _ _ Heval entry Hin) ].
    assert (Hentry : entry = (sym, const)).
    { destruct const; simpl in Hin; try contradiction;
        destruct Hin as [<- | []]; reflexivity. }
    subst entry. exists allocated. split; [ exact Hconst | ]. simpl.
    destruct (eval_static_bindings_extend vars bindings _ _ Heval)
      as [_ Hvalues].
    apply Hvalues. simpl. rewrite Pos.eqb_refl. reflexivity.
Qed.

Lemma closures_hold_under_statics (known : known_closures) (vars : env)
    (statics : static_env) (bindings : list static_binding)
    (extended : static_env) :
  closures_hold known vars statics ->
  eval_static_bindings vars bindings statics = Some extended ->
  closures_hold (closures_under_statics known bindings) vars extended.
Proof.
  intros Hhold Heval entry Hin. unfold closures_under_statics in Hin.
  apply in_app_or in Hin. destruct Hin as [Hin | Hin].
  - exact (static_closures_bound vars bindings statics extended Heval entry
      Hin).
  - exact (closures_hold_extend known vars statics extended Hhold
      (eval_static_bindings_extend vars bindings _ _ Heval) entry Hin).
Qed.

(* ======================================================================== *)
(* Reading a slot                                                           *)
(* ======================================================================== *)

(* What the projection is, by cases on whether a function slot is
   named. *)
Definition projected (statics : static_env) (vars : env)
    (project_from : option function_slot) (slot : value_slot)
    (closure : simple) : option value :=
  match project_from with
  | Some from =>
    eval_named statics vars (Project_value_slot_from from slot closure)
  | None => eval_named statics vars (Project_value_slot slot closure)
  end.

Lemma slot_read_sound (known : known_closures) (vars : env)
    (statics : static_env) (project_from : option function_slot)
    (slot : value_slot) (closure contents : simple) (v : value) :
  closures_hold known vars statics ->
  slot_read known project_from slot closure = Some contents ->
  projected statics vars project_from slot closure = Some v ->
  eval_simple vars contents = Some v.
Proof.
  intros Hhold. unfold slot_read.
  destruct closure as [x | s | c]; try discriminate.
  destruct (lookup s known) as [const | ] eqn:Hknown; [ | discriminate ].
  apply lookup_in in Hknown.
  destruct (Hhold (s, const) Hknown) as (allocated & Hconst & Hlookup).
  simpl in Hconst, Hlookup.
  destruct const as [code slots | fslot syms decls slots | tag fields
    | contents' | bytes | elements];
    simpl; try discriminate.
  - simpl in Hconst.
    destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots;
      [ | discriminate ].
    simpl in Hconst. injection Hconst as <-.
    destruct project_from as [from | ]; simpl.
    + destruct (Pos.eqb unnamed_function_slot from) eqn:Hfrom;
        [ | discriminate ].
      intros Hread. unfold project_value_slot, closure_in_set.
      rewrite Hlookup. simpl. rewrite Hfrom.
      rewrite (eval_value_slots_lookup vars slot contents slots vs Hslots
        Hread).
      intros Hv. exact Hv.
    + intros Hread. unfold closure_of. rewrite Hlookup. simpl.
      rewrite (eval_value_slots_lookup vars slot contents slots vs Hslots
        Hread).
      intros Hv. exact Hv.
  - simpl in Hconst.
    destruct (lookup fslot decls) as [code | ] eqn:Hdecl; [ | discriminate ].
    destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots;
      [ | discriminate ].
    simpl in Hconst. injection Hconst as <-.
    destruct project_from as [from | ]; simpl.
    + destruct (Pos.eqb fslot from) eqn:Hfrom; [ | discriminate ].
      intros Hread. unfold project_value_slot, closure_in_set.
      rewrite Hlookup, Hdecl. simpl. rewrite Hfrom.
      rewrite (eval_value_slots_lookup vars slot contents slots vs Hslots
        Hread).
      intros Hv. exact Hv.
    + intros Hread. unfold closure_of. rewrite Hlookup, Hdecl. simpl.
      rewrite (eval_value_slots_lookup vars slot contents slots vs Hslots
        Hread).
      intros Hv. exact Hv.
Qed.

Lemma read_slot_named_holds (known : known_closures) (statics : static_env)
    (vars : env) (d : named) (v : value) :
  closures_hold known vars statics ->
  eval_named statics vars d = Some v ->
  eval_named statics vars (read_slot_named known d) = Some v.
Proof.
  intros Hhold Hd.
  destruct d as
    [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    try exact Hd.
  - unfold read_slot_named.
    destruct (slot_read known None slot closure) as [contents | ] eqn:Hread;
      [ | exact Hd ].
    exact (slot_read_sound known vars statics None slot closure contents v
      Hhold Hread Hd).
  - unfold read_slot_named.
    destruct (slot_read known (Some pfrom) slot closure) as [contents | ]
      eqn:Hread;
      [ | exact Hd ].
    exact (slot_read_sound known vars statics (Some pfrom) slot closure
      contents v Hhold Hread Hd).
Qed.

(* The same of the run of the machine, which runs a [Let] with the heap
   ([exec_named]): the rewrite changes a projection only, and neither a
   projection nor what it is put for is a primitive, so the machine runs
   both without the heap. *)
Lemma read_slot_named_exec (known : known_closures) (statics : static_env)
    (hp hp' : heap) (rd rd' : read_record) (vars : env) (d : named)
    (v : value) :
  closures_hold known vars statics ->
  exec_named statics hp rd vars d = Some (v, hp', rd') ->
  exec_named statics hp rd vars (read_slot_named known d)
  = Some (v, hp', rd').
Proof.
  intros Hhold Hd.
  assert (Hnot :
            (∀ (p : prim) (args : list simple), d <> Prim p args) ->
            ∀ (p : prim) (args : list simple),
              read_slot_named known d <> Prim p args).
  { destruct d as
      [s | p args | code slots | fslot decls slots | slot closure
      | pfrom slot closure | move_from move_to closure];
      intros Hd' q args'; simpl; try discriminate.
    - exact (Hd' q args').
    - destruct (slot_read known None slot closure); discriminate.
    - destruct (slot_read known (Some pfrom) slot closure); discriminate. }
  destruct d as
    [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    try exact Hd;
    (rewrite exec_named_not_prim in Hd by (intros ? ?; discriminate);
     rewrite exec_named_not_prim
       by (apply Hnot; intros ? ?; discriminate);
     destruct (eval_named statics vars _) as [w | ] eqn:Hw in Hd;
       [ | discriminate ];
     injection Hd as <- <- <-;
     rewrite (read_slot_named_holds known statics vars _ w Hhold Hw);
     reflexivity).
Qed.

Theorem read_slot_named_sound : read_slot_named_sound_statement.
Proof.
  intros known statics vars d v Hhold.
  apply read_slot_named_holds. intros [s const] Hin.
  exact (Hhold s const Hin).
Qed.

(* ======================================================================== *)
(* Calling                                                                  *)
(* ======================================================================== *)

(* A callee known to be a closure is a symbol bound to a closure of the
   code that is known. *)
Lemma callee_code_sound (known : known_closures) (vars : env)
    (statics : static_env) (callee : simple) (code : code_id) :
  closures_hold known vars statics ->
  callee_code known callee = Some code ->
  ∃ (s : symbol) (slots : list (value_slot * value)),
    callee = Symbol s
    ∧ closure_of statics (Value_symbol s) = Some (code, slots).
Proof.
  intros Hhold. destruct callee as [x | s | c]; simpl; try discriminate.
  destruct (lookup s known) as [const | ] eqn:Hknown; [ | discriminate ].
  apply lookup_in in Hknown.
  destruct (Hhold (s, const) Hknown) as (allocated & Hconst & Hlookup).
  simpl in Hconst, Hlookup.
  destruct const as [code' slots | fslot syms decls slots | tag fields
    | contents | bytes | elements];
    simpl; try discriminate.
  - intros [= ->]. simpl in Hconst.
    destruct (eval_value_slots vars slots) as [vs | ]; [ | discriminate ].
    simpl in Hconst. injection Hconst as <-.
    exists s, vs. split; [ reflexivity | ].
    unfold closure_of. rewrite Hlookup. reflexivity.
  - intros Hcode. simpl in Hconst. rewrite Hcode in Hconst.
    simpl in Hconst.
    destruct (eval_value_slots vars slots) as [vs | ]; [ | discriminate ].
    simpl in Hconst. injection Hconst as <-.
    exists s, vs. split; [ reflexivity | ].
    unfold closure_of. rewrite Hlookup. cbv beta iota. rewrite Hcode.
    reflexivity.
Qed.

Lemma call_known_closure_holds (known : known_closures)
    (statics : static_env) (hp : heap) (rd : read_record) (vars : env)
      (conts : cont_env)
    (ap : apply_expr) :
  closures_hold known vars statics ->
  step (State (Apply (call_known_closure known ap)) vars conts statics hp rd)
  = step (State (Apply ap) vars conts statics hp rd).
Proof.
  intros Hhold. unfold call_known_closure.
  destruct (ap_call_kind ap) as [named | | code_ids] eqn:Hkind;
    [ reflexivity | | reflexivity ].
  destruct (callee_code known (ap_callee ap)) as [code | ] eqn:Hcode;
    [ | reflexivity ].
  destruct (callee_code_sound known vars statics (ap_callee ap) code Hhold
    Hcode) as (s & slots & Hcallee & Hclosure).
  rewrite !step_apply. unfold exn_extra_simples.
  cbn [ap_callee ap_call_kind ap_args ap_continuation ap_exn_continuation
       ap_exn_extra_args ap_args_arity ap_return_arity].
  rewrite Hkind, Hcallee. cbn [eval_simple].
  destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | reflexivity ].
  destruct (eval_simples vars (map fst (ap_exn_extra_args ap)))
    as [extra | ];
    [ | reflexivity ].
  unfold call_function.
  rewrite !(shape_of_call_closure statics (Value_symbol s) _ vs _ _ eq_refl),
    (shape_of_closure_call_kind statics (Value_symbol s) (Direct code) vs _ _
       code slots Hclosure (or_intror eq_refl)).
  reflexivity.
Qed.

Theorem call_known_closure_sound : call_known_closure_sound_statement.
Proof.
  intros known statics hp rd vars conts ap Hhold.
  apply call_known_closure_holds. intros [s const] Hin.
  exact (Hhold s const Hin).
Qed.

(* ======================================================================== *)
(* The pass                                                                 *)
(* ======================================================================== *)

(* At a [Let], the projection has the value of what the slot holds or
   none; then the parts under binders, by the induction, with what is
   known there. *)
Fixpoint use_closures_knowing_refines (e : expr) :
  ∀ (known : known_closures) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record),
    closures_hold known vars statics ->
    refines (State e vars conts statics hp rd)
      (State (use_closures_knowing known e) vars conts statics hp rd)
with use_closures_in_handler_refines (h : continuation_handler) :
  ∀ (known : known_closures) (k : continuation) (body : expr) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) (rd : read_record),
    closures_hold known vars statics ->
    refines (State (Let_cont_non_recursive k h body) vars conts statics hp rd)
      (State
         (Let_cont_non_recursive k (use_closures_in_handler known h) body)
         vars conts statics hp rd).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap
         | ac | scrutinee arms | ];
      intros known vars conts statics hp rd Hhold; simpl.
    + apply let_refines_step_forward.
      * intros v hp' rd' Hd.
        exact (read_slot_named_exec known statics hp hp' rd rd' vars d v
          Hhold Hd).
      * intros v hp' rd' _ _. apply use_closures_knowing_refines.
        apply closures_hold_forget_slot_var, Hhold.
    + apply let_static_refines_step.
      destruct (eval_static_bindings vars bindings statics)
        as [extended | ] eqn:Hbindings; [ | exact I ].
      apply use_closures_knowing_refines.
      exact (closures_hold_under_statics known vars statics bindings extended
        Hhold Hbindings).
    + apply (refines_trans _
        (State
           (Let_cont_non_recursive k (use_closures_in_handler known h) body)
           vars conts statics hp rd)).
      * apply use_closures_in_handler_refines, Hhold.
      * apply let_cont_non_recursive_refines_step,
          use_closures_knowing_refines, Hhold.
    + apply let_cont_recursive_refines_step; [ reflexivity | intros _ ].
      apply use_closures_knowing_refines, Hhold.
    + apply equivalent_refines, step_eq_equivalent. symmetry.
      apply call_known_closure_holds, Hhold.
    + apply refines_refl.
    + apply refines_refl.
    + apply refines_refl.
  - destruct h as [params metadata handler].
    intros known k body vars conts statics hp rd Hhold. simpl.
    apply (let_cont_handler_refines_where (closures_hold known vars)
      (closures_hold_extend known vars));
      [ exact Hhold | ].
    intros vs handler_vars conts' statics' hp' rd' Hhold' _ Hbind.
    apply use_closures_knowing_refines.
    exact (closures_hold_bind_params statics' params known vs vars
      handler_vars Hhold' Hbind).
Qed.

Theorem use_known_closures_refines : use_known_closures_refines_statement.
Proof.
  intros e vars conts statics hp rd. apply use_closures_knowing_refines.
  intros entry [].
Qed.

(* ======================================================================== *)
(* Renaming                                                                 *)
(* ======================================================================== *)

(* The law of the pass is proved from the definitions by [equivariance]
   (FlambdaEquivarianceTactic.v). What is known is a map from symbols to
   constants, and its type does not say how it is renamed: the instance
   says that it is by [rename_known_closures]. Besides it the tactic is
   given, to rewrite with:
   - [map_values] as the [map] that it is, so that the laws of lists
     apply to what is known renamed;
   - the lookup of a symbol in what is known renamed, which is a law of
     maps (FlambdaMapProofs.lookup_map_values) in that form;
   - the test of a variable in a constant of what is known, in the form
     that computation gives it;
   - the laws of [forget_slot_var] and [forget_slot_vars], stated from
     their types ([equivariance_of]; [Check] shows the statement):
     without the first the tactic does not close the second, whose
     induction needs it at what is known of each step;
   - the law of [slot_read], also stated from its type. Its lookups
     are the scrutinee of a [match], where only the step that tries
     every law on the whole goal finds them: it is proved with that
     step, and after it the database is applied by its one pass, the
     faster step, with which every later proof passes. *)

#[local] Instance known_closures_renaming : Renaming_of known_closures :=
  rename_known_closures.

Lemma map_values_as_map {A B : Type} (f : A -> B) (bindings : finite_map A) :
  map_values f bindings
  = map (fun binding => (fst binding, f (snd binding))) bindings.
Proof. reflexivity. Qed.
#[local] Hint Rewrite @map_values_as_map : equivariance.

Lemma lookup_renamed_known (r : renaming) (s : symbol)
    (known : known_closures) :
  lookup s
    (map (fun binding => (fst binding, rename_static_const r (snd binding)))
       known)
  = option_map (rename_static_const r) (lookup s known).
Proof. apply (lookup_map_values _ _ (rename_static_const r)). Qed.
#[local] Hint Rewrite lookup_renamed_known : equivariance.

Lemma slot_read_renamed : equivariance_of slot_read.
Proof. equivariance. Qed.
#[local] Hint Rewrite slot_read_renamed : equivariance.

#[local] Ltac eqv_database ::= eqv_database_by_one_pass.

Lemma var_in_static_const_renamed (r : renaming) (x : variable) (s : symbol)
    (const : static_const) :
  var_in_static_binding (rename_variable r x)
    (Bind_symbol s (rename_static_const r const))
  = var_in_static_binding x (Bind_symbol s const).
Proof. equivariance. Qed.
#[local] Hint Rewrite var_in_static_const_renamed : equivariance.

Lemma forget_slot_var_renamed : equivariance_of forget_slot_var.
Proof. equivariance. Qed.
#[local] Hint Rewrite forget_slot_var_renamed : equivariance.

Lemma forget_slot_vars_renamed : equivariance_of forget_slot_vars.
Proof. equivariance. Qed.
#[local] Hint Rewrite forget_slot_vars_renamed : equivariance.

Theorem use_closures_knowing_equivariant :
  use_closures_knowing_equivariant_statement.
Proof. equivariance. Qed.

Theorem use_known_closures_equivariant :
  use_known_closures_equivariant_statement.
Proof. intros r e. exact (use_closures_knowing_equivariant r [] e). Qed.
