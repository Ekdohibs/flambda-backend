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

(* Proofs of the statements of FlambdaAliasesSpec.v. Nothing here needs
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaHeapSpec FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaContextProofs
  FlambdaNominalSpec FlambdaNominalProofs FlambdaApplyProofs
  FlambdaNormaliseSpec FlambdaNormaliseProofs.
From OxCamlFlambda2Types Require Import FlambdaTypesSpec FlambdaTypesProofs
  FlambdaAliasesSpec.
Import ListNotations.

(* ======================================================================== *)
(* Forgetting a name                                                        *)
(* ======================================================================== *)

Lemma env_satisfies_incl (tenv tenv' : typing_env) (vars : env) :
  (∀ entry : variable * flambda_type, entry ∈ tenv' -> entry ∈ tenv) ->
  env_satisfies tenv vars -> env_satisfies tenv' vars.
Proof. intros Hincl Hsat x t Hin. apply Hsat, Hincl, Hin. Qed.

Lemma drop_var_incl (x : variable) (tenv : typing_env)
    (entry : variable * flambda_type) :
  entry ∈ drop_var x tenv -> entry ∈ tenv.
Proof. unfold drop_var. intros Hin. apply filter_In in Hin. tauto. Qed.

Lemma drop_var_fresh (x : variable) (tenv : typing_env) :
  ~ var_in_typing_env x (drop_var x tenv).
Proof.
  intros (y & t & Hin & Hmention). unfold drop_var in Hin.
  apply filter_In in Hin. destruct Hin as [_ Hkept].
  apply andb_true_iff in Hkept as [Hkey Htype].
  apply negb_true_iff in Hkey, Htype.
  destruct Hmention as [-> | Hfree].
  - rewrite Pos.eqb_refl in Hkey. discriminate.
  - destruct t as [ | | s | code slots]; simpl in Htype, Hfree;
      try contradiction.
    + apply var_in_simple_sound in Hfree. congruence.
    + apply var_in_value_slots_sound in Hfree. congruence.
Qed.

Lemma drop_vars_incl (xs : list variable) (tenv : typing_env)
    (entry : variable * flambda_type) :
  entry ∈ drop_vars xs tenv -> entry ∈ tenv.
Proof.
  induction xs as [ | x xs IH]; simpl; [ auto | ].
  intros Hin. apply IH, (drop_var_incl x), Hin.
Qed.

Lemma drop_vars_fresh (xs : list variable) (tenv : typing_env) (x : variable) :
  x ∈ xs -> ~ var_in_typing_env x (drop_vars xs tenv).
Proof.
  induction xs as [ | y xs IH]; simpl; [ contradiction | ].
  intros [-> | Hin]; [ apply drop_var_fresh | ].
  intros (z & t & Hz & Hmention). apply (IH Hin).
  exists z, t. split; [ exact (drop_var_incl y _ _ Hz) | exact Hmention ].
Qed.

(* ======================================================================== *)
(* Following aliases at the head of a term                                  *)
(* ======================================================================== *)

Lemma canonical_value_slots_sound (tenv : typing_env) (vars : env)
    (slots : list (value_slot * simple)) :
  env_satisfies tenv vars ->
  eval_value_slots vars (canonical_value_slots tenv slots)
  = eval_value_slots vars slots.
Proof.
  intros Hsat. induction slots as [ | [slot s] slots IH]; simpl;
    [ reflexivity | ].
  rewrite (canonical_simple_sound tenv vars s Hsat), IH. reflexivity.
Qed.

Lemma find_closure_in (tenv : typing_env) (x : variable) (code : code_id)
    (slots : list (value_slot * simple)) :
  find_closure tenv x = Some (code, slots) ->
  (x, Closure_of code slots) ∈ tenv.
Proof.
  induction tenv as [ | [y t] tenv IH]; simpl; [ discriminate | ].
  destruct t as [ | | s | code' slots'];
    try (intros Hfind; right; exact (IH Hfind)).
  destruct (Pos.eqb_spec x y) as [-> | _].
  - intros [= -> ->]. left. reflexivity.
  - intros Hfind. right. exact (IH Hfind).
Qed.

(* What a known closure is: the variable is bound to a closure of the
   code, whose slots have the values of what the type says they hold. *)
Lemma closure_of_simple_sound (tenv : typing_env) (vars : env) (s : simple)
    (code : code_id) (slots : list (value_slot * simple)) :
  env_satisfies tenv vars ->
  closure_of_simple tenv s = Some (code, slots) ->
  ∃ slot_values : list (value_slot * value),
    eval_value_slots vars slots = Some slot_values
    ∧ eval_simple vars s = Some (Value_closure code slot_values).
Proof.
  intros Hsat. destruct s as [x | sym | c]; simpl; try discriminate.
  intros Hfind.
  destruct (Hsat x _ (find_closure_in tenv x code slots Hfind))
    as (v & Hlookup & slot_values & Hslots & ->).
  exists slot_values. split; assumption.
Qed.

(* A slot of evaluated value slots has the value of what it held. *)
Lemma eval_value_slots_lookup (vars : env) (slot : value_slot)
    (slots : list (value_slot * simple)) :
  ∀ (slot_values : list (value_slot * value)) (contents : simple),
    eval_value_slots vars slots = Some slot_values ->
    lookup slot slots = Some contents ->
    lookup slot slot_values = eval_simple vars contents.
Proof.
  induction slots as [ | [slot' s] slots IH]; intros slot_values contents;
    simpl; [ discriminate | ].
  destruct (eval_simple vars s) as [v | ] eqn:Hs; [ | discriminate ].
  destruct (eval_value_slots vars slots) as [rest | ]; [ | discriminate ].
  simpl. intros [= <-]. simpl.
  destruct (Pos.eqb slot slot').
  - intros [= <-]. symmetry. exact Hs.
  - apply (IH rest contents eq_refl).
Qed.

Lemma canonical_project_sound (tenv : typing_env) (vars : env)
    (statics : static_env) (slot : value_slot) (closure : simple) :
  env_satisfies tenv vars ->
  eval_named statics vars (canonical_project tenv slot closure)
  = eval_named statics vars (Project_value_slot slot closure).
Proof.
  intros Hsat. unfold canonical_project.
  destruct (closure_of_simple tenv closure) as [[code slots] | ] eqn:Hclosure;
    [ | reflexivity ].
  destruct (lookup slot slots) as [contents | ] eqn:Hslot; [ | reflexivity ].
  destruct (closure_of_simple_sound tenv vars closure code slots Hsat
              Hclosure)
    as (slot_values & Hslots & Heval).
  simpl. rewrite Heval. simpl.
  symmetry.
  exact (eval_value_slots_lookup vars slot slots slot_values contents Hslots
    Hslot).
Qed.

Lemma canonical_named_sound (tenv : typing_env) (vars : env)
    (statics : static_env) (d : named) :
  env_satisfies tenv vars ->
  eval_named statics vars (canonical_named tenv d) = eval_named statics vars d.
Proof.
  intros Hsat.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure]; simpl.
  - apply canonical_simple_sound, Hsat.
  - rewrite (canonical_simples_sound tenv vars args Hsat). reflexivity.
  - rewrite (canonical_value_slots_sound tenv vars slots Hsat). reflexivity.
  - rewrite (canonical_value_slots_sound tenv vars slots Hsat). reflexivity.
  - rewrite (canonical_project_sound tenv vars statics slot _ Hsat). simpl.
    rewrite (canonical_simple_sound tenv vars closure Hsat). reflexivity.
  - rewrite (canonical_simple_sound tenv vars closure Hsat). reflexivity.
  - rewrite (canonical_simple_sound tenv vars closure Hsat). reflexivity.
Qed.

(* The same of the run of the named expression: a primitive is run on the
   values of its arguments, and what is not a primitive has the value that
   [eval_named] gives it, with the heap and the record as they came. *)
Lemma canonical_named_sound_exec (tenv : typing_env) (vars : env)
    (statics : static_env) (hp : heap) (rd : read_record) (d : named) :
  env_satisfies tenv vars ->
  exec_named statics hp rd vars (canonical_named tenv d)
  = exec_named statics hp rd vars d.
Proof.
  intros Hsat.
  assert (Hnot : ∀ d' : named,
    (∀ (p : prim) (args : list simple), d' <> Prim p args) ->
    (∀ (p : prim) (args : list simple), canonical_named tenv d' <> Prim p args)
    -> exec_named statics hp rd vars (canonical_named tenv d')
       = exec_named statics hp rd vars d').
  { intros d' Hd' Hcanonical.
    rewrite (exec_named_not_prim statics hp rd vars _ Hcanonical),
      (exec_named_not_prim statics hp rd vars d' Hd'),
      (canonical_named_sound tenv vars statics d' Hsat).
    reflexivity. }
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    try (apply Hnot; simpl; intros ? ?; discriminate).
  - simpl. apply exec_named_args.
    exact (canonical_simples_sound tenv vars args Hsat).
  - apply Hnot; [ intros ? ?; discriminate | ].
    simpl. unfold canonical_project. intros p args.
    destruct (closure_of_simple tenv (canonical_simple tenv closure))
      as [[code slots] | ]; [ destruct (lookup slot slots) | ]; discriminate.
Qed.

(* A call and the same call with its aliases followed and, if the callee
   is a known closure, made direct: they take the same step. *)
Lemma canonical_apply_equivalent (tenv : typing_env) (ap : apply_expr)
    (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
      (rd : read_record) :
  env_satisfies tenv vars ->
  equivalent (State (Apply ap) vars conts statics hp rd)
    (State (Apply (canonical_apply tenv ap)) vars conts statics hp rd).
Proof.
  intros Hsat. apply step_eq_equivalent. rewrite !step_apply.
  assert (Hextras : exn_extra_simples (canonical_apply tenv ap)
                    = map (canonical_simple tenv) (exn_extra_simples ap))
    by (unfold exn_extra_simples; simpl; rewrite !map_map; reflexivity).
  rewrite Hextras. simpl.
  rewrite (canonical_simples_sound tenv vars (ap_args ap) Hsat),
    (canonical_simples_sound tenv vars (exn_extra_simples ap) Hsat).
  unfold canonical_call_kind.
  destruct (ap_call_kind ap) as [code | | code_ids];
    [ rewrite (canonical_simple_sound tenv vars (ap_callee ap) Hsat);
      reflexivity
    |
    | rewrite (canonical_simple_sound tenv vars (ap_callee ap) Hsat);
      reflexivity ].
  destruct (closure_of_simple tenv (canonical_simple tenv (ap_callee ap)))
    as [[code slots] | ] eqn:Hclosure;
    [ | rewrite (canonical_simple_sound tenv vars (ap_callee ap) Hsat);
        reflexivity ].
  destruct (closure_of_simple_sound tenv vars _ code slots Hsat Hclosure)
    as (slot_values & _ & Heval).
  rewrite (canonical_simple_sound tenv vars (ap_callee ap) Hsat) in Heval.
  rewrite (canonical_simple_sound tenv vars (ap_callee ap) Hsat), Heval.
  destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | reflexivity ].
  destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
    [ | reflexivity ].
  unfold call_function, shape_of_call, shape_of_closure_call. simpl.
  rewrite Pos.eqb_refl. reflexivity.
Qed.

Lemma canonical_static_bindings_sound (tenv : typing_env) (vars : env)
    (bindings : list static_binding) :
  env_satisfies tenv vars ->
  ∀ statics : static_env,
    eval_static_bindings vars (map (canonical_static_binding tenv) bindings)
      statics
    = eval_static_bindings vars bindings statics.
Proof.
  intros Hsat.
  induction bindings as [ | binding bindings IH]; intros statics;
    [ reflexivity | ].
  destruct binding
    as [id c | id
       | sym [code slots | fslot syms decls slots | tag fields | s | bytes
              | elements]];
    simpl; try apply IH.
  - destruct (lookup id (static_code statics)); [ reflexivity | apply IH ].
  - destruct (lookup sym (static_values statics)); [ reflexivity | ].
    rewrite (canonical_value_slots_sound tenv vars slots Hsat).
    destruct (eval_value_slots vars slots); simpl; [ apply IH | reflexivity ].
  - destruct (lookup sym (static_values statics)); [ reflexivity | ].
    rewrite (canonical_value_slots_sound tenv vars slots Hsat).
    destruct (lookup fslot decls); simpl; [ | reflexivity ].
    destruct (eval_value_slots vars slots); simpl; [ apply IH | reflexivity ].
  - destruct (lookup sym (static_values statics)); [ reflexivity | ].
    rewrite (canonical_simples_sound tenv vars fields Hsat).
    destruct (eval_simples vars fields); simpl; [ apply IH | reflexivity ].
  - destruct (lookup sym (static_values statics)); [ reflexivity | ].
    rewrite (canonical_simple_sound tenv vars s Hsat).
    destruct (eval_simple vars s); simpl; [ apply IH | reflexivity ].
  - destruct (lookup sym (static_values statics)); [ reflexivity | ].
    apply IH.
  - destruct (lookup sym (static_values statics)); [ reflexivity | ].
    rewrite (canonical_simples_sound tenv vars elements Hsat).
    destruct (eval_simples vars elements); simpl; [ apply IH | reflexivity ].
Qed.

Lemma find_arm_canonical (tenv : typing_env) (i : Z)
    (arms : list (Z * apply_cont)) :
  find_arm i
    (map (fun arm => (fst arm, canonical_apply_cont tenv (snd arm))) arms)
  = option_map (canonical_apply_cont tenv) (find_arm i arms).
Proof.
  induction arms as [ | [discriminant ac] arms IH]; simpl; [ reflexivity | ].
  destruct (Z.eqb i discriminant); [ reflexivity | exact IH ].
Qed.

Lemma canonical_apply_cont_equivalent (tenv : typing_env) (ac : apply_cont)
    (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
      (rd : read_record) :
  env_satisfies tenv vars ->
  equivalent (State (Apply_cont ac) vars conts statics hp rd)
    (State (Apply_cont (canonical_apply_cont tenv ac)) vars conts statics hp
       rd).
Proof.
  intros Hsat. apply step_eq_equivalent. rewrite !step_apply_cont. simpl.
  rewrite (canonical_simples_sound tenv vars (ac_args ac) Hsat). reflexivity.
Qed.

(* ======================================================================== *)
(* Under binders                                                            *)
(* ======================================================================== *)

(* A store has a value only where that value is unit. *)
Lemma exec_heap_store_unit (statics : static_env) (h h' : heap)
    (q : heap_primitive) (vs : list value) (v : value) :
  exec_heap_primitive statics h q vs = Some (v, h') ->
  match q with
  | Block_set _ _ _ | Array_set _ _ => v = unit_value
  | Make_block_with_identity _ _ | Make_mutable_array _ => True
  end.
Proof.
  intros Hexec.
  destruct q as [mut tag | kind init field | kind | kind set_kind];
    try exact I; simpl in Hexec;
    repeat match type of Hexec with
           | match ?e with _ => _ end = Some _ =>
             destruct e; try discriminate Hexec
           end;
    injection Hexec as <- _; reflexivity.
Qed.

(* What a primitive of the heap gives at a [Let] is what
   [exec_heap_primitive] gives on the values of its arguments. *)
Lemma exec_named_heap_inv (statics : static_env) (hp hp' : heap)
    (rd rd' : read_record) (vars : env) (q : heap_primitive)
    (args : list simple) (v : value) :
  exec_named statics hp rd vars (Prim (Heap q) args) = Some (v, hp', rd') ->
  ∃ vs : list value, exec_heap_primitive statics hp q vs = Some (v, hp').
Proof.
  unfold exec_named, exec_prim. intros Hd.
  destruct (eval_simples vars args) as [vs | ]; [ | discriminate ].
  destruct (forallb (argument_bound statics hp) vs); [ | discriminate ].
  simpl in Hd.
  destruct (exec_heap_primitive statics hp q vs) as [[w hp''] | ] eqn:Hexec;
    [ | discriminate ].
  injection Hd as <- <- _. exists vs. exact Hexec.
Qed.

(* The environment under a [Let] satisfies what is known there. *)
Lemma env_satisfies_under_let (tenv : typing_env) (x : variable) (d : named)
    (vars : env) (statics : static_env) (v : value) :
  env_satisfies tenv vars ->
  eval_named statics vars d = Some v ->
  env_satisfies (typing_env_under_let tenv x d) ((x, v) :: vars).
Proof.
  intros Hsat Hd.
  assert (Hdropped : env_satisfies (drop_var x tenv) vars)
    by exact (env_satisfies_incl tenv _ vars (drop_var_incl x tenv) Hsat).
  assert (Hunknown : env_satisfies (drop_var x tenv) ((x, v) :: vars)).
  { apply (env_satisfies_incl ((x, Unknown) :: drop_var x tenv));
      [ intros entry Hin; right; exact Hin | ].
    exact (env_satisfies_extend _ vars x v Unknown Hdropped
      (drop_var_fresh x tenv) I). }
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure]; simpl;
    try exact Hunknown.
  - destruct (var_in_simple x s) eqn:Hmention; [ exact Hunknown | ].
    apply (env_satisfies_extend _ vars x v (Equals s) Hdropped
      (drop_var_fresh x tenv)).
    simpl in Hd |- *. rewrite <- Hd. apply eval_simple_extend.
    intros Hfree. apply var_in_simple_sound in Hfree. congruence.
  - (* A store has no value by [eval_named]. *)
    destruct p as [u | b | vr | q | r]; try exact Hunknown.
    destruct q as [mut tag | kind init field | kind | kind set_kind];
      try exact Hunknown;
      simpl in Hd; destruct (eval_simples vars args); discriminate Hd.
  - destruct (var_in_value_slots x slots) eqn:Hmention; [ exact Hunknown | ].
    apply (env_satisfies_extend _ vars x v (Closure_of code slots) Hdropped
      (drop_var_fresh x tenv)).
    simpl in Hd |- *.
    destruct (eval_value_slots vars slots) as [slot_values | ] eqn:Hslots;
      [ | discriminate ].
    simpl in Hd. injection Hd as <-. exists slot_values.
    split; [ | reflexivity ].
    rewrite <- Hslots. apply eval_value_slots_extend.
    intros Hfree. apply var_in_value_slots_sound in Hfree. congruence.
Qed.

(* The same, of the run of the named expression: of a primitive nothing
   is recorded but that a store returns unit, and what is not a
   primitive has the value that [eval_named] gives it. *)
Lemma env_satisfies_under_let_exec (tenv : typing_env) (x : variable)
    (d : named) (vars : env) (statics : static_env) (hp hp' : heap)
    (rd rd' : read_record) (v : value) :
  env_satisfies tenv vars ->
  exec_named statics hp rd vars d = Some (v, hp', rd') ->
  env_satisfies (typing_env_under_let tenv x d) ((x, v) :: vars).
Proof.
  intros Hsat Hd.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    try (rewrite exec_named_not_prim in Hd by (intros ? ?; discriminate);
         destruct (eval_named statics vars _) as [w | ] eqn:Hw;
           [ | discriminate ];
         injection Hd as <- <- <-;
         exact (env_satisfies_under_let tenv x _ vars statics w Hsat Hw)).
  assert (Hdropped : env_satisfies (drop_var x tenv) vars)
    by exact (env_satisfies_incl tenv _ vars (drop_var_incl x tenv) Hsat).
  assert (Hunknown : env_satisfies (drop_var x tenv) ((x, v) :: vars)).
  { apply (env_satisfies_incl ((x, Unknown) :: drop_var x tenv));
      [ intros entry Hin; right; exact Hin | ].
    exact (env_satisfies_extend _ vars x v Unknown Hdropped
      (drop_var_fresh x tenv) I). }
  destruct p as [u | b | vr | q | r]; try exact Hunknown.
  destruct (exec_named_heap_inv statics hp hp' rd rd' vars q args v Hd)
    as [vs Hexec].
  apply exec_heap_store_unit in Hexec.
  destruct q as [mut tag | kind init field | kind | kind set_kind];
    try exact Hunknown; subst v;
    exact (env_satisfies_extend _ vars x unit_value
      (Equals (Const (Tagged_immediate 0))) Hdropped (drop_var_fresh x tenv)
      eq_refl).
Qed.

Lemma let_equivalent_in (tenv : typing_env) (x : variable)
    (kind : flambda_kind) (d : named) (body body' : expr) :
  expr_equivalent_in (typing_env_under_let tenv x d) body body' ->
  expr_equivalent_in tenv (Let x kind d body) (Let x kind d body').
Proof.
  intros Hbody vars conts statics hp rd Hsat.
  apply let_equivalent_step_exec; [ reflexivity | intros v hp' rd' Hd _ ].
  apply Hbody.
  exact (env_satisfies_under_let_exec tenv x d vars statics hp hp' rd rd' v
    Hsat Hd).
Qed.

Lemma let_static_equivalent_in (tenv : typing_env)
    (bindings : list static_binding) (body body' : expr) :
  expr_equivalent_in tenv body body' ->
  expr_equivalent_in tenv (Let_static bindings body)
    (Let_static bindings body').
Proof.
  intros Hbody vars conts statics hp rd Hsat.
  apply let_static_equivalent_step.
  destruct (eval_static_bindings vars bindings statics) as [extended | ];
    [ apply Hbody, Hsat | exact I ].
Qed.

(* A handler may be replaced by one equivalent to it under what is known
   where it is defined, less what mentions its parameters. *)
Lemma let_cont_handler_equivalent_in (tenv : typing_env) (k : continuation)
    (params : list variable) (metadata : handler_metadata)
    (handler handler' body : expr) :
  expr_equivalent_in (drop_vars params tenv) handler handler' ->
  expr_equivalent_in tenv
    (Let_cont_non_recursive k (Handler params metadata handler) body)
    (Let_cont_non_recursive k (Handler params metadata handler') body).
Proof.
  intros Hhandler vars conts statics hp rd Hsat.
  assert (Hruns :
    ∀ (vs : list value) (handler_vars : env),
      bind_params params vs vars = Some handler_vars ->
      env_satisfies (drop_vars params tenv) handler_vars).
  { intros vs handler_vars Hbind.
    apply (env_satisfies_bind_params _ params vs vars handler_vars);
      [ | intros param Hin; apply drop_vars_fresh, Hin | exact Hbind ].
    exact (env_satisfies_incl tenv _ vars (drop_vars_incl params tenv) Hsat). }
  apply refines_antisymmetric; apply let_cont_handler_refines_at;
    intros vs handler_vars conts' statics' hp' rd' _ Hbind; apply
      equivalent_refines;
    [ | apply equivalent_sym ];
    apply Hhandler, (Hruns vs handler_vars Hbind).
Qed.

Lemma expr_equivalent_in_trans (tenv : typing_env) (e e' e'' : expr) :
  expr_equivalent_in tenv e e' -> expr_equivalent_in tenv e' e'' ->
  expr_equivalent_in tenv e e''.
Proof.
  intros H H' vars conts statics hp rd Hsat.
  exact (equivalent_trans _ _ _ (H vars conts statics hp rd_ Hsat)
    (H' vars conts statics hp rd_ Hsat)).
Qed.

(* ======================================================================== *)
(* Following aliases                                                        *)
(* ======================================================================== *)

(* At each node: the head with its aliases followed is equivalent, by
   evaluating the same; then the parts under binders, by the induction. *)
Fixpoint follow_aliases_equivalent_in (e : expr) :
  ∀ tenv : typing_env, expr_equivalent_in tenv e (follow_aliases tenv e)
with follow_aliases_in_handler_equivalent_in (h : continuation_handler) :
  ∀ (tenv : typing_env) (k : continuation) (body : expr),
    expr_equivalent_in tenv (Let_cont_non_recursive k h body)
      (Let_cont_non_recursive k (follow_aliases_in_handler tenv h) body).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros tenv; simpl.
    + (* Let *)
      apply (expr_equivalent_in_trans tenv _
        (Let x kind (canonical_named tenv d) body)).
      * intros vars conts statics hp rd Hsat.
        apply let_equivalent_step_exec;
          [ apply (canonical_named_sound_exec tenv vars statics hp rd d Hsat)
          | intros v hp' rd' _ _; apply equivalent_refl ].
      * apply let_equivalent_in, follow_aliases_equivalent_in.
    + (* Let_static *)
      apply (expr_equivalent_in_trans tenv _
        (Let_static (map (canonical_static_binding tenv) bindings) body)).
      * intros vars conts statics hp rd Hsat. apply step_eq_equivalent.
        rewrite !step_let_static,
          (canonical_static_bindings_sound tenv vars bindings Hsat).
        reflexivity.
      * apply let_static_equivalent_in, follow_aliases_equivalent_in.
    + (* Let_cont_non_recursive *)
      apply (expr_equivalent_in_trans tenv _
        (Let_cont_non_recursive k (follow_aliases_in_handler tenv h) body));
        [ apply follow_aliases_in_handler_equivalent_in | ].
      intros vars conts statics hp rd Hsat.
      apply let_cont_non_recursive_equivalent_step.
      apply (follow_aliases_equivalent_in body tenv), Hsat.
    + (* Let_cont_recursive *)
      intros vars conts statics hp rd Hsat.
      apply let_cont_recursive_equivalent_step; [ reflexivity | intros _ ].
      apply (follow_aliases_equivalent_in body tenv), Hsat.
    + (* Apply *)
      intros vars conts statics hp rd Hsat.
      apply canonical_apply_equivalent, Hsat.
    + (* Apply_cont *)
      intros vars conts statics hp rd Hsat.
      apply canonical_apply_cont_equivalent, Hsat.
    + (* Switch *)
      intros vars conts statics hp rd Hsat.
      destruct (eval_simple vars scrutinee) as [v | ] eqn:Hscrutinee;
        [ destruct (naked_immediate_of_value v) as [i | ] eqn:Himmediate;
          [ destruct (find_arm i arms) as [ac | ] eqn:Hfind | ]
        | ].
      * eapply equivalent_step;
          [ rewrite step_switch, Hscrutinee, Himmediate, Hfind; reflexivity
          | rewrite step_switch,
              (canonical_simple_sound tenv vars scrutinee Hsat),
              Hscrutinee, Himmediate, find_arm_canonical, Hfind;
            reflexivity
          | ].
        apply canonical_apply_cont_equivalent, Hsat.
      * apply step_eq_equivalent.
        rewrite !step_switch,
          (canonical_simple_sound tenv vars scrutinee Hsat),
          Hscrutinee, Himmediate, find_arm_canonical, Hfind.
        reflexivity.
      * apply step_eq_equivalent.
        rewrite !step_switch,
          (canonical_simple_sound tenv vars scrutinee Hsat),
          Hscrutinee, Himmediate.
        reflexivity.
      * apply step_eq_equivalent.
        rewrite !step_switch,
          (canonical_simple_sound tenv vars scrutinee Hsat),
          Hscrutinee.
        reflexivity.
    + (* Invalid *)
      intros vars conts statics hp rd Hsat. apply equivalent_refl.
  - destruct h as [params metadata handler]. intros tenv k body. simpl.
    apply let_cont_handler_equivalent_in, follow_aliases_equivalent_in.
Qed.

Theorem follow_aliases_equivalent : follow_aliases_equivalent_statement.
Proof.
  intros e vars conts statics hp rd.
  apply (follow_aliases_equivalent_in e []). intros x t [].
Qed.
