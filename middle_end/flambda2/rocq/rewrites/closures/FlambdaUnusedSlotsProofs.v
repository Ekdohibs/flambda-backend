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

(* Proofs of the statements of FlambdaUnusedSlotsSpec.v. Nothing here
   needs review.

   The unit and the unit without the slots are run side by side, in
   step. The second state is the first with the slots removed from the
   static sets of closures of its terms and from the closures that its
   symbols are bound to ([trim_state]); the values of its variables are
   those of the first. Nothing of the first run reads a slot removed:
   its terms have no projection of one, which a run keeps
   (FlambdaClosuresProofs.terms_inv_step), and no primitive looks at
   the value slots of a closure
   (FlambdaPrimitiveProofs.eval_prim_related, with equality for the
   relation). *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaPrimitiveModel FlambdaValueSpec
  FlambdaPrimitiveSpec FlambdaPrimitiveProofs FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaNominalSpec FlambdaClosuresProofs
  FlambdaUnusedStaticsSpec FlambdaImportsSpec FlambdaUnusedSlotsSpec
  FlambdaEquivarianceTactic.
Import ListNotations.

(* ======================================================================== *)
(* Lists and maps                                                           *)
(* ======================================================================== *)

Lemma lookup_map_snd {A B : Type} (f : A -> B) (name : positive)
    (l : list (positive * A)) :
  lookup name (map (fun binding => (fst binding, f (snd binding))) l)
  = option_map f (lookup name l).
Proof.
  induction l as [ | [k a] l IH]; simpl; [ reflexivity | ].
  destruct (Pos.eqb name k); [ reflexivity | exact IH ].
Qed.

Lemma forall2_eq_refl {A : Type} (l : list A) : Forall2 eq l l.
Proof. induction l; constructor; [ reflexivity | assumption ]. Qed.

Lemma forall2_eq {A : Type} (l l' : list A) : Forall2 eq l l' -> l = l'.
Proof. induction 1; [ reflexivity | congruence ]. Qed.

Section Dropped.

Variable D : list value_slot.

(* A slot that is kept is found where it was. *)
Lemma lookup_keep_slots {A : Type} (slot : value_slot)
    (slots : list (value_slot * A)) :
  among slot D = false ->
  lookup slot (keep_slots D slots) = lookup slot slots.
Proof.
  intros Hslot. unfold keep_slots.
  induction slots as [ | [k a] slots IH]; simpl; [ reflexivity | ].
  destruct (Pos.eqb_spec slot k) as [-> | Hne].
  - rewrite Hslot. simpl. rewrite Pos.eqb_refl. reflexivity.
  - destruct (among k D); simpl; [ exact IH | ].
    destruct (Pos.eqb_spec slot k); [ contradiction | exact IH ].
Qed.

(* ======================================================================== *)
(* The second state                                                         *)
(* ======================================================================== *)

Definition trim_allocated (allocated : statically_allocated_value)
    : statically_allocated_value :=
  match allocated with
  | Static_closure code slots => Static_closure code (keep_slots D slots)
  | Static_closure_of_set slot symbols decls slots =>
    Static_closure_of_set slot symbols decls (keep_slots D slots)
  | Static_block _ _ | Static_boxed_number _ | Static_string _
  | Static_immutable_array _ | Static_imported _ =>
    allocated
  end.

Definition trim_statics (statics : static_env) : static_env :=
  Static_env
    (map (fun binding => (fst binding, drop_slots_code D (snd binding)))
       (static_code statics))
    (map (fun binding => (fst binding, trim_allocated (snd binding)))
       (static_values statics)).

Fixpoint trim_cont (c : cont_value) : cont_value :=
  match c with
  | Closure h vars conts recursive =>
    Closure (drop_slots_handler D h) vars
      (map (fun binding => (fst binding, trim_cont (snd binding))) conts)
      (map (fun '(k, h) => (k, drop_slots_handler D h)) recursive)
  | Outer k => Outer k
  | Extra_args c extra => Extra_args (trim_cont c) extra
  | Over_application remaining remaining_arity return_arity return_to exn_to =>
    Over_application remaining remaining_arity return_arity
      (trim_cont return_to) (trim_cont exn_to)
  end.

Definition trim_conts (conts : cont_env) : cont_env :=
  map (fun binding => (fst binding, trim_cont (snd binding))) conts.

Definition trim_state (s : state) : state :=
  State (drop_slots D (state_expr s)) (state_vars s)
    (trim_conts (state_conts s)) (trim_statics (state_statics s))
      (state_heap s) (state_reads s).

Lemma lookup_trim_statics (s : symbol) (statics : static_env) :
  lookup s (static_values (trim_statics statics))
  = option_map trim_allocated (lookup s (static_values statics)).
Proof. unfold trim_statics. apply lookup_map_snd. Qed.

Lemma lookup_trim_code (id : code_id) (statics : static_env) :
  lookup id (static_code (trim_statics statics))
  = option_map (drop_slots_code D) (lookup id (static_code statics)).
Proof. unfold trim_statics. apply lookup_map_snd. Qed.

(* What a name is bound to in the second static environment, whether or
   not the environment has been unfolded. *)
Ltac lookup_trimmed :=
  first
    [ rewrite lookup_trim_statics | rewrite lookup_trim_code
    | rewrite lookup_map_snd ].

(* ======================================================================== *)
(* The observations                                                         *)
(* ======================================================================== *)

Lemma block_of_drop (statics : static_env) (v : value) :
  block_of (trim_statics statics) v = block_of statics v.
Proof.
  destruct v; simpl; try reflexivity. lookup_trimmed.
  destruct (lookup s (static_values statics)) as [[] | ]; reflexivity.
Qed.

Lemma boxed_number_of_drop (statics : static_env) (v : value) :
  boxed_number_of (trim_statics statics) v = boxed_number_of statics v.
Proof.
  destruct v; simpl; try reflexivity. lookup_trimmed.
  destruct (lookup s (static_values statics)) as [[] | ]; reflexivity.
Qed.

Lemma string_of_drop (statics : static_env) (v : value) :
  string_of (trim_statics statics) v = string_of statics v.
Proof.
  destruct v; simpl; try reflexivity. lookup_trimmed.
  destruct (lookup s (static_values statics)) as [[] | ]; reflexivity.
Qed.

Lemma array_of_drop (statics : static_env) (v : value) :
  array_of (trim_statics statics) v = array_of statics v.
Proof.
  destruct v; simpl; try reflexivity. lookup_trimmed.
  destruct (lookup s (static_values statics)) as [[] | ]; reflexivity.
Qed.

Lemma bound_if_symbol_drop (statics : static_env) (v : value) :
  bound_if_symbol (trim_statics statics) v = bound_if_symbol statics v.
Proof.
  destruct v; simpl; try reflexivity. lookup_trimmed.
  destruct (lookup s (static_values statics)); reflexivity.
Qed.

Lemma imported_of_drop (statics : static_env) (v : value) :
  imported_of (trim_statics statics) v = imported_of statics v.
Proof.
  destruct v; simpl; try reflexivity. lookup_trimmed.
  destruct (lookup s (static_values statics)) as [[] | ]; reflexivity.
Qed.

Lemma project_function_slot_drop (statics : static_env)
    (move_from move_to : function_slot) (v : value) :
  project_function_slot (trim_statics statics) move_from move_to v
  = project_function_slot statics move_from move_to v.
Proof.
  destruct v; simpl; try reflexivity. lookup_trimmed.
  destruct (lookup s (static_values statics)) as [[] | ]; reflexivity.
Qed.

(* A closure is a closure of the same code, at the same slot of the same
   set, and its slots that are kept have the values they had. *)
Definition same_kept_slots (slots slots' : list (value_slot * value))
    : Prop :=
  ∀ slot : value_slot,
    among slot D = false -> lookup slot slots' = lookup slot slots.

Lemma closure_of_drop (statics : static_env) (v : value) (code : code_id)
    (slots : list (value_slot * value)) :
  closure_of statics v = Some (code, slots) ->
  ∃ slots' : list (value_slot * value),
    closure_of (trim_statics statics) v = Some (code, slots')
    ∧ same_kept_slots slots slots'.
Proof.
  destruct v
    as [ c | vcode vslots | vslot vdecls vslots | s | tag fields | elements | c
       | loc | callee args args_arity ];
    simpl; try discriminate.
  - intros H. exists slots. split; [ exact H | intros kept _; reflexivity ].
  - intros H. exists slots. split; [ exact H | intros kept _; reflexivity ].
  - lookup_trimmed.
    destruct (lookup s (static_values statics))
      as [[ acode aslots | aslot asyms adecls aslots | | | | | ] | ];
      simpl; try discriminate.
    + intros [= <- <-]. eexists. split; [ reflexivity | ].
      intros kept Hkept. apply lookup_keep_slots, Hkept.
    + destruct (lookup aslot adecls); simpl; [ | discriminate ].
      intros [= <- <-]. eexists. split; [ reflexivity | ].
      intros kept Hkept. apply lookup_keep_slots, Hkept.
Qed.

Lemma closure_in_set_drop (statics : static_env) (v : value)
    (at_slot : function_slot) (decls : list (function_slot * code_id))
    (slots : list (value_slot * value)) :
  closure_in_set statics v = Some (at_slot, decls, slots) ->
  ∃ slots' : list (value_slot * value),
    closure_in_set (trim_statics statics) v = Some (at_slot, decls, slots')
    ∧ same_kept_slots slots slots'.
Proof.
  destruct v
    as [ c | vcode vslots | vslot vdecls vslots | s | tag fields | elements | c
       | loc | callee args args_arity ];
    simpl; try discriminate.
  - intros H. exists slots. split; [ exact H | intros kept _; reflexivity ].
  - intros H. exists slots. split; [ exact H | intros kept _; reflexivity ].
  - lookup_trimmed.
    destruct (lookup s (static_values statics))
      as [[ acode aslots | aslot asyms adecls aslots | | | | | ] | ];
      simpl; try discriminate.
    + intros [= <- <- <-]. eexists. split; [ reflexivity | ].
      intros kept Hkept. apply lookup_keep_slots, Hkept.
    + destruct (lookup aslot adecls); simpl; [ | discriminate ].
      intros [= <- <- <-]. eexists. split; [ reflexivity | ].
      intros kept Hkept. apply lookup_keep_slots, Hkept.
Qed.

(* No observation that a primitive makes tells the two static
   environments apart, so no primitive does. *)
Lemma drop_statics_respects (statics : static_env) :
  respects_observations eq statics (trim_statics statics).
Proof.
  split; [ | split ].
  - split; intros; subst; reflexivity.
  - split; [ intros c; split; reflexivity | split ].
    + intros tag fields fields' Hfields.
      rewrite (forall2_eq fields fields' Hfields). reflexivity.
    + intros elements elements' Helements.
      rewrite (forall2_eq elements elements' Helements). reflexivity.
  - intros v v' <-.
    split;
      [ | split; [ | split; [ | split; [ | split; [ | split; [ | split ] ] ] ] ]
      ].
    + intros tag fields H. exists fields. rewrite block_of_drop.
      split; [ exact H | apply forall2_eq_refl ].
    + intros c. rewrite boxed_number_of_drop. exact (fun H => H).
    + intros bytes. rewrite string_of_drop. exact (fun H => H).
    + destruct (closure_of statics v) as [[code slots] | ] eqn:Hc;
        [ | congruence ].
      destruct (closure_of_drop statics v code slots Hc) as (slots' & -> & _).
      intros _. discriminate.
    + exact (fun H => H).
    + rewrite bound_if_symbol_drop. exact (fun H => H).
    + intros s. rewrite imported_of_drop. exact (fun H => H).
    + intros elements H. exists elements. rewrite array_of_drop.
      split; [ exact H | apply forall2_eq_refl ].
Qed.

Lemma eval_prim_drop (statics : static_env) (p : prim) (vs : list value)
    (v : value) :
  eval_prim statics p vs = Some v ->
  eval_prim (trim_statics statics) p vs = Some v.
Proof.
  intros Hp.
  destruct
    (eval_prim_related eq statics (trim_statics statics) p vs vs v
       (drop_statics_respects statics) (forall2_eq_refl vs) Hp)
    as (v' & Hp' & <-).
  exact Hp'.
Qed.

(* ======================================================================== *)
(* Evaluation                                                               *)
(* ======================================================================== *)

(* A defining expression that projects no slot removed has the value it
   had. *)
Lemma eval_named_drop (statics : static_env) (vars : env) (d : named)
    (v : value) :
  named_projects_none D d = true ->
  eval_named statics vars d = Some v ->
  eval_named (trim_statics statics) vars d = Some v.
Proof.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure]; simpl;
    intros Hd; try exact (fun H => H).
  - destruct (eval_simples vars args) as [vs | ]; simpl; [ | discriminate ].
    apply eval_prim_drop.
  - destruct (eval_simple vars closure) as [w | ]; simpl; [ | discriminate ].
    destruct (closure_of statics w) as [[called cslots] | ] eqn:Hc; simpl;
      [ | discriminate ].
    destruct (closure_of_drop statics w called cslots Hc)
      as (cslots' & -> & Hsame).
    simpl. rewrite (Hsame slot (proj1 (negb_true_iff _) Hd)).
    exact (fun H => H).
  - destruct (eval_simple vars closure) as [w | ]; simpl; [ | discriminate ].
    unfold project_value_slot.
    destruct (closure_in_set statics w) as [[[at_slot decls] cslots] | ]
      eqn:Hc; simpl; [ | discriminate ].
    destruct (closure_in_set_drop statics w at_slot decls cslots Hc)
      as (cslots' & -> & Hsame).
    simpl. destruct (Pos.eqb at_slot pfrom); [ | discriminate ].
    rewrite (Hsame slot (proj1 (negb_true_iff _) Hd)). exact (fun H => H).
  - destruct (eval_simple vars closure) as [w | ]; simpl; [ | discriminate ].
    rewrite project_function_slot_drop. exact (fun H => H).
Qed.

(* The same of the run of the machine, which runs a [Let] with the
   heap: a primitive does on the heap what it did
   (FlambdaHeapSpec.exec_prim_related_statement at equality, as
   [eval_prim_drop] is of the law of [eval_prim]), and nothing else
   looks at the heap. *)
Lemma exec_prim_drop (statics : static_env) (hp hp' : heap) (p : prim)
    (vs : list value) (v : value) :
  exec_prim statics hp p vs = Some (v, hp') ->
  exec_prim (trim_statics statics) hp p vs = Some (v, hp').
Proof.
  intros Hp.
  assert (Hlocations : FlambdaHeapSpec.relates_locations eq).
  { split; [ intros l; reflexivity | split ].
    - intros l v' <-. reflexivity.
    - intros w l H. exact H. }
  destruct
    (FlambdaHeapProofs.exec_prim_related eq statics (trim_statics statics) hp
       hp hp' p vs vs v (drop_statics_respects statics) Hlocations
       (proj2 (FlambdaHeapProofs.heap_alike_eq hp hp) eq_refl)
       (forall2_eq_refl vs) Hp)
    as (v' & hp1 & Hp' & <- & Halike).
  apply FlambdaHeapProofs.heap_alike_eq in Halike. subst hp1. exact Hp'.
Qed.

Lemma exec_named_drop (statics : static_env) (hp hp' : heap)
    (rd rd' : read_record) (vars : env) (d : named) (v : value) :
  named_projects_none D d = true ->
  exec_named statics hp rd vars d = Some (v, hp', rd') ->
  exec_named (trim_statics statics) hp rd vars d = Some (v, hp', rd').
Proof.
  intros Hd Hexec.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure].
  2: { apply exec_named_prim_some in Hexec as (vs & Hvs & Hp & ->).
       apply exec_named_prim_some. exists vs.
       split; [ exact Hvs | ].
       split; [ exact (exec_prim_drop statics hp hp' p vs v Hp) | ].
       reflexivity. }
  all:
    rewrite exec_named_not_prim in Hexec |- * by (intros ? ?; discriminate);
    destruct (eval_named statics vars _) as [w | ] eqn:Hw;
      [ | discriminate ];
    injection Hexec as <- <- <-;
    rewrite (eval_named_drop statics vars _ w Hd Hw); reflexivity.
Qed.

Lemma eval_value_slots_keep (vars : env)
    (slots : list (value_slot * simple)) :
  ∀ vs : list (value_slot * value),
    eval_value_slots vars slots = Some vs ->
    eval_value_slots vars (keep_slots D slots) = Some (keep_slots D vs).
Proof.
  unfold keep_slots.
  induction slots as [ | [slot s] slots IH]; intros vs; simpl.
  - intros [= <-]. reflexivity.
  - destruct (eval_simple vars s) as [v | ] eqn:Hs; simpl; [ | discriminate ].
    destruct (eval_value_slots vars slots) as [rest | ]; simpl;
      [ | discriminate ].
    intros [= <-]. simpl.
    destruct (among slot D); simpl;
      [ exact (IH rest eq_refl) | ].
    rewrite Hs, (IH rest eq_refl). reflexivity.
Qed.

Lemma eval_static_const_drop (vars : env) (const : static_const)
    (allocated : statically_allocated_value) :
  eval_static_const vars const = Some allocated ->
  eval_static_const vars (drop_slots_static_const D const)
  = Some (trim_allocated allocated).
Proof.
  destruct const
    as [code slots | fslot syms decls slots | tag fields | contents | bytes
        | elements];
    simpl.
  - destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots; simpl;
      [ | discriminate ].
    intros [= <-]. rewrite (eval_value_slots_keep vars slots vs Hslots).
    reflexivity.
  - destruct (lookup fslot decls); simpl; [ | discriminate ].
    destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots; simpl;
      [ | discriminate ].
    intros [= <-]. rewrite (eval_value_slots_keep vars slots vs Hslots).
    reflexivity.
  - destruct (eval_simples vars fields); simpl; [ | discriminate ].
    intros [= <-]. reflexivity.
  - destruct (eval_simple vars contents); simpl; [ | discriminate ].
    intros [= <-]. reflexivity.
  - intros [= <-]. reflexivity.
  - destruct (eval_simples vars elements); simpl; [ | discriminate ].
    intros [= <-]. reflexivity.
Qed.

Lemma eval_static_bindings_drop (vars : env)
    (bindings : list static_binding) :
  ∀ statics statics' : static_env,
    eval_static_bindings vars bindings statics = Some statics' ->
    eval_static_bindings vars (map (drop_slots_binding D) bindings)
      (trim_statics statics)
    = Some (trim_statics statics').
Proof.
  induction bindings as [ | [id c | id | sym const] bindings IH];
    intros statics statics'; simpl.
  - intros [= <-]. reflexivity.
  - lookup_trimmed.
    destruct (lookup id (static_code statics)); simpl; [ discriminate | ].
    intros H. exact (IH _ _ H).
  - apply IH.
  - lookup_trimmed.
    destruct (lookup sym (static_values statics)); simpl; [ discriminate | ].
    destruct (eval_static_const vars const) as [allocated | ] eqn:Hconst;
      simpl; [ | discriminate ].
    rewrite (eval_static_const_drop vars const allocated Hconst). simpl.
    intros H. exact (IH _ _ H).
Qed.

(* ======================================================================== *)
(* Terms and continuations                                                  *)
(* ======================================================================== *)

(* The rewrite keeps whether a term binds something static. *)
Fixpoint drop_slots_binds_statics (e : expr) {struct e} :
  binds_statics (drop_slots D e) = binds_statics e
with drop_slots_handler_binds_statics (h : continuation_handler)
    {struct h} :
  handler_binds_statics (drop_slots_handler D h) = handler_binds_statics h.
Proof.
  - destruct e
      as [ x xkind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try reflexivity.
    + apply drop_slots_binds_statics.
    + rewrite drop_slots_binds_statics. f_equal.
      induction bindings as [ | binding bindings IH]; simpl; [ reflexivity | ].
      rewrite IH. destruct binding; reflexivity.
    + rewrite drop_slots_handler_binds_statics, drop_slots_binds_statics.
      reflexivity.
    + rewrite drop_slots_binds_statics. f_equal.
      induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
      rewrite drop_slots_handler_binds_statics, IH. reflexivity.
  - destruct h as [params metadata handler]. simpl.
    apply drop_slots_binds_statics.
Qed.

Lemma drop_slots_handlers_bind_statics
    (handlers : list (continuation * continuation_handler)) :
  handlers_bind_statics
    (map (fun '(k, h) => (k, drop_slots_handler D h)) handlers)
  = handlers_bind_statics handlers.
Proof.
  unfold handlers_bind_statics.
  induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
  rewrite drop_slots_handler_binds_statics, IH. reflexivity.
Qed.

Lemma resolve_drop (k : continuation) (conts : cont_env) :
  resolve k (trim_conts conts) = trim_cont (resolve k conts).
Proof.
  unfold resolve, trim_conts. rewrite lookup_map_snd.
  destruct (lookup k conts); reflexivity.
Qed.

Lemma rec_closures_drop
    (handlers : list (continuation * continuation_handler)) (vars : env)
    (conts : cont_env) :
  rec_closures (map (fun '(k, h) => (k, drop_slots_handler D h)) handlers)
    vars (trim_conts conts)
  = trim_conts (rec_closures handlers vars conts).
Proof.
  unfold rec_closures, trim_conts. rewrite !map_map.
  apply map_ext. intros [k h]. reflexivity.
Qed.

Lemma with_extra_args_drop (c : cont_value) (extra : list value) :
  trim_cont (with_extra_args c extra) = with_extra_args (trim_cont c) extra.
Proof. destruct extra; reflexivity. Qed.

Lemma drop_slots_code_parts (c : code) :
  code_params (drop_slots_code D c) = code_params c
  ∧ code_my_closure (drop_slots_code D c) = code_my_closure c
  ∧ code_return_continuation (drop_slots_code D c)
    = code_return_continuation c
  ∧ code_exn_continuation (drop_slots_code D c) = code_exn_continuation c
  ∧ code_params_arity (drop_slots_code D c) = code_params_arity c
  ∧ code_result_arity (drop_slots_code D c) = code_result_arity c
  ∧ code_body (drop_slots_code D c) = drop_slots D (code_body c).
Proof. destruct c. simpl. auto 8. Qed.

(* What the second run does where the first does something: the same
   step, to the second state of the result, or the same exit. *)
Definition result_trimmed (result result' : step_result) : Prop :=
  match result with
  | Step t => result' = Step (trim_state t)
  | Exit k vs hp rd => result' = Exit k vs hp rd
  | Undefined => True
  end.

(* A call of the second run has the shape of the call of the first,
   with the same values, of the code rewritten. *)
Definition trim_shape (shape : call_shape) : call_shape :=
  match shape with
  | Full_call closure c vs args_arity =>
    Full_call closure (drop_slots_code D c) vs args_arity
  | Over_call closure c vs remaining args_arity remaining_arity =>
    Over_call closure (drop_slots_code D c) vs remaining args_arity
      remaining_arity
  | Partial_call partial => Partial_call partial
  end.

Lemma shape_of_closure_call_drop (statics : static_env) (closure : value)
    (kind : call_kind) (vs : list value) (args_arity return_arity : arity)
    (shape : call_shape) :
  shape_of_closure_call statics closure kind vs args_arity return_arity
  = Some shape ->
  shape_of_closure_call (trim_statics statics) closure kind vs args_arity
    return_arity
  = Some (trim_shape shape).
Proof.
  unfold shape_of_closure_call.
  destruct (closure_of statics closure) as [[called slots] | ] eqn:Hc;
    [ | discriminate ].
  destruct (closure_of_drop statics closure called slots Hc)
    as (slots' & -> & _).
  destruct
    (match kind with
     | Direct named => Pos.eqb called named
     | Indirect => true
     end);
    [ | discriminate ].
  lookup_trimmed.
  destruct (lookup called (static_code statics)) as [c | ]; [ | discriminate ].
  simpl option_map. cbv iota zeta.
  destruct (drop_slots_code_parts c)
    as (_ & _ & _ & _ & Hparams_arity & Hresult_arity & _).
  assert (Htupled : code_is_tupled (drop_slots_code D c) = code_is_tupled c)
    by (destruct c; reflexivity).
  destruct (arguments_for_code statics c vs args_arity)
    as [[vs' args_arity'] | ] eqn:Hargs; [ | discriminate ].
  rewrite (arguments_for_code_mono statics (trim_statics statics) c
             (drop_slots_code D c) vs args_arity _ Htupled Hparams_arity
             (fun tuple rest p v _ => eval_prim_drop statics p [tuple] v)
             Hargs).
  simpl. unfold shape_of_code_call. cbv zeta.
  rewrite Hparams_arity, Hresult_arity, Htupled.
  destruct (Nat.eqb (length args_arity') (length (code_params_arity c)));
    [ intros [= <-]; reflexivity | ].
  destruct
    (negb
       (kinds_eqb (map kind_of_value vs') (unarize args_arity')
        && partial_or_over_application args_arity' return_arity
             (code_params_arity c) (code_result_arity c)));
    [ discriminate | ].
  destruct (Nat.ltb (length (code_params_arity c)) (length args_arity'));
    [ intros [= <-]; reflexivity | ].
  destruct (code_is_tupled c); [ discriminate | ].
  intros [= <-]. reflexivity.
Qed.

Lemma shape_of_call_drop (statics : static_env) (callee : value)
    (kind : call_kind) (vs : list value) (args_arity return_arity : arity)
    (shape : call_shape) :
  shape_of_call statics callee kind vs args_arity return_arity = Some shape ->
  shape_of_call (trim_statics statics) callee kind vs args_arity return_arity
  = Some (trim_shape shape).
Proof.
  unfold shape_of_call.
  destruct kind as [named | ]; [ apply shape_of_closure_call_drop | ].
  destruct (partial_application_of callee)
    as [[[closure applied] applied_arity] | ];
    [ | apply shape_of_closure_call_drop ].
  destruct args_arity as [ | component args_arity]; [ discriminate | ].
  destruct (kinds_eqb _ _); [ | discriminate ].
  apply shape_of_closure_call_drop.
Qed.

(* A full call. *)
Lemma call_code_drop (closure : value) (c : code) (vs : list value)
    (args_arity return_arity : arity) (return_to exn_to : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record) :
  result_trimmed
    (call_code closure c vs args_arity return_arity return_to exn_to statics hp
       rd)
    (call_code closure (drop_slots_code D c) vs args_arity return_arity
       (trim_cont return_to) (trim_cont exn_to) (trim_statics statics) hp rd).
Proof.
  unfold result_trimmed, call_code.
  destruct (drop_slots_code_parts c)
    as (Hparams & Hmy_closure & Hret & Hexn & Hparams_arity & Hresult_arity
        & Hbody).
  rewrite Hparams, Hmy_closure, Hret, Hexn, Hbody, drop_slots_binds_statics.
  destruct (binds_statics (code_body c)); [ exact I | ].
  rewrite Hparams_arity, Hresult_arity.
  match goal with
  | |- context [negb ?agrees] => destruct (negb agrees)
  end;
    [ exact I | ].
  destruct (bind_params (code_params c) vs [(code_my_closure c, closure)]);
    [ | exact I ].
  reflexivity.
Qed.

(* A call of a given shape, given what the two runs do with a partial
   application. *)
Lemma run_call_drop (return_partial return_partial' : value -> step_result)
    (shape : call_shape) (return_arity : arity)
    (return_to exn_to : cont_value) (statics : static_env) (hp : heap)
      (rd : read_record) :
  (∀ partial : value,
    result_trimmed (return_partial partial) (return_partial' partial)) ->
  result_trimmed
    (run_call return_partial (Some shape) return_arity return_to exn_to
       statics hp rd)
    (run_call return_partial' (Some (trim_shape shape)) return_arity
       (trim_cont return_to) (trim_cont exn_to) (trim_statics statics) hp rd).
Proof.
  intros Hpartial.
  destruct shape
    as [closure c vs args_arity
       | closure c vs remaining args_arity remaining_arity | partial];
    simpl.
  - apply call_code_drop.
  - destruct (drop_slots_code_parts c) as (_ & _ & _ & _ & _ & -> & _).
    exact (call_code_drop closure c vs args_arity (code_result_arity c)
             (Over_application remaining remaining_arity return_arity
                return_to exn_to)
             exn_to statics hp rd).
  - apply Hpartial.
Qed.

(* A call, given what the two runs do when the return continuation is
   applied to a partial application. *)
Lemma call_function_drop_from (callee : value) (kind : call_kind)
    (vs : list value) (args_arity return_arity : arity)
    (return_to exn_to : cont_value) (statics : static_env) (hp : heap)
      (rd : read_record) :
  (∀ partial : value,
    result_trimmed (apply_cont_value return_to [partial] statics hp rd)
      (apply_cont_value (trim_cont return_to) [partial]
         (trim_statics statics) hp rd)) ->
  result_trimmed
    (call_function callee kind vs args_arity return_arity return_to exn_to
       statics hp rd)
    (call_function callee kind vs args_arity return_arity
       (trim_cont return_to) (trim_cont exn_to) (trim_statics statics) hp rd).
Proof.
  intros Hpartial. unfold call_function.
  destruct (shape_of_call statics callee kind vs args_arity return_arity)
    as [shape | ] eqn:Hshape;
    [ | exact I ].
  rewrite (shape_of_call_drop statics callee kind vs args_arity return_arity
             shape Hshape).
  apply run_call_drop. exact Hpartial.
Qed.

Lemma apply_cont_value_drop (c : cont_value) :
  ∀ (vs : list value) (statics : static_env) (hp : heap) (rd : read_record),
    result_trimmed (apply_cont_value c vs statics hp rd)
      (apply_cont_value (trim_cont c) vs (trim_statics statics) hp rd).
Proof.
  induction c
    as [[params metadata handler] def_vars def_conts recursive | k
       | c IH extra
       | remaining remaining_arity return_arity return_to exn_to IHreturn _]
    using cont_value_over_ind;
    intros vs statics hp rd.
  - unfold result_trimmed. simpl.
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
      [ | exact I ].
    destruct (bind_params params vs def_vars) as [handler_vars | ];
      [ | exact I ].
    unfold trim_state. simpl_state. f_equal. f_equal.
    fold (trim_conts def_conts). rewrite rec_closures_drop.
    unfold trim_conts. rewrite map_app. reflexivity.
  - reflexivity.
  - apply IH.
  - simpl trim_cont. rewrite !apply_cont_value_over_application.
    destruct vs as [ | v [ | v' vs]]; [ exact I | | exact I ].
    apply call_function_drop_from. intros partial. apply IHreturn.
Qed.

Lemma call_function_drop (callee : value) (kind : call_kind)
    (vs : list value) (args_arity return_arity : arity)
    (return_to exn_to : cont_value) (statics : static_env) (hp : heap)
      (rd : read_record) :
  result_trimmed
    (call_function callee kind vs args_arity return_arity return_to exn_to
       statics hp rd)
    (call_function callee kind vs args_arity return_arity
       (trim_cont return_to) (trim_cont exn_to) (trim_statics statics) hp rd).
Proof.
  apply call_function_drop_from. intros partial. apply apply_cont_value_drop.
Qed.

(* ======================================================================== *)
(* A step                                                                   *)
(* ======================================================================== *)

(* A step of a state whose expression projects no slot removed is a
   step of the second state, to the second state of the result. *)
Lemma drop_step_components (e : expr) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) :
  slots_unused D e = true ->
  match step (State e vars conts statics hp rd) with
  | Step t =>
    step
      (State (drop_slots D e) vars (trim_conts conts) (trim_statics statics) hp
         rd)
    = Step (trim_state t)
  | Exit k vs hp' rd' =>
    step
      (State (drop_slots D e) vars (trim_conts conts) (trim_statics statics) hp
         rd)
    = Exit k vs hp' rd'
  | Undefined => True
  end.
Proof.
  intros Hunused.
  destruct e
    as [ x xkind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    simpl drop_slots; autorewrite with step.
  - (* Let *)
    simpl in Hunused. apply andb_true_iff in Hunused as [Hd _].
    destruct (exec_named statics hp rd vars d) as [[[v hp'] rd'] | ] eqn:Hv;
      [ | exact I ].
    destruct (kind_eqb (kind_of_value v) xkind) eqn:Hkind; [ | exact I ].
    rewrite (exec_named_drop statics hp hp' rd rd' vars d v Hd Hv), Hkind.
    reflexivity.
  - (* Let_static *)
    destruct (eval_static_bindings vars bindings statics) as [statics' | ]
      eqn:Heval; [ | exact I ].
    rewrite (eval_static_bindings_drop vars bindings statics statics' Heval).
    reflexivity.
  - (* Let_cont_non_recursive *)
    reflexivity.
  - (* Let_cont_recursive *)
    rewrite drop_slots_handlers_bind_statics.
    destruct (handlers_bind_statics handlers); [ exact I | ].
    unfold trim_state. simpl_state. rewrite rec_closures_drop.
    unfold trim_conts. rewrite map_app. reflexivity.
  - (* Apply *)
    destruct (eval_simple vars (ap_callee ap)) as [callee | ]; [ | exact I ].
    destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | exact I ].
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
      [ | exact I ].
    rewrite !resolve_drop, <- with_extra_args_drop.
    exact
      (call_function_drop callee (ap_call_kind ap) vs (ap_args_arity ap)
         (ap_return_arity ap) (resolve (ap_continuation ap) conts)
         (with_extra_args (resolve (ap_exn_continuation ap) conts) extra)
         statics hp rd).
  - (* Apply_cont *)
    destruct (eval_simples vars (ac_args ac)) as [vs | ]; [ | exact I ].
    unfold apply_continuation. rewrite resolve_drop.
    exact
      (apply_cont_value_drop (resolve (ac_continuation ac) conts) vs statics
         hp rd).
  - (* Switch *)
    destruct (eval_simple vars scrutinee) as [v | ]; [ | exact I ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | exact I ].
    destruct (find_arm i arms) as [ac | ]; [ | exact I ].
    reflexivity.
  - exact I.
Qed.

Lemma drop_step (s : state) :
  slots_unused D (state_expr s) = true ->
  match step s with
  | Step t => step (trim_state s) = Step (trim_state t)
  | Exit k vs hp rd => step (trim_state s) = Exit k vs hp rd
  | Undefined => True
  end.
Proof.
  intros Hunused.
  pose proof
    (drop_step_components (state_expr s) (state_vars s) (state_conts s)
       (state_statics s) (state_heap s) (state_reads s) Hunused)
    as Hstep.
  rewrite <- (state_eta s) in Hstep. exact Hstep.
Qed.

(* ======================================================================== *)
(* The run of a unit                                                        *)
(* ======================================================================== *)

(* The terms of a state project no slot removed. *)
Definition slots_unused_inv : state -> Prop :=
  terms_inv (slots_unused D) (handler_slots_unused D) (code_slots_unused D).

Lemma slots_unused_inv_step (s s' : state) :
  slots_unused_inv s -> step s = Step s' -> slots_unused_inv s'.
Proof.
  apply terms_inv_step.
  - intros x xkind d body H. simpl in H. apply andb_true_iff in H. tauto.
  - intros bindings body H. simpl in H.
    apply andb_true_iff in H as [Hbindings Hbody].
    split; [ exact Hbody | ]. intros id c Hin.
    exact (proj1 (forallb_forall _ _) Hbindings (Bind_code id c) Hin).
  - intros k h body H. simpl in H. apply andb_true_iff in H. tauto.
  - intros handlers body H. simpl in H. apply andb_true_iff in H. tauto.
  - intros ac. reflexivity.
  - intros params metadata handler H. exact H.
  - intros [params my_closure ret exn metadata body] H. exact H.
Qed.

Definition dropped_related (s s' : state) : Prop :=
  s' = trim_state s ∧ slots_unused_inv s.

Lemma dropped_related_step (s s' : state) :
  dropped_related s s' -> result_simulated dropped_related (step s) (step s').
Proof.
  intros [-> Hinv].
  pose proof (drop_step s (proj1 Hinv)) as Hstep.
  pose proof (slots_unused_inv_step s) as Hkept.
  destruct (step s) as [t | k vs | ]; simpl; [ | | exact I ].
  - rewrite Hstep. simpl. split; [ reflexivity | ].
    exact (Hkept t Hinv eq_refl).
  - rewrite Hstep. simpl. auto.
Qed.

End Dropped.

(* What [drop_unused_slots] gives refines the unit, the two run from a
   static environment that holds no code and binds no closure with a
   slot removed, and from any heap. *)
Lemma drop_unused_slots_refines_from (dropped : list value_slot)
    (e e' : expr) (statics : static_env) (hp : heap) (rd : read_record) :
  drop_unused_slots dropped e = Some e' ->
  static_code statics = [] ->
  trim_statics dropped statics = statics ->
  refines (State e [] [] statics hp rd) (State e' [] [] statics hp rd).
Proof.
  unfold drop_unused_slots.
  destruct (slots_unused dropped e) eqn:Hunused; [ | discriminate ].
  intros [= <-] Hno_code Htrim.
  apply (simulation_refines (dropped_related dropped)
           (dropped_related_step dropped)).
  split.
  - unfold trim_state. simpl_state. rewrite Htrim. reflexivity.
  - unfold slots_unused_inv, terms_inv. simpl_state.
    refine (conj Hunused (conj I _)).
    intros id c Hlookup. rewrite Hno_code in Hlookup. discriminate Hlookup.
Qed.

(* The static environment of the imports of a unit is such: an import
   is no closure. *)
Lemma trim_statics_imports (dropped : list value_slot)
    (imports : list (symbol * import_class)) :
  trim_statics dropped (imports_statics imports) = imports_statics imports.
Proof.
  unfold trim_statics, imports_statics. cbn [static_code static_values map].
  f_equal.
  induction imports as [ | [s c] imports IH];
    cbn [map fst snd trim_allocated]; [ reflexivity | ].
  rewrite IH. reflexivity.
Qed.

Theorem drop_unused_slots_program_refines :
  drop_unused_slots_program_refines_statement.
Proof.
  intros dropped e e' Hdrop. unfold program_refines.
  exact (drop_unused_slots_refines_from dropped e e' empty_static_env
           empty_heap no_reads Hdrop eq_refl eq_refl).
Qed.

Theorem drop_unused_slots_refines_imports :
  drop_unused_slots_refines_imports_statement.
Proof.
  intros imports dropped e e' Hdrop. unfold unit_refines.
  exact (drop_unused_slots_refines_from dropped e e' (imports_statics imports)
           empty_heap no_reads Hdrop eq_refl
             (trim_statics_imports dropped imports)).
Qed.

(* ======================================================================== *)
(* The rewrite and the check                                                *)
(* ======================================================================== *)

Fixpoint drop_slots_slots_unused_expr (dropped dropped' : list value_slot)
    (e : expr) {struct e} :
  slots_unused dropped' (drop_slots dropped e) = slots_unused dropped' e
with drop_slots_slots_unused_handler (dropped dropped' : list value_slot)
    (h : continuation_handler) {struct h} :
  handler_slots_unused dropped' (drop_slots_handler dropped h)
  = handler_slots_unused dropped' h
with drop_slots_slots_unused_binding (dropped dropped' : list value_slot)
    (binding : static_binding) {struct binding} :
  binding_slots_unused dropped' (drop_slots_binding dropped binding)
  = binding_slots_unused dropped' binding
with drop_slots_slots_unused_code (dropped dropped' : list value_slot)
    (c : code) {struct c} :
  code_slots_unused dropped' (drop_slots_code dropped c)
  = code_slots_unused dropped' c.
Proof.
  - destruct e
      as [ x xkind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try reflexivity.
    + rewrite drop_slots_slots_unused_expr. reflexivity.
    + rewrite drop_slots_slots_unused_expr. f_equal.
      induction bindings as [ | binding bindings IH]; simpl; [ reflexivity | ].
      rewrite drop_slots_slots_unused_binding, IH. reflexivity.
    + rewrite drop_slots_slots_unused_handler, drop_slots_slots_unused_expr.
      reflexivity.
    + rewrite drop_slots_slots_unused_expr. f_equal.
      induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
      rewrite drop_slots_slots_unused_handler, IH. reflexivity.
  - destruct h as [params metadata handler]. simpl.
    apply drop_slots_slots_unused_expr.
  - destruct binding as [id c | id | sym const]; simpl; try reflexivity.
    apply drop_slots_slots_unused_code.
  - destruct c as [params my_closure ret exn metadata body]. simpl.
    apply drop_slots_slots_unused_expr.
Qed.

Theorem drop_slots_slots_unused : drop_slots_slots_unused_statement.
Proof. exact drop_slots_slots_unused_expr. Qed.

(* Renaming renames no value slot: the rewrite and the check commute with
   it. Proved from the definitions by [equivariance]
   (FlambdaEquivarianceTactic.v). *)
Theorem drop_slots_equivariant : drop_slots_equivariant_statement.
Proof. equivariance. Qed.

Theorem slots_unused_equivariant : slots_unused_equivariant_statement.
Proof. equivariance. Qed.
