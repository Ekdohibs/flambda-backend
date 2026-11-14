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

(* Proofs of the statements of FlambdaHeapEquivarianceSpec.v. This file
   needs no review.

   The machine with a heap is that of FlambdaSemanticsSpec.v with the
   heap carried along, and the action leaves the heap alone, so each
   case of a step is the case of FlambdaEquivarianceProofs.v. The one
   new thing is a primitive, which reads the static environment for its
   statically allocated values only ([exec_prim_renamed]). The relation
   on outcomes reads the same and no more ([value_related_code]). *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaNominalSpec FlambdaNominalProofs FlambdaAlphaSpec FlambdaAlphaProofs
  FlambdaNameAbstractionSpec FlambdaEquivarianceSpec FlambdaEquivarianceProofs
  FlambdaStateSupportSpec FlambdaStateSupportProofs FlambdaHeapSpec
  FlambdaHeapMachineSpec FlambdaHeapProofs FlambdaHeapMachineProofs
  FlambdaHeapEquivarianceSpec.
Import ListNotations.

(* ======================================================================== *)
(* The action is that of a group                                            *)
(* ======================================================================== *)

(* A state with a heap is a state and a heap. *)
Definition with_heap (t : state) (h : heap) : heap_state :=
  Heap_state (state_expr t) (state_vars t) (state_conts t) (state_statics t)
    h.

Lemma rename_heap_state_with_heap (r : renaming) (s : heap_state) :
  rename_heap_state r s
  = with_heap (rename_state r (forget_heap s)) (hs_heap s).
Proof. reflexivity. Qed.

Lemma forget_with_heap (t : state) (h : heap) :
  state_heap t = h -> state_reads t = no_reads ->
  forget_heap (with_heap t h) = t.
Proof. intros <-. destruct t. simpl. intros ->. reflexivity. Qed.

Lemma hs_heap_with_heap (t : state) (h : heap) :
  hs_heap (with_heap t h) = h.
Proof. reflexivity. Qed.

Lemma with_heap_forget (s : heap_state) :
  with_heap (forget_heap s) (hs_heap s) = s.
Proof. destruct s. reflexivity. Qed.

Theorem forget_heap_equivariant : forget_heap_equivariant_statement.
Proof. intros r s. reflexivity. Qed.

Theorem rename_heap_state_identity : rename_heap_state_identity_statement.
Proof.
  intros s. rewrite rename_heap_state_with_heap, rename_state_identity.
  apply with_heap_forget.
Qed.

Theorem rename_heap_state_compose : rename_heap_state_compose_statement.
Proof.
  intros second first s. rewrite !rename_heap_state_with_heap.
  rewrite forget_with_heap by reflexivity.
  rewrite hs_heap_with_heap, rename_state_compose. reflexivity.
Qed.

Theorem rename_heap_state_inverse : rename_heap_state_inverse_statement.
Proof.
  intros r s.
  destruct (rename_state_inverse r (forget_heap s)) as [Hleft Hright].
  split; rewrite !rename_heap_state_with_heap;
    rewrite forget_with_heap by reflexivity; rewrite hs_heap_with_heap.
  - rewrite Hleft. apply with_heap_forget.
  - rewrite Hright. apply with_heap_forget.
Qed.

Theorem heap_state_finitely_supported :
  heap_state_finitely_supported_statement.
Proof.
  intros s.
  destruct (state_finitely_supported (forget_heap s)) as (vs & ks & Hfixed).
  exists vs, ks. intros r Hvs Hks.
  rewrite rename_heap_state_with_heap, (Hfixed r Hvs Hks).
  apply with_heap_forget.
Qed.

(* ======================================================================== *)
(* What a step is made of commutes with the action                          *)
(* ======================================================================== *)

(* A primitive reads the static environment for its statically allocated
   values, which the action leaves alone. *)
Lemma exec_prim_renamed (r : renaming) (statics : static_env) (h : heap)
    (p : prim) (vs : list value) :
  exec_prim (rename_static_env r statics) h p vs = exec_prim statics h p vs.
Proof. destruct statics as [codes values]. reflexivity. Qed.

Lemma heap_exec_named_equivariant (r : renaming) (statics : static_env)
    (h : heap) (vars : env) (d : named) :
  heap_exec_named (rename_static_env r statics) h (rename_env r vars)
    (rename_named r d)
  = heap_exec_named statics h vars d.
Proof.
  pose proof (eval_named_equivariant r statics vars d) as Heval.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    unfold heap_exec_named; cbn [rename_named] in *;
    try (rewrite Heval; reflexivity).
  rewrite eval_simples_equivariant.
  destruct (eval_simples vars args) as [vs | ]; [ | reflexivity ].
  apply exec_prim_renamed.
Qed.

(* A call of code. *)
Lemma heap_call_code_equivariant (r : renaming) (closure : value) (c : code)
    (vs : list value) (args_arity return_arity : arity)
    (return_to exn_to : cont_value) (statics : static_env) (h : heap) :
  heap_call_code closure (rename_code r c) vs args_arity return_arity
    (rename_cont_value r return_to) (rename_cont_value r exn_to)
    (rename_static_env r statics) h
  = rename_heap_result r
      (heap_call_code closure c vs args_arity return_arity return_to exn_to
         statics h).
Proof.
  destruct c as [params my_closure ret exn metadata body].
  unfold heap_call_code. simpl.
  rewrite rename_binds_statics.
  destruct (binds_statics body); [ reflexivity | ].
  destruct (negb _); [ reflexivity | ].
  change [(rename_variable r my_closure, closure)]
    with (rename_env r [(my_closure, closure)]).
  rewrite bind_params_equivariant.
  destruct (bind_params params vs [(my_closure, closure)]); reflexivity.
Qed.

(* A call by its shape: the shape of the renamed call is the shape with
   its code renamed (FlambdaEquivarianceProofs.shape_of_call_equivariant). *)
Lemma heap_run_call_equivariant (r : renaming)
    (return_partial return_partial' : value -> heap_step_result)
    (shape : option call_shape) (return_arity : arity)
    (return_to exn_to : cont_value) (statics : static_env) (h : heap) :
  (∀ partial : value,
    return_partial' partial
    = rename_heap_result r (return_partial partial)) ->
  heap_run_call return_partial' (option_map (rename_shape r) shape)
    return_arity (rename_cont_value r return_to)
    (rename_cont_value r exn_to) (rename_static_env r statics) h
  = rename_heap_result r
      (heap_run_call return_partial shape return_arity return_to exn_to
         statics h).
Proof.
  intros Hpartial.
  destruct shape
    as [[closure c vs args_arity
        | closure c vs remaining args_arity remaining_arity | partial] | ];
    simpl.
  - apply heap_call_code_equivariant.
  - replace (code_result_arity (rename_code r c)) with (code_result_arity c)
      by (destruct c; reflexivity).
    exact
      (heap_call_code_equivariant r closure c vs args_arity
         (code_result_arity c)
         (Over_application remaining remaining_arity return_arity return_to
            exn_to)
         exn_to statics h).
  - apply Hpartial.
  - reflexivity.
Qed.

Lemma heap_apply_cont_value_equivariant (r : renaming) (c : cont_value) :
  ∀ (vs : list value) (statics : static_env) (h : heap),
    heap_apply_cont_value (rename_cont_value r c) vs
      (rename_static_env r statics) h
    = rename_heap_result r (heap_apply_cont_value c vs statics h).
Proof.
  induction c
    as [[params metadata handler] def_vars def_conts recursive | k
       | c IH extra
       | remaining remaining_arity return_arity return_to exn_to IHreturn _]
    using cont_value_over_ind;
    intros vs statics h.
  - simpl.
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
      [ | reflexivity ].
    rewrite bind_params_equivariant.
    destruct (bind_params params vs def_vars) as [handler_vars | ]; simpl;
      [ | reflexivity ].
    unfold rename_heap_state. simpl.
    rewrite rename_cont_env_app, <- rec_closures_equivariant.
    reflexivity.
  - reflexivity.
  - simpl. apply IH.
  - destruct vs as [ | v [ | v' vs]]; try reflexivity.
    cbn [rename_cont_value heap_apply_cont_value].
    rewrite shape_of_call_equivariant.
    apply heap_run_call_equivariant. intros partial. apply IHreturn.
Qed.

Lemma heap_call_function_equivariant (r : renaming) (callee : value)
    (kind : call_kind) (vs : list value) (args_arity return_arity : arity)
    (return_to exn_to : cont_value) (statics : static_env) (h : heap) :
  heap_call_function callee kind vs args_arity return_arity
    (rename_cont_value r return_to) (rename_cont_value r exn_to)
    (rename_static_env r statics) h
  = rename_heap_result r
      (heap_call_function callee kind vs args_arity return_arity return_to
         exn_to statics h).
Proof.
  unfold heap_call_function. rewrite shape_of_call_equivariant.
  apply heap_run_call_equivariant. intros partial.
  apply heap_apply_cont_value_equivariant.
Qed.

(* ======================================================================== *)
(* Execution is equivariant                                                 *)
(* ======================================================================== *)

Theorem heap_step_equivariant : heap_step_equivariant_statement.
Proof.
  intros r [e vars conts statics h]. unfold rename_heap_state. simpl.
  destruct e
    as [ x kind d body | bindings body | k hd body | handlers body | ap | ac
       | scrutinee arms | ];
    simpl.
  - (* Let *)
    rewrite heap_exec_named_equivariant.
    destruct (heap_exec_named statics h vars d) as [[v h'] | ];
      [ | reflexivity ].
    destruct (kind_eqb (kind_of_value v) kind); reflexivity.
  - (* Let_static *)
    rewrite eval_static_bindings_equivariant.
    destruct (eval_static_bindings vars bindings statics); reflexivity.
  - (* Let_cont_non_recursive *)
    reflexivity.
  - (* Let_cont_recursive *)
    fold (rename_handlers r handlers).
    rewrite rename_handlers_bind_statics.
    destruct (handlers_bind_statics handlers); [ reflexivity | ].
    rewrite rec_closures_equivariant, <- rename_cont_env_app.
    reflexivity.
  - (* Apply *)
    rewrite eval_simple_equivariant, eval_simples_equivariant.
    destruct (eval_simple vars (ap_callee ap)) as [callee | ];
      [ | reflexivity ].
    destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | reflexivity ].
    rewrite exn_extra_simples_rename, eval_simples_equivariant.
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
      [ | reflexivity ].
    rewrite !resolve_equivariant, with_extra_args_equivariant.
    apply heap_call_function_equivariant.
  - (* Apply_cont *)
    rewrite eval_simples_equivariant.
    destruct (eval_simples vars (ac_args ac)) as [vs | ]; [ | reflexivity ].
    rewrite resolve_equivariant. apply heap_apply_cont_value_equivariant.
  - (* Switch *)
    rewrite eval_simple_equivariant.
    destruct (eval_simple vars scrutinee) as [v | ]; [ | reflexivity ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | reflexivity ].
    rewrite find_arm_renamed.
    destruct (find_arm i arms); reflexivity.
  - (* Invalid *)
    reflexivity.
Qed.

Theorem heap_run_equivariant : heap_run_equivariant_statement.
Proof.
  intros r fuel. induction fuel as [ | fuel IH]; intros s;
    [ reflexivity | ].
  cbn [heap_run]. rewrite heap_step_equivariant.
  destruct (heap_step s) as [s' | k o | ]; cbn [rename_heap_result];
    [ apply IH | | ]; reflexivity.
Qed.

(* ======================================================================== *)
(* Behaviours                                                               *)
(* ======================================================================== *)

Lemma rename_static_env_inverse_l (r : renaming) (statics : static_env) :
  rename_static_env (inverse_renaming r) (rename_static_env r statics)
  = statics.
Proof.
  pose proof
    (rename_state_inverse r (State Invalid [] [] statics empty_heap no_reads))
    as [Hleft _].
  unfold rename_state in Hleft. simpl_state.
  injection Hleft as Hstatics. exact Hstatics.
Qed.

Lemma rename_static_env_inverse_r (r : renaming) (statics : static_env) :
  rename_static_env r (rename_static_env (inverse_renaming r) statics)
  = statics.
Proof.
  pose proof
    (rename_state_inverse r (State Invalid [] [] statics empty_heap no_reads))
    as [_ Hright].
  unfold rename_state in Hright. simpl_state.
  injection Hright as Hstatics. exact Hstatics.
Qed.

Lemma rename_outcome_injective (r : renaming) (o o' : outcome) :
  rename_outcome r o = rename_outcome r o' -> o = o'.
Proof.
  intros Heq.
  pose proof (f_equal outcome_values Heq) as Hvs.
  pose proof (f_equal outcome_statics Heq) as Hstatics.
  pose proof (f_equal outcome_heap Heq) as Hheap.
  destruct o as [vs statics h], o' as [vs' statics' h'].
  cbn [rename_outcome outcome_values outcome_statics outcome_heap]
    in Hvs, Hstatics, Hheap.
  subst vs' h'. f_equal.
  rewrite <- (rename_static_env_inverse_l r statics),
    <- (rename_static_env_inverse_l r statics'), Hstatics.
  reflexivity.
Qed.

Lemma rename_outcome_inverse_r (r : renaming) (o : outcome) :
  rename_outcome r (rename_outcome (inverse_renaming r) o) = o.
Proof.
  destruct o as [vs statics h]. unfold rename_outcome.
  cbn [outcome_values outcome_statics outcome_heap].
  rewrite rename_static_env_inverse_r. reflexivity.
Qed.

Lemma rename_outcome_inverse_l (r : renaming) (o : outcome) :
  rename_outcome (inverse_renaming r) (rename_outcome r o) = o.
Proof.
  destruct o as [vs statics h]. unfold rename_outcome.
  cbn [outcome_values outcome_statics outcome_heap].
  rewrite rename_static_env_inverse_l. reflexivity.
Qed.

Lemma rename_heap_behaviour_inverse (r : renaming) (b : heap_behaviour) :
  rename_heap_behaviour r (rename_heap_behaviour (inverse_renaming r) b)
  = b.
Proof.
  destruct b as [k o | | ]; cbn [rename_heap_behaviour]; try reflexivity.
  rewrite rename_continuation_inverse_r, rename_outcome_inverse_r.
  reflexivity.
Qed.

Lemma heap_run_renamed_exit (r : renaming) (fuel : nat) (s : heap_state)
    (k : continuation) (o : outcome) :
  heap_run fuel (rename_heap_state r s)
  = Heap_exit (rename_continuation r k) (rename_outcome r o)
  <-> heap_run fuel s = Heap_exit k o.
Proof.
  rewrite heap_run_equivariant. split.
  - destruct (heap_run fuel s) as [s' | k' o' | ]; cbn [rename_heap_result];
      try discriminate.
    intros Heq.
    assert (Hk : rename_continuation r k' = rename_continuation r k)
      by congruence.
    assert (Ho : rename_outcome r o' = rename_outcome r o) by congruence.
    apply rename_continuation_injective in Hk.
    apply rename_outcome_injective in Ho. subst. reflexivity.
  - intros ->. reflexivity.
Qed.

Lemma heap_run_renamed_undefined (r : renaming) (fuel : nat)
    (s : heap_state) :
  heap_run fuel (rename_heap_state r s) = Heap_undefined
  <-> heap_run fuel s = Heap_undefined.
Proof.
  rewrite heap_run_equivariant.
  destruct (heap_run fuel s); cbn [rename_heap_result]; split; intros Hrun;
    try discriminate; reflexivity.
Qed.

Lemma heap_run_renamed_step (r : renaming) (fuel : nat) (s : heap_state) :
  (∃ s' : heap_state, heap_run fuel (rename_heap_state r s) = Heap_step s')
  <-> ∃ s' : heap_state, heap_run fuel s = Heap_step s'.
Proof.
  rewrite heap_run_equivariant.
  destruct (heap_run fuel s) as [s' | k o | ]; cbn [rename_heap_result];
    split; intros [t Ht]; try discriminate; eauto.
Qed.

Theorem heap_behaves_equivariant : heap_behaves_equivariant_statement.
Proof.
  intros r s [k o | | ]; cbn [rename_heap_behaviour heap_behaves].
  - split; intros [fuel Hrun]; exists fuel;
      apply (heap_run_renamed_exit r fuel s k o); exact Hrun.
  - split; intros [fuel Hrun]; exists fuel;
      apply (heap_run_renamed_undefined r fuel s); exact Hrun.
  - split; intros Hdiverges fuel;
      apply (heap_run_renamed_step r fuel s), Hdiverges.
Qed.

(* A behaviour of a renamed state is the renaming of one of the state. *)
Lemma heap_behaves_renamed (r : renaming) (s : heap_state)
    (b : heap_behaviour) :
  heap_behaves (rename_heap_state r s) b
  <-> heap_behaves s (rename_heap_behaviour (inverse_renaming r) b).
Proof.
  rewrite <- (rename_heap_behaviour_inverse r b) at 1.
  apply heap_behaves_equivariant.
Qed.

(* ======================================================================== *)
(* The relation on outcomes does not read code                              *)
(* ======================================================================== *)

Section Code_changed.

Variable poison_refines : bool.
Variable corr : correspondence.
Variables codes1 codes1' codes2 codes2' : list (code_id * code).
Variables values values' : list (symbol * statically_allocated_value).

Let st1 := Static_env codes1 values.
Let st1' := Static_env codes1' values'.
Let st2 := Static_env codes2 values.
Let st2' := Static_env codes2' values'.

Lemma value_step_code (R : value -> value -> Prop) (v v' : value) :
  value_step poison_refines corr st1 st1' R v v' ->
  value_step poison_refines corr st2 st2' R v v'.
Proof.
  intros Hstep.
  destruct Hstep
    as [c | l l' Hl | v v' tag fields fields' Hv Hv' Hfields
       | v v' c c' Hv Hv' Hc | c | v v' bytes Hv Hv'
       | v v' fslot fdecls slots slots' Hv Hv' Hnames Hslots Hsiblings
       | gslot gdecls gslots Hgslot | s Hs Hs'
       | kind v' Hpoison Hkind
       | callee callee' args args' args_arity Hcallee Hargs
       | v v' elements elements' Hv Hv' Helements].
  - apply Step_const.
  - apply Step_location. exact Hl.
  - exact (Step_block poison_refines corr st2 st2' R v v' tag fields fields'
             Hv Hv' Hfields).
  - exact (Step_boxed_number poison_refines corr st2 st2' R v v' c c' Hv Hv'
             Hc).
  - apply Step_same_boxed_number.
  - exact (Step_string poison_refines corr st2 st2' R v v' bytes Hv Hv').
  - exact (Step_closure poison_refines corr st2 st2' R v v' fslot fdecls slots
             slots' Hv Hv' Hnames Hslots Hsiblings).
  - apply Step_same_closure_of_set. exact Hgslot.
  - apply Step_opaque_symbol; [ exact Hs | exact Hs' ].
  - apply Step_poison; assumption.
  - apply Step_partial_application; assumption.
  - exact (Step_array poison_refines corr st2 st2' R v v' elements elements'
             Hv Hv' Helements).
Qed.

Lemma value_related_code (v v' : value) :
  value_related poison_refines corr st1 st1' v v' ->
  value_related poison_refines corr st2 st2' v v'.
Proof.
  revert v v'. apply value_related_induction.
  - intros v v' Hcorr. apply Related_corr. exact Hcorr.
  - intros v v' Hstep. apply Related_step. apply value_step_code.
    revert Hstep. apply value_step_mono. intros a a' [_ Ha]. exact Ha.
Qed.

End Code_changed.

Lemma outcome_related_code (poison_refines : bool)
    (codes1 codes1' codes2 codes2' : list (code_id * code))
    (values values' : list (symbol * statically_allocated_value))
    (vs vs' : list value) (h h' : heap) :
  outcome_related poison_refines (Outcome vs (Static_env codes1 values) h)
    (Outcome vs' (Static_env codes1' values') h') ->
  outcome_related poison_refines (Outcome vs (Static_env codes2 values) h)
    (Outcome vs' (Static_env codes2' values') h').
Proof.
  intros (corr & (Hbijection & Hcells & Hvalues) & Hrelated).
  exists corr. simpl in *. split; [ split; [ | split ] | ].
  - exact Hbijection.
  - intros l l' Hl. destruct (Hcells l l' Hl) as (c & c' & Hc & Hc' & Hcell).
    exists c, c'. split; [ exact Hc | ]. split; [ exact Hc' | ].
    destruct c as [mut tag fields | kind elements],
      c' as [mut' tag' fields' | kind' elements'];
      try (exfalso; exact Hcell).
    + destruct Hcell as (Hmut & Htag & Hfields).
      split; [ exact Hmut | ]. split; [ exact Htag | ].
      revert Hfields. apply Forall2_impl. intros a a'.
      apply value_related_code.
    + destruct Hcell as (Hkind & Helements).
      split; [ exact Hkind | ].
      revert Helements. apply Forall2_impl. intros a a'.
      apply value_related_code.
  - intros v v' Hcorr.
    apply (value_step_code poison_refines corr codes1 codes1' codes2 codes2').
    generalize (Hvalues v v' Hcorr). apply value_step_mono.
    intros a a'. apply value_related_code.
  - revert Hrelated. apply Forall2_impl. intros a a'.
    apply value_related_code.
Qed.

Theorem outcome_related_equivariant : outcome_related_equivariant_statement.
Proof.
  intros poison_refines r r' [vs [codes values] h] [vs' [codes' values'] h'].
  unfold rename_outcome, rename_static_env. simpl.
  split; apply outcome_related_code.
Qed.

(* ======================================================================== *)
(* Refinement is preserved                                                  *)
(* ======================================================================== *)

Lemma outcome_related_left (poison_refines : bool) (r : renaming)
    (o o' : outcome) :
  outcome_related poison_refines (rename_outcome r o) o'
  <-> outcome_related poison_refines o o'.
Proof.
  destruct o as [vs [codes values] h], o' as [vs' [codes' values'] h'].
  unfold rename_outcome, rename_static_env. simpl.
  split; apply outcome_related_code.
Qed.

Lemma outcome_related_right (poison_refines : bool) (r : renaming)
    (o o' : outcome) :
  outcome_related poison_refines o (rename_outcome r o')
  <-> outcome_related poison_refines o o'.
Proof.
  destruct o as [vs [codes values] h], o' as [vs' [codes' values'] h'].
  unfold rename_outcome, rename_static_env. simpl.
  split; apply outcome_related_code.
Qed.

Lemma outcome_refines_renamed (r : renaming) (o o' : outcome) :
  outcome_refines (rename_outcome (inverse_renaming r) o) o' ->
  outcome_refines o (rename_outcome r o').
Proof.
  unfold outcome_refines. intros Hrefines.
  apply outcome_related_right. apply outcome_related_left in Hrefines.
  exact Hrefines.
Qed.

Theorem heap_refines_equivariant : heap_refines_equivariant_statement.
Proof.
  intros r source target [Hexits Hends]. split.
  - intros k o Hsource. apply heap_behaves_renamed in Hsource.
    cbn [rename_heap_behaviour] in Hsource.
    destruct (Hexits _ _ Hsource) as (o' & Htarget & Ho).
    exists (rename_outcome r o'). split.
    + apply heap_behaves_renamed. cbn [rename_heap_behaviour].
      rewrite rename_outcome_inverse_l. exact Htarget.
    + apply outcome_refines_renamed, Ho.
  - intros b' Hb' Htarget. apply heap_behaves_renamed in Htarget.
    assert (Hnot :
      rename_heap_behaviour (inverse_renaming r) b' <> Heap_diverges).
    { destruct b'; cbn [rename_heap_behaviour];
        [ discriminate | discriminate | exact Hb' ]. }
    destruct (Hends _ Hnot Htarget) as [(b & Hsource & Hb) | Hwrong].
    + left. exists (rename_heap_behaviour r b). split.
      * apply heap_behaves_equivariant, Hsource.
      * destruct b as [k o | | ], b' as [k' o' | | ];
          cbn [rename_heap_behaviour heap_behaviour_refines] in Hb |- *;
          try exact Hb.
        destruct Hb as [Hk Ho]. split.
        -- rewrite Hk. apply rename_continuation_inverse_r.
        -- unfold outcome_refines in Ho |- *. apply outcome_related_left.
           apply outcome_related_right in Ho. exact Ho.
    + right.
      apply (heap_behaves_equivariant r source Heap_goes_wrong), Hwrong.
Qed.

(* A state with a renamed expression is the renaming of the state whose
   environments are renamed back. *)
Lemma heap_state_of_renamed_expr (r : renaming) (e : expr) (vars : env)
    (conts : cont_env) (statics : static_env) (h : heap) :
  Heap_state (rename_expr r e) vars conts statics h
  = rename_heap_state r
      (Heap_state e (rename_env (inverse_renaming r) vars)
         (rename_cont_env (inverse_renaming r) conts)
         (rename_static_env (inverse_renaming r) statics) h).
Proof.
  pose proof (state_of_renamed_expr r e vars conts statics h no_reads)
    as Hstate.
  unfold rename_state in Hstate. simpl_state.
  injection Hstate as Hvars Hconts Hstatics.
  unfold rename_heap_state. simpl.
  rewrite <- Hvars, <- Hconts, <- Hstatics. reflexivity.
Qed.

(* Whether a state is closed is said of its values, which the action
   leaves alone. *)
Lemma env_closed_renamed (r : renaming) (h : heap) (vars : env) :
  (∀ (x : variable) (v : value), (x, v) ∈ vars -> value_closed h v) ->
  ∀ (x : variable) (v : value),
    (x, v) ∈ rename_env r vars -> value_closed h v.
Proof.
  intros Hclosed x v Hin. unfold rename_env in Hin.
  apply in_map_iff in Hin. destruct Hin as [[y w] [Heq Hin]].
  injection Heq as _ <-. exact (Hclosed y w Hin).
Qed.

Fixpoint cont_value_closed_renamed (r : renaming) (h : heap)
    (c : cont_value) {struct c} :
  cont_value_closed h c -> cont_value_closed h (rename_cont_value r c).
Proof.
  destruct c
    as [hd vars conts recursive | k | c extra
       | remaining remaining_arity return_arity return_to exn_to];
    simpl;
    [ | exact (fun H => H)
    | intros [Hextra Hc]; split;
      [ exact Hextra | exact (cont_value_closed_renamed r h c Hc) ]
    | intros (Hremaining & Hreturn & Hexn); split; [ exact Hremaining | ];
      split;
      [ exact (cont_value_closed_renamed r h return_to Hreturn)
      | exact (cont_value_closed_renamed r h exn_to Hexn) ] ].
  intros [Hvars Hconts]. split.
  - exact (env_closed_renamed r h vars Hvars).
  - induction conts as [ | [k c] conts IH]; simpl; [ exact I | ].
    destruct Hconts as [Hc Hrest].
    split; [ exact (cont_value_closed_renamed r h c Hc) | exact (IH Hrest) ].
Qed.

Lemma state_closed_renamed (r : renaming) (s : heap_state) :
  state_closed s -> state_closed (rename_heap_state r s).
Proof.
  destruct s as [e vars conts [codes values] h].
  intros (Hmemory & Hvars & Hconts). unfold state_closed. simpl.
  split; [ exact Hmemory | ]. split.
  - exact (env_closed_renamed r h vars Hvars).
  - intros k c Hin. unfold rename_cont_env in Hin.
    apply in_map_iff in Hin. destruct Hin as [[k0 c0] [Heq Hin]].
    injection Heq as _ <-.
    exact (cont_value_closed_renamed r h c0 (Hconts k0 c0 Hin)).
Qed.

Theorem heap_expr_refines_equivariant :
  heap_expr_refines_equivariant_statement.
Proof.
  intros r source target Hrefines vars conts statics h Hclosed.
  rewrite !heap_state_of_renamed_expr.
  apply heap_refines_equivariant, Hrefines.
  rewrite heap_state_of_renamed_expr in Hclosed.
  apply (state_closed_renamed (inverse_renaming r)) in Hclosed.
  destruct (rename_heap_state_inverse r
              (Heap_state source (rename_env (inverse_renaming r) vars)
                 (rename_cont_env (inverse_renaming r) conts)
                 (rename_static_env (inverse_renaming r) statics) h))
    as [Hleft _].
  rewrite Hleft in Hclosed. exact Hclosed.
Qed.
