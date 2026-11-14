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

(* Proofs of the statements of FlambdaRegionsInvariantSpec.v that are
   about the test [no_poison]: the state that a run starts in passes
   it, a step and a run keep it, and in a state that passes it a value
   which may be evaluated is not of the kind [Region]. Nothing here
   needs review. *)

From Stdlib Require Import ZArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaPrimitiveModel FlambdaPrimitiveSpec
  FlambdaPrimitiveProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaClosuresProofs FlambdaRegionsSpec FlambdaRegionsProofs
  FlambdaRegionsInvariantSpec FlambdaRegionsReadsProofs
  FlambdaRegionsStepsProofs.
Import ListNotations.

(* ======================================================================== *)
(* The simples of a term that has no such constant                          *)
(* ======================================================================== *)

Lemma eval_simple_no_region_poison (vars : env) (s : simple) (v : value) :
  simple_no_region_poison s = true ->
  env_values_all (fun v => value_no_region_poison v = true) vars ->
  eval_simple vars s = Some v ->
  value_no_region_poison v = true.
Proof.
  intros Hs Hvars. destruct s as [x | sym | c]; simpl.
  - exact (env_values_all_lookup _ vars x v Hvars).
  - intros [= <-]. reflexivity.
  - intros [= <-]. exact Hs.
Qed.

Lemma evaluated_simples_no_region_poison (e : expr) (sim : simple) :
  no_region_poison e = true ->
  In sim (evaluated_simples e) ->
  simple_no_region_poison sim = true.
Proof.
  destruct e
    as [x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ].
  - intros He Hin. apply evaluated_simples_let in Hin. simpl in He.
    apply andb_true_iff in He as [Hd _]. exact (forallb_in _ _ sim Hd Hin).
  - intros He Hin. simpl in He, Hin. apply andb_true_iff in He as [Hb _].
    apply in_flat_map in Hin as (binding & Hbinding & Hin).
    pose proof (forallb_in _ _ binding Hb Hbinding) as Hsimples.
    destruct binding as [id c | id | s const]; simpl in Hin;
      try contradiction.
    exact (forallb_in _ _ sim Hsimples Hin).
  - intros _ [].
  - intros _ [].
  - intros He Hin. exact (forallb_in _ _ sim He Hin).
  - intros He Hin. exact (forallb_in _ _ sim He Hin).
  - intros He Hin. simpl in He, Hin. apply andb_true_iff in He as [Hs _].
    destruct Hin as [<- | []]. exact Hs.
  - intros _ [].
Qed.

Lemma named_simples_no_region_poison (vars : env) (x : variable)
    (kind : flambda_kind) (d : named) (body : expr) (sim : simple)
    (w : value) :
  no_region_poison (Let x kind d body) = true ->
  env_values_all (fun v => value_no_region_poison v = true) vars ->
  In sim (simples_of_named d) ->
  eval_simple vars sim = Some w ->
  value_no_region_poison w = true.
Proof.
  intros He Hvars Hin. simpl in He. apply andb_true_iff in He as [Hd _].
  exact
    (eval_simple_no_region_poison vars sim w (forallb_in _ _ sim Hd Hin)
       Hvars).
Qed.

(* ======================================================================== *)
(* The test of what is written                                              *)
(* ======================================================================== *)

Lemma no_region_poison_let (x : variable) (kind : flambda_kind) (d : named)
    (body : expr) :
  no_region_poison (Let x kind d body) = true -> no_region_poison body = true.
Proof. simpl. intros H. apply andb_true_iff in H as [_ H]. exact H. Qed.

Lemma no_region_poison_let_static (bindings : list static_binding)
    (body : expr) :
  no_region_poison (Let_static bindings body) = true ->
  no_region_poison body = true
  ∧ ∀ (id : code_id) (c : code),
      In (Bind_code id c) bindings -> code_no_region_poison c = true.
Proof.
  simpl. intros H. apply andb_true_iff in H as [Hb H]. split; [ exact H | ].
  intros id c Hin. exact (forallb_in _ _ _ Hb Hin).
Qed.

Lemma no_region_poison_let_cont (k : continuation)
    (h : continuation_handler) (body : expr) :
  no_region_poison (Let_cont_non_recursive k h body) = true ->
  handler_no_region_poison h = true ∧ no_region_poison body = true.
Proof. simpl. intros H. apply andb_true_iff in H. exact H. Qed.

Lemma no_region_poison_let_cont_rec
    (handlers : list (continuation * continuation_handler)) (body : expr) :
  no_region_poison (Let_cont_recursive handlers body) = true ->
  forallb (fun '(_, h) => handler_no_region_poison h) handlers = true
  ∧ no_region_poison body = true.
Proof. simpl. intros H. apply andb_true_iff in H. exact H. Qed.

Lemma no_region_poison_switch (scrutinee : simple)
    (arms : list (Z * apply_cont)) (i : Z) (ac : apply_cont) :
  no_region_poison (Switch scrutinee arms) = true ->
  find_arm i arms = Some ac ->
  no_region_poison (Apply_cont ac) = true.
Proof.
  simpl. intros H Harm. apply andb_true_iff in H as [_ Harms].
  destruct (find_arm_some_in i arms ac Harm) as (i' & Hin).
  exact (forallb_in _ _ (i', ac) Harms Hin).
Qed.

Lemma handler_no_region_poison_handler (params : list variable)
    (metadata : handler_metadata) (handler : expr) :
  handler_no_region_poison (Handler params metadata handler) = true ->
  no_region_poison handler = true.
Proof. exact (fun H => H). Qed.

Lemma code_no_region_poison_body (c : code) :
  code_no_region_poison c = true -> no_region_poison (code_body c) = true.
Proof. destruct c. exact (fun H => H). Qed.

(* What a state has written has no such constant: its expression, the
   handlers that its continuations hold, the code of its static
   environment. *)
Definition written_no_poison (s : state) : Prop :=
  written_ok no_region_poison handler_no_region_poison code_no_region_poison
    s.

Lemma region_step_keeps_written_no_poison (rs rs' : region_state) :
  written_no_poison (rs_state rs) ->
  region_step rs = Region_step rs' ->
  written_no_poison (rs_state rs').
Proof.
  exact
    (region_step_written no_region_poison handler_no_region_poison
       code_no_region_poison no_region_poison_let no_region_poison_let_static
       no_region_poison_let_cont no_region_poison_let_cont_rec
       no_region_poison_switch handler_no_region_poison_handler
       code_no_region_poison_body rs rs').
Qed.

(* ======================================================================== *)
(* The parts of the test                                                    *)
(* ======================================================================== *)

(* What a state holds has no such constant: the values of its variables
   and of its continuations. *)
Definition held_no_poison (s : state) : Prop :=
  vals_ok (fun v => value_no_region_poison v = true) s.

(* What is bound, in a frame or in the static environment, has no such
   constant. *)
Definition bound_no_poison
    (values : list (symbol * statically_allocated_value))
    (frames : list frame) : Prop :=
  ∀ (s : symbol) (allocated : statically_allocated_value),
    In (s, allocated) (live_locals frames ++ values) ->
    allocated_no_region_poison allocated = true.

(* The test [no_poison], in three parts. *)
Lemma no_poison_parts (rs : region_state) :
  no_poison rs = true
  <-> written_no_poison (rs_state rs)
      ∧ held_no_poison (rs_state rs)
      ∧ bound_no_poison (static_values (state_statics (rs_state rs)))
          (rs_frames rs).
Proof.
  unfold no_poison, statics_no_region_poison. cbv zeta. split.
  - intros H.
    apply andb_true_iff in H as [H H5]. apply andb_true_iff in H as [H H4].
    apply andb_true_iff in H as [H H3]. apply andb_true_iff in H as [H1 H2].
    apply andb_true_iff in H2 as [Hcode Hvalues].
    apply conts_no_region_poison_split in H4 as [H4t H4v].
    split; [ exact (conj H1 (conj H4t Hcode)) | ].
    split; [ exact (conj (proj1 (forallb_pairs_Forall _ _) H3) H4v) | ].
    intros s allocated Hin. apply in_app_or in Hin as [Hin | Hin].
    + exact (forallb_in _ _ (s, allocated) H5 Hin).
    + exact (forallb_in _ _ (s, allocated) Hvalues Hin).
  - intros ((H1 & H4t & Hcode) & (H3 & H4v) & Hbound).
    repeat (apply andb_true_iff; split).
    + exact H1.
    + exact Hcode.
    + apply forallb_forall. intros [s allocated] Hin.
      apply (Hbound s allocated). apply in_or_app. right. exact Hin.
    + exact (proj2 (forallb_pairs_Forall _ _) H3).
    + apply conts_no_region_poison_split. exact (conj H4t H4v).
    + apply forallb_forall. intros [s allocated] Hin.
      apply (Hbound s allocated). apply in_or_app. left. exact Hin.
Qed.

(* ======================================================================== *)
(* What is held after a step                                                *)
(* ======================================================================== *)

Lemma value_no_region_poison_parts_in (v w : value) :
  value_no_region_poison v = true ->
  In w (value_parts v) ->
  value_no_region_poison w = true.
Proof.
  intros Hv. exact (forallb_in _ _ w (value_no_region_poison_parts v Hv)).
Qed.

Lemma value_no_region_poison_move (slot slot' : function_slot)
    (function_decls : list (function_slot * code_id))
    (value_slots : list (value_slot * value)) :
  value_no_region_poison
    (Value_closure_of_set slot function_decls value_slots)
  = true ->
  value_no_region_poison
    (Value_closure_of_set slot' function_decls value_slots)
  = true.
Proof. exact (fun H => H). Qed.

Lemma value_no_region_poison_partial (callee : value) (args : list value)
    (args_arity : arity) :
  value_no_region_poison (Value_partial_application callee args args_arity)
  = true
  <-> value_no_region_poison callee = true
      ∧ Forall (fun v => value_no_region_poison v = true) args.
Proof. simpl. rewrite andb_true_iff, forallb_Forall. reflexivity. Qed.

(* What a symbol is bound to has no such constant, if that is so of
   everything bound. *)
Lemma value_no_region_poison_static (rs : region_state) :
  bound_no_poison (static_values (state_statics (rs_state rs)))
    (rs_frames rs) ->
  ∀ (s : symbol) (allocated : statically_allocated_value),
    value_no_region_poison (Value_symbol s) = true ->
    lookup s (static_values (live_statics rs)) = Some allocated ->
    (∀ w : value,
      In w (allocated_values allocated) -> value_no_region_poison w = true)
    ∧ (∀ s' : symbol,
        In s' (allocated_symbols allocated) ->
        value_no_region_poison (Value_symbol s') = true).
Proof.
  intros Hbound s allocated _ Hlookup.
  pose proof (Hbound s allocated (lookup_in _ _ _ Hlookup)) as Ha.
  rewrite allocated_no_region_poison_parts in Ha. split.
  - exact (fun w => forallb_in _ _ w Ha).
  - intros s' _. reflexivity.
Qed.

(* The values of a state after a step, and those of an exit, have no
   such constant. *)
Lemma region_step_no_poison_values (rs : region_state) :
  no_poison rs = true ->
  region_step_vals_ok (fun v => value_no_region_poison v = true) rs
    (region_step rs).
Proof.
  intros Hpoison.
  pose proof (proj1 (no_poison_parts rs) Hpoison)
    as ((He & _ & _) & Hheld & Hbound).
  apply (region_step_values _ value_no_region_poison_partial rs).
  - apply reads_fields;
      [ exact value_no_region_poison_parts_in
      | exact (value_no_region_poison_static rs Hbound) ].
  - exact Hheld.
  - intros x kind d body v Hexpr _ _ Hnamed. rewrite Hexpr in He.
    assert (Hsimples :
      ∀ (sim : simple) (w : value),
        In sim (simples_of_named d) ->
        eval_simple (state_vars (rs_state rs)) sim = Some w ->
        value_no_region_poison w = true).
    { intros sim w.
      exact
        (named_simples_no_region_poison _ x kind d body sim w He
           (proj1 Hheld)). }
    destruct (is_allocation d) eqn:His.
    + destruct (eval_named_makes _ _ d v His Hnamed) as [Hmade Hparts].
      apply (value_no_region_poison_made v Hmade). apply forallb_forall.
      intros w Hin. destruct (Hparts w Hin) as (sim & Hsim & Hw).
      exact (Hsimples sim w Hsim Hw).
    + exact
        (eval_named_reads (fun v => value_no_region_poison v = true)
           (live_statics rs) (fun c H => H) value_no_region_poison_parts_in
           value_no_region_poison_move
           (value_no_region_poison_static rs Hbound)
           (state_vars (rs_state rs)) d v His Hsimples Hnamed).
  - intros sim v Hin Hv.
    exact
      (eval_simple_no_region_poison _ sim v
         (evaluated_simples_no_region_poison _ sim He Hin) (proj1 Hheld) Hv).
  - reflexivity.
Qed.

(* ======================================================================== *)
(* What is bound after a step                                               *)
(* ======================================================================== *)

Lemma region_step_bound_no_poison (rs rs' : region_state) :
  no_poison rs = true ->
  region_step rs = Region_step rs' ->
  bound_no_poison (static_values (state_statics (rs_state rs')))
    (rs_frames rs').
Proof.
  intros Hpoison.
  pose proof (proj1 (no_poison_parts rs) Hpoison)
    as ((He & _ & _) & [Hvars _] & Hbound).
  apply
    (region_step_cases rs
       (fun result =>
          result = Region_step rs' ->
          bound_no_poison (static_values (state_statics (rs_state rs')))
            (rs_frames rs'))).
  - discriminate.
  - discriminate.
  - intros x kind d mark_kind body _ Hstep.
    rewrite (begin_region_step _ _ _ _ _ _ Hstep). exact Hbound.
  - intros x kind d ended_by region body _ Hstep.
    destruct (end_region_step _ _ _ _ _ _ _ Hstep)
      as (m & above & f & below & counts & _ & Hframes & ->).
    change (rs_frames (count_uses rs)) with (rs_frames rs) in Hframes.
    intros s allocated Hin. apply (Hbound s allocated). rewrite Hframes.
    apply in_app_or in Hin as [Hin | Hin]; apply in_or_app;
      [ left | right; exact Hin ].
    rewrite live_locals_app. apply in_or_app. right.
    change (live_locals (f :: below))
      with (frame_locals f ++ live_locals below).
    apply in_or_app. right. exact Hin.
  - intros x kind d body y region body' Hexpr _ _ _ Hstep.
    destruct (local_allocation_step _ _ _ _ _ _ _ _ Hstep)
      as (v & sym & bindings & His & Hnamed & Hlocalise & _ & _ & _ & _ & _
          & _ & Hframes).
    destruct (localise_makes _ _ _ _ Hlocalise) as (_ & _ & Hmade).
    destruct (eval_named_makes _ _ d v His Hnamed) as [_ Hparts].
    cbn [count_uses with_counts rs_state rs_frames rs_base rs_next]
      in Hframes, Hparts.
    rewrite Hexpr in He.
    assert (Hbindings :
      ∀ (s : symbol) (allocated : statically_allocated_value),
        In (s, allocated) bindings ->
        allocated_no_region_poison allocated = true).
    { intros s allocated Hin.
      destruct (Hmade s allocated Hin) as (_ & _ & Hvalues & _).
      rewrite allocated_no_region_poison_parts, Hvalues.
      apply forallb_forall. intros w Hw.
      destruct (Hparts w Hw) as (sim & Hsim & Heval).
      exact
        (named_simples_no_region_poison _ x kind d body sim w He Hvars Hsim
           Heval). }
    revert Hframes Hbound.
    destruct (rs_frames rs) as [ | [m mark_kind locals] frames];
      intros [Hf Hs] Hbound s allocated Hin; rewrite Hf, Hs in Hin.
    + simpl in Hin. apply in_app_or in Hin as [Hin | Hin];
        [ exact (Hbindings s allocated Hin) | exact (Hbound s allocated Hin) ].
    + change (live_locals (Frame m mark_kind (bindings ++ locals) :: frames))
        with ((bindings ++ locals) ++ live_locals frames) in Hin.
      rewrite <- !app_assoc in Hin. apply in_app_or in Hin as [Hin | Hin];
        [ exact (Hbindings s allocated Hin) | ].
      apply (Hbound s allocated).
      change (live_locals (Frame m mark_kind locals :: frames))
        with (locals ++ live_locals frames).
      rewrite <- app_assoc. exact Hin.
  - intros x kind d body _ _ _ _ Hstep.
    destruct (plain_step_step _ _ Hstep) as (s' & _ & ->). exact Hbound.
  - intros bindings body Hexpr _ _ Hstep.
    destruct (static_step_step (count_uses rs) rs' bindings body Hexpr Hstep)
      as (statics' & Heval & ->).
    destruct (eval_static_bindings_parts _ _ _ _ Heval)
      as (new_code & new_values & _ & Hvalues & _ & Hnew).
    cbn [count_uses with_counts rs_state rs_frames rs_base rs_next
         state_statics]
      in Hvalues |- *.
    unfold bound_no_poison. rewrite Hvalues.
    rewrite Hexpr in He. simpl in He. apply andb_true_iff in He as [Hb _].
    intros s allocated Hin. apply in_app_or in Hin as [Hin | Hin];
      [ apply (Hbound s allocated); apply in_or_app; left; exact Hin | ].
    apply in_app_or in Hin as [Hin | Hin];
      [ | apply (Hbound s allocated); apply in_or_app; right; exact Hin ].
    destruct (Hnew s allocated Hin) as (const & Hbinding & Hconst).
    pose proof (forallb_in _ _ _ Hb Hbinding) as Hsimples.
    destruct (eval_static_const_makes _ const allocated Hconst)
      as [Hmakes _].
    rewrite allocated_no_region_poison_parts. apply forallb_forall.
    intros w Hw. destruct (Hmakes w Hw) as (sim & Hsim & Hsimval).
    exact
      (eval_simple_no_region_poison _ sim w
         (forallb_in _ _ sim Hsimples Hsim) Hvars Hsimval).
  - intros ac _ Hstep.
    destruct (region_apply_cont_step _ _ ac Hstep) as (Hf & _ & _ & Hs).
    rewrite Hf, Hs. exact Hbound.
  - intros _ _ Hstep.
    destruct (plain_step_step _ _ Hstep) as (s' & _ & ->). exact Hbound.
Qed.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Theorem no_poison_initial : no_poison_initial_statement.
Proof.
  intros e statics base He Hstatics.
  unfold statics_no_region_poison in Hstatics.
  apply andb_true_iff in Hstatics as [Hcode Hvalues].
  apply no_poison_parts. split; [ exact (conj He (conj I Hcode)) | ].
  split; [ exact (conj (Forall_nil _) I) | ].
  intros s allocated Hin. exact (forallb_in _ _ (s, allocated) Hvalues Hin).
Qed.

Theorem region_step_keeps_no_poison : region_step_keeps_no_poison_statement.
Proof.
  intros rs rs' Hpoison Hstep.
  pose proof (proj1 (no_poison_parts rs) Hpoison) as (Hwritten & _ & _).
  apply no_poison_parts. split; [ | split ].
  - exact (region_step_keeps_written_no_poison rs rs' Hwritten Hstep).
  - pose proof (region_step_no_poison_values rs Hpoison) as Hvalues.
    rewrite Hstep in Hvalues. apply Hvalues. intros s _ _. reflexivity.
  - exact (region_step_bound_no_poison rs rs' Hpoison Hstep).
Qed.

Theorem region_run_keeps_no_poison : region_run_keeps_no_poison_statement.
Proof.
  intros fuel. induction fuel as [ | fuel IH]; intros rs rs' Hpoison; simpl.
  - intros [= <-]. exact Hpoison.
  - destruct (region_step rs) as [rs1 | k vs rs1 | fault | ] eqn:Hstep;
      try discriminate.
    exact (IH rs1 rs' (region_step_keeps_no_poison rs rs1 Hpoison Hstep)).
Qed.

(* A value that has no such constant is not of the kind [Region]. *)
Lemma value_no_region_poison_kind (v : value) :
  value_no_region_poison v = true -> kind_of_value v <> Region.
Proof.
  destruct v as [c | | | | | | | | ]; try (intros _; discriminate).
  destruct c as [ | | | | | | | | | | | | | | kind];
    try (intros _; discriminate).
  simpl. intros H Hkind. rewrite Hkind in H. discriminate H.
Qed.

Theorem usable_not_region : usable_not_region_statement.
Proof.
  intros rs s v Hpoison Hs Heval _.
  apply value_no_region_poison_kind.
  destruct (proj1 (no_poison_parts rs) Hpoison) as (_ & [Hvars _] & _).
  exact (eval_simple_no_region_poison _ s v Hs Hvars Heval).
Qed.

(* The state that a step exits with passes the test, and the values of
   the exit have no such constant. *)
Lemma region_step_exit_no_poison (rs : region_state) (k : continuation)
    (vs : list value) (rs' : region_state) :
  no_poison rs = true ->
  region_step rs = Region_exit k vs rs' ->
  no_poison rs' = true ∧ forallb value_no_region_poison vs = true.
Proof.
  intros Hpoison Hexit. split.
  - rewrite (region_step_exit rs rs' k vs Hexit). exact Hpoison.
  - pose proof (region_step_no_poison_values rs Hpoison) as Hvalues.
    rewrite Hexit in Hvalues. exact (proj2 (forallb_Forall _ vs) Hvalues).
Qed.
