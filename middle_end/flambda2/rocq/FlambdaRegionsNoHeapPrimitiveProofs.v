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

(* Proof of
   FlambdaRegionsInvariantSpec.no_heap_primitive_no_let_not_run_statement:
   a run from the first state of a term with no primitive of the heap,
   in a static environment whose code has none and whose values hold
   no location, never comes to a [Let] that the machine of regions
   does not run (decision D53). What a run keeps: nothing written has
   a primitive of the heap, and no value that is held or bound holds a
   location. It is the proof of FlambdaRegionsNoPoisonProofs.v with
   another test of terms (FlambdaHeapFreeSpec.expr_free_of_heap) and
   another property of values.
   FlambdaRegionsInvariantProofs.v states the theorem again under its
   own name. Nothing here needs review. *)

From Stdlib Require Import ZArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaPrimitiveModel FlambdaPrimitiveSpec
  FlambdaPrimitiveProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaHeapFreeSpec FlambdaClosuresProofs FlambdaRegionsSpec
  FlambdaRegionsProofs
  FlambdaRegionsNeedsHeapProofs FlambdaRegionsInvariantSpec
  FlambdaRegionsReadsProofs FlambdaRegionsStepsProofs.
Import ListNotations.

(* ======================================================================== *)
(* Lists                                                                    *)
(* ======================================================================== *)

Lemma flat_map_nil_in {A B : Type} (f : A -> list B) (l : list A) (x : A) :
  flat_map f l = [] -> In x l -> f x = [].
Proof.
  induction l as [ | y l IH]; simpl; [ contradiction | ].
  intros H. apply app_eq_nil in H as [Hy Hl].
  intros [<- | Hin]; [ exact Hy | exact (IH Hl Hin) ].
Qed.

Lemma flat_map_nil_of {A B : Type} (f : A -> list B) (l : list A) :
  (∀ x : A, In x l -> f x = []) -> flat_map f l = [].
Proof.
  induction l as [ | y l IH]; simpl; [ reflexivity | ].
  intros H. rewrite (H y (or_introl eq_refl)), IH; [ reflexivity | ].
  intros x Hin. exact (H x (or_intror Hin)).
Qed.

(* ======================================================================== *)
(* A value that holds no location                                           *)
(* ======================================================================== *)

Definition no_location (v : value) : Prop := locations_of_value v = [].

Lemma no_location_parts_in (v w : value) :
  no_location v -> In w (value_parts v) -> no_location w.
Proof.
  unfold no_location.
  destruct v as [c | code slots | slot decls slots | s | tag fields
                | elements | c | l | callee args args_arity];
    simpl; try contradiction.
  - intros H Hin. apply in_map_iff in Hin as (slot & <- & Hin).
    exact (flat_map_nil_in _ _ slot H Hin).
  - intros H Hin. apply in_map_iff in Hin as (slot0 & <- & Hin).
    exact (flat_map_nil_in _ _ slot0 H Hin).
  - intros H Hin. exact (flat_map_nil_in _ _ w H Hin).
  - intros H Hin. exact (flat_map_nil_in _ _ w H Hin).
  - intros _ [<- | []]. reflexivity.
Qed.

Lemma no_location_made (v : value) :
  is_made v = true ->
  (∀ w : value, In w (value_parts v) -> no_location w) ->
  no_location v.
Proof.
  unfold no_location.
  destruct v as [c | code slots | slot decls slots | s | tag fields
                | elements | c | l | callee args args_arity];
    simpl; try discriminate; intros _ H; try reflexivity.
  - apply flat_map_nil_of. intros slot Hin. apply H. apply in_map. exact Hin.
  - apply flat_map_nil_of. intros slot0 Hin. apply H. apply in_map.
    exact Hin.
  - apply flat_map_nil_of. exact H.
  - apply flat_map_nil_of. exact H.
Qed.

Lemma no_location_move (slot slot' : function_slot)
    (function_decls : list (function_slot * code_id))
    (value_slots : list (value_slot * value)) :
  no_location (Value_closure_of_set slot function_decls value_slots) ->
  no_location (Value_closure_of_set slot' function_decls value_slots).
Proof. exact (fun H => H). Qed.

Lemma no_location_partial (callee : value) (args : list value)
    (args_arity : arity) :
  no_location (Value_partial_application callee args args_arity)
  <-> no_location callee ∧ Forall no_location args.
Proof.
  unfold no_location. simpl. split.
  - intros H. apply app_eq_nil in H as [Hcallee Hargs].
    split; [ exact Hcallee | ]. apply Forall_forall. intros w Hin.
    exact (flat_map_nil_in _ _ w Hargs Hin).
  - intros [Hcallee Hargs]. rewrite Hcallee. simpl.
    apply flat_map_nil_of. intros w Hin. rewrite Forall_forall in Hargs.
    exact (Hargs w Hin).
Qed.

Lemma eval_simple_no_location (vars : env) (s : simple) (v : value) :
  env_values_all no_location vars ->
  eval_simple vars s = Some v ->
  no_location v.
Proof.
  intros Hvars. destruct s as [x | sym | c]; simpl.
  - exact (env_values_all_lookup _ vars x v Hvars).
  - intros [= <-]. reflexivity.
  - intros [= <-]. reflexivity.
Qed.

(* ======================================================================== *)
(* The test of what is written                                              *)
(* ======================================================================== *)

(* The test is FlambdaHeapFreeSpec.expr_free_of_heap, with its like
   for a handler and for code. *)

Lemma expr_free_of_heap_let (x : variable) (kind : flambda_kind) (d : named)
    (body : expr) :
  expr_free_of_heap (Let x kind d body) = true ->
  expr_free_of_heap body = true.
Proof.
  simpl. intros H. apply andb_true_iff in H as [_ H]. exact H.
Qed.

Lemma expr_free_of_heap_let_static (bindings : list static_binding)
    (body : expr) :
  expr_free_of_heap (Let_static bindings body) = true ->
  expr_free_of_heap body = true
  ∧ ∀ (id : code_id) (c : code),
      In (Bind_code id c) bindings -> code_free_of_heap c = true.
Proof.
  simpl. intros H. apply andb_true_iff in H as [Hb H].
  split; [ exact H | ].
  intros id c Hin.
  pose proof (proj1 (forallb_forall _ _) Hb (Bind_code id c) Hin) as Hc.
  simpl in Hc. exact Hc.
Qed.

Lemma expr_free_of_heap_let_cont (k : continuation)
    (h : continuation_handler) (body : expr) :
  expr_free_of_heap (Let_cont_non_recursive k h body) = true ->
  handler_free_of_heap h = true ∧ expr_free_of_heap body = true.
Proof.
  simpl. intros H. apply andb_true_iff in H. exact H.
Qed.

Lemma expr_free_of_heap_let_cont_rec
    (handlers : list (continuation * continuation_handler)) (body : expr) :
  expr_free_of_heap (Let_cont_recursive handlers body) = true ->
  forallb (fun '(_, h) => handler_free_of_heap h) handlers = true
  ∧ expr_free_of_heap body = true.
Proof.
  simpl. intros H. apply andb_true_iff in H. exact H.
Qed.

Lemma expr_free_of_heap_switch (scrutinee : simple)
    (arms : list (Z * apply_cont)) (i : Z) (ac : apply_cont) :
  expr_free_of_heap (Switch scrutinee arms) = true ->
  find_arm i arms = Some ac ->
  expr_free_of_heap (Apply_cont ac) = true.
Proof. reflexivity. Qed.

Lemma handler_free_of_heap_handler (params : list variable)
    (metadata : handler_metadata) (handler : expr) :
  handler_free_of_heap (Handler params metadata handler) = true ->
  expr_free_of_heap handler = true.
Proof. exact (fun H => H). Qed.

Lemma code_free_of_heap_body (c : code) :
  code_free_of_heap c = true -> expr_free_of_heap (code_body c) = true.
Proof. destruct c. exact (fun H => H). Qed.

(* What a state has written has no primitive of the heap: its
   expression, the handlers that its continuations hold, the code of
   its static environment. *)
Definition written_no_heap (s : state) : Prop :=
  written_ok expr_free_of_heap handler_free_of_heap code_free_of_heap s.

Lemma region_step_keeps_written_no_heap (rs rs' : region_state) :
  written_no_heap (rs_state rs) ->
  region_step rs = Region_step rs' ->
  written_no_heap (rs_state rs').
Proof.
  exact
    (region_step_written expr_free_of_heap handler_free_of_heap
       code_free_of_heap expr_free_of_heap_let expr_free_of_heap_let_static
       expr_free_of_heap_let_cont expr_free_of_heap_let_cont_rec
       expr_free_of_heap_switch handler_free_of_heap_handler
       code_free_of_heap_body rs rs').
Qed.

(* ======================================================================== *)
(* What a run keeps                                                         *)
(* ======================================================================== *)

(* What a state holds has no location: the values of its variables and
   of its continuations. *)
Definition held_no_location (s : state) : Prop := vals_ok no_location s.

(* What is bound, in a frame or in the static environment, holds no
   location. *)
Definition bound_no_location
    (values : list (symbol * statically_allocated_value))
    (frames : list frame) : Prop :=
  ∀ (s : symbol) (allocated : statically_allocated_value) (w : value),
    In (s, allocated) (live_locals frames ++ values) ->
    In w (allocated_values allocated) ->
    no_location w.

Definition no_heap_primitive (rs : region_state) : Prop :=
  written_no_heap (rs_state rs)
  ∧ held_no_location (rs_state rs)
  ∧ bound_no_location (static_values (state_statics (rs_state rs)))
      (rs_frames rs).

(* What a symbol is bound to holds no location, if that is so of
   everything bound. *)
Lemma no_location_static (rs : region_state) :
  bound_no_location (static_values (state_statics (rs_state rs)))
    (rs_frames rs) ->
  ∀ (s : symbol) (allocated : statically_allocated_value),
    no_location (Value_symbol s) ->
    lookup s (static_values (live_statics rs)) = Some allocated ->
    (∀ w : value, In w (allocated_values allocated) -> no_location w)
    ∧ (∀ s' : symbol,
        In s' (allocated_symbols allocated) ->
        no_location (Value_symbol s')).
Proof.
  intros Hbound s allocated _ Hlookup. split.
  - intros w Hw. exact (Hbound s allocated w (lookup_in _ _ _ Hlookup) Hw).
  - intros s' _. reflexivity.
Qed.

(* The values of a state after a step, and those of an exit, hold no
   location. *)
Lemma region_step_no_location_values (rs : region_state) :
  no_heap_primitive rs ->
  region_step_vals_ok no_location rs (region_step rs).
Proof.
  intros (_ & Hheld & Hbound).
  assert (Hsimples :
    ∀ (sim : simple) (w : value),
      eval_simple (state_vars (rs_state rs)) sim = Some w -> no_location w).
  { intros sim w. exact (eval_simple_no_location _ sim w (proj1 Hheld)). }
  apply (region_step_values _ no_location_partial rs).
  - apply reads_fields;
      [ exact no_location_parts_in | exact (no_location_static rs Hbound) ].
  - exact Hheld.
  - intros x kind d body v Hexpr _ _ Hnamed.
    destruct (is_allocation d) eqn:His.
    + destruct (eval_named_makes _ _ d v His Hnamed) as [Hmade Hparts].
      apply (no_location_made v Hmade). intros w Hin.
      destruct (Hparts w Hin) as (sim & _ & Hw).
      exact (Hsimples sim w Hw).
    + exact
        (eval_named_reads no_location (live_statics rs)
           (fun c _ => eq_refl) no_location_parts_in no_location_move
           (no_location_static rs Hbound) (state_vars (rs_state rs)) d v His
           (fun sim w _ => Hsimples sim w) Hnamed).
  - intros sim v _ Hv. exact (Hsimples sim v Hv).
  - reflexivity.
Qed.

(* What is bound after a step holds no location. *)
Lemma region_step_bound_no_location (rs rs' : region_state) :
  no_heap_primitive rs ->
  region_step rs = Region_step rs' ->
  bound_no_location (static_values (state_statics (rs_state rs')))
    (rs_frames rs').
Proof.
  intros (_ & [Hvars _] & Hbound).
  apply
    (region_step_cases rs
       (fun result =>
          result = Region_step rs' ->
          bound_no_location (static_values (state_statics (rs_state rs')))
            (rs_frames rs'))).
  - discriminate.
  - discriminate.
  - intros x kind d mark_kind body _ Hstep.
    rewrite (begin_region_step _ _ _ _ _ _ Hstep). exact Hbound.
  - intros x kind d ended_by region body _ Hstep.
    destruct (end_region_step _ _ _ _ _ _ _ Hstep)
      as (m & above & f & below & counts & _ & Hframes & ->).
    change (rs_frames (count_uses rs)) with (rs_frames rs) in Hframes.
    intros s allocated w Hin. apply (Hbound s allocated w). rewrite Hframes.
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
    assert (Hbindings :
      ∀ (s : symbol) (allocated : statically_allocated_value) (w : value),
        In (s, allocated) bindings ->
        In w (allocated_values allocated) ->
        no_location w).
    { intros s allocated w Hin Hw.
      destruct (Hmade s allocated Hin) as (_ & _ & Hvalues & _).
      rewrite Hvalues in Hw.
      destruct (Hparts w Hw) as (sim & _ & Heval).
      exact (eval_simple_no_location _ sim w Hvars Heval). }
    revert Hframes Hbound.
    destruct (rs_frames rs) as [ | [m mark_kind locals] frames];
      intros [Hf Hs] Hbound s allocated w Hin; rewrite Hf, Hs in Hin.
    + simpl in Hin. apply in_app_or in Hin as [Hin | Hin];
        [ exact (Hbindings s allocated w Hin)
        | exact (Hbound s allocated w Hin) ].
    + change (live_locals (Frame m mark_kind (bindings ++ locals) :: frames))
        with ((bindings ++ locals) ++ live_locals frames) in Hin.
      rewrite <- !app_assoc in Hin. apply in_app_or in Hin as [Hin | Hin];
        [ exact (Hbindings s allocated w Hin) | ].
      apply (Hbound s allocated w).
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
    unfold bound_no_location. rewrite Hvalues.
    intros s allocated w Hin. apply in_app_or in Hin as [Hin | Hin];
      [ apply (Hbound s allocated w); apply in_or_app; left; exact Hin | ].
    apply in_app_or in Hin as [Hin | Hin];
      [ | apply (Hbound s allocated w); apply in_or_app; right; exact Hin ].
    destruct (Hnew s allocated Hin) as (const & _ & Hconst).
    destruct (eval_static_const_makes _ const allocated Hconst)
      as [Hmakes _].
    intros Hw. destruct (Hmakes w Hw) as (sim & _ & Hsimval).
    exact (eval_simple_no_location _ sim w Hvars Hsimval).
  - intros ac _ Hstep.
    destruct (region_apply_cont_step _ _ ac Hstep) as (Hf & _ & _ & Hs).
    rewrite Hf, Hs. exact Hbound.
  - intros _ _ Hstep.
    destruct (plain_step_step _ _ Hstep) as (s' & _ & ->). exact Hbound.
Qed.

Lemma region_step_keeps_no_heap_primitive (rs rs' : region_state) :
  no_heap_primitive rs ->
  region_step rs = Region_step rs' ->
  no_heap_primitive rs'.
Proof.
  intros Hno Hstep. pose proof Hno as (Hwritten & _ & _).
  split; [ | split ].
  - exact (region_step_keeps_written_no_heap rs rs' Hwritten Hstep).
  - pose proof (region_step_no_location_values rs Hno) as Hvalues.
    rewrite Hstep in Hvalues. apply Hvalues. intros s _ _. reflexivity.
  - exact (region_step_bound_no_location rs rs' Hno Hstep).
Qed.

(* ======================================================================== *)
(* The machine runs every [Let] of such a state                             *)
(* ======================================================================== *)

Lemma no_heap_primitive_step_run (rs : region_state) :
  no_heap_primitive rs -> region_step rs <> Region_let_not_run.
Proof.
  intros ((He & _ & _) & [Hvars _] & _) Hstep.
  destruct (region_step_let_not_run rs Hstep)
    as (x & kind & d & body & Hexpr & Hneeds & _).
  rewrite Hexpr in He. simpl in He.
  apply andb_true_iff in He as [Hd _]. apply negb_true_iff in Hd.
  destruct d as [s | p args | | | | | ]; try discriminate Hneeds.
  destruct p as [u | b | va | hp | rp]; try discriminate Hd;
    simpl in Hneeds;
    (destruct (eval_simples (state_vars (rs_state rs)) args) as [vs | ]
       eqn:Hvs; [ | discriminate Hneeds ]);
    apply existsb_exists in Hneeds as (v & Hin & Hv);
    (assert (Hno : no_location v)
       by exact
            (proj1 (Forall_forall _ _)
               (eval_simples_forall no_location _ args vs
                  (fun sim w _ => eval_simple_no_location _ sim w Hvars) Hvs)
               v Hin));
    destruct v; try discriminate Hv; discriminate Hno.
Qed.

Lemma no_heap_primitive_run (fuel : nat) :
  ∀ rs : region_state,
    no_heap_primitive rs -> region_run fuel rs <> Region_let_not_run.
Proof.
  induction fuel as [ | fuel IH]; intros rs Hno; simpl; [ discriminate | ].
  destruct (region_step rs) as [rs1 | k vs rs1 | fault | ] eqn:Hstep;
    try discriminate.
  - exact (IH rs1 (region_step_keeps_no_heap_primitive rs rs1 Hno Hstep)).
  - intros _. exact (no_heap_primitive_step_run rs Hno Hstep).
Qed.

Lemma no_heap_primitive_initial (e : expr) (statics : static_env)
    (base : symbol) :
  expr_free_of_heap e = true ->
  statics_free_of_heap statics = true ->
  (∀ (s : symbol) (allocated : statically_allocated_value) (v : value),
     (s, allocated) ∈ static_values statics ->
     v ∈ values_of_static allocated ->
     locations_of_value v = []) ->
  no_heap_primitive (initial_region_state e statics base).
Proof.
  intros He Hcode Hvalues. split; [ | split ].
  - split; [ exact He | ].
    split; [ exact I | ].
    exact Hcode.
  - exact (conj (Forall_nil _) I).
  - intros s allocated w Hin Hw. simpl in Hin.
    apply (Hvalues s allocated w Hin).
    destruct allocated; exact Hw.
Qed.

Theorem no_heap_primitive_no_let_not_run :
  no_heap_primitive_no_let_not_run_statement.
Proof.
  intros e statics base fuel He Hcode Hvalues.
  exact
    (no_heap_primitive_run fuel _
       (no_heap_primitive_initial e statics base He Hcode Hvalues)).
Qed.
