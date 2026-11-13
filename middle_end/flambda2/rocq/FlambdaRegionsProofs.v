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

(* Proofs of the statements of FlambdaRegionsSpec.v. Nothing here needs
   review. *)

From Stdlib Require Import ZArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaRegionsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The steps of the machine of FlambdaSemanticsSpec.v                       *)
(* ======================================================================== *)

(* Only a [Let_static] changes the static environment. *)
Theorem step_keeps_statics : step_keeps_statics_statement.
Proof.
  intros s s' Hstep Hnot. revert Hstep.
  destruct s as [e vars conts statics hp]. simpl in Hnot. simpl.
  destruct e
    as [x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ].
  - rewrite step_let_exec.
    destruct (exec_named statics hp vars d) as [[v hp'] | ];
      [ | discriminate ].
    destruct (kind_eqb (kind_of_value v) kind); [ | discriminate ].
    intros [= <-]. reflexivity.
  - contradiction (Hnot bindings body eq_refl).
  - rewrite step_let_cont_non_recursive. intros [= <-]. reflexivity.
  - rewrite step_let_cont_recursive.
    destruct (handlers_bind_statics handlers); [ discriminate | ].
    intros [= <-]. reflexivity.
  - rewrite step_apply.
    destruct (eval_simple vars (ap_callee ap)); [ | discriminate ].
    destruct (eval_simples vars (ap_args ap)); [ | discriminate ].
    destruct (eval_simples vars (exn_extra_simples ap)); [ | discriminate ].
    apply call_function_statics.
  - rewrite step_apply_cont.
    destruct (eval_simples vars (ac_args ac)); [ | discriminate ].
    unfold apply_continuation. apply apply_cont_value_statics.
  - rewrite step_switch.
    destruct (eval_simple vars scrutinee) as [v | ]; [ | discriminate ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | discriminate ].
    destruct (find_arm i arms); [ | discriminate ].
    intros [= <-]. reflexivity.
  - rewrite step_invalid. discriminate.
Qed.

Lemma with_statics_same (s : state) : with_statics s (state_statics s) = s.
Proof. destruct s. reflexivity. Qed.

Lemma with_statics_of (s' : state) (statics : static_env) :
  state_statics s' = statics -> with_statics s' statics = s'.
Proof. intros <-. apply with_statics_same. Qed.

(* ======================================================================== *)
(* With no frame                                                            *)
(* ======================================================================== *)

Lemma live_statics_no_frames (rs : region_state) :
  rs_frames rs = [] -> live_statics rs = state_statics (rs_state rs).
Proof.
  unfold live_statics. intros ->. simpl.
  destruct (state_statics (rs_state rs)). reflexivity.
Qed.

Lemma value_local_no_frames (rs : region_state) (v : value) :
  rs_frames rs = [] -> value_local rs v = false.
Proof. unfold value_local. intros ->. destruct v; reflexivity. Qed.

Lemma simple_local_no_frames (rs : region_state) (s : simple) :
  rs_frames rs = [] -> simple_local rs s = false.
Proof.
  intros Hframes. unfold simple_local.
  destruct (eval_simple (state_vars (rs_state rs)) s);
    [ apply value_local_no_frames; exact Hframes | reflexivity ].
Qed.

Lemma existsb_simple_local_no_frames (rs : region_state) (l : list simple) :
  rs_frames rs = [] -> existsb (simple_local rs) l = false.
Proof.
  intros Hframes. induction l as [ | s l IH]; simpl; [ reflexivity | ].
  rewrite (simple_local_no_frames rs s Hframes). exact IH.
Qed.

Lemma value_open_mark_no_frames (rs : region_state) (v : value) :
  rs_frames rs = [] -> value_open_mark rs v = false.
Proof.
  unfold value_open_mark. intros ->. destruct v; try reflexivity.
  apply andb_false_r.
Qed.

Lemma existsb_simple_open_mark_no_frames (rs : region_state)
    (l : list simple) :
  rs_frames rs = [] -> existsb (simple_open_mark rs) l = false.
Proof.
  intros Hframes. induction l as [ | s l IH]; simpl; [ reflexivity | ].
  rewrite IH, orb_false_r. unfold simple_open_mark.
  destruct (eval_simple (state_vars (rs_state rs)) s);
    [ apply value_open_mark_no_frames; exact Hframes | reflexivity ].
Qed.

(* The test of the simples passes where each may be evaluated. *)
Lemma simples_fault_usable (rs : region_state) (simples : list simple) :
  ∀ places : list bool,
    forallb (simple_usable rs) simples = true ->
    simples_fault rs simples places = None.
Proof.
  induction simples as [ | s simples IH]; intros places; simpl;
    [ reflexivity | ].
  intros Husable. apply andb_true_iff in Husable as [Hs Husable].
  rewrite Hs. apply IH. exact Husable.
Qed.

Lemma plain_step_no_frames (rs : region_state) :
  rs_frames rs = [] ->
  (∀ (bindings : list static_binding) (body : expr),
    state_expr (rs_state rs) <> Let_static bindings body) ->
  plain_step rs = of_plain_result rs (rs_counts rs) (step (rs_state rs)).
Proof.
  intros Hframes Hnot. unfold plain_step.
  rewrite (live_statics_no_frames rs Hframes), with_statics_same.
  destruct (step (rs_state rs)) as [s' | k vs hp | ] eqn:Hstep; simpl.
  - rewrite (with_statics_of s' _ (step_keeps_statics _ _ Hstep Hnot)).
    reflexivity.
  - destruct rs. reflexivity.
  - reflexivity.
Qed.

Theorem region_step_plain : region_step_plain_statement.
Proof.
  intros rs Hframes Hhead Husable.
  unfold region_step. rewrite (simples_fault_usable _ _ _ Husable). cbv zeta.
  set (rs1 := count_uses rs).
  assert (Hframes1 : rs_frames rs1 = []) by exact Hframes.
  assert (Hplain :
            (∀ (bindings : list static_binding) (body : expr),
              state_expr (rs_state rs) <> Let_static bindings body) ->
            plain_step rs1
            = of_plain_result rs (rs_counts rs1) (step (rs_state rs))).
  { intros Hnot. exact (plain_step_no_frames rs1 Hframes1 Hnot). }
  assert (Hlocal :
            ∀ l : list simple, existsb (simple_local rs1) l = false).
  { intros l. exact (existsb_simple_local_no_frames rs1 l Hframes1). }
  destruct (state_expr (rs_state rs))
    as [x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ] eqn:He;
    try (apply Hplain; intros bindings0 body0; try rewrite He; discriminate).
  - assert (Hrest :
              match marker_of x body with
              | Marker y region body' =>
                local_allocation rs1 x y kind d region body'
              | Bad_marker => Region_undefined Not_of_the_conversion
              | No_marker =>
                if is_allocation d
                   && existsb (simple_local rs1) (simples_of_named d)
                then Region_undefined Heap_holds_local
                else plain_step rs1
              end
              = of_plain_result rs (rs_counts rs1) (step (rs_state rs))).
    { assert (Hmarker : marker_of x body = No_marker).
      { destruct d as [s | [u | b | v | hp | rp] args | | | | | ];
          simpl in Hhead;
          try discriminate Hhead;
          destruct (marker_of x body); try discriminate Hhead; reflexivity. }
      rewrite Hmarker, Hlocal, andb_false_r.
      apply Hplain. intros bindings0 body0. try rewrite He. discriminate. }
    destruct d as [s | [u | b | v | hp | rp] args | | | | | ];
      try exact Hrest.
    discriminate Hhead.
  - rewrite Hlocal. unfold static_step.
    change (rs_state rs1) with (rs_state rs).
    destruct (step (rs_state rs)); reflexivity.
  - rewrite (existsb_simple_open_mark_no_frames rs1 (ac_args ac) Hframes1).
    apply Hplain. intros bindings0 body0. try rewrite He. discriminate.
Qed.

(* ======================================================================== *)
(* The shape of a step                                                      *)
(* ======================================================================== *)

(* What a step does to the frames, to the counter and to the static
   environment of the state: nothing; or the static environment grows by
   a group of static bindings; or a frame is pushed; or frames are
   popped; or symbols are bound, in the newest frame or, with no frame,
   in the static environment. *)
Inductive step_shape (rs rs' : region_state) : Prop :=
  | Shape_plain :
      rs_frames rs' = rs_frames rs ->
      rs_next rs' = rs_next rs ->
      state_statics (rs_state rs') = state_statics (rs_state rs) ->
      ends_region (state_expr (rs_state rs)) = false ->
      step_shape rs rs'
  | Shape_static (bindings : list static_binding) (body : expr) :
      state_expr (rs_state rs) = Let_static bindings body ->
      rs_frames rs' = rs_frames rs ->
      rs_next rs' = rs_next rs ->
      eval_static_bindings (state_vars (rs_state rs)) bindings
        (state_statics (rs_state rs))
      = Some (state_statics (rs_state rs')) ->
      step_shape rs rs'
  | Shape_begin (kind : mark_kind) :
      rs_frames rs' = Frame (rs_next rs) kind [] :: rs_frames rs ->
      rs_next rs' = Pos.succ (rs_next rs) ->
      state_statics (rs_state rs') = state_statics (rs_state rs) ->
      ends_region (state_expr (rs_state rs)) = false ->
      step_shape rs rs'
  | Shape_end (above : list frame) (f : frame) :
      rs_frames rs = above ++ f :: rs_frames rs' ->
      rs_next rs' = rs_next rs ->
      state_statics (rs_state rs') = state_statics (rs_state rs) ->
      ends_region (state_expr (rs_state rs)) = true ->
      step_shape rs rs'
  | Shape_local (v : value) (sym : symbol)
      (bindings : list (symbol * statically_allocated_value)) :
      localise (rs_next rs) v = Some (sym, bindings) ->
      rs_next rs' = advance (rs_next rs) (length bindings) ->
      match rs_frames rs with
      | f :: frames =>
        rs_frames rs'
        = Frame (frame_mark f) (frame_kind f) (bindings ++ frame_locals f)
          :: frames
        ∧ state_statics (rs_state rs') = state_statics (rs_state rs)
      | [] =>
        rs_frames rs' = []
        ∧ state_statics (rs_state rs')
          = Static_env (static_code (state_statics (rs_state rs)))
              (bindings ++ static_values (state_statics (rs_state rs)))
      end ->
      ends_region (state_expr (rs_state rs)) = false ->
      step_shape rs rs'.

Lemma cut_app (m : mark) (frames above below : list frame) :
  cut m frames = Some (above, below) ->
  ∃ f : frame, frames = above ++ f :: below.
Proof.
  revert above below.
  induction frames as [ | f frames IH]; intros above below; simpl;
    [ discriminate | ].
  destruct (Pos.eqb (frame_mark f) m).
  - intros [= <- <-]. exists f. reflexivity.
  - destruct (cut m frames) as [[above' below'] | ]; simpl;
      [ | discriminate ].
    intros [= <- <-]. destruct (IH _ _ eq_refl) as (f' & ->).
    exists f'. reflexivity.
Qed.

Lemma plain_step_shape (rs rs' : region_state) :
  (∀ (bindings : list static_binding) (body : expr),
    state_expr (rs_state rs) <> Let_static bindings body) ->
  ends_region (state_expr (rs_state rs)) = false ->
  plain_step rs = Region_step rs' ->
  rs_base rs' = rs_base rs ∧ step_shape rs rs'.
Proof.
  intros Hnot Hends. unfold plain_step.
  destruct (step (with_statics (rs_state rs) (live_statics rs)))
    as [s' | k vs hp | ]; [ | discriminate | discriminate ].
  intros [= <-]. split; [ reflexivity | ].
  apply Shape_plain; try reflexivity. exact Hends.
Qed.

Lemma static_step_shape (rs rs' : region_state)
    (bindings : list static_binding) (body : expr) :
  state_expr (rs_state rs) = Let_static bindings body ->
  static_step rs = Region_step rs' ->
  rs_base rs' = rs_base rs ∧ step_shape rs rs'.
Proof.
  intros He. unfold static_step.
  destruct (step (rs_state rs)) as [s' | k vs hp | ] eqn:Hstep;
    [ | discriminate | discriminate ].
  intros [= <-]. split; [ reflexivity | ].
  apply (Shape_static _ _ bindings body); try reflexivity; [ exact He | ].
  simpl. destruct (rs_state rs) as [e vars conts statics hp].
  simpl in He. subst e. rewrite step_let_static in Hstep. simpl.
  destruct (eval_static_bindings vars bindings statics) as [statics' | ];
    [ | discriminate ].
  injection Hstep as <-. reflexivity.
Qed.

Lemma begin_region_shape (rs rs' : region_state) (x : variable)
    (kind : flambda_kind) (mark_kind : mark_kind) (body : expr) :
  ends_region (state_expr (rs_state rs)) = false ->
  begin_region rs x kind mark_kind body = Region_step rs' ->
  rs_base rs' = rs_base rs ∧ step_shape rs rs'.
Proof.
  intros Hends. unfold begin_region.
  destruct (kind_eqb kind Region); [ | discriminate ].
  intros [= <-]. split; [ reflexivity | ].
  apply (Shape_begin _ _ mark_kind); try reflexivity. exact Hends.
Qed.

Lemma end_region_shape (rs rs' : region_state) (x : variable)
    (kind : flambda_kind) (ended_by : mark_kind) (region : simple)
    (body : expr) :
  ends_region (state_expr (rs_state rs)) = true ->
  end_region rs x kind ended_by region body = Region_step rs' ->
  rs_base rs' = rs_base rs ∧ step_shape rs rs'.
Proof.
  intros Hends. unfold end_region.
  destruct (kind_eqb kind Value); [ | discriminate ].
  destruct (eval_simple (state_vars (rs_state rs)) region) as [v | ];
    [ | discriminate ].
  destruct v as [ | | | m | | | | | ]; try discriminate.
  destruct (cut m (rs_frames rs)) as [[above below] | ] eqn:Hcut;
    [ | discriminate ].
  destruct (cut_app _ _ _ _ Hcut) as (f & Hframes).
  intros [= <-]. split; [ reflexivity | ].
  apply (Shape_end _ _ above f); try reflexivity; [ exact Hframes | ].
  exact Hends.
Qed.

Lemma local_allocation_shape (rs rs' : region_state) (x y : variable)
    (kind : flambda_kind) (d : named) (region : option simple)
    (body : expr) :
  ends_region (state_expr (rs_state rs)) = false ->
  local_allocation rs x y kind d region body = Region_step rs' ->
  rs_base rs' = rs_base rs ∧ step_shape rs rs'.
Proof.
  intros Hends. unfold local_allocation.
  destruct (is_allocation d && kind_eqb kind Value); [ | discriminate ].
  destruct (named_region rs region) as [other | ]; [ | discriminate ].
  destruct (eval_named (live_statics rs) (state_vars (rs_state rs)) d)
    as [v | ]; [ | discriminate ].
  destruct (localise (rs_next rs) v) as [[sym bindings] | ] eqn:Hlocalise;
    [ | discriminate ].
  destruct (rs_frames rs) as [ | [m mark_kind locals] frames] eqn:Hframes;
    intros [= <-]; (split; [ reflexivity | ]);
    apply (Shape_local _ _ v sym bindings);
    try reflexivity; try exact Hlocalise; try exact Hends;
    rewrite Hframes; split; reflexivity.
Qed.

Lemma region_apply_cont_shape (rs rs' : region_state) (ac : apply_cont) :
  (∀ (bindings : list static_binding) (body : expr),
    state_expr (rs_state rs) <> Let_static bindings body) ->
  ends_region (state_expr (rs_state rs)) = false ->
  region_apply_cont rs ac = Region_step rs' ->
  rs_base rs' = rs_base rs ∧ step_shape rs rs'.
Proof.
  intros Hnot Hends. unfold region_apply_cont.
  destruct (resolve (ac_continuation ac) (state_conts (rs_state rs)))
    as [[params metadata handler] def_vars def_conts recursive | k | c extra
       | remaining remaining_arity return_arity return_to exn_to];
    try (apply plain_step_shape; assumption).
  destruct (eval_simples (state_vars (rs_state rs)) (ac_args ac)) as [vs | ];
    [ | discriminate ].
  destruct
    (kinds_eqb (map (region_kind_of_value rs) vs) (hm_params_kinds metadata));
    [ | discriminate ].
  destruct (bind_params params vs def_vars) as [handler_vars | ];
    [ | discriminate ].
  intros [= <-]. split; [ reflexivity | ].
  apply Shape_plain; try reflexivity. exact Hends.
Qed.

(* The components that [count_uses] leaves. *)
Lemma count_uses_shape (rs rs' : region_state) :
  rs_base rs' = rs_base (count_uses rs) ∧ step_shape (count_uses rs) rs' ->
  rs_base rs' = rs_base rs ∧ step_shape rs rs'.
Proof.
  intros [Hbase Hshape]. split; [ exact Hbase | ].
  destruct Hshape
    as [Hframes Hnext Hstatics Hends
       | bindings body He Hframes Hnext Hstatics
       | kind Hframes Hnext Hstatics Hends
       | above f Hframes Hnext Hstatics Hends
       | v sym bindings Hlocalise Hnext Hframes Hends];
    cbn [count_uses with_counts rs_state rs_frames rs_base rs_next
         rs_counts] in *.
  - exact (Shape_plain rs rs' Hframes Hnext Hstatics Hends).
  - exact (Shape_static rs rs' bindings body He Hframes Hnext Hstatics).
  - exact (Shape_begin rs rs' kind Hframes Hnext Hstatics Hends).
  - exact (Shape_end rs rs' above f Hframes Hnext Hstatics Hends).
  - exact (Shape_local rs rs' v sym bindings Hlocalise Hnext Hframes Hends).
Qed.

Lemma region_step_shape (rs rs' : region_state) :
  region_step rs = Region_step rs' ->
  rs_base rs' = rs_base rs ∧ step_shape rs rs'.
Proof.
  unfold region_step. cbv zeta.
  destruct
    (simples_fault rs (evaluated_simples (state_expr (rs_state rs)))
       (region_places rs));
    [ discriminate | ].
  intros Hstep. apply count_uses_shape. revert Hstep.
  set (rs1 := count_uses rs).
  change (state_expr (rs_state rs)) with (state_expr (rs_state rs1)).
  destruct (state_expr (rs_state rs1))
    as [x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ] eqn:He;
    try (apply plain_step_shape;
         [ intros bindings0 body0; try rewrite He; discriminate
         | rewrite He; reflexivity ]).
  - assert (Hrest :
              ends_region (Let x kind d body) = false ->
              match marker_of x body with
              | Marker y region body' =>
                local_allocation rs1 x y kind d region body'
              | Bad_marker => Region_undefined Not_of_the_conversion
              | No_marker =>
                if is_allocation d
                   && existsb (simple_local rs1) (simples_of_named d)
                then Region_undefined Heap_holds_local
                else plain_step rs1
              end
              = Region_step rs' ->
              rs_base rs' = rs_base rs1 ∧ step_shape rs1 rs').
    { intros Hends.
      destruct (marker_of x body) as [ | y region body' | ];
        [ | apply local_allocation_shape; rewrite He; exact Hends
          | discriminate ].
      destruct
        (is_allocation d && existsb (simple_local rs1) (simples_of_named d));
        [ discriminate | ].
      apply plain_step_shape;
        [ intros bindings0 body0; try rewrite He; discriminate
        | rewrite He; exact Hends ]. }
    destruct d as [s | [u | b | v | hp | rp] args | | | | | ];
      try (apply Hrest; reflexivity).
    destruct rp, args as [ | arg [ | arg' args]]; try discriminate.
    + apply begin_region_shape. rewrite He. reflexivity.
    + apply begin_region_shape. rewrite He. reflexivity.
    + apply end_region_shape. rewrite He. reflexivity.
    + apply end_region_shape. rewrite He. reflexivity.
  - destruct
      (existsb (simple_local rs1)
         (evaluated_simples (Let_static bindings body)));
      [ discriminate | ].
    apply (static_step_shape _ _ bindings body). exact He.
  - destruct (existsb (simple_open_mark rs1) (ac_args ac)).
    + apply region_apply_cont_shape;
        [ intros bindings0 body0; try rewrite He; discriminate
        | rewrite He; reflexivity ].
    + apply plain_step_shape;
        [ intros bindings0 body0; try rewrite He; discriminate
        | rewrite He; reflexivity ].
Qed.

(* ======================================================================== *)
(* The marks                                                                *)
(* ======================================================================== *)

Theorem region_step_keeps_marks : region_step_keeps_marks_statement.
Proof.
  intros rs rs' m Hstep Hnot_end Hon.
  destruct (region_step_shape _ _ Hstep) as [_ Hshape].
  destruct Hshape
    as [Hframes _ _ _
       | bindings body _ Hframes _ _
       | kind Hframes _ _ _
       | above f _ _ _ Hends
       | v sym bindings _ _ Hframes _].
  - rewrite Hframes. exact Hon.
  - rewrite Hframes. exact Hon.
  - rewrite Hframes. simpl. rewrite Hon. apply orb_true_r.
  - rewrite Hends in Hnot_end. discriminate.
  - destruct (rs_frames rs) as [ | f frames].
    + discriminate Hon.
    + destruct Hframes as [-> _]. exact Hon.
Qed.

Theorem region_step_end_pops : region_step_end_pops_statement.
Proof.
  intros rs rs' Hstep Hends.
  destruct (region_step_shape _ _ Hstep) as [_ Hshape].
  destruct Hshape
    as [_ _ _ Hnot
       | bindings body He _ _ _
       | kind _ _ _ Hnot
       | above f Hframes _ _ _
       | v sym bindings _ _ _ Hnot];
    try (rewrite Hnot in Hends; discriminate).
  - rewrite He in Hends. discriminate.
  - exists (above ++ [f]). rewrite <- app_assoc. exact Hframes.
Qed.

(* ======================================================================== *)
(* The symbols drawn                                                        *)
(* ======================================================================== *)

Lemma advance_le (next : symbol) (count : nat) :
  (next <= advance next count)%positive.
Proof.
  unfold advance. induction count as [ | count IH]; simpl; lia.
Qed.

Theorem region_step_draws_upwards : region_step_draws_upwards_statement.
Proof.
  intros rs rs' Hstep.
  destruct (region_step_shape _ _ Hstep) as [Hbase Hshape].
  split; [ exact Hbase | ]. apply Pos.leb_le.
  destruct Hshape
    as [_ Hnext _ _
       | bindings body _ _ Hnext _
       | kind _ Hnext _ _
       | above f _ Hnext _ _
       | v sym bindings _ Hnext _ _];
    rewrite Hnext; try lia.
  apply advance_le.
Qed.

(* The symbols that [number_slots] gives are [next] or above. *)
Lemma number_slots_from (function_decls : list (function_slot * code_id)) :
  ∀ (next : symbol) (slot : function_slot) (s : symbol),
    In (slot, s) (number_slots function_decls next) -> (next <= s)%positive.
Proof.
  induction function_decls as [ | [slot0 code] function_decls IH];
    intros next slot s; simpl; [ contradiction | ].
  intros [[= _ <-] | Hin]; [ lia | ].
  specialize (IH _ _ _ Hin). lia.
Qed.

(* The symbols that [localise] binds are [next] or above. *)
Lemma localise_from (next : symbol) (v : value) (sym : symbol)
    (bindings : list (symbol * statically_allocated_value)) :
  localise next v = Some (sym, bindings) ->
  ∀ (s : symbol) (allocated : statically_allocated_value),
    In (s, allocated) bindings -> (next <= s)%positive.
Proof.
  destruct v as [c | code value_slots | slot function_decls value_slots | s0
                | tag fields | elements | c | l | callee args arity];
    simpl; try discriminate;
    try (intros [= _ <-] s allocated [[= <- _] | []]; lia).
  destruct (lookup slot (number_slots function_decls next)) as [s1 | ];
    simpl; [ | discriminate ].
  intros [= _ <-] s allocated Hin.
  apply in_map_iff in Hin as ([slot' s'] & [= <- _] & Hin).
  exact (number_slots_from _ _ _ _ Hin).
Qed.

Lemma lookup_not_in {A : Type} (name : positive)
    (bindings : list (positive * A)) :
  (∀ a : A, ~ In (name, a) bindings) -> lookup name bindings = None.
Proof.
  intros Hnot. destruct (lookup name bindings) as [a | ] eqn:Hlookup;
    [ | reflexivity ].
  contradiction (Hnot a (lookup_in _ _ _ Hlookup)).
Qed.

Lemma localise_lookup_below (next : symbol) (v : value) (sym : symbol)
    (bindings : list (symbol * statically_allocated_value))
    (s : symbol) :
  localise next v = Some (sym, bindings) ->
  (s < next)%positive ->
  lookup s bindings = None.
Proof.
  intros Hlocalise Hbelow. apply lookup_not_in. intros allocated Hin.
  pose proof (localise_from _ _ _ _ Hlocalise _ _ Hin). lia.
Qed.

(* The symbols that a group of static bindings binds. *)
Lemma eval_static_bindings_lookup (vars : env)
    (bindings : list static_binding) :
  ∀ (statics statics' : static_env) (base s : symbol),
    forallb (binding_below base) bindings = true ->
    (base <= s)%positive ->
    eval_static_bindings vars bindings statics = Some statics' ->
    lookup s (static_values statics') = lookup s (static_values statics).
Proof.
  induction bindings as [ | binding bindings IH];
    intros statics statics' base s Hbelow Hs; simpl.
  - intros [= <-]. reflexivity.
  - simpl in Hbelow. apply andb_true_iff in Hbelow as [Hbinding Hbelow].
    destruct binding as [id c | id | s0 const].
    + destruct (lookup id (static_code statics)); [ discriminate | ].
      intros Heval. rewrite (IH _ _ _ _ Hbelow Hs Heval). reflexivity.
    + apply (IH _ _ _ _ Hbelow Hs).
    + destruct (lookup s0 (static_values statics)); [ discriminate | ].
      destruct (eval_static_const vars const) as [allocated | ]; simpl;
        [ | discriminate ].
      intros Heval. rewrite (IH _ _ _ _ Hbelow Hs Heval). simpl.
      simpl in Hbinding. apply andb_true_iff in Hbinding as [Hs0 _].
      unfold symbol_below in Hs0. apply Pos.ltb_lt in Hs0.
      destruct (Pos.eqb s s0) eqn:Heq; [ | reflexivity ].
      apply Pos.eqb_eq in Heq. lia.
Qed.

Lemma on_stack_app (m : mark) (frames frames' : list frame) :
  on_stack m (frames ++ frames')
  = on_stack m frames || on_stack m frames'.
Proof. unfold on_stack. apply existsb_app. Qed.

Lemma live_locals_app (frames frames' : list frame) :
  live_locals (frames ++ frames')
  = live_locals frames ++ live_locals frames'.
Proof. unfold live_locals. apply flat_map_app. Qed.

Lemma lookup_app_none {A : Type} (name : positive)
    (bindings bindings' : list (positive * A)) :
  lookup name (bindings ++ bindings') = None
  <-> lookup name bindings = None ∧ lookup name bindings' = None.
Proof.
  rewrite lookup_app. destruct (lookup name bindings); split.
  - discriminate.
  - intros [H _]. discriminate H.
  - intros H. split; [ reflexivity | exact H ].
  - intros [_ H]. exact H.
Qed.

Theorem gone_stays_gone : gone_stays_gone_statement.
Proof.
  intros rs rs' s Hstep Hfresh (Hdrawn & Hnext & Hstack & Hlookup).
  destruct (region_step_shape _ _ Hstep) as [Hbase Hshape].
  unfold gone, drawn. rewrite Hbase. split; [ exact Hdrawn | ].
  apply Pos.ltb_lt in Hnext.
  unfold drawn in Hdrawn. apply Pos.leb_le in Hdrawn.
  unfold live_statics in Hlookup |- *. simpl in Hlookup |- *.
  apply lookup_app_none in Hlookup as [Hlocals Hstatics].
  destruct Hshape
    as [Hframes Hnext' Hstatics' _
       | bindings body He Hframes Hnext' Hstatics'
       | kind Hframes Hnext' Hstatics' _
       | above f Hframes Hnext' Hstatics' _
       | v sym bindings Hlocalise Hnext' Hframes _].
  - rewrite Hframes, Hnext', Hstatics'.
    split; [ apply Pos.ltb_lt; exact Hnext | ].
    split; [ exact Hstack | ].
    apply lookup_app_none. split; assumption.
  - rewrite Hframes, Hnext'.
    split; [ apply Pos.ltb_lt; exact Hnext | ].
    split; [ exact Hstack | ].
    apply lookup_app_none. split; [ exact Hlocals | ].
    unfold region_names_fresh in Hfresh.
    apply andb_true_iff in Hfresh as [Hexpr _].
    rewrite He in Hexpr. simpl in Hexpr.
    apply andb_true_iff in Hexpr as [Hbindings _].
    rewrite
      (eval_static_bindings_lookup _ _ _ _ _ s Hbindings Hdrawn Hstatics').
    exact Hstatics.
  - rewrite Hframes, Hnext', Hstatics'.
    split; [ apply Pos.ltb_lt; lia | ].
    split.
    + simpl. rewrite Hstack, orb_false_r. apply Pos.eqb_neq. lia.
    + simpl. apply lookup_app_none. split; assumption.
  - rewrite Hnext', Hstatics'. rewrite Hframes in Hstack, Hlocals.
    split; [ apply Pos.ltb_lt; exact Hnext | ].
    rewrite on_stack_app in Hstack.
    apply orb_false_iff in Hstack as [_ Hstack]. simpl in Hstack.
    apply orb_false_iff in Hstack as [_ Hstack].
    split; [ exact Hstack | ].
    rewrite live_locals_app in Hlocals.
    apply lookup_app_none in Hlocals as [_ Hlocals]. simpl in Hlocals.
    apply lookup_app_none in Hlocals as [_ Hlocals].
    apply lookup_app_none. split; assumption.
  - rewrite Hnext'.
    pose proof (advance_le (rs_next rs) (length bindings)) as Hadvance.
    split; [ apply Pos.ltb_lt; lia | ].
    pose proof (localise_lookup_below _ _ _ _ s Hlocalise Hnext)
      as Hbindings.
    destruct (rs_frames rs) as [ | f frames].
    + destruct Hframes as [-> ->]. split; [ reflexivity | ].
      simpl. apply lookup_app_none. split; assumption.
    + destruct Hframes as [-> ->]. split; [ exact Hstack | ].
      simpl in Hlocals |- *.
      apply lookup_app_none in Hlocals as [Hf Hlocals].
      apply lookup_app_none. split; [ | exact Hstatics ].
      rewrite <- app_assoc.
      apply lookup_app_none. split; [ exact Hbindings | ].
      apply lookup_app_none. split; assumption.
Qed.

(* ======================================================================== *)
(* A region given to a continuation                                         *)
(* ======================================================================== *)

(* Where no argument is a mark on the stack, the kinds of the values are
   those of FlambdaValueSpec.kind_of_value. *)
Lemma region_kinds_plain (rs : region_state) (args : list simple) :
  ∀ vs : list value,
    eval_simples (state_vars (rs_state rs)) args = Some vs ->
    existsb (simple_open_mark rs) args = false ->
    map (region_kind_of_value rs) vs = map kind_of_value vs.
Proof.
  induction args as [ | s args IH]; intros vs; simpl.
  - intros [= <-] _. reflexivity.
  - unfold simple_open_mark at 1.
    destruct (eval_simple (state_vars (rs_state rs)) s) as [v | ]; simpl;
      [ | discriminate ].
    destruct (eval_simples (state_vars (rs_state rs)) args) as [vs0 | ];
      simpl; [ | discriminate ].
    intros [= <-] Hmarks. apply orb_false_iff in Hmarks as [Hv Hmarks].
    simpl. unfold region_kind_of_value at 1. rewrite Hv. f_equal.
    apply IH; [ reflexivity | exact Hmarks ].
Qed.

Theorem region_apply_cont_plain : region_apply_cont_plain_statement.
Proof.
  intros rs ac He Hmarks. unfold region_apply_cont.
  destruct (resolve (ac_continuation ac) (state_conts (rs_state rs)))
    as [[params metadata handler] def_vars def_conts recursive | k | c extra
       | remaining remaining_arity return_arity return_to exn_to]
    eqn:Hresolve;
    try reflexivity.
  unfold plain_step.
  destruct (eval_simples (state_vars (rs_state rs)) (ac_args ac)) as [vs | ]
    eqn:Hargs.
  - rewrite (region_kinds_plain _ _ _ Hargs Hmarks).
    destruct rs as [[e vars conts statics hp] frames base next counts].
    cbn [rs_state rs_frames rs_base rs_next rs_counts state_expr state_vars
         state_conts state_statics state_heap] in *.
    subst e. unfold with_statics.
    cbn [state_expr state_vars state_conts state_statics state_heap].
    rewrite step_apply_cont, Hargs. unfold apply_continuation.
    rewrite Hresolve. cbn [apply_cont_value].
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
      [ | reflexivity ].
    destruct (bind_params params vs def_vars) as [handler_vars | ];
      reflexivity.
  - destruct rs as [[e vars conts statics hp] frames base next counts].
    cbn [rs_state rs_frames rs_base rs_next rs_counts state_expr state_vars
         state_conts state_statics state_heap] in *.
    subst e. unfold with_statics.
    cbn [state_expr state_vars state_conts state_statics state_heap].
    rewrite step_apply_cont, Hargs. reflexivity.
Qed.
