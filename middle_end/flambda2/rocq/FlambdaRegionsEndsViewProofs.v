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

(* Lemmas for the proofs of the statements of FlambdaRegionsEndsSpec.v:
   a step of the machine of regions, taken apart once
   ([region_step_view]). What a step does is said by what it leaves:
   the expression, the variables, the continuations, the frames, the
   counter, and whether it counts a plain frame cut by the end of
   another region. The other files of these proofs read a step through
   this lemma only, and not through the definition of
   FlambdaRegionsSpec.region_step. Nothing here needs review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaPrimitiveModel
  FlambdaModel FlambdaMapSpec FlambdaValueSpec FlambdaSemanticsSpec
  FlambdaRegionsSpec FlambdaRegionsProofs FlambdaRegionsStepsProofs
  FlambdaRegionsCallsSpec FlambdaRegionsCallsProofs
  FlambdaRegionsCheckedViewsProofs.
Import ListNotations.

(* The kind of the frame that a primitive pushes. *)
Definition begin_mark_kind (p : region_primitive) : option mark_kind :=
  match p with
  | Begin_region => Some Plain_mark
  | Begin_try_region => Some Try_mark
  | End_region | End_try_region | Local_allocation => None
  end.

Definition is_end (p : region_primitive) : bool :=
  match p with
  | End_region | End_try_region => true
  | Begin_region | Begin_try_region | Local_allocation => false
  end.

(* The step counts no plain frame cut by the end of another region. *)
Definition cuts_kept (rs rs' : region_state) : Prop :=
  count_cut_by_end_region (rs_counts rs')
  = count_cut_by_end_region (rs_counts rs)
  ∧ count_cut_by_end_try_region (rs_counts rs')
    = count_cut_by_end_try_region (rs_counts rs).

(* What a step is. A frame is pushed; the stack is cut at the frame of
   a mark; a local allocation, of a value or of a cell, which leaves
   the marks and the kinds of the frames; a [Let_static]; an
   [Apply_cont] to a handler, which binds its parameters; any other
   step is the step of [step] under a static environment that has the
   code of the state's, and leaves the frames and the counter. *)
Inductive step_view (rs : region_state) : region_step_result -> Prop :=
  | View_undefined (fault : region_fault) :
    step_view rs (Region_undefined fault)
  | View_begin (x : variable) (kind : flambda_kind) (p : region_primitive)
      (mark_kind : mark_kind) (body : expr) (rs' : region_state) :
    state_expr (rs_state rs) = Let x kind (Prim (Region_primitive p) []) body ->
    begin_mark_kind p = Some mark_kind ->
    state_expr (rs_state rs') = body ->
    state_vars (rs_state rs')
    = (x, Value_symbol (rs_next rs)) :: state_vars (rs_state rs) ->
    state_conts (rs_state rs') = state_conts (rs_state rs) ->
    state_statics (rs_state rs') = state_statics (rs_state rs) ->
    rs_frames rs' = Frame (rs_next rs) mark_kind [] :: rs_frames rs ->
    rs_next rs' = Pos.succ (rs_next rs) ->
    cuts_kept rs rs' ->
    step_view rs (Region_step rs')
  | View_end (x : variable) (kind : flambda_kind) (p : region_primitive)
      (region : simple) (body : expr) (m : mark) (above : list frame)
      (f : frame) (below : list frame) (rs' : region_state) :
    state_expr (rs_state rs)
    = Let x kind (Prim (Region_primitive p) [region]) body ->
    is_end p = true ->
    eval_simple (state_vars (rs_state rs)) region = Some (Value_symbol m) ->
    rs_frames rs = above ++ f :: below ->
    frame_mark f = m ->
    state_expr (rs_state rs') = body ->
    state_vars (rs_state rs') = (x, unit_value) :: state_vars (rs_state rs) ->
    state_conts (rs_state rs') = state_conts (rs_state rs) ->
    state_statics (rs_state rs') = state_statics (rs_state rs) ->
    rs_frames rs' = below ->
    rs_next rs' = rs_next rs ->
    (filter is_plain above = [] -> cuts_kept rs rs') ->
    step_view rs (Region_step rs')
  | View_local (x : variable) (kind : flambda_kind) (d : named) (y : variable)
      (kind' : flambda_kind) (args : list simple) (body' : expr) (v : value)
      (rs' : region_state) :
    state_expr (rs_state rs)
    = Let x kind d
        (Let y kind' (Prim (Region_primitive Local_allocation) args) body') ->
    region_named d = false ->
    state_expr (rs_state rs') = body' ->
    state_vars (rs_state rs') = (y, v) :: (x, v) :: state_vars (rs_state rs) ->
    state_conts (rs_state rs') = state_conts (rs_state rs) ->
    static_code (state_statics (rs_state rs'))
    = static_code (state_statics (rs_state rs)) ->
    map (fun f => (frame_mark f, frame_kind f)) (rs_frames rs')
    = map (fun f => (frame_mark f, frame_kind f)) (rs_frames rs) ->
    (rs_next rs <= rs_next rs')%positive ->
    cuts_kept rs rs' ->
    step_view rs (Region_step rs')
  | View_static (bindings : list static_binding) (body : expr)
      (statics' : static_env) (rs' : region_state) :
    state_expr (rs_state rs) = Let_static bindings body ->
    eval_static_bindings (state_vars (rs_state rs)) bindings
      (state_statics (rs_state rs))
    = Some statics' ->
    state_expr (rs_state rs') = body ->
    state_vars (rs_state rs') = state_vars (rs_state rs) ->
    state_conts (rs_state rs') = state_conts (rs_state rs) ->
    state_statics (rs_state rs') = statics' ->
    rs_frames rs' = rs_frames rs ->
    rs_next rs' = rs_next rs ->
    cuts_kept rs rs' ->
    step_view rs (Region_step rs')
  | View_jump (ac : apply_cont) (params : list variable)
      (metadata : handler_metadata) (handler : expr) (def_vars : env)
      (def_conts : cont_env)
      (recursive : list (continuation * continuation_handler))
      (vs : list value) (handler_vars : env) (rs' : region_state) :
    state_expr (rs_state rs) = Apply_cont ac ->
    resolve (ac_continuation ac) (state_conts (rs_state rs))
    = Closure (Handler params metadata handler) def_vars def_conts recursive ->
    eval_simples (state_vars (rs_state rs)) (ac_args ac) = Some vs ->
    bind_params params vs def_vars = Some handler_vars ->
    state_expr (rs_state rs') = handler ->
    state_vars (rs_state rs') = handler_vars ->
    state_conts (rs_state rs')
    = rec_closures recursive def_vars def_conts ++ def_conts ->
    state_statics (rs_state rs') = state_statics (rs_state rs) ->
    rs_frames rs' = rs_frames rs ->
    rs_next rs' = rs_next rs ->
    cuts_kept rs rs' ->
    step_view rs (Region_step rs')
  | View_step (live : static_env) (s' : state) (rs' : region_state) :
    (∀ (x : variable) (kind : flambda_kind) (p : region_primitive)
       (args : list simple) (body : expr),
       state_expr (rs_state rs)
       <> Let x kind (Prim (Region_primitive p) args) body) ->
    (∀ (bindings : list static_binding) (body : expr),
       state_expr (rs_state rs) <> Let_static bindings body) ->
    static_code live = static_code (state_statics (rs_state rs)) ->
    step (with_statics (rs_state rs) live) = Step s' ->
    rs_state rs' = with_statics s' (state_statics (rs_state rs)) ->
    rs_frames rs' = rs_frames rs ->
    rs_next rs' = rs_next rs ->
    cuts_kept rs rs' ->
    step_view rs (Region_step rs')
  | View_exit (live : static_env) (k : continuation) (vs : list value)
      (hp : heap) (rd : read_record) (rs' : region_state) :
    static_code live = static_code (state_statics (rs_state rs)) ->
    step (with_statics (rs_state rs) live) = Exit k vs hp rd ->
    rs_frames rs' = rs_frames rs ->
    step_view rs (Region_exit k vs rs').

Section View.

Variable rs : region_state.

Lemma cuts_kept_count_uses : cuts_kept rs (count_uses rs).
Proof. split; reflexivity. Qed.

Lemma view_plain :
  (∀ (x : variable) (kind : flambda_kind) (p : region_primitive)
     (args : list simple) (body : expr),
     state_expr (rs_state rs)
     <> Let x kind (Prim (Region_primitive p) args) body) ->
  (∀ (bindings : list static_binding) (body : expr),
     state_expr (rs_state rs) <> Let_static bindings body) ->
  step_view rs (plain_step (count_uses rs)).
Proof.
  intros Hnot_region Hnot_static. unfold plain_step.
  change (rs_state (count_uses rs)) with (rs_state rs).
  change (live_statics (count_uses rs)) with (live_statics rs).
  destruct (step (with_statics (rs_state rs) (live_statics rs)))
    as [s' | k vs hp rd | ] eqn:Hstep.
  - eapply (View_step rs (live_statics rs) s');
      [ exact Hnot_region | exact Hnot_static | reflexivity | exact Hstep
      | reflexivity | reflexivity | reflexivity | exact cuts_kept_count_uses ].
  - eapply (View_exit rs (live_statics rs) k vs hp rd);
      [ reflexivity | exact Hstep | reflexivity ].
  - apply View_undefined.
Qed.

Lemma view_begin (x : variable) (kind : flambda_kind) (p : region_primitive)
    (mark_kind : mark_kind) (body : expr) :
  state_expr (rs_state rs) = Let x kind (Prim (Region_primitive p) []) body ->
  begin_mark_kind p = Some mark_kind ->
  step_view rs (begin_region (count_uses rs) x kind mark_kind body).
Proof.
  intros He Hp. unfold begin_region.
  destruct (kind_eqb kind Region); [ | apply View_undefined ].
  eapply (View_begin rs x kind p mark_kind body);
    [ exact He | exact Hp | reflexivity | reflexivity | reflexivity
    | reflexivity | reflexivity | reflexivity | exact cuts_kept_count_uses ].
Qed.

Lemma view_end (x : variable) (kind : flambda_kind) (p : region_primitive)
    (ended_by : mark_kind) (region : simple) (body : expr) :
  state_expr (rs_state rs)
  = Let x kind (Prim (Region_primitive p) [region]) body ->
  is_end p = true ->
  step_view rs (end_region (count_uses rs) x kind ended_by region body).
Proof.
  intros He Hp. unfold end_region.
  destruct (kind_eqb kind Value); [ | apply View_undefined ].
  change (rs_state (count_uses rs)) with (rs_state rs).
  change (rs_frames (count_uses rs)) with (rs_frames rs).
  destruct (eval_simple (state_vars (rs_state rs)) region)
    as [[c | code slots | slot decls slots | m | tag fields | elements
        | contents | l | callee args arity] | ] eqn:Heval;
    try apply View_undefined.
  destruct (cut m (rs_frames rs)) as [[above below] | ] eqn:Hcut;
    [ | apply View_undefined ].
  destruct (cut_mark m (rs_frames rs) above below Hcut) as (f & Hframes & Hf).
  eapply (View_end rs x kind p region body m above f below);
    [ exact He | exact Hp | exact Heval | exact Hframes | exact Hf
    | reflexivity | reflexivity | reflexivity | reflexivity | reflexivity
    | reflexivity | ].
  intros Hnone. unfold cuts_kept. cbn. rewrite Hnone.
  destruct ended_by; split; reflexivity.
Qed.

Lemma view_local (x : variable) (kind : flambda_kind) (d : named)
    (body : expr) (y : variable) (region : option simple) (body' : expr) :
  state_expr (rs_state rs) = Let x kind d body ->
  region_named d = false ->
  marker_of x body = Marker y region body' ->
  step_view rs (local_allocation (count_uses rs) x y kind d region body').
Proof.
  intros He Hd Hmarker.
  destruct (marker_of_marker x y region body body' Hmarker)
    as (kind' & args & Hbody).
  rewrite Hbody in He.
  destruct (is_cell_allocation d && kind_eqb kind Value) eqn:Hcell.
  { (* A cell: both variables are bound to its location, and the
       frames and the counter are left. *)
    rewrite (local_allocation_cell _ _ _ _ _ _ _ Hcell).
    destruct (local_cell_allocation (count_uses rs) x y d region body')
      as [rs' | k vs rs' | fault] eqn:Hstep.
    - assert (Hcuts : cuts_kept rs rs').
      { revert Hstep. unfold local_cell_allocation. cbv zeta.
        destruct (named_region (count_uses rs) region); [ | discriminate ].
        destruct
          (exec_named (live_statics (count_uses rs))
             (state_heap (rs_state (count_uses rs)))
             (state_reads (rs_state (count_uses rs)))
             (state_vars (rs_state (count_uses rs))) d)
          as [[[v hp] rd] | ]; [ | discriminate ].
        destruct v; try discriminate.
        intros [= <-]; split; reflexivity. }
      destruct (local_cell_allocation_step _ _ _ _ _ _ _ Hstep)
        as (l & hp & rd & _ & _ & Hstate & Hframes & _ & Hnext & _).
      refine
        (View_local rs x kind d y kind' args body' (Value_location l) rs' He
           Hd _ _ _ _ _ _ Hcuts).
      + rewrite Hstate. reflexivity.
      + rewrite Hstate. reflexivity.
      + rewrite Hstate. reflexivity.
      + rewrite Hstate. reflexivity.
      + rewrite Hframes. reflexivity.
      + rewrite Hnext. exact (Pos.le_refl _).
    - exfalso.
      exact (local_cell_allocation_no_exit _ _ _ _ _ _ _ _ _ Hstep).
    - apply View_undefined. }
  destruct (local_allocation (count_uses rs) x y kind d region body')
    as [rs' | k vs rs' | fault] eqn:Hstep.
  - assert (Hcuts : cuts_kept rs rs').
    { revert Hstep. unfold local_allocation. cbv zeta. rewrite Hcell.
      destruct (is_allocation d && kind_eqb kind Value); [ | discriminate ].
      destruct (named_region (count_uses rs) region); [ | discriminate ].
      destruct
        (eval_named (live_statics (count_uses rs))
           (state_vars (rs_state (count_uses rs))) d);
        [ | discriminate ].
      destruct (localise (rs_next (count_uses rs)) v) as [[sym bindings] | ];
        [ | discriminate ].
      destruct (rs_frames (count_uses rs)) as [ | [m mark_kind locals] frames];
        intros [= <-]; split; reflexivity. }
    destruct (local_allocation_step _ _ _ _ _ _ _ _ Hcell Hstep)
      as (v & sym & bindings & _ & _ & _ & Hexpr & Hvars & Hconts & _ & Hnext
          & Hcode & Hframes).
    refine
      (View_local rs x kind d y kind' args body' (Value_symbol sym) rs' He Hd
         Hexpr Hvars Hconts Hcode _ _ Hcuts).
    + change (rs_frames (count_uses rs)) with (rs_frames rs) in Hframes.
      destruct (rs_frames rs) as [ | [m mark_kind locals] frames].
      * destruct Hframes as [-> _]. reflexivity.
      * destruct Hframes as [-> _]. reflexivity.
    + rewrite Hnext. exact (advance_le _ _).
  - exfalso. exact (local_allocation_no_exit _ _ _ _ _ _ _ _ _ _ Hstep).
  - apply View_undefined.
Qed.

Lemma view_static (bindings : list static_binding) (body : expr) :
  state_expr (rs_state rs) = Let_static bindings body ->
  step_view rs (static_step (count_uses rs)).
Proof.
  intros He.
  destruct (static_step (count_uses rs)) as [rs' | k vs rs' | fault]
    eqn:Hstep.
  - destruct (static_step_step (count_uses rs) rs' bindings body He Hstep)
      as (statics' & Heval & ->).
    eapply (View_static rs bindings body statics');
      [ exact He | exact Heval | reflexivity | reflexivity | reflexivity
      | reflexivity | reflexivity | reflexivity | exact cuts_kept_count_uses ].
  - exfalso.
    exact (static_step_no_exit (count_uses rs) bindings body k vs rs' He Hstep).
  - apply View_undefined.
Qed.

Lemma view_region_apply_cont (ac : apply_cont) :
  state_expr (rs_state rs) = Apply_cont ac ->
  step_view rs (region_apply_cont (count_uses rs) ac).
Proof.
  intros He. unfold region_apply_cont. cbv zeta.
  change (rs_state (count_uses rs)) with (rs_state rs).
  destruct (resolve (ac_continuation ac) (state_conts (rs_state rs)))
    as [[params metadata handler] def_vars def_conts recursive | k | c extra
       | remaining remaining_arity return_arity return_to exn_to]
    eqn:Hresolve;
    try (apply view_plain; intros; rewrite He; discriminate).
  destruct (eval_simples (state_vars (rs_state rs)) (ac_args ac)) as [vs | ]
    eqn:Heval;
    [ | apply View_undefined ].
  destruct
    (kinds_eqb (map (region_kind_of_value (count_uses rs)) vs)
       (hm_params_kinds metadata));
    [ | apply View_undefined ].
  destruct (bind_params params vs def_vars) as [handler_vars | ] eqn:Hbind;
    [ | apply View_undefined ].
  eapply
    (View_jump rs ac params metadata handler def_vars def_conts recursive vs
       handler_vars);
    [ exact He | exact Hresolve | exact Heval | exact Hbind | reflexivity
    | reflexivity | reflexivity | reflexivity | reflexivity | reflexivity
    | exact cuts_kept_count_uses ].
Qed.

(* A step of the machine of regions is one of these. *)
Lemma region_step_view : step_view rs (region_step rs).
Proof.
  unfold region_step. cbv zeta.
  destruct
    (simples_fault rs (evaluated_simples (state_expr (rs_state rs)))
       (region_places rs));
    [ apply View_undefined | ].
  destruct (state_expr (rs_state rs))
    as [x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ] eqn:He;
    try (apply view_plain; intros; rewrite He; discriminate).
  - assert (Hrest :
              region_named d = false ->
              step_view rs
                match marker_of x body with
                | Marker y region body' =>
                  local_allocation (count_uses rs) x y kind d region body'
                | Bad_marker => Region_undefined Not_of_the_conversion
                | No_marker =>
                  if (is_allocation d || is_cell_allocation d)
                     && existsb (simple_local (count_uses rs))
                          (simples_of_named d)
                  then Region_undefined Heap_holds_local
                  else
                    match store_fault (count_uses rs) d with
                    | Some fault => Region_undefined fault
                    | None => plain_step (count_uses rs)
                    end
                end).
    { intros Hd.
      destruct (marker_of x body) as [ | y region body' | ] eqn:Hmarker.
      - destruct
          ((is_allocation d || is_cell_allocation d)
           && existsb (simple_local (count_uses rs)) (simples_of_named d));
          [ apply View_undefined | ].
        destruct (store_fault (count_uses rs) d); [ apply View_undefined | ].
        apply view_plain; [ | intros; rewrite He; discriminate ].
        intros x0 kind0 p args body0 Heq. rewrite He in Heq.
        injection Heq as _ _ Hd' _. rewrite Hd' in Hd. discriminate.
      - exact (view_local x kind d body y region body' He Hd Hmarker).
      - apply View_undefined. }
    destruct d as [s | [u | b | v | hp | rp] args | | | | | ];
      try (apply Hrest; reflexivity).
    destruct rp; destruct args as [ | region [ | region' args]];
      try apply View_undefined.
    + exact (view_begin x kind Begin_region Plain_mark body He eq_refl).
    + exact (view_begin x kind Begin_try_region Try_mark body He eq_refl).
    + exact (view_end x kind End_region Plain_mark region body He eq_refl).
    + exact (view_end x kind End_try_region Try_mark region body He eq_refl).
  - destruct
      (existsb (simple_local (count_uses rs))
         (evaluated_simples (Let_static bindings body)));
      [ apply View_undefined | ].
    exact (view_static bindings body He).
  - destruct (existsb (simple_open_mark (count_uses rs)) (ac_args ac)).
    + exact (view_region_apply_cont ac He).
    + apply view_plain; intros; rewrite He; discriminate.
Qed.

End View.

(* The view of a step that is known. *)
Lemma region_step_viewed (rs : region_state) (result : region_step_result) :
  region_step rs = result -> step_view rs result.
Proof. intros <-. apply region_step_view. Qed.
