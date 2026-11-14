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

(* What follows of decision D53 (FlambdaRegionsSpec.named_needs_heap):
   a [Let] that the machine of regions runs has the value that
   [eval_named] gives its defining expression, and leaves the heap and
   the read record as they are. It proves
   FlambdaRegionsSpec.region_step_keeps_memory_statement, which
   FlambdaRegionsProofs.v states again under its own name, and has the
   lemmas on locations that the proofs of the erasure use. Nothing here
   needs review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaSemanticsProofs FlambdaRegionsSpec
  FlambdaRegionsErasureSpec.
Import ListNotations.

(* ======================================================================== *)
(* A [Let] that needs no heap                                               *)
(* ======================================================================== *)

Lemma existsb_is_location_none (vs : list value) :
  existsb is_location vs = false -> ∀ l : location, Value_location l ∉ vs.
Proof.
  intros Hnone l Hin.
  assert (Hex : existsb is_location vs = true).
  { apply existsb_exists. exists (Value_location l). split; [ exact Hin | ].
    reflexivity. }
  rewrite Hnone in Hex. discriminate Hex.
Qed.

(* What a [Let] that the machine runs gives to [step]: the value that
   [eval_named] has, the same heap and the same record. *)
Lemma exec_named_needs_no_heap (statics : static_env) (hp : heap)
    (rd : read_record) (vars : env) (d : named) :
  named_needs_heap vars d = false ->
  exec_named statics hp rd vars d
  = let* v := eval_named statics vars d in
    Some (v, hp, rd).
Proof.
  intros Hrun.
  destruct d as [s | p args | | | | | ];
    try (apply exec_named_not_prim; intros ? ?; discriminate).
  destruct (eval_simples vars args) as [vs | ] eqn:Hvs.
  - apply (exec_named_no_location statics hp rd vars p args vs).
    + intros q ->. discriminate Hrun.
    + exact Hvs.
    + apply existsb_is_location_none.
      destruct p; simpl in Hrun; rewrite ?Hvs in Hrun;
        try exact Hrun; discriminate Hrun.
  - rewrite (exec_named_no_args statics hp rd vars p args Hvs).
    rewrite eval_named_prim, Hvs. reflexivity.
Qed.

(* The same, of a [Let] that binds a value. *)
Lemma exec_named_needs_no_heap_some (statics : static_env) (hp hp' : heap)
    (rd rd' : read_record) (vars : env) (d : named) (v : value) :
  named_needs_heap vars d = false ->
  exec_named statics hp rd vars d = Some (v, hp', rd') ->
  eval_named statics vars d = Some v ∧ hp' = hp ∧ rd' = rd.
Proof.
  intros Hrun Hexec.
  rewrite (exec_named_needs_no_heap statics hp rd vars d Hrun) in Hexec.
  destruct (eval_named statics vars d) as [w | ]; [ | discriminate ].
  injection Hexec as -> <- <-. auto.
Qed.

(* The step of such a [Let]: the equation that [step] had before it ran
   a [Let] on the heap. *)
Lemma step_let_needs_no_heap (x : variable) (kind : flambda_kind)
    (d : named) (body : expr) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) :
  named_needs_heap vars d = false ->
  step (State (Let x kind d body) vars conts statics hp rd)
  = match eval_named statics vars d with
    | Some v =>
      if kind_eqb (kind_of_value v) kind
      then Step (State body ((x, v) :: vars) conts statics hp rd)
      else Undefined
    | None => Undefined
    end.
Proof.
  intros Hrun.
  rewrite step_let_exec, (exec_named_needs_no_heap statics hp rd vars d Hrun).
  destruct (eval_named statics vars d); reflexivity.
Qed.

(* ======================================================================== *)
(* No step changes the heap or the read record                              *)
(* ======================================================================== *)

(* A state that is not at a [Let] that needs the heap. *)
Definition needs_no_heap (s : state) : Prop :=
  ∀ (x : variable) (kind : flambda_kind) (d : named) (body : expr),
    state_expr s = Let x kind d body ->
    named_needs_heap (state_vars s) d = false.

Lemma step_keeps_memory (s s' : state) :
  needs_no_heap s ->
  step s = Step s' ->
  state_heap s' = state_heap s ∧ state_reads s' = state_reads s.
Proof.
  intros Hruns Hstep. split.
  - destruct (step_heap s s' Hstep)
      as [Hsame | (x & kind & d & body & v & He & Hexec)];
      [ exact Hsame | ].
    destruct
      (exec_named_needs_no_heap_some _ _ _ _ _ _ _ _
         (Hruns x kind d body He) Hexec)
      as (_ & Hheap & _).
    exact Hheap.
  - pose proof (step_keeps_reads s) as Hreads. rewrite Hstep in Hreads.
    simpl in Hreads. injection Hreads as Hreads.
    rewrite <- Hreads. reflexivity.
Qed.

Lemma plain_step_keeps_memory (rs rs' : region_state) :
  needs_no_heap (rs_state rs) ->
  plain_step rs = Region_step rs' ->
  state_heap (rs_state rs') = state_heap (rs_state rs)
  ∧ state_reads (rs_state rs') = state_reads (rs_state rs).
Proof.
  intros Hruns. unfold plain_step.
  destruct (step (with_statics (rs_state rs) (live_statics rs)))
    as [s' | k vs hp rd | ] eqn:Hstep; try discriminate.
  intros [= <-]. simpl.
  exact
    (step_keeps_memory (with_statics (rs_state rs) (live_statics rs)) s' Hruns
       Hstep).
Qed.

Lemma static_step_keeps_memory (rs rs' : region_state) :
  needs_no_heap (rs_state rs) ->
  static_step rs = Region_step rs' ->
  state_heap (rs_state rs') = state_heap (rs_state rs)
  ∧ state_reads (rs_state rs') = state_reads (rs_state rs).
Proof.
  intros Hruns. unfold static_step.
  destruct (step (rs_state rs)) as [s' | k vs hp rd | ] eqn:Hstep;
    try discriminate.
  intros [= <-]. simpl.
  exact (step_keeps_memory _ s' Hruns Hstep).
Qed.

Theorem region_step_keeps_memory : region_step_keeps_memory_statement.
Proof.
  intros rs rs'. unfold region_step.
  destruct (simples_fault rs _ _); [ discriminate | ].
  assert (Hcount : rs_state (count_uses rs) = rs_state rs) by reflexivity.
  set (rs1 := count_uses rs) in *. rewrite <- Hcount. clearbody rs1.
  clear Hcount rs.
  destruct (state_expr (rs_state rs1))
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ] eqn:He.
  - (* Let *)
    assert (Hrest :
              (if named_needs_heap (state_vars (rs_state rs1)) d
               then Region_let_not_run
               else
                 match marker_of x body with
                 | Marker y region body' =>
                   local_allocation rs1 x y kind d region body'
                 | Bad_marker => Region_undefined Not_of_the_conversion
                 | No_marker =>
                   if is_allocation d
                      && existsb (simple_local rs1) (simples_of_named d)
                   then Region_undefined Heap_holds_local
                   else plain_step rs1
                 end)
              = Region_step rs' ->
              state_heap (rs_state rs') = state_heap (rs_state rs1)
              ∧ state_reads (rs_state rs') = state_reads (rs_state rs1)).
    { destruct (named_needs_heap (state_vars (rs_state rs1)) d) eqn:Hrun;
        [ discriminate | ].
      destruct (marker_of x body) as [ | y region body' | ].
      - destruct (is_allocation d && _); [ discriminate | ].
        apply plain_step_keeps_memory.
        intros x0 kind0 d0 body0 He0. rewrite He in He0.
        injection He0 as _ _ <- _. exact Hrun.
      - unfold local_allocation.
        destruct (is_allocation d && kind_eqb kind Value); [ | discriminate ].
        destruct (named_region rs1 region); [ | discriminate ].
        destruct (eval_named _ _ d); [ | discriminate ].
        destruct (localise _ _) as [[symbol bindings'] | ]; [ | discriminate ].
        destruct (rs_frames rs1) as [ | [m mark_kind locals] frames];
          intros [= <-]; split; reflexivity.
      - discriminate. }
    destruct d as [s | p args | | | | | ]; try exact Hrest.
    destruct p as [p | p | p | p | p]; try exact Hrest.
    destruct p, args as [ | region [ | ? ?]]; try discriminate.
    + unfold begin_region. destruct (kind_eqb kind Region); [ | discriminate ].
      intros [= <-]. split; reflexivity.
    + unfold begin_region. destruct (kind_eqb kind Region); [ | discriminate ].
      intros [= <-]. split; reflexivity.
    + unfold end_region. destruct (kind_eqb kind Value); [ | discriminate ].
      destruct (eval_simple _ region) as [[] | ]; try discriminate.
      destruct (cut _ _) as [[above below] | ]; [ | discriminate ].
      intros [= <-]. split; reflexivity.
    + unfold end_region. destruct (kind_eqb kind Value); [ | discriminate ].
      destruct (eval_simple _ region) as [[] | ]; try discriminate.
      destruct (cut _ _) as [[above below] | ]; [ | discriminate ].
      intros [= <-]. split; reflexivity.
  - (* Let_static *)
    destruct (existsb _ _); [ discriminate | ].
    apply static_step_keeps_memory. intros ? ? ? ? He0. rewrite He in He0.
    discriminate He0.
  - apply plain_step_keeps_memory. intros ? ? ? ? He0. rewrite He in He0.
    discriminate He0.
  - apply plain_step_keeps_memory. intros ? ? ? ? He0. rewrite He in He0.
    discriminate He0.
  - apply plain_step_keeps_memory. intros ? ? ? ? He0. rewrite He in He0.
    discriminate He0.
  - (* Apply_cont *)
    assert (Hplain :
              plain_step rs1 = Region_step rs' ->
              state_heap (rs_state rs') = state_heap (rs_state rs1)
              ∧ state_reads (rs_state rs') = state_reads (rs_state rs1)).
    { apply plain_step_keeps_memory. intros ? ? ? ? He0. rewrite He in He0.
      discriminate He0. }
    destruct (existsb _ _); [ | exact Hplain ].
    unfold region_apply_cont.
    destruct (resolve _ _) as [[params metadata handler] ? ? ? | | | ];
      try exact Hplain.
    destruct (eval_simples _ _); [ | discriminate ].
    destruct (kinds_eqb _ _); [ | discriminate ].
    destruct (bind_params _ _ _); [ | discriminate ].
    intros [= <-]. split; reflexivity.
  - apply plain_step_keeps_memory. intros ? ? ? ? He0. rewrite He in He0.
    discriminate He0.
  - apply plain_step_keeps_memory. intros ? ? ? ? He0. rewrite He in He0.
    discriminate He0.
Qed.

(* ======================================================================== *)
(* Locations and the erasure                                                *)
(* ======================================================================== *)

(* A value is erased to a location only if it is that location. *)
Lemma value_erased_is_location (rs : region_state) (v v' : value) :
  value_erased rs v v' -> is_location v' = is_location v.
Proof. intros Hv. destruct Hv; reflexivity. Qed.

Lemma values_erased_is_location (rs : region_state) (vs vs' : list value) :
  Forall2 (value_erased rs) vs vs' ->
  existsb is_location vs' = existsb is_location vs.
Proof.
  induction 1 as [ | v v' vs vs' Hv _ IH]; [ reflexivity | ].
  simpl. rewrite (value_erased_is_location rs v v' Hv), IH. reflexivity.
Qed.
