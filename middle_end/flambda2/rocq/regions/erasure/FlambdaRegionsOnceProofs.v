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

(* Lemmas for the proofs of the statements of
   FlambdaRegionsErasureSpec.v: the symbols that the frames bind are
   bound once. FlambdaRegionsInvariantSpec.v does not ask it ("what is
   not here"). Nothing here needs review. *)

From Stdlib Require Import ZArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaRegionsSpec FlambdaRegionsProofs FlambdaRegionsInvariantSpec
  FlambdaRegionsReadsProofs FlambdaRegionsStepsProofs
  FlambdaRegionsFreshProofs.
Import ListNotations.

(* No symbol is bound twice in the frames. *)
Definition locals_once (rs : region_state) : Prop :=
  NoDup (map fst (live_locals (rs_frames rs))).

(* ======================================================================== *)
(* The symbols that a local allocation binds are distinct                   *)
(* ======================================================================== *)

Lemma number_slots_once (function_decls : list (function_slot * code_id)) :
  ∀ next : symbol, NoDup (map snd (number_slots function_decls next)).
Proof.
  induction function_decls as [ | [slot code] function_decls IH];
    intros next; simpl; [ apply NoDup_nil | ].
  apply NoDup_cons; [ | apply IH ].
  intros Hin. apply in_map_iff in Hin as ([slot' s] & Hs & Hin).
  simpl in Hs. subst s.
  pose proof (number_slots_from _ _ _ _ Hin). lia.
Qed.

Lemma localise_once (next : symbol) (v : value) (sym : symbol)
    (bindings : list (symbol * statically_allocated_value)) :
  localise next v = Some (sym, bindings) -> NoDup (map fst bindings).
Proof.
  assert (Hone : ∀ s : symbol, NoDup [s])
    by (intros s; apply NoDup_cons; [ intros [] | apply NoDup_nil ]).
  destruct v as [c | code value_slots | slot function_decls value_slots | s0
                | tag fields | elements | c | l | callee args arity];
    simpl; try discriminate; try (intros [= _ <-]; apply Hone).
  destruct (lookup slot (number_slots function_decls next)) as [s1 | ];
    simpl; [ | discriminate ].
  intros [= _ <-]. rewrite map_map.
  rewrite (map_ext _ snd);
    [ apply number_slots_once | intros [slot' s']; reflexivity ].
Qed.

(* ======================================================================== *)
(* A step keeps it                                                          *)
(* ======================================================================== *)

Lemma locals_once_initial (e : expr) (statics : static_env) (base : symbol) :
  locals_once (initial_region_state e statics base).
Proof. apply NoDup_nil. Qed.

(* The symbols drawn are from the counter on, and those that the frames
   bind are below it: that is where [symbols_fresh] is used. *)
Theorem region_step_keeps_locals_once (rs rs' : region_state) :
  symbols_fresh rs = true ->
  locals_once rs ->
  region_step rs = Region_step rs' ->
  locals_once rs'.
Proof.
  intros Hfresh Honce Hstep. unfold locals_once in *.
  pose proof (proj1 (symbols_fresh_parts rs) Hfresh) as (_ & _ & Hstore).
  destruct (region_step_shape _ _ Hstep) as [_ Hshape].
  destruct Hshape
    as [Hframes _ _ _
       | bindings body _ Hframes _ _
       | kind Hframes _ _ _
       | above f Hframes _ _ _
       | v sym bindings Hlocalise _ Hframes _].
  - rewrite Hframes. exact Honce.
  - rewrite Hframes. exact Honce.
  - rewrite Hframes. exact Honce.
  - rewrite Hframes, live_locals_app, map_app in Honce.
    apply NoDup_app_remove_l in Honce.
    change (live_locals (f :: rs_frames rs'))
      with (frame_locals f ++ live_locals (rs_frames rs')) in Honce.
    rewrite map_app in Honce. exact (NoDup_app_remove_l _ _ Honce).
  - revert Hframes Honce Hstore.
    destruct (rs_frames rs) as [ | f frames]; intros [Hf _] Honce Hstore;
      rewrite Hf; [ apply NoDup_nil | ].
    change
      (live_locals
         (Frame (frame_mark f) (frame_kind f) (bindings ++ frame_locals f)
          :: frames))
      with ((bindings ++ frame_locals f) ++ live_locals frames).
    rewrite <- app_assoc, map_app.
    apply NoDup_app;
      [ exact (localise_once _ _ _ _ Hlocalise) | exact Honce | ].
    intros s Hnew Hold.
    apply in_map_iff in Hnew as ([s1 a1] & Hs1 & Hnew).
    apply in_map_iff in Hold as ([s2 a2] & Hs2 & Hold).
    simpl in Hs1, Hs2. subst s1 s2.
    pose proof (localise_from _ _ _ _ Hlocalise _ _ Hnew) as Hfrom.
    destruct
      (store_bound_below _ _ _ _ s a2 Hstore
         (in_or_app _ _ _ (or_introl Hold)))
      as [Hbelow _].
    lia.
Qed.

(* The state that a step exits with has it. *)
Lemma region_step_exit_locals_once (rs : region_state) (k : continuation)
    (vs : list value) (rs' : region_state) :
  locals_once rs ->
  region_step rs = Region_exit k vs rs' ->
  locals_once rs'.
Proof.
  intros Honce Hexit. rewrite (region_step_exit rs rs' k vs Hexit).
  exact Honce.
Qed.
