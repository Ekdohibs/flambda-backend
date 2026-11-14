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

(* Where a run on the machine of regions stops because [step] has no
   step, the run of the erased term goes wrong, if in every state of
   the run what a step reads may be read
   (FlambdaRegionsCheckedSpec.reads_live) and the regions of the
   handler that is applied are given
   (FlambdaRegionsCheckedSpec.regions_given). Nothing here needs
   review. This file does not import FlambdaRegionsCheckSoundSpec.v:
   [no_plain_step_erased_run] is stated with the text of
   no_plain_step_erased_statement of that file, which is what is to be
   reviewed, so that FlambdaRegionsCheckSoundProofs.v can prove that
   statement by it.

   The two states are related as in FlambdaRegionsErasureProofs.v
   ([related]), which a run keeps. At the state where the machine of
   regions stops so, the second state has no step either
   ([region_step_back]): had it one, the machine of regions would not
   stop because [step] has none, by FlambdaRegionsErasedReadsProofs.v
   and FlambdaRegionsErasedCallsProofs.v. It might stop there for a
   rule of regions, which is another fault and is not what is
   supposed. By the expression of the state:

   - a primitive of regions never stops so;
   - a [Let] that needs the heap is not run (decision D53): the result
     is [Region_let_not_run], which is not what is supposed;
   - a local allocation stops so where its defining expression has no
     value, and the erased term has the same [Let]
     ([local_allocation_back]);
   - a group of static bindings ([static_step_back]);
   - a continuation given a region: the erased term gives the handler
     the arguments that are no regions. The handler has as many kinds
     as the [Apply_cont] has arguments and as it has parameters,
     because every kind [Region] has an argument and a parameter and
     the erased handler has as many of each as it is given
     ([kinds_given], [drop_regions_lengths], [region_apply_cont_back]);
   - anything else is a step of [step] ([step_back],
     [plain_step_back]); an [Apply_cont] none of whose arguments is a
     mark on the stack applies a continuation that takes no region,
     since every kind [Region] of a handler is given a mark
     ([plain_simples_live]).

   Then the run, by induction on the steps of the machine of regions,
   as FlambdaRegionsErasureProofs.run_exits ([run_no_plain_step]). *)

From Stdlib Require Import ZArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaValueSpec FlambdaPrimitiveModel FlambdaPrimitiveSpec
  FlambdaPrimitiveProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaClosuresSpec FlambdaClosuresProofs FlambdaLiftProofs
  FlambdaRegionsSpec FlambdaRegionsProofs FlambdaRegionsNeedsHeapProofs
  FlambdaRegionsInvariantSpec FlambdaRegionsErasureSpec
  FlambdaRegionsReadsProofs FlambdaRegionsStepsProofs
  FlambdaRegionsInvariantProofs
  FlambdaRegionsErasedProofs FlambdaRegionsPlainStepProofs
  FlambdaRegionsErasureProofs FlambdaRegionsCheckedSpec
  FlambdaRegionsErasedReadsProofs FlambdaRegionsErasedCallsProofs.
Import ListNotations.

(* The static environment that [step] is given is not unfolded: the
   rules of FlambdaRegionsErasureSpec.value_erased name it. *)
#[local] Opaque live_statics.

(* ======================================================================== *)
(* The counts change nothing that is read here                              *)
(* ======================================================================== *)

#[local] Transparent live_statics.

Lemma simples_all_live_with_counts (rs : region_state)
    (counts : region_counts) (vars : env) (simples : list simple) :
  simples_all_live rs vars simples ->
  simples_all_live (with_counts rs counts) vars simples.
Proof. exact (fun H => H). Qed.

Lemma statics_live_with_counts (rs : region_state) (counts : region_counts) :
  forallb (fun '(_, allocated) => allocated_all (live_symbol rs) allocated)
    (static_values (live_statics rs))
  = true ->
  forallb
    (fun '(_, allocated) =>
       allocated_all (live_symbol (with_counts rs counts)) allocated)
    (static_values (live_statics (with_counts rs counts)))
  = true.
Proof. exact (fun H => H). Qed.

Lemma values_live_all_with_counts (rs : region_state)
    (counts : region_counts) (vs : list value) :
  forallb (value_all (live_symbol rs)) vs = true ->
  forallb (value_all (live_symbol (with_counts rs counts))) vs = true.
Proof. exact (fun H => H). Qed.

Lemma simples_live_with_counts (rs : region_state) (counts : region_counts)
    (simples : list simple) :
  ∀ places : list bool,
    simples_live (with_counts rs counts) simples places
    = simples_live rs simples places.
Proof.
  induction simples as [ | s simples IH]; intros places; [ reflexivity | ].
  simpl. rewrite IH. reflexivity.
Qed.

#[local] Opaque live_statics.

(* ======================================================================== *)
(* What the two properties of the state give                                *)
(* ======================================================================== *)

(* The test of the simples never stops for [step]. *)
Lemma simples_fault_not_plain (rs : region_state) (simples : list simple) :
  ∀ (places : list bool) (fault : region_fault),
    simples_fault rs simples places = Some fault -> fault <> No_plain_step.
Proof.
  induction simples as [ | s simples IH]; intros places fault; simpl;
    [ discriminate | ].
  destruct (simple_usable rs s); [ apply IH | ].
  destruct (simple_open_mark rs s).
  - destruct (hd false places); [ apply IH | intros [= <-]; discriminate ].
  - destruct (hd false places); intros [= <-]; discriminate.
Qed.

(* With no place of a region, the value of every simple reaches only
   symbols that may be evaluated. *)
Lemma simples_live_no_place (rs : region_state) (simples : list simple) :
  simples_live rs simples [] = true ->
  simples_all_live rs (state_vars (rs_state rs)) simples.
Proof.
  induction simples as [ | s simples IH]; simpl; [ intros _ s0 v [] | ].
  intros H. apply andb_true_iff in H as [Hs Hrest].
  intros s0 v [<- | Hin] Hv.
  - rewrite Hv in Hs. exact Hs.
  - exact (IH Hrest s0 v Hin Hv).
Qed.

(* With no mark on the stack among the simples, the same, and none of
   the simples is at the place of a region. *)
Lemma simples_live_no_mark (rs : region_state) (simples : list simple) :
  ∀ places : list bool,
    simples_live rs simples places = true ->
    existsb (simple_open_mark rs) simples = false ->
    simples_all_live rs (state_vars (rs_state rs)) simples
    ∧ existsb (fun place => place) (firstn (length simples) places) = false.
Proof.
  induction simples as [ | s simples IH]; intros places Hlive Hmark.
  - split; [ intros s0 v [] | reflexivity ].
  - simpl in Hlive, Hmark. apply andb_true_iff in Hlive as [Hs Hrest].
    apply orb_false_iff in Hmark as [Hs_mark Hmark].
    destruct (IH (tl places) Hrest Hmark) as [Hall Hplaces].
    assert (Hplace : hd false places = false).
    { destruct (hd false places); [ | reflexivity ].
      rewrite Hs_mark in Hs. discriminate. }
    rewrite Hplace in Hs.
    split.
    + intros s0 v [<- | Hin] Hv;
        [ rewrite Hv in Hs; exact Hs | exact (Hall s0 v Hin Hv) ].
    + destruct places as [ | place places]; [ reflexivity | ].
      simpl in Hplace, Hplaces |- *. rewrite Hplace, Hplaces. reflexivity.
Qed.

Lemma none_of_all_not (places : list bool) :
  forallb negb places = true -> existsb (fun place => place) places = false.
Proof.
  induction places as [ | place places IH]; simpl; [ reflexivity | ].
  intros H. apply andb_true_iff in H as [Hplace Hrest].
  apply negb_true_iff in Hplace. rewrite Hplace. exact (IH Hrest).
Qed.

Lemma no_place (places : list bool) (n : nat) :
  existsb (fun place => place) (firstn n places) = false ->
  forallb negb (skipn n places) = true ->
  existsb (fun place => place) places = false.
Proof.
  intros Hfirst Hrest.
  rewrite <- (firstn_skipn n places), existsb_app, Hfirst.
  exact (none_of_all_not _ Hrest).
Qed.

(* At an expression none of whose simples is a mark on the stack: the
   values of the simples reach only symbols that may be evaluated, and
   a continuation that an [Apply_cont] applies takes no region. Every
   kind [Region] of a handler is at the place of an argument, which
   is then a mark. *)
Lemma plain_simples_live (rs : region_state) :
  simples_live rs (evaluated_simples (state_expr (rs_state rs)))
    (region_places rs)
  = true ->
  regions_given rs = true ->
  (∀ ac : apply_cont,
    state_expr (rs_state rs) = Apply_cont ac ->
    existsb (simple_open_mark rs) (ac_args ac) = false) ->
  simples_all_live rs (state_vars (rs_state rs))
    (evaluated_simples (state_expr (rs_state rs)))
  ∧ ∀ ac : apply_cont,
      state_expr (rs_state rs) = Apply_cont ac ->
      cont_takes_no_region
        (resolve (ac_continuation ac) (state_conts (rs_state rs))).
Proof.
  unfold region_places, regions_given. cbv zeta.
  destruct (state_expr (rs_state rs))
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ]
    eqn:Hexpr;
    intros Hlive Hgiven Hmarks;
    try (split;
         [ exact (simples_live_no_place rs _ Hlive)
         | intros ac0 Heq; discriminate Heq ]).
  specialize (Hmarks ac eq_refl).
  destruct (simples_live_no_mark rs (ac_args ac) _ Hlive Hmarks)
    as [Hall Hfirst].
  split; [ exact Hall | ].
  intros ac0 Heq. injection Heq as <-.
  destruct (resolve (ac_continuation ac) (state_conts (rs_state rs)))
    as [ [params metadata handler] def_vars def_conts recursive | | | ];
    try exact I.
  apply andb_true_iff in Hgiven as [Hgiven_args _].
  unfold cont_takes_no_region, handler_takes_no_region.
  apply negb_true_iff.
  exact (no_place _ (length (ac_args ac)) Hfirst Hgiven_args).
Qed.

(* ======================================================================== *)
(* A step of [step]                                                         *)
(* ======================================================================== *)

Lemma naked_immediate_back (rs : region_state) (v v' : value) (i : Z) :
  value_erased rs v v' ->
  naked_immediate_of_value v' = Some i ->
  naked_immediate_of_value v = Some i.
Proof.
  intros Hv. erased_cases Hv; simpl; try discriminate. exact (fun H => H).
Qed.

Lemma find_arm_back (rconts : region_conts) (i : Z)
    (arms : list (Z * apply_cont)) (ac' : apply_cont) :
  find_arm i (map (fun '(i0, ac0) => (i0, erase_apply_cont rconts ac0)) arms)
  = Some ac' ->
  ∃ ac : apply_cont, find_arm i arms = Some ac.
Proof.
  induction arms as [ | [d ac0] arms IH]; simpl; [ discriminate | ].
  destruct (Z.eqb i d); [ intros _; eauto | exact IH ].
Qed.

(* One step of [step], at an expression at whose head the machine of
   regions does nothing of its own, that is not a [Let_static] and that
   is not a [Let] that needs the heap
   (FlambdaRegionsSpec.named_needs_heap): where the second state has a
   step, [step] has one under the static environment of the frames. The
   premises beside the relation: the values of the simples that the
   step evaluates, and the values that the continuation applied gives
   on, reach only symbols that may be evaluated; and a continuation
   that an [Apply_cont] applies takes no region. *)
Lemma step_back (rs : region_state) (rconts : region_conts) (e : expr)
    (vars vars' : env) (conts conts' : cont_env) (statics' : static_env)
    (hp : heap) (rd : read_record) :
  symbols_fresh rs = true ->
  forallb (fun '(_, allocated) => allocated_all (live_symbol rs) allocated)
    (static_values (live_statics rs))
  = true ->
  calls_return_no_region rconts e = true ->
  conts_agree rconts conts ->
  env_erased rs vars vars' ->
  conts_erased rs conts conts' ->
  statics_erased rs statics' ->
  head_plain e = true ->
  (∀ (bindings : list static_binding) (body : expr),
    e <> Let_static bindings body) ->
  (∀ (x : variable) (kind : flambda_kind) (d : named) (body : expr),
    e = Let x kind d body -> named_needs_heap vars d = false) ->
  expr_below (rs_base rs) e = true ->
  simples_all_live rs vars (evaluated_simples e) ->
  (∀ ap : apply_expr,
    e = Apply ap ->
    forallb (value_all (live_symbol rs))
      (cont_spine_values (resolve (ap_continuation ap) conts))
    = true) ->
  (∀ ac : apply_cont,
    e = Apply_cont ac ->
    cont_takes_no_region (resolve (ac_continuation ac) conts)
    ∧ forallb (value_all (live_symbol rs))
        (cont_spine_values (resolve (ac_continuation ac) conts))
      = true) ->
  step (State (erase rconts e) vars' conts' statics' hp rd) <> Undefined ->
  step (State e vars conts (live_statics rs) hp rd) <> Undefined.
Proof.
  intros Hfresh Hlive Hreturns Hagree Hvars Hconts Hstatics Hhead Hnot_static
    Hneeds Hbelow Hall Hspine_apply Hspine_cont.
  destruct e
    as [ x xkind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ].
  - (* A [Let]. *)
    destruct (head_plain_let rconts x xkind d body Hhead) as [-> Hsimples].
    rewrite Hsimples in Hall. simpl in Hbelow.
    apply andb_true_iff in Hbelow as [Hd_below _].
    pose proof (Hneeds x xkind d body eq_refl) as Hrun.
    rewrite
      (step_let_needs_no_heap x xkind d _ vars' conts' statics' hp rd
         (named_needs_heap_back rs vars vars' d Hvars Hd_below
            (simples_all_live_usable rs vars _ Hall) Hrun)),
      (step_let_needs_no_heap x xkind d body vars conts (live_statics rs) hp
         rd Hrun).
    destruct (eval_named statics' vars' d) as [v' | ] eqn:Hd';
      [ | intros H; contradiction (H eq_refl) ].
    destruct
      (eval_named_back rs statics' vars vars' d v' Hfresh Hstatics Hlive Hvars
         Hd_below Hall Hd')
      as (v & -> & Hv).
    rewrite (kind_of_value_erased rs v v' Hv).
    destruct (kind_eqb (kind_of_value v) xkind);
      [ intros _; discriminate | exact (fun H => H) ].
  - (* A [Let_static]: not here. *)
    contradiction (Hnot_static bindings body eq_refl).
  - (* A non-recursive [Let_cont]. *)
    cbn [erase]. rewrite !step_let_cont_non_recursive. intros _. discriminate.
  - (* A recursive [Let_cont]. *)
    change (erase rconts (Let_cont_recursive handlers body))
      with
        (Let_cont_recursive
           (erase_handlers (region_conts_of handlers ++ rconts) handlers)
           (erase (region_conts_of handlers ++ rconts) body)).
    rewrite !step_let_cont_recursive, erase_handlers_bind_statics.
    destruct (handlers_bind_statics handlers);
      [ exact (fun H => H) | intros _; discriminate ].
  - (* A call. *)
    cbn [erase]. rewrite !step_apply.
    simpl in Hbelow, Hreturns.
    apply andb_true_iff in Hbelow as [Hcallee_below Hbelow].
    rewrite forallb_app in Hbelow.
    apply andb_true_iff in Hbelow as [Hargs_below Hextra_below].
    apply andb_true_iff in Hreturns as [Hreturn Hexn].
    destruct (eval_simple vars' (ap_callee ap)) as [callee' | ] eqn:Hcallee';
      [ | intros H; contradiction (H eq_refl) ].
    destruct (eval_simples vars' (ap_args ap)) as [vs' | ] eqn:Hargs';
      [ | intros H; contradiction (H eq_refl) ].
    destruct (eval_simples vars' (exn_extra_simples ap)) as [extra' | ]
      eqn:Hextra';
      [ | intros H; contradiction (H eq_refl) ].
    destruct
      (eval_simple_related_back rs vars vars' (ap_callee ap) callee' Hvars
         Hcallee_below (fun v => Hall (ap_callee ap) v (or_introl eq_refl))
         Hcallee')
      as (callee & -> & Hcallee).
    destruct
      (eval_simples_related_back rs vars vars' (ap_args ap) vs' Hvars
         Hargs_below
         (fun s v Hin =>
            Hall s v (or_intror (in_or_app _ _ _ (or_introl Hin))))
         Hargs')
      as (vs & -> & Hvs).
    destruct
      (eval_simples_back rs vars vars' (exn_extra_simples ap) Hvars extra'
         Hextra')
      as [extra ->].
    apply (call_function_back rs statics' Hfresh Hstatics Hlive);
      try assumption.
    + apply resolve_erased, Hconts.
    + exact (takes_no_region_resolve rconts conts _ Hagree Hreturn).
    + exact (Hspine_apply ap eq_refl).
  - (* An [Apply_cont]. *)
    destruct (Hspine_cont ac eq_refl) as [Htakes Hspine].
    cbn [erase].
    rewrite (erase_apply_cont_no_region rconts conts ac Hagree Htakes).
    rewrite !step_apply_cont. simpl in Hbelow.
    destruct (eval_simples vars' (ac_args ac)) as [vs' | ] eqn:Hargs';
      [ | intros H; contradiction (H eq_refl) ].
    destruct
      (eval_simples_related_back rs vars vars' (ac_args ac) vs' Hvars Hbelow
         Hall Hargs')
      as (vs & -> & Hvs).
    unfold apply_continuation.
    apply
      (apply_cont_value_back rs statics' Hfresh Hstatics Hlive _ _
         (resolve_erased rs _ _ _ Hconts) Htakes Hspine).
    exact Hvs.
  - (* A [Switch]. *)
    cbn [erase]. rewrite !step_switch. simpl in Hbelow.
    apply andb_true_iff in Hbelow as [Hscrutinee_below _].
    destruct (eval_simple vars' scrutinee) as [v' | ] eqn:Hv';
      [ | intros H; contradiction (H eq_refl) ].
    destruct
      (eval_simple_related_back rs vars vars' scrutinee v' Hvars
         Hscrutinee_below (fun v => Hall scrutinee v (or_introl eq_refl)) Hv')
      as (v & -> & [Hv _]).
    destruct (naked_immediate_of_value v') as [i | ] eqn:Hi';
      [ | intros H; contradiction (H eq_refl) ].
    rewrite (naked_immediate_back rs v v' i Hv Hi').
    destruct
      (find_arm i
         (map (fun '(i0, ac0) => (i0, erase_apply_cont rconts ac0)) arms))
      as [ac' | ] eqn:Harm';
      [ | intros H; contradiction (H eq_refl) ].
    destruct (find_arm_back rconts i arms ac' Harm') as [ac ->].
    intros _. discriminate.
  - (* [Invalid]. *)
    exact (fun H => H).
Qed.

(* The step of [step] under the static environment of the frames, of a
   state that is not at a [Let] that needs the heap. *)
Lemma plain_step_back (rs : region_state) (s' : state) :
  symbols_fresh rs = true ->
  forallb (fun '(_, allocated) => allocated_all (live_symbol rs) allocated)
    (static_values (live_statics rs))
  = true ->
  state_erased rs s' ->
  head_plain (state_expr (rs_state rs)) = true ->
  (∀ (bindings : list static_binding) (body : expr),
    state_expr (rs_state rs) <> Let_static bindings body) ->
  needs_no_heap (rs_state rs) ->
  simples_all_live rs (state_vars (rs_state rs))
    (evaluated_simples (state_expr (rs_state rs))) ->
  forallb (value_all (live_symbol rs))
    (flat_map cont_spine_values (head_conts rs))
  = true ->
  (∀ ac : apply_cont,
    state_expr (rs_state rs) = Apply_cont ac ->
    cont_takes_no_region
      (resolve (ac_continuation ac) (state_conts (rs_state rs)))) ->
  step s' <> Undefined ->
  plain_step rs <> Region_undefined No_plain_step.
Proof.
  intros Hfresh Hlive
    (rconts & He & Hreturns & Hagree & Hvars & Hconts & Hstatics & Hheap
     & Hreads)
    Hhead Hnot_static Hneeds Hall Hspine Htakes Hstep'.
  destruct s' as [e' vars' conts' statics' hp' rd'].
  simpl in He, Hvars, Hconts, Hstatics, Hheap, Hreads.
  subst e' hp' rd'.
  unfold head_conts in Hspine. cbv zeta in Hspine.
  assert (Hcore :
    step
      (State (state_expr (rs_state rs)) (state_vars (rs_state rs))
         (state_conts (rs_state rs)) (live_statics rs)
         (state_heap (rs_state rs)) (state_reads (rs_state rs)))
    <> Undefined).
  { apply
      (step_back rs rconts _ _ vars' _ conts' statics' _ rd_ Hfresh Hlive
         Hreturns Hagree Hvars Hconts Hstatics Hhead Hnot_static Hneeds
         (symbols_fresh_expr rs Hfresh) Hall);
      [ | | exact Hstep' ].
    - intros ap Hexpr. rewrite Hexpr in Hspine. simpl in Hspine.
      rewrite app_nil_r in Hspine. exact Hspine.
    - intros ac Hexpr. split; [ exact (Htakes ac Hexpr) | ].
      rewrite Hexpr in Hspine. simpl in Hspine.
      rewrite app_nil_r in Hspine. exact Hspine. }
  unfold plain_step, with_statics. cbv zeta.
  destruct
    (step
       (State (state_expr (rs_state rs)) (state_vars (rs_state rs))
          (state_conts (rs_state rs)) (live_statics rs)
          (state_heap (rs_state rs)) (state_reads (rs_state rs))))
    as [s1 | k vs hp rd | ];
    [ discriminate | discriminate | contradiction (Hcore eq_refl) ].
Qed.

(* ======================================================================== *)
(* A group of static bindings                                               *)
(* ======================================================================== *)

Lemma static_step_back (rs : region_state) (s' : state)
    (bindings : list static_binding) (body : expr) :
  symbols_fresh rs = true ->
  state_erased rs s' ->
  state_expr (rs_state rs) = Let_static bindings body ->
  step s' <> Undefined ->
  static_step rs <> Region_undefined No_plain_step.
Proof.
  intros Hfresh
    (rconts & He & Hreturns & Hagree & Hvars & Hconts & Hstatics & Hheap
     & Hreads)
    Hexpr Hstep'.
  pose proof (symbols_fresh_expr rs Hfresh) as Hbelow.
  rewrite Hexpr in Hbelow. simpl in Hbelow.
  apply andb_true_iff in Hbelow as [Hbindings_below _].
  destruct s' as [e' vars' conts' statics' hp' rd'].
  simpl in He, Hvars, Hconts, Hstatics, Hheap, Hreads.
  subst e' hp' rd'.
  rewrite Hexpr in Hstep'. cbn [erase] in Hstep'.
  rewrite step_let_static in Hstep'.
  destruct (eval_static_bindings vars' (map erase_binding bindings) statics')
    as [statics1' | ] eqn:Heval';
    [ | contradiction (Hstep' eq_refl) ].
  destruct
    (eval_static_bindings_back rs (state_vars (rs_state rs)) vars' bindings
       Hvars Hbindings_below (state_statics (rs_state rs)) statics' statics1')
    as [statics1 Heval]; [ | | exact Heval' | ].
  - intros id Hid'. destruct Hstatics as (Hcode & _). rewrite Hcode in Hid'.
    destruct (lookup id (static_code (state_statics (rs_state rs))));
      [ discriminate | reflexivity ].
  - intros s Hs Hs'. destruct Hstatics as (_ & _ & Hsymbols & _).
    destruct (lookup s (static_values (state_statics (rs_state rs))))
      as [allocated | ] eqn:Hl; [ | reflexivity ].
    pose proof (drawn_below rs s Hs) as Hnot_drawn.
    assert (Hheap_symbol : heap_symbol rs s = true)
      by (unfold heap_symbol; rewrite Hnot_drawn; reflexivity).
    rewrite <- (heap_lookup rs s Hfresh Hheap_symbol) in Hl.
    destruct (Hsymbols s allocated Hnot_drawn Hl) as (allocated' & Hl' & _).
    rewrite Hs' in Hl'. discriminate.
  - unfold static_step. revert Hexpr Heval.
    destruct (rs_state rs) as [e vars conts statics hp rd]. simpl.
    intros -> Heval. rewrite step_let_static, Heval. discriminate.
Qed.

(* ======================================================================== *)
(* A local allocation                                                       *)
(* ======================================================================== *)

Lemma evaluated_simples_let_plain (x : variable) (kind : flambda_kind)
    (d : named) (body : expr) :
  (∀ (p : region_primitive) (args : list simple),
    d <> Prim (Region_primitive p) args) ->
  evaluated_simples (Let x kind d body) = simples_of_named d.
Proof.
  intros Hnot.
  destruct d
    as [ s | p args | code slots | slot decls slots | slot closure
       | project_from slot closure | move_from move_to closure ];
    try reflexivity.
  destruct p as [ | | | | rp]; try reflexivity.
  contradiction (Hnot rp args eq_refl).
Qed.

(* The erased term has the [Let] of the allocation, with the same
   defining expression: where that has a value in the second state it
   has one on the machine of regions, and the local allocation does
   not stop for [step]. The [Let] needs no heap: the machine of regions
   comes to a local allocation of no other (decision D53). *)
Lemma local_allocation_back (rs : region_state) (s' : state)
    (x y : variable) (kind : flambda_kind) (d : named)
    (region : option simple) (body body' : expr) :
  symbols_fresh rs = true ->
  forallb (fun '(_, allocated) => allocated_all (live_symbol rs) allocated)
    (static_values (live_statics rs))
  = true ->
  state_erased rs s' ->
  state_expr (rs_state rs) = Let x kind d body ->
  (∀ (p : region_primitive) (args : list simple),
    d <> Prim (Region_primitive p) args) ->
  named_needs_heap (state_vars (rs_state rs)) d = false ->
  simples_all_live rs (state_vars (rs_state rs)) (simples_of_named d) ->
  step s' <> Undefined ->
  local_allocation rs x y kind d region body' <> Region_undefined No_plain_step.
Proof.
  intros Hfresh Hlive
    (rconts & He & Hreturns & Hagree & Hvars & Hconts & Hstatics & Hheap
     & Hreads)
    Hexpr Hnot Hneeds Hall Hstep'.
  pose proof (symbols_fresh_expr rs Hfresh) as Hbelow.
  rewrite Hexpr in Hbelow. simpl in Hbelow.
  apply andb_true_iff in Hbelow as [Hd_below _].
  destruct s' as [e' vars' conts' statics' hp' rd'].
  simpl in He, Hvars, Hconts, Hstatics, Hheap, Hreads. subst e' hp' rd'.
  rewrite Hexpr, (erase_let_plain rconts x kind d body Hnot),
    (step_let_needs_no_heap x kind d _ vars' conts' statics' _ _
       (named_needs_heap_back rs _ vars' d Hvars Hd_below
          (simples_all_live_usable rs _ _ Hall) Hneeds))
    in Hstep'.
  destruct (eval_named statics' vars' d) as [v' | ] eqn:Hd';
    [ | contradiction (Hstep' eq_refl) ].
  destruct
    (eval_named_back rs statics' _ vars' d v' Hfresh Hstatics Hlive Hvars
       Hd_below Hall Hd')
    as (v & Hd & _).
  unfold local_allocation. cbv zeta.
  destruct (is_allocation d && kind_eqb kind Value); [ | discriminate ].
  destruct (named_region rs region); [ | discriminate ].
  rewrite Hd.
  destruct (localise (rs_next rs) v) as [[sym bindings] | ];
    [ | discriminate ].
  destruct (rs_frames rs) as [ | [m mark_kind locals] frames]; discriminate.
Qed.

(* ======================================================================== *)
(* A region given to a continuation                                         *)
(* ======================================================================== *)

Lemma eval_simples_as_many (vars : env) (args : list simple) :
  ∀ vs : list value,
    eval_simples vars args = Some vs -> length vs = length args.
Proof.
  induction args as [ | s args IH]; intros vs; simpl.
  - intros [= <-]. reflexivity.
  - destruct (eval_simple vars s); [ | discriminate ].
    destruct (eval_simples vars args) as [rest | ]; [ | discriminate ].
    intros [= <-]. simpl. rewrite (IH rest eq_refl). reflexivity.
Qed.

Lemma bind_params_as_many (params : list variable) :
  ∀ (vs : list value) (vars bound : env),
    bind_params params vs vars = Some bound -> length params = length vs.
Proof.
  induction params as [ | x params IH]; intros [ | v vs] vars bound; simpl;
    try discriminate; [ reflexivity | ].
  intros Hbound. f_equal. exact (IH vs _ bound Hbound).
Qed.

Lemma bind_params_of_as_many (params : list variable) :
  ∀ (vs : list value) (vars : env),
    length params = length vs ->
    ∃ bound : env, bind_params params vs vars = Some bound.
Proof.
  induction params as [ | p params IH]; intros [ | v vs] vars; simpl;
    try discriminate; [ eauto | ].
  intros [= Hlength]. apply IH, Hlength.
Qed.

(* The arguments have values on the machine of regions where those
   that are no regions have values in the second state: one at the
   place of a region is a mark, which has a value. *)
Lemma eval_simples_given (rs : region_state) (vars' : env)
    (args : list simple) :
  env_erased rs (state_vars (rs_state rs)) vars' ->
  ∀ (places : list bool) (vs' : list value),
    simples_live rs args places = true ->
    eval_simples vars' (drop_regions places args) = Some vs' ->
    ∃ vs : list value,
      eval_simples (state_vars (rs_state rs)) args = Some vs.
Proof.
  intros Hvars. induction args as [ | s args IH]; intros places vs' Hlive Hvs';
    [ exists []; reflexivity | ].
  simpl in Hlive. apply andb_true_iff in Hlive as [Hs Hrest].
  destruct places as [ | place places].
  - simpl in Hvs'.
    exact (eval_simples_back rs _ vars' (s :: args) Hvars vs' Hvs').
  - simpl in Hs, Hrest. destruct place; simpl in Hvs'.
    + destruct (IH places vs' Hrest Hvs') as [rest Hrest_eval].
      unfold simple_open_mark in Hs.
      destruct (eval_simple (state_vars (rs_state rs)) s) as [v | ] eqn:Hv;
        [ | discriminate ].
      exists (v :: rest). simpl. rewrite Hv, Hrest_eval. reflexivity.
    + destruct (eval_simple vars' s) as [v' | ] eqn:Hv'; [ | discriminate ].
      destruct (eval_simples vars' (drop_regions places args)) as [rest' | ]
        eqn:Hrest'; [ | discriminate ].
      destruct (eval_simple_back rs _ vars' s v' Hvars Hv') as [v Hv].
      destruct (IH places rest' Hrest Hrest') as [rest Hrest_eval].
      exists (v :: rest). simpl. rewrite Hv, Hrest_eval. reflexivity.
Qed.

(* The values of the arguments are of the kinds of the handler, where
   those that are no regions are of the kinds that are no [Region]:
   every kind [Region] has an argument, which is a mark on the stack;
   an argument at any other place may be evaluated, so it is no such
   mark. *)
Lemma kinds_given (rs : region_state) (args : list simple) :
  ∀ (kinds : list flambda_kind) (vs : list value),
    simples_live rs args (map (fun kind => kind_eqb kind Region) kinds)
    = true ->
    simples_fault rs args (map (fun kind => kind_eqb kind Region) kinds)
    = None ->
    forallb negb
      (skipn (length args) (map (fun kind => kind_eqb kind Region) kinds))
    = true ->
    eval_simples (state_vars (rs_state rs)) args = Some vs ->
    map kind_of_value
      (drop_regions (map (fun kind => kind_eqb kind Region) kinds) vs)
    = drop_regions (map (fun kind => kind_eqb kind Region) kinds) kinds ->
    map (region_kind_of_value rs) vs = kinds.
Proof.
  induction args as [ | s args IH]; intros kinds vs Hlive Hfault Hrest Hvs
    Hkinds.
  - simpl in Hvs. injection Hvs as <-.
    destruct kinds as [ | kind kinds]; [ reflexivity | ].
    simpl in Hrest, Hkinds. apply andb_true_iff in Hrest as [Hkind _].
    apply negb_true_iff in Hkind. rewrite Hkind in Hkinds. discriminate.
  - simpl in Hvs.
    destruct (eval_simple (state_vars (rs_state rs)) s) as [v | ] eqn:Hv;
      [ | discriminate ].
    destruct (eval_simples (state_vars (rs_state rs)) args) as [rest | ]
      eqn:Hrest_eval; [ | discriminate ].
    injection Hvs as <-.
    destruct kinds as [ | kind kinds]; [ simpl in Hkinds; discriminate | ].
    simpl in Hlive, Hfault, Hrest, Hkinds |- *.
    apply andb_true_iff in Hlive as [Hs Hlive].
    destruct (kind_eqb kind Region) eqn:Hkind.
    + apply kind_eqb_eq in Hkind. subst kind. f_equal.
      * unfold simple_open_mark in Hs. rewrite Hv in Hs.
        unfold region_kind_of_value. rewrite Hs. reflexivity.
      * apply (IH kinds rest Hlive); try assumption; try reflexivity.
        destruct (simple_usable rs s); [ exact Hfault | ].
        destruct (simple_open_mark rs s); [ exact Hfault | discriminate ].
    + assert (Husable : simple_usable rs s = true).
      { destruct (simple_usable rs s); [ reflexivity | ].
        destruct (simple_open_mark rs s); discriminate. }
      rewrite Husable in Hfault.
      injection Hkinds as Hhead Htail. f_equal.
      * unfold region_kind_of_value.
        assert (Hmark : value_open_mark rs v = false).
        { unfold simple_usable in Husable. rewrite Hv in Husable.
          destruct v; try reflexivity.
          unfold value_open_mark. rewrite Husable. reflexivity. }
        rewrite Hmark. exact Hhead.
      * exact (IH kinds rest Hlive Hfault Hrest eq_refl Htail).
Qed.

(* Two lists are as long as each other if they are once the regions
   are dropped and every region has an element in both. *)
Lemma drop_regions_lengths {A B : Type} (places : list bool) :
  ∀ (l1 : list A) (l2 : list B),
    forallb negb (skipn (length l1) places) = true ->
    forallb negb (skipn (length l2) places) = true ->
    length (drop_regions places l1) = length (drop_regions places l2) ->
    length l1 = length l2.
Proof.
  induction places as [ | place places IH]; intros l1 l2 H1 H2 Hlength;
    [ exact Hlength | ].
  destruct l1 as [ | a l1]; destruct l2 as [ | b l2]; simpl in *.
  - reflexivity.
  - apply andb_true_iff in H1 as [Hplace _]. apply negb_true_iff in Hplace.
    subst place. discriminate.
  - apply andb_true_iff in H2 as [Hplace _]. apply negb_true_iff in Hplace.
    subst place. discriminate.
  - destruct place; simpl in Hlength.
    + f_equal. exact (IH l1 l2 H1 H2 Hlength).
    + injection Hlength as Hlength. f_equal. exact (IH l1 l2 H1 H2 Hlength).
Qed.

(* The erased term gives the erased handler the arguments that are no
   regions. Where that has a step, the arguments have values, of the
   kinds of the handler, and are as many as its parameters. *)
Lemma region_apply_cont_back (rs : region_state) (s' : state)
    (ac : apply_cont) :
  symbols_fresh rs = true ->
  state_erased rs s' ->
  state_expr (rs_state rs) = Apply_cont ac ->
  simples_fault rs (ac_args ac) (region_places rs) = None ->
  simples_live rs (ac_args ac) (region_places rs) = true ->
  regions_given rs = true ->
  existsb (simple_open_mark rs) (ac_args ac) = true ->
  step s' <> Undefined ->
  region_apply_cont rs ac <> Region_undefined No_plain_step.
Proof.
  intros Hfresh
    (rconts & He & Hcalls & Hagree & Hvars & Hconts & Hstatics & Hheap
     & Hreads)
    Hexpr Hfault Hlive Hgiven Hmark.
  pose proof (symbols_fresh_expr rs Hfresh) as Hbelow.
  rewrite Hexpr in Hbelow.
  change (forallb (simple_below (rs_base rs)) (ac_args ac) = true) in Hbelow.
  pose proof (resolve_erased rs (ac_continuation ac) _ _ Hconts) as Hc.
  unfold region_places in Hfault, Hlive. rewrite Hexpr in Hfault, Hlive.
  unfold regions_given in Hgiven. cbv zeta in Hgiven.
  rewrite Hexpr in Hgiven.
  unfold region_apply_cont.
  destruct (resolve (ac_continuation ac) (state_conts (rs_state rs)))
    as [ h def_vars def_conts recursive | k | c extra
       | remaining remaining_arity return_arity return_to exn_to ]
    eqn:Hresolve;
    try (exfalso;
         pose proof (simples_fault_mark_place rs _ [] Hfault Hmark) as H;
         discriminate H).
  destruct h as [params [kinds is_exn] handler].
  unfold region_params, handler_kinds in Hfault, Hlive, Hgiven.
  cbn [hm_params_kinds] in Hfault, Hlive, Hgiven.
  cbn [hm_params_kinds].
  apply andb_true_iff in Hgiven as [Hgiven_args Hgiven_params].
  (* The continuation is one that [erase] lists. *)
  pose proof (Hagree (ac_continuation ac)) as Hk.
  destruct (lookup (ac_continuation ac) rconts) as [places | ] eqn:Hlisted.
  2: { exfalso. rewrite Hresolve in Hk.
       unfold cont_takes_no_region, handler_takes_no_region, region_params,
         handler_kinds in Hk.
       cbn [hm_params_kinds] in Hk.
       rewrite (simples_fault_mark_place rs _ _ Hfault Hmark) in Hk.
       discriminate Hk. }
  destruct Hk as (h0 & vars0 & conts0 & recursive0 & Hlookup & ->).
  unfold resolve in Hresolve. rewrite Hlookup in Hresolve.
  injection Hresolve as -> -> -> ->.
  unfold region_params, handler_kinds in Hlisted.
  cbn [hm_params_kinds] in Hlisted.
  (* The closure of the second state. *)
  inversion Hc
    as [
       | rconts0 h1 vars1 vars1' conts1 conts1' recursive1 Hh Hrecursive
           Hagree0 Hvars0 Hconts0
       |
       | ].
  subst h1 vars1 conts1 recursive1.
  match goal with
  | H : Closure _ _ _ _ = resolve _ _ |- _ => rename H into Hc'
  end.
  destruct s' as [e' vars' conts' statics' hp' rd'].
  cbn [state_expr state_vars state_conts state_statics state_heap
       state_reads] in *.
  rewrite Hexpr in He. cbn [erase] in He. unfold erase_apply_cont in He.
  rewrite Hlisted in He. subst e' hp' rd'.
  rewrite step_apply_cont. cbn [ac_args ac_continuation].
  destruct (eval_simples vars' (drop_regions _ (ac_args ac))) as [vs' | ]
    eqn:Hvs';
    [ | intros H; contradiction (H eq_refl) ].
  unfold apply_continuation. rewrite <- Hc'.
  cbn [erase_handler apply_cont_value hm_params_kinds].
  destruct (kinds_eqb (map kind_of_value vs') (drop_regions _ kinds))
    eqn:Hkinds';
    [ | intros H; contradiction (H eq_refl) ].
  destruct (bind_params (drop_regions _ params) vs' vars1') as [bound' | ]
    eqn:Hbind';
    [ | intros H; contradiction (H eq_refl) ].
  intros _.
  (* The machine of regions. *)
  destruct (eval_simples_given rs vars' (ac_args ac) Hvars _ vs' Hlive Hvs')
    as [vs Hvs].
  rewrite Hvs.
  destruct
    (eval_simples_dropped rs vars' (ac_args ac) Hvars _ vs Hbelow Hfault Hvs)
    as (vs0 & Hvs0 & Hrelated).
  rewrite Hvs' in Hvs0. injection Hvs0 as <-.
  apply kinds_eqb_eq in Hkinds'.
  rewrite (kinds_of_values_erased rs _ vs' Hrelated) in Hkinds'.
  rewrite
    (kinds_given rs (ac_args ac) kinds vs Hlive Hfault Hgiven_args Hvs
       Hkinds'),
    kinds_eqb_refl.
  assert (Hlength : length params = length vs).
  { apply (drop_regions_lengths (map (fun kind => kind_eqb kind Region) kinds)).
    - exact Hgiven_params.
    - rewrite (eval_simples_as_many _ _ vs Hvs). exact Hgiven_args.
    - rewrite (Forall2_length Hrelated).
      exact (bind_params_as_many _ vs' vars1' bound' Hbind'). }
  destruct (bind_params_of_as_many params vs def_vars Hlength) as [bound ->].
  discriminate.
Qed.

(* ======================================================================== *)
(* A step of the machine of regions                                         *)
(* ======================================================================== *)

Lemma begin_region_not_plain (rs : region_state) (x : variable)
    (kind : flambda_kind) (mark_kind : mark_kind) (body : expr) :
  begin_region rs x kind mark_kind body <> Region_undefined No_plain_step.
Proof. unfold begin_region. destruct (kind_eqb kind Region); discriminate. Qed.

Lemma end_region_not_plain (rs : region_state) (x : variable)
    (kind : flambda_kind) (ended_by : mark_kind) (region : simple)
    (body : expr) :
  end_region rs x kind ended_by region body <> Region_undefined No_plain_step.
Proof.
  unfold end_region. destruct (kind_eqb kind Value); [ | discriminate ].
  destruct (eval_simple (state_vars (rs_state rs)) region)
    as [[ | | | m | | | | | ] | ]; try discriminate.
  destruct (cut m (rs_frames rs)) as [[above below] | ]; discriminate.
Qed.

(* Where the machine of regions does what [step] does: at an expression
   at whose head it does nothing of its own, and that is not an
   [Apply_cont] one of whose arguments is a mark on the stack. *)
Lemma region_step_plain_back (rs : region_state) (s' : state) :
  state_erased rs s' ->
  symbols_fresh rs = true ->
  reads_live rs ->
  regions_given rs = true ->
  head_plain (state_expr (rs_state rs)) = true ->
  (∀ ac : apply_cont,
    state_expr (rs_state rs) = Apply_cont ac ->
    existsb (simple_open_mark rs) (ac_args ac) = false) ->
  step s' <> Undefined ->
  region_step rs <> Region_undefined No_plain_step.
Proof.
  intros Hstate Hfresh (R1 & R2 & R3) Hgiven Hhead Hmarks Hstep'.
  destruct (plain_simples_live rs R1 Hgiven Hmarks) as [Hall Htakes].
  unfold region_step. cbv zeta.
  destruct
    (simples_fault rs (evaluated_simples (state_expr (rs_state rs)))
       (region_places rs))
    as [fault | ] eqn:Hfault;
    [ intros [= Heq]; exact (simples_fault_not_plain rs _ _ _ Hfault Heq) | ].
  destruct (count_uses_with_counts rs) as [counts Hcounts]. rewrite Hcounts.
  pose proof
    (state_erased_same rs (with_counts rs counts) s'
       (same_statics_with_counts rs counts) eq_refl Hstate)
    as Hstate_c.
  assert (Hfresh_c : symbols_fresh (with_counts rs counts) = true)
    by (rewrite symbols_fresh_with_counts; exact Hfresh).
  assert (Hplain :
    (∀ (bindings : list static_binding) (body : expr),
      state_expr (rs_state rs) <> Let_static bindings body) ->
    (∀ (x : variable) (kind : flambda_kind) (d : named) (body : expr),
      state_expr (rs_state rs) = Let x kind d body ->
      named_needs_heap (state_vars (rs_state rs)) d = false) ->
    plain_step (with_counts rs counts) <> Region_undefined No_plain_step).
  { intros Hnot_static Hneeds.
    exact
      (plain_step_back (with_counts rs counts) s' Hfresh_c
         (statics_live_with_counts rs counts R2) Hstate_c Hhead Hnot_static
         Hneeds (simples_all_live_with_counts rs counts _ _ Hall)
         (values_live_all_with_counts rs counts _ R3) Htakes Hstep'). }
  assert (Hstatic :
    ∀ (bindings : list static_binding) (body : expr),
      state_expr (rs_state rs) = Let_static bindings body ->
      static_step (with_counts rs counts) <> Region_undefined No_plain_step).
  { intros bindings body He.
    exact
      (static_step_back (with_counts rs counts) s' bindings body Hfresh_c
         Hstate_c He Hstep'). }
  destruct (state_expr (rs_state rs))
    as [ x xkind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ].
  - (* A [Let]. *)
    assert (Hnot_static :
      ∀ (bindings : list static_binding) (body0 : expr),
        Let x xkind d body <> Let_static bindings body0)
      by (intros bindings body0; discriminate).
    (* A [Let] that needs the heap is not run (decision D53), which is
       not to stop because [step] has no step; of another, the test is
       what [plain_step_back] asks. *)
    destruct d as [ s | p args | | | | | ];
      try (destruct p; try discriminate Hhead);
      cbv beta iota; simpl in Hhead;
      (destruct (named_needs_heap _ _) eqn:Hneeds; [ discriminate | ]);
      (destruct (marker_of x body); try discriminate Hhead);
      cbv beta iota;
      (destruct (is_allocation _ && existsb _ _);
       [ discriminate
       | apply (Hplain Hnot_static);
         intros x0 kind0 d0 body0 [= <- <- <- <-]; exact Hneeds ]).
  - (* A [Let_static]. *)
    destruct
      (existsb (simple_local (with_counts rs counts))
         (evaluated_simples (Let_static bindings body)));
      [ discriminate | ].
    exact (Hstatic bindings body eq_refl).
  - apply Hplain; [ intros bindings body0 | intros x kind d body0 ];
      discriminate.
  - apply Hplain; [ intros bindings body0 | intros x kind d body0 ];
      discriminate.
  - apply Hplain; [ intros bindings body0 | intros x kind d body0 ];
      discriminate.
  - (* An [Apply_cont]. *)
    rewrite existsb_open_mark_with_counts, (Hmarks ac eq_refl).
    apply Hplain; [ intros bindings body0 | intros x kind d body0 ];
      discriminate.
  - apply Hplain; [ intros bindings body0 | intros x kind d body0 ];
      discriminate.
  - apply Hplain; [ intros bindings body0 | intros x kind d body0 ];
      discriminate.
Qed.

(* One step of the machine of regions, from a state that has the
   invariants and the two properties: where the second state has a
   step, the machine of regions does not stop because [step] has
   none. *)
Theorem region_step_back (rs : region_state) (s' : state) :
  related rs s' ->
  reads_live rs ->
  regions_given rs = true ->
  step s' <> Undefined ->
  region_step rs <> Region_undefined No_plain_step.
Proof.
  intros (Herased & Hinv & Hpoison & Honce) Hreads Hgiven Hstep'.
  apply andb_true_iff in Hinv as [Hfresh _].
  pose proof Hreads as (R1 & R2 & R3).
  destruct (count_uses_with_counts rs) as [counts Hcounts].
  pose proof
    (state_erased_same rs (with_counts rs counts) s'
       (same_statics_with_counts rs counts) eq_refl Herased)
    as Herased_c.
  assert (Hfresh_c : symbols_fresh (with_counts rs counts) = true)
    by (rewrite symbols_fresh_with_counts; exact Hfresh).
  unfold region_places in R1.
  destruct (state_expr (rs_state rs))
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ]
    eqn:Hexpr.
  - (* A [Let]. *)
    destruct (named_region_primitive_dec d) as [(p & args & ->) | Hnot].
    + (* Of a primitive of regions. *)
      rewrite (region_step_region_primitive rs x kind p args body Hexpr).
      destruct p; destruct args as [ | region [ | region' args]];
        try discriminate;
        first [ apply begin_region_not_plain | apply end_region_not_plain ].
    + destruct (marker_of x body) as [ | y region body' | ] eqn:Hmarker.
      * (* With nothing of regions after it. *)
        apply (region_step_plain_back rs s' Herased Hfresh Hreads Hgiven);
          [ | | exact Hstep' ].
        -- rewrite Hexpr, (head_plain_let_plain x kind d body Hnot), Hmarker.
           reflexivity.
        -- intros ac Heq. rewrite Hexpr in Heq. discriminate Heq.
      * (* With a [Local_allocation] after it. *)
        rewrite (region_step_let rs x kind d body Hexpr Hnot), Hmarker.
        destruct (simples_fault rs (simples_of_named d) []) as [fault | ]
          eqn:Hfault;
          [ intros [= Heq];
            exact (simples_fault_not_plain rs _ _ _ Hfault Heq)
          | ].
        destruct (named_needs_heap (state_vars (rs_state rs)) d) eqn:Hneeds;
          [ discriminate | ].
        rewrite Hcounts.
        apply
          (local_allocation_back (with_counts rs counts) s' x y kind d region
             body body' Hfresh_c (statics_live_with_counts rs counts R2)
             Herased_c Hexpr Hnot Hneeds);
          [ | exact Hstep' ].
        apply simples_all_live_with_counts.
        rewrite (evaluated_simples_let_plain x kind d body Hnot) in R1.
        exact (simples_live_no_place rs _ R1).
      * rewrite (region_step_let rs x kind d body Hexpr Hnot), Hmarker.
        destruct (simples_fault rs (simples_of_named d) []) as [fault | ]
          eqn:Hfault;
          [ intros [= Heq];
            exact (simples_fault_not_plain rs _ _ _ Hfault Heq)
          | destruct (named_needs_heap (state_vars (rs_state rs)) d);
            discriminate ].
  - apply (region_step_plain_back rs s' Herased Hfresh Hreads Hgiven);
      [ rewrite Hexpr; reflexivity
      | intros ac Heq; rewrite Hexpr in Heq; discriminate Heq
      | exact Hstep' ].
  - apply (region_step_plain_back rs s' Herased Hfresh Hreads Hgiven);
      [ rewrite Hexpr; reflexivity
      | intros ac Heq; rewrite Hexpr in Heq; discriminate Heq
      | exact Hstep' ].
  - apply (region_step_plain_back rs s' Herased Hfresh Hreads Hgiven);
      [ rewrite Hexpr; reflexivity
      | intros ac Heq; rewrite Hexpr in Heq; discriminate Heq
      | exact Hstep' ].
  - apply (region_step_plain_back rs s' Herased Hfresh Hreads Hgiven);
      [ rewrite Hexpr; reflexivity
      | intros ac Heq; rewrite Hexpr in Heq; discriminate Heq
      | exact Hstep' ].
  - (* An [Apply_cont]. *)
    destruct (existsb (simple_open_mark rs) (ac_args ac)) eqn:Hmark.
    + rewrite (region_step_apply_cont rs ac Hexpr Hmark).
      destruct (simples_fault rs (ac_args ac) (region_places rs))
        as [fault | ] eqn:Hfault;
        [ intros [= Heq]; exact (simples_fault_not_plain rs _ _ _ Hfault Heq)
        | ].
      rewrite Hcounts.
      apply
        (region_apply_cont_back (with_counts rs counts) s' ac Hfresh_c
           Herased_c Hexpr);
        [ | | exact Hgiven | | exact Hstep' ].
      * change (region_places (with_counts rs counts))
          with (region_places rs).
        rewrite simples_fault_with_counts. exact Hfault.
      * change (region_places (with_counts rs counts))
          with (region_places rs).
        rewrite simples_live_with_counts. unfold region_places.
        rewrite Hexpr. exact R1.
      * rewrite existsb_open_mark_with_counts. exact Hmark.
    + apply (region_step_plain_back rs s' Herased Hfresh Hreads Hgiven);
        [ rewrite Hexpr; reflexivity | | exact Hstep' ].
      intros ac0 Heq. rewrite Hexpr in Heq. injection Heq as <-. exact Hmark.
  - apply (region_step_plain_back rs s' Herased Hfresh Hreads Hgiven);
      [ rewrite Hexpr; reflexivity
      | intros ac Heq; rewrite Hexpr in Heq; discriminate Heq
      | exact Hstep' ].
  - apply (region_step_plain_back rs s' Herased Hfresh Hreads Hgiven);
      [ rewrite Hexpr; reflexivity
      | intros ac Heq; rewrite Hexpr in Heq; discriminate Heq
      | exact Hstep' ].
Qed.

(* ======================================================================== *)
(* Runs                                                                     *)
(* ======================================================================== *)

(* Where the run of the first state stops because [step] has no step,
   that of the second goes wrong. The two properties are asked of the
   states that the run of the first goes through. *)
Lemma run_no_plain_step (fuel : nat) :
  ∀ (rs : region_state) (s' : state),
    related rs s' ->
    (∀ (fuel' : nat) (rs' : region_state),
      region_run fuel' rs = Region_step rs' ->
      reads_live rs' ∧ regions_given rs' = true) ->
    region_run fuel rs = Region_undefined No_plain_step ->
    behaves s' Goes_wrong.
Proof.
  induction fuel as [ | fuel IH]; intros rs s' Hrelated Hstates Hrun;
    [ discriminate | ].
  rewrite region_run_succ in Hrun.
  pose proof (related_sim rs s' Hrelated) as Hsim.
  destruct (region_step rs) as [rs1 | k0 vs0 rs0 | fault | ] eqn:Hstep;
    [ | discriminate | | discriminate ].
  - assert (Hstates1 :
      ∀ (fuel' : nat) (rs' : region_state),
        region_run fuel' rs1 = Region_step rs' ->
        reads_live rs' ∧ regions_given rs' = true).
    { intros fuel' rs' Hrun'. apply (Hstates (S fuel') rs').
      rewrite region_run_succ, Hstep. exact Hrun'. }
    destruct Hsim
      as [ [Herased1 _]
         | [ (s1' & Hstep' & Herased1)
           | (s1' & s2' & Hstep1' & Hstep2' & Herased1) ] ].
    + exact
        (IH rs1 s' (related_step rs rs1 s' s' Hrelated Hstep Herased1)
           Hstates1 Hrun).
    + destruct
        (IH rs1 s1' (related_step rs rs1 s' s1' Hrelated Hstep Herased1)
           Hstates1 Hrun)
        as [n Hn].
      exists (S n). rewrite run_succ, Hstep'. exact Hn.
    + destruct
        (IH rs1 s2' (related_step rs rs1 s' s2' Hrelated Hstep Herased1)
           Hstates1 Hrun)
        as [n Hn].
      exists (S (S n)). rewrite run_succ, Hstep1', run_succ, Hstep2'.
      exact Hn.
  - injection Hrun as ->.
    destruct (Hstates 0 rs eq_refl) as [Hreads Hgiven].
    exists 1. rewrite run_succ.
    destruct (step s') as [s1' | k vs hp rd | ] eqn:Hstep';
      [ exfalso | exfalso | reflexivity ];
      (apply (region_step_back rs s' Hrelated Hreads Hgiven);
       [ rewrite Hstep'; discriminate | exact Hstep ]).
Qed.

(* ======================================================================== *)
(* The statement                                                            *)
(* ======================================================================== *)

(* The text of
   FlambdaRegionsCheckSoundSpec.no_plain_step_erased_statement. *)
Theorem no_plain_step_erased_run :
  ∀ (e : expr) (statics : static_env) (base : symbol) (fuel : nat),
    region_names_fresh base e statics = true ->
    erasable e statics = true ->
    (∀ (fuel' : nat) (rs : region_state),
       region_run fuel' (initial_region_state e statics base)
       = Region_step rs ->
       reads_live rs ∧ regions_given rs = true) ->
    region_run fuel (initial_region_state e statics base)
    = Region_undefined No_plain_step ->
    behaves (erased_initial_state e statics) Goes_wrong.
Proof.
  intros e statics base fuel Hfresh Herasable Hstates Hrun.
  exact
    (run_no_plain_step fuel _ _
       (related_initial e statics base Hfresh Herasable) Hstates Hrun).
Qed.
