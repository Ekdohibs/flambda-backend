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

(* The erasure of regions (FlambdaRegionsErasureSpec.v), read from the
   state that runs the erased term to the state of the machine of
   regions: calls and continuations. Nothing here needs review: no
   statement of a Spec file is proved here.

   FlambdaRegionsPlainStepProofs.v has that a call, or the application
   of a continuation, that has a step of [step] under the static
   environment that the machine of regions gives it
   (FlambdaRegionsSpec.live_statics) has one in the second state.
   Here is the other way: where the second state has one, [step] has
   one under that static environment. Nothing here is of the tests
   that the machine of regions makes before it asks [step]. As in
   FlambdaRegionsErasedReadsProofs.v it is of values that reach only
   symbols that may be evaluated
   (FlambdaRegionsErasedReadsProofs.erased_back): the callee and the
   arguments of the call, and the values that the continuation applied
   gives on (FlambdaRegionsCheckedSpec.cont_spine_values).

   - [shape_of_call_back]: what the call is. A partial application
     that the call makes reaches only symbols that may be evaluated.
   - [call_code_back]: a full call that has a step has one whatever
     the values are, if they are of the same kinds and as many.
   - [apply_cont_value_back] and [call_function_back]: a continuation
     that takes no region, and a call. *)

From Stdlib Require Import ZArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaValueSpec FlambdaPrimitiveModel FlambdaPrimitiveSpec
  FlambdaPrimitiveProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaClosuresSpec FlambdaClosuresProofs FlambdaLiftProofs
  FlambdaRegionsSpec FlambdaRegionsProofs FlambdaRegionsInvariantSpec
  FlambdaRegionsErasureSpec FlambdaRegionsReadsProofs
  FlambdaRegionsErasedProofs FlambdaRegionsPlainStepProofs
  FlambdaRegionsErasureProofs FlambdaRegionsCheckedSpec
  FlambdaRegionsErasedReadsProofs.
Import ListNotations.

(* The static environment that [step] is given is not unfolded: the
   rules of FlambdaRegionsErasureSpec.value_erased name it. *)
#[local] Opaque live_statics.

(* ======================================================================== *)
(* Values related both ways                                                 *)
(* ======================================================================== *)

Lemma back_values_usable (rs : region_state) (vs' vs : list value) :
  Forall2 (erased_back rs) vs' vs ->
  Forall2 (value_erased_if_usable rs) vs vs'.
Proof.
  intros H.
  exact
    (values_erased_if_usable_intro rs vs vs'
       (proj1 (back_values rs vs' vs H))).
Qed.

Lemma back_values_kinds (rs : region_state) (vs' vs : list value) :
  Forall2 (erased_back rs) vs' vs ->
  map kind_of_value vs' = map kind_of_value vs.
Proof.
  intros H.
  exact (kinds_of_values_erased rs vs vs' (proj1 (back_values rs vs' vs H))).
Qed.

(* Values that are related where they may be evaluated, and that reach
   only symbols that may be evaluated. *)
Lemma usable_values_back (rs : region_state) (vs vs' : list value) :
  Forall2 (value_erased_if_usable rs) vs vs' ->
  forallb (value_all (live_symbol rs)) vs = true ->
  Forall2 (erased_back rs) vs' vs.
Proof.
  induction 1 as [ | v v' vs vs' [_ Hv] _ IH]; simpl; intros Hall;
    [ constructor | ].
  apply andb_true_iff in Hall as [Hhead Htail].
  constructor; [ | exact (IH Htail) ].
  exact (conj (Hv (value_all_live_usable rs v Hhead)) Hhead).
Qed.

(* Parameters that can be bound to some values can be bound to as many
   others. *)
Lemma bind_params_back {R : value -> value -> Prop} (params : list variable) :
  ∀ (vs vs' : list value) (vars vars' bound' : env),
    Forall2 R vs vs' ->
    bind_params params vs' vars' = Some bound' ->
    ∃ bound : env, bind_params params vs vars = Some bound.
Proof.
  induction params as [ | p params IH]; intros vs vs' vars vars' bound' Hvs;
    destruct Hvs as [ | v v' vs vs' _ Hvs]; simpl; try discriminate.
  - intros _. eauto.
  - exact (IH vs vs' _ _ bound' Hvs).
Qed.

(* A partial application that a call makes reaches only symbols that
   may be evaluated. *)
Definition partial_shape_live (rs : region_state) (shape : call_shape)
    : Prop :=
  match shape with
  | Partial_call partial => value_all (live_symbol rs) partial = true
  | Full_call _ _ _ _ | Over_call _ _ _ _ _ _ => True
  end.

(* ======================================================================== *)
(* Calls                                                                    *)
(* ======================================================================== *)

Section Calls.

Variable rs : region_state.
Variable statics' : static_env.
Hypothesis Hfresh : symbols_fresh rs = true.
Hypothesis Hstatics : statics_erased rs statics'.
Hypothesis Hlive :
  forallb (fun '(_, allocated) => allocated_all (live_symbol rs) allocated)
    (static_values (live_statics rs))
  = true.

(* The fields of a tuple. *)
Lemma tuple_fields_back (size : nat) (tuple' tuple : value)
    (fields' : list value) :
  erased_back rs tuple' tuple ->
  tuple_fields statics' size tuple' = Some fields' ->
  ∃ fields : list value,
    tuple_fields (live_statics rs) size tuple = Some fields
    ∧ Forall2 (erased_back rs) fields' fields.
Proof.
  intros Htuple. unfold tuple_fields.
  exact
    (tuple_fields_from_related (erased_back rs) statics' (live_statics rs)
       size tuple' tuple
       (erased_back_respects rs statics' Hfresh Hstatics Hlive) Htuple size 0
       fields').
Qed.

(* The arguments as the code takes them. *)
Lemma arguments_for_code_back (c : code) (vs' vs : list value)
    (args_arity : arity) (ws' : list value) (args_arity' : arity) :
  Forall2 (erased_back rs) vs' vs ->
  arguments_for_code statics' (erase_code c) vs' args_arity
  = Some (ws', args_arity') ->
  ∃ ws : list value,
    arguments_for_code (live_statics rs) c vs args_arity
    = Some (ws, args_arity')
    ∧ Forall2 (erased_back rs) ws' ws.
Proof.
  intros Hvs. unfold arguments_for_code.
  rewrite erase_code_is_tupled, erase_code_params_arity.
  destruct (code_is_tupled c);
    [ | intros [= <- <-]; exists vs; split; [ reflexivity | exact Hvs ] ].
  unfold detuple.
  pose proof (back_values_kinds rs vs' vs Hvs) as Hkinds.
  destruct Hvs as [ | tuple' tuple rest' rest Htuple Hrest]; [ discriminate | ].
  destruct args_arity as [ | first rest_arity]; [ discriminate | ].
  rewrite Hkinds. destruct (negb _); [ discriminate | ].
  destruct
    (tuple_fields statics' (cardinal_unarized (code_params_arity c)) tuple')
    as [fields' | ] eqn:Hfields'; [ | discriminate ].
  destruct (tuple_fields_back _ tuple' tuple fields' Htuple Hfields')
    as (fields & -> & Hfields).
  intros [= <- <-]. exists (fields ++ rest).
  split; [ reflexivity | apply Forall2_app; assumption ].
Qed.

(* What the call of a closure is in the second state, the call of the
   closure is on the machine of regions. *)
Lemma shape_of_closure_call_back (closure' closure : value) (kind : call_kind)
    (vs' vs : list value) (args_arity return_arity : arity)
    (shape' : call_shape) :
  erased_back rs closure' closure ->
  Forall2 (erased_back rs) vs' vs ->
  shape_of_closure_call statics' closure' kind vs' args_arity return_arity
  = Some shape' ->
  ∃ shape : call_shape,
    shape_of_closure_call (live_statics rs) closure kind vs args_arity
      return_arity
    = Some shape
    ∧ shape_erased rs shape shape'
    ∧ partial_shape_live rs shape.
Proof.
  pose proof Hstatics as (Hcode & Hreturns & _ & _).
  intros [Hclosure Hclosure_live] Hvs. unfold shape_of_closure_call.
  destruct (closure_of statics' closure') as [[called slots'] | ] eqn:Hc';
    [ | discriminate ].
  destruct
    (closure_of_back rs statics' Hfresh Hstatics closure' closure called
       slots' Hclosure Hc')
    as (slots & -> & _).
  destruct
    (match kind with
     | Direct named => Pos.eqb called named
     | Indirect => true
     end);
    [ | discriminate ].
  rewrite live_statics_code, Hcode.
  destruct (lookup called (static_code (state_statics (rs_state rs))))
    as [c | ] eqn:Hcalled;
    [ | discriminate ].
  simpl option_map. cbv iota zeta.
  destruct (arguments_for_code statics' (erase_code c) vs' args_arity)
    as [[ws' args_arity'] | ] eqn:Hargs'; [ | discriminate ].
  destruct
    (arguments_for_code_back c vs' vs args_arity ws' args_arity' Hvs Hargs')
    as (ws & -> & Hws).
  simpl. unfold shape_of_code_call. cbv zeta.
  rewrite erase_code_params_arity, erase_code_result_arity,
    erase_code_is_tupled, (back_values_kinds rs ws' ws Hws).
  pose proof (Hreturns called c Hcalled) as Hc_returns.
  pose proof (back_values_usable rs ws' ws Hws) as Hws_usable.
  destruct (Nat.eqb (length args_arity') (length (code_params_arity c))).
  - intros [= <-]. eexists. split; [ reflexivity | ]. split; [ | exact I ].
    apply Full_call_erased; assumption.
  - destruct
      (negb
         (kinds_eqb (map kind_of_value ws) (unarize args_arity')
          && partial_or_over_application args_arity' return_arity
               (code_params_arity c) (code_result_arity c)));
      [ discriminate | ].
    destruct
      (Forall2_firstn_skipn (value_erased_if_usable rs)
         (cardinal_unarized (code_params_arity c)) ws ws' Hws_usable)
      as [Hfirst Hrest].
    destruct (Nat.ltb (length (code_params_arity c)) (length args_arity')).
    + intros [= <-]. eexists. split; [ reflexivity | ]. split; [ | exact I ].
      apply Over_call_erased; assumption.
    + destruct (code_is_tupled c); [ discriminate | ].
      intros [= <-]. eexists. split; [ reflexivity | ]. split.
      * apply Partial_call_erased, Erased_partial_application;
          [ intros _; exact Hclosure | exact Hws_usable ].
      * simpl. rewrite Hclosure_live.
        exact (proj2 (back_values rs ws' ws Hws)).
Qed.

(* The same of a call. A partial application that reaches only symbols
   that may be evaluated keeps a closure and arguments that are
   related. *)
Lemma shape_of_call_back (callee' callee : value) (kind : call_kind)
    (vs' vs : list value) (args_arity return_arity : arity)
    (shape' : call_shape) :
  erased_back rs callee' callee ->
  Forall2 (erased_back rs) vs' vs ->
  shape_of_call statics' callee' kind vs' args_arity return_arity
  = Some shape' ->
  ∃ shape : call_shape,
    shape_of_call (live_statics rs) callee kind vs args_arity return_arity
    = Some shape
    ∧ shape_erased rs shape shape'
    ∧ partial_shape_live rs shape.
Proof.
  intros Hcallee Hvs. unfold shape_of_call.
  destruct kind as [named | ];
    [ apply shape_of_closure_call_back; assumption | ].
  pose proof Hcallee as [Hv Hall].
  pose proof (partial_application_erased rs callee callee' Hv) as Hpartial.
  destruct (partial_application_of callee)
    as [[[closure applied] applied_arity] | ] eqn:Hpa.
  - destruct Hpartial as (closure' & applied' & -> & Hclosure & Happlied).
    destruct callee; try discriminate Hpa.
    simpl in Hpa. injection Hpa as -> -> ->.
    simpl in Hall. apply andb_true_iff in Hall as [Hclosure_live Happlied_live].
    simpl.
    destruct args_arity as [ | component args_arity]; [ discriminate | ].
    rewrite (back_values_kinds rs vs' vs Hvs).
    destruct (kinds_eqb _ _); [ | discriminate ].
    apply shape_of_closure_call_back.
    + exact
        (conj (Hclosure (value_all_live_usable rs _ Hclosure_live))
           Hclosure_live).
    + apply Forall2_app;
        [ exact (usable_values_back rs _ _ Happlied Happlied_live)
        | exact Hvs ].
  - rewrite Hpartial. apply shape_of_closure_call_back; assumption.
Qed.

(* A full call that has a step in the second state has one under the
   static environment of the frames: the tests are of the code and of
   the kinds of the arguments, and the parameters are as many as the
   arguments. *)
Lemma call_code_back (closure closure' : value) (c : code)
    (vs vs' : list value) (args_arity return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value) (hp : heap)
      (rd : read_record) :
  Forall2 (value_erased_if_usable rs) vs vs' ->
  call_code closure' (erase_code c) vs' args_arity return_arity return_to'
    exn_to' statics' hp rd
  <> Undefined ->
  call_code closure c vs args_arity return_arity return_to exn_to
    (live_statics rs) hp rd
  <> Undefined.
Proof.
  intros Hvs. unfold call_code.
  rewrite erase_code_body, erase_binds_statics, erase_code_params_arity,
    erase_code_result_arity, erase_code_params, erase_code_my_closure,
    erase_code_return_continuation, erase_code_exn_continuation,
    (kinds_of_values_if_usable rs vs vs' Hvs).
  destruct (binds_statics (code_body c)); [ exact (fun H => H) | ].
  destruct
    (negb
       (kinds_eqb (map kind_of_value vs) (unarize args_arity)
        && kinds_per_parameter_eqb (unarize_per_parameter args_arity)
             (unarize_per_parameter (code_params_arity c))
        && arity_eqb return_arity (code_result_arity c)));
    [ exact (fun H => H) | ].
  destruct (bind_params (code_params c) vs' [(code_my_closure c, closure')])
    as [bound' | ] eqn:Hbind';
    [ | intros H; contradiction (H eq_refl) ].
  destruct
    (bind_params_back (code_params c) vs vs' [(code_my_closure c, closure)] _
       bound' Hvs Hbind')
    as [bound ->].
  intros _. discriminate.
Qed.

(* A call, given what the two states do when the return continuation is
   applied to a partial application. *)
Lemma call_function_back_from (callee' callee : value) (kind : call_kind)
    (vs' vs : list value) (args_arity return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value) (hp : heap)
      (rd : read_record) :
  erased_back rs callee' callee ->
  Forall2 (erased_back rs) vs' vs ->
  (∀ partial' partial : value,
    erased_back rs partial' partial ->
    apply_cont_value return_to' [partial'] statics' hp rd <> Undefined ->
    apply_cont_value return_to [partial] (live_statics rs) hp rd <> Undefined)
      ->
  call_function callee' kind vs' args_arity return_arity return_to' exn_to'
    statics' hp rd
  <> Undefined ->
  call_function callee kind vs args_arity return_arity return_to exn_to
    (live_statics rs) hp rd
  <> Undefined.
Proof.
  intros Hcallee Hvs Hpartial. unfold call_function.
  destruct (shape_of_call statics' callee' kind vs' args_arity return_arity)
    as [shape' | ] eqn:Hshape';
    [ | intros H; contradiction (H eq_refl) ].
  destruct
    (shape_of_call_back callee' callee kind vs' vs args_arity return_arity
       shape' Hcallee Hvs Hshape')
    as (shape & -> & Hshapes & Hshape_live).
  destruct Hshapes
    as [ closure closure' c ws ws' args_arity' Hclosure Hc Hws
       | closure closure' c ws ws' remaining remaining' args_arity'
           remaining_arity Hclosure Hc Hws Hremaining
       | partial partial' Hp ];
    simpl.
  - apply call_code_back. exact Hws.
  - rewrite erase_code_result_arity. apply call_code_back. exact Hws.
  - simpl in Hshape_live. apply Hpartial. exact (conj Hp Hshape_live).
Qed.

(* Applying a continuation that takes no region. What is left of an
   over-application calls the value that it is given with the
   arguments that remain: they are among the values that the
   continuation gives on. *)
Lemma apply_cont_value_back (c c' : cont_value) :
  cont_erased rs c c' ->
  cont_takes_no_region c ->
  forallb (value_all (live_symbol rs)) (cont_spine_values c) = true ->
  ∀ (vs' vs : list value) (hp : heap) (rd : read_record),
    Forall2 (erased_back rs) vs' vs ->
    apply_cont_value c' vs' statics' hp rd <> Undefined ->
    apply_cont_value c vs (live_statics rs) hp rd <> Undefined.
Proof.
  intros Hc.
  induction Hc
    as [ k
       | rconts [params metadata handler] vars vars' conts conts' recursive
           Hh Hrecursive Hagree Hvars Hconts
       | c c' extra extra' Htakes Hc IH Hextra
       | remaining remaining' remaining_arity return_arity return_to
           return_to' exn_to exn_to' Hremaining Hreturn_takes Hexn_takes
           Hreturn IHreturn Hexn _ ];
    intros Htop Hspine vs' vs hp rd Hvs.
  - simpl. intros _. discriminate.
  - simpl in Htop.
    rewrite (erase_handler_no_region _ params metadata handler Htop).
    cbn [apply_cont_value].
    rewrite (back_values_kinds rs vs' vs Hvs).
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
      [ | exact (fun H => H) ].
    destruct (bind_params params vs' vars') as [bound' | ] eqn:Hbind';
      [ | intros H; contradiction (H eq_refl) ].
    destruct
      (bind_params_back params vs vs' vars vars' bound'
         (back_values_usable rs vs' vs Hvs) Hbind')
      as [bound ->].
    intros _. discriminate.
  - cbn [apply_cont_value]. cbn [cont_spine_values] in Hspine.
    rewrite forallb_app in Hspine.
    apply andb_true_iff in Hspine as [Hextra_live Hspine].
    apply (IH Htakes Hspine).
    apply Forall2_app;
      [ exact Hvs
      | exact (usable_values_back rs extra extra' Hextra Hextra_live) ].
  - rewrite !apply_cont_value_over_application.
    cbn [cont_spine_values] in Hspine.
    rewrite forallb_app in Hspine.
    apply andb_true_iff in Hspine as [Hremaining_live Hspine].
    destruct Hvs as [ | v' v vs' vs Hv Hvs]; [ exact (fun H => H) | ].
    destruct Hvs as [ | w' w vs' vs Hw Hvs]; [ | exact (fun H => H) ].
    apply call_function_back_from.
    + exact Hv.
    + exact
        (usable_values_back rs remaining remaining' Hremaining
           Hremaining_live).
    + intros partial' partial Hpartial.
      apply (IHreturn Hreturn_takes Hspine).
      constructor; [ exact Hpartial | constructor ].
Qed.

(* A call. *)
Lemma call_function_back (callee' callee : value) (kind : call_kind)
    (vs' vs : list value) (args_arity return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value) (hp : heap)
      (rd : read_record) :
  erased_back rs callee' callee ->
  Forall2 (erased_back rs) vs' vs ->
  cont_erased rs return_to return_to' ->
  cont_takes_no_region return_to ->
  forallb (value_all (live_symbol rs)) (cont_spine_values return_to) = true ->
  call_function callee' kind vs' args_arity return_arity return_to' exn_to'
    statics' hp rd
  <> Undefined ->
  call_function callee kind vs args_arity return_arity return_to exn_to
    (live_statics rs) hp rd
  <> Undefined.
Proof.
  intros Hcallee Hvs Hreturn Htakes Hspine.
  apply call_function_back_from; try assumption.
  intros partial' partial Hpartial.
  apply (apply_cont_value_back return_to return_to' Hreturn Htakes Hspine).
  constructor; [ exact Hpartial | constructor ].
Qed.

End Calls.
