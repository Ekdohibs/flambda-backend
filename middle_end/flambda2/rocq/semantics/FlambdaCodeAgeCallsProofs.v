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

(* Proofs of the statements of FlambdaCodeAgeCallsSpec.v. Nothing here
   needs review. *)

From Stdlib Require Import PArith ZArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaStubCallsSpec FlambdaCodeAgeCallsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaCodeAgeRelation FlambdaCodeAgeRelationSpec
  FlambdaCodeAgeRelationProofs.
Import ListNotations.

(* ======================================================================== *)
(* A machine given by its step                                              *)
(* ======================================================================== *)

Section Machine.

Variable machine_step : state -> step_result.

(* The behaviours of what a step or a run comes to. *)
Definition result_behaves_by (result : step_result) (b : behaviour) : Prop :=
  match result with
  | Step s => behaves_by machine_step s b
  | Exit k vs hp rd => b = Exits k vs hp
  | Undefined => b = Goes_wrong
  end.

(* The behaviours of a state are those of what its first step comes
   to. *)
Lemma step_behaves_by (s : state) (b : behaviour) :
  behaves_by machine_step s b <-> result_behaves_by (machine_step s) b.
Proof.
  destruct b as [k vs | | ]; simpl; split.
  - intros [[ | n] [rd Hrun]]; [ discriminate | ]. simpl in Hrun.
    destruct (machine_step s); simpl; [ eauto | congruence | discriminate ].
  - destruct (machine_step s) eqn:Hstep; simpl.
    + intros [n [rd Hrun]]. exists (S n), rd. simpl. rewrite Hstep.
      exact Hrun.
    + intros [= <- <- <-]. exists 1. eexists. simpl. rewrite Hstep.
      reflexivity.
    + discriminate.
  - intros [[ | n] Hrun]; [ discriminate | ]. simpl in Hrun.
    destruct (machine_step s); simpl; [ eauto | discriminate | reflexivity ].
  - destruct (machine_step s) eqn:Hstep; simpl.
    + intros [n Hrun]. exists (S n). simpl. rewrite Hstep. exact Hrun.
    + discriminate.
    + intros _. exists 1. simpl. rewrite Hstep. reflexivity.
  - intros Hdiv. destruct (machine_step s) eqn:Hstep; simpl.
    + intros n. specialize (Hdiv (S n)). simpl in Hdiv.
      rewrite Hstep in Hdiv. exact Hdiv.
    + destruct (Hdiv 1) as [s' Hrun]. simpl in Hrun.
      rewrite Hstep in Hrun. discriminate.
    + destruct (Hdiv 1) as [s' Hrun]. simpl in Hrun.
      rewrite Hstep in Hrun. discriminate.
  - destruct (machine_step s) eqn:Hstep; simpl;
      [ | discriminate | discriminate ].
    intros Hdiv [ | n]; simpl; [ eauto | ]. rewrite Hstep. apply Hdiv.
Qed.

(* The same for a run of any length. *)
Lemma run_behaves_by (n : nat) :
  ∀ (s : state) (b : behaviour),
    behaves_by machine_step s b
    <-> result_behaves_by (run_by machine_step n s) b.
Proof.
  induction n as [ | n IH]; intros s b; [ reflexivity | ].
  rewrite step_behaves_by. simpl.
  destruct (machine_step s); simpl; [ apply IH | reflexivity | reflexivity ].
Qed.

Lemma refines_by_refl (s : state) : refines_by machine_step s s.
Proof.
  split; [ intros k vs hp Hb; exact Hb | intros b _ Hb; left; exact Hb ].
Qed.

Lemma refines_by_trans (s s' s'' : state) :
  refines_by machine_step s s' ->
  refines_by machine_step s' s'' ->
  refines_by machine_step s s''.
Proof.
  intros [Hexits Hends] [Hexits' Hends']. split.
  - intros k vs hp Hb. apply Hexits', Hexits, Hb.
  - intros b Hterm Hb.
    destruct (Hends' b Hterm Hb) as [Hb' | Hwrong];
      [ exact (Hends b Hterm Hb') | ].
    destruct (Hends Goes_wrong ltac:(discriminate) Hwrong); right; assumption.
Qed.

Lemma result_refines_by_refl (result : step_result) :
  result_refines_by machine_step result result.
Proof.
  destruct result; simpl; [ apply refines_by_refl | auto | exact I ].
Qed.

Lemma result_refines_by_trans (result result' result'' : step_result) :
  result_refines_by machine_step result result' ->
  result_refines_by machine_step result' result'' ->
  result_refines_by machine_step result result''.
Proof.
  destruct result as [s | k vs hp rd | ], result' as [s' | k' vs' hp' rd' | ],
    result'' as [s'' | k'' vs'' hp'' rd'' | ]; simpl; intros H H';
    try contradiction; try exact I.
  - exact (refines_by_trans s s' s'' H H').
  - destruct H as [-> [-> ->]]. exact H'.
Qed.

(* Two states whose first steps are so related: the second refines the
   first. *)
Lemma result_refines_by_step (s s' : state) :
  result_refines_by machine_step (machine_step s) (machine_step s') ->
  refines_by machine_step s s'.
Proof.
  intros H. split.
  - intros k vs hp Hb. apply step_behaves_by in Hb. apply step_behaves_by.
    destruct (machine_step s) as [t | k0 vs0 hp0 rd0 | ],
      (machine_step s') as [t' | k1 vs1 hp1 rd1 | ]; simpl in *;
      try contradiction; try discriminate.
    + exact (proj1 H k vs hp Hb).
    + destruct H as [<- [<- <-]]. exact Hb.
  - intros b Hterm Hb. apply step_behaves_by in Hb.
    rewrite (step_behaves_by s b), (step_behaves_by s Goes_wrong).
    destruct (machine_step s) as [t | k0 vs0 hp0 rd0 | ],
      (machine_step s') as [t' | k1 vs1 hp1 rd1 | ]; simpl in *;
      try contradiction; try (right; reflexivity).
    + exact (proj2 H b Hterm Hb).
    + left. destruct H as [-> [-> ->]]. exact Hb.
Qed.

End Machine.

Theorem run_by_step : run_by_step_statement.
Proof.
  intros fuel. induction fuel as [ | fuel IH]; intros s; simpl;
    [ reflexivity | ].
  destruct (step s); [ apply IH | reflexivity | reflexivity ].
Qed.

Lemma behaves_by_step (s : state) (b : behaviour) :
  behaves_by step s b <-> behaves s b.
Proof.
  destruct b as [k vs hp | | ]; simpl; split.
  - intros [fuel [rd H]]. exists fuel, rd. rewrite <- run_by_step. exact H.
  - intros [fuel [rd H]]. exists fuel, rd. rewrite run_by_step. exact H.
  - intros [fuel H]. exists fuel. rewrite <- run_by_step. exact H.
  - intros [fuel H]. exists fuel. rewrite run_by_step. exact H.
  - intros H fuel. destruct (H fuel) as [s' H']. exists s'.
    rewrite <- run_by_step. exact H'.
  - intros H fuel. destruct (H fuel) as [s' H']. exists s'.
    rewrite run_by_step. exact H'.
Qed.

Theorem refines_by_step : refines_by_step_statement.
Proof.
  intros source target. split; intros [Hexits Hends]; split.
  - intros k vs hp H. apply behaves_by_step, Hexits, behaves_by_step, H.
  - intros b Hterm H. apply behaves_by_step in H.
    destruct (Hends b Hterm H) as [H' | H']; [ left | right ];
      apply behaves_by_step; exact H'.
  - intros k vs hp H. apply behaves_by_step, Hexits, behaves_by_step, H.
  - intros b Hterm H. apply behaves_by_step in H.
    destruct (Hends b Hterm H) as [H' | H']; [ left | right ];
      apply behaves_by_step; exact H'.
Qed.

(* ======================================================================== *)
(* The machine that is given a code age relation                            *)
(* ======================================================================== *)

(* Its step at a direct call. *)
Lemma step_aged_apply_direct (age_of : static_env -> Code_age_relation.t)
    (ap : apply_expr) (named : code_id) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) :
  ap_call_kind ap = Direct named ->
  step_aged age_of (State (Apply ap) vars conts statics hp rd)
  = match
      eval_simple vars (ap_callee ap), eval_simples vars (ap_args ap),
      eval_simples vars (exn_extra_simples ap)
    with
    | Some callee, Some vs, Some extra =>
      run_call
        (fun partial =>
           apply_cont_value (resolve (ap_continuation ap) conts) [partial]
             statics hp rd)
        (shape_of_aged_direct_call (age_of statics) statics callee named vs
           (ap_args_arity ap) (ap_return_arity ap))
        (ap_return_arity ap) (resolve (ap_continuation ap) conts)
        (with_extra_args (resolve (ap_exn_continuation ap) conts) extra)
        statics hp rd
    | _, _, _ => Undefined
    end.
Proof. intros Hkind. unfold step_aged. rewrite Hkind. reflexivity. Qed.

(* A direct call that [step] gives a shape has that shape on the
   machine: its callee is a closure of the code named. *)
Lemma shape_of_aged_direct_call_extends (rel : Code_age_relation.t)
    (statics : static_env) (closure : value) (named : code_id)
    (vs : list value) (args_arity return_arity : arity) :
  shape_of_closure_call statics closure (Direct named) vs args_arity
    return_arity
  <> None ->
  shape_of_aged_direct_call rel statics closure named vs args_arity
    return_arity
  = shape_of_closure_call statics closure (Direct named) vs args_arity
      return_arity.
Proof.
  unfold shape_of_aged_direct_call, shape_of_closure_call.
  destruct (closure_of statics closure) as [[called slots] | ];
    [ | intros Hdefined; exfalso; apply Hdefined; reflexivity ].
  destruct (Pos.eqb_spec called named) as [-> | Hne]; intros Hdefined.
  - rewrite same_or_newer_b_refl.
    destruct (lookup named (static_code statics)); reflexivity.
  - exfalso. apply Hdefined.
    destruct (lookup called (static_code statics)); reflexivity.
Qed.

Theorem step_aged_extends : step_aged_extends_statement.
Proof.
  intros age_of [e vars conts statics hp rd] Hdefined.
  destruct e as [ | | | | ap | | | ]; try reflexivity.
  destruct (ap_call_kind ap) as [named | | code_ids] eqn:Hkind;
    [ | unfold step_aged; rewrite Hkind; reflexivity .. ].
  rewrite (step_aged_apply_direct age_of ap named vars conts statics hp rd
             Hkind).
  rewrite step_apply in *. rewrite Hkind in *.
  destruct (eval_simple vars (ap_callee ap)) as [callee | ];
    [ | reflexivity ].
  destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | reflexivity ].
  destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
    [ | reflexivity ].
  unfold call_function in *. rewrite shape_of_call_direct in *.
  destruct (shape_of_closure_call statics callee (Direct named) vs
              (ap_args_arity ap) (ap_return_arity ap))
    as [shape | ] eqn:Hshape; [ | exfalso; apply Hdefined; reflexivity ].
  rewrite shape_of_aged_direct_call_extends; rewrite Hshape;
    [ reflexivity | discriminate ].
Qed.

Lemma shape_of_aged_direct_call_empty (statics : static_env)
    (closure : value) (named : code_id) (vs : list value)
    (args_arity return_arity : arity) :
  shape_of_aged_direct_call Code_age_relation.empty statics closure named vs
    args_arity return_arity
  = shape_of_closure_call statics closure (Direct named) vs args_arity
      return_arity.
Proof.
  unfold shape_of_aged_direct_call, shape_of_closure_call.
  destruct (closure_of statics closure) as [[called slots] | ];
    [ | reflexivity ].
  rewrite same_or_newer_b_empty.
  destruct (Pos.eqb_spec called named) as [-> | Hne].
  - destruct (lookup named (static_code statics)); reflexivity.
  - destruct (lookup called (static_code statics)); reflexivity.
Qed.

Theorem step_aged_empty : step_aged_empty_statement.
Proof.
  intros [e vars conts statics hp rd].
  destruct e as [ | | | | ap | | | ]; try reflexivity.
  destruct (ap_call_kind ap) as [named | | code_ids] eqn:Hkind;
    [ | unfold step_aged; rewrite Hkind; reflexivity .. ].
  rewrite (step_aged_apply_direct _ ap named vars conts statics hp rd Hkind).
  cbv beta. rewrite step_apply, Hkind.
  destruct (eval_simple vars (ap_callee ap)) as [callee | ];
    [ | reflexivity ].
  destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | reflexivity ].
  destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
    [ | reflexivity ].
  unfold call_function.
  rewrite shape_of_call_direct, shape_of_aged_direct_call_empty.
  reflexivity.
Qed.

(* A run that [step] does not make undefined is the run of the
   machine. *)
Lemma run_aged_extends (age_of : static_env -> Code_age_relation.t)
    (fuel : nat) :
  ∀ s : state,
    run fuel s <> Undefined -> run_aged age_of fuel s = run fuel s.
Proof.
  unfold run_aged.
  induction fuel as [ | fuel IH]; intros s Hdefined; [ reflexivity | ].
  simpl in *.
  assert (Hstep : step s <> Undefined).
  { intros Hstep. rewrite Hstep in Hdefined. apply Hdefined. reflexivity. }
  rewrite (step_aged_extends age_of s Hstep).
  destruct (step s); [ apply IH; exact Hdefined | reflexivity | reflexivity ].
Qed.

Theorem behaves_aged_extends : behaves_aged_extends_statement.
Proof.
  intros age_of s b Hb H.
  destruct b as [k vs hp | | ]; [ | contradiction | ].
  - destruct H as [fuel [rd H]]. exists fuel, rd.
    change (run_aged age_of fuel s = Exit k vs hp rd).
    rewrite run_aged_extends; [ exact H | rewrite H; discriminate ].
  - intros fuel. destruct (H fuel) as [s' H']. exists s'.
    change (run_aged age_of fuel s = Step s').
    rewrite run_aged_extends; [ exact H' | rewrite H'; discriminate ].
Qed.

(* ======================================================================== *)
(* A direct call may name a newer version                                   *)
(* ======================================================================== *)

Lemma code_refines_for_refl (age_of : static_env -> Code_age_relation.t)
    (statics : static_env) (v : value) (c : code) :
  code_refines_for age_of statics v c c.
Proof.
  repeat split. intros vs args_arity return_arity return_to exn_to hp rd.
  apply result_refines_by_refl.
Qed.

Lemma code_refines_for_trans (age_of : static_env -> Code_age_relation.t)
    (statics : static_env) (v : value) (c c' c'' : code) :
  code_refines_for age_of statics v c c' ->
  code_refines_for age_of statics v c' c'' ->
  code_refines_for age_of statics v c c''.
Proof.
  intros [Hparams [Hresult [Htupled Hcalls]]]
    [Hparams' [Hresult' [Htupled' Hcalls']]].
  split; [ congruence | ]. split; [ congruence | ]. split; [ congruence | ].
  intros vs args_arity return_arity return_to exn_to hp rd.
  exact (result_refines_by_trans _ _ _ _
           (Hcalls vs args_arity return_arity return_to exn_to hp rd)
           (Hcalls' vs args_arity return_arity return_to exn_to hp rd)).
Qed.

(* Down the versions from [newer] to [named], link by link: the code of
   [newer] refines that of [named] for the closure. *)
Lemma versions_refine (age_of : static_env -> Code_age_relation.t)
    (statics : static_env) (v : value) :
  sound_for age_of statics v ->
  ∀ newer named : code_id,
    same_or_newer (age_of statics) newer named ->
    closure_at_least (age_of statics) statics v newer ->
    versions_bound (age_of statics) statics newer named ->
    ∀ c c' : code,
      lookup named (static_code statics) = Some c ->
      lookup newer (static_code statics) = Some c' ->
      code_refines_for age_of statics v c c'.
Proof.
  intros Hsound newer named H.
  induction H as [id | id older id' Holder H IH];
    intros Hclosure Hbound c c' Hc Hc'.
  - rewrite Hc in Hc'. injection Hc' as <-. apply code_refines_for_refl.
  - destruct (Hbound older (same_or_newer_older _ id older Holder) H)
      as [c_older Hc_older].
    apply (code_refines_for_trans age_of statics v c c_older c').
    + apply IH; [ | | exact Hc | exact Hc_older ].
      * destruct Hclosure as [called [slots [Hcalled Hge]]].
        exists called, slots. split; [ exact Hcalled | ].
        exact (same_or_newer_trans _ called id older Hge
                 (same_or_newer_older _ id older Holder)).
      * intros between Hfrom Hto. apply Hbound; [ | exact Hto ].
        exact (Newer_version _ id older between Holder Hfrom).
    + exact (Hsound id older c_older c' Hclosure Holder Hc_older Hc').
Qed.

(* A call of a closure by one code and by another that refines it for
   the closure: the two calls have one shape, and the second refines the
   first. *)
Lemma run_call_code_refines (age_of : static_env -> Code_age_relation.t)
    (statics : static_env) (callee : value) (c c' : code) (vs : list value)
    (args_arity return_arity : arity) (return_to exn_to : cont_value)
    (hp : heap) (rd : read_record) (return_partial : value -> step_result) :
  code_refines_for age_of statics callee c c' ->
  result_refines_aged age_of
    (run_call return_partial
       (let* '(vs', args_arity') :=
          arguments_for_code statics c vs args_arity
        in
        shape_of_code_call callee c vs' args_arity' return_arity)
       return_arity return_to exn_to statics hp rd)
    (run_call return_partial
       (let* '(vs', args_arity') :=
          arguments_for_code statics c' vs args_arity
        in
        shape_of_code_call callee c' vs' args_arity' return_arity)
       return_arity return_to exn_to statics hp rd).
Proof.
  intros [Hparams [Hresult [Htupled Hcalls]]].
  unfold arguments_for_code. rewrite Htupled, Hparams.
  match goal with
  | |- context [if code_is_tupled c then ?tupled else ?plain] =>
    destruct (if code_is_tupled c then tupled else plain)
      as [[vs' args_arity'] | ]
  end; [ | exact I ].
  unfold shape_of_code_call. cbv zeta. rewrite Hparams, Hresult, Htupled.
  destruct (Nat.eqb (length args_arity') (length (code_params_arity c)));
    [ apply Hcalls | ].
  destruct (negb _); [ exact I | ].
  destruct (Nat.ltb (length (code_params_arity c)) (length args_arity')).
  - simpl. rewrite Hresult. apply Hcalls.
  - destruct (code_is_tupled c); [ exact I | ].
    apply result_refines_by_refl.
Qed.

(* The two direct calls, of one callee with the same arguments. *)
Lemma aged_direct_call_refines (age_of : static_env -> Code_age_relation.t)
    (statics : static_env) (callee : value) (named newer : code_id)
    (vs : list value) (args_arity return_arity : arity)
    (return_to exn_to : cont_value) (hp : heap) (rd : read_record)
    (return_partial : value -> step_result) :
  closure_at_least (age_of statics) statics callee newer ->
  same_or_newer (age_of statics) newer named ->
  versions_bound (age_of statics) statics newer named ->
  sound_for age_of statics callee ->
  result_refines_aged age_of
    (run_call return_partial
       (shape_of_aged_direct_call (age_of statics) statics callee named vs
          args_arity return_arity)
       return_arity return_to exn_to statics hp rd)
    (run_call return_partial
       (shape_of_aged_direct_call (age_of statics) statics callee newer vs
          args_arity return_arity)
       return_arity return_to exn_to statics hp rd).
Proof.
  intros Hclosure Horder Hbound Hsound.
  pose proof Hclosure as [called [slots [Hcalled Hge]]].
  unfold shape_of_aged_direct_call. rewrite Hcalled.
  rewrite (proj2 (same_or_newer_b_decides _ called newer) Hge).
  rewrite (proj2 (same_or_newer_b_decides _ called named)
             (same_or_newer_trans _ called newer named Hge Horder)).
  destruct (Hbound named Horder (Same_version _ named)) as [c Hc].
  destruct (Hbound newer (Same_version _ newer) Horder) as [c' Hc'].
  rewrite Hc, Hc'. apply run_call_code_refines.
  exact (versions_refine age_of statics callee Hsound newer named Horder
           Hclosure Hbound c c' Hc Hc').
Qed.

Theorem direct_call_newer_version_refines :
  direct_call_newer_version_refines_statement.
Proof.
  intros age_of ap named newer vars conts statics hp rd callee Hkind Hcallee
    Hclosure Horder Hbound Hsound.
  apply result_refines_by_step.
  rewrite (step_aged_apply_direct age_of ap named vars conts statics hp rd
             Hkind).
  rewrite (step_aged_apply_direct age_of (with_call_kind ap (Direct newer))
             newer vars conts statics hp rd eq_refl).
  unfold with_call_kind, exn_extra_simples. cbn [ap_callee ap_args
    ap_continuation ap_exn_continuation ap_exn_extra_args ap_args_arity
    ap_return_arity].
  rewrite Hcallee.
  destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | exact I ].
  destruct (eval_simples vars (map fst (ap_exn_extra_args ap)))
    as [extra | ]; [ | exact I ].
  apply aged_direct_call_refines; assumption.
Qed.

Theorem direct_call_meet_set_refines :
  direct_call_meet_set_refines_statement.
Proof.
  intros age_of steps ap named from_type met vars conts statics hp rd callee
    Hlen Hmeet Hkind Hcallee Hclosure Hbound Hsound.
  pose proof (meet_set_in_unit_ok steps (age_of statics) [named] [from_type]
                [met] Hlen Hmeet) as Hallows.
  assert (Hmet : set_allows (age_of statics) [met] met)
    by (apply set_allows_singleton; constructor).
  apply Hallows in Hmet. destruct Hmet as [Hnamed _].
  apply set_allows_singleton in Hnamed.
  destruct Hclosure as [called [slots [Hcalled Hge]]].
  destruct (same_or_newer_b (age_of statics) called named) eqn:Hallowed.
  - apply same_or_newer_b_decides in Hallowed.
    apply (direct_call_newer_version_refines age_of ap named met vars conts
             statics hp rd callee Hkind Hcallee); try assumption.
    exists called, slots. split; [ exact Hcalled | ].
    apply set_allows_singleton, Hallows.
    split; apply set_allows_singleton; assumption.
  - apply result_refines_by_step.
    rewrite (step_aged_apply_direct age_of ap named vars conts statics hp rd
               Hkind).
    rewrite Hcallee.
    destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | exact I ].
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
      [ | exact I ].
    unfold shape_of_aged_direct_call. rewrite Hcalled, Hallowed. exact I.
Qed.

Theorem direct_call_meet_set_bottom : direct_call_meet_set_bottom_statement.
Proof.
  intros age_of steps ap named from_type vars conts statics hp rd callee Hlen
    Hmeet Hkind Hcallee [called [slots [Hcalled Hge]]].
  rewrite (step_aged_apply_direct age_of ap named vars conts statics hp rd
             Hkind).
  rewrite Hcallee.
  destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | reflexivity ].
  destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
    [ | reflexivity ].
  unfold shape_of_aged_direct_call. rewrite Hcalled.
  destruct (same_or_newer_b (age_of statics) called named) eqn:Hallowed;
    [ | reflexivity ].
  exfalso. apply same_or_newer_b_decides in Hallowed.
  apply (meet_set_in_unit_bottom steps (age_of statics) [named] [from_type]
           Hlen Hmeet called).
  split; apply set_allows_singleton; assumption.
Qed.

(* ======================================================================== *)
(* Two units                                                                *)
(* ======================================================================== *)

Theorem direct_call_older_code_example :
  direct_call_older_code_example_statement.
Proof. cbv zeta. repeat split; vm_compute; reflexivity. Qed.

(* The code that the first unit binds is one code. *)
Lemma age_example_same_code_lookup (id : code_id) (c : code) :
  lookup id age_example_same_code = Some c ->
  c = age_example_code (Var 1%positive).
Proof.
  unfold age_example_same_code. cbv zeta. intros H.
  rewrite !lookup_cons in H.
  repeat (destruct (Pos.eqb id _); [ injection H as <-; reflexivity | ]).
  discriminate H.
Qed.

Theorem direct_call_other_version : direct_call_other_version_statement.
Proof.
  cbv zeta. split; [ | split; [ | split ] ].
  - apply same_or_newer_b_decides. vm_compute. reflexivity.
  - intros id Hfrom _.
    inversion Hfrom as [ id0 | id0 older id0' Holder Hrest ]; subst;
      [ eexists; reflexivity | ].
    vm_compute in Holder. injection Holder as <-.
    inversion Hrest as [ id0 | id0 older id0' Holder Hrest' ]; subst;
      [ eexists; reflexivity | ].
    vm_compute in Holder. discriminate.
  - intros newer older c c' _ _ Hc Hc'. simpl in Hc, Hc'.
    apply age_example_same_code_lookup in Hc, Hc'. subst c c'.
    apply code_refines_for_refl.
  - intros [Hexits _].
    assert (Hsource :
              behaves_by (step_aged (fun _ => age_example_relation))
                (age_example_state age_example_same_code (Direct 1%positive))
                (Exits 1%positive [Value_const (Tagged_immediate 5)]
                   empty_heap))
      by (exists 2, no_reads; vm_compute; reflexivity).
    apply Hexits in Hsource.
    apply (run_behaves_by _ 1) in Hsource. vm_compute in Hsource.
    discriminate Hsource.
Qed.

Theorem direct_call_runs_code_named : direct_call_runs_code_named_statement.
Proof. cbv zeta. split; vm_compute; reflexivity. Qed.

(* ======================================================================== *)
(* Three statements of FlambdaStubCallsSpec.v, on the second machine        *)
(* ======================================================================== *)

Theorem demote_call_step_aged_fails : demote_call_step_aged_fails_statement.
Proof.
  intros H.
  specialize (H (fun _ => age_example_unsound_relation) [1%positive]
                (age_example_call (Direct 1%positive))
                [(3%positive, Value_closure 2%positive [])]
                [(1%positive, Outer 1%positive);
                 (2%positive, Outer 2%positive)]
                {| static_code := age_example_other_code;
                   static_values := [] |}
                empty_heap no_reads).
  assert (Hdefined :
            step_aged (fun _ => age_example_unsound_relation)
              (State (Apply (age_example_call (Direct 1%positive)))
                 [(3%positive, Value_closure 2%positive [])]
                 [(1%positive, Outer 1%positive);
                  (2%positive, Outer 2%positive)]
                 {| static_code := age_example_other_code;
                    static_values := [] |}
                 empty_heap no_reads)
            <> Undefined)
    by (vm_compute; discriminate).
  specialize (H Hdefined). vm_compute in H. discriminate H.
Qed.

Theorem demote_calls_refines_aged_fails :
  demote_calls_refines_aged_fails_statement.
Proof.
  intros H.
  specialize (H (fun _ => age_example_unsound_relation) [1%positive]
                (Apply (age_example_call (Direct 1%positive)))
                [(3%positive, Value_closure 2%positive [])]
                [(1%positive, Outer 1%positive);
                 (2%positive, Outer 2%positive)]
                {| static_code := age_example_other_code;
                   static_values := [] |}
                empty_heap no_reads).
  destruct H as [Hexits _].
  assert (Hsource :
            behaves_by (step_aged (fun _ => age_example_unsound_relation))
              (State (Apply (age_example_call (Direct 1%positive)))
                 [(3%positive, Value_closure 2%positive [])]
                 [(1%positive, Outer 1%positive);
                  (2%positive, Outer 2%positive)]
                 {| static_code := age_example_other_code;
                    static_values := [] |}
                 empty_heap no_reads)
              (Exits 1%positive [Value_const (Tagged_immediate 0)]
                 empty_heap))
    by (exists 2, no_reads; vm_compute; reflexivity).
  apply Hexits in Hsource.
  apply (run_behaves_by _ 2) in Hsource. vm_compute in Hsource.
  discriminate Hsource.
Qed.

Theorem demote_full_call_step_aged_fails :
  demote_full_call_step_aged_fails_statement.
Proof.
  intros H.
  specialize (H (fun _ => age_example_unsound_relation)
                age_example_other_code
                (age_example_call (Direct 1%positive))
                [(3%positive, Value_closure 2%positive [])]
                [(1%positive, Outer 1%positive);
                 (2%positive, Outer 2%positive)]
                {| static_code := age_example_other_code;
                   static_values := [] |}
                empty_heap no_reads).
  assert (Hknown :
            statics_know_signatures age_example_other_code
              {| static_code := age_example_other_code;
                 static_values := [] |}).
  { intros id c Hc. exists c. split; [ exact Hc | split; reflexivity ]. }
  assert (Hdefined :
            step_aged (fun _ => age_example_unsound_relation)
              (State (Apply (age_example_call (Direct 1%positive)))
                 [(3%positive, Value_closure 2%positive [])]
                 [(1%positive, Outer 1%positive);
                  (2%positive, Outer 2%positive)]
                 {| static_code := age_example_other_code;
                    static_values := [] |}
                 empty_heap no_reads)
            <> Undefined)
    by (vm_compute; discriminate).
  specialize (H Hknown Hdefined). vm_compute in H. discriminate H.
Qed.
