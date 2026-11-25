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

(* Proofs of the statements of FlambdaReaperCodeAgeSpec.v. Nothing here
   needs review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec FlambdaSemanticsSpec
  FlambdaWellFormedSpec FlambdaReaperGraphSpec FlambdaReaperTraversal
  FlambdaReaperSolution FlambdaReaperFlowSpec FlambdaReaperRebuild
  FlambdaReaperRebuildSpec FlambdaReaperCallingConventions
  FlambdaReaperCallingConventionsSpec FlambdaReaperLeastSpec
  FlambdaCodeAgeCallsSpec FlambdaCodeAgeCallsProofs
  FlambdaReaperCodeAgeSpec.
From OxCamlFlambda2 Require FlambdaReaperLeast FlambdaReaperLeastProofs
  FlambdaReaperCallingConventionsProofs FlambdaReaperFlowProofs.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaCodeAgeRelation FlambdaCodeAgeRelationSpec
  FlambdaCodeAgeRelationProofs.
Import ListNotations.

(* ======================================================================== *)
(* Runs of a machine                                                        *)
(* ======================================================================== *)

Section Machine.

Variable machine_step : state -> step_result.

(* What a step result comes to after [n] more steps. *)
Definition run_result (n : nat) (result : step_result) : step_result :=
  match result with
  | Step s => run_by machine_step n s
  | _ => result
  end.

Lemma result_behaves_run (n : nat) (result : step_result) (b : behaviour) :
  result_behaves_by machine_step result b
  <-> result_behaves_by machine_step (run_result n result) b.
Proof.
  destruct result as [s | k vs hp rd | ]; simpl;
    [ | reflexivity | reflexivity ].
  apply run_behaves_by.
Qed.

Lemma behaves_by_run_exit (n : nat) (s : state) (k : continuation)
    (vs : list value) (hp : heap) (rd : read_record) :
  run_by machine_step n s = Exit k vs hp rd ->
  ∀ b : behaviour, behaves_by machine_step s b <-> b = Exits k vs hp.
Proof.
  intros Hrun b. rewrite (run_behaves_by machine_step n s b), Hrun.
  reflexivity.
Qed.

Lemma behaves_by_run_undefined (n : nat) (s : state) :
  run_by machine_step n s = Undefined ->
  ∀ b : behaviour, behaves_by machine_step s b <-> b = Goes_wrong.
Proof.
  intros Hrun b. rewrite (run_behaves_by machine_step n s b), Hrun.
  reflexivity.
Qed.

(* Two runs that exit in the same way. *)
Lemma refines_by_runs (n m : nat) (s t : state) (k : continuation)
    (vs : list value) (hp : heap) (rd rd' : read_record) :
  run_by machine_step n s = Exit k vs hp rd ->
  run_by machine_step m t = Exit k vs hp rd' ->
  refines_by machine_step s t.
Proof.
  intros Hs Ht. split.
  - intros k' vs' hp' Hb.
    apply (behaves_by_run_exit n s k vs hp rd Hs) in Hb.
    apply (behaves_by_run_exit m t k vs hp rd' Ht). exact Hb.
  - intros b _ Hb. left.
    apply (behaves_by_run_exit m t k vs hp rd' Ht) in Hb.
    apply (behaves_by_run_exit n s k vs hp rd Hs). exact Hb.
Qed.

(* A run that exits and one that goes wrong. *)
Lemma not_refines_by_exit_undefined (n m : nat) (s t : state)
    (k : continuation) (vs : list value) (hp : heap) (rd : read_record) :
  run_by machine_step n s = Exit k vs hp rd ->
  run_by machine_step m t = Undefined ->
  ~ refines_by machine_step s t.
Proof.
  intros Hs Ht [Hexits _].
  assert (Hb : behaves_by machine_step t (Exits k vs hp)).
  { apply Hexits. apply (behaves_by_run_exit n s k vs hp rd Hs).
    reflexivity. }
  apply (behaves_by_run_undefined m t Ht) in Hb. discriminate Hb.
Qed.

(* Two results that come to exits with other values. *)
Lemma not_result_refines_by_exits (n : nat) (result result' : step_result)
    (k : continuation) (vs vs' : list value) (hp hp' : heap)
    (rd rd' : read_record) :
  run_result n result = Exit k vs hp rd ->
  run_result n result' = Exit k vs' hp' rd' ->
  vs <> vs' ->
  ~ result_refines_by machine_step result result'.
Proof.
  intros Hr Hr' Hne.
  destruct result as [s | k0 vs0 hp0 rd0 | ]; cbn [run_result] in Hr;
    [ | | discriminate Hr ].
  - destruct result' as [s' | k1 vs1 hp1 rd1 | ]; cbn [run_result] in Hr';
      cbn [result_refines_by]; [ | intros [] | intros [] ].
    intros [Hexits _].
    assert (H : behaves_by machine_step s (Exits k vs hp)).
    { rewrite (run_behaves_by machine_step n s), Hr. reflexivity. }
    apply Hexits in H. rewrite (run_behaves_by machine_step n s'), Hr' in H.
    cbn [result_behaves_by] in H. injection H as Hv _. exact (Hne Hv).
  - injection Hr as -> -> -> ->.
    destruct result' as [s' | k1 vs1 hp1 rd1 | ]; cbn [run_result] in Hr';
      cbn [result_refines_by]; [ intros [] | | discriminate Hr' ].
    injection Hr' as -> -> -> ->. intros (_ & Hvs & _). exact (Hne Hvs).
Qed.

(* A state whose first step is undefined is refined by any. *)
Lemma refines_by_undefined (s t : state) :
  machine_step s = Undefined -> refines_by machine_step s t.
Proof.
  intros H. split.
  - intros k vs hp Hb. apply step_behaves_by in Hb. rewrite H in Hb.
    discriminate Hb.
  - intros b _ _. right. apply step_behaves_by. rewrite H. reflexivity.
Qed.

End Machine.

(* A state that exits on [step] does not go wrong. *)
Lemma not_wrong_of_run_exit (n : nat) (s : state) (k : continuation)
    (vs : list value) (hp : heap) (rd : read_record) :
  run n s = Exit k vs hp rd -> ~ behaves s Goes_wrong.
Proof.
  intros Hrun Hw. apply behaves_by_step in Hw.
  rewrite <- run_by_step in Hrun.
  apply (behaves_by_run_exit step n s k vs hp rd Hrun) in Hw.
  discriminate Hw.
Qed.

(* ======================================================================== *)
(* A refinement of a state that does not go wrong, on the machine           *)
(* ======================================================================== *)

Section Defined.

Variable age_of : static_env -> Code_age_relation.t.

(* The runs of a state that does not go wrong on [step] are those of
   [step]. *)
Lemma run_aged_defined (fuel : nat) :
  ∀ s : state,
    ~ behaves s Goes_wrong -> run_aged age_of fuel s = run_by step fuel s.
Proof.
  induction fuel as [ | fuel IH]; intros s Hnw; [ reflexivity | ].
  unfold run_aged in *. simpl.
  assert (Hstep : step s <> Undefined).
  { intros H. apply Hnw. exists 1. simpl. rewrite H. reflexivity. }
  rewrite (step_aged_extends age_of s Hstep).
  destruct (step s) as [s' | k vs hp rd | ] eqn:Hs;
    [ | reflexivity | contradiction ].
  apply IH. intros [n Hn]. apply Hnw. exists (S n). simpl. rewrite Hs.
  exact Hn.
Qed.

Lemma behaves_aged_defined (s : state) :
  ~ behaves s Goes_wrong ->
  ∀ b : behaviour, behaves_aged age_of s b <-> behaves s b.
Proof.
  intros Hnw b. rewrite <- behaves_by_step. unfold behaves_aged.
  pose proof (run_aged_defined) as Hrun. unfold run_aged in Hrun.
  destruct b as [k vs hp | | ]; simpl; split.
  - intros [n [rd H]]. exists n, rd. rewrite <- (Hrun n s Hnw). exact H.
  - intros [n [rd H]]. exists n, rd. rewrite (Hrun n s Hnw). exact H.
  - intros [n H]. exists n. rewrite <- (Hrun n s Hnw). exact H.
  - intros [n H]. exists n. rewrite (Hrun n s Hnw). exact H.
  - intros H n. rewrite <- (Hrun n s Hnw). exact (H n).
  - intros H n. rewrite (Hrun n s Hnw). exact (H n).
Qed.

End Defined.

Theorem refines_aged_of_defined : refines_aged_of_defined_statement.
Proof.
  intros age_of s t Href Hnw.
  assert (Htnw : ~ behaves t Goes_wrong).
  { intros Ht. destruct Href as [_ Hends].
    destruct (Hends Goes_wrong ltac:(discriminate) Ht); contradiction. }
  destruct Href as [Hexits Hends]. split.
  - intros k vs hp Hb.
    change (behaves_aged age_of t (Exits k vs hp)).
    apply (behaves_aged_defined age_of t Htnw), Hexits.
    apply (behaves_aged_defined age_of s Hnw). exact Hb.
  - intros b Hd Hb.
    change (behaves_aged age_of t b) in Hb.
    apply (behaves_aged_defined age_of t Htnw) in Hb.
    destruct (Hends b Hd Hb) as [H | H]; [ left | right ];
      apply (behaves_aged_defined age_of s Hnw); exact H.
Qed.

Theorem refines_observably_aged_of_defined :
  refines_observably_aged_of_defined_statement.
Proof.
  intros age_of s t Href Hnw.
  assert (Htnw : ~ behaves t Goes_wrong).
  { intros Ht. destruct Href as (_ & _ & Hwrong). exact (Hnw (Hwrong Ht)). }
  pose proof (behaves_aged_defined age_of s Hnw) as Hs.
  pose proof (behaves_aged_defined age_of t Htnw) as Ht.
  destruct Href as (H1 & H2 & H3). split; [ | split ].
  - intros k vs hp Hb. apply Hs, H1 in Hb.
    destruct Hb as (vs' & Hb & Hrel). exists vs'. split; [ | exact Hrel ].
    apply Ht. exact Hb.
  - intros k vs' hp Hb. apply Ht, H2 in Hb.
    destruct Hb as [(vs & Hb & Hrel) | Hb]; [ left | right ].
    + exists vs. split; [ | exact Hrel ]. apply Hs. exact Hb.
    + apply Hs. exact Hb.
  - intros Hb. apply Ht, H3 in Hb. apply Hs. exact Hb.
Qed.

Theorem least_rebuild_refines_aged : least_rebuild_refines_aged_statement.
Proof.
  intros flag is_local u e T preserve_direct_calls age_of Hslots Hlocal
    Hfrag Hkinds Hres Hwf Hexits Hleast Hnw.
  apply refines_observably_aged_of_defined; [ | exact Hnw ].
  exact (FlambdaReaperLeastProofs.least_rebuild_refines_observably flag
           is_local u e T preserve_direct_calls Hslots Hlocal Hfrag Hkinds
           Hres Hwf Hexits Hleast).
Qed.

Theorem least_rebuild_cc_auto_refines_aged :
  least_rebuild_cc_auto_refines_aged_statement.
Proof.
  intros is_local u e T preserve_direct_calls age_of Hslots Hlocal Hfrag
    Hkinds Hres Hwf Hexits Hleast Hnw.
  apply refines_observably_aged_of_defined; [ | exact Hnw ].
  exact (FlambdaReaperLeastProofs.least_rebuild_cc_auto_refines_observably
           is_local u e T preserve_direct_calls Hslots Hlocal Hfrag Hkinds
           Hres Hwf Hexits Hleast).
Qed.

(* ======================================================================== *)
(* The code of a unit and its conventions                                   *)
(* ======================================================================== *)

Theorem unit_statics_step : unit_statics_step_statement.
Proof.
  intros bindings body statics H. unfold unit_statics. rewrite H.
  split; [ reflexivity | ]. cbn [step]. rewrite H. reflexivity.
Qed.

Lemma lookup_some_in {A : Type} (id : positive) (a : A)
    (l : list (positive * A)) :
  lookup id l = Some a -> (id, a) ∈ l.
Proof.
  induction l as [ | [id' a'] l IH]; simpl; [ discriminate | ].
  destruct (Pos.eqb_spec id id') as [<- | _].
  - intros [= <-]. left. reflexivity.
  - intros H. right. exact (IH H).
Qed.

(* The code that a group of bindings adds is bound by one of them. *)
Lemma eval_static_bindings_codes (vars : env) (bindings : list static_binding)
    : ∀ statics statics' : static_env,
        eval_static_bindings vars bindings statics = Some statics' ->
        ∀ (id : code_id) (c : code),
          lookup id (static_code statics') = Some c ->
          lookup id (static_code statics) = Some c
          ∨ Bind_code id c ∈ bindings.
Proof.
  induction bindings as [ | b bindings IH]; intros statics statics' H id c Hc.
  - injection H as <-. left. exact Hc.
  - destruct b as [id' c' | id' | s const]; simpl in H.
    + destruct (lookup id' (static_code statics)); [ discriminate H | ].
      destruct (IH _ _ H id c Hc) as [Hc' | Hin]; [ | right; right; exact Hin ].
      simpl in Hc'. destruct (Pos.eqb_spec id id') as [<- | _].
      * injection Hc' as <-. right. left. reflexivity.
      * left. exact Hc'.
    + destruct (IH _ _ H id c Hc) as [Hc' | Hin]; [ left; exact Hc' | ].
      right. right. exact Hin.
    + destruct (lookup s (static_values statics)); [ discriminate H | ].
      destruct (eval_static_const vars const); [ | discriminate H ].
      destruct (IH _ _ H id c Hc) as [Hc' | Hin]; [ left; exact Hc' | ].
      right. right. exact Hin.
Qed.

Theorem unit_statics_codes : unit_statics_codes_statement.
Proof.
  intros e id c H.
  destruct e as [ | bindings body | | | | | | ]; try discriminate H.
  unfold unit_statics in H.
  destruct (eval_static_bindings [] bindings empty_static_env) as [statics | ]
    eqn:He; [ | discriminate H ].
  destruct (eval_static_bindings_codes [] bindings _ _ He id c H)
    as [Hc | Hin]; [ discriminate Hc | ].
  simpl. apply in_or_app. left. apply in_flat_map.
  exists (Bind_code id c). split; [ exact Hin | left; reflexivity ].
Qed.

Theorem sound_for_conventions : sound_for_conventions_statement.
Proof.
  intros age_of statics v H newer older c c' Hat Hl Hc Hc'.
  destruct (H newer older c c' Hat Hl Hc Hc') as (Hp & Hr & Ht & _).
  split; [ exact Hp | split; [ exact Hr | exact Ht ] ].
Qed.

Theorem conventions_agree_statics : conventions_agree_statics_statement.
Proof.
  intros age_of e v H newer older c c' _ Hl Hc Hc'.
  exact (H newer older c c' Hl (unit_statics_codes e older c Hc)
           (unit_statics_codes e newer c' Hc')).
Qed.

(* ======================================================================== *)
(* The code of the rebuilt unit                                             *)
(* ======================================================================== *)

Section Rebuilt.

Variable F : facts.
Variable sol : solution.
Variable cannot_change : node -> bool.
Variable code_deps : finite_map code_dep.
Variable preserve_direct_calls : bool.
Variable unit_code_ids : list code_id.
Variable codes : finite_map code_metadata.
Variable first_fresh : positive.
Variable types : subkinds_rewriting.

Abbreviation rebuild :=
  (rebuild_expr_cc F sol cannot_change code_deps preserve_direct_calls
     unit_code_ids codes first_fresh types).
Abbreviation rebuild_h :=
  (rebuild_handler_cc F sol cannot_change code_deps preserve_direct_calls
     unit_code_ids codes first_fresh types).
Abbreviation rebuild_c :=
  (rebuild_code_cc F sol cannot_change code_deps preserve_direct_calls
     unit_code_ids codes first_fresh types).

(* A call binds no code. *)
Lemma unit_codes_rebuild_apply_cc (en : rebuild_env) (ap : apply_expr) :
  unit_codes
    (rebuild_apply_cc F sol cannot_change code_deps preserve_direct_calls
       unit_code_ids codes first_fresh types en ap)
  = [].
Proof.
  unfold rebuild_apply_cc.
  destruct (apply_calling_convention_change _ _ _ _ _ _ ap).
  - unfold rebuild_apply. destruct (_ || _); [ reflexivity | ].
    unfold make_apply_wrapper. destruct (lookup _ _); [ | reflexivity ].
    destruct (forallb _ _); reflexivity.
  - destruct (_ || _); [ reflexivity | ].
    unfold make_apply_wrapper_cc. cbv zeta.
    destruct (forallb _ _); reflexivity.
Qed.

(* Each code that the rebuilt unit binds is the rebuild of a code that the
   unit binds, by the same code id. *)
Lemma unit_codes_rebuild_cc :
  (∀ e : expr,
     ∀ (en : rebuild_env) (x : code_id * code),
       x ∈ unit_codes (rebuild en e) ->
       ∃ (id : code_id) (c : code),
         x = (id, rebuild_c id c) ∧ (id, c) ∈ unit_codes e)
  ∧ (∀ h : continuation_handler,
       ∀ (en : rebuild_env) (dead_check : bool) (mask : list bool)
         (x : code_id * code),
         x ∈ handler_unit_codes (rebuild_h en dead_check mask h) ->
         ∃ (id : code_id) (c : code),
           x = (id, rebuild_c id c) ∧ (id, c) ∈ handler_unit_codes h).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - intros x kind d body IH en y Hin. cbn [rebuild_expr_cc] in Hin.
    destruct (is_pending_projection en x d);
      [ exact (IH _ _ Hin) | ].
    destruct d; cbv zeta in Hin;
      repeat match type of Hin with
             | context [if ?b then _ else _] => destruct b
             end;
      solve [ destruct Hin | exact (IH _ _ Hin) ].
  - intros bindings body _ IH en x Hin. cbn [rebuild_expr_cc unit_codes] in Hin.
    apply in_app_or in Hin as [Hin | Hin].
    + apply in_flat_map in Hin as (b' & Hb' & Hx).
      apply in_flat_map in Hb' as (b & Hb & Hb').
      destruct b as [id c | id | s const]; cbn [rebuild_binding_cc] in Hb'.
      * destruct (is_code_id_used sol id);
          destruct Hb' as [<- | []]; simpl in Hx; [ | destruct Hx ].
        destruct Hx as [<- | []]. exists id, c. split; [ reflexivity | ].
        cbn [unit_codes]. apply in_or_app. left. apply in_flat_map.
        exists (Bind_code id c). split; [ exact Hb | left; reflexivity ].
      * destruct Hb' as [<- | []]. destruct Hx.
      * destruct (static_const_is_used sol s const);
          [ destruct Hb' as [<- | []]; destruct Hx | destruct Hb' ].
    + destruct (IH _ _ Hin) as (id & c & -> & Hc). exists id, c.
      split; [ reflexivity | ]. cbn [unit_codes]. apply in_or_app. right.
      exact Hc.
  - intros k h body IHh IH en x Hin. cbn [rebuild_expr_cc] in Hin.
    cbv zeta in Hin. destruct (among k _); cbn [unit_codes] in Hin |- *.
    + apply in_app_or in Hin as [Hin | Hin].
      * destruct (IHh _ _ _ _ Hin) as (id & c & -> & Hc). exists id, c.
        split; [ reflexivity | ]. apply in_or_app. left. exact Hc.
      * destruct (IH _ _ Hin) as (id & c & -> & Hc). exists id, c.
        split; [ reflexivity | ]. apply in_or_app. right. exact Hc.
    + destruct (IH _ _ Hin) as (id & c & -> & Hc). exists id, c.
      split; [ reflexivity | ]. apply in_or_app. right. exact Hc.
  - intros handlers body IHh IH en x Hin.
    cbn [rebuild_expr_cc unit_codes] in Hin.
    cbn [unit_codes]. apply in_app_or in Hin as [Hin | Hin].
    + apply in_flat_map in Hin as ([k h'] & Hkh & Hx).
      apply in_map_iff in Hkh as ([k0 h0] & Heq & Hin0).
      injection Heq as <- <-.
      destruct (IHh k0 h0 Hin0 _ _ _ _ Hx) as (id & c & -> & Hc).
      exists id, c. split; [ reflexivity | ]. apply in_or_app. left.
      apply in_flat_map. exists (k0, h0). split; [ exact Hin0 | exact Hc ].
    + destruct (IH _ _ Hin) as (id & c & -> & Hc). exists id, c.
      split; [ reflexivity | ]. apply in_or_app. right. exact Hc.
  - intros ap en x Hin. cbn [rebuild_expr_cc] in Hin.
    rewrite unit_codes_rebuild_apply_cc in Hin. destruct Hin.
  - intros ac en x Hin. cbn [rebuild_expr_cc] in Hin.
    unfold rebuild_apply_cont in Hin. destruct (rewrite_apply_cont _ _ _);
      destruct Hin.
  - intros scrutinee arms en x Hin. cbn [rebuild_expr_cc] in Hin.
    unfold rebuild_switch in Hin. cbv zeta in Hin.
    destruct (Keyed.is_empty _); destruct Hin.
  - intros en x Hin. destruct Hin.
  - intros params metadata handler IH en dead_check mask x Hin.
    cbn [rebuild_handler_cc handler_unit_codes] in Hin.
    destruct (_ && _); [ destruct Hin | exact (IH _ _ Hin) ].
Qed.

(* Code whose code id is in the table keeps its convention. *)
Lemma rebuild_code_cc_unchanged (id : code_id) (c : code) :
  cannot_change (Node_code_id id) = true ->
  same_convention c (rebuild_c id c).
Proof.
  intros H. destruct c as [params my_closure ret exn metadata body].
  cbn [rebuild_code_cc]. cbv zeta.
  unfold code_calling_convention_change, cannot_change_calling_convention.
  rewrite H, orb_true_r.
  unfold same_convention, kept_code_metadata.
  destruct types as [t | ];
    [ destruct (lookup id (st_code_deps t)) | ]; simpl;
    (split; [ reflexivity | split; reflexivity ]).
Qed.

End Rebuilt.

(* ======================================================================== *)
(* The fixes                                                                *)
(* ======================================================================== *)

Theorem rebuild_cc_keeps_conventions : rebuild_cc_keeps_conventions_statement.
Proof.
  intros types F sol cannot_change code_deps preserve_direct_calls e rel Hfix
    Hagree newer older c c' Hl Hc Hc'.
  unfold rebuild_unit_cc_with in Hc, Hc'.
  set (uci := static_names code_ids_bound e) in Hc, Hc'.
  set (cm := bound_code_metadata e) in Hc, Hc'.
  set (ff := FlambdaNameAbstractionSpec.fresh_name
               (FlambdaRenameApartSpec.names_of e)) in Hc, Hc'.
  destruct (proj1 (unit_codes_rebuild_cc F sol cannot_change code_deps
                     preserve_direct_calls uci cm ff types) e _ _ Hc)
    as (id & c0 & Heq & Hc0).
  injection Heq as <- ->.
  destruct (proj1 (unit_codes_rebuild_cc F sol cannot_change code_deps
                     preserve_direct_calls uci cm ff types) e _ _ Hc')
    as (id & c0' & Heq & Hc0').
  injection Heq as <- ->.
  destruct (Hfix newer older Hl) as [Hn Ho].
  destruct (Hagree newer older c0 c0' Hl Hc0 Hc0') as (Hp & Hr & Ht).
  destruct (rebuild_code_cc_unchanged F sol cannot_change code_deps
              preserve_direct_calls uci cm ff types older c0 Ho)
    as (Hp1 & Hr1 & Ht1).
  destruct (rebuild_code_cc_unchanged F sol cannot_change code_deps
              preserve_direct_calls uci cm ff types newer c0' Hn)
    as (Hp2 & Hr2 & Ht2).
  unfold same_convention. rewrite Hp1, Hr1, Ht1, Hp2, Hr2, Ht2.
  split; [ exact Hp | split; [ exact Hr | exact Ht ] ].
Qed.

(* A link of the cut relation is a link of the relation, between codes
   that [keep] keeps. *)
Lemma cut_relation_link (keep : code_id -> bool) (rel : Code_age_relation.t)
    (newer older : code_id) :
  Code_age_relation.get_older_version_of (cut_relation keep rel) newer
  = Some older ->
  Code_age_relation.get_older_version_of rel newer = Some older
  ∧ keep newer = true ∧ keep older = true.
Proof.
  unfold Code_age_relation.get_older_version_of, cut_relation.
  intros H. apply lookup_some_in, filter_In in H as [_ Hp]. cbn beta iota in Hp.
  apply andb_true_iff in Hp as [Hp Hl].
  apply andb_true_iff in Hp as [Hn Ho].
  assert (Hlk : ∀ o : option code_id,
            match o with
            | Some older' => Pos.eqb older' older
            | None => false
            end = true ->
            o = Some older).
  { intros [o | ] Ho'; [ apply Pos.eqb_eq in Ho' as ->; reflexivity
                       | discriminate Ho' ]. }
  split; [ exact (Hlk _ Hl) | split; assumption ].
Qed.

Theorem rebuild_cc_cut_keeps_conventions :
  rebuild_cc_cut_keeps_conventions_statement.
Proof.
  intros types F sol cannot_change code_deps preserve_direct_calls e rel
    Hagree.
  assert (Hcut : conventions_agree
                   (cut_relation (fun id => cannot_change (Node_code_id id))
                      rel)
                   (unit_codes e)).
  { intros newer older c c' Hl. apply cut_relation_link in Hl as [Hl _].
    exact (Hagree newer older c c' Hl). }
  apply (rebuild_cc_keeps_conventions types F sol cannot_change code_deps
           preserve_direct_calls e); [ | exact Hcut ].
  intros newer older Hl. apply cut_relation_link in Hl as (_ & Hn & Ho).
  auto.
Qed.

Lemma mem_node_app_l (n : node) (l l' : list node) :
  mem_node n l = true -> mem_node n (l ++ l') = true.
Proof. unfold mem_node. rewrite existsb_app. intros ->. reflexivity. Qed.

Theorem age_fixed_table_links : age_fixed_table_links_statement.
Proof.
  intros rel cannot_change newer older Hl.
  apply lookup_some_in in Hl.
  unfold age_fixed_table, mem_node. rewrite !existsb_app.
  split; apply orb_true_intro; right; apply existsb_exists.
  - exists (Node_code_id newer).
    split; [ | apply FlambdaReaperSolutionProofs.node_eqb_refl ].
    apply in_flat_map. exists (newer, older).
    split; [ exact Hl | left; reflexivity ].
  - exists (Node_code_id older).
    split; [ | apply FlambdaReaperSolutionProofs.node_eqb_refl ].
    apply in_flat_map. exists (newer, older).
    split; [ exact Hl | right; left; reflexivity ].
Qed.

Theorem age_fixed_table_check : age_fixed_table_check_statement.
Proof.
  intros F T cannot_change rel.
  unfold check_cc_closed, check_code_id_of_witness,
    check_code_id_unknown_arity, age_fixed_table.
  rewrite !andb_true_iff, !forallb_forall. intros [H1 H2].
  split; intros n Hn; apply mem_node_app_l; auto.
Qed.

Theorem rebuild_cc_fixed_sound_conventions :
  rebuild_cc_fixed_sound_conventions_statement.
Proof.
  intros F T cannot_change code_deps preserve_direct_calls e rel age_of v.
  cbv zeta. intros Hagree Hage. apply conventions_agree_statics.
  rewrite Hage.
  apply (rebuild_cc_keeps_conventions None F (solution_of_tables T)
           (fun n => mem_node n (age_fixed_table rel cannot_change))
           code_deps preserve_direct_calls e rel); [ | exact Hagree ].
  intros newer older Hl. exact (age_fixed_table_links rel _ _ _ Hl).
Qed.

Theorem rebuild_cc_fixed_refines_observably :
  rebuild_cc_fixed_refines_observably_statement.
Proof.
  intros flag is_local u e T cannot_change preserve_direct_calls rel Hslots
    Hlocal Hfrag Hkinds Hres Hwf Hexits Hclosed Hexact Hcc Hwidgets.
  exact (FlambdaReaperCallingConventionsProofs.rebuild_cc_refines_observably
           flag is_local u e T (age_fixed_table rel cannot_change)
           preserve_direct_calls Hslots Hlocal Hfrag Hkinds Hres Hwf Hexits
           Hclosed Hexact (age_fixed_table_check _ _ _ _ Hcc) Hwidgets).
Qed.

Theorem least_rebuild_cc_fixed_refines_observably :
  least_rebuild_cc_fixed_refines_observably_statement.
Proof.
  intros is_local u e T preserve_direct_calls rel Hslots Hlocal Hfrag Hkinds
    Hres Hwf Hexits Hleast.
  apply (rebuild_cc_fixed_refines_observably Auto is_local u e T
           (derive_cannot_change (traverse_unit Auto u e) T)
           preserve_direct_calls rel Hslots Hlocal Hfrag Hkinds Hres Hwf
           Hexits).
  - exact (proj1 Hleast).
  - exact (FlambdaReaperLeastProofs.least_field_used_exact _ _ _ Hleast).
  - exact (proj1
             (FlambdaReaperCallingConventionsProofs.derive_cannot_change_least
                _ _)).
  - exact (FlambdaReaperLeastProofs.least_apply_widgets _ _ _ Hslots Hleast
             (FlambdaReaperLeastProofs.traverse_widgets_read_only u e)).
Qed.

Theorem least_rebuild_cc_fixed_refines_aged :
  least_rebuild_cc_fixed_refines_aged_statement.
Proof.
  intros is_local u e T preserve_direct_calls rel age_of Hslots Hlocal Hfrag
    Hkinds Hres Hwf Hexits Hleast Hnw.
  apply refines_observably_aged_of_defined; [ | exact Hnw ].
  exact (least_rebuild_cc_fixed_refines_observably is_local u e T
           preserve_direct_calls rel Hslots Hlocal Hfrag Hkinds Hres Hwf
           Hexits Hleast).
Qed.

(* ======================================================================== *)
(* Tools for the examples                                                   *)
(* ======================================================================== *)

(* An element of a list that computes. *)
Ltac in_computed := vm_compute; repeat first [ left; reflexivity | right ].

Lemma refines_observably_aged_of_runs
    (age_of : static_env -> Code_age_relation.t) (n m : nat) (s t : state)
    (k : continuation) (vs vs' : list value) (hp : heap)
    (rd rd' : read_record) :
  run_aged age_of n s = Exit k vs hp rd ->
  run_aged age_of m t = Exit k vs' hp rd' ->
  values_related observed_equal vs vs' ->
  refines_observably_aged age_of s t.
Proof.
  unfold refines_observably_aged, behaves_aged. intros Hs Ht Hrel.
  pose proof (behaves_by_run_exit (step_aged age_of) n s k vs hp rd Hs) as Bs.
  pose proof (behaves_by_run_exit (step_aged age_of) m t k vs' hp rd' Ht)
    as Bt.
  split; [ | split ].
  - intros k0 vs0 hp0 Hb. apply Bs in Hb. injection Hb as -> -> ->.
    exists vs'. split; [ apply Bt; reflexivity | exact Hrel ].
  - intros k0 vs0 hp0 Hb. apply Bt in Hb. injection Hb as -> -> ->. left.
    exists vs. split; [ apply Bs; reflexivity | exact Hrel ].
  - intros Hb. apply Bt in Hb. discriminate Hb.
Qed.

Lemma not_refines_observably_aged_of_runs
    (age_of : static_env -> Code_age_relation.t) (n m : nat) (s t : state)
    (k k' : continuation) (vs vs' : list value) (hp hp' : heap)
    (rd rd' : read_record) :
  run_aged age_of n s = Exit k vs hp rd ->
  run_aged age_of m t = Exit k' vs' hp' rd' ->
  ~ values_related observed_equal vs vs' ->
  ~ refines_observably_aged age_of s t.
Proof.
  unfold refines_observably_aged, behaves_aged.
  intros Hs Ht Hne (H1 & _ & _).
  destruct (H1 k vs hp) as (vs'' & Hb & Hrel).
  { apply (behaves_by_run_exit (step_aged age_of) n s k vs hp rd Hs).
    reflexivity. }
  apply (behaves_by_run_exit (step_aged age_of) m t k' vs' hp' rd' Ht) in Hb.
  injection Hb as _ <- _. exact (Hne Hrel).
Qed.

(* Every code of [codes] has the convention of [c0]: the conventions
   agree, whatever the relation. *)
Lemma conventions_agree_uniform (rel : Code_age_relation.t)
    (codes : list (code_id * code)) (c0 : code) :
  Forall (fun x => same_convention c0 (snd x)) codes ->
  conventions_agree rel codes.
Proof.
  intros Hall newer older c c' _ Hc Hc'.
  rewrite Forall_forall in Hall.
  destruct (Hall _ Hc) as (Hp & Hr & Ht).
  destruct (Hall _ Hc') as (Hp' & Hr' & Ht').
  simpl in *. split; [ congruence | split; congruence ].
Qed.

(* Two codes of one convention whose bodies, entered with the same
   arguments, take steps that are so related. *)
Lemma code_refines_for_steps (age_of : static_env -> Code_age_relation.t)
    (statics : static_env) (v : value) (c c' : code) :
  same_convention c c' ->
  code_params_subkinds c' = code_params_subkinds c ->
  binds_statics (code_body c) = false ->
  binds_statics (code_body c') = false ->
  (∀ (vs : list value) (vars : env) (return_to exn_to : cont_value)
     (hp : heap) (rd : read_record),
     bind_params (code_params c) vs [(code_my_closure c, v)] = Some vars ->
     ∃ vars' : env,
       bind_params (code_params c') vs [(code_my_closure c', v)]
       = Some vars'
       ∧ result_refines_aged age_of
           (step_aged age_of
              (State (code_body c) vars
                 [(code_return_continuation c, return_to);
                  (code_exn_continuation c, exn_to)]
                 statics hp rd))
           (step_aged age_of
              (State (code_body c') vars'
                 [(code_return_continuation c', return_to);
                  (code_exn_continuation c', exn_to)]
                 statics hp rd))) ->
  code_refines_for age_of statics v c c'.
Proof.
  intros (Hp & Hr & Ht) Hsk Hb Hb' Hsteps.
  split; [ exact Hp | split; [ exact Hr | split; [ exact Ht | ] ] ].
  intros vs args_arity return_arity return_to exn_to hp rd.
  unfold call_code. rewrite Hb, Hb', Hp, Hr, Hsk.
  destruct (negb (_ && _ && _)); [ exact I | ].
  destruct (negb (values_of_subkinds _ _ _)); [ exact I | ].
  destruct (bind_params (code_params c) vs _) as [vars | ] eqn:Hbind;
    [ | exact I ].
  destruct (Hsteps vs vars return_to exn_to hp rd Hbind)
    as (vars' & Hbind' & Hres).
  rewrite Hbind'.
  exact (result_refines_by_step (step_aged age_of) _ _ Hres).
Qed.

(* The links of a relation of one link. *)
Lemma one_link (newer older newer' older' : code_id) :
  Code_age_relation.get_older_version_of
    (Code_age_relation.add Code_age_relation.empty newer older) newer'
  = Some older' ->
  newer' = newer ∧ older' = older.
Proof.
  unfold Code_age_relation.get_older_version_of, Code_age_relation.add,
    Code_age_relation.empty.
  simpl. destruct (Pos.eqb_spec newer' newer) as [-> | _];
    [ | discriminate ].
  intros [= ->]. split; reflexivity.
Qed.

(* ======================================================================== *)
(* The bug                                                                  *)
(* ======================================================================== *)

Lemma conventions_codes_refine (age_of : static_env -> Code_age_relation.t)
    (statics : static_env) (v : value) :
  code_refines_for age_of statics v ConventionsExample.older
    ConventionsExample.newer.
Proof.
  apply code_refines_for_steps; try reflexivity.
  - split; [ reflexivity | split; reflexivity ].
  - intros vs vars return_to exn_to hp rd Hb.
    destruct vs as [ | a [ | b [ | ? ? ] ] ]; try discriminate Hb.
    injection Hb as <-. eexists. split; [ reflexivity | ].
    simpl. apply result_refines_by_refl.
Qed.

Lemma conventions_sound_for (e : expr) :
  lookup 50%positive (static_code (unit_statics e))
  = Some ConventionsExample.older ->
  lookup 51%positive (static_code (unit_statics e))
  = Some ConventionsExample.newer ->
  sound_for ConventionsExample.age_of (unit_statics e)
    (Value_symbol 80%positive).
Proof.
  intros H50 H51 newer older c c' _ Hl Hc Hc'.
  apply one_link in Hl as [-> ->].
  rewrite H50 in Hc. rewrite H51 in Hc'.
  injection Hc as <-. injection Hc' as <-.
  apply conventions_codes_refine.
Qed.

Lemma conventions_closure_at_least (e : expr) :
  closure_of (unit_statics e) (Value_symbol 80%positive)
  = Some (51%positive, []) ->
  closure_at_least ConventionsExample.relation (unit_statics e)
    (Value_symbol 80%positive) 51%positive.
Proof. intros H. exists 51%positive, []. split; [ exact H | constructor ]. Qed.

Theorem conventions_counterexample : conventions_counterexample_statement.
Proof.
  cbv zeta.
  assert (Hleast : least_solution Examples.no_local ConventionsExample.facts
                     ConventionsExample.tables).
  { apply (FlambdaReaperLeastProofs.compute_least_sound _ _ 40).
    vm_compute. reflexivity. }
  assert (Hnw : ~ behaves
                  (State ConventionsExample.unit_expr [] [] empty_static_env
                     empty_heap no_reads)
                  Goes_wrong).
  { apply (not_wrong_of_run_exit 10 _ 1%positive
             [Value_block 0
                [Value_const (Tagged_immediate 9);
                 Value_closure_of_set 71%positive [(71%positive, 50%positive)]
                   []]]
             empty_heap no_reads).
    vm_compute. reflexivity. }
  assert (Hslots : is_local_slots_only Examples.no_local).
  { intros f H. discriminate H. }
  assert (Hlocal : (∀ f : field, Examples.no_local f = false)
                   ∨ projection_free ConventionsExample.unit_expr = true).
  { left. intros f. reflexivity. }
  assert (Hc50 : lookup 50%positive
                   (static_code (unit_statics ConventionsExample.rebuilt))
                 = Some ConventionsExample.older).
  { vm_compute. reflexivity. }
  assert (Hc51 : lookup 51%positive
                   (static_code (unit_statics ConventionsExample.rebuilt))
                 = Some ConventionsExample.newer_rebuilt).
  { vm_compute. reflexivity. }
  split; [ exact Hleast | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ exact Hnw | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split.
  { exact (FlambdaReaperLeastProofs.least_rebuild_cc_auto_refines_observably
             Examples.no_local Examples.interface ConventionsExample.unit_expr
             ConventionsExample.tables true Hslots Hlocal
             ltac:(vm_compute; reflexivity) ltac:(vm_compute; reflexivity)
             ltac:(vm_compute; reflexivity) ltac:(vm_compute; reflexivity)
             ltac:(vm_compute; reflexivity) Hleast). }
  split.
  { intros age_of.
    exact (least_rebuild_cc_auto_refines_aged Examples.no_local
             Examples.interface ConventionsExample.unit_expr
             ConventionsExample.tables true age_of Hslots Hlocal
             ltac:(vm_compute; reflexivity) ltac:(vm_compute; reflexivity)
             ltac:(vm_compute; reflexivity) ltac:(vm_compute; reflexivity)
             ltac:(vm_compute; reflexivity) Hleast Hnw). }
  split.
  { apply (conventions_agree_uniform _ _ ConventionsExample.older).
    vm_compute. repeat constructor. }
  split.
  { apply conventions_sound_for; vm_compute; reflexivity. }
  split; [ exact Hc50 | ].
  split; [ exact Hc51 | ].
  split.
  { intros H.
    destruct (H 51%positive 50%positive ConventionsExample.older
                ConventionsExample.newer_rebuilt eq_refl)
      as (Hp & _ & _).
    - in_computed.
    - in_computed.
    - vm_compute in Hp. discriminate Hp. }
  intros H.
  destruct (H 51%positive 50%positive ConventionsExample.older
              ConventionsExample.newer_rebuilt
              (conventions_closure_at_least ConventionsExample.rebuilt
                 ltac:(vm_compute; reflexivity))
              eq_refl Hc50 Hc51)
    as (Hp & _ & _ & _).
  vm_compute in Hp. discriminate Hp.
Qed.

Theorem conventions_client : conventions_client_statement.
Proof.
  cbv zeta.
  assert (H1 : run_aged ConventionsExample.age_of 2
                 (ConventionsExample.client
                    (unit_statics ConventionsExample.unit_expr) 50%positive)
               = Exit 1%positive [Value_const (Tagged_immediate 5)]
                   empty_heap no_reads).
  { vm_compute. reflexivity. }
  assert (H2 : run_aged ConventionsExample.age_of 2
                 (ConventionsExample.client
                    (unit_statics ConventionsExample.unit_expr) 51%positive)
               = Exit 1%positive [Value_const (Tagged_immediate 5)]
                   empty_heap no_reads).
  { vm_compute. reflexivity. }
  assert (H3 : run_aged ConventionsExample.age_of 2
                 (ConventionsExample.client
                    (unit_statics ConventionsExample.rebuilt) 50%positive)
               = Exit 1%positive [Value_const (Tagged_immediate 5)]
                   empty_heap no_reads).
  { vm_compute. reflexivity. }
  assert (H4 : run_aged ConventionsExample.age_of 1
                 (ConventionsExample.client
                    (unit_statics ConventionsExample.rebuilt) 51%positive)
               = Undefined).
  { vm_compute. reflexivity. }
  split; [ exact H1 | ].
  split; [ exact H2 | ].
  split; [ exact (refines_by_runs _ 2 2 _ _ _ _ _ _ _ H1 H2) | ].
  split; [ vm_compute; reflexivity | ].
  split; [ exact H3 | ].
  split; [ exact H4 | ].
  exact (not_refines_by_exit_undefined _ 2 1 _ _ _ _ _ _ H3 H4).
Qed.

Theorem conventions_fixed : conventions_fixed_statement.
Proof.
  cbv zeta.
  assert (Hleast : least_solution Examples.no_local ConventionsExample.facts
                     ConventionsExample.tables).
  { apply (FlambdaReaperLeastProofs.compute_least_sound _ _ 40).
    vm_compute. reflexivity. }
  assert (Hslots : is_local_slots_only Examples.no_local).
  { intros f H. discriminate H. }
  assert (Hlocal : (∀ f : field, Examples.no_local f = false)
                   ∨ projection_free ConventionsExample.unit_expr = true).
  { left. intros f. reflexivity. }
  assert (H1 : run_aged ConventionsExample.age_of 2
                 (ConventionsExample.client
                    (unit_statics ConventionsExample.rebuilt_fixed)
                    50%positive)
               = Exit 1%positive [Value_const (Tagged_immediate 5)]
                   empty_heap no_reads).
  { vm_compute. reflexivity. }
  assert (H2 : run_aged ConventionsExample.age_of 2
                 (ConventionsExample.client
                    (unit_statics ConventionsExample.rebuilt_fixed)
                    51%positive)
               = Exit 1%positive [Value_const (Tagged_immediate 5)]
                   empty_heap no_reads).
  { vm_compute. reflexivity. }
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split.
  { exact (least_rebuild_cc_fixed_refines_observably Examples.no_local
             Examples.interface ConventionsExample.unit_expr
             ConventionsExample.tables true ConventionsExample.relation
             Hslots Hlocal
             ltac:(vm_compute; reflexivity) ltac:(vm_compute; reflexivity)
             ltac:(vm_compute; reflexivity) ltac:(vm_compute; reflexivity)
             ltac:(vm_compute; reflexivity) Hleast). }
  split.
  { apply (conventions_agree_uniform _ _ ConventionsExample.older).
    vm_compute. repeat constructor. }
  split.
  { apply conventions_sound_for; vm_compute; reflexivity. }
  split; [ exact (refines_by_runs _ 2 2 _ _ _ _ _ _ _ H1 H2) | ].
  vm_compute. reflexivity.
Qed.

(* ======================================================================== *)
(* A direct call of a closure of a newer version, within the unit           *)
(* ======================================================================== *)

Lemma aged_premises (e : expr) :
  in_flow_fragment e = true -> any_kind_arguments e = true ->
  results_kinded [] [] e = true ->
  well_formed_unit Examples.interface e = true ->
  exits_one_value Examples.interface e = true ->
  least_solution Examples.no_local (AgedCallExample.facts_of e)
    (AgedCallExample.tables_of e) ->
  program_refines_observably e (AgedCallExample.rebuilt e).
Proof.
  intros Hfrag Hkinds Hres Hwf Hexits Hleast.
  exact (FlambdaReaperLeastProofs.least_rebuild_refines_observably Auto
           Examples.no_local Examples.interface e
           (AgedCallExample.tables_of e) true
           ltac:(intros f H; discriminate H)
           ltac:(left; intros f; reflexivity)
           Hfrag Hkinds Hres Hwf Hexits Hleast).
Qed.

Theorem aged_call_counterexample : aged_call_counterexample_statement.
Proof.
  cbv zeta.
  assert (Hleast : least_solution Examples.no_local
                     (AgedCallExample.facts_of AgedCallExample.unsound_unit)
                     (AgedCallExample.tables_of
                        AgedCallExample.unsound_unit)).
  { apply (FlambdaReaperLeastProofs.compute_least_sound _ _ 40).
    vm_compute. reflexivity. }
  assert (H0 : run_aged AgedCallExample.age_of 5
                 (AgedCallExample.initial AgedCallExample.unsound_unit)
               = Exit 1%positive [Value_const (Tagged_immediate 0)]
                   empty_heap no_reads).
  { vm_compute. reflexivity. }
  assert (H1 : run_aged AgedCallExample.age_of 5
                 (AgedCallExample.initial
                    (AgedCallExample.rebuilt AgedCallExample.unsound_unit))
               = Exit 1%positive [Value_const (Tagged_immediate 1)]
                   empty_heap no_reads).
  { vm_compute. reflexivity. }
  assert (H2 : run_aged AgedCallExample.age_of 5
                 (AgedCallExample.initial
                    (AgedCallExample.rebuilt_fixed
                       AgedCallExample.unsound_unit))
               = Exit 1%positive [Value_const (Tagged_immediate 1)]
                   empty_heap no_reads).
  { vm_compute. reflexivity. }
  assert (Hne : ~ values_related observed_equal
                  [Value_const (Tagged_immediate 0)]
                  [Value_const (Tagged_immediate 1)]).
  { simpl. intros [H _]. discriminate H. }
  split; [ exact Hleast | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ exists 3; vm_compute; reflexivity | ].
  split.
  { apply aged_premises; try (vm_compute; reflexivity). exact Hleast. }
  split; [ exact H0 | ].
  split; [ exact H1 | ].
  split; [ exact H2 | ].
  split; [ exact (not_refines_observably_aged_of_runs _ 5 5 _ _ _ _ _ _ _ _
                    _ _ H0 H1 Hne) | ].
  exact (not_refines_observably_aged_of_runs _ 5 5 _ _ _ _ _ _ _ _ _ _ H0 H2
           Hne).
Qed.

Lemma aged_codes_refine (age_of : static_env -> Code_age_relation.t)
    (statics : static_env) (v : value) :
  code_refines_for age_of statics v
    (AgedCallExample.code_returning 40%positive 41%positive 3%positive
       4%positive (Var 40%positive))
    (AgedCallExample.code_returning 43%positive 44%positive 5%positive
       6%positive (Var 43%positive)).
Proof.
  apply code_refines_for_steps; try reflexivity.
  - split; [ reflexivity | split; reflexivity ].
  - intros vs vars return_to exn_to hp rd Hb.
    destruct vs as [ | a [ | ? ? ] ]; try discriminate Hb.
    injection Hb as <-. eexists. split; [ reflexivity | ].
    simpl. apply result_refines_by_refl.
Qed.

Theorem aged_call_sound_example : aged_call_sound_example_statement.
Proof.
  cbv zeta.
  set (e := AgedCallExample.sound_unit).
  set (age_of := AgedCallExample.age_of).
  assert (H0 : run_aged age_of 5 (AgedCallExample.initial e)
               = Exit 1%positive [Value_const (Tagged_immediate 9)]
                   empty_heap no_reads).
  { vm_compute. reflexivity. }
  assert (H1 : run_aged age_of 5
                 (AgedCallExample.initial (AgedCallExample.rebuilt e))
               = Exit 1%positive [Value_const (Tagged_immediate 9)]
                   empty_heap no_reads).
  { vm_compute. reflexivity. }
  assert (H2 : run_aged age_of 5
                 (AgedCallExample.initial (AgedCallExample.rebuilt_fixed e))
               = Exit 1%positive [Value_const (Tagged_immediate 9)]
                   empty_heap no_reads).
  { vm_compute. reflexivity. }
  assert (Hrel : values_related observed_equal
                   [Value_const (Tagged_immediate 9)]
                   [Value_const (Tagged_immediate 9)]).
  { simpl. split; [ reflexivity | exact I ]. }
  split.
  { intros newer older c c' _ Hl Hc Hc'.
    apply one_link in Hl as [-> ->].
    vm_compute in Hc, Hc'. injection Hc as <-. injection Hc' as <-.
    apply aged_codes_refine. }
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split.
  { exists
      (match run_aged age_of 3 (AgedCallExample.initial e) with
       | Step s => s
       | _ => AgedCallExample.initial e
       end).
    split; [ vm_compute; reflexivity | ].
    split; vm_compute; reflexivity. }
  split; [ vm_compute; reflexivity | ].
  split; [ exact (refines_observably_aged_of_runs _ 5 5 _ _ _ _ _ _ _ _ H0 H1
                    Hrel) | ].
  exact (refines_observably_aged_of_runs _ 5 5 _ _ _ _ _ _ _ _ H0 H2 Hrel).
Qed.

(* ======================================================================== *)
(* What the fix does not keep                                               *)
(* ======================================================================== *)

Lemma behaviour_codes_refine :
  code_refines_for BehaviourExample.age_of
    (unit_statics BehaviourExample.unit_expr) (Value_symbol 80%positive)
    BehaviourExample.older BehaviourExample.newer.
Proof.
  apply code_refines_for_steps; try reflexivity.
  - split; [ reflexivity | split; reflexivity ].
  - intros vs vars return_to exn_to hp rd Hb.
    destruct vs as [ | a [ | ? ? ] ]; try discriminate Hb.
    injection Hb as <-. eexists. split; [ reflexivity | ].
    unfold result_refines_aged.
    match goal with
    | |- result_refines_by _ ?r _ =>
      assert (Hr : r = Undefined) by (vm_compute; reflexivity)
    end.
    rewrite Hr. exact I.
Qed.

Theorem sound_for_not_kept : sound_for_not_kept_statement.
Proof.
  cbv zeta.
  assert (Hleast : least_solution Examples.no_local BehaviourExample.facts
                     BehaviourExample.tables).
  { apply (FlambdaReaperLeastProofs.compute_least_sound _ _ 40).
    vm_compute. reflexivity. }
  assert (Hnw : ~ behaves
                  (State BehaviourExample.unit_expr [] [] empty_static_env
                     empty_heap no_reads)
                  Goes_wrong).
  { apply (not_wrong_of_run_exit 10 _ 1%positive
             [Value_block 0
                [Value_closure_of_set 71%positive
                   [(71%positive, 50%positive)]
                   [(90%positive, Value_const (Tagged_immediate 3))];
                 Value_symbol 80%positive]]
             empty_heap no_reads).
    vm_compute. reflexivity. }
  assert (Hslots : is_local_slots_only Examples.no_local).
  { intros f H. discriminate H. }
  assert (Hlocal : (∀ f : field, Examples.no_local f = false)
                   ∨ projection_free BehaviourExample.unit_expr = true).
  { left. intros f. reflexivity. }
  assert (Heq : BehaviourExample.rebuilt
                = rebuild_unit_cc_from_tables BehaviourExample.facts
                    BehaviourExample.tables BehaviourExample.table
                    (unit_code_deps Auto Examples.interface
                       BehaviourExample.unit_expr)
                    true BehaviourExample.unit_expr).
  { vm_compute. reflexivity. }
  assert (Hc50 : lookup 50%positive
                   (static_code (unit_statics BehaviourExample.rebuilt))
                 = Some BehaviourExample.older_rebuilt).
  { vm_compute. reflexivity. }
  assert (Hc51 : lookup 51%positive
                   (static_code (unit_statics BehaviourExample.rebuilt))
                 = Some BehaviourExample.newer).
  { vm_compute. reflexivity. }
  split; [ exact Hleast | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ exact Hnw | ].
  split.
  { intros newer older Hl. apply one_link in Hl as [-> ->].
    split; vm_compute; reflexivity. }
  split; [ exact Heq | ].
  split.
  { rewrite Heq.
    exact (FlambdaReaperLeastProofs.least_rebuild_cc_auto_refines_observably
             Examples.no_local Examples.interface BehaviourExample.unit_expr
             BehaviourExample.tables true Hslots Hlocal
             ltac:(vm_compute; reflexivity) ltac:(vm_compute; reflexivity)
             ltac:(vm_compute; reflexivity) ltac:(vm_compute; reflexivity)
             ltac:(vm_compute; reflexivity) Hleast). }
  split.
  { intros age_of. rewrite Heq.
    exact (least_rebuild_cc_auto_refines_aged Examples.no_local
             Examples.interface BehaviourExample.unit_expr
             BehaviourExample.tables true age_of Hslots Hlocal
             ltac:(vm_compute; reflexivity) ltac:(vm_compute; reflexivity)
             ltac:(vm_compute; reflexivity) ltac:(vm_compute; reflexivity)
             ltac:(vm_compute; reflexivity) Hleast Hnw). }
  split.
  { apply (conventions_agree_uniform _ _ BehaviourExample.older).
    vm_compute. repeat constructor. }
  split.
  { apply (conventions_agree_uniform _ _ BehaviourExample.older).
    vm_compute. repeat constructor. }
  split.
  { intros newer older c c' _ Hl Hc Hc'.
    apply one_link in Hl as [-> ->].
    assert (H50 : lookup 50%positive
                    (static_code (unit_statics BehaviourExample.unit_expr))
                  = Some BehaviourExample.older).
    { vm_compute. reflexivity. }
    assert (H51 : lookup 51%positive
                    (static_code (unit_statics BehaviourExample.unit_expr))
                  = Some BehaviourExample.newer).
    { vm_compute. reflexivity. }
    rewrite H50 in Hc. rewrite H51 in Hc'.
    injection Hc as <-. injection Hc' as <-.
    exact behaviour_codes_refine. }
  split; [ exact Hc50 | ].
  intros H.
  assert (Hat : closure_at_least BehaviourExample.relation
                  (unit_statics BehaviourExample.rebuilt)
                  (Value_symbol 80%positive) 51%positive).
  { exists 51%positive, []. split; [ vm_compute; reflexivity | constructor ]. }
  destruct (H 51%positive 50%positive _ _ Hat eq_refl Hc50 Hc51)
    as (_ & _ & _ & Hcalls).
  specialize (Hcalls [Value_const (Tagged_immediate 5)] [Singleton Value]
                [Singleton Value] (Outer 1%positive) (Outer 2%positive)
                empty_heap no_reads).
  revert Hcalls.
  apply (not_result_refines_by_exits _ 3 _ _ 1%positive
           [Value_const (Tagged_immediate 0)]
           [Value_const (Tagged_immediate 1)] empty_heap empty_heap no_reads
           no_reads).
  - vm_compute. reflexivity.
  - vm_compute. reflexivity.
  - intros H'. discriminate H'.
Qed.
