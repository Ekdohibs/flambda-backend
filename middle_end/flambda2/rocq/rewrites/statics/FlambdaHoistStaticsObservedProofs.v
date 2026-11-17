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


(* Proofs of the statement of FlambdaHoistStaticsObservedSpec.v. Nothing
   here needs review.

   The simulation of FlambdaHoistStaticsProofs.v is of a unit run on
   its own. So the two units are run on their own to their exits
   (FlambdaRunUnderSpec.behaves_under_statement), where the second
   static environment is the first with the bindings taken to the head
   bound beside it, and what follows is run from there by the same
   simulation, with nothing left to take out ([hoist_inv_after]). The
   invariant of the first run gives that the values of the exit hold
   only names that are bound. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaValueSpec FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaBindersSpec FlambdaWellFormedSpec
  FlambdaWellFormedProofs FlambdaStaticsInScopeSpec
  FlambdaStaticsInScopeProofs FlambdaImportsSpec FlambdaUnitFromProofs
  FlambdaExitWorldSpec FlambdaExitWorldProofs FlambdaRunUnderSpec
  FlambdaRunUnderProofs FlambdaObserversSpec FlambdaObserversProofs
  FlambdaNameAbstractionSpec FlambdaNameAbstractionProofs
  FlambdaHoistStaticsSpec FlambdaHoistStaticsProofs
  FlambdaHoistStaticsObservedSpec.
From OxCamlFlambda2 Require FlambdaStaticsScopeProofs.
Import ListNotations.

(* ======================================================================== *)
(* Two units followed by one thing                                          *)
(* ======================================================================== *)

(* Two units that behave alike on their own, and after whose exits what
   stands for the continuation of the exit behaves alike on what each
   left, behave alike from that continuation environment. One way. *)
Lemma under_one_way (conts : cont_env) (e e' : expr) (statics : static_env)
    (hp : heap) (rd : read_record) :
  code_closed e = true ->
  code_closed e' = true ->
  static_code_closed statics ->
  equivalent (State e [] [] statics hp rd) (State e' [] [] statics hp rd) ->
  (∀ (w w' : exit_world) (b : behaviour),
    exits_world (State e [] [] statics hp rd) w ->
    exits_world (State e' [] [] statics hp rd) w' ->
    step_result_behaves
      (apply_cont_value (resolve (exit_cont w) conts) (exit_values w)
         (exit_statics w) (exit_heap w) (exit_reads w))
      b ->
    step_result_behaves
      (apply_cont_value (resolve (exit_cont w') conts) (exit_values w')
         (exit_statics w') (exit_heap w') (exit_reads w'))
      b) ->
  ∀ b : behaviour,
    behaves (State e [] conts statics hp rd) b ->
    behaves (State e' [] conts statics hp rd) b.
Proof.
  intros Hc Hc' Hs Heq Hworlds b Hb.
  assert (Hvia :
    ∀ w : exit_world,
      exits_world (State e [] [] statics hp rd) w ->
      behaves (State e' [] conts statics hp rd) b).
  { intros w Hw.
    pose proof (proj1 (Heq _) (exits_world_exits _ w Hw)) as Hexit'.
    destruct (proj1 (exits_world_behaves _ _ _ _) Hexit') as
      [statics' [rd' Hw']].
    refine (proj2 (under_world conts e' statics hp rd _ b Hc' Hs Hw') _).
    apply (Hworlds w _ b Hw Hw').
    exact (proj1 (under_world conts e statics hp rd w b Hc Hs Hw) Hb). }
  assert (Hdec : b = Diverges ∨ b <> Diverges)
    by (destruct b; [ right | right | left ]; congruence).
  destruct Hdec as [-> | Hnot].
  - intros m.
    destruct (run m (State e' [] [] statics hp rd)) as [s | k vs hp' rd' | ]
      eqn:Hrun.
    + destruct (run_under conts e' [] statics hp rd m s Hc' Hs Hrun)
        as (s' & Hrun' & _).
      exists s'. exact Hrun'.
    + assert (Hexit : behaves (State e [] [] statics hp rd) (Exits k vs hp'))
        by (apply Heq; exists m; eexists; exact Hrun).
      destruct (proj1 (exits_world_behaves _ _ _ _) Hexit) as
        [statics0 [rd'' Hw]].
      exact (Hvia _ Hw m).
    + assert (Hwrong : behaves (State e [] [] statics hp rd) Goes_wrong)
        by (apply Heq; exists m; exact Hrun).
      pose proof (under_wrong conts e statics hp rd Hc Hs Hwrong) as Hwrong'.
      discriminate (behaviour_unique _ _ _ Hb Hwrong').
  - destruct (under_ends conts e statics hp rd b Hc Hs Hnot Hb)
      as [Hwrong | [w Hw]]; [ | exact (Hvia w Hw) ].
    pose proof (under_wrong conts e statics hp rd Hc Hs Hwrong) as Hwrong'.
    rewrite (behaviour_unique _ _ _ Hb Hwrong').
    apply (under_wrong conts e' statics hp rd Hc' Hs). apply Heq. exact Hwrong.
Qed.

Lemma equivalent_under (conts : cont_env) (e e' : expr) (statics : static_env)
    (hp : heap) (rd : read_record) :
  code_closed e = true ->
  code_closed e' = true ->
  static_code_closed statics ->
  equivalent (State e [] [] statics hp rd) (State e' [] [] statics hp rd) ->
  (∀ (w w' : exit_world) (b : behaviour),
    exits_world (State e [] [] statics hp rd) w ->
    exits_world (State e' [] [] statics hp rd) w' ->
    step_result_behaves
      (apply_cont_value (resolve (exit_cont w) conts) (exit_values w)
         (exit_statics w) (exit_heap w) (exit_reads w))
      b
    <-> step_result_behaves
          (apply_cont_value (resolve (exit_cont w') conts) (exit_values w')
             (exit_statics w') (exit_heap w') (exit_reads w'))
          b) ->
  equivalent (State e [] conts statics hp rd) (State e' [] conts statics hp rd).
Proof.
  intros Hc Hc' Hs Heq Hworlds b. split.
  - apply (under_one_way conts e e' statics hp rd Hc Hc' Hs Heq).
    intros w w' b' Hw Hw'. exact (proj1 (Hworlds w w' b' Hw Hw')).
  - apply (under_one_way conts e' e statics hp rd Hc' Hc Hs
             (equivalent_sym _ _ Heq)).
    intros w' w b' Hw' Hw. exact (proj2 (Hworlds w w' b' Hw Hw')).
Qed.

(* ======================================================================== *)
(* What follows, from what the two units left                               *)
(* ======================================================================== *)

Section After.

Variable deleted : list code_id.
Variable src_code : list (code_id * code).
Variable src_values : list (symbol * statically_allocated_value).

Hypothesis deleted_not_src :
  ∀ id : code_id, among id deleted = true -> lookup id src_code = None.

(* What follows a unit, run on values that hold only names that are
   bound from the static environment of a state of the first run,
   satisfies the invariant of the first run. *)
Lemma hoist_inv_after (last : state) (body : expr) (vars : env) (hp : heap)
    (rd : read_record) :
  hoist_inv deleted src_code src_values last ->
  env_values_all (value_statics_bound deleted (state_statics last)) vars ->
  heap_values_all (value_statics_bound deleted (state_statics last)) hp ->
  binds_no_static body = true ->
  expr_statics_in_scope [] [] body = true ->
  hoist_inv deleted src_code src_values
    (State body vars [] (state_statics last) hp rd).
Proof.
  intros (Hcodes & Hsymbols & Hterms & Hbound & Hknown) Hvars Hheap Hno
    Hscope.
  destruct
    (statics_nowhere code_ids_of_binding body (no_static_nowhere body Hno))
    as (P & B & _).
  split; [ exact (inv_after _ _ last body vars hp rd Hcodes Hno) | ].
  split; [ exact (inv_after _ _ last body vars hp rd Hsymbols Hno) | ].
  split; [ | split ].
  - destruct Hterms as (_ & _ & _ & _ & Hfine).
    unfold terms_inv. simpl_state.
    refine (conj P (conj _ (conj _ (conj _ Hfine)))).
    + exact (expr_scope_sub body [] _ [] _ (sub_nil _) (sub_nil _) Hscope).
    + rewrite (no_static_names deleted_code_ids_of_binding body Hno).
      reflexivity.
    + intros k syms ids _ _. apply Good_outer.
  - destruct Hbound as (_ & _ & Hstatics & _).
    exact (conj Hvars (conj I (conj Hstatics Hheap))).
  - destruct Hknown as (_ & _ & Hstatics).
    refine (conj _ (conj I Hstatics)).
    apply expr_known_nothing. exact B.
Qed.

(* After an exit of a state of the first run, what follows behaves the
   same from the static environment of that state and from the one of
   the second run. *)
Lemma after_equivalent (last : state) (k : continuation) (vs : list value)
    (hp : heap) (rd : read_record) (slots : list value_slot) (c : cont_value) :
  hoist_inv deleted src_code src_values last ->
  step last = Exit k vs hp rd ->
  observer slots c ->
  ∀ b : behaviour,
    step_result_behaves (apply_cont_value c vs (state_statics last) hp rd) b
    <-> step_result_behaves
          (apply_cont_value c vs
             (hoist_env src_code src_values (state_statics last)) hp rd)
          b.
Proof.
  intros Hinv Hexit Hc b.
  destruct c
    as [ [params metadata body] vars conts recursive | k0 | c0 extra
       | remaining remaining_arity return_arity return_to exn_to ];
    simpl in Hc; try contradiction; [ | reflexivity ].
  destruct vars; [ | contradiction ].
  destruct conts; [ | contradiction ].
  destruct recursive; [ | contradiction ].
  destruct Hc as (Hno & Hscope & _).
  simpl.
  destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
    [ | reflexivity ].
  destruct (bind_params params vs []) as [bound | ] eqn:Hbind;
    [ | reflexivity ].
  pose proof Hinv as (_ & _ & Hterms & Hbound & _).
  pose proof (values_step deleted last Hterms Hbound) as Hvs.
  rewrite Hexit in Hvs.
  assert (Hvars :
    env_values_all (value_statics_bound deleted (state_statics last)) bound).
  { exact (bind_params_values_all _ params vs [] bound Hvs (Forall_nil _)
             Hbind). }
  assert (Hheap :
    heap_values_all (value_statics_bound deleted (state_statics last)) hp).
  { rewrite (step_exit_heap last k vs hp rd Hexit).
    exact (proj2 (proj2 (proj2 Hbound))). }
  pose proof
    (hoist_inv_after last body bound hp rd Hinv Hvars Hheap Hno Hscope)
    as Hafter.
  destruct
    (statics_nowhere code_ids_of_binding body (no_static_nowhere body Hno))
    as (_ & B & _).
  refine
    (bisimulation_equivalent (hoisted_related deleted src_code src_values)
       (hoisted_related_step deleted src_code src_values deleted_not_src)
       _ _ _ b).
  split; [ | exact Hafter ].
  unfold hoist_state. simpl_state.
  rewrite (proj1 (without_hoisted_nothing body B)). reflexivity.
Qed.

End After.

(* ======================================================================== *)
(* The statement                                                            *)
(* ======================================================================== *)

Lemma hoist_statics_equivalent_under (slots : list value_slot)
    (u : unit_interface) (e e' : expr) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) :
  hoist_statics u e = Some e' ->
  static_code statics = [] ->
  static_names_free_in statics e = true ->
  Forall
    (fun binding =>
       allocated_statics_among (map fst (static_values statics)) []
         (snd binding)
       = true)
    (static_values statics) ->
  heap_values_all
    (fun v =>
       value_statics_among (map fst (static_values statics)) [] v = true)
    hp ->
  observers slots conts ->
  equivalent (State e [] conts statics hp rd) (State e' [] conts statics hp rd).
Proof.
  intros Hhoist Hno_code Hfree Hamong Hheap Hobservers.
  pose proof
    (hoist_statics_equivalent_from u e e' statics hp rd Hhoist Hno_code Hfree
       Hamong Hheap)
    as Halone.
  unfold hoist_statics in Hhoist.
  destruct (well_formed_unit u e) eqn:Hwf; simpl in Hhoist;
    [ | discriminate ].
  destruct (statics_in_scope [] e) eqn:Hscope; simpl in Hhoist;
    [ | discriminate ].
  destruct (interface_conts_not_rebound u e) eqn:Hconts; simpl in Hhoist;
    [ | discriminate ].
  destruct (static_names_apart [] e) eqn:Hapart; [ | discriminate ].
  injection Hhoist as <-.
  destruct
    (hoist_statics_related_from u e statics hp rd Hwf Hscope Hconts Hapart
       Hno_code Hfree Hamong Hheap)
    as (src_code & src_values & Hdeleted & Hsrc & Hrel).
  pose proof (well_formed_unit_code_closed u e Hwf) as Hclosed.
  pose proof
    (proj1
       (FlambdaStaticsScopeProofs.scoped_let_static_around _ _ _
          (FlambdaStaticsScopeProofs.hoisted_bindings_no_free_names e Hclosed)
          (FlambdaStaticsScopeProofs.without_hoisted_scoped e) Hclosed))
    as Hclosed'.
  assert (Hstatics : static_code_closed statics).
  { unfold static_code_closed. rewrite Hno_code. constructor. }
  apply (equivalent_under conts _ _ statics hp rd Hclosed Hclosed' Hstatics
           Halone).
  intros w w' b (n & last & Hrun & Hexit & Hw_statics) Hw'.
  pose proof
    (bisimulation_run
       (hoisted_related (deleted_code_ids e) src_code src_values)
       (hoisted_related_step (deleted_code_ids e) src_code src_values
          Hdeleted)
       n _ _ Hrel)
    as Hn.
  rewrite Hrun in Hn.
  destruct
    (run n (State (without_hoisted e) [] [] (Static_env src_code src_values)
              hp rd))
    as [last' | | ] eqn:Hrun'; simpl in Hn; try contradiction.
  destruct Hn as [-> Hinv].
  pose proof
    (hoisted_related_step (deleted_code_ids e) src_code src_values Hdeleted
       last _ (conj eq_refl Hinv))
    as Hstep.
  rewrite Hexit in Hstep.
  destruct (step (hoist_state src_code src_values last))
    as [ | k' vs' hp' rd' | ] eqn:Hexit'; simpl in Hstep; try contradiction.
  destruct Hstep as (<- & <- & <- & <-).
  assert (Hw0 :
    exits_world
      (State (Let_static (hoisted_bindings e) (without_hoisted e)) [] []
         statics hp rd)
      (Exit_world (exit_cont w) (exit_values w)
         (hoist_env src_code src_values (state_statics last)) (exit_heap w)
           (exit_reads w))).
  { exists (S n), (hoist_state src_code src_values last).
    split; [ | split; [ exact Hexit' | reflexivity ] ].
    rewrite run_succ, step_let_static, Hsrc. exact Hrun'. }
  rewrite (exits_world_unique _ _ _ Hw' Hw0).
  cbn [exit_cont exit_values exit_statics exit_heap].
  rewrite <- Hw_statics.
  exact (after_equivalent (deleted_code_ids e) src_code
           src_values Hdeleted last (exit_cont w) (exit_values w)
           (exit_heap w) rd_ slots _ Hinv Hexit
           (observers_resolve slots conts (exit_cont w) Hobservers) b).
Qed.

Theorem hoist_statics_equivalent_observed :
  hoist_statics_equivalent_observed_statement.
Proof.
  intros imports conts u e e' Hhoist Hfree Hobservers.
  exact (hoist_statics_equivalent_under [] u e e' conts
           (imports_statics imports) empty_heap no_reads Hhoist eq_refl Hfree
           (imports_statics_among _ imports) (heap_values_all_empty _)
           Hobservers).
Qed.
