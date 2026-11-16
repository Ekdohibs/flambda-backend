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


(* The proofs of the statements of FlambdaJoinInTermsSpec.v. This file
   needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaMapProofs FlambdaSemanticsSpec FlambdaClosuresSpec
  FlambdaNameAbstractionSpec FlambdaNameAbstractionProofs.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeWellFormedSpec
  FlambdaTypeMeaningSpec FlambdaTypeEqualSpec FlambdaTypeEqualProofs
  FlambdaBindingTime FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaTypingEnvInvariantSpec FlambdaTypingEnvInvariantProofs
  FlambdaTypingEnvKindedSpec FlambdaTypingEnvKindedProofs FlambdaJoinEnv
  FlambdaJoinSpec FlambdaJoinEnvSpec FlambdaJoinEnvProofs
  FlambdaJoinStateSpec FlambdaJoinStateProofs FlambdaJoinDefineSpec
  FlambdaJoinDefineProofs FlambdaJoinFitsSpec FlambdaJoinFitsProofs
  FlambdaJoinAddsSpec FlambdaJoinAddsProofs FlambdaJoinKindsSpec
  FlambdaJoinKindsProofs FlambdaJoinNamesSpec FlambdaJoinNamesProofs
  FlambdaTypeAliasKindsSpec FlambdaTypeAliasKindsProofs FlambdaJoinTypes
  FlambdaJoinTypesSpec FlambdaJoinTypesWithEnvSpec
  FlambdaJoinTypesWithEnvProofs FlambdaMeetGrammar FlambdaExpandHead
  FlambdaExpandHeadSpec FlambdaExpandHeadProofs FlambdaMeetEnv FlambdaMeet
  FlambdaMeetSpec FlambdaMeetStepsSpec FlambdaMeetStepsProofs
  FlambdaMeetEnvSpec FlambdaMeetEnvProofs FlambdaJoinMeet
  FlambdaJoinMeetSpec FlambdaJoinMeetProofs FlambdaJoinWellFormedSpec
  FlambdaJoinWellFormedProofs FlambdaJoinEquationNamesSpec
  FlambdaJoinEquationNamesProofs FlambdaMeetKeepsSpec FlambdaMeetKeepsProofs
  FlambdaMeetTermsSpec FlambdaMeetTermsProofs FlambdaJoinMeetFitsSpec
  FlambdaJoinMeetFitsProofs FlambdaMeetExamplesSpec FlambdaJoinEnvSpec
  FlambdaJoinProofs FlambdaJoinRepairSpec FlambdaJoinRepairProofs
  FlambdaJoinInTermsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The join with a meet of expanded types that is given: five parts         *)
(* ======================================================================== *)

(* The proofs of FlambdaJoinMeetProofs.v and FlambdaJoinMeetFitsProofs.v
   for the join at the meet of the model, with the meet given. *)

Lemma join_with_external_ids {A : Type} (join_choices : Join_env.choices)
    (join_depth : nat) (fuel : fuels) (mh : meet_expanded_head)
    (avoid : list variable) (inputs : join_inputs A) (target : Typing_env.t)
    (analysis : Analysis.t A) :
  model_join_with join_choices join_depth fuel mh avoid inputs
  = Some (target, analysis) ->
  Analysis.external_ids analysis = ids_of_uses inputs.
Proof.
  intros Hjoin.
  unfold model_join_with, join_with, cut_and_n_way_join_with_analysis in Hjoin.
  cbv zeta in Hjoin.
  destruct (Join_env.cut_for_joins _ _) as [envs | ]; [ | discriminate ].
  destruct (Join_env.cut_and_n_way_join0 _ _ _ _ _ _ _ _ _ _)
    as [[[menv_t definitions] names] | ]; [ | discriminate ].
  destruct (Analysis.create _ envs definitions) as [analysis' | ]
    eqn:Hanalysis; [ | discriminate ].
  injection Hjoin as _ <-.
  unfold Analysis.create in Hanalysis.
  destruct (Analysis.definitions_at_normal_mode envs (compact definitions))
    as [at_normal_mode | ]; [ | discriminate ].
  injection Hanalysis as <-. reflexivity.
Qed.

Lemma join_with_keeps {A : Type} (join_choices : Join_env.choices)
    (join_depth : nat) (fuel : fuels) (mh : meet_expanded_head)
    (avoid : list variable) (inputs : join_inputs A) (target : Typing_env.t)
    (analysis : Analysis.t A) (envs : Joined_envs.t) (s3 : Join_env.t)
    (relations3 : Join_env.inverse_relations)
    (kept : list (variable * flambda_kind))
    (equations relation_equations : list (name * flambda_type))
    (definitions : finite_map (Simples_in_joined_envs.t * flambda_kind)) :
  meet_expanded_head_keeps mh ->
  join_inputs_well_formed typing_env_invariant avoid inputs ->
  uses_kinded inputs ->
  uses_agree_with_source_on_kinds inputs ->
  typing_env_fits (source inputs) ->
  model_join_with join_choices join_depth fuel mh avoid inputs
  = Some (target, analysis) ->
  model_join_run join_choices join_depth fuel avoid inputs envs s3 relations3
    kept equations relation_equations definitions ->
  types_fit_otherwise (Meet_env.typing_env (model_join_defined inputs s3 kept))
    (equations ++ relation_equations) ->
  typing_env_fits target
  ∧ join_keeps_source (source inputs) target
  ∧ join_introduces avoid inputs target.
Proof.
  intros Hmh Hinputs Hkinded Hagree Hfits Hjoin Hrun Hrest.
  pose proof Hrun
    as [s1 [s2 [relations2 [Hcut [Haliases [Hdelayed [Hloop Hadds]]]]]]].
  change (Join_env.cut_for_joins (cut_after inputs) (numbered_uses inputs)
          = Some envs) in Hcut.
  set (initial_env := Meet_env.create (source inputs)) in *.
  destruct (Keyed.is_empty envs) eqn:Hempty.
  { (* No environment to join: the target is bottom. *)
    destruct (join_with_unfold join_choices join_depth fuel mh
                avoid inputs target analysis envs Hjoin Hcut)
      as [target_env [definitions' [names [Hjoin0 ->]]]].
    unfold Join_env.cut_and_n_way_join0 in Hjoin0. rewrite Hempty in Hjoin0.
    injection Hjoin0 as <- _ _.
    cbn [Join_env.make_bottom meet_env_functions].
    destruct (make_bottom_keeps initial_env Hfits) as [Hextends Hfits'].
    split; [ exact Hfits' | ].
    split; [ exact (extends_keeps_source _ _ Hextends) | ].
    intros x Hx Hnot. exfalso. apply Hnot. exact Hx. }
  destruct (join_with_replayed join_choices join_depth fuel mh
              avoid inputs target analysis envs s3 relations3 kept equations
              relation_equations definitions Hjoin Hrun Hempty)
    as [menv1 [target_env [Hadd1 [Hadd2 [-> _]]]]].
  unfold model_join_defined in Hrest, Hadd1. fold initial_env in Hrest, Hadd1.
  set (names :=
         Bindings_in_target_env.names_to_avoid (Join_env.bindings s3)) in *.
  set (menv_a := Meet_env.with_names_to_avoid initial_env names) in *.
  set (menv0 :=
         variables_defined meet_env_functions menv_a
           (Bindings_in_target_env.imported_variables_in_order
              (Join_env.bindings s3)
            ++ kept)) in *.
  destruct (inputs_first_phases join_choices join_depth fuel avoid inputs envs
              s1 s2 s3 relations2 relations3 Hinputs Hkinded Hagree Hcut
              Haliases Hdelayed Hloop)
    as (Htrue3 & Hrelations3 & Hgrows3 & Hstate3 & Hnonempty3 & Hscoped3 & HX
        & HR & Hnames_kept & Hmade & Huse3 & Hinvariant3 & Hkinded3
        & Hsymbols3).
  destruct (last_phase_kept _ (fuel_of_env fuel) (source inputs) s3 relations3
              mh menv_a menv1 target_env kept equations relation_equations
              definitions Htrue3 Hrelations3 Hnonempty3 Hscoped3 Hstate3
              Hinvariant3 Hkinded3 Hsymbols3 HX HR eq_refl Hfits Hadds Hrest
              Hmh Hadd1 Hadd2) as [Hkept0 [Hstep1 Hstep2]].
  fold menv0 in Hkept0, Hstep1.
  assert (Hextends_a : meet_env_extends initial_env menv_a).
  { apply with_names_extends. intros x Hx. discriminate Hx. }
  pose proof (meet_env_extends_trans _ _ _ Hextends_a (proj1 Hkept0))
    as Hextends0.
  pose proof (meet_env_extends_trans _ _ _ Hextends0 (proj1 (proj1 Hstep1)))
    as Hextends1.
  pose proof (meet_env_extends_trans _ _ _ Hextends1 (proj1 (proj1 Hstep2)))
    as Hextends2.
  destruct (last_phase_adds_kept _ (fuel_of_env fuel) s3 relations3 kept
              equations relation_equations definitions Hadds)
    as [Hdistinct Hkept].
  assert (Hsource3 : Meet_env.typing_env menv_a = Join_env.source_env s3).
  { rewrite (proj2 (proj2 Htrue3)). reflexivity. }
  destruct (target_variables_defined s3 menv_a kept (proj1 Htrue3) Hsource3
              Hfits Hkept Hdistinct) as [_ [Hmem0 [_ Hnames0]]].
  fold menv0 in Hmem0, Hnames0.
  assert (Hkept_made :
            ∀ x : variable, x ∈ map fst kept -> is_existential s3 x).
  { intros x Hx. apply in_map_iff in Hx. destruct Hx as [[y k] [Heq Hin]].
    cbn in Heq. subst y. apply is_bound_lookup. exists k.
    exact (in_order_lookup _ _ _ (Hkept _ Hin)). }
  destruct (proj1 Htrue3) as [_ [_ [_ [_ Himports]]]].
  split; [ exact (proj2 (proj1 Hstep2)) | ].
  split; [ exact (extends_keeps_source _ _ Hextends2) | ].
  intros x Hx Hnot. unfold knows in Hx, Hnot.
  assert (Hnot' : Typing_env.mem None (source inputs) (Name_var x) = false).
  { destruct (Typing_env.mem None (source inputs) (Name_var x));
      [ exfalso; exact (Hnot eq_refl) | reflexivity ]. }
  split.
  - intros Hterms. unfold in_scope_in_terms in Hterms.
    rewrite (last_phase_not_in_terms _ (fuel_of_env fuel) s3 relations3 menv_a
               menv1 target_env kept equations relation_equations definitions
               (proj1 Htrue3) Hsource3 Hfits Hadds (proj1 (proj1 Hstep1))
               (proj1 (proj1 Hstep2))
               (fun y H1 H0 => proj2 (proj2 (proj2 Hstep1 y H1 H0)))
               (fun y H2 H1 => proj2 (proj2 (proj2 Hstep2 y H2 H1))) x Hx
               Hnot') in Hterms.
    discriminate Hterms.
  - assert (Hnew :
              among x (Meet_env.names_to_avoid menv0) = false ->
              among x avoid = false).
    { intros Hav. destruct (among x avoid) eqn:Hx_avoid; [ | reflexivity ].
      rewrite Hnames0 in Hav.
      change (among x names = false) in Hav.
      pose proof (proj2 (among_In _ _)
                    (Hnames_kept x (proj1 (among_In _ _) Hx_avoid))) as Hin.
      change (among x names = true) in Hin. rewrite Hin in Hav.
      discriminate Hav. }
    destruct (Typing_env.mem None (Meet_env.typing_env menv0) (Name_var x))
      eqn:H0.
    + apply Hmem0 in H0. destruct H0 as [H0 | [H0 | H0]].
      * exfalso. rewrite (proj2 (proj2 Htrue3)) in H0. exact (Hnot H0).
      * left. destruct (Himports x H0) as [_ [index [use [Huse Hknows]]]].
        exists index, use.
        destruct (proj1 (Huse3 index use) Huse) as [Hat Hb].
        split; [ exact Hat | ]. split; [ exact Hb | exact Hknows ].
      * right. destruct (among x avoid) eqn:Hav; [ | reflexivity ].
        exfalso.
        exact (Hmade x (Hkept_made x H0) (proj1 (among_In _ _) Hav)).
    + right. apply Hnew.
      destruct (Typing_env.mem None (Meet_env.typing_env menv1) (Name_var x))
        eqn:H1; [ exact (proj1 (proj2 Hstep1 x H1 H0)) | ].
      destruct (among x (Meet_env.names_to_avoid menv0)) eqn:Hav0;
        [ | reflexivity ].
      pose proof (proj1 (proj2 Hstep2 x Hx H1)) as Hnew2.
      rewrite (extends_avoids menv0 menv1 x (proj1 (proj1 Hstep1)) Hav0)
        in Hnew2.
      discriminate Hnew2.
Qed.

Lemma join_with_describes_uses {A : Type} (join_choices : Join_env.choices)
    (join_depth : nat) (fuel : fuels) (mh : meet_expanded_head)
    (avoid : list variable) (inputs : join_inputs A) (target : Typing_env.t)
    (analysis : Analysis.t A) (envs : Joined_envs.t) (s3 : Join_env.t)
    (relations3 : Join_env.inverse_relations)
    (kept : list (variable * flambda_kind))
    (equations relation_equations : list (name * flambda_type))
    (definitions : finite_map (Simples_in_joined_envs.t * flambda_kind)) :
  meet_expanded_head_keeps mh ->
  join_inputs_well_formed typing_env_invariant avoid inputs ->
  uses_kinded inputs ->
  uses_agree_with_source_on_kinds inputs ->
  typing_env_fits (source inputs) ->
  model_join_with join_choices join_depth fuel mh avoid inputs
  = Some (target, analysis) ->
  model_join_run join_choices join_depth fuel avoid inputs envs s3 relations3
    kept equations relation_equations definitions ->
  types_fit_otherwise (Meet_env.typing_env (model_join_defined inputs s3 kept))
    (equations ++ relation_equations) ->
  analysis_describes_uses inputs target analysis.
Proof.
  intros Hmh Hinputs Hkinded Hagree Hfits Hjoin Hrun Hrest.
  pose proof Hrun
    as [s1 [s2 [relations2 [Hcut [Haliases [Hdelayed [Hloop Hadds]]]]]]].
  change (Join_env.cut_for_joins (cut_after inputs) (numbered_uses inputs)
          = Some envs) in Hcut.
  pose proof (join_with_external_ids join_choices join_depth
                fuel mh avoid inputs target analysis Hjoin) as Hids.
  assert (Hids_found :
            ∀ (index : Index.t) (id : A),
              Keyed.lookup Index.equal index (Analysis.external_ids analysis)
              = Some id ->
              ∃ use : Typing_env.t,
                nth_error (uses inputs) index = Some (id, use)
                ∧ Typing_env.is_bottom use = false).
  { intros index id Hlookup. rewrite Hids in Hlookup.
    exact (ids_of_uses_found inputs index id Hlookup). }
  set (initial_env := Meet_env.create (source inputs)) in *.
  destruct (Keyed.is_empty envs) eqn:Hempty.
  { (* No environment to join: the analysis has no definition. *)
    destruct (join_with_unfold join_choices join_depth fuel mh
                avoid inputs target analysis envs Hjoin Hcut)
      as [target_env [definitions' [names [Hjoin0 Htarget]]]].
    destruct (join_with_analysis join_choices join_depth fuel mh
                avoid inputs target analysis envs target_env definitions'
                names Hjoin Hcut Hjoin0)
      as [at_normal_mode [Hnormal [Hdefs Hat]]].
    unfold Join_env.cut_and_n_way_join0 in Hjoin0. rewrite Hempty in Hjoin0.
    injection Hjoin0 as _ <- _.
    split; [ exact Hids_found | ]. split.
    - intros x simples kind index simple Hx. rewrite Hdefs in Hx.
      discriminate Hx.
    - intros x simples kind index simple Hx. rewrite Hat in Hx.
      destruct (definitions_at_normal_mode_lookup envs _ _ Hnormal x simples
                  kind Hx) as [simples0 [Hin _]].
      apply in_compact in Hin. discriminate Hin. }
  destruct (join_with_replayed join_choices join_depth fuel mh
              avoid inputs target analysis envs s3 relations3 kept equations
              relation_equations definitions Hjoin Hrun Hempty)
    as [menv1 [target_env [Hadd1 [Hadd2 [-> [at_normal_mode [Hnormal
          [Hdefinitions Hat_normal]]]]]]]].
  unfold model_join_defined in Hrest, Hadd1. fold initial_env in Hrest, Hadd1.
  set (names :=
         Bindings_in_target_env.names_to_avoid (Join_env.bindings s3)) in *.
  set (menv_a := Meet_env.with_names_to_avoid initial_env names) in *.
  set (menv0 :=
         variables_defined meet_env_functions menv_a
           (Bindings_in_target_env.imported_variables_in_order
              (Join_env.bindings s3)
            ++ kept)) in *.
  destruct (inputs_first_phases join_choices join_depth fuel avoid inputs envs
              s1 s2 s3 relations2 relations3 Hinputs Hkinded Hagree Hcut
              Haliases Hdelayed Hloop)
    as (Htrue3 & Hrelations3 & Hgrows3 & Hstate3 & Hnonempty3 & Hscoped3 & HX
        & HR & Hnames_kept & Hmade & Huse3 & Hinvariant3 & Hkinded3
        & Hsymbols3).
  destruct (last_phase_kept _ (fuel_of_env fuel) (source inputs) s3 relations3
              mh menv_a menv1 target_env kept equations relation_equations
              definitions Htrue3 Hrelations3 Hnonempty3 Hscoped3 Hstate3
              Hinvariant3 Hkinded3 Hsymbols3 HX HR eq_refl Hfits Hadds Hrest
              Hmh Hadd1 Hadd2) as [Hkept0 [Hstep1 Hstep2]].
  fold menv0 in Hkept0, Hstep1.
  destruct (last_phase_adds_kept _ (fuel_of_env fuel) s3 relations3 kept
              equations relation_equations definitions Hadds)
    as [Hdistinct Hkept].
  assert (Hsource3 : Meet_env.typing_env menv_a = Join_env.source_env s3).
  { rewrite (proj2 (proj2 Htrue3)). reflexivity. }
  destruct (target_variables_defined s3 menv_a kept (proj1 Htrue3) Hsource3
              Hfits Hkept Hdistinct) as [_ [_ [Hkinds0 _]]].
  fold menv0 in Hkinds0.
  assert (Hcore :
            ∀ (x : variable) (simples : Simples_in_joined_envs.t)
              (kind : flambda_kind) (index : Index.t) (simple : simple),
              lookup x definitions = Some (simples, kind) ->
              Simples_in_joined_envs.find index simples = Some simple ->
              simple_of_kind (Meet_env.typing_env target_env) (Var x) kind
              ∧ ∃ (id : A) (use : Typing_env.t),
                  nth_error (uses inputs) index = Some (id, use)
                  ∧ Keyed.lookup Index.equal index
                      (Analysis.external_ids analysis)
                    = Some id
                  ∧ simple_of_kind use simple kind
                  ∧ Join_env.joined_env s3 index = Some use).
  { intros x simples kind index simple Hx Hfind.
    pose proof (last_phase_adds_definitions _ (fuel_of_env fuel)
                  (source inputs) s3 relations3 kept equations
                  relation_equations definitions Htrue3
                  (scoped_aliases_not_made s3 (proj1 Htrue3) (proj2 Hscoped3))
                  Hadds x (simples, kind) Hx) as Hfinal.
    destruct (last_phase_adds_definitions_kinded _ (fuel_of_env fuel)
                (source inputs) s3 relations3 kept equations
                relation_equations definitions Htrue3
                (proj1 (proj2 (proj2 (proj2 HX)))) Hadds x (simples, kind) Hx)
      as [Hkind [e He]].
    cbn [snd] in Hkind. split.
    - apply (extends_kinds menv0 target_env (Var x) kind
               (meet_env_extends_trans _ _ _ (proj1 (proj1 Hstep1))
                  (proj1 (proj1 Hstep2)))).
      exact (Hkinds0 x kind Hfinal Hkind).
    - pose proof (proj2 (proj2 (proj2 (proj1 (proj2 Htrue3))))) as Hin_envs.
      destruct (Hin_envs e simples kind He index simple Hfind)
        as [use [Huse _]].
      destruct (proj1 (Huse3 index use) Huse) as [[id Hnth] Hb].
      exists id, use. split; [ exact Hnth | ]. split.
      + rewrite Hids. exact (ids_of_uses_has inputs index id use Hnth Hb).
      + split; [ | exact Huse ].
        exact (proj2 (proj2 Hstate3) e simples kind index simple use He Hfind
                 Huse). }
  split; [ exact Hids_found | ]. split.
  - intros x simples kind index simple Hx Hfind. rewrite Hdefinitions in Hx.
    destruct (Hcore x simples kind index simple Hx Hfind)
      as [Htarget [id [use [Hnth [Hid [Hk _]]]]]].
    split; [ exact Htarget | ]. exists id, use.
    split; [ exact Hnth | ]. split; [ exact Hid | exact Hk ].
  - intros x simples' kind index canonical Hx Hfind.
    rewrite Hat_normal in Hx.
    destruct (definitions_at_normal_mode_lookup envs _ _ Hnormal x simples'
                kind Hx) as [simples [Hin Hat]].
    apply in_compact in Hin.
    destruct (simples_at_normal_mode_find envs _ _ Hat index canonical Hfind)
      as [simple [use' [Hin' [Huse' Hcanonical]]]].
    apply (keyed_in_compact Index.equal index_equal_decides) in Hin'.
    destruct (Hcore x simples kind index simple Hin Hin')
      as [Htarget [id [use [Hnth [Hid [Hk Huse]]]]]].
    pose proof Huse as Huse0. rewrite (proj1 Hgrows3) in Huse0.
    change (Joined_envs.get_nth_joined_env envs index = Some use) in Huse0.
    rewrite Huse' in Huse0. injection Huse0 as ->.
    split; [ exact Htarget | ]. exists id, use.
    split; [ exact Hnth | ]. split; [ exact Hid | ].
    exact (canonical_at_mode_kinded use Name_mode.normal simple canonical kind
             (Hinvariant3 index use Huse) (Hkinded3 index use Huse) Hk
             Hcanonical).
Qed.

Lemma join_with_run_found {A : Type} (join_choices : Join_env.choices)
    (join_depth : nat) (fuel : fuels) (mh : meet_expanded_head)
    (avoid : list variable) (inputs : join_inputs A) (target : Typing_env.t)
    (analysis : Analysis.t A) :
  model_join_with join_choices join_depth fuel mh avoid inputs
  = Some (target, analysis) ->
  ∃ envs : Joined_envs.t,
    Join_env.cut_for_joins (cut_after inputs) (numbered_uses inputs)
    = Some envs
    ∧ (Keyed.is_empty envs = true
       ∨ ∃ (s3 : Join_env.t) (relations3 : Join_env.inverse_relations)
           (kept : list (variable * flambda_kind))
           (equations relation_equations : list (name * flambda_type))
           (definitions :
              finite_map (Simples_in_joined_envs.t * flambda_kind)),
           model_join_run join_choices join_depth fuel avoid inputs envs s3
             relations3 kept equations relation_equations definitions).
Proof.
  intros Hjoin.
  pose proof Hjoin as Hjoin'.
  unfold model_join_with, join_with, cut_and_n_way_join_with_analysis in Hjoin'.
  cbv zeta in Hjoin'.
  destruct (Join_env.cut_for_joins _ _) as [envs | ] eqn:Hcut;
    [ | discriminate ].
  clear Hjoin'.
  change (Join_env.cut_for_joins (cut_after inputs) (numbered_uses inputs)
          = Some envs) in Hcut.
  exists envs. split; [ reflexivity | ].
  destruct (Keyed.is_empty envs) eqn:Hempty; [ left; reflexivity | right ].
  destruct (join_with_unfold join_choices join_depth fuel mh avoid
              inputs target analysis envs Hjoin Hcut)
    as [target_env [definitions [names [Hjoin0 _]]]].
  destruct (cut_and_n_way_join0_replayed Meet_env.t meet_expanded_head
              meet_env_functions (join_parameters join_depth)
              (fuel_of_env fuel) join_choices
              (model_join_of_types join_depth fuel)
              mh avoid (Meet_env.create (source inputs)) target_env
              (source inputs) envs definitions names Hempty Hjoin0)
    as [s1 [s2 [s3 [relations2 [relations3 [kept [equations
          [relation_equations [menv1 [Haliases [Hdelayed [Hloop [Hnames
            [Hadds _]]]]]]]]]]]]]].
  subst names.
  exists s3, relations3, kept, equations, relation_equations, definitions.
  exists s1, s2, relations2.
  split; [ exact Hcut | ]. split; [ exact Haliases | ].
  split; [ exact Hdelayed | ]. split; [ exact Hloop | exact Hadds ].
Qed.


Theorem model_join_with_guarantees_on_inputs
  : model_join_with_guarantees_on_inputs_statement.
Proof.
  intros A join_choices join_depth fuel mh avoid inputs target analysis Hmh
    Hinputs Hkinded Hagree Hfits Hfit_uses Hjoin.
  destruct (join_with_run_found join_choices join_depth fuel mh avoid inputs
              target analysis Hjoin)
    as [envs [Hcut [Hempty | [s3 [relations3 [kept [equations
          [relation_equations [definitions Hrun]]]]]]]]].
  { (* No environment to join: the target is bottom, and the analysis
       has no definition. *)
    pose proof (join_with_external_ids join_choices join_depth fuel mh avoid
                  inputs target analysis Hjoin) as Hids.
    destruct (join_with_unfold join_choices join_depth fuel mh avoid inputs
                target analysis envs Hjoin Hcut)
      as [target_env [definitions' [names [Hjoin0 Htarget]]]].
    destruct (join_with_analysis join_choices join_depth fuel mh avoid inputs
                target analysis envs target_env definitions' names Hjoin Hcut
                Hjoin0)
      as [at_normal_mode [Hnormal [Hdefs Hat]]].
    subst target.
    unfold Join_env.cut_and_n_way_join0 in Hjoin0. rewrite Hempty in Hjoin0.
    injection Hjoin0 as <- <- _.
    cbn [Join_env.make_bottom meet_env_functions].
    destruct (make_bottom_keeps (Meet_env.create (source inputs)) Hfits)
      as [Hextends Hfits'].
    split; [ exact Hfits' | ].
    split; [ exact (extends_keeps_source _ _ Hextends) | ].
    split; [ intros x Hx Hnot; exfalso; apply Hnot; exact Hx | ].
    split.
    { intros index id Hlookup. rewrite Hids in Hlookup.
      exact (ids_of_uses_found inputs index id Hlookup). }
    split.
    - intros x simples kind index simple Hx. rewrite Hdefs in Hx.
      discriminate Hx.
    - intros x simples kind index simple Hx. rewrite Hat in Hx.
      destruct (definitions_at_normal_mode_lookup envs _ _ Hnormal x simples
                  kind Hx) as [simples0 [Hin _]].
      apply in_compact in Hin. discriminate Hin. }
  pose proof (model_join_run_adds_types_that_fit_otherwise A join_choices
                join_depth fuel avoid inputs envs s3 relations3 kept equations
                relation_equations definitions Hinputs Hkinded Hagree Hfits
                Hfit_uses Hrun) as Hrest.
  destruct (join_with_keeps join_choices join_depth fuel mh avoid inputs
              target analysis envs s3 relations3 kept equations
              relation_equations definitions Hmh Hinputs Hkinded Hagree Hfits
              Hjoin Hrun Hrest) as [Htarget [Hkeeps Hintroduces]].
  split; [ exact Htarget | ]. split; [ exact Hkeeps | ].
  split; [ exact Hintroduces | ].
  exact (join_with_describes_uses join_choices join_depth fuel mh avoid inputs
           target analysis envs s3 relations3 kept equations
           relation_equations definitions Hmh Hinputs Hkinded Hagree Hfits
           Hjoin Hrun Hrest).
Qed.

(* ======================================================================== *)
(* The definitions of the variables that the last phase defines             *)
(* ======================================================================== *)

(* Through the fold of the last phase, a variable that the join has
   made and that the fold has not come to, or has defined, keeps its
   definition. *)
Lemma recording_fold_definitions (tenv : Typing_env.t) (s : Join_env.t)
    (tenv_of_aliases : Typing_env.t) (unique : variable -> bool) :
  aliases_not_made s ->
  ∀ (l : list (variable * flambda_kind)) (acc acc' : define_acc recorded),
    NoDup (map fst l) ->
    List.fold_left (define_step (recording tenv) s tenv_of_aliases unique) l
      (Some acc)
    = Some acc' ->
    ∃ kept : list (variable * flambda_kind),
      recorded_variables (acc_env acc')
      = recorded_variables (acc_env acc) ++ kept
      ∧ (∀ binding : variable * flambda_kind, binding ∈ kept -> binding ∈ l)
      ∧ (∀ x : variable,
           is_existential s x ->
           x ∉ map fst l ∨ x ∈ map fst kept ->
           lookup x (snd acc') = lookup x (snd acc))
      ∧ ∀ binding : variable * flambda_kind,
          binding ∈ l ->
          Aliases_of_existentials.aliases_of_existential_var
            (Join_env.aliases_of_existentials s) (fst binding)
          = [] ->
          unique (fst binding) = false ->
          binding ∈ kept.
Proof.
  intros Hnot_made.
  induction l as [ | binding l IH]; intros acc acc' Hnodup Hfold.
  - injection Hfold as <-. exists [].
    split; [ rewrite app_nil_r; reflexivity | ].
    split; [ intros b [] | ].
    split; [ intros x _ _; reflexivity | intros b [] ].
  - cbn [List.fold_left] in Hfold.
    destruct (define_step (recording tenv) s tenv_of_aliases unique (Some acc)
                binding) as [acc1 | ] eqn:Hstep.
    2:{ rewrite (fold_left_none _ (fun b => eq_refl)) in Hfold.
        discriminate. }
    cbn [map] in Hnodup. apply NoDup_cons_iff in Hnodup.
    destruct Hnodup as [Hnotin Hnodup].
    destruct (IH acc1 acc' Hnodup Hfold)
      as [kept [Hvars [Hin [Hdefs Hall]]]].
    destruct acc as [[[[r to_expand] equations] relations] definitions].
    destruct binding as [var var_kind].
    cbn [define_step fst snd] in Hstep.
    assert (Hkept_ne : ∀ x : variable, x ∈ map fst kept -> x <> var).
    { intros x Hx ->. apply Hnotin. apply in_map_iff in Hx.
      destruct Hx as [b [Heq Hb]]. apply in_map_iff. exists b.
      split; [ exact Heq | exact (Hin b Hb) ]. }
    (* A variable of the conclusion, where the step defines none. *)
    assert (Hrest :
              ∀ x : variable,
                x ∉ map fst ((var, var_kind) :: l) ∨ x ∈ map fst kept ->
                (x ∉ map fst l ∨ x ∈ map fst kept) ∧ x <> var).
    { intros x [Hx | Hx].
      - split; [ left; intros H; apply Hx; right; exact H | ].
        intros ->. apply Hx. left. reflexivity.
      - split; [ right; exact Hx | exact (Hkept_ne x Hx) ]. }
    destruct (Join_env.choose_alias tenv_of_aliases _) as [[canon_var | ] | ]
      eqn:Hchoose; [ | | discriminate ].
    + (* an alias in the target stands for the variable *)
      destruct (Join_env.move_equation equations var canon_var)
        as [equations1 | ]; [ | discriminate ].
      injection Hstep as <-.
      pose proof (Hnot_made var canon_var (choose_alias_in _ _ _ Hchoose))
        as Hcanon.
      exists kept. split; [ exact Hvars | ].
      split; [ intros b Hb; right; exact (Hin b Hb) | ].
      split.
      { intros x Hx Hcases. destruct (Hrest x Hcases) as [Hcases' Hne].
        rewrite (Hdefs x Hx Hcases'). cbn [snd].
        unfold Join_env.move_definition.
        destruct (lookup var definitions) as [defn | ]; [ | reflexivity ].
        rewrite lookup_cons.
        destruct (Pos.eqb x canon_var) eqn:Hxc.
        { apply Pos.eqb_eq in Hxc. subst x. exfalso. exact (Hcanon Hx). }
        rewrite lookup_unbind.
        destruct (Pos.eqb x var) eqn:Hxv; [ | reflexivity ].
        apply Pos.eqb_eq in Hxv. exfalso. exact (Hne Hxv). }
      intros b [<- | Hb] Hnone Hunique; [ | exact (Hall b Hb Hnone Hunique) ].
      cbn [fst] in Hnone. rewrite Hnone in Hchoose. discriminate Hchoose.
    + destruct (unique var) eqn:Hunique_var.
      * (* the variable is replaced by its type *)
        injection Hstep as <-.
        exists kept. split; [ exact Hvars | ].
        split; [ intros b Hb; right; exact (Hin b Hb) | ].
        split.
        { intros x Hx Hcases. destruct (Hrest x Hcases) as [Hcases' Hne].
          rewrite (Hdefs x Hx Hcases'). cbn [snd].
          rewrite lookup_unbind.
          destruct (Pos.eqb x var) eqn:Hxv; [ | reflexivity ].
          apply Pos.eqb_eq in Hxv. exfalso. exact (Hne Hxv). }
        intros b [<- | Hb] Hnone Hunique;
          [ | exact (Hall b Hb Hnone Hunique) ].
        cbn [fst] in Hunique. rewrite Hunique in Hunique_var.
        discriminate Hunique_var.
      * (* the variable is defined *)
        injection Hstep as <-.
        exists ((var, var_kind) :: kept).
        unfold acc_env in Hvars |- *. cbn [fst] in Hvars |- *.
        cbn [Join_env.add_variable_definition recording recorded_variables]
          in Hvars.
        split; [ rewrite Hvars, <- app_assoc; reflexivity | ].
        split.
        { intros b [<- | Hb];
            [ left; reflexivity | right; exact (Hin b Hb) ]. }
        split.
        { intros x Hx Hcases. cbn [snd] in Hdefs |- *. apply (Hdefs x Hx).
          destruct Hcases as [Hx_out | Hx_kept].
          - left. intros H. apply Hx_out. right. exact H.
          - cbn [map fst] in Hx_kept. destruct Hx_kept as [<- | Hx_kept];
              [ left; exact Hnotin | right; exact Hx_kept ]. }
        intros b [<- | Hb] Hnone Hunique;
          [ left; reflexivity | right; exact (Hall b Hb Hnone Hunique) ].
Qed.

(* A variable that the last phase defines has, in the definitions that
   the phase answers, the definition that the state has for it. *)
Lemma kept_definitions (tenv : Typing_env.t) (fuel : nat) (s : Join_env.t)
    (relations : Join_env.inverse_relations)
    (kept : list (variable * flambda_kind))
    (equations relation_equations : list (name * flambda_type))
    (definitions : finite_map (Simples_in_joined_envs.t * flambda_kind))
    (x : variable) (kind : flambda_kind) :
  aliases_not_made s ->
  last_phase_adds tenv fuel s relations kept equations relation_equations
    definitions ->
  (x, kind) ∈ kept ->
  lookup x definitions = lookup x (Join_env.definitions_of_existentials s).
Proof.
  intros Hnot_made Hadds Hin.
  destruct (last_phase_adds_kept tenv fuel s relations kept equations
              relation_equations definitions Hadds) as [_ Hkept].
  destruct (last_phase_adds_parts tenv fuel s relations _ definitions Hadds)
    as [r1 [to_expand [equations1 [relations1 [kept0 [equations0
          [relation_equations0
             [Hfold [Hvars [_ [_ [_ [_ [_ Heq]]]]]]]]]]]]]].
  injection Heq as Hkept0 _ _. apply app_inv_head in Hkept0. subst kept0.
  destruct (recording_fold_definitions tenv s tenv (unique_of s relations)
              Hnot_made _ _ _ (in_order_distinct _) Hfold)
    as [kept' [Hvars' [_ [Hdefs _]]]].
  unfold acc_env in Hvars'. cbn [fst recorded_variables] in Hvars'.
  rewrite Hvars in Hvars'. apply app_inv_head in Hvars'. subst kept'.
  assert (Hx : is_existential s x).
  { apply is_bound_lookup. exists kind.
    exact (in_order_lookup _ _ _ (Hkept _ Hin)). }
  apply (Hdefs x Hx). right. apply in_map_iff. exists (x, kind).
  split; [ reflexivity | exact Hin ].
Qed.

(* A variable that the join has made, that no variable of the target
   is to be an alias of, and that is part of a relation, is one that
   the last phase defines. *)
Lemma related_kept (tenv : Typing_env.t) (fuel : nat) (s : Join_env.t)
    (relations : Join_env.inverse_relations)
    (kept : list (variable * flambda_kind))
    (equations relation_equations : list (name * flambda_type))
    (definitions : finite_map (Simples_in_joined_envs.t * flambda_kind))
    (x : variable) :
  aliases_not_made s ->
  last_phase_adds tenv fuel s relations kept equations relation_equations
    definitions ->
  is_existential s x ->
  Aliases_of_existentials.aliases_of_existential_var
    (Join_env.aliases_of_existentials s) x
  = [] ->
  Join_env.in_inverse_relations relations x = true ->
  x ∈ map fst kept.
Proof.
  intros Hnot_made Hadds Hx Hnone Hrelated.
  destruct (last_phase_adds_parts tenv fuel s relations _ definitions Hadds)
    as [r1 [to_expand [equations1 [relations1 [kept0 [equations0
          [relation_equations0
             [Hfold [Hvars [_ [_ [_ [_ [_ Heq]]]]]]]]]]]]]].
  injection Heq as Hkept0 _ _. apply app_inv_head in Hkept0. subst kept0.
  destruct (recording_fold_definitions tenv s tenv (unique_of s relations)
              Hnot_made _ _ _ (in_order_distinct _) Hfold)
    as [kept' [Hvars' [_ [_ Hall]]]].
  unfold acc_env in Hvars'. cbn [fst recorded_variables] in Hvars'.
  rewrite Hvars in Hvars'. apply app_inv_head in Hvars'. subst kept'.
  apply is_bound_lookup in Hx. destruct Hx as [kind Hkind].
  apply in_map_iff. exists (x, kind). split; [ reflexivity | ].
  apply Hall.
  - apply (proj1 (in_rev _ _)). apply (proj2 (in_compact _ _ _)).
    exact Hkind.
  - exact Hnone.
  - cbn [fst]. unfold unique_of. rewrite Hrelated. reflexivity.
Qed.

(* A variable of the target so far is one that the source knows, one
   that the join has made, or one brought in. *)
Lemma in_target_cases (s : Join_env.t) (x : variable) :
  exists_in_target_env kind_in_target_env s x = true ->
  knows (Join_env.source_env s) x ∨ is_existential s x ∨ is_imported s x.
Proof.
  unfold exists_in_target_env, kind_in_target_env, knows, Typing_env.mem,
    is_existential, is_imported, Join_env.source_env.
  destruct (Keyed.lookup name_eqb (Name_var x)
              (Typing_env.names_to_types
                 (Bindings_in_target_env.source_env (Join_env.bindings s))))
    as [[ty btm] | ].
  - intros _. left. reflexivity.
  - destruct (lookup x
                (Bindings_in_target_env.existential_variables
                   (Join_env.bindings s))) as [kind | ] eqn:Hmade.
    + intros _. right. left. apply is_bound_lookup. exists kind. exact Hmade.
    + destruct (lookup x
                  (Bindings_in_target_env.imported_variables
                     (Join_env.bindings s))) as [kind | ] eqn:Himported;
        [ | discriminate ].
      intros _. right. right. apply is_bound_lookup. exists kind.
      exact Himported.
Qed.

(* ======================================================================== *)
(* The test on the answer, and the run                                      *)
(* ======================================================================== *)

Theorem kept_defined_at_uses_b_sound : kept_defined_at_uses_b_sound_statement.
Proof.
  intros A inputs analysis index use Hb Hat Hbottom x simples kind Hx Hnot.
  unfold kept_defined_at_uses_b in Hb. rewrite forallb_forall in Hb.
  specialize (Hb (x, (simples, kind)) (lookup_in _ _ _ Hx)).
  cbn [fst snd] in Hb. apply orb_true_iff in Hb. destruct Hb as [Hb | Hb].
  { exfalso. exact (Hnot Hb). }
  rewrite forallb_forall in Hb.
  apply (numbered_uses_in inputs index use) in Hat.
  unfold numbered_uses in Hat. apply in_map_iff in Hat.
  destruct Hat as [[index' [id use']] [Heq Hin]]. cbn [fst snd] in Heq.
  injection Heq as -> ->.
  specialize (Hb _ Hin). cbn [fst snd] in Hb. rewrite Hbottom in Hb.
  cbn [orb] in Hb.
  destruct (Simples_in_joined_envs.find index simples) as [simple | ];
    [ exists simple; reflexivity | discriminate Hb ].
Qed.

Theorem known_in_terms_have_values : known_in_terms_have_values_statement.
Proof.
  intros tenv valuation Hterms Hbound x Hx. exact (Hbound x (Hterms x Hx)).
Qed.

(* ======================================================================== *)
(* The sixth part                                                           *)
(* ======================================================================== *)

Theorem model_join_with_admits_use_valued
  : model_join_with_admits_use_valued_statement.
Proof.
  intros A join_choices join_depth fuel mh avoid inputs target analysis index
    use statics valuation Hmh Hsound Hinputs Hkinded Hagree Hfits Hfit_uses
    Hknow Hjoin Hat Hbottom Hdefined Hsat_use Hsat_source
    [Hstatics [Hvalues Htags]].
  assert (Hsource_values :
            ∀ x : variable,
              knows (source inputs) x ->
              ∃ v : value, lookup x valuation = Some v).
  { intros x Hx. apply Hvalues. apply (Hknow index use x Hat). exact Hx. }
  destruct (model_join_with_guarantees_on_inputs A join_choices join_depth
              fuel mh avoid inputs target analysis Hmh Hinputs Hkinded Hagree
              Hfits Hfit_uses Hjoin) as [_ [_ [_ [_ [Hdescribes _]]]]].
  destruct (join_with_run_found join_choices join_depth fuel mh avoid inputs
              target analysis Hjoin)
    as [envs [Hcut [Hempty | [s3 [relations3 [kept [equations
          [relation_equations [definitions Hrun]]]]]]]]].
  { (* The use is not bottom, so there is an environment to join. *)
    exfalso.
    destruct (cut_for_joins_has (cut_after inputs) (numbered_uses inputs) envs
                Hcut (numbered_uses_distinct inputs) index use
                (proj2 (numbered_uses_in inputs index use) Hat) Hbottom)
      as [level [_ Hlookup]].
    destruct envs; [ discriminate Hlookup | discriminate Hempty ]. }
  pose proof Hrun
    as [s1 [s2 [relations2 [Hcut' [Haliases [Hdelayed [Hloop Hadds]]]]]]].
  change (Join_env.cut_for_joins (cut_after inputs) (numbered_uses inputs)
          = Some envs) in Hcut'.
  destruct (inputs_first_phases join_choices join_depth fuel avoid inputs envs
              s1 s2 s3 relations2 relations3 Hinputs Hkinded Hagree Hcut'
              Haliases Hdelayed Hloop)
    as (Htrue3 & Hrelations3 & Hgrows3 & Hstate3 & Hnonempty3 & Hscoped3 & HX
        & HR & Hnames_kept & Hmade & Huse3 & Hinvariant3 & Hkinded3
        & Hsymbols3).
  assert (Huse : Join_env.joined_env s3 index = Some use).
  { apply Huse3. split; [ exact Hat | exact Hbottom ]. }
  assert (Hempty : Keyed.is_empty envs = false).
  { pose proof Huse as Huse0. rewrite (proj1 Hgrows3) in Huse0.
    destruct envs; [ discriminate Huse0 | reflexivity ]. }
  pose proof (proj2 (proj2 Htrue3)) as Hsource3.
  destruct (join_with_replayed join_choices join_depth fuel mh avoid inputs
              target analysis envs s3 relations3 kept equations
              relation_equations definitions Hjoin Hrun Hempty)
    as [_ [_ [_ [_ [_ [at_normal_mode [_ [Hdefinitions _]]]]]]]].
  destruct (target_valuation_extends s3 index use valuation (proj1 Htrue3)
              Huse) as [[use' [Huse' [_ Hdefs_tv]]] Hagree_source].
  pose proof (scoped_aliases_not_made s3 (proj1 Htrue3) (proj2 Hscoped3))
    as Hnot_made.
  set (tv := target_valuation s3 index use valuation) in *.
  (* What the valuation of the target gives: to a variable of the
     source, to one brought in (there is none), and to one that the
     last phase defines. *)
  assert (Hsource_tv :
            ∀ x : variable,
              knows (Join_env.source_env s3) x ->
              ∃ v : value, lookup x tv = Some v).
  { intros x Hx. rewrite (Hagree_source x Hx). rewrite Hsource3 in Hx.
    exact (Hsource_values x Hx). }
  assert (Hnot_imported : ∀ x : variable, ~ is_imported s3 x).
  { intros x Himp. destruct (proj1 Htrue3) as [_ [_ [_ [_ Himports]]]].
    destruct (Himports x Himp) as [Hnot [index0 [use0 [Huse0 Hknows0]]]].
    apply Hnot. rewrite Hsource3.
    destruct (proj1 (Huse3 index0 use0) Huse0) as [Hat0 _].
    exact (proj1 (Hknow index0 use0 x Hat0) Hknows0). }
  assert (Hkept_tv :
            ∀ x : variable,
              x ∈ map fst kept -> ∃ v : value, lookup x tv = Some v).
  { intros x Hkept. apply in_map_iff in Hkept.
    destruct Hkept as [[y k] [Heq Hin_kept]]. cbn [fst] in Heq. subst y.
    pose proof (kept_definitions _ (fuel_of_env fuel) s3 relations3 kept
                  equations relation_equations definitions x k Hnot_made Hadds
                  Hin_kept) as Hdef.
    destruct (last_phase_adds_kept _ (fuel_of_env fuel) s3 relations3 kept
                equations relation_equations definitions Hadds)
      as [_ Hkept_in].
    assert (Hex : is_existential s3 x).
    { apply is_bound_lookup. exists k.
      exact (in_order_lookup _ _ _ (Hkept_in _ Hin_kept)). }
    destruct (proj1 Htrue3) as [_ [Hnew [[_ [_ Hrecorded]] _]]].
    pose proof (Hrecorded x Hex) as Hbound_def.
    apply is_bound_lookup in Hbound_def.
    destruct Hbound_def as [[simples kind] Hlookup_def].
    destruct (Hdefined x simples kind) as [simple Hfind].
    { rewrite Hdefinitions, Hdef. exact Hlookup_def. }
    { rewrite <- Hsource3. exact (proj1 (Hnew x Hex)). }
    pose proof (proj2 (proj2 (proj2 (proj1 (proj2 Htrue3))))) as Hin_envs.
    destruct (Hin_envs x simples kind Hlookup_def index simple Hfind)
      as [use0 [Huse0 Hscope]].
    rewrite Huse in Huse0. injection Huse0 as <-.
    assert (Hvalue : ∃ v : value, eval_simple valuation simple = Some v).
    { destruct simple as [y | sym | c]; cbn [eval_simple].
      - apply Hvalues. apply Hscope. left. reflexivity.
      - eexists. reflexivity.
      - eexists. reflexivity. }
    destruct Hvalue as [v Hv]. exists v.
    exact (Hdefs_tv x simples kind simple v Hlookup_def Hfind Hv). }
  destruct (model_join_with_admits_use_outright A join_choices join_depth fuel
              mh avoid inputs target analysis envs s3 relations3 kept
              equations relation_equations definitions index use statics
              valuation Hmh Hsound Hinputs Hkinded Hagree Hfits Hfit_uses
              Hjoin Hrun Hat Hbottom Hstatics Hsat_use Hsat_source)
    as [valuation' [Hsat [Hfrom_use [Hfrom_source [Hdefs Hdefs_normal]]]]].
  - (* the blocks have tags *)
    intros simple v tag fields Hknown Heval Hblock.
    apply (Htags simple v tag fields); [ | exact Heval | exact Hblock ].
    intros x Hx. specialize (Hknown x Hx). unfold known_to in Hknown.
    apply orb_true_iff in Hknown. destruct Hknown as [H | H]; [ exact H | ].
    rewrite Hsource3 in H. exact (proj2 (Hknow index use x Hat) H).
  - (* the aliases have values *)
    apply (aliases_have_values_of_values (source inputs) s3 index use
             valuation Htrue3 Huse).
    + intros x _ Hx. exact (Hvalues x Hx).
    + intros x Hx _. rewrite Hsource3 in Hx. exact (Hsource_values x Hx).
  - (* the variables that the relations are on have values *)
    intros r relation x v Hrecorded _.
    destruct (Hrelations3 r relation (Name_var x) Hrecorded)
      as [x' [Heq [Hx_target _]]].
    injection Heq as <-.
    destruct (in_target_cases s3 x Hx_target) as [Hsrc | [Hex | Himp]];
      [ exact (Hsource_tv x Hsrc) | | exfalso; exact (Hnot_imported x Himp) ].
    destruct (Aliases_of_existentials.aliases_of_existential_var
                (Join_env.aliases_of_existentials s3) x)
      as [ | alias aliases] eqn:Haliases_of.
    + (* no alias: the last phase defines the variable *)
      apply Hkept_tv.
      apply (related_kept _ (fuel_of_env fuel) s3 relations3 kept equations
               relation_equations definitions x Hnot_made Hadds Hex
               Haliases_of).
      unfold Join_env.in_inverse_relations. apply orb_true_iff. right.
      destruct Hrecorded as [recorded [Hlookup Hin]].
      apply existsb_exists. exists (r, recorded).
      split; [ apply (proj2 (in_compact _ _ _)); exact Hlookup | ].
      cbn [snd]. apply existsb_exists. exists (relation, Name_var x).
      split; [ exact Hin | ]. cbn [snd name_eqb]. apply Pos.eqb_refl.
    + (* an alias, which is a variable of the source *)
      assert (Hin_alias :
                alias
                ∈ Aliases_of_existentials.aliases_of_existential_var
                    (Join_env.aliases_of_existentials s3) x).
      { rewrite Haliases_of. left. reflexivity. }
      destruct (proj1 (proj2 (proj2 (proj1 (proj2 Htrue3)))) x alias
                  Hin_alias) as [Halias_target [_ [Hsame _]]].
      destruct (in_target_cases s3 alias Halias_target)
        as [Hsrc | [Hex' | Himp]];
        [ | exfalso; exact (Hnot_made x alias Hin_alias Hex')
          | exfalso; exact (Hnot_imported alias Himp) ].
      destruct (Hsource_tv alias Hsrc) as [w Hw]. exists w.
      exact (Hsame index use statics valuation w Huse Hsat_use Hw).
  - (* the names of the equations have values *)
    intros n ty Hin.
    destruct (last_phase_adds_on_variables _ (fuel_of_env fuel) s3 relations3
                kept equations relation_equations definitions Hadds n ty Hin)
      as [x ->].
    destruct (last_phase_adds_in_scope _ (fuel_of_env fuel) (source inputs) s3
                relations3 kept equations relation_equations definitions
                Htrue3 Hrelations3 Hnonempty3 (proj1 Hscoped3) Hnot_made
                Hadds) as [Hscope1 Hscope2].
    assert (Hfinal : in_final_target s3 kept x).
    { apply in_app_or in Hin. destruct Hin as [Hin | Hin].
      - apply (proj1 (Hscope1 _ _ Hin)). left. reflexivity.
      - apply (proj1 (Hscope2 _ _ Hin)). left. reflexivity. }
    change (∃ v : value, lookup x tv = Some v).
    destruct Hfinal as [Hsrc | [Himp | Hkept]];
      [ exact (Hsource_tv x Hsrc) | exfalso; exact (Hnot_imported x Himp)
        | exact (Hkept_tv x Hkept) ].
  - (* the witness *)
    exists valuation'. split; [ exact Hsat | ].
    split; [ exact Hfrom_use | ]. split; [ exact Hfrom_source | ].
    split; [ exact Hdefs | ]. split; [ exact Hdefs_normal | ].
    intros x Hx.
    destruct (Typing_env.mem None (source inputs) (Name_var x)) eqn:Hmem.
    + destruct (Hsource_values x Hmem) as [v Hv]. exists v.
      rewrite (Hfrom_source x Hmem). exact Hv.
    + destruct Hx as [Hx | Hx].
      { unfold knows in Hx. rewrite Hmem in Hx. discriminate Hx. }
      apply is_bound_lookup in Hx. destruct Hx as [[simples kind] Hlookup].
      destruct (Hdefined x simples kind Hlookup) as [simple Hfind].
      { unfold knows. rewrite Hmem. discriminate. }
      destruct (Hdescribes x simples kind index simple Hlookup Hfind)
        as [_ [id0 [use0 [Hnth0 [_ Hkind0]]]]].
      destruct Hat as [id Hnth]. rewrite Hnth in Hnth0.
      injection Hnth0 as _ <-.
      assert (Hvalue : ∃ v : value, eval_simple valuation simple = Some v).
      { destruct simple as [y | sym | c]; cbn [eval_simple].
        - apply Hvalues. exact (simple_of_kind_var_known use y kind Hkind0).
        - eexists. reflexivity.
        - eexists. reflexivity. }
      destruct Hvalue as [v Hv]. exists v.
      exact (Hdefs x simples kind simple v Hlookup Hfind Hv).
Qed.

Theorem model_join_with_admits_use_on
  : model_join_with_admits_use_on_statement.
Proof.
  intros A join_choices join_depth fuel mh avoid inputs target analysis index
    use Hmh Hsound Hinputs Hkinded Hagree Hfits Hfit_uses Hknow Hjoin Hat
    Hbottom Hdefined statics valuation Hsat_use Hsat_source Hrun.
  destruct (model_join_with_admits_use_valued A join_choices join_depth fuel
              mh avoid inputs target analysis index use statics valuation Hmh
              Hsound Hinputs Hkinded Hagree Hfits Hfit_uses Hknow Hjoin Hat
              Hbottom Hdefined Hsat_use Hsat_source Hrun)
    as [valuation' [H1 [H2 [H3 [H4 [H5 _]]]]]].
  exists valuation'. split; [ exact H1 | ]. split; [ exact H2 | ].
  split; [ exact H3 | ]. split; [ exact H4 | exact H5 ].
Qed.

Theorem model_join_with_in_machine : model_join_with_in_machine_statement.
Proof.
  intros A join_choices join_depth fuel mh avoid inputs target analysis index
    use statics vars valuation Hmh Hsound Hinputs Hkinded Hagree Hfits
    Hfit_uses Hknow Hterms Hjoin Hat Hbottom Hdefined Hbound Hsat_use
    Hsat_source Hagrees Hrun.
  destruct (model_join_with_guarantees_on_inputs A join_choices join_depth
              fuel mh avoid inputs target analysis Hmh Hinputs Hkinded Hagree
              Hfits Hfit_uses Hjoin)
    as [_ [Hkeeps [Hintroduces _]]].
  assert (Hscope :
            ∀ x : variable,
              in_scope_in_terms target x -> in_scope_in_terms use x).
  { intros x Hscope. apply (Hterms index use x Hat).
    destruct (Typing_env.mem None (source inputs) (Name_var x)) eqn:Hknown.
    - exact (in_scope_in_terms_kept _ _ x Hkeeps Hknown Hscope).
    - destruct (Hintroduces x (in_scope_in_terms_knows _ _ Hscope))
        as [Hnot_in_scope _].
      + unfold knows. rewrite Hknown. discriminate.
      + contradiction. }
  destruct (model_join_with_admits_use_valued A join_choices join_depth fuel
              mh avoid inputs target analysis index use statics valuation Hmh
              Hsound Hinputs Hkinded Hagree Hfits Hfit_uses Hknow Hjoin Hat
              Hbottom Hdefined Hsat_use Hsat_source Hrun)
    as [valuation' [Hsat [Hfrom_use [_ [_ [_ Hvalues]]]]]].
  split.
  { intros x Hx. apply Hbound. apply Hscope. exact Hx. }
  exists valuation'. split; [ exact Hsat | ]. split.
  { intros x Hx. specialize (Hscope x Hx).
    rewrite (Hfrom_use x (in_scope_in_terms_knows _ _ Hscope)).
    exact (Hagrees x Hscope). }
  split; [ exact Hfrom_use | exact Hvalues ].
Qed.

(* ======================================================================== *)
(* The instances that refute the contracts                                  *)
(* ======================================================================== *)

Theorem counterexamples_excluded : counterexamples_excluded_statement.
Proof.
  split; [ | split; [ | split ] ].
  - intros Hvalues.
    assert (Hx : knows Witness.use_env Witness.x) by (vm_compute; reflexivity).
    destruct (Hvalues Witness.x Hx) as [v Hv]. discriminate Hv.
  - intros Hvalues.
    assert (Hx : knows In_terms.use_env Witness.x)
      by (vm_compute; reflexivity).
    destruct (Hvalues Witness.x Hx) as [v Hv]. discriminate Hv.
  - vm_compute. reflexivity.
  - intros Hx. vm_compute in Hx. discriminate Hx.
Qed.

(* ======================================================================== *)
(* What the join computes on the arms of a [Switch]                         *)
(* ======================================================================== *)

Theorem arms_without_equations : arms_without_equations_statement.
Proof. intros choices [<- | [<- | []]]; vm_compute; reflexivity. Qed.

Theorem arms_block_and_immediate : arms_block_and_immediate_statement.
Proof. intros choices [<- | [<- | []]]; vm_compute; reflexivity. Qed.

Theorem arms_shared_field : arms_shared_field_statement.
Proof.
  intros choices [<- | [<- | []]];
    (eexists; split; [ vm_compute; reflexivity | ];
     split; [ vm_compute; reflexivity | ];
     split; [ reflexivity | ];
     split; [ vm_compute; reflexivity | ];
     split; [ vm_compute; reflexivity | ];
     split; vm_compute; reflexivity).
Qed.
