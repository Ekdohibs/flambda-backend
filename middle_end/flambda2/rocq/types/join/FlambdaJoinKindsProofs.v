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


(* The proofs of the statements of FlambdaJoinKindsSpec.v. This file
   needs no review.

   As for the variables (FlambdaJoinFitsProofs.v): a relation between
   states that every function of the join of environments keeps
   ([keeps_kinds]), then the rounds, which follow those of
   FlambdaJoinStateProofs.v. The relation has the kinds of the later
   state as a premise; the kinds of each state in between follow
   ([kinded_earlier]). *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaMapProofs FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeMeaningSpec FlambdaTypeMeaningProofs
  FlambdaTypeAliasKindsSpec FlambdaTypeAliasKindsProofs FlambdaTypeProject
  FlambdaBindingTime
  FlambdaTypingEnv FlambdaTypingEnvSpec FlambdaTypingEnvProofs
  FlambdaTypingEnvInvariantSpec FlambdaTypingEnvInvariantProofs
  FlambdaTypingEnvKindedSpec FlambdaTypingEnvKindedProofs FlambdaJoinEnv
  FlambdaJoinSpec FlambdaJoinEnvSpec FlambdaJoinEnvProofs
  FlambdaJoinStateSpec FlambdaJoinStateProofs FlambdaJoinDefineSpec
  FlambdaJoinDefineProofs FlambdaJoinFitsSpec FlambdaJoinFitsProofs
  FlambdaJoinAddsSpec FlambdaJoinAddsProofs FlambdaJoinTypes
  FlambdaJoinTypesKeepsSpec FlambdaJoinTypesKeepsProofs
  FlambdaJoinKindsSpec.
Import ListNotations.

(* The type that a constant is expanded to is of the kind of the
   constant: it follows from what is asked of the parameters
   ([expand_const_sound]), since the constant has the type. *)
Definition expand_const_of_kind (p : Join_env.parameters) : Prop :=
  ∀ c : const,
    kind (Join_env.expand_const p c) = kind (type_for_const c).

Lemma expand_const_sound_of_kind (p : Join_env.parameters) :
  expand_const_sound p -> expand_const_of_kind p.
Proof.
  intros Hconst c.
  destruct (Hconst c) as [_ [_ Hsatisfied]].
  rewrite <- (type_satisfied_kind empty_static_env [] (Value_const c) _
                (Hsatisfied empty_static_env [])).
  exact (type_satisfied_kind empty_static_env [] (Value_const c)
           (type_for_const c)
           (type_for_const_satisfied empty_static_env [] c)).
Qed.

Lemma reach_grows (s s' : Join_env.t) (simple : simple)
    (kind : flambda_kind) :
  grows s s' ->
  simple_of_kind_in_reach s simple kind ->
  simple_of_kind_in_reach s' simple kind.
Proof.
  intros [Henvs [_ Hkinds]] [H | [index [use [Huse H]]]].
  - left. destruct simple as [x | sym | c];
      [ exact (Hkinds x kind H) | | exact H ].
    destruct H as [Hkind [index [use [Huse Hmem]]]].
    split; [ exact Hkind | ]. exists index, use. rewrite Henvs.
    split; assumption.
  - right. exists index, use. rewrite Henvs. split; assumption.
Qed.

Lemma kinds_grow (s s' : Join_env.t) :
  grows s s' ->
  Join_env.types_in_target_env s' = Join_env.types_in_target_env s ->
  Join_env.types_in_joined_envs s' = Join_env.types_in_joined_envs s ->
  Join_env.aliases_of_existentials s' = Join_env.aliases_of_existentials s ->
  equations_kinded_in_reach s
  ∧ equations_of_kinds s ∧ types_to_join_of_kinds s ∧ aliases_of_kinds s ->
  equations_kinded_in_reach s'
  ∧ equations_of_kinds s' ∧ types_to_join_of_kinds s' ∧ aliases_of_kinds s'.
Proof.
  intros Hgrows Htypes Hjoin Haliases [H1 [H4 [H4' H5]]].
  pose proof (proj2 (proj2 Hgrows)) as Hkinds.
  split; [ | split; [ | split ] ].
  - intros x ty Hx simple kind Hin. rewrite Htypes in Hx.
    exact (reach_grows _ _ _ _ Hgrows (H1 x ty Hx simple kind Hin)).
  - intros x ty Hx. rewrite Htypes in Hx. exact (Hkinds _ _ (H4 x ty Hx)).
  - intros x types Hx. rewrite Hjoin in Hx.
    destruct (H4' x types Hx) as [Hsome Hall].
    split; [ exact Hsome | ].
    intros index t Ht. exact (Hkinds _ _ (Hall index t Ht)).
  - intros e x Hx. rewrite Haliases in Hx.
    destruct (H5 e x Hx) as [k [Hk Hk']]. exists k.
    split; [ exact (Hkinds _ _ Hk) | exact (Hkinds _ _ Hk') ].
Qed.

Lemma same_readings_sym (s s' : Join_env.t) :
  same_readings s s' -> same_readings s' s.
Proof.
  intros [H1 [H2 [H3 H4]]]. repeat split; symmetry; assumption.
Qed.

(* The same maps, in a state that has grown: all is kept. *)
Lemma kinds_same_maps (s s' : Join_env.t) :
  grows s s' ->
  Join_env.types_in_target_env s' = Join_env.types_in_target_env s ->
  Join_env.types_in_joined_envs s' = Join_env.types_in_joined_envs s ->
  Join_env.aliases_of_existentials s' = Join_env.aliases_of_existentials s ->
  Join_env.definitions_of_existentials s'
  = Join_env.definitions_of_existentials s ->
  equations_kinded s ->
  equations_kinded s'.
Proof.
  intros Hgrows Htypes Hjoin Haliases Hdefinitions [H1 [H4 [H4' [H5 HN]]]].
  destruct (kinds_grow s s' Hgrows Htypes Hjoin Haliases
              (conj H1 (conj H4 (conj H4' H5)))) as [H1' [H4a [H4b H5']]].
  split; [ exact H1' | ]. split; [ exact H4a | ]. split; [ exact H4b | ].
  split; [ exact H5' | ].
  intros x simples k Hx. rewrite Hdefinitions in Hx. exact (HN x simples k Hx).
Qed.

Lemma existential_keeps_kinds (p : Join_env.parameters) (s s' : Join_env.t)
    (canonicals : Simples_in_joined_envs.t) (kind : flambda_kind)
    (e : variable) :
  fresh_avoids p ->
  state_valid s ->
  (∃ (index : Index.t) (simple : simple),
     Simples_in_joined_envs.find index canonicals = Some simple) ->
  Join_env.existential_for_these_canonicals p s canonicals kind = (e, s') ->
  equations_kinded s ->
  equations_kinded s'.
Proof.
  intros Hfresh Hvalid Hsome Hmake [H1 [H4 [H4' [H5 HN]]]].
  destruct (existential_keeps _ _ _ _ _ _ Hfresh Hvalid Hmake)
    as [_ [Hgrows _]].
  destruct (existential_scope_grows _ _ _ _ _ _ Hmake)
    as [_ [Htypes Haliases]].
  assert (Hjoin :
            Join_env.types_in_joined_envs s' = Join_env.types_in_joined_envs s).
  { unfold Join_env.existential_for_these_canonicals in Hmake.
    destruct (Bindings_in_target_env.existential_for_these_simples
                (Join_env.fresh p) (Join_env.bindings s) canonicals kind)
      as [x0 b0].
    injection Hmake as _ <-. reflexivity. }
  destruct (kinds_grow s s' Hgrows Htypes Hjoin Haliases
              (conj H1 (conj H4 (conj H4' H5)))) as [H1' [H4a [H4b H5']]].
  split; [ exact H1' | ]. split; [ exact H4a | ]. split; [ exact H4b | ].
  split; [ exact H5' | ].
  intros x simples k Hx.
  unfold Join_env.existential_for_these_canonicals in Hmake.
  destruct (Bindings_in_target_env.existential_for_these_simples
              (Join_env.fresh p) (Join_env.bindings s) canonicals kind)
    as [x0 b0].
  injection Hmake as _ <-. cbn [Join_env.definitions_of_existentials] in Hx.
  destruct (is_bound x0 (Join_env.definitions_of_existentials s));
    [ exact (HN x simples k Hx) | ].
  rewrite lookup_cons in Hx. destruct (Pos.eqb x x0);
    [ injection Hx as <- _; exact Hsome | exact (HN x simples k Hx) ].
Qed.

(* A vector has a [Simple] at an index that has an environment. *)
Definition vector_located (s : Join_env.t)
    (simples : Simples_in_joined_envs.t) : Prop :=
  ∃ (index : Index.t) (simple : simple) (use : Typing_env.t),
    Simples_in_joined_envs.find index simples = Some simple
    ∧ Join_env.joined_env s index = Some use.

Lemma existential_kind (p : Join_env.parameters) (s s' : Join_env.t)
    (canonicals : Simples_in_joined_envs.t) (kind : flambda_kind)
    (e : variable) :
  fresh_avoids p ->
  state_valid s ->
  state_kinded s' ->
  vector_of_kind s canonicals kind ->
  vector_located s canonicals ->
  Join_env.existential_for_these_canonicals p s canonicals kind = (e, s') ->
  kind_in_target_env s' e = Some kind.
Proof.
  intros Hfresh Hvalid Hkinded' Hvector [index [simple [use [Hfind Huse]]]]
    Hmake.
  destruct (existential_keeps _ _ _ _ _ _ Hfresh Hvalid Hmake)
    as [Hvalid1 [Hgrows1 [Hexistential
                            [simples' [kind' [Hdefinition Hequal]]]]]].
  assert (Huse1 : Join_env.joined_env s' index = Some use).
  { destruct Hgrows1 as [Henvs1 _]. rewrite Henvs1. exact Huse. }
  pose proof (Hvector _ _ _ Hfind Huse) as Hkind0.
  destruct Hkinded' as [_ [_ Hdefinitions1]].
  rewrite (simples_equal_find _ _ Hequal) in Hfind.
  pose proof (Hdefinitions1 _ _ _ _ _ _ Hdefinition Hfind Huse1) as Hkind'.
  rewrite (simple_of_kind_unique _ _ _ _ Hkind' Hkind0) in Hdefinition.
  destruct Hvalid1 as [_ [Hnew1 [[Hrecorded1 _] _]]].
  unfold kind_in_target_env.
  destruct (Hnew1 e Hexistential) as [Hunknown _].
  unfold knows, Typing_env.mem, Join_env.source_env in Hunknown.
  destruct (Keyed.lookup name_eqb (Name_var e)
              (Typing_env.names_to_types
                 (Bindings_in_target_env.source_env (Join_env.bindings s'))))
    as [[ty time] | ]; [ exfalso; apply Hunknown; reflexivity | ].
  rewrite (Hrecorded1 _ _ _ Hdefinition). reflexivity.
Qed.

Lemma add_equation_keeps_kinds (s s' : Join_env.t) (x : variable)
    (ty : flambda_type) :
  Join_env.add_equation_in_target_env s x ty = Some s' ->
  (∀ (simple : simple) (kind : flambda_kind),
     (simple, kind) ∈ aliases_with_kinds ty ->
     simple_of_kind_in_reach s simple kind) ->
  kind_in_target_env s x = Some (kind ty) ->
  equations_kinded s ->
  equations_kinded s'.
Proof.
  intros Hadd Hty Hkind [H1 [H4 [H4' [H5 HN]]]].
  unfold Join_env.add_equation_in_target_env in Hadd.
  destruct (is_bound x (Join_env.types_in_target_env s)); [ discriminate | ].
  injection Hadd as <-.
  split; [ | split; [ | split; [ exact H4' | ] ] ];
    [ | | split; [ exact H5 | exact HN ] ].
  - intros x0 ty0 Hx0. cbn [Join_env.types_in_target_env] in Hx0.
    rewrite lookup_cons in Hx0. destruct (Pos.eqb x0 x).
    + injection Hx0 as <-. exact Hty.
    + exact (H1 x0 ty0 Hx0).
  - intros x0 ty0 Hx0. cbn [Join_env.types_in_target_env] in Hx0.
    rewrite lookup_cons in Hx0. destruct (Pos.eqb x0 x) eqn:Heq.
    + injection Hx0 as <-. apply Pos.eqb_eq in Heq. subst x0. exact Hkind.
    + exact (H4 x0 ty0 Hx0).
Qed.

Lemma imported_kind (s : Join_env.t) (x : variable) (kind : flambda_kind) :
  state_valid s ->
  lookup x (Bindings_in_target_env.imported_variables (Join_env.bindings s))
  = Some kind ->
  kind_in_target_env s x = Some kind.
Proof.
  intros [_ [Hnew [_ [_ Himports]]]] Hx.
  assert (Himported : is_imported s x).
  { apply is_bound_lookup. exists kind. exact Hx. }
  destruct (Himports x Himported) as [Hsource [index [use [Huse Hknows]]]].
  unfold kind_in_target_env.
  destruct (Keyed.lookup name_eqb (Name_var x)
              (Typing_env.names_to_types
                 (Bindings_in_target_env.source_env (Join_env.bindings s))))
    as [[ty btm] | ] eqn:Hlookup.
  - exfalso. apply Hsource. apply mem_var_lookup. exists ty, btm.
    exact Hlookup.
  - destruct (lookup x
                (Bindings_in_target_env.existential_variables
                   (Join_env.bindings s))) as [k | ] eqn:Hexistential;
      [ | exact Hx ].
    exfalso.
    assert (Hex : is_existential s x).
    { apply is_bound_lookup. exists k. exact Hexistential. }
    exact (proj2 (Hnew x Hex) index use Huse Hknows).
Qed.

(* A later state of one join, with what [scope_grows] says. *)
Definition steps (s s' : Join_env.t) : Prop :=
  state_valid s ∧ state_valid s' ∧ grows s s' ∧ scope_grows s s'.

Lemma steps_refl (s : Join_env.t) : state_valid s -> steps s s.
Proof.
  intros Hvalid. split; [ exact Hvalid | ]. split; [ exact Hvalid | ].
  split; [ apply grows_refl | apply scope_grows_refl ].
Qed.

Lemma steps_trans (s1 s2 s3 : Join_env.t) :
  steps s1 s2 -> steps s2 s3 -> steps s1 s3.
Proof.
  intros [Hv1 [_ [Hg12 Hs12]]] [_ [Hv3 [Hg23 Hs23]]].
  split; [ exact Hv1 | ]. split; [ exact Hv3 | ].
  split; [ exact (grows_trans _ _ _ Hg12 Hg23)
         | exact (scope_grows_trans _ _ _ Hs12 Hs23) ].
Qed.

(* The kinds of an earlier state agree with the joined environments if
   those of a later state do. *)
Lemma kinded_earlier (s s' : Join_env.t) :
  steps s s' -> state_kinded s' -> state_kinded s.
Proof.
  intros [Hvalid [Hvalid' [[Henvs [Hdefs Hkinds]] [_ [Hsource [_ Himported]]]]]]
    [Hsources [Himports Hdefinitions]].
  split; [ | split ].
  - intros index use x kind ty btm Huse Hkind Hlookup.
    apply (Hsources index use x kind ty btm);
      [ rewrite Henvs; exact Huse | exact Hkind | ].
    rewrite Hsource. exact Hlookup.
  - intros x kind index use Hx Huse Hknows.
    apply (Himports x kind index use);
      [ | rewrite Henvs; exact Huse | exact Hknows ].
    assert (Hi : is_imported s x).
    { apply is_bound_lookup. exists kind. exact Hx. }
    pose proof (Himported x Hi) as Hi'. apply is_bound_lookup in Hi'.
    destruct Hi' as [k' Hk'].
    pose proof (Hkinds _ _ (imported_kind s x kind Hvalid Hx)) as Hkind'.
    rewrite (imported_kind s' x k' Hvalid' Hk') in Hkind'.
    injection Hkind' as ->. exact Hk'.
  - intros x simples kind index simple use Hx Hfind Huse.
    apply (Hdefinitions x simples kind index simple use (Hdefs _ _ Hx) Hfind).
    rewrite Henvs. exact Huse.
Qed.

Lemma aliases_alias_type_of (kind : flambda_kind) (simple : simple) :
  aliases_with_kinds (alias_type_of kind simple) = [(simple, kind)].
Proof. destruct kind as [ | k | | ]; [ | destruct k | | ]; reflexivity. Qed.

Lemma kind_of_alias_type (kind : flambda_kind) (simple : simple) :
  FlambdaTypeGrammar.kind (alias_type_of kind simple) = kind.
Proof. destruct kind as [ | k | | ]; [ | destruct k | | ]; reflexivity. Qed.

Lemma stands_kind (s : Join_env.t) (canonicals : Simples_in_joined_envs.t)
    (joined : simple) (kind : flambda_kind) :
  joined_envs_are typing_env_invariant s ->
  joined_envs_are typing_env_kinded s ->
  vector_of_kind s canonicals kind ->
  vector_located s canonicals ->
  stands_in_for s canonicals joined ->
  ∃ (index : Index.t) (use : Typing_env.t),
    Join_env.joined_env s index = Some use ∧ simple_of_kind use joined kind.
Proof.
  intros Hinvariant Henvs Hvector [index [simple [use [Hfind Huse]]]] Hstands.
  exists index, use. split; [ exact Huse | ].
  pose proof (Hvector _ _ _ Hfind Huse) as Hkind.
  destruct (Hstands _ _ Hfind) as [Heq | [env [Henv Heq]]].
  - rewrite <- Heq. exact Hkind.
  - rewrite Huse in Henv. injection Henv as <-.
    apply (kind_of_canonical use joined kind (Hinvariant _ _ Huse)
             (Henvs _ _ Huse)).
    rewrite <- Heq. exact Hkind.
Qed.

Lemma add_alias_keeps_kinds (p : Join_env.parameters) (s : Join_env.t)
    (kind : flambda_kind) (var : variable)
    (canonicals : Simples_in_joined_envs.t) :
  fresh_avoids p ->
  state_valid s ->
  state_kinded (Join_env.add_alias_of_existential p s kind var canonicals) ->
  vector_of_kind s canonicals kind ->
  vector_located s canonicals ->
  kind_in_target_env s var = Some kind ->
  equations_kinded s ->
  equations_kinded (Join_env.add_alias_of_existential p s kind var canonicals).
Proof.
  intros Hfresh Hvalid Hkinded' Hvector Hlocated Hvar HX.
  unfold Join_env.add_alias_of_existential in Hkinded' |- *.
  destruct (Join_env.existential_for_these_canonicals p s canonicals kind)
    as [e s1] eqn:Hmake.
  assert (Hkinded1 : state_kinded s1).
  { refine (same_readings_kinded _ s1 _ Hkinded'). repeat split. }
  assert (Hsome :
            ∃ (index : Index.t) (simple : simple),
              Simples_in_joined_envs.find index canonicals = Some simple).
  { destruct Hlocated as [index [simple [use [Hfind _]]]].
    exists index, simple. exact Hfind. }
  destruct (existential_keeps_kinds _ _ _ _ _ _ Hfresh Hvalid Hsome Hmake HX)
    as [H1 [H4 [H4' [H5 HN]]]].
  pose proof (existential_kind _ _ _ _ _ _ Hfresh Hvalid Hkinded1 Hvector
                Hlocated Hmake) as He.
  destruct (existential_keeps _ _ _ _ _ _ Hfresh Hvalid Hmake)
    as [_ [Hgrows _]].
  split; [ exact H1 | ]. split; [ exact H4 | ]. split; [ exact H4' | ].
  split; [ | exact HN ].
  intros e' x Hx. cbn [Join_env.aliases_of_existentials] in Hx.
  apply aliases_add_cases in Hx. destruct Hx as [Hx | [-> ->]].
  - exact (H5 e' x Hx).
  - exists kind. split; [ exact (proj2 (proj2 Hgrows) _ _ Hvar) | exact He ].
Qed.

Definition import_keeps_kinds
    (import_var : Join_env.t -> variable -> option Join_env.t) : Prop :=
  ∀ (s s' : Join_env.t) (x : variable),
    state_valid s ->
    joined_envs_are typing_env_invariant s ->
    joined_envs_are typing_env_kinded s ->
    import_var s x = Some s' ->
    state_kinded s' ->
    equations_kinded s ->
    equations_kinded s'.

Lemma import_steps
    (import_var : Join_env.t -> variable -> option Join_env.t) :
  import_keeps import_var ->
  import_keeps_scope import_var ->
  ∀ (s s' : Join_env.t) (x : variable),
    state_valid s -> import_var s x = Some s' -> steps s s'.
Proof.
  intros Hk Hs s s' x Hvalid Hi.
  destruct (Hk _ _ _ Hvalid Hi) as [Hvalid' [Hgrows _]].
  split; [ exact Hvalid | ]. split; [ exact Hvalid' | ].
  split; [ exact Hgrows | exact (proj1 (proj1 (Hs _ _ _ Hi))) ].
Qed.

Lemma import_vars_steps
    (import_var : Join_env.t -> variable -> option Join_env.t)
    (vars : list variable) :
  import_keeps import_var ->
  import_keeps_scope import_var ->
  ∀ s s' : Join_env.t,
    state_valid s ->
    List.fold_left
      (fun env var =>
         let* env := env in
         import_var env var)
      vars (Some s)
    = Some s' ->
    steps s s'.
Proof.
  intros Hk Hs s s' Hvalid Hfold.
  destruct (import_vars_keeps _ vars Hk _ _ Hvalid Hfold)
    as [Hvalid' [Hgrows _]].
  split; [ exact Hvalid | ]. split; [ exact Hvalid' | ].
  split; [ exact Hgrows | ].
  exact (proj1 (proj1 (import_vars_keeps_scope _ vars Hs _ _ Hfold))).
Qed.

Lemma import_vars_keeps_kinds
    (import_var : Join_env.t -> variable -> option Join_env.t)
    (vars : list variable) :
  import_keeps import_var ->
  import_keeps_scope import_var ->
  import_keeps_kinds import_var ->
  ∀ s s' : Join_env.t,
    state_valid s ->
    joined_envs_are typing_env_invariant s ->
    joined_envs_are typing_env_kinded s ->
    List.fold_left
      (fun env var =>
         let* env := env in
         import_var env var)
      vars (Some s)
    = Some s' ->
    state_kinded s' ->
    equations_kinded s ->
    equations_kinded s'.
Proof.
  intros Hk Hs Hkinds.
  induction vars as [ | var vars IH];
    intros s s' Hvalid Hinvariant Henvs Hfold Hkinded' HX.
  - injection Hfold as <-. exact HX.
  - cbn [List.fold_left] in Hfold.
    destruct (import_var s var) as [s1 | ] eqn:Hvar.
    + destruct (Hk _ _ _ Hvalid Hvar) as [Hvalid1 [Hgrows1 _]].
      pose proof (import_vars_steps _ vars Hk Hs _ _ Hvalid1 Hfold) as Hsteps.
      pose proof (kinded_earlier _ _ Hsteps Hkinded') as Hkinded1.
      apply (IH s1 s' Hvalid1 (grows_envs_invariant _ _ Hgrows1 Hinvariant)
               (grows_envs_kinded _ _ Hgrows1 Henvs) Hfold Hkinded').
      exact (Hkinds _ _ _ Hvalid Hinvariant Henvs Hvar Hkinded1 HX).
    + exfalso. clear -Hfold.
      induction vars as [ | var' vars IH]; [ discriminate | exact (IH Hfold) ].
Qed.

Lemma add_equals_keeps_kinds (p : Join_env.parameters)
    (import_var : Join_env.t -> variable -> option Join_env.t)
    (s s' : Join_env.t) (kind : flambda_kind) (var : variable)
    (simples : Simples_in_joined_envs.t) :
  fresh_avoids p ->
  import_keeps import_var ->
  import_keeps_scope import_var ->
  import_keeps_kinds import_var ->
  state_valid s ->
  joined_envs_are typing_env_invariant s ->
  joined_envs_are typing_env_kinded s ->
  vector_of_kind s simples kind ->
  vector_located s simples ->
  kind_in_target_env s var = Some kind ->
  Join_env.add_equals_in_joined_envs_with p import_var s kind var simples
  = Some s' ->
  state_kinded s' ->
  equations_kinded s ->
  equations_kinded s'.
Proof.
  intros Hfresh Hk Hs Hkinds Hvalid Hinvariant Henvs Hvector Hlocated Hvar
    Hadd Hkinded' HX.
  unfold Join_env.add_equals_in_joined_envs_with in Hadd.
  destruct (Join_env.n_way_join_canonical_simples s simples)
    as [[canonical | imported | canonicals] | ] eqn:Hcanonical;
    [ | | | discriminate ].
  - destruct (canonical_simples_cases s simples _ Hcanonical) as [Hstands _].
    destruct (stands_kind s simples canonical kind Hinvariant Henvs Hvector
                Hlocated Hstands) as [index [use [Huse Hkind]]].
    apply (add_equation_keeps_kinds _ _ _ _ Hadd);
      [ | rewrite kind_of_alias_type; exact Hvar | exact HX ].
    intros simple k Hin. rewrite aliases_alias_type_of in Hin.
    destruct Hin as [Heq | []]. injection Heq as <- <-.
    right. exists index, use. split; [ exact Huse | exact Hkind ].
  - destruct (import_var s imported) as [s1 | ] eqn:Hi; [ | discriminate ].
    pose proof (canonical_simples_cases s simples _ Hcanonical) as Hstands.
    cbn in Hstands.
    destruct (stands_kind s simples (Var imported) kind Hinvariant Henvs
                Hvector Hlocated Hstands) as [index [use [Huse Hkind]]].
    destruct (Hk _ _ _ Hvalid Hi) as [Hvalid1 [Hgrows1 _]].
    assert (Hkinded1 : state_kinded s1).
    { exact (same_readings_kinded _ _
               (same_readings_sym _ _
                  (add_equation_same_readings _ _ _ _ Hadd))
               Hkinded'). }
    pose proof (Hkinds _ _ _ Hvalid Hinvariant Henvs Hi Hkinded1 HX) as HX1.
    apply (add_equation_keeps_kinds _ _ _ _ Hadd);
      [ | rewrite kind_of_alias_type;
          exact (proj2 (proj2 Hgrows1) _ _ Hvar)
        | exact HX1 ].
    intros simple k Hin. rewrite aliases_alias_type_of in Hin.
    destruct Hin as [Heq | []]. injection Heq as <- <-.
    right. exists index, use.
    split; [ rewrite (proj1 Hgrows1); exact Huse | exact Hkind ].
  - injection Hadd as <-.
    pose proof (canonical_simples_cases _ _ _ Hcanonical) as Hcases.
    cbn in Hcases. subst canonicals.
    exact (add_alias_keeps_kinds p s kind var simples Hfresh Hvalid Hkinded'
             Hvector Hlocated Hvar HX).
Qed.

Lemma concrete_keeps_kinds (p : Join_env.parameters)
    (import_var : Join_env.t -> variable -> option Join_env.t)
    (s s' : Join_env.t) (var : variable)
    (types : Keyed.finite_map Index.t flambda_type) (kind : flambda_kind) :
  import_keeps import_var ->
  import_keeps_scope import_var ->
  import_keeps_kinds import_var ->
  state_valid s ->
  joined_envs_are typing_env_invariant s ->
  joined_envs_are typing_env_kinded s ->
  kind_in_target_env s var = Some kind ->
  (∀ (index : Index.t) (t : flambda_type),
     Keyed.lookup Index.equal index types = Some t ->
     FlambdaTypeGrammar.kind t = kind
     ∧ ∃ use : Typing_env.t,
         Join_env.joined_env s index = Some use ∧ type_kinded use t) ->
  Join_env.import_or_delay_n_way_join_of_concrete_type_with p import_var s var
    types
  = Some s' ->
  state_kinded s' ->
  equations_kinded s ->
  equations_kinded s'.
Proof.
  intros Hk Hs Hkinds Hvalid Hinvariant Henvs Hvar Htypes Hconcrete Hkinded'
    HX.
  unfold Join_env.import_or_delay_n_way_join_of_concrete_type_with
    in Hconcrete.
  destruct (Index.choose types) as [[index any_ty] | ] eqn:Hchoose;
    [ | discriminate ].
  pose proof (choose_lookup _ _ _ Hchoose) as Hany.
  destruct (Htypes _ _ Hany) as [Hkind_any [use [Huse Hkinded_any]]].
  destruct (forallb _ _).
  - destruct (Join_env.add_equation_in_target_env s var any_ty)
      as [s1 | ] eqn:Hadd; [ | discriminate ].
    pose proof (add_equation_same_readings _ _ _ _ Hadd) as Hsame.
    assert (HX1 : equations_kinded s1).
    { apply (add_equation_keeps_kinds _ _ _ _ Hadd);
        [ | rewrite Hkind_any; exact Hvar | exact HX ].
      intros simple k Hin. right. exists index, use.
      split; [ exact Huse | exact (Hkinded_any _ _ Hin) ]. }
    exact (import_vars_keeps_kinds import_var (free_variables any_ty) Hk Hs
             Hkinds s1 s' (same_readings_valid _ _ Hsame Hvalid)
             (grows_envs_invariant _ _ (same_readings_grows _ _ Hsame)
                Hinvariant)
             (grows_envs_kinded _ _ (same_readings_grows _ _ Hsame) Henvs)
             Hconcrete Hkinded' HX1).
  - injection Hconcrete as <-. destruct HX as [H1 [H4 [H4' [H5 HN]]]].
    split; [ exact H1 | ]. split; [ exact H4 | ].
    split; [ | split; [ exact H5 | exact HN ] ].
    intros x types0 Hx.
    cbn [Join_env.add_equations_in_joined_envs Join_env.types_in_joined_envs]
      in Hx.
    rewrite lookup_cons in Hx. destruct (Pos.eqb x var) eqn:Heq.
    + injection Hx as <-. apply Pos.eqb_eq in Heq. subst x. split.
      * exists index, any_ty. exact Hany.
      * intros index0 t Ht. rewrite (proj1 (Htypes _ _ Ht)). exact Hvar.
    + exact (H4' x types0 Hx).
Qed.

Lemma canonicals_located (envs : Joined_envs.t) :
  ∀ (types : list (Index.t * flambda_type))
    (canonicals : Simples_in_joined_envs.t),
    Joined_envs.canonicals_of_alias_types envs types = Some (Some canonicals)
    ->
    types <> [] ->
    ∃ (index : Index.t) (simple : simple) (env : Typing_env.t),
      Simples_in_joined_envs.find index canonicals = Some simple
      ∧ Joined_envs.get_nth_joined_env envs index = Some env.
Proof.
  intros [ | [index ty] types] canonicals H Hne; [ contradiction | ].
  cbn [Joined_envs.canonicals_of_alias_types] in H.
  destruct (Joined_envs.get_nth_joined_env envs index) as [env | ] eqn:Hnth;
    [ | discriminate ].
  destruct (Joined_envs.canonicals_of_alias_types envs types) as [rest | ];
    [ | discriminate ].
  cbn in H. injection H as H.
  destruct (get_alias_opt ty) as [alias | ]; [ | discriminate ].
  destruct rest as [rest | ]; [ | discriminate ].
  injection H as <-.
  exists index, (Typing_env.get_canonical_simple_ignoring_name_mode env alias),
    env.
  split; [ | exact Hnth ].
  unfold Simples_in_joined_envs.find. rewrite keyed_lookup_cons.
  rewrite (proj2 (index_equal_decides index index) eq_refl). reflexivity.
Qed.

Lemma import_body_keeps_kinds (p : Join_env.parameters)
    (import_var : Join_env.t -> variable -> option Join_env.t) :
  fresh_avoids p ->
  import_keeps import_var ->
  import_keeps_scope import_var ->
  import_keeps_kinds import_var ->
  import_keeps_kinds (Join_env.import_var_transitive_body p import_var).
Proof.
  intros Hfresh Hk Hs Hkinds s s' x Hvalid Hinvariant Henvs Hbody Hkinded' HX.
  pose proof (import_steps _ (import_body_keeps p _ Hfresh Hk)
                (import_body_keeps_scope p _ Hs) _ _ _ Hvalid Hbody)
    as Hsteps.
  pose proof (kinded_earlier _ _ Hsteps Hkinded') as Hkinded.
  unfold Join_env.import_var_transitive_body in Hbody.
  destruct (Source_env.exists_in_source_env (Join_env.source_env s) x)
    eqn:Hsource; [ injection Hbody as <-; exact HX | ].
  destruct (among x (Join_env.imported_vars s));
    [ injection Hbody as <-; exact HX | ].
  destruct (Joined_envs.kind_of_imported_var (Join_env.joined_envs s) x)
    as [kind | ] eqn:Hkind; [ | discriminate ].
  cbv zeta in Hbody. fold (with_import s x kind) in Hbody.
  assert (Hunknown : ~ knows (Join_env.source_env s) x).
  { unfold knows. unfold Source_env.exists_in_source_env in Hsource.
    rewrite Hsource. discriminate. }
  destruct (kind_of_imported_var_known _ _ _ Hkind)
    as [indexk [usek [Husek Hknowsk]]].
  destruct (with_import_keeps s x kind Hvalid Hunknown
              (kind_of_imported_var_known _ _ _ Hkind))
    as [Hvalid1 [Hgrows1 Hexists1]].
  destruct (Joined_envs.find_imported_var _ x kind) as [types | ] eqn:Htypes;
    [ | discriminate ].
  destruct (types_of_imported_var_spec _ _ _ _ Htypes) as [Hknown Hfound].
  assert (Hx :
            ∀ (index : Index.t) (use : Typing_env.t),
              Join_env.joined_env s index = Some use ->
              knows use x ->
              simple_of_kind use (Var x) kind).
  { intros index use Huse Hknows.
    destruct (joined_env_in_compact _ _ _ Huse) as [equations Hin].
    exact (Hknown _ _ _ Hin Hknows). }
  pose proof (with_import_keeps_kinded s x kind Hx Hkinded) as Hkinded1.
  pose proof (grows_envs_invariant _ _ Hgrows1 Hinvariant) as Hinvariant1.
  pose proof (grows_envs_kinded _ _ Hgrows1 Henvs) as Henvs1.
  assert (Hvar1 : kind_in_target_env (with_import s x kind) x = Some kind).
  { apply (kind_in_target_of_use _ indexk usek x kind Hvalid1 Hkinded1);
      [ rewrite (proj1 Hgrows1); exact Husek
      | exact (Hx _ _ Husek Hknowsk)
      | exact Hexists1 ]. }
  assert (HX1 : equations_kinded (with_import s x kind)).
  { apply (kinds_same_maps s _ Hgrows1); try reflexivity. exact HX. }
  assert (Htypes_kinded :
            ∀ (index : Index.t) (t : flambda_type),
              Keyed.lookup Index.equal index types = Some t ->
              FlambdaTypeGrammar.kind t = kind
              ∧ ∃ use : Typing_env.t,
                  Join_env.joined_env (with_import s x kind) index = Some use
                  ∧ type_kinded use t).
  { intros index t Ht.
    apply (keyed_lookup_in Index.equal index_equal_decides) in Ht.
    destruct (Hfound _ _ Ht) as [env [equations [Hin Hfind]]].
    pose proof (in_compact_joined_env _ _ _ _ Hin) as Huse.
    split; [ exact (find_kind _ _ _ _ Hfind) | ].
    exists env. split; [ exact Huse | ].
    exact (find_kinded_in _ _ _ _ (Henvs _ _ Huse) Hfind). }
  destruct (existsb _ _) eqn:Hexists.
  - destruct
      (Joined_envs.get_alias_then_canonical_simple_ignoring_name_mode_exn
         _ _) as [[canonical_simples | ] | ] eqn:Hcanonicals;
      try discriminate.
    apply (add_equals_keeps_kinds p import_var (with_import s x kind) s' kind
             x canonical_simples Hfresh Hk Hs Hkinds Hvalid1 Hinvariant1
             Henvs1);
      [ | | exact Hvar1 | exact Hbody | exact Hkinded' | exact HX1 ].
    + intros index c use Hfind Huse.
      apply (keyed_lookup_in Index.equal index_equal_decides) in Hfind.
      destruct (canonicals_of_alias_types_spec _ _ _ Hcanonicals _ _ Hfind)
        as [ty' [env [alias [Hin' [Hnth [Halias ->]]]]]].
      change (Join_env.joined_env s index = Some use) in Huse.
      change (Join_env.joined_env s index = Some env) in Hnth.
      rewrite Huse in Hnth. injection Hnth as <-.
      apply (keyed_in_compact Index.equal index_equal_decides) in Hin'.
      rewrite (keyed_lookup_map_values Index.t Index.equal) in Hin'.
      destruct (Keyed.lookup Index.equal index types) as [ty0 | ] eqn:Hty0;
        [ | discriminate ].
      injection Hin' as Hty'.
      apply (keyed_lookup_in Index.equal index_equal_decides) in Hty0.
      destruct (Hfound _ _ Hty0) as [env0 [equations0 [Hin0 Hfind0]]].
      pose proof (in_compact_joined_env _ _ _ _ Hin0) as Huse0.
      change (Join_env.joined_env s index = Some env0) in Huse0.
      rewrite Huse in Huse0. injection Huse0 as <-.
      apply (canonical_kinded use alias kind (Hinvariant _ _ Huse)
               (Henvs _ _ Huse)).
      destruct (get_alias_opt ty0) as [alias0 | ] eqn:Halias0.
      * subst ty'. rewrite Halias0 in Halias. injection Halias as <-.
        pose proof (get_alias_in_aliases _ _ Halias0) as Hmember.
        rewrite (find_kind _ _ _ _ Hfind0) in Hmember.
        exact (find_kinded_in _ _ _ _ (Henvs _ _ Huse) Hfind0 _ _ Hmember).
      * subst ty'. rewrite get_alias_alias_type_of in Halias.
        injection Halias as <-. exists ty0. exact Hfind0.
    + apply existsb_exists in Hexists. destruct Hexists as [[index t] [Hin _]].
      apply (keyed_in_compact Index.equal index_equal_decides) in Hin.
      refine (canonicals_located _ _ _ Hcanonicals _).
      match goal with
      | |- Keyed.compact Index.equal ?m <> [] =>
        assert (Hlookup :
                  ∃ t' : flambda_type,
                    Keyed.lookup Index.equal index m = Some t')
      end.
      { rewrite (keyed_lookup_map_values Index.t Index.equal), Hin.
        eexists. reflexivity. }
      destruct Hlookup as [t' Ht'].
      apply (keyed_in_compact Index.equal index_equal_decides) in Ht'.
      intros Hempty. rewrite Hempty in Ht'. destruct Ht'.
  - exact (concrete_keeps_kinds p import_var _ s' x types kind Hk Hs Hkinds
             Hvalid1 Hinvariant1 Henvs1 Hvar1 Htypes_kinded Hbody Hkinded'
             HX1).
Qed.

Lemma import_var_keeps_kinds (p : Join_env.parameters) (fuel : nat) :
  fresh_avoids p -> import_keeps_kinds (Join_env.import_var_transitive p fuel).
Proof.
  intros Hfresh. induction fuel as [ | fuel IH].
  - intros s s' x _ _ _ H. discriminate.
  - exact (import_body_keeps_kinds p _ Hfresh (import_var_keeps p fuel Hfresh)
             (import_var_keeps_scope p fuel) IH).
Qed.

Lemma canonical_simples_fold_some (envs : Joined_envs.t) :
  ∀ (simples : list (Index.t * simple))
    (acc canonicals : Simples_in_joined_envs.t),
    List.fold_left
      (fun acc binding =>
         let* acc := acc in
         let* env := Joined_envs.get_nth_joined_env envs (fst binding) in
         Some
           ((fst binding,
             Typing_env.get_canonical_simple_ignoring_name_mode env
               (snd binding))
            :: acc))
      simples (Some acc)
    = Some canonicals ->
    acc <> [] ∨ simples <> [] ->
    canonicals <> [].
Proof.
  induction simples as [ | binding simples IH]; intros acc canonicals Hfold Hne.
  - injection Hfold as <-. destruct Hne as [H | H]; [ exact H | ].
    exfalso. apply H. reflexivity.
  - cbn [List.fold_left] in Hfold.
    destruct (Joined_envs.get_nth_joined_env envs (fst binding)) as [env | ].
    + cbn in Hfold. apply (IH _ _ Hfold). left. discriminate.
    + exfalso. cbn in Hfold. clear -Hfold.
      induction simples as [ | b simples IH];
        [ discriminate | exact (IH Hfold) ].
Qed.

Lemma canonical_simples_some (envs : Joined_envs.t)
    (simples : list (Index.t * simple))
    (canonicals : Simples_in_joined_envs.t) :
  Joined_envs.get_canonical_simples_ignoring_name_mode envs simples
  = Some canonicals ->
  simples <> [] ->
  ∃ (index : Index.t) (simple : simple),
    Simples_in_joined_envs.find index canonicals = Some simple.
Proof.
  intros Hfold Hne.
  pose proof (canonical_simples_fold_some envs simples [] canonicals Hfold
                (or_intror Hne)) as Hsome.
  destruct canonicals as [ | [index simple] rest]; [ contradiction | ].
  exists index, simple. unfold Simples_in_joined_envs.find.
  rewrite keyed_lookup_cons.
  rewrite (proj2 (index_equal_decides index index) eq_refl). reflexivity.
Qed.

(* What every function of the join of environments keeps of kinds,
   between any two states. *)
Definition keeps_kinds (s s' : Join_env.t) : Prop :=
  state_valid s ->
  steps s s'
  ∧ (joined_envs_are typing_env_invariant s ->
     joined_envs_are typing_env_kinded s ->
     state_kinded s' ->
     equations_kinded s ->
     equations_kinded s').

Lemma keeps_kinds_refl (s : Join_env.t) : keeps_kinds s s.
Proof.
  intros Hvalid. split; [ exact (steps_refl s Hvalid) | ].
  intros _ _ _ HX. exact HX.
Qed.

Lemma keeps_kinds_trans (s1 s2 s3 : Join_env.t) :
  keeps_kinds s1 s2 -> keeps_kinds s2 s3 -> keeps_kinds s1 s3.
Proof.
  intros H12 H23 Hvalid1. destruct (H12 Hvalid1) as [Hsteps12 K12].
  destruct (H23 (proj1 (proj2 Hsteps12))) as [Hsteps23 K23].
  split; [ exact (steps_trans _ _ _ Hsteps12 Hsteps23) | ].
  intros Hinvariant Henvs Hkinded3 HX.
  pose proof (proj1 (proj2 (proj2 Hsteps12))) as Hgrows12.
  apply (K23 (grows_envs_invariant _ _ Hgrows12 Hinvariant)
           (grows_envs_kinded _ _ Hgrows12 Henvs) Hkinded3).
  exact (K12 Hinvariant Henvs (kinded_earlier _ _ Hsteps23 Hkinded3) HX).
Qed.

Lemma import_var_keeps_kinds_between (p : Join_env.parameters) (fuel : nat)
    (s s' : Join_env.t) (x : variable) :
  fresh_avoids p ->
  Join_env.import_var_transitive p fuel s x = Some s' ->
  keeps_kinds s s'.
Proof.
  intros Hfresh Hi Hvalid. split.
  - exact (import_steps _ (import_var_keeps p fuel Hfresh)
             (import_var_keeps_scope p fuel) _ _ _ Hvalid Hi).
  - intros Hinvariant Henvs Hkinded' HX.
    exact (import_var_keeps_kinds p fuel Hfresh _ _ _ Hvalid Hinvariant Henvs
             Hi Hkinded' HX).
Qed.

Lemma import_type_keeps_kinds_between (p : Join_env.parameters) (fuel : nat)
    (s s' : Join_env.t) (ty : flambda_type) :
  fresh_avoids p ->
  Join_env.import_type_transitive p fuel s ty = Some s' ->
  keeps_kinds s s'.
Proof.
  intros Hfresh Hi Hvalid. split.
  - exact (import_vars_steps _ (free_variables ty)
             (import_var_keeps p fuel Hfresh) (import_var_keeps_scope p fuel)
             _ _ Hvalid Hi).
  - intros Hinvariant Henvs Hkinded' HX.
    exact (import_vars_keeps_kinds _ (free_variables ty)
             (import_var_keeps p fuel Hfresh) (import_var_keeps_scope p fuel)
             (import_var_keeps_kinds p fuel Hfresh) s s' Hvalid Hinvariant
             Henvs Hi Hkinded' HX).
Qed.

Lemma n_way_join_simples_keeps_kinds (p : Join_env.parameters) (fuel : nat)
    (s s' : Join_env.t) (kind : flambda_kind)
    (simples : list (Index.t * simple)) (result : Or_bottom.t simple) :
  fresh_avoids p ->
  Join_env.n_way_join_simples p fuel s kind simples = Some (result, s') ->
  keeps_kinds s s'.
Proof.
  intros Hfresh Hjoin. unfold Join_env.n_way_join_simples in Hjoin.
  destruct simples as [ | first rest].
  { injection Hjoin as _ <-. apply keeps_kinds_refl. }
  destruct (Joined_envs.get_canonical_simples_ignoring_name_mode _ _)
    as [canonicals | ] eqn:Hcanonicals; [ | discriminate ].
  destruct (Join_env.n_way_join_canonical_simples s canonicals)
    as [[simple0 | var | canonicals'] | ] eqn:Hcanonical;
    [ | | | discriminate ].
  - injection Hjoin as _ <-. apply keeps_kinds_refl.
  - destruct (Join_env.import_var_transitive p fuel s var) as [s1 | ]
      eqn:Himport; [ | discriminate ].
    injection Hjoin as _ <-.
    exact (import_var_keeps_kinds_between p fuel _ _ _ Hfresh Himport).
  - destruct (Join_env.existential_for_these_canonicals p s canonicals' kind)
      as [x s1] eqn:Hmake.
    injection Hjoin as _ <-.
    pose proof (canonical_simples_cases _ _ _ Hcanonical) as Hcases.
    cbn in Hcases. subst canonicals'.
    intros Hvalid.
    destruct (existential_keeps _ _ _ _ _ _ Hfresh Hvalid Hmake)
      as [Hvalid1 [Hgrows1 _]].
    split.
    + split; [ exact Hvalid | ]. split; [ exact Hvalid1 | ].
      split; [ exact Hgrows1 | ].
      exact (proj1 (existential_keeps_scope _ _ _ _ _ _ Hmake)).
    + intros _ _ _ HX.
      apply (existential_keeps_kinds _ _ _ _ _ _ Hfresh Hvalid) with (3 := HX)
        (2 := Hmake).
      apply (canonical_simples_some _ _ _ Hcanonicals). discriminate.
Qed.

Lemma n_way_join_env_extension_keeps_kinds (p : Join_env.parameters)
    (fuel : nat) (s s' : Join_env.t)
    (extensions : list (Index.t * env_extension)) (joined : env_extension) :
  fresh_avoids p ->
  Join_env.n_way_join_env_extension p fuel s extensions
  = Some (Or_bottom.Ok (joined, s')) ->
  keeps_kinds s s'.
Proof.
  intros Hfresh Hjoin. unfold Join_env.n_way_join_env_extension in Hjoin.
  destruct extensions as [ | [index0 first] others]; [ discriminate | ].
  destruct (forallb _ others).
  - destruct (List.fold_left _ (Join_env.equations_of first) (Some s))
      as [s1 | ] eqn:Hfold; [ | discriminate ].
    injection Hjoin as _ <-. revert s Hfold.
    generalize (Join_env.equations_of first) as equations.
    induction equations as [ | [n ty] equations IH]; intros s Hfold.
    + injection Hfold as <-. apply keeps_kinds_refl.
    + cbn [List.fold_left fst snd] in Hfold.
      destruct (Join_env.import_name_transitive p fuel s n) as [s2 | ]
        eqn:Hname.
      * assert (Hfirst : keeps_kinds s s2).
        { destruct n as [x | sym].
          - exact (import_var_keeps_kinds_between p fuel _ _ _ Hfresh Hname).
          - injection Hname as <-. apply keeps_kinds_refl. }
        destruct (Join_env.import_type_transitive p fuel s2 ty) as [s3 | ]
          eqn:Htype.
        -- exact (keeps_kinds_trans _ _ _ Hfirst
                    (keeps_kinds_trans _ _ _
                       (import_type_keeps_kinds_between p fuel _ _ _ Hfresh
                          Htype)
                       (IH _ Hfold))).
        -- exfalso. clear -Hfold.
           induction equations as [ | e equations IH];
             [ discriminate | exact (IH Hfold) ].
      * exfalso. clear -Hfold.
        induction equations as [ | e equations IH];
          [ discriminate | exact (IH Hfold) ].
  - injection Hjoin as _ <-. apply keeps_kinds_refl.
Qed.

Theorem n_way_join_keeps_kinds : n_way_join_keeps_kinds_statement.
Proof.
  intros p fuel_of_env fuel expand_head reduce_head_of_kind_naked_immediate
    type_eqb block_shape_eqb Hfresh s s' ts joined Hstep Hvalid Hinvariant
    Henvs Hkinded' HX.
  refine
    (proj2
       (n_way_join_keeps Join_env.t keeps_kinds Join_env.joined_env
          (Join_env.n_way_join_simples p fuel_of_env)
          (Join_env.import_type p fuel_of_env)
          (Join_env.n_way_join_env_extension p fuel_of_env) expand_head
          reduce_head_of_kind_naked_immediate type_eqb block_shape_eqb fuel
          keeps_kinds_refl keeps_kinds_trans _ _ _ s s' ts joined Hstep
          Hvalid)
       Hinvariant Henvs Hkinded' HX).
  - intros s0 s0' kind simples result Hsimples.
    exact (n_way_join_simples_keeps_kinds p fuel_of_env s0 s0' kind simples
             result Hfresh Hsimples).
  - intros s0 s0' ty Himport.
    exact (import_type_keeps_kinds_between p fuel_of_env s0 s0' ty Hfresh
             Himport).
  - intros s0 s0' extensions joined0 Hextension.
    exact (n_way_join_env_extension_keeps_kinds p fuel_of_env s0 s0'
             extensions joined0 Hfresh Hextension).
Qed.

Lemma recover_kind (relations relations' : Join_env.inverse_relations)
    (n : name) (ty ty' : flambda_type) :
  Join_env.recover_inverse_relations relations n ty = Some (ty', relations')
  ->
  kind ty' = kind ty.
Proof.
  intros Hrecover.
  destruct (recover_type_cases _ _ _ _ _ Hrecover)
    as [-> | [is_int [immediates [g [blocks [extensions [is_unique
          [when_block [-> [_ ->]]]]]]]]]]; reflexivity.
Qed.

Lemma in_app_tail {A : Type} (x : A) (l m m' : list A) :
  (x ∈ m -> x ∈ m') -> x ∈ l ++ m -> x ∈ l ++ m'.
Proof.
  intros Htail H. apply in_app_or in H. apply in_or_app.
  destruct H as [H | H]; [ left; exact H | right; exact (Htail H) ].
Qed.

(* The aliases of the type answered are aliases of the type given. (The
   equation that is added has no alias; the equation that it replaces
   is no longer there.) *)
Lemma recover_aliases (relations relations' : Join_env.inverse_relations)
    (n : name) (ty ty' : flambda_type) :
  Join_env.recover_inverse_relations relations n ty = Some (ty', relations')
  ->
  ∀ alias : simple * flambda_kind,
    alias ∈ aliases_with_kinds ty' -> alias ∈ aliases_with_kinds ty.
Proof.
  intros Hrecover alias.
  destruct (recover_type_cases _ _ _ _ _ Hrecover)
    as [-> | [is_int [immediates [g [blocks [extensions [is_unique
          [when_block [-> [Hadd ->]]]]]]]]]]; [ intros H; exact H | ].
  assert (Hext :
            alias ∈ aliases_of_env_extension when_block ->
            alias
            ∈ aliases_of_env_extension (snd (extensions_pair extensions))).
  { destruct (add_inverse_relation_spec _ _ _ _ _ _ _ Hadd)
      as [-> | [n' [head [_ [_ ->]]]]]; [ intros H; exact H | ].
    destruct (snd (extensions_pair extensions)) as [equations].
    cbn [Join_env.add_or_replace_equation aliases_of_env_extension flat_map
           snd].
    intros H. apply in_app_or in H. destruct H as [[] | H].
    apply in_flat_map in H. destruct H as [binding [Hbinding H]].
    apply filter_In in Hbinding. apply in_flat_map. exists binding.
    split; [ exact (proj1 Hbinding) | exact H ]. }
  unfold variant_type.
  cbn [aliases_with_kinds aliases_of_descr aliases_of_head_of_kind_value
         aliases_of_head_of_kind_value_non_null].
  rewrite !app_nil_r. do 4 apply in_app_tail.
  destruct extensions as [ | a b]; cbn [extensions_pair fst snd] in Hext;
    intros H.
  - apply in_app_or in H. destruct H as [H | H]; [ exact H | exact (Hext H) ].
  - apply in_app_or in H. apply in_or_app.
    destruct H as [H | H]; [ left; exact H | right; exact (Hext H) ].
Qed.

Lemma recover_relations_results
    (relations relations' : Join_env.inverse_relations) (n : name)
    (ty ty' : flambda_type) (r : variable)
    (names : list (Relation.t * name)) :
  Join_env.recover_inverse_relations relations n ty = Some (ty', relations')
  ->
  lookup r relations' = Some names ->
  (∃ names0 : list (Relation.t * name), lookup r relations = Some names0)
  ∨ (Var r, Naked_number Kind_naked_immediate) ∈ aliases_with_kinds ty.
Proof.
  intros Hrecover Hr.
  destruct (recover_relations_cases _ _ _ _ _ Hrecover)
    as [-> | [H | [H | H]]].
  - left. exists names. exact Hr.
  - destruct H as [is_int [imm_ty [g [blocks [extensions [is_unique
                    [-> [_ ->]]]]]]]].
    unfold Join_env.add_to_inverse_relations in Hr. rewrite lookup_cons in Hr.
    destruct (Pos.eqb r g) eqn:Heq; [ | left; exists names; exact Hr ].
    apply Pos.eqb_eq in Heq. subst r. right. unfold variant_type.
    cbn [aliases_with_kinds aliases_of_descr aliases_of_head_of_kind_value
           aliases_of_head_of_kind_value_non_null].
    rewrite app_nil_r. apply in_or_app. right. apply in_or_app. right.
    left. reflexivity.
  - destruct H as [r0 [immediates [get_tag [blocks [extensions [is_unique
                    [-> ->]]]]]]].
    unfold Join_env.add_to_inverse_relations in Hr. rewrite lookup_cons in Hr.
    destruct (Pos.eqb r r0) eqn:Heq; [ | left; exists names; exact Hr ].
    apply Pos.eqb_eq in Heq. subst r. right. unfold variant_type.
    cbn [aliases_with_kinds aliases_of_descr aliases_of_head_of_kind_value
           aliases_of_head_of_kind_value_non_null].
    rewrite app_nil_r. left. reflexivity.
  - destruct H as [non_null [r0 [-> ->]]].
    unfold Join_env.add_to_inverse_relations in Hr. rewrite lookup_cons in Hr.
    destruct (Pos.eqb r r0) eqn:Heq; [ | left; exists names; exact Hr ].
    apply Pos.eqb_eq in Heq. subst r. right.
    cbn [aliases_with_kinds aliases_of_descr aliases_of_head_of_kind_value].
    apply in_or_app. right. left. reflexivity.
Qed.

Lemma types_of_kind_grows (s s' : Join_env.t) (x : variable)
    (types : Keyed.finite_map Index.t flambda_type) :
  grows s s' -> types_of_kind s x types -> types_of_kind s' x types.
Proof.
  intros [_ [_ Hkinds]] [Hsome Hall]. split; [ exact Hsome | ].
  intros index t Ht. exact (Hkinds _ _ (Hall index t Ht)).
Qed.

Lemma import_or_delay_steps (p : Join_env.parameters) (fuel : nat)
    (s s' : Join_env.t) (var : variable)
    (types : Keyed.finite_map Index.t flambda_type) :
  fresh_avoids p ->
  state_valid s ->
  knows (Join_env.source_env s) var ∨ is_imported s var ->
  Join_env.import_or_delay_n_way_join_type p fuel s var types = Some s' ->
  steps s s'.
Proof.
  intros Hfresh Hvalid Hvar H.
  pose proof (proj1 (import_or_delay_keeps_scope p fuel s s' var types Hvar H))
    as Hscope.
  unfold Join_env.import_or_delay_n_way_join_type,
    Join_env.import_or_delay_n_way_join_type_with in H.
  destruct (Index.choose types) as [[index any_ty] | ]; [ | discriminate ].
  destruct
    (Joined_envs.get_alias_then_canonical_simple_ignoring_name_mode_exn _ _)
    as [[canonical_simples | ] | ]; [ | | discriminate ].
  - destruct (add_equals_keeps p _ s s' _ var canonical_simples Hfresh
                (import_var_keeps p fuel Hfresh) Hvalid H)
      as [Hvalid' Hgrows].
    split; [ exact Hvalid | ]. split; [ exact Hvalid' | ].
    split; [ exact Hgrows | exact Hscope ].
  - destruct (concrete_keeps p _ s s' var types
                (import_var_keeps p fuel Hfresh) Hvalid H)
      as [Hvalid' Hgrows].
    split; [ exact Hvalid | ]. split; [ exact Hvalid' | ].
    split; [ exact Hgrows | exact Hscope ].
Qed.

Lemma import_or_delay_keeps_kinds (p : Join_env.parameters) (fuel : nat)
    (s s' : Join_env.t) (var : variable)
    (types : Keyed.finite_map Index.t flambda_type) (kind : flambda_kind) :
  fresh_avoids p ->
  state_valid s ->
  joined_envs_are typing_env_invariant s ->
  joined_envs_are typing_env_kinded s ->
  kind_in_target_env s var = Some kind ->
  (∀ (index : Index.t) (t : flambda_type),
     Keyed.lookup Index.equal index types = Some t ->
     FlambdaTypeGrammar.kind t = kind
     ∧ ∃ use : Typing_env.t,
         Join_env.joined_env s index = Some use ∧ type_kinded use t) ->
  Join_env.import_or_delay_n_way_join_type p fuel s var types = Some s' ->
  state_kinded s' ->
  equations_kinded s ->
  equations_kinded s'.
Proof.
  intros Hfresh Hvalid Hinvariant Henvs Hvar Htypes H Hkinded' HX.
  unfold Join_env.import_or_delay_n_way_join_type,
    Join_env.import_or_delay_n_way_join_type_with in H.
  destruct (Index.choose types) as [[index any_ty] | ] eqn:Hchoose;
    [ | discriminate ].
  pose proof (choose_lookup _ _ _ Hchoose) as Hany.
  destruct
    (Joined_envs.get_alias_then_canonical_simple_ignoring_name_mode_exn _ _)
    as [[canonical_simples | ] | ] eqn:Hcanonicals; [ | | discriminate ].
  - rewrite (proj1 (Htypes _ _ Hany)) in H.
    apply (add_equals_keeps_kinds p _ s s' kind var canonical_simples Hfresh
             (import_var_keeps p fuel Hfresh) (import_var_keeps_scope p fuel)
             (import_var_keeps_kinds p fuel Hfresh) Hvalid Hinvariant Henvs);
      [ | | exact Hvar | exact H | exact Hkinded' | exact HX ].
    + intros index0 c use Hfind Huse.
      apply (keyed_lookup_in Index.equal index_equal_decides) in Hfind.
      destruct (canonicals_of_alias_types_spec _ _ _ Hcanonicals _ _ Hfind)
        as [ty' [env [alias [Hin' [Hnth [Halias ->]]]]]].
      change (Join_env.joined_env s index0 = Some env) in Hnth.
      rewrite Huse in Hnth. injection Hnth as <-.
      apply (keyed_in_compact Index.equal index_equal_decides) in Hin'.
      destruct (Htypes _ _ Hin') as [Hkind' [use' [Huse' Hkinded_ty']]].
      rewrite Huse in Huse'. injection Huse' as <-.
      apply (canonical_kinded use alias kind (Hinvariant _ _ Huse)
               (Henvs _ _ Huse)).
      pose proof (get_alias_in_aliases _ _ Halias) as Hmember.
      rewrite Hkind' in Hmember. exact (Hkinded_ty' _ _ Hmember).
    + refine (canonicals_located _ _ _ Hcanonicals _).
      apply (keyed_in_compact Index.equal index_equal_decides) in Hany.
      intros Hempty. rewrite Hempty in Hany. destruct Hany.
  - exact (concrete_keeps_kinds p _ s s' var types kind
             (import_var_keeps p fuel Hfresh) (import_var_keeps_scope p fuel)
             (import_var_keeps_kinds p fuel Hfresh) Hvalid Hinvariant Henvs
             Hvar Htypes H Hkinded' HX).
Qed.

(* The step of the fold of the first phase. *)
Definition aliases_step (p : Join_env.parameters) (fuel : nat)
    (envs : Joined_envs.t) (test : variable -> bool)
    (env : option Join_env.t) (name : name) : option Join_env.t :=
  let* env := env in
  match name with
  | Name_symbol _ => Some env
  | Name_var var =>
    if test var
    then
      Join_env.import_or_delay_n_way_join_type p fuel env var
        (Join_env.types_of_name_in_levels envs name)
    else Some env
  end.

Lemma aliases_fold_steps (p : Join_env.parameters) (fuel : nat)
    (envs : Joined_envs.t) (test : variable -> bool) :
  fresh_avoids p ->
  ∀ (names : list name) (s s' : Join_env.t),
    state_valid s ->
    (∀ x : variable, test x = true -> knows (Join_env.source_env s) x) ->
    List.fold_left (aliases_step p fuel envs test) names (Some s) = Some s' ->
    steps s s'.
Proof.
  intros Hfresh.
  induction names as [ | n names IH]; intros s s' Hvalid Htest Hfold.
  - injection Hfold as <-. exact (steps_refl s Hvalid).
  - cbn [List.fold_left] in Hfold.
    destruct (aliases_step p fuel envs test (Some s) n) as [s1 | ] eqn:Hstep;
      [ | rewrite (fold_left_none _ (fun b => eq_refl)) in Hfold;
          discriminate ].
    assert (Hfirst : steps s s1).
    { destruct n as [var | sym]; cbn [aliases_step] in Hstep.
      - destruct (test var) eqn:Hvar.
        + exact (import_or_delay_steps p fuel s s1 var _ Hfresh Hvalid
                   (or_introl (Htest var Hvar)) Hstep).
        + injection Hstep as <-. exact (steps_refl s Hvalid).
      - injection Hstep as <-. exact (steps_refl s Hvalid). }
    apply (steps_trans _ _ _ Hfirst).
    apply (IH s1 s' (proj1 (proj2 Hfirst))); [ | exact Hfold ].
    intros x Hx.
    rewrite (proj1 (proj2 (proj2 (proj2 (proj2 Hfirst))))).
    exact (Htest x Hx).
Qed.

Lemma aliases_fold_keeps_kinds (p : Join_env.parameters) (fuel : nat)
    (envs : Joined_envs.t) (test : variable -> bool) (s0 : Join_env.t) :
  fresh_avoids p ->
  joined_envs_are typing_env_invariant s0 ->
  joined_envs_are typing_env_kinded s0 ->
  (∀ x : variable, test x = true -> knows (Join_env.source_env s0) x) ->
  ∀ (names : list name) (s s' : Join_env.t),
    (∀ x : variable,
       Name_var x ∈ names ->
       test x = true ->
       ∃ kind : flambda_kind,
         kind_in_target_env s0 x = Some kind
         ∧ ∀ (index : Index.t) (t : flambda_type),
             Keyed.lookup Index.equal index
               (Join_env.types_of_name_in_levels envs (Name_var x))
             = Some t ->
             FlambdaTypeGrammar.kind t = kind
             ∧ ∃ use : Typing_env.t,
                 Join_env.joined_env s0 index = Some use ∧ type_kinded use t) ->
    steps s0 s ->
    List.fold_left (aliases_step p fuel envs test) names (Some s) = Some s' ->
    state_kinded s' ->
    equations_kinded s ->
    equations_kinded s'.
Proof.
  intros Hfresh Hinvariant0 Henvs0 Htest.
  induction names as [ | n names IH];
    intros s s' Hvars Hsteps Hfold Hkinded' HX.
  - injection Hfold as <-. exact HX.
  - cbn [List.fold_left] in Hfold.
    destruct (aliases_step p fuel envs test (Some s) n) as [s1 | ] eqn:Hstep;
      [ | rewrite (fold_left_none _ (fun b => eq_refl)) in Hfold;
          discriminate ].
    destruct Hsteps as [Hvalid0 [Hvalid [Hgrows Hscope]]].
    assert (Hsource : Join_env.source_env s = Join_env.source_env s0).
    { exact (proj1 (proj2 Hscope)). }
    assert (Htest_s :
              ∀ x : variable, test x = true -> knows (Join_env.source_env s) x).
    { intros x Hx. rewrite Hsource. exact (Htest x Hx). }
    assert (Hrest :
              ∀ x : variable,
                Name_var x ∈ names ->
                test x = true ->
                ∃ kind : flambda_kind,
                  kind_in_target_env s0 x = Some kind
                  ∧ ∀ (index : Index.t) (t : flambda_type),
                      Keyed.lookup Index.equal index
                        (Join_env.types_of_name_in_levels envs (Name_var x))
                      = Some t ->
                      FlambdaTypeGrammar.kind t = kind
                      ∧ ∃ use : Typing_env.t,
                          Join_env.joined_env s0 index = Some use
                          ∧ type_kinded use t).
    { intros x Hx. exact (Hvars x (or_intror Hx)). }
    assert (Hfirst : steps s s1 ∧ (state_kinded s1 -> equations_kinded s1)).
    { destruct n as [var | sym]; cbn [aliases_step] in Hstep.
      - destruct (test var) eqn:Hvar.
        + split.
          * exact (import_or_delay_steps p fuel s s1 var _ Hfresh Hvalid
                     (or_introl (Htest_s var Hvar)) Hstep).
          * intros Hkinded1.
            destruct (Hvars var (or_introl eq_refl) Hvar)
              as [kind [Hkind Htypes]].
            apply (import_or_delay_keeps_kinds p fuel s s1 var
                     (Join_env.types_of_name_in_levels envs (Name_var var))
                     kind Hfresh Hvalid
                     (grows_envs_invariant _ _ Hgrows Hinvariant0)
                     (grows_envs_kinded _ _ Hgrows Henvs0)
                     (proj2 (proj2 Hgrows) _ _ Hkind));
              [ | exact Hstep | exact Hkinded1 | exact HX ].
            intros index t Ht. destruct (Htypes index t Ht)
              as [Hk [use [Huse Hkinded_t]]].
            split; [ exact Hk | ]. exists use.
            split; [ rewrite (proj1 Hgrows); exact Huse | exact Hkinded_t ].
        + injection Hstep as <-.
          split; [ exact (steps_refl s Hvalid) | intros _; exact HX ].
      - injection Hstep as <-.
        split; [ exact (steps_refl s Hvalid) | intros _; exact HX ]. }
    destruct Hfirst as [Hsteps1 HX1].
    assert (Htest1 :
              ∀ x : variable,
                test x = true -> knows (Join_env.source_env s1) x).
    { intros x Hx.
      rewrite (proj1 (proj2 (proj2 (proj2 (proj2 Hsteps1))))).
      exact (Htest_s x Hx). }
    pose proof (aliases_fold_steps p fuel envs test Hfresh names s1 s'
                  (proj1 (proj2 Hsteps1)) Htest1 Hfold) as Hsteps'.
    apply (IH s1 s' Hrest
             (steps_trans _ _ _
                (conj Hvalid0 (conj Hvalid (conj Hgrows Hscope))) Hsteps1)
             Hfold Hkinded').
    exact (HX1 (kinded_earlier _ _ Hsteps' Hkinded')).
Qed.

Lemma existential_definition_kind (s : Join_env.t) (x : variable)
    (simples : Simples_in_joined_envs.t) (kind : flambda_kind) :
  state_valid s ->
  lookup x (Join_env.definitions_of_existentials s) = Some (simples, kind) ->
  kind_in_target_env s x = Some kind.
Proof.
  intros [_ [Hnew [[Hrecorded _] _]]] Hdefinition.
  pose proof (Hrecorded _ _ _ Hdefinition) as Hlookup.
  assert (Hex : is_existential s x).
  { apply is_bound_lookup. exists kind. exact Hlookup. }
  destruct (Hnew x Hex) as [Hunknown _].
  unfold kind_in_target_env.
  unfold knows, Typing_env.mem, Join_env.source_env in Hunknown.
  destruct (Keyed.lookup name_eqb (Name_var x)
              (Typing_env.names_to_types
                 (Bindings_in_target_env.source_env (Join_env.bindings s))))
    as [[ty time] | ]; [ exfalso; apply Hunknown; reflexivity | ].
  rewrite Hlookup. reflexivity.
Qed.

Lemma existential_types_of_kind (p : Join_env.parameters) (s : Join_env.t)
    (x : variable) (simples : Simples_in_joined_envs.t) (kind : flambda_kind)
    (types : Keyed.finite_map Index.t flambda_type) :
  expand_const_of_kind p ->
  state_valid s ->
  state_holds s ->
  state_kinded s ->
  definitions_nonempty s ->
  lookup x (Join_env.definitions_of_existentials s) = Some (simples, kind) ->
  Joined_envs.find_simples_in_joined_envs (Join_env.expand_const p)
    (Join_env.joined_envs s) simples kind
  = Some types ->
  types_of_kind s x types.
Proof.
  intros Hconst Hvalid Hholds Hstate HN Hdefinition Hfind.
  unfold Joined_envs.find_simples_in_joined_envs in Hfind.
  destruct (types_of_simples_lookup _ _ _ _ _ Hfind) as [Hfound Hcovered].
  pose proof (proj2 (proj2 (proj2 Hholds)) x simples kind Hdefinition)
    as Hvector.
  pose proof (existential_definition_kind s x simples kind Hvalid Hdefinition)
    as Hkind.
  split.
  - destruct (HN x simples kind Hdefinition) as [index [simple Hsimple]].
    unfold Simples_in_joined_envs.find in Hsimple.
    rewrite <- (keyed_lookup_compact Index.t Index.equal index_equal_decides)
      in Hsimple.
    destruct (Hcovered index simple Hsimple) as [t [Ht _]].
    exists index, t. exact Ht.
  - intros index t Ht. rewrite Hkind. f_equal. symmetry.
    destruct (Hfound index t Ht) as [simple [Hsimple Hty]].
    rewrite (keyed_lookup_compact Index.t Index.equal index_equal_decides)
      in Hsimple.
    destruct (Hvector index simple Hsimple) as [use [Huse _]].
    destruct simple as [y | sym | c]; cbn [type_of_simple_at] in Hty.
    + change (Joined_envs.get_nth_joined_env (Join_env.joined_envs s) index)
        with (Join_env.joined_env s index) in Hty.
      rewrite Huse in Hty. cbn in Hty. exact (find_kind _ _ _ _ Hty).
    + change (Joined_envs.get_nth_joined_env (Join_env.joined_envs s) index)
        with (Join_env.joined_env s index) in Hty.
      rewrite Huse in Hty. cbn in Hty. exact (find_kind _ _ _ _ Hty).
    + injection Hty as <-. rewrite Hconst.
      destruct Hstate as [_ [_ Hdefinitions]].
      exact (Hdefinitions x simples kind index (Const c) use Hdefinition
               Hsimple Huse).
Qed.

Section Rounds_kinds.

Context (source : Typing_env.t) (jt : Join_env.n_way_join_type)
  (choices : Join_env.choices).

Hypothesis jt_sound :
  FlambdaJoinSpec.n_way_join_type_sound typing_env_invariant
    Join_env.joined_env Join_env.definitions_of_existentials
    kind_in_target_env (state_true source) state_kinded jt.

Hypothesis jt_keeps : join_of_types_keeps_kinds jt.

Lemma join_call_kinds (s s0 : Join_env.t) (var : variable)
    (types : Keyed.finite_map Index.t flambda_type)
    (joined : Or_unknown.t flambda_type) :
  state_true source s ->
  joined_envs_are typing_env_invariant s ->
  joined_envs_are typing_env_kinded s ->
  types_hold s var types ->
  state_kinded s ->
  jt s (Index.bindings types) = Some (joined, s0) ->
  ∀ ty : flambda_type,
    joined = Or_unknown.Known ty ->
    (∀ (index : Index.t) (t : flambda_type),
       Keyed.lookup Index.equal index types = Some t -> kind ty = kind t)
    ∧ type_kinded_in_target Join_env.joined_env kind_in_target_env s0 ty.
Proof.
  intros Htrue Hinvariant Hkinded [Hvar [Hfound _]] Hstate Hjoin.
  assert (Hscope :
            join_args_in_scope Join_env.joined_env type_in_scope s
              (Index.bindings types)).
  { intros index t Hin. apply bindings_lookup in Hin.
    destruct (Hfound _ _ Hin) as [use [Huse [Hscope _]]].
    exists use. split; [ exact Huse | exact Hscope ]. }
  assert (Hkinds :
            join_args_in_scope Join_env.joined_env type_kinded s
              (Index.bindings types)).
  { intros index t Hin. apply bindings_lookup in Hin.
    destruct (Hfound _ _ Hin) as [use [Huse [_ Hkind]]].
    exists use. split; [ exact Huse | exact Hkind ]. }
  destruct (jt_sound s s0 (Index.bindings types) joined Htrue Hinvariant
              Hkinded (bindings_distinct types) Hscope Hkinds Hjoin)
    as [_ [_ [Hknown Hkinded0]]].
  intros ty Hty. destruct (Hknown ty Hty) as [_ [Hkind _]].
  split; [ | exact (proj2 (Hkinded0 Hstate) ty Hty) ].
  intros index t Ht.
  exact (Hkind index t (proj2 (bindings_lookup types index t) Ht)).
Qed.

Lemma round_fold_kinds :
  ∀ (bindings : list (variable * Keyed.finite_map Index.t flambda_type))
    (s s' : Join_env.t) (relations relations' : Join_env.inverse_relations),
    (∀ (var : variable) (types : Keyed.finite_map Index.t flambda_type),
       In (var, types) bindings ->
       types_hold s var types ∧ types_of_kind s var types) ->
    state_true source s ->
    joined_envs_are typing_env_invariant s ->
    joined_envs_are typing_env_kinded s ->
    relations_hold s relations ->
    state_kinded s ->
    equations_kinded s ->
    relations_kinded s relations ->
    List.fold_left (round_step jt choices) bindings (Some (relations, s))
    = Some (relations', s') ->
    equations_kinded s' ∧ relations_kinded s' relations'.
Proof.
  induction bindings as [ | [var types] bindings IH];
    intros s s' relations relations' Htypes Htrue Hinvariant Hkinded
      Hrelations Hstate HX HR Hfold.
  - cbn in Hfold. injection Hfold as <- <-. split; assumption.
  - cbn [List.fold_left] in Hfold.
    destruct (round_step jt choices (Some (relations, s)) (var, types))
      as [[relations1 s1] | ] eqn:Hstep;
      [ | rewrite round_step_none in Hfold; discriminate ].
    destruct (Htypes var types (or_introl eq_refl))
      as [Hvar_types Hvar_kind].
    assert (Hsingle :
              ∀ (v : variable) (t : Keyed.finite_map Index.t flambda_type),
                In (v, t) [(var, types)] -> types_hold s v t).
    { intros v t [Heq | []]. injection Heq as <- <-. exact Hvar_types. }
    destruct (round_fold_holds source jt choices jt_sound [(var, types)] s s1
                relations relations1 Hsingle Htrue Hinvariant Hkinded
                Hrelations Hstep)
      as [Hlater1 [Hholds1 Hrelations1]].
    pose proof (round_fold_kinded source jt choices jt_sound [(var, types)] s
                  s1 relations relations1 Hsingle Htrue Hinvariant Hkinded
                  Hrelations Hstate Hstep) as Hstate1.
    assert (Hfirst : equations_kinded s1 ∧ relations_kinded s1 relations1).
    { unfold round_step in Hstep. cbn [fst snd] in Hstep.
      destruct (jt s (Index.bindings types)) as [[joined s0] | ] eqn:Hjoin;
        [ | discriminate ].
      destruct (join_call_holds source jt jt_sound s s0 var types joined Htrue
                  Hinvariant Hkinded Hvar_types Hjoin)
        as [Hlater0 [Hholds0 _]].
      pose proof (join_call_kinded source jt jt_sound s s0 var types joined
                    Htrue Hinvariant Hkinded Hvar_types Hstate Hjoin)
        as Hstate0.
      pose proof (join_call_kinds s s0 var types joined Htrue Hinvariant
                    Hkinded Hvar_types Hstate Hjoin) as Hknown.
      pose proof (jt_keeps s s0 _ joined Hjoin (proj1 Htrue) Hinvariant
                    Hkinded Hstate0 HX) as HX0.
      pose proof (proj2 (proj2 (proj1 (proj2 (proj2 Hlater0))))) as Hkinds0.
      assert (HR0 : relations_kinded s0 relations).
      { intros r names Hr. exact (Hkinds0 _ _ (HR r names Hr)). }
      destruct joined as [ty | ].
      - destruct (Hknown ty eq_refl) as [Hkind_ty Hkinded_ty].
        assert (Hvar0 : kind_in_target_env s0 var = Some (kind ty)).
        { destruct Hvar_kind as [[index [t Ht]] Hall].
          rewrite (Hkind_ty index t Ht).
          exact (Hkinds0 _ _ (Hall index t Ht)). }
        assert (Hadd :
                  ∀ (ty' : flambda_type)
                    (relations0 : Join_env.inverse_relations)
                    (s2 : Join_env.t),
                    (ty' = ty ∧ relations0 = relations)
                    ∨ Join_env.recover_inverse_relations relations
                        (Name_var var) ty
                      = Some (ty', relations0) ->
                    Join_env.add_equation_in_target_env s0 var ty' = Some s2
                    ->
                    equations_kinded s2 ∧ relations_kinded s2 relations0).
        { intros ty' relations0 s2 Hcase Hadd.
          pose proof (proj2 (proj2 (same_readings_grows _ _
                                      (add_equation_same_readings _ _ _ _
                                         Hadd)))) as Hkinds2.
          split.
          - apply (add_equation_keeps_kinds _ _ _ _ Hadd); [ | | exact HX0 ].
            + intros simple k Hin. left. apply Hkinded_ty.
              destruct Hcase as [[-> _] | Hrecover]; [ exact Hin | ].
              exact (recover_aliases _ _ _ _ _ Hrecover _ Hin).
            + destruct Hcase as [[-> _] | Hrecover]; [ exact Hvar0 | ].
              rewrite (recover_kind _ _ _ _ _ Hrecover). exact Hvar0.
          - intros r names Hr. apply Hkinds2.
            destruct Hcase as [[_ ->] | Hrecover];
              [ exact (HR0 r names Hr) | ].
            destruct (recover_relations_results _ _ _ _ _ r names Hrecover Hr)
              as [[names0 Hold] | Hnew].
            + exact (HR0 r names0 Hold).
            + exact (Hkinded_ty _ _ Hnew). }
        destruct (Join_env.recover_relations_on choices s0 var types).
        + destruct (Join_env.recover_inverse_relations relations
                      (Name_var var) ty)
            as [[ty' relations0] | ] eqn:Hrecover; [ | discriminate ].
          destruct (Join_env.add_equation_in_target_env s0 var ty')
            as [s2 | ] eqn:Hadd2; [ | discriminate ].
          injection Hstep as <- <-.
          exact (Hadd ty' relations0 s2 (or_intror eq_refl) Hadd2).
        + destruct (Join_env.add_equation_in_target_env s0 var ty)
            as [s2 | ] eqn:Hadd2; [ | discriminate ].
          injection Hstep as <- <-.
          exact (Hadd ty relations s2 (or_introl (conj eq_refl eq_refl))
                   Hadd2).
      - injection Hstep as <- <-. split; [ exact HX0 | exact HR0 ]. }
    destruct Hfirst as [HX1 HR1].
    apply (IH s1 s' relations1 relations');
      [ | exact (later_true source s s1 Hlater1 Hholds1
                   (proj2 (proj2 Htrue)))
        | exact (later_envs_invariant _ _ Hlater1 Hinvariant)
        | exact (later_envs_kinded _ _ Hlater1 Hkinded)
        | exact Hrelations1 | exact Hstate1 | exact HX1 | exact HR1
        | exact Hfold ].
    intros v t Hin. destruct (Htypes v t (or_intror Hin)) as [Hh Hk].
    split; [ exact (later_types_hold _ _ _ _ Hlater1 Hh) | ].
    exact (types_of_kind_grows _ _ _ _ (proj1 (proj2 (proj2 Hlater1))) Hk).
Qed.

Lemma round_kinds (s s' : Join_env.t)
    (equations : finite_map (Keyed.finite_map Index.t flambda_type))
    (relations relations' : Join_env.inverse_relations) :
  (∀ (var : variable) (types : Keyed.finite_map Index.t flambda_type),
     lookup var equations = Some types ->
     types_hold s var types ∧ types_of_kind s var types) ->
  state_true source s ->
  joined_envs_are typing_env_invariant s ->
  joined_envs_are typing_env_kinded s ->
  relations_hold s relations ->
  state_kinded s ->
  equations_kinded s ->
  relations_kinded s relations ->
  Join_env.n_way_join_round choices jt s equations relations
  = Some (relations', s') ->
  equations_kinded s' ∧ relations_kinded s' relations'.
Proof.
  intros Htypes Htrue Hinvariant Hkinded Hrelations Hstate HX HR Hround.
  apply (round_fold_kinds (compact equations) s s' relations relations');
    [ | exact Htrue | exact Hinvariant | exact Hkinded | exact Hrelations
      | exact Hstate | exact HX | exact HR | exact Hround ].
  intros var types Hin. apply Htypes. apply in_compact. exact Hin.
Qed.

Lemma delayed_equations_kinds :
  ∀ (fuel : nat) (s s' : Join_env.t)
    (equations : finite_map (Keyed.finite_map Index.t flambda_type))
    (relations relations' : Join_env.inverse_relations),
    (∀ (var : variable) (types : Keyed.finite_map Index.t flambda_type),
       lookup var equations = Some types ->
       types_hold s var types ∧ types_of_kind s var types) ->
    state_true source s ->
    joined_envs_are typing_env_invariant s ->
    joined_envs_are typing_env_kinded s ->
    relations_hold s relations ->
    state_kinded s ->
    equations_kinded s ->
    relations_kinded s relations ->
    Join_env.n_way_join_delayed_equations fuel choices jt s relations
      equations
    = Some (relations', s') ->
    equations_kinded s' ∧ relations_kinded s' relations'.
Proof.
  induction fuel as [ | fuel IH];
    intros s s' equations relations relations' Htypes Htrue Hinvariant Hkinded
      Hrelations Hstate HX HR Hloop; [ discriminate | ].
  cbn [Join_env.n_way_join_delayed_equations] in Hloop.
  destruct (Keyed.is_empty equations).
  - injection Hloop as <- <-. split; assumption.
  - destruct (Join_env.n_way_join_round choices jt s equations relations)
      as [[relations1 s1] | ] eqn:Hround; [ | discriminate ].
    assert (Hholds_types :
              ∀ (var : variable)
                (types : Keyed.finite_map Index.t flambda_type),
                lookup var equations = Some types -> types_hold s var types).
    { intros var types Hvar. exact (proj1 (Htypes var types Hvar)). }
    destruct (round_holds source jt choices jt_sound s s1 equations relations
                relations1 Hholds_types Htrue Hinvariant Hkinded Hrelations
                Hround)
      as [Hlater1 [Hholds1 Hrelations1]].
    pose proof (round_fold_kinded source jt choices jt_sound
                  (compact equations) s s1 relations relations1
                  (fun var types Hin =>
                     Hholds_types var types (proj1 (in_compact _ _ _) Hin))
                  Htrue Hinvariant Hkinded Hrelations Hstate Hround)
      as Hstate1.
    destruct (round_kinds s s1 equations relations relations1 Htypes Htrue
                Hinvariant Hkinded Hrelations Hstate HX HR Hround)
      as [HX1 HR1].
    assert (Htypes1 :
              ∀ (var : variable)
                (types : Keyed.finite_map Index.t flambda_type),
                lookup var (Join_env.new_equations_in_joined_envs s1 s)
                = Some types ->
                types_hold s1 var types ∧ types_of_kind s1 var types).
    { intros var types Hvar.
      unfold Join_env.new_equations_in_joined_envs in Hvar.
      rewrite lookup_filter_bindings in Hvar.
      destruct (lookup var (Join_env.types_in_joined_envs s1))
        as [types1 | ] eqn:Hlookup; [ | discriminate ].
      destruct (negb (is_bound var (Join_env.types_in_joined_envs s)));
        [ | discriminate ].
      injection Hvar as <-.
      split; [ exact (proj1 (proj2 Hholds1) var types1 Hlookup) | ].
      exact (proj1 (proj2 (proj2 HX1)) var types1 Hlookup). }
    exact (IH s1 s' _ relations1 relations' Htypes1
             (later_true source s s1 Hlater1 Hholds1 (proj2 (proj2 Htrue)))
             (later_envs_invariant _ _ Hlater1 Hinvariant)
             (later_envs_kinded _ _ Hlater1 Hkinded) Hrelations1 Hstate1 HX1
             HR1 Hloop).
Qed.

Lemma loop_kinds (p : Join_env.parameters) :
  expand_const_sound p ->
  expand_const_of_kind p ->
  ∀ (depth fuel : nat) (s s' : Join_env.t)
    (relations relations' : Join_env.inverse_relations)
    (existentials : finite_map (Simples_in_joined_envs.t * flambda_kind)),
    (∀ (x : variable) (definition : Simples_in_joined_envs.t * flambda_kind),
       lookup x existentials = Some definition ->
       lookup x (Join_env.definitions_of_existentials s) = Some definition) ->
    state_true source s ->
    joined_envs_are typing_env_invariant s ->
    joined_envs_are typing_env_kinded s ->
    relations_hold s relations ->
    state_kinded s ->
    equations_kinded s ->
    relations_kinded s relations ->
    Join_env.n_way_join_loop depth p fuel choices jt s relations existentials
    = Some (relations', s') ->
    equations_kinded s' ∧ relations_kinded s' relations'.
Proof.
  intros Hconst Hconst_kind.
  induction depth as [ | depth IH];
    intros fuel s s' relations relations' existentials Hdefinitions Htrue
      Hinvariant Hkinded Hrelations Hstate HX HR Hloop;
    cbn [Join_env.n_way_join_loop] in Hloop.
  - injection Hloop as <- <-. split; assumption.
  - destruct (Keyed.is_empty existentials).
    { injection Hloop as <- <-. split; assumption. }
    destruct (Join_env.equations_of_existentials p s (compact existentials))
      as [equations | ] eqn:Hequations; [ | discriminate ].
    destruct (Join_env.n_way_join_delayed_equations fuel choices jt s
                relations equations)
      as [[relations1 s1] | ] eqn:Hdelayed; [ | discriminate ].
    assert (Htypes :
              ∀ (var : variable)
                (types : Keyed.finite_map Index.t flambda_type),
                lookup var equations = Some types ->
                types_hold s var types ∧ types_of_kind s var types).
    { intros var types Hvar.
      destruct (equations_of_existentials_lookup p s _ _ Hequations _ _ Hvar)
        as [simples [kind [Hin Hfind]]].
      apply in_compact in Hin. split.
      - exact (existential_types_hold p s var simples kind types Hconst
                 (proj1 Htrue) (proj1 (proj2 Htrue)) Hinvariant Hkinded
                 (Hdefinitions _ _ Hin) Hfind).
      - exact (existential_types_of_kind p s var simples kind types
                 Hconst_kind (proj1 Htrue) (proj1 (proj2 Htrue)) Hstate
                 (proj2 (proj2 (proj2 (proj2 HX)))) (Hdefinitions _ _ Hin)
                 Hfind). }
    assert (Hholds_types :
              ∀ (var : variable)
                (types : Keyed.finite_map Index.t flambda_type),
                lookup var equations = Some types -> types_hold s var types).
    { intros var types Hvar. exact (proj1 (Htypes var types Hvar)). }
    destruct (delayed_equations_hold source jt choices jt_sound fuel s s1
                equations relations relations1 Hholds_types Htrue Hinvariant
                Hkinded Hrelations Hdelayed)
      as [Hlater1 [Hholds1 Hrelations1]].
    pose proof (delayed_equations_kinded source jt choices jt_sound fuel s s1
                  equations relations relations1 Hholds_types Htrue Hinvariant
                  Hkinded Hrelations Hstate Hdelayed) as Hstate1.
    destruct (delayed_equations_kinds fuel s s1 equations relations relations1
                Htypes Htrue Hinvariant Hkinded Hrelations Hstate HX HR
                Hdelayed) as [HX1 HR1].
    assert (Hdefinitions1 :
              ∀ (x : variable)
                (definition : Simples_in_joined_envs.t * flambda_kind),
                lookup x (Join_env.new_definitions_of_existentials s1 s)
                = Some definition ->
                lookup x (Join_env.definitions_of_existentials s1)
                = Some definition).
    { intros x definition Hx.
      unfold Join_env.new_definitions_of_existentials in Hx.
      rewrite lookup_filter_bindings in Hx.
      destruct (lookup x (Join_env.definitions_of_existentials s1))
        as [definition1 | ]; [ | discriminate ].
      destruct (negb (is_bound x (Join_env.definitions_of_existentials s)));
        [ | discriminate ].
      exact Hx. }
    exact (IH fuel s1 s' relations1 relations' _ Hdefinitions1
             (later_true source s s1 Hlater1 Hholds1 (proj2 (proj2 Htrue)))
             (later_envs_invariant _ _ Hlater1 Hinvariant)
             (later_envs_kinded _ _ Hlater1 Hkinded) Hrelations1 Hstate1 HX1
             HR1 Hloop).
Qed.

Lemma first_phases_kinds_lemma (p : Join_env.parameters) (fuel : nat)
    (avoid : list variable) (envs : Joined_envs.t) (s1 s2 s3 : Join_env.t)
    (relations2 relations3 : Join_env.inverse_relations) :
  fresh_avoids p ->
  parameters_type_eqb_sound p ->
  expand_const_sound p ->
  expand_const_of_kind p ->
  (∀ x : variable, knows source x -> x ∈ avoid) ->
  (∀ (index : Index.t) (use : Typing_env.t) (x : variable),
     Joined_envs.get_nth_joined_env envs index = Some use ->
     knows use x ->
     x ∈ avoid) ->
  (∀ (index : Index.t) (use : Typing_env.t),
     Joined_envs.get_nth_joined_env envs index = Some use ->
     typing_env_invariant use) ->
  (∀ (index : Index.t) (use : Typing_env.t),
     Joined_envs.get_nth_joined_env envs index = Some use ->
     typing_env_kinded use) ->
  levels_found envs ->
  (∀ (index : Index.t) (use : Typing_env.t) (x : variable)
     (kind : flambda_kind) (ty : flambda_type)
     (binding_time_and_mode : Binding_time.With_name_mode.t),
     Joined_envs.get_nth_joined_env envs index = Some use ->
     simple_of_kind use (Var x) kind ->
     Keyed.lookup name_eqb (Name_var x) (Typing_env.names_to_types source)
     = Some (ty, binding_time_and_mode) ->
     FlambdaTypeGrammar.kind ty = kind) ->
  Join_env.join_aliases_into_bindings p fuel
    (Join_env.create envs
       (Bindings_in_target_env.from_source_env avoid source))
    envs
  = Some s1 ->
  Join_env.n_way_join_delayed_equations fuel choices jt s1 []
    (Join_env.new_equations_in_joined_envs s1
       (Join_env.create envs
          (Bindings_in_target_env.from_source_env avoid source)))
  = Some (relations2, s2) ->
  Join_env.n_way_join_loop (Join_env.join_depth p) p fuel choices jt s2
    relations2
    (Join_env.new_definitions_of_existentials s2
       (Join_env.create envs
          (Bindings_in_target_env.from_source_env avoid source)))
  = Some (relations3, s3) ->
  equations_kinded s3 ∧ relations_kinded s3 relations3.
Proof.
  intros Hfresh Heqb Hconst Hconst_kind Hsource Huses Hinvariant Hkinded
    Hlevels Hkinds Haliases Hdelayed Hloop.
  set (s0 :=
         Join_env.create envs
           (Bindings_in_target_env.from_source_env avoid source)) in *.
  pose proof (created_state_true avoid source envs Hsource Huses) as Htrue0.
  fold s0 in Htrue0.
  pose proof (created_state_kinded avoid source envs Hkinds) as Hstate0.
  fold s0 in Hstate0.
  assert (Hknown0 :
            ∀ x : variable, exists_in_target s0 x = true -> knows source x).
  { intros x Hx. rewrite kind_in_target_env_exists in Hx.
    unfold s0, Join_env.create, Bindings_in_target_env.exists_in_target_env,
      Bindings_in_target_env.from_source_env in Hx.
    cbn in Hx. rewrite !orb_false_r in Hx. exact Hx. }
  assert (Hinvariant0 : joined_envs_are typing_env_invariant s0)
    by exact Hinvariant.
  assert (Hkinded0 : joined_envs_are typing_env_kinded s0) by exact Hkinded.
  destruct (join_aliases_true p fuel source s0 s1 Hfresh Heqb Htrue0 Hstate0
              Hinvariant Hkinded Hlevels Hknown0 Haliases)
    as [Htrue1 Hgrows1].
  pose proof (join_aliases_kinded p fuel s0 s1 Hfresh Heqb (proj1 Htrue0)
                Hstate0 Hinvariant0 Hkinded0 (proj1 (proj2 Htrue0)) Hlevels
                Hknown0 Haliases) as Hstate1.
  assert (Hlater1 : later s0 s1).
  { split; [ exact (proj1 Htrue0) | ]. split; [ exact (proj1 Htrue1) | ].
    split; [ exact Hgrows1 | ].
    rewrite (proj2 (proj2 Htrue1)). symmetry. exact (proj2 (proj2 Htrue0)). }
  assert (HX0 : equations_kinded s0).
  { split; [ intros x ty Hx; discriminate | ].
    split; [ intros x ty Hx; discriminate | ].
    split; [ intros x types Hx; discriminate | ].
    split; [ intros e x Hx; destruct Hx | ].
    intros x simples kind Hx. discriminate. }
  assert (HX1 : equations_kinded s1).
  { unfold Join_env.join_aliases_into_bindings in Haliases.
    refine (aliases_fold_keeps_kinds p fuel envs _ s0 Hfresh Hinvariant0
              Hkinded0 _ _ s0 s1 _ (steps_refl s0 (proj1 Htrue0)) Haliases
              Hstate1 HX0).
    - intros x Hx. rewrite <- kind_in_target_env_exists in Hx.
      rewrite (proj2 (proj2 Htrue0)). exact (Hknown0 x Hx).
    - intros x _ Hx. rewrite <- kind_in_target_env_exists in Hx.
      pose proof (Hknown0 x Hx) as Hknows.
      apply mem_var_lookup in Hknows. destruct Hknows as [ty [btm Hlookup]].
      exists (kind ty). split.
      + unfold kind_in_target_env.
        change (Bindings_in_target_env.source_env (Join_env.bindings s0))
          with source.
        rewrite Hlookup. reflexivity.
      + intros index t Ht. rewrite types_of_name_lookup in Ht.
        destruct (Keyed.lookup Index.equal index envs)
          as [[use equations] | ] eqn:Hindex; [ | discriminate ].
        destruct (Hlevels _ _ _ _ _ Hindex Ht) as [btm' Hfound].
        assert (Huse : Join_env.joined_env s0 index = Some use).
        { apply joined_env_lookup. exists equations. exact Hindex. }
        split.
        * symmetry.
          apply (proj1 Hstate0 index use x (kind t) ty btm Huse);
            [ | exact Hlookup ].
          apply (simple_of_kind_name_iff use (Name_var x)). left.
          exists t, btm'. split; [ exact Hfound | reflexivity ].
        * exists use. split; [ exact Huse | ].
          exact (proj1 (Hkinded _ _ Huse) _ _ _ Hfound). }
  assert (Htypes1 :
            ∀ (var : variable)
              (types : Keyed.finite_map Index.t flambda_type),
              lookup var (Join_env.new_equations_in_joined_envs s1 s0)
              = Some types ->
              types_hold s1 var types ∧ types_of_kind s1 var types).
  { intros var types Hvar.
    unfold Join_env.new_equations_in_joined_envs in Hvar.
    rewrite lookup_filter_bindings in Hvar.
    destruct (lookup var (Join_env.types_in_joined_envs s1))
      as [types1 | ] eqn:Hlookup; [ | discriminate ].
    destruct (negb (is_bound var (Join_env.types_in_joined_envs s0)));
      [ | discriminate ].
    injection Hvar as <-.
    split; [ exact (proj1 (proj2 (proj1 (proj2 Htrue1))) var types1 Hlookup)
           | ].
    exact (proj1 (proj2 (proj2 HX1)) var types1 Hlookup). }
  assert (Hholds_types1 :
            ∀ (var : variable)
              (types : Keyed.finite_map Index.t flambda_type),
              lookup var (Join_env.new_equations_in_joined_envs s1 s0)
              = Some types ->
              types_hold s1 var types).
  { intros var types Hvar. exact (proj1 (Htypes1 var types Hvar)). }
  assert (Hrelations1 : relations_hold s1 []).
  { intros r relation n [recorded [Hlookup _]]. discriminate. }
  assert (HR1 : relations_kinded s1 []).
  { intros r names Hr. discriminate. }
  pose proof (later_envs_invariant _ _ Hlater1 Hinvariant0) as Hinvariant1.
  pose proof (later_envs_kinded _ _ Hlater1 Hkinded0) as Hkinded1.
  destruct (delayed_equations_hold source jt choices jt_sound fuel s1 s2 _ []
              relations2 Hholds_types1 Htrue1 Hinvariant1 Hkinded1
              Hrelations1 Hdelayed)
    as [Hlater2 [Hholds2 Hrelations2]].
  pose proof (delayed_equations_kinded source jt choices jt_sound fuel s1 s2 _
                [] relations2 Hholds_types1 Htrue1 Hinvariant1 Hkinded1
                Hrelations1 Hstate1 Hdelayed) as Hstate2.
  destruct (delayed_equations_kinds fuel s1 s2 _ [] relations2 Htypes1 Htrue1
              Hinvariant1 Hkinded1 Hrelations1 Hstate1 HX1 HR1 Hdelayed)
    as [HX2 HR2].
  pose proof (later_trans _ _ _ Hlater1 Hlater2) as Hlater02.
  assert (Hdefinitions2 :
            ∀ (x : variable)
              (definition : Simples_in_joined_envs.t * flambda_kind),
              lookup x (Join_env.new_definitions_of_existentials s2 s0)
              = Some definition ->
              lookup x (Join_env.definitions_of_existentials s2)
              = Some definition).
  { intros x definition Hx.
    unfold Join_env.new_definitions_of_existentials in Hx.
    rewrite lookup_filter_bindings in Hx.
    destruct (lookup x (Join_env.definitions_of_existentials s2))
      as [definition2 | ]; [ | discriminate ].
    destruct (negb (is_bound x (Join_env.definitions_of_existentials s0)));
      [ | discriminate ].
    exact Hx. }
  exact (loop_kinds p Hconst Hconst_kind (Join_env.join_depth p) fuel s2 s3
           relations2 relations3 _ Hdefinitions2
           (later_true source s1 s2 Hlater2 Hholds2 (proj2 (proj2 Htrue1)))
           (later_envs_invariant _ _ Hlater02 Hinvariant0)
           (later_envs_kinded _ _ Hlater02 Hkinded0) Hrelations2 Hstate2 HX2
           HR2 Hloop).
Qed.

End Rounds_kinds.

Theorem first_phases_kinds : first_phases_kinds_statement.
Proof.
  intros source jt choices p fuel avoid envs s1 s2 s3 relations2 relations3
    Hjt Hkeeps Hfresh Heqb Hconst Hsource Huses Hinvariant Hkinded Hlevels
    Hkinds Haliases Hdelayed Hloop.
  exact (first_phases_kinds_lemma source jt choices Hjt Hkeeps p fuel avoid
           envs s1 s2 s3 relations2 relations3 Hfresh Heqb Hconst
           (expand_const_sound_of_kind p Hconst) Hsource Huses Hinvariant
           Hkinded Hlevels Hkinds Haliases Hdelayed Hloop).
Qed.

Theorem scoped_equations_kinded : scoped_equations_kinded_statement.
Proof.
  intros s Hvalid Hstate Hinvariant Henvs Hscope Hreach x ty Hx simple kind
    Hin.
  destruct (Hreach x ty Hx simple kind Hin) as [H | [index [use [Huse H]]]];
    [ exact H | ].
  apply (simple_kind_in_target_of_use s index use simple kind Hvalid Hstate
           Hinvariant Henvs Huse H).
  intros y Hy. apply (Hscope x ty Hx).
  exact (proj1 aliases_free_variables ty simple kind y Hin Hy).
Qed.

(* ======================================================================== *)
(* The last phase                                                           *)
(* ======================================================================== *)

Lemma kind_of_unknown (k : flambda_kind) : kind (unknown k) = k.
Proof. destruct k as [ | [] | | ]; reflexivity. Qed.

Lemma aliases_of_unknown (k : flambda_kind) :
  aliases_with_kinds (unknown k) = [].
Proof. destruct k as [ | [] | | ]; reflexivity. Qed.

Section Define_kinds.

Context {meet_env meet_expanded_head : Type}
  (ops : Join_env.meet_env_functions meet_env meet_expanded_head)
  (s : Join_env.t) (tenv_of_aliases : Typing_env.t)
  (unique : variable -> bool).

Hypothesis s_valid : state_valid s.
Hypothesis aliases_kinds : aliases_of_kinds s.

(* A variable and a type: the variable is of the kind of the type, and
   the aliases of the type are of their kinds, in the target. *)
Definition typed (x : variable) (ty : flambda_type) : Prop :=
  kind_in_target_env s x = Some (kind ty)
  ∧ type_kinded_in_target Join_env.joined_env kind_in_target_env s ty.

Definition kinds_invariant (acc : define_acc meet_env) : Prop :=
  let '(_, to_expand, equations, relations, _) := acc in
  (∀ (x : variable) (ty : flambda_type),
     lookup x equations = Some ty -> typed x ty)
  ∧ (∀ (e : variable) (ty : flambda_type),
       lookup e to_expand = Some ty -> typed e ty)
  ∧ ∀ (r : variable) (names : list (Relation.t * name)),
      lookup r relations = Some names ->
      kind_in_target_env s r = Some (Naked_number Kind_naked_immediate).

Lemma define_step_kinds (acc acc' : define_acc meet_env)
    (binding : variable * flambda_kind) :
  binding
  ∈ Bindings_in_target_env.existential_variables_in_order (Join_env.bindings s)
  ->
  kinds_invariant acc ->
  define_step ops s tenv_of_aliases unique (Some acc) binding = Some acc' ->
  kinds_invariant acc'.
Proof.
  intros Hbinding Hinv Hstep.
  destruct acc as [[[[menv to_expand] equations] relations] definitions].
  destruct binding as [var var_kind]. cbn [define_step fst snd] in Hstep.
  destruct Hinv as [Hequations [Hexpand Hrelations]].
  assert (Hvar : kind_in_target_env s var = Some var_kind).
  { pose proof (in_order_lookup _ _ _ Hbinding) as Hlookup.
    assert (Hex : is_existential s var).
    { apply is_bound_lookup. exists var_kind. exact Hlookup. }
    destruct s_valid as [_ [Hnew _]].
    destruct (Hnew var Hex) as [Hunknown _].
    unfold kind_in_target_env.
    unfold knows, Typing_env.mem, Join_env.source_env in Hunknown.
    destruct (Keyed.lookup name_eqb (Name_var var)
                (Typing_env.names_to_types
                   (Bindings_in_target_env.source_env (Join_env.bindings s))))
      as [[ty time] | ]; [ exfalso; apply Hunknown; reflexivity | ].
    rewrite Hlookup. reflexivity. }
  destruct (Join_env.choose_alias tenv_of_aliases _) as [[canon_var | ] | ]
    eqn:Hchoose; [ | | discriminate ].
  - destruct (Join_env.move_equation equations var canon_var)
      as [equations1 | ] eqn:Hmove; [ | discriminate ].
    injection Hstep as <-.
    pose proof (choose_alias_in _ _ _ Hchoose) as Hcanon_in.
    assert (Halias_kind :
              ∀ x : variable,
                x
                ∈ Aliases_of_existentials.aliases_of_existential_var
                    (Join_env.aliases_of_existentials s) var ->
                kind_in_target_env s x = Some var_kind).
    { intros x Hx. destruct (aliases_kinds var x Hx) as [k [Hk Hk']].
      rewrite Hvar in Hk'. injection Hk' as <-. exact Hk. }
    pose proof (Halias_kind canon_var Hcanon_in) as Hcanon.
    assert (Htyped_alias :
              ∀ x : variable,
                kind_in_target_env s x = Some var_kind ->
                typed x (alias_type_of var_kind (Var canon_var))).
    { intros x Hx. split; [ rewrite kind_of_alias_type; exact Hx | ].
      intros simple k Hin. rewrite aliases_alias_type_of in Hin.
      destruct Hin as [Heq | []]. injection Heq as <- <-. exact Hcanon. }
    assert (Hequations1 :
              ∀ (x : variable) (ty : flambda_type),
                lookup x equations1 = Some ty -> typed x ty).
    { unfold Join_env.move_equation in Hmove.
      destruct (is_bound canon_var equations); [ discriminate | ].
      destruct (lookup var equations) as [ty_var | ] eqn:Hlookup_var.
      - injection Hmove as <-. intros x ty Hx. rewrite lookup_cons in Hx.
        destruct (Pos.eqb x canon_var) eqn:Heq.
        + apply Pos.eqb_eq in Heq. subst x. injection Hx as <-.
          destruct (Hequations var ty_var Hlookup_var) as [Hk Ha].
          split; [ | exact Ha ]. rewrite Hvar in Hk. injection Hk as Hk.
          rewrite <- Hk. exact Hcanon.
        + rewrite lookup_unbind in Hx.
          destruct (Pos.eqb x var); [ discriminate | ].
          exact (Hequations x ty Hx).
      - injection Hmove as <-. exact Hequations. }
    split; [ | split ].
    + intros x ty Hx.
      destruct (lookup_fold_cons _ _ _ _ _ Hx) as [[Hin ->] | Hlookup].
      * apply filter_In in Hin. destruct Hin as [Hin _].
        exact (Htyped_alias x (Halias_kind x Hin)).
      * exact (Hequations1 x ty Hlookup).
    + intros e ty He. rewrite lookup_cons in He.
      destruct (Pos.eqb e var) eqn:Heq.
      * apply Pos.eqb_eq in Heq. subst e. injection He as <-.
        exact (Htyped_alias var Hvar).
      * exact (Hexpand e ty He).
    + intros r names Hr. unfold Join_env.move_inverse_relation in Hr.
      destruct (lookup var relations) as [names_var | ] eqn:Hlookup_var.
      * rewrite lookup_cons in Hr. destruct (Pos.eqb r canon_var) eqn:Heq.
        -- apply Pos.eqb_eq in Heq. subst r.
           pose proof (Hrelations var names_var Hlookup_var) as Hk.
           rewrite Hvar in Hk. injection Hk as Hk. rewrite <- Hk.
           exact Hcanon.
        -- rewrite lookup_unbind in Hr.
           destruct (Pos.eqb r var); [ discriminate | ].
           exact (Hrelations r names Hr).
      * exact (Hrelations r names Hr).
  - destruct (unique var).
    + injection Hstep as <-. split; [ | split; [ | exact Hrelations ] ].
      * intros x ty Hx. rewrite lookup_unbind in Hx.
        destruct (Pos.eqb x var); [ discriminate | ].
        exact (Hequations x ty Hx).
      * intros e ty He. rewrite lookup_cons in He.
        destruct (Pos.eqb e var) eqn:Heq; [ | exact (Hexpand e ty He) ].
        apply Pos.eqb_eq in Heq. subst e. injection He as <-.
        destruct (lookup var equations) as [ty_var | ] eqn:Hlookup_var.
        -- exact (Hequations var ty_var Hlookup_var).
        -- split; [ rewrite kind_of_unknown; exact Hvar | ].
           intros simple k Hin. rewrite aliases_of_unknown in Hin.
           destruct Hin.
    + injection Hstep as <-.
      split; [ exact Hequations | split; [ exact Hexpand | exact Hrelations ] ].
Qed.

Lemma define_fold_kinds (acc acc' : define_acc meet_env) :
  kinds_invariant acc ->
  List.fold_left (define_step ops s tenv_of_aliases unique)
    (Bindings_in_target_env.existential_variables_in_order
       (Join_env.bindings s))
    (Some acc)
  = Some acc' ->
  kinds_invariant acc'.
Proof.
  apply (fold_left_invariant (define_step ops s tenv_of_aliases unique)
           kinds_invariant (fun b => eq_refl)).
  intros a a' b Hb Ha Hstep.
  exact (define_step_kinds a a' b Hb Ha Hstep).
Qed.

(* What is supposed of the aliases of a type that is projected, from
   the kinds of the types put. *)
Lemma aliases_before_of (to_project : list variable)
    (expand : variable -> option flambda_type) (t : flambda_type) :
  (∀ (e : variable) (ty : flambda_type),
     expand e = Some ty -> kind_in_target_env s e = Some (kind ty)) ->
  type_kinded_in_target Join_env.joined_env kind_in_target_env s t ->
  ∀ alias : simple * flambda_kind,
    alias ∈ aliases_with_kinds t ->
    alias_before to_project expand
      (simple_of_kind_in_target Join_env.joined_env kind_in_target_env s) alias.
Proof.
  intros Hexpand Ht [simple k] Hin. unfold alias_before. cbn [fst snd].
  pose proof (Ht simple k Hin) as HK.
  destruct simple as [x | sym | c]; [ | exact HK | exact HK ].
  destruct (among x to_project); [ | exact HK ].
  intros ty1 H1. pose proof (Hexpand x ty1 H1) as Hk1.
  change (kind_in_target_env s x = Some k) in HK.
  rewrite Hk1 in HK. injection HK as HK. exact HK.
Qed.

Lemma expand_variable_kinds (to_expand : finite_map flambda_type) :
  (∀ (e : variable) (ty : flambda_type),
     lookup e to_expand = Some ty -> typed e ty) ->
  ∀ (fuel : nat) (e : variable) (ty : flambda_type),
    Join_env.expand_variable fuel to_expand e = Some ty -> typed e ty.
Proof.
  intros Hbase.
  induction fuel as [ | fuel IH]; intros e ty Hexpand; [ discriminate | ].
  cbn [Join_env.expand_variable] in Hexpand.
  destruct (lookup e to_expand) as [ty0 | ] eqn:Hlookup; [ | discriminate ].
  cbn in Hexpand.
  destruct (Hbase e ty0 Hlookup) as [Hkind0 Haliases0].
  destruct (get_alias_opt ty0) as [[y | sym | c] | ] eqn:Halias.
  - destruct (among y (domain to_expand)) eqn:Hy.
    + destruct (IH y ty Hexpand) as [Hkind_y Haliases_y].
      split; [ | exact Haliases_y ].
      pose proof (Haliases0 _ _ (get_alias_in_aliases _ _ Halias)) as Hy_kind.
      change (kind_in_target_env s y = Some (kind ty0)) in Hy_kind.
      rewrite Hkind_y in Hy_kind. injection Hy_kind as Heq.
      rewrite Heq. exact Hkind0.
    + injection Hexpand as <-. split; assumption.
  - injection Hexpand as <-. split; assumption.
  - injection Hexpand as <-. split; assumption.
  - split.
    + rewrite (project_variables_out_kind _ _ _ _ Hexpand). exact Hkind0.
    + destruct (project_variables_out_kinded (domain to_expand)
                  (Join_env.expand_variable fuel to_expand)
                  (simple_of_kind_in_target Join_env.joined_env
                     kind_in_target_env s))
        as [Htype _].
      * intros e' ty' simple k _ He' Hin.
        exact (proj2 (IH e' ty' He') simple k Hin).
      * intros simple k Hin. apply (Htype ty0 ty Hexpand); [ | exact Hin ].
        apply aliases_before_of; [ | exact Haliases0 ].
        intros e' ty' He'. exact (proj1 (IH e' ty' He')).
Qed.

End Define_kinds.

Theorem last_phase_adds_kinded : last_phase_adds_kinded_statement.
Proof.
  intros tenv fuel s relations kept equations relation_equations definitions
    Hvalid Hkinded Hkinds Haliases Hrelations Hadds n ty Hin.
  destruct (last_phase_adds_parts tenv fuel s relations _ definitions Hadds)
    as [r1 [to_expand [equations1 [relations1 [kept0 [equations0
          [relation_equations0
             [Hfold [_ [_ [_ [_ [Hequations [Hrelation_equations
                                              Heq]]]]]]]]]]]]]].
  injection Heq as _ -> ->.
  assert (Hstart :
            kinds_invariant s
              ({| recorded_variables :=
                    Bindings_in_target_env.imported_variables_in_order
                      (Join_env.bindings s);
                  recorded_extensions := [] |},
               [], Join_env.types_in_target_env s, relations,
               Join_env.definitions_of_existentials s)).
  { split; [ | split ].
    - intros x ty0 Hx. split; [ exact (Hkinds x ty0 Hx) | ].
      exact (Hkinded x ty0 Hx).
    - intros e ty0 He. discriminate.
    - exact Hrelations. }
  destruct (define_fold_kinds (recording tenv) s tenv (unique_of s relations)
              Hvalid Haliases _ _ Hstart Hfold)
    as [Hequations_inv [Hexpand_inv Hrelations_inv]].
  pose proof (expand_variable_kinds s to_expand Hexpand_inv) as Hexp.
  destruct (project_variables_out_kinded (domain to_expand)
              (Join_env.expand_variable fuel to_expand)
              (simple_of_kind_in_target Join_env.joined_env
                 kind_in_target_env s)
              (fun e ty0 simple k _ He Hin0 =>
                 proj2 (Hexp fuel e ty0 He) simple k Hin0))
    as [Htype _].
  assert (Hbefore :
            ∀ t : flambda_type,
              type_kinded_in_target Join_env.joined_env kind_in_target_env s
                t ->
              ∀ alias : simple * flambda_kind,
                alias ∈ aliases_with_kinds t ->
                alias_before (domain to_expand)
                  (Join_env.expand_variable fuel to_expand)
                  (simple_of_kind_in_target Join_env.joined_env
                     kind_in_target_env s)
                  alias).
  { intros t Ht. apply aliases_before_of; [ | exact Ht ].
    intros e ty0 He. exact (proj1 (Hexp fuel e ty0 He)). }
  apply in_app_or in Hin. destruct Hin as [Hin | Hin].
  - destruct (traverse_in _ _ _ Hequations (n, ty) Hin)
      as [[x ty0] [Hbinding Hf]].
    cbn [fst snd] in Hf.
    destruct (project_variables_out (domain to_expand)
                (Join_env.expand_variable fuel to_expand) ty0) as [ty1 | ]
      eqn:Hty; [ | discriminate ].
    injection Hf as <- <-. apply in_compact in Hbinding.
    destruct (Hequations_inv x ty0 Hbinding) as [Hkind0 Haliases0].
    split.
    + exact (Htype ty0 ty1 Hty (Hbefore ty0 Haliases0)).
    + exists x. split; [ reflexivity | ].
      rewrite (project_variables_out_kind _ _ _ _ Hty). exact Hkind0.
  - destruct (traverse_in _ _ _ Hrelation_equations (n, ty) Hin)
      as [[r names] [Hbinding Hf]].
    cbn [fst snd] in Hf.
    match type of Hf with
    | context [project_variables_out ?to_project ?expand ?ty0] =>
      destruct (project_variables_out to_project expand ty0) as [ty1 | ]
        eqn:Hty; [ | discriminate ]
    end.
    injection Hf as <- <-. apply in_compact in Hbinding.
    split.
    + refine (Htype _ ty1 Hty (Hbefore _ _)). intros simple k [].
    + exists r. split; [ reflexivity | ].
      rewrite (project_variables_out_kind _ _ _ _ Hty).
      exact (Hrelations_inv r names Hbinding).
Qed.

(* The definitions, through the fold of the last phase. *)
Definition definitions_kinded_invariant {meet_env : Type} (s : Join_env.t)
    (acc : define_acc meet_env) : Prop :=
  let '(_, _, _, _, definitions) := acc in
  ∀ (x : variable) (definition : Simples_in_joined_envs.t * flambda_kind),
    lookup x definitions = Some definition ->
    kind_in_target_env s x = Some (snd definition)
    ∧ ∃ e : variable,
        lookup e (Join_env.definitions_of_existentials s) = Some definition.

Lemma define_step_definitions_kinded {meet_env meet_expanded_head : Type}
    (ops : Join_env.meet_env_functions meet_env meet_expanded_head)
    (s : Join_env.t) (tenv_of_aliases : Typing_env.t)
    (unique : variable -> bool) (acc acc' : define_acc meet_env)
    (binding : variable * flambda_kind) :
  aliases_of_kinds s ->
  definitions_kinded_invariant s acc ->
  define_step ops s tenv_of_aliases unique (Some acc) binding = Some acc' ->
  definitions_kinded_invariant s acc'.
Proof.
  intros Hkinds Hinv Hstep.
  destruct acc as [[[[menv to_expand] equations] relations] definitions].
  destruct binding as [var var_kind]. cbn [define_step fst snd] in Hstep.
  cbn [definitions_kinded_invariant] in Hinv.
  assert (Hremoved :
            ∀ (x : variable)
              (definition : Simples_in_joined_envs.t * flambda_kind),
              lookup x (unbind var definitions) = Some definition ->
              kind_in_target_env s x = Some (snd definition)
              ∧ ∃ e : variable,
                  lookup e (Join_env.definitions_of_existentials s)
                  = Some definition).
  { intros x d Hx. rewrite lookup_unbind in Hx.
    destruct (Pos.eqb x var); [ discriminate | ]. exact (Hinv x d Hx). }
  destruct (Join_env.choose_alias tenv_of_aliases _) as [[canon_var | ] | ]
    eqn:Hchoose; [ | | discriminate ].
  - destruct (Join_env.move_equation equations var canon_var)
      as [equations1 | ]; [ | discriminate ].
    injection Hstep as <-.
    pose proof (choose_alias_in _ _ _ Hchoose) as Hcanon_in.
    intros x d Hx. unfold Join_env.move_definition in Hx.
    destruct (lookup var definitions) as [defn | ] eqn:Hvar_def;
      [ | exact (Hinv x d Hx) ].
    rewrite lookup_cons in Hx.
    destruct (Pos.eqb x canon_var) eqn:Heq; [ | exact (Hremoved x d Hx) ].
    apply Pos.eqb_eq in Heq. subst x. injection Hx as <-.
    destruct (Hinv var defn Hvar_def) as [Hkind_var Horigin].
    split; [ | exact Horigin ].
    destruct (Hkinds var canon_var Hcanon_in) as [kind [Hk1 Hk2]].
    rewrite Hk1, <- Hk2. exact Hkind_var.
  - destruct (unique var); injection Hstep as <-;
      [ exact Hremoved | exact Hinv ].
Qed.

Theorem last_phase_adds_definitions_kinded
  : last_phase_adds_definitions_kinded_statement.
Proof.
  intros tenv fuel source s relations kept equations relation_equations
    definitions Htrue Hkinds Hadds x definition Hx.
  destruct (last_phase_adds_parts tenv fuel s relations _ definitions Hadds)
    as [r1 [to_expand [equations1 [relations1 [kept' [equations'
          [relation_equations' [Hfold _]]]]]]]].
  assert (Hinv :
            definitions_kinded_invariant s
              (r1, to_expand, equations1, relations1, definitions)).
  { refine (fold_left_invariant
              (define_step (recording tenv) s tenv (unique_of s relations))
              (definitions_kinded_invariant s) (fun b => eq_refl) _ _ _ _ _
              Hfold).
    - intros a a' b _ Ha Hstep.
      exact (define_step_definitions_kinded (recording tenv) s tenv
               (unique_of s relations) a a' b Hkinds Ha Hstep).
    - intros y d Hy.
      destruct Htrue as [[_ [Hnew [[Hrecorded _] _]]] _].
      destruct d as [simples k].
      pose proof (Hrecorded y simples k Hy) as He.
      split; [ | exists y; exact Hy ]. cbn [snd].
      unfold kind_in_target_env.
      destruct (Keyed.lookup name_eqb (Name_var y)
                  (Typing_env.names_to_types
                     (Bindings_in_target_env.source_env
                        (Join_env.bindings s)))) as [[ty btm] | ]
        eqn:Hsource; [ exfalso | rewrite He; reflexivity ].
      apply (proj1 (Hnew y (proj2 (is_bound_lookup _ _)
                               (ex_intro _ k He)))).
      apply bound_knows. unfold Keyed.is_bound, Join_env.source_env.
      rewrite Hsource. reflexivity. }
  exact (Hinv x definition Hx).
Qed.
