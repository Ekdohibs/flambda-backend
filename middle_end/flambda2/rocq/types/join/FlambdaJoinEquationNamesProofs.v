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


(* The proofs of the statements of FlambdaJoinEquationNamesSpec.v.
   This file needs no review.

   As for the kinds of the aliases (FlambdaJoinKindsProofs.v): each
   function of the join of environments keeps
   [equation_names_kinded], then the rounds, which follow those of
   FlambdaJoinStateProofs.v, and the last phase. *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaMapProofs FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeAliasKindsSpec
  FlambdaTypeAliasKindsProofs FlambdaTypeProject FlambdaBindingTime
  FlambdaTypingEnv FlambdaTypingEnvSpec FlambdaTypingEnvProofs
  FlambdaTypingEnvInvariantSpec FlambdaTypingEnvInvariantProofs
  FlambdaTypingEnvKindedSpec FlambdaTypingEnvKindedProofs FlambdaJoinEnv
  FlambdaJoinSpec FlambdaJoinEnvSpec FlambdaJoinEnvProofs
  FlambdaJoinStateSpec FlambdaJoinStateProofs FlambdaJoinDefineSpec
  FlambdaJoinDefineProofs FlambdaJoinFitsSpec FlambdaJoinFitsProofs
  FlambdaJoinAddsSpec FlambdaJoinAddsProofs FlambdaJoinKindsSpec
  FlambdaJoinKindsProofs FlambdaJoinTypes FlambdaJoinTypesSpec
  FlambdaJoinTypesKeepsSpec FlambdaJoinTypesKeepsProofs
  FlambdaJoinTypesWithEnvSpec FlambdaJoinMeet FlambdaJoinEquationNamesSpec.
From OxCamlFlambda2Types Require FlambdaExpandHead FlambdaExpandHeadSpec
  FlambdaExpandHeadProofs.
Import ListNotations.

Abbreviation envs_equations_kinded :=
  (joined_envs_are types_equations_kinded).

Abbreviation of_kind_in_target :=
  (simple_of_kind_in_target Join_env.joined_env kind_in_target_env).

(* ======================================================================== *)
(* The functions of the join of environments                                *)
(* ======================================================================== *)

Lemma grows_envs_equations_kinded (s s' : Join_env.t) :
  grows s s' -> envs_equations_kinded s -> envs_equations_kinded s'.
Proof.
  intros [Henvs _] H index use Huse. rewrite Henvs in Huse.
  exact (H _ _ Huse).
Qed.

(* The same types, in a state that has grown: all is kept. *)
Lemma names_grow (s s' : Join_env.t) :
  grows s s' ->
  Join_env.types_in_target_env s' = Join_env.types_in_target_env s ->
  Join_env.types_in_joined_envs s' = Join_env.types_in_joined_envs s ->
  equation_names_kinded s ->
  equation_names_kinded s'.
Proof.
  intros Hgrows Htarget Hjoined [H1 H2]. split.
  - intros x ty Hx n kind Hin. rewrite Htarget in Hx.
    exact (reach_grows _ _ _ _ Hgrows (H1 x ty Hx n kind Hin)).
  - intros x types Hx index t Ht. rewrite Hjoined in Hx.
    destruct (H2 x types Hx index t Ht) as [use [Huse Hk]].
    exists use. split; [ rewrite (proj1 Hgrows); exact Huse | exact Hk ].
Qed.

Lemma existential_keeps_names (p : Join_env.parameters) (s s' : Join_env.t)
    (canonicals : Simples_in_joined_envs.t) (kind : flambda_kind)
    (e : variable) :
  fresh_avoids p ->
  state_valid s ->
  Join_env.existential_for_these_canonicals p s canonicals kind = (e, s') ->
  equation_names_kinded s ->
  equation_names_kinded s'.
Proof.
  intros Hfresh Hvalid Hmake H.
  destruct (existential_keeps _ _ _ _ _ _ Hfresh Hvalid Hmake)
    as [_ [Hgrows _]].
  destruct (existential_scope_grows _ _ _ _ _ _ Hmake) as [_ [Htypes _]].
  apply (names_grow s s' Hgrows Htypes); [ | exact H ].
  unfold Join_env.existential_for_these_canonicals in Hmake.
  destruct (Bindings_in_target_env.existential_for_these_simples
              (Join_env.fresh p) (Join_env.bindings s) canonicals kind)
    as [x0 b0].
  injection Hmake as _ <-. reflexivity.
Qed.

Lemma add_equation_keeps_names (s s' : Join_env.t) (x : variable)
    (ty : flambda_type) :
  Join_env.add_equation_in_target_env s x ty = Some s' ->
  equation_names_in_reach s ty ->
  equation_names_kinded s ->
  equation_names_kinded s'.
Proof.
  intros Hadd Hty [H1 H2].
  pose proof (same_readings_grows _ _
                (add_equation_same_readings _ _ _ _ Hadd)) as Hgrows.
  unfold Join_env.add_equation_in_target_env in Hadd.
  destruct (is_bound x (Join_env.types_in_target_env s)); [ discriminate | ].
  injection Hadd as <-. split.
  - intros x0 ty0 Hx0 n kind Hin. cbn [Join_env.types_in_target_env] in Hx0.
    rewrite lookup_cons in Hx0. apply (reach_grows _ _ _ _ Hgrows).
    destruct (Pos.eqb x0 x).
    + injection Hx0 as <-. exact (Hty n kind Hin).
    + exact (H1 x0 ty0 Hx0 n kind Hin).
  - intros x0 types Hx0 index t Ht. exact (H2 x0 types Hx0 index t Ht).
Qed.

Lemma alias_no_equation_names (s : Join_env.t) (kind : flambda_kind)
    (simple : simple) :
  equation_names_in_reach s (alias_type_of kind simple).
Proof.
  intros n k Hin.
  rewrite (proj1 (proj2 (proj2 equation_names_none)) kind simple) in Hin.
  destruct Hin.
Qed.

Lemma add_alias_keeps_names (p : Join_env.parameters) (s : Join_env.t)
    (kind : flambda_kind) (var : variable)
    (canonicals : Simples_in_joined_envs.t) :
  fresh_avoids p ->
  state_valid s ->
  equation_names_kinded s ->
  equation_names_kinded
    (Join_env.add_alias_of_existential p s kind var canonicals).
Proof.
  intros Hfresh Hvalid H. unfold Join_env.add_alias_of_existential.
  destruct (Join_env.existential_for_these_canonicals p s canonicals kind)
    as [e s1] eqn:Hmake.
  pose proof (existential_keeps_names _ _ _ _ _ _ Hfresh Hvalid Hmake H)
    as H1.
  revert H1. apply names_grow; try reflexivity.
  apply same_readings_grows. repeat split.
Qed.

(* What is shown of a function that imports a variable. *)
Definition import_keeps_names
    (import_var : Join_env.t -> variable -> option Join_env.t) : Prop :=
  ∀ (s s' : Join_env.t) (x : variable),
    state_valid s ->
    envs_equations_kinded s ->
    import_var s x = Some s' ->
    equation_names_kinded s ->
    equation_names_kinded s'.

Lemma import_vars_keeps_names
    (import_var : Join_env.t -> variable -> option Join_env.t)
    (vars : list variable) :
  import_keeps import_var ->
  import_keeps_names import_var ->
  ∀ s s' : Join_env.t,
    state_valid s ->
    envs_equations_kinded s ->
    List.fold_left
      (fun env var =>
         let* env := env in
         import_var env var)
      vars (Some s)
    = Some s' ->
    equation_names_kinded s ->
    equation_names_kinded s'.
Proof.
  intros Hk Hnames.
  induction vars as [ | var vars IH]; intros s s' Hvalid Henvs Hfold H.
  - injection Hfold as <-. exact H.
  - cbn [List.fold_left] in Hfold.
    destruct (import_var s var) as [s1 | ] eqn:Hvar.
    + destruct (Hk _ _ _ Hvalid Hvar) as [Hvalid1 [Hgrows1 _]].
      exact (IH s1 s' Hvalid1 (grows_envs_equations_kinded _ _ Hgrows1 Henvs)
               Hfold (Hnames _ _ _ Hvalid Henvs Hvar H)).
    + exfalso. clear -Hfold.
      induction vars as [ | var' vars IH]; [ discriminate | exact (IH Hfold) ].
Qed.

Lemma add_equals_keeps_names (p : Join_env.parameters)
    (import_var : Join_env.t -> variable -> option Join_env.t)
    (s s' : Join_env.t) (kind : flambda_kind) (var : variable)
    (simples : Simples_in_joined_envs.t) :
  fresh_avoids p ->
  import_keeps_names import_var ->
  state_valid s ->
  envs_equations_kinded s ->
  Join_env.add_equals_in_joined_envs_with p import_var s kind var simples
  = Some s' ->
  equation_names_kinded s ->
  equation_names_kinded s'.
Proof.
  intros Hfresh Hnames Hvalid Henvs Hadd H.
  unfold Join_env.add_equals_in_joined_envs_with in Hadd.
  destruct (Join_env.n_way_join_canonical_simples s simples)
    as [[canonical | imported | canonicals] | ]; [ | | | discriminate ].
  - exact (add_equation_keeps_names _ _ _ _ Hadd
             (alias_no_equation_names s kind canonical) H).
  - destruct (import_var s imported) as [s1 | ] eqn:Hi; [ | discriminate ].
    exact (add_equation_keeps_names _ _ _ _ Hadd
             (alias_no_equation_names s1 kind (Var imported))
             (Hnames _ _ _ Hvalid Henvs Hi H)).
  - injection Hadd as <-.
    exact (add_alias_keeps_names p s kind var canonicals Hfresh Hvalid H).
Qed.

Lemma concrete_keeps_names (p : Join_env.parameters)
    (import_var : Join_env.t -> variable -> option Join_env.t)
    (s s' : Join_env.t) (var : variable)
    (types : Keyed.finite_map Index.t flambda_type) :
  import_keeps import_var ->
  import_keeps_names import_var ->
  state_valid s ->
  envs_equations_kinded s ->
  types_equations_kinded_in_envs s types ->
  Join_env.import_or_delay_n_way_join_of_concrete_type_with p import_var s var
    types
  = Some s' ->
  equation_names_kinded s ->
  equation_names_kinded s'.
Proof.
  intros Hk Hnames Hvalid Henvs Htypes Hconcrete H.
  unfold Join_env.import_or_delay_n_way_join_of_concrete_type_with
    in Hconcrete.
  destruct (Index.choose types) as [[index any_ty] | ] eqn:Hchoose;
    [ | discriminate ].
  pose proof (choose_lookup _ _ _ Hchoose) as Hany.
  destruct (Htypes _ _ Hany) as [use [Huse Hkinded_any]].
  destruct (forallb _ _).
  - destruct (Join_env.add_equation_in_target_env s var any_ty)
      as [s1 | ] eqn:Hadd; [ | discriminate ].
    pose proof (add_equation_same_readings _ _ _ _ Hadd) as Hsame.
    apply (import_vars_keeps_names import_var (free_variables any_ty) Hk
             Hnames s1 s' (same_readings_valid _ _ Hsame Hvalid)
             (grows_envs_equations_kinded _ _ (same_readings_grows _ _ Hsame)
                Henvs)
             Hconcrete).
    apply (add_equation_keeps_names _ _ _ _ Hadd); [ | exact H ].
    intros n kind Hin. right. exists index, use.
    split; [ exact Huse | exact (Hkinded_any n kind Hin) ].
  - injection Hconcrete as <-. destruct H as [H1 H2].
    split; [ exact H1 | ].
    intros x types0 Hx.
    cbn [Join_env.add_equations_in_joined_envs Join_env.types_in_joined_envs]
      in Hx.
    rewrite lookup_cons in Hx. destruct (Pos.eqb x var).
    + injection Hx as <-. exact Htypes.
    + exact (H2 x types0 Hx).
Qed.

(* The types that the joined environments have for a variable brought
   in have the names of the equations inside them at their kinds. *)
Lemma imported_types_equations_kinded (s : Join_env.t) (x : variable)
    (kind : flambda_kind) (types : Keyed.finite_map Index.t flambda_type) :
  envs_equations_kinded s ->
  Joined_envs.find_imported_var (Join_env.joined_envs s) x kind = Some types
  ->
  types_equations_kinded_in_envs s types.
Proof.
  intros Henvs Htypes index t Ht.
  destruct (types_of_imported_var_spec _ _ _ _ Htypes) as [_ Hfound].
  apply (keyed_lookup_in Index.equal index_equal_decides) in Ht.
  destruct (Hfound _ _ Ht) as [env [equations [Hin Hfind]]].
  pose proof (in_compact_joined_env _ _ _ _ Hin) as Huse.
  exists env. split; [ exact Huse | ].
  exact (find_equations_kinded env (Name_var x) (Some kind) t
           (Henvs _ _ Huse) Hfind).
Qed.

Lemma import_body_keeps_names (p : Join_env.parameters)
    (import_var : Join_env.t -> variable -> option Join_env.t) :
  fresh_avoids p ->
  import_keeps import_var ->
  import_keeps_names import_var ->
  import_keeps_names (Join_env.import_var_transitive_body p import_var).
Proof.
  intros Hfresh Hk Hnames s s' x Hvalid Henvs Hbody H.
  unfold Join_env.import_var_transitive_body in Hbody.
  destruct (Source_env.exists_in_source_env (Join_env.source_env s) x)
    eqn:Hsource; [ injection Hbody as <-; exact H | ].
  destruct (among x (Join_env.imported_vars s));
    [ injection Hbody as <-; exact H | ].
  destruct (Joined_envs.kind_of_imported_var (Join_env.joined_envs s) x)
    as [kind | ] eqn:Hkind; [ | discriminate ].
  cbv zeta in Hbody. fold (with_import s x kind) in Hbody.
  assert (Hunknown : ~ knows (Join_env.source_env s) x).
  { unfold knows. unfold Source_env.exists_in_source_env in Hsource.
    rewrite Hsource. discriminate. }
  destruct (with_import_keeps s x kind Hvalid Hunknown
              (kind_of_imported_var_known _ _ _ Hkind))
    as [Hvalid1 [Hgrows1 _]].
  pose proof (grows_envs_equations_kinded _ _ Hgrows1 Henvs) as Henvs1.
  assert (H1 : equation_names_kinded (with_import s x kind)).
  { apply (names_grow s _ Hgrows1); try reflexivity. exact H. }
  destruct (Joined_envs.find_imported_var _ x kind) as [types | ] eqn:Htypes;
    [ | discriminate ].
  pose proof (imported_types_equations_kinded s x kind types Henvs Htypes)
    as Htypes_kinded.
  destruct (existsb _ _).
  - destruct
      (Joined_envs.get_alias_then_canonical_simple_ignoring_name_mode_exn
         _ _) as [[canonical_simples | ] | ]; try discriminate.
    exact (add_equals_keeps_names p import_var _ s' kind x canonical_simples
             Hfresh Hnames Hvalid1 Henvs1 Hbody H1).
  - exact (concrete_keeps_names p import_var _ s' x types Hk Hnames Hvalid1
             Henvs1 Htypes_kinded Hbody H1).
Qed.

Lemma import_var_keeps_names (p : Join_env.parameters) (fuel : nat) :
  fresh_avoids p -> import_keeps_names (Join_env.import_var_transitive p fuel).
Proof.
  intros Hfresh. induction fuel as [ | fuel IH].
  - intros s s' x _ _ H. discriminate.
  - exact (import_body_keeps_names p _ Hfresh (import_var_keeps p fuel Hfresh)
             IH).
Qed.

(* ======================================================================== *)
(* The join of types                                                        *)
(* ======================================================================== *)

(* What every function of the join of environments keeps, between any
   two states. *)
Definition keeps_names (s s' : Join_env.t) : Prop :=
  state_valid s ->
  steps s s'
  ∧ (envs_equations_kinded s ->
     equation_names_kinded s ->
     equation_names_kinded s').

Lemma keeps_names_refl (s : Join_env.t) : keeps_names s s.
Proof.
  intros Hvalid. split; [ exact (steps_refl s Hvalid) | ].
  intros _ H. exact H.
Qed.

Lemma keeps_names_trans (s1 s2 s3 : Join_env.t) :
  keeps_names s1 s2 -> keeps_names s2 s3 -> keeps_names s1 s3.
Proof.
  intros H12 H23 Hvalid1. destruct (H12 Hvalid1) as [Hsteps12 K12].
  destruct (H23 (proj1 (proj2 Hsteps12))) as [Hsteps23 K23].
  split; [ exact (steps_trans _ _ _ Hsteps12 Hsteps23) | ].
  intros Henvs H.
  pose proof (proj1 (proj2 (proj2 Hsteps12))) as Hgrows12.
  exact (K23 (grows_envs_equations_kinded _ _ Hgrows12 Henvs) (K12 Henvs H)).
Qed.

Lemma import_var_keeps_names_between (p : Join_env.parameters) (fuel : nat)
    (s s' : Join_env.t) (x : variable) :
  fresh_avoids p ->
  Join_env.import_var_transitive p fuel s x = Some s' ->
  keeps_names s s'.
Proof.
  intros Hfresh Hi Hvalid. split.
  - exact (import_steps _ (import_var_keeps p fuel Hfresh)
             (import_var_keeps_scope p fuel) _ _ _ Hvalid Hi).
  - intros Henvs H.
    exact (import_var_keeps_names p fuel Hfresh _ _ _ Hvalid Henvs Hi H).
Qed.

Lemma import_type_keeps_names_between (p : Join_env.parameters) (fuel : nat)
    (s s' : Join_env.t) (ty : flambda_type) :
  fresh_avoids p ->
  Join_env.import_type_transitive p fuel s ty = Some s' ->
  keeps_names s s'.
Proof.
  intros Hfresh Hi Hvalid. split.
  - exact (import_vars_steps _ (free_variables ty)
             (import_var_keeps p fuel Hfresh) (import_var_keeps_scope p fuel)
             _ _ Hvalid Hi).
  - intros Henvs H.
    exact (import_vars_keeps_names _ (free_variables ty)
             (import_var_keeps p fuel Hfresh)
             (import_var_keeps_names p fuel Hfresh) s s' Hvalid Henvs Hi H).
Qed.

Lemma n_way_join_simples_keeps_names (p : Join_env.parameters) (fuel : nat)
    (s s' : Join_env.t) (kind : flambda_kind)
    (simples : list (Index.t * simple)) (result : Or_bottom.t simple) :
  fresh_avoids p ->
  Join_env.n_way_join_simples p fuel s kind simples = Some (result, s') ->
  keeps_names s s'.
Proof.
  intros Hfresh Hjoin Hvalid. split.
  { exact (proj1 (n_way_join_simples_keeps_kinds p fuel s s' kind simples
                    result Hfresh Hjoin Hvalid)). }
  intros Henvs H. unfold Join_env.n_way_join_simples in Hjoin.
  destruct simples as [ | first rest].
  { injection Hjoin as _ <-. exact H. }
  destruct (Joined_envs.get_canonical_simples_ignoring_name_mode _ _)
    as [canonicals | ]; [ | discriminate ].
  destruct (Join_env.n_way_join_canonical_simples s canonicals)
    as [[simple0 | var | canonicals'] | ]; [ | | | discriminate ].
  - injection Hjoin as _ <-. exact H.
  - destruct (Join_env.import_var_transitive p fuel s var) as [s1 | ]
      eqn:Himport; [ | discriminate ].
    injection Hjoin as _ <-.
    exact (import_var_keeps_names p fuel Hfresh _ _ _ Hvalid Henvs Himport H).
  - destruct (Join_env.existential_for_these_canonicals p s canonicals' kind)
      as [x s1] eqn:Hmake.
    injection Hjoin as _ <-.
    exact (existential_keeps_names _ _ _ _ _ _ Hfresh Hvalid Hmake H).
Qed.

Lemma n_way_join_env_extension_keeps_names (p : Join_env.parameters)
    (fuel : nat) (s s' : Join_env.t)
    (extensions : list (Index.t * env_extension)) (joined : env_extension) :
  fresh_avoids p ->
  Join_env.n_way_join_env_extension p fuel s extensions
  = Some (Or_bottom.Ok (joined, s')) ->
  keeps_names s s'.
Proof.
  intros Hfresh Hjoin. unfold Join_env.n_way_join_env_extension in Hjoin.
  destruct extensions as [ | [index0 first] others]; [ discriminate | ].
  destruct (forallb _ others).
  - destruct (List.fold_left _ (Join_env.equations_of first) (Some s))
      as [s1 | ] eqn:Hfold; [ | discriminate ].
    injection Hjoin as _ <-. revert s Hfold.
    generalize (Join_env.equations_of first) as equations.
    induction equations as [ | [n ty] equations IH]; intros s Hfold.
    + injection Hfold as <-. apply keeps_names_refl.
    + cbn [List.fold_left fst snd] in Hfold.
      destruct (Join_env.import_name_transitive p fuel s n) as [s2 | ]
        eqn:Hname.
      * assert (Hfirst : keeps_names s s2).
        { destruct n as [x | sym].
          - exact (import_var_keeps_names_between p fuel _ _ _ Hfresh Hname).
          - injection Hname as <-. apply keeps_names_refl. }
        destruct (Join_env.import_type_transitive p fuel s2 ty) as [s3 | ]
          eqn:Htype.
        -- exact (keeps_names_trans _ _ _ Hfirst
                    (keeps_names_trans _ _ _
                       (import_type_keeps_names_between p fuel _ _ _ Hfresh
                          Htype)
                       (IH _ Hfold))).
        -- exfalso. clear -Hfold.
           induction equations as [ | e equations IH];
             [ discriminate | exact (IH Hfold) ].
      * exfalso. clear -Hfold.
        induction equations as [ | e equations IH];
          [ discriminate | exact (IH Hfold) ].
  - injection Hjoin as _ <-. apply keeps_names_refl.
Qed.

Theorem n_way_join_keeps_equation_names
  : n_way_join_keeps_equation_names_statement.
Proof.
  intros p fuel_of_env fuel expand_head reduce_head_of_kind_naked_immediate
    type_eqb block_shape_eqb Hfresh s s' ts joined Hstep Hvalid Henvs H.
  refine
    (proj2
       (n_way_join_keeps Join_env.t keeps_names Join_env.joined_env
          (Join_env.n_way_join_simples p fuel_of_env)
          (Join_env.import_type p fuel_of_env)
          (Join_env.n_way_join_env_extension p fuel_of_env) expand_head
          reduce_head_of_kind_naked_immediate type_eqb block_shape_eqb fuel
          keeps_names_refl keeps_names_trans _ _ _ s s' ts joined Hstep
          Hvalid)
       Henvs H).
  - intros s0 s0' kind simples result Hsimples.
    exact (n_way_join_simples_keeps_names p fuel_of_env s0 s0' kind simples
             result Hfresh Hsimples).
  - intros s0 s0' ty Himport.
    exact (import_type_keeps_names_between p fuel_of_env s0 s0' ty Hfresh
             Himport).
  - intros s0 s0' extensions joined0 Hextension.
    exact (n_way_join_env_extension_keeps_names p fuel_of_env s0 s0'
             extensions joined0 Hfresh Hextension).
Qed.

(* ======================================================================== *)
(* The relations recovered                                                  *)
(* ======================================================================== *)

Lemma alias_chain_aliased (extension : env_extension) (n n' : name) :
  alias_chain extension n n' -> aliased_in extension n n'.
Proof.
  intros H. induction H as [n | n n1 n2 simple Hlookup Hname _ IH].
  - apply Aliased_here.
  - exact (Aliased_step _ _ _ _ _ Hlookup Hname IH).
Qed.

Lemma variant_equation_names (is_int : option variable)
    (immediates : Or_unknown.t flambda_type) (get_tag : option variable)
    (blocks : Or_unknown.t row_like_for_blocks)
    (extensions : variant_extensions) (is_unique : bool) :
  equation_names_with_kinds
    (variant_type is_int immediates get_tag blocks extensions is_unique)
  = match immediates with
    | Or_unknown.Known ty => equation_names_with_kinds ty
    | Or_unknown.Unknown => []
    end
    ++ match blocks with
       | Or_unknown.Known blocks =>
         equation_names_of_row_like_for_blocks blocks
       | Or_unknown.Unknown => []
       end
    ++ equation_names_of_env_extension (fst (extensions_pair extensions))
    ++ equation_names_of_env_extension (snd (extensions_pair extensions)).
Proof. destruct extensions; reflexivity. Qed.

Theorem recover_equation_names : recover_equation_names_statement.
Proof.
  intros relations relations' n ty ty' Hrecover equation.
  destruct (recover_type_cases _ _ _ _ _ Hrecover)
    as [-> | [is_int [immediates [g [blocks [extensions [is_unique
          [when_block [-> [Hadd ->]]]]]]]]]]; [ intros H; left; exact H | ].
  rewrite !variant_equation_names. cbn [extensions_pair fst snd].
  intros H. rewrite !in_app_iff in H.
  destruct H as [H | [H | [H | H]]].
  - left. rewrite !in_app_iff. left. exact H.
  - left. rewrite !in_app_iff. right. left. exact H.
  - left. rewrite !in_app_iff. right. right. left. exact H.
  - destruct (add_inverse_relation_spec _ _ _ _ _ _ _ Hadd)
      as [-> | [n' [head [Hchain [_ ->]]]]].
    + left. rewrite !in_app_iff. right. right. right. exact H.
    + destruct (snd (extensions_pair extensions)) as [equations] eqn:Hsnd.
      cbn [Join_env.add_or_replace_equation equation_names_of_env_extension
             flat_map fst snd] in H.
      apply in_app_or in H. destruct H as [[Heq | []] | H].
      * right. exists g, (Env_extension equations), n'.
        split; [ | split; [ | split ] ].
        -- unfold variant_type.
           cbn [aliases_with_kinds aliases_of_descr
                  aliases_of_head_of_kind_value
                  aliases_of_head_of_kind_value_non_null].
           rewrite !in_app_iff. left. right. right. left. left. reflexivity.
        -- intros alias Halias. unfold variant_type.
           destruct extensions as [ | a b]; cbn [extensions_pair snd] in Hsnd.
           ++ unfold empty_env_extension in Hsnd. injection Hsnd as <-.
              destruct Halias.
           ++ subst b.
              cbn [aliases_with_kinds aliases_of_descr
                     aliases_of_head_of_kind_value
                     aliases_of_head_of_kind_value_non_null].
              rewrite !in_app_iff. tauto.
        -- exact (alias_chain_aliased _ _ _ Hchain).
        -- symmetry. exact Heq.
      * left. rewrite !in_app_iff. right. right. right.
        cbn [equation_names_of_env_extension]. apply in_flat_map in H.
        destruct H as [binding [Hbinding H]]. apply filter_In in Hbinding.
        apply in_flat_map. exists binding.
        split; [ exact (proj1 Hbinding) | exact H ].
Qed.

(* A name reached through the aliases of an extension is a naked
   immediate where the first is, if the aliases of the extension are
   of their kinds. *)
Lemma aliased_of_kind (of_kind : simple -> flambda_kind -> Prop)
    (extension : env_extension) (n n' : name) :
  aliased_in extension n n' ->
  (∀ alias : simple * flambda_kind,
     alias ∈ aliases_of_env_extension extension ->
     of_kind (fst alias) (snd alias)) ->
  of_kind (simple_of_name n) (Naked_number Kind_naked_immediate) ->
  of_kind (simple_of_name n') (Naked_number Kind_naked_immediate).
Proof.
  intros Hchain Haliases.
  induction Hchain as [n | n n1 n2 simple Hlookup Hname _ IH]; intros H;
    [ exact H | ].
  apply IH. rewrite <- (name_of_simple_some _ _ Hname).
  apply (Haliases (simple, Naked_number Kind_naked_immediate)).
  destruct extension as [equations]. cbn [aliases_of_env_extension].
  apply in_flat_map.
  exists (n, Type_naked_immediate (Ok (Equals simple))). split.
  - exact (keyed_lookup_in name_eqb name_eqb_eq _ _ _ Hlookup).
  - left. reflexivity.
Qed.

(* ======================================================================== *)
(* The first phase                                                          *)
(* ======================================================================== *)

Lemma import_or_delay_keeps_names (p : Join_env.parameters) (fuel : nat)
    (s s' : Join_env.t) (var : variable)
    (types : Keyed.finite_map Index.t flambda_type) :
  fresh_avoids p ->
  state_valid s ->
  envs_equations_kinded s ->
  types_equations_kinded_in_envs s types ->
  Join_env.import_or_delay_n_way_join_type p fuel s var types = Some s' ->
  equation_names_kinded s ->
  equation_names_kinded s'.
Proof.
  intros Hfresh Hvalid Henvs Htypes H HX.
  unfold Join_env.import_or_delay_n_way_join_type,
    Join_env.import_or_delay_n_way_join_type_with in H.
  destruct (Index.choose types) as [[index any_ty] | ]; [ | discriminate ].
  destruct
    (Joined_envs.get_alias_then_canonical_simple_ignoring_name_mode_exn _ _)
    as [[canonical_simples | ] | ]; [ | | discriminate ].
  - exact (add_equals_keeps_names p _ s s' _ var canonical_simples Hfresh
             (import_var_keeps_names p fuel Hfresh) Hvalid Henvs H HX).
  - exact (concrete_keeps_names p _ s s' var types
             (import_var_keeps p fuel Hfresh)
             (import_var_keeps_names p fuel Hfresh) Hvalid Henvs Htypes H HX).
Qed.

Lemma aliases_fold_keeps_names (p : Join_env.parameters) (fuel : nat)
    (envs : Joined_envs.t) (test : variable -> bool) (s0 : Join_env.t) :
  fresh_avoids p ->
  envs_equations_kinded s0 ->
  (∀ x : variable, test x = true -> knows (Join_env.source_env s0) x) ->
  ∀ (names : list name) (s s' : Join_env.t),
    (∀ x : variable,
       Name_var x ∈ names ->
       test x = true ->
       types_equations_kinded_in_envs s0
         (Join_env.types_of_name_in_levels envs (Name_var x))) ->
    steps s0 s ->
    List.fold_left (aliases_step p fuel envs test) names (Some s) = Some s' ->
    equation_names_kinded s ->
    equation_names_kinded s'.
Proof.
  intros Hfresh Henvs0 Htest.
  induction names as [ | n names IH]; intros s s' Hvars Hsteps Hfold HX.
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
    assert (Hfirst : steps s s1 ∧ equation_names_kinded s1).
    { destruct n as [var | sym]; cbn [aliases_step] in Hstep.
      - destruct (test var) eqn:Hvar.
        + split.
          * exact (import_or_delay_steps p fuel s s1 var _ Hfresh Hvalid
                     (or_introl (Htest_s var Hvar)) Hstep).
          * apply (import_or_delay_keeps_names p fuel s s1 var
                     (Join_env.types_of_name_in_levels envs (Name_var var))
                     Hfresh Hvalid
                     (grows_envs_equations_kinded _ _ Hgrows Henvs0));
              [ | exact Hstep | exact HX ].
            intros index t Ht.
            destruct (Hvars var (or_introl eq_refl) Hvar index t Ht)
              as [use [Huse Hk]].
            exists use.
            split; [ rewrite (proj1 Hgrows); exact Huse | exact Hk ].
        + injection Hstep as <-.
          split; [ exact (steps_refl s Hvalid) | exact HX ].
      - injection Hstep as <-.
        split; [ exact (steps_refl s Hvalid) | exact HX ]. }
    destruct Hfirst as [Hsteps1 HX1].
    exact (IH s1 s' (fun x Hx => Hvars x (or_intror Hx))
             (steps_trans _ _ _
                (conj Hvalid0 (conj Hvalid (conj Hgrows Hscope))) Hsteps1)
             Hfold HX1).
Qed.

(* The types that the joined environments have for the [Simple]s of a
   vector have the names of the equations inside them at their
   kinds. *)
Lemma simples_types_equations_kinded (p : Join_env.parameters)
    (s : Join_env.t) (x : variable) (simples : Simples_in_joined_envs.t)
    (kind : flambda_kind) (types : Keyed.finite_map Index.t flambda_type) :
  expand_const_no_equations p ->
  envs_equations_kinded s ->
  types_hold s x types ->
  Joined_envs.find_simples_in_joined_envs (Join_env.expand_const p)
    (Join_env.joined_envs s) simples kind
  = Some types ->
  types_equations_kinded_in_envs s types.
Proof.
  intros Hconst Henvs [_ [Hfound_use _]] Hfind index t Ht.
  destruct (Hfound_use index t Ht) as [use [Huse _]].
  exists use. split; [ exact Huse | ].
  unfold Joined_envs.find_simples_in_joined_envs in Hfind.
  destruct (types_of_simples_lookup _ _ _ _ _ Hfind) as [Hfound _].
  destruct (Hfound index t Ht) as [simple [_ Hty]].
  destruct simple as [y | sym | c]; cbn [type_of_simple_at] in Hty.
  - change (Joined_envs.get_nth_joined_env (Join_env.joined_envs s) index)
      with (Join_env.joined_env s index) in Hty.
    rewrite Huse in Hty. cbn in Hty.
    exact (find_equations_kinded use _ _ t (Henvs _ _ Huse) Hty).
  - change (Joined_envs.get_nth_joined_env (Join_env.joined_envs s) index)
      with (Join_env.joined_env s index) in Hty.
    rewrite Huse in Hty. cbn in Hty.
    exact (find_equations_kinded use _ _ t (Henvs _ _ Huse) Hty).
  - injection Hty as <-. intros n k Hin. rewrite Hconst in Hin. destruct Hin.
Qed.

(* ======================================================================== *)
(* The rounds                                                               *)
(* ======================================================================== *)

Lemma later_envs_equations_kinded (s s' : Join_env.t) :
  later s s' -> envs_equations_kinded s -> envs_equations_kinded s'.
Proof.
  intros Hlater H index use Huse. rewrite (later_env _ _ _ Hlater) in Huse.
  exact (H _ _ Huse).
Qed.

Lemma later_types_equations_kinded (s s' : Join_env.t)
    (types : Keyed.finite_map Index.t flambda_type) :
  later s s' ->
  types_equations_kinded_in_envs s types ->
  types_equations_kinded_in_envs s' types.
Proof.
  intros Hlater H index t Ht. destruct (H index t Ht) as [use [Huse Hk]].
  exists use.
  split; [ rewrite (later_env _ _ _ Hlater); exact Huse | exact Hk ].
Qed.

Section Rounds_names.

Context (source : Typing_env.t) (jt : Join_env.n_way_join_type)
  (choices : Join_env.choices).

Hypothesis jt_sound :
  FlambdaJoinSpec.n_way_join_type_sound typing_env_invariant
    Join_env.joined_env Join_env.definitions_of_existentials
    kind_in_target_env (state_true source) state_kinded jt.

Hypothesis jt_names :
  n_way_join_type_equations_kinded typing_env_invariant Join_env.joined_env
    kind_in_target_env state_valid state_kinded jt.

Hypothesis jt_keeps : join_of_types_keeps_equation_names jt.

Lemma join_call_names (s s0 : Join_env.t) (var : variable)
    (types : Keyed.finite_map Index.t flambda_type) (ty : flambda_type) :
  state_true source s ->
  envs_invariant s ->
  envs_kinded s ->
  envs_equations_kinded s ->
  types_hold s var types ->
  types_equations_kinded_in_envs s types ->
  state_kinded s ->
  jt s (Index.bindings types) = Some (Or_unknown.Known ty, s0) ->
  type_equations_kinded_in_target Join_env.joined_env kind_in_target_env s0
    ty.
Proof.
  intros Htrue Hinvariant Hkinded Henvs [Hvar [Hfound _]] Htypes Hstate Hjoin.
  apply (jt_names s s0 (Index.bindings types) ty (proj1 Htrue) Hinvariant
           Hkinded Henvs (bindings_distinct types));
    [ | | | exact Hstate | exact Hjoin ].
  - intros index t Hin. apply bindings_lookup in Hin.
    destruct (Hfound _ _ Hin) as [use [Huse [Hscope _]]].
    exists use. split; [ exact Huse | exact Hscope ].
  - intros index t Hin. apply bindings_lookup in Hin.
    destruct (Hfound _ _ Hin) as [use [Huse [_ Hkind]]].
    exists use. split; [ exact Huse | exact Hkind ].
  - intros index t Hin. apply bindings_lookup in Hin.
    exact (Htypes _ _ Hin).
Qed.

Lemma round_fold_names :
  ∀ (bindings : list (variable * Keyed.finite_map Index.t flambda_type))
    (s s' : Join_env.t) (relations relations' : Join_env.inverse_relations),
    (∀ (var : variable) (types : Keyed.finite_map Index.t flambda_type),
       In (var, types) bindings ->
       types_hold s var types ∧ types_equations_kinded_in_envs s types) ->
    state_true source s ->
    envs_invariant s ->
    envs_kinded s ->
    envs_equations_kinded s ->
    relations_hold s relations ->
    state_kinded s ->
    equation_names_kinded s ->
    List.fold_left (round_step jt choices) bindings (Some (relations, s))
    = Some (relations', s') ->
    equation_names_kinded s'.
Proof.
  induction bindings as [ | [var types] bindings IH];
    intros s s' relations relations' Htypes Htrue Hinvariant Hkinded Henvs
      Hrelations Hstate HX Hfold.
  - cbn in Hfold. injection Hfold as _ <-. exact HX.
  - cbn [List.fold_left] in Hfold.
    destruct (round_step jt choices (Some (relations, s)) (var, types))
      as [[relations1 s1] | ] eqn:Hstep;
      [ | rewrite round_step_none in Hfold; discriminate ].
    destruct (Htypes var types (or_introl eq_refl))
      as [Hvar_types Hvar_names].
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
    assert (HX1 : equation_names_kinded s1).
    { unfold round_step in Hstep. cbn [fst snd] in Hstep.
      destruct (jt s (Index.bindings types)) as [[joined s0] | ] eqn:Hjoin;
        [ | discriminate ].
      pose proof (jt_keeps s s0 _ joined Hjoin (proj1 Htrue) Henvs HX) as HX0.
      destruct joined as [ty | ].
      - pose proof (join_call_names s s0 var types ty Htrue Hinvariant Hkinded
                      Henvs Hvar_types Hvar_names Hstate Hjoin) as Hty.
        pose proof (proj2 (join_call_kinds _ _ jt_sound _ _ _ _ _ Htrue
                             Hinvariant Hkinded Hvar_types Hstate Hjoin ty
                             eq_refl)) as Hkinded_ty.
        assert (Hadd :
                  ∀ (ty' : flambda_type) (s2 : Join_env.t),
                    ty' = ty
                    ∨ (∃ relations0 : Join_env.inverse_relations,
                         Join_env.recover_inverse_relations relations
                           (Name_var var) ty
                         = Some (ty', relations0)) ->
                    Join_env.add_equation_in_target_env s0 var ty' = Some s2
                    ->
                    equation_names_kinded s2).
        { intros ty' s2 Hcase Hadd.
          apply (add_equation_keeps_names _ _ _ _ Hadd); [ | exact HX0 ].
          intros n kind Hin. left.
          destruct Hcase as [-> | [relations0 Hrecover]];
            [ exact (Hty n kind Hin) | ].
          destruct (recover_equation_names _ _ _ _ _ Hrecover (n, kind) Hin)
            as [Hold | [g [extension [n' [Hg [Hext [Hchain Heq]]]]]]].
          - exact (Hty n kind Hold).
          - injection Heq as -> ->.
            apply (aliased_of_kind (of_kind_in_target s0) extension
                     (Name_var g) n' Hchain).
            + intros [simple k] Halias.
              exact (Hkinded_ty simple k (Hext _ Halias)).
            + exact (Hkinded_ty _ _ Hg). }
        destruct (Join_env.recover_relations_on choices s0 var types).
        + destruct (Join_env.recover_inverse_relations relations
                      (Name_var var) ty)
            as [[ty' relations0] | ] eqn:Hrecover; [ | discriminate ].
          destruct (Join_env.add_equation_in_target_env s0 var ty')
            as [s2 | ] eqn:Hadd2; [ | discriminate ].
          injection Hstep as _ <-.
          exact (Hadd ty' s2 (or_intror (ex_intro _ relations0 eq_refl))
                   Hadd2).
        + destruct (Join_env.add_equation_in_target_env s0 var ty)
            as [s2 | ] eqn:Hadd2; [ | discriminate ].
          injection Hstep as _ <-.
          exact (Hadd ty s2 (or_introl eq_refl) Hadd2).
      - injection Hstep as _ <-. exact HX0. }
    apply (IH s1 s' relations1 relations');
      [ | exact (later_true source s s1 Hlater1 Hholds1
                   (proj2 (proj2 Htrue)))
        | exact (later_envs_invariant _ _ Hlater1 Hinvariant)
        | exact (later_envs_kinded _ _ Hlater1 Hkinded)
        | exact (later_envs_equations_kinded _ _ Hlater1 Henvs)
        | exact Hrelations1 | exact Hstate1 | exact HX1 | exact Hfold ].
    intros v t Hin. destruct (Htypes v t (or_intror Hin)) as [Hh Hk].
    split; [ exact (later_types_hold _ _ _ _ Hlater1 Hh) | ].
    exact (later_types_equations_kinded _ _ _ Hlater1 Hk).
Qed.

Lemma delayed_equations_names :
  ∀ (fuel : nat) (s s' : Join_env.t)
    (equations : finite_map (Keyed.finite_map Index.t flambda_type))
    (relations relations' : Join_env.inverse_relations),
    (∀ (var : variable) (types : Keyed.finite_map Index.t flambda_type),
       lookup var equations = Some types ->
       types_hold s var types ∧ types_equations_kinded_in_envs s types) ->
    state_true source s ->
    envs_invariant s ->
    envs_kinded s ->
    envs_equations_kinded s ->
    relations_hold s relations ->
    state_kinded s ->
    equation_names_kinded s ->
    Join_env.n_way_join_delayed_equations fuel choices jt s relations
      equations
    = Some (relations', s') ->
    equation_names_kinded s'.
Proof.
  induction fuel as [ | fuel IH];
    intros s s' equations relations relations' Htypes Htrue Hinvariant Hkinded
      Henvs Hrelations Hstate HX Hloop; [ discriminate | ].
  cbn [Join_env.n_way_join_delayed_equations] in Hloop.
  destruct (Keyed.is_empty equations).
  - injection Hloop as _ <-. exact HX.
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
    assert (HX1 : equation_names_kinded s1).
    { apply (round_fold_names (compact equations) s s1 relations relations1);
        [ | exact Htrue | exact Hinvariant | exact Hkinded | exact Henvs
          | exact Hrelations | exact Hstate | exact HX | exact Hround ].
      intros var types Hin. apply Htypes. apply in_compact. exact Hin. }
    assert (Htypes1 :
              ∀ (var : variable)
                (types : Keyed.finite_map Index.t flambda_type),
                lookup var (Join_env.new_equations_in_joined_envs s1 s)
                = Some types ->
                types_hold s1 var types
                ∧ types_equations_kinded_in_envs s1 types).
    { intros var types Hvar.
      unfold Join_env.new_equations_in_joined_envs in Hvar.
      rewrite lookup_filter_bindings in Hvar.
      destruct (lookup var (Join_env.types_in_joined_envs s1))
        as [types1 | ] eqn:Hlookup; [ | discriminate ].
      destruct (negb (is_bound var (Join_env.types_in_joined_envs s)));
        [ | discriminate ].
      injection Hvar as <-.
      split; [ exact (proj1 (proj2 Hholds1) var types1 Hlookup) | ].
      exact (proj2 HX1 var types1 Hlookup). }
    exact (IH s1 s' _ relations1 relations' Htypes1
             (later_true source s s1 Hlater1 Hholds1 (proj2 (proj2 Htrue)))
             (later_envs_invariant _ _ Hlater1 Hinvariant)
             (later_envs_kinded _ _ Hlater1 Hkinded)
             (later_envs_equations_kinded _ _ Hlater1 Henvs) Hrelations1
             Hstate1 HX1 Hloop).
Qed.

Lemma loop_names (p : Join_env.parameters) :
  expand_const_sound p ->
  expand_const_no_equations p ->
  ∀ (depth fuel : nat) (s s' : Join_env.t)
    (relations relations' : Join_env.inverse_relations)
    (existentials : finite_map (Simples_in_joined_envs.t * flambda_kind)),
    (∀ (x : variable) (definition : Simples_in_joined_envs.t * flambda_kind),
       lookup x existentials = Some definition ->
       lookup x (Join_env.definitions_of_existentials s) = Some definition) ->
    state_true source s ->
    envs_invariant s ->
    envs_kinded s ->
    envs_equations_kinded s ->
    relations_hold s relations ->
    state_kinded s ->
    equation_names_kinded s ->
    Join_env.n_way_join_loop depth p fuel choices jt s relations existentials
    = Some (relations', s') ->
    equation_names_kinded s'.
Proof.
  intros Hconst Hconst_names.
  induction depth as [ | depth IH];
    intros fuel s s' relations relations' existentials Hdefinitions Htrue
      Hinvariant Hkinded Henvs Hrelations Hstate HX Hloop;
    cbn [Join_env.n_way_join_loop] in Hloop.
  - injection Hloop as _ <-. exact HX.
  - destruct (Keyed.is_empty existentials).
    { injection Hloop as _ <-. exact HX. }
    destruct (Join_env.equations_of_existentials p s (compact existentials))
      as [equations | ] eqn:Hequations; [ | discriminate ].
    destruct (Join_env.n_way_join_delayed_equations fuel choices jt s
                relations equations)
      as [[relations1 s1] | ] eqn:Hdelayed; [ | discriminate ].
    assert (Htypes :
              ∀ (var : variable)
                (types : Keyed.finite_map Index.t flambda_type),
                lookup var equations = Some types ->
                types_hold s var types
                ∧ types_equations_kinded_in_envs s types).
    { intros var types Hvar.
      destruct (equations_of_existentials_lookup p s _ _ Hequations _ _ Hvar)
        as [simples [kind [Hin Hfind]]].
      apply in_compact in Hin.
      pose proof (existential_types_hold p s var simples kind types Hconst
                    (proj1 Htrue) (proj1 (proj2 Htrue)) Hinvariant Hkinded
                    (Hdefinitions _ _ Hin) Hfind) as Hholds.
      split; [ exact Hholds | ].
      exact (simples_types_equations_kinded p s var simples kind types
               Hconst_names Henvs Hholds Hfind). }
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
    pose proof (delayed_equations_names fuel s s1 equations relations
                  relations1 Htypes Htrue Hinvariant Hkinded Henvs Hrelations
                  Hstate HX Hdelayed) as HX1.
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
             (later_envs_kinded _ _ Hlater1 Hkinded)
             (later_envs_equations_kinded _ _ Hlater1 Henvs) Hrelations1
             Hstate1 HX1 Hloop).
Qed.

End Rounds_names.

Theorem first_phases_equation_names : first_phases_equation_names_statement.
Proof.
  intros source jt choices p fuel avoid envs s1 s2 s3 relations2 relations3
    Hjt Hjt_names Hkeeps Hfresh Heqb Hconst Hconst_names Hsource Huses
    Hinvariant Hkinded Hequations Hlevels Hkinds Haliases Hdelayed Hloop.
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
  assert (Hinvariant0 : envs_invariant s0) by exact Hinvariant.
  assert (Hkinded0 : envs_kinded s0) by exact Hkinded.
  assert (Henvs0 : envs_equations_kinded s0) by exact Hequations.
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
  assert (HX0 : equation_names_kinded s0).
  { split; [ intros x ty Hx; discriminate | ].
    intros x types Hx. discriminate. }
  assert (HX1 : equation_names_kinded s1).
  { unfold Join_env.join_aliases_into_bindings in Haliases.
    refine (aliases_fold_keeps_names p fuel envs _ s0 Hfresh Henvs0 _ _ s0 s1
              _ (steps_refl s0 (proj1 Htrue0)) Haliases HX0).
    - intros x Hx. rewrite <- kind_in_target_env_exists in Hx.
      rewrite (proj2 (proj2 Htrue0)). exact (Hknown0 x Hx).
    - intros x _ _ index t Ht. rewrite types_of_name_lookup in Ht.
      destruct (Keyed.lookup Index.equal index envs)
        as [[use equations] | ] eqn:Hindex; [ | discriminate ].
      destruct (Hlevels _ _ _ _ _ Hindex Ht) as [btm' Hfound].
      assert (Huse : Join_env.joined_env s0 index = Some use).
      { apply joined_env_lookup. exists equations. exact Hindex. }
      exists use. split; [ exact Huse | ].
      exact (Henvs0 index use Huse _ _ _ Hfound). }
  assert (Htypes1 :
            ∀ (var : variable)
              (types : Keyed.finite_map Index.t flambda_type),
              lookup var (Join_env.new_equations_in_joined_envs s1 s0)
              = Some types ->
              types_hold s1 var types
              ∧ types_equations_kinded_in_envs s1 types).
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
    exact (proj2 HX1 var types1 Hlookup). }
  assert (Hholds_types1 :
            ∀ (var : variable)
              (types : Keyed.finite_map Index.t flambda_type),
              lookup var (Join_env.new_equations_in_joined_envs s1 s0)
              = Some types ->
              types_hold s1 var types).
  { intros var types Hvar. exact (proj1 (Htypes1 var types Hvar)). }
  assert (Hrelations1 : relations_hold s1 []).
  { intros r relation n [recorded [Hlookup _]]. discriminate. }
  pose proof (later_envs_invariant _ _ Hlater1 Hinvariant0) as Hinvariant1.
  pose proof (later_envs_kinded _ _ Hlater1 Hkinded0) as Hkinded1.
  pose proof (later_envs_equations_kinded _ _ Hlater1 Henvs0) as Henvs1.
  destruct (delayed_equations_hold source jt choices Hjt fuel s1 s2 _ []
              relations2 Hholds_types1 Htrue1 Hinvariant1 Hkinded1
              Hrelations1 Hdelayed)
    as [Hlater2 [Hholds2 Hrelations2]].
  pose proof (delayed_equations_kinded source jt choices Hjt fuel s1 s2 _ []
                relations2 Hholds_types1 Htrue1 Hinvariant1 Hkinded1
                Hrelations1 Hstate1 Hdelayed) as Hstate2.
  pose proof (delayed_equations_names source jt choices Hjt Hjt_names Hkeeps
                fuel s1 s2 _ [] relations2 Htypes1 Htrue1 Hinvariant1 Hkinded1
                Henvs1 Hrelations1 Hstate1 HX1 Hdelayed) as HX2.
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
  exact (loop_names source jt choices Hjt Hjt_names Hkeeps p Hconst
           Hconst_names (Join_env.join_depth p) fuel s2 s3 relations2
           relations3 _ Hdefinitions2
           (later_true source s1 s2 Hlater2 Hholds2 (proj2 (proj2 Htrue1)))
           (later_envs_invariant _ _ Hlater02 Hinvariant0)
           (later_envs_kinded _ _ Hlater02 Hkinded0)
           (later_envs_equations_kinded _ _ Hlater02 Henvs0) Hrelations2
           Hstate2 HX2 Hloop).
Qed.

Theorem scoped_equation_names : scoped_equation_names_statement.
Proof.
  intros s Hvalid Hstate Hinvariant Henvs Hscope [Hreach _] x ty Hx n kind
    Hin.
  destruct (Hreach x ty Hx n kind Hin) as [H | [index [use [Huse H]]]];
    [ exact H | ].
  apply (simple_kind_in_target_of_use s index use (simple_of_name n) kind
           Hvalid Hstate Hinvariant Henvs Huse H).
  intros y Hy. apply (Hscope x ty Hx).
  apply (proj1 equation_names_free_variables ty n kind y Hin).
  destruct n as [z | sym]; exact Hy.
Qed.

(* ======================================================================== *)
(* The last phase                                                           *)
(* ======================================================================== *)

Section Define_names.

Context {meet_env meet_expanded_head : Type}
  (ops : Join_env.meet_env_functions meet_env meet_expanded_head)
  (s : Join_env.t) (tenv_of_aliases : Typing_env.t)
  (unique : variable -> bool).

(* The names of the equations inside a type are of their kinds in the
   target. *)
Definition named (ty : flambda_type) : Prop :=
  type_equations_kinded_in_target Join_env.joined_env kind_in_target_env s
    ty.

Definition names_invariant (acc : define_acc meet_env) : Prop :=
  let '(_, to_expand, equations, _, _) := acc in
  (∀ (x : variable) (ty : flambda_type),
     lookup x equations = Some ty -> named ty)
  ∧ ∀ (e : variable) (ty : flambda_type),
      lookup e to_expand = Some ty -> named ty.

Lemma named_alias (kind : flambda_kind) (simple : simple) :
  named (alias_type_of kind simple).
Proof.
  intros n k Hin.
  rewrite (proj1 (proj2 (proj2 equation_names_none)) kind simple) in Hin.
  destruct Hin.
Qed.

Lemma named_unknown (kind : flambda_kind) : named (unknown kind).
Proof.
  intros n k Hin. rewrite (proj1 equation_names_none kind) in Hin.
  destruct Hin.
Qed.

Lemma define_step_names (acc acc' : define_acc meet_env)
    (binding : variable * flambda_kind) :
  names_invariant acc ->
  define_step ops s tenv_of_aliases unique (Some acc) binding = Some acc' ->
  names_invariant acc'.
Proof.
  intros Hinv Hstep.
  destruct acc as [[[[menv to_expand] equations] relations] definitions].
  destruct binding as [var var_kind]. cbn [define_step fst snd] in Hstep.
  destruct Hinv as [Hequations Hexpand].
  destruct (Join_env.choose_alias tenv_of_aliases _) as [[canon_var | ] | ];
    [ | | discriminate ].
  - destruct (Join_env.move_equation equations var canon_var)
      as [equations1 | ] eqn:Hmove; [ | discriminate ].
    injection Hstep as <-.
    assert (Hequations1 :
              ∀ (x : variable) (ty : flambda_type),
                lookup x equations1 = Some ty -> named ty).
    { unfold Join_env.move_equation in Hmove.
      destruct (is_bound canon_var equations); [ discriminate | ].
      destruct (lookup var equations) as [ty_var | ] eqn:Hlookup_var.
      - injection Hmove as <-. intros x ty Hx. rewrite lookup_cons in Hx.
        destruct (Pos.eqb x canon_var).
        + injection Hx as <-. exact (Hequations var ty_var Hlookup_var).
        + rewrite lookup_unbind in Hx.
          destruct (Pos.eqb x var); [ discriminate | ].
          exact (Hequations x ty Hx).
      - injection Hmove as <-. exact Hequations. }
    split.
    + intros x ty Hx.
      destruct (lookup_fold_cons _ _ _ _ _ Hx) as [[_ ->] | Hlookup].
      * apply named_alias.
      * exact (Hequations1 x ty Hlookup).
    + intros e ty He. rewrite lookup_cons in He.
      destruct (Pos.eqb e var).
      * injection He as <-. apply named_alias.
      * exact (Hexpand e ty He).
  - destruct (unique var).
    + injection Hstep as <-. split.
      * intros x ty Hx. rewrite lookup_unbind in Hx.
        destruct (Pos.eqb x var); [ discriminate | ].
        exact (Hequations x ty Hx).
      * intros e ty He. rewrite lookup_cons in He.
        destruct (Pos.eqb e var); [ | exact (Hexpand e ty He) ].
        injection He as <-.
        destruct (lookup var equations) as [ty_var | ] eqn:Hlookup_var.
        -- exact (Hequations var ty_var Hlookup_var).
        -- apply named_unknown.
    + injection Hstep as <-. split; [ exact Hequations | exact Hexpand ].
Qed.

Lemma define_fold_names (acc acc' : define_acc meet_env) :
  names_invariant acc ->
  List.fold_left (define_step ops s tenv_of_aliases unique)
    (Bindings_in_target_env.existential_variables_in_order
       (Join_env.bindings s))
    (Some acc)
  = Some acc' ->
  names_invariant acc'.
Proof.
  apply (fold_left_invariant (define_step ops s tenv_of_aliases unique)
           names_invariant (fun b => eq_refl)).
  intros a a' b _ Ha Hstep.
  exact (define_step_names a a' b Ha Hstep).
Qed.

Lemma expand_variable_names (to_expand : finite_map flambda_type) :
  (∀ (e : variable) (ty : flambda_type),
     lookup e to_expand = Some ty -> named ty) ->
  ∀ (fuel : nat) (e : variable) (ty : flambda_type),
    Join_env.expand_variable fuel to_expand e = Some ty -> named ty.
Proof.
  intros Hbase.
  induction fuel as [ | fuel IH]; intros e ty Hexpand; [ discriminate | ].
  cbn [Join_env.expand_variable] in Hexpand.
  destruct (lookup e to_expand) as [ty0 | ] eqn:Hlookup; [ | discriminate ].
  cbn in Hexpand.
  pose proof (Hbase e ty0 Hlookup) as Hnamed0.
  destruct (get_alias_opt ty0) as [[y | sym | c] | ] eqn:Halias.
  - destruct (among y (domain to_expand)).
    + exact (IH y ty Hexpand).
    + injection Hexpand as <-. exact Hnamed0.
  - injection Hexpand as <-. exact Hnamed0.
  - injection Hexpand as <-. exact Hnamed0.
  - destruct (project_variables_out_equations_kinded (domain to_expand)
                (Join_env.expand_variable fuel to_expand)
                (fun n k => of_kind_in_target s (simple_of_name n) k))
      as [Htype _].
    + intros e' ty' n k _ He' Hin. exact (IH e' ty' He' n k Hin).
    + intros n k Hin. apply (Htype ty0 ty Hexpand); [ | exact Hin ].
      intros n0 k0 Hin0 _. exact (Hnamed0 n0 k0 Hin0).
Qed.

End Define_names.

Theorem last_phase_adds_equation_names
  : last_phase_adds_equation_names_statement.
Proof.
  intros tenv fuel s relations kept equations relation_equations definitions
    Htypes Hadds n ty Hin.
  destruct (last_phase_adds_parts tenv fuel s relations _ definitions Hadds)
    as [r1 [to_expand [equations1 [relations1 [kept0 [equations0
          [relation_equations0
             [Hfold [_ [_ [_ [_ [Hequations [Hrelation_equations
                                              Heq]]]]]]]]]]]]]].
  injection Heq as _ -> ->.
  assert (Hstart :
            names_invariant s
              ({| recorded_variables :=
                    Bindings_in_target_env.imported_variables_in_order
                      (Join_env.bindings s);
                  recorded_extensions := [] |},
               [], Join_env.types_in_target_env s, relations,
               Join_env.definitions_of_existentials s)).
  { split; [ exact Htypes | intros e ty0 He; discriminate ]. }
  destruct (define_fold_names (recording tenv) s tenv (unique_of s relations)
              _ _ Hstart Hfold)
    as [Hequations_inv Hexpand_inv].
  pose proof (expand_variable_names s to_expand Hexpand_inv) as Hexp.
  destruct (project_variables_out_equations_kinded (domain to_expand)
              (Join_env.expand_variable fuel to_expand)
              (fun n0 k => of_kind_in_target s (simple_of_name n0) k)
              (fun e ty0 n0 k _ He Hin0 => Hexp fuel e ty0 He n0 k Hin0))
    as [Htype _].
  apply in_app_or in Hin. destruct Hin as [Hin | Hin].
  - destruct (traverse_in _ _ _ Hequations (n, ty) Hin)
      as [[x ty0] [Hbinding Hf]].
    cbn [fst snd] in Hf.
    destruct (project_variables_out (domain to_expand)
                (Join_env.expand_variable fuel to_expand) ty0) as [ty1 | ]
      eqn:Hty; [ | discriminate ].
    injection Hf as _ <-. apply in_compact in Hbinding.
    intros n0 k Hin0. apply (Htype ty0 ty1 Hty); [ | exact Hin0 ].
    intros n1 k1 Hin1 _. exact (Hequations_inv x ty0 Hbinding n1 k1 Hin1).
  - destruct (traverse_in _ _ _ Hrelation_equations (n, ty) Hin)
      as [[r names] [Hbinding Hf]].
    cbn [fst snd] in Hf.
    match type of Hf with
    | context [project_variables_out ?to_project ?expand ?ty0] =>
      destruct (project_variables_out to_project expand ty0) as [ty1 | ]
        eqn:Hty; [ | discriminate ]
    end.
    injection Hf as _ <-.
    intros n0 k Hin0. refine (Htype _ ty1 Hty _ n0 k Hin0).
    intros n1 k1 [].
Qed.

(* ======================================================================== *)
(* What the join is given in FlambdaJoinMeet.v                              *)
(* ======================================================================== *)

Theorem n_way_join_of_types_keeps_equation_names
  : n_way_join_of_types_keeps_equation_names_statement.
Proof.
  intros p fuel_of_env expand_head reduce_head_of_kind_naked_immediate fuel
    Hfresh.
  exact
    (n_way_join_keeps_equation_names p fuel_of_env fuel expand_head
       reduce_head_of_kind_naked_immediate FlambdaTypeEqualSpec.type_eqb
       FlambdaTypeEqualSpec.block_shape_eqb Hfresh).
Qed.

Theorem join_parameters_no_equations : join_parameters_no_equations_statement.
Proof.
  intros join_depth c.
  exact (FlambdaExpandHeadProofs.create_const_equation_names c).
Qed.
