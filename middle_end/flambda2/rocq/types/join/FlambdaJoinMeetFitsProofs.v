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


(* The proofs of the statements of FlambdaJoinMeetFitsSpec.v. This
   file needs no review. *)

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
  FlambdaMeetTermsSpec FlambdaMeetTermsProofs FlambdaJoinMeetFitsSpec.
Import ListNotations.

Theorem expand_head_for_join_equations_kinded
  : expand_head_for_join_equations_kinded_statement.
Proof.
  intros tenv ty expanded Hexpand Htypes Hty.
  unfold expand_head_for_join in Hexpand.
  destruct (FlambdaExpandHead.expand_head tenv ty) as [expanded0 | ] eqn:He;
    [ | discriminate ].
  injection Hexpand as <-.
  rewrite (proj1 (join_expanded_to_type expanded0)).
  exact (FlambdaExpandHeadProofs.expand_head_equations_kinded tenv ty
           expanded0 Htypes Hty He).
Qed.

Theorem last_phase_equation_names_known
  : last_phase_equation_names_known_statement.
Proof.
  intros tenv fuel source s relations menv kept equations relation_equations
    definitions Htrue Hrelations Hnonempty Hscoped Hstate Hinvariant Henvs
    Hsymbols HN Hsource Hfits Hadds menv' n ty Hin.
  destruct (last_phase_adds_kept tenv fuel s relations kept equations
              relation_equations definitions Hadds) as [Hdistinct Hkept].
  destruct (last_phase_adds_in_scope tenv fuel source s relations kept
              equations relation_equations definitions Htrue Hrelations
              Hnonempty (proj1 Hscoped)
              (scoped_aliases_not_made s (proj1 Htrue) (proj2 Hscoped)) Hadds)
    as [Hscope1 Hscope2].
  assert (Hsource' : Meet_env.typing_env menv = Join_env.source_env s).
  { rewrite (proj2 (proj2 Htrue)). exact Hsource. }
  destruct (target_variables_defined s menv kept (proj1 Htrue) Hsource' Hfits
              Hkept Hdistinct) as [Hkept' [_ [Hkinds _]]].
  fold menv' in Hkept', Hkinds.
  pose proof (scoped_equation_names s (proj1 Htrue) Hstate Hinvariant Henvs
                (proj1 Hscoped) HN) as Hnamed.
  pose proof (last_phase_adds_equation_names tenv fuel s relations kept
                equations relation_equations definitions Hnamed Hadds n ty
                Hin) as Hty.
  assert (Hfree :
            ∀ y : variable,
              y ∈ free_variables ty -> in_final_target s kept y).
  { apply in_app_or in Hin. destruct Hin as [Hin | Hin];
      [ exact (proj2 (Hscope1 _ _ Hin)) | exact (proj2 (Hscope2 _ _ Hin)) ]. }
  intros n0 k Hmember.
  pose proof (Hty n0 k Hmember) as HK.
  destruct n0 as [y | sym]; cbn [simple_of_name] in HK |- *.
  - apply (Hkinds y k); [ | exact HK ].
    apply Hfree.
    exact (proj1 equation_names_free_variables ty (Name_var y) k y Hmember
             (or_introl eq_refl)).
  - destruct HK as [-> [index [use [Huse Hmem]]]].
    assert (Hknown :
              Typing_env.mem None (Meet_env.typing_env menv')
                (Name_symbol sym)
              = true).
    { apply (tenv_extends_mem _ _ _
               (meet_env_extends_tenv _ _ (proj1 Hkept'))).
      rewrite Hsource. exact (Hsymbols index use sym Huse Hmem). }
    destruct (bound_has_kind _ _ Hknown) as [k' Hk'].
    cbn [simple_of_name] in Hk'.
    rewrite (symbol_kinded _ _ _ (proj1 (proj2 (proj2 (proj2 Hkept'))))
               Hk') in Hk'.
    exact Hk'.
Qed.

(* The types that the last phase adds fit otherwise, for any
   environments joined that have what the first phases suppose: the
   two joins of this file give them from their own premises. *)
Lemma phases_add_types_that_fit (join_choices : Join_env.choices)
    (join_depth : nat) (fuel : fuels) (avoid : list variable)
    (initial_env : Meet_env.t) (envs : Joined_envs.t)
    (s1 s2 s3 : Join_env.t)
    (relations2 relations3 : Join_env.inverse_relations)
    (tenv : Typing_env.t) (kept : list (variable * flambda_kind))
    (equations relation_equations : list (name * flambda_type))
    (definitions : finite_map (Simples_in_joined_envs.t * flambda_kind)) :
  let initial_tenv := Meet_env.typing_env initial_env in
  let p := join_parameters join_depth in
  let jt := model_join_of_types join_depth fuel in
  let s0 :=
    Join_env.create envs
      (Bindings_in_target_env.from_source_env avoid initial_tenv)
  in
  typing_env_fits initial_tenv ->
  (∀ x : variable, knows initial_tenv x -> x ∈ avoid) ->
  (∀ (index : Index.t) (use : Typing_env.t) (x : variable),
     Joined_envs.get_nth_joined_env envs index = Some use ->
     knows use x ->
     x ∈ avoid) ->
  (∀ (index : Index.t) (use : Typing_env.t),
     Joined_envs.get_nth_joined_env envs index = Some use ->
     typing_env_invariant use) ->
  (∀ (index : Index.t) (use : Typing_env.t),
     Joined_envs.get_nth_joined_env envs index = Some use ->
     types_well_formed use) ->
  (∀ (index : Index.t) (use : Typing_env.t),
     Joined_envs.get_nth_joined_env envs index = Some use ->
     typing_env_kinded use) ->
  (∀ (index : Index.t) (use : Typing_env.t),
     Joined_envs.get_nth_joined_env envs index = Some use ->
     types_equations_kinded use) ->
  levels_found envs ->
  (∀ (index : Index.t) (use : Typing_env.t) (x : variable)
     (kind : flambda_kind) (ty : flambda_type)
     (binding_time_and_mode : Binding_time.With_name_mode.t),
     Joined_envs.get_nth_joined_env envs index = Some use ->
     simple_of_kind use (Var x) kind ->
     Keyed.lookup name_eqb (Name_var x)
       (Typing_env.names_to_types initial_tenv)
     = Some (ty, binding_time_and_mode) ->
     FlambdaTypeGrammar.kind ty = kind) ->
  (∀ (index : Index.t) (use : Typing_env.t) (sym : symbol),
     Joined_envs.get_nth_joined_env envs index = Some use ->
     Typing_env.mem None use (Name_symbol sym) = true ->
     Typing_env.mem None initial_tenv (Name_symbol sym) = true) ->
  Join_env.join_aliases_into_bindings p (fuel_of_env fuel) s0 envs
  = Some s1 ->
  Join_env.n_way_join_delayed_equations (fuel_of_env fuel) join_choices jt
    s1 [] (Join_env.new_equations_in_joined_envs s1 s0)
  = Some (relations2, s2) ->
  Join_env.n_way_join_loop join_depth p (fuel_of_env fuel) join_choices jt
    s2 relations2 (Join_env.new_definitions_of_existentials s2 s0)
  = Some (relations3, s3) ->
  last_phase_adds tenv (fuel_of_env fuel) s3 relations3 kept equations
    relation_equations definitions ->
  types_fit_otherwise
    (Meet_env.typing_env
       (variables_defined meet_env_functions
          (Meet_env.with_names_to_avoid initial_env
             (Bindings_in_target_env.names_to_avoid (Join_env.bindings s3)))
          (Bindings_in_target_env.imported_variables_in_order
             (Join_env.bindings s3)
           ++ kept)))
    (equations ++ relation_equations).
Proof.
  intros initial_tenv p jt s0 Hfits Hsource Huses Hinvariant Hwf_envs
    Hkinded_envs Hnames_envs Hlevels Hkinds Hsymbols Haliases Hdelayed Hloop
    Hadds.
  destruct (join_parameters_sound join_depth)
    as [Hfresh [Heqb [Hextension Hconst]]].
  fold p in Hfresh, Heqb, Hextension, Hconst.
  assert (Hreduce :
            FlambdaJoinTypesSpec.reduce_head_of_kind_naked_immediate_sound
              reduce_head_of_kind_naked_immediate).
  { intros tenv0 statics valuation v head Hsat Hhead.
    exact (FlambdaMeetStepsProofs.reduce_head_of_kind_naked_immediate_sound
             statics valuation tenv0 head v Hsat Hhead). }
  pose proof (n_way_join_of_types_true initial_tenv p (fuel_of_env fuel)
                expand_head_for_join reduce_head_of_kind_naked_immediate
                (fuel_of_types fuel) Hfresh Heqb Hextension
                expand_head_for_join_sound Hreduce) as Hjt.
  (* The first phases. *)
  destruct (first_phases_true initial_tenv jt join_choices p (fuel_of_env fuel)
              avoid envs s1 s2 s3 relations2 relations3 Hjt Hfresh Heqb Hconst
              Hsource Huses Hinvariant Hkinded_envs Hlevels Hkinds Haliases
              Hdelayed Hloop) as [Htrue3 [Hrelations3 Hgrows3]].
  pose proof (first_phases_kinded initial_tenv jt join_choices p
                (fuel_of_env fuel) avoid envs s1 s2 s3 relations2 relations3
                Hjt Hfresh Heqb Hconst Hsource Huses Hinvariant Hkinded_envs
                Hlevels Hkinds Haliases Hdelayed Hloop) as Hstate3.
  assert (Hnonempty1 : relations_nonempty []).
  { intros r names' Hr. discriminate Hr. }
  pose proof (loop_relations_nonempty _ _ _ _ _ _ _ _ _ _
                (delayed_equations_relations_nonempty _ _ _ _ _ _ _ _
                   Hnonempty1 Hdelayed)
                Hloop) as Hnonempty3.
  pose proof (first_phases_scoped initial_tenv jt join_choices p
                (fuel_of_env fuel) avoid envs s1 s2 s3 relations2 relations3
                Hjt
                (n_way_join_keeps_scoped p (fuel_of_env fuel)
                   (fuel_of_types fuel) expand_head_for_join
                   reduce_head_of_kind_naked_immediate type_eqb
                   block_shape_eqb)
                Hfresh Heqb Hconst Hsource Huses Hinvariant Hkinded_envs
                Hlevels Hkinds Haliases Hdelayed Hloop) as Hscoped3.
  pose proof (first_phases_well_formed initial_tenv jt join_choices p
                (fuel_of_env fuel) avoid envs s1 s2 s3 relations2 relations3
                Hjt
                (n_way_join_of_types_well_formed p (fuel_of_env fuel)
                   expand_head_for_join reduce_head_of_kind_naked_immediate
                   (fuel_of_types fuel) Hfresh Hextension
                   expand_head_for_join_sound expand_head_for_join_well_formed
                   Hreduce)
                (n_way_join_of_types_keeps_well_formed p (fuel_of_env fuel)
                   expand_head_for_join reduce_head_of_kind_naked_immediate
                   (fuel_of_types fuel))
                Hfresh Heqb Hconst (join_parameters_well_formed join_depth)
                Hsource Huses Hinvariant Hkinded_envs Hwf_envs Hlevels Hkinds
                Haliases Hdelayed Hloop) as HW.
  pose proof (first_phases_equation_names initial_tenv jt join_choices p
                (fuel_of_env fuel) avoid envs s1 s2 s3 relations2 relations3
                Hjt
                (n_way_join_of_types_equations_kinded p (fuel_of_env fuel)
                   expand_head_for_join reduce_head_of_kind_naked_immediate
                   (fuel_of_types fuel) Hfresh Hextension
                   expand_head_for_join_sound
                   expand_head_for_join_equations_kinded Hreduce)
                (n_way_join_of_types_keeps_equation_names p (fuel_of_env fuel)
                   expand_head_for_join reduce_head_of_kind_naked_immediate
                   (fuel_of_types fuel) Hfresh)
                Hfresh Heqb Hconst (join_parameters_no_equations join_depth)
                Hsource Huses Hinvariant Hkinded_envs Hnames_envs Hlevels
                Hkinds Haliases Hdelayed Hloop) as HN.
  assert (Hinvariant3 : joined_envs_are typing_env_invariant s3).
  { intros index use Huse. rewrite (proj1 Hgrows3) in Huse.
    exact (Hinvariant index use Huse). }
  assert (Hkinded3 : joined_envs_are typing_env_kinded s3).
  { intros index use Huse. rewrite (proj1 Hgrows3) in Huse.
    exact (Hkinded_envs index use Huse). }
  assert (Hsymbols3 : symbols_of_joined_envs_in_source initial_tenv s3).
  { intros index use sym Huse Hmem. rewrite (proj1 Hgrows3) in Huse.
    exact (Hsymbols index use sym Huse Hmem). }
  (* The last phase. *)
  intros n ty Hin. split.
  - exact (last_phase_adds_well_formed tenv (fuel_of_env fuel) s3 relations3
             kept equations relation_equations definitions (proj1 HW) Hadds n
             ty Hin).
  - exact (last_phase_equation_names_known tenv (fuel_of_env fuel)
             initial_tenv s3 relations3
             (Meet_env.with_names_to_avoid initial_env
                (Bindings_in_target_env.names_to_avoid (Join_env.bindings s3)))
             kept equations relation_equations definitions Htrue3 Hrelations3
             Hnonempty3 Hscoped3 Hstate3 Hinvariant3 Hkinded3 Hsymbols3 HN
             eq_refl Hfits Hadds n ty Hin).
Qed.

Theorem model_join_adds_types_that_fit_otherwise
  : model_join_adds_types_that_fit_otherwise_statement.
Proof.
  intros join_choices join_depth fuel initial_env scoped_envs envs s1 s2 s3
    relations2 relations3 tenv kept equations relation_equations definitions
    initial_tenv p jt avoid s0 Hfits Hscoped Hagree Hcut Haliases Hdelayed
    Hloop Hadds.
  (* The environments joined. *)
  assert (Hfound :
            ∀ (index : Index.t) (use : Typing_env.t),
              Joined_envs.get_nth_joined_env envs index = Some use ->
              use ∈ scoped_envs).
  { intros index use Huse. apply get_nth_lookup in Huse.
    destruct Huse as [equations0 Hlookup].
    destruct (cut_for_joins_found _ _ _ Hcut _ _ _ Hlookup) as [Hin _].
    apply number_in in Hin. exact (nth_error_In _ _ Hin). }
  assert (Hsource : ∀ x : variable, knows initial_tenv x -> x ∈ avoid).
  { intros x Hx. apply in_or_app. right. apply in_or_app. left.
    apply variables_of_env_knows. exact Hx. }
  assert (Huses :
            ∀ (index : Index.t) (use : Typing_env.t) (x : variable),
              Joined_envs.get_nth_joined_env envs index = Some use ->
              knows use x ->
              x ∈ avoid).
  { intros index use x Huse Hx. apply in_or_app. right. apply in_or_app.
    right. apply in_flat_map. exists use.
    split; [ exact (Hfound index use Huse)
           | apply variables_of_env_knows; exact Hx ]. }
  assert (Hinvariant :
            ∀ (index : Index.t) (use : Typing_env.t),
              Joined_envs.get_nth_joined_env envs index = Some use ->
              typing_env_invariant use).
  { intros index use Huse.
    exact (proj1 (proj1 (Hscoped use (Hfound index use Huse)))). }
  assert (Hwf_envs :
            ∀ (index : Index.t) (use : Typing_env.t),
              Joined_envs.get_nth_joined_env envs index = Some use ->
              types_well_formed use).
  { intros index use Huse.
    exact (proj1 (proj2 (proj1 (Hscoped use (Hfound index use Huse))))). }
  assert (Hkinded_envs :
            ∀ (index : Index.t) (use : Typing_env.t),
              Joined_envs.get_nth_joined_env envs index = Some use ->
              typing_env_kinded use).
  { intros index use Huse.
    exact (proj1 (proj2 (proj2 (proj1 (Hscoped use
                                         (Hfound index use Huse)))))). }
  assert (Hnames_envs :
            ∀ (index : Index.t) (use : Typing_env.t),
              Joined_envs.get_nth_joined_env envs index = Some use ->
              types_equations_kinded use).
  { intros index use Huse.
    exact (proj1 (proj2 (proj2 (proj2 (proj1 (Hscoped use
                                                (Hfound index use
                                                   Huse))))))). }
  assert (Hlevels : levels_found envs).
  { intros index use equations0 n ty Hlookup Hn.
    destruct (cut_for_joins_found _ _ _ Hcut _ _ _ Hlookup)
      as [Hin [_ [level [Hlevel ->]]]].
    apply number_in in Hin. apply nth_error_In in Hin.
    exact (cut_equations_found use (Typing_env.current_scope initial_tenv)
             level n ty (proj2 (proj1 (proj1 (Hscoped use Hin)))) Hlevel
             Hn). }
  assert (Hkinds :
            ∀ (index : Index.t) (use : Typing_env.t) (x : variable)
              (kind : flambda_kind) (ty : flambda_type)
              (binding_time_and_mode : Binding_time.With_name_mode.t),
              Joined_envs.get_nth_joined_env envs index = Some use ->
              simple_of_kind use (Var x) kind ->
              Keyed.lookup name_eqb (Name_var x)
                (Typing_env.names_to_types initial_tenv)
              = Some (ty, binding_time_and_mode) ->
              FlambdaTypeGrammar.kind ty = kind).
  { intros index use x kind ty btm Huse Hk Hx.
    assert (Hknown : Typing_env.mem_simple None initial_tenv (Var x) = true).
    { apply (proj2 (mem_var_lookup initial_tenv x)). exists ty, btm.
      exact Hx. }
    pose proof (scoped_env_of_kind initial_tenv use (Var x) kind
                  (Hscoped use (Hfound index use Huse)) Hknown Hk) as Hk0.
    apply (simple_of_kind_unique initial_tenv (Var x)); [ | exact Hk0 ].
    change (Var x) with (simple_of_name (Name_var x)).
    apply simple_of_kind_name_iff. left. exists ty, btm.
    split; [ exact Hx | reflexivity ]. }
  exact (phases_add_types_that_fit join_choices join_depth fuel avoid
           initial_env envs s1 s2 s3 relations2 relations3 tenv kept equations
           relation_equations definitions Hfits Hsource Huses Hinvariant
           Hwf_envs Hkinded_envs Hnames_envs Hlevels Hkinds
           (fun index use sym Huse Hmem =>
              proj2 (proj2 (proj2 (Hscoped use (Hfound index use Huse)))) sym
                Hmem)
           Haliases Hdelayed Hloop Hadds).
Qed.

Theorem model_join_scoped_envs_keeps_outright
  : model_join_scoped_envs_keeps_outright_statement.
Proof.
  intros join_choices join_depth fuel.
  exact (model_join_scoped_envs_keeps join_choices join_depth fuel
           (model_join_adds_types_that_fit_otherwise join_choices join_depth
              fuel)).
Qed.

Theorem model_join_scoped_envs_satisfied_outright
  : model_join_scoped_envs_satisfied_outright_statement.
Proof.
  intros join_choices join_depth fuel.
  exact (model_join_scoped_envs_satisfied join_choices join_depth fuel
           (model_join_adds_types_that_fit_otherwise join_choices join_depth
              fuel)).
Qed.

(* ---------------------------------------------------------------------- *)
(* The join that Simplify calls                                            *)
(* ---------------------------------------------------------------------- *)

Theorem model_join_run_adds_types_that_fit_otherwise
  : model_join_run_adds_types_that_fit_otherwise_statement.
Proof.
  intros A join_choices join_depth fuel avoid inputs envs s3 relations3 kept
    equations relation_equations definitions Hinputs Hkinded Hagree Hfits
    Hfit_uses
    [s1 [s2 [relations2 [Hcut [Haliases [Hdelayed [Hloop Hadds]]]]]]].
  change (Join_env.cut_for_joins (cut_after inputs) (numbered_uses inputs)
          = Some envs) in Hcut.
  destruct Hinputs
    as [[_ Huses_wf] [_ [_ [Hsymbols [Havoid_source Havoid_uses]]]]].
  assert (Hfound :
            ∀ (index : Index.t) (use : Typing_env.t),
              Joined_envs.get_nth_joined_env envs index = Some use ->
              use_at inputs index use ∧ Typing_env.is_bottom use = false).
  { intros index use Huse. apply get_nth_lookup in Huse.
    destruct Huse as [equations0 Hlookup].
    destruct (cut_for_joins_found _ _ _ Hcut _ _ _ Hlookup) as [Hin [Hb _]].
    split; [ apply numbered_uses_in; exact Hin | exact Hb ]. }
  assert (Hsource : ∀ x : variable, knows (source inputs) x -> x ∈ avoid).
  { intros x Hx. apply among_In. exact (Havoid_source x Hx). }
  assert (Huses :
            ∀ (index : Index.t) (use : Typing_env.t) (x : variable),
              Joined_envs.get_nth_joined_env envs index = Some use ->
              knows use x ->
              x ∈ avoid).
  { intros index use x Huse Hx. apply among_In.
    exact (Havoid_uses index use x (proj1 (Hfound index use Huse)) Hx). }
  assert (Hinvariant :
            ∀ (index : Index.t) (use : Typing_env.t),
              Joined_envs.get_nth_joined_env envs index = Some use ->
              typing_env_invariant use).
  { intros index use Huse.
    exact (Huses_wf index use (proj1 (Hfound index use Huse))). }
  assert (Hwf_envs :
            ∀ (index : Index.t) (use : Typing_env.t),
              Joined_envs.get_nth_joined_env envs index = Some use ->
              types_well_formed use).
  { intros index use Huse. destruct (Hfound index use Huse) as [Hat Hb].
    exact (proj1 (Hfit_uses index use Hat Hb)). }
  assert (Hkinded_envs :
            ∀ (index : Index.t) (use : Typing_env.t),
              Joined_envs.get_nth_joined_env envs index = Some use ->
              typing_env_kinded use).
  { intros index use Huse.
    exact (Hkinded index use (proj1 (Hfound index use Huse))). }
  assert (Hnames_envs :
            ∀ (index : Index.t) (use : Typing_env.t),
              Joined_envs.get_nth_joined_env envs index = Some use ->
              types_equations_kinded use).
  { intros index use Huse. destruct (Hfound index use Huse) as [Hat Hb].
    exact (proj2 (Hfit_uses index use Hat Hb)). }
  assert (Hlevels : levels_found envs).
  { intros index use equations0 n ty Hlookup Hn.
    destruct (cut_for_joins_found _ _ _ Hcut _ _ _ Hlookup)
      as [Hin [_ [level [Hlevel ->]]]].
    apply numbered_uses_in in Hin.
    exact (cut_equations_found use (cut_after inputs) level n ty
             (proj2 (Huses_wf index use Hin)) Hlevel Hn). }
  assert (Hkinds :
            ∀ (index : Index.t) (use : Typing_env.t) (x : variable)
              (kind : flambda_kind) (ty : flambda_type)
              (binding_time_and_mode : Binding_time.With_name_mode.t),
              Joined_envs.get_nth_joined_env envs index = Some use ->
              simple_of_kind use (Var x) kind ->
              Keyed.lookup name_eqb (Name_var x)
                (Typing_env.names_to_types (source inputs))
              = Some (ty, binding_time_and_mode) ->
              FlambdaTypeGrammar.kind ty = kind).
  { intros index use x kind ty btm Huse Hk Hx.
    assert (Hsrc :
              simple_of_kind (source inputs) (Var x)
                (FlambdaTypeGrammar.kind ty)).
    { apply (simple_of_kind_name (source inputs) (Name_var x)). left.
      exists ty, btm. split; [ exact Hx | reflexivity ]. }
    destruct Hsrc as [ty0 Hfind0].
    destruct (Hagree index use (Name_var x) (FlambdaTypeGrammar.kind ty) ty0
                (proj1 (Hfound index use Huse))
                (simple_of_kind_var_known use x kind Hk) Hfind0)
      as [ty' Hfind'].
    exact (simple_of_kind_unique use (Var x) (FlambdaTypeGrammar.kind ty) kind
             (ex_intro _ ty' Hfind') Hk). }
  exact (phases_add_types_that_fit join_choices join_depth fuel avoid
           (Meet_env.create (source inputs)) envs s1 s2 s3 relations2
           relations3 _ kept equations relation_equations definitions Hfits
           Hsource Huses Hinvariant Hwf_envs Hkinded_envs Hnames_envs Hlevels
           Hkinds
           (fun index use sym Huse Hmem =>
              Hsymbols index use sym (proj1 (Hfound index use Huse)) Hmem)
           Haliases Hdelayed Hloop Hadds).
Qed.

(* The run of the join that Simplify calls, found from its answer,
   where there is an environment to join. *)
Lemma model_join_run_found {A : Type} (join_choices : Join_env.choices)
    (meet_choices : meet_choices) (join_depth : nat) (fuel : fuels)
    (avoid : list variable) (inputs : join_inputs A) (target : Typing_env.t)
    (analysis : Analysis.t A) :
  model_join join_choices meet_choices join_depth fuel avoid inputs
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
  unfold model_join, join_with, cut_and_n_way_join_with_analysis in Hjoin'.
  cbv zeta in Hjoin'.
  destruct (Join_env.cut_for_joins _ _) as [envs | ] eqn:Hcut;
    [ | discriminate ].
  clear Hjoin'.
  change (Join_env.cut_for_joins (cut_after inputs) (numbered_uses inputs)
          = Some envs) in Hcut.
  exists envs. split; [ reflexivity | ].
  destruct (Keyed.is_empty envs) eqn:Hempty; [ left; reflexivity | right ].
  destruct (model_join_unfold join_choices meet_choices join_depth fuel avoid
              inputs target analysis envs Hjoin Hcut)
    as [target_env [definitions [names [Hjoin0 _]]]].
  destruct (cut_and_n_way_join0_replayed Meet_env.t meet_expanded_head
              meet_env_functions (join_parameters join_depth)
              (fuel_of_env fuel) join_choices
              (model_join_of_types join_depth fuel)
              (model_meet_expanded_head join_choices meet_choices join_depth
                 fuel)
              avoid (Meet_env.create (source inputs)) target_env
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

Theorem model_join_keeps_outright : model_join_keeps_outright_statement.
Proof.
  intros A join_choices meet_choices join_depth fuel avoid inputs target
    analysis Hmh Hinputs Hkinded Hagree Hfits Hfit_uses Hjoin.
  destruct (model_join_run_found join_choices meet_choices join_depth fuel
              avoid inputs target analysis Hjoin)
    as [envs [Hcut [Hempty | [s3 [relations3 [kept [equations
          [relation_equations [definitions Hrun]]]]]]]]].
  { (* No environment to join: the target is bottom. *)
    destruct (model_join_unfold join_choices meet_choices join_depth fuel
                avoid inputs target analysis envs Hjoin Hcut)
      as [target_env [definitions [names [Hjoin0 ->]]]].
    unfold Join_env.cut_and_n_way_join0 in Hjoin0. rewrite Hempty in Hjoin0.
    injection Hjoin0 as <- _ _.
    cbn [Join_env.make_bottom meet_env_functions].
    destruct (make_bottom_keeps (Meet_env.create (source inputs)) Hfits)
      as [Hextends Hfits'].
    split; [ exact Hfits' | ].
    split; [ exact (extends_keeps_source _ _ Hextends) | ].
    intros x Hx Hnot. exfalso. apply Hnot. exact Hx. }
  exact (model_join_keeps A join_choices meet_choices join_depth fuel avoid
           inputs target analysis envs s3 relations3 kept equations
           relation_equations definitions Hmh Hinputs Hkinded Hagree Hfits
           Hjoin Hrun
           (model_join_run_adds_types_that_fit_otherwise A join_choices
              join_depth fuel avoid inputs envs s3 relations3 kept equations
              relation_equations definitions Hinputs Hkinded Hagree Hfits
              Hfit_uses Hrun)).
Qed.

Theorem model_join_describes_uses_outright
  : model_join_describes_uses_outright_statement.
Proof.
  intros A join_choices meet_choices join_depth fuel avoid inputs target
    analysis Hmh Hinputs Hkinded Hagree Hfits Hfit_uses Hjoin.
  destruct (model_join_run_found join_choices meet_choices join_depth fuel
              avoid inputs target analysis Hjoin)
    as [envs [Hcut [Hempty | [s3 [relations3 [kept [equations
          [relation_equations [definitions Hrun]]]]]]]]].
  { (* No environment to join: the analysis has no definition. *)
    pose proof (model_join_external_ids join_choices meet_choices join_depth
                  fuel avoid inputs target analysis Hjoin) as Hids.
    destruct (model_join_unfold join_choices meet_choices join_depth fuel
                avoid inputs target analysis envs Hjoin Hcut)
      as [target_env [definitions' [names [Hjoin0 Htarget]]]].
    destruct (model_join_analysis join_choices meet_choices join_depth fuel
                avoid inputs target analysis envs target_env definitions'
                names Hjoin Hcut Hjoin0)
      as [at_normal_mode [Hnormal [Hdefs Hat]]].
    unfold Join_env.cut_and_n_way_join0 in Hjoin0. rewrite Hempty in Hjoin0.
    injection Hjoin0 as _ <- _.
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
  exact (model_join_describes_uses A join_choices meet_choices join_depth
           fuel avoid inputs target analysis envs s3 relations3 kept equations
           relation_equations definitions Hmh Hinputs Hkinded Hagree Hfits
           Hjoin Hrun
           (model_join_run_adds_types_that_fit_otherwise A join_choices
              join_depth fuel avoid inputs envs s3 relations3 kept equations
              relation_equations definitions Hinputs Hkinded Hagree Hfits
              Hfit_uses Hrun)).
Qed.

Theorem model_join_with_admits_use_outright
  : model_join_with_admits_use_outright_statement.
Proof.
  intros A join_choices join_depth fuel mh avoid inputs target analysis envs
    s3 relations3 kept equations relation_equations definitions index use
    statics valuation Hmh Hsound Hinputs Hkinded Hagree Hfits Hfit_uses Hjoin
    Hrun.
  exact (model_join_with_admits_use A join_choices join_depth fuel mh avoid
           inputs target analysis envs s3 relations3 kept equations
           relation_equations definitions index use statics valuation Hmh
           Hsound Hinputs Hkinded Hagree Hfits Hjoin Hrun
           (model_join_run_adds_types_that_fit_otherwise A join_choices
              join_depth fuel avoid inputs envs s3 relations3 kept equations
              relation_equations definitions Hinputs Hkinded Hagree Hfits
              Hfit_uses Hrun)).
Qed.

(* ---------------------------------------------------------------------- *)
(* The join and the meet of the model together                             *)
(* ---------------------------------------------------------------------- *)

Theorem model_meet_expanded_head_keeps
  : model_meet_expanded_head_keeps_statement.
Proof.
  intros join_choices meet_choices join_depth fuel Htag Hagain Hlift.
  exact (meet_expanded_head_fuel_keeps
           (model_join_scoped_envs join_choices join_depth fuel) meet_choices
           (fuel_of_meet fuel)
           (model_join_scoped_envs_keeps_outright join_choices join_depth
              fuel)
           Htag Hagain Hlift).
Qed.

Theorem model_meet_keeps_terms : model_meet_keeps_terms_statement.
Proof.
  intros join_choices meet_choices join_depth fuel Htag Hagain Hlift.
  exact (meet_of_fuel_keeps_terms
           (model_join_scoped_envs join_choices join_depth fuel) meet_choices
           (fuel_of_meet fuel)
           (model_meet_expanded_head_keeps join_choices meet_choices
              join_depth fuel Htag Hagain Hlift)).
Qed.

Theorem model_join_guarantees_on_inputs
  : model_join_guarantees_on_inputs_statement.
Proof.
  intros A join_choices meet_choices join_depth fuel avoid inputs target
    analysis Htag Hagain Hlift Hinputs Hkinded Hagree Hfits Hfit_uses Hjoin.
  pose proof (model_meet_expanded_head_keeps join_choices meet_choices
                join_depth fuel Htag Hagain Hlift) as Hmh.
  destruct (model_join_keeps_outright A join_choices meet_choices join_depth
              fuel avoid inputs target analysis Hmh Hinputs Hkinded Hagree
              Hfits Hfit_uses Hjoin) as [Htarget [Hkeeps Hintroduces]].
  split; [ exact Htarget | ]. split; [ exact Hkeeps | ].
  split; [ exact Hintroduces | ].
  exact (model_join_describes_uses_outright A join_choices meet_choices
           join_depth fuel avoid inputs target analysis Hmh Hinputs Hkinded
           Hagree Hfits Hfit_uses Hjoin).
Qed.
