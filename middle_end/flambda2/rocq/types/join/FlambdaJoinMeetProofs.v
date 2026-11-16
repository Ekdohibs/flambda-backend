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


(* The proofs of the statements of FlambdaJoinMeetSpec.v. This file
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
  FlambdaTypeAliasKindsSpec
  FlambdaTypeAliasKindsProofs FlambdaJoinTypes
  FlambdaJoinTypesSpec FlambdaJoinTypesWithEnvSpec
  FlambdaJoinTypesWithEnvProofs FlambdaMeetGrammar FlambdaExpandHead
  FlambdaExpandHeadSpec FlambdaExpandHeadProofs FlambdaMeetEnv FlambdaMeet
  FlambdaMeetSpec FlambdaMeetStepsSpec FlambdaMeetStepsProofs
  FlambdaMeetEnvSpec FlambdaMeetEnvProofs FlambdaMeetScopesProofs
  FlambdaJoinMeet
  FlambdaJoinMeetSpec FlambdaMeetFuelSpec FlambdaMeetFuelProofs
  FlambdaMeetClosuresSpec FlambdaMeetClosuresProofs FlambdaMeetExamplesSpec
  FlambdaMeetExamplesProofs FlambdaEnvAliases FlambdaEnvAliasesSpec
  FlambdaEnvAliasesProofs FlambdaTypingEnvOperationsSpec
  FlambdaTypingEnvOperationsProofs FlambdaTypingEnvProofs
  FlambdaTypeMeaningProofs.
Import ListNotations.

Theorem join_expanded_to_type : join_expanded_to_type_statement.
Proof.
  intros [k [ | [head | head | k' head | head | head] | ]];
    split; reflexivity.
Qed.

Theorem join_parameters_sound : join_parameters_sound_statement.
Proof.
  intros join_depth. split; [ | split; [ | split ] ].
  - intros names Hin. cbn [Join_env.fresh join_parameters] in Hin.
    pose proof (fresh_name_fresh names) as Hfresh.
    apply (proj2 (among_In _ _)) in Hin. congruence.
  - intros t t' statics valuation v Heqb Hsat.
    exact (proj2 (type_equal_satisfied statics valuation v t t' Heqb) Hsat).
  - intros extension extension' statics valuation Heqb Hsat.
    exact (proj1 (env_extension_equal_satisfied statics valuation extension
                    extension' Heqb) Hsat).
  - intros c. cbn [Join_env.expand_const join_parameters].
    split; [ exact (create_const_free_variables c) | ].
    split; [ intros tenv; exact (create_const_kinded tenv c) | ].
    intros statics valuation. exact (create_const_sound statics valuation c).
Qed.

Theorem expand_head_for_join_sound : expand_head_for_join_sound_statement.
Proof.
  intros tenv ty expanded Hexpand. unfold expand_head_for_join in Hexpand.
  destruct (FlambdaExpandHead.expand_head tenv ty) as [expanded0 | ] eqn:H0;
    [ | discriminate ].
  injection Hexpand as <-.
  destruct (join_expanded_to_type expanded0) as [Htype Hkind].
  rewrite Htype, Hkind.
  split; [ | split; [ | split ] ].
  - intros statics valuation v Hsat Hty.
    exact (FlambdaExpandHeadProofs.expand_head_sound statics valuation tenv
             ty expanded0 v Hsat Hty H0).
  - intros Hinvariant Hscope x Hx.
    exact (expand_head_closed tenv ty expanded0
             (invariant_types_closed tenv Hinvariant) Hscope H0 x Hx).
  - intros Hkinded Hty.
    exact (expand_head_kinded tenv ty expanded0 (proj1 Hkinded) Hty H0).
  - intros Hconstants Hty.
    exact (expand_head_kind tenv ty expanded0 Hconstants Hty H0).
Qed.

Theorem model_join_true : model_join_true_statement.
Proof.
  intros A join_choices meet_choices join_depth fuel avoid inputs target
    analysis Hinputs Hkinded Hagree Hsome Hjoin.
  destruct (join_parameters_sound join_depth)
    as [Hfresh [Heqb [Hextension Hconst]]].
  assert (Hreduce :
            FlambdaJoinTypesSpec.reduce_head_of_kind_naked_immediate_sound
              reduce_head_of_kind_naked_immediate).
  { intros tenv statics valuation v head Hsat Hhead.
    exact (FlambdaMeetStepsProofs.reduce_head_of_kind_naked_immediate_sound
             statics valuation tenv head v Hsat Hhead). }
  exact (join_with_true A Meet_env.t meet_expanded_head meet_env_functions
           (join_parameters join_depth) (fuel_of_env fuel) join_choices
           (model_join_of_types join_depth fuel)
           (model_meet_expanded_head join_choices meet_choices join_depth
              fuel)
           avoid inputs target analysis
           (n_way_join_of_types_true (source inputs)
              (join_parameters join_depth) (fuel_of_env fuel)
              expand_head_for_join reduce_head_of_kind_naked_immediate
              (fuel_of_types fuel) Hfresh Heqb Hextension
              expand_head_for_join_sound Hreduce)
           (n_way_join_keeps_envs (join_parameters join_depth)
              (fuel_of_env fuel) (fuel_of_types fuel) expand_head_for_join
              reduce_head_of_kind_naked_immediate type_eqb block_shape_eqb)
           Hfresh Heqb Hconst Hinputs Hkinded Hagree Hsome Hjoin).
Qed.

(* ---------------------------------------------------------------------- *)
(* The variables that the join defines, in the environment of the meet     *)
(* ---------------------------------------------------------------------- *)

Theorem variables_defined_kept : variables_defined_kept_statement.
Proof.
  intros vars.
  induction vars as [ | [x k] vars IH]; intros menv Hfits Hdistinct Hfresh.
  - cbn. split; [ split; [ apply meet_env_extends_refl | exact Hfits ] | ].
    split; [ intros x kind [] | ].
    split; [ | reflexivity ].
    intros x. split; [ intros H; left; exact H | intros [H | []]; exact H ].
  - change (variables_defined meet_env_functions menv ((x, k) :: vars))
      with (variables_defined meet_env_functions
              (Meet_env.add_variable_definition menv x k Name_mode.in_types)
              vars).
    set (menv1 := Meet_env.add_variable_definition menv x k Name_mode.in_types).
    assert (Hx : Typing_env.mem None (Meet_env.typing_env menv) (Name_var x)
                 = false).
    { apply Hfresh. left. reflexivity. }
    destruct (add_variable_definition_keeps menv x k Name_mode.in_types Hfits
                Hx) as [Hextends1 Hfits1].
    fold menv1 in Hextends1, Hfits1.
    cbn [map fst] in Hdistinct. apply NoDup_cons_iff in Hdistinct.
    destruct Hdistinct as [Hnotin Hdistinct].
    assert (Hother :
              ∀ y : variable,
                y <> x ->
                Typing_env.mem None (Meet_env.typing_env menv1) (Name_var y)
                = Typing_env.mem None (Meet_env.typing_env menv) (Name_var y)).
    { intros y Hne.
      exact (mem_add_variable_definition (Meet_env.typing_env menv) x k
               Name_mode.in_types y Hne). }
    assert (Hkind1 : simple_of_kind (Meet_env.typing_env menv1) (Var x) k).
    { apply (proj2 (add_variable_definition_simple_of_kind
                      (Meet_env.typing_env menv) x k Name_mode.in_types
                      (Var x) k)).
      left. split; reflexivity. }
    assert (Hmem1 : Typing_env.mem None (Meet_env.typing_env menv1)
                      (Name_var x)
                    = true).
    { exact (simple_of_kind_mem (Meet_env.typing_env menv1) (Name_var x) k
               (proj1 Hfits1) Hkind1). }
    assert (Hfresh1 :
              ∀ y : variable,
                y ∈ map fst vars ->
                Typing_env.mem None (Meet_env.typing_env menv1) (Name_var y)
                = false).
    { intros y Hy. rewrite Hother.
      - apply Hfresh. right. exact Hy.
      - intros ->. exact (Hnotin Hy). }
    destruct (IH menv1 Hfits1 Hdistinct Hfresh1)
      as [[Hextends Hfits'] [Hkinds [Hmem Hnames]]].
    split.
    { split; [ exact (meet_env_extends_trans _ _ _ Hextends1 Hextends)
             | exact Hfits' ]. }
    split.
    { intros y kind [Heq | Hin].
      - injection Heq as <- <-.
        destruct Hextends as (_ & _ & _ & _ & Hkind & _).
        exact (Hkind _ _ Hkind1).
      - exact (Hkinds y kind Hin). }
    split; [ | rewrite Hnames; reflexivity ].
    intros y. rewrite Hmem. cbn [map fst].
    destruct (Pos.eq_dec y x) as [-> | Hne].
    + split; [ intros _; right; left; reflexivity | ].
      intros _. left. exact Hmem1.
    + rewrite (Hother y Hne). split.
      * intros [H | H]; [ left; exact H | right; right; exact H ].
      * intros [H | [Heq | H]];
          [ left; exact H | congruence | right; exact H ].
Qed.

(* A variable that an environment knows is known at the same modes to
   an environment that extends it. *)
Lemma extends_mode (menv menv' : Meet_env.t) (x : variable)
    (mode : Name_mode.t) :
  meet_env_extends menv menv' ->
  Typing_env.mem None (Meet_env.typing_env menv) (Name_var x) = true ->
  Typing_env.mem (Some mode) (Meet_env.typing_env menv') (Name_var x)
  = Typing_env.mem (Some mode) (Meet_env.typing_env menv) (Name_var x).
Proof.
  intros (_ & _ & Hmin & _ & _ & _ & _ & Hkeeps) Hmem.
  apply mem_var_lookup in Hmem. destruct Hmem as [ty [btm Hlookup]].
  destruct (Hkeeps _ _ _ Hlookup) as [ty' [Hlookup' _]].
  unfold Typing_env.mem. rewrite Hlookup, Hlookup', Hmin. reflexivity.
Qed.

Lemma defined_in_types (tenv : Typing_env.t) (x : variable)
    (k : flambda_kind) :
  Typing_env.mem (Some Name_mode.normal)
    (Typing_env.add_variable_definition tenv x k Name_mode.in_types)
    (Name_var x)
  = false.
Proof.
  unfold Typing_env.mem. rewrite add_variable_definition_lookup.
  rewrite (proj2 (name_eqb_eq (Name_var x) (Name_var x)) eq_refl).
  unfold Binding_time.With_name_mode.scoped_name_mode.
  cbn [Binding_time.With_name_mode.name_mode].
  destruct (_ || _); reflexivity.
Qed.

Theorem variables_defined_not_in_terms
  : variables_defined_not_in_terms_statement.
Proof.
  intros vars.
  induction vars as [ | [y k] vars IH];
    intros menv x Hfits Hdistinct Hfresh Hx; [ destruct Hx | ].
  change (variables_defined meet_env_functions menv ((y, k) :: vars))
    with (variables_defined meet_env_functions
            (Meet_env.add_variable_definition menv y k Name_mode.in_types)
            vars).
  set (menv1 := Meet_env.add_variable_definition menv y k Name_mode.in_types).
  assert (Hy : Typing_env.mem None (Meet_env.typing_env menv) (Name_var y)
               = false).
  { apply Hfresh. left. reflexivity. }
  destruct (add_variable_definition_keeps menv y k Name_mode.in_types Hfits
              Hy) as [Hextends1 Hfits1].
  fold menv1 in Hextends1, Hfits1.
  cbn [map fst] in Hdistinct. apply NoDup_cons_iff in Hdistinct.
  destruct Hdistinct as [Hnotin Hdistinct].
  assert (Hfresh1 :
            ∀ z : variable,
              z ∈ map fst vars ->
              Typing_env.mem None (Meet_env.typing_env menv1) (Name_var z)
              = false).
  { intros z Hz.
    assert (Hne : z <> y) by (intros ->; exact (Hnotin Hz)).
    pose proof (mem_add_variable_definition (Meet_env.typing_env menv) y k
                  Name_mode.in_types z Hne) as Hsame.
    change (Typing_env.mem None (Meet_env.typing_env menv1) (Name_var z)
            = Typing_env.mem None (Meet_env.typing_env menv) (Name_var z))
      in Hsame.
    rewrite Hsame. apply Hfresh. right. exact Hz. }
  destruct Hx as [Heq | Hx];
    [ | exact (IH menv1 x Hfits1 Hdistinct Hfresh1 Hx) ].
  cbn in Heq. subst y.
  destruct (variables_defined_kept vars menv1 Hfits1 Hdistinct Hfresh1)
    as [[Hextends _] _].
  rewrite (extends_mode menv1 _ x Name_mode.normal Hextends).
  - exact (defined_in_types (Meet_env.typing_env menv) x k).
  - assert (Hkind1 : simple_of_kind (Meet_env.typing_env menv1) (Var x) k).
    { apply (proj2 (add_variable_definition_simple_of_kind
                      (Meet_env.typing_env menv) x k Name_mode.in_types
                      (Var x) k)).
      left. split; reflexivity. }
    exact (simple_of_kind_mem (Meet_env.typing_env menv1) (Name_var x) k
             (proj1 Hfits1) Hkind1).
Qed.

Lemma in_order_keys {A : Type} (m : finite_map A) (x : variable) :
  x ∈ map fst (rev (compact m)) <-> is_bound x m = true.
Proof.
  rewrite map_rev, <- in_rev. split.
  - intros Hx. apply in_map_iff in Hx. destruct Hx as [[y a] [Heq Hin]].
    cbn in Heq. subst y. apply is_bound_lookup. exists a.
    apply in_compact. exact Hin.
  - intros Hx. apply is_bound_lookup in Hx. destruct Hx as [a Ha].
    apply in_map_iff. exists (x, a).
    split; [ reflexivity | apply in_compact; exact Ha ].
Qed.

(* The variables brought in and variables that the join has made,
   each once: no variable twice, and none that the source knows. *)
Lemma target_variables_fresh (s : Join_env.t) (menv : Meet_env.t)
    (kept : list (variable * flambda_kind)) :
  state_valid s ->
  Meet_env.typing_env menv = Join_env.source_env s ->
  (∀ binding : variable * flambda_kind,
     binding ∈ kept ->
     binding
     ∈ Bindings_in_target_env.existential_variables_in_order
         (Join_env.bindings s)) ->
  NoDup (map fst kept) ->
  let vars :=
    Bindings_in_target_env.imported_variables_in_order (Join_env.bindings s)
    ++ kept
  in
  NoDup (map fst vars)
  ∧ ∀ x : variable,
      x ∈ map fst vars ->
      Typing_env.mem None (Meet_env.typing_env menv) (Name_var x) = false.
Proof.
  intros Hvalid Hsource Hkept Hdistinct.
  destruct Hvalid as [_ [Hnew [_ [_ Himports]]]].
  set (imported :=
         Bindings_in_target_env.imported_variables_in_order
           (Join_env.bindings s)) in *.
  cbv zeta.
  assert (Himported_keys :
            ∀ x : variable, x ∈ map fst imported <-> is_imported s x).
  { intros x. exact (in_order_keys _ x). }
  assert (Hkept_existential :
            ∀ x : variable, x ∈ map fst kept -> is_existential s x).
  { intros x Hx. apply in_map_iff in Hx. destruct Hx as [[y k] [Heq Hin]].
    cbn in Heq. subst y. apply is_bound_lookup. exists k.
    exact (in_order_lookup _ _ _ (Hkept _ Hin)). }
  assert (Hapart :
            ∀ x : variable, is_imported s x -> is_existential s x -> False).
  { intros x Hi He. destruct (Hnew x He) as [_ Huses].
    destruct (Himports x Hi) as [_ [index [use [Huse Hknows]]]].
    exact (Huses index use Huse Hknows). }
  split.
  - rewrite map_app. apply NoDup_app.
    + apply in_order_distinct.
    + exact Hdistinct.
    + intros x Hi Hk.
      exact (Hapart x (proj1 (Himported_keys x) Hi)
               (Hkept_existential x Hk)).
  - intros x Hx. rewrite Hsource. rewrite map_app in Hx.
    apply in_app_or in Hx.
    destruct (Typing_env.mem None (Join_env.source_env s) (Name_var x))
      eqn:Hmem; [ exfalso | reflexivity ].
    destruct Hx as [Hx | Hx].
    + exact (proj1 (Himports x (proj1 (Himported_keys x) Hx)) Hmem).
    + exact (proj1 (Hnew x (Hkept_existential x Hx)) Hmem).
Qed.

Theorem target_variables_defined : target_variables_defined_statement.
Proof.
  intros s menv kept Hvalid Hsource Hfits Hkept Hdistinct menv'.
  pose proof Hvalid as Hvalid0.
  destruct Hvalid as [_ [Hnew [_ [_ Himports]]]].
  set (imported :=
         Bindings_in_target_env.imported_variables_in_order
           (Join_env.bindings s)) in *.
  assert (Himported_keys :
            ∀ x : variable, x ∈ map fst imported <-> is_imported s x).
  { intros x. exact (in_order_keys _ x). }
  assert (Hkept_existential :
            ∀ x : variable, x ∈ map fst kept -> is_existential s x).
  { intros x Hx. apply in_map_iff in Hx. destruct Hx as [[y k] [Heq Hin]].
    cbn in Heq. subst y. apply is_bound_lookup. exists k.
    exact (in_order_lookup _ _ _ (Hkept _ Hin)). }
  assert (Hapart :
            ∀ x : variable, is_imported s x -> is_existential s x -> False).
  { intros x Hi He. destruct (Hnew x He) as [_ Huses].
    destruct (Himports x Hi) as [_ [index [use [Huse Hknows]]]].
    exact (Huses index use Huse Hknows). }
  destruct (target_variables_fresh s menv kept Hvalid0 Hsource Hkept
              Hdistinct) as [Hnodup Hfresh].
  fold imported in Hnodup, Hfresh.
  destruct (variables_defined_kept (imported ++ kept) menv Hfits Hnodup
              Hfresh) as [Hkept' [Hkinds [Hmem Hnames]]].
  fold menv' in Hkept', Hkinds, Hmem, Hnames.
  split; [ exact Hkept' | ]. split.
  { intros x. rewrite Hmem, Hsource, map_app.
    unfold in_final_target, knows. split.
    - intros [H | [H | H]];
        [ left; exact H
        | right; apply in_or_app; left; exact (proj2 (Himported_keys x) H)
        | right; apply in_or_app; right; exact H ].
    - intros [H | H]; [ left; exact H | ]. apply in_app_or in H.
      destruct H as [H | H];
        [ right; left; exact (proj1 (Himported_keys x) H)
        | right; right; exact H ]. }
  split; [ | exact Hnames ].
  intros x kind Hin Hkind. unfold kind_in_target_env in Hkind.
  destruct (Keyed.lookup name_eqb (Name_var x)
              (Typing_env.names_to_types
                 (Bindings_in_target_env.source_env (Join_env.bindings s))))
    as [[ty btm] | ] eqn:Hlookup.
  - injection Hkind as <-.
    destruct Hkept' as [Hextends _].
    destruct Hextends as (_ & _ & _ & _ & Hkind_kept & _).
    apply Hkind_kept. rewrite Hsource.
    change (Var x) with (simple_of_name (Name_var x)).
    apply simple_of_kind_name_iff. left. exists ty, btm.
    split; [ exact Hlookup | reflexivity ].
  - assert (Hunknown : ~ knows (Join_env.source_env s) x).
    { intros Hknows. apply mem_var_lookup in Hknows.
      destruct Hknows as [ty [btm H]]. unfold Join_env.source_env in H.
      rewrite Hlookup in H. discriminate. }
    destruct (lookup x
                (Bindings_in_target_env.existential_variables
                   (Join_env.bindings s))) as [k | ] eqn:Hexistential.
    + injection Hkind as <-.
      assert (Hex : is_existential s x).
      { apply is_bound_lookup. exists k. exact Hexistential. }
      destruct Hin as [H | [H | H]];
        [ exfalso; exact (Hunknown H) | exfalso; exact (Hapart x H Hex) | ].
      apply in_map_iff in H. destruct H as [[y k'] [Heq Hin']].
      cbn in Heq. subst y.
      pose proof (in_order_lookup _ _ _ (Hkept _ Hin')) as Hk'.
      rewrite Hexistential in Hk'. injection Hk' as <-.
      apply Hkinds. apply in_or_app. right. exact Hin'.
    + apply Hkinds. apply in_or_app. left.
      unfold imported, Bindings_in_target_env.imported_variables_in_order.
      apply in_rev. rewrite rev_involutive. apply in_compact. exact Hkind.
Qed.

Theorem last_phase_names_known : last_phase_names_known_statement.
Proof.
  intros tenv fuel source s relations menv kept equations relation_equations
    definitions Htrue Hrelations Hnonempty Hscoped Hsource Hfits Hadds menv'.
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
              Hkept Hdistinct) as [Hkept' [Hmem _]].
  split; [ exact Hkept' | ].
  intros n ty Hin.
  destruct (last_phase_adds_on_variables tenv fuel s relations kept equations
              relation_equations definitions Hadds n ty Hin) as [x ->].
  assert (Hboth :
            (∀ y : variable,
               y ∈ free_variables_of_name (Name_var x) ->
               in_final_target s kept y)
            ∧ ∀ y : variable,
                y ∈ free_variables ty -> in_final_target s kept y).
  { apply in_app_or in Hin. destruct Hin as [Hin | Hin];
      [ exact (Hscope1 _ _ Hin) | exact (Hscope2 _ _ Hin) ]. }
  split.
  - apply Hmem. apply (proj1 Hboth). left. reflexivity.
  - intros y Hy. apply Hmem. exact (proj2 Hboth y Hy).
Qed.

Theorem last_phase_kinds_known : last_phase_kinds_known_statement.
Proof.
  intros tenv fuel source s relations menv kept equations relation_equations
    definitions Htrue Hrelations Hnonempty Hscoped Hstate Hinvariant Henvs
    Hsymbols HX HR Hsource Hfits Hadds menv' n ty Hin.
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
  destruct HX as [H1 [H4 [_ [H5 _]]]].
  pose proof (scoped_equations_kinded s (proj1 Htrue) Hstate Hinvariant Henvs
                (proj1 Hscoped) H1) as Hkinded.
  destruct (last_phase_adds_kinded tenv fuel s relations kept equations
              relation_equations definitions (proj1 Htrue) Hkinded H4 H5 HR
              Hadds n ty Hin) as [Haliases [x [-> Hx]]].
  assert (Hboth :
            (∀ y : variable,
               y ∈ free_variables_of_name (Name_var x) ->
               in_final_target s kept y)
            ∧ ∀ y : variable,
                y ∈ free_variables ty -> in_final_target s kept y).
  { apply in_app_or in Hin. destruct Hin as [Hin | Hin];
      [ exact (Hscope1 _ _ Hin) | exact (Hscope2 _ _ Hin) ]. }
  split.
  - exact (Hkinds x (kind ty) (proj1 Hboth x (or_introl eq_refl)) Hx).
  - intros simple k Hmember.
    pose proof (Haliases simple k Hmember) as HK.
    destruct simple as [y | sym | c].
    + apply (Hkinds y k); [ | exact HK ].
      apply (proj2 Hboth).
      exact (proj1 aliases_free_variables ty (Var y) k y Hmember
               (or_introl eq_refl)).
    + destruct HK as [-> [index [use [Huse Hmem]]]].
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
    + exact HK.
Qed.

(* An equation that is found is one of the list. *)
Lemma lookup_among (n : name) (ty : flambda_type)
    (equations : list (name * flambda_type)) :
  Keyed.lookup name_eqb n equations = Some ty -> (n, ty) ∈ equations.
Proof. exact (keyed_lookup_in name_eqb name_eqb_eq n ty equations). Qed.

Theorem last_phase_kept : last_phase_kept_statement.
Proof.
  intros tenv fuel source s relations mh menv menv1 target kept equations
    relation_equations definitions Htrue Hrelations Hnonempty Hscoped Hstate
    Hinvariant Henvs Hsymbols HX HR Hsource Hfits Hadds menv0 Hrest Hmh Hadd1
    Hadd2.
  destruct (last_phase_names_known tenv fuel source s relations menv kept
              equations relation_equations definitions Htrue Hrelations
              Hnonempty Hscoped Hsource Hfits Hadds) as [Hkept0 Hnames].
  fold menv0 in Hkept0, Hnames.
  pose proof (last_phase_kinds_known tenv fuel source s relations menv kept
                equations relation_equations definitions Htrue Hrelations
                Hnonempty Hscoped Hstate Hinvariant Henvs Hsymbols HX HR
                Hsource Hfits Hadds)
    as Hkinds.
  cbv zeta in Hkinds. fold menv0 in Hkinds.
  assert (Hall :
            ∀ (n : name) (ty : flambda_type),
              (n, ty) ∈ equations ++ relation_equations ->
              Typing_env.mem None (Meet_env.typing_env menv0) n = true
              ∧ type_fits (Meet_env.typing_env menv0) ty
              ∧ simple_of_kind (Meet_env.typing_env menv0) (simple_of_name n)
                  (kind ty)).
  { intros n ty Hin.
    destruct (Hnames n ty Hin) as [Hn Hvars].
    destruct (Hkinds n ty Hin) as [Hkind Hkinded].
    destruct (Hrest n ty Hin) as [Hwf Heqs].
    split; [ exact Hn | ]. split; [ | exact Hkind ].
    split; [ exact Hwf | ]. split; [ exact Hvars | ].
    split; [ exact Hkinded | exact Heqs ]. }
  split; [ exact Hkept0 | ].
  assert (Hstep1 : meet_env_kept_by_meet menv0 menv1).
  { apply (add_env_extension_keeps mh menv0 menv1 (Env_extension equations)
             Hmh (proj2 Hkept0)); [ | | exact Hadd1 ].
    - intros n ty Hlookup.
      destruct (Hall n ty
                  (in_or_app _ _ _ (or_introl (lookup_among _ _ _ Hlookup))))
        as [Hn [Hty _]].
      split; assumption.
    - intros n ty Hlookup.
      exact (proj2 (proj2 (Hall n ty
                             (in_or_app _ _ _
                                (or_introl (lookup_among _ _ _ Hlookup)))))). }
  split; [ exact Hstep1 | ].
  pose proof (meet_env_extends_tenv _ _ (proj1 (proj1 Hstep1))) as Hextends.
  apply (add_env_extension_keeps mh menv1 target
           (Env_extension relation_equations) Hmh (proj2 (proj1 Hstep1)));
    [ | | exact Hadd2 ].
  - intros n ty Hlookup.
    destruct (Hall n ty
                (in_or_app _ _ _ (or_intror (lookup_among _ _ _ Hlookup))))
      as [Hn [Hty _]].
    split.
    + exact (tenv_extends_mem _ _ _ Hextends Hn).
    + exact (tenv_extends_type_fits _ _ _ Hextends Hty).
  - intros n ty Hlookup.
    apply (proj1 (proj2 (proj2 (proj2 (proj2 Hextends))))).
    exact (proj2 (proj2 (Hall n ty
                           (in_or_app _ _ _
                              (or_intror (lookup_among _ _ _ Hlookup)))))).
Qed.

(* ---------------------------------------------------------------------- *)
(* The target is satisfied, on the runs that have the conditions           *)
(* ---------------------------------------------------------------------- *)

(* A variable that an environment mentions is one that it knows. *)
Lemma mentions_known (tenv : Typing_env.t) (x : variable) :
  typing_env_invariant tenv ->
  typing_env_mentions tenv x ->
  Typing_env.mem None tenv (Name_var x) = true.
Proof.
  intros [Hwf _] [[ty [btm Hlookup]] | [[n [ty [btm [Hlookup Hx]]]] | Hx]].
  - apply mem_var_lookup. exists ty, btm. exact Hlookup.
  - pose proof Hwf as Hconditions.
    apply typing_env_well_formed_iff in Hconditions.
    destruct Hconditions as [[_ [_ [_ [Hclosed _]]]] _].
    rewrite mem_var_is_bound. exact (Hclosed n ty btm x Hlookup Hx).
  - exact (well_formed_mentioned_known tenv (Name_var x) Hwf Hx).
Qed.

(* A valuation that satisfies the environment of a meet satisfies it
   with variables defined, if the values that it has for them are of
   their kinds. *)
Lemma variables_defined_satisfied (statics : static_env) (valuation : env) :
  ∀ (vars : list (variable * flambda_kind)) (menv : Meet_env.t),
    typing_env_satisfied_strictly statics valuation (Meet_env.typing_env menv)
    ->
    (∀ (x : variable) (k : flambda_kind) (v : value),
       (x, k) ∈ vars -> lookup x valuation = Some v -> kind_of_value v = k) ->
    typing_env_satisfied_strictly statics valuation
      (Meet_env.typing_env (variables_defined meet_env_functions menv vars)).
Proof.
  induction vars as [ | [x k] vars IH]; intros menv Hsatisfied Hkinds;
    [ exact Hsatisfied | ].
  change (variables_defined meet_env_functions menv ((x, k) :: vars))
    with (variables_defined meet_env_functions
            (Meet_env.add_variable_definition menv x k Name_mode.in_types)
            vars).
  apply IH.
  - exact (add_variable_definition_satisfied (fun _ => False) statics
             valuation (Meet_env.typing_env menv) x k Name_mode.in_types
             Hsatisfied
             (fun v Hv => Hkinds x k v (or_introl eq_refl) Hv)).
  - intros y k' v Hin Hv. exact (Hkinds y k' v (or_intror Hin) Hv).
Qed.

(* A valuation that has what one has for the variables that an
   environment knows, and what another has for the others. *)
Definition patched (tenv : Typing_env.t) (inside outside : env) : env :=
  filter_bindings (fun x _ => Typing_env.mem None tenv (Name_var x)) inside
  ++ filter_bindings (fun x _ => negb (Typing_env.mem None tenv (Name_var x)))
       outside.

Lemma lookup_patched (tenv : Typing_env.t) (inside outside : env)
    (x : variable) :
  lookup x (patched tenv inside outside)
  = if Typing_env.mem None tenv (Name_var x)
    then lookup x inside
    else lookup x outside.
Proof.
  unfold patched. rewrite lookup_app, !lookup_filter_bindings.
  destruct (Typing_env.mem None tenv (Name_var x)); cbn [negb].
  - destruct (lookup x inside); [ reflexivity | ].
    destruct (lookup x outside); reflexivity.
  - destruct (lookup x inside); destruct (lookup x outside); reflexivity.
Qed.

(* Defining variables that the environment does not know, each once,
   adds no equation. *)
Lemma variables_defined_valued (valuation : env) :
  ∀ (vars : list (variable * flambda_kind)) (menv : Meet_env.t),
    typing_env_fits (Meet_env.typing_env menv) ->
    NoDup (map fst vars) ->
    (∀ x : variable,
       x ∈ map fst vars ->
       Typing_env.mem None (Meet_env.typing_env menv) (Name_var x) = false) ->
    keeps_equations_valued (Meet_env.typing_env menv)
      (Meet_env.typing_env (variables_defined meet_env_functions menv vars))
      valuation valuation.
Proof.
  induction vars as [ | [y k] vars IH]; intros menv Hfits Hdistinct Hfresh;
    [ apply keeps_equations_valued_refl | ].
  change (variables_defined meet_env_functions menv ((y, k) :: vars))
    with (variables_defined meet_env_functions
            (Meet_env.add_variable_definition menv y k Name_mode.in_types)
            vars).
  assert (Hy : Typing_env.mem None (Meet_env.typing_env menv) (Name_var y)
               = false).
  { apply Hfresh. left. reflexivity. }
  destruct (define_fresh menv y k Hfits Hy)
    as [[_ Hfits1] [_ [_ [_ Hvalued1]]]].
  cbn [map fst] in Hdistinct. apply NoDup_cons_iff in Hdistinct.
  destruct Hdistinct as [Hnotin Hdistinct].
  apply (keeps_equations_valued_trans _ _ _ _ _ _ (Hvalued1 valuation)).
  apply IH; [ exact Hfits1 | exact Hdistinct | ].
  intros z Hz.
  assert (Hne : z <> y) by (intros ->; exact (Hnotin Hz)).
  pose proof (mem_add_variable_definition (Meet_env.typing_env menv) y k
                Name_mode.in_types z Hne) as Hsame.
  change (Typing_env.mem None
            (Meet_env.typing_env
               (Meet_env.add_variable_definition menv y k Name_mode.in_types))
            (Name_var z)
          = Typing_env.mem None (Meet_env.typing_env menv) (Name_var z))
    in Hsame.
  rewrite Hsame. apply Hfresh. right. exact Hz.
Qed.

Theorem last_phase_satisfied : last_phase_satisfied_statement.
Proof.
  intros tenv fuel source s relations mh menv menv1 target kept equations
    relation_equations definitions index use statics valuation Htrue
    Hrelations Hnonempty Hscoped Hstate Hinvariant Henvs Hsymbols HX HR
    Hsource Hfits Hnames Hadds menv0 Hrest Hmh Hsound Hadd1 Hadd2 Huse
    Hstatics Hsat_source Hsat_use Htags Haliases Hscrutinees tv Hvalued.
  pose proof (proj1 Htrue) as Hvalid.
  (* The equations are true. *)
  destruct (last_phase_adds_true tenv fuel source s relations kept equations
              relation_equations definitions Htrue Hstate Henvs Hrelations
              Hnonempty Hadds index use statics valuation Huse Hsat_use Htags
              Haliases Hscrutinees) as [Htrue1 [Htrue2 Hdefs_tv]].
  fold tv in Htrue1, Htrue2, Hdefs_tv.
  (* The extensions fit. *)
  destruct (last_phase_kept tenv fuel source s relations mh menv menv1 target
              kept equations relation_equations definitions Htrue Hrelations
              Hnonempty Hscoped Hstate Hinvariant Henvs Hsymbols HX HR Hsource
              Hfits Hadds Hrest Hmh Hadd1 Hadd2) as [Hkept0 [Hstep1 Hstep2]].
  fold menv0 in Hkept0, Hstep1.
  destruct (last_phase_names_known tenv fuel source s relations menv kept
              equations relation_equations definitions Htrue Hrelations
              Hnonempty Hscoped Hsource Hfits Hadds) as [_ Hknown].
  fold menv0 in Hknown.
  pose proof (last_phase_kinds_known tenv fuel source s relations menv kept
                equations relation_equations definitions Htrue Hrelations
                Hnonempty Hscoped Hstate Hinvariant Henvs Hsymbols HX HR
                Hsource Hfits Hadds) as Hkinds.
  cbv zeta in Hkinds. fold menv0 in Hkinds.
  assert (Hall :
            ∀ (n : name) (ty : flambda_type),
              (n, ty) ∈ equations ++ relation_equations ->
              Typing_env.mem None (Meet_env.typing_env menv0) n = true
              ∧ type_fits (Meet_env.typing_env menv0) ty
              ∧ simple_of_kind (Meet_env.typing_env menv0) (simple_of_name n)
                  (kind ty)).
  { intros n ty Hin.
    destruct (Hknown n ty Hin) as [Hn Hvars].
    destruct (Hkinds n ty Hin) as [Hkind Hkinded].
    destruct (Hrest n ty Hin) as [Hwf Heqs].
    split; [ exact Hn | ]. split; [ | exact Hkind ].
    split; [ exact Hwf | ]. split; [ exact Hvars | ].
    split; [ exact Hkinded | exact Heqs ]. }
  pose proof (meet_env_extends_tenv _ _ (proj1 (proj1 Hstep1))) as Hextends1.
  (* The valuation of the target satisfies the source. *)
  destruct (target_valuation_extends_at s index use valuation Hvalid Huse)
    as [_ Hagrees].
  rewrite (proj2 (proj2 Htrue)) in Hagrees. fold tv in Hagrees.
  assert (Hsat_tv : typing_env_satisfied_strictly statics tv source).
  { apply (satisfied_but_coincide (fun _ => False) statics valuation tv source);
      [ | exact Hsat_source ].
    intros x Hx. symmetry. apply Hagrees.
    apply mentions_known; [ | exact Hx ].
    rewrite <- Hsource. exact (proj1 Hfits). }
  (* And the environment with the variables defined. *)
  destruct (last_phase_adds_kept tenv fuel s relations kept equations
              relation_equations definitions Hadds) as [Hdistinct Hkept].
  destruct Hvalid
    as [[_ [Hknown_avoided _]] [Hnew [[Hrecorded _] [_ Himports]]]].
  assert (Hsat0 :
            typing_env_satisfied_strictly statics tv
              (Meet_env.typing_env menv0)).
  { apply variables_defined_satisfied.
    - rewrite Hsource. exact Hsat_tv.
    - intros x k v Hin Hv. apply in_app_or in Hin.
      destruct Hin as [Hin | Hin].
      + (* A variable brought in. *)
        pose proof (in_order_lookup _ _ _ Hin) as Himported.
        assert (Hi : is_imported s x).
        { apply is_bound_lookup. exists k. exact Himported. }
        destruct (Himports x Hi) as [Hunknown _].
        unfold tv in Hv. rewrite lookup_target_valuation in Hv.
        destruct (value_of_definition s index valuation x) as [w | ]
          eqn:Hdefinition.
        { exfalso. unfold value_of_definition in Hdefinition.
          destruct (lookup x (Join_env.definitions_of_existentials s))
            as [[simples k'] | ] eqn:Hd; [ | discriminate ].
          pose proof (Hrecorded x simples k' Hd) as He.
          destruct (Hnew x (proj2 (is_bound_lookup _ _) (ex_intro _ k' He)))
            as [_ Huses].
          destruct (Himports x Hi) as [_ [index' [use' [Huse' Hknows']]]].
          exact (Huses index' use' Huse' Hknows'). }
        destruct (known_to s use x) eqn:Hknown_to; [ | discriminate ].
        unfold known_to in Hknown_to. apply orb_true_iff in Hknown_to.
        destruct Hknown_to as [Hk | Hk]; [ | exfalso; exact (Hunknown Hk) ].
        exact (simple_of_kind_value statics valuation use (Var x) k v
                 (Henvs index use Huse) Hsat_use
                 (proj1 (proj2 Hstate) x k index use Himported Huse Hk) Hv).
      + (* A variable made. *)
        exact (existential_value_kind s x k index use statics valuation v
                 (proj1 Htrue) Hstate Henvs
                 (in_order_lookup _ _ _ (Hkept _ Hin)) Huse Hsat_use Hv). }
  (* The first extension. *)
  assert (Hext1 :
            env_extension_fits (Meet_env.typing_env menv0)
              (Env_extension equations)
            ∧ env_extension_kinded (Meet_env.typing_env menv0)
                (Env_extension equations)).
  { split; intros n ty Hlookup;
      destruct (Hall n ty
                  (in_or_app _ _ _ (or_introl (lookup_among _ _ _ Hlookup))))
      as [Hn [Hty Hk]]; [ split; assumption | exact Hk ]. }
  destruct (meet_env_add_env_extension_sound mh statics tv menv0 menv1
              (Env_extension equations) Hmh Hsound (proj2 Hkept0)
              (proj1 Hext1) (proj2 Hext1) Hstatics Hsat0
              (equations_true_with_values statics tv equations Htrue1
                 (fun n ty Hin =>
                    Hvalued n ty (in_or_app _ _ _ (or_introl Hin))))
              Hadd1)
    as [valuation1 [Hfrom1 [Hsat1 Hvalued1]]].
  (* The second. *)
  assert (Hext2 :
            env_extension_fits (Meet_env.typing_env menv1)
              (Env_extension relation_equations)
            ∧ env_extension_kinded (Meet_env.typing_env menv1)
                (Env_extension relation_equations)).
  { split; intros n ty Hlookup;
      destruct (Hall n ty
                  (in_or_app _ _ _ (or_intror (lookup_among _ _ _ Hlookup))))
      as [Hn [Hty Hk]].
    - split; [ exact (tenv_extends_mem _ _ _ Hextends1 Hn)
             | exact (tenv_extends_type_fits _ _ _ Hextends1 Hty) ].
    - exact (proj1 (proj2 (proj2 (proj2 (proj2 Hextends1)))) _ _ Hk). }
  assert (Hsatisfied2 :
            env_extension_satisfied statics valuation1
              (Env_extension relation_equations)).
  { assert (Hcoincide :
              ∀ x : variable,
                x
                ∈ free_variables_of_env_extension
                    (Env_extension relation_equations) ->
                lookup x tv = lookup x valuation1).
    { intros x Hx. symmetry. apply Hfrom1.
      cbn [free_variables_of_env_extension] in Hx.
      apply in_flat_map in Hx. destruct Hx as [[n ty] [Hin Hx]].
      cbn [fst snd] in Hx.
      destruct (Hall n ty (in_or_app _ _ _ (or_intror Hin)))
        as [Hn [[_ [Hvars _]] _]].
      apply in_app_or in Hx. destruct Hx as [Hx | Hx].
      - destruct n as [y | sym]; [ | destruct Hx ].
        destruct Hx as [<- | []]. exact Hn.
      - exact (Hvars x Hx). }
    apply (proj1 (env_extension_satisfied_coincide statics tv valuation1
                    (Env_extension relation_equations) Hcoincide)).
    exact (equations_true_with_values statics tv relation_equations Htrue2
             (fun n ty Hin =>
                Hvalued n ty (in_or_app _ _ _ (or_intror Hin)))). }
  destruct (meet_env_add_env_extension_sound mh statics valuation1 menv1
              target (Env_extension relation_equations) Hmh Hsound
              (proj2 (proj1 Hstep1)) (proj1 Hext2) (proj2 Hext2) Hstatics
              Hsat1 Hsatisfied2 Hadd2)
    as [valuation2 [Hfrom2 [Hsat2 Hvalued2]]].
  (* The valuation: that one on what the target knows, the one given
     elsewhere. *)
  set (tenv2 := Meet_env.typing_env target) in *.
  exists (patched tenv2 valuation2 valuation).
  split; [ | split; [ | split; [ | split ] ] ].
  - intros x Hx. rewrite lookup_patched.
    destruct (Typing_env.mem None tenv2 (Name_var x)) eqn:H2;
      [ | reflexivity ].
    assert (Hnames_x : among x (Meet_env.names_to_avoid menv0) = true).
    { destruct (target_variables_defined s menv kept (proj1 Htrue)
                  (eq_trans Hsource (eq_sym (proj2 (proj2 Htrue)))) Hfits
                  Hkept Hdistinct) as [_ [_ [_ Hnames0]]].
      fold menv0 in Hnames0. rewrite Hnames0. apply Hnames.
      exact (Hknown_avoided index use x Huse Hx). }
    destruct (Typing_env.mem None (Meet_env.typing_env menv0) (Name_var x))
      eqn:H0.
    + rewrite (Hfrom2 x (tenv_extends_mem _ _ _ Hextends1 H0)).
      rewrite (Hfrom1 x H0).
      apply (target_valuation_known s index use valuation x (proj1 Htrue)
               Huse).
      unfold known_to. rewrite Hx. reflexivity.
    + exfalso.
      destruct (Typing_env.mem None (Meet_env.typing_env menv1) (Name_var x))
        eqn:H1.
      * destruct (proj2 Hstep1 x H1 H0) as [Hnew1 _].
        rewrite Hnames_x in Hnew1. discriminate Hnew1.
      * destruct (proj2 Hstep2 x H2 H1) as [Hnew2 _].
        destruct (proj1 (proj1 Hstep1)) as (_ & _ & _ & _ & _ & _ & Hav & _).
        rewrite (Hav x Hnames_x) in Hnew2. discriminate Hnew2.
  - apply (satisfied_but_coincide (fun _ => False) statics valuation2
             (patched tenv2 valuation2 valuation) tenv2); [ | exact Hsat2 ].
    intros x Hx. rewrite lookup_patched.
    rewrite (mentions_known tenv2 x (proj1 (proj2 (proj1 Hstep2))) Hx).
    reflexivity.
  - (* The equations are on names that have values. *)
    intros cut_after Hcut.
    apply (equations_valued_extended tenv2 cut_after valuation2);
      [ exact (proj1 (proj2 (proj1 Hstep2))) | | ].
    { intros x Hx. rewrite lookup_patched, Hx. reflexivity. }
    apply Hvalued2. apply Hvalued1.
    destruct (target_variables_fresh s menv kept (proj1 Htrue)
                (eq_trans Hsource (eq_sym (proj2 (proj2 Htrue)))) Hkept
                Hdistinct) as [Hnodup Hfresh].
    apply (variables_defined_valued tv _ menv Hfits Hnodup Hfresh).
    rewrite Hsource.
    apply (equations_valued_extended source cut_after valuation tv);
      [ rewrite <- Hsource; exact (proj1 Hfits) | | exact Hcut ].
    intros x Hx. exact (Hagrees x Hx).
  - (* What the source knows. *)
    intros x Hx. rewrite lookup_patched.
    assert (H0 : Typing_env.mem None (Meet_env.typing_env menv0) (Name_var x)
                 = true).
    { apply (tenv_extends_mem _ _ _
               (meet_env_extends_tenv _ _ (proj1 Hkept0))).
      rewrite Hsource. exact Hx. }
    pose proof (tenv_extends_mem _ _ _ Hextends1 H0) as H1.
    pose proof (tenv_extends_mem _ _ _
                  (meet_env_extends_tenv _ _ (proj1 (proj1 Hstep2))) H1)
      as H2.
    change (Typing_env.mem None tenv2 (Name_var x) = true) in H2.
    rewrite H2, (Hfrom2 x H1), (Hfrom1 x H0). exact (Hagrees x Hx).
  - (* The definitions: each is of a variable that the environment in
       which the variables are defined knows. *)
    intros x simples kind simple v Hx Hfind Hv. rewrite lookup_patched.
    pose proof (last_phase_adds_definitions tenv fuel source s relations kept
                  equations relation_equations definitions Htrue
                  (scoped_aliases_not_made s (proj1 Htrue) (proj2 Hscoped))
                  Hadds x _ Hx) as Hfinal.
    destruct (target_variables_defined s menv kept (proj1 Htrue)
                (eq_trans Hsource (eq_sym (proj2 (proj2 Htrue)))) Hfits Hkept
                Hdistinct) as [_ [Hmem0 _]].
    fold menv0 in Hmem0.
    pose proof (proj1 (Hmem0 x) Hfinal) as H0.
    pose proof (tenv_extends_mem _ _ _ Hextends1 H0) as H1.
    pose proof (tenv_extends_mem _ _ _
                  (meet_env_extends_tenv _ _ (proj1 (proj1 Hstep2))) H1)
      as H2.
    change (Typing_env.mem None tenv2 (Name_var x) = true) in H2.
    rewrite H2, (Hfrom2 x H1), (Hfrom1 x H0).
    exact (Hdefs_tv x simples kind simple v Hx Hfind Hv).
Qed.

(* ---------------------------------------------------------------------- *)
(* The contract that the meet has for its join, but for the values         *)
(* ---------------------------------------------------------------------- *)

(* A variable of the target of the last phase that the environment
   given does not know may not occur in a term, if the variables that
   the meets of the two additions define may not. *)
Lemma last_phase_not_in_terms (tenv : Typing_env.t) (fuel : nat)
    (s : Join_env.t) (relations : Join_env.inverse_relations)
    (menv menv1 target : Meet_env.t) (kept : list (variable * flambda_kind))
    (equations relation_equations : list (name * flambda_type))
    (definitions : finite_map (Simples_in_joined_envs.t * flambda_kind)) :
  state_valid s ->
  Meet_env.typing_env menv = Join_env.source_env s ->
  typing_env_fits (Meet_env.typing_env menv) ->
  last_phase_adds tenv fuel s relations kept equations relation_equations
    definitions ->
  let menv0 :=
    variables_defined meet_env_functions menv
      (Bindings_in_target_env.imported_variables_in_order (Join_env.bindings s)
       ++ kept)
  in
  meet_env_extends menv0 menv1 ->
  meet_env_extends menv1 target ->
  (∀ x : variable,
     Typing_env.mem None (Meet_env.typing_env menv1) (Name_var x) = true ->
     Typing_env.mem None (Meet_env.typing_env menv0) (Name_var x) = false ->
     Typing_env.mem (Some Name_mode.normal) (Meet_env.typing_env menv1)
       (Name_var x)
     = false) ->
  (∀ x : variable,
     Typing_env.mem None (Meet_env.typing_env target) (Name_var x) = true ->
     Typing_env.mem None (Meet_env.typing_env menv1) (Name_var x) = false ->
     Typing_env.mem (Some Name_mode.normal) (Meet_env.typing_env target)
       (Name_var x)
     = false) ->
  ∀ x : variable,
    Typing_env.mem None (Meet_env.typing_env target) (Name_var x) = true ->
    Typing_env.mem None (Meet_env.typing_env menv) (Name_var x) = false ->
    Typing_env.mem (Some Name_mode.normal) (Meet_env.typing_env target)
      (Name_var x)
    = false.
Proof.
  intros Hvalid Hsource Hfits Hadds menv0 Hextends1 Hextends2 Hnew1 Hnew2 x Hx
    Hnot.
  destruct (last_phase_adds_kept tenv fuel s relations kept equations
              relation_equations definitions Hadds) as [Hdistinct Hkept].
  destruct (target_variables_fresh s menv kept Hvalid Hsource Hkept Hdistinct)
    as [Hnodup Hfresh].
  destruct (variables_defined_kept _ menv Hfits Hnodup Hfresh)
    as [_ [_ [Hmem _]]].
  fold menv0 in Hmem.
  destruct (Typing_env.mem None (Meet_env.typing_env menv1) (Name_var x))
    eqn:H1; [ | exact (Hnew2 x Hx H1) ].
  rewrite (extends_mode menv1 target x Name_mode.normal Hextends2 H1).
  destruct (Typing_env.mem None (Meet_env.typing_env menv0) (Name_var x))
    eqn:H0; [ | exact (Hnew1 x H1 H0) ].
  rewrite (extends_mode menv0 menv1 x Name_mode.normal Hextends1 H0).
  apply (variables_defined_not_in_terms _ menv x Hfits Hnodup Hfresh).
  apply Hmem in H0. destruct H0 as [H0 | H0]; [ | exact H0 ].
  rewrite H0 in Hnot. discriminate Hnot.
Qed.

(* The environment of a meet with more names to avoid. *)
Lemma with_names_extends (menv : Meet_env.t) (names : list variable) :
  (∀ x : variable,
     among x (Meet_env.names_to_avoid menv) = true -> among x names = true) ->
  meet_env_extends menv (Meet_env.with_names_to_avoid menv names).
Proof.
  intros Hnames. repeat split; try reflexivity.
  - intros s kind Hkind. exact Hkind.
  - exact Hnames.
  - apply keeps_names_refl.
Qed.

(* The names to avoid of an environment of a meet, extended. *)
Lemma extends_avoids (menv menv' : Meet_env.t) (x : variable) :
  meet_env_extends menv menv' ->
  among x (Meet_env.names_to_avoid menv) = true ->
  among x (Meet_env.names_to_avoid menv') = true.
Proof. intros (_ & _ & _ & _ & _ & _ & H & _). exact (H x). Qed.

Lemma extends_kinds (menv menv' : Meet_env.t) (s : simple)
    (kind : flambda_kind) :
  meet_env_extends menv menv' ->
  simple_of_kind (Meet_env.typing_env menv) s kind ->
  simple_of_kind (Meet_env.typing_env menv') s kind.
Proof. intros (_ & _ & _ & _ & H & _). exact (H s kind). Qed.

(* The first phases of the join that the meet calls, on inputs that
   have what the contract supposes. *)
Lemma scoped_envs_first_phases (join_choices : Join_env.choices)
    (join_depth : nat) (fuel : fuels) (initial_env : Meet_env.t)
    (scoped_envs : list Typing_env.t) (envs : Joined_envs.t)
    (s1 s2 s3 : Join_env.t)
    (relations2 relations3 : Join_env.inverse_relations) :
  let initial_tenv := Meet_env.typing_env initial_env in
  let p := join_parameters join_depth in
  let jt := model_join_of_types join_depth fuel in
  let avoid :=
    Meet_env.names_to_avoid initial_env
    ++ Join_env.variables_of_env initial_tenv
    ++ flat_map Join_env.variables_of_env scoped_envs
  in
  let s0 :=
    Join_env.create envs
      (Bindings_in_target_env.from_source_env avoid initial_tenv)
  in
  typing_env_fits initial_tenv ->
  (∀ scoped_env : Typing_env.t,
    scoped_env ∈ scoped_envs -> scoped_env_of initial_tenv scoped_env) ->
  Join_env.cut_for_joins (Typing_env.current_scope initial_tenv)
    (Index.number scoped_envs)
  = Some envs ->
  Join_env.join_aliases_into_bindings p (fuel_of_env fuel) s0 envs
  = Some s1 ->
  Join_env.n_way_join_delayed_equations (fuel_of_env fuel) join_choices jt
    s1 [] (Join_env.new_equations_in_joined_envs s1 s0)
  = Some (relations2, s2) ->
  Join_env.n_way_join_loop join_depth p (fuel_of_env fuel) join_choices jt
    s2 relations2 (Join_env.new_definitions_of_existentials s2 s0)
  = Some (relations3, s3) ->
  state_true initial_tenv s3
  ∧ relations_hold s3 relations3
  ∧ grows s0 s3
  ∧ state_kinded s3
  ∧ relations_nonempty relations3
  ∧ state_scoped s3
  ∧ equations_kinded s3
  ∧ relations_kinded s3 relations3
  ∧ (∀ x : variable, x ∈ avoid -> x ∈ names_to_avoid s3)
  ∧ (∀ x : variable, is_existential s3 x -> x ∉ avoid)
  ∧ (∀ (index : Index.t) (use : Typing_env.t),
       Join_env.joined_env s3 index = Some use -> use ∈ scoped_envs)
  ∧ joined_envs_are typing_env_invariant s3
  ∧ joined_envs_are typing_env_kinded s3
  ∧ symbols_of_joined_envs_in_source initial_tenv s3.
Proof.
  intros initial_tenv p jt avoid s0 Hfits Hscoped Hcut Haliases Hdelayed
    Hloop.
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
  assert (Havoid_scoped :
            ∀ (scoped_env : Typing_env.t) (x : variable),
              scoped_env ∈ scoped_envs -> knows scoped_env x -> x ∈ avoid).
  { intros scoped_env x Hin Hx. apply in_or_app. right. apply in_or_app.
    right. apply in_flat_map. exists scoped_env.
    split; [ exact Hin | apply variables_of_env_knows; exact Hx ]. }
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
  assert (Huses :
            ∀ (index : Index.t) (use : Typing_env.t) (x : variable),
              Joined_envs.get_nth_joined_env envs index = Some use ->
              knows use x ->
              x ∈ avoid).
  { intros index use x Huse Hx.
    exact (Havoid_scoped use x (Hfound index use Huse) Hx). }
  assert (Hinvariant :
            ∀ (index : Index.t) (use : Typing_env.t),
              Joined_envs.get_nth_joined_env envs index = Some use ->
              typing_env_invariant use).
  { intros index use Huse.
    exact (proj1 (proj1 (Hscoped use (Hfound index use Huse)))). }
  assert (Hkinded_envs :
            ∀ (index : Index.t) (use : Typing_env.t),
              Joined_envs.get_nth_joined_env envs index = Some use ->
              typing_env_kinded use).
  { intros index use Huse.
    exact (proj1 (proj2 (proj2 (proj1 (Hscoped use
                                         (Hfound index use Huse)))))). }
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
  destruct (first_phases_kinds initial_tenv jt join_choices p
              (fuel_of_env fuel) avoid envs s1 s2 s3 relations2 relations3 Hjt
              (n_way_join_keeps_kinds p (fuel_of_env fuel)
                 (fuel_of_types fuel) expand_head_for_join
                 reduce_head_of_kind_naked_immediate type_eqb block_shape_eqb
                 Hfresh)
              Hfresh Heqb Hconst Hsource Huses Hinvariant Hkinded_envs Hlevels
              Hkinds Haliases Hdelayed Hloop) as [HX HR].
  destruct (first_phases_names jt join_choices p (fuel_of_env fuel) avoid
              initial_tenv envs s1 s2 s3 relations2 relations3
              (n_way_join_keeps_names p (fuel_of_env fuel)
                 (fuel_of_types fuel) expand_head_for_join
                 reduce_head_of_kind_naked_immediate type_eqb block_shape_eqb
                 Hfresh)
              Hfresh Haliases Hdelayed Hloop) as [Hnames_kept Hmade].
  assert (Hjoined :
            ∀ (index : Index.t) (use : Typing_env.t),
              Join_env.joined_env s3 index = Some use -> use ∈ scoped_envs).
  { intros index use Huse. rewrite (proj1 Hgrows3) in Huse.
    exact (Hfound index use Huse). }
  assert (Hinvariant3 : joined_envs_are typing_env_invariant s3).
  { intros index use Huse. rewrite (proj1 Hgrows3) in Huse.
    exact (Hinvariant index use Huse). }
  assert (Hkinded3 : joined_envs_are typing_env_kinded s3).
  { intros index use Huse. rewrite (proj1 Hgrows3) in Huse.
    exact (Hkinded_envs index use Huse). }
  assert (Hsymbols3 : symbols_of_joined_envs_in_source initial_tenv s3).
  { intros index use sym Huse Hmem.
    exact (proj2 (proj2 (proj2 (Hscoped use (Hjoined index use Huse)))) sym
             Hmem). }
  repeat (split; [ assumption | ]). exact Hsymbols3.
Qed.

Theorem model_join_scoped_envs_keeps : model_join_scoped_envs_keeps_statement.
Proof.
  intros join_choices join_depth fuel Hrest mh initial_env result scoped_envs
    initial_tenv Hmh Hfits Hscoped Hagree Hjoin.
  unfold model_join_scoped_envs, Join_env.cut_and_n_way_join in Hjoin.
  cbv zeta in Hjoin.
  destruct (Join_env.cut_for_joins _ _) as [envs | ] eqn:Hcut;
    [ | discriminate ].
  cbn [Join_env.names_to_avoid meet_env_functions] in Hjoin.
  fold initial_tenv in Hjoin, Hcut.
  set (avoid :=
         Meet_env.names_to_avoid initial_env
         ++ Join_env.variables_of_env initial_tenv
         ++ flat_map Join_env.variables_of_env scoped_envs) in *.
  set (p := join_parameters join_depth) in *.
  set (jt := model_join_of_types join_depth fuel) in *.
  destruct (Join_env.cut_and_n_way_join0 _ _ _ _ _ _ _ _ _ _)
    as [[[target definitions] names] | ] eqn:Hjoin0; [ | discriminate ].
  injection Hjoin as <-.
  (* The environments joined. *)
  assert (Hfound :
            ∀ (index : Index.t) (use : Typing_env.t),
              Joined_envs.get_nth_joined_env envs index = Some use ->
              use ∈ scoped_envs).
  { intros index use Huse. apply get_nth_lookup in Huse.
    destruct Huse as [equations0 Hlookup].
    destruct (cut_for_joins_found _ _ _ Hcut _ _ _ Hlookup) as [Hin _].
    apply number_in in Hin. exact (nth_error_In _ _ Hin). }
  assert (Havoid_initial :
            ∀ x : variable,
              among x (Meet_env.names_to_avoid initial_env) = true ->
              x ∈ avoid).
  { intros x Hx. apply in_or_app. left. apply among_In. exact Hx. }
  assert (Hsource : ∀ x : variable, knows initial_tenv x -> x ∈ avoid).
  { intros x Hx. apply in_or_app. right. apply in_or_app. left.
    apply variables_of_env_knows. exact Hx. }
  assert (Havoid_scoped :
            ∀ (scoped_env : Typing_env.t) (x : variable),
              scoped_env ∈ scoped_envs -> knows scoped_env x -> x ∈ avoid).
  { intros scoped_env x Hin Hx. apply in_or_app. right. apply in_or_app.
    right. apply in_flat_map. exists scoped_env.
    split; [ exact Hin | apply variables_of_env_knows; exact Hx ]. }
  (* What the result knows of a simple that the first environment
     knows. *)
  assert (Hkind_initial :
            ∀ (menv : Meet_env.t) (scoped_env : Typing_env.t) (s : simple)
              (kind : flambda_kind),
              meet_env_extends initial_env menv ->
              scoped_env ∈ scoped_envs ->
              Typing_env.mem_simple None initial_tenv s = true ->
              simple_of_kind scoped_env s kind ->
              simple_of_kind (Meet_env.typing_env menv) s kind).
  { intros menv scoped_env s kind Hextends Hin Hknown Hk.
    apply (extends_kinds initial_env menv s kind Hextends).
    exact (scoped_env_of_kind initial_tenv scoped_env s kind
             (Hscoped scoped_env Hin) Hknown Hk). }
  assert (Hkind_symbol :
            ∀ (menv : Meet_env.t) (scoped_env : Typing_env.t) (sym : symbol)
              (kind : flambda_kind),
              typing_env_fits (Meet_env.typing_env menv) ->
              scoped_env ∈ scoped_envs ->
              Typing_env.mem_simple None (Meet_env.typing_env menv)
                (Symbol sym)
              = true ->
              simple_of_kind scoped_env (Symbol sym) kind ->
              simple_of_kind (Meet_env.typing_env menv) (Symbol sym) kind).
  { intros menv scoped_env sym kind Hfits_menv Hin Hknown Hk.
    destruct (mem_simple_of_some_kind _ _ Hknown) as [kind' Hk'].
    rewrite (symbol_kinded scoped_env sym kind
               (proj1 (proj2 (proj2 (proj1 (Hscoped scoped_env Hin))))) Hk).
    rewrite (symbol_kinded _ sym kind' (proj1 (proj2 (proj2 Hfits_menv)))
               Hk') in Hk'.
    exact Hk'. }
  destruct (Keyed.is_empty envs) eqn:Hempty.
  { (* No environment to join: the target is bottom. *)
    unfold Join_env.cut_and_n_way_join0 in Hjoin0. rewrite Hempty in Hjoin0.
    injection Hjoin0 as <- _ _.
    cbn [Join_env.make_bottom meet_env_functions].
    destruct (make_bottom_keeps initial_env Hfits) as [Hextends Hfits'].
    split; [ exact Hextends | ]. split; [ exact Hfits' | ].
    split; [ | split ].
    - intros x Hx Hnot _.
      change (Typing_env.mem None initial_tenv (Name_var x) = true) in Hx.
      rewrite Hx in Hnot. discriminate Hnot.
    - intros x Hx Hnot.
      change (Typing_env.mem None initial_tenv (Name_var x) = true) in Hx.
      rewrite Hx in Hnot. discriminate Hnot.
    - intros scoped_env s kind Hin Hknown Hk.
      destruct s as [x | sym | c].
      + exact (Hkind_initial _ scoped_env (Var x) kind Hextends Hin Hknown
                 Hk).
      + exact (Hkind_symbol _ scoped_env sym kind Hfits' Hin Hknown Hk).
      + exact Hk. }
  (* The first phases. *)
  destruct (cut_and_n_way_join0_replayed Meet_env.t meet_expanded_head
              meet_env_functions p (fuel_of_env fuel) join_choices jt mh avoid
              initial_env target initial_tenv envs definitions names Hempty
              Hjoin0)
    as [s1 [s2 [s3 [relations2 [relations3 [kept [equations
          [relation_equations [menv1 [Haliases [Hdelayed [Hloop [Hnames
            [Hadds [Hadd1 Hadd2]]]]]]]]]]]]]]].
  subst names.
  set (names :=
         Bindings_in_target_env.names_to_avoid (Join_env.bindings s3)) in *.
  cbn [Join_env.with_names_to_avoid Join_env.add_env_extension
         meet_env_functions] in Hadd1, Hadd2.
  set (menv_a := Meet_env.with_names_to_avoid initial_env names) in *.
  set (menv0 :=
         variables_defined meet_env_functions menv_a
           (Bindings_in_target_env.imported_variables_in_order
              (Join_env.bindings s3)
            ++ kept)) in *.
  destruct (scoped_envs_first_phases join_choices join_depth fuel initial_env
              scoped_envs envs s1 s2 s3 relations2 relations3 Hfits Hscoped
              Hcut Haliases Hdelayed Hloop)
    as (Htrue3 & Hrelations3 & Hgrows3 & Hstate3 & Hnonempty3 & Hscoped3 & HX
        & HR & Hnames_kept & Hmade & Hjoined & Hinvariant3 & Hkinded3
        & Hsymbols3).
  fold initial_tenv p jt avoid in Htrue3, Hrelations3, Hgrows3, Hnames_kept,
    Hmade, Hsymbols3.
  (* The last phase. *)
  destruct (last_phase_kept _ (fuel_of_env fuel) initial_tenv s3 relations3
              mh menv_a menv1 target kept equations relation_equations
              definitions Htrue3 Hrelations3 Hnonempty3 Hscoped3 Hstate3
              Hinvariant3 Hkinded3 Hsymbols3 HX HR eq_refl Hfits Hadds
              (Hrest initial_env scoped_envs envs s1 s2 s3 relations2
                 relations3 _ kept equations relation_equations definitions
                 Hfits Hscoped Hagree Hcut Haliases Hdelayed Hloop Hadds)
              Hmh Hadd1 Hadd2) as [Hkept0 [Hstep1 Hstep2]].
  fold menv0 in Hkept0, Hstep1.
  assert (Hextends_a : meet_env_extends initial_env menv_a).
  { apply with_names_extends. intros x Hx. apply among_In.
    exact (Hnames_kept x (Havoid_initial x Hx)). }
  pose proof (meet_env_extends_trans _ _ _ Hextends_a (proj1 Hkept0))
    as Hextends0.
  pose proof (meet_env_extends_trans _ _ _ Hextends0 (proj1 (proj1 Hstep1)))
    as Hextends1.
  pose proof (meet_env_extends_trans _ _ _ Hextends1 (proj1 (proj1 Hstep2)))
    as Hextends2.
  (* The variables that the join defined. *)
  destruct (last_phase_adds_kept _ (fuel_of_env fuel) s3 relations3 kept
              equations relation_equations definitions Hadds)
    as [Hdistinct Hkept].
  assert (Hsource3 : Meet_env.typing_env menv_a = Join_env.source_env s3).
  { rewrite (proj2 (proj2 Htrue3)). reflexivity. }
  destruct (target_variables_defined s3 menv_a kept (proj1 Htrue3) Hsource3
              Hfits Hkept Hdistinct) as [_ [Hmem0 [Hkinds0 Hnames0]]].
  fold menv0 in Hmem0, Hkinds0, Hnames0.
  assert (Hkept_made :
            ∀ x : variable, x ∈ map fst kept -> is_existential s3 x).
  { intros x Hx. apply in_map_iff in Hx. destruct Hx as [[y k] [Heq Hin]].
    cbn in Heq. subst y. apply is_bound_lookup. exists k.
    exact (in_order_lookup _ _ _ (Hkept _ Hin)). }
  destruct (proj1 Htrue3) as [[_ [_ Hmade_avoided]] [_ [_ [_ Himports]]]].
  (* A variable that the environment with the variables defined knows
     and the first does not: brought in from an environment joined, or
     made by the join. *)
  assert (Hdefined :
            ∀ x : variable,
              Typing_env.mem None (Meet_env.typing_env menv0) (Name_var x)
              = true ->
              Typing_env.mem None initial_tenv (Name_var x) = false ->
              (is_imported s3 x
               ∧ ∃ use : Typing_env.t, use ∈ scoped_envs ∧ knows use x)
              ∨ (is_existential s3 x ∧ x ∈ map fst kept)).
  { intros x Hx Hnot. apply Hmem0 in Hx.
    destruct Hx as [Hx | [Hx | Hx]].
    - exfalso. rewrite (proj2 (proj2 Htrue3)) in Hx. unfold knows in Hx.
      rewrite Hx in Hnot. discriminate Hnot.
    - left. split; [ exact Hx | ].
      destruct (Himports x Hx) as [_ [index [use [Huse Hknows]]]].
      exists use. split; [ exact (Hjoined index use Huse) | exact Hknows ].
    - right. split; [ exact (Hkept_made x Hx) | exact Hx ]. }
  split; [ exact Hextends2 | ].
  split; [ exact (proj2 (proj1 Hstep2)) | ].
  split; [ | split ].
  - (* The variables that are new. *)
    intros x Hx Hnot_initial Hnot_scoped.
    destruct (Typing_env.mem None (Meet_env.typing_env menv0) (Name_var x))
      eqn:H0.
    + destruct (Hdefined x H0 Hnot_initial)
        as [[_ [use [Hin Hknows]]] | [Hmade_x _]].
      * unfold knows in Hknows. rewrite (Hnot_scoped use Hin) in Hknows.
        discriminate Hknows.
      * split.
        -- destruct (among x (Meet_env.names_to_avoid initial_env)) eqn:Hav;
             [ | reflexivity ].
           exfalso. exact (Hmade x Hmade_x (Havoid_initial x Hav)).
        -- apply (extends_avoids menv_a target x).
           ++ exact (meet_env_extends_trans _ _ _
                       (meet_env_extends_trans _ _ _ (proj1 Hkept0)
                          (proj1 (proj1 Hstep1)))
                       (proj1 (proj1 Hstep2))).
           ++ apply among_In. exact (Hmade_avoided x Hmade_x).
    + destruct (Typing_env.mem None (Meet_env.typing_env menv1) (Name_var x))
        eqn:H1.
      * destruct (proj2 Hstep1 x H1 H0) as [Hnew [Havoided _]].
        split.
        -- destruct (among x (Meet_env.names_to_avoid initial_env)) eqn:Hav;
             [ | reflexivity ].
           rewrite (extends_avoids initial_env menv0 x Hextends0 Hav) in Hnew.
           discriminate Hnew.
        -- exact (extends_avoids menv1 target x (proj1 (proj1 Hstep2))
                    Havoided).
      * destruct (proj2 Hstep2 x Hx H1) as [Hnew [Havoided _]].
        split; [ | exact Havoided ].
        destruct (among x (Meet_env.names_to_avoid initial_env)) eqn:Hav;
          [ | reflexivity ].
        rewrite (extends_avoids initial_env menv1 x Hextends1 Hav) in Hnew.
        discriminate Hnew.
  - (* The variables that the first environment does not know may not
       occur in terms. *)
    intros x Hx Hnot.
    exact (last_phase_not_in_terms _ (fuel_of_env fuel) s3 relations3 menv_a
             menv1 target kept equations relation_equations definitions
             (proj1 Htrue3) Hsource3 Hfits Hadds (proj1 (proj1 Hstep1))
             (proj1 (proj1 Hstep2))
             (fun y H1 H0 => proj2 (proj2 (proj2 Hstep1 y H1 H0)))
             (fun y H2 H1 => proj2 (proj2 (proj2 Hstep2 y H2 H1))) x Hx Hnot).
  - (* The kinds. *)
    intros scoped_env s kind Hin Hknown Hk.
    destruct s as [x | sym | c];
      [ | exact (Hkind_symbol _ scoped_env sym kind (proj2 (proj1 Hstep2)) Hin
                   Hknown Hk)
        | exact Hk ].
    destruct (Typing_env.mem None initial_tenv (Name_var x)) eqn:Hinitial.
    { exact (Hkind_initial _ scoped_env (Var x) kind Hextends2 Hin Hinitial
               Hk). }
    pose proof (simple_of_kind_var_known scoped_env x kind Hk) as Hknows.
    assert (Havoid_x : x ∈ avoid).
    { exact (Havoid_scoped scoped_env x Hin Hknows). }
    assert (Hnames_x : among x (Meet_env.names_to_avoid menv0) = true).
    { rewrite Hnames0. apply among_In. exact (Hnames_kept x Havoid_x). }
    destruct (Typing_env.mem None (Meet_env.typing_env menv0) (Name_var x))
      eqn:H0.
    + apply (extends_kinds menv0 target (Var x) kind).
      { exact (meet_env_extends_trans _ _ _ (proj1 (proj1 Hstep1))
                 (proj1 (proj1 Hstep2))). }
      destruct (Hdefined x H0 Hinitial)
        as [[Himported [use [Huse_in Huse_knows]]] | [Hmade_x _]].
      * apply is_bound_lookup in Himported.
        destruct Himported as [kind0 Hkind0].
        apply (Hkinds0 x kind (proj2 (Hmem0 x) H0)).
        (* The kind recorded is the kind at the environments. *)
        destruct (bound_has_kind use (Name_var x) Huse_knows)
          as [kind1 Hkind1].
        cbn [simple_of_name] in Hkind1.
        rewrite (Hagree scoped_env use (Var x) kind kind1 Hin Huse_in Hk
                   Hkind1).
        destruct (Himports x (proj2 (is_bound_lookup _ _)
                                      (ex_intro _ kind0 Hkind0)))
          as [Hunknown [index [use' [Huse' Hknows']]]].
        pose proof (proj1 (proj2 Hstate3) x kind0 index use' Hkind0 Huse'
                      Hknows') as Hkind'.
        rewrite (Hagree use use' (Var x) kind1 kind0 Huse_in
                   (Hjoined index use' Huse') Hkind1 Hkind').
        unfold kind_in_target_env.
        destruct (Keyed.lookup name_eqb (Name_var x)
                    (Typing_env.names_to_types
                       (Bindings_in_target_env.source_env
                          (Join_env.bindings s3)))) as [[ty btm] | ]
          eqn:Hlookup.
        { exfalso. apply Hunknown. apply mem_var_lookup. exists ty, btm.
          exact Hlookup. }
        destruct (lookup x
                    (Bindings_in_target_env.existential_variables
                       (Join_env.bindings s3))) as [k | ] eqn:Hexistential;
          [ | exact Hkind0 ].
        exfalso.
        destruct (proj1 (proj2 (proj1 Htrue3)) x
                    (proj2 (is_bound_lookup _ _)
                       (ex_intro _ k Hexistential))) as [_ Hnot_known].
        exact (Hnot_known index use' Huse' Hknows').
      * exfalso. exact (Hmade x Hmade_x Havoid_x).
    + exfalso.
      destruct (Typing_env.mem None (Meet_env.typing_env menv1) (Name_var x))
        eqn:H1.
      * destruct (proj2 Hstep1 x H1 H0) as [Hnew _].
        rewrite Hnames_x in Hnew. discriminate Hnew.
      * assert (Hx : Typing_env.mem None (Meet_env.typing_env target)
                       (Name_var x) = true) by exact Hknown.
        destruct (proj2 Hstep2 x Hx H1) as [Hnew _].
        rewrite (extends_avoids menv0 menv1 x (proj1 (proj1 Hstep1))
                   Hnames_x) in Hnew.
        discriminate Hnew.
Qed.

Theorem model_join_scoped_envs_satisfied
  : model_join_scoped_envs_satisfied_statement.
Proof.
  intros join_choices join_depth fuel Hrest mh initial_env result scoped_envs
    envs s1 s2 s3 relations2 relations3 kept equations relation_equations
    definitions index scoped_env statics valuation initial_tenv p jt avoid s0
    menv Hsound Hmh Hfits Hscoped Hagree Hjoin Hcut Haliases Hdelayed Hloop
    Hadds Huse Hstatics Hsat_initial Hsat_use Htags Haliases_v Hscrutinees
    Hvalued.
  destruct (scoped_envs_first_phases join_choices join_depth fuel initial_env
              scoped_envs envs s1 s2 s3 relations2 relations3 Hfits Hscoped
              Hcut Haliases Hdelayed Hloop)
    as (Htrue3 & Hrelations3 & Hgrows3 & Hstate3 & Hnonempty3 & Hscoped3 & HX
        & HR & Hnames_kept & Hmade & Hjoined & Hinvariant3 & Hkinded3
        & Hsymbols3).
  fold initial_tenv p jt avoid s0 in Htrue3, Hrelations3, Hgrows3, Hnames_kept,
    Hmade, Hsymbols3.
  unfold model_join_scoped_envs, Join_env.cut_and_n_way_join in Hjoin.
  cbv zeta in Hjoin. fold initial_tenv in Hjoin. rewrite Hcut in Hjoin.
  cbn [Join_env.names_to_avoid meet_env_functions] in Hjoin.
  fold avoid p jt in Hjoin.
  destruct (Join_env.cut_and_n_way_join0 _ _ _ _ _ _ _ _ _ _)
    as [[[target definitions'] names] | ] eqn:Hjoin0; [ | discriminate ].
  injection Hjoin as <-.
  assert (Hempty : Keyed.is_empty envs = false).
  { pose proof Huse as Huse0. rewrite (proj1 Hgrows3) in Huse0.
    destruct envs; [ discriminate Huse0 | reflexivity ]. }
  destruct (cut_and_n_way_join0_replayed Meet_env.t meet_expanded_head
              meet_env_functions p (fuel_of_env fuel) join_choices jt mh avoid
              initial_env target initial_tenv envs definitions' names Hempty
              Hjoin0)
    as [s1' [s2' [s3' [relations2' [relations3' [kept' [equations'
          [relation_equations' [menv1 [Haliases' [Hdelayed' [Hloop' [Hnames
            [Hadds' [Hadd1 Hadd2]]]]]]]]]]]]]]].
  fold s0 in Haliases', Hdelayed', Hloop'.
  rewrite Haliases in Haliases'. injection Haliases' as <-.
  rewrite Hdelayed in Hdelayed'. injection Hdelayed' as <- <-.
  change (Join_env.join_depth p) with join_depth in Hloop'.
  rewrite Hloop in Hloop'. injection Hloop' as <- <-.
  subst names.
  cbn [Join_env.with_names_to_avoid Join_env.add_env_extension
         Join_env.current_typing_env meet_env_functions] in Hadds', Hadd1,
    Hadd2.
  fold menv in Hadds', Hadd1, Hadd2.
  destruct (last_phase_adds_functional _ _ _ _ _ _ _ _ _ _ _ _ Hadds Hadds')
    as (-> & -> & -> & ->).
  destruct (last_phase_satisfied _ (fuel_of_env fuel) initial_tenv s3
              relations3
           mh menv menv1 target kept equations relation_equations definitions
           index scoped_env statics valuation Htrue3 Hrelations3 Hnonempty3
           Hscoped3 Hstate3 Hinvariant3 Hkinded3 Hsymbols3 HX HR eq_refl Hfits
           (fun x Hx => proj2 (among_In _ _) Hx) Hadds
           (Hrest initial_env scoped_envs envs s1 s2 s3 relations2 relations3
              _ kept equations relation_equations definitions Hfits Hscoped
              Hagree Hcut Haliases Hdelayed Hloop Hadds)
           Hmh Hsound Hadd1 Hadd2 Huse Hstatics Hsat_initial Hsat_use Htags
           Haliases_v Hscrutinees Hvalued)
    as [valuation' [Hfrom [Hsat [Hkeeps _]]]].
  exists valuation'. split; [ exact Hfrom | ].
  split; [ exact Hsat | exact Hkeeps ].
Qed.

(* ---------------------------------------------------------------------- *)
(* The join that Simplify calls, in the environment of the meet            *)
(* ---------------------------------------------------------------------- *)

(* An environment of a meet, extended, keeps the typing environment as
   the contract of the join says of a source. *)
Lemma extends_keeps_source (menv menv' : Meet_env.t) :
  meet_env_extends menv menv' ->
  join_keeps_source (Meet_env.typing_env menv) (Meet_env.typing_env menv').
Proof.
  intros (Hscope & Hprev & Hmin & Hsymbols & _ & _ & _ & Hnames).
  split; [ | split; [ | split; [ exact Hprev | split; assumption ] ] ].
  - intros n ty btm Hlookup.
    destruct (Hnames n ty btm Hlookup) as [ty' [H _]]. exists ty'. exact H.
  - intros s. cbn [Typing_env.mem]. rewrite Hsymbols. reflexivity.
Qed.

(* The first phases of the join that Simplify calls, on inputs of the
   contract. *)
Lemma inputs_first_phases {A : Type} (join_choices : Join_env.choices)
    (join_depth : nat) (fuel : fuels) (avoid : list variable)
    (inputs : join_inputs A) (envs : Joined_envs.t) (s1 s2 s3 : Join_env.t)
    (relations2 relations3 : Join_env.inverse_relations) :
  let p := join_parameters join_depth in
  let jt := model_join_of_types join_depth fuel in
  let s0 :=
    Join_env.create envs
      (Bindings_in_target_env.from_source_env avoid (source inputs))
  in
  join_inputs_well_formed typing_env_invariant avoid inputs ->
  uses_kinded inputs ->
  uses_agree_with_source_on_kinds inputs ->
  Join_env.cut_for_joins (cut_after inputs) (numbered_uses inputs)
  = Some envs ->
  Join_env.join_aliases_into_bindings p (fuel_of_env fuel) s0 envs
  = Some s1 ->
  Join_env.n_way_join_delayed_equations (fuel_of_env fuel) join_choices jt
    s1 [] (Join_env.new_equations_in_joined_envs s1 s0)
  = Some (relations2, s2) ->
  Join_env.n_way_join_loop join_depth p (fuel_of_env fuel) join_choices jt
    s2 relations2 (Join_env.new_definitions_of_existentials s2 s0)
  = Some (relations3, s3) ->
  state_true (source inputs) s3
  ∧ relations_hold s3 relations3
  ∧ grows s0 s3
  ∧ state_kinded s3
  ∧ relations_nonempty relations3
  ∧ state_scoped s3
  ∧ equations_kinded s3
  ∧ relations_kinded s3 relations3
  ∧ (∀ x : variable, x ∈ avoid -> x ∈ names_to_avoid s3)
  ∧ (∀ x : variable, is_existential s3 x -> x ∉ avoid)
  ∧ (∀ (index : Index.t) (use : Typing_env.t),
       Join_env.joined_env s3 index = Some use
       <-> use_at inputs index use ∧ Typing_env.is_bottom use = false)
  ∧ joined_envs_are typing_env_invariant s3
  ∧ joined_envs_are typing_env_kinded s3
  ∧ symbols_of_joined_envs_in_source (source inputs) s3.
Proof.
  intros p jt s0 Hinputs Hkinded Hagree Hcut Haliases Hdelayed Hloop.
  destruct Hinputs
    as [[_ Huses_wf] [_ [_ [Hsymbols [Havoid_source Havoid_uses]]]]].
  assert (Hfound :
            ∀ (index : Index.t) (use : Typing_env.t),
              Joined_envs.get_nth_joined_env envs index = Some use ->
              use_at inputs index use ∧ Typing_env.is_bottom use = false).
  { intros index use Huse. apply get_nth_lookup in Huse.
    destruct Huse as [equations Hlookup].
    destruct (cut_for_joins_found _ _ _ Hcut _ _ _ Hlookup) as [Hin [Hb _]].
    split; [ apply numbered_uses_in; exact Hin | exact Hb ]. }
  assert (Hhas :
            ∀ (index : Index.t) (use : Typing_env.t),
              use_at inputs index use ->
              Typing_env.is_bottom use = false ->
              ∃ level : Typing_env_level.t,
                Typing_env.cut use (cut_after inputs) = Some level
                ∧ Keyed.lookup Index.equal index envs
                  = Some (use, Typing_env_level.equations level)).
  { intros index use Hat Hb.
    exact (cut_for_joins_has _ _ _ Hcut (numbered_uses_distinct inputs) index
             use (proj2 (numbered_uses_in inputs index use) Hat) Hb). }
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
  assert (Hkinded_envs :
            ∀ (index : Index.t) (use : Typing_env.t),
              Joined_envs.get_nth_joined_env envs index = Some use ->
              typing_env_kinded use).
  { intros index use Huse.
    exact (Hkinded index use (proj1 (Hfound index use Huse))). }
  assert (Hlevels : levels_found envs).
  { intros index use equations n ty Hlookup Hn.
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
  destruct (join_parameters_sound join_depth)
    as [Hfresh [Heqb [Hextension Hconst]]].
  fold p in Hfresh, Heqb, Hextension, Hconst.
  assert (Hreduce :
            FlambdaJoinTypesSpec.reduce_head_of_kind_naked_immediate_sound
              reduce_head_of_kind_naked_immediate).
  { intros tenv0 statics valuation v head Hsat Hhead.
    exact (FlambdaMeetStepsProofs.reduce_head_of_kind_naked_immediate_sound
             statics valuation tenv0 head v Hsat Hhead). }
  pose proof (n_way_join_of_types_true (source inputs) p (fuel_of_env fuel)
                expand_head_for_join reduce_head_of_kind_naked_immediate
                (fuel_of_types fuel) Hfresh Heqb Hextension
                expand_head_for_join_sound Hreduce) as Hjt.
  destruct (first_phases_true (source inputs) jt join_choices p
              (fuel_of_env fuel)
              avoid envs s1 s2 s3 relations2 relations3 Hjt Hfresh Heqb Hconst
              Hsource Huses Hinvariant Hkinded_envs Hlevels Hkinds Haliases
              Hdelayed Hloop) as [Htrue3 [Hrelations3 Hgrows3]].
  pose proof (first_phases_kinded (source inputs) jt join_choices p
                (fuel_of_env fuel) avoid envs s1 s2 s3 relations2 relations3
                Hjt Hfresh Heqb Hconst Hsource Huses Hinvariant Hkinded_envs
                Hlevels Hkinds Haliases Hdelayed Hloop) as Hstate3.
  assert (Hnonempty1 : relations_nonempty []).
  { intros r names' Hr. discriminate Hr. }
  pose proof (loop_relations_nonempty _ _ _ _ _ _ _ _ _ _
                (delayed_equations_relations_nonempty _ _ _ _ _ _ _ _
                   Hnonempty1 Hdelayed)
                Hloop) as Hnonempty3.
  pose proof (first_phases_scoped (source inputs) jt join_choices p
                (fuel_of_env fuel) avoid envs s1 s2 s3 relations2 relations3
                Hjt
                (n_way_join_keeps_scoped p (fuel_of_env fuel)
                   (fuel_of_types fuel) expand_head_for_join
                   reduce_head_of_kind_naked_immediate type_eqb
                   block_shape_eqb)
                Hfresh Heqb Hconst Hsource Huses Hinvariant Hkinded_envs
                Hlevels Hkinds Haliases Hdelayed Hloop) as Hscoped3.
  destruct (first_phases_kinds (source inputs) jt join_choices p
              (fuel_of_env fuel) avoid envs s1 s2 s3 relations2 relations3 Hjt
              (n_way_join_keeps_kinds p (fuel_of_env fuel)
                 (fuel_of_types fuel) expand_head_for_join
                 reduce_head_of_kind_naked_immediate type_eqb block_shape_eqb
                 Hfresh)
              Hfresh Heqb Hconst Hsource Huses Hinvariant Hkinded_envs Hlevels
              Hkinds Haliases Hdelayed Hloop) as [HX HR].
  destruct (first_phases_names jt join_choices p (fuel_of_env fuel) avoid
              (source inputs) envs s1 s2 s3 relations2 relations3
              (n_way_join_keeps_names p (fuel_of_env fuel)
                 (fuel_of_types fuel) expand_head_for_join
                 reduce_head_of_kind_naked_immediate type_eqb block_shape_eqb
                 Hfresh)
              Hfresh Haliases Hdelayed Hloop) as [Hnames_kept Hmade].
  assert (Huse3 :
            ∀ (index : Index.t) (use : Typing_env.t),
              Join_env.joined_env s3 index = Some use
              <-> use_at inputs index use ∧ Typing_env.is_bottom use = false).
  { intros index use. rewrite (proj1 Hgrows3). split.
    - exact (Hfound index use).
    - intros [Hat Hb]. destruct (Hhas index use Hat Hb) as [level [_ Hl]].
      apply get_nth_lookup. exists (Typing_env_level.equations level).
      exact Hl. }
  assert (Hinvariant3 : joined_envs_are typing_env_invariant s3).
  { intros index use Huse. rewrite (proj1 Hgrows3) in Huse.
    exact (Hinvariant index use Huse). }
  assert (Hkinded3 : joined_envs_are typing_env_kinded s3).
  { intros index use Huse. rewrite (proj1 Hgrows3) in Huse.
    exact (Hkinded_envs index use Huse). }
  assert (Hsymbols3 : symbols_of_joined_envs_in_source (source inputs) s3).
  { intros index use sym Huse Hmem.
    exact (Hsymbols index use sym (proj1 (proj1 (Huse3 index use) Huse))
             Hmem). }
  repeat (split; [ assumption | ]). exact Hsymbols3.
Qed.

(* The join that Simplify calls, unfolded. *)
Lemma model_join_unfold {A : Type} (join_choices : Join_env.choices)
    (meet_choices : meet_choices) (join_depth : nat) (fuel : fuels)
    (avoid : list variable) (inputs : join_inputs A) (target : Typing_env.t)
    (analysis : Analysis.t A) (envs : Joined_envs.t) :
  model_join join_choices meet_choices join_depth fuel avoid inputs
  = Some (target, analysis) ->
  Join_env.cut_for_joins (cut_after inputs) (numbered_uses inputs)
  = Some envs ->
  ∃ (target_env : Meet_env.t)
    (definitions : finite_map (Simples_in_joined_envs.t * flambda_kind))
    (names : list variable),
    Join_env.cut_and_n_way_join0 meet_env_functions
      (join_parameters join_depth) (fuel_of_env fuel) join_choices
      (model_join_of_types join_depth fuel)
      (model_meet_expanded_head join_choices meet_choices join_depth fuel)
      avoid (Meet_env.create (source inputs)) (source inputs) envs
    = Some (target_env, definitions, names)
    ∧ target = Meet_env.typing_env target_env.
Proof.
  intros Hjoin.
  unfold model_join, join_with, cut_and_n_way_join_with_analysis in Hjoin.
  cbv zeta in Hjoin.
  destruct (Join_env.cut_for_joins _ _) as [envs' | ] eqn:Hcut';
    [ | discriminate ].
  intros Hcut. injection Hcut as ->.
  destruct (Join_env.cut_and_n_way_join0 _ _ _ _ _ _ _ _ _ _)
    as [[[menv_t definitions] names] | ] eqn:Hjoin0; [ | discriminate ].
  destruct (Analysis.create _ envs definitions) as [analysis' | ]
    eqn:Hanalysis; [ | discriminate ].
  injection Hjoin as <- <-.
  exists menv_t, definitions, names. split; reflexivity.
Qed.

(* The two maps of the analysis that the join that Simplify calls
   answers: the definitions that the join up to what it adds answers,
   and those of them at the normal mode. *)
Lemma model_join_analysis {A : Type} (join_choices : Join_env.choices)
    (meet_choices : meet_choices) (join_depth : nat) (fuel : fuels)
    (avoid : list variable) (inputs : join_inputs A) (target : Typing_env.t)
    (analysis : Analysis.t A) (envs : Joined_envs.t) (target_env : Meet_env.t)
    (definitions : finite_map (Simples_in_joined_envs.t * flambda_kind))
    (names : list variable) :
  model_join join_choices meet_choices join_depth fuel avoid inputs
  = Some (target, analysis) ->
  Join_env.cut_for_joins (cut_after inputs) (numbered_uses inputs)
  = Some envs ->
  Join_env.cut_and_n_way_join0 meet_env_functions
    (join_parameters join_depth) (fuel_of_env fuel) join_choices
    (model_join_of_types join_depth fuel)
    (model_meet_expanded_head join_choices meet_choices join_depth fuel)
    avoid (Meet_env.create (source inputs)) (source inputs) envs
  = Some (target_env, definitions, names) ->
  ∃ at_normal_mode : finite_map (Simples_in_joined_envs.t * flambda_kind),
    Analysis.definitions_at_normal_mode envs (compact definitions)
    = Some at_normal_mode
    ∧ Analysis.definitions_in_joined_envs analysis = definitions
    ∧ Analysis.canonical_definitions_at_normal_mode analysis = at_normal_mode.
Proof.
  intros Hjoin.
  unfold model_join, join_with, cut_and_n_way_join_with_analysis in Hjoin.
  cbv zeta in Hjoin.
  destruct (Join_env.cut_for_joins _ _) as [envs' | ] eqn:Hcut';
    [ | discriminate ].
  intros Hcut. injection Hcut as ->.
  destruct (Join_env.cut_and_n_way_join0 _ _ _ _ _ _ _ _ _ _)
    as [[[menv_t definitions'] names'] | ] eqn:Hjoin0'; [ | discriminate ].
  intros Hjoin0. injection Hjoin0 as -> -> ->.
  destruct (Analysis.create _ envs definitions) as [analysis' | ]
    eqn:Hanalysis; [ | discriminate ].
  injection Hjoin as _ <-.
  unfold Analysis.create in Hanalysis.
  destruct (Analysis.definitions_at_normal_mode envs (compact definitions))
    as [at_normal_mode | ] eqn:Hnormal; [ | discriminate ].
  injection Hanalysis as <-.
  exists at_normal_mode. split; [ reflexivity | ]. split; reflexivity.
Qed.

(* The join that Simplify calls, where there is an environment to
   join: its target is what the two additions answer. *)
Lemma model_join_replayed {A : Type} (join_choices : Join_env.choices)
    (meet_choices : meet_choices) (join_depth : nat) (fuel : fuels)
    (avoid : list variable) (inputs : join_inputs A) (target : Typing_env.t)
    (analysis : Analysis.t A) (envs : Joined_envs.t) (s3 : Join_env.t)
    (relations3 : Join_env.inverse_relations)
    (kept : list (variable * flambda_kind))
    (equations relation_equations : list (name * flambda_type))
    (definitions : finite_map (Simples_in_joined_envs.t * flambda_kind)) :
  let mh :=
    model_meet_expanded_head join_choices meet_choices join_depth fuel
  in
  model_join join_choices meet_choices join_depth fuel avoid inputs
  = Some (target, analysis) ->
  model_join_run join_choices join_depth fuel avoid inputs envs s3 relations3
    kept equations relation_equations definitions ->
  Keyed.is_empty envs = false ->
  ∃ menv1 target_env : Meet_env.t,
    Meet_env.add_env_extension (model_join_defined inputs s3 kept)
      (Env_extension equations) mh
    = Some menv1
    ∧ Meet_env.add_env_extension menv1 (Env_extension relation_equations) mh
      = Some target_env
    ∧ target = Meet_env.typing_env target_env
    ∧ ∃ at_normal_mode : finite_map (Simples_in_joined_envs.t * flambda_kind),
        Analysis.definitions_at_normal_mode envs (compact definitions)
        = Some at_normal_mode
        ∧ Analysis.definitions_in_joined_envs analysis = definitions
        ∧ Analysis.canonical_definitions_at_normal_mode analysis
          = at_normal_mode.
Proof.
  intros mh Hjoin
    [s1 [s2 [relations2 [Hcut [Haliases [Hdelayed [Hloop Hadds]]]]]]] Hempty.
  change (Join_env.cut_for_joins (cut_after inputs) (numbered_uses inputs)
          = Some envs) in Hcut.
  destruct (model_join_unfold join_choices meet_choices join_depth fuel avoid
              inputs target analysis envs Hjoin Hcut)
    as [target_env [definitions' [names [Hjoin0 Htarget]]]].
  pose proof (model_join_analysis join_choices meet_choices join_depth fuel
                avoid inputs target analysis envs target_env definitions'
                names Hjoin Hcut Hjoin0) as Hanalysis.
  subst target.
  fold mh in Hjoin0.
  set (p := join_parameters join_depth) in *.
  set (jt := model_join_of_types join_depth fuel) in *.
  destruct (cut_and_n_way_join0_replayed Meet_env.t meet_expanded_head
              meet_env_functions p (fuel_of_env fuel) join_choices jt mh avoid
              (Meet_env.create (source inputs)) target_env (source inputs)
              envs definitions' names Hempty Hjoin0)
    as [s1' [s2' [s3' [relations2' [relations3' [kept' [equations'
          [relation_equations' [menv1 [Haliases' [Hdelayed' [Hloop' [Hnames
            [Hadds' [Hadd1 Hadd2]]]]]]]]]]]]]]].
  rewrite Haliases in Haliases'. injection Haliases' as <-.
  rewrite Hdelayed in Hdelayed'. injection Hdelayed' as <- <-.
  change (Join_env.join_depth p) with join_depth in Hloop'.
  rewrite Hloop in Hloop'. injection Hloop' as <- <-.
  subst names.
  cbn [Join_env.with_names_to_avoid Join_env.add_env_extension
         Join_env.current_typing_env meet_env_functions] in Hadds', Hadd1,
    Hadd2.
  destruct (last_phase_adds_functional _ _ _ _ _ _ _ _ _ _ _ _ Hadds Hadds')
    as (-> & -> & -> & ->).
  exists menv1, target_env.
  split; [ exact Hadd1 | ]. split; [ exact Hadd2 | ].
  split; [ reflexivity | exact Hanalysis ].
Qed.

Theorem model_join_keeps : model_join_keeps_statement.
Proof.
  intros A join_choices meet_choices join_depth fuel avoid inputs target
    analysis envs s3 relations3 kept equations relation_equations definitions
    Hmh Hinputs Hkinded Hagree Hfits Hjoin Hrun Hrest.
  pose proof Hrun
    as [s1 [s2 [relations2 [Hcut [Haliases [Hdelayed [Hloop Hadds]]]]]]].
  change (Join_env.cut_for_joins (cut_after inputs) (numbered_uses inputs)
          = Some envs) in Hcut.
  set (mh :=
         model_meet_expanded_head join_choices meet_choices join_depth fuel)
    in *.
  set (initial_env := Meet_env.create (source inputs)) in *.
  destruct (Keyed.is_empty envs) eqn:Hempty.
  { (* No environment to join: the target is bottom. *)
    destruct (model_join_unfold join_choices meet_choices join_depth fuel
                avoid inputs target analysis envs Hjoin Hcut)
      as [target_env [definitions' [names [Hjoin0 ->]]]].
    unfold Join_env.cut_and_n_way_join0 in Hjoin0. rewrite Hempty in Hjoin0.
    injection Hjoin0 as <- _ _.
    cbn [Join_env.make_bottom meet_env_functions].
    destruct (make_bottom_keeps initial_env Hfits) as [Hextends Hfits'].
    split; [ exact Hfits' | ].
    split; [ exact (extends_keeps_source _ _ Hextends) | ].
    intros x Hx Hnot. exfalso. apply Hnot. exact Hx. }
  destruct (model_join_replayed join_choices meet_choices join_depth fuel
              avoid inputs target analysis envs s3 relations3 kept equations
              relation_equations definitions Hjoin Hrun Hempty)
    as [menv1 [target_env [Hadd1 [Hadd2 [-> _]]]]].
  fold mh in Hadd1, Hadd2.
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

Theorem model_join_admits_use : model_join_admits_use_statement.
Proof.
  intros A join_choices meet_choices join_depth fuel avoid inputs target
    analysis envs s3 relations3 kept equations relation_equations definitions
    index use statics valuation Hmh Hsound Hinputs Hkinded Hagree Hfits Hjoin
    Hrun Hrest Hat Hbottom Hstatics Hsat_use Hsat_source Htags Haliases_v
    Hscrutinees Hvalued.
  pose proof Hrun
    as [s1 [s2 [relations2 [Hcut [Haliases [Hdelayed [Hloop Hadds]]]]]]].
  change (Join_env.cut_for_joins (cut_after inputs) (numbered_uses inputs)
          = Some envs) in Hcut.
  destruct (inputs_first_phases join_choices join_depth fuel avoid inputs envs
              s1 s2 s3 relations2 relations3 Hinputs Hkinded Hagree Hcut
              Haliases Hdelayed Hloop)
    as (Htrue3 & Hrelations3 & Hgrows3 & Hstate3 & Hnonempty3 & Hscoped3 & HX
        & HR & Hnames_kept & Hmade & Huse3 & Hinvariant3 & Hkinded3
        & Hsymbols3).
  assert (Huse : Join_env.joined_env s3 index = Some use).
  { apply Huse3. split; [ exact Hat | exact Hbottom ]. }
  assert (Hempty : Keyed.is_empty envs = false).
  { pose proof Huse as Huse0. rewrite (proj1 Hgrows3) in Huse0.
    destruct envs; [ discriminate Huse0 | reflexivity ]. }
  destruct (model_join_replayed join_choices meet_choices join_depth fuel
              avoid inputs target analysis envs s3 relations3 kept equations
              relation_equations definitions Hjoin Hrun Hempty)
    as [menv1 [target_env [Hadd1 [Hadd2 [-> [at_normal_mode [Hnormal
          [Hdefinitions Hat_normal]]]]]]]].
  unfold model_join_defined in Hrest, Hadd1.
  set (mh :=
         model_meet_expanded_head join_choices meet_choices join_depth fuel)
    in *.
  set (menv_a :=
         Meet_env.with_names_to_avoid (Meet_env.create (source inputs))
           (Bindings_in_target_env.names_to_avoid (Join_env.bindings s3)))
    in *.
  destruct (last_phase_satisfied _ (fuel_of_env fuel) (source inputs) s3
              relations3 mh menv_a menv1 target_env kept equations
              relation_equations definitions index use statics valuation
              Htrue3 Hrelations3 Hnonempty3 Hscoped3 Hstate3 Hinvariant3
              Hkinded3 Hsymbols3 HX HR eq_refl Hfits
              (fun x Hx => proj2 (among_In _ _) Hx) Hadds Hrest Hmh Hsound
              Hadd1 Hadd2 Huse Hstatics Hsat_source Hsat_use Htags Haliases_v
              Hscrutinees Hvalued)
    as [valuation' [Hfrom [Hsat [_ [Hfrom_source Hdefs]]]]].
  exists valuation'. split; [ exact Hsat | ].
  split; [ exact Hfrom | ]. split; [ exact Hfrom_source | ].
  rewrite Hdefinitions, Hat_normal. split; [ exact Hdefs | ].
  apply (definitions_at_normal_mode_hold envs definitions at_normal_mode index
           use statics valuation valuation' Hnormal); [ | exact Hsat_use
                                                        | exact Hdefs ].
  rewrite (proj1 Hgrows3) in Huse. exact Huse.
Qed.

(* The identifiers of the uses that are not bottom, by index. *)
Definition ids_of_uses {A : Type} (inputs : join_inputs A)
    : Keyed.finite_map Index.t A :=
  flat_map
    (fun binding : Index.t * (A * Typing_env.t) =>
       if Typing_env.is_bottom (snd (snd binding))
       then []
       else [(fst binding, fst (snd binding))])
    (Index.number (uses inputs)).

Lemma model_join_external_ids {A : Type} (join_choices : Join_env.choices)
    (meet_choices : meet_choices) (join_depth : nat) (fuel : fuels)
    (avoid : list variable) (inputs : join_inputs A) (target : Typing_env.t)
    (analysis : Analysis.t A) :
  model_join join_choices meet_choices join_depth fuel avoid inputs
  = Some (target, analysis) ->
  Analysis.external_ids analysis = ids_of_uses inputs.
Proof.
  intros Hjoin.
  unfold model_join, join_with, cut_and_n_way_join_with_analysis in Hjoin.
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

Lemma ids_of_uses_found {A : Type} (inputs : join_inputs A) (index : Index.t)
    (id : A) :
  Keyed.lookup Index.equal index (ids_of_uses inputs) = Some id ->
  ∃ use : Typing_env.t,
    nth_error (uses inputs) index = Some (id, use)
    ∧ Typing_env.is_bottom use = false.
Proof.
  intros Hlookup.
  apply (keyed_lookup_in Index.equal index_equal_decides) in Hlookup.
  unfold ids_of_uses in Hlookup. apply in_flat_map in Hlookup.
  destruct Hlookup as [[i [id' use]] [Hin Hmem]]. cbn [fst snd] in Hmem.
  destruct (Typing_env.is_bottom use) eqn:Hb; [ destruct Hmem | ].
  destruct Hmem as [Heq | []]. injection Heq as -> ->.
  exists use. split; [ apply number_in; exact Hin | exact Hb ].
Qed.

Lemma ids_distinct {A : Type} :
  ∀ numbered : list (Index.t * (A * Typing_env.t)),
    NoDup (map fst numbered) ->
    NoDup
      (map fst
         (flat_map
            (fun binding : Index.t * (A * Typing_env.t) =>
               if Typing_env.is_bottom (snd (snd binding))
               then []
               else [(fst binding, fst (snd binding))])
            numbered))
    ∧ ∀ i : Index.t,
        i
        ∈ map fst
            (flat_map
               (fun binding : Index.t * (A * Typing_env.t) =>
                  if Typing_env.is_bottom (snd (snd binding))
                  then []
                  else [(fst binding, fst (snd binding))])
               numbered) ->
        i ∈ map fst numbered.
Proof.
  induction numbered as [ | [i [id use]] numbered IH]; intros Hnodup.
  - split; [ constructor | intros i [] ].
  - cbn [map fst] in Hnodup. apply NoDup_cons_iff in Hnodup.
    destruct Hnodup as [Hfresh Hnodup].
    destruct (IH Hnodup) as [Hdistinct Hsub].
    cbn [flat_map fst snd]. destruct (Typing_env.is_bottom use).
    + split; [ exact Hdistinct | ].
      intros j Hj. right. exact (Hsub j Hj).
    + cbn [app map fst]. split.
      * constructor; [ | exact Hdistinct ].
        intros Hin. exact (Hfresh (Hsub i Hin)).
      * intros j [<- | Hj]; [ left; reflexivity | right; exact (Hsub j Hj) ].
Qed.

Lemma ids_of_uses_has {A : Type} (inputs : join_inputs A) (index : Index.t)
    (id : A) (use : Typing_env.t) :
  nth_error (uses inputs) index = Some (id, use) ->
  Typing_env.is_bottom use = false ->
  Keyed.lookup Index.equal index (ids_of_uses inputs) = Some id.
Proof.
  intros Hnth Hb.
  apply (keyed_lookup_of_in Index.equal index_equal_decides).
  - exact (proj1 (ids_distinct _ (combine_seq_distinct (uses inputs) 0))).
  - unfold ids_of_uses. apply in_flat_map. exists (index, (id, use)).
    split; [ apply number_in; exact Hnth | ].
    cbn [fst snd]. rewrite Hb. left. reflexivity.
Qed.

Lemma mem_at_mode_mem (tenv : Typing_env.t) (mode : Name_mode.t)
    (s : simple) :
  Typing_env.mem_simple (Some mode) tenv s = true ->
  Typing_env.mem_simple None tenv s = true.
Proof.
  destruct s as [x | sym | c]; cbn [Typing_env.mem_simple];
    [ | intros H; exact H | reflexivity ].
  unfold Typing_env.mem.
  destruct (Keyed.lookup name_eqb (Name_var x) (Typing_env.names_to_types tenv))
    as [[ty btm] | ]; [ reflexivity | discriminate ].
Qed.

(* The canonical element at a name mode of a [Simple] of some kind is
   of that kind. *)
Lemma canonical_at_mode_kinded (tenv : Typing_env.t) (mode : Name_mode.t)
    (s canonical : simple) (kind : flambda_kind) :
  typing_env_invariant tenv ->
  typing_env_kinded tenv ->
  simple_of_kind tenv s kind ->
  Typing_env.get_canonical_simple_exn tenv (Some mode) None s
  = Some (Some canonical) ->
  simple_of_kind tenv canonical kind
  ∧ Typing_env.mem_simple (Some mode) tenv canonical = true.
Proof.
  intros Hinvariant Hkinded Hk Hcanonical.
  pose proof (simple_of_kind_mem_simple tenv s kind Hinvariant Hk) as Hmem.
  pose proof (get_canonical_simple_bound tenv mode s canonical
                (proj1 Hinvariant) Hmem Hcanonical) as Hbound.
  split; [ | exact Hbound ].
  destruct (mem_simple_of_some_kind tenv canonical
              (mem_at_mode_mem tenv mode canonical Hbound)) as [kind' Hk'].
  destruct (proj1 (typing_env_well_formed_iff tenv) (proj1 Hinvariant))
    as [_ [_ [Hconsistent _]]].
  unfold Typing_env.get_canonical_simple_exn in Hcanonical.
  destruct (option_map _ _) as [name_mode | ]; [ | discriminate ].
  cbn in Hcanonical. injection Hcanonical as Hcan.
  pose proof (canonical_at_mode_aliased _ _ _ _ _ _ _ _ _ Hconsistent Hcan)
    as Haliased.
  rewrite (proj1 (proj2 Hkinded) canonical s kind' kind Haliased Hk' Hk)
    in Hk'.
  exact Hk'.
Qed.

Theorem model_join_describes_uses : model_join_describes_uses_statement.
Proof.
  intros A join_choices meet_choices join_depth fuel avoid inputs target
    analysis envs s3 relations3 kept equations relation_equations definitions
    Hmh Hinputs Hkinded Hagree Hfits Hjoin Hrun Hrest.
  pose proof Hrun
    as [s1 [s2 [relations2 [Hcut [Haliases [Hdelayed [Hloop Hadds]]]]]]].
  change (Join_env.cut_for_joins (cut_after inputs) (numbered_uses inputs)
          = Some envs) in Hcut.
  pose proof (model_join_external_ids join_choices meet_choices join_depth
                fuel avoid inputs target analysis Hjoin) as Hids.
  assert (Hids_found :
            ∀ (index : Index.t) (id : A),
              Keyed.lookup Index.equal index (Analysis.external_ids analysis)
              = Some id ->
              ∃ use : Typing_env.t,
                nth_error (uses inputs) index = Some (id, use)
                ∧ Typing_env.is_bottom use = false).
  { intros index id Hlookup. rewrite Hids in Hlookup.
    exact (ids_of_uses_found inputs index id Hlookup). }
  set (mh :=
         model_meet_expanded_head join_choices meet_choices join_depth fuel)
    in *.
  set (initial_env := Meet_env.create (source inputs)) in *.
  destruct (Keyed.is_empty envs) eqn:Hempty.
  { (* No environment to join: the analysis has no definition. *)
    destruct (model_join_unfold join_choices meet_choices join_depth fuel
                avoid inputs target analysis envs Hjoin Hcut)
      as [target_env [definitions' [names [Hjoin0 Htarget]]]].
    destruct (model_join_analysis join_choices meet_choices join_depth fuel
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
  destruct (model_join_replayed join_choices meet_choices join_depth fuel
              avoid inputs target analysis envs s3 relations3 kept equations
              relation_equations definitions Hjoin Hrun Hempty)
    as [menv1 [target_env [Hadd1 [Hadd2 [-> [at_normal_mode [Hnormal
          [Hdefinitions Hat_normal]]]]]]]].
  fold mh in Hadd1, Hadd2.
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

(* ---------------------------------------------------------------------- *)
(* The same join, with a meet of expanded types that is given              *)
(* ---------------------------------------------------------------------- *)

(* The join with a meet of expanded types that is given, unfolded. *)
Lemma join_with_unfold {A : Type} (join_choices : Join_env.choices)
    (join_depth : nat) (fuel : fuels) (mh : meet_expanded_head)
    (avoid : list variable) (inputs : join_inputs A) (target : Typing_env.t)
    (analysis : Analysis.t A) (envs : Joined_envs.t) :
  model_join_with join_choices join_depth fuel mh avoid inputs
  = Some (target, analysis) ->
  Join_env.cut_for_joins (cut_after inputs) (numbered_uses inputs)
  = Some envs ->
  ∃ (target_env : Meet_env.t)
    (definitions : finite_map (Simples_in_joined_envs.t * flambda_kind))
    (names : list variable),
    Join_env.cut_and_n_way_join0 meet_env_functions
      (join_parameters join_depth) (fuel_of_env fuel) join_choices
      (model_join_of_types join_depth fuel)
      mh
      avoid (Meet_env.create (source inputs)) (source inputs) envs
    = Some (target_env, definitions, names)
    ∧ target = Meet_env.typing_env target_env.
Proof.
  intros Hjoin.
  unfold model_join_with, join_with, cut_and_n_way_join_with_analysis in Hjoin.
  cbv zeta in Hjoin.
  destruct (Join_env.cut_for_joins _ _) as [envs' | ] eqn:Hcut';
    [ | discriminate ].
  intros Hcut. injection Hcut as ->.
  destruct (Join_env.cut_and_n_way_join0 _ _ _ _ _ _ _ _ _ _)
    as [[[menv_t definitions] names] | ] eqn:Hjoin0; [ | discriminate ].
  destruct (Analysis.create _ envs definitions) as [analysis' | ]
    eqn:Hanalysis; [ | discriminate ].
  injection Hjoin as <- <-.
  exists menv_t, definitions, names. split; reflexivity.
Qed.

(* The two maps of the analysis that the join with a meet that is
   given answers: the definitions that the join up to what it adds answers,
   and those of them at the normal mode. *)
Lemma join_with_analysis {A : Type} (join_choices : Join_env.choices)
    (join_depth : nat) (fuel : fuels) (mh : meet_expanded_head)
    (avoid : list variable) (inputs : join_inputs A) (target : Typing_env.t)
    (analysis : Analysis.t A) (envs : Joined_envs.t) (target_env : Meet_env.t)
    (definitions : finite_map (Simples_in_joined_envs.t * flambda_kind))
    (names : list variable) :
  model_join_with join_choices join_depth fuel mh avoid inputs
  = Some (target, analysis) ->
  Join_env.cut_for_joins (cut_after inputs) (numbered_uses inputs)
  = Some envs ->
  Join_env.cut_and_n_way_join0 meet_env_functions
    (join_parameters join_depth) (fuel_of_env fuel) join_choices
    (model_join_of_types join_depth fuel)
    mh
    avoid (Meet_env.create (source inputs)) (source inputs) envs
  = Some (target_env, definitions, names) ->
  ∃ at_normal_mode : finite_map (Simples_in_joined_envs.t * flambda_kind),
    Analysis.definitions_at_normal_mode envs (compact definitions)
    = Some at_normal_mode
    ∧ Analysis.definitions_in_joined_envs analysis = definitions
    ∧ Analysis.canonical_definitions_at_normal_mode analysis = at_normal_mode.
Proof.
  intros Hjoin.
  unfold model_join_with, join_with, cut_and_n_way_join_with_analysis in Hjoin.
  cbv zeta in Hjoin.
  destruct (Join_env.cut_for_joins _ _) as [envs' | ] eqn:Hcut';
    [ | discriminate ].
  intros Hcut. injection Hcut as ->.
  destruct (Join_env.cut_and_n_way_join0 _ _ _ _ _ _ _ _ _ _)
    as [[[menv_t definitions'] names'] | ] eqn:Hjoin0'; [ | discriminate ].
  intros Hjoin0. injection Hjoin0 as -> -> ->.
  destruct (Analysis.create _ envs definitions) as [analysis' | ]
    eqn:Hanalysis; [ | discriminate ].
  injection Hjoin as _ <-.
  unfold Analysis.create in Hanalysis.
  destruct (Analysis.definitions_at_normal_mode envs (compact definitions))
    as [at_normal_mode | ] eqn:Hnormal; [ | discriminate ].
  injection Hanalysis as <-.
  exists at_normal_mode. split; [ reflexivity | ]. split; reflexivity.
Qed.

(* The join with a meet that is given, where there is an environment
   to join: its target is what the two additions answer. *)
Lemma join_with_replayed {A : Type} (join_choices : Join_env.choices)
    (join_depth : nat) (fuel : fuels) (mh : meet_expanded_head)
    (avoid : list variable) (inputs : join_inputs A) (target : Typing_env.t)
    (analysis : Analysis.t A) (envs : Joined_envs.t) (s3 : Join_env.t)
    (relations3 : Join_env.inverse_relations)
    (kept : list (variable * flambda_kind))
    (equations relation_equations : list (name * flambda_type))
    (definitions : finite_map (Simples_in_joined_envs.t * flambda_kind)) :
  model_join_with join_choices join_depth fuel mh avoid inputs
  = Some (target, analysis) ->
  model_join_run join_choices join_depth fuel avoid inputs envs s3 relations3
    kept equations relation_equations definitions ->
  Keyed.is_empty envs = false ->
  ∃ menv1 target_env : Meet_env.t,
    Meet_env.add_env_extension (model_join_defined inputs s3 kept)
      (Env_extension equations) mh
    = Some menv1
    ∧ Meet_env.add_env_extension menv1 (Env_extension relation_equations) mh
      = Some target_env
    ∧ target = Meet_env.typing_env target_env
    ∧ ∃ at_normal_mode : finite_map (Simples_in_joined_envs.t * flambda_kind),
        Analysis.definitions_at_normal_mode envs (compact definitions)
        = Some at_normal_mode
        ∧ Analysis.definitions_in_joined_envs analysis = definitions
        ∧ Analysis.canonical_definitions_at_normal_mode analysis
          = at_normal_mode.
Proof.
  intros Hjoin
    [s1 [s2 [relations2 [Hcut [Haliases [Hdelayed [Hloop Hadds]]]]]]] Hempty.
  change (Join_env.cut_for_joins (cut_after inputs) (numbered_uses inputs)
          = Some envs) in Hcut.
  destruct (join_with_unfold join_choices join_depth fuel mh avoid
              inputs target analysis envs Hjoin Hcut)
    as [target_env [definitions' [names [Hjoin0 Htarget]]]].
  pose proof (join_with_analysis join_choices join_depth fuel mh
                avoid inputs target analysis envs target_env definitions'
                names Hjoin Hcut Hjoin0) as Hanalysis.
  subst target.
  set (p := join_parameters join_depth) in *.
  set (jt := model_join_of_types join_depth fuel) in *.
  destruct (cut_and_n_way_join0_replayed Meet_env.t meet_expanded_head
              meet_env_functions p (fuel_of_env fuel) join_choices jt mh avoid
              (Meet_env.create (source inputs)) target_env (source inputs)
              envs definitions' names Hempty Hjoin0)
    as [s1' [s2' [s3' [relations2' [relations3' [kept' [equations'
          [relation_equations' [menv1 [Haliases' [Hdelayed' [Hloop' [Hnames
            [Hadds' [Hadd1 Hadd2]]]]]]]]]]]]]]].
  rewrite Haliases in Haliases'. injection Haliases' as <-.
  rewrite Hdelayed in Hdelayed'. injection Hdelayed' as <- <-.
  change (Join_env.join_depth p) with join_depth in Hloop'.
  rewrite Hloop in Hloop'. injection Hloop' as <- <-.
  subst names.
  cbn [Join_env.with_names_to_avoid Join_env.add_env_extension
         Join_env.current_typing_env meet_env_functions] in Hadds', Hadd1,
    Hadd2.
  destruct (last_phase_adds_functional _ _ _ _ _ _ _ _ _ _ _ _ Hadds Hadds')
    as (-> & -> & -> & ->).
  exists menv1, target_env.
  split; [ exact Hadd1 | ]. split; [ exact Hadd2 | ].
  split; [ reflexivity | exact Hanalysis ].
Qed.

Theorem model_join_with_admits_use
  : model_join_with_admits_use_statement.
Proof.
  intros A join_choices join_depth fuel mh avoid inputs target
    analysis envs s3 relations3 kept equations relation_equations definitions
    index use statics valuation Hmh Hsound Hinputs Hkinded Hagree Hfits Hjoin
    Hrun Hrest Hat Hbottom Hstatics Hsat_use Hsat_source Htags Haliases_v
    Hscrutinees Hvalued.
  pose proof Hrun
    as [s1 [s2 [relations2 [Hcut [Haliases [Hdelayed [Hloop Hadds]]]]]]].
  change (Join_env.cut_for_joins (cut_after inputs) (numbered_uses inputs)
          = Some envs) in Hcut.
  destruct (inputs_first_phases join_choices join_depth fuel avoid inputs envs
              s1 s2 s3 relations2 relations3 Hinputs Hkinded Hagree Hcut
              Haliases Hdelayed Hloop)
    as (Htrue3 & Hrelations3 & Hgrows3 & Hstate3 & Hnonempty3 & Hscoped3 & HX
        & HR & Hnames_kept & Hmade & Huse3 & Hinvariant3 & Hkinded3
        & Hsymbols3).
  assert (Huse : Join_env.joined_env s3 index = Some use).
  { apply Huse3. split; [ exact Hat | exact Hbottom ]. }
  assert (Hempty : Keyed.is_empty envs = false).
  { pose proof Huse as Huse0. rewrite (proj1 Hgrows3) in Huse0.
    destruct envs; [ discriminate Huse0 | reflexivity ]. }
  destruct (join_with_replayed join_choices join_depth fuel mh
              avoid inputs target analysis envs s3 relations3 kept equations
              relation_equations definitions Hjoin Hrun Hempty)
    as [menv1 [target_env [Hadd1 [Hadd2 [-> [at_normal_mode [Hnormal
          [Hdefinitions Hat_normal]]]]]]]].
  unfold model_join_defined in Hrest, Hadd1.
  set (menv_a :=
         Meet_env.with_names_to_avoid (Meet_env.create (source inputs))
           (Bindings_in_target_env.names_to_avoid (Join_env.bindings s3)))
    in *.
  destruct (last_phase_satisfied _ (fuel_of_env fuel) (source inputs) s3
              relations3 mh menv_a menv1 target_env kept equations
              relation_equations definitions index use statics valuation
              Htrue3 Hrelations3 Hnonempty3 Hscoped3 Hstate3 Hinvariant3
              Hkinded3 Hsymbols3 HX HR eq_refl Hfits
              (fun x Hx => proj2 (among_In _ _) Hx) Hadds Hrest Hmh Hsound
              Hadd1 Hadd2 Huse Hstatics Hsat_source Hsat_use Htags Haliases_v
              Hscrutinees Hvalued)
    as [valuation' [Hfrom [Hsat [_ [Hfrom_source Hdefs]]]]].
  exists valuation'. split; [ exact Hsat | ].
  split; [ exact Hfrom | ]. split; [ exact Hfrom_source | ].
  rewrite Hdefinitions, Hat_normal. split; [ exact Hdefs | ].
  apply (definitions_at_normal_mode_hold envs definitions at_normal_mode index
           use statics valuation valuation' Hnormal); [ | exact Hsat_use
                                                        | exact Hdefs ].
  rewrite (proj1 Hgrows3) in Huse. exact Huse.
Qed.

Theorem meet_with_forgetful_join_has_contracts
  : meet_with_forgetful_join_has_contracts_statement.
Proof.
  intros fuel.
  exact (meet_expanded_head_fuel_ok forgetful_join repaired_choices eq_refl
           (fun mh Hkeeps Hsound _ =>
              value_heads_of_join forgetful_join mh forgetful_join_sound
                Hkeeps Hsound)
           fuel).
Qed.
