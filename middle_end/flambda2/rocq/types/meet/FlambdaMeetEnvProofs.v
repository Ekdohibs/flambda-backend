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



(* Proofs of the statements of FlambdaMeetEnvSpec.v. Nothing here
   needs review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaPrimitiveProofs
  FlambdaClosuresSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeMeaningSpec FlambdaTypeMeaningProofs
  FlambdaTypeWellFormedSpec FlambdaTypeWellFormedProofs
  FlambdaTypeAliasKindsSpec FlambdaTypeAliasKindsProofs
  FlambdaBindingTime FlambdaEnvAliases FlambdaEnvAliasesSpec
  FlambdaEnvAliasesProofs FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaTypingEnvProofs FlambdaTypingEnvInvariantSpec
  FlambdaTypingEnvInvariantProofs FlambdaTypingEnvOperationsSpec
  FlambdaTypingEnvOperationsProofs FlambdaTypingEnvKindedSpec
  FlambdaTypingEnvKindedProofs FlambdaMeetGrammar FlambdaExpandHead
  FlambdaMeetEnv FlambdaMeet FlambdaMeetSpec FlambdaMeetProofs
  FlambdaExpandHeadSpec FlambdaExpandHeadProofs FlambdaMeetStepsSpec
  FlambdaMeetStepsProofs FlambdaMeetEnvSpec.
Import ListNotations.

(* The proofs below, as far as the end of the section, are for any
   predicate on static environments that implies validity
   (FlambdaMeetValidity.v): in them the names of the four contracts
   that suppose a valid static environment stand for the same with
   that predicate. The theorems of the signed statements are after the
   section, at validity itself. *)
From OxCamlFlambda2Types Require Import FlambdaMeetValidity.

Section Validity.

Context {V : validity}.

Abbreviation statics_valid := valid.
Abbreviation meet_expanded_head_sound := meet_expanded_head_sound_for.
Abbreviation join_scoped_envs_sound := join_scoped_envs_sound_for.
Abbreviation meet_call_ok := meet_call_ok_for.
Abbreviation part_meet_ok := part_meet_ok_for.
Abbreviation meet_disjunction_right := meet_disjunction_right_for.
Abbreviation type_part_meet_ok := type_part_meet_ok_for.
Abbreviation meet_row_like_for_closures_right :=
  meet_row_like_for_closures_right_for.
Abbreviation value_heads_right := value_heads_right_for.

(* ======================================================================== *)
(* From one environment to another that has the same names                  *)
(* ======================================================================== *)

(* A type that fits an environment fits another in which the variables
   that were bound are bound and every simple has the kinds that it
   had. *)
Lemma type_fits_kept (tenv tenv' : Typing_env.t) (ty : flambda_type) :
  (∀ x : variable,
     Typing_env.mem None tenv (Name_var x) = true ->
     Typing_env.mem None tenv' (Name_var x) = true) ->
  (∀ (s : simple) (kind : flambda_kind),
     simple_of_kind tenv s kind -> simple_of_kind tenv' s kind) ->
  type_fits tenv ty ->
  type_fits tenv' ty.
Proof.
  intros Hmem Hkind (Hwf & Hclosed & Hkinded & Hequations).
  split; [ exact Hwf | ]. split.
  { intros x Hin. exact (Hmem x (Hclosed x Hin)). }
  split.
  - intros s kind Hin. exact (Hkind s kind (Hkinded s kind Hin)).
  - intros n kind Hin. exact (Hkind _ kind (Hequations n kind Hin)).
Qed.

Lemma is_obviously_bottom_unknown (kind : flambda_kind) :
  is_obviously_bottom (unknown kind) = false.
Proof. destruct kind as [ | [] | | ]; reflexivity. Qed.

Lemma aliases_with_kinds_unknown (kind : flambda_kind) :
  aliases_with_kinds (unknown kind) = [].
Proof. destruct kind as [ | [] | | ]; reflexivity. Qed.

(* ======================================================================== *)
(* [make_bottom]                                                            *)
(* ======================================================================== *)

Lemma make_bottom_keeps_for :
  for_validity make_bottom_keeps_statement.
Proof.
  intros menv (Hinvariant & Hwf & Hkinded & Hequations & Hbottom).
  split.
  - repeat split; try reflexivity.
    + intros s kind Hkind. apply make_bottom_simple_of_kind. exact Hkind.
    + intros x Hamong. exact Hamong.
    + intros n ty binding_time_and_mode Hlookup. exists ty.
      split; [ exact Hlookup | reflexivity ].
  - change (Meet_env.typing_env (Meet_env.make_bottom menv))
      with (Typing_env.make_bottom (Meet_env.typing_env menv)).
    split; [ exact (make_bottom_invariant _ Hinvariant) | ].
    split; [ exact (make_bottom_types_well_formed _ Hwf) | ].
    split; [ exact (make_bottom_kinded _ Hkinded) | ].
    split; [ exact (make_bottom_equations_kinded _ Hequations) | ].
    intros Hfalse. discriminate Hfalse.
Qed.

(* ======================================================================== *)
(* [add_variable_definition]                                                *)
(* ======================================================================== *)

Lemma add_variable_definition_names (tenv : Typing_env.t) (var : variable)
    (kind : flambda_kind) (name_mode : Name_mode.t) :
  Typing_env.names_to_types
    (Typing_env.add_variable_definition tenv var kind name_mode)
  = (Name_var var,
     (unknown kind,
      Binding_time.With_name_mode.create (Typing_env.next_binding_time tenv)
        name_mode))
    :: Typing_env.names_to_types tenv.
Proof. reflexivity. Qed.

Lemma add_variable_definition_keeps_for :
  for_validity add_variable_definition_keeps_statement.
Proof.
  intros menv var kind name_mode
    (Hinvariant & Hwf & Hkinded & Hequations & Hbottom) Hfresh.
  set (menv' := Meet_env.add_variable_definition menv var kind name_mode).
  set (tenv := Meet_env.typing_env menv) in *.
  set (tenv' := Typing_env.add_variable_definition tenv var kind name_mode).
  assert (Hnone :
    Keyed.lookup name_eqb (Name_var var) (Typing_env.names_to_types tenv)
    = None).
  { destruct (Keyed.lookup name_eqb (Name_var var)
                (Typing_env.names_to_types tenv))
      as [ [ty binding_time_and_mode] | ] eqn:Hlookup; [ | reflexivity ].
    rewrite (proj2 (mem_var_lookup tenv var)) in Hfresh;
      [ discriminate Hfresh | ].
    exists ty, binding_time_and_mode. exact Hlookup. }
  assert (Hother :
    ∀ n : name,
      n <> Name_var var ->
      Keyed.lookup name_eqb n (Typing_env.names_to_types tenv')
      = Keyed.lookup name_eqb n (Typing_env.names_to_types tenv)).
  { intros n Hn. unfold tenv'. rewrite add_variable_definition_names.
    apply (keyed_lookup_other name_eqb name_eqb_eq). exact Hn. }
  assert (Hkeeps : keeps_names tenv tenv').
  { intros n ty binding_time_and_mode Hlookup. exists ty.
    split; [ | reflexivity ]. rewrite Hother; [ exact Hlookup | ].
    intros ->. rewrite Hnone in Hlookup. discriminate Hlookup. }
  assert (Hkind :
    ∀ (s : simple) (kind' : flambda_kind),
      simple_of_kind tenv s kind' -> simple_of_kind tenv' s kind').
  { intros s kind' Hs. apply add_variable_definition_simple_of_kind.
    destruct (simple_equal s (Var var)) eqn:Hsame.
    - apply simple_equal_decides_equality in Hsame. subst s.
      destruct Hs as [ty Hfind]. exfalso.
      destruct (find_cases tenv (Name_var var) (Some kind') ty Hfind)
        as [ [binding_time_and_mode Hlookup] | Hrest ].
      + rewrite Hnone in Hlookup. discriminate Hlookup.
      + unfold Typing_env.find,
          Typing_env.find_with_binding_time_and_mode' in Hfind.
        rewrite Hnone in Hfind. discriminate Hfind.
    - right. split; [ | exact Hs ]. intros ->.
      rewrite (proj2 (simple_equal_decides_equality _ _) eq_refl) in Hsame.
      discriminate Hsame. }
  split.
  - repeat split; try reflexivity.
    + exact Hkind.
    + intros x Hamong. exact Hamong.
    + exact Hkeeps.
  - change (Meet_env.typing_env menv') with tenv'.
    split.
    { exact (add_variable_definition_invariant tenv var kind name_mode
               Hinvariant Hfresh). }
    split.
    { exact (add_variable_definition_types_well_formed tenv var kind
               name_mode Hwf). }
    split.
    { exact (add_variable_definition_kinded tenv var kind name_mode
               (proj1 Hinvariant) Hfresh Hkinded). }
    split.
    { exact (add_variable_definition_equations_kinded tenv var kind
               name_mode Hfresh Hequations). }
    intros Hnot_bottom n ty binding_time_and_mode Hlookup.
    destruct (name_eqb n (Name_var var)) eqn:Hn.
    + apply name_eqb_eq in Hn. subst n. unfold tenv' in Hlookup.
      rewrite add_variable_definition_names in Hlookup.
      rewrite (keyed_lookup_head name_eqb name_eqb_eq) in Hlookup.
      injection Hlookup as <- _. apply is_obviously_bottom_unknown.
    + rewrite Hother in Hlookup.
      * exact (Hbottom Hnot_bottom n ty binding_time_and_mode Hlookup).
      * intros ->. rewrite (proj2 (name_eqb_eq _ _) eq_refl) in Hn.
        discriminate Hn.
Qed.

(* ======================================================================== *)
(* The typing environment of a meet, extended                               *)
(* ======================================================================== *)

(* What [meet_env_extends] says of the two typing environments. *)
Definition tenv_extends (tenv tenv' : Typing_env.t) : Prop :=
  Typing_env.current_scope tenv' = Typing_env.current_scope tenv
  ∧ Typing_env.prev_levels tenv' = Typing_env.prev_levels tenv
  ∧ Typing_env.min_binding_time tenv' = Typing_env.min_binding_time tenv
  ∧ Typing_env.defined_symbols tenv' = Typing_env.defined_symbols tenv
  ∧ (∀ (s : simple) (kind : flambda_kind),
      simple_of_kind tenv s kind -> simple_of_kind tenv' s kind)
  ∧ keeps_names tenv tenv'.

Lemma tenv_extends_refl (tenv : Typing_env.t) : tenv_extends tenv tenv.
Proof.
  repeat split; try reflexivity.
  - intros s kind Hkind. exact Hkind.
  - apply keeps_names_refl.
Qed.

Lemma tenv_extends_trans (tenv1 tenv2 tenv3 : Typing_env.t) :
  tenv_extends tenv1 tenv2 -> tenv_extends tenv2 tenv3 ->
  tenv_extends tenv1 tenv3.
Proof.
  intros (Hs12 & Hp12 & Hm12 & Hd12 & Hkind12 & Hk12)
    (Hs23 & Hp23 & Hm23 & Hd23 & Hkind23 & Hk23).
  repeat split; try congruence.
  - intros s kind Hkind. exact (Hkind23 s kind (Hkind12 s kind Hkind)).
  - exact (keeps_names_trans _ _ _ Hk12 Hk23).
Qed.

Lemma meet_env_extends_tenv (menv menv' : Meet_env.t) :
  meet_env_extends menv menv' ->
  tenv_extends (Meet_env.typing_env menv) (Meet_env.typing_env menv').
Proof.
  intros (Hs & Hp & Hm & Hd & Hkind & _ & _ & Hk).
  repeat split; assumption.
Qed.

(* The environment of a meet with another typing environment that
   extends its own. *)
Lemma meet_env_extends_with (menv : Meet_env.t) (tenv' : Typing_env.t) :
  tenv_extends (Meet_env.typing_env menv) tenv' ->
  meet_env_extends menv (Meet_env.with_typing_env menv tenv').
Proof.
  intros (Hs & Hp & Hm & Hd & Hkind & Hk).
  repeat split; try assumption; try reflexivity.
  intros x Hamong. exact Hamong.
Qed.

(* The second environment knows no variable that the first does not. *)
Definition no_new_variables (tenv tenv' : Typing_env.t) : Prop :=
  ∀ x : variable,
    Typing_env.mem None tenv' (Name_var x) = true ->
    Typing_env.mem None tenv (Name_var x) = true.

(* What a function that runs the meet of expanded types does to the
   environment of a meet: it extends it, and the variables that it
   defines are new. *)
Definition meet_env_steps (menv menv' : Meet_env.t) : Prop :=
  meet_env_extends menv menv' ∧ new_names_avoided menv menv'.

Lemma meet_env_steps_refl (menv : Meet_env.t) : meet_env_steps menv menv.
Proof.
  split; [ apply meet_env_extends_refl | apply new_names_avoided_refl ].
Qed.

Lemma meet_env_steps_trans (menv1 menv2 menv3 : Meet_env.t) :
  meet_env_steps menv1 menv2 -> meet_env_steps menv2 menv3 ->
  meet_env_steps menv1 menv3.
Proof.
  intros [Hextends12 Hnames12] [Hextends23 Hnames23]. split.
  - exact (meet_env_extends_trans _ _ _ Hextends12 Hextends23).
  - exact (new_names_avoided_trans _ _ _ Hextends12 Hextends23 Hnames12
             Hnames23).
Qed.

(* The environment of a meet with another typing environment, which
   extends its own and knows no other variable. *)
Lemma meet_env_steps_with (menv : Meet_env.t) (tenv' : Typing_env.t) :
  tenv_extends (Meet_env.typing_env menv) tenv' ->
  no_new_variables (Meet_env.typing_env menv) tenv' ->
  meet_env_steps menv (Meet_env.with_typing_env menv tenv').
Proof.
  intros Hextends Hnone.
  split; [ exact (meet_env_extends_with menv tenv' Hextends) | ].
  intros x Hknown Hunknown.
  change (Meet_env.typing_env (Meet_env.with_typing_env menv tenv'))
    with tenv' in Hknown.
  rewrite (Hnone x Hknown) in Hunknown. discriminate Hunknown.
Qed.

(* What the statements say, from what the lemmas give. *)
Lemma meet_env_steps_kept (menv menv' : Meet_env.t) :
  meet_env_steps menv menv'
  ∧ typing_env_fits (Meet_env.typing_env menv') ->
  meet_env_kept_by_meet menv menv'.
Proof.
  intros [[Hextends Hnames] Hfits].
  split; [ split; assumption | exact Hnames ].
Qed.

(* A name that is bound stays bound. *)
Lemma tenv_extends_mem (tenv tenv' : Typing_env.t) (n : name) :
  tenv_extends tenv tenv' ->
  Typing_env.mem None tenv n = true ->
  Typing_env.mem None tenv' n = true.
Proof.
  intros (_ & _ & _ & Hd & _ & Hk) Hmem. destruct n as [ x | sym ].
  - exact (keeps_names_mem _ _ _ Hk Hmem).
  - unfold Typing_env.mem in *. rewrite Hd. exact Hmem.
Qed.

Lemma tenv_extends_type_fits (tenv tenv' : Typing_env.t)
    (ty : flambda_type) :
  tenv_extends tenv tenv' -> type_fits tenv ty -> type_fits tenv' ty.
Proof.
  intros Hextends. apply type_fits_kept.
  - intros x Hmem. exact (tenv_extends_mem _ _ _ Hextends Hmem).
  - exact (proj1 (proj2 (proj2 (proj2 (proj2 Hextends))))).
Qed.

(* What the invariant says of the names. *)
Lemma invariant_names (tenv : Typing_env.t) :
  typing_env_invariant tenv ->
  types_closed tenv ∧ symbols_defined tenv ∧ symbols_recorded tenv.
Proof.
  intros [Hwf _]. apply typing_env_well_formed_iff in Hwf.
  destruct Hwf as [[_ [_ [_ [Hclosed [Hsymbols Hrecorded]]]]] _].
  auto.
Qed.

(* A name that is of a kind is bound. *)
Lemma simple_of_kind_mem (tenv : Typing_env.t) (n : name)
    (kind : flambda_kind) :
  typing_env_invariant tenv ->
  simple_of_kind tenv (simple_of_name n) kind ->
  Typing_env.mem None tenv n = true.
Proof.
  intros Hinvariant Hkind.
  destruct (invariant_names tenv Hinvariant) as (_ & Hsymbols & _).
  apply simple_of_kind_name in Hkind.
  destruct Hkind as [ (ty & binding_time_and_mode & Hlookup & _)
                    | (_ & Hmem & _) ]; [ | exact Hmem ].
  destruct n as [ x | sym ].
  - apply mem_var_lookup. exists ty, binding_time_and_mode. exact Hlookup.
  - exact (Hsymbols sym ty binding_time_and_mode Hlookup).
Qed.

(* ======================================================================== *)
(* A name is given a type                                                   *)
(* ======================================================================== *)

Section Replace_equation.

Variable tenv : Typing_env.t.
Variable n : name.
Variable ty : flambda_type.

Hypothesis Hinvariant : typing_env_invariant tenv.
Hypothesis Hkind : simple_of_kind tenv (simple_of_name n) (kind ty).

Let tenv' := Typing_env.replace_equation tenv n ty.

(* The type that the name had is of the kind of the new one. *)
Lemma replace_equation_same_kind (ty0 : flambda_type)
    (binding_time_and_mode : Binding_time.With_name_mode.t) :
  Keyed.lookup name_eqb n (Typing_env.names_to_types tenv)
  = Some (ty0, binding_time_and_mode) ->
  kind ty = kind ty0.
Proof.
  intros Hlookup. apply simple_of_kind_name in Hkind.
  destruct Hkind as [ (ty1 & binding_time_and_mode1 & Hlookup1 & Hkind1)
                    | (Hnone & _) ].
  - rewrite Hlookup in Hlookup1. injection Hlookup1 as <- _.
    symmetry. exact Hkind1.
  - rewrite Hlookup in Hnone. discriminate Hnone.
Qed.

Lemma replace_equation_keeps_names : keeps_names tenv tenv'.
Proof.
  destruct (invariant_names tenv Hinvariant) as (_ & _ & Hrecorded).
  intros m ty0 binding_time_and_mode Hlookup.
  destruct (name_eqb m n) eqn:Hm.
  - apply name_eqb_eq in Hm. subst m. exists ty.
    split; [ | exact (replace_equation_same_kind _ _ Hlookup) ].
    unfold tenv'. rewrite replace_equation_lookup.
    destruct n as [ x | sym ].
    + rewrite Hlookup. reflexivity.
    + rewrite (Hrecorded sym ty0 binding_time_and_mode Hlookup).
      reflexivity.
  - exists ty0. split; [ | reflexivity ].
    unfold tenv'. rewrite replace_equation_lookup_other; [ exact Hlookup | ].
    intros ->. rewrite (proj2 (name_eqb_eq _ _) eq_refl) in Hm.
    discriminate Hm.
Qed.

Lemma replace_equation_tenv_extends : tenv_extends tenv tenv'.
Proof.
  assert (Hframe := replace_equation_frame tenv n ty). cbv zeta in Hframe.
  destruct Hframe as (Hsymbols & Hprev & Hscope & _ & _ & _ & Hmin & _).
  split; [ exact Hscope | ]. split; [ exact Hprev | ].
  split; [ exact Hmin | ]. split; [ exact Hsymbols | ].
  split; [ | exact replace_equation_keeps_names ].
  intros s kind0 Hs.
  exact (proj2 (replace_equation_simple_of_kind tenv n ty s kind0 Hkind) Hs).
Qed.

(* An environment that fits, with the new type for the name, fits: if
   the type fits, is not obviously bottom nor obviously unknown, and is
   not an alias unless the name is not canonical. *)
Lemma replace_equation_fits :
  typing_env_fits tenv ->
  type_fits tenv ty ->
  (∀ alias : simple,
     get_alias_opt ty = Some alias ->
     ~ is_canonical (Typing_env.aliases tenv) (simple_of_name n)) ->
  is_obviously_unknown ty = false ->
  is_obviously_bottom ty = false ->
  typing_env_fits tenv'.
Proof.
  intros (_ & Hwf & Hkinded & Hequations & Hbottom)
    (Hty_wf & Hty_closed & Hty_kinded & Hty_equations) Halias Hunknown
    Hnot_bottom.
  assert (Hmem := simple_of_kind_mem tenv n (kind ty) Hinvariant Hkind).
  split.
  { apply replace_equation_invariant; try assumption.
    - intros ty0 binding_time_and_mode Hlookup.
      exact (replace_equation_same_kind _ _ Hlookup).
    - left. exact Hunknown. }
  split; [ exact (replace_equation_types_well_formed tenv n ty Hty_wf Hwf) | ].
  split.
  { exact (replace_equation_kinded tenv n ty Hkind Hty_kinded Hkinded). }
  split.
  { exact (replace_equation_equations_kinded tenv n ty Hkind Hty_equations
             Hequations). }
  intros His_bottom m ty0 binding_time_and_mode Hlookup.
  destruct (name_eqb m n) eqn:Hm.
  - apply name_eqb_eq in Hm. subst m.
    unfold tenv' in Hlookup. rewrite replace_equation_lookup in Hlookup.
    assert (Hsame : ty0 = ty).
    { destruct n as [ x | sym ].
      - destruct (Keyed.lookup name_eqb (Name_var x)
                    (Typing_env.names_to_types tenv)) as [ entry | ];
          [ | discriminate Hlookup ].
        injection Hlookup as <- _. reflexivity.
      - injection Hlookup as <- _. reflexivity. }
    rewrite Hsame. exact Hnot_bottom.
  - unfold tenv' in Hlookup.
    rewrite replace_equation_lookup_other in Hlookup.
    + apply (Hbottom) with (n := m)
        (binding_time_and_mode := binding_time_and_mode);
        [ | exact Hlookup ].
      assert (Hframe := replace_equation_frame tenv n ty). cbv zeta in Hframe.
      destruct Hframe as (_ & _ & _ & _ & _ & _ & _ & Hsame & _).
      unfold tenv' in His_bottom. rewrite Hsame in His_bottom.
      exact His_bottom.
    + intros ->. rewrite (proj2 (name_eqb_eq _ _) eq_refl) in Hm.
      discriminate Hm.
Qed.

(* No equation is added but on the name. *)
Lemma replace_equation_valued (valuation : env) (v : value) :
  value_of_name valuation n = Some v ->
  keeps_equations_valued tenv tenv' valuation valuation.
Proof.
  intros Hv cut_after Hvalued one_level m ty0 Hin Hscope Hequation.
  assert (Hframe := replace_equation_frame tenv n ty). cbv zeta in Hframe.
  destruct Hframe as (_ & Hprev & Hcurrent_scope & _).
  fold tenv' in Hprev, Hcurrent_scope.
  destruct Hin as [ <- | Hin ].
  - destruct (replace_equation_equations tenv n m ty ty0 Hequation)
      as [ [-> _] | Hold ]; [ exists v; exact Hv | ].
    apply (Hvalued (Typing_env.current_level tenv) m ty0);
      [ left; reflexivity | | exact Hold ].
    exact (eq_ind _ (fun scope => (cut_after < scope)%Z) Hscope _
             Hcurrent_scope).
  - rewrite Hprev in Hin.
    exact (Hvalued one_level m ty0 (or_intror Hin) Hscope Hequation).
Qed.

End Replace_equation.

(* ======================================================================== *)
(* Two canonical elements are made aliases                                  *)
(* ======================================================================== *)

Lemma invariant_aliases_consistent (tenv : Typing_env.t) :
  typing_env_invariant tenv ->
  aliases_consistent (Typing_env.aliases tenv).
Proof.
  intros [Hwf _]. apply typing_env_well_formed_iff in Hwf.
  destruct Hwf as [_ [_ [Hconsistent _]]]. exact Hconsistent.
Qed.

Section Add_alias.

Variable tenv : Typing_env.t.
Variable canonical_element1 canonical_element2 : simple.
Variable result : Typing_env.add_alias_result.

Hypothesis Hadd :
  Typing_env.add_alias tenv canonical_element1 canonical_element2
  = Some (Ok result).

Let tenv' := Typing_env.add_alias_result_t result.

Lemma add_alias_tenv_extends : tenv_extends tenv tenv'.
Proof.
  assert (Hframe := add_alias_frame tenv _ _ result Hadd).
  cbv zeta in Hframe.
  destruct Hframe as (_ & Hsymbols & Hprev & Hcurrent & Hnames & _ & Hmin & _).
  fold tenv' in Hsymbols, Hprev, Hcurrent, Hnames, Hmin.
  split; [ unfold Typing_env.current_scope; rewrite Hcurrent; reflexivity | ].
  split; [ exact Hprev | ]. split; [ exact Hmin | ].
  split; [ exact Hsymbols | ]. split.
  - intros s kind Hs.
    exact (proj2 (add_alias_simple_of_kind tenv _ _ result s kind Hadd) Hs).
  - intros n ty binding_time_and_mode Hlookup. exists ty.
    split; [ | reflexivity ]. rewrite Hnames. exact Hlookup.
Qed.

Lemma add_alias_valued (valuation : env) :
  keeps_equations_valued tenv tenv' valuation valuation.
Proof.
  assert (Hframe := add_alias_frame tenv _ _ result Hadd).
  cbv zeta in Hframe.
  destruct Hframe as (_ & _ & Hprev & Hcurrent & _).
  fold tenv' in Hprev, Hcurrent.
  intros cut_after Hvalued one_level n ty Hin.
  rewrite Hprev, Hcurrent in Hin. exact (Hvalued one_level n ty Hin).
Qed.

Lemma add_alias_fits (kind : flambda_kind) :
  typing_env_fits tenv ->
  is_canonical (Typing_env.aliases tenv) canonical_element1 ->
  is_canonical (Typing_env.aliases tenv) canonical_element2 ->
  Typing_env.mem_simple None tenv canonical_element1 = true ->
  Typing_env.mem_simple None tenv canonical_element2 = true ->
  simple_of_kind tenv canonical_element1 kind ->
  simple_of_kind tenv canonical_element2 kind ->
  typing_env_fits tenv'.
Proof.
  intros (Hinvariant & Hwf & Hkinded & Hequations & Hbottom) Hc1 Hc2 Hm1 Hm2
    Hk1 Hk2.
  assert (Hframe := add_alias_frame tenv _ _ result Hadd).
  cbv zeta in Hframe.
  destruct Hframe as (_ & _ & _ & _ & Hnames & _ & _ & His_bottom).
  fold tenv' in Hnames, His_bottom.
  split.
  { exact (add_alias_invariant tenv _ _ result Hinvariant Hc1 Hc2 Hm1 Hm2
             Hadd). }
  split; [ exact (add_alias_types_well_formed tenv _ _ result Hadd Hwf) | ].
  split.
  { exact (add_alias_kinded tenv _ _ result kind
             (invariant_aliases_consistent tenv Hinvariant) Hc1 Hc2 Hk1 Hk2
             Hadd Hkinded). }
  split.
  { exact (add_alias_equations_kinded tenv _ _ result Hadd Hequations). }
  intros Hnot_bottom n ty binding_time_and_mode Hlookup.
  rewrite Hnames in Hlookup. rewrite His_bottom in Hnot_bottom.
  exact (Hbottom Hnot_bottom n ty binding_time_and_mode Hlookup).
Qed.

(* The two that come out of it are the two that went in, and the one
   demoted is no longer canonical. *)
Lemma add_alias_demoted :
  typing_env_invariant tenv ->
  is_canonical (Typing_env.aliases tenv) canonical_element1 ->
  is_canonical (Typing_env.aliases tenv) canonical_element2 ->
  canonical_element1 <> canonical_element2 ->
  ((Typing_env.canonical_element result = canonical_element1
    ∧ simple_of_name (Typing_env.demoted_name result) = canonical_element2)
   ∨ (Typing_env.canonical_element result = canonical_element2
      ∧ simple_of_name (Typing_env.demoted_name result)
        = canonical_element1))
  ∧ ~ is_canonical (Typing_env.aliases tenv')
        (simple_of_name (Typing_env.demoted_name result)).
Proof.
  intros Hinvariant Hc1 Hc2 Hdifferent.
  assert (Hconsistent := invariant_aliases_consistent tenv Hinvariant).
  assert (Hframe := add_alias_frame tenv _ _ result Hadd).
  cbv zeta in Hframe.
  destruct Hframe
    as ((added & Hadded & Haliases & Hcanonical & Hdemoted) & _).
  fold tenv' in Haliases.
  assert (Hresult := add_result _ _ _ _ _ _ added Hadded).
  assert (Hwhich :
    (Typing_env.canonical_element result = canonical_element1
     ∧ simple_of_name (Typing_env.demoted_name result) = canonical_element2)
    ∨ (Typing_env.canonical_element result = canonical_element2
       ∧ simple_of_name (Typing_env.demoted_name result)
         = canonical_element1)).
  { rewrite Hcanonical, Hdemoted.
    destruct (Aliases.defined_earlier _ _ canonical_element1
                canonical_element2); [ left | right ]; exact Hresult. }
  split; [ exact Hwhich | ].
  intros Hstill.
  assert (Hof :=
    add_canonical_element _ _ _ _ _ _ added
      (simple_of_name (Typing_env.demoted_name result)) Hconsistent Hc1 Hc2
      Hadded).
  rewrite <- Haliases in Hof.
  rewrite (canonical_element_of_canonical _ _ Hstill) in Hof.
  assert (Hwas :
    canonical_element_of (Typing_env.aliases tenv)
      (simple_of_name (Typing_env.demoted_name result))
    = simple_of_name (Typing_env.demoted_name result)).
  { apply canonical_element_of_canonical.
    destruct Hwhich as [ [_ ->] | [_ ->] ]; assumption. }
  rewrite Hwas, <- Hdemoted in Hof.
  rewrite (proj2 (simple_equal_eq _ _) eq_refl) in Hof.
  rewrite <- Hcanonical in Hof.
  destruct Hwhich as [ [H1 H2] | [H1 H2] ]; congruence.
Qed.

End Add_alias.

(* ======================================================================== *)
(* Facts about types                                                        *)
(* ======================================================================== *)

Lemma kind_alias_type_of (k : flambda_kind) (s : simple) :
  kind (alias_type_of k s) = k.
Proof. destruct k as [ | [] | | ]; reflexivity. Qed.

Lemma get_alias_opt_alias_type_of (k : flambda_kind) (s : simple) :
  get_alias_opt (alias_type_of k s) = Some s.
Proof. destruct k as [ | [] | | ]; reflexivity. Qed.

Lemma aliases_with_kinds_alias_type_of (k : flambda_kind) (s : simple) :
  aliases_with_kinds (alias_type_of k s) = [(s, k)].
Proof. destruct k as [ | [] | | ]; reflexivity. Qed.

Lemma free_variables_alias_type_of (k : flambda_kind) (s : simple)
    (x : variable) :
  x ∈ free_variables (alias_type_of k s) -> s = Var x.
Proof.
  destruct k as [ | [] | | ]; destruct s as [ y | sym | c ]; cbn;
    intros Hin; try contradiction; destruct Hin as [ -> | [] ]; reflexivity.
Qed.

Lemma is_obviously_unknown_alias_type_of (k : flambda_kind) (s : simple) :
  is_obviously_unknown (alias_type_of k s) = false.
Proof. destruct k as [ | [] | | ]; reflexivity. Qed.

Lemma is_obviously_bottom_alias_type_of (k : flambda_kind) (s : simple) :
  is_obviously_bottom (alias_type_of k s) = false.
Proof. destruct k as [ | [] | | ]; reflexivity. Qed.

Lemma kind_type_for_const (c : const) :
  kind (type_for_const c) = kind_of_const c.
Proof.
  destruct c as [i | i | f | f | i | i | i | i | i | i | i | i | i | | k];
    try reflexivity.
  apply kind_unknown.
Qed.

(* The constant that a type that is not an alias has for only value is
   of the kind of the type. *)
Lemma must_be_singleton_kind (ty : flambda_type) (c : const) :
  get_alias_opt ty = None ->
  must_be_singleton ty = Some (Some c) ->
  kind_of_const c = kind ty.
Proof.
  destruct ty as [ descr | descr | k descr | descr | descr ];
    destruct descr as [ | [ head | s ] | ]; cbn;
    try discriminate.
  - intros _.
    destruct head as [ [ | [ ? immediates ? blocks ? ? | | | | ] | ] [ | ] ];
      try discriminate.
    destruct blocks as [ blocks | ]; [ | discriminate ].
    destruct (negb (row_like_for_blocks_is_bottom blocks));
      [ discriminate | ].
    destruct immediates as [ immediates | ]; [ | discriminate ].
    destruct (must_be_singleton immediates) as [ [ c0 | ] | ];
      try discriminate.
    destruct c0 as [i | i | f | f | i | i | i | i | i | i | i | i | i | | k];
      try discriminate; try (intros [= <-]; reflexivity).
    destruct k as [ | [] | | ]; try discriminate.
    intros [= <-]. reflexivity.
  - intros _. destruct (naked_immediates head) as [ numbers | ];
      [ | discriminate ].
    destruct (Keyed.get_singleton Z.eqb numbers); [ | discriminate ].
    intros [= <-]. reflexivity.
  - intros _. destruct (Keyed.get_singleton Z.eqb head); [ | discriminate ].
    intros [= <-]. destruct k; reflexivity.
Qed.

(* A type made of an expanded type is not an alias. *)
Lemma get_alias_opt_to_type (expanded : Expanded_type.t) :
  get_alias_opt (Expanded_type.to_type expanded) = None.
Proof.
  destruct expanded as [k [ | [ ] | ]]; cbn; try reflexivity;
    destruct k as [ | [] | | ]; reflexivity.
Qed.

(* ======================================================================== *)
(* The functions that add an equation: what they keep                       *)
(* ======================================================================== *)

(* Giving a name a type, and making two canonical elements aliases,
   define no variable. *)
Lemma replace_equation_no_new (tenv : Typing_env.t) (n : name)
    (ty : flambda_type) :
  no_new_variables tenv (Typing_env.replace_equation tenv n ty).
Proof.
  intros x. rewrite !mem_var_is_bound. unfold Keyed.is_bound.
  destruct (name_eqb (Name_var x) n) eqn:Hx.
  - apply name_eqb_eq in Hx. subst n. rewrite replace_equation_lookup.
    destruct (Keyed.lookup name_eqb (Name_var x)
                (Typing_env.names_to_types tenv));
      [ reflexivity | discriminate ].
  - rewrite replace_equation_lookup_other; [ intros H; exact H | ].
    intros Heq. rewrite Heq, (proj2 (name_eqb_eq _ _) eq_refl) in Hx.
    discriminate Hx.
Qed.

Lemma add_alias_no_new (tenv : Typing_env.t)
    (canonical_element1 canonical_element2 : simple)
    (result : Typing_env.add_alias_result) :
  Typing_env.add_alias tenv canonical_element1 canonical_element2
  = Some (Ok result) ->
  no_new_variables tenv (Typing_env.add_alias_result_t result).
Proof.
  intros Hadd x.
  assert (Hframe := add_alias_frame tenv _ _ result Hadd).
  cbv zeta in Hframe.
  destruct Hframe as (_ & _ & _ & _ & Hnames & _).
  rewrite !mem_var_is_bound, Hnames. intros H. exact H.
Qed.

Lemma replace_concrete_equation_no_new (tenv tenv' : Typing_env.t)
    (n : name) (ty : flambda_type) :
  Meet_env.replace_concrete_equation tenv n ty = Some tenv' ->
  no_new_variables tenv tenv'.
Proof.
  intros Hreplace x Hknown.
  unfold Meet_env.replace_concrete_equation in Hreplace.
  destruct (must_be_singleton ty) as [ [ const | ] | ];
    [ | | discriminate Hreplace ]; cbn in Hreplace.
  - destruct (Aliases.canonical (Typing_env.aliases tenv) (simple_of_name n));
      [ | discriminate Hreplace ].
    destruct (Typing_env.add_alias tenv (simple_of_name n) (Const const))
      as [ [ | result | ] | ] eqn:Hadd; try discriminate Hreplace.
    cbn in Hreplace.
    destruct (name_eqb (Typing_env.demoted_name result) n
              && simple_equal (Typing_env.canonical_element result)
                   (Const const)); [ | discriminate Hreplace ].
    injection Hreplace as <-.
    apply (add_alias_no_new tenv _ _ result Hadd).
    exact (replace_equation_no_new _ _ _ x Hknown).
  - destruct (get_alias_opt ty); [ discriminate Hreplace | ].
    injection Hreplace as <-.
    exact (replace_equation_no_new _ _ _ x Hknown).
Qed.

Section Keeps.

Lemma mem_simple_of_name (tenv : Typing_env.t) (n : name) :
  Typing_env.mem_simple None tenv (simple_of_name n)
  = Typing_env.mem None tenv n.
Proof. destruct n; reflexivity. Qed.

Lemma type_fits_alias (tenv : Typing_env.t) (k : flambda_kind) (s : simple) :
  Typing_env.mem_simple None tenv s = true ->
  simple_of_kind tenv s k ->
  type_fits tenv (alias_type_of k s).
Proof.
  intros Hmem Hkind. split; [ apply alias_type_of_well_formed | ]. split.
  { intros x Hin. apply free_variables_alias_type_of in Hin. subst s.
    exact Hmem. }
  split.
  - intros s' k' Hin. rewrite aliases_with_kinds_alias_type_of in Hin.
    destruct Hin as [ [= <- <-] | [] ]. exact Hkind.
  - intros n k' Hin.
    rewrite (proj1 (proj2 (proj2 equation_names_none))) in Hin.
    destruct Hin.
Qed.

Lemma type_fits_of_non_alias (tenv : Typing_env.t) (ty : flambda_type)
    (expanded : Expanded_type.t) :
  Expanded_type.of_non_alias_type ty = Some expanded ->
  type_fits tenv ty ->
  type_fits tenv (Expanded_type.to_type expanded).
Proof.
  intros Hexpanded (Hwf & Hclosed & Hkinded & Hequations).
  split;
    [ exact (expanded_of_non_alias_well_formed ty expanded Hexpanded Hwf) | ].
  split.
  { intros x Hin.
    rewrite (of_non_alias_type_free_variables ty expanded Hexpanded) in Hin.
    exact (Hclosed x Hin). }
  split.
  - intros s k Hin.
    rewrite (of_non_alias_type_aliases ty expanded Hexpanded) in Hin.
    exact (Hkinded s k Hin).
  - intros n k Hin.
    rewrite (of_non_alias_type_equation_names ty expanded Hexpanded) in Hin.
    exact (Hequations n k Hin).
Qed.

(* The type of a constant, expanded, fits any environment. *)
Lemma type_fits_create_const (tenv : Typing_env.t) (c : const) :
  type_fits tenv (Expanded_type.to_type (Expanded_type.create_const c)).
Proof.
  split; [ apply expanded_create_const_well_formed | ]. split.
  { intros x Hin. rewrite create_const_free_variables in Hin. destruct Hin. }
  split; [ apply create_const_kinded | ].
  intros n k Hin. rewrite create_const_equation_names in Hin. destruct Hin.
Qed.

(* The type that an environment that fits answers for a name fits. *)
Lemma find_type_fits (tenv : Typing_env.t) (n : name)
    (kind : option flambda_kind) (ty : flambda_type) :
  typing_env_fits tenv ->
  Typing_env.find tenv n kind = Some ty ->
  type_fits tenv ty.
Proof.
  intros Hfits Hfind.
  destruct (find_cases tenv n kind ty Hfind)
    as [ [binding_time_and_mode Hlookup] | [ [k ->] | -> ] ].
  - exact (fits_types tenv n ty binding_time_and_mode Hfits Hlookup).
  - split; [ apply bottom_well_formed | ]. split.
    { intros x Hin. rewrite free_variables_bottom in Hin. destruct Hin. }
    split.
    + intros s k' Hin. rewrite bottom_aliases in Hin. destruct Hin.
    + intros m k' Hin.
      rewrite (proj1 (proj2 equation_names_none)) in Hin. destruct Hin.
  - split; [ exact any_non_null_value_well_formed | ]. split.
    { intros x Hin. destruct Hin. }
    split.
    + intros s k' Hin. destruct Hin.
    + intros m k' Hin. destruct Hin.
Qed.

(* [replace_concrete_equation]. *)
Lemma replace_concrete_equation_keeps (tenv tenv' : Typing_env.t) (n : name)
    (ty : flambda_type) :
  typing_env_fits tenv ->
  type_fits tenv ty ->
  simple_of_kind tenv (simple_of_name n) (kind ty) ->
  get_alias_opt ty = None ->
  is_obviously_unknown ty = false ->
  is_obviously_bottom ty = false ->
  Meet_env.replace_concrete_equation tenv n ty = Some tenv' ->
  tenv_extends tenv tenv' ∧ typing_env_fits tenv'.
Proof.
  intros Hfits Hty Hkind Hconcrete Hunknown Hbottom Hreplace.
  assert (Hinvariant := proj1 Hfits).
  unfold Meet_env.replace_concrete_equation in Hreplace.
  destruct (must_be_singleton ty) as [ [ const | ] | ] eqn:Hsingleton;
    [ | | discriminate Hreplace ]; cbn in Hreplace.
  - (* The name becomes an alias of the constant. *)
    destruct (Aliases.canonical (Typing_env.aliases tenv) (simple_of_name n))
      eqn:Hcanonical; [ | discriminate Hreplace ].
    destruct (Typing_env.add_alias tenv (simple_of_name n) (Const const))
      as [ [ | result | ] | ] eqn:Hadd; try discriminate Hreplace.
    cbn in Hreplace.
    destruct (name_eqb (Typing_env.demoted_name result) n
              && simple_equal (Typing_env.canonical_element result)
                   (Const const)) eqn:Hcheck; [ | discriminate Hreplace ].
    injection Hreplace as <-.
    apply andb_true_iff in Hcheck. destruct Hcheck as [Hdemoted Hconst].
    apply name_eqb_eq in Hdemoted. apply simple_equal_eq in Hconst.
    rewrite Hdemoted, Hconst.
    assert (Hkind_const : kind_of_const const = kind ty).
    { exact (must_be_singleton_kind ty const Hconcrete Hsingleton). }
    assert (Hconst_kind : simple_of_kind tenv (Const const) (kind ty)).
    { cbn. rewrite kind_type_for_const. exact Hkind_const. }
    assert (Hmem : Typing_env.mem_simple None tenv (simple_of_name n) = true).
    { rewrite mem_simple_of_name.
      exact (simple_of_kind_mem tenv n (kind ty) Hinvariant Hkind). }
    assert (Hextends1 := add_alias_tenv_extends tenv _ _ result Hadd).
    assert (Hfits1 :=
      add_alias_fits tenv _ _ result Hadd (kind ty) Hfits
        Hcanonical eq_refl Hmem eq_refl Hkind Hconst_kind).
    set (tenv1 := Typing_env.add_alias_result_t result) in *.
    set (alias := alias_type_of (kind_of_const const) (Const const)).
    assert (Hkind1 : simple_of_kind tenv1 (simple_of_name n) (kind alias)).
    { unfold alias. rewrite kind_alias_type_of, Hkind_const.
      exact (proj1 (proj2 (proj2 (proj2 (proj2 Hextends1)))) _ _ Hkind). }
    split.
    + exact (tenv_extends_trans _ _ _ Hextends1
               (replace_equation_tenv_extends tenv1 n alias (proj1 Hfits1)
                  Hkind1)).
    + apply (replace_equation_fits tenv1 n alias (proj1 Hfits1) Hkind1
               Hfits1).
      * apply type_fits_alias; [ reflexivity | ].
        cbn. apply kind_type_for_const.
      * intros s _.
        destruct (add_alias_demoted tenv _ _ result Hadd Hinvariant
                    Hcanonical eq_refl) as [_ Hnot].
        { destruct n; discriminate. }
        rewrite Hdemoted in Hnot. exact Hnot.
      * apply is_obviously_unknown_alias_type_of.
      * apply is_obviously_bottom_alias_type_of.
  - (* The name gets the type. *)
    rewrite Hconcrete in Hreplace. injection Hreplace as <-.
    split.
    + exact (replace_equation_tenv_extends tenv n ty Hinvariant Hkind).
    + apply (replace_equation_fits tenv n ty Hinvariant Hkind Hfits
               Hty); try assumption.
      intros s Hs. rewrite Hconcrete in Hs. discriminate Hs.
Qed.

(* The meet that the functions are given. *)
Variable mh : meet_expanded_head.
Hypothesis Hmh : meet_expanded_head_keeps mh.

Lemma simple_of_kind_of_find (tenv : Typing_env.t) (n : name)
    (kind : flambda_kind) (ty : flambda_type) :
  Typing_env.find tenv n (Some kind) = Some ty ->
  simple_of_kind tenv (simple_of_name n) kind.
Proof. intros Hfind. destruct n; exists ty; exact Hfind. Qed.

(* [add_concrete_equation_on_canonical_name]. *)
Lemma add_concrete_equation_on_canonical_name_keeps (menv menv' : Meet_env.t)
    (n : name) (ty : Expanded_type.t) :
  typing_env_fits (Meet_env.typing_env menv) ->
  type_fits (Meet_env.typing_env menv)
    (Expanded_type.to_type ty) ->
  Expanded_type.kind ty = kind (Expanded_type.to_type ty) ->
  Meet_env.add_concrete_equation_on_canonical_name true menv n ty mh
  = Some (Or_bottom.Ok menv') ->
  meet_env_steps menv menv'
  ∧ typing_env_fits (Meet_env.typing_env menv').
Proof.
  intros Hfits Hty Hcoherent Hadd.
  unfold Meet_env.add_concrete_equation_on_canonical_name in Hadd.
  destruct (Typing_env.find (Meet_env.typing_env menv) n
              (Some (Expanded_type.kind ty)))
    as [ existing | ] eqn:Hfind; [ | discriminate Hadd ].
  cbn in Hadd.
  destruct (Expanded_type.of_non_alias_type existing)
    as [ existing_ty | ] eqn:Hexisting; [ | discriminate Hadd ].
  cbn in Hadd.
  destruct (mh menv ty existing_ty) as [ result | ] eqn:Hmeet;
    [ | discriminate Hadd ].
  cbn in Hadd.
  destruct result as [ r0 | r menv1 ]; [ discriminate Hadd | ].
  assert (Hinputs :
    meet_inputs_well_formed (Meet_env.typing_env menv)
      (Expanded_type.to_type ty) (Expanded_type.to_type existing_ty)).
  { split; [ exact Hfits | ]. split; [ exact Hty | ]. split.
    - exact (type_fits_of_non_alias _ _ _ Hexisting
               (find_type_fits _ _ _ _ Hfits Hfind)).
    - rewrite (proj2 (of_non_alias_type_kind _ _ Hexisting)).
      rewrite (find_kind _ _ _ _ Hfind). symmetry. exact Hcoherent. }
  destruct (Hmh menv menv1 ty existing_ty r Hinputs Hmeet)
    as (Hextends1 & Hfits1 & Hanswer & Hkind_answer & Hnot_bottom
        & Hunknown & Hnames1).
  assert (Hreplace :
    ∀ (answer : Expanded_type.t) (tenv2 : Typing_env.t),
      extract_value r ty existing_ty = answer ->
      (r = Left_input ∨ r = New_result answer) ->
      Meet_env.replace_concrete_equation (Meet_env.typing_env menv1) n
        (Expanded_type.to_type answer)
      = Some tenv2 ->
      meet_env_steps menv (Meet_env.with_typing_env menv1 tenv2)
      ∧ typing_env_fits tenv2).
  { intros answer tenv2 Hextract Hr Hreplace. rewrite Hextract in *.
    assert (Hkind_n :
      simple_of_kind (Meet_env.typing_env menv1) (simple_of_name n)
        (kind (Expanded_type.to_type answer))).
    { rewrite Hkind_answer, <- Hcoherent.
      apply (proj1 (proj2 (proj2 (proj2 (proj2
               (meet_env_extends_tenv _ _ Hextends1)))))).
      exact (simple_of_kind_of_find _ _ _ _ Hfind). }
    destruct (replace_concrete_equation_keeps _ tenv2 n _ Hfits1 Hanswer
                Hkind_n (get_alias_opt_to_type answer))
      as [Hextends2 Hfits2]; try assumption.
    - destruct (is_obviously_unknown (Expanded_type.to_type answer));
        [ | reflexivity ].
      destruct (Hunknown eq_refl) as [ -> | -> ];
        destruct Hr as [ Hr | Hr ]; discriminate Hr.
    - split; [ | exact Hfits2 ].
      exact (meet_env_steps_trans _ _ _ (conj Hextends1 Hnames1)
               (meet_env_steps_with menv1 tenv2 Hextends2
                  (replace_concrete_equation_no_new _ _ _ _ Hreplace))). }
  destruct r as [ | | | answer ].
  - destruct (Meet_env.replace_concrete_equation (Meet_env.typing_env menv1)
                n (Expanded_type.to_type ty))
      as [ tenv2 | ] eqn:Hreplaced; [ | discriminate Hadd ].
    injection Hadd as <-.
    exact (Hreplace ty tenv2 eq_refl (or_introl eq_refl) Hreplaced).
  - injection Hadd as <-. exact (conj (conj Hextends1 Hnames1) Hfits1).
  - injection Hadd as <-. exact (conj (conj Hextends1 Hnames1) Hfits1).
  - destruct (Meet_env.replace_concrete_equation (Meet_env.typing_env menv1)
                n (Expanded_type.to_type answer))
      as [ tenv2 | ] eqn:Hreplaced; [ | discriminate Hadd ].
    injection Hadd as <-.
    exact (Hreplace answer tenv2 eq_refl (or_intror eq_refl) Hreplaced).
Qed.

(* [adding_equation_for_name]. *)
Lemma adding_equation_for_name_keeps (menv menv' : Meet_env.t) (n : name)
    (f : Meet_env.t -> option (Or_bottom.t Meet_env.t)) :
  (∀ menv1 menv2 : Meet_env.t,
     Meet_env.typing_env menv1 = Meet_env.typing_env menv ->
     f menv1 = Some (Or_bottom.Ok menv2) ->
     meet_env_steps menv1 menv2
     ∧ typing_env_fits (Meet_env.typing_env menv2)) ->
  Meet_env.adding_equation_for_name menv n f = Some (Or_bottom.Ok menv') ->
  meet_env_steps menv menv'
  ∧ typing_env_fits (Meet_env.typing_env menv').
Proof.
  intros Hf Hadding. unfold Meet_env.adding_equation_for_name in Hadding.
  match type of Hadding with
  | context [ f ?env ] =>
    destruct (f env) as [ [ menv2 | ] | ] eqn:Hrun;
      try discriminate Hadding;
      destruct (Hf env menv2 eq_refl Hrun) as [[Hextends Hnames] Hfits]
  end.
  injection Hadding as <-. split; [ | exact Hfits ].
  destruct Hextends as (Hs & Hp & Hm & Hd & Hkind & _ & Hv & Hk).
  split; [ | exact Hnames ].
  exact (conj Hs (conj Hp (conj Hm (conj Hd (conj Hkind (conj eq_refl
           (conj Hv Hk))))))).
Qed.

(* [add_or_delay_concrete_equation_on_canonical_name]. *)
Lemma add_or_delay_concrete_equation_on_canonical_name_keeps
    (menv menv' : Meet_env.t) (n : name) (ty : Expanded_type.t) :
  typing_env_fits (Meet_env.typing_env menv) ->
  type_fits (Meet_env.typing_env menv)
    (Expanded_type.to_type ty) ->
  Expanded_type.kind ty = kind (Expanded_type.to_type ty) ->
  Meet_env.add_or_delay_concrete_equation_on_canonical_name true menv n ty mh
  = Some (Or_bottom.Ok menv') ->
  meet_env_steps menv menv'
  ∧ typing_env_fits (Meet_env.typing_env menv').
Proof.
  intros Hfits Hty Hcoherent Hadd.
  unfold Meet_env.add_or_delay_concrete_equation_on_canonical_name in Hadd.
  destruct (Keyed.among name_eqb n (Meet_env.adding_equations_for_names menv)).
  - injection Hadd as <-.
    split; [ apply meet_env_steps_refl | exact Hfits ].
  - apply (adding_equation_for_name_keeps menv menv' n _) in Hadd;
      [ exact Hadd | ].
    intros menv1 menv2 Hsame Hrun.
    apply (add_concrete_equation_on_canonical_name_keeps menv1 menv2 n ty);
      try assumption; rewrite Hsame; assumption.
Qed.

(* [add_concrete_equation_on_const]. *)
Lemma add_concrete_equation_on_const_keeps (menv menv' : Meet_env.t)
    (const : const) (ty : Expanded_type.t) :
  typing_env_fits (Meet_env.typing_env menv) ->
  type_fits (Meet_env.typing_env menv)
    (Expanded_type.to_type ty) ->
  kind (type_for_const const) = kind (Expanded_type.to_type ty) ->
  Meet_env.add_concrete_equation_on_const true menv const ty mh
  = Some (Or_bottom.Ok menv') ->
  meet_env_steps menv menv'
  ∧ typing_env_fits (Meet_env.typing_env menv').
Proof.
  intros Hfits Hty Hkind Hadd.
  unfold Meet_env.add_concrete_equation_on_const in Hadd.
  destruct (mh menv ty (Expanded_type.create_const const))
    as [ [ r0 | r menv1 ] | ] eqn:Hmeet; try discriminate Hadd.
  injection Hadd as <-.
  assert (Hinputs :
    meet_inputs_well_formed (Meet_env.typing_env menv)
      (Expanded_type.to_type ty)
      (Expanded_type.to_type (Expanded_type.create_const const))).
  { split; [ exact Hfits | ]. split; [ exact Hty | ]. split.
    - apply type_fits_create_const.
    - rewrite (proj2 (create_const_kind const)). symmetry. exact Hkind. }
  destruct (Hmh menv menv1 ty _ r Hinputs Hmeet)
    as (Hextends & Hfits1 & _ & _ & _ & _ & Hnames).
  exact (conj (conj Hextends Hnames) Hfits1).
Qed.

(* [add_or_delay_concrete_equation_on_canonical]. *)
Lemma add_or_delay_concrete_equation_on_canonical_keeps
    (menv menv' : Meet_env.t) (canonical : simple) (ty : Expanded_type.t) :
  typing_env_fits (Meet_env.typing_env menv) ->
  type_fits (Meet_env.typing_env menv)
    (Expanded_type.to_type ty) ->
  Expanded_type.kind ty = kind (Expanded_type.to_type ty) ->
  simple_of_kind (Meet_env.typing_env menv) canonical
    (kind (Expanded_type.to_type ty)) ->
  Meet_env.add_or_delay_concrete_equation_on_canonical true menv canonical ty
    mh
  = Some (Or_bottom.Ok menv') ->
  meet_env_steps menv menv'
  ∧ typing_env_fits (Meet_env.typing_env menv').
Proof.
  intros Hfits Hty Hcoherent Hkind Hadd.
  destruct canonical as [ x | sym | const ]; cbn in Hadd.
  - exact (add_or_delay_concrete_equation_on_canonical_name_keeps _ _ _ _
             Hfits Hty Hcoherent Hadd).
  - exact (add_or_delay_concrete_equation_on_canonical_name_keeps _ _ _ _
             Hfits Hty Hcoherent Hadd).
  - exact (add_concrete_equation_on_const_keeps _ _ _ _ Hfits Hty Hkind Hadd).
Qed.

(* ------------------------------------------------------------------------ *)
(* Canonical elements                                                       *)
(* ------------------------------------------------------------------------ *)

Lemma invariant_aliases_known (tenv : Typing_env.t) :
  typing_env_invariant tenv -> aliases_known tenv.
Proof.
  intros [Hwf _]. apply typing_env_well_formed_iff in Hwf.
  destruct Hwf as (_ & _ & _ & _ & Hknown & _). exact Hknown.
Qed.

(* A simple that is of a kind is bound. *)
Lemma simple_of_kind_mem_simple (tenv : Typing_env.t) (s : simple)
    (kind : flambda_kind) :
  typing_env_invariant tenv ->
  simple_of_kind tenv s kind ->
  Typing_env.mem_simple None tenv s = true.
Proof.
  intros Hinvariant Hkind. destruct s as [ x | sym | c ]; [ | | reflexivity ].
  - exact (simple_of_kind_mem tenv (Name_var x) kind Hinvariant Hkind).
  - exact (simple_of_kind_mem tenv (Name_symbol sym) kind Hinvariant Hkind).
Qed.

(* A simple that is bound is of some kind. *)
Lemma mem_simple_of_some_kind (tenv : Typing_env.t) (s : simple) :
  Typing_env.mem_simple None tenv s = true ->
  ∃ kind : flambda_kind, simple_of_kind tenv s kind.
Proof.
  intros Hmem.
  assert (Hname :
    ∀ n : name,
      Typing_env.mem None tenv n = true ->
      ∃ kind : flambda_kind, simple_of_kind tenv (simple_of_name n) kind).
  { intros n Hn.
    destruct (Keyed.lookup name_eqb n (Typing_env.names_to_types tenv))
      as [ [ty binding_time_and_mode] | ] eqn:Hlookup.
    - exists (kind ty). apply simple_of_kind_name. left.
      exists ty, binding_time_and_mode. split; [ exact Hlookup | reflexivity ].
    - exists Value. apply simple_of_kind_name. right. auto. }
  destruct s as [ x | sym | c ].
  - exact (Hname (Name_var x) Hmem).
  - exact (Hname (Name_symbol sym) Hmem).
  - exists (kind (type_for_const c)). reflexivity.
Qed.

(* A simple that an environment knows is of the kind that an
   environment made from it has it at. *)
Lemma scoped_env_of_kind (initial_tenv scoped_env : Typing_env.t)
    (s : simple) (kind : flambda_kind) :
  scoped_env_of initial_tenv scoped_env ->
  Typing_env.mem_simple None initial_tenv s = true ->
  simple_of_kind scoped_env s kind ->
  simple_of_kind initial_tenv s kind.
Proof.
  intros (Hfits & _ & Hnames & _) Hknown Hkind.
  assert (Hsymbols : symbols_of_kind_value scoped_env).
  { exact (proj2 (proj2 (proj1 (proj2 (proj2 Hfits))))). }
  assert (Hname :
    ∀ n : name,
      Typing_env.mem None initial_tenv n = true ->
      simple_of_kind scoped_env (simple_of_name n) kind ->
      simple_of_kind initial_tenv (simple_of_name n) kind).
  { intros n Hmem Hn. apply simple_of_kind_name.
    destruct (Keyed.lookup name_eqb n (Typing_env.names_to_types initial_tenv))
      as [ [ty binding_time_and_mode] | ] eqn:Hlookup.
    - left. exists ty, binding_time_and_mode. split; [ reflexivity | ].
      destruct (Hnames n ty binding_time_and_mode Hlookup)
        as (ty' & Hlookup' & Hsame).
      apply simple_of_kind_name in Hn.
      destruct Hn as [ (ty'' & binding_time_and_mode'' & Hfound & Hkind'')
                     | (Hnone & _) ].
      + rewrite Hlookup' in Hfound. injection Hfound as <- _.
        rewrite <- Hsame. exact Hkind''.
      + rewrite Hlookup' in Hnone. discriminate Hnone.
    - right. split; [ reflexivity | ]. split; [ exact Hmem | ].
      destruct n as [ x | sym ].
      + unfold Typing_env.mem in Hmem. rewrite Hlookup in Hmem.
        discriminate Hmem.
      + exact (symbol_of_kind_value scoped_env sym kind Hsymbols Hn). }
  destruct s as [ x | sym | c ].
  - exact (Hname (Name_var x) Hknown Hkind).
  - exact (Hname (Name_symbol sym) Hknown Hkind).
  - exact Hkind.
Qed.

Lemma tenv_extends_mem_simple (tenv tenv' : Typing_env.t) (s : simple) :
  tenv_extends tenv tenv' ->
  Typing_env.mem_simple None tenv s = true ->
  Typing_env.mem_simple None tenv' s = true.
Proof.
  intros Hextends Hmem. destruct s as [ x | sym | c ]; [ | | reflexivity ].
  - exact (tenv_extends_mem _ _ (Name_var x) Hextends Hmem).
  - exact (tenv_extends_mem _ _ (Name_symbol sym) Hextends Hmem).
Qed.

Lemma get_canonical_simple_ignoring_name_mode_eq (tenv : Typing_env.t)
    (s : simple) :
  Typing_env.get_canonical_simple_ignoring_name_mode tenv s
  = canonical_element_of (Typing_env.aliases tenv) s.
Proof.
  destruct s as [ x | sym | c ]; [ | | reflexivity ].
  - exact (get_canonical_ignoring_name_mode _ (Name_var x)).
  - exact (get_canonical_ignoring_name_mode _ (Name_symbol sym)).
Qed.

(* The canonical element of a simple that is of a kind is canonical and
   of that kind. *)
Lemma canonical_of_kind (tenv : Typing_env.t) (s : simple)
    (kind : flambda_kind) :
  typing_env_invariant tenv ->
  aliases_kinded tenv ->
  simple_of_kind tenv s kind ->
  is_canonical (Typing_env.aliases tenv)
    (Typing_env.get_canonical_simple_ignoring_name_mode tenv s)
  ∧ simple_of_kind tenv
      (Typing_env.get_canonical_simple_ignoring_name_mode tenv s) kind.
Proof.
  intros Hinvariant Hkinded Hkind.
  rewrite get_canonical_simple_ignoring_name_mode_eq.
  assert (Hconsistent := invariant_aliases_consistent tenv Hinvariant).
  split; [ exact (canonical_element_canonical _ s Hconsistent) | ].
  destruct (canonical_cases (Typing_env.aliases tenv) s)
    as [ [_ ->] | (n & canonical_element & -> & Hdemoted & _ & Hof) ];
    [ exact Hkind | ].
  destruct (invariant_aliases_known tenv Hinvariant n canonical_element
              Hdemoted) as [_ Hmem].
  destruct (mem_simple_of_some_kind tenv canonical_element Hmem)
    as [kind' Hkind'].
  rewrite Hof.
  assert (Hsame : kind = kind').
  { apply (Hkinded (simple_of_name n) canonical_element kind kind');
      [ | exact Hkind | exact Hkind' ].
    unfold aliased. rewrite Hof. symmetry.
    apply canonical_element_of_canonical. rewrite <- Hof.
    exact (canonical_element_canonical _ _ Hconsistent). }
  rewrite Hsame. exact Hkind'.
Qed.

(* ------------------------------------------------------------------------ *)
(* A name is demoted                                                        *)
(* ------------------------------------------------------------------------ *)

(* [record_demotion]. *)
Lemma record_demotion_keeps (menv menv' : Meet_env.t) (kind : flambda_kind)
    (demoted : name) (canonical : simple) :
  typing_env_fits (Meet_env.typing_env menv) ->
  ~ is_canonical (Typing_env.aliases (Meet_env.typing_env menv))
      (simple_of_name demoted) ->
  simple_of_kind (Meet_env.typing_env menv) (simple_of_name demoted) kind ->
  simple_of_kind (Meet_env.typing_env menv) canonical kind ->
  Meet_env.record_demotion true menv kind demoted canonical mh
  = Some (Or_bottom.Ok menv') ->
  meet_env_steps menv menv'
  ∧ typing_env_fits (Meet_env.typing_env menv').
Proof.
  intros Hfits Hnot_canonical Hkind_demoted Hkind_canonical Hrecord.
  assert (Hinvariant := proj1 Hfits).
  set (tenv := Meet_env.typing_env menv) in *.
  unfold Meet_env.record_demotion in Hrecord.
  fold tenv in Hrecord.
  destruct (Typing_env.find tenv demoted (Some kind))
    as [ ty_of_demoted | ] eqn:Hfind; [ | discriminate Hrecord ].
  cbv beta iota in Hrecord.
  set (alias := alias_type_of kind canonical) in *.
  set (menv1 :=
    Meet_env.map_typing_env menv
      (fun tenv => Typing_env.replace_equation tenv demoted alias)) in *.
  set (tenv1 := Typing_env.replace_equation tenv demoted alias).
  assert (Hkind_alias :
    simple_of_kind tenv (simple_of_name demoted)
      (FlambdaTypeGrammar.kind alias)).
  { unfold alias. rewrite kind_alias_type_of. exact Hkind_demoted. }
  assert (Hextends1 : tenv_extends tenv tenv1).
  { exact (replace_equation_tenv_extends tenv demoted alias Hinvariant
             Hkind_alias). }
  assert (Hfits1 : typing_env_fits tenv1).
  { apply (replace_equation_fits tenv demoted alias Hinvariant Hkind_alias
             Hfits).
    - apply type_fits_alias; [ | exact Hkind_canonical ].
      exact (simple_of_kind_mem_simple tenv canonical kind Hinvariant
               Hkind_canonical).
    - intros s _. exact Hnot_canonical.
    - apply is_obviously_unknown_alias_type_of.
    - apply is_obviously_bottom_alias_type_of. }
  change (Meet_env.typing_env menv1) with tenv1 in Hrecord.
  destruct (expand_head0 tenv1 ty_of_demoted (Some (simple_of_name demoted)))
    as [ expanded | ] eqn:Hexpand; [ | discriminate Hrecord ].
  cbv beta iota in Hrecord.
  assert (Hkind_found : FlambdaTypeGrammar.kind ty_of_demoted = kind).
  { exact (find_kind _ _ _ _ Hfind). }
  assert (Hexpanded :
    type_fits tenv1 (Expanded_type.to_type expanded)
    ∧ Expanded_type.kind expanded = kind
    ∧ FlambdaTypeGrammar.kind (Expanded_type.to_type expanded) = kind).
  { unfold expand_head0 in Hexpand.
    destruct (get_alias_opt ty_of_demoted) as [ s | ].
    - unfold expand_head_of_alias_type in Hexpand.
      assert (Hfound :
        ∃ found : flambda_type,
          Typing_env.find tenv1 demoted
            (Some (FlambdaTypeGrammar.kind ty_of_demoted))
          = Some found
          ∧ Expanded_type.of_non_alias_type found = Some expanded).
      { destruct demoted as [ x | sym ]; cbn [simple_of_name] in Hexpand;
          destruct (Typing_env.find tenv1 _
                      (Some (FlambdaTypeGrammar.kind ty_of_demoted)))
            as [ found | ] eqn:Hfound; try discriminate Hexpand;
          exists found; (split; [ reflexivity | exact Hexpand ]). }
      destruct Hfound as (found & Hfound & Hof).
      split.
      + exact (type_fits_of_non_alias _ _ _ Hof
                 (find_type_fits _ _ _ _ Hfits1 Hfound)).
      + destruct (of_non_alias_type_kind _ _ Hof) as [-> ->].
        rewrite (find_kind _ _ _ _ Hfound). auto.
    - split.
      + apply (type_fits_of_non_alias _ _ _ Hexpand).
        apply (tenv_extends_type_fits _ _ _ Hextends1).
        exact (find_type_fits _ _ _ _ Hfits Hfind).
      + destruct (of_non_alias_type_kind _ _ Hexpand) as [-> ->]. auto. }
  destruct Hexpanded as (Hty & Hkind1 & Hkind2).
  destruct (add_or_delay_concrete_equation_on_canonical_keeps menv1 menv'
              canonical expanded Hfits1 Hty) as [Hextends2 Hfits2].
  - rewrite Hkind1, Hkind2. reflexivity.
  - rewrite Hkind2.
    exact (proj1 (proj2 (proj2 (proj2 (proj2 Hextends1)))) _ _
             Hkind_canonical).
  - exact Hrecord.
  - split; [ | exact Hfits2 ].
    exact (meet_env_steps_trans _ _ _
             (meet_env_steps_with menv tenv1 Hextends1
                (replace_equation_no_new tenv demoted alias))
             Hextends2).
Qed.

(* [add_alias_between_canonicals]. *)
Lemma add_alias_between_canonicals_keeps (menv menv' : Meet_env.t)
    (kind : flambda_kind) (canonical_element1 canonical_element2 : simple) :
  typing_env_fits (Meet_env.typing_env menv) ->
  is_canonical (Typing_env.aliases (Meet_env.typing_env menv))
    canonical_element1 ->
  is_canonical (Typing_env.aliases (Meet_env.typing_env menv))
    canonical_element2 ->
  simple_of_kind (Meet_env.typing_env menv) canonical_element1 kind ->
  simple_of_kind (Meet_env.typing_env menv) canonical_element2 kind ->
  Meet_env.add_alias_between_canonicals true menv kind canonical_element1
    canonical_element2 mh
  = Some (Or_bottom.Ok menv') ->
  meet_env_steps menv menv'
  ∧ typing_env_fits (Meet_env.typing_env menv').
Proof.
  intros Hfits Hc1 Hc2 Hkind1 Hkind2 Hadd.
  assert (Hinvariant := proj1 Hfits).
  set (tenv := Meet_env.typing_env menv) in *.
  unfold Meet_env.add_alias_between_canonicals in Hadd. fold tenv in Hadd.
  destruct (simple_equal canonical_element1 canonical_element2) eqn:Hsame.
  { injection Hadd as <-.
    split; [ apply meet_env_steps_refl | exact Hfits ]. }
  destruct (Typing_env.add_alias tenv canonical_element1 canonical_element2)
    as [ [ | result | ] | ] eqn:Halias; try discriminate Hadd.
  - injection Hadd as <-.
    split; [ apply meet_env_steps_refl | exact Hfits ].
  - cbv beta iota in Hadd.
    assert (Hdifferent : canonical_element1 <> canonical_element2).
    { intros Heq. apply simple_equal_eq in Heq. rewrite Heq in Hsame.
      discriminate Hsame. }
    assert (Hextends1 := add_alias_tenv_extends tenv _ _ result Halias).
    assert (Hfits1 :=
      add_alias_fits tenv _ _ result Halias kind Hfits Hc1
        Hc2
        (simple_of_kind_mem_simple tenv _ kind Hinvariant Hkind1)
        (simple_of_kind_mem_simple tenv _ kind Hinvariant Hkind2)
        Hkind1 Hkind2).
    destruct (add_alias_demoted tenv _ _ result Halias Hinvariant Hc1 Hc2
                Hdifferent) as [Hwhich Hnot].
    set (tenv1 := Typing_env.add_alias_result_t result) in *.
    assert (Hkept :
      ∀ s : simple, simple_of_kind tenv s kind -> simple_of_kind tenv1 s kind).
    { intros s.
      exact (proj1 (proj2 (proj2 (proj2 (proj2 Hextends1)))) s kind). }
    destruct (record_demotion_keeps (Meet_env.with_typing_env menv tenv1)
                menv' kind (Typing_env.demoted_name result)
                (Typing_env.canonical_element result) Hfits1 Hnot)
      as [Hextends2 Hfits2].
    + apply Hkept. destruct Hwhich as [ [_ ->] | [_ ->] ]; assumption.
    + apply Hkept. destruct Hwhich as [ [-> _] | [-> _] ]; assumption.
    + exact Hadd.
    + split; [ | exact Hfits2 ].
      exact (meet_env_steps_trans _ _ _
               (meet_env_steps_with menv tenv1 Hextends1
                  (add_alias_no_new tenv _ _ result Halias))
               Hextends2).
Qed.

(* ------------------------------------------------------------------------ *)
(* A simple is given a type                                                 *)
(* ------------------------------------------------------------------------ *)

(* [add_equation_on_simple ~raise_on_bottom:true]. *)
Lemma add_equation_on_simple0_keeps (menv menv' : Meet_env.t) (s : simple)
    (ty : flambda_type) :
  typing_env_fits (Meet_env.typing_env menv) ->
  type_fits (Meet_env.typing_env menv) ty ->
  simple_of_kind (Meet_env.typing_env menv) s (kind ty) ->
  Meet_env.add_equation_on_simple0 true menv s ty mh
  = Some (Or_bottom.Ok menv') ->
  meet_env_steps menv menv'
  ∧ typing_env_fits (Meet_env.typing_env menv').
Proof.
  intros Hfits Hty Hkind Hadd.
  assert (Hinvariant := proj1 Hfits).
  assert (Hkinded := proj1 (proj2 (proj1 (proj2 (proj2 Hfits))))).
  set (tenv := Meet_env.typing_env menv) in *.
  unfold Meet_env.add_equation_on_simple0 in Hadd. fold tenv in Hadd.
  destruct (canonical_of_kind tenv s (kind ty) Hinvariant Hkinded Hkind)
    as [Hcanonical Hkind_canonical].
  destruct (get_alias_opt ty) as [ alias | ] eqn:Halias.
  - assert (Hkind_alias : simple_of_kind tenv alias (kind ty)).
    { exact (proj1 (proj2 (proj2 Hty)) alias (kind ty)
               (alias_among_aliases ty alias Halias)). }
    destruct (canonical_of_kind tenv alias (kind ty) Hinvariant Hkinded
                Hkind_alias) as [Hcanonical2 Hkind_canonical2].
    exact (add_alias_between_canonicals_keeps menv menv' (kind ty) _ _ Hfits
             Hcanonical Hcanonical2 Hkind_canonical Hkind_canonical2 Hadd).
  - destruct (Expanded_type.of_non_alias_type ty) as [ expanded | ]
      eqn:Hexpanded; [ | discriminate Hadd ].
    cbv beta iota in Hadd.
    unfold Meet_env.add_concrete_equation_on_simple in Hadd.
    fold tenv in Hadd.
    destruct (of_non_alias_type_kind _ _ Hexpanded) as [Hkind1 Hkind2].
    apply (add_or_delay_concrete_equation_on_canonical_keeps menv menv'
             (Typing_env.get_canonical_simple_ignoring_name_mode tenv s)
             expanded Hfits).
    + exact (type_fits_of_non_alias _ _ _ Hexpanded Hty).
    + rewrite Hkind1, Hkind2. reflexivity.
    + rewrite Hkind2. exact Hkind_canonical.
    + exact Hadd.
Qed.

(* The equations of a list, each on a name of the kind of its type. *)
Definition equations_fit (tenv : Typing_env.t)
    (equations : list (name * flambda_type)) : Prop :=
  ∀ (n : name) (ty : flambda_type),
    (n, ty) ∈ equations ->
    type_fits tenv ty
    ∧ simple_of_kind tenv (simple_of_name n) (kind ty).

Lemma add_equations0_keeps (equations : list (name * flambda_type)) :
  ∀ menv menv' : Meet_env.t,
    typing_env_fits (Meet_env.typing_env menv) ->
    equations_fit (Meet_env.typing_env menv) equations ->
    Meet_env.add_equations0 true menv equations mh
    = Some (Or_bottom.Ok menv') ->
    meet_env_steps menv menv'
    ∧ typing_env_fits (Meet_env.typing_env menv').
Proof.
  induction equations as [ | [n ty] equations IH ];
    intros menv menv' Hfits Hequations Hadd.
  - injection Hadd as <-.
    split; [ apply meet_env_steps_refl | exact Hfits ].
  - cbn [Meet_env.add_equations0] in Hadd.
    destruct (Meet_env.add_equation0 true menv n ty mh)
      as [ [ menv1 | ] | ] eqn:Hfirst; try discriminate Hadd.
    cbv beta iota in Hadd.
    destruct (Hequations n ty (or_introl eq_refl)) as [Hty Hkind].
    destruct (add_equation_on_simple0_keeps menv menv1 (simple_of_name n) ty
                Hfits Hty Hkind Hfirst) as [Hextends1 Hfits1].
    assert (Htenv := meet_env_extends_tenv _ _ (proj1 Hextends1)).
    destruct (IH menv1 menv' Hfits1) as [Hextends2 Hfits2].
    + intros n' ty' Hin.
      destruct (Hequations n' ty' (or_intror Hin)) as [Hty' Hkind'].
      split.
      * exact (tenv_extends_type_fits _ _ _ Htenv Hty').
      * exact (proj1 (proj2 (proj2 (proj2 (proj2 Htenv)))) _ _ Hkind').
    + exact Hadd.
    + split; [ | exact Hfits2 ].
      exact (meet_env_steps_trans _ _ _ Hextends1 Hextends2).
Qed.

(* [add_env_extension ~raise_on_bottom:true]. *)
Lemma add_env_extension0_keeps (menv menv' : Meet_env.t)
    (extension : env_extension) :
  typing_env_fits (Meet_env.typing_env menv) ->
  env_extension_fits (Meet_env.typing_env menv) extension ->
  env_extension_kinded (Meet_env.typing_env menv) extension ->
  Meet_env.add_env_extension0 true menv extension mh
  = Some (Or_bottom.Ok menv') ->
  meet_env_kept_by_meet menv menv'.
Proof.
  intros Hfits Hextension Hkinded Hadd.
  destruct extension as [equations]. cbn in Hextension, Hkinded.
  apply meet_env_steps_kept.
  apply (add_equations0_keeps (Keyed.compact name_eqb equations) menv menv'
           Hfits); [ | exact Hadd ].
  intros n ty Hin. apply (keyed_in_compact name_eqb name_eqb_eq) in Hin.
  split; [ exact (proj2 (Hextension n ty Hin)) | exact (Hkinded n ty Hin) ].
Qed.

(* Where the compiler catches the exception and makes the environment
   bottom. *)
Lemma bottom_on_exception_keeps (menv menv' : Meet_env.t)
    (result : option (Or_bottom.t Meet_env.t)) :
  typing_env_fits (Meet_env.typing_env menv) ->
  (∀ menv1 : Meet_env.t,
     result = Some (Or_bottom.Ok menv1) ->
     meet_env_kept_by_meet menv menv1) ->
  Meet_env.bottom_on_exception menv result = Some menv' ->
  meet_env_kept_by_meet menv menv'.
Proof.
  intros Hfits Hresult Hbottom.
  destruct result as [ [ menv1 | ] | ]; cbn in Hbottom;
    [ | | discriminate Hbottom ]; injection Hbottom as <-.
  - exact (Hresult menv1 eq_refl).
  - split; [ exact (make_bottom_keeps_for menv Hfits) | ].
    intros x Hknown Hunknown.
    change (Typing_env.mem None (Meet_env.typing_env menv) (Name_var x)
            = true) in Hknown.
    rewrite Hknown in Hunknown. discriminate Hunknown.
Qed.

(* ------------------------------------------------------------------------ *)
(* The meet of two types                                                    *)
(* ------------------------------------------------------------------------ *)

(* The canonical element that is asked for at the name mode of the
   variables that exist only in types is the canonical element. *)
Lemma get_canonical_simple_in_types (tenv : Typing_env.t) (s : simple)
    (result : option simple) :
  Typing_env.get_canonical_simple_exn tenv (Some Name_mode.in_types) None s
  = Some result ->
  result = Some (Typing_env.get_canonical_simple_ignoring_name_mode tenv s).
Proof.
  unfold Typing_env.get_canonical_simple_exn.
  destruct (option_map Binding_time.With_name_mode.name_mode
              (Typing_env.binding_time_and_mode_of_simple tenv s))
    as [ name_mode | ]; [ | discriminate ].
  cbv beta iota. intros [= <-].
  rewrite get_canonical_simple_ignoring_name_mode_eq.
  apply canonical_at_mode. apply name_mode_at_least_in_types.
Qed.

(* What [get_alias_then_canonical_simple_exn] answers at that name
   mode. *)
Lemma alias_then_canonical_cases (tenv : Typing_env.t) (ty : flambda_type)
    (result : option simple) :
  Typing_env.get_alias_then_canonical_simple_exn tenv
    (Some Name_mode.in_types) None ty
  = Some result ->
  (get_alias_opt ty = None ∧ result = None)
  ∨ ∃ alias : simple,
      get_alias_opt ty = Some alias
      ∧ result
        = Some (Typing_env.get_canonical_simple_ignoring_name_mode tenv alias).
Proof.
  unfold Typing_env.get_alias_then_canonical_simple_exn.
  destruct (get_alias_opt ty) as [ alias | ].
  - intros Hcanonical. right. exists alias. split; [ reflexivity | ].
    exact (get_canonical_simple_in_types tenv alias result Hcanonical).
  - intros [= <-]. left. split; reflexivity.
Qed.

(* A type that is not an alias, expanded with no canonical element
   given. *)
Lemma expand_head0_concrete (tenv : Typing_env.t) (ty : flambda_type) :
  get_alias_opt ty = None ->
  expand_head0 tenv ty None = Expanded_type.of_non_alias_type ty.
Proof. intros Halias. unfold expand_head0. rewrite Halias. reflexivity. Qed.

(* The canonical element of the alias that a type that fits is. *)
Lemma alias_canonical_of_kind (tenv : Typing_env.t) (ty : flambda_type)
    (alias : simple) :
  typing_env_fits tenv ->
  type_fits tenv ty ->
  get_alias_opt ty = Some alias ->
  is_canonical (Typing_env.aliases tenv)
    (Typing_env.get_canonical_simple_ignoring_name_mode tenv alias)
  ∧ simple_of_kind tenv
      (Typing_env.get_canonical_simple_ignoring_name_mode tenv alias)
      (kind ty).
Proof.
  intros Hfits Hty Halias.
  apply (canonical_of_kind tenv alias (kind ty) (proj1 Hfits)
           (proj1 (proj2 (proj1 (proj2 (proj2 Hfits)))))).
  exact (proj1 (proj2 (proj2 Hty)) alias (kind ty)
           (alias_among_aliases ty alias Halias)).
Qed.

(* [Meet_env.meet]. *)
Lemma meet_keeps (menv menv' : Meet_env.t) (t1 t2 : flambda_type)
    (r : meet_return_value Expanded_type.t) :
  typing_env_fits (Meet_env.typing_env menv) ->
  type_fits (Meet_env.typing_env menv) t1 ->
  type_fits (Meet_env.typing_env menv) t2 ->
  Meet_env.meet menv t1 t2 mh = Some (Meet_result.Ok r menv') ->
  let answered :=
    match r with
    | New_result expanded => Expanded_type.to_type expanded
    | Right_input => t2
    | Left_input | Both_inputs => t1
    end
  in
  meet_env_kept_by_meet menv menv'
  ∧ type_fits (Meet_env.typing_env menv') answered
  ∧ kind answered = kind t1.
Proof.
  intros Hfits Ht1 Ht2 Hmeet. cbv zeta.
  set (tenv := Meet_env.typing_env menv) in *.
  unfold Meet_env.meet in Hmeet. fold tenv in Hmeet.
  destruct (negb (kind_eqb (kind t1) (kind t2))) eqn:Hkinds;
    [ discriminate Hmeet | ].
  apply negb_false_iff, kind_eqb_eq in Hkinds.
  destruct (Typing_env.get_alias_then_canonical_simple_exn tenv
              (Some Name_mode.in_types) None t1)
    as [ simple1 | ] eqn:Hsimple1; [ | discriminate Hmeet ].
  destruct (Typing_env.get_alias_then_canonical_simple_exn tenv
              (Some Name_mode.in_types) None t2)
    as [ simple2 | ] eqn:Hsimple2; [ | discriminate Hmeet ].
  cbv beta iota zeta in Hmeet.
  (* What is kept by a function that adds an equation. *)
  assert (Hkept :
    ∀ ty : flambda_type,
      type_fits tenv ty ->
      kind ty = kind t1 ->
      meet_env_steps menv menv'
      ∧ typing_env_fits (Meet_env.typing_env menv') ->
      meet_env_kept_by_meet menv menv'
      ∧ type_fits (Meet_env.typing_env menv') ty
      ∧ kind ty = kind t1).
  { intros ty Hty Hkind Hsteps.
    split; [ exact (meet_env_steps_kept _ _ Hsteps) | ].
    split; [ | exact Hkind ].
    exact (tenv_extends_type_fits _ _ _
             (meet_env_extends_tenv _ _ (proj1 (proj1 Hsteps))) Hty). }
  destruct (alias_then_canonical_cases tenv t1 simple1 Hsimple1)
    as [ [Halias1 ->] | (alias1 & Halias1 & ->) ];
    destruct (alias_then_canonical_cases tenv t2 simple2 Hsimple2)
      as [ [Halias2 ->] | (alias2 & Halias2 & ->) ].
  - (* Neither is an alias. *)
    rewrite (expand_head0_concrete tenv t1 Halias1) in Hmeet.
    destruct (Expanded_type.of_non_alias_type t1) as [ expanded1 | ]
      eqn:Hexpanded1; [ | discriminate Hmeet ].
    cbv beta iota in Hmeet.
    rewrite (expand_head0_concrete tenv t2 Halias2) in Hmeet.
    destruct (Expanded_type.of_non_alias_type t2) as [ expanded2 | ]
      eqn:Hexpanded2; [ | discriminate Hmeet ].
    cbv beta iota in Hmeet.
    assert (Hinputs :
      meet_inputs_well_formed tenv (Expanded_type.to_type expanded1)
        (Expanded_type.to_type expanded2)).
    { split; [ exact Hfits | ].
      split; [ exact (type_fits_of_non_alias _ _ _ Hexpanded1 Ht1) | ].
      split; [ exact (type_fits_of_non_alias _ _ _ Hexpanded2 Ht2) | ].
      rewrite (proj2 (of_non_alias_type_kind _ _ Hexpanded1)).
      rewrite (proj2 (of_non_alias_type_kind _ _ Hexpanded2)).
      exact Hkinds. }
    destruct (Hmh menv menv' expanded1 expanded2 r Hinputs Hmeet)
      as (Hextends & Hfits' & Hanswer & Hkind_answer & _ & _ & Hnames).
    split; [ exact (conj (conj Hextends Hfits') Hnames) | ].
    assert (Htenv := meet_env_extends_tenv _ _ Hextends).
    destruct r as [ | | | answer ].
    + split; [ | reflexivity ].
      exact (tenv_extends_type_fits _ _ _ Htenv Ht1).
    + split; [ | symmetry; exact Hkinds ].
      exact (tenv_extends_type_fits _ _ _ Htenv Ht2).
    + split; [ | reflexivity ].
      exact (tenv_extends_type_fits _ _ _ Htenv Ht1).
    + split; [ exact Hanswer | ].
      rewrite <- (proj2 (of_non_alias_type_kind _ _ Hexpanded1)).
      exact Hkind_answer.
  - (* The second is an alias: the first is added to what it is an
       alias of. *)
    rewrite (expand_head0_concrete tenv t1 Halias1) in Hmeet.
    destruct (Expanded_type.of_non_alias_type t1) as [ expanded1 | ]
      eqn:Hexpanded1; [ | discriminate Hmeet ].
    cbv beta iota in Hmeet.
    match type of Hmeet with
    | context [ Meet_env.add_or_delay_concrete_equation_on_canonical
                  ?rob ?env ?canonical ?expanded ?meet ] =>
      destruct (Meet_env.add_or_delay_concrete_equation_on_canonical rob env
                  canonical expanded meet)
        as [ [ menv1 | ] | ] eqn:Hadd; try discriminate Hmeet
    end.
    injection Hmeet as <- <-.
    destruct (of_non_alias_type_kind _ _ Hexpanded1) as [Hkind1 Hkind2].
    destruct (alias_canonical_of_kind tenv t2 alias2 Hfits Ht2 Halias2)
      as [_ Hkind_canonical].
    apply (Hkept t2 Ht2 (eq_sym Hkinds)).
    apply (add_or_delay_concrete_equation_on_canonical_keeps menv menv1
             (Typing_env.get_canonical_simple_ignoring_name_mode tenv alias2)
             expanded1 Hfits).
    + exact (type_fits_of_non_alias _ _ _ Hexpanded1 Ht1).
    + rewrite Hkind1, Hkind2. reflexivity.
    + rewrite Hkind2, Hkinds. exact Hkind_canonical.
    + exact Hadd.
  - (* The first is an alias. *)
    rewrite (expand_head0_concrete tenv t2 Halias2) in Hmeet.
    destruct (Expanded_type.of_non_alias_type t2) as [ expanded2 | ]
      eqn:Hexpanded2; [ | discriminate Hmeet ].
    cbv beta iota in Hmeet.
    match type of Hmeet with
    | context [ Meet_env.add_or_delay_concrete_equation_on_canonical
                  ?rob ?env ?canonical ?expanded ?meet ] =>
      destruct (Meet_env.add_or_delay_concrete_equation_on_canonical rob env
                  canonical expanded meet)
        as [ [ menv1 | ] | ] eqn:Hadd; try discriminate Hmeet
    end.
    injection Hmeet as <- <-.
    destruct (of_non_alias_type_kind _ _ Hexpanded2) as [Hkind1 Hkind2].
    destruct (alias_canonical_of_kind tenv t1 alias1 Hfits Ht1 Halias1)
      as [_ Hkind_canonical].
    apply (Hkept t1 Ht1 eq_refl).
    apply (add_or_delay_concrete_equation_on_canonical_keeps menv menv1
             (Typing_env.get_canonical_simple_ignoring_name_mode tenv alias1)
             expanded2 Hfits).
    + exact (type_fits_of_non_alias _ _ _ Hexpanded2 Ht2).
    + rewrite Hkind1, Hkind2. reflexivity.
    + rewrite Hkind2, <- Hkinds. exact Hkind_canonical.
    + exact Hadd.
  - (* Both are aliases: the two are made aliases. *)
    match type of Hmeet with
    | context [ Meet_env.add_alias_between_canonicals ?rob ?env ?kind ?c1
                  ?c2 ?meet ] =>
      destruct (Meet_env.add_alias_between_canonicals rob env kind c1 c2
                  meet)
        as [ [ menv1 | ] | ] eqn:Hadd; try discriminate Hmeet
    end.
    injection Hmeet as <- <-.
    destruct (alias_canonical_of_kind tenv t1 alias1 Hfits Ht1 Halias1)
      as [Hcanonical1 Hkind_canonical1].
    destruct (alias_canonical_of_kind tenv t2 alias2 Hfits Ht2 Halias2)
      as [Hcanonical2 Hkind_canonical2].
    apply (Hkept t1 Ht1 eq_refl).
    rewrite <- Hkinds in Hkind_canonical2.
    exact (add_alias_between_canonicals_keeps menv menv1 (kind t1) _ _ Hfits
             Hcanonical1 Hcanonical2 Hkind_canonical1 Hkind_canonical2 Hadd).
Qed.

(* The kind that [kind_of_simple] reads is the kind that the simple is
   of, in an environment where a symbol is of the kind of values. *)
Lemma kind_of_simple_of_kind (tenv : Typing_env.t) (s : simple)
    (kind kind' : flambda_kind) :
  symbols_of_kind_value tenv ->
  simple_of_kind tenv s kind ->
  kind_of_simple tenv s = Some kind' ->
  kind' = kind.
Proof.
  intros Hsymbols Hkind Hread. destruct s as [ x | sym | c ].
  - apply (simple_of_kind_name tenv (Name_var x)) in Hkind.
    destruct Hkind as [ (ty & binding_time_and_mode & Hlookup & Hkind)
                      | (Hnone & Hmem & _) ].
    + unfold kind_of_simple, Typing_env.find,
        Typing_env.find_with_binding_time_and_mode' in Hread.
      rewrite Hlookup in Hread. cbn in Hread.
      destruct (Typing_env.is_bottom tenv); cbn in Hread;
        injection Hread as <-; [ rewrite kind_bottom | ]; exact Hkind.
    + unfold Typing_env.mem in Hmem. rewrite Hnone in Hmem.
      discriminate Hmem.
  - injection Hread as <-. symmetry.
    exact (symbol_of_kind_value tenv sym kind Hsymbols Hkind).
  - injection Hread as <-. cbn in Hkind. rewrite kind_type_for_const in Hkind.
    exact Hkind.
Qed.

End Keeps.

(* ======================================================================== *)
(* The statements about the functions that add an equation                  *)
(* ======================================================================== *)

Lemma add_equation_on_simple_keeps_for :
  for_validity add_equation_on_simple_keeps_statement.
Proof.
  intros mh menv menv' s ty Hmh Hfits Hty Hkind Hadd.
  apply (bottom_on_exception_keeps menv menv' _ Hfits) in Hadd;
    [ exact Hadd | ].
  intros menv1 Hresult. apply meet_env_steps_kept.
  exact (add_equation_on_simple0_keeps mh Hmh menv menv1 s ty Hfits Hty Hkind
           Hresult).
Qed.

Lemma add_equation_keeps_for :
  for_validity add_equation_keeps_statement.
Proof.
  intros mh menv menv' n ty Hmh Hfits Hty Hkind Hadd.
  exact (add_equation_on_simple_keeps_for mh menv menv' (simple_of_name n) ty
           Hmh
           Hfits Hty Hkind Hadd).
Qed.

Lemma add_equation_strict_keeps_for :
  for_validity add_equation_strict_keeps_statement.
Proof.
  intros mh menv menv' n ty Hmh Hfits Hty Hkind Hadd.
  unfold Meet_env.add_equation_strict in Hadd.
  destruct (Typing_env.is_bottom (Meet_env.typing_env menv));
    [ discriminate Hadd | ].
  apply meet_env_steps_kept.
  exact (add_equation_on_simple0_keeps mh Hmh menv menv' (simple_of_name n) ty
           Hfits Hty Hkind Hadd).
Qed.

Lemma add_env_extension_keeps_for :
  for_validity add_env_extension_keeps_statement.
Proof.
  intros mh menv menv' extension Hmh Hfits Hextension Hkinded Hadd.
  apply (bottom_on_exception_keeps menv menv' _ Hfits) in Hadd;
    [ exact Hadd | ].
  intros menv1 Hresult.
  exact (add_env_extension0_keeps mh Hmh menv menv1 extension Hfits
           Hextension Hkinded Hresult).
Qed.

Lemma add_env_extension_strict_keeps_for :
  for_validity add_env_extension_strict_keeps_statement.
Proof.
  intros mh menv menv' extension Hmh Hfits Hextension Hkinded Hadd.
  unfold Meet_env.add_env_extension_strict in Hadd.
  destruct (Typing_env.is_bottom (Meet_env.typing_env menv));
    [ discriminate Hadd | ].
  exact (add_env_extension0_keeps mh Hmh menv menv' extension Hfits
           Hextension Hkinded Hadd).
Qed.

Lemma add_alias_keeps_for :
  for_validity add_alias_keeps_statement.
Proof.
  intros mh menv menv' s1 s2 kind Hmh Hfits Hkind1 Hkind2 Hadd.
  assert (Hinvariant := proj1 Hfits).
  destruct (proj1 (proj2 (proj2 Hfits))) as (_ & Haliases & Hsymbols).
  set (tenv := Meet_env.typing_env menv) in *.
  unfold Meet_env.add_alias, Meet_env.add_alias0 in Hadd. fold tenv in Hadd.
  destruct (canonical_of_kind tenv s1 kind Hinvariant Haliases Hkind1)
    as [Hcanonical1 Hkind_canonical1].
  destruct (canonical_of_kind tenv s2 kind Hinvariant Haliases Hkind2)
    as [Hcanonical2 Hkind_canonical2].
  destruct (kind_of_simple tenv
              (Typing_env.get_canonical_simple_ignoring_name_mode tenv s1))
    as [ kind' | ] eqn:Hread; [ | discriminate Hadd ].
  cbv beta iota in Hadd.
  rewrite (kind_of_simple_of_kind tenv _ kind kind' Hsymbols Hkind_canonical1
             Hread) in Hadd.
  apply meet_env_steps_kept.
  exact (add_alias_between_canonicals_keeps mh Hmh menv menv' kind _ _ Hfits
           Hcanonical1 Hcanonical2 Hkind_canonical1 Hkind_canonical2 Hadd).
Qed.

Lemma add_equations_on_params_keeps_for :
  for_validity add_equations_on_params_keeps_statement.
Proof.
  intros mh menv menv' params param_types Hmh. revert param_types menv menv'.
  induction params as [ | param params IH ];
    intros param_types menv menv' Hfits Hparams Hadd.
  - destruct param_types as [ | ty param_types ]; [ | discriminate Hadd ].
    injection Hadd as <-.
    exact (meet_env_steps_kept _ _ (conj (meet_env_steps_refl menv) Hfits)).
  - destruct param_types as [ | ty param_types ]; [ discriminate Hadd | ].
    apply Forall2_cons_iff in Hparams. destruct Hparams as [[Hty Hkind] Hrest].
    cbn [Meet_env.add_equations_on_params] in Hadd.
    destruct (Meet_env.add_equation menv (Name_var (fst param)) ty mh)
      as [ menv1 | ] eqn:Hfirst; [ | discriminate Hadd ].
    cbv beta iota in Hadd.
    destruct (add_equation_keeps_for mh menv menv1 (Name_var (fst param)) ty Hmh
                Hfits Hty Hkind Hfirst) as [[Hextends1 Hfits1] Hnames1].
    assert (Htenv := meet_env_extends_tenv _ _ Hextends1).
    destruct (IH param_types menv1 menv' Hfits1)
      as [[Hextends2 Hfits2] Hnames2].
    + eapply Forall2_impl; [ | exact Hrest ].
      intros param' ty' [Hty' Hkind']. split.
      * exact (tenv_extends_type_fits _ _ _ Htenv Hty').
      * exact (proj1 (proj2 (proj2 (proj2 (proj2 Htenv)))) _ _ Hkind').
    + exact Hadd.
    + apply meet_env_steps_kept. split; [ | exact Hfits2 ].
      exact (meet_env_steps_trans _ _ _ (conj Hextends1 Hnames1)
               (conj Hextends2 Hnames2)).
Qed.

Lemma meet_type_keeps_for :
  for_validity meet_type_keeps_statement.
Proof.
  intros mh menv menv' t1 t2 r Hmh Hfits Ht1 Ht2 Hmeet.
  unfold Meet_env.meet_type in Hmeet.
  destruct (Meet_env.meet menv t1 t2 mh) as [ [ r0 | r0 menv0 ] | ]
    eqn:Hmeet0; cbn in Hmeet; try discriminate Hmeet.
  assert (Hkeeps := meet_keeps mh Hmh menv menv0 t1 t2 r0 Hfits Ht1 Ht2
                      Hmeet0).
  destruct r0 as [ | | | answer ]; injection Hmeet as <- <-; exact Hkeeps.
Qed.

(* ======================================================================== *)
(* What the functions add is true                                           *)
(* ======================================================================== *)

(* A simple that the environment knows has the same value, or none,
   under a valuation extended from another. *)
Lemma eval_simple_extended (tenv : Typing_env.t) (valuation valuation' : env)
    (s : simple) :
  extended_from tenv valuation valuation' ->
  Typing_env.mem_simple None tenv s = true ->
  eval_simple valuation' s = eval_simple valuation s.
Proof.
  intros Hextended Hmem.
  destruct s as [ x | sym | c ]; [ exact (Hextended x Hmem) | | ]; reflexivity.
Qed.

(* A value that has a type that fits has it under a valuation extended
   from the first. *)
Lemma type_satisfied_extended (statics : static_env) (tenv : Typing_env.t)
    (valuation valuation' : env) (v : value) (ty : flambda_type) :
  type_fits tenv ty ->
  extended_from tenv valuation valuation' ->
  type_satisfied statics valuation v ty ->
  type_satisfied statics valuation' v ty.
Proof.
  intros (_ & Hclosed & _) Hextended.
  apply (type_satisfied_coincide statics valuation valuation' v ty).
  intros x Hin. symmetry. exact (Hextended x (Hclosed x Hin)).
Qed.

Lemma meet_env_added_refl (statics : static_env) (valuation : env)
    (menv : Meet_env.t) :
  typing_env_satisfied_strictly statics valuation (Meet_env.typing_env menv) ->
  meet_env_added statics valuation menv menv.
Proof.
  intros Hsatisfied. exists valuation.
  split; [ apply extended_from_refl | ].
  split; [ exact Hsatisfied | apply keeps_equations_valued_refl ].
Qed.

(* What was added, after something else was. *)
Lemma meet_env_added_trans (statics : static_env) (valuation valuation1 : env)
    (menv menv1 menv2 : Meet_env.t) :
  keeps_names (Meet_env.typing_env menv) (Meet_env.typing_env menv1) ->
  extended_from (Meet_env.typing_env menv) valuation valuation1 ->
  keeps_equations_valued (Meet_env.typing_env menv)
    (Meet_env.typing_env menv1) valuation valuation1 ->
  meet_env_added statics valuation1 menv1 menv2 ->
  meet_env_added statics valuation menv menv2.
Proof.
  intros Hnames Hextended1 Hvalued1
    (valuation2 & Hextended2 & Hsatisfied2 & Hvalued2).
  exists valuation2.
  split; [ exact (extended_from_trans _ _ _ _ _ Hnames Hextended1 Hextended2)
         | ].
  split; [ exact Hsatisfied2 | ].
  exact (keeps_equations_valued_trans _ _ _ _ _ _ Hvalued1 Hvalued2).
Qed.

Section Sound.

Variable mh : meet_expanded_head.
Hypothesis Hkeeps : meet_expanded_head_keeps mh.
Hypothesis Hsound : meet_expanded_head_sound mh.
Variable statics : static_env.
Hypothesis Hstatics : statics_valid statics.

(* [replace_concrete_equation]: the name has a value, of the type. *)
Lemma replace_concrete_equation_sound (tenv tenv' : Typing_env.t) (n : name)
    (ty : flambda_type) (valuation : env) (v : value) :
  typing_env_satisfied_strictly statics valuation tenv ->
  value_of_name valuation n = Some v ->
  type_satisfied statics valuation v ty ->
  Meet_env.replace_concrete_equation tenv n ty = Some tenv' ->
  typing_env_satisfied_strictly statics valuation tenv'
  ∧ keeps_equations_valued tenv tenv' valuation valuation.
Proof.
  intros Hsatisfied Hvalue Hty Hreplace.
  unfold Meet_env.replace_concrete_equation in Hreplace.
  destruct (must_be_singleton ty) as [ [ const | ] | ] eqn:Hsing;
    [ | | discriminate Hreplace ]; cbn in Hreplace.
  - destruct (Aliases.canonical (Typing_env.aliases tenv) (simple_of_name n))
      eqn:Hcanonical; [ | discriminate Hreplace ].
    destruct (Typing_env.add_alias tenv (simple_of_name n) (Const const))
      as [ [ | result | ] | ] eqn:Hadd; try discriminate Hreplace.
    cbn in Hreplace.
    destruct (name_eqb (Typing_env.demoted_name result) n
              && simple_equal (Typing_env.canonical_element result)
                   (Const const)) eqn:Hcheck; [ | discriminate Hreplace ].
    injection Hreplace as <-.
    apply andb_true_iff in Hcheck. destruct Hcheck as [Hdemoted Hconst].
    apply name_eqb_eq in Hdemoted. apply simple_equal_eq in Hconst.
    rewrite Hdemoted, Hconst.
    assert (Hv : v = Value_const const).
    { exact (must_be_singleton_satisfied statics valuation v ty const Hsing
               Hty). }
    subst v.
    assert (Hsatisfied1 :
      typing_env_satisfied_strictly statics valuation
        (Typing_env.add_alias_result_t result)).
    { apply (add_alias_satisfied _ statics valuation tenv _ _ result
               Hsatisfied Hadd).
      exact Hvalue. }
    split.
    + apply replace_equation_satisfied; [ exact Hsatisfied1 | ].
      intros v' Hv'. right. rewrite Hvalue in Hv'. injection Hv' as <-.
      apply alias_satisfied. split; reflexivity.
    + apply (keeps_equations_valued_trans _ _ _ _ _ _
               (add_alias_valued tenv _ _ result Hadd valuation)).
      exact (replace_equation_valued _ n _ valuation _ Hvalue).
  - destruct (get_alias_opt ty); [ discriminate Hreplace | ].
    injection Hreplace as <-. split.
    + apply replace_equation_satisfied; [ exact Hsatisfied | ].
      intros v' Hv'. right. rewrite Hvalue in Hv'. injection Hv' as <-.
      exact Hty.
    + exact (replace_equation_valued tenv n ty valuation v Hvalue).
Qed.

(* [add_concrete_equation_on_canonical_name]. *)
Lemma add_concrete_equation_on_canonical_name_sound (menv : Meet_env.t)
    (n : name) (ty : Expanded_type.t) (valuation : env) (v : value)
    (result : Or_bottom.t Meet_env.t) :
  typing_env_fits (Meet_env.typing_env menv) ->
  type_fits (Meet_env.typing_env menv) (Expanded_type.to_type ty) ->
  Expanded_type.kind ty = kind (Expanded_type.to_type ty) ->
  typing_env_satisfied_strictly statics valuation (Meet_env.typing_env menv) ->
  value_of_name valuation n = Some v ->
  type_satisfied statics valuation v (Expanded_type.to_type ty) ->
  Meet_env.add_concrete_equation_on_canonical_name true menv n ty mh
  = Some result ->
  meet_env_sound statics valuation menv result.
Proof.
  intros Hfits Hty Hcoherent Hsatisfied Hvalue Hv Hadd.
  unfold Meet_env.add_concrete_equation_on_canonical_name in Hadd.
  destruct (Typing_env.find (Meet_env.typing_env menv) n
              (Some (Expanded_type.kind ty)))
    as [ existing | ] eqn:Hfind; [ | discriminate Hadd ].
  cbn in Hadd.
  destruct (Expanded_type.of_non_alias_type existing)
    as [ existing_ty | ] eqn:Hexisting; [ | discriminate Hadd ].
  cbn in Hadd.
  destruct (mh menv ty existing_ty) as [ result0 | ] eqn:Hmeet;
    [ | discriminate Hadd ].
  cbn in Hadd.
  assert (Hinputs :
    meet_inputs_well_formed (Meet_env.typing_env menv)
      (Expanded_type.to_type ty) (Expanded_type.to_type existing_ty)).
  { split; [ exact Hfits | ]. split; [ exact Hty | ]. split.
    - exact (type_fits_of_non_alias _ _ _ Hexisting
               (find_type_fits _ _ _ _ Hfits Hfind)).
    - rewrite (proj2 (of_non_alias_type_kind _ _ Hexisting)).
      rewrite (find_kind _ _ _ _ Hfind). symmetry. exact Hcoherent. }
  assert (Hv2 :
    type_satisfied statics valuation v (Expanded_type.to_type existing_ty)).
  { apply (of_non_alias_type_sound _ _ _ _ _ Hexisting).
    exact (find_satisfied_strictly statics valuation _ n _ existing v
             Hsatisfied Hfind Hvalue). }
  destruct (Hsound statics valuation menv ty existing_ty v result0 Hinputs
              Hstatics Hsatisfied Hv Hv2 Hmeet)
    as (r & menv1 & valuation1 & -> & Hextended1 & Hsatisfied1 & Hanswer
        & Hvalued1).
  assert (Hvalue1 : value_of_name valuation1 n = Some v).
  { unfold value_of_name in *.
    rewrite (eval_simple_extended _ _ _ _ Hextended1); [ exact Hvalue | ].
    rewrite mem_simple_of_name.
    exact (simple_of_kind_mem _ n _ (proj1 Hfits)
             (simple_of_kind_of_find _ _ _ _ Hfind)). }
  assert (Hreplace :
    ∀ (answer : Expanded_type.t) (tenv2 : Typing_env.t),
      extract_value r ty existing_ty = answer ->
      Meet_env.replace_concrete_equation (Meet_env.typing_env menv1) n
        (Expanded_type.to_type answer)
      = Some tenv2 ->
      meet_env_sound statics valuation menv
        (Or_bottom.Ok (Meet_env.with_typing_env menv1 tenv2))).
  { intros answer tenv2 Hextract Hreplace. rewrite Hextract in Hanswer.
    destruct (replace_concrete_equation_sound _ tenv2 n _ valuation1 v
                Hsatisfied1 Hvalue1 Hanswer Hreplace)
      as [Hsatisfied2 Hvalued2].
    eexists. split; [ reflexivity | ]. exists valuation1.
    split; [ exact Hextended1 | ]. split; [ exact Hsatisfied2 | ].
    exact (keeps_equations_valued_trans _ _ _ _ _ _ Hvalued1 Hvalued2). }
  assert (Hsame :
    meet_env_sound statics valuation menv (Or_bottom.Ok menv1)).
  { eexists. split; [ reflexivity | ]. exists valuation1. auto. }
  destruct r as [ | | | answer ].
  - destruct (Meet_env.replace_concrete_equation (Meet_env.typing_env menv1)
                n (Expanded_type.to_type ty))
      as [ tenv2 | ] eqn:Hreplaced; [ | discriminate Hadd ].
    injection Hadd as <-. exact (Hreplace ty tenv2 eq_refl Hreplaced).
  - injection Hadd as <-. exact Hsame.
  - injection Hadd as <-. exact Hsame.
  - destruct (Meet_env.replace_concrete_equation (Meet_env.typing_env menv1)
                n (Expanded_type.to_type answer))
      as [ tenv2 | ] eqn:Hreplaced; [ | discriminate Hadd ].
    injection Hadd as <-. exact (Hreplace answer tenv2 eq_refl Hreplaced).
Qed.

(* [add_or_delay_concrete_equation_on_canonical_name]. *)
Lemma add_or_delay_concrete_equation_on_canonical_name_sound
    (menv : Meet_env.t) (n : name) (ty : Expanded_type.t) (valuation : env)
    (v : value) (result : Or_bottom.t Meet_env.t) :
  typing_env_fits (Meet_env.typing_env menv) ->
  type_fits (Meet_env.typing_env menv) (Expanded_type.to_type ty) ->
  Expanded_type.kind ty = kind (Expanded_type.to_type ty) ->
  typing_env_satisfied_strictly statics valuation (Meet_env.typing_env menv) ->
  value_of_name valuation n = Some v ->
  type_satisfied statics valuation v (Expanded_type.to_type ty) ->
  Meet_env.add_or_delay_concrete_equation_on_canonical_name true menv n ty mh
  = Some result ->
  meet_env_sound statics valuation menv result.
Proof.
  intros Hfits Hty Hcoherent Hsatisfied Hvalue Hv Hadd.
  unfold Meet_env.add_or_delay_concrete_equation_on_canonical_name in Hadd.
  destruct (Keyed.among name_eqb n (Meet_env.adding_equations_for_names menv)).
  - injection Hadd as <-. exists menv. split; [ reflexivity | ].
    exact (meet_env_added_refl statics valuation menv Hsatisfied).
  - unfold Meet_env.adding_equation_for_name in Hadd.
    cbv beta zeta in Hadd.
    match type of Hadd with
    | context [ Meet_env.add_concrete_equation_on_canonical_name ?rob ?env ?n
                  ?ty ?meet ] =>
      destruct (Meet_env.add_concrete_equation_on_canonical_name rob env n ty
                  meet) as [ result0 | ] eqn:Hrun; [ | discriminate Hadd ];
        destruct (add_concrete_equation_on_canonical_name_sound env n ty
                    valuation v result0 Hfits Hty Hcoherent Hsatisfied Hvalue
                    Hv Hrun) as (menv2 & -> & Hadded)
    end.
    cbv beta iota in Hadd. injection Hadd as <-.
    eexists. split; [ reflexivity | exact Hadded ].
Qed.

(* [add_concrete_equation_on_const]. *)
Lemma add_concrete_equation_on_const_sound (menv : Meet_env.t) (c : const)
    (ty : Expanded_type.t) (valuation : env)
    (result : Or_bottom.t Meet_env.t) :
  typing_env_fits (Meet_env.typing_env menv) ->
  type_fits (Meet_env.typing_env menv) (Expanded_type.to_type ty) ->
  kind (type_for_const c) = kind (Expanded_type.to_type ty) ->
  typing_env_satisfied_strictly statics valuation (Meet_env.typing_env menv) ->
  type_satisfied statics valuation (Value_const c) (Expanded_type.to_type ty)
  ->
  Meet_env.add_concrete_equation_on_const true menv c ty mh = Some result ->
  meet_env_sound statics valuation menv result.
Proof.
  intros Hfits Hty Hkind Hsatisfied Hv Hadd.
  unfold Meet_env.add_concrete_equation_on_const in Hadd.
  destruct (mh menv ty (Expanded_type.create_const c)) as [ result0 | ]
    eqn:Hmeet; [ | discriminate Hadd ].
  assert (Hinputs :
    meet_inputs_well_formed (Meet_env.typing_env menv)
      (Expanded_type.to_type ty)
      (Expanded_type.to_type (Expanded_type.create_const c))).
  { split; [ exact Hfits | ]. split; [ exact Hty | ]. split.
    - apply type_fits_create_const.
    - rewrite (proj2 (create_const_kind c)). symmetry. exact Hkind. }
  destruct (Hsound statics valuation menv ty _ (Value_const c) result0
              Hinputs Hstatics Hsatisfied Hv
              (create_const_sound statics valuation c) Hmeet)
    as (r & menv1 & valuation1 & -> & Hextended1 & Hsatisfied1 & _
        & Hvalued1).
  cbn in Hadd. injection Hadd as <-.
  exists menv1. split; [ reflexivity | ]. exists valuation1. auto.
Qed.

(* [add_or_delay_concrete_equation_on_canonical]. *)
Lemma add_or_delay_concrete_equation_on_canonical_sound (menv : Meet_env.t)
    (canonical : simple) (ty : Expanded_type.t) (valuation : env) (v : value)
    (result : Or_bottom.t Meet_env.t) :
  typing_env_fits (Meet_env.typing_env menv) ->
  type_fits (Meet_env.typing_env menv) (Expanded_type.to_type ty) ->
  Expanded_type.kind ty = kind (Expanded_type.to_type ty) ->
  simple_of_kind (Meet_env.typing_env menv) canonical
    (kind (Expanded_type.to_type ty)) ->
  typing_env_satisfied_strictly statics valuation (Meet_env.typing_env menv) ->
  eval_simple valuation canonical = Some v ->
  type_satisfied statics valuation v (Expanded_type.to_type ty) ->
  Meet_env.add_or_delay_concrete_equation_on_canonical true menv canonical ty
    mh
  = Some result ->
  meet_env_sound statics valuation menv result.
Proof.
  intros Hfits Hty Hcoherent Hkind Hsatisfied Hvalue Hv Hadd.
  destruct canonical as [ x | sym | c ]; cbn in Hadd.
  - exact (add_or_delay_concrete_equation_on_canonical_name_sound menv
             (Name_var x) ty valuation v result Hfits Hty Hcoherent
             Hsatisfied Hvalue Hv Hadd).
  - exact (add_or_delay_concrete_equation_on_canonical_name_sound menv
             (Name_symbol sym) ty valuation v result Hfits Hty Hcoherent
             Hsatisfied Hvalue Hv Hadd).
  - cbn in Hvalue. injection Hvalue as <-.
    exact (add_concrete_equation_on_const_sound menv c ty valuation result
             Hfits Hty Hkind Hsatisfied Hv Hadd).
Qed.

(* [record_demotion]: the name demoted and the canonical element have
   one value. *)
Lemma record_demotion_sound (menv : Meet_env.t) (kind : flambda_kind)
    (demoted : name) (canonical : simple) (valuation : env) (v : value)
    (result : Or_bottom.t Meet_env.t) :
  typing_env_fits (Meet_env.typing_env menv) ->
  ~ is_canonical (Typing_env.aliases (Meet_env.typing_env menv))
      (simple_of_name demoted) ->
  simple_of_kind (Meet_env.typing_env menv) (simple_of_name demoted) kind ->
  simple_of_kind (Meet_env.typing_env menv) canonical kind ->
  typing_env_satisfied_strictly statics valuation (Meet_env.typing_env menv) ->
  value_of_name valuation demoted = Some v ->
  eval_simple valuation canonical = Some v ->
  Meet_env.record_demotion true menv kind demoted canonical mh = Some result ->
  meet_env_sound statics valuation menv result.
Proof.
  intros Hfits Hnot_canonical Hkind_demoted Hkind_canonical Hsatisfied
    Hvalue_demoted Hvalue_canonical Hrecord.
  assert (Hinvariant := proj1 Hfits).
  set (tenv := Meet_env.typing_env menv) in *.
  unfold Meet_env.record_demotion in Hrecord.
  fold tenv in Hrecord.
  destruct (Typing_env.find tenv demoted (Some kind))
    as [ ty_of_demoted | ] eqn:Hfind; [ | discriminate Hrecord ].
  cbv beta iota in Hrecord.
  set (alias := alias_type_of kind canonical) in *.
  set (menv1 :=
    Meet_env.map_typing_env menv
      (fun tenv => Typing_env.replace_equation tenv demoted alias)) in *.
  set (tenv1 := Typing_env.replace_equation tenv demoted alias).
  assert (Hkind_alias :
    simple_of_kind tenv (simple_of_name demoted)
      (FlambdaTypeGrammar.kind alias)).
  { unfold alias. rewrite kind_alias_type_of. exact Hkind_demoted. }
  assert (Hextends1 : tenv_extends tenv tenv1).
  { exact (replace_equation_tenv_extends tenv demoted alias Hinvariant
             Hkind_alias). }
  assert (Hfits1 : typing_env_fits tenv1).
  { apply (replace_equation_fits tenv demoted alias Hinvariant Hkind_alias
             Hfits).
    - apply type_fits_alias; [ | exact Hkind_canonical ].
      exact (simple_of_kind_mem_simple tenv canonical kind Hinvariant
               Hkind_canonical).
    - intros s _. exact Hnot_canonical.
    - apply is_obviously_unknown_alias_type_of.
    - apply is_obviously_bottom_alias_type_of. }
  assert (Hfound :
    type_satisfied statics valuation v ty_of_demoted
    ∧ FlambdaTypeGrammar.kind ty_of_demoted = kind).
  { split; [ | exact (find_kind _ _ _ _ Hfind) ].
    exact (find_satisfied_strictly statics valuation tenv demoted _ _ v
             Hsatisfied Hfind Hvalue_demoted). }
  destruct Hfound as [Hv_found Hkind_found].
  assert (Hkind_v : kind_of_value v = kind).
  { rewrite <- Hkind_found.
    exact (type_satisfied_kind statics valuation v _ Hv_found). }
  assert (Hsatisfied1 : typing_env_satisfied_strictly statics valuation tenv1).
  { apply replace_equation_satisfied; [ exact Hsatisfied | ].
    intros v' Hv'. right. rewrite Hvalue_demoted in Hv'. injection Hv' as <-.
    apply alias_satisfied. split; assumption. }
  assert (Hvalued1 : keeps_equations_valued tenv tenv1 valuation valuation).
  { exact (replace_equation_valued tenv demoted alias valuation v
             Hvalue_demoted). }
  change (Meet_env.typing_env menv1) with tenv1 in Hrecord.
  destruct (expand_head0 tenv1 ty_of_demoted (Some (simple_of_name demoted)))
    as [ expanded | ] eqn:Hexpand; [ | discriminate Hrecord ].
  cbv beta iota in Hrecord.
  assert (Hexpanded :
    type_fits tenv1 (Expanded_type.to_type expanded)
    ∧ Expanded_type.kind expanded = kind
    ∧ FlambdaTypeGrammar.kind (Expanded_type.to_type expanded) = kind
    ∧ type_satisfied statics valuation v (Expanded_type.to_type expanded)).
  { unfold expand_head0 in Hexpand.
    destruct (get_alias_opt ty_of_demoted) as [ s | ].
    - unfold expand_head_of_alias_type in Hexpand.
      assert (Hfound :
        ∃ found : flambda_type,
          Typing_env.find tenv1 demoted
            (Some (FlambdaTypeGrammar.kind ty_of_demoted))
          = Some found
          ∧ Expanded_type.of_non_alias_type found = Some expanded).
      { destruct demoted as [ x | sym ]; cbn [simple_of_name] in Hexpand;
          destruct (Typing_env.find tenv1 _
                      (Some (FlambdaTypeGrammar.kind ty_of_demoted)))
            as [ found | ] eqn:Hfound; try discriminate Hexpand;
          exists found; (split; [ reflexivity | exact Hexpand ]). }
      destruct Hfound as (found & Hfound & Hof).
      split.
      { exact (type_fits_of_non_alias _ _ _ Hof
                 (find_type_fits _ _ _ _ Hfits1 Hfound)). }
      destruct (of_non_alias_type_kind _ _ Hof) as [-> ->].
      rewrite (find_kind _ _ _ _ Hfound).
      split; [ exact Hkind_found | ]. split; [ exact Hkind_found | ].
      apply (of_non_alias_type_sound _ _ _ _ _ Hof).
      exact (find_satisfied_strictly statics valuation tenv1 demoted _ _ v
               Hsatisfied1 Hfound Hvalue_demoted).
    - split.
      { apply (type_fits_of_non_alias _ _ _ Hexpand).
        apply (tenv_extends_type_fits _ _ _ Hextends1).
        exact (find_type_fits _ _ _ _ Hfits Hfind). }
      destruct (of_non_alias_type_kind _ _ Hexpand) as [-> ->].
      split; [ exact Hkind_found | ]. split; [ exact Hkind_found | ].
      exact (of_non_alias_type_sound _ _ _ _ _ Hexpand Hv_found). }
  destruct Hexpanded as (Hty & Hkind1 & Hkind2 & Hv_expanded).
  destruct (add_or_delay_concrete_equation_on_canonical_sound menv1 canonical
              expanded valuation v result Hfits1 Hty)
    as (menv2 & -> & Hadded); try assumption.
  - rewrite Hkind1, Hkind2. reflexivity.
  - rewrite Hkind2.
    exact (proj1 (proj2 (proj2 (proj2 (proj2 Hextends1)))) _ _
             Hkind_canonical).
  - exists menv2. split; [ reflexivity | ].
    apply (meet_env_added_trans statics valuation valuation menv menv1 menv2).
    + exact (proj2 (proj2 (proj2 (proj2 (proj2 Hextends1))))).
    + apply extended_from_refl.
    + exact Hvalued1.
    + exact Hadded.
Qed.

(* [add_alias_between_canonicals]: the two canonical elements have one
   value. *)
Lemma add_alias_between_canonicals_sound (menv : Meet_env.t)
    (kind : flambda_kind) (canonical_element1 canonical_element2 : simple)
    (valuation : env) (v : value) (result : Or_bottom.t Meet_env.t) :
  typing_env_fits (Meet_env.typing_env menv) ->
  is_canonical (Typing_env.aliases (Meet_env.typing_env menv))
    canonical_element1 ->
  is_canonical (Typing_env.aliases (Meet_env.typing_env menv))
    canonical_element2 ->
  simple_of_kind (Meet_env.typing_env menv) canonical_element1 kind ->
  simple_of_kind (Meet_env.typing_env menv) canonical_element2 kind ->
  typing_env_satisfied_strictly statics valuation (Meet_env.typing_env menv) ->
  eval_simple valuation canonical_element1 = Some v ->
  eval_simple valuation canonical_element2 = Some v ->
  Meet_env.add_alias_between_canonicals true menv kind canonical_element1
    canonical_element2 mh
  = Some result ->
  meet_env_sound statics valuation menv result.
Proof.
  intros Hfits Hc1 Hc2 Hkind1 Hkind2 Hsatisfied Hvalue1 Hvalue2 Hadd.
  assert (Hinvariant := proj1 Hfits).
  set (tenv := Meet_env.typing_env menv) in *.
  assert (Hsame_env :
    meet_env_sound statics valuation menv (Or_bottom.Ok menv)).
  { exists menv. split; [ reflexivity | ].
    exact (meet_env_added_refl statics valuation menv Hsatisfied). }
  unfold Meet_env.add_alias_between_canonicals in Hadd. fold tenv in Hadd.
  destruct (simple_equal canonical_element1 canonical_element2) eqn:Hsame.
  { injection Hadd as <-. exact Hsame_env. }
  destruct (Typing_env.add_alias tenv canonical_element1 canonical_element2)
    as [ [ | result0 | ] | ] eqn:Halias; [ | | | discriminate Hadd ].
  - injection Hadd as <-. exact Hsame_env.
  - cbv beta iota in Hadd.
    assert (Hdifferent : canonical_element1 <> canonical_element2).
    { intros Heq. apply simple_equal_eq in Heq. rewrite Heq in Hsame.
      discriminate Hsame. }
    assert (Hextends1 := add_alias_tenv_extends tenv _ _ result0 Halias).
    assert (Hfits1 :=
      add_alias_fits tenv _ _ result0 Halias kind Hfits Hc1 Hc2
        (simple_of_kind_mem_simple tenv _ kind Hinvariant Hkind1)
        (simple_of_kind_mem_simple tenv _ kind Hinvariant Hkind2)
        Hkind1 Hkind2).
    destruct (add_alias_demoted tenv _ _ result0 Halias Hinvariant Hc1 Hc2
                Hdifferent) as [Hwhich Hnot].
    assert (Hsatisfied1 :
      typing_env_satisfied_strictly statics valuation
        (Typing_env.add_alias_result_t result0)).
    { apply (add_alias_satisfied _ statics valuation tenv _ _ result0
               Hsatisfied Halias).
      rewrite Hvalue1, Hvalue2. reflexivity. }
    set (tenv1 := Typing_env.add_alias_result_t result0) in *.
    assert (Hkept :
      ∀ s : simple, simple_of_kind tenv s kind -> simple_of_kind tenv1 s kind).
    { intros s.
      exact (proj1 (proj2 (proj2 (proj2 (proj2 Hextends1)))) s kind). }
    destruct (record_demotion_sound (Meet_env.with_typing_env menv tenv1) kind
                (Typing_env.demoted_name result0)
                (Typing_env.canonical_element result0) valuation v result
                Hfits1 Hnot)
      as (menv2 & -> & Hadded); try assumption.
    + apply Hkept. destruct Hwhich as [ [_ ->] | [_ ->] ]; assumption.
    + apply Hkept. destruct Hwhich as [ [-> _] | [-> _] ]; assumption.
    + unfold value_of_name.
      destruct Hwhich as [ [_ ->] | [_ ->] ]; assumption.
    + destruct Hwhich as [ [-> _] | [-> _] ]; assumption.
    + exists menv2. split; [ reflexivity | ].
      apply (meet_env_added_trans statics valuation valuation menv
               (Meet_env.with_typing_env menv tenv1) menv2).
      * exact (proj2 (proj2 (proj2 (proj2 (proj2 Hextends1))))).
      * apply extended_from_refl.
      * exact (add_alias_valued tenv _ _ result0 Halias valuation).
      * exact Hadded.
  - exfalso.
    apply (add_alias_bottom valuation tenv _ _ Halias).
    rewrite Hvalue1, Hvalue2. reflexivity.
Qed.

(* [add_equation_on_simple ~raise_on_bottom:true]: the simple has a
   value, of the type. *)
Lemma add_equation_on_simple0_sound (menv : Meet_env.t) (s : simple)
    (ty : flambda_type) (valuation : env) (v : value)
    (result : Or_bottom.t Meet_env.t) :
  typing_env_fits (Meet_env.typing_env menv) ->
  type_fits (Meet_env.typing_env menv) ty ->
  simple_of_kind (Meet_env.typing_env menv) s (kind ty) ->
  typing_env_satisfied_strictly statics valuation (Meet_env.typing_env menv) ->
  eval_simple valuation s = Some v ->
  type_satisfied statics valuation v ty ->
  Meet_env.add_equation_on_simple0 true menv s ty mh = Some result ->
  meet_env_sound statics valuation menv result.
Proof.
  intros Hfits Hty Hkind Hsatisfied Hvalue Hv Hadd.
  assert (Hinvariant := proj1 Hfits).
  assert (Hkinded := proj1 (proj2 (proj1 (proj2 (proj2 Hfits))))).
  set (tenv := Meet_env.typing_env menv) in *.
  unfold Meet_env.add_equation_on_simple0 in Hadd. fold tenv in Hadd.
  destruct (canonical_of_kind tenv s (kind ty) Hinvariant Hkinded Hkind)
    as [Hcanonical Hkind_canonical].
  assert (Hvalue_canonical :
    eval_simple valuation
      (Typing_env.get_canonical_simple_ignoring_name_mode tenv s)
    = Some v).
  { rewrite (canonical_ignoring_name_mode_satisfied _ statics valuation tenv s
               Hsatisfied).
    exact Hvalue. }
  destruct (get_alias_opt ty) as [ alias | ] eqn:Halias.
  - assert (Hkind_alias : simple_of_kind tenv alias (kind ty)).
    { exact (proj1 (proj2 (proj2 Hty)) alias (kind ty)
               (alias_among_aliases ty alias Halias)). }
    destruct (canonical_of_kind tenv alias (kind ty) Hinvariant Hkinded
                Hkind_alias) as [Hcanonical2 Hkind_canonical2].
    apply (add_alias_between_canonicals_sound menv (kind ty) _ _ valuation v
             result Hfits Hcanonical Hcanonical2 Hkind_canonical
             Hkind_canonical2 Hsatisfied Hvalue_canonical); [ | exact Hadd ].
    rewrite (canonical_ignoring_name_mode_satisfied _ statics valuation tenv
               alias Hsatisfied).
    exact (proj1 (proj1 (get_alias_satisfied statics valuation v ty alias
                           Halias) Hv)).
  - destruct (Expanded_type.of_non_alias_type ty) as [ expanded | ]
      eqn:Hexpanded; [ | discriminate Hadd ].
    cbv beta iota in Hadd.
    unfold Meet_env.add_concrete_equation_on_simple in Hadd.
    fold tenv in Hadd.
    destruct (of_non_alias_type_kind _ _ Hexpanded) as [Hkind1 Hkind2].
    apply (add_or_delay_concrete_equation_on_canonical_sound menv
             (Typing_env.get_canonical_simple_ignoring_name_mode tenv s)
             expanded valuation v result Hfits); try assumption.
    + exact (type_fits_of_non_alias _ _ _ Hexpanded Hty).
    + rewrite Hkind1, Hkind2. reflexivity.
    + rewrite Hkind2. exact Hkind_canonical.
    + exact (of_non_alias_type_sound _ _ _ _ _ Hexpanded Hv).
Qed.

(* The equations of a list are true: each name has a value, of the
   type. *)
Definition equations_true (valuation : env)
    (equations : list (name * flambda_type)) : Prop :=
  ∀ (n : name) (ty : flambda_type),
    (n, ty) ∈ equations ->
    ∃ v : value,
      value_of_name valuation n = Some v
      ∧ type_satisfied statics valuation v ty.

Lemma add_equations0_sound (equations : list (name * flambda_type)) :
  ∀ (menv : Meet_env.t) (valuation : env) (result : Or_bottom.t Meet_env.t),
    typing_env_fits (Meet_env.typing_env menv) ->
    equations_fit (Meet_env.typing_env menv) equations ->
    typing_env_satisfied_strictly statics valuation (Meet_env.typing_env menv)
    ->
    equations_true valuation equations ->
    Meet_env.add_equations0 true menv equations mh = Some result ->
    meet_env_sound statics valuation menv result.
Proof.
  induction equations as [ | [n ty] equations IH ];
    intros menv valuation result Hfits Hequations Hsatisfied Htrue Hadd.
  - injection Hadd as <-. exists menv. split; [ reflexivity | ].
    exact (meet_env_added_refl statics valuation menv Hsatisfied).
  - cbn [Meet_env.add_equations0] in Hadd.
    destruct (Meet_env.add_equation0 true menv n ty mh)
      as [ result1 | ] eqn:Hfirst; [ | discriminate Hadd ].
    cbv beta iota in Hadd.
    destruct (Hequations n ty (or_introl eq_refl)) as [Hty Hkind].
    destruct (Htrue n ty (or_introl eq_refl)) as (v & Hvalue & Hv).
    destruct (add_equation_on_simple0_sound menv (simple_of_name n) ty
                valuation v result1 Hfits Hty Hkind Hsatisfied Hvalue Hv
                Hfirst)
      as (menv1 & -> & valuation1 & Hextended1 & Hsatisfied1 & Hvalued1).
    destruct (add_equation_on_simple0_keeps mh Hkeeps menv menv1
                (simple_of_name n) ty Hfits Hty Hkind Hfirst)
      as [[Hextends1 _] Hfits1].
    assert (Htenv := meet_env_extends_tenv _ _ Hextends1).
    destruct (IH menv1 valuation1 result Hfits1)
      as (menv2 & -> & Hadded); try assumption.
    + intros n' ty' Hin.
      destruct (Hequations n' ty' (or_intror Hin)) as [Hty' Hkind'].
      split.
      * exact (tenv_extends_type_fits _ _ _ Htenv Hty').
      * exact (proj1 (proj2 (proj2 (proj2 (proj2 Htenv)))) _ _ Hkind').
    + intros n' ty' Hin.
      destruct (Hequations n' ty' (or_intror Hin)) as [Hty' Hkind'].
      destruct (Htrue n' ty' (or_intror Hin)) as (v' & Hvalue' & Hv').
      exists v'. split.
      * unfold value_of_name in *.
        rewrite (eval_simple_extended _ _ _ _ Hextended1);
          [ exact Hvalue' | ].
        exact (simple_of_kind_mem_simple _ _ _ (proj1 Hfits) Hkind').
      * exact (type_satisfied_extended statics _ _ _ v' ty' Hty' Hextended1
                 Hv').
    + exists menv2. split; [ reflexivity | ].
      exact (meet_env_added_trans statics valuation valuation1 menv menv1
               menv2 (proj2 (proj2 (proj2 (proj2 (proj2 Htenv))))) Hextended1
               Hvalued1 Hadded).
Qed.

(* [add_env_extension ~raise_on_bottom:true]. *)
Lemma add_env_extension0_sound (menv : Meet_env.t)
    (extension : env_extension) (valuation : env)
    (result : Or_bottom.t Meet_env.t) :
  typing_env_fits (Meet_env.typing_env menv) ->
  env_extension_fits (Meet_env.typing_env menv) extension ->
  env_extension_kinded (Meet_env.typing_env menv) extension ->
  typing_env_satisfied_strictly statics valuation (Meet_env.typing_env menv) ->
  env_extension_satisfied statics valuation extension ->
  Meet_env.add_env_extension0 true menv extension mh = Some result ->
  meet_env_sound statics valuation menv result.
Proof.
  intros Hfits Hextension Hkinded Hsatisfied Htrue Hadd.
  destruct extension as [equations]. cbn in Hextension, Hkinded.
  apply (add_equations0_sound (Keyed.compact name_eqb equations) menv
           valuation result Hfits); try assumption.
  - intros n ty Hin. apply (keyed_in_compact name_eqb name_eqb_eq) in Hin.
    split; [ exact (proj2 (Hextension n ty Hin)) | exact (Hkinded n ty Hin) ].
  - intros n ty Hin. apply (keyed_in_compact name_eqb name_eqb_eq) in Hin.
    exact (proj1 (env_extension_lookup statics valuation equations) Htrue n ty
             Hin).
Qed.

(* Where the compiler catches the exception: there is none. *)
Lemma bottom_on_exception_sound (menv menv' : Meet_env.t)
    (result : option (Or_bottom.t Meet_env.t)) (valuation : env) :
  (∀ result0 : Or_bottom.t Meet_env.t,
     result = Some result0 -> meet_env_sound statics valuation menv result0)
  ->
  Meet_env.bottom_on_exception menv result = Some menv' ->
  meet_env_added statics valuation menv menv'.
Proof.
  intros Hresult Hbottom.
  destruct result as [ result0 | ]; [ | discriminate Hbottom ].
  destruct (Hresult result0 eq_refl) as (menv1 & -> & Hadded).
  cbn in Hbottom. injection Hbottom as <-. exact Hadded.
Qed.

End Sound.

(* ------------------------------------------------------------------------ *)
(* The meet of two types loses no value                                     *)
(* ------------------------------------------------------------------------ *)

Section Meet_sound.

Variable mh : meet_expanded_head.
Hypothesis Hkeeps : meet_expanded_head_keeps mh.
Hypothesis Hsound : meet_expanded_head_sound mh.
Variable statics : static_env.
Hypothesis Hstatics : statics_valid statics.

(* The canonical element of the alias that a type is has the value
   that has the type. *)
Lemma alias_canonical_value (tenv : Typing_env.t) (ty : flambda_type)
    (alias : simple) (valuation : env) (v : value) :
  typing_env_satisfied_strictly statics valuation tenv ->
  get_alias_opt ty = Some alias ->
  type_satisfied statics valuation v ty ->
  eval_simple valuation
    (Typing_env.get_canonical_simple_ignoring_name_mode tenv alias)
  = Some v.
Proof.
  intros Hsatisfied Halias Hv.
  rewrite (canonical_ignoring_name_mode_satisfied _ statics valuation tenv
             alias Hsatisfied).
  exact (proj1 (proj1 (get_alias_satisfied statics valuation v ty alias
                         Halias) Hv)).
Qed.

(* [Meet_env.meet]. *)
Lemma meet_env_meet_sound (menv : Meet_env.t) (t1 t2 : flambda_type)
    (valuation : env) (v : value) (result : Meet_result.t Expanded_type.t) :
  typing_env_fits (Meet_env.typing_env menv) ->
  type_fits (Meet_env.typing_env menv) t1 ->
  type_fits (Meet_env.typing_env menv) t2 ->
  typing_env_satisfied_strictly statics valuation (Meet_env.typing_env menv) ->
  type_satisfied statics valuation v t1 ->
  type_satisfied statics valuation v t2 ->
  Meet_env.meet menv t1 t2 mh = Some result ->
  ∃ (r : meet_return_value Expanded_type.t) (menv' : Meet_env.t)
    (valuation' : env),
    result = Meet_result.Ok r menv'
    ∧ extended_from (Meet_env.typing_env menv) valuation valuation'
    ∧ typing_env_satisfied_strictly statics valuation'
        (Meet_env.typing_env menv')
    ∧ type_satisfied statics valuation' v
        match r with
        | New_result expanded => Expanded_type.to_type expanded
        | Right_input => t2
        | Left_input | Both_inputs => t1
        end
    ∧ keeps_equations_valued (Meet_env.typing_env menv)
        (Meet_env.typing_env menv') valuation valuation'.
Proof.
  intros Hfits Ht1 Ht2 Hsatisfied Hv1 Hv2 Hmeet.
  set (tenv := Meet_env.typing_env menv) in *.
  unfold Meet_env.meet in Hmeet. fold tenv in Hmeet.
  destruct (negb (kind_eqb (kind t1) (kind t2))) eqn:Hkinds;
    [ discriminate Hmeet | ].
  apply negb_false_iff, kind_eqb_eq in Hkinds.
  destruct (Typing_env.get_alias_then_canonical_simple_exn tenv
              (Some Name_mode.in_types) None t1)
    as [ simple1 | ] eqn:Hsimple1; [ | discriminate Hmeet ].
  destruct (Typing_env.get_alias_then_canonical_simple_exn tenv
              (Some Name_mode.in_types) None t2)
    as [ simple2 | ] eqn:Hsimple2; [ | discriminate Hmeet ].
  cbv beta iota zeta in Hmeet.
  (* What a function that adds an equation gives, for the input that
     the answer is. *)
  assert (Hadded :
    ∀ (ty : flambda_type) (menv1 : Meet_env.t),
      type_fits tenv ty ->
      type_satisfied statics valuation v ty ->
      meet_env_added statics valuation menv menv1 ->
      ∃ valuation' : env,
        extended_from tenv valuation valuation'
        ∧ typing_env_satisfied_strictly statics valuation'
            (Meet_env.typing_env menv1)
        ∧ type_satisfied statics valuation' v ty
        ∧ keeps_equations_valued tenv (Meet_env.typing_env menv1) valuation
            valuation').
  { intros ty menv1 Hty Hv (valuation1 & Hextended1 & Hsatisfied1 & Hvalued1).
    exists valuation1. split; [ exact Hextended1 | ].
    split; [ exact Hsatisfied1 | ]. split; [ | exact Hvalued1 ].
    exact (type_satisfied_extended statics tenv _ _ v ty Hty Hextended1 Hv). }
  destruct (alias_then_canonical_cases tenv t1 simple1 Hsimple1)
    as [ [Halias1 ->] | (alias1 & Halias1 & ->) ];
    destruct (alias_then_canonical_cases tenv t2 simple2 Hsimple2)
      as [ [Halias2 ->] | (alias2 & Halias2 & ->) ].
  - (* Neither is an alias. *)
    rewrite (expand_head0_concrete tenv t1 Halias1) in Hmeet.
    destruct (Expanded_type.of_non_alias_type t1) as [ expanded1 | ]
      eqn:Hexpanded1; [ | discriminate Hmeet ].
    cbv beta iota in Hmeet.
    rewrite (expand_head0_concrete tenv t2 Halias2) in Hmeet.
    destruct (Expanded_type.of_non_alias_type t2) as [ expanded2 | ]
      eqn:Hexpanded2; [ | discriminate Hmeet ].
    cbv beta iota in Hmeet.
    assert (Hinputs :
      meet_inputs_well_formed tenv (Expanded_type.to_type expanded1)
        (Expanded_type.to_type expanded2)).
    { split; [ exact Hfits | ].
      split; [ exact (type_fits_of_non_alias _ _ _ Hexpanded1 Ht1) | ].
      split; [ exact (type_fits_of_non_alias _ _ _ Hexpanded2 Ht2) | ].
      rewrite (proj2 (of_non_alias_type_kind _ _ Hexpanded1)).
      rewrite (proj2 (of_non_alias_type_kind _ _ Hexpanded2)).
      exact Hkinds. }
    destruct (Hsound statics valuation menv expanded1 expanded2 v result
                Hinputs Hstatics Hsatisfied
                (of_non_alias_type_sound _ _ _ _ _ Hexpanded1 Hv1)
                (of_non_alias_type_sound _ _ _ _ _ Hexpanded2 Hv2) Hmeet)
      as (r & menv' & valuation' & -> & Hextended & Hsatisfied' & Hanswer
          & Hvalued).
    exists r, menv', valuation'. split; [ reflexivity | ].
    split; [ exact Hextended | ]. split; [ exact Hsatisfied' | ].
    split; [ | exact Hvalued ].
    destruct r as [ | | | answer ].
    + exact (type_satisfied_extended statics tenv _ _ v t1 Ht1 Hextended Hv1).
    + exact (type_satisfied_extended statics tenv _ _ v t2 Ht2 Hextended Hv2).
    + exact (type_satisfied_extended statics tenv _ _ v t1 Ht1 Hextended Hv1).
    + exact Hanswer.
  - (* The second is an alias. *)
    rewrite (expand_head0_concrete tenv t1 Halias1) in Hmeet.
    destruct (Expanded_type.of_non_alias_type t1) as [ expanded1 | ]
      eqn:Hexpanded1; [ | discriminate Hmeet ].
    cbv beta iota in Hmeet.
    match type of Hmeet with
    | context [ Meet_env.add_or_delay_concrete_equation_on_canonical
                  ?rob ?env ?canonical ?expanded ?meet ] =>
      destruct (Meet_env.add_or_delay_concrete_equation_on_canonical rob env
                  canonical expanded meet)
        as [ result0 | ] eqn:Hadd; [ | discriminate Hmeet ]
    end.
    destruct (of_non_alias_type_kind _ _ Hexpanded1) as [Hkind1 Hkind2].
    destruct (alias_canonical_of_kind tenv t2 alias2 Hfits Ht2 Halias2)
      as [_ Hkind_canonical].
    destruct (add_or_delay_concrete_equation_on_canonical_sound mh Hsound
                statics Hstatics menv
                (Typing_env.get_canonical_simple_ignoring_name_mode tenv
                   alias2)
                expanded1 valuation v result0 Hfits)
      as (menv1 & -> & Hadded1); try assumption.
    + exact (type_fits_of_non_alias _ _ _ Hexpanded1 Ht1).
    + rewrite Hkind1, Hkind2. reflexivity.
    + rewrite Hkind2, Hkinds. exact Hkind_canonical.
    + exact (alias_canonical_value tenv t2 alias2 valuation v Hsatisfied
               Halias2 Hv2).
    + exact (of_non_alias_type_sound _ _ _ _ _ Hexpanded1 Hv1).
    + cbv beta iota in Hmeet. injection Hmeet as <-.
      destruct (Hadded t2 menv1 Ht2 Hv2 Hadded1) as (valuation' & Hresult).
      exists Right_input, menv1, valuation'.
      split; [ reflexivity | exact Hresult ].
  - (* The first is an alias. *)
    rewrite (expand_head0_concrete tenv t2 Halias2) in Hmeet.
    destruct (Expanded_type.of_non_alias_type t2) as [ expanded2 | ]
      eqn:Hexpanded2; [ | discriminate Hmeet ].
    cbv beta iota in Hmeet.
    match type of Hmeet with
    | context [ Meet_env.add_or_delay_concrete_equation_on_canonical
                  ?rob ?env ?canonical ?expanded ?meet ] =>
      destruct (Meet_env.add_or_delay_concrete_equation_on_canonical rob env
                  canonical expanded meet)
        as [ result0 | ] eqn:Hadd; [ | discriminate Hmeet ]
    end.
    destruct (of_non_alias_type_kind _ _ Hexpanded2) as [Hkind1 Hkind2].
    destruct (alias_canonical_of_kind tenv t1 alias1 Hfits Ht1 Halias1)
      as [_ Hkind_canonical].
    destruct (add_or_delay_concrete_equation_on_canonical_sound mh Hsound
                statics Hstatics menv
                (Typing_env.get_canonical_simple_ignoring_name_mode tenv
                   alias1)
                expanded2 valuation v result0 Hfits)
      as (menv1 & -> & Hadded1); try assumption.
    + exact (type_fits_of_non_alias _ _ _ Hexpanded2 Ht2).
    + rewrite Hkind1, Hkind2. reflexivity.
    + rewrite Hkind2, <- Hkinds. exact Hkind_canonical.
    + exact (alias_canonical_value tenv t1 alias1 valuation v Hsatisfied
               Halias1 Hv1).
    + exact (of_non_alias_type_sound _ _ _ _ _ Hexpanded2 Hv2).
    + cbv beta iota in Hmeet. injection Hmeet as <-.
      destruct (Hadded t1 menv1 Ht1 Hv1 Hadded1) as (valuation' & Hresult).
      exists Left_input, menv1, valuation'.
      split; [ reflexivity | exact Hresult ].
  - (* Both are aliases. *)
    match type of Hmeet with
    | context [ Meet_env.add_alias_between_canonicals ?rob ?env ?kind ?c1
                  ?c2 ?meet ] =>
      destruct (Meet_env.add_alias_between_canonicals rob env kind c1 c2
                  meet)
        as [ result0 | ] eqn:Hadd; [ | discriminate Hmeet ]
    end.
    destruct (alias_canonical_of_kind tenv t1 alias1 Hfits Ht1 Halias1)
      as [Hcanonical1 Hkind_canonical1].
    destruct (alias_canonical_of_kind tenv t2 alias2 Hfits Ht2 Halias2)
      as [Hcanonical2 Hkind_canonical2].
    rewrite <- Hkinds in Hkind_canonical2.
    destruct (add_alias_between_canonicals_sound mh Hsound
                statics Hstatics menv (kind t1) _ _ valuation v result0 Hfits
                Hcanonical1 Hcanonical2 Hkind_canonical1 Hkind_canonical2
                Hsatisfied
                (alias_canonical_value tenv t1 alias1 valuation v Hsatisfied
                   Halias1 Hv1)
                (alias_canonical_value tenv t2 alias2 valuation v Hsatisfied
                   Halias2 Hv2)
                Hadd)
      as (menv1 & -> & Hadded1).
    cbv beta iota in Hmeet. injection Hmeet as <-.
    destruct (Hadded t1 menv1 Ht1 Hv1 Hadded1) as (valuation' & Hresult).
    exists Both_inputs, menv1, valuation'.
    split; [ reflexivity | exact Hresult ].
Qed.

End Meet_sound.

(* ======================================================================== *)
(* The statements about what the functions add                              *)
(* ======================================================================== *)

Lemma meet_env_add_equation_on_simple_sound_for :
  for_validity meet_env_add_equation_on_simple_sound_statement.
Proof.
  intros mh statics valuation menv menv' s ty v Hkeeps Hsound Hfits
    Hty Hkind Hstatics Hsatisfied Hvalue Hv Hadd.
  apply (bottom_on_exception_sound statics menv menv' _ valuation) in Hadd;
    [ exact Hadd | ].
  intros result0 Hresult.
  exact (add_equation_on_simple0_sound mh Hsound statics Hstatics menv s ty
           valuation v result0 Hfits Hty Hkind Hsatisfied Hvalue Hv Hresult).
Qed.

Lemma meet_env_add_equation_sound_for :
  for_validity meet_env_add_equation_sound_statement.
Proof.
  intros mh statics valuation menv menv' n ty v Hkeeps Hsound Hfits
    Hty Hkind Hstatics Hsatisfied Hvalue Hv Hadd.
  exact (meet_env_add_equation_on_simple_sound_for mh statics valuation
           menv menv' (simple_of_name n) ty v Hkeeps Hsound Hfits Hty Hkind
           Hstatics Hsatisfied Hvalue Hv Hadd).
Qed.

(* An environment that a valuation satisfies is not bottom. *)
Lemma satisfied_not_bottom (statics : static_env) (valuation : env)
    (tenv : Typing_env.t) :
  typing_env_satisfied_strictly statics valuation tenv ->
  Typing_env.is_bottom tenv = false.
Proof. intros [Hnot_bottom _]. exact Hnot_bottom. Qed.

Lemma meet_env_add_equation_strict_sound_for :
  for_validity meet_env_add_equation_strict_sound_statement.
Proof.
  intros mh statics valuation menv n ty v result Hkeeps Hsound
    Hfits Hty Hkind Hstatics Hsatisfied Hvalue Hv Hadd.
  unfold Meet_env.add_equation_strict in Hadd.
  rewrite (satisfied_not_bottom statics valuation _ Hsatisfied) in Hadd.
  exact (add_equation_on_simple0_sound mh Hsound statics Hstatics menv
           (simple_of_name n) ty valuation v result Hfits Hty Hkind Hsatisfied
           Hvalue Hv Hadd).
Qed.

Lemma meet_env_add_env_extension_sound_for :
  for_validity meet_env_add_env_extension_sound_statement.
Proof.
  intros mh statics valuation menv menv' extension Hkeeps Hsound
    Hfits Hextension Hkinded Hstatics Hsatisfied Htrue Hadd.
  apply (bottom_on_exception_sound statics menv menv' _ valuation) in Hadd;
    [ exact Hadd | ].
  intros result0 Hresult.
  exact (add_env_extension0_sound mh Hkeeps Hsound statics Hstatics menv
           extension valuation result0 Hfits Hextension Hkinded Hsatisfied
           Htrue Hresult).
Qed.

Lemma meet_env_add_env_extension_strict_sound_for :
  for_validity meet_env_add_env_extension_strict_sound_statement.
Proof.
  intros mh statics valuation menv extension result Hkeeps Hsound
    Hfits Hextension Hkinded Hstatics Hsatisfied Htrue Hadd.
  unfold Meet_env.add_env_extension_strict in Hadd.
  rewrite (satisfied_not_bottom statics valuation _ Hsatisfied) in Hadd.
  exact (add_env_extension0_sound mh Hkeeps Hsound statics Hstatics menv
           extension valuation result Hfits Hextension Hkinded Hsatisfied
           Htrue Hadd).
Qed.

Lemma meet_env_add_alias_sound_for :
  for_validity meet_env_add_alias_sound_statement.
Proof.
  intros mh statics valuation menv s1 s2 kind v result Hkeeps
    Hsound Hfits Hkind1 Hkind2 Hstatics Hsatisfied Hvalue1 Hvalue2 Hadd.
  assert (Hinvariant := proj1 Hfits).
  destruct (proj1 (proj2 (proj2 Hfits))) as (_ & Haliases & Hsymbols).
  set (tenv := Meet_env.typing_env menv) in *.
  unfold Meet_env.add_alias, Meet_env.add_alias0 in Hadd. fold tenv in Hadd.
  destruct (canonical_of_kind tenv s1 kind Hinvariant Haliases Hkind1)
    as [Hcanonical1 Hkind_canonical1].
  destruct (canonical_of_kind tenv s2 kind Hinvariant Haliases Hkind2)
    as [Hcanonical2 Hkind_canonical2].
  destruct (kind_of_simple tenv
              (Typing_env.get_canonical_simple_ignoring_name_mode tenv s1))
    as [ kind' | ] eqn:Hread; [ | discriminate Hadd ].
  cbv beta iota in Hadd.
  rewrite (kind_of_simple_of_kind tenv _ kind kind' Hsymbols Hkind_canonical1
             Hread) in Hadd.
  apply (add_alias_between_canonicals_sound mh Hsound
           statics Hstatics menv kind _ _ valuation v result Hfits Hcanonical1
           Hcanonical2 Hkind_canonical1 Hkind_canonical2 Hsatisfied);
    [ | | exact Hadd ].
  - rewrite (canonical_ignoring_name_mode_satisfied _ statics valuation tenv
               s1 Hsatisfied).
    exact Hvalue1.
  - rewrite (canonical_ignoring_name_mode_satisfied _ statics valuation tenv
               s2 Hsatisfied).
    exact Hvalue2.
Qed.

Lemma Forall2_and {A B : Type} (P Q : A -> B -> Prop) (l1 : list A)
    (l2 : list B) :
  Forall2 P l1 l2 -> Forall2 Q l1 l2 ->
  Forall2 (fun a b => P a b ∧ Q a b) l1 l2.
Proof.
  intros HP. induction HP as [ | a b l1 l2 Hab HP IH ]; intros HQ.
  - constructor.
  - apply Forall2_cons_iff in HQ. destruct HQ as [Hq HQ].
    constructor; [ split; assumption | exact (IH HQ) ].
Qed.

Lemma meet_env_add_equations_on_params_sound_for :
  for_validity meet_env_add_equations_on_params_sound_statement.
Proof.
  intros mh statics valuation menv menv' params param_types Hkeeps
    Hsound.
  revert param_types menv menv' valuation.
  induction params as [ | param params IH ];
    intros param_types menv menv' valuation Hfits Hparams Hstatics Hsatisfied
      Htrue Hadd.
  - destruct param_types as [ | ty param_types ]; [ | discriminate Hadd ].
    injection Hadd as <-.
    exact (meet_env_added_refl statics valuation menv Hsatisfied).
  - destruct param_types as [ | ty param_types ]; [ discriminate Hadd | ].
    apply Forall2_cons_iff in Hparams. destruct Hparams as [[Hty Hkind] Hrest].
    apply Forall2_cons_iff in Htrue.
    destruct Htrue as [(v & Hvalue & Hv) Htrue_rest].
    cbn [Meet_env.add_equations_on_params] in Hadd.
    destruct (Meet_env.add_equation menv (Name_var (fst param)) ty mh)
      as [ menv1 | ] eqn:Hfirst; [ | discriminate Hadd ].
    cbv beta iota in Hadd.
    destruct (add_equation_keeps_for mh menv menv1 (Name_var (fst param)) ty
                Hkeeps Hfits Hty Hkind Hfirst) as [[Hextends1 Hfits1] _].
    destruct (meet_env_add_equation_sound_for mh statics valuation menv
                menv1 (Name_var (fst param)) ty v Hkeeps Hsound Hfits Hty
                Hkind Hstatics Hsatisfied Hvalue Hv Hfirst)
      as (valuation1 & Hextended1 & Hsatisfied1 & Hvalued1).
    assert (Htenv := meet_env_extends_tenv _ _ Hextends1).
    assert (Hboth := Forall2_and _ _ _ _ Hrest Htrue_rest).
    apply (meet_env_added_trans statics valuation valuation1 menv menv1 menv'
             (proj2 (proj2 (proj2 (proj2 (proj2 Htenv))))) Hextended1
             Hvalued1).
    apply (IH param_types menv1 menv' valuation1 Hfits1); try assumption.
    + eapply Forall2_impl; [ | exact Hrest ].
      intros param' ty' [Hty' Hkind']. split.
      * exact (tenv_extends_type_fits _ _ _ Htenv Hty').
      * exact (proj1 (proj2 (proj2 (proj2 (proj2 Htenv)))) _ _ Hkind').
    + eapply Forall2_impl; [ | exact Hboth ].
      intros param' ty' [[Hty' Hkind'] (v' & Hvalue' & Hv')].
      exists v'. split.
      * rewrite (Hextended1 (fst param')); [ exact Hvalue' | ].
        exact (simple_of_kind_mem _ (Name_var (fst param')) _ (proj1 Hfits)
                 Hkind').
      * exact (type_satisfied_extended statics _ _ _ v' ty' Hty' Hextended1
                 Hv').
Qed.

Lemma meet_type_ok_for :
  for_validity meet_type_ok_statement.
Proof.
  intros mh v Hkeeps Hsound menv t1 t2 result Hfits Ht1 Ht2 Hmeet.
  split.
  - intros r menv' ->.
    destruct (meet_type_keeps_for mh menv menv' t1 t2 r Hkeeps Hfits Ht1 Ht2
                Hmeet) as ([[Hextends Hfits'] Hnames] & Hanswer & _).
    split; [ exact Hextends | ]. split; [ exact Hfits' | ].
    split; [ exact Hanswer | exact Hnames ].
  - intros statics valuation Hstatics Hsatisfied Hv1 Hv2.
    unfold Meet_env.meet_type in Hmeet.
    destruct (Meet_env.meet menv t1 t2 mh) as [ result0 | ] eqn:Hmeet0;
      [ | discriminate Hmeet ].
    cbn in Hmeet. injection Hmeet as <-.
    destruct (meet_env_meet_sound mh Hsound statics Hstatics menv t1 t2
                valuation v result0 Hfits Ht1 Ht2 Hsatisfied Hv1 Hv2 Hmeet0)
      as (r & menv' & valuation' & -> & Hextended & Hsatisfied' & Hanswer
          & Hvalued).
    destruct r as [ | | | answer ];
      [ exists Left_input | exists Right_input | exists Both_inputs
      | exists (New_result (Expanded_type.to_type answer)) ];
      exists menv', valuation'; (split; [ reflexivity | ]);
      (split; [ exact Hextended | ]); (split; [ exact Hsatisfied' | ]);
      (split; [ exact Hanswer | exact Hvalued ]).
Qed.

End Validity.

Theorem make_bottom_keeps :
  make_bottom_keeps_statement.
Proof.
  first
    [ exact (@make_bottom_keeps_for standard_validity)
    | exact make_bottom_keeps_for ].
Qed.

Theorem add_variable_definition_keeps :
  add_variable_definition_keeps_statement.
Proof.
  first
    [ exact (@add_variable_definition_keeps_for standard_validity)
    | exact add_variable_definition_keeps_for ].
Qed.

Theorem add_equation_on_simple_keeps :
  add_equation_on_simple_keeps_statement.
Proof.
  first
    [ exact (@add_equation_on_simple_keeps_for standard_validity)
    | exact add_equation_on_simple_keeps_for ].
Qed.

Theorem add_equation_keeps :
  add_equation_keeps_statement.
Proof.
  first
    [ exact (@add_equation_keeps_for standard_validity)
    | exact add_equation_keeps_for ].
Qed.

Theorem add_equation_strict_keeps :
  add_equation_strict_keeps_statement.
Proof.
  first
    [ exact (@add_equation_strict_keeps_for standard_validity)
    | exact add_equation_strict_keeps_for ].
Qed.

Theorem add_env_extension_keeps :
  add_env_extension_keeps_statement.
Proof.
  first
    [ exact (@add_env_extension_keeps_for standard_validity)
    | exact add_env_extension_keeps_for ].
Qed.

Theorem add_env_extension_strict_keeps :
  add_env_extension_strict_keeps_statement.
Proof.
  first
    [ exact (@add_env_extension_strict_keeps_for standard_validity)
    | exact add_env_extension_strict_keeps_for ].
Qed.

Theorem add_alias_keeps :
  add_alias_keeps_statement.
Proof.
  first
    [ exact (@add_alias_keeps_for standard_validity)
    | exact add_alias_keeps_for ].
Qed.

Theorem add_equations_on_params_keeps :
  add_equations_on_params_keeps_statement.
Proof.
  first
    [ exact (@add_equations_on_params_keeps_for standard_validity)
    | exact add_equations_on_params_keeps_for ].
Qed.

Theorem meet_type_keeps :
  meet_type_keeps_statement.
Proof.
  first
    [ exact (@meet_type_keeps_for standard_validity)
    | exact meet_type_keeps_for ].
Qed.

Theorem meet_env_add_equation_on_simple_sound :
  meet_env_add_equation_on_simple_sound_statement.
Proof.
  first
    [ exact (@meet_env_add_equation_on_simple_sound_for standard_validity)
    | exact meet_env_add_equation_on_simple_sound_for ].
Qed.

Theorem meet_env_add_equation_sound :
  meet_env_add_equation_sound_statement.
Proof.
  first
    [ exact (@meet_env_add_equation_sound_for standard_validity)
    | exact meet_env_add_equation_sound_for ].
Qed.

Theorem meet_env_add_equation_strict_sound :
  meet_env_add_equation_strict_sound_statement.
Proof.
  first
    [ exact (@meet_env_add_equation_strict_sound_for standard_validity)
    | exact meet_env_add_equation_strict_sound_for ].
Qed.

Theorem meet_env_add_env_extension_sound :
  meet_env_add_env_extension_sound_statement.
Proof.
  first
    [ exact (@meet_env_add_env_extension_sound_for standard_validity)
    | exact meet_env_add_env_extension_sound_for ].
Qed.

Theorem meet_env_add_env_extension_strict_sound :
  meet_env_add_env_extension_strict_sound_statement.
Proof.
  first
    [ exact (@meet_env_add_env_extension_strict_sound_for standard_validity)
    | exact meet_env_add_env_extension_strict_sound_for ].
Qed.

Theorem meet_env_add_alias_sound :
  meet_env_add_alias_sound_statement.
Proof.
  first
    [ exact (@meet_env_add_alias_sound_for standard_validity)
    | exact meet_env_add_alias_sound_for ].
Qed.

Theorem meet_env_add_equations_on_params_sound :
  meet_env_add_equations_on_params_sound_statement.
Proof.
  first
    [ exact (@meet_env_add_equations_on_params_sound_for standard_validity)
    | exact meet_env_add_equations_on_params_sound_for ].
Qed.

Theorem meet_type_ok :
  meet_type_ok_statement.
Proof.
  first
    [ exact (@meet_type_ok_for standard_validity)
    | exact meet_type_ok_for ].
Qed.
