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


(* Proofs of the statements of FlambdaMeetParamsSpec.v. Nothing here
   needs review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaClosuresSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeMeaningSpec FlambdaTypeMeaningProofs
  FlambdaBindingTime FlambdaEnvAliases FlambdaEnvAliasesSpec
  FlambdaEnvAliasesProofs FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaTypingEnvProofs FlambdaTypingEnvExamplesProofs
  FlambdaTypingEnvInvariantSpec FlambdaTypingEnvInvariantProofs
  FlambdaTypingEnvKindedSpec FlambdaTypingEnvKindedProofs
  FlambdaTypingEnvOperationsSpec FlambdaTypingEnvOperationsProofs
  FlambdaMeetGrammar FlambdaExpandHead FlambdaMeetEnv FlambdaMeet
  FlambdaMeetSpec FlambdaMeetProofs FlambdaMeetEnvSpec FlambdaMeetEnvProofs
  FlambdaMeetScopesProofs FlambdaMeetImmediateSpec FlambdaMeetImmediateProofs
  FlambdaMeetFuelSpec FlambdaMeetFuelProofs FlambdaMeetClosuresSpec
  FlambdaMeetClosuresProofs FlambdaMeetExamplesSpec FlambdaMeetExamplesProofs
  FlambdaSimplifySwitchProofs FlambdaMeetParamsSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* The join that makes a variable                                           *)
(* ======================================================================== *)

(* A type that mentions no name fits any environment. *)
Ltac closed_type_fits :=
  split; [ vm_compute; reflexivity | ];
  split; [ intros x Hin; vm_compute in Hin; destruct Hin | ];
  split;
  [ intros s k Hin; vm_compute in Hin; destruct Hin
  | intros n k Hin; vm_compute in Hin; destruct Hin ].

Lemma naming_join_type_fits (tenv : Typing_env.t) :
  type_fits tenv naming_join_type.
Proof. closed_type_fits. Qed.

(* The valuation with the variable bound agrees with the valuation on
   what an environment that does not know the variable knows. *)
Lemma extended_from_bind (tenv : Typing_env.t) (valuation : env)
    (q : variable) (v : value) :
  Typing_env.mem None tenv (Name_var q) = false ->
  extended_from tenv valuation ((q, v) :: valuation).
Proof.
  intros Hq x Hx. cbn [lookup].
  destruct (Pos.eqb x q) eqn:Hxq; [ | reflexivity ].
  apply Pos.eqb_eq in Hxq. subst x. rewrite Hx in Hq. discriminate Hq.
Qed.

Theorem naming_join_sound : naming_join_sound_statement.
Proof.
  intros q meet_head initial_env result scoped_envs initial_tenv Hsound Hkeeps
    Hfits Hscoped Hagree Hjoin.
  unfold naming_join in Hjoin. fold initial_tenv in Hjoin.
  destruct (Typing_env.mem None initial_tenv (Name_var q)
            || existsb
                 (fun scoped_env =>
                    Typing_env.mem None scoped_env (Name_var q))
                 scoped_envs
            || among q (Meet_env.names_to_avoid initial_env)) eqn:Htest.
  { exact (forgetful_join_sound meet_head initial_env result scoped_envs
             Hsound Hkeeps Hfits Hscoped Hagree Hjoin). }
  apply orb_false_iff in Htest. destruct Htest as [Htest Havoid].
  apply orb_false_iff in Htest. destruct Htest as [Hunknown Hno_scoped].
  assert (Hscoped_unknown :
    ∀ scoped_env : Typing_env.t,
      scoped_env ∈ scoped_envs ->
      Typing_env.mem None scoped_env (Name_var q) = false).
  { intros scoped_env Hin.
    destruct (Typing_env.mem None scoped_env (Name_var q)) eqn:Hq;
      [ | reflexivity ].
    rewrite (proj2 (existsb_exists _ _)
               (ex_intro _ scoped_env (conj Hin Hq))) in Hno_scoped.
    discriminate Hno_scoped. }
  injection Hjoin as <-.
  set (K := Naked_number Kind_naked_immediate) in *.
  destruct (define_fresh initial_env q K Hfits Hunknown)
    as ([Hextends1 Hfits1] & Hkind1 & Hknown1 & Hsat1 & Hvalued1).
  cbv zeta in Hfits1, Hkind1, Hknown1, Hsat1, Hvalued1.
  set (tenv1 :=
    Typing_env.add_variable_definition initial_tenv q K Name_mode.in_types).
  change (Meet_env.typing_env
            (Meet_env.add_variable_definition initial_env q K
               Name_mode.in_types))
    with tenv1 in Hextends1, Hfits1, Hkind1, Hknown1, Hsat1, Hvalued1.
  change (Meet_env.typing_env initial_env)
    with initial_tenv in Hknown1, Hsat1, Hvalued1.
  assert (Hinvariant1 := proj1 Hfits1).
  assert (Hkindq :
    simple_of_kind tenv1 (simple_of_name (Name_var q))
      (kind naming_join_type)) by exact Hkind1.
  set (tenv2 :=
    Typing_env.replace_equation tenv1 (Name_var q) naming_join_type).
  assert (Hextends2 : tenv_extends tenv1 tenv2).
  { exact (replace_equation_tenv_extends tenv1 _ _ Hinvariant1 Hkindq). }
  assert (Hfits2 : typing_env_fits tenv2).
  { apply (replace_equation_fits tenv1 _ _ Hinvariant1 Hkindq Hfits1
             (naming_join_type_fits tenv1)).
    - intros alias Halias. vm_compute in Halias. discriminate Halias.
    - reflexivity.
    - reflexivity. }
  assert (Hextends : tenv_extends initial_tenv tenv2).
  { exact (tenv_extends_trans _ _ _ (meet_env_extends_tenv _ _ Hextends1)
             Hextends2). }
  assert (Hnew :
    ∀ x : variable,
      Typing_env.mem None tenv2 (Name_var x) = true ->
      Typing_env.mem None initial_tenv (Name_var x) = false ->
      x = q).
  { intros x Hknown Hunknown_x.
    apply (replace_equation_no_new tenv1 (Name_var q) naming_join_type)
      in Hknown.
    destruct (Hknown1 x Hknown) as [ -> | Hinitial ]; [ reflexivity | ].
    rewrite Hinitial in Hunknown_x. discriminate Hunknown_x. }
  cbn [Meet_env.typing_env Meet_env.with_names_to_avoid
         Meet_env.with_typing_env Meet_env.names_to_avoid].
  split; [ | split; [ | split; [ | split; [ | split ] ] ] ].
  - intros scoped_env statics valuation Hin Hstatics Hinitial Hscoped_env
      Hvalued.
    set (v5 := Value_const (Naked_immediate 5)).
    set (valuation1 := (q, v5) :: valuation).
    assert (Hq_value : value_of_name valuation1 (Name_var q) = Some v5).
    { unfold valuation1. cbn. rewrite Pos.eqb_refl. reflexivity. }
    assert (Hwf : typing_env_well_formed initial_tenv = true).
    { exact (proj1 (proj1 Hfits)). }
    exists valuation1. split; [ | split ].
    + exact (extended_from_bind scoped_env valuation q v5
               (Hscoped_unknown scoped_env Hin)).
    + apply replace_equation_satisfied.
      * apply Hsat1.
        -- exact (satisfied_strictly_extended statics initial_tenv valuation
                    valuation1 Hwf
                    (extended_from_bind initial_tenv valuation q v5 Hunknown)
                    Hinitial).
        -- intros v Hv. unfold valuation1 in Hv. cbn in Hv.
           rewrite Pos.eqb_refl in Hv. injection Hv as <-. reflexivity.
      * intros v Hv. right. rewrite Hq_value in Hv. injection Hv as <-.
        unfold naming_join_type, v5. open_type.
        exists 5. split; [ reflexivity | ].
        split; [ left; reflexivity | intros relation n [] ].
    + intros cut_after Hvalued0.
      apply (replace_equation_valued tenv1 (Name_var q) naming_join_type
               valuation1 v5 Hq_value).
      apply Hvalued1.
      exact (equations_valued_extended initial_tenv cut_after valuation
               valuation1 (proj1 Hfits)
               (extended_from_bind initial_tenv valuation q v5 Hunknown)
               Hvalued0).
  - destruct Hextends as (Hs & Hp & Hm & Hd & Hkind & Hk).
    repeat split; try assumption; try reflexivity.
    intros x Hamong.
    cbn [Meet_env.names_to_avoid Meet_env.with_names_to_avoid].
    unfold among in *. cbn [existsb]. rewrite Hamong. apply orb_true_r.
  - exact Hfits2.
  - intros x Hknown Hunknown_x _.
    rewrite (Hnew x Hknown Hunknown_x).
    split; [ exact Havoid | ].
    unfold among. cbn [existsb]. rewrite Pos.eqb_refl. reflexivity.
  - intros x Hknown Hunknown_x.
    rewrite (Hnew x Hknown Hunknown_x).
    unfold Typing_env.mem, tenv2. rewrite replace_equation_lookup.
    unfold tenv1. rewrite add_variable_definition_names.
    rewrite (keyed_lookup_head name_eqb name_eqb_eq). cbn.
    unfold Binding_time.With_name_mode.scoped_name_mode. cbn.
    destruct (_ || _); reflexivity.
  - intros scoped_env s kind0 Hin Hmem Hkind_s.
    destruct (Typing_env.mem_simple None initial_tenv s) eqn:Hs_initial.
    + apply (proj1 (proj2 (proj2 (proj2 (proj2 Hextends))))).
      exact (scoped_env_of_kind initial_tenv scoped_env s kind0
               (Hscoped scoped_env Hin) Hs_initial Hkind_s).
    + exfalso. destruct s as [ x | sym | c ].
      * rewrite (Hnew x Hmem Hs_initial) in Hkind_s.
        assert (Hq := Hscoped_unknown scoped_env Hin).
        rewrite (simple_of_kind_mem scoped_env (Name_var q) kind0
                   (proj1 (proj1 (Hscoped scoped_env Hin))) Hkind_s)
          in Hq.
        discriminate Hq.
      * destruct Hextends as (_ & _ & _ & Hd & _).
        unfold Typing_env.mem_simple, Typing_env.mem in Hmem, Hs_initial.
        rewrite Hd in Hmem. rewrite Hmem in Hs_initial.
        discriminate Hs_initial.
      * discriminate Hs_initial.
Qed.

(* ======================================================================== *)
(* The example                                                              *)
(* ======================================================================== *)

Lemma example_params_env_fits : typing_env_fits example_params_env.
Proof.
  apply plain_types_fits; [ | | vm_compute; reflexivity ].
  - split; [ vm_compute; reflexivity | ].
    intros n ty Hlookup.
    apply (keyed_lookup_in name_eqb name_eqb_eq) in Hlookup.
    vm_compute in Hlookup. destruct Hlookup as [[= <- <-] | []].
    eexists. vm_compute. reflexivity.
  - apply replace_equation_aliases_kinded.
    + eexists. vm_compute. reflexivity.
    + apply add_variable_definition_aliases_kinded.
      * vm_compute. reflexivity.
      * reflexivity.
      * apply create_aliases_kinded.
Qed.

Theorem example_params_premises : example_params_premises_statement.
Proof.
  split; [ exact example_params_env_fits | ].
  split.
  { intros ty [ <- | [ <- | [] ] ]; closed_type_fits. }
  split; [ exact (Forall_nil _) | ].
  split.
  { split; [ reflexivity | ]. split; [ | apply empty_respected ].
    intros n ty btm v Hlookup Hvalue.
    apply (keyed_lookup_in name_eqb name_eqb_eq) in Hlookup.
    vm_compute in Hlookup.
    repeat (destruct Hlookup as [Hlookup | Hlookup];
            [ injection Hlookup as <- <- <- | ]); [ .. | contradiction ].
    all: cbv in Hvalue; injection Hvalue as <-; right; open_type.
    all: try reflexivity.
    right. split; [ discriminate | ]. split; [ | exact I ].
    left. exists 3. repeat split. }
  constructor; [ | constructor; [ | constructor ] ].
  - exists (Value_const (Tagged_immediate 3)). split; [ reflexivity | ].
    open_type.
    right. split; [ discriminate | ]. split; [ | exact I ].
    left. exists 3. repeat split.
  - exists (Value_const (Naked_immediate 7)). split; [ reflexivity | ].
    open_type.
    exists 7. split; [ reflexivity | ].
    split; [ left; reflexivity | intros relation n [] ].
Qed.

Theorem example_params_answer : example_params_answer_statement.
Proof. eexists. split; [ vm_compute; reflexivity | reflexivity ]. Qed.

Theorem add_equations_on_params_not_sound
  : add_equations_on_params_not_sound_statement.
Proof.
  split; [ exact (naming_join_sound example_unknown_param) | ].
  intros Hsound.
  destruct example_params_premises
    as (Hfits & Htypes & Hstatics & Hsatisfied & Hvalues).
  destruct example_params_answer as (tenv' & Hanswer & Hbottom).
  destruct (Hsound empty_static_env example_params_valuation
              example_params_env tenv' example_params example_param_types
              Hfits Htypes Hstatics Hsatisfied Hvalues Hanswer)
    as (valuation' & _ & Hsatisfied' & _).
  rewrite (satisfied_not_bottom _ _ _ Hsatisfied') in Hbottom.
  discriminate Hbottom.
Qed.

(* ======================================================================== *)
(* The premise under which the contract is true                             *)
(* ======================================================================== *)

(* Making a variable an alias never answers bottom: only two different
   constants do. *)
Lemma add_alias_var_not_bottom (tenv : Typing_env.t) (x : variable)
    (c : simple) :
  Typing_env.add_alias tenv (Var x) c <> Some Bottom.
Proof.
  intros Hadd.
  set (v0 := Value_const (Tagged_immediate 0)).
  destruct c as [ y | sym | k ].
  - apply (add_alias_bottom [(x, v0); (y, v0)]) in Hadd. apply Hadd.
    cbn. rewrite Pos.eqb_refl.
    destruct (Pos.eqb y x); [ reflexivity | ].
    rewrite Pos.eqb_refl. reflexivity.
  - apply (add_alias_bottom [(x, Value_symbol sym)]) in Hadd. apply Hadd.
    cbn. rewrite Pos.eqb_refl. reflexivity.
  - apply (add_alias_bottom [(x, Value_const k)]) in Hadd. apply Hadd.
    cbn. rewrite Pos.eqb_refl. reflexivity.
Qed.

Theorem add_equation_on_unknown : add_equation_on_unknown_statement.
Proof.
  intros mh menv menv' x ty Hother Hfits Hty Hadding Hmem Hadd.
  unfold Meet_env.add_equation in Hadd.
  destruct (Meet_env.add_equation0 true menv (Name_var x) ty mh)
    as [ result | ] eqn:Hadd0; [ | discriminate Hadd ].
  assert (Hnot_adding :
    Keyed.among name_eqb (Name_var x)
      (Meet_env.adding_equations_for_names menv)
    = false) by (rewrite Hadding; reflexivity).
  destruct (add_equation0_unknown mh menv (Name_var x) ty result Hother Hfits
              Hty Hmem Hnot_adding Hadd0)
    as [ -> | (-> & alias & _ & Hbottom) ].
  - cbn in Hadd. injection Hadd as <-. reflexivity.
  - exfalso. exact (add_alias_var_not_bottom _ _ _ Hbottom).
Qed.

(* What is asked of a parameter and its type, in an environment and
   under a valuation: the type fits; if the environment knows the
   parameter, it has a value of the type; if not, it is to be
   avoided. *)
Definition param_ok (statics : static_env) (valuation : env)
    (menv : Meet_env.t) (param : variable * flambda_kind)
    (ty : flambda_type) : Prop :=
  type_fits (Meet_env.typing_env menv) ty
  ∧ (Typing_env.mem None (Meet_env.typing_env menv) (Name_var (fst param))
     = true ->
     ∃ v : value,
       lookup (fst param) valuation = Some v
       ∧ type_satisfied statics valuation v ty)
  ∧ (Typing_env.mem None (Meet_env.typing_env menv) (Name_var (fst param))
     = false ->
     among (fst param) (Meet_env.names_to_avoid menv) = true).

Lemma add_equations_on_params_avoided (mh : meet_expanded_head)
    (statics : static_env) :
  meet_expanded_head_keeps mh ->
  meet_expanded_head_sound mh ->
  meet_expanded_head_other_kinds mh ->
  statics_valid statics ->
  ∀ (params : list (variable * flambda_kind))
    (param_types : list flambda_type) (menv menv' : Meet_env.t)
    (valuation : env),
    typing_env_fits (Meet_env.typing_env menv) ->
    Meet_env.adding_equations_for_names menv = [] ->
    typing_env_satisfied_strictly statics valuation (Meet_env.typing_env menv)
    ->
    Forall2 (param_ok statics valuation menv) params param_types ->
    Meet_env.add_equations_on_params menv params param_types mh = Some menv' ->
    meet_env_added statics valuation menv menv'.
Proof.
  intros Hkeeps Hsound Hother Hstatics.
  induction params as [ | param params IH ];
    intros param_types menv menv' valuation Hfits Hadding Hsatisfied Hparams
      Hadd.
  - destruct param_types as [ | ty param_types ]; [ | discriminate Hadd ].
    injection Hadd as <-.
    exact (meet_env_added_refl statics valuation menv Hsatisfied).
  - destruct param_types as [ | ty param_types ]; [ discriminate Hadd | ].
    apply Forall2_cons_iff in Hparams.
    destruct Hparams as [(Hty & Hknown & Havoided) Hrest].
    cbn [Meet_env.add_equations_on_params] in Hadd.
    destruct (Meet_env.add_equation menv (Name_var (fst param)) ty mh)
      as [ menv1 | ] eqn:Hfirst; [ | discriminate Hadd ].
    cbv beta iota in Hadd.
    set (tenv := Meet_env.typing_env menv) in *.
    destruct (Typing_env.mem None tenv (Name_var (fst param))) eqn:Hmem.
    + (* The parameter is known: as where every parameter is. *)
      destruct (Hknown eq_refl) as (v & Hvalue & Hv).
      assert (Hkind :=
        known_name_of_kind statics valuation tenv (Name_var (fst param)) ty v
          Hsatisfied Hmem Hvalue Hv).
      destruct (add_equation_keeps mh menv menv1 (Name_var (fst param)) ty
                  Hkeeps Hfits Hty Hkind Hfirst)
        as [[Hextends1 Hfits1] Hnew1].
      destruct (meet_env_add_equation_sound mh statics valuation menv menv1
                  (Name_var (fst param)) ty v Hkeeps Hsound Hfits Hty Hkind
                  Hstatics Hsatisfied Hvalue Hv Hfirst)
        as (valuation1 & Hextended1 & Hsatisfied1 & Hvalued1).
      assert (Htenv := meet_env_extends_tenv _ _ Hextends1).
      apply (meet_env_added_trans statics valuation valuation1 menv menv1
               menv' (proj2 (proj2 (proj2 (proj2 (proj2 Htenv))))) Hextended1
               Hvalued1).
      apply (IH param_types menv1 menv' valuation1 Hfits1);
        [ | exact Hsatisfied1 | | exact Hadd ].
      * destruct Hextends1 as (_ & _ & _ & _ & _ & Hsame & _).
        rewrite Hsame. exact Hadding.
      * eapply Forall2_impl; [ | exact Hrest ].
        intros param' ty' (Hty' & Hknown' & Havoided').
        fold tenv in Hty', Hknown', Havoided'.
        assert (Hstays :
          Typing_env.mem None tenv (Name_var (fst param')) = false ->
          Typing_env.mem None (Meet_env.typing_env menv1)
            (Name_var (fst param'))
          = false).
        { intros Hunknown.
          destruct (Typing_env.mem None (Meet_env.typing_env menv1)
                      (Name_var (fst param'))) eqn:Hnow; [ | reflexivity ].
          destruct (Hnew1 (fst param') Hnow Hunknown) as [Hnot _].
          rewrite (Havoided' Hunknown) in Hnot. discriminate Hnot. }
        split; [ exact (tenv_extends_type_fits _ _ _ Htenv Hty') | ].
        split.
        -- intros Hnow.
           destruct (Typing_env.mem None tenv (Name_var (fst param')))
             eqn:Hbefore.
           ++ destruct (Hknown' eq_refl) as (v' & Hvalue' & Hv').
              exists v'. split.
              ** rewrite (Hextended1 (fst param') Hbefore). exact Hvalue'.
              ** exact (type_satisfied_extended statics _ _ _ v' ty' Hty'
                          Hextended1 Hv').
           ++ rewrite (Hstays eq_refl) in Hnow. discriminate Hnow.
        -- intros Hnow.
           destruct (Typing_env.mem None tenv (Name_var (fst param')))
             eqn:Hbefore.
           ++ rewrite (tenv_extends_mem _ _ _ Htenv Hbefore) in Hnow.
              discriminate Hnow.
           ++ destruct Hextends1 as (_ & _ & _ & _ & _ & _ & Havoid1 & _).
              exact (Havoid1 _ (Havoided' eq_refl)).
    + (* The parameter is not known: nothing is added. *)
      unfold Meet_env.add_equation in Hfirst.
      destruct (Meet_env.add_equation0 true menv (Name_var (fst param)) ty mh)
        as [ result | ] eqn:Hadd0; [ | discriminate Hfirst ].
      assert (Hnot_adding :
        Keyed.among name_eqb (Name_var (fst param))
          (Meet_env.adding_equations_for_names menv)
        = false) by (rewrite Hadding; reflexivity).
      destruct (add_equation0_unknown mh menv (Name_var (fst param)) ty result
                  Hother Hfits Hty Hmem Hnot_adding Hadd0)
        as [ -> | (-> & alias & _ & Hbottom) ].
      * cbn in Hfirst. injection Hfirst as <-.
        exact (IH param_types menv menv' valuation Hfits Hadding Hsatisfied
                 Hrest Hadd).
      * exfalso. exact (add_alias_var_not_bottom _ _ _ Hbottom).
Qed.

Theorem meet_env_add_equations_on_params_avoided
  : meet_env_add_equations_on_params_avoided_statement.
Proof.
  intros mh statics valuation menv menv' params param_types Hkeeps Hsound
    Hother Hfits Hadding Htypes Havoided Hstatics Hsatisfied Hvalues Hadd.
  apply (add_equations_on_params_avoided mh statics Hkeeps Hsound Hother
           Hstatics params param_types menv menv' valuation Hfits Hadding
           Hsatisfied);
    [ | exact Hadd ].
  clear Hadd. revert Htypes Havoided.
  induction Hvalues as [ | param ty params param_types Hvalue Hrest IH ];
    intros Htypes Havoided; [ constructor | ].
  constructor.
  - split; [ apply Htypes; left; reflexivity | ].
    split; [ intros _; exact Hvalue | ].
    exact (Havoided param (or_introl eq_refl)).
  - apply IH.
    + intros ty' Hin. apply Htypes. right. exact Hin.
    + intros param' Hin. apply Havoided. right. exact Hin.
Qed.

Theorem add_equations_on_params_avoiding_sound
  : add_equations_on_params_avoiding_sound_statement.
Proof.
  intros join fuel Hjoin statics valuation tenv tenv' params param_types
    Hfits Htypes Hstatics Hsatisfied Hvalues Hadd.
  unfold add_equations_on_params_avoiding in Hadd.
  set (mh := Meet.meet_expanded_head join repaired_choices fuel) in *.
  set (menv :=
    Meet_env.with_names_to_avoid (Meet_env.create tenv) (map fst params))
    in *.
  destruct (Meet_env.add_equations_on_params menv params param_types mh)
    as [ menv' | ] eqn:Hadd'; [ | discriminate Hadd ].
  cbn in Hadd. injection Hadd as <-.
  destruct (meet_to_prove_expanded_heads join fuel Hjoin) as [Hkeeps Hsound].
  apply (meet_env_add_equations_on_params_avoided mh statics valuation menv
           menv' params param_types Hkeeps Hsound
           (meet_expanded_head_fuel_other_kinds join repaired_choices fuel)
           Hfits eq_refl Htypes);
    [ | exact Hstatics | exact Hsatisfied | exact Hvalues | exact Hadd' ].
  intros param Hin _.
  cbn [Meet_env.names_to_avoid Meet_env.with_names_to_avoid menv].
  unfold among. apply existsb_exists. exists (fst param).
  split; [ exact (in_map fst params param Hin) | apply Pos.eqb_refl ].
Qed.

Theorem example_params_avoiding_answer
  : example_params_avoiding_answer_statement.
Proof. vm_compute. reflexivity. Qed.
