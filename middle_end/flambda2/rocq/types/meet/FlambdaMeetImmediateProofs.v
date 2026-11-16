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

(* Proofs of the statements of FlambdaMeetImmediateSpec.v. Nothing here
   needs review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaClosuresSpec
  FlambdaPrimitiveProofs.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeMeaningSpec FlambdaTypeMeaningProofs
  FlambdaTypeWellFormedSpec FlambdaTypeWellFormedProofs
  FlambdaBindingTime FlambdaEnvAliases FlambdaEnvAliasesSpec
  FlambdaEnvAliasesProofs FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaTypingEnvProofs FlambdaTypingEnvInvariantSpec
  FlambdaTypingEnvInvariantProofs FlambdaTypingEnvOperationsSpec
  FlambdaTypingEnvOperationsProofs FlambdaTypingEnvKindedSpec
  FlambdaTypingEnvKindedProofs FlambdaMeetGrammar FlambdaExpandHead
  FlambdaExpandHeadSpec FlambdaExpandHeadProofs
  FlambdaMeetEnv FlambdaMeet FlambdaMeetSpec FlambdaMeetProofs
  FlambdaMeetStepsSpec FlambdaMeetStepsProofs FlambdaMeetEnvSpec
  FlambdaMeetEnvProofs FlambdaMeetImmediateSpec.
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
(* Two types of different kinds                                             *)
(* ======================================================================== *)

Lemma meet_expanded_head_fuel_other_kinds_for :
  for_validity meet_expanded_head_fuel_other_kinds_statement.
Proof.
  intros join choices fuel menv menv' expanded1 expanded2 r Hkinds Hmeet.
  destruct fuel as [ | fuel ]; [ discriminate Hmeet | ].
  cbn [Meet.meet_expanded_head] in Hmeet.
  unfold meet_expanded_head_step, meet_or_unknown_or_bottom in Hmeet.
  unfold Expanded_type.to_type in Hkinds.
  destruct (Expanded_type.descr expanded1) as [ | head1 | ],
    (Expanded_type.descr expanded2) as [ | head2 | ];
    try discriminate Hmeet;
    try (injection Hmeet as _ Hmenv; symmetry; exact Hmenv).
  exfalso.
  destruct head1 as [ head1 | head1 | k1 head1 | head1 | head1 ],
    head2 as [ head2 | head2 | k2 head2 | head2 | head2 ];
    cbn in Hmeet; try discriminate Hmeet;
    try (apply Hkinds; reflexivity).
  destruct k1, k2; cbn in Hmeet; try discriminate Hmeet;
    apply Hkinds; reflexivity.
Qed.

(* ======================================================================== *)
(* A head that fits                                                         *)
(* ======================================================================== *)

(* The names of the relations that are variables are known. *)
Definition relations_known (tenv : Typing_env.t)
    (relations : list (Relation.t * name)) : Prop :=
  ∀ (relation : Relation.t) (x : variable),
    (relation, Name_var x) ∈ relations ->
    Typing_env.mem None tenv (Name_var x) = true.

Lemma head_fits_iff (tenv : Typing_env.t)
    (head : head_of_kind_naked_immediate) :
  type_fits tenv (naked_immediate_head_type head)
  <-> head_of_kind_naked_immediate_well_formed head = true
      ∧ relations_known tenv (inverse_relations head).
Proof.
  unfold type_fits, naked_immediate_head_type.
  cbn [type_well_formed descr_well_formed free_variables
       free_variables_of_descr].
  split.
  - intros (Hwf & Hclosed & _ & _). split; [ exact Hwf | ].
    intros relation x Hin. apply Hclosed.
    unfold free_variables_of_naked_immediate. apply in_flat_map.
    exists (relation, Name_var x). split; [ exact Hin | ].
    left. reflexivity.
  - intros [Hwf Hknown]. split; [ exact Hwf | ]. split.
    { intros x Hin. unfold free_variables_of_naked_immediate in Hin.
      apply in_flat_map in Hin. destruct Hin as ([relation n] & Hin & Hx).
      destruct n as [ y | sym ]; cbn in Hx; [ | destruct Hx ].
      destruct Hx as [ -> | [] ]. exact (Hknown relation x Hin). }
    split; intros ? ? Hin; cbn in Hin; destruct Hin.
Qed.

Lemma relations_known_extended (tenv tenv' : Typing_env.t)
    (relations : list (Relation.t * name)) :
  tenv_extends tenv tenv' ->
  relations_known tenv relations ->
  relations_known tenv' relations.
Proof.
  intros Hextends Hknown relation x Hin.
  exact (tenv_extends_mem _ _ _ Hextends (Hknown relation x Hin)).
Qed.

Lemma in_dedup_relations (inverse : Relation.t * name)
    (relations : list (Relation.t * name)) :
  inverse ∈ Keyed.dedup inverse_relation_eqb relations <-> inverse ∈ relations.
Proof.
  rewrite <- (keyed_among_in _ inverse_relation_eqb inverse_relation_eqb_eq),
    (keyed_among_dedup _ inverse_relation_eqb inverse_relation_eqb_eq).
  apply (keyed_among_in _ inverse_relation_eqb inverse_relation_eqb_eq).
Qed.

(* The canonical element of a name, if it is a variable, is known: it
   is the name, or the name was demoted to it. *)
Lemma canonical_variable_known (tenv : Typing_env.t) (n : name)
    (x : variable) :
  typing_env_invariant tenv ->
  (∀ y : variable,
     n = Name_var y -> Typing_env.mem None tenv (Name_var y) = true) ->
  Typing_env.get_canonical_simple_ignoring_name_mode tenv (simple_of_name n)
  = Var x ->
  Typing_env.mem None tenv (Name_var x) = true.
Proof.
  intros Hinvariant Hknown Hcanonical.
  rewrite get_canonical_simple_ignoring_name_mode_eq in Hcanonical.
  destruct (canonical_cases (Typing_env.aliases tenv) (simple_of_name n))
    as [ [_ Hself] | (n' & canonical_element & _ & Hdemoted & _ & Hof) ].
  - rewrite Hself in Hcanonical.
    destruct n as [ y | sym ]; [ | discriminate Hcanonical ].
    cbn in Hcanonical. injection Hcanonical as ->. apply Hknown. reflexivity.
  - rewrite Hof in Hcanonical. subst canonical_element.
    exact (proj2 (invariant_aliases_known tenv Hinvariant n' (Var x)
                    Hdemoted)).
Qed.

Lemma reduce_head_of_kind_naked_immediate_fits_for :
  for_validity reduce_head_of_kind_naked_immediate_fits_statement.
Proof.
  intros tenv head head' Hfits Hhead Hreduce.
  assert (Hinvariant : typing_env_invariant tenv) by exact (proj1 Hfits).
  apply head_fits_iff in Hhead. destruct Hhead as [Hwf Hknown].
  apply head_fits_iff.
  unfold reduce_head_of_kind_naked_immediate in Hreduce.
  set (step := fun (state : Or_bottom.t reduce_acc)
                   (inverse : Relation.t * name) => _) in Hreduce.
  assert (Hfold :
    ∀ (relations : list (Relation.t * name)) (state : Or_bottom.t reduce_acc)
      (acc' : reduce_acc),
      relations_known tenv relations ->
      match state with
      | Or_bottom.Bottom => True
      | Or_bottom.Ok acc => relations_known tenv (reduce_relations acc)
      end ->
      List.fold_left step relations state = Or_bottom.Ok acc' ->
      relations_known tenv (reduce_relations acc')).
  { induction relations as [ | [relation n] relations IH ];
      intros state acc' Hrelations Hstate Hfolded.
    - cbn in Hfolded. subst state. exact Hstate.
    - cbn [List.fold_left] in Hfolded.
      apply (IH _ acc') in Hfolded; [ exact Hfolded | | ].
      { intros r x Hin. apply (Hrelations r x). right. exact Hin. }
      destruct state as [ acc | ]; [ | exact I ].
      unfold step. cbn [fst snd].
      destruct (Typing_env.get_canonical_simple_ignoring_name_mode tenv
                  (simple_of_name n)) as [ x | sym | c ] eqn:Hcanonical.
      + cbn [reduce_relations].
        intros r y [ Heq | Hin ]; [ | exact (Hstate r y Hin) ].
        injection Heq as _ Hy. subst y.
        apply (canonical_variable_known tenv n x Hinvariant);
          [ | exact Hcanonical ].
        intros y ->. apply (Hrelations relation y). left. reflexivity.
      + cbn [reduce_relations].
        intros r y [ Heq | Hin ]; [ discriminate Heq | exact (Hstate r y Hin) ].
      + destruct (Relation.of_const relation c) as [ imm | ];
          [ | exact Hstate ].
        destruct (reduce_imms acc) as [ imms | ]; [ | exact Hstate ].
        destruct (Keyed.same_keys_b Z.eqb [imm] imms); [ exact Hstate | ].
        destruct (negb (Keyed.among Z.eqb imm imms));
          [ exact I | exact Hstate ]. }
  destruct (List.fold_left step (inverse_relations head) _)
    as [ acc | ] eqn:Hfolded; [ | discriminate Hreduce ].
  apply Hfold in Hfolded; [ | exact Hknown | intros r x [] ].
  destruct (negb (reduce_changed acc)).
  - injection Hreduce as <-. split; assumption.
  - unfold head_of_kind_naked_immediate_from_descr in Hreduce.
    destruct (reduce_imms acc) as [ [ | number numbers ] | ];
      [ discriminate Hreduce | | ];
      injection Hreduce as <-; (split; [ reflexivity | ]);
      cbn [inverse_relations]; intros r x Hin; apply (Hfolded r x);
      apply in_rev; apply in_dedup_relations; exact Hin.
Qed.

(* ======================================================================== *)
(* The numbers                                                              *)
(* ======================================================================== *)

(* A number is one of the numbers that a head has, if they are
   known. *)
Definition numbers_have (numbers : Or_unknown.t (list Z)) (i : Z) : Prop :=
  match numbers with
  | Or_unknown.Unknown => True
  | Or_unknown.Known numbers => i ∈ numbers
  end.

(* The meet of the numbers of two heads, as
   [meet_head_of_kind_naked_immediate] makes it. *)
Definition meet_numbers (env : Meet_env.t)
    (numbers1 numbers2 : Or_unknown.t (list Z))
    : option (Meet_result.t (Or_unknown.t (list Z))) :=
  meet_unknown
    (fun env s1 s2 => Some (set_meet Z.eqb (fun s => s) env s1 s2))
    Keyed.is_empty env numbers1 numbers2.

Lemma meet_numbers_cases (env : Meet_env.t)
    (numbers1 numbers2 : Or_unknown.t (list Z)) :
  ∃ result : Meet_result.t (Or_unknown.t (list Z)),
    meet_numbers env numbers1 numbers2 = Some result
    ∧ match result with
      | Meet_result.Bottom _ =>
        ∀ i : Z, ~ (numbers_have numbers1 i ∧ numbers_have numbers2 i)
      | Meet_result.Ok r env' =>
        env' = env
        ∧ ∀ i : Z,
            numbers_have (extract_value r numbers1 numbers2) i
            <-> numbers_have numbers1 i ∧ numbers_have numbers2 i
      end.
Proof.
  unfold meet_numbers, meet_unknown.
  destruct numbers1 as [ [ | i1 s1 ] | ], numbers2 as [ [ | i2 s2 ] | ];
    cbn [Keyed.is_empty andb];
    try (eexists; split; [ reflexivity | ]; cbn;
         intros i [H1 H2]; (exact H1 || exact H2)).
  - cbn [option_map].
    assert (Hcases :=
      set_meet_cases_for Z (list Z) Z.eqb (fun s => s) env (i1 :: s1) (i2 :: s2)
        z_eqb_decides_equality).
    destruct (set_meet Z.eqb (fun s => s) env (i1 :: s1) (i2 :: s2))
      as [ r | r env' ].
    + eexists. split; [ reflexivity | ]. exact Hcases.
    + destruct Hcases as (-> & s & Hs & Hboth & _).
      exists (Meet_result.Ok (map_return_value Or_unknown.Known r) env).
      split; [ destruct r; reflexivity | ].
      split; [ reflexivity | ]. intros i.
      rewrite (extract_value_map Or_unknown.Known r), Hs. exact (Hboth i).
  - eexists. split; [ reflexivity | ]. split; [ reflexivity | ].
    intros i. cbn. tauto.
  - eexists. split; [ reflexivity | ]. split; [ reflexivity | ].
    intros i. cbn. tauto.
  - eexists. split; [ reflexivity | ]. split; [ reflexivity | ].
    intros i. cbn. tauto.
Qed.

(* ======================================================================== *)
(* An equation on a name that may be of another kind                        *)
(* ======================================================================== *)

(* A simple is of the kind of its canonical element. *)
Lemma kind_of_canonical (tenv : Typing_env.t) (s : simple)
    (kind : flambda_kind) :
  typing_env_invariant tenv ->
  aliases_kinded tenv ->
  simple_of_kind tenv
    (Typing_env.get_canonical_simple_ignoring_name_mode tenv s) kind ->
  simple_of_kind tenv s kind.
Proof.
  intros Hinvariant Hkinded Hkind.
  assert (Heq := get_canonical_simple_ignoring_name_mode_eq tenv s).
  destruct (canonical_cases (Typing_env.aliases tenv) s)
    as [ [_ Hself] | (n & canonical_element & -> & Hdemoted & _ & Hof) ].
  - rewrite Heq, Hself in Hkind. exact Hkind.
  - destruct (invariant_aliases_known tenv Hinvariant n canonical_element
                Hdemoted) as [Hmem _].
    rewrite <- mem_simple_of_name in Hmem.
    destruct (mem_simple_of_some_kind tenv _ Hmem) as [kind' Hkind'].
    destruct (canonical_of_kind tenv _ kind' Hinvariant Hkinded Hkind')
      as [_ Hcanonical].
    rewrite (simple_of_kind_unique tenv _ kind kind' Hkind Hcanonical).
    exact Hkind'.
Qed.

Section Other_kinds.

Variable mh : meet_expanded_head.
Hypothesis Hkeeps : meet_expanded_head_keeps mh.
Hypothesis Hother : meet_expanded_head_other_kinds mh.

(* Where an equation whose type is not an alias is added to a name:
   the name is of the kind of the type; or nothing is added; or the
   answer is bottom, and the canonical element of the name is a
   constant of another kind. *)
Lemma add_equation0_kind_cases (menv : Meet_env.t) (n : name)
    (ty : flambda_type) (result : Or_bottom.t Meet_env.t) :
  typing_env_fits (Meet_env.typing_env menv) ->
  get_alias_opt ty = None ->
  Meet_env.add_equation0 true menv n ty mh = Some result ->
  simple_of_kind (Meet_env.typing_env menv) (simple_of_name n) (kind ty)
  ∨ result = Or_bottom.Ok menv
  ∨ (result = Or_bottom.Bottom
     ∧ ∃ c : const,
         Typing_env.get_canonical_simple_ignoring_name_mode
           (Meet_env.typing_env menv) (simple_of_name n)
         = Const c
         ∧ kind_of_const c <> kind ty).
Proof.
  intros Hfits Halias Hadd.
  assert (Hinvariant := proj1 Hfits).
  assert (Hkinded := proj1 (proj2 (proj1 (proj2 (proj2 Hfits))))).
  set (tenv := Meet_env.typing_env menv) in *.
  unfold Meet_env.add_equation0, Meet_env.add_equation_on_simple0 in Hadd.
  rewrite Halias in Hadd.
  destruct (Expanded_type.of_non_alias_type ty) as [ expanded | ] eqn:Hexpanded;
    [ | discriminate Hadd ].
  cbv beta iota in Hadd.
  destruct (of_non_alias_type_kind ty expanded Hexpanded)
    as [Hkind_expanded Hkind_to_type].
  unfold Meet_env.add_concrete_equation_on_simple,
    Meet_env.add_or_delay_concrete_equation_on_canonical in Hadd.
  fold tenv in Hadd.
  assert (Hname :
    ∀ n' : name,
      Typing_env.get_canonical_simple_ignoring_name_mode tenv
        (simple_of_name n)
      = simple_of_name n' ->
      Meet_env.add_or_delay_concrete_equation_on_canonical_name true menv n'
        expanded mh
      = Some result ->
      simple_of_kind tenv (simple_of_name n) (kind ty)
      ∨ result = Or_bottom.Ok menv).
  { intros n' Hcanonical Hdelay.
    unfold Meet_env.add_or_delay_concrete_equation_on_canonical_name in Hdelay.
    destruct (Keyed.among name_eqb n'
                (Meet_env.adding_equations_for_names menv)).
    { right. injection Hdelay as <-. reflexivity. }
    left.
    unfold Meet_env.adding_equation_for_name,
      Meet_env.add_concrete_equation_on_canonical_name in Hdelay.
    cbn [Meet_env.typing_env] in Hdelay. fold tenv in Hdelay.
    destruct (Typing_env.find tenv n' (Some (Expanded_type.kind expanded)))
      as [ existing | ] eqn:Hfind; [ | discriminate Hdelay ].
    apply (kind_of_canonical tenv _ _ Hinvariant Hkinded).
    rewrite Hcanonical, <- Hkind_expanded.
    exact (simple_of_kind_of_find tenv n' _ existing Hfind). }
  destruct (Typing_env.get_canonical_simple_ignoring_name_mode tenv
              (simple_of_name n)) as [ x | sym | c ] eqn:Hcanonical.
  - destruct (Hname (Name_var x) eq_refl Hadd) as [ H | H ]; auto.
  - destruct (Hname (Name_symbol sym) eq_refl Hadd) as [ H | H ]; auto.
  - unfold Meet_env.add_concrete_equation_on_const in Hadd.
    destruct (kind_eqb (kind_of_const c) (kind ty)) eqn:Hkinds.
    { left. apply kind_eqb_eq in Hkinds.
      apply (kind_of_canonical tenv _ _ Hinvariant Hkinded).
      rewrite Hcanonical. cbn. rewrite kind_type_for_const. exact Hkinds. }
    assert (Hdiffer : kind_of_const c <> kind ty).
    { intros Heq. apply kind_eqb_eq in Heq. rewrite Heq in Hkinds.
      discriminate Hkinds. }
    right.
    destruct (mh menv expanded (Expanded_type.create_const c))
      as [ [ r | r menv' ] | ] eqn:Hmeet; [ | | discriminate Hadd ].
    + right. injection Hadd as <-. split; [ reflexivity | ].
      exists c. split; [ reflexivity | exact Hdiffer ].
    + left. injection Hadd as <-. f_equal.
      apply (Hother menv menv' expanded (Expanded_type.create_const c) r);
        [ | exact Hmeet ].
      rewrite Hkind_to_type, (proj2 (create_const_kind c)),
        kind_type_for_const.
      intros Heq. apply Hdiffer. symmetry. exact Heq.
Qed.

(* [Meet_env.add_equation_strict] of a type that is not an alias keeps
   what it is to keep, whatever the kind of the name. *)
Lemma add_equation_strict_keeps_any_kind (menv menv' : Meet_env.t) (n : name)
    (ty : flambda_type) :
  typing_env_fits (Meet_env.typing_env menv) ->
  type_fits (Meet_env.typing_env menv) ty ->
  get_alias_opt ty = None ->
  Meet_env.add_equation_strict menv n ty mh = Some (Or_bottom.Ok menv') ->
  meet_env_kept_by_meet menv menv'.
Proof.
  intros Hfits Hty Halias Hadd.
  assert (Hadd0 := Hadd). unfold Meet_env.add_equation_strict in Hadd0.
  destruct (Typing_env.is_bottom (Meet_env.typing_env menv));
    [ discriminate Hadd0 | ].
  destruct (add_equation0_kind_cases menv n ty _ Hfits Halias Hadd0)
    as [ Hkind | [ Heq | [Heq _] ] ].
  - exact (add_equation_strict_keeps_for mh menv menv' n ty Hkeeps Hfits Hty
             Hkind
             Hadd).
  - injection Heq as ->. apply meet_env_steps_kept.
    split; [ apply meet_env_steps_refl | exact Hfits ].
  - discriminate Heq.
Qed.

(* And it loses no value. *)
Lemma add_equation_strict_sound_any_kind (statics : static_env)
    (valuation : env) (menv : Meet_env.t) (n : name) (ty : flambda_type)
    (v : value) (result : Or_bottom.t Meet_env.t) :
  meet_expanded_head_sound mh ->
  typing_env_fits (Meet_env.typing_env menv) ->
  type_fits (Meet_env.typing_env menv) ty ->
  get_alias_opt ty = None ->
  statics_valid statics ->
  typing_env_satisfied_strictly statics valuation (Meet_env.typing_env menv) ->
  value_of_name valuation n = Some v ->
  type_satisfied statics valuation v ty ->
  Meet_env.add_equation_strict menv n ty mh = Some result ->
  meet_env_sound statics valuation menv result.
Proof.
  intros Hsound Hfits Hty Halias Hstatics Hsatisfied Hvalue Hv Hadd.
  assert (Hadd0 := Hadd). unfold Meet_env.add_equation_strict in Hadd0.
  rewrite (satisfied_not_bottom statics valuation _ Hsatisfied) in Hadd0.
  destruct (add_equation0_kind_cases menv n ty _ Hfits Halias Hadd0)
    as [ Hkind | [ -> | (_ & c & Hcanonical & Hdiffer) ] ].
  - exact (meet_env_add_equation_strict_sound_for mh statics valuation menv n ty
             v
             result Hkeeps Hsound Hfits Hty Hkind Hstatics Hsatisfied Hvalue
             Hv Hadd).
  - exists menv. split; [ reflexivity | ].
    exact (meet_env_added_refl statics valuation menv Hsatisfied).
  - exfalso. apply Hdiffer.
    assert (Hvalue_canonical :=
      canonical_ignoring_name_mode_satisfied (fun _ => False) statics
        valuation (Meet_env.typing_env menv) (simple_of_name n) Hsatisfied).
    rewrite Hcanonical in Hvalue_canonical. unfold value_of_name in Hvalue.
    rewrite Hvalue in Hvalue_canonical. cbn in Hvalue_canonical.
    injection Hvalue_canonical as <-.
    exact (type_satisfied_kind statics valuation _ ty Hv).
Qed.

End Other_kinds.

(* ======================================================================== *)
(* What the numbers say of the name of a relation                           *)
(* ======================================================================== *)

(* A type that has no name fits every environment. *)
Lemma closed_type_fits (tenv : Typing_env.t) (ty : flambda_type) :
  type_well_formed ty = true ->
  free_variables ty = [] ->
  aliases_with_kinds ty = [] ->
  equation_names_with_kinds ty = [] ->
  type_fits tenv ty.
Proof.
  intros Hwf Hfree Haliases Hequations. split; [ exact Hwf | ]. split.
  { intros x Hin. rewrite Hfree in Hin. destruct Hin. }
  split.
  - intros s kind Hin. rewrite Haliases in Hin. destruct Hin.
  - intros n kind Hin. rewrite Hequations in Hin. destruct Hin.
Qed.

Lemma unknown_cases_closed {A : Type} (f : row_like_block_case -> list A)
    (tags : list Z) :
  flat_map
    (fun case : Z * Or_unknown.t row_like_block_case =>
       match snd case with
       | Or_unknown.Known case => f case
       | Or_unknown.Unknown => []
       end)
    (map (fun tag => (tag, Or_unknown.Unknown)) tags)
  = [].
Proof. induction tags as [ | tag tags IH ]; [ reflexivity | exact IH ]. Qed.

Lemma immediate_or_blocks_with_these_tags_fits (tenv : Typing_env.t)
    (tags : list Z) (alloc_mode : Alloc_mode.t) :
  Keyed.distinct Z.eqb tags = true ->
  type_fits tenv (immediate_or_blocks_with_these_tags tags alloc_mode).
Proof.
  intros Hdistinct. apply closed_type_fits.
  - unfold immediate_or_blocks_with_these_tags.
    apply (create_variant_well_formed false Or_unknown.Unknown
             (Or_unknown.Known
                (Row_like_for_blocks
                   (map (fun tag => (tag, Or_unknown.Unknown)) tags)
                   Or_bottom.Bottom alloc_mode))
             No_extensions).
    split; [ exact I | ]. split; [ | exact I ].
    apply (row_like_for_blocks_well_formed_iff _ _ _).
    rewrite keys_of_unknown_cases.
    split; [ exact Hdistinct | ]. split; [ | exact I ].
    intros tag case. rewrite lookup_unknown_cases.
    destruct (existsb (Z.eqb tag) tags); discriminate.
  - cbn. rewrite unknown_cases_closed. reflexivity.
  - cbn. rewrite unknown_cases_closed. reflexivity.
  - cbn. rewrite unknown_cases_closed. reflexivity.
Qed.

(* What [reduce_inverse_relations] does for one relation, given the
   numbers: nothing, bottom, or an equation on the name. *)
Inductive relation_fact : Type :=
  | Fact_none
  | Fact_bottom
  | Fact_type (ty : flambda_type).

Definition fact_of_relation (choices : meet_choices) (imms : list Z)
    (relation : Relation.t) : relation_fact :=
  match relation with
  | Relation.Is_null =>
    match Keyed.among Z.eqb 0%Z imms, Keyed.among Z.eqb 1%Z imms with
    | false, false => Fact_bottom
    | true, true => Fact_none
    | true, false => Fact_type any_non_null_value
    | false, true => Fact_type null
    end
  | Relation.Is_int =>
    match Keyed.among Z.eqb 0%Z imms, Keyed.among Z.eqb 1%Z imms with
    | false, false => Fact_bottom
    | true, true => Fact_none
    | true, false => Fact_type any_block
    | false, true => Fact_type any_tagged_immediate
    end
  | Relation.Get_tag =>
    match
      type_of_get_tag_scrutinee_or_unknown choices
        (Keyed.dedup Z.eqb (filter is_tag imms)) Alloc_mode.Heap_or_local
    with
    | Or_unknown.Known shape => Fact_type shape
    | Or_unknown.Unknown => Fact_none
    end
  end.

Section Facts.

Variable choices : meet_choices.
Hypothesis Hchoice :
  type_of_get_tag_scrutinee choices = immediate_or_blocks_with_these_tags.

(* The type of a fact fits every environment and is not an alias. *)
Lemma fact_type_fits (tenv : Typing_env.t) (imms : list Z)
    (relation : Relation.t) (ty : flambda_type) :
  fact_of_relation choices imms relation = Fact_type ty ->
  type_fits tenv ty ∧ get_alias_opt ty = None.
Proof.
  unfold fact_of_relation. intros Hfact.
  destruct relation.
  - destruct (Keyed.among Z.eqb 0%Z imms), (Keyed.among Z.eqb 1%Z imms);
      try discriminate Hfact; injection Hfact as <-;
      (split; [ apply closed_type_fits | ]); reflexivity.
  - destruct (Keyed.among Z.eqb 0%Z imms), (Keyed.among Z.eqb 1%Z imms);
      try discriminate Hfact; injection Hfact as <-;
      (split; [ apply closed_type_fits | ]); reflexivity.
  - unfold type_of_get_tag_scrutinee_or_unknown in Hfact.
    rewrite Hchoice in Hfact.
    destruct (forallb is_structured_block
                (Keyed.dedup Z.eqb (filter is_tag imms)));
      [ | discriminate Hfact ].
    injection Hfact as <-. split; [ | reflexivity ].
    apply immediate_or_blocks_with_these_tags_fits.
    exact (keyed_dedup_distinct Z Z.eqb z_eqb_decides_equality _).
Qed.

(* Where the relation relates a value to one of the numbers, the fact
   is not bottom, and its type is true of the value. *)
Lemma fact_true (statics : static_env) (valuation : env) (imms : list Z)
    (relation : Relation.t) (w : value) (i : Z) :
  relation_holds statics relation w i ->
  i ∈ imms ->
  match fact_of_relation choices imms relation with
  | Fact_none => True
  | Fact_bottom => False
  | Fact_type ty => type_satisfied statics valuation w ty
  end.
Proof.
  intros Hholds Hin.
  assert (Hamong : Keyed.among Z.eqb i imms = true)
    by (apply (keyed_among_in Z Z.eqb z_eqb_decides_equality); exact Hin).
  unfold fact_of_relation. destruct relation.
  - destruct (relation_is_null statics valuation w i Hholds)
      as [ [-> Hw] | [-> Hw] ]; rewrite Hamong.
    + destruct (Keyed.among Z.eqb 0%Z imms); [ exact I | exact Hw ].
    + destruct (Keyed.among Z.eqb 1%Z imms); [ exact I | exact Hw ].
  - destruct (relation_is_int statics valuation w i Hholds)
      as [ [-> Hw] | [-> Hw] ]; rewrite Hamong.
    + destruct (Keyed.among Z.eqb 0%Z imms); [ exact I | exact Hw ].
    + destruct (Keyed.among Z.eqb 1%Z imms); [ exact I | exact Hw ].
  - unfold type_of_get_tag_scrutinee_or_unknown. rewrite Hchoice.
    destruct (forallb is_structured_block
                (Keyed.dedup Z.eqb (filter is_tag imms)));
      [ | exact I ].
    exact (get_tag_reduction_repaired statics valuation w i imms _ Hholds Hin).
Qed.

End Facts.

(* ======================================================================== *)
(* [reduce_inverse_relations]                                               *)
(* ======================================================================== *)

Section Reduce.

Variable choices : meet_choices.
Hypothesis Hchoice :
  type_of_get_tag_scrutinee choices = immediate_or_blocks_with_these_tags.
Variable mh : meet_expanded_head.
Hypothesis Hkeeps : meet_expanded_head_keeps mh.
Hypothesis Hother : meet_expanded_head_other_kinds mh.

(* The fold of [reduce_inverse_relations], by the facts. *)
Fixpoint add_facts (imms : list Z) (relations : list (Relation.t * name))
    (menv : Meet_env.t) : option (Or_bottom.t Meet_env.t) :=
  match relations with
  | [] => Some (Or_bottom.Ok menv)
  | (relation, n) :: relations =>
    match fact_of_relation choices imms relation with
    | Fact_none => add_facts imms relations menv
    | Fact_bottom => Some Or_bottom.Bottom
    | Fact_type ty =>
      match Meet_env.add_equation_strict menv n ty mh with
      | None => None
      | Some Or_bottom.Bottom => Some Or_bottom.Bottom
      | Some (Or_bottom.Ok menv) => add_facts imms relations menv
      end
    end
  end.

Lemma reduce_inverse_relations_eq (menv : Meet_env.t)
    (numbers : Or_unknown.t (list Z))
    (relations : list (Relation.t * name)) :
  reduce_inverse_relations choices mh menv numbers relations
  = match numbers with
    | Or_unknown.Unknown => Some (Or_bottom.Ok menv)
    | Or_unknown.Known imms => add_facts imms relations menv
    end.
Proof.
  unfold reduce_inverse_relations.
  destruct numbers as [ imms | ]; [ | reflexivity ].
  set (step := fun (state : option (Or_bottom.t Meet_env.t))
                   (inverse : Relation.t * name) => _).
  assert (Hnone :
    ∀ relations : list (Relation.t * name),
      List.fold_left step relations None = None).
  { intros l. induction l as [ | inverse l IH ]; [ reflexivity | exact IH ]. }
  assert (Hbottom :
    ∀ relations : list (Relation.t * name),
      List.fold_left step relations (Some Or_bottom.Bottom)
      = Some Or_bottom.Bottom).
  { intros l. induction l as [ | inverse l IH ]; [ reflexivity | exact IH ]. }
  revert menv.
  induction relations as [ | [relation n] relations IH ]; intros menv;
    [ reflexivity | ].
  cbn [List.fold_left add_facts].
  assert (Hstep :
    step (Some (Or_bottom.Ok menv)) (relation, n)
    = match fact_of_relation choices imms relation with
      | Fact_none => Some (Or_bottom.Ok menv)
      | Fact_bottom => Some Or_bottom.Bottom
      | Fact_type ty => Meet_env.add_equation_strict menv n ty mh
      end).
  { unfold step, fact_of_relation. cbn [fst snd]. destruct relation.
    - destruct (Keyed.among Z.eqb 0%Z imms), (Keyed.among Z.eqb 1%Z imms);
        reflexivity.
    - destruct (Keyed.among Z.eqb 0%Z imms), (Keyed.among Z.eqb 1%Z imms);
        reflexivity.
    - destruct (type_of_get_tag_scrutinee_or_unknown choices _ _);
        reflexivity. }
  rewrite Hstep.
  destruct (fact_of_relation choices imms relation) as [ | | ty ].
  - apply IH.
  - apply Hbottom.
  - destruct (Meet_env.add_equation_strict menv n ty mh)
      as [ [ menv1 | ] | ]; [ apply IH | apply Hbottom | apply Hnone ].
Qed.

(* What the fold keeps. *)
Lemma add_facts_keeps (imms : list Z) :
  ∀ (relations : list (Relation.t * name)) (menv menv' : Meet_env.t),
    typing_env_fits (Meet_env.typing_env menv) ->
    add_facts imms relations menv = Some (Or_bottom.Ok menv') ->
    meet_env_steps menv menv' ∧ typing_env_fits (Meet_env.typing_env menv').
Proof.
  induction relations as [ | [relation n] relations IH ];
    intros menv menv' Hfits Hadd.
  - injection Hadd as <-. split; [ apply meet_env_steps_refl | exact Hfits ].
  - cbn [add_facts] in Hadd.
    destruct (fact_of_relation choices imms relation) as [ | | ty ] eqn:Hfact.
    + exact (IH menv menv' Hfits Hadd).
    + discriminate Hadd.
    + destruct (Meet_env.add_equation_strict menv n ty mh)
        as [ [ menv1 | ] | ] eqn:Hstrict; try discriminate Hadd.
      destruct (fact_type_fits choices Hchoice (Meet_env.typing_env menv) imms
                  relation ty Hfact) as [Hty Halias].
      destruct (add_equation_strict_keeps_any_kind mh Hkeeps Hother menv menv1
                  n ty Hfits Hty Halias Hstrict)
        as [[Hextends1 Hfits1] Hnew1].
      destruct (IH menv1 menv' Hfits1 Hadd) as [Hsteps Hfits'].
      split; [ | exact Hfits' ].
      exact (meet_env_steps_trans _ _ _ (conj Hextends1 Hnew1) Hsteps).
Qed.

(* And it loses no value: under a valuation that satisfies the
   environment, where each relation relates the value of its name to
   a number that is among the numbers. *)
Lemma add_facts_sound (statics : static_env) (imms : list Z) (i : Z) :
  meet_expanded_head_sound mh ->
  statics_valid statics ->
  i ∈ imms ->
  ∀ (relations : list (Relation.t * name)) (menv : Meet_env.t)
    (valuation : env) (result : Or_bottom.t Meet_env.t),
    typing_env_fits (Meet_env.typing_env menv) ->
    typing_env_satisfied_strictly statics valuation (Meet_env.typing_env menv)
    ->
    relations_known (Meet_env.typing_env menv) relations ->
    (∀ (relation : Relation.t) (n : name),
       (relation, n) ∈ relations ->
       ∃ w : value,
         value_of_name valuation n = Some w
         ∧ relation_holds statics relation w i) ->
    add_facts imms relations menv = Some result ->
    meet_env_sound statics valuation menv result.
Proof.
  intros Hsound Hstatics Hin.
  induction relations as [ | [relation n] relations IH ];
    intros menv valuation result Hfits Hsatisfied Hknown Hholds Hadd.
  - injection Hadd as <-. exists menv. split; [ reflexivity | ].
    exact (meet_env_added_refl statics valuation menv Hsatisfied).
  - cbn [add_facts] in Hadd.
    destruct (Hholds relation n (or_introl eq_refl)) as (w & Hw & Hrelation).
    assert (Hfact :=
      fact_true choices Hchoice statics valuation imms relation w i Hrelation
        Hin).
    assert (Hknown_rest : relations_known (Meet_env.typing_env menv) relations).
    { intros r x Hin'. apply (Hknown r x). right. exact Hin'. }
    assert (Hholds_rest :
      ∀ (relation : Relation.t) (n : name),
        (relation, n) ∈ relations ->
        ∃ w : value,
          value_of_name valuation n = Some w
          ∧ relation_holds statics relation w i).
    { intros r n' Hin'. apply Hholds. right. exact Hin'. }
    destruct (fact_of_relation choices imms relation) as [ | | ty ]
      eqn:Hfact_eq.
    + exact (IH menv valuation result Hfits Hsatisfied Hknown_rest Hholds_rest
               Hadd).
    + destruct Hfact.
    + destruct (fact_type_fits choices Hchoice (Meet_env.typing_env menv) imms
                  relation ty Hfact_eq) as [Hty Halias].
      destruct (Meet_env.add_equation_strict menv n ty mh)
        as [ result1 | ] eqn:Hstrict; [ | discriminate Hadd ].
      destruct (add_equation_strict_sound_any_kind mh Hkeeps Hother statics
                  valuation menv n ty w result1 Hsound Hfits Hty Halias
                  Hstatics Hsatisfied Hw Hfact Hstrict)
        as (menv1 & -> & valuation1 & Hextended1 & Hsatisfied1 & Hvalued1).
      destruct (add_equation_strict_keeps_any_kind mh Hkeeps Hother menv menv1
                  n ty Hfits Hty Halias Hstrict)
        as [[Hextends1 Hfits1] _].
      assert (Htenv := meet_env_extends_tenv _ _ Hextends1).
      apply (IH menv1 valuation1 result Hfits1 Hsatisfied1) in Hadd.
      * destruct Hadd as (menv' & -> & Hadded). exists menv'.
        split; [ reflexivity | ].
        exact (meet_env_added_trans statics valuation valuation1 menv menv1
                 menv' (meet_env_extends_keeps_names _ _ Hextends1) Hextended1
                 Hvalued1 Hadded).
      * exact (relations_known_extended _ _ _ Htenv Hknown_rest).
      * intros r n' Hin'. destruct (Hholds_rest r n' Hin') as (w' & Hw' & Hr').
        exists w'. split; [ | exact Hr' ]. rewrite <- Hw'.
        unfold value_of_name.
        destruct n' as [ x | sym ]; [ | reflexivity ].
        exact (Hextended1 x (Hknown_rest r x Hin')).
Qed.

End Reduce.

(* ======================================================================== *)
(* The head answered                                                        *)
(* ======================================================================== *)

Lemma head_satisfied_iff (statics : static_env) (valuation : env) (v : value)
    (head : head_of_kind_naked_immediate) :
  type_satisfied statics valuation v (naked_immediate_head_type head)
  <-> ∃ i : Z,
        v = Value_const (Naked_immediate i)
        ∧ numbers_have (naked_immediates head) i
        ∧ ∀ (relation : Relation.t) (n : name),
            (relation, n) ∈ inverse_relations head ->
            ∃ w : value,
              value_of_name valuation n = Some w
              ∧ relation_holds statics relation w i.
Proof. reflexivity. Qed.

(* A head made of the numbers and the relations of two heads that fit
   fits, and is true where the two are. *)
Lemma new_head_fits (tenv : Typing_env.t)
    (head1 head2 : head_of_kind_naked_immediate)
    (numbers : Or_unknown.t (list Z))
    (relations : list (Relation.t * name)) :
  type_fits tenv (naked_immediate_head_type head1) ->
  type_fits tenv (naked_immediate_head_type head2) ->
  numbers <> Or_unknown.Known [] ->
  (∀ inverse : Relation.t * name,
     inverse ∈ relations ->
     inverse ∈ inverse_relations head1 ∨ inverse ∈ inverse_relations head2) ->
  type_fits tenv
    (naked_immediate_head_type
       (Head_of_kind_naked_immediate numbers relations)).
Proof.
  intros Hhead1 Hhead2 Hnumbers Hsubset.
  apply head_fits_iff in Hhead1, Hhead2. apply head_fits_iff.
  destruct Hhead1 as [_ Hknown1], Hhead2 as [_ Hknown2]. split.
  - destruct numbers as [ [ | number numbers ] | ];
      [ destruct (Hnumbers eq_refl) | reflexivity | reflexivity ].
  - intros relation x Hin. cbn [inverse_relations] in Hin.
    destruct (Hsubset _ Hin) as [ H | H ];
      [ exact (Hknown1 relation x H) | exact (Hknown2 relation x H) ].
Qed.

Lemma new_head_satisfied (statics : static_env) (valuation : env) (v : value)
    (head1 head2 : head_of_kind_naked_immediate)
    (numbers : Or_unknown.t (list Z))
    (relations : list (Relation.t * name)) :
  type_satisfied statics valuation v (naked_immediate_head_type head1) ->
  type_satisfied statics valuation v (naked_immediate_head_type head2) ->
  (∀ i : Z,
     numbers_have (naked_immediates head1) i ->
     numbers_have (naked_immediates head2) i ->
     numbers_have numbers i) ->
  (∀ inverse : Relation.t * name,
     inverse ∈ relations ->
     inverse ∈ inverse_relations head1 ∨ inverse ∈ inverse_relations head2) ->
  type_satisfied statics valuation v
    (naked_immediate_head_type
       (Head_of_kind_naked_immediate numbers relations)).
Proof.
  intros Hhead1 Hhead2 Hnumbers Hsubset.
  apply head_satisfied_iff in Hhead1, Hhead2. apply head_satisfied_iff.
  destruct Hhead1 as (i & Hv & Hnumbers1 & Hrelations1).
  destruct Hhead2 as (i2 & Hv2 & Hnumbers2 & Hrelations2).
  rewrite Hv in Hv2. injection Hv2 as <-.
  exists i. split; [ exact Hv | ].
  split; [ exact (Hnumbers i Hnumbers1 Hnumbers2) | ].
  intros relation n Hin. cbn [inverse_relations] in Hin.
  destruct (Hsubset _ Hin) as [ H | H ];
    [ exact (Hrelations1 relation n H) | exact (Hrelations2 relation n H) ].
Qed.

(* The relations that [meet_inverse_relations] answers are relations
   of the inputs. *)
Lemma meet_inverse_relations_subsets
    (relations1 relations2 : list (Relation.t * name)) :
  let result := meet_inverse_relations relations1 relations2 in
  (∀ inverse : Relation.t * name,
     inverse ∈ left_inputs_only result -> inverse ∈ relations1)
  ∧ (∀ inverse : Relation.t * name,
       inverse ∈ right_inputs_only result -> inverse ∈ relations2)
  ∧ (∀ inverse : Relation.t * name,
       inverse
       ∈ extract_value (meet_return_value_of result) relations1 relations2 ->
       inverse ∈ relations1 ∨ inverse ∈ relations2).
Proof.
  unfold meet_inverse_relations.
  cbn [left_inputs_only right_inputs_only meet_return_value_of].
  split; [ | split ].
  - intros inverse Hin. unfold Keyed.without in Hin.
    apply filter_In in Hin. exact (proj1 Hin).
  - intros inverse Hin. unfold Keyed.without in Hin.
    apply filter_In in Hin. exact (proj1 Hin).
  - intros inverse.
    destruct (Keyed.is_empty _), (Keyed.is_empty _); cbn [extract_value];
      auto.
    intros Hin. apply (proj1 (in_dedup_relations _ _)) in Hin.
    apply in_app_or. exact Hin.
Qed.

(* What the head answered is: an input, or a head made of the numbers
   and the relations given, of which the numbers are not the empty
   set. *)
Lemma head_answered_cases {A B : Type} (a : meet_return_value A)
    (b : meet_return_value B) (numbers : Or_unknown.t (list Z))
    (relations : list (Relation.t * name))
    (head : meet_return_value head_of_kind_naked_immediate) :
  sequence_return_value
    (combine_meet_return_values a b
       (fun _ =>
          head_of_kind_naked_immediate_from_descr_non_empty numbers relations))
  = Some head ->
  head = Left_input ∨ head = Right_input ∨ head = Both_inputs
  ∨ (head = New_result (Head_of_kind_naked_immediate numbers relations)
     ∧ numbers <> Or_unknown.Known []).
Proof.
  assert (Hnew :
    sequence_return_value
      (New_result
         (head_of_kind_naked_immediate_from_descr_non_empty numbers relations))
    = Some head ->
    head = New_result (Head_of_kind_naked_immediate numbers relations)
    ∧ numbers <> Or_unknown.Known []).
  { unfold head_of_kind_naked_immediate_from_descr_non_empty,
      head_of_kind_naked_immediate_from_descr.
    destruct numbers as [ [ | number numbers ] | ]; cbn; intros H;
      try discriminate H; injection H as <-;
      (split; [ reflexivity | discriminate ]). }
  destruct a, b; cbn [combine_meet_return_values];
    try (intros H; injection H as <-; auto; fail);
    intros H; right; right; right; exact (Hnew H).
Qed.

(* ======================================================================== *)
(* The meet of two heads                                                    *)
(* ======================================================================== *)

Lemma meet_head_of_kind_naked_immediate_ok_for :
  for_validity meet_head_of_kind_naked_immediate_ok_statement.
Proof.
  intros choices mh v Hchoice Hkeeps Hsound Hother menv t1 t2 result Hfits
    Ht1 Ht2 Hmeet.
  set (tenv := Meet_env.typing_env menv) in *.
  unfold meet_head_of_kind_naked_immediate in Hmeet.
  change (Meet_env.current_typing_env menv) with tenv in Hmeet.
  assert (Hreduced :
    ∀ (statics : static_env) (valuation : env)
      (head : head_of_kind_naked_immediate),
      typing_env_satisfied_strictly statics valuation tenv ->
      type_satisfied statics valuation v (naked_immediate_head_type head) ->
      ∃ head' : head_of_kind_naked_immediate,
        reduce_head_of_kind_naked_immediate tenv head = Or_bottom.Ok head'
        ∧ type_satisfied statics valuation v
            (naked_immediate_head_type head')).
  { intros statics valuation head Hsatisfied Hhead.
    exact (reduce_head_of_kind_naked_immediate_sound_for statics valuation tenv
             head v Hsatisfied Hhead). }
  destruct (reduce_head_of_kind_naked_immediate tenv t1) as [ t1' | ] eqn:Hr1,
    (reduce_head_of_kind_naked_immediate tenv t2) as [ t2' | ] eqn:Hr2.
  2: { injection Hmeet as <-. apply meet_call_ok_bottom.
       intros statics valuation Hstatics Hsatisfied H1 H2.
       destruct (Hreduced statics valuation t2 Hsatisfied H2) as (h & Heq & _).
       rewrite Hr2 in Heq. discriminate Heq. }
  2: { injection Hmeet as <-. apply meet_call_ok_bottom.
       intros statics valuation Hstatics Hsatisfied H1 H2.
       destruct (Hreduced statics valuation t1 Hsatisfied H1) as (h & Heq & _).
       rewrite Hr1 in Heq. discriminate Heq. }
  2: { injection Hmeet as <-. apply meet_call_ok_bottom.
       intros statics valuation Hstatics Hsatisfied H1 H2.
       destruct (Hreduced statics valuation t1 Hsatisfied H1) as (h & Heq & _).
       rewrite Hr1 in Heq. discriminate Heq. }
  assert (Ht1' :=
    reduce_head_of_kind_naked_immediate_fits_for tenv t1 t1' Hfits Ht1 Hr1).
  assert (Ht2' :=
    reduce_head_of_kind_naked_immediate_fits_for tenv t2 t2' Hfits Ht2 Hr2).
  cbv zeta in Hmeet.
  fold (meet_numbers menv (naked_immediates t1') (naked_immediates t2'))
    in Hmeet.
  destruct (meet_numbers_cases menv (naked_immediates t1')
              (naked_immediates t2')) as (numbers & Hnumbers & Hcases).
  rewrite Hnumbers in Hmeet. cbv beta iota in Hmeet.
  destruct numbers as [ r | ni menv0 ].
  { injection Hmeet as <-. apply meet_call_ok_bottom.
    intros statics valuation Hstatics Hsatisfied H1 H2.
    destruct (Hreduced statics valuation t1 Hsatisfied H1)
      as (h1 & Heq1 & Hh1).
    destruct (Hreduced statics valuation t2 Hsatisfied H2)
      as (h2 & Heq2 & Hh2).
    rewrite Hr1 in Heq1. injection Heq1 as <-.
    rewrite Hr2 in Heq2. injection Heq2 as <-.
    apply head_satisfied_iff in Hh1, Hh2.
    destruct Hh1 as (i & Hv & Hn1 & _), Hh2 as (i2 & Hv2 & Hn2 & _).
    rewrite Hv in Hv2. injection Hv2 as <-.
    exact (Hcases i (conj Hn1 Hn2)). }
  destruct Hcases as [-> Hboth].
  destruct (meet_inverse_relations_subsets (inverse_relations t1')
              (inverse_relations t2')) as (Hleft & Hright & Hall).
  remember (meet_inverse_relations (inverse_relations t1')
              (inverse_relations t2')) as ir eqn:Hir.
  remember (extract_value (meet_return_value_of ir) (inverse_relations t1')
              (inverse_relations t2')) as R eqn:HR.
  remember (extract_value ni (naked_immediates t1') (naked_immediates t2'))
    as N eqn:HN.
  destruct (sequence_return_value _) as [ head | ] eqn:Hhead in Hmeet;
    [ | discriminate Hmeet ].
  cbv beta iota in Hmeet.
  apply head_answered_cases in Hhead.
  assert (Hresult :
    ∃ (rels : list (Relation.t * name)) (reduced : Or_bottom.t Meet_env.t),
      (∀ inverse : Relation.t * name,
         inverse ∈ rels ->
         inverse ∈ inverse_relations t1' ∨ inverse ∈ inverse_relations t2')
      ∧ match N with
        | Or_unknown.Unknown => Some (Or_bottom.Ok menv)
        | Or_unknown.Known imms => add_facts choices mh imms rels menv
        end
        = Some reduced
      ∧ result
        = match reduced with
          | Or_bottom.Ok env => Meet_result.Ok head env
          | Or_bottom.Bottom => Meet_result.Bottom (New_result tt)
          end).
  { destruct ni as [ | | | ni' ]; cbn [extract_value] in HN; subst N.
    - destruct (reduce_inverse_relations choices mh menv
                  (naked_immediates t1') (right_inputs_only ir))
        as [ reduced | ] eqn:Hreduced_eq; [ | discriminate Hmeet ].
      injection Hmeet as <-.
      rewrite reduce_inverse_relations_eq in Hreduced_eq.
      exists (right_inputs_only ir), reduced.
      split; [ intros inverse Hin; right; exact (Hright inverse Hin) | ].
      split; [ exact Hreduced_eq | reflexivity ].
    - destruct (reduce_inverse_relations choices mh menv
                  (naked_immediates t2') (left_inputs_only ir))
        as [ reduced | ] eqn:Hreduced_eq; [ | discriminate Hmeet ].
      injection Hmeet as <-.
      rewrite reduce_inverse_relations_eq in Hreduced_eq.
      exists (left_inputs_only ir), reduced.
      split; [ intros inverse Hin; left; exact (Hleft inverse Hin) | ].
      split; [ exact Hreduced_eq | reflexivity ].
    - injection Hmeet as <-. exists [], (Or_bottom.Ok menv).
      split; [ intros inverse [] | ].
      split; [ destruct (naked_immediates t1'); reflexivity | reflexivity ].
    - destruct (reduce_inverse_relations choices mh menv ni' R)
        as [ reduced | ] eqn:Hreduced_eq; [ | discriminate Hmeet ].
      injection Hmeet as <-.
      rewrite reduce_inverse_relations_eq in Hreduced_eq.
      exists R, reduced. split; [ exact Hall | ].
      split; [ exact Hreduced_eq | reflexivity ]. }
  clear Hmeet HN.
  destruct Hresult as (rels & reduced & Hsubset & Hreduced_eq & ->).
  split.
  - intros r menv' Heq.
    destruct reduced as [ menv1 | ]; [ | discriminate Heq ].
    injection Heq as <- <-.
    assert (Hsteps :
      meet_env_steps menv menv1
      ∧ typing_env_fits (Meet_env.typing_env menv1)).
    { destruct N as [ imms | ].
      - exact (add_facts_keeps choices Hchoice mh Hkeeps Hother imms rels menv
                 menv1 Hfits Hreduced_eq).
      - injection Hreduced_eq as <-.
        split; [ apply meet_env_steps_refl | exact Hfits ]. }
    destruct Hsteps as [[Hextends Hnew] Hfits1].
    split; [ exact Hextends | ]. split; [ exact Hfits1 | ].
    split; [ | exact Hnew ].
    assert (Htenv := meet_env_extends_tenv _ _ Hextends).
    destruct Hhead as [ -> | [ -> | [ -> | [-> Hnonempty] ] ] ];
      cbn [extract_value].
    + exact (tenv_extends_type_fits _ _ _ Htenv Ht1).
    + exact (tenv_extends_type_fits _ _ _ Htenv Ht2).
    + exact (tenv_extends_type_fits _ _ _ Htenv Ht1).
    + apply (new_head_fits _ t1' t2');
        [ exact (tenv_extends_type_fits _ _ _ Htenv Ht1')
        | exact (tenv_extends_type_fits _ _ _ Htenv Ht2')
        | exact Hnonempty
        | exact Hall ].
  - intros statics valuation Hstatics Hsatisfied H1 H2.
    destruct (Hreduced statics valuation t1 Hsatisfied H1)
      as (h1 & Heq1 & Hh1).
    destruct (Hreduced statics valuation t2 Hsatisfied H2)
      as (h2 & Heq2 & Hh2).
    rewrite Hr1 in Heq1. injection Heq1 as <-.
    rewrite Hr2 in Heq2. injection Heq2 as <-.
    assert (Hh1' := Hh1). assert (Hh2' := Hh2).
    apply head_satisfied_iff in Hh1', Hh2'.
    destruct Hh1' as (i & Hv & Hn1 & Hrel1), Hh2' as (i2 & Hv2 & Hn2 & Hrel2).
    rewrite Hv in Hv2. injection Hv2 as <-.
    assert (Hi : numbers_have N i) by (apply Hboth; split; assumption).
    assert (Hsound_env : meet_env_sound statics valuation menv reduced).
    { destruct N as [ imms | ].
      - apply (add_facts_sound choices Hchoice mh Hkeeps Hother statics imms i
                 Hsound Hstatics Hi rels menv valuation reduced Hfits
                 Hsatisfied); [ | | exact Hreduced_eq ].
        + intros relation x Hin.
          destruct (Hsubset _ Hin) as [ H | H ];
            [ exact (proj2 (proj1 (head_fits_iff tenv t1') Ht1') relation x H)
            | exact (proj2 (proj1 (head_fits_iff tenv t2') Ht2') relation x H)
            ].
        + intros relation n Hin.
          destruct (Hsubset _ Hin) as [ H | H ];
            [ exact (Hrel1 relation n H) | exact (Hrel2 relation n H) ].
      - injection Hreduced_eq as <-. exists menv. split; [ reflexivity | ].
        exact (meet_env_added_refl statics valuation menv Hsatisfied). }
    destruct Hsound_env
      as (menv1 & -> & valuation1 & Hextended & Hsatisfied1 & Hvalued).
    exists head, menv1, valuation1. split; [ reflexivity | ].
    split; [ exact Hextended | ]. split; [ exact Hsatisfied1 | ].
    split; [ | exact Hvalued ].
    assert (Hstable :
      ∀ head0 : head_of_kind_naked_immediate,
        type_fits tenv (naked_immediate_head_type head0) ->
        type_satisfied statics valuation v (naked_immediate_head_type head0) ->
        type_satisfied statics valuation1 v
          (naked_immediate_head_type head0)).
    { intros head0 Hfit Hsat.
      exact (type_satisfied_extended statics tenv valuation valuation1 v _
               Hfit Hextended Hsat). }
    destruct Hhead as [ -> | [ -> | [ -> | [-> Hnonempty] ] ] ];
      cbn [extract_value].
    + exact (Hstable t1 Ht1 H1).
    + exact (Hstable t2 Ht2 H2).
    + exact (Hstable t1 Ht1 H1).
    + apply (new_head_satisfied statics valuation1 v t1' t2');
        [ exact (Hstable t1' Ht1' Hh1)
        | exact (Hstable t2' Ht2' Hh2)
        | intros j Hj1 Hj2; apply Hboth; split; assumption
        | exact Hall ].
Qed.

End Validity.

Theorem meet_expanded_head_fuel_other_kinds :
  meet_expanded_head_fuel_other_kinds_statement.
Proof.
  first
    [ exact (@meet_expanded_head_fuel_other_kinds_for standard_validity)
    | exact meet_expanded_head_fuel_other_kinds_for ].
Qed.

Theorem reduce_head_of_kind_naked_immediate_fits :
  reduce_head_of_kind_naked_immediate_fits_statement.
Proof.
  first
    [ exact (@reduce_head_of_kind_naked_immediate_fits_for standard_validity)
    | exact reduce_head_of_kind_naked_immediate_fits_for ].
Qed.

Theorem meet_head_of_kind_naked_immediate_ok :
  meet_head_of_kind_naked_immediate_ok_statement.
Proof.
  first
    [ exact (@meet_head_of_kind_naked_immediate_ok_for standard_validity)
    | exact meet_head_of_kind_naked_immediate_ok_for ].
Qed.
